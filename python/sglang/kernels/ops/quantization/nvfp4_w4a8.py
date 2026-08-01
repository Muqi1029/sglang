# SPDX-License-Identifier: Apache-2.0
"""NVFP4-weight / FP8-activation GEMM kernels for Ada and Hopper.

The dense path and the Ada MoE fallback keep the ModelOpt checkpoint layout in
device memory: two E2M1 values per byte, one E4M3 scale per 16 weights, and an
FP32 outer scale.  Hopper's performance-first MoE path instead requantizes the
same allocation in place to signed INT4/group-32 during loading so the existing
CUTLASS FP8 x INT4 WGMMA kernel can consume it.  Both representations retain
4-bit weight storage and avoid a persistent FP8 weight copy.

Hopper/Ada do not have an FP4 tensor-core instruction.  Two adjacent NVFP4
groups are therefore normalized to a common scale and represented as an FP8
K=32 tile.  The common scale is applied to the FP32 accumulator.  Activations
are dynamically quantized per token and their scale is also applied after the
dot product.  The public contract is thus BF16 -> FP8 x NVFP4 -> FP32 acc ->
BF16.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl

NVFP4_GROUP_SIZE = 16
CUTLASS_W4A8_GROUP_SIZE = 32
_FP8_DOT_K = 32
# E2M1 has a maximum magnitude of 6 while E4M3 can represent 448.  Use a
# power-of-two boost before the E4M3 cast so small block-scale ratios do not
# underflow; compensate exactly in the post-MMA scale.  64 keeps 6 * 64 = 384
# below the finite E4M3 maximum.
_NVFP4_TO_FP8_BOOST = 64.0
_JIT_CUTLASS_W4A8_GROUP_SIZE = tl.constexpr(CUTLASS_W4A8_GROUP_SIZE)
_JIT_NVFP4_TO_FP8_BOOST = tl.constexpr(_NVFP4_TO_FP8_BOOST)


def _check_supported_device(tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError("nvfp4_w4a8 requires CUDA tensors")
    major, minor = torch.cuda.get_device_capability(tensor.device)
    if (major, minor) not in ((8, 9), (9, 0)):
        raise ValueError(
            "nvfp4_w4a8 currently supports Ada (SM89) and Hopper (SM90), "
            f"got SM{major}{minor}."
        )


def nvfp4_w4a8_moe_block_size(
    tensor: torch.Tensor, routes_per_expert: Optional[float] = None
) -> int:
    """Choose route padding without forcing sparse decode experts to M=64."""
    _check_supported_device(tensor)
    major, _ = torch.cuda.get_device_capability(tensor.device)
    if major != 9 or routes_per_expert is None:
        return 16
    if routes_per_expert <= 16:
        return 16
    if routes_per_expert <= 32:
        return 32
    return 64


def repack_nvfp4_for_w4a8(
    weight: torch.Tensor, weight_scale: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reorder NVFP4 storage for coalesced KxN tensor-core tile loads.

    The logical checkpoint shapes remain ``[..., N, K/2]`` and
    ``[..., N, K/16]``.  Only the physical strides change from N-major to
    K-major, analogous to Marlin's one-time weight repack.  Keeping the
    logical shape unchanged also keeps weight inspection and the portable
    dequantization helper working without a backend-specific code path.
    """
    if weight.dtype != torch.uint8 or weight.ndim not in (2, 3):
        raise TypeError("NVFP4 weight must be a 2-D or 3-D packed uint8 tensor")
    if weight_scale.dtype != torch.float8_e4m3fn:
        raise TypeError("NVFP4 block scale must use float8_e4m3fn")
    if weight_scale.ndim != weight.ndim:
        raise ValueError("NVFP4 weight and scale ranks must match")
    if weight.shape[:-1] != weight_scale.shape[:-1]:
        raise ValueError("NVFP4 weight and scale output dimensions must match")
    if weight.shape[-1] * 2 != weight_scale.shape[-1] * NVFP4_GROUP_SIZE:
        raise ValueError("NVFP4 packed weight and block-scale K dimensions mismatch")

    # Transpose-copy-transpose returns the original logical shape backed by a
    # K-major allocation (N has stride 1).  It does not retain a second copy.
    weight_k_major = weight.transpose(-1, -2).contiguous().transpose(-1, -2)
    scale_k_major = weight_scale.transpose(-1, -2).contiguous().transpose(-1, -2)
    return weight_k_major, scale_k_major


@triton.jit
def _decode_e2m1(code):
    magnitude_code = code & 0x7
    magnitude = tl.where(
        magnitude_code == 0,
        0.0,
        tl.where(
            magnitude_code == 1,
            0.5,
            tl.where(
                magnitude_code == 2,
                1.0,
                tl.where(
                    magnitude_code == 3,
                    1.5,
                    tl.where(
                        magnitude_code == 4,
                        2.0,
                        tl.where(
                            magnitude_code == 5,
                            3.0,
                            tl.where(magnitude_code == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return tl.where((code & 0x8) != 0, -magnitude, magnitude)


@triton.jit
def _round_symmetric_int4(value, scale):
    magnitude = tl.floor(tl.abs(value) / scale + 0.5)
    signed = tl.where(value < 0.0, -magnitude, magnitude)
    return tl.maximum(-7.0, tl.minimum(7.0, signed)).to(tl.int32)


@triton.jit
def _nvfp4_to_cutlass_int4_kernel(
    weight_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    cutlass_scale_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    weight_stride_e,
    weight_stride_n,
    weight_stride_k,
    scale_stride_e,
    scale_stride_n,
    scale_stride_k,
    global_scale_stride_e,
    global_scale_stride_s,
    GLOBAL_SCALE_SHARDS: tl.constexpr,
    NUM_ROWS: tl.constexpr,
    BLOCK_ROWS: tl.constexpr,
):
    """In-place E2M1/group16 -> signed INT4/group32 conversion for CUTLASS."""
    pid = tl.program_id(0)
    groups_per_row = K // _JIT_CUTLASS_W4A8_GROUP_SIZE
    rows = pid * BLOCK_ROWS + tl.arange(0, BLOCK_ROWS)
    row_mask = rows < NUM_ROWS
    expert = rows // N
    out_channel = rows % N

    packed_offsets = tl.arange(0, _JIT_CUTLASS_W4A8_GROUP_SIZE // 2)
    global_scale_shard = out_channel // (N // GLOBAL_SCALE_SHARDS)
    global_scale = tl.load(
        weight_global_scale_ptr
        + expert * global_scale_stride_e
        + global_scale_shard * global_scale_stride_s,
        mask=row_mask,
        other=0.0,
    ).to(tl.float32)

    # One program converts several complete rows.  This keeps the launch count
    # small for GLM-5.2 (hundreds of experts) while preserving coalesced 16-byte
    # group accesses and race-free in-place writes.
    for group in tl.range(0, groups_per_row):
        packed_ptrs = (
            weight_ptr
            + expert[:, None] * weight_stride_e
            + out_channel[:, None] * weight_stride_n
            + (group * (_JIT_CUTLASS_W4A8_GROUP_SIZE // 2) + packed_offsets[None, :])
            * weight_stride_k
        )
        packed = tl.load(packed_ptrs, mask=row_mask[:, None], other=0).to(tl.int32)
        low_code = packed & 0xF
        high_code = (packed >> 4) & 0xF

        # The first eight packed bytes belong to the first group16 scale and
        # the remaining eight to the second.  Fold ModelOpt's outer scale into
        # the CUTLASS dequant scale, leaving one BF16 lookup in the GEMM.
        block_scale0 = tl.load(
            weight_scale_ptr
            + expert * scale_stride_e
            + out_channel * scale_stride_n
            + (group * 2) * scale_stride_k,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        block_scale1 = tl.load(
            weight_scale_ptr
            + expert * scale_stride_e
            + out_channel * scale_stride_n
            + (group * 2 + 1) * scale_stride_k,
            mask=row_mask,
            other=0.0,
        ).to(tl.float32)
        block_scale = tl.where(
            packed_offsets[None, :] < 8,
            block_scale0[:, None],
            block_scale1[:, None],
        )

        low = _decode_e2m1(low_code) * block_scale * global_scale[:, None]
        high = _decode_e2m1(high_code) * block_scale * global_scale[:, None]
        max_abs = tl.maximum(tl.max(tl.abs(low), axis=1), tl.max(tl.abs(high), axis=1))
        dequant_scale = max_abs / 7.0
        safe_scale = tl.where(max_abs > 0.0, dequant_scale, 1.0)

        low_q = _round_symmetric_int4(low, safe_scale[:, None])
        high_q = _round_symmetric_int4(high, safe_scale[:, None])
        packed_q = (low_q & 0xF) | ((high_q & 0xF) << 4)
        tl.store(packed_ptrs, packed_q, mask=row_mask[:, None])

        # interleave_scales layout: [E, groups/4, N*4].  Four adjacent K
        # groups form one 64-bit scale element used by the CUTLASS LUT.
        scale_offset = (
            expert * (groups_per_row * N)
            + (group // 4) * (N * 4)
            + out_channel * 4
            + group % 4
        )
        tl.store(
            cutlass_scale_ptr + scale_offset,
            dequant_scale,
            mask=row_mask,
        )


def prepare_nvfp4_for_cutlass_w4a8(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Prepare ModelOpt NVFP4 weights for the SM90 CUTLASS W4A8 kernel.

    Two group-16 E2M1 blocks are requantized into one signed-INT4 group-32
    block.  The packed weight is converted in place; the returned BF16 scale
    tensor occupies exactly as many bytes as the original E4M3 group-16 scale.
    """
    _check_supported_device(weight)
    if torch.cuda.get_device_capability(weight.device)[0] != 9:
        raise ValueError("CUTLASS NVFP4-W4A8 preparation requires SM90")
    if weight.dtype != torch.uint8 or weight.ndim != 3:
        raise TypeError("CUTLASS NVFP4-W4A8 expects W[E,N,K/2] uint8")
    if weight_scale.dtype != torch.float8_e4m3fn or weight_scale.ndim != 3:
        raise TypeError("CUTLASS NVFP4-W4A8 expects S[E,N,K/16] E4M3")

    experts, n, packed_k = weight.shape
    k = packed_k * 2
    groups = k // CUTLASS_W4A8_GROUP_SIZE
    if k % (CUTLASS_W4A8_GROUP_SIZE * 4) != 0:
        raise ValueError(
            "CUTLASS NVFP4-W4A8 requires K divisible by 128 for scale "
            f"interleaving, got K={k}"
        )
    if tuple(weight_scale.shape) != (experts, n, k // NVFP4_GROUP_SIZE):
        raise ValueError(
            "NVFP4 scale shape mismatch: expected "
            f"{(experts, n, k // NVFP4_GROUP_SIZE)}, got {tuple(weight_scale.shape)}"
        )

    global_scale = weight_global_scale.to(torch.float32).contiguous()
    if global_scale.ndim == 1:
        global_scale = global_scale.view(experts, 1)
    if global_scale.ndim != 2 or global_scale.shape[0] != experts:
        raise ValueError(
            f"invalid NVFP4 global scale shape {tuple(global_scale.shape)}"
        )
    if global_scale.shape[1] not in (1, 2):
        raise ValueError("NVFP4 global scale must have one value, or gate/up values")
    if global_scale.shape[1] == 2 and n % 2:
        raise ValueError("gate/up global scales require an even output dimension")

    # ModelOpt tensors normally arrive contiguous.  Keep an explicit guard so
    # the in-place write can never mutate a strided view with aliased elements.
    packed_weight = weight.contiguous()
    source_scale = weight_scale.contiguous()
    cutlass_scale = torch.empty(
        (experts, groups // 4, n * 4),
        device=weight.device,
        dtype=torch.bfloat16,
    )
    block_rows = 8
    grid = (triton.cdiv(experts * n, block_rows),)
    _nvfp4_to_cutlass_int4_kernel[grid](
        packed_weight,
        source_scale,
        global_scale,
        cutlass_scale,
        n,
        k,
        packed_weight.stride(0),
        packed_weight.stride(1),
        packed_weight.stride(2),
        source_scale.stride(0),
        source_scale.stride(1),
        source_scale.stride(2),
        global_scale.stride(0),
        global_scale.stride(1),
        GLOBAL_SCALE_SHARDS=global_scale.shape[1],
        NUM_ROWS=experts * n,
        BLOCK_ROWS=block_rows,
        num_warps=4,
    )
    return packed_weight.view(torch.int8), cutlass_scale


def requantize_nvfp4_to_int4_reference(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Portable reference for the group16 E2M1 -> group32 INT4 conversion."""
    if weight.ndim != 3 or weight.dtype != torch.uint8:
        raise TypeError("reference conversion expects W[E,N,K/2] uint8")
    experts, n, packed_k = weight.shape
    k = packed_k * 2
    if k % (CUTLASS_W4A8_GROUP_SIZE * 4) != 0:
        raise ValueError("reference conversion requires K divisible by 128")

    low = weight & 0xF
    high = (weight >> 4) & 0xF
    codes = torch.stack((low, high), dim=-1).flatten(-2)
    table = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=weight.device,
        dtype=torch.float32,
    )
    values = table[(codes & 0x7).long()]
    values = torch.where((codes & 0x8) != 0, -values, values)
    values = values * weight_scale.float().repeat_interleave(NVFP4_GROUP_SIZE, dim=-1)

    global_scale = weight_global_scale.float()
    if global_scale.ndim == 1:
        global_scale = global_scale.view(experts, 1)
    if global_scale.shape[1] == 2:
        global_scale = global_scale.repeat_interleave(n // 2, dim=1)
    values = values * global_scale.unsqueeze(-1)

    groups = k // CUTLASS_W4A8_GROUP_SIZE
    values = values.reshape(experts, n, groups, CUTLASS_W4A8_GROUP_SIZE)
    scales = values.abs().amax(dim=-1) / 7.0
    safe_scales = torch.where(scales > 0, scales, torch.ones_like(scales))
    quant = torch.round(values / safe_scales.unsqueeze(-1)).clamp(-7, 7).to(torch.int8)
    low_q = quant[..., 0::2].to(torch.int16) & 0xF
    high_q = (quant[..., 1::2].to(torch.int16) & 0xF) << 4
    packed = (low_q | high_q).to(torch.int8).flatten(-2).contiguous()
    interleaved = (
        scales.to(torch.bfloat16)
        .reshape(experts, n, groups // 4, 4)
        .permute(0, 2, 1, 3)
        .reshape(experts, groups // 4, n * 4)
        .contiguous()
    )
    return packed, interleaved


@triton.jit
def _e2m1_to_fp32(nibble):
    """Decode an E2M1 nibble without a lookup table."""
    magnitude = nibble & 0x7
    value = tl.where(
        magnitude == 0,
        0.0,
        tl.where(
            magnitude == 1,
            0.5,
            tl.where(
                magnitude == 2,
                1.0,
                tl.where(
                    magnitude == 3,
                    1.5,
                    tl.where(
                        magnitude == 4,
                        2.0,
                        tl.where(
                            magnitude == 5,
                            3.0,
                            tl.where(magnitude == 6, 4.0, 6.0),
                        ),
                    ),
                ),
            ),
        ),
    )
    return tl.where((nibble & 0x8) != 0, -value, value)


@triton.jit
def _load_normalized_nvfp4_k32(
    weight_ptr,
    weight_scale_ptr,
    weight_base,
    scale_base,
    k_base,
    offs_k,
    offs_n,
    stride_wk,
    stride_wn,
    stride_sk,
    stride_sn,
    K: tl.constexpr,
    N: tl.constexpr,
):
    """Load K=32 packed NVFP4 values and return (FP8 tile, common scale).

    Each half of the K tile has a distinct block-16 scale.  Normalizing both
    halves by their per-output-column maximum lets the tile use one FP32 scale
    after the FP8 dot while retaining the original NVFP4 storage.
    """
    k_mask = offs_k[:, None] < K
    n_mask = offs_n[None, :] < N
    packed = tl.load(
        weight_ptr
        + weight_base
        + (offs_k[:, None] // 2) * stride_wk
        + offs_n[None, :] * stride_wn,
        mask=k_mask & n_mask,
        other=0,
    ).to(tl.uint8)
    nibble = tl.where((offs_k[:, None] & 1) == 0, packed & 0xF, packed >> 4)
    fp4 = _e2m1_to_fp32(nibble)

    group0 = k_base // 16
    scale0 = tl.load(
        weight_scale_ptr + scale_base + group0 * stride_sk + offs_n * stride_sn,
        mask=offs_n < N,
        other=0.0,
    ).to(tl.float32)
    scale1 = tl.load(
        weight_scale_ptr + scale_base + (group0 + 1) * stride_sk + offs_n * stride_sn,
        mask=(offs_n < N) & (k_base + 16 < K),
        other=0.0,
    ).to(tl.float32)
    common_scale = tl.maximum(tl.abs(scale0), tl.abs(scale1))
    safe_scale = tl.where(common_scale == 0.0, 1.0, common_scale)
    element_scale = tl.where(
        (offs_k[:, None] & 31) < 16,
        scale0[None, :],
        scale1[None, :],
    )
    weight_fp8 = (
        fp4 * element_scale / safe_scale[None, :] * _JIT_NVFP4_TO_FP8_BOOST
    ).to(tl.float8e4nv)
    return weight_fp8, common_scale / _JIT_NVFP4_TO_FP8_BOOST


@triton.jit
def _nvfp4_w4a8_gemm_kernel(
    a_ptr,
    weight_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    a_scale_ptr,
    out_ptr,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    stride_am,
    stride_ak,
    stride_wn,
    stride_wk,
    stride_sn,
    stride_sk,
    stride_om,
    stride_on,
    GLOBAL_SCALE_PER_CHANNEL: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = offs_m < M
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    if GLOBAL_SCALE_PER_CHANNEL:
        weight_global_scale = tl.load(
            weight_global_scale_ptr + offs_n, mask=offs_n < N, other=0.0
        ).to(tl.float32)
    else:
        weight_global_scale = tl.load(weight_global_scale_ptr).to(tl.float32)
    for k_start in range(0, K, BLOCK_K):
        for k_pair in tl.static_range(0, BLOCK_K, 32):
            offs_k = k_start + k_pair + tl.arange(0, 32)
            a = tl.load(
                a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=m_mask[:, None] & (offs_k[None, :] < K),
                other=0.0,
            )
            weight, common_scale = _load_normalized_nvfp4_k32(
                weight_ptr,
                weight_scale_ptr,
                0,
                0,
                k_start + k_pair,
                offs_k,
                offs_n,
                stride_wk,
                stride_wn,
                stride_sk,
                stride_sn,
                K,
                N,
            )
            accumulator += (
                tl.dot(
                    a,
                    weight,
                    out_dtype=tl.float32,
                    max_num_imprecise_acc=32,
                )
                * common_scale[None, :]
                * weight_global_scale
            )

    a_scale = tl.load(a_scale_ptr + offs_m, mask=m_mask, other=0.0).to(tl.float32)
    accumulator *= a_scale[:, None]
    tl.store(
        out_ptr + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on,
        accumulator,
        mask=m_mask[:, None] & (offs_n[None, :] < N),
    )


def nvfp4_w4a8_linear(
    x: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run a dense NVFP4-W4A8 linear and return the input floating dtype."""
    _check_supported_device(x)
    if x.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"nvfp4_w4a8 input must be BF16/FP16, got {x.dtype}")
    if weight.dtype != torch.uint8 or weight.ndim != 2:
        raise TypeError("NVFP4 weight must be a 2-D packed uint8 tensor")
    if weight_scale.dtype != torch.float8_e4m3fn or weight_scale.ndim != 2:
        raise TypeError("NVFP4 block scale must be a 2-D float8_e4m3fn tensor")

    original_shape = x.shape
    x_2d = x.reshape(-1, original_shape[-1]).contiguous()
    m, k = x_2d.shape
    n = weight.shape[0]
    if k % _FP8_DOT_K != 0:
        raise ValueError(f"nvfp4_w4a8 requires K divisible by 32, got K={k}")
    if weight.shape[1] * 2 != k:
        raise ValueError(
            f"packed weight K mismatch: input K={k}, weight shape={tuple(weight.shape)}"
        )
    if tuple(weight_scale.shape) != (n, k // NVFP4_GROUP_SIZE):
        raise ValueError(
            "NVFP4 scale shape mismatch: expected "
            f"{(n, k // NVFP4_GROUP_SIZE)}, got {tuple(weight_scale.shape)}"
        )
    if weight_global_scale.numel() not in (1, n):
        raise ValueError(
            "dense NVFP4 global scale must be scalar or per-output-channel; "
            f"got {weight_global_scale.numel()} values for N={n}"
        )
    if m == 0:
        return x.new_empty((*original_shape[:-1], n))

    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

    x_fp8, x_scale = sglang_per_token_quant_fp8(x_2d)
    out = torch.empty((m, n), device=x.device, dtype=x.dtype)
    major, _ = torch.cuda.get_device_capability(x.device)
    block_m = 16 if m <= 32 else (64 if major == 9 else 32)
    # Marlin's small-M path uses an N=128 thread tile.  It amortizes route and
    # scale handling while retaining enough independent N tiles for the GLM
    # projection sizes.  Fall back to N=64 for larger M to control registers.
    block_n = 128 if m <= 16 and n >= 128 else 64
    block_k = 64
    _nvfp4_w4a8_gemm_kernel[(triton.cdiv(m, block_m), triton.cdiv(n, block_n))](
        x_fp8,
        weight,
        weight_scale,
        weight_global_scale.reshape(-1),
        x_scale,
        out,
        m,
        n,
        k,
        x_fp8.stride(0),
        x_fp8.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight_scale.stride(0),
        weight_scale.stride(1),
        out.stride(0),
        out.stride(1),
        GLOBAL_SCALE_PER_CHANNEL=weight_global_scale.numel() == n,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    if bias is not None:
        out.add_(bias)
    return out.view(*original_shape[:-1], n)


@triton.jit
def _nvfp4_w4a8_grouped_gemm_kernel(
    a_ptr,
    weight_ptr,
    weight_scale_ptr,
    weight_global_scale_ptr,
    a_scale_ptr,
    out_ptr,
    topk_weights_ptr,
    sorted_token_ids_ptr,
    expert_ids_ptr,
    num_tokens_post_padded_ptr,
    N: tl.constexpr,
    K: tl.constexpr,
    num_valid_routes,
    stride_am,
    stride_ak,
    stride_we,
    stride_wn,
    stride_wk,
    stride_se,
    stride_sn,
    stride_sk,
    stride_ge,
    stride_gs,
    stride_om,
    stride_on,
    top_k: tl.constexpr,
    global_scale_shards: tl.constexpr,
    MUL_ROUTED_WEIGHT: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    num_tokens_post_padded = tl.load(num_tokens_post_padded_ptr)
    if pid_m * BLOCK_M >= num_tokens_post_padded:
        return

    route_slots = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    routes = tl.load(sorted_token_ids_ptr + route_slots).to(tl.int64)
    route_mask = routes < num_valid_routes
    expert_i32 = tl.load(expert_ids_ptr + pid_m)
    expert = expert_i32.to(tl.int64)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if expert_i32 == -1:
        tl.store(
            out_ptr + routes[:, None] * stride_om + offs_n[None, :] * stride_on,
            0.0,
            mask=route_mask[:, None] & (offs_n[None, :] < N),
        )
        return

    source_rows = routes // top_k
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    if global_scale_shards == 1:
        global_scale = tl.load(weight_global_scale_ptr + expert * stride_ge).to(
            tl.float32
        )
    else:
        shard_size = N // global_scale_shards
        shard = offs_n // shard_size
        global_scale = tl.load(
            weight_global_scale_ptr + expert * stride_ge + shard * stride_gs,
            mask=offs_n < N,
            other=0.0,
        ).to(tl.float32)

    for k_start in range(0, K, BLOCK_K):
        for k_pair in tl.static_range(0, BLOCK_K, 32):
            offs_k = k_start + k_pair + tl.arange(0, 32)
            a = tl.load(
                a_ptr + source_rows[:, None] * stride_am + offs_k[None, :] * stride_ak,
                mask=route_mask[:, None] & (offs_k[None, :] < K),
                other=0.0,
            )
            weight, common_scale = _load_normalized_nvfp4_k32(
                weight_ptr,
                weight_scale_ptr,
                expert * stride_we,
                expert * stride_se,
                k_start + k_pair,
                offs_k,
                offs_n,
                stride_wk,
                stride_wn,
                stride_sk,
                stride_sn,
                K,
                N,
            )
            accumulator += (
                tl.dot(
                    a,
                    weight,
                    out_dtype=tl.float32,
                    max_num_imprecise_acc=32,
                )
                * common_scale[None, :]
                * global_scale
            )

    a_scale = tl.load(a_scale_ptr + source_rows, mask=route_mask, other=0.0).to(
        tl.float32
    )
    accumulator *= a_scale[:, None]
    if MUL_ROUTED_WEIGHT:
        routed_weight = tl.load(
            topk_weights_ptr + routes, mask=route_mask, other=0.0
        ).to(tl.float32)
        accumulator *= routed_weight[:, None]
    tl.store(
        out_ptr + routes[:, None] * stride_om + offs_n[None, :] * stride_on,
        accumulator,
        mask=route_mask[:, None] & (offs_n[None, :] < N),
    )


def nvfp4_w4a8_grouped_gemm(
    a: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    *,
    top_k: int,
    mul_routed_weight: bool,
    block_m: int,
    input_scale: Optional[torch.Tensor] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Grouped NVFP4-W4A8 GEMM over already aligned MoE route ids."""
    _check_supported_device(a)
    if a.ndim != 2 or weight.ndim != 3 or weight_scale.ndim != 3:
        raise ValueError(
            "grouped NVFP4-W4A8 expects " "A[rows,K], W[E,N,K/2], S[E,N,K/16]"
        )
    is_prequantized = input_scale is not None
    if is_prequantized:
        if a.dtype != torch.float8_e4m3fn:
            raise TypeError(
                "prequantized grouped nvfp4_w4a8 input must use float8_e4m3fn"
            )
        if output_dtype not in (torch.bfloat16, torch.float16):
            raise TypeError(
                "prequantized grouped GEMM requires a BF16/FP16 output_dtype"
            )
    elif a.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError(f"grouped nvfp4_w4a8 input must be BF16/FP16, got {a.dtype}")
    if weight.dtype != torch.uint8 or weight_scale.dtype != torch.float8_e4m3fn:
        raise TypeError("grouped NVFP4-W4A8 expects uint8 weights and E4M3 scales")
    routes, k = a.shape
    experts, n, packed_k = weight.shape
    if k % _FP8_DOT_K != 0 or packed_k * 2 != k:
        raise ValueError(
            "grouped nvfp4_w4a8 requires matching K divisible by 32, " f"got K={k}"
        )
    if tuple(weight_scale.shape) != (experts, n, k // NVFP4_GROUP_SIZE):
        raise ValueError(
            f"grouped NVFP4 scale shape mismatch: {tuple(weight_scale.shape)}"
        )
    global_scale = weight_global_scale.contiguous()
    if global_scale.ndim == 1:
        global_scale = global_scale.view(experts, 1)
    if global_scale.ndim != 2 or global_scale.shape[0] != experts:
        raise ValueError(
            f"invalid expert global scale shape {tuple(weight_global_scale.shape)}"
        )
    if global_scale.shape[1] not in (1, 2):
        raise ValueError("expert global scales must have one value, or gate/up values")
    if global_scale.shape[1] == 2 and n % 2:
        raise ValueError("gate/up global scales require an even output dimension")
    if routes == 0:
        return torch.empty((0, n), device=a.device, dtype=output_dtype or a.dtype)

    if block_m not in (16, 32, 64):
        raise ValueError(f"unsupported nvfp4_w4a8 MoE block_m={block_m}")
    if sorted_token_ids.shape[0] % block_m != 0:
        raise ValueError(
            "sorted route storage is not aligned to the requested block_m: "
            f"{sorted_token_ids.shape[0]} vs {block_m}"
        )
    expected_expert_blocks = sorted_token_ids.shape[0] // block_m
    if expert_ids.numel() < expected_expert_blocks:
        raise ValueError(
            "expert_ids was produced with a different MoE block size: "
            f"need {expected_expert_blocks}, got {expert_ids.numel()}"
        )

    if is_prequantized:
        a_fp8 = a.contiguous()
        a_scale = input_scale
        if a_scale.numel() != routes:
            raise ValueError(
                "expected one FP8 scale per input row "
                f"({routes}), got {a_scale.numel()}"
            )
        a_scale = a_scale.reshape(routes, 1).contiguous()
        result_dtype = output_dtype
    else:
        from sglang.kernels.ops.quantization.fp8_kernel import (
            sglang_per_token_quant_fp8,
        )

        a_fp8, a_scale = sglang_per_token_quant_fp8(a.contiguous())
        result_dtype = a.dtype
    out = torch.empty((routes * top_k, n), device=a.device, dtype=result_dtype)
    # Decode is dominated by many very small expert groups.  N=128 mirrors
    # Marlin's small-batch tile and halves program/metadata overhead for both
    # GLM-5.2 expert GEMMs (N=512 and N=6144 under TP8).
    block_n = 128 if block_m == 16 and n >= 128 else 64
    block_k = 64
    grid = (triton.cdiv(sorted_token_ids.shape[0], block_m) * triton.cdiv(n, block_n),)
    _nvfp4_w4a8_grouped_gemm_kernel[grid](
        a_fp8,
        weight,
        weight_scale,
        global_scale,
        a_scale,
        out,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        n,
        k,
        routes * top_k,
        a_fp8.stride(0),
        a_fp8.stride(1),
        weight.stride(0),
        weight.stride(1),
        weight.stride(2),
        weight_scale.stride(0),
        weight_scale.stride(1),
        weight_scale.stride(2),
        global_scale.stride(0),
        global_scale.stride(1),
        out.stride(0),
        out.stride(1),
        top_k=top_k,
        global_scale_shards=global_scale.shape[1],
        MUL_ROUTED_WEIGHT=mul_routed_weight,
        BLOCK_M=block_m,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        num_warps=4,
        num_stages=2,
    )
    return out


def dequantize_nvfp4_reference(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    weight_global_scale: torch.Tensor,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Portable reference decoder for tests and checkpoint inspection."""
    if weight.dtype != torch.uint8:
        raise TypeError("weight must use packed uint8 NVFP4 storage")
    low = weight & 0xF
    high = (weight >> 4) & 0xF
    nibbles = torch.stack((low, high), dim=-1).flatten(-2)
    values = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        device=weight.device,
        dtype=torch.float32,
    )
    magnitude = values[(nibbles & 0x7).long()]
    unpacked = torch.where((nibbles & 0x8) != 0, -magnitude, magnitude)
    scales = weight_scale.to(torch.float32).repeat_interleave(NVFP4_GROUP_SIZE, dim=-1)
    global_scale = weight_global_scale.to(torch.float32)
    if global_scale.numel() == weight.shape[-2] and global_scale.ndim == 1:
        global_scale = global_scale.unsqueeze(-1)
    return (unpacked * scales * global_scale).to(dtype)


__all__ = [
    "CUTLASS_W4A8_GROUP_SIZE",
    "NVFP4_GROUP_SIZE",
    "dequantize_nvfp4_reference",
    "nvfp4_w4a8_grouped_gemm",
    "nvfp4_w4a8_linear",
    "nvfp4_w4a8_moe_block_size",
    "prepare_nvfp4_for_cutlass_w4a8",
    "repack_nvfp4_for_w4a8",
    "requantize_nvfp4_to_int4_reference",
]
