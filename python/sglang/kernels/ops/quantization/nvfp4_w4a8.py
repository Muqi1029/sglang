# SPDX-License-Identifier: Apache-2.0
"""NVFP4-weight / FP8-activation GEMM kernels for Ada and Hopper.

The kernels in this module deliberately keep the ModelOpt checkpoint layout in
device memory: two E2M1 values per byte, one E4M3 scale per 16 weights, and an
FP32 outer scale.  A weight tile is expanded to E4M3 only in registers before
an FP8 tensor-core dot product; no persistent FP8 copy of the weight is made.

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
_FP8_DOT_K = 32


def _check_supported_device(tensor: torch.Tensor) -> None:
    if not tensor.is_cuda:
        raise ValueError("nvfp4_w4a8 requires CUDA tensors")
    major, minor = torch.cuda.get_device_capability(tensor.device)
    if (major, minor) not in ((8, 9), (9, 0)):
        raise ValueError(
            "nvfp4_w4a8 currently supports Ada (SM89) and Hopper (SM90), "
            f"got SM{major}{minor}."
        )


def nvfp4_w4a8_moe_block_size(tensor: torch.Tensor) -> int:
    """Routing alignment used by the architecture-specific FP8 MMA tile."""
    _check_supported_device(tensor)
    major, _ = torch.cuda.get_device_capability(tensor.device)
    # Hopper FP8 uses an m64 WGMMA tile.  Ada uses m16 FP8 mma.sync.
    return 64 if major == 9 else 16


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
    weight_fp8 = (fp4 * element_scale / safe_scale[None, :]).to(tl.float8e4nv)
    return weight_fp8, common_scale


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
    block_m = 64 if major == 9 else (16 if m <= 32 else 32)
    block_n = 64
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
) -> torch.Tensor:
    """Grouped NVFP4-W4A8 GEMM over already aligned MoE route ids."""
    _check_supported_device(a)
    if a.ndim != 2 or weight.ndim != 3 or weight_scale.ndim != 3:
        raise ValueError(
            "grouped NVFP4-W4A8 expects " "A[rows,K], W[E,N,K/2], S[E,N,K/16]"
        )
    if a.dtype not in (torch.bfloat16, torch.float16):
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
        return a.new_empty((0, n))

    from sglang.kernels.ops.quantization.fp8_kernel import sglang_per_token_quant_fp8

    a_fp8, a_scale = sglang_per_token_quant_fp8(a.contiguous())
    out = torch.empty((routes * top_k, n), device=a.device, dtype=a.dtype)
    block_m = nvfp4_w4a8_moe_block_size(a)
    block_n = 64
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
    "NVFP4_GROUP_SIZE",
    "dequantize_nvfp4_reference",
    "nvfp4_w4a8_grouped_gemm",
    "nvfp4_w4a8_linear",
    "nvfp4_w4a8_moe_block_size",
]
