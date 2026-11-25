"""NVFP4 dense and MoE weight preparation for the Humming GEMM backend.

Checkpoint weights stay group-16 FP4. SM80–SM87 consume those codes directly
and fold each projection's global scale into the group scales (W4A16). SM89
and newer dequantize and requantize to group-32 once at load time, because
FP8 MMA reads 32 K elements (W4A8).
"""

from __future__ import annotations

import json
import logging

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4
from sglang.srt.layers.utils import copy_or_rebind_param
from sglang.srt.utils.common import round_up
from sglang.srt.utils.device import get_device_capability, is_cuda

logger = logging.getLogger(__name__)

_SM80 = (8, 0)
_SM89 = (8, 9)
_PAD_N = 256
_PAD_K = 128
_REQUANT_CHUNK_N = 16384
_NUM_STREAM_K_LOCKS = 2048
HUMMING_SILU_AND_MUL_EXPR = "(a / (1.0f + __expf(-a))) * b"


def nvfp4_humming_uses_fp8_activations() -> bool:
    """SM89 and newer Humming NVFP4 uses dynamic FP8 activations."""
    return get_device_capability() >= _SM89


def humming_fp8_supported() -> bool:
    """Humming FP8 kernels require CUDA SM89+ tensor cores."""
    return is_cuda() and get_device_capability() >= _SM89


def nvfp4_humming_quant_mode() -> str:
    # Ampere has no FP8 tensor cores, so it keeps the layer's BF16/FP16 input.
    return "w4a8" if nvfp4_humming_uses_fp8_activations() else "w4a16"


def _should_use_f16_accum(
    layer: torch.nn.Module, *, use_fp8: bool, orig_dtype: torch.dtype
) -> bool:
    from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

    return (
        envs.SGLANG_HUMMING_USE_F16_ACCUM.get()
        and (use_fp8 or orig_dtype == torch.float16)
        and not isinstance(layer, VocabParallelEmbedding)
    )


def _finalize_humming_dense_layer(
    layer: torch.nn.Module,
    *,
    use_f16_accum: bool,
    output_dtype: torch.dtype | None = None,
) -> None:
    if output_dtype is not None:
        layer._humming_output_dtype = output_dtype
    elif hasattr(layer, "_humming_output_dtype"):
        delattr(layer, "_humming_output_dtype")
    layer._humming_compute_config = json.dumps(
        {
            "use_f16_accum": use_f16_accum,
            "gemm_type": "dense",
        }
    )
    layer.register_buffer(
        "locks",
        torch.zeros(_NUM_STREAM_K_LOCKS, dtype=torch.int32, device=layer.weight.device),
    )


def prepare_nvfp4_layer_for_humming(layer: torch.nn.Module, group_size: int) -> None:
    """Repack one dense NVFP4 linear into the Humming weight layout."""
    _validate_nvfp4_humming(group_size, "NVFP4 Humming")
    use_fp8 = nvfp4_humming_uses_fp8_activations()
    orig_dtype = layer.params_dtype
    use_f16_accum = _should_use_f16_accum(layer, use_fp8=use_fp8, orig_dtype=orig_dtype)
    compute_dtype = torch.float16 if use_f16_accum else orig_dtype
    # One global scale per fused projection, repeated across that projection's
    # rows. Taking the max would mix Q/K/V (or gate/up) scales.
    global_scale = layer.weight_scale_2.repeat_interleave(
        torch.tensor(layer.logical_widths, device=layer.weight.device)
    )[:, None]
    _prepare_sublayer(
        layer,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        global_scale=global_scale,
        shape_n=layer.output_size_per_partition,
        shape_k=layer.input_size_per_partition,
        params_dtype=compute_dtype,
        use_fp8=use_fp8,
    )
    if use_fp8 and layer.output_size_per_partition > _REQUANT_CHUNK_N:
        torch.cuda.empty_cache()
    _finalize_humming_dense_layer(
        layer,
        use_f16_accum=use_f16_accum,
        output_dtype=orig_dtype if compute_dtype != orig_dtype else None,
    )


def prepare_fp8_layer_for_humming(layer: torch.nn.Module) -> None:
    """Repack one dense per-tensor/channel FP8 linear into the Humming layout."""
    from humming.layer import HummingMethod
    from humming.schema import HummingInputSchema, HummingWeightSchema

    if not humming_fp8_supported():
        raise ValueError("FP8 Humming requires an NVIDIA SM89+ GPU.")

    orig_dtype = getattr(layer, "params_dtype", None) or layer.orig_dtype
    # W8A8 FP8 (group_size=0) accumulates across full K before epilogue
    # scaling and overflows FP16 accumulators on large K; keep FP32 accumulators.
    compute_dtype = orig_dtype
    shape_n = getattr(layer, "output_size_per_partition", layer.weight.shape[0])
    shape_k = getattr(layer, "input_size_per_partition", layer.weight.shape[1])

    weight = layer.weight.data.contiguous()
    meta_shape_n = round_up(shape_n, 8) if shape_n % 8 != 0 else shape_n
    # W8A8 tensor cores require K padded to a multiple of 128 // a_bits (= 16).
    meta_shape_k = round_up(shape_k, 16) if shape_k % 16 != 0 else shape_k
    if meta_shape_k != shape_k or meta_shape_n != shape_n:
        weight = torch.nn.functional.pad(
            weight, (0, meta_shape_k - shape_k, 0, meta_shape_n - shape_n)
        )
    if meta_shape_k != shape_k:
        layer._humming_padded_k = meta_shape_k
    elif hasattr(layer, "_humming_padded_k"):
        delattr(layer, "_humming_padded_k")

    weight_scale = layer.weight_scale.data
    if weight_scale.numel() == 1:
        weight_scale_type = "tensor"
        weight_scale_tensor = weight_scale.reshape(1).to(torch.float32).contiguous()
    else:
        weight_scale_type = "channel"
        weight_scale_tensor = weight_scale.reshape(shape_n, 1).to(compute_dtype)
        if meta_shape_n != shape_n:
            weight_scale_tensor = torch.nn.functional.pad(
                weight_scale_tensor, (0, 0, 0, meta_shape_n - shape_n)
            )
        weight_scale_tensor = weight_scale_tensor.contiguous()

    weight_schema = HummingWeightSchema(
        b_dtype="float8e4m3",
        weight_scale_group_size=0,
        weight_scale_type=weight_scale_type,
        weight_scale_2_type="none",
    )

    tensors: dict[str, torch.Tensor] = {
        "weight": weight.view(torch.int32),
        "weight_scale": weight_scale_tensor,
    }
    raw_input_scale = getattr(layer, "input_scale", None)
    if raw_input_scale is not None:
        input_schema = HummingInputSchema(
            a_dtype="float8e4m3", input_quant_mode="static_tensor"
        )
        tensors["input_scale"] = (
            raw_input_scale.data.reshape(1).to(torch.float32).contiguous()
        )
        stale_names = ("weight_scale_2", "weight_scale_inv", "input_scale_2")
    else:
        input_schema = HummingInputSchema(
            a_dtype="float8e4m3", input_quant_mode="dynamic_token"
        )
        stale_names = (
            "weight_scale_2",
            "weight_scale_inv",
            "input_scale",
            "input_scale_2",
        )

    for stale in stale_names:
        if hasattr(layer, stale):
            delattr(layer, stale)
    for name, tensor in tensors.items():
        copy_or_rebind_param(layer, name, tensor)

    HummingMethod.prepare_layer_meta(
        layer=layer,
        shape_n=meta_shape_n,
        shape_k=meta_shape_k,
        weight_schema=weight_schema,
        input_schema=input_schema,
        pad_n_to_multiple=_PAD_N,
        pad_k_to_multiple=_PAD_K,
        has_bias=False,
        torch_dtype=compute_dtype,
        sublayer_name="",
        tensors=tensors,
    )
    HummingMethod.transform_humming_layer(layer, sublayer_name="", already_padded=False)
    _finalize_humming_dense_layer(layer, use_f16_accum=False, output_dtype=None)


def _apply_humming_linear(
    layer: torch.nn.Module,
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    from humming.forward import humming_forward

    if isinstance(x, tuple):
        tensor_in, input_scale = x
    else:
        tensor_in = x
        input_scale = getattr(layer, "input_scale", None)

    leading_shape = tensor_in.shape[:-1]
    x_2d = tensor_in.reshape(-1, tensor_in.shape[-1])
    padded_k = getattr(layer, "_humming_padded_k", None)
    if padded_k is not None and padded_k > x_2d.shape[-1]:
        x_2d = torch.nn.functional.pad(x_2d, (0, padded_k - x_2d.shape[-1]))

    output = humming_forward(
        layer.humming_metas[""],
        inputs=x_2d,
        weight=layer.weight,
        weight_scale=layer.weight_scale,
        input_scale=input_scale,
        locks=layer.locks,
        compute_config=getattr(layer, "_humming_compute_config", None),
    )
    output_size = layer.output_size_per_partition
    if output.shape[-1] > output_size:
        output = output[:, :output_size]
    output = output.reshape(*leading_shape, output_size)
    output_dtype = getattr(layer, "_humming_output_dtype", None)
    if (
        output_dtype is not None
        and output.dtype != output_dtype
        and not getattr(layer, "_humming_keep_f16_output", False)
    ):
        output = output.to(output_dtype)
    if bias is not None:
        if bias.dtype != output.dtype:
            bias = bias.to(output.dtype)
        output = output + bias
    return output


def apply_nvfp4_humming_linear(
    layer: torch.nn.Module,
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    bias: torch.Tensor | None = None,
    *,
    is_awq: bool = False,
) -> torch.Tensor:
    if isinstance(x, tuple):
        if is_awq:
            raise ValueError(
                "Pre-quantized FP8 tuple input is not supported with AWQ pre_quant_scale."
            )
        return _apply_humming_linear(layer, x, bias)
    if is_awq:
        x = x * layer.pre_quant_scale
    return _apply_humming_linear(layer, x, bias)


def apply_fp8_humming_linear(
    layer: torch.nn.Module,
    x: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    return _apply_humming_linear(layer, x, bias)


def can_fuse_silu_and_mul_humming_fp8(down_proj: torch.nn.Module) -> bool:
    """Whether down_proj consumes a fused SiLU+Mul+DynamicTokenFP8 tuple."""
    from humming import dtypes
    from humming.config import InputQuantizationMode

    if (
        not hasattr(down_proj, "humming_metas")
        or "" not in down_proj.humming_metas
        or hasattr(down_proj, "base_layer")
        or hasattr(down_proj, "pre_quant_scale")
        or not getattr(down_proj, "input_is_parallel", True)
        or getattr(down_proj, "_humming_padded_k", None) is not None
    ):
        return False
    meta = down_proj.humming_metas[""]
    quant_method = getattr(down_proj, "quant_method", None)
    weight_schema = getattr(quant_method, "weight_schema", None)
    hadamard_block_size = (
        getattr(weight_schema, "hadamard_block_size", 0) or 0
        if weight_schema is not None
        else 0
    )
    return (
        meta.a_dtype == dtypes.float8e4m3
        and meta.input_quant_mode == InputQuantizationMode.DynamicToken
        and hadamard_block_size <= 1
    )


def silu_and_mul_humming_fp8_quant(
    down_proj: torch.nn.Module, gate_up: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute SiLU(gate) * up and dynamic per-token FP8 quant for down_proj."""
    from humming.forward import may_process_input

    q_fp8, _, token_scales = may_process_input(
        down_proj.humming_metas[""],
        gate_up.reshape(-1, gate_up.shape[-1]),
        activation_type="binary_split",
        activation_impl=HUMMING_SILU_AND_MUL_EXPR,
    )
    return q_fp8.reshape(*gate_up.shape[:-1], q_fp8.shape[-1]), token_scales


def prepare_moe_nvfp4_layer_for_humming(
    layer: torch.nn.Module, group_size: int | None = None
) -> None:
    """Repack NVFP4 expert projections into the Humming MoE weight layout."""
    if group_size is None:
        group_size = layer.quant_config.group_size
    _validate_nvfp4_humming(group_size, "NVFP4 Humming MoE")
    if layer.moe_runner_config.apply_router_weight_on_input:
        raise ValueError("Humming NVFP4 MoE does not support router weights on input.")
    if not layer.moe_runner_config.is_gated:
        raise ValueError("Humming NVFP4 MoE requires a gated activation.")

    from sglang.srt.layers.quantization.humming_utils import (
        configure_humming_deepep_dispatch,
    )

    use_deepep_fp8_dispatch = configure_humming_deepep_dispatch(layer)
    use_fp8 = nvfp4_humming_uses_fp8_activations()
    use_f16_accum = _should_use_f16_accum(
        layer, use_fp8=use_fp8, orig_dtype=layer.params_dtype
    )
    compute_dtype = torch.float16 if use_f16_accum else layer.params_dtype
    intermediate = layer.intermediate_size_per_partition
    # Gate and up keep their own global scales when stored separately as [E, 2];
    # a shared [E] or [E, 1] scale broadcasts over both halves.
    w13_scale_2 = layer.w13_weight_scale_2.reshape(layer.num_local_experts, -1)
    w13_global_scale = (
        w13_scale_2.repeat_interleave(intermediate, dim=1).unsqueeze(-1)
        if w13_scale_2.shape[1] > 1
        else w13_scale_2.unsqueeze(-1)
    )
    w2_global_scale = layer.w2_weight_scale_2.reshape(layer.num_local_experts, 1, 1)

    layer.weight_schemas = {}
    for sublayer_name, shape_n, shape_k, global_scale in (
        ("w13", 2 * intermediate, layer.hidden_size, w13_global_scale),
        ("w2", layer.hidden_size, intermediate, w2_global_scale),
    ):
        layer.weight_schemas[sublayer_name] = _prepare_sublayer(
            layer,
            weight=getattr(layer, f"{sublayer_name}_weight"),
            weight_scale=getattr(layer, f"{sublayer_name}_weight_scale"),
            global_scale=global_scale,
            shape_n=shape_n,
            shape_k=shape_k,
            params_dtype=compute_dtype,
            use_fp8=use_fp8,
            sublayer_name=sublayer_name,
            num_experts=layer.num_local_experts,
            use_deepep_fp8_dispatch=use_deepep_fp8_dispatch,
        )
        if use_fp8:
            torch.cuda.empty_cache()

    layer.register_buffer(
        "locks",
        torch.zeros(
            _NUM_STREAM_K_LOCKS, dtype=torch.int32, device=layer.w13_weight.device
        ),
    )


def _validate_nvfp4_humming(group_size: int, label: str) -> None:
    if not is_cuda() or get_device_capability() < _SM80:
        raise ValueError(f"{label} requires an NVIDIA SM80+ GPU.")
    if group_size != 16:
        raise ValueError(f"{label} requires group_size=16, got {group_size}.")


def _requant_nvfp4_to_group32(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    *,
    shape_n: int,
    shape_k: int,
    params_dtype: torch.dtype,
    schema,
) -> dict[str, torch.Tensor]:
    # Chunk large N (e.g. 248k-row lm_head) so FP32 dequant temporaries stay small.
    padded_n = round_up(shape_n, _PAD_N)
    pad_k = round_up(shape_k, _PAD_K) - shape_k
    w_chunks, s_chunks = [], []
    for start in range(0, shape_n, _REQUANT_CHUNK_N):
        end = min(start + _REQUANT_CHUNK_N, shape_n)
        g_slice = (
            global_scale[..., start:end, :]
            if global_scale.shape[-2] > 1
            else global_scale
        )
        deq = dequantize_nvfp4(
            weight[..., start:end, :],
            weight_scale[..., start:end, :],
            g_slice,
            out_dtype=params_dtype,
        )
        pad_n = (padded_n - start) - deq.shape[-2] if end == shape_n else 0
        if pad_n or pad_k:
            deq = torch.nn.functional.pad(deq, (0, pad_k, 0, pad_n))
        quant = schema.quant_tensor(deq, schema, params_dtype)
        del deq
        w_chunks.append(quant["weight"])
        s_chunks.append(quant["weight_scale"])
    if len(w_chunks) == 1:
        return {"weight": w_chunks[0], "weight_scale": s_chunks[0]}
    return {
        "weight": torch.cat(w_chunks, dim=-2),
        "weight_scale": torch.cat(s_chunks, dim=-2),
    }


def _prepare_sublayer(
    layer: torch.nn.Module,
    *,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    global_scale: torch.Tensor,
    shape_n: int,
    shape_k: int,
    params_dtype: torch.dtype,
    use_fp8: bool,
    sublayer_name: str = "",
    num_experts: int | None = None,
    use_deepep_fp8_dispatch: bool = False,
):
    from humming.layer import HummingMethod
    from humming.schema import HummingInputSchema, HummingWeightSchema

    schema = HummingWeightSchema(
        b_dtype="float4e2m1",
        weight_scale_group_size=32 if use_fp8 else 16,
        weight_scale_type="group",
        weight_scale_2_type="none",
    )
    if not use_fp8:
        tensors = {
            "weight": weight.view(torch.int32),
            "weight_scale": (weight_scale.float() * global_scale).to(params_dtype),
        }
    else:
        logger.warning_once(
            "Humming NVFP4 W4A8 requantizes group-16 checkpoint weights to "
            "group-32 at load time and uses dynamic FP8 activations. "
            "This introduces additional quantization error."
        )
        tensors = _requant_nvfp4_to_group32(
            weight,
            weight_scale,
            global_scale,
            shape_n=shape_n,
            shape_k=shape_k,
            params_dtype=params_dtype,
            schema=schema,
        )

    prefix = f"{sublayer_name}_" if sublayer_name else ""
    for stale in ("weight_scale_2", "weight_scale_inv", "input_scale", "input_scale_2"):
        if hasattr(layer, f"{prefix}{stale}"):
            delattr(layer, f"{prefix}{stale}")
    for name, tensor in tensors.items():
        copy_or_rebind_param(layer, f"{prefix}{name}", tensor)

    if use_deepep_fp8_dispatch:
        from sglang.srt.layers.quantization.humming_utils import (
            make_humming_deepep_input_schema,
        )

        input_schema = make_humming_deepep_input_schema(
            sublayer_name,
            shape_k,
            weight_schema=schema,
            param_dtype=params_dtype,
        )
    else:
        # Checkpoint input_scale describes FP4 activations and is not reused.
        input_schema = HummingInputSchema(a_dtype="float8e4m3" if use_fp8 else None)

    HummingMethod.prepare_layer_meta(
        layer=layer,
        shape_n=shape_n,
        shape_k=shape_k,
        weight_schema=schema,
        input_schema=input_schema,
        num_experts=num_experts,
        pad_n_to_multiple=_PAD_N,
        pad_k_to_multiple=_PAD_K,
        has_bias=False,
        torch_dtype=params_dtype,
        sublayer_name=sublayer_name,
        tensors=tensors,
    )
    HummingMethod.transform_humming_layer(
        layer, sublayer_name=sublayer_name, already_padded=use_fp8
    )
    return schema
