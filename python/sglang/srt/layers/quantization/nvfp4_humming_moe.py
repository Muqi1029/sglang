from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch
from torch.nn import Module, Parameter

from sglang.srt.environ import envs
from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptNvFp4FusedMoEMethod,
    deinterleave_w13,
)
from sglang.srt.utils import log_info_on_rank0

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import CombineInput, DispatchOutput

logger = logging.getLogger(__name__)

# Humming's WGMMA (SM90) kernels apply group scales once per MMA K-step:
# 256 / a_bits elements -> 32 for 8-bit activations, 16 for 16-bit ones.
# NVFP4 stores one scale per 16 weights, so 8-bit activations need the
# experts regrouped to >= 32 (see ``_requant_group_size``).
_WGMMA_MMA_K_BITS = 256
_REQUANT_EXPERT_CHUNK = 16


def _humming():
    from humming import dtypes
    from humming.config.enum import WeightScale2Type, WeightScaleType
    from humming.layer import HummingMethod
    from humming.schema import HummingInputSchema, HummingWeightSchema
    from humming.schema.modelopt import ModeloptNvfp4WeightSchema

    return (
        dtypes,
        WeightScaleType,
        WeightScale2Type,
        HummingMethod,
        HummingInputSchema,
        HummingWeightSchema,
        ModeloptNvfp4WeightSchema,
    )


def nvfp4_humming_input_schema(layer_name: str):
    """Activation schema for NVFP4 experts on the Humming runner.

    Default is W4A8: FP8-E4M3 activations with dynamic per-token scales, so the
    MoE GEMMs run on the FP8 tensor cores (twice the BF16 rate on Hopper).
    ``SGLANG_HUMMING_INPUT_QUANT_CONFIG`` overrides it; ``{"dtype":
    "bfloat16"}`` selects lossless W4A16 on the native NVFP4 layout.
    """
    dtypes, _, _, _, HummingInputSchema, _, _ = _humming()
    from sglang.srt.layers.quantization.humming_utils import (
        humming_is_layer_skipped,
    )

    input_quant_config = envs.SGLANG_HUMMING_INPUT_QUANT_CONFIG.get() or {}
    if not input_quant_config:
        return HummingInputSchema(a_dtype=dtypes.float8e4m3, input_scale_group_size=0)
    if humming_is_layer_skipped(input_quant_config, layer_name):
        return HummingInputSchema()
    return HummingInputSchema.from_config(input_quant_config)


def _requant_group_size(input_schema, weight_schema) -> int | None:
    """Group size the weights must be regrouped to for this activation dtype."""
    a_bits = input_schema.get_activation_bits()
    if a_bits >= 16:
        return None
    min_group = _WGMMA_MMA_K_BITS // a_bits
    if weight_schema.weight_scale_group_size >= min_group:
        return None
    return min_group


def _requant_experts_chunked(
    tensors: dict[str, torch.Tensor],
    source_schema,
    target_schema,
    param_dtype: torch.dtype,
    num_experts: int,
) -> dict[str, torch.Tensor]:
    """``HummingWeightSchema.requant_tensors`` in expert chunks to bound the
    FP32 dequantization temporaries (a full GLM-5.3-Flash w13 bank would need
    ~10 GB at once)."""
    outputs: dict[str, list[torch.Tensor]] = {}
    for start in range(0, num_experts, _REQUANT_EXPERT_CHUNK):
        end = min(start + _REQUANT_EXPERT_CHUNK, num_experts)
        chunk = {k: v[start:end] for k, v in tensors.items()}
        requant = source_schema.requant_tensors(
            tensors=chunk, target_weight_schema=target_schema, param_dtype=param_dtype
        )
        for k, v in requant.items():
            outputs.setdefault(k, []).append(v)
        del chunk, requant
    return {k: torch.cat(v, dim=0) for k, v in outputs.items()}


def prepare_nvfp4_humming_moe_layer(layer: Module, prefix: str) -> None:
    """Lay out ModelOpt NVFP4 expert weights for the Humming MoE runner.

    ``layer`` holds the raw checkpoint tensors created by
    ``ModelOptNvFp4FusedMoEMethod.create_weights`` (``w13_weight`` packed
    E2M1, ``w13_weight_scale`` E4M3 per-16, ``w13_weight_scale_2`` FP32
    global, ``w13_input_scale``; same for ``w2``). Humming's
    ``ModeloptNvfp4WeightSchema`` converts them; when the activation dtype is
    8-bit the experts are additionally regrouped to 32-wide BF16 scales, which
    is lossy (roughly +25% weight quantization error on Gaussian weights) but
    the only way onto Hopper's FP8 WGMMA path.
    """
    (
        dtypes,
        WeightScaleType,
        _,
        HummingMethod,
        _,
        HummingWeightSchema,
        ModeloptNvfp4WeightSchema,
    ) = _humming()
    from sglang.srt.layers.quantization.humming_utils import (
        configure_humming_deepep_dispatch,
        make_humming_deepep_input_schema,
    )

    checkpoint_schema = ModeloptNvfp4WeightSchema()
    input_schema = nvfp4_humming_input_schema(layer.layer_name)
    use_deepep_fp8_dispatch = configure_humming_deepep_dispatch(layer)

    num_experts = layer.num_local_experts
    param_dtype = layer.params_dtype
    shape_config = {
        "w13": (layer.intermediate_size_per_partition * 2, layer.hidden_size),
        "w2": (layer.hidden_size, layer.intermediate_size_per_partition),
    }
    layer.weight_schemas = {}
    layer.input_schemas = {}
    requant_log = None

    for sublayer_name, (shape_n, shape_k) in shape_config.items():
        sub_input_schema = input_schema
        if use_deepep_fp8_dispatch:
            sub_input_schema = make_humming_deepep_input_schema(sublayer_name, shape_k)

        tensors = {
            "weight": getattr(layer, f"{sublayer_name}_weight").data,
            "weight_scale": getattr(layer, f"{sublayer_name}_weight_scale").data,
            "weight_scale_2": getattr(layer, f"{sublayer_name}_weight_scale_2").data,
        }
        shape_n_stacks = [shape_n // 2] * 2 if sublayer_name == "w13" else [shape_n]
        weight_schema, tensors = checkpoint_schema.convert_humming(
            tensors=tensors,
            shape_n_stacks=shape_n_stacks,
            shape_k_stacks=[shape_k],
            param_dtype=param_dtype,
            num_experts=num_experts,
        )

        group_size = _requant_group_size(sub_input_schema, weight_schema)
        if group_size is not None:
            target_schema = HummingWeightSchema(
                b_dtype=dtypes.float4e2m1,
                bs_dtype=dtypes.DataType.from_torch_dtype(param_dtype),
                weight_scale_group_size=group_size,
                weight_scale_type=WeightScaleType.GROUP,
            )
            tensors = _requant_experts_chunked(
                tensors, weight_schema, target_schema, param_dtype, num_experts
            )
            weight_schema = target_schema
            requant_log = (
                f"regrouped NVFP4 experts to {group_size}-wide "
                f"{param_dtype} scales for {sub_input_schema.a_dtype} activations"
            )

        for name, _ in list(layer.named_parameters()):
            if name.startswith(sublayer_name + "_"):
                delattr(layer, name)
        for name, tensor in tensors.items():
            setattr(
                layer,
                f"{sublayer_name}_{name}",
                Parameter(tensor.contiguous(), requires_grad=False),
            )
        del tensors

        layer.weight_schemas[sublayer_name] = weight_schema
        layer.input_schemas[sublayer_name] = sub_input_schema

        HummingMethod.prepare_layer_meta(
            layer=layer,
            shape_n=shape_n,
            shape_k=shape_k,
            pad_n_to_multiple=256,
            pad_k_to_multiple=128,
            input_schema=sub_input_schema,
            weight_schema=weight_schema,
            has_bias=layer.with_bias,
            num_experts=num_experts,
            torch_dtype=param_dtype,
            sublayer_name=sublayer_name,
        )
        HummingMethod.transform_humming_layer(layer, sublayer_name=sublayer_name)

    if not hasattr(layer, "locks"):
        layer.register_buffer(
            "locks",
            torch.zeros(1024, dtype=torch.int32, device=layer.w13_weight.device),
        )

    a_dtype = layer.input_schemas["w13"].a_dtype
    log_info_on_rank0(
        logger,
        f"NVFP4 experts on Humming ({prefix}): activations={a_dtype or 'bf16'}"
        + (f"; {requant_log}" if requant_log else "; native NVFP4 layout"),
    )


class ModelOptNvFp4HummingMoEMethod(ModelOptNvFp4FusedMoEMethod):
    """ModelOpt NVFP4 MoE weights executed by the Humming grouped-GEMM runner.

    Selected by ``--moe-runner-backend humming`` for NVFP4 MoE layers
    (``modelopt_fp4`` checkpoints and the FP8 + NVFP4 hybrid
    ``HybridFp8NvFp4Config``). Subclassing keeps ``FusedMoE.weight_loader``'s
    ModelOpt branch (``weight_scale_2`` / ``input_scale`` per tensor,
    ``weight_scale`` per block) and the parent's ``create_weights``; only the
    post-load layout and ``apply`` are replaced. On Hopper this gives W4A8
    (FP4 weights dequantized in-kernel, FP8 activations) by default; see
    ``nvfp4_humming_input_schema``.
    """

    def __init__(self, quant_config: ModelOptFp4Config, prefix: str = ""):
        super().__init__(quant_config)
        self.prefix = prefix

    def create_moe_runner(self, layer: Module, moe_runner_config):
        from sglang.srt.layers.moe.moe_runner import MoeRunner

        self.moe_runner_config = moe_runner_config
        self._moe_runner_backend = MoeRunnerBackend.HUMMING
        self.runner = MoeRunner(MoeRunnerBackend.HUMMING, moe_runner_config)

    def process_weights_after_loading(self, layer: Module) -> None:
        if getattr(layer, "_nvfp4_humming_prepared", False):
            return

        if getattr(layer, "inference_moe_w13_interleaved", False) and not getattr(
            layer, "_w13_deinterleaved", False
        ):
            layer.w13_weight.data = deinterleave_w13(layer.w13_weight.data)
            layer.w13_weight_scale.data = deinterleave_w13(layer.w13_weight_scale.data)
            layer._w13_deinterleaved = True

        # The parent's FlashInfer-only swizzled scale copies are never read here.
        for name in ("w13_blockscale_swizzled", "w2_blockscale_swizzled"):
            if hasattr(layer, name):
                delattr(layer, name)

        prepare_nvfp4_humming_moe_layer(layer, self.prefix)
        layer._nvfp4_humming_prepared = True

    def apply(
        self,
        layer: Module,
        dispatch_output: DispatchOutput,
    ) -> CombineInput:
        from sglang.srt.layers.moe.moe_runner.humming import HummingMoeQuantInfo

        quant_info = HummingMoeQuantInfo(layer=layer)
        return self.runner.run(dispatch_output, quant_info)
