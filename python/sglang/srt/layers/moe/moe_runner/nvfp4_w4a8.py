# SPDX-License-Identifier: Apache-2.0
"""Fused MoE runner using packed ModelOpt NVFP4 weights and FP8 activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.quantization.nvfp4_w4a8 import (
    nvfp4_w4a8_grouped_gemm,
    nvfp4_w4a8_moe_block_size,
)
from sglang.srt.layers.moe.moe_runner.base import MoeQuantInfo, register_fused_func

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@dataclass
class Nvfp4W4A8MoeQuantInfo(MoeQuantInfo):
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    w13_weight_scale: torch.Tensor
    w2_weight_scale: torch.Tensor
    w13_weight_global_scale: torch.Tensor
    w2_weight_global_scale: torch.Tensor
    cutlass_metadata: Optional[Nvfp4W4A8CutlassMetadata] = None


@dataclass
class Nvfp4W4A8CutlassMetadata:
    a_strides1: torch.Tensor
    b_strides1: torch.Tensor
    c_strides1: torch.Tensor
    a_strides2: torch.Tensor
    b_strides2: torch.Tensor
    c_strides2: torch.Tensor
    s_strides13: torch.Tensor
    s_strides2: torch.Tensor
    expert_offsets: torch.Tensor
    problem_sizes1: torch.Tensor
    problem_sizes2: torch.Tensor
    group_size: int = 32


def _validate_config(config: MoeRunnerConfig) -> None:
    if config.activation != "silu" or not config.is_gated:
        raise ValueError("nvfp4_w4a8 MoE currently requires a gated SiLU/SwiGLU MLP")
    if config.num_fused_shared_experts not in (None, 0):
        raise ValueError("nvfp4_w4a8 does not support fused shared experts")
    if config.gemm1_alpha is not None or config.gemm1_clamp_limit is not None:
        raise ValueError("nvfp4_w4a8 does not support gemm1 alpha/clamp variants")
    if config.swiglu_limit is not None:
        raise ValueError("nvfp4_w4a8 does not yet support clipped SwiGLU")
    if config.apply_router_weight_on_input and config.top_k != 1:
        raise ValueError("router weight on input is valid only for top-k=1")


def _run_cutlass_sm90(
    dispatch_output: StandardDispatchOutput,
    quant_info: Nvfp4W4A8MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """SM90 fast path: FP8 activations x group-32 INT4 via CUTLASS WGMMA."""
    from sglang.srt.layers.moe.cutlass_w4a8_moe import cutlass_w4a8_moe
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    metadata = quant_info.cutlass_metadata
    assert metadata is not None
    if runner_config.no_combine:
        raise ValueError("SM90 CUTLASS nvfp4_w4a8 does not support no_combine")

    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "SM90 CUTLASS nvfp4_w4a8 keeps BF16 layer boundaries; got "
            f"{hidden_states.dtype}"
        )
    if hidden_states.shape[0] == 0:
        return StandardCombineInput(
            hidden_states=hidden_states.new_empty(hidden_states.shape)
        )
    if (
        quant_info.w13_weight.dtype != torch.int8
        or quant_info.w2_weight.dtype != torch.int8
    ):
        raise TypeError("SM90 CUTLASS nvfp4_w4a8 requires prepared packed INT4 weights")

    output = cutlass_w4a8_moe(
        hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        quant_info.w13_weight_scale,
        quant_info.w2_weight_scale,
        topk_weights,
        topk_ids,
        metadata.a_strides1,
        metadata.b_strides1,
        metadata.c_strides1,
        metadata.a_strides2,
        metadata.b_strides2,
        metadata.c_strides2,
        metadata.s_strides13,
        metadata.s_strides2,
        metadata.expert_offsets,
        metadata.problem_sizes1,
        metadata.problem_sizes2,
        apply_router_weight_on_input=runner_config.apply_router_weight_on_input,
        routed_scaling_factor=runner_config.routed_scaling_factor or 1.0,
        group_size=metadata.group_size,
    )
    return StandardCombineInput(hidden_states=output)


@register_fused_func("none", "nvfp4_w4a8")
def fused_experts_none_to_nvfp4_w4a8(
    dispatch_output: StandardDispatchOutput,
    quant_info: Nvfp4W4A8MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """BF16 -> dynamic FP8 -> NVFP4 GEMM -> SwiGLU -> FP8 -> NVFP4 GEMM."""
    if not isinstance(quant_info, Nvfp4W4A8MoeQuantInfo):
        raise TypeError(
            "nvfp4_w4a8 runner requires Nvfp4W4A8MoeQuantInfo, got "
            f"{type(quant_info).__name__}"
        )
    _validate_config(runner_config)

    if quant_info.cutlass_metadata is not None:
        return _run_cutlass_sm90(dispatch_output, quant_info, runner_config)

    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states = dispatch_output.hidden_states
    topk_weights, topk_ids, _ = dispatch_output.topk_output
    if hidden_states.dtype != torch.bfloat16:
        raise TypeError(
            "nvfp4_w4a8 MoE keeps BF16 layer boundaries; got " f"{hidden_states.dtype}"
        )

    num_tokens, hidden_size = hidden_states.shape
    top_k = topk_ids.shape[1]
    num_experts = quant_info.w13_weight.shape[0]
    if num_tokens == 0:
        shape = (
            (0, top_k, hidden_size) if runner_config.no_combine else (0, hidden_size)
        )
        return StandardCombineInput(hidden_states=hidden_states.new_empty(shape))
    if runner_config.apply_router_weight_on_input:
        hidden_states = hidden_states * topk_weights[:, :1].to(hidden_states.dtype)

    # The same aligned route order is valid for both expert GEMMs.
    routes_per_expert = num_tokens * top_k / num_experts
    block_m = nvfp4_w4a8_moe_block_size(hidden_states, routes_per_expert)
    sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
        topk_ids, block_m, num_experts
    )

    gate_up = nvfp4_w4a8_grouped_gemm(
        hidden_states,
        quant_info.w13_weight,
        quant_info.w13_weight_scale,
        quant_info.w13_weight_global_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k=top_k,
        mul_routed_weight=False,
        block_m=block_m,
    )
    intermediate_size = gate_up.shape[-1] // 2

    # GLM-5.2 TP8 has a 256-wide local expert intermediate.  Fuse SwiGLU and
    # per-route FP8 quantization so GEMM2 consumes FP8 directly, avoiding both
    # a BF16 intermediate allocation and the standalone activation launch.
    supported_fused_groups = (16, 32, 64, 128, 256)
    if intermediate_size in supported_fused_groups:
        from sglang.kernels.ops.quantization.per_token_group_quant import (
            per_token_group_quant,
        )

        down_input, down_scale = per_token_group_quant(
            gate_up.contiguous(),
            group_size=intermediate_size,
            fuse_silu_and_mul=True,
            out_dtype=torch.float8_e4m3fn,
        )
    else:
        from sglang.kernels.ops.activation.activation import silu_and_mul

        down_input = torch.empty(
            (num_tokens * top_k, intermediate_size),
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        silu_and_mul(gate_up, down_input)
        down_scale = None

    # nvfp4_w4a8_grouped_gemm dynamically quantizes down_input per route.  A
    # top_k of one here means its source row is the flattened route id.
    route_output = nvfp4_w4a8_grouped_gemm(
        down_input,
        quant_info.w2_weight,
        quant_info.w2_weight_scale,
        quant_info.w2_weight_global_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        top_k=1,
        mul_routed_weight=(
            not runner_config.apply_router_weight_on_input
            and not runner_config.no_combine
        ),
        block_m=block_m,
        input_scale=down_scale,
        output_dtype=hidden_states.dtype if down_scale is not None else None,
    )
    route_output = route_output.view(num_tokens, top_k, hidden_size)

    if runner_config.no_combine:
        output = route_output
    else:
        output = route_output.sum(dim=1)
    if not runner_config.no_combine and runner_config.routed_scaling_factor not in (
        None,
        1.0,
    ):
        output.mul_(runner_config.routed_scaling_factor)
    return StandardCombineInput(hidden_states=output)


__all__ = [
    "Nvfp4W4A8CutlassMetadata",
    "Nvfp4W4A8MoeQuantInfo",
    "fused_experts_none_to_nvfp4_w4a8",
]
