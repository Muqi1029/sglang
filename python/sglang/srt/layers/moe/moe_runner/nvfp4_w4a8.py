# SPDX-License-Identifier: Apache-2.0
"""Fused MoE runner using packed ModelOpt NVFP4 weights and FP8 activations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

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


@register_fused_func("none", "nvfp4_w4a8")
def fused_experts_none_to_nvfp4_w4a8(
    dispatch_output: StandardDispatchOutput,
    quant_info: Nvfp4W4A8MoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """BF16 -> dynamic FP8 -> NVFP4 GEMM -> SwiGLU -> FP8 -> NVFP4 GEMM."""
    from sglang.kernels.ops.activation.activation import silu_and_mul
    from sglang.srt.layers.moe.moe_runner.triton_utils.moe_align_block_size import (
        moe_align_block_size,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    if not isinstance(quant_info, Nvfp4W4A8MoeQuantInfo):
        raise TypeError(
            "nvfp4_w4a8 runner requires Nvfp4W4A8MoeQuantInfo, got "
            f"{type(quant_info).__name__}"
        )
    _validate_config(runner_config)

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
    block_m = nvfp4_w4a8_moe_block_size(hidden_states)
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
    )
    intermediate_size = gate_up.shape[-1] // 2
    down_input = torch.empty(
        (num_tokens * top_k, intermediate_size),
        device=hidden_states.device,
        dtype=hidden_states.dtype,
    )
    silu_and_mul(gate_up, down_input)

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


__all__ = ["Nvfp4W4A8MoeQuantInfo", "fused_experts_none_to_nvfp4_w4a8"]
