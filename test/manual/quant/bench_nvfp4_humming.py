"""Benchmark accuracy and latency of NVFP4 Marlin vs Humming (Dense & MoE).

Run:
  python test/manual/quant/bench_nvfp4_humming.py --n 4096 --k 4096
  python test/manual/quant/bench_nvfp4_humming.py --include-a30-w4a16 --moe

Timings include activation quantization, exclude load-time weight conversion,
and use CUDA graphs to avoid Python launch overhead. Humming uses W4A16 on
Ampere (e.g. A30 SM80) and W4A8 on SM89+ (e.g. L40).
"""

from __future__ import annotations

import argparse
from contextlib import ExitStack
from unittest.mock import patch

import torch

from sglang.srt.layers.quantization.dequantization import dequantize_nvfp4
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
)
from sglang.srt.layers.quantization.nvfp4_humming import (
    prepare_moe_nvfp4_layer_for_humming,
)


def _time_cuda_graph(
    fn, warmup: int = 5, graph_iters: int = 20, replays: int = 10
) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(graph_iters):
            fn()
    graph.replay()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    end.synchronize()
    return start.elapsed_time(end) * 1000 / (graph_iters * replays)


def _bench_dense(args) -> None:
    config = ModelOptFp4Config(is_checkpoint_nvfp4_serialized=True, group_size=16)
    with torch.device("cuda"):
        packed = torch.randint(0, 256, (args.n, args.k // 2), dtype=torch.uint8)
        scales = (torch.rand(args.n, args.k // 16) + 0.5).to(torch.float8_e4m3fn)
        global_scale = torch.tensor([0.1])
        inputs = {m: torch.randn(m, args.k, dtype=torch.bfloat16) for m in args.m}
        ref_w = dequantize_nvfp4(packed, scales, global_scale, out_dtype=torch.float32)
        refs = {m: x.float() @ ref_w.T for m, x in inputs.items()}

    modes = [
        ("marlin", Fp4GemmRunnerBackend.MARLIN, None),
        ("humming", Fp4GemmRunnerBackend.HUMMING, None),
    ]
    if args.include_a30_w4a16 and torch.cuda.get_device_capability() >= (8, 9):
        modes.append(("humming_sm80_w4a16", Fp4GemmRunnerBackend.HUMMING, (8, 0)))

    print(
        f"=== Dense NVFP4 GEMM ({torch.cuda.get_device_name()}, N={args.n}, K={args.k}) ==="
    )
    print("mode,act_bits,M,us,tflops,rel_l2_err,cos_sim")
    backend_path = (
        "sglang.srt.layers.quantization.modelopt_quant.get_fp4_gemm_runner_backend"
    )
    cap_path = "sglang.srt.layers.quantization.nvfp4_humming.get_device_capability"
    for label, backend, cap_override in modes:
        with ExitStack() as stack:
            stack.enter_context(patch(backend_path, return_value=backend))
            if cap_override is not None:
                stack.enter_context(patch(cap_path, return_value=cap_override))
            method = ModelOptFp4LinearMethod(config)
            with torch.device("cuda"):
                layer = torch.nn.Module()
                method.create_weights(
                    layer, args.k, [args.n], args.k, args.n, torch.bfloat16
                )
                layer.weight.data.copy_(packed)
                layer.weight_scale.data.copy_(scales)
                layer.weight_scale_2.data.copy_(global_scale)
            method.process_weights_after_loading(layer)
            act_bits = 8 if method.quant_mode == "w4a8" else 16
            for m, x in inputs.items():
                out = method.apply(layer, x).float()
                ref = refs[m]
                rel_err = ((out - ref).norm() / ref.norm()).item()
                cos_sim = torch.nn.functional.cosine_similarity(
                    out.flatten(), ref.flatten(), dim=0
                ).item()
                us = _time_cuda_graph(lambda: method.apply(layer, x))
                tflops = (2.0 * m * args.n * args.k) / (us * 1e6)
                print(
                    f"{label},{act_bits},{m},{us:.2f},{tflops:.2f},{rel_err:.5f},{cos_sim:.6f}",
                    flush=True,
                )


def _bench_moe(args) -> None:
    from humming.config import GemmType

    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
    from sglang.srt.layers.moe.moe_runner.humming import (
        HummingMoeQuantInfo,
        HummingRunnerCore,
        HummingRunnerInput,
    )
    from sglang.srt.layers.moe.moe_runner.marlin import (
        MarlinMoeQuantInfo,
        fused_experts_none_to_marlin,
    )
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardDispatchOutput
    from sglang.srt.layers.moe.topk import StandardTopKOutput
    from sglang.srt.layers.quantization.marlin_utils_fp4 import (
        prepare_moe_nvfp4_layer_for_marlin,
    )

    experts, hidden, inter, top_k = args.experts, args.k, args.moe_inter, 2
    cfg = MoeRunnerConfig(
        num_experts=experts,
        num_local_experts=experts,
        hidden_size=hidden,
        intermediate_size_per_partition=inter,
        top_k=top_k,
        params_dtype=torch.bfloat16,
        routed_scaling_factor=1.0,
    )
    with torch.device("cuda"):
        w13_p = torch.randint(
            0, 256, (experts, 2 * inter, hidden // 2), dtype=torch.uint8
        )
        w2_p = torch.randint(0, 256, (experts, hidden, inter // 2), dtype=torch.uint8)
        w13_s = (torch.rand(experts, 2 * inter, hidden // 16) + 0.5).to(
            torch.float8_e4m3fn
        )
        w2_s = (torch.rand(experts, hidden, inter // 16) + 0.5).to(torch.float8_e4m3fn)
        w13_g = torch.full((experts, 2), 0.05)
        w2_g = torch.full((experts,), 0.05)
        w13_ref = dequantize_nvfp4(
            w13_p, w13_s, w13_g[:, :1, None], out_dtype=torch.float32
        )
        w2_ref = dequantize_nvfp4(
            w2_p, w2_s, w2_g[:, None, None], out_dtype=torch.float32
        )

    def _make_raw_layer():
        layer = torch.nn.Module()
        layer.num_local_experts = experts
        layer.hidden_size = hidden
        layer.intermediate_size_per_partition = inter
        layer.params_dtype = torch.bfloat16
        layer.moe_runner_config = cfg
        layer.quant_config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True, group_size=16
        )
        for k_name, t in (
            ("w13_weight", w13_p.clone()),
            ("w2_weight", w2_p.clone()),
            ("w13_weight_scale", w13_s.clone()),
            ("w2_weight_scale", w2_s.clone()),
            ("w13_weight_scale_2", w13_g.clone()),
            ("w2_weight_scale_2", w2_g.clone()),
        ):
            layer.register_parameter(k_name, torch.nn.Parameter(t, requires_grad=False))
        return layer

    marlin_layer = _make_raw_layer()
    marlin_layer.w13_weight_scale_2 = torch.nn.Parameter(
        w13_g[:, 0].contiguous(), requires_grad=False
    )
    prepare_moe_nvfp4_layer_for_marlin(marlin_layer)
    marlin_qi = MarlinMoeQuantInfo(
        w13_qweight=marlin_layer.w13_weight,
        w2_qweight=marlin_layer.w2_weight,
        w13_scales=marlin_layer.w13_weight_scale,
        w2_scales=marlin_layer.w2_weight_scale,
        w13_g_idx_sort_indices=None,
        w2_g_idx_sort_indices=None,
        weight_bits=4,
        w13_global_scale=marlin_layer.w13_weight_scale_2,
        w2_global_scale=marlin_layer.w2_weight_scale_2,
        expert_map=None,
        global_num_experts=-1,
    )

    humming_modes = [("humming", None)]
    if args.include_a30_w4a16 and torch.cuda.get_device_capability() >= (8, 9):
        humming_modes.append(("humming_sm80_w4a16", (8, 0)))

    humming_layers = {}
    cap_path = "sglang.srt.layers.quantization.nvfp4_humming.get_device_capability"
    for label, cap_override in humming_modes:
        hl = _make_raw_layer()
        with ExitStack() as stack:
            if cap_override is not None:
                stack.enter_context(patch(cap_path, return_value=cap_override))
            prepare_moe_nvfp4_layer_for_humming(hl, group_size=16)
        humming_layers[label] = (hl, HummingRunnerCore(cfg))

    print(
        f"\n=== MoE NVFP4 GEMM ({torch.cuda.get_device_name()}, E={experts}, "
        f"H={hidden}, I={inter}, top_k={top_k}) ==="
    )
    print("mode,act_bits,M,us,rel_l2_err,cos_sim")
    for m in args.m:
        x = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
        ids = torch.stack(
            [
                torch.arange(m, device="cuda") % experts,
                (torch.arange(m, device="cuda") + 1) % experts,
            ],
            dim=1,
        ).int()
        weights = torch.softmax(torch.randn(m, top_k, device="cuda"), dim=-1)
        ref = torch.zeros(m, hidden, device="cuda")
        for s in range(top_k):
            gate, up = (
                torch.bmm(w13_ref[ids[:, s].long()], x.float().unsqueeze(-1))
                .squeeze(-1)
                .chunk(2, -1)
            )
            down = torch.bmm(
                w2_ref[ids[:, s].long()],
                (torch.nn.functional.silu(gate) * up).unsqueeze(-1),
            ).squeeze(-1)
            ref += down * weights[:, s, None]

        router_logits = torch.empty((m, experts), device="cuda", dtype=torch.bfloat16)
        disp = StandardDispatchOutput(
            hidden_states=x,
            hidden_states_scale=None,
            topk_output=StandardTopKOutput(
                topk_weights=weights, topk_ids=ids, router_logits=router_logits
            ),
        )
        m_out = fused_experts_none_to_marlin(disp, marlin_qi, cfg).hidden_states.float()
        m_us = _time_cuda_graph(
            lambda: fused_experts_none_to_marlin(disp, marlin_qi, cfg)
        )
        m_err = ((m_out - ref).norm() / ref.norm()).item()
        m_cos = torch.nn.functional.cosine_similarity(
            m_out.flatten(), ref.flatten(), dim=0
        ).item()
        print(f"marlin,16,{m},{m_us:.2f},{m_err:.5f},{m_cos:.6f}", flush=True)

        for label, (hl, runner) in humming_layers.items():
            act_bits = hl.humming_metas["w13"].a_dtype.num_bits
            h_in = HummingRunnerInput(x, weights, ids, GemmType.INDEXED)
            h_qi = HummingMoeQuantInfo(hl)
            h_out = runner.run(h_in, h_qi, {}).hidden_states.float()
            h_us = _time_cuda_graph(lambda: runner.run(h_in, h_qi, {}))
            h_err = ((h_out - ref).norm() / ref.norm()).item()
            h_cos = torch.nn.functional.cosine_similarity(
                h_out.flatten(), ref.flatten(), dim=0
            ).item()
            print(
                f"{label},{act_bits},{m},{h_us:.2f},{h_err:.5f},{h_cos:.6f}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--k", type=int, default=4096)
    parser.add_argument("--m", type=int, nargs="+", default=[1, 16, 64, 128, 512])
    parser.add_argument("--experts", type=int, default=16)
    parser.add_argument("--moe-inter", type=int, default=1536)
    parser.add_argument("--moe", action="store_true")
    parser.add_argument("--include-a30-w4a16", action="store_true")
    args = parser.parse_args()
    torch.manual_seed(42)
    _bench_dense(args)
    if args.moe:
        _bench_moe(args)


if __name__ == "__main__":
    main()
