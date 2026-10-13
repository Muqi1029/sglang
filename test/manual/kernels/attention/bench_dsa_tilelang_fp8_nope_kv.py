"""Accuracy and latency of NoPE sparse MLA decode under four KV-cache designs.

    bf16        BF16 KV rows, BF16 TileLang kernel (current CUDA default).
    dequant     group-scaled FP8 rows, dequantized into a BF16 workspace, then
                the BF16 kernel (the approach of sgl-project/sglang#39349).
    raw_fp8     unscaled FP8 rows and FP8 queries through the partial+combine
                TileLang kernel (the approach of sgl-project/sglang#36904).
    fused       group-scaled FP8 rows dequantized inside the kernel (this branch).

Accuracy is measured against an FP32 reference on the unquantized keys.

    python3 test/manual/kernels/attention/bench_dsa_tilelang_fp8_nope_kv.py
"""

import argparse
import math

import torch
import triton

from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
    dequantize_k_cache,
    dequantize_k_cache_paged,
)
from sglang.kernels.ops.attention.dsa.quant_k_cache import quantize_k_cache_separate
from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
    _pick_inner_iter,
    sparse_attention_fwd_kernel_v1,
    sparse_mla_fwd_decode_combine,
    sparse_mla_fwd_decode_partial_fp8,
    tilelang_sparse_fwd,
)

DIM = 512


def make_keys(num_rows, distribution, generator):
    keys = torch.randn(num_rows, 1, DIM, device="cuda", generator=generator)
    if distribution == "outlier_channels":
        channel_scale = torch.empty(DIM, device="cuda").log_normal_(
            0.0, 1.0, generator=generator
        )
        keys = keys * channel_scale
    elif distribution == "small_groups":
        keys.view(num_rows, 4, 128).mul_(
            torch.tensor([0.01, 0.1, 1.0, 4.0], device="cuda")[None, :, None]
        )
    return keys.bfloat16()


def dequant_workspace(cache, indices):
    # Mirrors dequantize_sparse_nope_cache in #39349.
    if indices.numel() >= cache.shape[0]:
        return dequantize_k_cache(cache), indices
    valid = indices >= 0
    kv = dequantize_k_cache_paged(cache, torch.where(valid, indices, 0).flatten())
    remapped = torch.arange(
        indices.numel(), device=indices.device, dtype=indices.dtype
    ).view_as(indices)
    return kv, torch.where(valid, remapped, -1)


class Designs:
    def __init__(self, heads, topk, sm_scale, num_stages):
        self.sm_scale = sm_scale
        self.bf16_kernel = sparse_attention_fwd_kernel_v1(
            heads, DIM, 0, topk, sm_scale=sm_scale, num_stages=num_stages
        )
        self.heads = heads
        self.topk = topk
        self.sm_count = torch.cuda.get_device_properties(0).multi_processor_count

    def bf16(self, q, kv_bf16, indices):
        return self.bf16_kernel(
            q.unsqueeze(0), kv_bf16.unsqueeze(0), indices.unsqueeze(0)
        )

    def dequant(self, q, cache, indices):
        kv, remapped = dequant_workspace(cache, indices)
        return self.bf16_kernel(q.unsqueeze(0), kv.unsqueeze(0), remapped.unsqueeze(0))

    def raw_fp8(self, q, kv_raw, indices):
        block_I, threads = 32, 128
        ni = self.topk // block_I
        inner_iter = _pick_inner_iter(q.shape[0], ni, self.sm_count, 1)
        partial = sparse_mla_fwd_decode_partial_fp8(
            self.heads,
            DIM,
            0,
            self.topk,
            sm_scale=self.sm_scale,
            block_I=block_I,
            inner_iter=inner_iter,
            threads=threads,
        )
        combine = sparse_mla_fwd_decode_combine(
            self.heads,
            DIM,
            (ni // inner_iter) * block_I,
            head_per_block=4,
            block_I=block_I,
            threads=threads,
        )
        o, lse = partial(
            q.to(torch.float8_e4m3fn).unsqueeze(0),
            kv_raw.unsqueeze(0),
            indices.unsqueeze(0),
        )
        return combine(o, lse)

    def fused(self, q, cache, indices):
        return tilelang_sparse_fwd(q, cache, indices, self.sm_scale, d_v=DIM)


def capture_graph(fn):
    # Decode replays CUDA graphs, so time kernels without per-call host work.
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(2):
            fn()
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        fn()
    return graph


def reference(q, keys, indices, sm_scale):
    rows = indices[:, 0].long()
    valid = rows >= 0
    selected = keys[rows.clamp_min(0), 0].float()
    scores = torch.einsum("thd,tkd->thk", q.float(), selected) * sm_scale
    scores.masked_fill_(~valid[:, None, :], float("-inf"))
    return torch.einsum("thk,tkd->thd", scores.softmax(-1), selected)


def errors(actual, expected):
    actual = actual.reshape(expected.shape).float()
    rel_l2 = ((actual - expected).norm() / expected.norm()).item()
    cos = torch.nn.functional.cosine_similarity(
        actual.flatten(1), expected.flatten(1), dim=-1
    )
    return rel_l2, (actual - expected).abs().max().item(), cos.min().item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--topk", type=int, default=2112)
    parser.add_argument("--pool", type=int, default=262144)
    parser.add_argument(
        "--batches", type=int, nargs="+", default=[1, 16, 64, 128, 512, 2048]
    )
    parser.add_argument("--skip-raw", action="store_true")
    args = parser.parse_args()

    smem_limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    num_stages = 2 if smem_limit >= 200 * 1024 else 1
    sm_scale = 1.0 / math.sqrt(256)
    designs = Designs(args.heads, args.topk, sm_scale, num_stages)
    names = ["bf16", "dequant", "fused"] + ([] if args.skip_raw else ["raw_fp8"])
    print(
        f"{torch.cuda.get_device_name(0)}  heads={args.heads} topk={args.topk} "
        f"pool={args.pool} bf16 num_stages={num_stages}"
    )

    generator = torch.Generator(device="cuda").manual_seed(0)
    print("\n== accuracy vs FP32 on unquantized keys (batch 32) ==")
    print(f"{'keys':<18}{'design':<10}{'rel_l2':>10}{'max_abs':>10}{'min_cos':>10}")
    for distribution in ("gaussian", "outlier_channels", "small_groups"):
        keys = make_keys(8192, distribution, generator)
        cache = quantize_k_cache_separate(keys, None)[0].view(torch.float8_e4m3fn)
        q = torch.randn(32, args.heads, DIM, device="cuda", generator=generator)
        q = q.bfloat16()
        indices = torch.randint(
            1, 8192, (32, 1, args.topk), device="cuda", generator=generator
        ).int()
        indices[..., 2051:] = -1
        expected = reference(q, keys, indices, sm_scale)
        inputs = {
            "bf16": keys,
            "dequant": cache,
            "fused": cache,
            "raw_fp8": keys.to(torch.float8_e4m3fn),
        }
        for name in names:
            out = getattr(designs, name)(q, inputs[name], indices)
            rel_l2, max_abs, min_cos = errors(out, expected)
            print(
                f"{distribution:<18}{name:<10}{rel_l2:>10.2e}{max_abs:>10.2e}"
                f"{min_cos:>10.6f}"
            )

    print(f"\n== CUDA-graph replay latency (us), pool {args.pool} rows, random rows ==")
    keys = make_keys(args.pool, "gaussian", generator)
    cache = quantize_k_cache_separate(keys, None)[0].view(torch.float8_e4m3fn)
    inputs = {
        "bf16": keys,
        "dequant": cache,
        "fused": cache,
        "raw_fp8": keys.to(torch.float8_e4m3fn),
    }
    print(f"{'batch':<8}" + "".join(f"{n:>10}" for n in names) + f"{'fused/bf16':>12}")
    for batch in args.batches:
        q = torch.randn(batch, args.heads, DIM, device="cuda").bfloat16()
        indices = torch.randint(
            1, args.pool, (batch, 1, args.topk), device="cuda"
        ).int()
        indices[..., 2051:] = -1
        times = {}
        for name in names:
            graph = capture_graph(
                lambda: getattr(designs, name)(q, inputs[name], indices)
            )
            times[name] = 1e3 * triton.testing.do_bench(
                graph.replay, warmup=50, rep=300
            )
        print(
            f"{batch:<8}"
            + "".join(f"{times[n]:>10.1f}" for n in names)
            + f"{times['fused'] / times['bf16']:>12.2f}"
        )


if __name__ == "__main__":
    main()
