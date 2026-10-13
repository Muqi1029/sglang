"""CUDA TileLang sparse MLA over group-scaled FP8 NoPE KV rows (GLM-5.3-Flash).

The kernel dequantizes the 528-byte rows written by ``quantize_k_cache_separate``
in shared memory and keeps the attention math in BF16, so on the dequantized
cache it must be as close to FP32 as the BF16 kernel is; the only budget beyond
BF16 rounding is the split-K combine of BF16 partials.
"""

import math
import unittest
from types import SimpleNamespace

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-large")

_CUDA = torch.cuda.is_available() and torch.version.hip is None
# BF16 kernel vs FP32 on identical BF16 KV is ~2e-3; a misapplied group scale,
# a leaked masked row or a wrong split merge is O(1).
_REL_L2_TOL = 6e-3


def _make_cache(num_rows, dim=512, seed=0):
    from sglang.kernels.ops.attention.dsa.dequant_k_cache import dequantize_k_cache
    from sglang.kernels.ops.attention.dsa.quant_k_cache import (
        quantize_k_cache_separate,
    )

    generator = torch.Generator(device="cuda").manual_seed(seed)
    k_nope = torch.randn(num_rows, 1, dim, device="cuda", generator=generator)
    # Per-group magnitudes spanning three decades exercise the per-group scales.
    group_gain = torch.tensor([0.01, 0.5, 2.0, 8.0], device="cuda")[: dim // 128]
    k_nope.view(num_rows, -1, 128).mul_(group_gain[None, :, None])
    packed, _ = quantize_k_cache_separate(k_nope.bfloat16(), None)
    cache = packed.view(torch.float8_e4m3fn)
    return cache, dequantize_k_cache(cache)


def _reference(q, kv_bf16, indices, sm_scale):
    rows = indices[:, 0].long()
    valid = rows >= 0
    selected = kv_bf16[rows.clamp_min(0), 0].float()
    scores = torch.einsum("thd,tkd->thk", q.float(), selected) * sm_scale
    scores.masked_fill_(~valid[:, None, :], float("-inf"))
    probs = torch.softmax(scores, dim=-1).nan_to_num(0.0)
    return torch.einsum("thk,tkd->thd", probs, selected)


def _max_heads_for_device():
    smem_limit = torch.cuda.get_device_properties(0).shared_memory_per_block_optin
    # 64 heads stage a 64x512 BF16 Q tile next to the 64x512 KV tile.
    return 64 if smem_limit >= 200 * 1024 else 16


@unittest.skipUnless(_CUDA, "CUDA TileLang path")
class TestTileLangFp8GroupScaledNope(CustomTestCase):
    def assert_rel_l2(self, actual, expected, tol=_REL_L2_TOL):
        actual = actual.reshape(expected.shape).float()
        self.assertTrue(torch.isfinite(actual).all())
        rel_l2 = ((actual - expected.float()).norm() / expected.float().norm()).item()
        self.assertLess(rel_l2, tol)

    def _run(self, q, cache, indices, sm_scale):
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            tilelang_sparse_fwd,
        )

        return tilelang_sparse_fwd(q, cache, indices, sm_scale, d_v=q.shape[-1])

    def _inputs(self, tokens, heads, topk, num_rows, seed):
        cache, kv_bf16 = _make_cache(num_rows, seed=seed)
        generator = torch.Generator(device="cuda").manual_seed(seed + 1)
        q = torch.randn(
            tokens, heads, 512, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        indices = torch.randint(
            1,
            num_rows,
            (tokens, 1, topk),
            device="cuda",
            dtype=torch.int32,
            generator=generator,
        )
        # KPool tail padding and short prefixes leave -1 holes, including mid-row.
        indices[..., topk - 61 :] = -1
        indices[0, 0, 5:37] = -1
        return q, cache, kv_bf16, indices

    def test_matches_bf16_kernel_and_reference(self):
        """Dequant-then-BF16 math must track the BF16 kernel on identical values."""
        from sglang.kernels.ops.attention.dsa.tilelang_kernel import (
            sparse_attention_fwd_kernel_v1,
        )

        sm_scale = 1.0 / math.sqrt(512)
        # 1-7 tokens split top-k across blocks; 300 tokens run unsplit. Two heads
        # (TP32 of 64) merge splits with a 2-head combine tile.
        cases = ((2, 3), (8, 7), (16, 1), (16, 300), (_max_heads_for_device(), 7))
        for heads, tokens in cases:
            with self.subTest(heads=heads, tokens=tokens):
                q, cache, kv_bf16, indices = self._inputs(
                    tokens=tokens, heads=heads, topk=2112, num_rows=4096, seed=heads
                )
                actual = self._run(q, cache, indices, sm_scale)
                self.assert_rel_l2(actual, _reference(q, kv_bf16, indices, sm_scale))
                # One stage fits ~100 KB parts; staging does not change the math.
                bf16_kernel = sparse_attention_fwd_kernel_v1(
                    heads,
                    512,
                    0,
                    indices.shape[-1],
                    sm_scale=sm_scale,
                    num_stages=1 if heads <= 16 else 2,
                )
                bf16 = bf16_kernel(
                    q.unsqueeze(0), kv_bf16.unsqueeze(0), indices.unsqueeze(0)
                )
                self.assert_rel_l2(actual, bf16.view_as(q))

    def test_masked_slots_and_zero_groups_do_not_leak(self):
        """Masked slots and all-zero groups carry NaN payloads that must never be read."""
        sm_scale = 1.0 / math.sqrt(512)
        q, cache, kv_bf16, indices = self._inputs(
            tokens=3, heads=16, topk=128, num_rows=256, seed=3
        )
        cache_u8 = cache.view(torch.uint8)
        # Slot 0 is the padding slot; fill it with NaN payload bytes.
        cache_u8[0] = 0x7F
        # An all-zero group quantizes to scale 0 and a NaN payload.
        zero_row = int(indices[1, 0, 0])
        cache_u8[zero_row, 0, 128:256] = 0x7F
        cache[zero_row, 0, 512:].view(torch.float32)[1] = 0.0
        kv_bf16[zero_row, 0, 128:256] = 0
        # A split made only of masked slots must not poison the merge.
        indices[2, 0, 64:] = -1

        actual = self._run(q, cache, indices, sm_scale)
        self.assert_rel_l2(actual, _reference(q, kv_bf16, indices, sm_scale))

    def test_cuda_graph_replay_follows_device_updates(self):
        """Captured decode must read fresh indices and rewritten FP8 rows."""
        sm_scale = 1.0 / math.sqrt(512)
        q, cache, kv_bf16, indices = self._inputs(
            tokens=4, heads=16, topk=256, num_rows=1024, seed=5
        )

        def run():
            return self._run(q, cache, indices, sm_scale)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                run()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run()

        new_cache, new_kv = _make_cache(1024, seed=11)
        cache.copy_(new_cache)
        indices.copy_(torch.flip(indices, dims=[-1]))
        graph.replay()
        self.assert_rel_l2(output, _reference(q, new_kv, indices, sm_scale))

    def test_pool_writer_feeds_backend_consumer(self):
        """Rows written by the fp8 DSA pool must be what the backend reads, incl. KPool padding."""
        from sglang.kernels.ops.attention.dsa.dequant_k_cache import (
            dequantize_k_cache,
        )
        from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
        from sglang.srt.mem_cache.memory_pool import MLATokenToKVPool
        from sglang.srt.runtime_context import get_parallel

        pool = MLATokenToKVPool(
            size=512,
            page_size=1,
            dtype=torch.float8_e4m3fn,
            kv_lora_rank=512,
            qk_rope_head_dim=0,
            layer_num=1,
            device="cuda",
            enable_memory_saver=False,
            use_dsa=True,
            override_kv_cache_dim=528,
        )
        generator = torch.Generator(device="cuda").manual_seed(13)
        keys = torch.randn(
            64, 1, 512, device="cuda", dtype=torch.bfloat16, generator=generator
        )
        locs = torch.arange(3, 3 + 2 * 64, 2, device="cuda")
        empty_rope = keys.new_empty((64, 1, 0))
        with get_parallel().override(attn_dcp_size=1, attn_dcp_rank=0):
            pool.set_mla_kv_buffer(SimpleNamespace(layer_id=0), locs, keys, empty_rope)
        cache = pool.get_key_buffer(0)
        self.assertEqual((cache.dtype, cache.shape[-1]), (torch.float8_e4m3fn, 528))

        kv_bf16 = dequantize_k_cache(cache)
        # Group scales keep the stored rows within FP8 rounding of the source keys.
        torch.testing.assert_close(
            kv_bf16[locs].float(), keys.float(), atol=0.07, rtol=0.07
        )
        q = torch.randn(4, 16, 512, device="cuda", dtype=torch.bfloat16)
        # 2048 + 3 KPool tail columns: the backend pads to a 64-column multiple.
        selection = torch.randint(0, 64, (4, 2051), device="cuda", generator=generator)
        page_table = locs[selection].int()
        page_table[:, 2048:] = -1
        sm_scale = 1.0 / math.sqrt(512)
        actual = DeepseekSparseAttnBackend._forward_tilelang(
            None, q, cache, 512, page_table, sm_scale
        )
        self.assert_rel_l2(
            actual, _reference(q, kv_bf16, page_table.unsqueeze(1), sm_scale)
        )

    def test_empty_batch_returns_empty_output(self):
        """An empty scattered attention slice must not launch a zero-sized grid."""
        cache, _ = _make_cache(64)
        q = torch.empty(0, 16, 512, device="cuda", dtype=torch.bfloat16)
        indices = torch.empty(0, 1, 2112, device="cuda", dtype=torch.int32)
        out = self._run(q, cache, indices, 1.0 / math.sqrt(512))
        self.assertEqual(tuple(out.shape), (1, 0, 16, 512))

    def test_rejects_non_nope_or_unscaled_rows(self):
        """A raw 512-byte FP8 row or a 656-byte RoPE row must not be misread."""
        sm_scale = 1.0 / math.sqrt(512)
        q = torch.randn(1, 16, 512, device="cuda", dtype=torch.bfloat16)
        indices = torch.zeros(1, 1, 64, device="cuda", dtype=torch.int32)
        for width in (512, 656):
            with self.subTest(width=width):
                cache = torch.zeros(8, 1, width, device="cuda").to(torch.float8_e4m3fn)
                with self.assertRaisesRegex(ValueError, "group-scaled"):
                    self._run(q, cache, indices, sm_scale)


if __name__ == "__main__":
    unittest.main()
