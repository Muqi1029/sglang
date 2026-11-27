"""Packed CP decode partitions must match independently loaded TP weights."""

import json
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.environ import envs
from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpContext
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.srt.layers.quantization.humming import HummingConfig
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-large")


@pytest.fixture(autouse=True)
def runtime_context():
    with get_context().override_server_args():
        yield


@pytest.mark.parametrize("rank", range(4))
@pytest.mark.parametrize("input_dtype", ["float8e4m3", "bfloat16"])
@pytest.mark.parametrize("rows", [1, 8])
@pytest.mark.parametrize("column", [True, False])
@torch.no_grad()
def test_packed_decode_partition_matches_tp(column, rows, rank, input_dtype):
    pytest.importorskip("humming")
    torch.manual_seed(712)
    n, k = (32768, 1280) if column else (5120, 8192)
    cls = ColumnParallelLinear if column else RowParallelLinear
    extra = {} if column else {"reduce_results": False}
    with (
        get_context().override_server_args(),
        envs.SGLANG_HUMMING_INPUT_QUANT_CONFIG.override(
            json.dumps({"dtype": input_dtype})
        ),
        torch.device("cuda"),
    ):
        config = HummingConfig(
            {
                "quant_method": "fp8",
                "weight_block_size": [32, 32],
                "activation_scheme": "dynamic",
            }
        )
        with get_parallel().override(
            tp_size=1,
            tp_rank=0,
            tp_group=SimpleNamespace(world_size=1, rank_in_group=0),
        ):
            full = cls(
                k,
                n,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=config,
                parallel_group="tp",
                **extra,
            )
        with get_parallel().override(
            tp_size=4,
            attn_tp_size=4,
            moe_tp_size=4,
            tp_rank=rank,
            tp_group=SimpleNamespace(world_size=4, rank_in_group=rank),
        ):
            ref = cls(
                k,
                n,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=config,
                parallel_group="tp",
                **extra,
            )
        weight = torch.randn(n, k).to(torch.float8_e4m3fn)
        scale = torch.rand(n // 32, k // 32) * 0.01
        full.weight.copy_(weight)
        full.weight_scale_inv.copy_(scale)
        dim = 0 if column else 1
        ref.weight.copy_(weight.chunk(4, dim=dim)[rank])
        ref.weight_scale_inv.copy_(scale.chunk(4, dim=dim)[rank])
        full.quant_method.process_weights_after_loading(full)
        ref.quant_method.process_weights_after_loading(ref)
        # Compare shards with deterministic kernels; the library's default
        # W8A16 stream-K reduction varies even for identical packed operands.
        for layer in (full, ref):
            config = json.loads(layer.quant_method.compute_config)
            config["use_batch_invariant"] = True
            layer.quant_method.compute_config = json.dumps(config)
        x = torch.randn(rows, k if column else k // 4, dtype=torch.bfloat16)

    originals = (full.weight, full.weight_scale, full.humming_metas)
    size_attr = "output_size_per_partition" if column else "input_size_per_partition"
    original_size = getattr(full, size_attr)
    with get_parallel().override(
        enable_cp_decode_attn_tp=True,
        tp_size=4,
        moe_tp_size=4,
        attn_tp_size=1,
        attn_cp_size=4,
        attn_cp_rank=rank,
    ):
        ctx = CpDecodeAttnTpContext()
    with (
        patch(
            "sglang.srt.layers.cp.cp_decode_attn_tp.is_cp_active", return_value=False
        ),
        patch(
            "sglang.srt.layers.cp.cp_decode_attn_tp.dsa_use_prefill_cp",
            return_value=False,
        ),
    ):

        def forward():
            with ctx.maybe_use_decode_attn_tp(SimpleNamespace(), [full]):
                return full.quant_method.apply(full, x)

        with ctx.maybe_use_decode_attn_tp(SimpleNamespace(), [full]):
            # Exact packed bytes catch incorrect N/K dimensions or tile order.
            torch.testing.assert_close(full.weight, ref.weight, rtol=0, atol=0)
            torch.testing.assert_close(
                full.weight_scale, ref.weight_scale, rtol=0, atol=0
            )
            assert full.humming_metas[""] == ref.humming_metas[""]
            assert getattr(full, size_attr) == getattr(ref, size_attr)
            if not column:
                assert full.use_decode_attn_tp
            actual = full.quant_method.apply(full, x)
        expected = ref.quant_method.apply(ref, x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for _ in range(3):
            forward()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            replay_out = forward()
        x.neg_()
        graph.replay()
        torch.testing.assert_close(
            replay_out, ref.quant_method.apply(ref, x), rtol=0, atol=0
        )

        with pytest.raises(RuntimeError, match="restore"):
            with ctx.maybe_use_decode_attn_tp(SimpleNamespace(), [full]):
                raise RuntimeError("restore")
        assert full.weight is originals[0]
        assert full.weight_scale is originals[1]
        assert full.humming_metas is originals[2]
        assert getattr(full, size_attr) == original_size
        if not column:
            assert not full.use_decode_attn_tp


@pytest.mark.parametrize("column", [True, False])
@pytest.mark.parametrize("rank", [0, 3])
@torch.no_grad()
def test_packed_decode_partition_block128_matches_tp(column, rank):
    pytest.importorskip("humming")
    torch.manual_seed(713)
    n, k = (32768, 1280) if column else (5120, 8192)
    cls = ColumnParallelLinear if column else RowParallelLinear
    extra = {} if column else {"reduce_results": False}
    with (
        get_context().override_server_args(),
        envs.SGLANG_HUMMING_INPUT_QUANT_CONFIG.override(
            json.dumps({"dtype": "float8e4m3"})
        ),
        torch.device("cuda"),
    ):
        config = HummingConfig(
            {
                "quant_method": "fp8",
                "weight_block_size": [128, 128],
                "activation_scheme": "dynamic",
            }
        )
        with get_parallel().override(
            tp_size=1,
            tp_rank=0,
            tp_group=SimpleNamespace(world_size=1, rank_in_group=0),
        ):
            full = cls(
                k,
                n,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=config,
                parallel_group="tp",
                **extra,
            )
        with get_parallel().override(
            tp_size=4,
            attn_tp_size=4,
            moe_tp_size=4,
            tp_rank=rank,
            tp_group=SimpleNamespace(world_size=4, rank_in_group=rank),
        ):
            ref = cls(
                k,
                n,
                bias=False,
                params_dtype=torch.bfloat16,
                quant_config=config,
                parallel_group="tp",
                **extra,
            )
        weight = torch.randn(n, k).to(torch.float8_e4m3fn)
        scale = torch.rand(n // 128, k // 128) * 0.01
        full.weight.copy_(weight)
        full.weight_scale_inv.copy_(scale)
        dim = 0 if column else 1
        ref.weight.copy_(weight.chunk(4, dim=dim)[rank])
        ref.weight_scale_inv.copy_(scale.chunk(4, dim=dim)[rank])
        full.quant_method.process_weights_after_loading(full)
        ref.quant_method.process_weights_after_loading(ref)
        for layer in (full, ref):
            cfg = json.loads(layer.quant_method.compute_config)
            cfg["use_batch_invariant"] = True
            layer.quant_method.compute_config = json.dumps(cfg)
        x = torch.randn(4, k if column else k // 4, dtype=torch.bfloat16)

    with get_parallel().override(
        enable_cp_decode_attn_tp=True,
        tp_size=4,
        moe_tp_size=4,
        attn_tp_size=1,
        attn_cp_size=4,
        attn_cp_rank=rank,
    ):
        ctx = CpDecodeAttnTpContext()
    with (
        patch(
            "sglang.srt.layers.cp.cp_decode_attn_tp.is_cp_active", return_value=False
        ),
        patch(
            "sglang.srt.layers.cp.cp_decode_attn_tp.dsa_use_prefill_cp",
            return_value=False,
        ),
    ):
        with ctx.maybe_use_decode_attn_tp(SimpleNamespace(), [full]):
            torch.testing.assert_close(full.weight, ref.weight, rtol=0, atol=0)
            torch.testing.assert_close(
                full.weight_scale, ref.weight_scale, rtol=0, atol=0
            )
            assert full.humming_metas[""] == ref.humming_metas[""]
            actual = full.quant_method.apply(full, x)
        expected = ref.quant_method.apply(ref, x)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
