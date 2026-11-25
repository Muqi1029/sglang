"""Unit coverage for the Humming FP8 backend and MoE SiLU+FP8 fusion."""

import unittest
from unittest.mock import patch

import pytest
import torch

pytest.importorskip("humming")

from humming import dtypes  # noqa: E402
from humming.config import GemmType  # noqa: E402

from sglang.srt.layers.linear import MergedColumnParallelLinear  # noqa: E402
from sglang.srt.layers.logits_processor import (  # noqa: E402
    should_apply_lm_head_quant_method,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE  # noqa: E402
from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig  # noqa: E402
from sglang.srt.layers.moe.moe_runner.humming import (  # noqa: E402
    HummingMoeQuantInfo,
    HummingRunnerCore,
    HummingRunnerInput,
)
from sglang.srt.layers.moe.utils import MoeA2ABackend, MoeRunnerBackend  # noqa: E402
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend  # noqa: E402
from sglang.srt.layers.quantization.fp8_utils import (  # noqa: E402
    Fp8GemmRunnerBackend,
    _dispatch_explicit_backend,
)
from sglang.srt.layers.quantization.modelopt_quant import (  # noqa: E402
    ModelOptFp4Config,
    ModelOptFp8Config,
    ModelOptFp8LinearMethod,
)
from sglang.srt.layers.quantization.nvfp4_humming import (  # noqa: E402
    prepare_fp8_layer_for_humming,
    prepare_moe_nvfp4_layer_for_humming,
)
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead  # noqa: E402
from sglang.srt.runtime_context import (  # noqa: E402
    get_context,
    get_flags,
    get_parallel,
)
from sglang.test.ci.ci_register import register_cuda_ci  # noqa: E402
from sglang.test.test_utils import CustomTestCase  # noqa: E402

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")


def _fill_nvfp4_linear(layer: torch.nn.Module) -> None:
    layer.weight.data = torch.randint(
        0, 256, layer.weight.shape, dtype=torch.uint8, device=layer.weight.device
    )
    layer.weight.data[..., ::8] = 0x77
    layer.weight_scale.data = torch.full(
        layer.weight_scale.shape,
        0.5,
        dtype=layer.weight_scale.dtype,
        device=layer.weight_scale.device,
    )
    layer.weight_scale_2.data = torch.tensor(
        [0.0625] * len(layer.weight_scale_2.data),
        dtype=torch.float32,
        device=layer.weight_scale_2.device,
    )
    layer.input_scale.data = torch.ones_like(layer.input_scale.data)


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9),
    "Humming FP8 / W4A8 SM89+ optimizations require SM89+",
)
class TestHummingFp8AndFusion(CustomTestCase):
    def test_fp8_humming_backend_static_channel_dynamic_and_lm_head(self):
        torch.manual_seed(0)
        cfg = ModelOptFp8Config(
            is_checkpoint_fp8_serialized=True,
            packed_modules_mapping={},
            exclude_modules=[],
        )
        fp8_backend_path = (
            "sglang.srt.layers.quantization.modelopt_quant.get_fp8_gemm_runner_backend"
        )
        with patch(fp8_backend_path, return_value=Fp8GemmRunnerBackend.HUMMING):
            # 1. Static FP8 with unaligned K, bias, and distinct shard scales
            #    (kept as channel scales, not requantized to the max scale).
            with torch.device("cuda"):
                layer = MergedColumnParallelLinear(
                    input_size=258,
                    output_sizes=[128, 130],
                    bias=True,
                    params_dtype=torch.bfloat16,
                    quant_config=cfg,
                    parallel_group="replicated",
                )
                self.assertIsInstance(layer.quant_method, ModelOptFp8LinearMethod)
                self.assertTrue(layer.quant_method.use_humming)
                raw_w = (torch.randn(258, 258, device="cuda") * 4.0).to(
                    torch.float8_e4m3fn
                )
                w_scale = torch.tensor([0.1, 0.25], dtype=torch.float32, device="cuda")
                in_scale = torch.tensor([0.5, 0.5], dtype=torch.float32, device="cuda")
                layer.weight.data.copy_(raw_w)
                layer.weight_scale.data.copy_(w_scale)
                layer.input_scale.data.copy_(in_scale)
                layer.bias.data = torch.randn(258, dtype=torch.bfloat16, device="cuda")
                layer.quant_method.process_weights_after_loading(layer)

                x = torch.randn(4, 258, dtype=torch.bfloat16, device="cuda")
                out = layer.quant_method.apply(layer, x, bias=layer.bias)
                x_q = (x.float() / 0.5).to(torch.float8_e4m3fn).float() * 0.5
                w_deq = torch.cat(
                    [
                        raw_w[:128].float() * 0.1,
                        raw_w[128:].float() * 0.25,
                    ],
                    dim=0,
                )
                ref = (x_q @ w_deq.t()).to(torch.bfloat16) + layer.bias
                rel = (out.float() - ref.float()).norm() / ref.float().norm()
                self.assertLess(rel.item(), 0.01)

            # 2. Channel scale & dynamic_token (no input_scale) on raw Module.
            dyn_mod = torch.nn.Module()
            dyn_mod.output_size_per_partition = 192
            dyn_mod.input_size_per_partition = 256
            dyn_mod.orig_dtype = torch.bfloat16
            raw_w2 = (torch.randn(192, 256, device="cuda") * 2.0).to(
                torch.float8_e4m3fn
            )
            ch_scale = torch.rand(192, dtype=torch.bfloat16, device="cuda") * 0.1 + 0.05
            dyn_mod.weight = torch.nn.Parameter(raw_w2, requires_grad=False)
            dyn_mod.weight_scale = torch.nn.Parameter(ch_scale, requires_grad=False)
            prepare_fp8_layer_for_humming(dyn_mod)
            out_dyn = layer.quant_method.apply(
                dyn_mod, torch.randn(3, 256, dtype=torch.bfloat16, device="cuda")
            )
            self.assertEqual(out_dyn.shape, (3, 192))

            # 3. Quantized lm_head detection for Humming ModelOptFp8LinearMethod.
            with torch.device("cuda"):
                lm_head = ParallelLMHead(
                    num_embeddings=512,
                    embedding_dim=256,
                    params_dtype=torch.bfloat16,
                    quant_config=cfg,
                    enable_tp=False,
                )
                lm_head.weight.data = (torch.randn(512, 256, device="cuda") * 2.0).to(
                    torch.float8_e4m3fn
                )
                lm_head.weight_scale.data.fill_(0.125)
                lm_head.input_scale.data.fill_(0.5)
                lm_head.quant_method.process_weights_after_loading(lm_head)
                self.assertTrue(
                    should_apply_lm_head_quant_method(lm_head, lm_head.quant_method)
                )

        # 4. Blockwise FP8 rejects explicit --fp8-gemm-backend humming loudly.
        with self.assertRaises(RuntimeError):
            _dispatch_explicit_backend(Fp8GemmRunnerBackend.HUMMING)

    def test_nvfp4_lm_head_excluded_from_f16_accum(self):
        fp4_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            packed_modules_mapping={},
            exclude_modules=[],
        )
        backend_path = (
            "sglang.srt.layers.quantization.modelopt_quant.get_fp4_gemm_runner_backend"
        )
        with (
            patch(backend_path, return_value=Fp4GemmRunnerBackend.HUMMING),
            patch.dict("os.environ", {"SGLANG_HUMMING_USE_F16_ACCUM": "1"}),
            torch.device("cuda"),
        ):
            lm_head = ParallelLMHead(
                num_embeddings=512,
                embedding_dim=256,
                params_dtype=torch.bfloat16,
                quant_config=fp4_cfg,
                enable_tp=False,
            )
            _fill_nvfp4_linear(lm_head)
            lm_head.quant_method.process_weights_after_loading(lm_head)
            self.assertIsNone(getattr(lm_head, "_humming_output_dtype", None))
            self.assertEqual(lm_head.humming_metas[""].c_dtype, dtypes.bfloat16)
            x = torch.randn(6, 256, dtype=torch.bfloat16)
            logits = lm_head.quant_method.apply(lm_head, x)
            self.assertEqual(logits.dtype, torch.bfloat16)
            self.assertEqual(logits.shape, (6, 512))

    def test_moe_runner_silu_fp8_fusion_non_aliasing_and_f16_accum(self):
        torch.manual_seed(11)
        experts, hidden, intermediate = 4, 256, 256
        config = MoeRunnerConfig(
            num_experts=experts,
            num_local_experts=experts,
            hidden_size=hidden,
            intermediate_size_per_partition=intermediate,
            top_k=2,
            params_dtype=torch.bfloat16,
            routed_scaling_factor=1.5,
        )
        for use_f16_accum in (False, True):
            with (
                self.subTest(use_f16_accum=use_f16_accum),
                patch.dict(
                    "os.environ",
                    {"SGLANG_HUMMING_USE_F16_ACCUM": "1" if use_f16_accum else "0"},
                ),
            ):
                layer = torch.nn.Module()
                layer.num_local_experts = experts
                layer.hidden_size = hidden
                layer.intermediate_size_per_partition = intermediate
                layer.params_dtype = torch.bfloat16
                layer.moe_runner_config = config
                for name, n, k in (
                    ("w13", intermediate * 2, hidden),
                    ("w2", hidden, intermediate),
                ):
                    packed = torch.randint(
                        0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
                    )
                    packed[..., ::8] = 0x77
                    scale = torch.ones(
                        experts, n, k // 16, device="cuda", dtype=torch.float8_e4m3fn
                    )
                    scale_2 = torch.full(
                        (experts,), 0.0625, device="cuda", dtype=torch.float32
                    )
                    in_scale = torch.ones(experts, device="cuda", dtype=torch.float32)
                    for suffix, tensor in (
                        ("weight", packed),
                        ("weight_scale", scale),
                        ("weight_scale_2", scale_2),
                        ("input_scale", in_scale),
                    ):
                        layer.register_parameter(
                            f"{name}_{suffix}",
                            torch.nn.Parameter(tensor, requires_grad=False),
                        )
                prepare_moe_nvfp4_layer_for_humming(layer, group_size=16)
                expected_c_dtype = dtypes.float16 if use_f16_accum else dtypes.bfloat16
                self.assertEqual(layer.humming_metas["w13"].c_dtype, expected_c_dtype)

                runner = HummingRunnerCore(config)
                runner.layer = layer
                self.assertTrue(runner._can_fuse_silu_and_mul_w2())

                x = torch.randn(8, hidden, device="cuda", dtype=torch.bfloat16)
                ids = torch.stack(
                    [
                        torch.arange(8, device="cuda") % experts,
                        (torch.arange(8, device="cuda") + 1) % experts,
                    ],
                    dim=1,
                ).int()
                weights = torch.softmax(torch.randn(8, 2, device="cuda"), -1)

                for gemm_type in (GemmType.INDEXED, GemmType.GROUPED_CONTIGUOUS):
                    _, required = runner.get_buffer_metas(x, ids, gemm_type)
                    self.assertNotIn("activation_output", required)
                    bufs = runner.prepare_buffers(x, ids, gemm_type)
                    # gate_up_output and quanted_down_input must never alias.
                    self.assertNotEqual(
                        bufs["gate_up_output"].data_ptr(),
                        bufs["quanted_down_input"].data_ptr(),
                    )
                    out_fused = runner.run(
                        HummingRunnerInput(x, weights, ids, gemm_type),
                        HummingMoeQuantInfo(layer),
                        {},
                    ).hidden_states
                    self.assertEqual(out_fused.dtype, torch.bfloat16)

                    # Compare against unfused W2 path.
                    with patch.object(
                        runner, "_can_fuse_silu_and_mul_w2", return_value=False
                    ):
                        out_unfused = runner.run(
                            HummingRunnerInput(x, weights, ids, gemm_type),
                            HummingMoeQuantInfo(layer),
                            {},
                        ).hidden_states
                    rel = (out_fused.float() - out_unfused.float()).norm() / (
                        out_unfused.float().norm() + 1e-6
                    )
                    self.assertLess(rel.item(), 0.05)

    def test_fused_moe_humming_sets_inplace_false(self):
        fp4_cfg = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            packed_modules_mapping={},
            exclude_modules=[],
        )
        with (
            get_context().override_server_args(model_path="dummy"),
            get_flags().moe.override(
                runner_backend=MoeRunnerBackend.HUMMING,
                a2a_backend=MoeA2ABackend.NONE,
            ),
            patch(
                "sglang.srt.layers.quantization.modelopt_quant.get_fp4_gemm_runner_backend",
                return_value=Fp4GemmRunnerBackend.HUMMING,
            ),
            get_parallel().override(
                tp_group=None,
                moe_ep_size=1,
                moe_ep_rank=0,
                moe_tp_size=1,
                moe_tp_rank=0,
                tp_size=1,
                tp_rank=0,
                attn_tp_size=1,
                attn_tp_rank=0,
            ),
            torch.device("cuda"),
        ):
            moe = FusedMoE(
                num_experts=4,
                top_k=2,
                hidden_size=256,
                intermediate_size=256,
                layer_id=0,
                params_dtype=torch.bfloat16,
                reduce_results=False,
                quant_config=fp4_cfg,
                inplace=True,
            )
            self.assertTrue(moe.runner_backend.is_humming())
            self.assertFalse(moe.moe_runner_config.inplace)


if __name__ == "__main__":
    unittest.main()
