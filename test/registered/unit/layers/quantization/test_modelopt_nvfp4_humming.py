"""Check architecture dispatch and W4A8/W4A16 numerics against FP32 GEMM."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.linear import MergedColumnParallelLinear
from sglang.srt.layers.logits_processor import should_apply_lm_head_quant_method
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.srt.layers.quantization.modelopt_quant import (
    ModelOptFp4Config,
    ModelOptFp4LinearMethod,
    ModelOptMixedPrecisionConfig,
)
from sglang.srt.layers.quantization.nvfp4_humming import (
    prepare_moe_nvfp4_layer_for_humming,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b", runner_config="1-gpu-large")


@unittest.skipUnless(
    torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 0),
    "Humming requires SM80+",
)
class TestModelOptNvfp4Humming(CustomTestCase):
    def test_moe_expert_projection_scales_routing_and_graph(self):
        from humming.config import GemmType

        from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
        from sglang.srt.layers.moe.moe_runner.humming import (
            HummingMoeQuantInfo,
            HummingRunnerCore,
            HummingRunnerInput,
        )

        torch.manual_seed(42)
        experts, hidden, intermediate = 4, 256, 256
        routed_scaling = 2.5
        config = MoeRunnerConfig(
            num_experts=experts,
            num_local_experts=experts,
            hidden_size=hidden,
            intermediate_size_per_partition=intermediate,
            top_k=2,
            params_dtype=torch.bfloat16,
            routed_scaling_factor=routed_scaling,
        )
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
        )
        # Run natively on Ampere too; on newer GPUs also exercise the W4A16
        # dispatch and group-16 scales used by SM80/SM86, plus 1D shared w13 scale.
        cases = [
            (cap, False)
            for cap in dict.fromkeys(
                (torch.cuda.get_device_capability(), (8, 0), (8, 6))
            )
        ] + [(torch.cuda.get_device_capability(), True)]
        for capability, shared_w13_scale in cases:
            with self.subTest(capability=capability, shared_w13_scale=shared_w13_scale):
                layer = torch.nn.Module()
                layer.num_local_experts = experts
                layer.hidden_size = hidden
                layer.intermediate_size_per_partition = intermediate
                layer.params_dtype = torch.bfloat16
                layer.moe_runner_config = config
                references = []
                for name, n, k in (
                    ("w13", intermediate * 2, hidden),
                    ("w2", hidden, intermediate),
                ):
                    packed = torch.randint(
                        0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
                    )
                    packed[..., ::8] = 7
                    scale = torch.ones(
                        experts, n, k // 16, device="cuda", dtype=torch.float8_e4m3fn
                    )
                    if capability < (8, 9):
                        scale[..., ::2] = 2
                    if name == "w13" and not shared_w13_scale:
                        global_scale = torch.tensor(
                            [
                                [0.015625, 0.03125],
                                [0.0625, 0.015625],
                                [0.03125, 0.0625],
                                [0.015625, 0.125],
                            ],
                            device="cuda",
                        )
                    else:
                        global_scale = torch.tensor(
                            [0.03125, 0.0625, 0.125, 0.015625],
                            device="cuda",
                        )
                    for suffix, tensor in (
                        ("weight", packed),
                        ("weight_scale", scale),
                        ("weight_scale_2", global_scale),
                        ("input_scale", torch.ones(experts, device="cuda")),
                    ):
                        layer.register_parameter(
                            f"{name}_{suffix}",
                            torch.nn.Parameter(tensor, requires_grad=False),
                        )
                    rows = (
                        global_scale.repeat_interleave(intermediate, dim=1)[..., None]
                        if global_scale.ndim == 2
                        else global_scale[:, None, None]
                    )
                    references.append(
                        lut[torch.stack([packed & 15, packed >> 4], -1).long()].flatten(
                            -2
                        )
                        * scale.float().repeat_interleave(16, dim=-1)
                        * rows
                    )
                with patch(
                    "sglang.srt.layers.quantization.nvfp4_humming.get_device_capability",
                    return_value=capability,
                ):
                    prepare_moe_nvfp4_layer_for_humming(layer, group_size=16)
                self.assertEqual(
                    layer.humming_metas["w13"].a_dtype.num_bits,
                    8 if capability >= (8, 9) else 16,
                )
                self.assertFalse(layer._humming_uses_deepep_fp8_dispatch)
                runner = HummingRunnerCore(config)
                for gemm_type in (GemmType.INDEXED, GemmType.GROUPED_CONTIGUOUS):
                    for m in (4, 32):
                        x = torch.randn(m, hidden, device="cuda", dtype=torch.bfloat16)
                        ids = torch.stack(
                            [
                                torch.arange(m, device="cuda") % experts,
                                (torch.arange(m, device="cuda") + 1) % experts,
                            ],
                            dim=1,
                        ).int()
                        weights = torch.softmax(torch.randn(m, 2, device="cuda"), -1)
                        expected = torch.zeros(m, hidden, device="cuda")
                        for slot in range(2):
                            w13 = references[0][ids[:, slot].long()]
                            gate, up = (
                                torch.bmm(w13, x.float().unsqueeze(-1))
                                .squeeze(-1)
                                .chunk(2, -1)
                            )
                            result = torch.bmm(
                                references[1][ids[:, slot].long()],
                                (torch.nn.functional.silu(gate) * up).unsqueeze(-1),
                            ).squeeze(-1)
                            expected += result * weights[:, slot, None] * routed_scaling
                        inputs = HummingRunnerInput(x, weights, ids, gemm_type)
                        actual = runner.run(
                            inputs, HummingMoeQuantInfo(layer), {}
                        ).hidden_states
                        relative_error = (
                            actual.float() - expected
                        ).norm() / expected.norm()
                        self.assertLess(
                            relative_error.item(),
                            0.06 if capability >= (8, 9) else 0.015,
                        )
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            captured = runner.run(
                                inputs, HummingMoeQuantInfo(layer), {}
                            ).hidden_states
                        graph.replay()
                        torch.testing.assert_close(captured, actual)

    def test_fused_scales_padding_awq_bias_and_graph(self):
        # Equal scales within each 32-element group make the weight conversion
        # exact. Different projection scales expose accidental max-reduction;
        # a stored bias must not be fused when apply() receives bias=None.
        torch.manual_seed(42)
        config = ModelOptFp4Config(
            is_checkpoint_nvfp4_serialized=True,
            group_size=16,
            is_awq=True,
            packed_modules_mapping={},
            exclude_modules=[],
        )
        backend_path = (
            "sglang.srt.layers.quantization.modelopt_quant.get_fp4_gemm_runner_backend"
        )
        capability_path = (
            "sglang.srt.layers.quantization.nvfp4_humming.get_device_capability"
        )
        mixed_config = ModelOptMixedPrecisionConfig.from_config(
            {
                "quant_algo": "MIXED_PRECISION",
                "quantized_layers": {"linear": {"quant_algo": "W4A16_NVFP4"}},
                "packed_modules_mapping": {},
            }
        )
        mixed_config.nvfp4a16_config.is_awq = True
        actual_capability = torch.cuda.get_device_capability()
        for dtype, w4a16, capability in (
            (torch.bfloat16, False, actual_capability),
            (torch.float16, False, actual_capability),
            (torch.bfloat16, True, actual_capability),
            (torch.bfloat16, "mixed", actual_capability),
            # Exercise Ampere dispatch on other GPUs too. Only SGLang's
            # capability query is overridden; Humming kernels run on real HW.
            (torch.bfloat16, "mixed", (8, 0)),
            (torch.float16, True, (8, 6)),
        ):
            config.is_w4a16 = w4a16
            with (
                self.subTest(dtype=dtype, w4a16=w4a16, capability=capability),
                patch(backend_path, return_value=Fp4GemmRunnerBackend.HUMMING),
                patch(capability_path, return_value=capability),
            ):
                with torch.device("cuda"):
                    layer = MergedColumnParallelLinear(
                        input_size=144,
                        output_sizes=[96, 64, 48],
                        bias=True,
                        params_dtype=dtype,
                        quant_config=mixed_config if w4a16 == "mixed" else config,
                        parallel_group="replicated",
                        prefix="linear",
                    )
                    method = layer.quant_method
                    self.assertIsInstance(method, ModelOptFp4LinearMethod)
                    layer.weight.data = torch.randint(
                        0, 256, layer.weight.shape, dtype=torch.uint8
                    )
                    # Include a positive maximum in every group, so requant
                    # has a unique exact scale even for the final short group.
                    layer.weight.data[:, ::8] = 7  # E2M1 code 7 represents +6.
                    layer.weight_scale.data.fill_(1)
                    if capability < (8, 9):
                        # Distinct group-16 scales catch accidental group-32
                        # requantization on Ampere, which must keep FP4 codes.
                        layer.weight_scale.data[:, ::2] = 4
                    layer.weight_scale_2.data.copy_(torch.tensor([0.125, 0.5, 2]))
                    layer.pre_quant_scale.data.copy_(torch.linspace(0.5, 1.5, 144))
                    layer.input_scale.data.fill_(123)  # FP4 calibration is irrelevant.
                    layer.register_parameter(
                        "bias", torch.nn.Parameter(torch.randn(208, dtype=dtype))
                    )

                lut = torch.tensor(
                    [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
                    device="cuda",
                )
                packed = layer.weight
                weight = lut[
                    torch.stack([packed & 15, packed >> 4], dim=-1).long()
                ].flatten(-2)
                scales = layer.weight_scale_2.repeat_interleave(
                    torch.tensor([96, 64, 48], device="cuda")
                )
                reference_weight = (
                    weight
                    * scales[:, None]
                    * layer.weight_scale.float().repeat_interleave(16, dim=1)
                )
                original_bias = layer.bias.detach().clone()
                method.process_weights_after_loading(layer)
                torch.testing.assert_close(layer.bias, original_bias)
                self.assertTrue(should_apply_lm_head_quant_method(layer, method))
                use_fp8 = capability >= (8, 9)
                self.assertEqual(method.quant_mode, "w4a8" if use_fp8 else "w4a16")
                self.assertEqual(
                    layer.humming_metas[""].a_dtype.num_bits, 8 if use_fp8 else 16
                )

                # Verify chunked N requantization produces identical weights/scales.
                if use_fp8:
                    from humming.schema import HummingWeightSchema

                    from sglang.srt.layers.quantization.nvfp4_humming import (
                        _requant_nvfp4_to_group32,
                    )

                    schema32 = HummingWeightSchema(
                        b_dtype="float4e2m1",
                        weight_scale_group_size=32,
                        weight_scale_type="group",
                        weight_scale_2_type="none",
                    )
                    big_w = torch.randint(
                        0, 256, (560, 72), device="cuda", dtype=torch.uint8
                    )
                    big_s = torch.ones(560, 9, device="cuda", dtype=torch.float8_e4m3fn)
                    big_g = torch.linspace(0.1, 1.0, 560, device="cuda")[:, None]
                    full = _requant_nvfp4_to_group32(
                        big_w,
                        big_s,
                        big_g,
                        shape_n=560,
                        shape_k=144,
                        params_dtype=dtype,
                        schema=schema32,
                    )
                    with patch(
                        "sglang.srt.layers.quantization.nvfp4_humming._REQUANT_CHUNK_N",
                        256,
                    ):
                        chunked = _requant_nvfp4_to_group32(
                            big_w,
                            big_s,
                            big_g,
                            shape_n=560,
                            shape_k=144,
                            params_dtype=dtype,
                            schema=schema32,
                        )
                    torch.testing.assert_close(chunked["weight"], full["weight"])
                    torch.testing.assert_close(
                        chunked["weight_scale"], full["weight_scale"]
                    )

                for m in (1, 16, 128):
                    x = torch.randn(2, m, 144, device="cuda", dtype=dtype)
                    actual = method.apply(layer, x)
                    reference = (x * layer.pre_quant_scale).float() @ reference_weight.T
                    relative_error = (
                        actual.float() - reference
                    ).norm() / reference.norm()
                    self.assertLess(relative_error.item(), 0.04 if use_fp8 else 0.005)
                    self.assertEqual(actual.shape, (2, m, 208))
                    self.assertEqual(actual.dtype, dtype)
                    torch.testing.assert_close(
                        method.apply(layer, x, layer.bias), actual + original_bias
                    )

                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        captured = method.apply(layer, x)
                    graph.replay()
                    torch.testing.assert_close(captured, actual)

    def test_compressed_tensors_nvfp4_and_a30_fp8_fallback(self):
        from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_w4a16_nvfp4 import (
            CompressedTensorsW4A16Fp4,
        )
        from sglang.srt.layers.quantization.humming import HummingConfig

        torch.manual_seed(7)
        lut = torch.tensor(
            [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6],
            device="cuda",
        )
        # 1. CompressedTensors NVFP4 with --fp4-gemm-backend humming on L40 & A30.
        ct_backend_path = (
            "sglang.srt.layers.quantization.compressed_tensors.schemes."
            "compressed_tensors_w4a16_nvfp4.get_fp4_gemm_runner_backend"
        )
        cap_path = "sglang.srt.layers.quantization.nvfp4_humming.get_device_capability"
        for capability in dict.fromkeys((torch.cuda.get_device_capability(), (8, 0))):
            with (
                self.subTest(ct_capability=capability),
                patch(ct_backend_path, return_value=Fp4GemmRunnerBackend.HUMMING),
                patch(cap_path, return_value=capability),
            ):
                scheme = CompressedTensorsW4A16Fp4(has_input_global_scale=True)
                with torch.device("cuda"):
                    layer = torch.nn.Module()
                    scheme.create_weights(
                        layer,
                        output_partition_sizes=[128, 128],
                        input_size_per_partition=256,
                        params_dtype=torch.bfloat16,
                        weight_loader=lambda *a, **kw: None,
                    )
                    layer.weight_packed.data = torch.randint(
                        0, 256, (256, 128), dtype=torch.uint8
                    )
                    layer.weight_packed.data[:, ::8] = 7
                    layer.weight_scale.data.fill_(1)
                    # compressed-tensors stores divisor 1 / scale
                    layer.weight_global_scale.data.copy_(torch.tensor([4.0, 2.0]))
                    layer.input_global_scale.data.fill_(1.0)
                packed = layer.weight_packed
                w_fp32 = lut[
                    torch.stack([packed & 15, packed >> 4], dim=-1).long()
                ].flatten(-2)
                row_scale = torch.tensor([0.25] * 128 + [0.5] * 128, device="cuda")
                ref_w = (
                    w_fp32
                    * row_scale[:, None]
                    * layer.weight_scale.float().repeat_interleave(16, dim=1)
                )
                scheme.process_weights_after_loading(layer)
                x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
                out = scheme.apply_weights(layer, x)
                ref = x.float() @ ref_w.T
                rel_err = (out.float() - ref).norm() / ref.norm()
                self.assertLess(rel_err.item(), 0.04 if capability >= (8, 9) else 0.005)

        # 2. HummingConfig FP8 on A30 (SM80): executes W8A16 without FP8 Tensor Cores.
        with patch(
            "sglang.srt.utils.device.get_device_capability", return_value=(8, 0)
        ):
            h_cfg = HummingConfig(
                {"quant_method": "fp8", "activation_scheme": "static"}
            )
            with torch.device("cuda"):
                fp8_layer = MergedColumnParallelLinear(
                    input_size=256,
                    output_sizes=[128, 128],
                    bias=False,
                    params_dtype=torch.bfloat16,
                    quant_config=h_cfg,
                    parallel_group="replicated",
                    prefix="model.layers.0.proj",
                )
                w_bf16 = torch.randn(256, 256, dtype=torch.bfloat16) * 0.25
                fp8_layer.weight.data.copy_(w_bf16.to(torch.float8_e4m3fn))
                fp8_layer.weight_scale.data.copy_(torch.tensor([0.5, 0.25]))
                fp8_layer.input_scale.data.fill_(1.0)
            ref_fp8_w = (
                fp8_layer.weight.float()
                * torch.tensor([0.5] * 128 + [0.25] * 128, device="cuda")[:, None]
            )
            fp8_layer.quant_method.process_weights_after_loading(fp8_layer)
            self.assertEqual(fp8_layer.humming_metas[""].a_dtype.num_bits, 16)
            x = torch.randn(16, 256, device="cuda", dtype=torch.bfloat16)
            actual, _ = fp8_layer(x)
            ref = x.float() @ ref_fp8_w.T
            self.assertLess(((actual.float() - ref).norm() / ref.norm()).item(), 0.01)


if __name__ == "__main__":
    unittest.main()
