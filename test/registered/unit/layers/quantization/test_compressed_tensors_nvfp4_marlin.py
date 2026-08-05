"""CPU tests for compressed-tensors NVFP4 conversion to Marlin layouts."""

import unittest
from unittest import mock

import torch

from sglang.srt.layers.moe.utils import MoeRunnerBackend
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w4a4_nvfp4 as linear_module,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes import (
    compressed_tensors_w4a4_nvfp4_moe as moe_module,
)
from sglang.srt.layers.quantization.fp4_utils import Fp4GemmRunnerBackend
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCompressedTensorsNvfp4Marlin(CustomTestCase):
    def test_dense_weights_are_adapted_to_modelopt_marlin_convention(self):
        layer = torch.nn.Module()
        layer.input_global_scale = torch.nn.Parameter(torch.tensor([2.0]))
        layer.weight_global_scale = torch.nn.Parameter(torch.tensor([4.0]))
        layer.weight_packed = torch.nn.Parameter(
            torch.zeros((64, 32), dtype=torch.uint8), requires_grad=False
        )
        layer.weight_scale = torch.nn.Parameter(
            torch.zeros((64, 4), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        layer.input_size_per_partition = 64
        layer.output_size_per_partition = 64
        layer.params_dtype = torch.bfloat16

        with (
            mock.patch.object(
                linear_module,
                "get_fp4_gemm_runner_backend",
                return_value=Fp4GemmRunnerBackend.MARLIN,
            ),
            mock.patch.object(
                linear_module, "is_blackwell_supported", return_value=False
            ),
            mock.patch.object(
                linear_module, "prepare_nvfp4_layer_for_marlin"
            ) as prepare,
        ):
            scheme = linear_module.CompressedTensorsW4A4Fp4()
            scheme.process_weights_after_loading(layer)

        self.assertFalse(hasattr(layer, "weight_packed"))
        self.assertEqual(layer.weight.shape, (64, 32))
        torch.testing.assert_close(layer.weight_global_scale, torch.tensor(0.25))
        prepare.assert_called_once_with(layer, group_size=16)

    def test_moe_weights_are_adapted_to_modelopt_marlin_convention(self):
        with (
            mock.patch.object(
                moe_module,
                "get_moe_runner_backend",
                return_value=MoeRunnerBackend.MARLIN,
            ),
            mock.patch.object(moe_module, "is_blackwell_supported", return_value=False),
        ):
            scheme = moe_module.CompressedTensorsW4A4Nvfp4MoE()

        self.assertFalse(scheme.load_up_proj_weight_first)

        layer = torch.nn.Module()
        layer.w13_weight_packed = torch.nn.Parameter(
            torch.zeros((2, 128, 32), dtype=torch.uint8), requires_grad=False
        )
        layer.w2_weight_packed = torch.nn.Parameter(
            torch.zeros((2, 64, 64), dtype=torch.uint8), requires_grad=False
        )
        layer.w13_weight_scale = torch.nn.Parameter(
            torch.zeros((2, 128, 4), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w2_weight_scale = torch.nn.Parameter(
            torch.zeros((2, 64, 8), dtype=torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w13_weight_global_scale = torch.nn.Parameter(
            torch.tensor([[4.0, 4.0], [8.0, 8.0]]), requires_grad=False
        )
        layer.w2_weight_global_scale = torch.nn.Parameter(
            torch.tensor([2.0, 4.0]), requires_grad=False
        )

        with mock.patch.object(
            moe_module, "prepare_moe_nvfp4_layer_for_marlin"
        ) as prepare:
            scheme.process_weights_after_loading(layer)

        torch.testing.assert_close(
            layer.w13_weight_scale_2, torch.tensor([0.25, 0.125])
        )
        torch.testing.assert_close(layer.w2_weight_scale_2, torch.tensor([0.5, 0.25]))
        prepare.assert_called_once_with(layer, group_size=16)


if __name__ == "__main__":
    unittest.main()
