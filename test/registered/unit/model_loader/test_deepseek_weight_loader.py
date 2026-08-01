import unittest

import torch

from sglang.kernels.ops.quantization.nvfp4_w4a8 import (
    dequantize_nvfp4_reference,
)
from sglang.srt.models.deepseek_common.deepseek_weight_loader import (
    _load_fused_indexer_wk,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestDeepseekWeightLoader(CustomTestCase):
    def test_load_fused_indexer_nvfp4_tensors_in_arbitrary_order(self):
        prefix = "model.layers.0.self_attn.indexer"
        fused_name = f"{prefix}.wk_weights_proj.weight"
        fused_weight = torch.full((3, 16), torch.nan, dtype=torch.bfloat16)
        params_dict = {fused_name: fused_weight}

        wk_weight = torch.tensor(
            [
                [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE],
                [0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01],
            ],
            dtype=torch.uint8,
        )
        weights_proj_weight = torch.tensor(
            [[0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF]],
            dtype=torch.uint8,
        )
        wk_scale = torch.tensor([[0.5], [1.0]], dtype=torch.float8_e4m3fn)
        weights_proj_scale = torch.tensor([[1.5]], dtype=torch.float8_e4m3fn)
        wk_global_scale = torch.tensor(2.0)
        weights_proj_global_scale = torch.tensor(0.25)

        tensors = {
            f"{prefix}.wk.weight": wk_weight,
            f"{prefix}.wk.weight_scale": wk_scale,
            f"{prefix}.wk.weight_scale_2": wk_global_scale,
            f"{prefix}.wk.input_scale": torch.tensor(1.0),
            f"{prefix}.weights_proj.weight": weights_proj_weight,
            f"{prefix}.weights_proj.weight_scale": weights_proj_scale,
            f"{prefix}.weights_proj.weight_scale_2": weights_proj_global_scale,
            f"{prefix}.weights_proj.input_scale": torch.tensor(1.0),
        }

        # Exercise scale-before-weight and weight-before-scale loading, while
        # interleaving wk and weights_proj as a real sharded iterator may do.
        order = [
            f"{prefix}.wk.weight_scale_2",
            f"{prefix}.weights_proj.weight",
            f"{prefix}.wk.input_scale",
            f"{prefix}.weights_proj.weight_scale_2",
            f"{prefix}.wk.weight_scale",
            f"{prefix}.weights_proj.input_scale",
            f"{prefix}.wk.weight",
            f"{prefix}.weights_proj.weight_scale",
        ]
        pending = {}
        for name in order:
            self.assertTrue(
                _load_fused_indexer_wk(
                    name,
                    tensors[name],
                    params_dict,
                    pending,
                    quant_config=None,
                )
            )

        expected_wk = dequantize_nvfp4_reference(
            wk_weight, wk_scale, wk_global_scale, dtype=torch.bfloat16
        )
        expected_weights_proj = dequantize_nvfp4_reference(
            weights_proj_weight,
            weights_proj_scale,
            weights_proj_global_scale,
            dtype=torch.bfloat16,
        )
        torch.testing.assert_close(fused_weight[:2], expected_wk)
        torch.testing.assert_close(fused_weight[-1:], expected_weights_proj)
        self.assertEqual(pending, {})


if __name__ == "__main__":
    unittest.main()
