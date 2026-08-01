import unittest

import torch

from sglang.kernels.ops.quantization.nvfp4_w4a8 import (
    dequantize_nvfp4_reference,
    nvfp4_w4a8_grouped_gemm,
    nvfp4_w4a8_linear,
    nvfp4_w4a8_moe_block_size,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


class TestNvfp4W4A8(unittest.TestCase):
    def test_reference_nibble_order_and_e2m1_values(self):
        weight = torch.tensor(
            [[0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE]],
            dtype=torch.uint8,
        )
        scale = torch.ones((1, 1), dtype=torch.float8_e4m3fn)
        result = dequantize_nvfp4_reference(weight, scale, torch.tensor(1.0))
        expected = torch.tensor(
            [
                [
                    0.0,
                    0.5,
                    1.0,
                    1.5,
                    2.0,
                    3.0,
                    4.0,
                    6.0,
                    -0.0,
                    -0.5,
                    -1.0,
                    -1.5,
                    -2.0,
                    -3.0,
                    -4.0,
                    -6.0,
                ]
            ]
        )
        torch.testing.assert_close(result, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_dense_matches_dequantized_reference(self):
        capability = torch.cuda.get_device_capability()
        if capability not in ((8, 9), (9, 0)):
            self.skipTest("nvfp4_w4a8 supports SM89/SM90")

        torch.manual_seed(7)
        m, n, k = 8, 48, 64
        x = torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.2
        weight = torch.randint(0, 256, (n, k // 2), device="cuda", dtype=torch.uint8)
        scale = (
            torch.rand((n, k // 16), device="cuda", dtype=torch.float32) * 4 + 1
        ).to(torch.float8_e4m3fn)
        global_scale = torch.tensor(0.005, device="cuda", dtype=torch.float32)

        actual = nvfp4_w4a8_linear(x, weight, scale, global_scale)
        dequant_weight = dequantize_nvfp4_reference(
            weight, scale, global_scale, torch.float32
        )
        expected = torch.matmul(x.float(), dequant_weight.T).to(torch.bfloat16)

        # The emulation converts each pair of block-16 weight groups to an FP8
        # tensor-core tile.  Allow the expected FP8 representation error.
        torch.testing.assert_close(actual, expected, rtol=0.12, atol=0.08)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_grouped_moe_matches_dequantized_reference(self):
        capability = torch.cuda.get_device_capability()
        if capability not in ((8, 9), (9, 0)):
            self.skipTest("nvfp4_w4a8 supports SM89/SM90")

        torch.manual_seed(11)
        tokens, top_k, experts, n, k = 3, 2, 3, 40, 64
        x = torch.randn((tokens, k), device="cuda", dtype=torch.bfloat16) * 0.2
        weight = torch.randint(
            0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
        )
        scale = (
            torch.rand((experts, n, k // 16), device="cuda", dtype=torch.float32) * 4
            + 1
        ).to(torch.float8_e4m3fn)
        global_scale = torch.tensor(
            [0.004, 0.005, 0.006], device="cuda", dtype=torch.float32
        )
        topk_ids = torch.tensor(
            [[0, 1], [2, 0], [1, 2]], device="cuda", dtype=torch.int32
        )
        topk_weights = torch.tensor(
            [[0.6, 0.4], [0.7, 0.3], [0.55, 0.45]],
            device="cuda",
            dtype=torch.float32,
        )

        # Build the same expert-major, block-padded route layout returned by
        # moe_align_block_size without depending on that auxiliary CUDA op.
        block_m = nvfp4_w4a8_moe_block_size(x)
        flat_experts = topk_ids.flatten().tolist()
        sorted_routes = []
        for expert in range(experts):
            expert_routes = [
                route
                for route, expert_id in enumerate(flat_experts)
                if expert_id == expert
            ]
            sorted_routes.extend(expert_routes)
            sorted_routes.extend([tokens * top_k] * (block_m - len(expert_routes)))
        sorted_token_ids = torch.tensor(sorted_routes, device="cuda", dtype=torch.int32)
        expert_ids = torch.arange(experts, device="cuda", dtype=torch.int32)
        num_tokens_post_padded = torch.tensor(
            [len(sorted_routes)], device="cuda", dtype=torch.int32
        )

        actual = nvfp4_w4a8_grouped_gemm(
            x,
            weight,
            scale,
            global_scale,
            topk_weights,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            top_k=top_k,
            mul_routed_weight=True,
        )

        expected = torch.empty_like(actual)
        for route, expert in enumerate(flat_experts):
            dequant_weight = dequantize_nvfp4_reference(
                weight[expert], scale[expert], global_scale[expert], torch.float32
            )
            token = route // top_k
            route_weight = topk_weights.flatten()[route]
            expected[route] = (
                torch.matmul(x[token].float(), dequant_weight.T) * route_weight
            ).to(torch.bfloat16)

        torch.testing.assert_close(actual, expected, rtol=0.12, atol=0.08)


if __name__ == "__main__":
    unittest.main()
