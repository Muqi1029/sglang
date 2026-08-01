import unittest

import torch

from sglang.kernels.ops.quantization.nvfp4_w4a8 import (
    CUTLASS_W4A8_GROUP_SIZE,
    dequantize_nvfp4_reference,
    nvfp4_w4a8_grouped_gemm,
    nvfp4_w4a8_linear,
    nvfp4_w4a8_moe_block_size,
    prepare_nvfp4_for_cutlass_w4a8,
    repack_nvfp4_for_w4a8,
    requantize_nvfp4_to_int4_reference,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")
register_cuda_ci(est_time=8, stage="base-b", runner_config="1-gpu-small")


class TestNvfp4W4A8(unittest.TestCase):
    def test_repack_keeps_logical_values_and_makes_n_contiguous(self):
        weight = torch.randint(0, 256, (3, 16), dtype=torch.uint8)
        scale = torch.randn(3, 2).to(torch.float8_e4m3fn)

        packed_weight, packed_scale = repack_nvfp4_for_w4a8(weight, scale)

        self.assertEqual(tuple(packed_weight.shape), tuple(weight.shape))
        self.assertEqual(tuple(packed_scale.shape), tuple(scale.shape))
        self.assertTrue(torch.equal(packed_weight, weight))
        self.assertTrue(torch.equal(packed_scale, scale))
        self.assertEqual(packed_weight.stride(-2), 1)
        self.assertEqual(packed_scale.stride(-2), 1)

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

    def test_cutlass_group32_requantization_layout_and_error(self):
        torch.manual_seed(17)
        experts, n, k = 2, 6, 128
        weight = torch.randint(0, 256, (experts, n, k // 2), dtype=torch.uint8)
        scale = (
            torch.rand((experts, n, k // 16), dtype=torch.float32) * 3.0 + 0.25
        ).to(torch.float8_e4m3fn)
        global_scale = torch.tensor([0.01, 0.02], dtype=torch.float32)

        packed, cutlass_scale = requantize_nvfp4_to_int4_reference(
            weight, scale, global_scale
        )

        self.assertEqual(packed.dtype, torch.int8)
        self.assertEqual(tuple(packed.shape), tuple(weight.shape))
        self.assertEqual(tuple(cutlass_scale.shape), (experts, (k // 32) // 4, n * 4))
        # E4M3 group16 and BF16 group32 consume exactly the same bytes.
        self.assertEqual(cutlass_scale.numel() * 2, scale.numel())

        packed_u8 = packed.to(torch.uint8)
        low = packed_u8 & 0xF
        high = (packed_u8 >> 4) & 0xF
        codes = torch.stack((low, high), dim=-1).flatten(-2).to(torch.int16)
        quant = torch.where(codes >= 8, codes - 16, codes).float()

        groups = k // 32
        deinterleaved_scale = (
            cutlass_scale.reshape(experts, groups // 4, n, 4)
            .permute(0, 2, 1, 3)
            .reshape(experts, n, groups)
            .float()
        )
        reconstructed = quant * deinterleaved_scale.repeat_interleave(32, dim=-1)
        original = dequantize_nvfp4_reference(
            weight,
            scale,
            torch.ones((experts, 1, 1), dtype=torch.float32),
        ) * global_scale.view(experts, 1, 1)

        group_error = (reconstructed - original).abs().reshape(experts, n, groups, 32)
        # Round-to-nearest signed INT4 is bounded by half a group scale; add a
        # small allowance for storing that scale as BF16.
        self.assertTrue(
            torch.all(group_error.amax(dim=-1) <= deinterleaved_scale * 0.51 + 1e-4)
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_sm90_cutlass_preparation_matches_reference(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("CUTLASS nvfp4_w4a8 preparation is SM90-only")

        torch.manual_seed(19)
        experts, n, k = 2, 8, 128
        weight = torch.randint(
            0, 256, (experts, n, k // 2), device="cuda", dtype=torch.uint8
        )
        scale = (
            torch.rand((experts, n, k // 16), device="cuda", dtype=torch.float32) * 3.0
            + 0.25
        ).to(torch.float8_e4m3fn)
        global_scale = torch.tensor(
            [[0.01, 0.02], [0.03, 0.04]], device="cuda", dtype=torch.float32
        )
        expected_weight, expected_scale = requantize_nvfp4_to_int4_reference(
            weight.clone(), scale, global_scale
        )
        source_ptr = weight.data_ptr()

        actual_weight, actual_scale = prepare_nvfp4_for_cutlass_w4a8(
            weight, scale, global_scale
        )

        self.assertEqual(actual_weight.data_ptr(), source_ptr)
        self.assertTrue(torch.equal(actual_weight, expected_weight))
        torch.testing.assert_close(actual_scale, expected_scale, rtol=0.01, atol=1e-4)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_sm90_cutlass_group32_gemm_matches_reference(self):
        if torch.cuda.get_device_capability() != (9, 0):
            self.skipTest("CUTLASS group-32 W4A8 GEMM is SM90-only")

        from sgl_kernel import cutlass_w4a8_moe_mm

        torch.manual_seed(23)
        experts, m, n, k = 1, 8, 64, 128
        activation = (
            torch.randn((m, k), device="cuda", dtype=torch.bfloat16) * 0.1
        ).to(torch.float8_e4m3fn)
        activation_scale = torch.ones(1, device="cuda", dtype=torch.float32)
        weight = torch.randint(-7, 8, (experts, n, k), device="cuda", dtype=torch.int8)
        weight_scale = (
            torch.rand(
                (experts, n, k // CUTLASS_W4A8_GROUP_SIZE),
                device="cuda",
                dtype=torch.float32,
            )
            * 0.01
            + 0.001
        ).to(torch.bfloat16)

        low = weight[..., 0::2].to(torch.int16) & 0xF
        high = (weight[..., 1::2].to(torch.int16) & 0xF) << 4
        packed_weight = (low | high).to(torch.int8).contiguous()
        groups = k // CUTLASS_W4A8_GROUP_SIZE
        packed_scale = (
            weight_scale.reshape(experts, n, groups // 4, 4)
            .permute(0, 2, 1, 3)
            .reshape(experts, groups // 4, n * 4)
            .contiguous()
        )

        expert_offsets = torch.zeros(experts, device="cuda", dtype=torch.int32)
        problem_sizes = torch.tensor([[n, m, k]], device="cuda", dtype=torch.int32)
        a_strides = torch.full((experts, 3), k, device="cuda", dtype=torch.int64)
        c_strides = torch.full((experts, 3), n, device="cuda", dtype=torch.int64)
        output = torch.empty((m, n), device="cuda", dtype=torch.bfloat16)
        cutlass_w4a8_moe_mm(
            output,
            activation,
            packed_weight,
            activation_scale,
            packed_scale,
            expert_offsets,
            problem_sizes,
            a_strides,
            a_strides,
            c_strides,
            c_strides,
            CUTLASS_W4A8_GROUP_SIZE,
            1,
        )

        dequant_weight = weight.float() * weight_scale.float().repeat_interleave(
            CUTLASS_W4A8_GROUP_SIZE, dim=-1
        )
        expected = torch.matmul(
            activation.float(), dequant_weight[0].transpose(0, 1)
        ).to(torch.bfloat16)
        torch.testing.assert_close(output, expected, rtol=0.03, atol=0.2)

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
    def test_dense_preserves_small_adjacent_group_scale(self):
        capability = torch.cuda.get_device_capability()
        if capability not in ((8, 9), (9, 0)):
            self.skipTest("nvfp4_w4a8 supports SM89/SM90")

        # Without the E4M3 range boost, the first group's normalized 0.5 value
        # falls below the minimum normal E4M3 value and incurs a very large
        # relative error when paired with the 448-scale second group.
        m, n, k = 4, 16, 32
        x = torch.ones((m, k), device="cuda", dtype=torch.bfloat16)
        weight = torch.full((n, k // 2), 0x11, device="cuda", dtype=torch.uint8)
        scale = torch.empty((n, 2), device="cuda", dtype=torch.float8_e4m3fn)
        scale[:, 0] = 1.0
        scale[:, 1] = 448.0
        global_scale = torch.tensor(0.001, device="cuda", dtype=torch.float32)

        actual = nvfp4_w4a8_linear(x, weight, scale, global_scale)
        dequant_weight = dequantize_nvfp4_reference(
            weight, scale, global_scale, torch.float32
        )
        expected = torch.matmul(x.float(), dequant_weight.T).to(torch.bfloat16)

        torch.testing.assert_close(actual, expected, rtol=0.04, atol=0.04)

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
            block_m=block_m,
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
