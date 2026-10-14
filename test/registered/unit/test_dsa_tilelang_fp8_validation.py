"""Validate which fp8_e4m3 KV + tilelang DSA combinations may start on CUDA.

Regression: unsupported combinations used to boot the server and crash at
decode CUDA-graph capture with ``kernel main input KV dtype expected bfloat16,
but got float8_e4m3fn``. BF16 NoPE models with both phases on tilelang read
group-scaled FP8 rows and are the one CUDA combination that may start.

Validate raw TileLang FP8 KV against incompatible backends and platforms."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.arg_groups.overrides import (
    ResolvedView,
    _check_dsa_backend_constraints,
    _check_tilelang_dsa_fp8_kv,
    _dsa_split_backend_resolution,
)
from sglang.srt.runtime_context import override_platform
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestDsaTilelangFp8Validation(CustomTestCase):
    def test_cuda_fp8_tilelang_decode_rejected(self):
        with self.assertRaises(ValueError):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "flashmla_kv", "tilelang", hip=False)

    def test_cuda_fp8_tilelang_prefill_rejected(self):
        with self.assertRaises(ValueError):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "trtllm", hip=False)

    @patch("torch.cuda.get_device_capability", return_value=(9, 0))
    def test_cuda_fp8_tilelang_allowed(self, _capability):
        _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "tilelang", hip=False)

    @patch("torch.cuda.get_device_capability", return_value=(8, 0))
    def test_cuda_fp8_requires_fp8_tensor_cores(self, _capability):
        with self.assertRaisesRegex(ValueError, "SM89"):
            _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "tilelang", hip=False)

    def test_cuda_fp8_tilelang_dcp_rejected(self):
        with self.assertRaisesRegex(ValueError, "dcp-size"):
            _check_tilelang_dsa_fp8_kv(
                "fp8_e4m3", "tilelang", "tilelang", hip=False, dcp_size=2
            )

    def test_hip_fp8_tilelang_allowed(self):
        # ROCm has a real fp8 tilelang kernel
        _check_tilelang_dsa_fp8_kv("fp8_e4m3", "tilelang", "tilelang", hip=True)

    def test_bf16_tilelang_allowed(self):
        _check_tilelang_dsa_fp8_kv("bfloat16", "tilelang", "tilelang", hip=False)

    def test_cuda_fp8_non_tilelang_allowed(self):
        # fp8-capable backends must pass
        _check_tilelang_dsa_fp8_kv("fp8_e4m3", "flashmla_kv", "trtllm", hip=False)

    def test_cuda_group_scaled_nope_allowed_only_when_both_phases_match(self):
        _check_dsa_backend_constraints(
            "fp8_e4m3", "tilelang", "tilelang", hip=False, nope_group_scaled=True
        )
        for prefill, decode in (
            ("tilelang", "trtllm"),
            ("flashmla_kv", "tilelang"),
            ("tilelang", "fa3"),
            ("tilelang", None),
        ):
            with self.subTest(prefill=prefill, decode=decode):
                with self.assertRaises(ValueError):
                    _check_dsa_backend_constraints(
                        "fp8_e4m3", prefill, decode, hip=False, nope_group_scaled=True
                    )

    def test_cuda_group_scaled_nope_rejects_dcp(self):
        with self.assertRaisesRegex(ValueError, "dcp-size 1"):
            _check_dsa_backend_constraints(
                "fp8_e4m3",
                "tilelang",
                "tilelang",
                hip=False,
                nope_group_scaled=True,
                dcp_size=2,
            )

    def test_resolution_derives_geometry_from_model_config(self):
        def make_view(rank=512, rope=0, dtype=torch.bfloat16, **options):
            model_config = SimpleNamespace(
                hf_config=SimpleNamespace(
                    architectures=["Glm5NextForConditionalGeneration"]
                ),
                kv_lora_rank=rank,
                qk_rope_head_dim=rope,
                dtype=dtype,
            )
            fields = dict(
                _model_config=model_config,
                kv_cache_dtype="fp8_e4m3",
                dsa_prefill_backend="tilelang",
                dsa_decode_backend="tilelang",
                enable_hisparse=False,
                dcp_size=1,
            )
            fields.update(options)
            return ResolvedView(SimpleNamespace(**fields))

        with (
            patch("sglang.srt.configs.model_config.is_deepseek_dsa", return_value=True),
            patch("torch.cuda.get_device_capability", return_value=(9, 0)),
            override_platform(is_npu=False, is_xpu=False, is_hip=False),
        ):
            self.assertEqual(_dsa_split_backend_resolution(make_view()), {})
            self.assertEqual(_dsa_split_backend_resolution(make_view(rank=256)), {})
            # Recognizing the geometry must not change the fp8 backend defaults.
            self.assertEqual(
                _dsa_split_backend_resolution(
                    make_view(dsa_prefill_backend=None, dsa_decode_backend=None)
                ),
                {
                    "dsa_prefill_backend": "flashmla_kv",
                    "dsa_decode_backend": "flashmla_kv",
                },
            )
            for options in (
                {"rope": 64},
                {"rank": 448},
                {"dtype": torch.float16},
                {"dcp_size": 2},
                {"enable_hisparse": True},
            ):
                with self.subTest(options=options):
                    with self.assertRaises(ValueError):
                        _dsa_split_backend_resolution(make_view(**options))


class TestMlaKvCacheDimForDsaTilelang(CustomTestCase):
    """The CUDA tilelang NoPE pool must allocate the row the writer emits.

    Regression: this returned the raw MLA width whenever both DSA backends were
    tilelang, so ``dsa_kv_cache_store_fp8`` stayed off and the pool sized rows
    at ``kv_lora_rank``. The backend then read those rows as group-scaled
    NoPE rows -- scales and all -- which desynchronized decode and produced
    NaN logits (sampled as token 0, ``!``).
    """

    @staticmethod
    def _model(kv_lora_rank, qk_rope_head_dim):
        return SimpleNamespace(
            hf_config={
                "architectures": ["Glm5NextForConditionalGeneration"],
                "index_topk": 2048,
            },
            kv_lora_rank=kv_lora_rank,
            qk_rope_head_dim=qk_rope_head_dim,
        )

    def _dim(self, kv_lora_rank, qk_rope_head_dim, kv_cache_dtype, prefill, decode):
        from sglang.srt.mem_cache import kv_cache_configurator

        exec_view = SimpleNamespace(
            kernel=SimpleNamespace(
                dsa_prefill_backend=prefill, dsa_decode_backend=decode
            )
        )
        with (
            patch.object(kv_cache_configurator, "get_exec", return_value=exec_view),
            patch.object(
                kv_cache_configurator,
                "get_disagg",
                return_value=SimpleNamespace(disaggregation_mode="null"),
            ),
            patch.object(kv_cache_configurator, "_is_hip", False),
        ):
            return kv_cache_configurator.calculate_mla_kv_cache_dim(
                model_config=self._model(kv_lora_rank, qk_rope_head_dim),
                kv_cache_dtype=kv_cache_dtype,
            )

    def test_nope_fp8_tilelang_allocates_scale_bytes(self):
        """NoPE fp8 rows carry four FP32 scales: 512 + 4 * 4 = 528."""
        self.assertEqual(
            self._dim(512, 0, torch.float8_e4m3fn, "tilelang", "tilelang"), 528
        )

    def test_rope_fp8_tilelang_keeps_raw_row(self):
        """A RoPE tail is only served by the raw layout, which has no scales."""
        self.assertEqual(
            self._dim(512, 64, torch.float8_e4m3fn, "tilelang", "tilelang"), 576
        )

    def test_nope_bf16_tilelang_keeps_raw_row(self):
        """Without fp8 there is no scale payload to reserve."""
        self.assertEqual(self._dim(512, 0, torch.bfloat16, "tilelang", "tilelang"), 512)

    def test_trtllm_keeps_raw_row(self):
        self.assertEqual(
            self._dim(512, 0, torch.float8_e4m3fn, "trtllm", "trtllm"), 512
        )


if __name__ == "__main__":
    unittest.main()
