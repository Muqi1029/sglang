import concurrent.futures
import json
import os
import sys
import tempfile
import threading
import unittest
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import torch
from safetensors.torch import save_file

import sglang.srt.model_loader.loader as loader_mod
import sglang.srt.model_loader.utils as model_loader_utils
import sglang.srt.model_loader.weight_utils as weight_utils
from sglang.srt.configs.device_config import DeviceConfig
from sglang.srt.configs.load_config import _DEFAULT_LOAD_GROUP, LoadConfig, LoadFormat
from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.models.deepseek_common import deepseek_weight_loader
from sglang.srt.models.deepseek_v4 import (
    _dequant_fp8_wo_a,
    _dequant_fp8_wo_a_streaming,
)
from sglang.srt.models.deepseek_v4_dspark import DeepseekV4ForCausalLMDSpark
from sglang.srt.utils import runai_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


class _FakeModel:
    def eval(self):
        return self


class TestRunaiModelStreamerLoader(CustomTestCase):
    def test_bundled_draft_file_selection(self):
        with tempfile.TemporaryDirectory() as model_path:
            for name in ("model.safetensors", "mtp.safetensors"):
                save_file({"weight": torch.ones(1)}, os.path.join(model_path, name))
            with open(
                os.path.join(model_path, "model.safetensors.index.json"), "w"
            ) as f:
                json.dump({"weight_map": {"weight": "model.safetensors"}}, f)

            config = SimpleNamespace(
                model_path=model_path,
                revision=None,
                hf_config=SimpleNamespace(architectures=["DraftModel"]),
            )
            model = SimpleNamespace(
                allow_patterns_overrides=["mtp.safetensors"],
                safetensors_index_file=None,
            )
            loader = loader_mod.RunaiModelStreamerLoader(
                LoadConfig(load_format=LoadFormat.RUNAI_STREAMER)
            )
            loader.target_device_str = "cpu"
            with (
                patch.object(loader_mod, "get_server_args", return_value=None),
                # The runai package is optional; list the local directory instead.
                patch.object(
                    runai_utils,
                    "list_safetensors",
                    side_effect=lambda path: sorted(
                        os.path.join(path, name)
                        for name in os.listdir(path)
                        if name.endswith(".safetensors")
                    ),
                ),
                patch.object(
                    weight_utils,
                    "runai_safetensors_weights_iterator",
                    return_value=iter(()),
                ) as streamer,
            ):
                list(loader._get_all_weights(config, model))
                streamer.assert_called_once_with(
                    [os.path.join(model_path, "mtp.safetensors")],
                    False,
                    "cpu",
                    load_group=_DEFAULT_LOAD_GROUP,
                )

    def test_local_directory_wildcard_rejected(self):
        """A directory wildcard must fail loudly on local paths too, not match nothing."""
        loader = loader_mod.RunaiModelStreamerLoader(
            LoadConfig(load_format=LoadFormat.RUNAI_STREAMER)
        )
        with (
            tempfile.TemporaryDirectory() as model_path,
            patch.object(loader_mod, "get_server_args", return_value=None),
            patch.object(runai_utils, "list_safetensors", return_value=[]),
            self.assertRaisesRegex(ValueError, "directory wildcards"),
        ):
            loader._prepare_weights(
                model_path, None, allow_patterns_overrides=["*/model.safetensors"]
            )

    def test_remote_checkpoint_index(self):
        for scheme in ("s3", "gs", "az"):
            for relative_cache in (False, True):
                with (
                    self.subTest(scheme=scheme, relative_cache=relative_cache),
                    tempfile.TemporaryDirectory() as cache_dir,
                ):
                    model_path = f"{scheme}://bucket/model"
                    shard = "model-00001-of-00001.safetensors"
                    files = [
                        f"{model_path}/{shard}",
                        f"{model_path}/mtp.safetensors",
                        f"{model_path}/unused.safetensors",
                    ]
                    cache_root = (
                        os.path.relpath(cache_dir) if relative_cache else cache_dir
                    )
                    with (
                        patch.object(
                            runai_utils.envs.SGLANG_CACHE_DIR,
                            "get",
                            return_value=cache_root,
                        ),
                        patch.object(loader_mod, "get_server_args", return_value=None),
                        patch.object(
                            runai_utils, "list_safetensors", return_value=files
                        ) as list_safetensors,
                        patch.object(
                            loader_mod, "download_weights_from_hf"
                        ) as download,
                        patch.object(
                            loader_mod, "download_safetensors_index_file_from_hf"
                        ) as download_index,
                        patch.object(
                            weight_utils,
                            "runai_safetensors_weights_iterator",
                            return_value=iter(()),
                        ) as streamer,
                    ):
                        metadata = runai_utils.ObjectStorageModel.get_path(model_path)
                        os.makedirs(metadata, exist_ok=True)
                        loader = loader_mod.RunaiModelStreamerLoader(
                            LoadConfig(load_format=LoadFormat.RUNAI_STREAMER)
                        )
                        self.assertEqual(
                            loader._prepare_weights(model_path, revision=None),
                            (model_path, files),
                        )
                        list_safetensors.assert_called_with(path=model_path)

                        with open(
                            os.path.join(metadata, "model.safetensors.index.json"), "w"
                        ) as f:
                            json.dump({"weight_map": {"weight": shard}}, f)
                        self.assertEqual(
                            loader._prepare_weights(model_path, revision=None),
                            (model_path, [files[0]]),
                        )

                        loader.target_device_str = "cpu"
                        source = loader_mod.RunaiModelStreamerLoader.Source(
                            model_or_path=model_path,
                            revision=None,
                            model_config=cast(
                                ModelConfig,
                                SimpleNamespace(
                                    hf_config=SimpleNamespace(
                                        architectures=["Glm4MoeForCausalLMNextN"],
                                        num_nextn_predict_layers=1,
                                    )
                                ),
                            ),
                        )
                        list(loader._get_weights_iterator(source))
                        streamer.assert_called_once_with(
                            files[:2], True, "cpu", load_group=_DEFAULT_LOAD_GROUP
                        )

                        streamer.reset_mock()
                        config = SimpleNamespace(
                            model_path=metadata,
                            model_weights=model_path,
                            revision="revision",
                            hf_config=SimpleNamespace(architectures=["DraftModel"]),
                        )
                        model = SimpleNamespace(
                            allow_patterns_overrides=["mtp.safetensors"],
                            safetensors_index_file=None,
                        )
                        list(loader._get_all_weights(config, model))
                        list_safetensors.assert_called_with(path=model_path)
                        streamer.assert_called_once_with(
                            [files[1]], True, "cpu", load_group=_DEFAULT_LOAD_GROUP
                        )
                        download.assert_not_called()
                        download_index.assert_not_called()

                        list_safetensors.return_value = [files[-1]]
                        with self.assertRaisesRegex(RuntimeError, shard):
                            loader._prepare_weights(model_path, revision=None)

    def test_passes_quant_config_to_model_init(self):
        quant_config = object()
        fake_model = _FakeModel()

        with (
            patch.object(
                loader_mod,
                "_get_quantization_config",
                return_value=quant_config,
            ),
            patch.object(loader_mod, "_initialize_model") as mock_initialize_model,
            patch.object(
                loader_mod.DefaultModelLoader,
                "load_weights_and_postprocess",
            ) as mock_load_weights,
        ):
            mock_initialize_model.return_value = fake_model
            runai_loader = loader_mod.RunaiModelStreamerLoader(
                LoadConfig(
                    load_format=LoadFormat.RUNAI_STREAMER,
                    model_loader_extra_config={},
                )
            )
            model_config = cast(
                ModelConfig,
                SimpleNamespace(dtype=torch.float16, modelopt_quant=False),
            )

            model = runai_loader.load_model(
                model_config=model_config,
                device_config=DeviceConfig("cpu"),
            )

        self.assertIs(model, fake_model)
        self.assertIs(mock_load_weights.call_args.args[0], fake_model)
        self.assertIs(mock_initialize_model.call_args.args[2], quant_config)

    def test_marks_streamer_tensors(self):
        source_tensor = torch.tensor([1], dtype=torch.int32)

        class FakeStreamer:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def stream_files(self, *_args, **_kwargs):
                self.files_to_tensors_metadata = {0: [object()]}

            def get_tensors(self):
                yield "weight", source_tensor

        with patch.dict(
            sys.modules,
            {"runai_model_streamer": SimpleNamespace(SafetensorsStreamer=FakeStreamer)},
        ):
            weights = list(
                weight_utils.runai_safetensors_weights_iterator(["model.safetensors"])
            )

        self.assertEqual(weights[0][0], "weight")
        self.assertTrue(getattr(weights[0][1], weight_utils.RUNAI_STREAMER_TENSOR_ATTR))

    def test_load_group_controls_distributed_streaming(self):
        streamed = {}
        source_tensor = torch.tensor([1])

        class FakeStreamer:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def stream_files(self, files, **kwargs):
                streamed.update(kwargs)
                self.files_to_tensors_metadata = {files[0]: [object()]}

            def get_tensors(self):
                yield "weight", source_tensor

        groups = (
            ("default", _DEFAULT_LOAD_GROUP, True, True),
            ("world", SimpleNamespace(world_size=4), True, True),
            ("stage", SimpleNamespace(world_size=2), True, False),
            ("single_rank", SimpleNamespace(world_size=1), True, False),
            ("local", None, True, False),
            ("uninitialized_default", _DEFAULT_LOAD_GROUP, False, True),
            ("uninitialized_local", None, False, False),
        )
        for name, group, initialized, collective in groups:
            for requested in (False, True):
                with (
                    self.subTest(group=name, requested=requested),
                    patch.object(loader_mod, "get_server_args", return_value=None),
                    patch.object(
                        torch.distributed, "is_initialized", return_value=initialized
                    ),
                    patch.object(
                        torch.distributed,
                        "get_world_size",
                        return_value=4,
                        side_effect=None if initialized else AssertionError,
                    ),
                    patch.object(
                        torch.distributed,
                        "get_rank",
                        return_value=3,
                        side_effect=None if initialized else AssertionError,
                    ),
                    patch.object(weight_utils, "is_cuda_alike", return_value=True),
                    patch.dict(
                        sys.modules,
                        {
                            "runai_model_streamer": SimpleNamespace(
                                SafetensorsStreamer=FakeStreamer
                            )
                        },
                    ),
                ):
                    loader = loader_mod.RunaiModelStreamerLoader(
                        LoadConfig(
                            load_format=LoadFormat.RUNAI_STREAMER, load_group=group
                        )
                    )
                    loader._is_distributed = requested
                    loader.target_device_str = "cuda:3"
                    source = loader.Source("unused", None, prefix="draft.")
                    with patch.object(
                        loader,
                        "_prepare_weights",
                        return_value=("unused", ["model.safetensors"]),
                    ):
                        weights = list(loader._get_weights_iterator(source))

                distributed = requested and collective
                self.assertEqual(streamed["is_distributed"], distributed)
                self.assertEqual(streamed["device"], "cuda:3" if distributed else "cpu")
                self.assertEqual(weights[0][0], "draft.weight")
                self.assertIs(weights[0][1], source_tensor)

    def test_deepseek_clone_only_clones_marked_tensors(self):
        unmarked = torch.tensor([1], dtype=torch.int32)

        self.assertIs(
            deepseek_weight_loader._clone_if_runai_streamed_tensor(unmarked),
            unmarked,
        )

        marked = torch.tensor([1], dtype=torch.int32)
        setattr(marked, weight_utils.RUNAI_STREAMER_TENSOR_ATTR, True)

        cloned = deepseek_weight_loader._clone_if_runai_streamed_tensor(marked)

        self.assertIsNot(cloned, marked)
        marked.fill_(2)
        self.assertEqual(cloned.item(), 1)

    def test_runai_streamed_tensor_is_consumed_before_buffer_reuse(self):
        def consume_view(mark_as_runai: bool):
            shared_buffer = torch.tensor([1], dtype=torch.int32)
            view = shared_buffer[:]
            if mark_as_runai:
                setattr(view, weight_utils.RUNAI_STREAMER_TENSOR_ATTR, True)

            release_worker = threading.Event()
            observed = []
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
                # Keep the sole worker busy so an async consumer cannot read the
                # zero-copy view until after the simulated streamer buffer reuse.
                blocker = executor.submit(release_worker.wait)
                model_loader_utils.maybe_executor_submit(
                    executor=executor,
                    futures=futures,
                    use_async=model_loader_utils.should_async_load(view),
                    func=lambda tensor: observed.append(tensor.item()),
                    func_args=(view,),
                )
                shared_buffer.fill_(2)
                release_worker.set()
                blocker.result()
                for future in futures:
                    future.result()

            return observed, len(futures)

        # The control demonstrates the race: an async consumer observes the
        # overwritten buffer rather than the value present when it was queued.
        self.assertEqual(consume_view(mark_as_runai=False), ([2], 1))
        # A RunAI-tagged view is consumed inline before the buffer is reused.
        self.assertEqual(consume_view(mark_as_runai=True), ([1], 0))

    def test_deepseek_v4_streaming_dequant_fp8_wo_a_pairs_weight_and_scale(self):
        weight = torch.eye(128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.ones((1, 1), dtype=torch.float32)

        for weights in (
            [
                ("layers.0.attn.wo_a.scale", scale),
                ("layers.0.attn.wo_a.weight", weight),
                ("layers.0.attn.wq.weight", torch.tensor([3])),
            ],
            [
                ("layers.0.attn.wo_a.weight", weight),
                ("layers.0.attn.wq.weight", torch.tensor([3])),
                ("layers.0.attn.wo_a.scale", scale),
            ],
        ):
            converted = list(_dequant_fp8_wo_a_streaming(weights))

            converted_names = [name for name, _ in converted]
            self.assertIn("layers.0.attn.wo_a.weight", converted_names)
            self.assertNotIn("layers.0.attn.wo_a.scale", converted_names)
            converted_weight = dict(converted)["layers.0.attn.wo_a.weight"]
            self.assertEqual(converted_weight.dtype, torch.bfloat16)

    def test_deepseek_v4_streaming_dequant_matches_legacy_by_name(self):
        weight = torch.eye(128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.ones((1, 1), dtype=torch.float32)
        ordinary = torch.tensor([3])
        weights = [
            ("layers.0.attn.wo_a.weight", weight),
            ("layers.0.attn.wq.weight", ordinary),
            ("layers.0.attn.wo_a.scale", scale),
        ]

        legacy = list(_dequant_fp8_wo_a(weights))
        streaming = list(_dequant_fp8_wo_a_streaming(weights))

        self.assertNotEqual(
            [name for name, _ in legacy], [name for name, _ in streaming]
        )
        self.assertEqual(set(dict(legacy)), set(dict(streaming)))
        for name, legacy_tensor in dict(legacy).items():
            torch.testing.assert_close(legacy_tensor, dict(streaming)[name])

    def test_deepseek_v4_streaming_dequant_clones_pending_runai_tensors(self):
        weight = torch.eye(128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.ones((1, 1), dtype=torch.float32)
        setattr(scale, weight_utils.RUNAI_STREAMER_TENSOR_ATTR, True)

        def weights():
            yield "layers.0.attn.wo_a.scale", scale
            scale.fill_(0)
            yield "layers.0.attn.wo_a.weight", weight

        converted = dict(_dequant_fp8_wo_a_streaming(weights()))

        converted_weight = converted["layers.0.attn.wo_a.weight"]
        self.assertGreater(converted_weight.abs().sum().item(), 0)

    def test_deepseek_v4_dspark_load_weights_streams_wo_a_dequant(self):
        weight = torch.eye(128, dtype=torch.float32).to(torch.float8_e4m3fn)
        scale = torch.ones((1, 1), dtype=torch.float32)
        setattr(scale, weight_utils.RUNAI_STREAMER_TENSOR_ATTR, True)
        loaded_weights = []

        def weight_loader(_param, loaded_weight):
            loaded_weights.append(loaded_weight)

        param = SimpleNamespace(weight_loader=weight_loader)
        remapper = SimpleNamespace(confidence_head=None)
        model = SimpleNamespace(
            config=SimpleNamespace(n_routed_experts=1),
            num_fused_shared_experts=0,
            named_parameters=lambda: [
                ("stages.0.self_attn.wo_a.weight", param),
            ],
            _remap_dspark_weight_name=lambda name: (
                DeepseekV4ForCausalLMDSpark._remap_dspark_weight_name(remapper, name)
            ),
            _assert_confidence_head_loaded=lambda **_kwargs: None,
            post_load_weights=lambda: None,
        )

        def weights():
            yield "mtp.0.attn.wo_a.scale", scale
            scale.fill_(0)
            yield "mtp.0.attn.wo_a.weight", weight

        DeepseekV4ForCausalLMDSpark.load_weights(model, weights())

        self.assertEqual(len(loaded_weights), 1)
        self.assertEqual(loaded_weights[0].dtype, torch.bfloat16)
        self.assertGreater(loaded_weights[0].abs().sum().item(), 0)

    def test_deepseek_v4_streaming_dequant_preserves_missing_scale_behavior(self):
        weight = torch.eye(128, dtype=torch.float32).to(torch.float8_e4m3fn)
        ordinary = torch.tensor([3])

        converted = dict(
            _dequant_fp8_wo_a_streaming(
                [
                    ("layers.0.attn.wo_a.weight", weight),
                    ("layers.0.attn.wq.weight", ordinary),
                ]
            )
        )

        self.assertIs(converted["layers.0.attn.wo_a.weight"], weight)
        self.assertIs(converted["layers.0.attn.wq.weight"], ordinary)

        with self.assertRaises(AssertionError):
            list(
                _dequant_fp8_wo_a_streaming(
                    [
                        ("layers.0.attn.wo_a.weight", weight),
                        ("layers.1.attn.wo_a.scale", torch.ones((1, 1))),
                    ]
                )
            )

    def test_get_model_loader_uses_runai_for_prequantized_modelopt(self):
        load_config = LoadConfig(
            load_format=LoadFormat.RUNAI_STREAMER,
            model_loader_extra_config={},
        )
        model_config = cast(
            ModelConfig,
            SimpleNamespace(
                quantization="modelopt_fp4",
                modelopt_quant=False,
                _is_already_quantized=lambda: True,
            ),
        )

        model_loader = loader_mod.get_model_loader(load_config, model_config)

        self.assertIsInstance(model_loader, loader_mod.RunaiModelStreamerLoader)

    def test_get_model_loader_uses_remote_instance_for_prequantized_modelopt(self):
        load_config = LoadConfig(
            load_format=LoadFormat.REMOTE_INSTANCE,
            model_loader_extra_config={},
        )
        model_config = cast(
            ModelConfig,
            SimpleNamespace(
                quantization="modelopt_fp4",
                modelopt_quant=False,
                _is_already_quantized=lambda: True,
            ),
        )

        model_loader = loader_mod.get_model_loader(load_config, model_config)

        self.assertIsInstance(model_loader, loader_mod.RemoteInstanceModelLoader)

    def test_cpu_offload_split_indexed_direct_engram_and_gpu_pass(self):
        with tempfile.TemporaryDirectory() as model_path:
            gpu_shard = os.path.join(model_path, "model-00001-of-00002.safetensors")
            engram_shard = os.path.join(model_path, "model-00002-of-00002.safetensors")
            save_file(
                {
                    "layers.0.attn.weight": torch.ones(4, 4),
                    "layers.1.engram.embed.scale": torch.arange(
                        16, dtype=torch.uint8
                    ).reshape(4, 4),
                },
                gpu_shard,
            )
            save_file(
                {
                    "layers.1.engram.embed.weight": torch.arange(
                        64, dtype=torch.uint8
                    ).reshape(4, 16),
                },
                engram_shard,
            )
            with open(
                os.path.join(model_path, "model.safetensors.index.json"), "w"
            ) as f:
                json.dump(
                    {
                        "weight_map": {
                            "layers.0.attn.weight": "model-00001-of-00002.safetensors",
                            "layers.1.engram.embed.scale": "model-00001-of-00002.safetensors",
                            "layers.1.engram.embed.weight": "model-00002-of-00002.safetensors",
                        }
                    },
                    f,
                )

            shared_weight = torch.zeros((4, 16), dtype=torch.uint8)
            shared_scale = torch.zeros((4, 4), dtype=torch.uint8)
            dst_w = shared_weight[1:3]
            dst_s = shared_scale[1:3]
            p_engram_w = torch.nn.Parameter(shared_weight, requires_grad=False)
            p_engram_s = torch.nn.Parameter(shared_scale, requires_grad=False)
            p_engram_w._runai_cpu_direct_load = ((1, 3), dst_w)
            p_engram_s._runai_cpu_direct_load = ((1, 3), dst_s)

            class _FakeGpuParam:
                device = torch.device("meta")

            # Use a fake model whose CPU params use HF-prefixed names while checkpoint uses native names.
            model = SimpleNamespace(
                allow_patterns_overrides=None,
                safetensors_index_file="model.safetensors.index.json",
                named_parameters=lambda: [
                    ("model.layers.1.engram.embed.weight", p_engram_w),
                    ("model.layers.1.engram.embed.scale", p_engram_s),
                ],
                remap_weight_name_to_dpsk_hf_format=lambda name: (
                    "model." + name if name.startswith("layers.") else name
                ),
            )
            config = SimpleNamespace(
                model_path=model_path,
                revision=None,
                hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLM"]),
            )
            loader = loader_mod.RunaiModelStreamerLoader(
                LoadConfig(
                    load_format=LoadFormat.RUNAI_STREAMER,
                    model_loader_extra_config={"distributed": True},
                )
            )
            loader.target_device_str = "cuda:0"

            calls = []

            def fake_iter(
                files, is_distributed, device, load_group=_DEFAULT_LOAD_GROUP, **kwargs
            ):
                calls.append((list(files), is_distributed, device, kwargs))
                if kwargs.get("tensor_dsts"):
                    for k, dst in kwargs["tensor_dsts"].items():
                        yield k, dst
                else:
                    yield "layers.0.attn.weight", torch.ones(4, 4)

            with (
                patch.object(loader_mod, "get_server_args", return_value=None),
                patch.object(
                    runai_utils,
                    "list_safetensors",
                    side_effect=lambda path: [gpu_shard, engram_shard],
                ),
                patch.object(
                    weight_utils,
                    "runai_streamer_capabilities",
                    return_value=(True, True, True),
                ),
                patch.object(
                    weight_utils,
                    "runai_safetensors_weights_iterator",
                    side_effect=fake_iter,
                ),
            ):
                yielded = list(loader._get_all_weights(config, model))

            self.assertEqual(len(calls), 2)
            # Pass 1 (GPU): only gpu_shard (engram-only shard omitted!), distributed=True, excludes shared scale.
            gpu_files, gpu_dist, gpu_dev, gpu_kw = calls[0]
            self.assertEqual(gpu_files, [gpu_shard])
            self.assertTrue(gpu_dist)
            self.assertEqual(gpu_dev, "cuda:0")
            self.assertEqual(
                gpu_kw["exclude_tensor_names"], ["layers.1.engram.embed.scale"]
            )

            # Pass 2 (CPU direct-dst): both shards carrying engram weight/scale, distributed=False, device="cpu".
            cpu_files, cpu_dist, cpu_dev, cpu_kw = calls[1]
            self.assertEqual(cpu_files, [gpu_shard, engram_shard])
            self.assertFalse(cpu_dist)
            self.assertEqual(cpu_dev, "cpu")
            self.assertEqual(
                set(cpu_kw["tensor_names"]),
                {"layers.1.engram.embed.weight", "layers.1.engram.embed.scale"},
            )
            self.assertEqual(
                cpu_kw["tensor_slices"],
                {
                    "layers.1.engram.embed.weight": (1, 3),
                    "layers.1.engram.embed.scale": (1, 3),
                },
            )
            self.assertIs(cpu_kw["tensor_dsts"]["layers.1.engram.embed.weight"], dst_w)
            self.assertIs(cpu_kw["tensor_dsts"]["layers.1.engram.embed.scale"], dst_s)
            self.assertEqual(
                {name for name, _ in yielded},
                {
                    "layers.0.attn.weight",
                    "layers.1.engram.embed.weight",
                    "layers.1.engram.embed.scale",
                },
            )

    def test_cpu_offload_split_unindexed_and_plain_cpu_param(self):
        with tempfile.TemporaryDirectory() as model_path:
            shard = os.path.join(model_path, "model.safetensors")
            save_file(
                {
                    "gpu.weight": torch.ones(2, 2),
                    "model.cpu_buf.weight": torch.ones(4, 4),
                },
                shard,
            )
            p_cpu = torch.nn.Parameter(torch.zeros(4, 4), requires_grad=False)
            model = SimpleNamespace(
                allow_patterns_overrides=None,
                safetensors_index_file=None,
                named_parameters=lambda: [("model.cpu_buf.weight", p_cpu)],
            )
            config = SimpleNamespace(
                model_path=model_path,
                revision=None,
                hf_config=SimpleNamespace(architectures=["TestModel"]),
            )
            loader = loader_mod.RunaiModelStreamerLoader(
                LoadConfig(
                    load_format=LoadFormat.RUNAI_STREAMER,
                    model_loader_extra_config={"distributed": True},
                )
            )
            loader.target_device_str = "cuda:0"
            calls = []

            def fake_iter(
                files, is_distributed, device, load_group=_DEFAULT_LOAD_GROUP, **kwargs
            ):
                calls.append((list(files), is_distributed, device, kwargs))
                if (
                    "excluded_names_out" in kwargs
                    and kwargs["excluded_names_out"] is not None
                ):
                    kwargs["excluded_names_out"].add("model.cpu_buf.weight")
                    yield "gpu.weight", torch.ones(2, 2)
                else:
                    yield "model.cpu_buf.weight", torch.ones(4, 4)

            with (
                patch.object(loader_mod, "get_server_args", return_value=None),
                patch.object(runai_utils, "list_safetensors", return_value=[shard]),
                patch.object(
                    weight_utils,
                    "runai_streamer_capabilities",
                    return_value=(True, True, True),
                ),
                patch.object(
                    weight_utils,
                    "runai_safetensors_weights_iterator",
                    side_effect=fake_iter,
                ),
            ):
                yielded = list(loader._get_all_weights(config, model))

            self.assertEqual(len(calls), 2)
            self.assertTrue(calls[0][1])
            self.assertEqual(calls[0][2], "cuda:0")
            self.assertIn("model.cpu_buf.weight", calls[0][3]["exclude_tensor_names"])
            self.assertFalse(calls[1][1])
            self.assertEqual(calls[1][2], "cpu")
            self.assertEqual(calls[1][3]["tensor_names"], ["model.cpu_buf.weight"])
            self.assertEqual(calls[1][3]["memory_limit"], 0)
            self.assertEqual(
                [n for n, _ in yielded], ["gpu.weight", "model.cpu_buf.weight"]
            )

    def test_engram_load_rows_in_place_fast_path(self):
        from sglang.srt.layers.engram import EngramEmbedding

        embed = object.__new__(EngramEmbedding)
        torch.nn.Module.__init__(embed)
        embed.row_start = 2
        embed.rows = 3
        embed.host_table = SimpleNamespace(dirty=False, replication="per_node")
        shared = torch.zeros((6, 8), dtype=torch.uint8)
        embed.weight = torch.nn.Parameter(shared, requires_grad=False)
        embed.scale = torch.nn.Parameter(
            torch.zeros((6, 1), dtype=torch.uint8), requires_grad=False
        )

        dst = shared[2:5]
        dst.fill_(7)
        embed._load_rows(embed.weight, dst)

        self.assertTrue(embed.host_table.dirty)
        self.assertTrue(torch.all(shared[2:5] == 7))
        self.assertEqual(shared[:2].sum().item(), 0)
        self.assertEqual(shared[5:].sum().item(), 0)

    def test_direct_dst_tensors_are_not_marked_as_reused_runai_buffers(self):
        dst = torch.zeros((2, 4), dtype=torch.float32)

        class FakeStreamer:
            def __enter__(self):
                return self

            def __exit__(self, *_args):
                pass

            def stream_files(self, files, **kwargs):
                self.files_to_tensors_metadata = {0: [object()]}
                self.kwargs = kwargs

            def get_tensors(self):
                dst.fill_(3.0)
                yield "engram.weight", dst

        with patch.dict(
            sys.modules,
            {"runai_model_streamer": SimpleNamespace(SafetensorsStreamer=FakeStreamer)},
        ):
            weights = list(
                weight_utils.runai_safetensors_weights_iterator(
                    ["model.safetensors"],
                    is_distributed=False,
                    device="cpu",
                    tensor_names=["engram.weight"],
                    tensor_slices={"engram.weight": (0, 2)},
                    tensor_dsts={"engram.weight": dst},
                )
            )

        self.assertEqual(len(weights), 1)
        self.assertIs(weights[0][1], dst)
        self.assertFalse(getattr(dst, weight_utils.RUNAI_STREAMER_TENSOR_ATTR, False))

    def test_nextn_filter_checkpoint_weight_names_and_indexed_shard_filtering(self):
        from sglang.srt.models.deepseek_v4_nextn import DeepseekV4ForCausalLMNextN

        draft_model = object.__new__(DeepseekV4ForCausalLMNextN)
        torch.nn.Module.__init__(draft_model)
        draft_model.config = SimpleNamespace(
            num_hidden_layers=60,
            num_nextn_predict_layers=1,
        )

        ckpt_names = [
            "embed.tok_emb.weight",
            "layers.0.engram.weight",
            "layers.0.engram.scale",
            "layers.59.ffn.experts.0.w1.weight",
            "norm.weight",
            "head.weight",
            "mtp.0.e_proj.weight",
            "mtp.0.e_proj.scale",
            "mtp.0.h_proj.weight",
            "mtp.0.enorm.weight",
            "mtp.0.hnorm.weight",
            "mtp.0.norm.weight",
            "mtp.0.hc_head_fn",
            "mtp.0.attn.wo_a.weight",
            "mtp.0.attn.wo_a.scale",
            "mtp.0.ffn.experts.0.w1.weight",
            "mtp.0.head.weight",
            "mtp.0.emb.tok_emb.weight",
            "model.layers.60.self_attn.rotary_emb.inv_freq",
        ]
        kept = draft_model.filter_checkpoint_weight_names(ckpt_names)
        self.assertEqual(
            kept,
            {
                "mtp.0.e_proj.weight",
                "mtp.0.e_proj.scale",
                "mtp.0.h_proj.weight",
                "mtp.0.enorm.weight",
                "mtp.0.hnorm.weight",
                "mtp.0.norm.weight",
                "mtp.0.hc_head_fn",
                "mtp.0.attn.wo_a.weight",
                "mtp.0.attn.wo_a.scale",
                "mtp.0.ffn.experts.0.w1.weight",
            },
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            shard_main = os.path.join(tmpdir, "model-00001-of-00003.safetensors")
            shard_engram = os.path.join(tmpdir, "model-00002-of-00003.safetensors")
            shard_mtp = os.path.join(tmpdir, "model-00003-of-00003.safetensors")
            for p in (shard_main, shard_engram, shard_mtp):
                with open(p, "wb") as f:
                    f.write(b"")
            index = {
                "weight_map": {
                    "layers.59.ffn.experts.0.w1.weight": os.path.basename(shard_main),
                    "layers.0.engram.weight": os.path.basename(shard_engram),
                    # Mix an Engram tensor into the MTP shard too to verify tensor_names excludes it.
                    "layers.1.engram.weight": os.path.basename(shard_mtp),
                    "mtp.0.e_proj.weight": os.path.basename(shard_mtp),
                    "mtp.0.norm.weight": os.path.basename(shard_mtp),
                }
            }
            with open(os.path.join(tmpdir, "model.safetensors.index.json"), "w") as f:
                json.dump(index, f)

            config = SimpleNamespace(
                model_path=tmpdir,
                revision=None,
                hf_config=SimpleNamespace(architectures=["DeepseekV4ForCausalLMNextN"]),
            )
            loader = loader_mod.RunaiModelStreamerLoader(
                LoadConfig(
                    load_format=LoadFormat.RUNAI_STREAMER,
                    model_loader_extra_config={"distributed": True},
                )
            )
            loader.target_device_str = "cuda:0"
            calls = []

            def fake_iter(
                files, is_distributed, device, load_group=_DEFAULT_LOAD_GROUP, **kwargs
            ):
                calls.append((list(files), is_distributed, device, kwargs))
                for name in kwargs.get("tensor_names", []):
                    yield name, torch.ones(2, 2)

            with (
                patch.object(loader_mod, "get_server_args", return_value=None),
                patch.object(
                    runai_utils,
                    "list_safetensors",
                    return_value=[shard_main, shard_engram, shard_mtp],
                ),
                patch.object(
                    weight_utils,
                    "runai_streamer_supports_tensor_names",
                    return_value=True,
                ),
                patch.object(
                    weight_utils,
                    "runai_safetensors_weights_iterator",
                    side_effect=fake_iter,
                ),
            ):
                yielded = list(loader._get_all_weights(config, draft_model))

            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0][0], [shard_mtp])
            self.assertTrue(calls[0][1])
            self.assertEqual(calls[0][2], "cuda:0")
            self.assertEqual(
                set(calls[0][3]["tensor_names"]),
                {"mtp.0.e_proj.weight", "mtp.0.norm.weight"},
            )
            self.assertEqual(
                {n for n, _ in yielded}, {"mtp.0.e_proj.weight", "mtp.0.norm.weight"}
            )


if __name__ == "__main__":
    unittest.main()
