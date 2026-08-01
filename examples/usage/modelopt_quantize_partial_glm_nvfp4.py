#!/usr/bin/env python3
"""Finish a partially quantized ModelOpt GLM checkpoint as NVFP4.

This converter is intended for checkpoints whose routed experts are already
ModelOpt NVFP4 while attention, dense/shared MLP, MTP, and lm_head weights are
still BF16.  It streams one tensor at a time, keeps existing packed tensors
bit-for-bit, and uses ModelOpt's NVFP4QTensor implementation for each remaining
eligible matrix.  No activation calibration is needed because SGLang's
``nvfp4_w4a8`` backend dynamically quantizes every GEMM input to FP8.

Embedding tables, normalization parameters, and MoE router weights remain
BF16.  They are not operands of the NVFP4-W4A8 GEMMs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

NVFP4_BLOCK_SIZE = 16
E2M1_MAX = 6.0
E4M3_MAX = 448.0
_CONFIG_FILES = {"config.json", "hf_quant_config.json", "glm_hf_quant_config.json"}


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2, ensure_ascii=False)
        f.write("\n")


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _is_floating_safetensors_dtype(dtype: str) -> bool:
    return dtype in {"BF16", "F16", "F32", "F64"}


def _is_router_or_embedding(module_name: str) -> bool:
    return module_name == "model.embed_tokens" or module_name.endswith(".mlp.gate")


def _should_quantize_weight(
    name: str,
    dtype: str,
    shape: list[int],
    source_names: set[str],
) -> bool:
    if not name.endswith(".weight") or len(shape) != 2:
        return False
    if not _is_floating_safetensors_dtype(dtype):
        return False
    module_name = name.removesuffix(".weight")
    if _is_router_or_embedding(module_name):
        return False
    if f"{module_name}.weight_scale" in source_names:
        return False
    # The runtime FP8 MMA consumes K=32 tiles.  GLM-5.2's eligible matrices
    # satisfy this naturally; reject rather than silently padding model dims.
    return shape[-1] % 32 == 0


def _quantize_slice(
    tensor_slice,
    shape: list[int],
    device: torch.device,
    chunk_rows: int,
    nvfp4_qtensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    rows, cols = shape
    global_amax = torch.zeros((), dtype=torch.float32, device=device)
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        chunk = tensor_slice[start:end].to(device=device)
        global_amax = torch.maximum(global_amax, chunk.abs().max().float())
        del chunk

    global_scale = global_amax / (E2M1_MAX * E4M3_MAX)
    if global_amax.item() == 0.0:
        global_scale.fill_(1.0)

    packed = torch.empty((rows, cols // 2), dtype=torch.uint8, device="cpu")
    block_scale = torch.empty(
        (rows, cols // NVFP4_BLOCK_SIZE),
        dtype=torch.float8_e4m3fn,
        device="cpu",
    )
    for start in range(0, rows, chunk_rows):
        end = min(start + chunk_rows, rows)
        chunk = tensor_slice[start:end].to(device=device)
        quantized, chunk_scale, _ = nvfp4_qtensor.quantize(
            chunk,
            NVFP4_BLOCK_SIZE,
            weights_scaling_factor_2=global_scale,
        )
        packed[start:end].copy_(quantized._quantized_data.to(device="cpu"))
        block_scale[start:end].copy_(chunk_scale.to(device="cpu"))
        del chunk, quantized, chunk_scale

    return packed, block_scale, global_scale.to(device="cpu")


def _updated_quant_config(
    source_path: Path, excluded_modules: list[str]
) -> dict[str, Any]:
    config = _load_json(source_path)
    if source_path.name == "config.json":
        quant = config.get("quantization_config")
        if not isinstance(quant, dict):
            raise ValueError("config.json has no ModelOpt quantization_config")
        quant["ignore"] = excluded_modules
        quant["quant_algo"] = "NVFP4"
        quant["quant_method"] = "modelopt"
    else:
        quant = config.get("quantization")
        if not isinstance(quant, dict):
            raise ValueError(f"{source_path.name} has no quantization object")
        quant["exclude_modules"] = excluded_modules
        quant["quant_algo"] = "NVFP4"
        quant["group_size"] = NVFP4_BLOCK_SIZE
    return config


def convert(args: argparse.Namespace) -> None:
    try:
        from modelopt.torch.quantization.qtensor import NVFP4QTensor
        from safetensors import safe_open
        from safetensors.torch import save_file
    except ImportError as exc:
        raise ImportError(
            "Install nvidia-modelopt and safetensors before running this converter "
            "(for this checkout: python -m pip install -e "
            "'/data/muqi/projects/Model-Optimizer')."
        ) from exc

    source_dir = Path(args.source_model).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not source_dir.is_dir():
        raise ValueError(f"source checkpoint does not exist: {source_dir}")
    if source_dir == output_dir or source_dir in output_dir.parents:
        raise ValueError("--output-dir must not be the source or a child of it")
    if output_dir.exists():
        raise FileExistsError(
            f"output directory already exists: {output_dir}; choose a new directory"
        )

    index_path = source_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        raise ValueError("streaming conversion requires model.safetensors.index.json")
    source_index = _load_json(index_path)
    source_weight_map = source_index.get("weight_map")
    if not isinstance(source_weight_map, dict):
        raise ValueError("invalid safetensors index: missing weight_map")
    source_names = set(source_weight_map)
    tensors_by_shard: dict[str, list[str]] = defaultdict(list)
    for name, shard in source_weight_map.items():
        tensors_by_shard[shard].append(name)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda was requested but CUDA is unavailable")
    output_dir.mkdir(parents=True)

    max_shard_bytes = int(args.max_shard_size_gb * 1024**3)
    pending: dict[str, torch.Tensor] = {}
    pending_bytes = 0
    temp_shards: list[Path] = []
    output_weight_map: dict[str, str] = {}
    total_size = 0
    quantized_modules: list[str] = []
    excluded_modules: list[str] = []

    def flush() -> None:
        nonlocal pending, pending_bytes
        if not pending:
            return
        filename = f"model-part-{len(temp_shards) + 1:05d}.safetensors"
        path = output_dir / filename
        save_file(pending, path, metadata={"format": "pt"})
        for tensor_name in pending:
            output_weight_map[tensor_name] = filename
        temp_shards.append(path)
        pending = {}
        pending_bytes = 0

    def add_tensor(name: str, tensor: torch.Tensor) -> None:
        nonlocal pending_bytes, total_size
        tensor = tensor.detach().to(device="cpu").contiguous()
        nbytes = _tensor_nbytes(tensor)
        if pending and pending_bytes + nbytes > max_shard_bytes:
            flush()
        pending[name] = tensor
        pending_bytes += nbytes
        total_size += nbytes

    for shard_index, (shard, names) in enumerate(tensors_by_shard.items(), start=1):
        print(f"[{shard_index}/{len(tensors_by_shard)}] reading {shard}", flush=True)
        with safe_open(source_dir / shard, framework="pt", device="cpu") as source_file:
            for name in names:
                tensor_slice = source_file.get_slice(name)
                shape = list(tensor_slice.get_shape())
                dtype = tensor_slice.get_dtype()
                if _should_quantize_weight(name, dtype, shape, source_names):
                    module_name = name.removesuffix(".weight")
                    print(f"  ModelOpt NVFP4: {name} {shape}", flush=True)
                    packed, scale, scale_2 = _quantize_slice(
                        tensor_slice,
                        shape,
                        device,
                        args.chunk_rows,
                        NVFP4QTensor,
                    )
                    add_tensor(name, packed)
                    add_tensor(f"{module_name}.weight_scale", scale)
                    add_tensor(f"{module_name}.weight_scale_2", scale_2)
                    # Dynamic FP8 runtime activation quantization ignores this
                    # legacy ModelOpt checkpoint scale, but the loader expects it.
                    add_tensor(f"{module_name}.input_scale", torch.tensor(1.0))
                    quantized_modules.append(module_name)
                else:
                    add_tensor(name, source_file.get_tensor(name))
                    if (
                        name.endswith(".weight")
                        and len(shape) == 2
                        and _is_floating_safetensors_dtype(dtype)
                    ):
                        excluded_modules.append(name.removesuffix(".weight"))
        if device.type == "cuda":
            torch.cuda.empty_cache()
    flush()

    total_shards = len(temp_shards)
    shard_renames: dict[str, str] = {}
    for index, temp_path in enumerate(temp_shards, start=1):
        final_name = f"model-{index:05d}-of-{total_shards:05d}.safetensors"
        os.replace(temp_path, output_dir / final_name)
        shard_renames[temp_path.name] = final_name
    output_weight_map = {
        name: shard_renames[shard] for name, shard in output_weight_map.items()
    }
    _dump_json(
        output_dir / "model.safetensors.index.json",
        {"metadata": {"total_size": total_size}, "weight_map": output_weight_map},
    )

    excluded_modules = sorted(set(excluded_modules))
    for entry in source_dir.iterdir():
        if entry.name == "model.safetensors.index.json" or entry.name in _CONFIG_FILES:
            continue
        if entry.name.startswith("model-") and entry.name.endswith(".safetensors"):
            continue
        destination = output_dir / entry.name
        if entry.is_dir():
            shutil.copytree(entry, destination)
        else:
            shutil.copy2(entry, destination)
    for config_name in _CONFIG_FILES:
        source_config = source_dir / config_name
        if source_config.is_file():
            _dump_json(
                output_dir / config_name,
                _updated_quant_config(source_config, excluded_modules),
            )

    manifest = {
        "source_model": str(source_dir),
        "modelopt_quantized_module_count": len(quantized_modules),
        "excluded_bf16_matrix_modules": excluded_modules,
        "activation_runtime": "dynamic_per_token_fp8_e4m3",
    }
    _dump_json(output_dir / "nvfp4_w4a8_conversion.json", manifest)
    print(
        f"Converted {len(quantized_modules)} BF16 matrices; kept "
        f"{len(excluded_modules)} matrix modules in BF16."
    )
    print(f"Wrote {total_shards} shards to {output_dir}")
    print("Serve with:")
    print(
        "python -m sglang.launch_server "
        f"--model-path {output_dir} --quantization modelopt_fp4 "
        "--fp4-gemm-backend nvfp4_w4a8 "
        "--moe-runner-backend nvfp4_w4a8 --moe-a2a-backend none"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--device",
        default="cuda",
        help="Device used for one-matrix-at-a-time ModelOpt quantization",
    )
    parser.add_argument("--chunk-rows", type=int, default=4096)
    parser.add_argument("--max-shard-size-gb", type=float, default=4.0)
    args = parser.parse_args()
    if args.chunk_rows <= 0:
        parser.error("--chunk-rows must be positive")
    if args.max_shard_size_gb <= 0:
        parser.error("--max-shard-size-gb must be positive")
    return args


if __name__ == "__main__":
    convert(parse_args())
