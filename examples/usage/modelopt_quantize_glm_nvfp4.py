#!/usr/bin/env python3
"""Quantize every supported GLM-5.2 Linear to ModelOpt NVFP4.

This utility intentionally starts from the original BF16 checkpoint.  ModelOpt
marks a model as globally quantized after any quantizers are restored, so using
an already-partial NVFP4 export as ``--source-model`` would silently leave its
excluded BF16 layers unchanged.

Embeddings remain BF16 because lookup tables are not GEMMs.  Norms and routers
also stay in their original floating format.  Attention projections, dense
MLPs, shared/routed experts, MTP Linear modules, and lm_head are selected by
ModelOpt's NVFP4 ``Linear`` rule.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path

import torch


def _read_config(model_path: str) -> dict:
    path = Path(model_path) / "config.json"
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def _reject_prequantized_source(model_path: str) -> None:
    config = _read_config(model_path)
    quant = config.get("quantization_config") or {}
    quant_algo = str(quant.get("quant_algo", "")).upper()
    producer = quant.get("producer") or {}
    if "NVFP4" in quant_algo or producer.get("name") == "modelopt":
        raise ValueError(
            "--source-model is already a ModelOpt/quantized checkpoint. Start "
            "from the original BF16 GLM-5.2 checkpoint; ModelOpt cannot resume "
            "only the BF16 exclusions of a partial NVFP4 export safely."
        )


def _build_nvfp4_config(mtq):
    quant_cfg = copy.deepcopy(mtq.NVFP4_DEFAULT_CFG)
    if not isinstance(quant_cfg, dict):
        raise TypeError("ModelOpt NVFP4_DEFAULT_CFG must be a dictionary")
    rules = quant_cfg.setdefault("quant_cfg", {})
    # SGLang's NVFP4-W4A8 GEMM also handles ParallelLMHead.  ModelOpt's default
    # recipe may disable lm_head for accuracy, so explicitly opt it in to retain
    # the requested persistent-memory saving.  Embedding lookup remains BF16.
    if isinstance(rules, list):
        # ModelOpt 0.31+ represents quantizer rules as an ordered list.  The
        # default recipe disables lm_head near the end, so append a later rule
        # to re-enable it.  The wildcard weight/input rules earlier in the list
        # already carry the NVFP4 quantizer attributes.
        rules.extend(
            [
                {"quantizer_name": "*lm_head*", "enable": True},
                {"quantizer_name": "*embed_tokens*", "enable": False},
            ]
        )
    elif isinstance(rules, dict):
        # Compatibility with older ModelOpt releases.
        rules["*lm_head*"] = {"enable": True}
        rules["*embed_tokens*"] = {"enable": False}
    else:
        raise TypeError(
            "ModelOpt NVFP4_DEFAULT_CFG['quant_cfg'] must be a list or dictionary"
        )
    return quant_cfg


def quantize(args: argparse.Namespace) -> None:
    _reject_prequantized_source(args.source_model)
    if os.path.abspath(args.source_model) == os.path.abspath(args.output_dir):
        raise ValueError("--output-dir must differ from --source-model")

    try:
        import modelopt.torch.quantization as mtq
        from modelopt.torch.export import export_hf_checkpoint
        from modelopt.torch.utils.dataset_utils import (
            create_forward_loop,
            get_dataset_dataloader,
        )
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Install a GLM-5.2-compatible transformers build and NVIDIA "
            "ModelOpt (nvidia-modelopt[torch]) before running this utility."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.source_model, trust_remote_code=args.trust_remote_code, use_fast=True
    )
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.source_model,
        trust_remote_code=args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map=args.device_map,
        low_cpu_mem_usage=True,
    ).eval()

    try:
        calibration_device = next(
            p.device for p in model.parameters() if p.device.type == "cuda"
        )
    except StopIteration as exc:
        raise RuntimeError(
            "NVFP4 calibration requires at least one CUDA device"
        ) from exc

    dataloader = get_dataset_dataloader(
        dataset_name=args.calibration_dataset,
        tokenizer=tokenizer,
        batch_size=args.batch_size,
        num_samples=args.num_calibration_samples,
        device=calibration_device,
        include_labels=False,
    )
    forward_loop = create_forward_loop(dataloader=dataloader)
    mtq.quantize(model, _build_nvfp4_config(mtq), forward_loop=forward_loop)
    mtq.print_quant_summary(model)

    os.makedirs(args.output_dir, exist_ok=True)
    export_hf_checkpoint(model, export_dir=args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Exported full-Linear NVFP4 checkpoint to {args.output_dir}")
    print("Serve with:")
    print(
        "python -m sglang.launch_server "
        f"--model-path {args.output_dir} --quantization modelopt_fp4 "
        "--fp4-gemm-backend nvfp4_w4a8 "
        "--moe-runner-backend nvfp4_w4a8 --moe-a2a-backend none"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-model",
        required=True,
        help="Original, unquantized BF16 GLM-5.2 checkpoint",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device-map", default="auto")
    parser.add_argument("--calibration-dataset", default="cnn_dailymail")
    parser.add_argument("--num-calibration-samples", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--trust-remote-code", action=argparse.BooleanOptionalAction, default=True
    )
    return parser.parse_args()


if __name__ == "__main__":
    quantize(parse_args())
