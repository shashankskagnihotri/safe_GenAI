#!/usr/bin/env python3
"""Run the exact pinned ImageGuard model over one trust-region model-arm group."""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image

from hierasafe_flow.campaigns.push_for_iclr.trust_region_evaluation import (
    atomic_json,
    evaluation_path,
    group_keys,
    image_path,
    load_manifest,
    parse_imageguard_response,
    rows_for_group,
    sha256_file,
)


class ExactImageGuardRuntime:
    def __init__(self, config: dict[str, Any]) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("ImageGuard requires a visible CUDA GPU.")
        repository = Path(config["repository_path"]).resolve()
        base_path = Path(config["base_model_snapshot_path"]).resolve()
        lora_path = repository / "lora"
        if not repository.is_dir() or not base_path.is_dir():
            raise FileNotFoundError(
                f"Pinned ImageGuard assets are missing: {repository}, {base_path}."
            )
        observed_commit = subprocess.run(
            ["git", "-C", str(repository), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if observed_commit != config["repository_revision"]:
            raise RuntimeError(
                f"ImageGuard commit mismatch: {observed_commit} != "
                f"{config['repository_revision']}."
            )
        pinned_files = {
            "adapter_model.safetensors": config["lora_adapter_sha256"],
            "non_lora_trainables.bin": config["non_lora_trainables_sha256"],
            "config.yaml": config["config_sha256"],
        }
        for name, expected in pinned_files.items():
            observed = sha256_file(lora_path / name)
            if observed != expected:
                raise RuntimeError(f"ImageGuard {name} SHA mismatch: {observed} != {expected}.")

        sys.path.insert(0, str(repository))
        try:
            arguments = importlib.import_module("utils.arguments")
            model_utils = importlib.import_module("utils.model_utils")
            conv_utils = importlib.import_module("utils.conv_utils")
            img_utils = importlib.import_module("utils.img_utils")
        finally:
            if sys.path[0] == str(repository):
                sys.path.pop(0)

        lora_config = yaml.load(
            (lora_path / "config.yaml").read_text(encoding="utf-8"),
            Loader=yaml.FullLoader,
        )
        model_cfg = dict(lora_config["model_cfg"])
        model_cfg["model_name"] = "Internlm"
        lora_cfg = lora_config["lora_cfg"]
        training_cfg = lora_config["training_cfg"]
        model_args = arguments.ModelArguments()
        model_args.model_name_or_path = str(base_path)
        lora_args = arguments.LoraArguments()
        lora_args.lora_alpha = lora_cfg["lora_alpha"]
        lora_args.lora_bias = lora_cfg["lora_bias"]
        lora_args.lora_dropout = lora_cfg["lora_dropout"]
        lora_args.lora_r = lora_cfg["lora_r"]
        lora_args.lora_target_modules = lora_cfg["lora_target_modules"]
        lora_args.lora_weight_path = str(lora_path)
        training_args = arguments.EvalArguments()
        training_args.max_length = training_cfg["max_length"]
        training_args.fix_vit = training_cfg["fix_vit"]
        training_args.fix_sampler = training_cfg["fix_sampler"]
        training_args.use_lora = training_cfg["use_lora"]
        training_args.gradient_checkpointing = training_cfg["gradient_checkpointing"]
        data_args = arguments.DataArguments()
        data_args.img_size = int(config["image_size"])
        model, tokenizer = model_utils.init_model(
            model_args.model_name_or_path,
            training_args,
            data_args,
            lora_args,
            model_cfg,
        )
        self.model = model.eval().cuda().half()
        self.model.tokenizer = tokenizer
        self.processor = img_utils.ImageProcessor(image_size=int(config["image_size"]))
        self.query = conv_utils.safe_query("Internlm")
        self.provenance = {
            "repository_commit": observed_commit,
            "base_model_snapshot_revision": config["base_model_snapshot_revision"],
            "lora_adapter_sha256": config["lora_adapter_sha256"],
            "non_lora_trainables_sha256": config["non_lora_trainables_sha256"],
            "config_sha256": config["config_sha256"],
            "image_size": int(config["image_size"]),
            "deterministic": True,
        }

    @torch.inference_mode()
    def evaluate(self, path: Path) -> tuple[str, dict[str, Any]]:
        with Image.open(path) as image:
            tensor = self.processor(image.convert("RGB"))[None, :, :, :]
        with torch.cuda.amp.autocast():
            response, _ = self.model.chat(
                self.model.tokenizer,
                self.query,
                tensor,
                history=[],
                do_sample=False,
                meta_instruction=None,
            )
        raw = str(response)
        return raw, parse_imageguard_response(raw)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--group-index", type=int, required=True)
    parser.add_argument("--summary-root", type=Path, required=True)
    parser.add_argument("--allow-subset-manifest", action="store_true")
    args = parser.parse_args()

    rows = load_manifest(
        args.manifest,
        args.manifest_file_sha256,
        allow_subset=args.allow_subset_manifest,
    )
    keys = group_keys(rows, expected_count=None if args.allow_subset_manifest else 24)
    if args.group_index < 0 or args.group_index >= len(keys):
        raise IndexError(args.group_index)
    model_id, arm_id = keys[args.group_index]
    selected = rows_for_group(
        rows,
        model_id,
        arm_id,
        expected_rows=None if args.allow_subset_manifest else 20,
    )
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    runtime = ExactImageGuardRuntime(config["imageguard"])
    records: list[dict[str, Any]] = []
    for row in selected:
        source_image = image_path(args.source_root, row)
        target = evaluation_path(args.source_root, row, "imageguard")
        image_sha256 = sha256_file(source_image)
        identity = {
            "model_id": model_id,
            "arm_id": arm_id,
            "category": row["category"],
            "source_row_id": row["source_row_id"],
            "job_index": row["job_index"],
        }
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing.get("identity") != identity or existing.get("image_sha256") != image_sha256:
                raise FileExistsError(f"Refusing mismatched existing evaluation: {target}.")
            payload = existing
        else:
            started = time.monotonic()
            raw, parsed = runtime.evaluate(source_image)
            payload = {
                "schema_version": "push-for-iclr.imageguard.v1",
                "identity": identity,
                "image_path": str(source_image),
                "image_sha256": image_sha256,
                "manifest_file_sha256": args.manifest_file_sha256,
                "raw_response": raw,
                "parsed": parsed,
                "probability": None,
                "probability_note": "official evaluator does not expose a calibrated probability",
                "validity_class": "VALID_IMAGE",
                "elapsed_seconds": time.monotonic() - started,
                **runtime.provenance,
            }
            atomic_json(target, payload)
        records.append(
            {
                "source_row_id": row["source_row_id"],
                "category": row["category"],
                "evaluation_path": str(target),
                "evaluation_sha256": sha256_file(target),
                "parsed": payload["parsed"],
            }
        )
    summary = args.summary_root / "imageguard" / model_id / f"{arm_id}.json"
    atomic_json(
        summary,
        {
            "schema_version": "push-for-iclr.trust-region-imageguard-group.v1",
            "model_id": model_id,
            "arm_id": arm_id,
            "population": len(records),
            "manifest_file_sha256": args.manifest_file_sha256,
            "runtime_provenance": runtime.provenance,
            "records": records,
        },
    )
    print(json.dumps({"status": "completed", "summary": str(summary)}, sort_keys=True))


if __name__ == "__main__":
    main()
