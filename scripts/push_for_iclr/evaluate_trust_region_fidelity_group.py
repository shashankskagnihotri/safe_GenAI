#!/usr/bin/env python3
"""Compute pinned CLIP prompt fidelity and DINO baseline retention for one group."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as functional
import yaml
from PIL import Image
from transformers import AutoImageProcessor, AutoModel, CLIPModel, CLIPProcessor

from hierasafe_flow.campaigns.push_for_iclr.trust_region_evaluation import (
    atomic_json,
    baseline_index,
    evaluation_path,
    group_keys,
    image_path,
    load_manifest,
    rows_for_group,
    sha256_file,
)


def _move(inputs: dict[str, torch.Tensor], device: torch.device) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in inputs.items()}


def _normalized(value: torch.Tensor) -> torch.Tensor:
    return functional.normalize(value.float(), dim=-1)


class FidelityRuntime:
    def __init__(self, config: dict[str, Any]) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("Fidelity evaluation requires a visible CUDA GPU.")
        torch_release = tuple(
            int(component)
            for component in torch.__version__.split("+")[0].split(".")[:2]
        )
        if torch_release < (2, 6):
            raise RuntimeError(
                "Torch >=2.6 is mandatory for CVE-2025-32434-safe loading of the "
                f"official CLIP checkpoint; observed {torch.__version__}."
            )
        self.device = torch.device("cuda")
        clip_cfg = config["clip"]
        dino_cfg = config["dino"]
        clip_path = Path(clip_cfg["snapshot_path"])
        dino_path = Path(dino_cfg["snapshot_path"])
        pinned_assets = (
            (clip_path / "config.json", clip_cfg["config_sha256"]),
            (clip_path / "pytorch_model.bin", clip_cfg["weights_sha256"]),
            (
                clip_path / "preprocessor_config.json",
                clip_cfg["preprocessor_config_sha256"],
            ),
            (clip_path / "tokenizer_config.json", clip_cfg["tokenizer_config_sha256"]),
            (clip_path / "vocab.json", clip_cfg["vocab_sha256"]),
            (clip_path / "merges.txt", clip_cfg["merges_sha256"]),
            (
                clip_path / "special_tokens_map.json",
                clip_cfg["special_tokens_map_sha256"],
            ),
            (dino_path / "config.json", dino_cfg["config_sha256"]),
            (dino_path / "model.safetensors", dino_cfg["weights_sha256"]),
            (
                dino_path / "preprocessor_config.json",
                dino_cfg["preprocessor_config_sha256"],
            ),
        )
        for path, expected in pinned_assets:
            observed = sha256_file(path)
            if observed != expected:
                raise RuntimeError(f"Pinned model config SHA mismatch: {path}: {observed}.")
        self.clip_processor = CLIPProcessor.from_pretrained(
            clip_path, local_files_only=True
        )
        self.clip_model = CLIPModel.from_pretrained(
            clip_path, local_files_only=True, torch_dtype=torch.float16
        ).eval().to(self.device)
        self.dino_processor = AutoImageProcessor.from_pretrained(
            dino_path, local_files_only=True
        )
        self.dino_model = AutoModel.from_pretrained(
            dino_path, local_files_only=True, torch_dtype=torch.float16
        ).eval().to(self.device)
        self.provenance = {
            "clip_snapshot_revision": clip_cfg["snapshot_revision"],
            "clip_config_sha256": clip_cfg["config_sha256"],
            "clip_weights_sha256": clip_cfg["weights_sha256"],
            "clip_preprocessor_config_sha256": clip_cfg[
                "preprocessor_config_sha256"
            ],
            "clip_tokenizer_config_sha256": clip_cfg["tokenizer_config_sha256"],
            "clip_vocab_sha256": clip_cfg["vocab_sha256"],
            "clip_merges_sha256": clip_cfg["merges_sha256"],
            "clip_special_tokens_map_sha256": clip_cfg[
                "special_tokens_map_sha256"
            ],
            "dino_snapshot_revision": dino_cfg["snapshot_revision"],
            "dino_config_sha256": dino_cfg["config_sha256"],
            "dino_weights_sha256": dino_cfg["weights_sha256"],
            "dino_preprocessor_config_sha256": dino_cfg[
                "preprocessor_config_sha256"
            ],
            "torch_dtype": "float16",
            "torch_version": torch.__version__,
        }

    @torch.inference_mode()
    def evaluate(self, target: Image.Image, baseline: Image.Image, prompt: str) -> dict[str, float]:
        clip_image_inputs = _move(
            self.clip_processor(images=target, return_tensors="pt"), self.device
        )
        clip_text_inputs = _move(
            self.clip_processor(
                text=[prompt], return_tensors="pt", padding=True, truncation=True
            ),
            self.device,
        )
        image_features = _normalized(self.clip_model.get_image_features(**clip_image_inputs))
        text_features = _normalized(self.clip_model.get_text_features(**clip_text_inputs))
        clip_cosine = float((image_features * text_features).sum(dim=-1).item())

        pair_inputs = _move(
            self.dino_processor(images=[target, baseline], return_tensors="pt"),
            self.device,
        )
        pair_outputs = self.dino_model(**pair_inputs)
        pair_features = _normalized(pair_outputs.last_hidden_state[:, 0, :])
        dino_cosine = float((pair_features[0] * pair_features[1]).sum().item())
        return {
            "clip_image_text_cosine": clip_cosine,
            "dino_cls_cosine_to_matched_r00": dino_cosine,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--group-index", type=int, required=True)
    parser.add_argument("--summary-root", type=Path, required=True)
    args = parser.parse_args()

    rows = load_manifest(args.manifest, args.manifest_file_sha256)
    keys = group_keys(rows)
    if args.group_index < 0 or args.group_index >= len(keys):
        raise IndexError(args.group_index)
    model_id, arm_id = keys[args.group_index]
    selected = rows_for_group(rows, model_id, arm_id)
    baselines = baseline_index(rows)
    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    runtime = FidelityRuntime(config["fidelity"])
    records: list[dict[str, Any]] = []
    for row in selected:
        source_image = image_path(args.source_root, row)
        baseline_row = baselines[(model_id, str(row["source_row_id"]))]
        baseline_image = image_path(args.source_root, baseline_row)
        target = evaluation_path(args.source_root, row, "fidelity")
        image_sha256 = sha256_file(source_image)
        baseline_sha256 = sha256_file(baseline_image)
        identity = {
            "model_id": model_id,
            "arm_id": arm_id,
            "category": row["category"],
            "source_row_id": row["source_row_id"],
            "job_index": row["job_index"],
        }
        if target.exists():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if (
                existing.get("identity") != identity
                or existing.get("image_sha256") != image_sha256
                or existing.get("baseline_image_sha256") != baseline_sha256
            ):
                raise FileExistsError(f"Refusing mismatched existing evaluation: {target}.")
            payload = existing
        else:
            started = time.monotonic()
            with Image.open(source_image) as target_image, Image.open(baseline_image) as base_image:
                metrics = runtime.evaluate(
                    target_image.convert("RGB"),
                    base_image.convert("RGB"),
                    str(row["original_prompt"]),
                )
            payload = {
                "schema_version": "push-for-iclr.trust-region-fidelity.v1",
                "identity": identity,
                "image_path": str(source_image),
                "image_sha256": image_sha256,
                "baseline_image_path": str(baseline_image),
                "baseline_image_sha256": baseline_sha256,
                "baseline_arm_id": "R00_BASELINE",
                "prompt_sha256": row["original_prompt_sha256"],
                "metrics": metrics,
                "elapsed_seconds": time.monotonic() - started,
                "provenance": runtime.provenance,
            }
            atomic_json(target, payload)
        records.append(
            {
                "source_row_id": row["source_row_id"],
                "category": row["category"],
                "evaluation_path": str(target),
                "evaluation_sha256": sha256_file(target),
                "metrics": payload["metrics"],
            }
        )
    summary = args.summary_root / "fidelity" / model_id / f"{arm_id}.json"
    atomic_json(
        summary,
        {
            "schema_version": "push-for-iclr.trust-region-fidelity-group.v1",
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
