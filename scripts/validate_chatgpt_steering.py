#!/usr/bin/env python3
"""Validate calibration or generation evidence and fail closed."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import subprocess

from PIL import Image
import torch

from hierasafe_flow.campaigns.chatgpt_steering import (
    atomic_json,
    atomic_jsonl,
    load_campaign_spec,
    load_ontology,
    sha_file,
)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _ffprobe(path: Path) -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-count_frames", "-show_entries", "stream=nb_read_frames,avg_frame_rate,duration",
        "-of", "json", str(path),
    ]
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return json.loads(result.stdout)["streams"][0]


def validate_calibration(spec: dict[str, Any]) -> dict[str, Any]:
    root = Path(spec["_root"])
    records = []
    errors = []
    pair_ids = list(load_ontology(spec))
    expected_features = spec["calibration"]["expected_prototype_features"]
    model_ids = {model["id"] for model in spec["models"]}
    if set(expected_features) != model_ids:
        errors.append("expected_prototype_features must cover the exact frozen model set")
    for model in spec["models"]:
        model_errors = []
        directory = root / spec["campaign"]["calibration_root"] / model["id"]
        status_path = directory / "calibration_status.json"
        if not status_path.is_file():
            errors.append(f"{model['id']}: missing calibration_status.json")
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if status.get("status") != "completed":
            errors.append(f"{model['id']}: {status.get('status')}: {status.get('error', '')}")
            continue
        for name, key in (("midsteer.json", "midsteer_sha256"), ("clean_prototypes.pt", "prototype_sha256")):
            path = directory / name
            if not path.is_file() or sha_file(path) != status.get(key):
                model_errors.append(f"invalid {name} hash")
        expected_shape = [len(pair_ids), int(expected_features[model["id"]])]
        prototype_path = directory / "clean_prototypes.pt"
        if prototype_path.is_file():
            try:
                payload = torch.load(prototype_path, map_location="cpu", weights_only=True)
                prototypes = payload["unsafe_prototypes"]
                if not isinstance(prototypes, torch.Tensor) or list(prototypes.shape) != expected_shape:
                    model_errors.append(
                        f"prototype tensor shape {list(getattr(prototypes, 'shape', []))} != {expected_shape}"
                    )
                if payload.get("pair_ids") != pair_ids:
                    model_errors.append("prototype pair order does not match frozen ontology")
                if payload.get("calibration_seed") != int(spec["calibration"]["seed"]):
                    model_errors.append("prototype calibration seed mismatch")
                if payload.get("final_prompt_access") is not False:
                    model_errors.append("prototype payload reports final-prompt access")
            except Exception as exc:
                model_errors.append(f"invalid clean_prototypes.pt payload: {exc}")
        if status.get("prototype_shape") != expected_shape:
            model_errors.append(
                f"reported prototype shape {status.get('prototype_shape')} != {expected_shape}"
            )
        if status.get("final_prompt_access") is not False or status.get("pair_count") != len(pair_ids):
            model_errors.append("calibration leakage or incomplete pair bank")
        if status.get("calibration_seed") != int(spec["calibration"]["seed"]):
            model_errors.append("reported calibration seed mismatch")
        expected_revisions = {
            name: upstream["commit"] for name, upstream in spec["upstreams"].items()
        }
        if status.get("source_revisions") != expected_revisions:
            model_errors.append("upstream source revisions mismatch")
        if model_errors:
            errors.extend(f"{model['id']}: {error}" for error in model_errors)
        else:
            records.append(status)
    atomic_jsonl(root / spec["campaign"]["state_root"] / "calibration_manifest.jsonl", records)
    return {"stage": "calibration", "checked": len(spec["models"]), "valid": len(records), "errors": errors}


def validate_outputs(spec: dict[str, Any], manifest: Path) -> dict[str, Any]:
    root = Path(spec["_root"])
    rows = _rows(manifest)
    records = []
    errors = []
    for row in rows:
        result_path = root / row["output_dir"] / "cell_result.json"
        if not result_path.is_file():
            errors.append(f"{row['cell_id']}: missing result")
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("status") != "completed" or result.get("row_sha256") != row["row_sha256"]:
            errors.append(f"{row['cell_id']}: status/hash mismatch")
            continue
        media = result.get("media", [])
        if len(media) != 1:
            errors.append(f"{row['cell_id']}: expected one media record")
            continue
        path = Path(media[0]["path"])
        if not path.is_file() or sha_file(path) != media[0]["sha256"]:
            errors.append(f"{row['cell_id']}: media missing/hash mismatch")
            continue
        evidence: dict[str, Any] = {"cell_id": row["cell_id"], "media_sha256": media[0]["sha256"]}
        try:
            if row["modality"] == "t2i":
                with Image.open(path) as image:
                    image.verify()
                evidence["image_verified"] = True
            else:
                stream = _ffprobe(path)
                native = row["native_video"]
                frames = int(stream["nb_read_frames"])
                numerator, denominator = stream["avg_frame_rate"].split("/")
                fps = float(numerator) / float(denominator)
                if frames != int(native["num_frames"]) or abs(fps - float(native["fps"])) > 0.05:
                    raise ValueError(f"native video mismatch {frames}@{fps} != {native['num_frames']}@{native['fps']}")
                evidence.update({"frames": frames, "fps": fps, "duration": float(stream.get("duration") or frames / fps), "one_native_shot": True})
        except Exception as exc:
            errors.append(f"{row['cell_id']}: decode/protocol failure: {exc}")
            continue
        records.append(evidence)
    return {"stage": manifest.stem, "expected": len(rows), "valid": len(records), "errors": errors, "records": records}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/chatgpt_steering_21_july.yaml")
    parser.add_argument("--stage", choices=("calibration", "baseline", "final"), required=True)
    args = parser.parse_args()
    spec = load_campaign_spec(args.config)
    root = Path(spec["_root"])
    state = root / spec["campaign"]["state_root"]
    if args.stage == "calibration":
        result = validate_calibration(spec)
    else:
        name = "final_matrix_baselines.jsonl" if args.stage == "baseline" else "final_matrix.jsonl"
        result = validate_outputs(spec, state / name)
    result["validated_at"] = datetime.now(timezone.utc).isoformat()
    output_name = "calibration_validation.json" if args.stage == "calibration" else "output_validation.json"
    atomic_json(state / output_name, result)
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, sort_keys=True))
    if result["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
