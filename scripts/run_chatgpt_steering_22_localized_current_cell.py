#!/usr/bin/env python3
"""Run one historical ConceptSteer cell with only localization enabled."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback
from typing import Any

from hierasafe_flow.campaigns.chatgpt_steering_22_july import (
    atomic_json,
    build_generation_config,
    load_campaign_spec,
    primary_media_files,
    serialize_result,
    sha_file,
)
from hierasafe_flow.generation.runner import GenerationRunner


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_row(path: str | Path, index: int) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"Manifest {path} has no row {index}")


def run_localized_cell(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    if row.get("variant") != "current_conceptsteer":
        raise ValueError("Localized historical runner accepts only current_conceptsteer")
    output_dir = Path(spec["_root"]) / row["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "cell_status.json"
    running = {
        "status": "running",
        "cell_id": row["cell_id"],
        "row_sha256": row["row_sha256"],
        "started_at": _now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "diagnostic_override": "steering.mask.enabled=false_to_true",
    }
    atomic_json(status_path, running)

    config = build_generation_config(row, spec)
    steering = config.get("steering")
    if not isinstance(steering, dict) or steering.get("enabled") is not True:
        raise RuntimeError("Historical ConceptSteer configuration is not enabled")
    mask = steering.get("mask")
    if not isinstance(mask, dict) or mask.get("enabled") is not False:
        raise RuntimeError("Expected sealed historical localization mask to start disabled")
    mask["enabled"] = True
    config.setdefault("_meta", {})["diagnostic_override"] = {
        "path": "steering.mask.enabled",
        "from": False,
        "to": True,
        "purpose": "localized_historical_cogvideox_diagnostic",
    }
    result = GenerationRunner(config).run(prompt=row["prompt"])
    media = primary_media_files(output_dir, row["modality"])
    if len(media) != 1:
        raise RuntimeError(
            f"Expected one media file in {output_dir}, found {len(media)}"
        )
    record = {
        "status": "completed",
        "cell_id": row["cell_id"],
        "row_sha256": row["row_sha256"],
        "completed_at": _now(),
        "model_id": row["model_id"],
        "prompt_id": row["prompt_id"],
        "prompt_sha256": row["prompt_sha256"],
        "variant": row["variant"],
        "seed": row["seed"],
        "attempt": row["attempt"],
        "diagnostic_override": "steering.mask.enabled=false_to_true",
        "media": [
            {
                "path": str(path),
                "sha256": sha_file(path),
                "bytes": path.stat().st_size,
            }
            for path in media
        ],
        "runner_result": serialize_result(result),
        "scheduler_step_unchanged": True,
    }
    atomic_json(output_dir / "cell_result.json", record)
    atomic_json(status_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/chatgpt_steering_22_july.yaml",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    spec = load_campaign_spec(args.config)
    row = _read_row(args.manifest, args.index)
    output_dir = Path(spec["_root"]) / row["output_dir"]
    try:
        print(json.dumps(run_localized_cell(row, spec), sort_keys=True))
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            output_dir / "cell_status.json",
            {
                "status": "failed",
                "cell_id": row["cell_id"],
                "row_sha256": row["row_sha256"],
                "failed_at": _now(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            },
        )
        raise


if __name__ == "__main__":
    main()
