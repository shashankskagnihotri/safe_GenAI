#!/usr/bin/env python3
"""Execute exactly one immutable final-matrix row and write terminal evidence."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import traceback

from hierasafe_flow.campaigns.chatgpt_steering import (
    CampaignGenerationRunner,
    atomic_json,
    build_generation_config,
    load_campaign_spec,
    media_files,
    serialize_result,
    sha_file,
)


def read_row(path: str | Path, index: int) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"Manifest {path} has no row {index}")


def _native_negative(config: dict[str, Any], row: dict[str, Any]) -> Any:
    from hierasafe_flow.cli.run_redteam_tri_condition import run_native_negative_prompt_baseline

    config["generation"]["negative_prompt"] = row["native_negative_prompt"]
    config["native_negative_prompt"] = row["native_negative_prompt"]
    return run_native_negative_prompt_baseline(config, row["prompt"])


def run_cell(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    root = Path(spec["_root"])
    output_dir = root / row["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    status_path = output_dir / "cell_status.json"
    atomic_json(
        status_path,
        {
            "status": "running",
            "cell_id": row["cell_id"],
            "row_sha256": row["row_sha256"],
            "started_at": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    )
    config = build_generation_config(row, spec)
    if row["variant"] == "native_negative_prompt":
        result = _native_negative(config, row)
    else:
        result = CampaignGenerationRunner(config).run(prompt=row["prompt"])
    media = media_files(output_dir)
    if len(media) != 1:
        raise RuntimeError(f"Expected exactly one final media file in {output_dir}, found {len(media)}")
    record = {
        "status": "completed",
        "cell_id": row["cell_id"],
        "row_sha256": row["row_sha256"],
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "model_id": row["model_id"],
        "prompt_id": row["prompt_id"],
        "prompt_sha256": row["prompt_sha256"],
        "variant": row["variant"],
        "seed": row["seed"],
        "native_video": row.get("native_video"),
        "media": [{"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size} for path in media],
        "runner_result": serialize_result(result),
        "scheduler_step_unchanged": True,
    }
    atomic_json(output_dir / "cell_result.json", record)
    atomic_json(status_path, record)
    return record


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/chatgpt_steering_21_july.yaml")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()
    index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    spec = load_campaign_spec(args.config)
    row = read_row(args.manifest, index)
    output_dir = Path(spec["_root"]) / row["output_dir"]
    try:
        print(json.dumps(run_cell(row, spec), sort_keys=True))
    except Exception as exc:
        atomic_json(
            output_dir / "cell_status.json",
            {
                "status": "failed",
                "cell_id": row["cell_id"],
                "row_sha256": row["row_sha256"],
                "failed_at": datetime.now(timezone.utc).isoformat(),
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
