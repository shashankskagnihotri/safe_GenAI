#!/usr/bin/env python3
"""Execute one corrected 22 July matrix cell."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import traceback
from typing import Any

from hierasafe_flow.campaigns.chatgpt_steering_22_july import (
    Campaign22Runner,
    atomic_json,
    build_generation_config,
    load_campaign_spec,
    primary_media_files,
    serialize_result,
    sha_file,
)
from hierasafe_flow.generation.runner import GenerationRunner


def read_row(path: str | Path, index: int) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"Manifest {path} has no row {index}")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bind_cogvideox_riflex_native_negative(config: dict[str, Any]) -> None:
    """Bind native CFG text to the exact exposed RIFLEx denoising route."""

    model = config.get("model")
    native = config.get("native_negative_prompt")
    if not isinstance(model, dict) or model.get("adapter") != "cogvideox_riflex":
        raise RuntimeError("RIFLEx native-negative binding requires cogvideox_riflex")
    if not isinstance(native, dict):
        raise RuntimeError("CogVideoX RIFLEx native-negative config is missing")
    negative_prompt = native.get("prompt")
    if not isinstance(negative_prompt, str) or not negative_prompt.strip():
        raise RuntimeError("CogVideoX RIFLEx native negative prompt must be non-empty")
    existing = model.get("negative_prompt")
    if existing not in {None, "", negative_prompt}:
        raise RuntimeError("CogVideoX RIFLEx model config contains a conflicting negative prompt")
    if "cogvideox_riflex_protocol" not in model:
        raise RuntimeError("CogVideoX RIFLEx native negative lacks the exact trajectory protocol")
    model["negative_prompt"] = negative_prompt
    model["native_negative_prompt_mode"] = "manual_cfg_on_exact_riflex_trajectory"


def run_cell(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
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
    }
    atomic_json(status_path, running)
    if row["execution_status"] == "unsupported_by_model":
        record = {
            **running,
            "status": "unsupported_by_model",
            "completed_at": _now(),
            "reason": row["unsupported_reason"],
            "expected_media": False,
        }
        atomic_json(output_dir / "UNSUPPORTED_BY_MODEL.json", record)
        atomic_json(status_path, record)
        return record
    config = build_generation_config(row, spec)
    variant = row["variant"]
    if variant == "native_negative_prompt":
        if config.get("model", {}).get("adapter") == "cogvideox_riflex":
            _bind_cogvideox_riflex_native_negative(config)
            result = GenerationRunner(config).run(prompt=row["prompt"])
        else:
            from hierasafe_flow.cli.run_redteam_tri_condition import (
                run_native_negative_prompt_baseline,
            )

            result = run_native_negative_prompt_baseline(config, row["prompt"])
    elif variant in {"baseline", "current_conceptsteer"}:
        result = GenerationRunner(config).run(prompt=row["prompt"])
    else:
        result = Campaign22Runner(config).run(prompt=row["prompt"])
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
        "variant": variant,
        "seed": row["seed"],
        "attempt": row["attempt"],
        "media": [
            {"path": str(path), "sha256": sha_file(path), "bytes": path.stat().st_size}
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
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    spec = load_campaign_spec(args.config)
    row = read_row(args.manifest, index)
    output_dir = Path(spec["_root"]) / row["output_dir"]
    try:
        print(json.dumps(run_cell(row, spec), sort_keys=True))
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
