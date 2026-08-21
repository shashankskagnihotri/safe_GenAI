#!/usr/bin/env python3
"""Submit each corrected calibration/generation row as an ordinary Slurm job."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any

from hierasafe_flow.campaigns.chatgpt_steering_22_july import (
    atomic_json,
    load_campaign_spec,
)


ALLOWED_ENVIRONMENTS = {"safe_genai_conceptsteer", "safe_genai_ltx23"}
CONDA_ROOT = Path("/ceph/sagnihot/miniconda3")


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def submit(script: Path, dependency: str | None = None) -> str:
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency=afterok:{dependency}")
    command.append(str(script))
    result = subprocess.run(command, check=True, text=True, capture_output=True)
    return result.stdout.strip().split(";", 1)[0]


def write_sbatch(
    *,
    path: Path,
    root: Path,
    job_name: str,
    manifest: Path,
    command_script: str,
    config_path: str,
    resources: dict[str, Any],
    environment_name: str,
    model_id: str,
) -> None:
    text = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={resources['partition']}
#SBATCH --gres={resources['gres']}
#SBATCH --cpus-per-task={int(resources['cpus'])}
#SBATCH --mem={int(resources['memory_gb'])}G
#SBATCH --time={resources['time']}
#SBATCH --output={root}/debugging/chatgpt_steering_22_july/slurm/%x_%j.out
#SBATCH --error={root}/debugging/chatgpt_steering_22_july/slurm/%x_%j.err

set -euo pipefail
cd {root}
if [ ! -f "{CONDA_ROOT}/etc/profile.d/conda.sh" ]; then
  echo "Required conda bootstrap is missing." >&2
  exit 1
fi
set +u
source "{CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "{environment_name}"
set -u
if [ "${{CONDA_DEFAULT_ENV:-}}" != "{environment_name}" ]; then
  echo "Wrong conda environment after activation: ${{CONDA_DEFAULT_ENV:-unset}}" >&2
  exit 1
fi
export PYTHONPATH="{root}/src:{root}:${{PYTHONPATH:-}}"
export PYTORCH_CUDA_ALLOC_CONF="${{PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}}"
export TOKENIZERS_PARALLELISM=false
python scripts/verify_chatgpt_steering_22_environment.py \\
  --model-id "{model_id}" --expected-environment "{environment_name}" --require-cuda
exec python {command_script} --config {config_path} --manifest {manifest} --index 0
"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    path.chmod(0o755)


def row_slug(row: dict[str, Any], *, stage: str) -> str:
    identity = "__".join(
        (
            stage,
            str(row.get("prompt_id", "unknown_prompt")),
            str(row["model_id"]),
            str(row.get("variant", "calibration")),
            f"attempt_{int(row.get('attempt', 1)):03d}",
        )
    )
    safe = "".join(character if character.isalnum() else "_" for character in identity)
    digest = hashlib.sha256(
        json.dumps(row, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return f"{safe[:160]}__{digest}"


def materialize_unsupported(root: Path, row: dict[str, Any]) -> Path:
    output_dir = root / str(row["output_dir"])
    status_path = output_dir / "cell_status.json"
    unsupported_path = output_dir / "UNSUPPORTED_BY_MODEL.json"
    if status_path.exists() or unsupported_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing unsupported cell: {output_dir}"
        )
    completed_at = datetime.now(timezone.utc).isoformat()
    record = {
        "status": "unsupported_by_model",
        "cell_id": row["cell_id"],
        "row_sha256": row["row_sha256"],
        "model_id": row["model_id"],
        "prompt_id": row["prompt_id"],
        "variant": row["variant"],
        "attempt": int(row["attempt"]),
        "completed_at": completed_at,
        "reason": row["unsupported_reason"],
        "expected_media": False,
        "slurm_job_id": None,
        "slurm_array_task_id": None,
    }
    atomic_json(unsupported_path, record)
    atomic_json(status_path, record)
    return status_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/chatgpt_steering_22_july.yaml",
    )
    parser.add_argument("--mode", choices=("diagnostic", "full"), required=True)
    args = parser.parse_args()
    spec = load_campaign_spec(args.config)
    root = Path(spec["_root"])
    state = root / spec["campaign"]["state_root"]
    matrix_name = "diagnostic_matrix.jsonl" if args.mode == "diagnostic" else "matrix.jsonl"
    calibration_name = (
        "diagnostic_calibration.jsonl"
        if args.mode == "diagnostic"
        else "calibration.jsonl"
    )
    matrix = read_rows(state / matrix_name)
    calibration = read_rows(state / calibration_name)
    environment_spec = spec["slurm"].get("environments", {})
    default_environment = environment_spec.get("default")
    by_model_environment = environment_spec.get("by_model", {})
    if default_environment not in ALLOWED_ENVIRONMENTS:
        raise ValueError(f"Invalid default environment: {default_environment!r}")
    model_ids = {model["id"] for model in spec["models"]}
    if set(by_model_environment) - model_ids:
        raise ValueError(
            "Environment overrides name unknown models: "
            f"{sorted(set(by_model_environment) - model_ids)}"
        )
    model_environments = {
        model_id: by_model_environment.get(model_id, default_environment)
        for model_id in model_ids
    }
    invalid_environments = {
        model_id: environment
        for model_id, environment in model_environments.items()
        if environment not in ALLOWED_ENVIRONMENTS
    }
    if invalid_environments:
        raise ValueError(f"Invalid model environments: {invalid_environments}")
    by_model_cal: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_independent: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_model_related: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in calibration:
        by_model_cal[row["model_id"]].append(row)
    for row in matrix:
        execution_status = str(row["execution_status"])
        if execution_status == "unsupported_by_model":
            continue
        if execution_status != "runnable":
            raise ValueError(
                f"Unexpected execution status for {row['cell_id']}: {execution_status!r}"
            )
        destination = (
            by_model_related
            if row["variant"]
            in {
                "midsteer",
                "sgf_switch_adaptation",
                "safe_denoiser_switch_adaptation",
            }
            else by_model_independent
        )
        destination[row["model_id"]].append(row)
    receipt: dict[str, Any] = {
        "mode": args.mode,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "slurm_arrays": False,
        "array_throttles": False,
        "gemini_stage": False,
        "jobs": [],
        "unsupported_cells": [],
    }
    for row in matrix:
        if row["execution_status"] != "unsupported_by_model":
            continue
        status_path = materialize_unsupported(root, row)
        receipt["unsupported_cells"].append(
            {
                "cell_id": row["cell_id"],
                "model_id": row["model_id"],
                "prompt_id": row["prompt_id"],
                "variant": row["variant"],
                "row_sha256": row["row_sha256"],
                "status_path": str(status_path),
            }
        )

    calibration_jobs: dict[str, list[str]] = defaultdict(list)
    for model_id, rows in by_model_cal.items():
        resources = {
            **spec["slurm"]["defaults"],
            **spec["slurm"].get("by_model", {}).get(model_id, {}),
        }
        for row in rows:
            slug = row_slug(row, stage="calibration")
            model_dir = state / "submission" / args.mode / model_id / "calibration"
            manifest = model_dir / f"{slug}.jsonl"
            script = model_dir / f"{slug}.sbatch"
            write_rows(manifest, [row])
            write_sbatch(
                path=script,
                root=root,
                job_name=f"chs22-cal-{model_id[:20]}-{slug[-12:]}"[:128],
                manifest=manifest,
                command_script="scripts/run_chatgpt_steering_22_calibration.py",
                config_path=args.config,
                resources=resources,
                environment_name=model_environments[model_id],
                model_id=model_id,
            )
            job_id = submit(script)
            calibration_jobs[model_id].append(job_id)
            receipt["jobs"].append(
                {
                    "stage": "calibration",
                    "cell_id": row.get("cell_id"),
                    "model_id": model_id,
                    "job_id": job_id,
                    "row_sha256": row["row_sha256"],
                    "manifest": str(manifest),
                    "script": str(script),
                    "environment": model_environments[model_id],
                }
            )
    for stage, grouped in (
        ("independent", by_model_independent),
        ("related", by_model_related),
    ):
        for model_id, rows in grouped.items():
            resources = {
                **spec["slurm"]["defaults"],
                **spec["slurm"].get("by_model", {}).get(model_id, {}),
            }
            dependency = (
                ":".join(calibration_jobs.get(model_id, []))
                if stage == "related"
                else None
            )
            for row in rows:
                slug = row_slug(row, stage=stage)
                model_dir = state / "submission" / args.mode / model_id / stage
                manifest = model_dir / f"{slug}.jsonl"
                script = model_dir / f"{slug}.sbatch"
                write_rows(manifest, [row])
                write_sbatch(
                    path=script,
                    root=root,
                    job_name=f"chs22-{stage[:3]}-{model_id[:20]}-{slug[-12:]}"[:128],
                    manifest=manifest,
                    command_script="scripts/run_chatgpt_steering_22_cell.py",
                    config_path=args.config,
                    resources=resources,
                    environment_name=model_environments[model_id],
                    model_id=model_id,
                )
                job_id = submit(script, dependency=dependency)
                receipt["jobs"].append(
                    {
                        "stage": stage,
                        "cell_id": row["cell_id"],
                        "model_id": model_id,
                        "job_id": job_id,
                        "dependency": dependency,
                        "row_sha256": row["row_sha256"],
                        "manifest": str(manifest),
                        "script": str(script),
                        "environment": model_environments[model_id],
                    }
                )
    target = (
        state
        / f"submission_{args.mode}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    atomic_json(target, receipt)
    print(json.dumps({**receipt, "receipt": str(target)}, sort_keys=True))


if __name__ == "__main__":
    main()
