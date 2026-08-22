#!/usr/bin/env python3
"""Submit the exact DES SD1.5 quality reproduction chain."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/experiments/push_for_iclr/des_sd15_quality_v1.json"
REGISTRY = Path(
    "/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_REGISTRY"
)


def command_output(command: List[str]) -> str:
    completed = subprocess.run(
        command,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed ({completed.returncode}): {' '.join(command)}\n"
            f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )
    return completed.stdout.strip()


def submit(script: Path, commit: str, dependency: Optional[str] = None) -> str:
    text = script.read_text(encoding="utf-8")
    array_lines = [line for line in text.splitlines() if line.startswith("#SBATCH --array=")]
    if any("%" in line for line in array_lines):
        raise RuntimeError(f"Array throttle is prohibited: {script}")
    command = ["sbatch", "--parsable"]
    if dependency:
        command.append(f"--dependency={dependency}")
    command.extend(
        [
            "--export=ALL,"
            f"PUSH_FOR_ICLR_EXECUTION_ROOT={ROOT},"
            f"PUSH_FOR_ICLR_EXPECTED_COMMIT={commit}",
            str(script),
        ]
    )
    return command_output(command).split(";")[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generation-job-id",
        required=True,
        help="Parent array job ID for the exact DES six-suite generation",
    )
    args = parser.parse_args()

    commit = command_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"])
    if command_output(["git", "-C", str(ROOT), "status", "--porcelain"]):
        raise RuntimeError("DES quality worktree must be clean before submission")
    if not re.fullmatch(r"\d+", args.generation_job_id):
        raise RuntimeError("Generation job ID must be numeric")
    command_output(["scontrol", "show", "job", args.generation_job_id])

    slurm = ROOT / "slurm"
    environment = submit(slurm / "push_for_iclr_des_quality_env.sbatch", commit)
    metrics = submit(
        slurm / "push_for_iclr_des_quality_metrics.sbatch",
        commit,
        dependency=f"afterok:{environment}:{args.generation_job_id}",
    )
    aggregate = submit(
        slurm / "push_for_iclr_des_quality_aggregate.sbatch",
        commit,
        dependency=f"afterok:{metrics}",
    )

    config_sha = hashlib.sha256(CONFIG.read_bytes()).hexdigest()
    payload: Dict[str, object] = {
        "method": "DES",
        "protocol": "SD15_PAPER_QUALITY_EXACT_V1",
        "implementation_commit": commit,
        "config_sha256": config_sha,
        "generation_job_id": args.generation_job_id,
        "jobs": {
            "environment": environment,
            "metrics": metrics,
            "aggregate": aggregate,
        },
        "array_throttles": None,
        "nice": 10000,
        "fallbacks_allowed": False,
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "worktree": str(ROOT),
    }
    REGISTRY.mkdir(parents=True, exist_ok=True)
    path = REGISTRY / f"des_sd15_quality_{environment}.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
