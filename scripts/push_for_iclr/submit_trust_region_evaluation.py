#!/usr/bin/env python3
"""Submit all non-Gemini trust-region evaluators with strict dependencies."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from hierasafe_flow.campaigns.push_for_iclr.trust_region_evaluation import (
    atomic_json,
    sha256_file,
)


def _submit(command: list[str]) -> str:
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";")[0]
    if not job_id.isdigit():
        raise RuntimeError(f"Could not parse sbatch result: {result.stdout!r}.")
    return job_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation-job-id", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--evaluation-worktree", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    source_root = args.source_root.resolve()
    worktree = args.evaluation_worktree.resolve()
    manifest = args.manifest.resolve()
    config = args.config.resolve()
    manifest_sha = sha256_file(manifest)
    commit = subprocess.run(
        ["git", "-C", str(worktree), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "-C", str(worktree), "status", "--porcelain"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError("Evaluation worktree must be clean before Slurm submission.")
    campaign_root = source_root / "outputs" / "PUSH_FOR_ICLR"
    summary_root = campaign_root / "EVALUATIONS" / "trust_region_development"
    sheet_root = campaign_root / "CONTACT_SHEETS" / "trust_region_development"
    aggregate_root = campaign_root / "AGGREGATES" / "trust_region_development"
    logs = campaign_root / "JOB_LOGS"
    logs.mkdir(parents=True, exist_ok=True)
    common = (
        f"TR_EVAL_WORKTREE={worktree},TR_SOURCE_ROOT={source_root},"
        f"TR_MANIFEST={manifest},TR_MANIFEST_SHA256={manifest_sha},"
        f"TR_EVAL_CONFIG={config},TR_SUMMARY_ROOT={summary_root},"
        f"TR_SHEET_ROOT={sheet_root},TR_AGGREGATE_ROOT={aggregate_root},"
        f"TR_EVAL_COMMIT={commit}"
    )
    imageguard_command = [
        "sbatch",
        "--parsable",
        "--array=0-23",
        f"--dependency=afterok:{args.generation_job_id}",
        f"--export=ALL,{common}",
        str(worktree / "slurm/push_for_iclr_trust_region_imageguard.sbatch"),
    ]
    imageguard_job = _submit(imageguard_command)
    fidelity_command = [
        "sbatch",
        "--parsable",
        "--array=0-23",
        f"--dependency=afterok:{args.generation_job_id}",
        f"--export=ALL,{common}",
        str(worktree / "slurm/push_for_iclr_trust_region_fidelity.sbatch"),
    ]
    fidelity_job = _submit(fidelity_command)
    sheet_command = [
        "sbatch",
        "--parsable",
        "--array=0-47",
        f"--dependency=afterok:{imageguard_job}:{fidelity_job}",
        f"--export=ALL,{common}",
        str(worktree / "slurm/push_for_iclr_trust_region_contact_sheets.sbatch"),
    ]
    sheet_job = _submit(sheet_command)
    aggregate_command = [
        "sbatch",
        "--parsable",
        f"--dependency=afterok:{sheet_job}",
        f"--export=ALL,{common}",
        str(worktree / "slurm/push_for_iclr_trust_region_aggregate.sbatch"),
    ]
    aggregate_job = _submit(aggregate_command)
    payload: dict[str, Any] = {
        "schema_version": "push-for-iclr.trust-region-evaluation-submission.v1",
        "submitted_at_utc": datetime.now(timezone.utc).isoformat(),
        "generation_job_id": args.generation_job_id,
        "evaluation_commit": commit,
        "evaluation_worktree": str(worktree),
        "source_root": str(source_root),
        "manifest_path": str(manifest),
        "manifest_file_sha256": manifest_sha,
        "config_path": str(config),
        "config_sha256": sha256_file(config),
        "array_throttles": None,
        "jobs": {
            "imageguard": {"job_id": imageguard_job, "array": "0-23", "command": imageguard_command},
            "fidelity": {"job_id": fidelity_job, "array": "0-23", "command": fidelity_command},
            "contact_sheets": {"job_id": sheet_job, "array": "0-47", "command": sheet_command},
            "provisional_aggregate": {"job_id": aggregate_job, "command": aggregate_command},
        },
        "manual_review_required_after_aggregate": True,
        "gemini_used": False,
    }
    registry = campaign_root / "JOB_REGISTRY" / f"trust_region_evaluation_submitted_{imageguard_job}_{fidelity_job}.json"
    atomic_json(registry, payload)
    print(json.dumps({"status": "submitted", "registry": str(registry), "jobs": payload["jobs"]}, sort_keys=True))


if __name__ == "__main__":
    main()
