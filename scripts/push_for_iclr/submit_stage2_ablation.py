#!/usr/bin/env python3
"""Validate and submit the real unthrottled Stage-2 Slurm array."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hierasafe_flow.campaigns.push_for_iclr.ablation_manifests import (  # noqa: E402
    AblationManifestError,
    sha256_file,
    validate_sealed_job_manifest,
)


def _write_immutable(path: Path, value: dict) -> None:
    if path.exists():
        raise AblationManifestError(f"Refusing to overwrite registry: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    with partial.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/PUSH_FOR_ICLR/MANIFESTS/stage2_ablation_780.jsonl",
    )
    parser.add_argument(
        "--sbatch",
        type=Path,
        default=REPOSITORY_ROOT / "slurm/push_for_iclr_stage2_ablation.sbatch",
    )
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    try:
        manifest = args.manifest.resolve()
        rows, manifest_sha256 = validate_sealed_job_manifest(manifest)
        if len(rows) != 780:
            raise AblationManifestError(f"Expected 780 rows, got {len(rows)}")
        if [row["job_index"] for row in rows] != list(range(780)):
            raise AblationManifestError("Job indices are not exactly 0..779")
        if {row["code_commit"] for row in rows} != {args.code_commit}:
            raise AblationManifestError("Code commit mismatch")
        if any("%" in str(value) for value in ["0-779"]):
            raise AblationManifestError("Array throttle is forbidden")
        completed = sum(
            (
                REPOSITORY_ROOT
                / "outputs/PUSH_FOR_ICLR"
                / row["expected_output_relative_path"]
                / "_SUCCESS"
            ).exists()
            for row in rows
        )
        if completed:
            raise AblationManifestError(f"Initial manifest has {completed} completed cells")
        manifest_file_sha256 = sha256_file(manifest)
        command = [
            "sbatch",
            "--parsable",
            "--array=0-779",
            "--export=ALL,"
            f"PUSH_FOR_ICLR_MANIFEST={manifest},"
            f"PUSH_FOR_ICLR_MANIFEST_FILE_SHA256={manifest_file_sha256},"
            f"PUSH_FOR_ICLR_CODE_COMMIT={args.code_commit}",
            str(args.sbatch.resolve()),
        ]
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        slurm_job_id = result.stdout.strip().split(";", 1)[0]
        if not slurm_job_id.isdigit():
            raise AblationManifestError(f"Unexpected sbatch output: {result.stdout!r}")
        registry = {
            "schema_version": "push-for-iclr.slurm-submission.v1",
            "stage": 2,
            "status": "SUBMITTED",
            "submitted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "slurm_job_id": slurm_job_id,
            "array_expression": "0-779",
            "array_throttle": None,
            "user_hold": False,
            "job_count": 780,
            "manifest_path": str(manifest),
            "manifest_sha256": manifest_sha256,
            "manifest_file_sha256": manifest_file_sha256,
            "code_commit": args.code_commit,
            "sbatch_path": str(args.sbatch.resolve()),
            "sbatch_file_sha256": sha256_file(args.sbatch.resolve()),
            "submission_command": command,
        }
        registry_path = (
            REPOSITORY_ROOT
            / "outputs/PUSH_FOR_ICLR/JOB_REGISTRY"
            / f"stage2_ablation_780_submitted_{slurm_job_id}.json"
        )
        _write_immutable(registry_path, registry)
        print(json.dumps(registry, indent=2, sort_keys=True))
        return 0
    except (AblationManifestError, subprocess.CalledProcessError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
