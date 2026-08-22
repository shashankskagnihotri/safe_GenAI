#!/usr/bin/env python3
"""Submit the exact unthrottled 480-cell trust-region development array."""

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
        default=REPOSITORY_ROOT / "outputs/PUSH_FOR_ICLR/MANIFESTS/trust_region_development_480.jsonl",
    )
    parser.add_argument(
        "--sbatch",
        type=Path,
        default=REPOSITORY_ROOT / "slurm/push_for_iclr_trust_region_development.sbatch",
    )
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    try:
        manifest = args.manifest.resolve()
        rows, manifest_sha256 = validate_sealed_job_manifest(manifest)
        if len(rows) != 480:
            raise AblationManifestError(f"Expected 480 rows, got {len(rows)}")
        if [row["job_index"] for row in rows] != list(range(480)):
            raise AblationManifestError("Job indices are not exactly 0..479")
        if {row["code_commit"] for row in rows} != {args.code_commit}:
            raise AblationManifestError("Code commit mismatch")
        if {row["ablation_split"] for row in rows} != {"development"}:
            raise AblationManifestError("Non-development row in redesign manifest")
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
            "--array=0-479",
            "--export=ALL,"
            f"PUSH_TR_MANIFEST={manifest},"
            f"PUSH_TR_MANIFEST_FILE_SHA256={manifest_file_sha256},"
            f"PUSH_TR_CODE_COMMIT={args.code_commit}",
            str(args.sbatch.resolve()),
        ]
        result = subprocess.run(
            command,
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise AblationManifestError(f"Unexpected sbatch output: {result.stdout!r}")
        registry = {
            "schema_version": "push-for-iclr.trust-region-submission.v1",
            "stage": 7,
            "split_role": "development_only_method_redesign",
            "status": "SUBMITTED",
            "submitted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "slurm_job_id": job_id,
            "array_expression": "0-479",
            "array_throttle": None,
            "user_hold": False,
            "job_count": 480,
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
            / f"trust_region_development_480_submitted_{job_id}.json"
        )
        _write_immutable(registry_path, registry)
        print(json.dumps(registry, indent=2, sort_keys=True))
        return 0
    except (AblationManifestError, subprocess.CalledProcessError) as exc:
        if isinstance(exc, subprocess.CalledProcessError) and exc.stderr:
            print(exc.stderr.strip(), file=sys.stderr)
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
