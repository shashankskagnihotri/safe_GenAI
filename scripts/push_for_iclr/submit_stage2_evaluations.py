#!/usr/bin/env python3
"""Submit dependency-correct Gemini, ImageGuard, and contact-sheet jobs."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hierasafe_flow.campaigns.push_for_iclr.ablation_manifests import (  # noqa: E402
    AblationManifestError,
    sha256_file,
    validate_sealed_job_manifest,
)


def _submit(arguments: list[str]) -> str:
    result = subprocess.run(
        ["sbatch", "--parsable", *arguments],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    job_id = result.stdout.strip().split(";", 1)[0]
    if not job_id.isdigit():
        raise AblationManifestError(f"Unexpected sbatch output: {result.stdout!r}")
    return job_id


def _write_registry(path: Path, value: dict) -> None:
    if path.exists():
        raise AblationManifestError(f"Refusing to overwrite evaluation registry: {path}")
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
    parser.add_argument("--generation-job-id", required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPOSITORY_ROOT / "outputs/PUSH_FOR_ICLR/MANIFESTS/stage2_ablation_780.jsonl",
    )
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    try:
        if not args.generation_job_id.isdigit():
            raise AblationManifestError("generation job ID must be numeric")
        current_commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if current_commit != args.code_commit:
            raise AblationManifestError("Evaluation worktree HEAD changed")
        subprocess.run(
            ["scontrol", "show", "job", args.generation_job_id],
            check=True,
            capture_output=True,
            text=True,
        )
        manifest = args.manifest.resolve()
        rows, manifest_sha = validate_sealed_job_manifest(manifest)
        if len(rows) != 780 or [row["job_index"] for row in rows] != list(range(780)):
            raise AblationManifestError("Generation manifest is not the exact 780-cell matrix")
        manifest_file_sha = sha256_file(manifest)
        common_export = (
            "ALL,"
            f"PUSH_FOR_ICLR_MANIFEST={manifest},"
            f"PUSH_FOR_ICLR_MANIFEST_FILE_SHA256={manifest_file_sha},"
            f"PUSH_FOR_ICLR_EVAL_COMMIT={args.code_commit}"
        )
        gemini_script = REPOSITORY_ROOT / "slurm/push_for_iclr_stage2_gemini.sbatch"
        imageguard_script = REPOSITORY_ROOT / "slurm/push_for_iclr_stage2_imageguard.sbatch"
        contact_script = REPOSITORY_ROOT / "slurm/push_for_iclr_stage2_contact_sheet.sbatch"
        gemini_job = _submit(
            [
                "--array=0-779",
                f"--dependency=aftercorr:{args.generation_job_id}",
                f"--export={common_export}",
                str(gemini_script),
            ]
        )

        group_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)
        category_rows: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
        for row in rows:
            group_rows[(row["model_id"], row["ablation_id"])].append(row)
            category_rows[(row["model_id"], row["ablation_id"], row["category"])].append(row)
        if len(group_rows) != 26 or any(len(value) != 30 for value in group_rows.values()):
            raise AblationManifestError("Expected 26 complete model/ablation groups")
        if len(category_rows) != 52 or any(len(value) != 15 for value in category_rows.values()):
            raise AblationManifestError("Expected 52 complete category groups")

        imageguard_jobs = {}
        for (model, ablation), group in sorted(group_rows.items()):
            dependency = ":".join(
                ["afterok", *(f"{args.generation_job_id}_{row['job_index']}" for row in group)]
            )
            exported = (
                common_export
                + f",PUSH_FOR_ICLR_EVAL_MODEL={model}"
                + f",PUSH_FOR_ICLR_EVAL_ABLATION={ablation}"
            )
            imageguard_jobs[f"{model}/{ablation}"] = _submit(
                [f"--dependency={dependency}", f"--export={exported}", str(imageguard_script)]
            )

        contact_jobs = {}
        for (model, ablation, category), group in sorted(category_rows.items()):
            dependency = ":".join(
                ["afterok", *(f"{args.generation_job_id}_{row['job_index']}" for row in group)]
            )
            exported = (
                common_export
                + f",PUSH_FOR_ICLR_EVAL_MODEL={model}"
                + f",PUSH_FOR_ICLR_EVAL_ABLATION={ablation}"
                + f",PUSH_FOR_ICLR_EVAL_CATEGORY={category}"
            )
            contact_jobs[f"{model}/{ablation}/{category}"] = _submit(
                [f"--dependency={dependency}", f"--export={exported}", str(contact_script)]
            )

        registry = {
            "schema_version": "push-for-iclr.stage2-evaluation-submission.v1",
            "status": "SUBMITTED_WITH_GENERATION_DEPENDENCIES",
            "submitted_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "generation_job_id": args.generation_job_id,
            "generation_manifest_path": str(manifest),
            "generation_manifest_sha256": manifest_sha,
            "generation_manifest_file_sha256": manifest_file_sha,
            "evaluation_code_commit": args.code_commit,
            "gemini": {
                "job_id": gemini_job,
                "array_expression": "0-779",
                "array_throttle": None,
                "dependency": f"aftercorr:{args.generation_job_id}",
            },
            "imageguard_jobs": imageguard_jobs,
            "contact_sheet_jobs": contact_jobs,
            "counts": {
                "gemini_cells": 780,
                "imageguard_groups": len(imageguard_jobs),
                "contact_sheet_groups": len(contact_jobs),
            },
        }
        registry_path = (
            REPOSITORY_ROOT
            / "outputs/PUSH_FOR_ICLR/JOB_REGISTRY"
            / f"stage2_evaluations_for_{args.generation_job_id}.json"
        )
        _write_registry(registry_path, registry)
        print(json.dumps(registry, indent=2, sort_keys=True))
        return 0
    except (AblationManifestError, subprocess.CalledProcessError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
