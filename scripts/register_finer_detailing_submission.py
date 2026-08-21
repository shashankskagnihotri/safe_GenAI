#!/usr/bin/env python
"""Create an immutable registry for one submitted finer-detailing Slurm array."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.benchmarks.finer_detailing_correction import (
    project_root,
    read_manifest,
)
from hierasafe_flow.benchmarks.slurm_tracking import (
    build_submission_registry,
    write_submission_registry,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically register manifest job indices to a Slurm array job. "
            "The output is immutable and cannot be overwritten."
        )
    )
    parser.add_argument("--manifest", required=True, help="Verified immutable manifest JSON.")
    parser.add_argument(
        "--slurm-array-job-id",
        required=True,
        help="Numeric job ID returned by sbatch --parsable.",
    )
    parser.add_argument(
        "--array-spec",
        required=True,
        help="Exact submitted array subset, for example 0-35%%8 or 0,2,5-9.",
    )
    parser.add_argument("--slurm-job-name", default=None)
    parser.add_argument("--output", required=True, help="New immutable registry JSON path.")
    parser.add_argument("--project-root", default=str(project_root()))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    manifest_path = _resolve(Path(args.manifest), root)
    output_path = _resolve(Path(args.output), root)
    # Registry publication is prospective scheduler state, not historical audit.
    # The strict reader reauthenticates all live launch inputs, including the
    # complete accepted FLUX-v3 native-equivalence DAG, before any immutable
    # registry can be written.
    manifest = read_manifest(manifest_path, root)
    registry = build_submission_registry(
        manifest=manifest,
        manifest_path=manifest_path,
        slurm_array_job_id=args.slurm_array_job_id,
        array_spec=args.array_spec,
        slurm_job_name=args.slurm_job_name,
    )
    write_submission_registry(registry, output_path)
    print(
        json.dumps(
            {
                "path": str(output_path),
                "registry_sha256": registry["registry_sha256"],
                "manifest_sha256": registry["manifest_sha256"],
                "slurm_array_job_id": registry["slurm_array_job_id"],
                "num_registered_tasks": registry["num_registered_tasks"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


def _resolve(path: Path, root: Path) -> Path:
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


if __name__ == "__main__":
    raise SystemExit(main())
