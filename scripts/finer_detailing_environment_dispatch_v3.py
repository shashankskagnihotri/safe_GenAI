#!/usr/bin/env python3
"""Fail-closed environment dispatch for Flux common-seed v3 only.

This small entry point deliberately reuses the registered environment checks
without broadening the ordinary dispatcher.  It always uses the v3 audit
reader and requires the complete held-phase authorization before persisting a
preflight record.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import finer_detailing_environment_dispatch as environment

from hierasafe_flow.evaluation import flux1_common_seed_v3 as common


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("resolve", "verify-job"))
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", required=True, type=int)
    parser.add_argument("--phase-plan", required=True)
    parser.add_argument("--project-root", default=str(common.project_root()))
    parser.add_argument("--expected-model")
    parser.add_argument("--expected-environment")
    parser.add_argument("--expected-diffusers-revision")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser()
    manifest_path = (
        manifest_path if manifest_path.is_absolute() else root / manifest_path
    ).resolve()
    phase_plan = Path(args.phase_plan).expanduser()
    phase_plan = (phase_plan if phase_plan.is_absolute() else root / phase_plan).resolve()
    manifest, job, contract = environment.authenticated_manifest_job(
        manifest_path,
        args.index,
        root=root,
        manifest_reader=common.read_seed_manifest_for_audit,
    )
    if job.get("stage") != common.STAGE or manifest.get("stage") != common.STAGE:
        raise ValueError("Flux common-seed v3 dispatcher received a non-v3 manifest.")
    for candidate in manifest.get("jobs", ()):
        common.reject_reserved_common_seed_output_without_stage(candidate, root)
    authorization = common.build_common_seed_launch_authorization(
        phase_plan_path=phase_plan,
        manifest_path=manifest_path,
        job_index=args.index,
        root=root,
    )
    model_name = str(job["model_name"])
    if args.command == "resolve":
        forbidden = ("\t", "\n")
        fields = (contract.name, model_name, contract.diffusers_revision)
        if any(token in value for value in fields for token in forbidden):
            raise RuntimeError("Unsafe character in v3 environment dispatch record.")
        print("\t".join(fields))
        return 0

    expected = (
        args.expected_environment,
        args.expected_model,
        args.expected_diffusers_revision,
    )
    if any(value is None for value in expected):
        raise ValueError("verify-job requires all three --expected-* arguments.")
    actual = (contract.name, model_name, contract.diffusers_revision)
    if expected != actual:
        raise RuntimeError(
            "Environment dispatch changed between authentication and preflight: "
            f"expected={expected}, authenticated={actual}."
        )
    validation = environment.validate_job_environment(job, contract)
    validation["common_seed_launch_authorization"] = authorization
    persisted = environment.persist_validated_job_environment(
        job=job,
        manifest=manifest,
        job_index=args.index,
        validation=validation,
        root=root,
    )
    print(json.dumps(persisted, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
