#!/usr/bin/env python
"""Dedicated, plan-lineage-aware environment dispatch for engineering smokes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.benchmarks.finer_detailing_correction import project_root
from hierasafe_flow.benchmarks.finer_detailing_production_smoke import validate_smoke_plan
from hierasafe_flow.benchmarks.finer_detailing_smoke_launch import (
    publish_smoke_launch_authorization,
    read_smoke_launch_bundle,
    require_exact_smoke_registry_path,
    smoke_plan_manifest_job,
    validate_live_smoke_dispatch_identity,
)
from scripts.finer_detailing_environment_dispatch import (
    contract_for_model,
    persist_validated_job_environment,
    validate_job_environment,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("resolve", "verify-job"):
        child = subparsers.add_parser(command)
        child.add_argument("--plan", required=True)
        child.add_argument("--role", required=True)
        child.add_argument("--index", required=True, type=int)
        child.add_argument("--submission-registry", required=True)
        child.add_argument("--project-root", default=str(project_root()))
    verify = subparsers.choices["verify-job"]
    verify.add_argument("--expected-model", required=True)
    verify.add_argument("--expected-environment", required=True)
    verify.add_argument("--expected-diffusers-revision", required=True)
    verify.add_argument("--expected-manifest", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    plan = validate_smoke_plan(args.plan, root=root)
    read_smoke_launch_bundle(plan.path, root=root)
    manifest_path, manifest, job = smoke_plan_manifest_job(plan, args.role, args.index)
    registry_path = require_exact_smoke_registry_path(plan, args.role, args.submission_registry)
    validate_live_smoke_dispatch_identity(
        plan=plan,
        role=args.role,
        index=args.index,
        submission_registry_path=registry_path,
        root=root,
    )
    contract = contract_for_model(str(job["model_name"]))
    if args.command == "resolve":
        fields = (
            contract.name,
            str(job["model_name"]),
            contract.diffusers_revision,
            str(manifest_path),
        )
        if any("\t" in field or "\n" in field for field in fields):
            raise RuntimeError("Unsafe character in smoke dispatch record.")
        print("\t".join(fields))
        return 0

    expected = (
        args.expected_environment,
        args.expected_model,
        args.expected_diffusers_revision,
        str(Path(args.expected_manifest).resolve()),
    )
    actual = (
        contract.name,
        str(job["model_name"]),
        contract.diffusers_revision,
        str(manifest_path),
    )
    if expected != actual:
        raise RuntimeError(
            f"Smoke dispatch changed between resolve and verify: expected={expected}, actual={actual}."
        )
    validation = validate_job_environment(job, contract)
    persisted = persist_validated_job_environment(
        job=job,
        manifest=manifest,
        job_index=args.index,
        validation=validation,
        root=root,
    )
    authorization = publish_smoke_launch_authorization(
        plan_path=plan.path,
        role=args.role,
        index=args.index,
        submission_registry_path=args.submission_registry,
        root=root,
    )
    print(
        json.dumps(
            {
                "environment_preflight": persisted,
                "smoke_launch_authorization": authorization,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
