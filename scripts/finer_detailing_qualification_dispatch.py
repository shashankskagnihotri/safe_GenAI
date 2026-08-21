#!/usr/bin/env python
"""Authenticate one committed qualification task before model loading."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.benchmarks.finer_detailing_correction import project_root
from hierasafe_flow.benchmarks.finer_detailing_qualification import (
    MANIFEST_ROLE_ORDER,
    canonical_qualification_manifest_paths,
    validate_qualification_plan,
)
from hierasafe_flow.benchmarks.finer_detailing_qualification_launch import (
    build_qualification_runtime_authorization,
    validate_live_qualification_dispatch_identity,
)
from scripts.finer_detailing_environment_dispatch import (
    contract_for_model,
    persist_validated_job_environment,
    validate_job_environment,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("resolve", "verify-job"):
        child = subparsers.add_parser(command)
        child.add_argument("--plan", required=True)
        child.add_argument("--role", required=True, choices=MANIFEST_ROLE_ORDER)
        child.add_argument("--registry", required=True)
        child.add_argument("--index", required=True, type=int)
        child.add_argument("--project-root", default=str(project_root()))
    verify = subparsers.choices["verify-job"]
    verify.add_argument("--expected-model", required=True)
    verify.add_argument("--expected-environment", required=True)
    verify.add_argument("--expected-diffusers-revision", required=True)
    verify.add_argument("--expected-manifest", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    plan = validate_qualification_plan(args.plan, root=root)
    role = str(args.role)
    manifest = plan.manifests[role]
    if isinstance(args.index, bool) or args.index not in range(len(manifest["jobs"])):
        raise IndexError("Qualification task index is outside its exact role manifest.")
    manifest_path = canonical_qualification_manifest_paths(plan.phase, root)[role]
    job = manifest["jobs"][args.index]
    contract = contract_for_model(str(job["model_name"]))
    validate_live_qualification_dispatch_identity(
        plan=plan,
        role=role,
        index=args.index,
        registry_path=args.registry,
        root=root,
    )
    if args.command == "resolve":
        fields = (
            contract.name,
            str(job["model_name"]),
            contract.diffusers_revision,
            str(manifest_path),
        )
        if any("\t" in field or "\n" in field for field in fields):
            raise RuntimeError("Unsafe character in qualification dispatch record.")
        print("\t".join(fields))
        return 0

    expected = (
        args.expected_environment,
        args.expected_model,
        args.expected_diffusers_revision,
        str(Path(args.expected_manifest).expanduser()),
    )
    actual = (
        contract.name,
        str(job["model_name"]),
        contract.diffusers_revision,
        str(manifest_path),
    )
    if expected != actual:
        raise RuntimeError(
            "Qualification dispatch changed between resolution and preflight: "
            f"expected={expected}, authenticated={actual}."
        )
    validation = validate_job_environment(job, contract)
    validation["qualification_launch_authorization"] = (
        build_qualification_runtime_authorization(
            plan=plan,
            role=role,
            index=args.index,
            registry_path=args.registry,
            root=root,
        )
    )
    persisted = persist_validated_job_environment(
        job=job,
        manifest=manifest,
        job_index=args.index,
        validation=validation,
        root=root,
    )
    print(json.dumps(persisted, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
