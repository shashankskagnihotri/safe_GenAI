#!/usr/bin/env python
"""Dedicated cohort/registry/live-Slurm dispatcher for production campaigns."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.benchmarks.finer_detailing_campaign_launch import (
    COHORT_KINDS,
    campaign_manifest_job,
    publish_campaign_launch_authorization,
    read_campaign_cohort,
    read_campaign_launch_bundle,
    require_exact_registry_path,
    validate_live_campaign_dispatch_identity,
)
from hierasafe_flow.benchmarks.finer_detailing_correction import project_root
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
        child.add_argument("--cohort-kind", required=True, choices=COHORT_KINDS)
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
    cohort = read_campaign_cohort(args.cohort_kind, root=root)
    read_campaign_launch_bundle(args.cohort_kind, root=root)
    registry_path = require_exact_registry_path(
        cohort, args.role, args.submission_registry
    )
    manifest_path, manifest, job = campaign_manifest_job(
        cohort, args.role, args.index
    )
    # This scheduler query deliberately precedes dependency inspection and every
    # preflight/authorization write.
    validate_live_campaign_dispatch_identity(
        kind=args.cohort_kind,
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
            raise RuntimeError("Unsafe character in campaign dispatch record.")
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
            "Campaign dispatch changed between resolve and verify: "
            f"expected={expected}, actual={actual}."
        )
    validation = validate_job_environment(job, contract)
    persisted = persist_validated_job_environment(
        job=job,
        manifest=manifest,
        job_index=args.index,
        validation=validation,
        root=root,
    )
    authorization = publish_campaign_launch_authorization(
        kind=args.cohort_kind,
        role=args.role,
        index=args.index,
        submission_registry_path=registry_path,
        root=root,
    )
    print(
        json.dumps(
            {
                "environment_preflight": persisted,
                "campaign_launch_authorization": authorization,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
