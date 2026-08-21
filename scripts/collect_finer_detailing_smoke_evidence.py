#!/usr/bin/env python
"""Collect and atomically seal production-smoke gate evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.benchmarks.finer_detailing_correction import project_root
from hierasafe_flow.evaluation.production_smoke_collection import (
    ROW_GATE_CONTRACTS,
    collect_full_pair_non_regression_evidence_bundle,
    collect_no_generation_evidence_bundle,
    seal_ideogram_conditioning_evidence_bundle,
    seal_selected_row_evidence_bundle,
)


def _manual_bindings(values: list[str]) -> dict[str, str]:
    bindings: dict[str, str] = {}
    for raw in values:
        if "=" not in raw:
            raise ValueError("--manual-ledger must use CONDITION_ID=PATH.")
        condition_id, path = raw.split("=", 1)
        if not condition_id or not path or condition_id in bindings:
            raise ValueError("Manual-ledger bindings must be nonempty and unique.")
        bindings[condition_id] = path
    return bindings


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=str(project_root()))
    subparsers = parser.add_subparsers(dest="command", required=True)

    no_generation = subparsers.add_parser(
        "no-generation", help="Execute the exact ten registered no-generation checks."
    )
    no_generation.add_argument("--subject-plan", required=True)
    no_generation.add_argument("--destination", required=True)
    no_generation.add_argument("--timeout-seconds", type=int, default=43_200)

    ideogram = subparsers.add_parser(
        "ideogram-conditioning",
        help="Extract schema-3 provenance from the exact executed Ideogram sample report.",
    )
    ideogram.add_argument("--subject-plan", required=True)
    ideogram.add_argument("--destination", required=True)

    non_regression = subparsers.add_parser(
        "full-pair-nonregression",
        help="Run the exact registered Shapley tests with structured JUnit capture.",
    )
    non_regression.add_argument("--subject-plan", required=True)
    non_regression.add_argument("--destination", required=True)
    non_regression.add_argument("--timeout-seconds", type=int, default=1_800)

    rows = subparsers.add_parser(
        "rows", help="Seal completed row hashes and explicit human review ledgers."
    )
    rows.add_argument("--contract", required=True, choices=sorted(ROW_GATE_CONTRACTS))
    rows.add_argument("--subject-plan", required=True)
    rows.add_argument("--destination", required=True)
    rows.add_argument(
        "--manual-ledger",
        action="append",
        default=[],
        metavar="CONDITION_ID=PATH",
        help="Repeat once, in plan order, for every selected media row.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    destination = Path(args.destination)
    if not destination.is_absolute():
        destination = root / destination
    if args.command == "no-generation":
        result = collect_no_generation_evidence_bundle(
            subject_plan_path=args.subject_plan,
            destination=destination,
            root=root,
            timeout_seconds=args.timeout_seconds,
        )
    elif args.command == "ideogram-conditioning":
        result = seal_ideogram_conditioning_evidence_bundle(
            subject_plan_path=args.subject_plan,
            destination=destination,
            root=root,
        )
    elif args.command == "full-pair-nonregression":
        result = collect_full_pair_non_regression_evidence_bundle(
            subject_plan_path=args.subject_plan,
            destination=destination,
            root=root,
            timeout_seconds=args.timeout_seconds,
        )
    else:
        result = seal_selected_row_evidence_bundle(
            args.contract,
            subject_plan_path=args.subject_plan,
            manual_ledger_paths=_manual_bindings(args.manual_ledger),
            destination=destination,
            root=root,
        )
    evaluation = result["evaluation"]
    print(
        json.dumps(
            {
                "bundle": result["bundle"],
                "gate_contract": result["gate_contract"],
                "subject_plan_sha256": result["subject_plan_sha256"],
                "decision": evaluation["decision"],
                "evaluation_sha256": evaluation["evaluation_sha256"],
                "counts": evaluation["derived"]["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
