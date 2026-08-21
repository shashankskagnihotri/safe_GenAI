#!/usr/bin/env python
"""Derive one immutable production-smoke gate from structured evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.benchmarks.finer_detailing_correction import project_root
from hierasafe_flow.benchmarks.finer_detailing_production_smoke import GATE_CONTRACTS
from hierasafe_flow.evaluation.production_smoke import evaluate_smoke_gate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", required=True, choices=GATE_CONTRACTS)
    parser.add_argument("--subject-plan", required=True)
    parser.add_argument("--evidence-index", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=str(project_root()))
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    report = evaluate_smoke_gate(
        args.contract,
        subject_plan_path=args.subject_plan,
        evidence_index_path=args.evidence_index,
        output_path=args.output,
        root=Path(args.project_root).resolve(),
    )
    print(
        json.dumps(
            {
                "mode": "evidence_derived_gate_evaluation",
                "contract": report["contract"],
                "decision": report["decision"],
                "evaluation_sha256": report["evaluation_sha256"],
                "counts": report["derived"]["counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
