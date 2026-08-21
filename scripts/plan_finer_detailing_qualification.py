#!/usr/bin/env python
"""Build and immutably publish one exact finer-detailing Q1/Q2 plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.benchmarks.finer_detailing_correction import project_root
from hierasafe_flow.benchmarks.finer_detailing_qualification import (
    MANIFEST_ROLE_ORDER,
    PHASES,
    publish_qualification_phase,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build four manifest-builder-derived Q1/Q2 manifests and one authenticated "
            "qualification plan. The supplied paths must be the exact five direct children "
            "of the canonical phase cohort. They, their sidecars/snapshots, and the cohort "
            "commit become admissible through one commit-last O_EXCL hard-link transaction; "
            "this command never "
            "submits jobs."
        )
    )
    parser.add_argument("--phase", required=True, choices=PHASES)
    parser.add_argument("--project-root", default=str(project_root()))
    parser.add_argument("--plan-out", required=True, help="New qualification-plan JSON path.")
    parser.add_argument(
        "--image-seed000-manifest-out",
        required=True,
        help="New image seed-0 manifest JSON path.",
    )
    parser.add_argument(
        "--video-seed000-manifest-out",
        required=True,
        help="New video seed-0 manifest JSON path.",
    )
    parser.add_argument(
        "--video-seed001-manifest-out",
        required=True,
        help="New video seed-1 manifest JSON path.",
    )
    parser.add_argument(
        "--video-seed002-manifest-out",
        required=True,
        help="New video seed-2 manifest JSON path.",
    )
    parser.add_argument(
        "--output-root",
        required=True,
        help="Exact shared generation output root frozen into every manifest.",
    )
    parser.add_argument("--attempt", required=True, type=int)
    parser.add_argument(
        "--flux1-native-equivalence-acceptance-receipt",
        required=True,
        help=(
            "Absolute canonical acceptance_receipt.json for the admitted FLUX-v3 "
            "native-equivalence execution DAG."
        ),
    )
    parser.add_argument(
        "--allow-unvalidated-temporal-pilot",
        action="store_true",
        help=(
            "Explicitly authorize configs still marked pilot for qualification. This is "
            "forwarded to the existing benchmark builder and recorded per temporal job."
        ),
    )
    parser.add_argument(
        "--q1-plan",
        default=None,
        help="Q2 only: authenticated upstream Q1 qualification-plan path.",
    )
    parser.add_argument(
        "--exact-one-gate-receipt",
        default=None,
        help="Q2 only: evaluator-produced all-66 exact-one gate receipt and sidecar.",
    )
    parser.add_argument(
        "--non-regression-gate-receipt",
        default=None,
        help="Q2 only: evaluator/test-produced full-pair non-regression receipt and sidecar.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    manifest_paths = dict(
        zip(
            MANIFEST_ROLE_ORDER,
            (
                args.image_seed000_manifest_out,
                args.video_seed000_manifest_out,
                args.video_seed001_manifest_out,
                args.video_seed002_manifest_out,
            ),
            strict=True,
        )
    )
    validated = publish_qualification_phase(
        args.phase,
        plan_path=args.plan_out,
        manifest_paths=manifest_paths,
        output_root=args.output_root,
        attempt=args.attempt,
        allow_unvalidated_temporal_pilot=args.allow_unvalidated_temporal_pilot,
        flux1_native_equivalence_acceptance_receipt_path=(
            args.flux1_native_equivalence_acceptance_receipt
        ),
        q1_plan_path=args.q1_plan,
        exact_one_gate_receipt_path=args.exact_one_gate_receipt,
        non_regression_gate_receipt_path=args.non_regression_gate_receipt,
        root=root,
    )
    print(
        json.dumps(
            {
                "mode": "published_without_submission",
                "phase": validated.phase,
                "qualification_plan_path": str(validated.path),
                "qualification_plan_sha256": validated.digest,
                "topology_proof": validated.topology_proof,
                "qualification_union_proof": validated.plan["qualification_union_proof"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
