#!/usr/bin/env python
"""Publish one gated production-shaped engineering-smoke stage without submission."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.benchmarks.finer_detailing_correction import project_root
from hierasafe_flow.benchmarks.finer_detailing_production_smoke import (
    EXACT_ONE_PATH_GATE,
    FINAL_CUMULATIVE_ADMISSION_GATE,
    FULL_PAIR_NON_REGRESSION_GATE,
    IDEOGRAM_CONDITIONING_GATE,
    NATIVE_BASELINE_GATE,
    NO_GENERATION_GATE,
    STAGES,
    WAN_TRANSITION_GATE,
    publish_smoke_stage,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build ordinary-builder-derived, production-shaped smoke manifests and publish "
            "one authenticated stage plan. This command never submits or runs a job."
        )
    )
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--project-root", default=str(project_root()))
    parser.add_argument(
        "--bundle-dir",
        required=True,
        help="New directory for component manifests and the final plan commit marker.",
    )
    parser.add_argument(
        "--output-root",
        default=str(project_root() / "outputs/finer_detailing_engineering_smoke"),
        help=(
            "Must remain the canonical engineering-smoke root; stage names are appended "
            "automatically and never count toward qualification/final outputs."
        ),
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
        "--upstream-plan",
        default=None,
        help="Required after no_generation_preflight; exact prior stage plan and sidecar.",
    )
    parser.add_argument("--no-generation-gate-evaluation", default=None)
    parser.add_argument("--wan-transition-gate-evaluation", default=None)
    parser.add_argument("--native-baseline-gate-evaluation", default=None)
    parser.add_argument("--ideogram-conditioning-gate-evaluation", default=None)
    parser.add_argument("--exact-one-path-gate-evaluation", default=None)
    parser.add_argument("--full-pair-non-regression-gate-evaluation", default=None)
    parser.add_argument("--final-cumulative-admission-gate-evaluation", default=None)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    supplied = {
        contract: path
        for contract, path in (
            (NO_GENERATION_GATE, args.no_generation_gate_evaluation),
            (WAN_TRANSITION_GATE, args.wan_transition_gate_evaluation),
            (NATIVE_BASELINE_GATE, args.native_baseline_gate_evaluation),
            (IDEOGRAM_CONDITIONING_GATE, args.ideogram_conditioning_gate_evaluation),
            (EXACT_ONE_PATH_GATE, args.exact_one_path_gate_evaluation),
            (FULL_PAIR_NON_REGRESSION_GATE, args.full_pair_non_regression_gate_evaluation),
            (
                FINAL_CUMULATIVE_ADMISSION_GATE,
                args.final_cumulative_admission_gate_evaluation,
            ),
        )
        if path is not None
    }
    validated = publish_smoke_stage(
        args.stage,
        bundle_dir=args.bundle_dir,
        output_root=args.output_root,
        attempt=args.attempt,
        flux1_native_equivalence_acceptance_receipt_path=(
            args.flux1_native_equivalence_acceptance_receipt
        ),
        upstream_plan_path=args.upstream_plan,
        gate_evaluation_paths=supplied,
        root=Path(args.project_root).expanduser().resolve(),
    )
    print(
        json.dumps(
            {
                "mode": "published_without_submission",
                "stage": validated.stage,
                "plan_path": str(validated.path),
                "smoke_plan_sha256": validated.digest,
                "topology_proof": validated.topology_proof,
                "cumulative_proof": validated.plan["cumulative_proof"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
