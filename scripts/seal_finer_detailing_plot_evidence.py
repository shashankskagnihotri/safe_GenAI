#!/usr/bin/env python
"""Atomically seal the complete fresh finer-detailing plot-evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.evaluation.finer_detailing_plot_bundle import (
    seal_fresh_plot_evidence,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--qualification-q2-plan", required=True)
    parser.add_argument(
        "--submission-registry",
        action="append",
        required=True,
        help="Repeat for every exact qualification, ladder, and final Slurm registry.",
    )
    parser.add_argument("--objective-temporal-raw", required=True)
    parser.add_argument("--manual-semantic-raw", required=True)
    parser.add_argument("--shapley-diagnostics-raw", required=True)
    parser.add_argument(
        "--destination",
        required=True,
        help="New immutable directory; it and any symlink alias must not exist.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = seal_fresh_plot_evidence(
        root=args.root,
        qualification_q2_plan_path=args.qualification_q2_plan,
        submission_registry_paths=args.submission_registry,
        objective_temporal_raw_path=args.objective_temporal_raw,
        manual_semantic_raw_path=args.manual_semantic_raw,
        shapley_diagnostics_raw_path=args.shapley_diagnostics_raw,
        destination=args.destination,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
