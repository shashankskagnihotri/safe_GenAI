#!/usr/bin/env python
"""Validate or execute one exact all-held production campaign launch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.benchmarks.finer_detailing_campaign_launch import (
    COHORT_KINDS,
    campaign_launch_preview,
    orchestrate_campaign_launch,
    read_campaign_launch_bundle,
)
from hierasafe_flow.benchmarks.finer_detailing_correction import project_root


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cohort-kind", required=True, choices=COHORT_KINDS)
    parser.add_argument("--project-root", default=str(project_root()))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Authenticate exact arrays and launch files without scheduler mutation.",
    )
    mode.add_argument(
        "--read-committed",
        action="store_true",
        help="Reopen an existing complete launch bundle without scheduler mutation.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    if args.validate_only:
        payload = campaign_launch_preview(args.cohort_kind, root=root)
    elif args.read_committed:
        bundle = read_campaign_launch_bundle(args.cohort_kind, root=root)
        payload = {
            "schema_version": 1,
            "status": "committed_campaign_launch_bundle_authenticated",
            "cohort_kind": bundle["cohort"].kind,
            "cohort_commit_sha256": bundle["cohort"].digest,
            "launch_commit_sha256": bundle["commit"]["document_sha256"],
            "held_release_receipt_sha256": bundle["receipt"]["document_sha256"],
        }
    else:
        payload = orchestrate_campaign_launch(args.cohort_kind, root=root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
