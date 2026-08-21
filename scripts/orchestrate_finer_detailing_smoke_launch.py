#!/usr/bin/env python
"""Validate or execute one all-held, atomically committed production-smoke launch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.benchmarks.finer_detailing_correction import project_root
from hierasafe_flow.benchmarks.finer_detailing_smoke_launch import (
    orchestrate_smoke_launch,
    read_smoke_launch_bundle,
    smoke_launch_preview,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, help="Authenticated production-smoke plan.")
    parser.add_argument("--project-root", default=str(project_root()))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate exact arrays and launch files without filesystem or scheduler writes.",
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
    plan = Path(args.plan)
    if not plan.is_absolute():
        plan = root / plan
    if args.validate_only:
        payload = smoke_launch_preview(plan, root=root)
    elif args.read_committed:
        reopened = read_smoke_launch_bundle(plan, root=root)
        payload = {
            "schema_version": 1,
            "status": "committed_launch_bundle_authenticated",
            "plan_sha256": reopened["plan"].digest,
            "launch_commit_sha256": reopened["commit"]["document_sha256"],
            "held_release_receipt_sha256": reopened["receipt"]["document_sha256"],
        }
    else:
        payload = orchestrate_smoke_launch(plan, root=root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
