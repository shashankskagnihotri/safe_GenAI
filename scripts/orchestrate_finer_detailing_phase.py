#!/usr/bin/env python
"""Validate, preview, submit, or resume one immutable finer-detailing phase plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.benchmarks.finer_detailing_correction import project_root
from hierasafe_flow.benchmarks.finer_detailing_phase import (
    phase_preview,
    submit_phase,
    validate_phase_plan,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fail-closed phase orchestration for already-frozen finer-detailing manifests. "
            "The plan must be a schema-v1 JSON file with an authenticated .sha256 sidecar."
        )
    )
    parser.add_argument("--plan", required=True, help="Immutable phase-plan JSON path.")
    parser.add_argument("--project-root", default=str(project_root()))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate all manifests, launchers, arrays, environments, and union coverage.",
    )
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print exact sbatch commands without creating state or submitting.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    plan_path = Path(args.plan)
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    validated = validate_phase_plan(plan_path, root=root)
    preview = phase_preview(validated)
    if args.validate_only:
        preview["mode"] = "validate_only"
        preview.pop("commands", None)
        payload = preview
    elif args.dry_run:
        payload = preview
    else:
        payload = submit_phase(validated, root=root)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
