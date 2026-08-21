#!/usr/bin/env python
"""Preview or execute the exact four-array held Q1/Q2 launch."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.benchmarks.finer_detailing_correction import project_root
from hierasafe_flow.benchmarks.finer_detailing_qualification_launch import (
    orchestrate_qualification_launch,
    qualification_launch_preview,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--project-root", default=str(project_root()))
    parser.add_argument("--submit-held-and-release", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).expanduser().resolve()
    payload = (
        orchestrate_qualification_launch(args.plan, root=root)
        if args.submit_held_and_release
        else qualification_launch_preview(args.plan, root=root)
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
