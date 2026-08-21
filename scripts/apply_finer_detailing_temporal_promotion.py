#!/usr/bin/env python3
"""Apply, resume, or authenticate the committed five-model temporal promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.evaluation.temporal_promotion_apply import (
    apply_temporal_production_promotion,
    read_temporal_promotion_apply,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("apply", "reopen"))
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--no-wait-lock",
        action="store_true",
        help="Fail immediately instead of waiting for another apply process.",
    )
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.action == "apply":
        result = apply_temporal_production_promotion(
            root=args.root, wait_for_lock=not args.no_wait_lock
        )
    else:
        if args.no_wait_lock:
            raise SystemExit("--no-wait-lock is valid only with the apply action")
        result = read_temporal_promotion_apply(root=args.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
