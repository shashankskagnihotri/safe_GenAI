#!/usr/bin/env python3
"""Prepare the immutable five-model temporal promotion admission cohort."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.evaluation.temporal_promotion import (
    VIDEO_MODEL_ORDER,
    publish_temporal_promotion_cohort,
)


def _mapping(values: list[str], *, label: str) -> dict[str, Path]:
    parsed: dict[str, Path] = {}
    for value in values:
        model, separator, raw_path = value.partition("=")
        if not separator or not model or not raw_path:
            raise ValueError(f"{label} entries must use MODEL=PATH syntax.")
        if model in parsed:
            raise ValueError(f"{label} repeats model {model!r}.")
        parsed[model] = Path(raw_path)
    if set(parsed) != set(VIDEO_MODEL_ORDER):
        raise ValueError(f"{label} must cover exactly {list(VIDEO_MODEL_ORDER)}.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate five Q1/Q2 decisions and five proposed production YAML files, then "
            "publish one immutable commit-last admission cohort. Live configs are never edited."
        )
    )
    parser.add_argument(
        "--candidate-config",
        action="append",
        required=True,
        metavar="MODEL=PATH",
        help="Repeat exactly once for each of the five video models.",
    )
    parser.add_argument(
        "--decision",
        action="append",
        required=True,
        metavar="MODEL=PATH",
        help="Repeat exactly once for each of the five temporal qualification decisions.",
    )
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    commit = publish_temporal_promotion_cohort(
        candidate_config_paths=_mapping(
            args.candidate_config, label="--candidate-config"
        ),
        decision_paths=_mapping(args.decision, label="--decision"),
        root=Path(args.root),
    )
    print(json.dumps(commit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
