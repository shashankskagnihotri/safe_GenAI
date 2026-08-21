#!/usr/bin/env python3
"""Build full and diagnostic manifests for the corrected 22 July campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.campaigns.chatgpt_steering_22_july import (
    atomic_json,
    build_matrix,
    load_campaign_spec,
    write_matrix,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/chatgpt_steering_22_july.yaml",
    )
    parser.add_argument("--attempt", type=int, default=1)
    args = parser.parse_args()
    spec = load_campaign_spec(args.config)
    state = Path(spec["_root"]) / spec["campaign"]["state_root"]
    rows = build_matrix(spec, attempt=args.attempt)
    summary = write_matrix(rows, state / "matrix.jsonl")
    calibration = [row for row in rows if row["variant"] == "baseline"]
    _write_jsonl(state / "calibration.jsonl", calibration)
    diagnostic = [
        row
        for row in rows
        if (
            row["prompt_id"] == "01_sad_young_girl"
            and row["model_id"] == "flux2_dev"
        )
        or (
            row["prompt_id"] == "01_sad_young_girl"
            and row["model_id"] == "cogvideox_5b"
            and row["variant"]
            in {"baseline", "native_negative_prompt", "current_conceptsteer"}
        )
    ]
    _write_jsonl(state / "diagnostic_matrix.jsonl", diagnostic)
    diagnostic_calibration = [
        row
        for row in calibration
        if row["prompt_id"] == "01_sad_young_girl"
        and row["model_id"] == "flux2_dev"
    ]
    _write_jsonl(state / "diagnostic_calibration.jsonl", diagnostic_calibration)
    atomic_json(
        state / "manifest_build.json",
        {
            **summary,
            "calibration_rows": len(calibration),
            "diagnostic_rows": len(diagnostic),
            "diagnostic_calibration_rows": len(diagnostic_calibration),
        },
    )
    print(json.dumps({**summary, "state_root": str(state)}, sort_keys=True))


if __name__ == "__main__":
    main()
