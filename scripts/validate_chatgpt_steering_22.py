#!/usr/bin/env python3
"""Structural validator for the corrected 252-cell campaign."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from hierasafe_flow.campaigns.chatgpt_steering_22_july import (
    load_campaign_spec,
    media_files,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/chatgpt_steering_22_july.yaml",
    )
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    spec = load_campaign_spec(args.config)
    root = Path(spec["_root"])
    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    failures = []
    counts: dict[str, int] = {}
    for row in rows:
        output = root / row["output_dir"]
        final = output.parent / "FINAL_ATTEMPT"
        active = final if final.is_dir() else output
        status_path = active / "cell_status.json"
        if not status_path.is_file():
            failures.append({"cell_id": row["cell_id"], "reason": "missing status"})
            continue
        status = json.loads(status_path.read_text(encoding="utf-8"))
        value = str(status.get("status"))
        counts[value] = counts.get(value, 0) + 1
        if row["execution_status"] == "unsupported_by_model":
            if value != "unsupported_by_model" or media_files(active):
                failures.append(
                    {"cell_id": row["cell_id"], "reason": "invalid unsupported cell"}
                )
            continue
        if value != "completed":
            failures.append({"cell_id": row["cell_id"], "reason": f"status={value}"})
            continue
        if len(media_files(active)) != 1:
            failures.append(
                {"cell_id": row["cell_id"], "reason": "media count is not one"}
            )
        if not list(active.rglob("steering_trace.json")):
            failures.append({"cell_id": row["cell_id"], "reason": "missing trace"})
    result = {
        "rows": len(rows),
        "status_counts": counts,
        "failure_count": len(failures),
        "failures": failures,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
