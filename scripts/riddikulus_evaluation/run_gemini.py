#!/usr/bin/env python3
"""Explicit production Gemini request for one prepared stage; no implicit dispatch."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from riddikulus_evaluation.gemini import configuration, run_task
from riddikulus_evaluation.protocol import read_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path)
    parser.add_argument("--show-config", action="store_true")
    parser.add_argument("--campaign-state", type=Path,
                        help="Optional explicit campaign root; otherwise task's EVALUATION parent is used")
    args = parser.parse_args()
    if args.show_config:
        print(json.dumps(configuration(), indent=2))
    elif args.task:
        evaluation = args.task.resolve().parents[2]
        campaign = args.campaign_state.resolve().parent if args.campaign_state else evaluation.parent
        stop_paths = (campaign / "STOP", evaluation / "STOP")
        if any(path.exists() for path in stop_paths):
            print(json.dumps({"status": "STOPPED", "task_status": read_json(args.task.with_name("state.json"))["status"],
                              "reason": "Campaign or evaluation STOP marker; no API call made"}))
            raise SystemExit(2)
        result = run_task(args.task, stop_paths=stop_paths)
        print(json.dumps(result, indent=2))
        if result.get("status") not in {"COMPLETE", "COMPLETE_WITH_ISSUES"}:
            raise SystemExit(2)
    else:
        parser.error("Supply --show-config or one prepared --task")


if __name__ == "__main__":
    main()
