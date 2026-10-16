#!/usr/bin/env python3
"""Export authenticated coverage, raw/derived tables and PNG/PDF/SVG plots."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from riddikulus_evaluation.plots import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluation", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-state", type=Path)
    parser.add_argument("--scheduler-timezone", default="Europe/Berlin")
    args = parser.parse_args()
    summary = run(args.evaluation, args.images, args.expected, args.output, args.campaign_state, args.scheduler_timezone)
    print(json.dumps({k: v for k, v in summary.items() if k != "evidence_sha256"}, indent=2))


if __name__ == "__main__":
    main()
