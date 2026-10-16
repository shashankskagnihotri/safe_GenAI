#!/usr/bin/env python3
"""Prepare neutral tasks; Sol output contains exact fresh-agent spawn arguments."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from riddikulus_evaluation.protocol import prepare_next


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--judge", choices=("sol", "gemini"), required=True)
    parser.add_argument("--limit", type=int, default=3)
    args = parser.parse_args()
    print(json.dumps(prepare_next(args.images, args.expected, args.output, args.judge, args.limit), indent=2))


if __name__ == "__main__":
    main()
