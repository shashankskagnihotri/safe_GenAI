#!/usr/bin/env python3
"""Ingest the exact JSON and completed-spawn evidence from a fresh HERE Sol agent."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from riddikulus_evaluation.protocol import ingest_sol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--agent-id", required=True)
    parser.add_argument("--spawn-receipt", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(ingest_sol(args.task.resolve(), args.response, args.agent_id, args.spawn_receipt), indent=2))


if __name__ == "__main__":
    main()
