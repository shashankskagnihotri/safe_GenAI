#!/usr/bin/env python3
"""Record an actual failed assigned agent; never use this to reroll valid scores."""
from pathlib import Path
import argparse
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from riddikulus_evaluation.protocol import TECHNICAL_FAILURES, read_json, record_failure


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--assignment-token", required=True)
    parser.add_argument("--failure-class", choices=sorted(TECHNICAL_FAILURES | {"provider_refusal", "ambiguous_execution", "invalid_input"}), required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    args = parser.parse_args()
    detail = read_json(args.evidence)
    if not isinstance(detail, dict) or not detail:
        raise ValueError("An actual failure requires nonempty saved evidence")
    print(json.dumps(record_failure(args.task, args.assignment_token, args.failure_class, detail), indent=2))


if __name__ == "__main__":
    main()
