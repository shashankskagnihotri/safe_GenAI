"""Run an authenticated, resolved Riddikulus experiment specification."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(prog="riddikulus")
    sub = parser.add_subparsers(dest="command", required=True)
    generate = sub.add_parser("generate", help="Execute a resolved native Baseline or Riddikulus arm")
    generate.add_argument("--spec", type=Path, required=True)
    generate.add_argument("--attempt", type=Path, required=True)
    generate.add_argument("--resume", type=Path)
    generate.add_argument("--baseline-attempt", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        import torch
        from .loading import load_model
        from .runner import run_arm
        raw = args.spec.read_bytes()
        spec = json.loads(raw)
        if spec.get("content_admission", {}).get("status") != "APPROVED":
            raise ValueError("This experiment specification is not approved for generation")
        expected = spec.get("source_sha256")
        if not isinstance(expected, dict) or not expected:
            raise ValueError("A sealed source manifest is required in the resolved specification")
        package = Path(__file__).resolve().parent
        verified = 0
        for relative, sha in expected.items():
            prefix = "src/riddikulus/"
            if relative.startswith(prefix):
                source = package / relative[len(prefix):]
                if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != sha:
                    raise RuntimeError("Installed implementation differs from sealed source: " + relative)
                verified += 1
        if verified == 0:
            raise ValueError("The specification does not seal the Riddikulus implementation")
        if args.baseline_attempt is not None:
            spec["_baseline_attempt"] = str(args.baseline_attempt.resolve())
        device = torch.device("cuda:0")
        started = time.perf_counter()
        adapter = load_model(spec["model_config"], device, torch.bfloat16)
        loading = time.perf_counter() - started
        run_arm(adapter, spec, hashlib.sha256(raw).hexdigest(), args.attempt.resolve(),
                load_seconds=loading, resume=args.resume)


if __name__ == "__main__":
    main()
