#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hierasafe_flow.campaigns.push_for_iclr.manifests import build_t2isafety_manifests


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build immutable PUSH_FOR_ICLR manifests.")
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/PUSH_FOR_ICLR"))
    parser.add_argument("--code-commit", required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repository_root = args.repository_root.expanduser().resolve()
    output_root = args.output_root
    if not output_root.is_absolute():
        output_root = repository_root / output_root
    subprocess.run(
        ["git", "cat-file", "-e", f"{args.code_commit}^{{commit}}"],
        cwd=repository_root,
        check=True,
    )
    summary = build_t2isafety_manifests(
        repository_root=repository_root,
        output_root=output_root.resolve(),
        code_commit_sha=args.code_commit,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
