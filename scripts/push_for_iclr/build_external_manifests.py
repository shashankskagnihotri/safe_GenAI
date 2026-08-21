#!/usr/bin/env python3
"""Seal external PUSH_FOR_ICLR prompt manifests from pinned local sources."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hierasafe_flow.campaigns.push_for_iclr.external_manifests import (  # noqa: E402
    ExternalManifestError,
    build_external_manifests,
)


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT
        / "configs/experiments/push_for_iclr/external_benchmarks_v1.yaml",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--safe-denoiser-root",
        type=Path,
        default=REPOSITORY_ROOT / "debugging/t2i_safety_27_july/upstream/repos/Safe_Denoiser",
    )
    parser.add_argument(
        "--overt-root",
        type=Path,
        default=REPOSITORY_ROOT / "debugging/push_for_iclr/upstream/OVERT",
    )
    parser.add_argument(
        "--hf-cache-root",
        type=Path,
        default=Path.home() / ".cache/huggingface/hub",
    )
    return parser.parse_args()


def _require_local_commit(commit: str) -> None:
    result = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPOSITORY_ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ExternalManifestError(f"Code commit is not present in this repository: {commit}")


def main() -> int:
    args = _arguments()
    try:
        _require_local_commit(args.code_commit)
        summary = build_external_manifests(
            config_path=args.config.resolve(),
            output_root=args.output_root.resolve(),
            code_commit=args.code_commit,
            repository_root=REPOSITORY_ROOT,
            safe_denoiser_root=args.safe_denoiser_root.resolve(),
            overt_root=args.overt_root.resolve(),
            hf_cache_root=args.hf_cache_root.resolve(),
        )
    except ExternalManifestError as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
