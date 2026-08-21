#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hierasafe_flow.campaigns.push_for_iclr.ablation_manifests import (  # noqa: E402
    AblationManifestError,
    build_stage2_ablation_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the immutable 780-cell Stage-2 manifest")
    parser.add_argument("--code-commit", required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs/experiments/push_for_iclr/ablation_stage2_v1.yaml",
    )
    parser.add_argument(
        "--output-root", type=Path, default=REPOSITORY_ROOT / "outputs/PUSH_FOR_ICLR"
    )
    args = parser.parse_args()
    try:
        subprocess.run(
            ["git", "cat-file", "-e", f"{args.code_commit}^{{commit}}"],
            cwd=REPOSITORY_ROOT,
            check=True,
        )
        summary = build_stage2_ablation_manifest(
            repository_root=REPOSITORY_ROOT,
            config_path=args.config.resolve(),
            output_root=args.output_root.resolve(),
            code_commit=args.code_commit,
        )
    except (AblationManifestError, subprocess.CalledProcessError) as exc:
        print(f"FATAL: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
