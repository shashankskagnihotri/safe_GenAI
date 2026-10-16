#!/usr/bin/env python3
"""Seal only evaluation source and local dependency versions; never copy credentials."""
import argparse
import importlib.metadata
from pathlib import Path
import platform
import shutil
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from riddikulus_evaluation.protocol import atomic_json, sha256, now


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    destination = args.output.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for subtree in ("src/riddikulus_evaluation", "scripts/riddikulus_evaluation"):
        for source in sorted((ROOT / subtree).iterdir()):
            if source.suffix not in {".py", ".md", ".sbatch", ".json"} or not source.is_file():
                continue
            relative = source.relative_to(ROOT)
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
            hashes[str(relative)] = sha256(target)
    manifest = {"schema": "riddikulus.evaluation-source.v1", "sealed_at": now(),
                "source_root": str(destination), "source_sha256": hashes,
                "interpreter": sys.executable, "python_version": platform.python_version(),
                "dependencies": {name: importlib.metadata.version(name) for name in ("google-genai", "pillow", "numpy", "matplotlib")}}
    atomic_json(destination / "source_manifest.json", manifest, immutable=True)
    print(destination / "source_manifest.json")


if __name__ == "__main__":
    main()
