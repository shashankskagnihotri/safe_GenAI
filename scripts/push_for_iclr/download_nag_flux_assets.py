#!/usr/bin/env python3
"""Download and inventory only the immutable Diffusers FLUX.1-schnell snapshot."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import snapshot_download


ALLOW_PATTERNS = [
    "model_index.json",
    "scheduler/*",
    "text_encoder/*",
    "text_encoder_2/*",
    "tokenizer/*",
    "tokenizer_2/*",
    "transformer/*",
    "vae/*",
]
REQUIRED_PATHS = [
    "model_index.json",
    "scheduler/scheduler_config.json",
    "text_encoder/config.json",
    "text_encoder_2/config.json",
    "tokenizer/tokenizer_config.json",
    "tokenizer_2/tokenizer_config.json",
    "transformer/config.json",
    "vae/config.json",
]
FORBIDDEN_MONOLITHS = ["flux1-schnell.safetensors", "ae.safetensors"]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    upstream = Path(config["upstream"]["path"])
    expected_upstream = config["upstream"]["commit"]
    actual_upstream = git_head(upstream)
    if actual_upstream != expected_upstream:
        raise RuntimeError(
            f"NAG upstream mismatch: expected {expected_upstream}, got {actual_upstream}"
        )

    model = config["model"]
    local_dir = Path(model["local_dir"])
    local_dir.mkdir(parents=True, exist_ok=True)
    started = utc_now()
    resolved = snapshot_download(
        repo_id=model["repo_id"],
        revision=model["revision"],
        local_dir=local_dir,
        allow_patterns=ALLOW_PATTERNS,
    )
    if Path(resolved).resolve() != local_dir.resolve():
        raise RuntimeError(f"Unexpected snapshot location: {resolved}")

    missing = [name for name in REQUIRED_PATHS if not (local_dir / name).is_file()]
    forbidden = [name for name in FORBIDDEN_MONOLITHS if (local_dir / name).exists()]
    if missing or forbidden:
        raise RuntimeError(f"Asset contract failed: missing={missing}, forbidden={forbidden}")

    files = []
    for path in sorted(local_dir.rglob("*")):
        if not path.is_file() or ".cache" in path.relative_to(local_dir).parts:
            continue
        if path.name in {"ADMISSION.json", "FILES.sha256.json"}:
            continue
        files.append(
            {
                "path": path.relative_to(local_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    if not files:
        raise RuntimeError("Downloaded snapshot inventory is empty")

    inventory = {
        "schema_version": 1,
        "repo_id": model["repo_id"],
        "revision": model["revision"],
        "allow_patterns": ALLOW_PATTERNS,
        "files": files,
        "total_bytes": sum(item["bytes"] for item in files),
    }
    atomic_json(local_dir / "FILES.sha256.json", inventory)
    atomic_json(
        local_dir / "ADMISSION.json",
        {
            "schema_version": 1,
            "status": "ADMITTED_IMMUTABLE_DIFFUSERS_SNAPSHOT",
            "started_at_utc": started,
            "completed_at_utc": utc_now(),
            "repo_id": model["repo_id"],
            "revision": model["revision"],
            "upstream_commit": actual_upstream,
            "file_count": len(files),
            "total_bytes": inventory["total_bytes"],
            "host": platform.node(),
            "config": str(args.config.resolve()),
        },
    )
    print(json.dumps({"status": "admitted", "files": len(files)}, sort_keys=True))


if __name__ == "__main__":
    main()
