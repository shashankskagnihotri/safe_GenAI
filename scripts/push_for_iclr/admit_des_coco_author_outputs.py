#!/usr/bin/env python3
"""Admit every DES COCO author output against frozen prompts and hashes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    config_bytes = args.config.read_bytes()
    config: Dict[str, Any] = json.loads(config_bytes)
    require(config.get("method") == "DES", "configuration is not DES")
    policy = config["author_output_admission"]
    require(policy.get("fallbacks_allowed") is False, "author-output fallback is forbidden")
    require(policy.get("verify_every_source_sha256") is True, "source hash verification is required")
    require(policy.get("verify_every_prompt_against_released_csv") is True, "prompt verification is required")

    paths = {key: Path(value) for key, value in config["paths"].items()}
    manifest = paths["author_generation_manifest"]
    author_images = paths["author_generation_images"]
    admission_path = paths["author_generation_admission"]
    expected_manifest_sha = config["expected_sha256"]["author_generation_manifest"]
    require(manifest.is_file(), f"missing author generation manifest: {manifest}")
    manifest_bytes = manifest.read_bytes()
    require(hashlib.sha256(manifest_bytes).hexdigest() == expected_manifest_sha, "author generation manifest hash mismatch")
    require(author_images.is_dir(), f"missing author image directory: {author_images}")
    require(not admission_path.exists(), f"refusing to overwrite admission: {admission_path}")

    with paths["prompt_csv"].open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        require(reader.fieldnames == ["prompt"], "released prompt CSV schema mismatch")
        prompts = [row["prompt"] for row in reader]

    rows = [json.loads(line) for line in manifest_bytes.decode("utf-8").splitlines() if line.strip()]
    expected_count = int(config["generated_images"]["count"])
    require(len(rows) == expected_count == len(prompts), "DES denominator mismatch")
    require([row.get("prompt_index") for row in rows] == list(range(expected_count)), "prompt indices are not exact 0..9968")

    filename_digest = hashlib.sha256()
    image_sha_digest = hashlib.sha256()
    byte_count = 0
    for index, (row, released_prompt) in enumerate(zip(rows, prompts)):
        require(row.get("prompt") == released_prompt, f"released prompt mismatch at index {index}")
        expected_source = author_images / policy["source_filename_template"].format(index=index)
        source = Path(row["image_path"])
        require(source == expected_source, f"author filename contract mismatch at index {index}")
        require(source.is_file() and source.stat().st_size > 0, f"missing author image at index {index}")
        observed_sha = sha256_file(source)
        require(observed_sha == row.get("image_sha256"), f"author image hash mismatch at index {index}")
        byte_count += source.stat().st_size
        filename_digest.update((source.name + "\n").encode())
        image_sha_digest.update((observed_sha + "\n").encode())

    receipt = {
        "schema_version": "push-for-iclr.des-author-generation-admission.v1",
        "status": "AUTHOR_GENERATION_ADMITTED",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "config": str(args.config.resolve()),
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "author_generation_manifest": str(manifest),
        "author_generation_manifest_sha256": expected_manifest_sha,
        "author_image_directory": str(author_images),
        "image_count": expected_count,
        "start_index": 0,
        "stop_index_exclusive": expected_count,
        "total_image_bytes": byte_count,
        "all_source_hashes_verified": True,
        "all_prompts_verified_against_released_csv": True,
        "author_filename_contract_preserved": True,
        "filename_digest": filename_digest.hexdigest(),
        "image_sha256_digest": image_sha_digest.hexdigest(),
        "fallback_used": False,
    }
    admission_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(admission_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
