#!/usr/bin/env python3
"""Verify completeness and cross-arm distinctness before manual NAG admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: object) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    root = Path(config["output_root"])
    gate_path = root / "MACHINE_INTEGRITY_GATE.json"
    if gate_path.exists():
        raise FileExistsError(f"Refusing to overwrite {gate_path}")

    expected_per_arm = len(config["prompts"]) * len(config["generation"]["seeds"])
    expected_total = expected_per_arm * len(config["arms"])
    errors = []
    records_by_cell: dict[tuple[int, int], dict[str, str]] = {}
    observed_total = 0
    for arm in config["arms"]:
        arm_id = arm["id"]
        manifest_path = root / arm_id / "MANIFEST.json"
        if not manifest_path.is_file():
            errors.append(f"missing manifest: {manifest_path}")
            continue
        manifest = load_json(manifest_path)
        records = manifest.get("records", [])
        if len(records) != expected_per_arm:
            errors.append(f"{arm_id}: expected {expected_per_arm} records, got {len(records)}")
        for record in records:
            path = Path(record["path"])
            if not path.is_file():
                errors.append(f"missing image: {path}")
                continue
            with Image.open(path) as image:
                if image.size != (
                    config["generation"]["width"],
                    config["generation"]["height"],
                ):
                    errors.append(f"wrong dimensions: {path}: {image.size}")
                image.verify()
            digest = sha256_file(path)
            if digest != record.get("sha256"):
                errors.append(f"hash mismatch: {path}")
            key = (record["prompt_index"], record["seed"])
            records_by_cell.setdefault(key, {})[arm_id] = digest
            observed_total += 1

    distinctness = []
    for key, arm_hashes in sorted(records_by_cell.items()):
        unique_hashes = len(set(arm_hashes.values()))
        distinctness.append(
            {
                "prompt_index": key[0],
                "seed": key[1],
                "arm_hashes": arm_hashes,
                "unique_hash_count": unique_hashes,
            }
        )
        if len(arm_hashes) != len(config["arms"]):
            errors.append(f"incomplete cross-arm cell: {key}: {sorted(arm_hashes)}")
        elif unique_hashes != len(config["arms"]):
            errors.append(f"byte-identical arm outputs in cell {key}")

    contact = root / "CONTACT_SHEETS" / "page_000.jpg"
    if not contact.is_file():
        errors.append(f"missing contact sheet: {contact}")
    if observed_total != expected_total:
        errors.append(f"expected {expected_total} total records, got {observed_total}")

    passed = not errors
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": (
            "MACHINE_COMPLETE_AWAITING_MANUAL_VISUAL_REVIEW"
            if passed
            else "FAILED_MACHINE_INTEGRITY"
        ),
        "passed": passed,
        "claim_scope": config["claim_scope"],
        "expected_images": expected_total,
        "observed_images": observed_total,
        "errors": errors,
        "distinctness": distinctness,
        "contact_sheet": str(contact),
        "manual_visual_admission": False,
    }
    atomic_json(gate_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
