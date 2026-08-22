#!/usr/bin/env python3
"""Build the immutable 24-cell V8 legacy-bottleneck development manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

import yaml


HEX40 = re.compile(r"^[0-9a-f]{40}$")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_bytes(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def write_immutable(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        raise
    return hashlib.sha256(payload).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                row = json.loads(line)
                require(isinstance(row, dict), f"Source line {line_number} is not an object")
                rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--code-commit", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    config_path = args.config.resolve()
    output_root = args.output_root.resolve()
    require(bool(HEX40.fullmatch(args.code_commit)), "code commit must be a full SHA")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    require(config["schema_version"] == "push-for-iclr.legacy-bottleneck-development.v1", "bad config schema")
    require(config["split_role"] == "development_only_method_redesign", "bad split role")
    require(config["slurm"]["array_throttle"] is None, "array throttle is forbidden")
    require(config["slurm"]["user_hold"] is False, "user hold is forbidden")

    source_spec = config["source_manifest"]
    source_path = repository_root / source_spec["path"]
    require(sha256_file(source_path) == source_spec["file_sha256"], "source file hash mismatch")
    source_rows = load_jsonl(source_path)
    selected_by_id = {
        row["row_id"]: row
        for row in source_rows
        if row.get("row_id") in config["prompt_ids"]
    }
    require(set(selected_by_id) == set(config["prompt_ids"]), "missing development prompt")
    require(all(row["ablation_split"] == "development" for row in selected_by_id.values()), "non-development prompt selected")

    hierarchy_path = repository_root / config["hierarchy_path"]
    hierarchy_sha256 = sha256_file(hierarchy_path)
    config_sha256 = sha256_file(config_path)
    dependencies = [
        Path("scripts/push_for_iclr/build_legacy_bottleneck_manifest.py"),
        Path("scripts/push_for_iclr/run_legacy_bottleneck_cell.py"),
        Path("src/hierasafe_flow/generation/runner.py"),
        Path("src/hierasafe_flow/steering/bottleneck.py"),
        Path("src/hierasafe_flow/steering/concept_graph.py"),
        Path("src/hierasafe_flow/steering/vector_fields.py"),
        Path("src/hierasafe_flow/steering/local_masks.py"),
        Path("src/hierasafe_flow/adapters/flux_adapter.py"),
    ]
    dependency_sha256 = {
        str(path): sha256_file(repository_root / path) for path in dependencies
    }

    arms = config["arms"]
    require(len(arms) == 12, "V8 requires exactly 12 arms")
    require(len({arm["id"] for arm in arms}) == 12, "duplicate arm id")
    hierarchy = yaml.safe_load(hierarchy_path.read_text(encoding="utf-8"))
    pair_ids = {pair["id"] for pair in hierarchy["pairs"]}
    for arm in arms:
        require(float(arm["strength"]) >= 0.0, f"negative strength in {arm['id']}")
        require(set(arm["active_pair_ids"]).issubset(pair_ids), f"unknown pair in {arm['id']}")

    rows: list[dict[str, Any]] = []
    stage_directory = config["output"]["stage_directory"]
    for arm in arms:
        for prompt_id in config["prompt_ids"]:
            source = selected_by_id[prompt_id]
            expected = (
                output_root
                / "BY_MODEL"
                / config["model"]["id"]
                / stage_directory
                / arm["id"]
                / source["category"]
                / prompt_id
                / f"seed_{int(config['seed'])}"
            )
            rows.append(
                {
                    "schema_version": "push-for-iclr.legacy-bottleneck-job.v1",
                    "job_index": len(rows),
                    "campaign_id": config["campaign_id"],
                    "stage": int(config["stage"]),
                    "split_role": config["split_role"],
                    "source_row_id": source["row_id"],
                    "source_release_index": source["release_index"],
                    "prompt_id": prompt_id,
                    "category": source["category"],
                    "difficulty_stratum": source["difficulty_stratum"],
                    "original_prompt": source["original_prompt"],
                    "original_prompt_sha256": source["original_prompt_sha256"],
                    "seed": int(config["seed"]),
                    "model": config["model"],
                    "generation": config["generation"],
                    "arm": arm,
                    "legacy_controller": config["legacy_controller"],
                    "hierarchy_path": config["hierarchy_path"],
                    "hierarchy_sha256": hierarchy_sha256,
                    "source_manifest_path": source_spec["path"],
                    "source_manifest_sha256": source_spec["manifest_sha256"],
                    "source_manifest_file_sha256": source_spec["file_sha256"],
                    "stage_config_path": str(config_path.relative_to(repository_root)),
                    "stage_config_sha256": config_sha256,
                    "runtime_dependency_sha256": dependency_sha256,
                    "code_commit": args.code_commit,
                    "expected_output_dir": str(expected),
                    "execution_policy": "RUN_EXACT_LEGACY_BOTTLENECK_DEVELOPMENT_CELL_NO_FALLBACK",
                }
            )

    require(len(rows) == 24, "V8 manifest must contain 24 rows")
    manifest_payload_sha256 = hashlib.sha256(canonical_bytes(rows)).hexdigest()
    manifest_bytes = b"".join(canonical_bytes(row) for row in rows)
    manifest_path = output_root / "MANIFESTS" / config["output"]["manifest_filename"]
    manifest_file_sha256 = write_immutable(manifest_path, manifest_bytes)
    summary = {
        "schema_version": "push-for-iclr.legacy-bottleneck-registry.v1",
        "status": "PREPARED_NOT_SUBMITTED",
        "stage": int(config["stage"]),
        "split_role": config["split_role"],
        "code_commit": args.code_commit,
        "job_count": len(rows),
        "array_expression": "0-23",
        "array_throttle": None,
        "manifest_path": str(manifest_path),
        "manifest_payload_sha256": manifest_payload_sha256,
        "manifest_file_sha256": manifest_file_sha256,
        "source_manifest_sha256": source_spec["manifest_sha256"],
        "stage_config_sha256": config_sha256,
        "hierarchy_sha256": hierarchy_sha256,
        "model": config["model"]["id"],
        "prompt_ids": config["prompt_ids"],
        "arms": [arm["id"] for arm in arms],
        "output_stage_directory": stage_directory,
    }
    registry_path = output_root / "JOB_REGISTRY" / config["output"]["prepared_registry_filename"]
    summary["registry_file_sha256"] = write_immutable(registry_path, canonical_bytes(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
