"""Immutable Stage-2 job-manifest construction."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

from .ablation_controller import ABLATION_IDS


HEX40 = re.compile(r"^[0-9a-f]{40}$")


class AblationManifestError(RuntimeError):
    pass


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AblationManifestError(message)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    _require(isinstance(value, dict), f"Expected YAML mapping: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            _require(isinstance(value, dict), f"Expected object at {path}:{line_number}")
            rows.append(value)
    return rows


def _write_immutable(path: Path, payload: bytes) -> str:
    _require(not path.exists(), f"Refusing to overwrite immutable artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    _require(not partial.exists(), f"Stale partial exists: {partial}")
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise
    return sha256_bytes(payload)


def _relative(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise AblationManifestError(f"Path is outside repository: {path}") from exc


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    _require(bool(result), f"Cannot derive path component from {value!r}")
    return result


def build_stage2_ablation_manifest(
    *,
    repository_root: Path,
    config_path: Path,
    output_root: Path,
    code_commit: str,
) -> dict[str, Any]:
    _require(bool(HEX40.fullmatch(code_commit)), "code_commit must be a full lowercase SHA")
    config = _load_yaml(config_path)
    _require(config.get("schema_version") == "push-for-iclr.ablation-stage2.v1", "Bad schema")
    _require(tuple(config["ablations"]) == ABLATION_IDS, "Ablation IDs/order changed")
    _require(config["slurm"]["array_throttle"] is None, "Array throttle is forbidden")
    _require(config["slurm"]["user_hold"] is False, "User hold is forbidden")

    source_spec = config["source_manifest"]
    source_path = repository_root / source_spec["path"]
    _require(sha256_file(source_path) == source_spec["file_sha256"], "Source file hash mismatch")
    source_rows = _read_jsonl(source_path)
    _require(len(source_rows) == source_spec["row_count"] == 30, "Expected exactly 30 prompts")
    _require(len({row["row_id"] for row in source_rows}) == 30, "Duplicate source row IDs")
    _require(
        {row["manifest_sha256"] for row in source_rows} == {source_spec["manifest_sha256"]},
        "Source manifest payload hash mismatch",
    )
    _require(all(row["execution_policy"] == "RUNNABLE" for row in source_rows), "Blocked row selected")
    _require(
        {row["category"] for row in source_rows} == {"nudity", "violence"},
        "Ablation categories changed",
    )

    dependency_paths = [
        "src/hierasafe_flow/adapters/base.py",
        "src/hierasafe_flow/adapters/registry.py",
        "src/hierasafe_flow/adapters/flux_adapter.py",
        "src/hierasafe_flow/adapters/sd35_adapter.py",
        "src/hierasafe_flow/steering/local_masks.py",
        "src/hierasafe_flow/steering/vector_fields.py",
        "src/hierasafe_flow/campaigns/push_for_iclr/ablation_controller.py",
        "src/hierasafe_flow/campaigns/push_for_iclr/ablation_manifests.py",
        "scripts/push_for_iclr/run_ablation_cell.py",
    ]
    dependency_sha256 = {
        relative: sha256_file(repository_root / relative) for relative in dependency_paths
    }
    config_sha256 = sha256_file(config_path)
    rows: list[dict[str, Any]] = []
    for model_id, model_spec in config["models"].items():
        model_config_path = repository_root / model_spec["config"]
        model_config = _load_yaml(model_config_path)
        _require(model_config["model"]["revision"] == model_spec["revision"], "Model revision mismatch")
        model_config_sha256 = sha256_file(model_config_path)
        for ablation_id in config["ablations"]:
            for source_row in source_rows:
                ontology_path = Path(source_row["ontology_path"])
                ontology_relative = _relative(ontology_path, repository_root)
                _require(
                    sha256_file(ontology_path) == source_row["ontology_sha256"],
                    f"Ontology changed for {source_row['row_id']}",
                )
                prompt_id = _slug(source_row["row_id"])
                relative_output = (
                    Path("BY_MODEL")
                    / model_id
                    / config["output"]["stage_directory"]
                    / ablation_id
                    / source_row["category"]
                    / prompt_id
                    / f"seed_{config['seed']}"
                )
                row = {
                    "job_schema_version": "push-for-iclr.ablation-job.v1",
                    "campaign_id": config["campaign_id"],
                    "stage": 2,
                    "job_index": len(rows),
                    "job_key": f"{model_id}/{ablation_id}/{source_row['row_id']}/seed_{config['seed']}",
                    "model_id": model_id,
                    "model_config_path": model_spec["config"],
                    "model_config_sha256": model_config_sha256,
                    "model_revision": model_spec["revision"],
                    "generation": model_config["generation"],
                    "ablation_id": ablation_id,
                    "category": source_row["category"],
                    "prompt_id": prompt_id,
                    "source_row_id": source_row["row_id"],
                    "original_prompt": source_row["original_prompt"],
                    "original_prompt_sha256": source_row["original_prompt_sha256"],
                    "difficulty_stratum": source_row["difficulty_stratum"],
                    "ablation_split": source_row["ablation_split"],
                    "seed": int(config["seed"]),
                    "ontology_id": source_row["ontology_id"],
                    "ontology_path": ontology_relative,
                    "ontology_sha256": source_row["ontology_sha256"],
                    "prompt_specific_ontology_used": False,
                    "relative_rms_strength": float(model_spec["relative_rms_strength"]),
                    "strength_provenance": model_spec["strength_provenance"],
                    "margin": float(model_spec["margin"]),
                    "mask": model_spec["mask"],
                    "early_window_unified_diffusion_time": config["schedule"][
                        "early_window_unified_diffusion_time"
                    ],
                    "expected_output_relative_path": str(relative_output),
                    "source_manifest_path": source_spec["path"],
                    "source_manifest_sha256": source_spec["manifest_sha256"],
                    "source_manifest_file_sha256": source_spec["file_sha256"],
                    "stage_config_path": _relative(config_path, repository_root),
                    "stage_config_sha256": config_sha256,
                    "runtime_dependency_sha256": dependency_sha256,
                    "code_commit": code_commit,
                    "execution_policy": "RUN_EXACT_CELL_NO_REWRITE_NO_FALLBACK",
                }
                rows.append(row)

    expected_count = len(config["models"]) * len(ABLATION_IDS) * len(source_rows)
    _require(expected_count == config["slurm"]["array_count"] == 780, "Matrix arithmetic mismatch")
    _require(len(rows) == 780, f"Expected 780 jobs, got {len(rows)}")
    _require([row["job_index"] for row in rows] == list(range(780)), "Non-contiguous indices")
    _require(len({row["job_key"] for row in rows}) == 780, "Duplicate job keys")
    _require(len({row["expected_output_relative_path"] for row in rows}) == 780, "Output collision")

    payload = "".join(f"{canonical_json(row)}\n" for row in rows).encode("utf-8")
    manifest_sha256 = sha256_bytes(payload)
    sealed_rows = [{**row, "job_manifest_sha256": manifest_sha256} for row in rows]
    physical = "".join(f"{canonical_json(row)}\n" for row in sealed_rows).encode("utf-8")
    manifest_path = output_root / "MANIFESTS/stage2_ablation_780.jsonl"
    file_sha256 = _write_immutable(manifest_path, physical)
    summary = {
        "schema_version": "push-for-iclr.ablation-job-registry.v1",
        "status": "PREPARED_NOT_SUBMITTED",
        "code_commit": code_commit,
        "job_count": 780,
        "array_expression": "0-779",
        "array_throttle": None,
        "user_hold": False,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_file_sha256": file_sha256,
        "source_prompt_manifest_sha256": source_spec["manifest_sha256"],
        "stage_config_sha256": config_sha256,
        "models": list(config["models"]),
        "ablations": list(ABLATION_IDS),
    }
    registry_path = output_root / "JOB_REGISTRY/stage2_ablation_780_prepared.json"
    registry_payload = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    summary["registry_file_sha256"] = _write_immutable(registry_path, registry_payload)
    return summary


def validate_sealed_job_manifest(path: Path) -> tuple[list[dict[str, Any]], str]:
    rows = _read_jsonl(path)
    _require(bool(rows), "Empty job manifest")
    hashes = {row.get("job_manifest_sha256") for row in rows}
    _require(len(hashes) == 1 and None not in hashes, "Inconsistent job manifest hash")
    expected = next(iter(hashes))
    payload_rows = [
        {key: value for key, value in row.items() if key != "job_manifest_sha256"} for row in rows
    ]
    payload = "".join(f"{canonical_json(row)}\n" for row in payload_rows).encode("utf-8")
    _require(sha256_bytes(payload) == expected, "Job manifest payload hash mismatch")
    return rows, expected
