"""Immutable manifest construction for the development-only trust-region redesign."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import yaml

from .ablation_manifests import canonical_json, sha256_bytes, sha256_file
from .trust_region_controller import TrustRegionArm


HEX40 = re.compile(r"^[0-9a-f]{40}$")


class TrustRegionManifestError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TrustRegionManifestError(message)


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    _require(isinstance(value, dict), f"Expected YAML mapping: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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
        raise TrustRegionManifestError(f"Path is outside repository: {path}") from exc


def _slug(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    _require(bool(result), f"Cannot derive path component from {value!r}")
    return result


def build_trust_region_manifest(
    *,
    repository_root: Path,
    config_path: Path,
    output_root: Path,
    code_commit: str,
) -> dict[str, Any]:
    _require(bool(HEX40.fullmatch(code_commit)), "code_commit must be a full lowercase SHA")
    config = _load_yaml(config_path)
    _require(
        config.get("schema_version") == "push-for-iclr.trust-region-development.v1",
        "Bad trust-region config schema",
    )
    _require(config["split_role"] == "development_only_method_redesign", "Bad split role")
    _require(config["slurm"]["array_throttle"] is None, "Array throttle is forbidden")
    _require(config["slurm"]["user_hold"] is False, "User hold is forbidden")
    conditioning_memory_policy = config["generation"].get(
        "conditioning_memory_policy", "persistent_cpu_per_call_gpu_materialization_v1"
    )
    _require(
        conditioning_memory_policy == "persistent_cpu_per_call_gpu_materialization_v1",
        "Unsupported conditioning memory policy",
    )
    probe_context_conditioning = config["generation"].get(
        "probe_context_conditioning",
        "safety_concept_prefix_plus_exact_original_prompt",
    )
    _require(
        probe_context_conditioning
        in {
            "safety_concept_prefix_plus_exact_original_prompt",
            "concept_only_pair_endpoint_v1",
        },
        "Unsupported probe context conditioning",
    )

    source_spec = config["source_manifest"]
    _require(
        source_spec["path"] == "outputs/PUSH_FOR_ICLR/MANIFESTS/ablation_20_development.jsonl",
        "Only the frozen development manifest is allowed",
    )
    _require(
        source_spec["forbidden_manifest_substring"] not in source_spec["path"],
        "Locked validation manifest is forbidden during redesign",
    )
    source_path = repository_root / source_spec["path"]
    _require(sha256_file(source_path) == source_spec["file_sha256"], "Source file hash mismatch")
    source_rows = _read_jsonl(source_path)
    _require(len(source_rows) == source_spec["row_count"] == 20, "Expected 20 development rows")
    _require(
        {row["manifest_sha256"] for row in source_rows} == {source_spec["manifest_sha256"]},
        "Source payload hash mismatch",
    )
    _require(
        {row["ablation_split"] for row in source_rows} == {source_spec["required_split"]},
        "Non-development row selected",
    )
    _require(
        {row["category"] for row in source_rows} == {"nudity", "violence"},
        "Category coverage changed",
    )
    _require(all(row["execution_policy"] == "RUNNABLE" for row in source_rows), "Blocked row selected")

    arms = [TrustRegionArm.from_mapping(value) for value in config["arms"]]
    _require(len(arms) == 12, "Expected exactly 12 redesign arms")
    _require(len({arm.arm_id for arm in arms}) == len(arms), "Duplicate arm IDs")
    _require(arms[0].arm_id == "R00_BASELINE" and not arms[0].enabled, "Bad baseline arm")

    ontology_records: dict[str, dict[str, Any]] = {}
    for category, relative in config["ontologies"].items():
        path = repository_root / relative
        ontology = _load_yaml(path)
        _require(ontology["schema_version"] == 2, f"Bad ontology schema for {category}")
        _require(ontology["category"] == category, f"Ontology category mismatch: {category}")
        _require(ontology["prompt_independent"] is True, "Ontology must be prompt-independent")
        _require(bool(ontology["runtime_probe_pairs"]), f"No probe pairs for {category}")
        _require(bool(ontology["neutral_probes"]), f"No neutral probes for {category}")
        ontology_records[category] = {
            "path": relative,
            "sha256": sha256_file(path),
            "id": ontology["ontology_id"],
        }

    dependency_paths = [
        "src/hierasafe_flow/adapters/base.py",
        "src/hierasafe_flow/adapters/registry.py",
        "src/hierasafe_flow/adapters/flux_adapter.py",
        "src/hierasafe_flow/adapters/sd35_adapter.py",
        "src/hierasafe_flow/steering/local_masks.py",
        "src/hierasafe_flow/steering/vector_fields.py",
        "src/hierasafe_flow/campaigns/push_for_iclr/ablation_controller.py",
        "src/hierasafe_flow/campaigns/push_for_iclr/ablation_manifests.py",
        "src/hierasafe_flow/campaigns/push_for_iclr/trust_region_controller.py",
        "src/hierasafe_flow/campaigns/push_for_iclr/trust_region_manifests.py",
        "scripts/push_for_iclr/run_trust_region_cell.py",
    ]
    dependency_sha256 = {
        relative: sha256_file(repository_root / relative) for relative in dependency_paths
    }
    config_sha256 = sha256_file(config_path)
    rows: list[dict[str, Any]] = []
    for model_id, model_spec in config["models"].items():
        model_path = repository_root / model_spec["config"]
        model_config = _load_yaml(model_path)
        _require(model_config["model"]["revision"] == model_spec["revision"], "Revision mismatch")
        model_sha256 = sha256_file(model_path)
        for arm_value, arm in zip(config["arms"], arms):
            for source_row in source_rows:
                ontology_record = ontology_records[source_row["category"]]
                prompt_id = _slug(source_row["row_id"])
                relative_output = (
                    Path("BY_MODEL")
                    / model_id
                    / config["output"]["stage_directory"]
                    / arm.arm_id
                    / source_row["category"]
                    / prompt_id
                    / f"seed_{config['seed']}"
                )
                rows.append(
                    {
                        "job_schema_version": "push-for-iclr.trust-region-job.v1",
                        "campaign_id": config["campaign_id"],
                        "stage": 7,
                        "split_role": config["split_role"],
                        "job_index": len(rows),
                        "job_key": f"{model_id}/{arm.arm_id}/{source_row['row_id']}/seed_{config['seed']}",
                        "model_id": model_id,
                        "model_config_path": model_spec["config"],
                        "model_config_sha256": model_sha256,
                        "model_revision": model_spec["revision"],
                        "generation": model_config["generation"],
                        "arm_id": arm.arm_id,
                        "arm": arm_value,
                        "category": source_row["category"],
                        "prompt_id": prompt_id,
                        "source_row_id": source_row["row_id"],
                        "original_prompt": source_row["original_prompt"],
                        "original_prompt_sha256": source_row["original_prompt_sha256"],
                        "difficulty_stratum": source_row["difficulty_stratum"],
                        "ablation_split": source_row["ablation_split"],
                        "seed": int(config["seed"]),
                        "ontology_id": ontology_record["id"],
                        "ontology_path": ontology_record["path"],
                        "ontology_sha256": ontology_record["sha256"],
                        "prompt_specific_ontology_used": False,
                        "probe_context_conditioning": probe_context_conditioning,
                        "conditioning_memory_policy": conditioning_memory_policy,
                        "margin": float(model_spec["margin"]),
                        "mask": model_spec["mask"],
                        "expected_output_relative_path": str(relative_output),
                        "source_manifest_path": source_spec["path"],
                        "source_manifest_sha256": source_spec["manifest_sha256"],
                        "source_manifest_file_sha256": source_spec["file_sha256"],
                        "stage_config_path": _relative(config_path, repository_root),
                        "stage_config_sha256": config_sha256,
                        "runtime_dependency_sha256": dependency_sha256,
                        "code_commit": code_commit,
                        "execution_policy": "RUN_EXACT_DEVELOPMENT_CELL_NO_REWRITE_NO_FALLBACK",
                    }
                )

    expected = len(config["models"]) * len(arms) * len(source_rows)
    _require(expected == config["slurm"]["array_count"] == 480, "Matrix arithmetic mismatch")
    _require(len(rows) == 480, "Expected 480 jobs")
    _require([row["job_index"] for row in rows] == list(range(480)), "Bad indices")
    _require(len({row["job_key"] for row in rows}) == 480, "Duplicate job keys")
    _require(len({row["expected_output_relative_path"] for row in rows}) == 480, "Collision")

    payload = "".join(f"{canonical_json(row)}\n" for row in rows).encode("utf-8")
    manifest_sha256 = sha256_bytes(payload)
    sealed = [{**row, "job_manifest_sha256": manifest_sha256} for row in rows]
    physical = "".join(f"{canonical_json(row)}\n" for row in sealed).encode("utf-8")
    manifest_filename = config["output"].get(
        "manifest_filename", "trust_region_development_480.jsonl"
    )
    prepared_registry_filename = config["output"].get(
        "prepared_registry_filename", "trust_region_development_480_prepared.json"
    )
    _require(
        Path(manifest_filename).name == manifest_filename and manifest_filename.endswith(".jsonl"),
        "Manifest filename must be a JSONL basename",
    )
    _require(
        Path(prepared_registry_filename).name == prepared_registry_filename
        and prepared_registry_filename.endswith(".json"),
        "Prepared registry filename must be a JSON basename",
    )
    manifest_path = output_root / "MANIFESTS" / manifest_filename
    file_sha256 = _write_immutable(manifest_path, physical)
    summary = {
        "schema_version": "push-for-iclr.trust-region-registry.v1",
        "status": "PREPARED_NOT_SUBMITTED",
        "stage": 7,
        "split_role": config["split_role"],
        "code_commit": code_commit,
        "job_count": 480,
        "array_expression": "0-479",
        "array_throttle": None,
        "user_hold": False,
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256,
        "manifest_file_sha256": file_sha256,
        "source_prompt_manifest_sha256": source_spec["manifest_sha256"],
        "stage_config_sha256": config_sha256,
        "output_stage_directory": config["output"]["stage_directory"],
        "conditioning_memory_policy": conditioning_memory_policy,
        "probe_context_conditioning": probe_context_conditioning,
        "models": list(config["models"]),
        "arms": [arm.arm_id for arm in arms],
        "ontologies": ontology_records,
    }
    registry_path = output_root / "JOB_REGISTRY" / prepared_registry_filename
    registry_payload = (json.dumps(summary, indent=2, sort_keys=True) + "\n").encode("utf-8")
    summary["registry_file_sha256"] = _write_immutable(registry_path, registry_payload)
    return summary
