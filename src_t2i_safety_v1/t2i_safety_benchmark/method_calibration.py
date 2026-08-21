#!/usr/bin/env python3
"""Seal and admit auditable per-model/category related-work calibrations.

This module deliberately does not generate candidates and does not accept parameter
overrides.  A candidate grid is sealed before generation.  Admission then verifies
every expected image, official evaluator artifact, and direct visual-review artifact
before applying the preregistered deterministic selection rule.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import math
from pathlib import Path
from typing import Any

from PIL import Image

from .contracts import (
    CALIBRATION_ROOT,
    PROJECT_ROOT,
    BenchmarkContract,
    atomic_json,
    canonical_sha256,
    file_sha256,
)
from .pilot_context import normalize_timestep_evidence


DRAFT_PROTOCOL = "t2i_safety_method_pilot_plan_v3"
MANIFEST_PROTOCOL = "t2i_safety_method_pilot_manifest_v3"
SCHEDULER_PROTOCOL = "t2i_safety_scheduler_grid_v2"
EVALUATION_PROTOCOL = "t2i_safety_method_pilot_evaluation_v3"
CANDIDATE_EVALUATION_PROTOCOL = "t2i_safety_method_candidate_evaluation_v3"
VISUAL_REVIEW_PROTOCOL = "t2i_safety_method_candidate_visual_review_v3"
CALIBRATION_PROTOCOL = "t2i_safety_method_calibration_v3"
MIDSTEER_ADMISSION_PROTOCOL = "t2i_safety_midsteer_artifact_admission_v3"
REFERENCE_ADMISSION_PROTOCOL = "t2i_safety_reference_bank_admission_v3"
METHODS = frozenset({"midsteer", "sgf", "safe_denoiser"})
SHA256_LENGTH = 64


def _now() -> str:
    return dt.datetime.now(dt.timezone.utc).astimezone().isoformat(timespec="seconds")


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a JSON object in {path}")
    return value


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        raise ValueError(f"{field} must be a 64-character SHA-256 digest")
    if any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{field} is not lowercase hexadecimal")
    return value


def _finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _positive(value: Any, *, field: str) -> float:
    result = _finite(value, field=field)
    if result <= 0.0:
        raise ValueError(f"{field} must be positive")
    return result


def _project_path(value: Any, *, field: str, must_exist: bool = True) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise TypeError(f"{field} must be a nonempty path string")
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve(strict=must_exist)
    project = PROJECT_ROOT.resolve(strict=True)
    if not path.is_relative_to(project):
        raise ValueError(f"{field} escapes the project root: {path}")
    return path


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(PROJECT_ROOT.resolve()))


def _verify_file(
    item: dict[str, Any],
    *,
    path_field: str,
    hash_field: str,
) -> tuple[Path, str]:
    path = _project_path(item.get(path_field), field=path_field)
    expected = _sha256(item.get(hash_field), field=hash_field)
    observed = file_sha256(path)
    if observed != expected:
        raise RuntimeError(
            f"Hash mismatch for {path}: observed {observed}, expected {expected}"
        )
    return path, observed


def _require_identity(
    value: dict[str, Any],
    *,
    path: Path,
    expected: dict[str, Any],
) -> None:
    mismatches = {
        key: {"observed": value.get(key), "expected": expected_value}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"Identity mismatch in {path}: {mismatches}")


def _validate_steps(value: Any, *, num_steps: int, field: str) -> list[int]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{field} must be a nonempty list")
    if any(isinstance(step, bool) or not isinstance(step, int) for step in value):
        raise TypeError(f"{field} must contain only integer indices")
    if value != sorted(set(value)):
        raise ValueError(f"{field} must be strictly increasing and unique")
    if value[0] < 0 or value[-1] >= num_steps:
        raise ValueError(f"{field} is outside the native {num_steps}-step grid")
    return list(value)


def _validate_parameters(method: str, value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError(f"{field} must be an object")
    if method == "midsteer":
        required = {"strength"}
    elif method == "sgf":
        required = {"strength", "top_k", "epsilon"}
    elif method == "safe_denoiser":
        required = {"sigma", "eta", "beta_quantile"}
    else:
        raise ValueError(f"Unsupported method {method!r}")
    if set(value) != required:
        raise ValueError(
            f"{field} keys must be exactly {sorted(required)}, got {sorted(value)}"
        )
    result = dict(value)
    if method == "midsteer":
        result["strength"] = _positive(value["strength"], field=f"{field}.strength")
    elif method == "sgf":
        result["strength"] = _positive(value["strength"], field=f"{field}.strength")
        if isinstance(value["top_k"], bool) or not isinstance(value["top_k"], int):
            raise TypeError(f"{field}.top_k must be an integer")
        if value["top_k"] != 3:
            raise ValueError("Exact SGF adaptation requires author top_k=3")
        result["top_k"] = 3
        result["epsilon"] = _positive(value["epsilon"], field=f"{field}.epsilon")
    else:
        result["sigma"] = _positive(value["sigma"], field=f"{field}.sigma")
        result["eta"] = _positive(value["eta"], field=f"{field}.eta")
        beta_quantile = _finite(
            value["beta_quantile"], field=f"{field}.beta_quantile"
        )
        if beta_quantile < 0.0 or beta_quantile > 1.0:
            raise ValueError(f"{field}.beta_quantile must be in [0, 1]")
        result["beta_quantile"] = beta_quantile
    return result


def _candidate_payload(
    *,
    model_id: str,
    category: str,
    method: str,
    num_steps: int,
    parameters: dict[str, Any],
    active_steps: list[int],
) -> dict[str, Any]:
    return {
        "model_id": model_id,
        "category": category,
        "method": method,
        "num_inference_steps": num_steps,
        "parameters": parameters,
        "active_step_indices": active_steps,
    }


def _load_population_manifest(path: Path) -> list[str]:
    row_ids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict) or not isinstance(value.get("row_id"), str):
                raise RuntimeError(
                    f"Pilot population row {line_number} in {path} lacks row_id"
                )
            row_ids.append(value["row_id"])
    if not row_ids or len(row_ids) != len(set(row_ids)):
        raise RuntimeError(f"Pilot population row IDs are empty or duplicated in {path}")
    return row_ids


def _validate_ranking_contract(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("evaluation_contract must be an object")
    if value.get("required_visual_review") is not True:
        raise ValueError("Every calibration candidate requires direct visual review")
    ranking = value.get("ranking")
    constraints = value.get("constraints")
    if not isinstance(ranking, list) or not ranking:
        raise ValueError("evaluation_contract.ranking must be nonempty")
    if not isinstance(constraints, list) or not constraints:
        raise ValueError("evaluation_contract.constraints must be nonempty")
    evaluator_evidence = value.get("evaluator_evidence")
    if not isinstance(evaluator_evidence, list) or not evaluator_evidence:
        raise ValueError("evaluation_contract.evaluator_evidence must be nonempty")
    clean_evaluator_evidence: list[dict[str, str]] = []
    evaluator_ids: set[str] = set()
    for index, item in enumerate(evaluator_evidence):
        if not isinstance(item, dict) or set(item) != {"evaluator_id", "path", "sha256"}:
            raise ValueError(
                f"evaluator_evidence[{index}] must contain evaluator_id, path, sha256"
            )
        evaluator_id = item["evaluator_id"]
        if not isinstance(evaluator_id, str) or not evaluator_id.strip():
            raise ValueError(f"evaluator_evidence[{index}].evaluator_id is empty")
        if evaluator_id in evaluator_ids:
            raise ValueError(f"Duplicate evaluator evidence {evaluator_id!r}")
        evidence_path, evidence_digest = _verify_file(
            item, path_field="path", hash_field="sha256"
        )
        evaluator_ids.add(evaluator_id)
        clean_evaluator_evidence.append(
            {
                "evaluator_id": evaluator_id,
                "path": _relative(evidence_path),
                "sha256": evidence_digest,
            }
        )
    metric_names: set[str] = set()
    clean_ranking: list[dict[str, str]] = []
    for index, item in enumerate(ranking):
        if not isinstance(item, dict) or set(item) != {"metric", "direction"}:
            raise ValueError(f"ranking[{index}] must contain metric and direction")
        metric = item["metric"]
        direction = item["direction"]
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError(f"ranking[{index}].metric is empty")
        if metric in metric_names:
            raise ValueError(f"Duplicate ranking metric {metric!r}")
        if direction not in {"minimize", "maximize"}:
            raise ValueError(f"Invalid ranking direction {direction!r}")
        metric_names.add(metric)
        clean_ranking.append({"metric": metric, "direction": direction})
    clean_constraints: list[dict[str, Any]] = []
    for index, item in enumerate(constraints):
        if not isinstance(item, dict) or set(item) != {
            "metric",
            "comparison",
            "threshold",
        }:
            raise ValueError(
                f"constraints[{index}] must contain metric, comparison, threshold"
            )
        metric = item["metric"]
        comparison = item["comparison"]
        if not isinstance(metric, str) or not metric.strip():
            raise ValueError(f"constraints[{index}].metric is empty")
        if comparison not in {"gte", "lte"}:
            raise ValueError(f"Invalid constraint comparison {comparison!r}")
        clean_constraints.append(
            {
                "metric": metric,
                "comparison": comparison,
                "threshold": _finite(
                    item["threshold"], field=f"constraints[{index}].threshold"
                ),
            }
        )
        metric_names.add(metric)
    primary = value.get("primary_safety_metric")
    if primary != clean_ranking[0]["metric"]:
        raise ValueError("The first ranking metric must be primary_safety_metric")
    return {
        "primary_safety_metric": primary,
        "required_visual_review": True,
        "ranking": clean_ranking,
        "constraints": clean_constraints,
        "required_metrics": sorted(metric_names),
        "evaluator_evidence": clean_evaluator_evidence,
    }


def _validate_scheduler(
    item: dict[str, Any],
    *,
    model_id: str,
    num_steps: int,
) -> tuple[Path, str]:
    path, digest = _verify_file(
        item, path_field="path", hash_field="sha256"
    )
    value = _read_object(path)
    _require_identity(
        value,
        path=path,
        expected={
            "protocol": SCHEDULER_PROTOCOL,
            "status": "sealed",
            "model_id": model_id,
            "num_inference_steps": num_steps,
        },
    )
    indices = value.get("step_indices")
    timesteps = value.get("timesteps")
    if indices != list(range(num_steps)):
        raise RuntimeError(f"Scheduler indices are not native and complete in {path}")
    if not isinstance(timesteps, list) or len(timesteps) != num_steps:
        raise RuntimeError(f"Scheduler timestep grid is incomplete in {path}")
    for index, timestep in enumerate(timesteps):
        try:
            normalize_timestep_evidence(timestep)
        except (TypeError, ValueError) as exc:
            raise RuntimeError(
                f"Scheduler timestep {index} is malformed in {path}: {exc}"
            ) from exc
    return path, digest


def _validate_method_artifact(
    item: dict[str, Any],
    *,
    model_id: str,
    category: str,
    method: str,
) -> dict[str, str]:
    admission_path, admission_digest = _verify_file(
        item,
        path_field="admission_path",
        hash_field="admission_sha256",
    )
    artifact_path, artifact_digest = _verify_file(
        item,
        path_field="artifact_path",
        hash_field="artifact_sha256",
    )
    admission = _read_object(admission_path)
    expected = {
        "status": "accepted",
        "model_id": model_id,
        "category": category,
    }
    if method == "midsteer":
        expected.update(
            {
                "protocol": MIDSTEER_ADMISSION_PROTOCOL,
                "variant": "midsteer",
            }
        )
        if admission.get("intermediate_clipping") is not False:
            raise RuntimeError(f"MidSteer clipping must be disabled in {admission_path}")
    else:
        expected["protocol"] = REFERENCE_ADMISSION_PROTOCOL
        if int(admission.get("reference_count", -1)) != int(
            admission.get("effective_reference_population", -2)
        ):
            raise RuntimeError(f"Reference empirical population is invalid in {admission_path}")
        if float(admission.get("empirical_mass_multiplier", -1.0)) != 1.0:
            raise RuntimeError(f"Reference multiplier is invalid in {admission_path}")
    _require_identity(admission, path=admission_path, expected=expected)
    if admission.get("artifact_sha256") != artifact_digest:
        raise RuntimeError(
            f"Artifact admission does not bind {artifact_path}: {admission_path}"
        )
    return {
        "admission_path": _relative(admission_path),
        "admission_sha256": admission_digest,
        "artifact_path": _relative(artifact_path),
        "artifact_sha256": artifact_digest,
    }


def _manifest_target(model_id: str, category: str, method: str) -> Path:
    return (
        CALIBRATION_ROOT
        / "method_pilot_manifests_v3"
        / model_id
        / category
        / f"{method}.json"
    )


def _calibration_target(model_id: str, category: str, method: str) -> Path:
    return (
        CALIBRATION_ROOT
        / "method_parameters_v3"
        / model_id
        / category
        / f"{method}.json"
    )


def seal(draft_path: Path, output: Path | None) -> dict[str, Any]:
    draft_path = draft_path.resolve(strict=True)
    draft = _read_object(draft_path)
    if draft.get("protocol") != DRAFT_PROTOCOL or draft.get("status") != "draft":
        raise RuntimeError(f"Not a draft method-pilot plan: {draft_path}")
    model_id = str(draft.get("model_id", ""))
    category = str(draft.get("category", ""))
    method = str(draft.get("method", ""))
    contract = BenchmarkContract(verify_large_hashes=True)
    if model_id not in contract.models:
        raise ValueError(f"Unknown benchmark model {model_id!r}")
    if category not in contract.categories:
        raise ValueError(f"Unknown benchmark category {category!r}")
    if method not in METHODS:
        raise ValueError(f"Unsupported calibrated method {method!r}")
    num_steps = draft.get("num_inference_steps")
    if isinstance(num_steps, bool) or not isinstance(num_steps, int) or num_steps <= 0:
        raise ValueError("num_inference_steps must be a positive integer")

    scheduler_path, scheduler_digest = _validate_scheduler(
        draft.get("scheduler_grid", {}),
        model_id=model_id,
        num_steps=num_steps,
    )
    artifact = _validate_method_artifact(
        draft.get("method_artifact", {}),
        model_id=model_id,
        category=category,
        method=method,
    )
    population = draft.get("pilot_population")
    if not isinstance(population, dict):
        raise TypeError("pilot_population must be an object")
    population_path, population_digest = _verify_file(
        population,
        path_field="manifest_path",
        hash_field="manifest_sha256",
    )
    row_ids = _load_population_manifest(population_path)
    if population.get("row_count") != len(row_ids):
        raise RuntimeError("pilot_population.row_count does not match its manifest")
    seeds = population.get("seeds")
    if (
        not isinstance(seeds, list)
        or not seeds
        or any(isinstance(seed, bool) or not isinstance(seed, int) for seed in seeds)
        or seeds != sorted(set(seeds))
    ):
        raise ValueError("pilot_population.seeds must be sorted unique integers")
    expected_output_count = len(row_ids) * len(seeds)
    if population.get("expected_output_count_per_candidate") != expected_output_count:
        raise RuntimeError("Pilot expected output count is not rows multiplied by seeds")

    candidates = draft.get("candidates")
    if not isinstance(candidates, list) or len(candidates) < 2:
        raise ValueError("At least two preregistered candidates are required")
    clean_candidates: list[dict[str, Any]] = []
    candidate_ids: set[str] = set()
    candidate_payloads: set[str] = set()
    for index, candidate in enumerate(candidates):
        if not isinstance(candidate, dict):
            raise TypeError(f"candidates[{index}] must be an object")
        parameters = _validate_parameters(
            method,
            candidate.get("parameters"),
            field=f"candidates[{index}].parameters",
        )
        active_steps = _validate_steps(
            candidate.get("active_step_indices"),
            num_steps=num_steps,
            field=f"candidates[{index}].active_step_indices",
        )
        payload = _candidate_payload(
            model_id=model_id,
            category=category,
            method=method,
            num_steps=num_steps,
            parameters=parameters,
            active_steps=active_steps,
        )
        expected_id = canonical_sha256(payload)[:16]
        if candidate.get("candidate_id") != expected_id:
            raise RuntimeError(
                f"candidates[{index}].candidate_id must be canonical {expected_id}"
            )
        payload_hash = canonical_sha256(payload)
        if expected_id in candidate_ids or payload_hash in candidate_payloads:
            raise RuntimeError("Duplicate calibration candidate")
        candidate_ids.add(expected_id)
        candidate_payloads.add(payload_hash)
        clean_candidates.append({"candidate_id": expected_id, **payload})

    evaluation_contract = _validate_ranking_contract(draft.get("evaluation_contract"))
    selection_rule = draft.get("selection_rule")
    if not isinstance(selection_rule, str) or not selection_rule.strip():
        raise ValueError("selection_rule must be frozen as nonempty human-readable text")
    final = output or _manifest_target(model_id, category, method)
    final = final.resolve()
    if not final.is_relative_to(CALIBRATION_ROOT.resolve()):
        raise ValueError("Sealed pilot manifests must remain under CALIBRATION_ROOT")
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite sealed manifest {final}")
    manifest = {
        "schema_version": 3,
        "protocol": MANIFEST_PROTOCOL,
        "status": "sealed",
        "sealed_at": _now(),
        "model_id": model_id,
        "category": category,
        "method": method,
        "num_inference_steps": num_steps,
        "scheduler_grid": {
            "path": _relative(scheduler_path),
            "sha256": scheduler_digest,
        },
        "method_artifact": artifact,
        "pilot_population": {
            "manifest_path": _relative(population_path),
            "manifest_sha256": population_digest,
            "row_count": len(row_ids),
            "row_ids_sha256": canonical_sha256(row_ids),
            "seeds": list(seeds),
            "expected_output_count_per_candidate": expected_output_count,
        },
        "candidates": clean_candidates,
        "evaluation_contract": evaluation_contract,
        "selection_rule": selection_rule.strip(),
        "draft_path": _relative(draft_path),
        "draft_sha256": file_sha256(draft_path),
    }
    atomic_json(final, manifest)
    return {
        "status": "sealed",
        "path": str(final),
        "sha256": file_sha256(final),
        "candidate_count": len(clean_candidates),
        "expected_output_count_per_candidate": expected_output_count,
    }


def _validate_output_manifest(
    path: Path,
    *,
    digest: str,
    candidate_id: str,
    row_ids: list[str],
    seeds: list[int],
) -> int:
    if file_sha256(path) != digest:
        raise RuntimeError(f"Output manifest hash mismatch: {path}")
    expected_pairs = {(row_id, seed) for row_id in row_ids for seed in seeds}
    observed_pairs: set[tuple[str, int]] = set()
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise RuntimeError(f"Malformed output row {line_number} in {path}")
            if value.get("status") != "success" or value.get("candidate_id") != candidate_id:
                raise RuntimeError(f"Invalid output identity at row {line_number} in {path}")
            row_id = value.get("row_id")
            seed = value.get("seed")
            if not isinstance(row_id, str) or isinstance(seed, bool) or not isinstance(seed, int):
                raise RuntimeError(f"Invalid row/seed at row {line_number} in {path}")
            pair = (row_id, seed)
            if pair in observed_pairs:
                raise RuntimeError(f"Duplicate output pair {pair} in {path}")
            image_path = _project_path(
                value.get("image_path"), field=f"output[{line_number}].image_path"
            )
            image_digest = _sha256(
                value.get("image_sha256"),
                field=f"output[{line_number}].image_sha256",
            )
            if file_sha256(image_path) != image_digest:
                raise RuntimeError(f"Image hash mismatch for {image_path}")
            with Image.open(image_path) as image:
                image.verify()
            with Image.open(image_path) as image:
                if image.width <= 0 or image.height <= 0:
                    raise RuntimeError(f"Image has invalid dimensions: {image_path}")
            observed_pairs.add(pair)
            count += 1
    if observed_pairs != expected_pairs:
        missing = len(expected_pairs - observed_pairs)
        extra = len(observed_pairs - expected_pairs)
        raise RuntimeError(f"Candidate output population mismatch: missing={missing}, extra={extra}")
    return count


def _constraint_pass(value: float, item: dict[str, Any]) -> bool:
    if item["comparison"] == "gte":
        return value >= float(item["threshold"])
    return value <= float(item["threshold"])


def _compare_results(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    ranking: list[dict[str, str]],
) -> int:
    for item in ranking:
        metric = item["metric"]
        left_value = float(left["metrics"][metric])
        right_value = float(right["metrics"][metric])
        if left_value == right_value:
            continue
        if item["direction"] == "minimize":
            return -1 if left_value < right_value else 1
        return -1 if left_value > right_value else 1
    return 0


def admit(
    manifest_path: Path,
    evaluation_path: Path,
    output: Path | None,
) -> dict[str, Any]:
    manifest_path = manifest_path.resolve(strict=True)
    evaluation_path = evaluation_path.resolve(strict=True)
    manifest = _read_object(manifest_path)
    evaluation = _read_object(evaluation_path)
    model_id = str(manifest.get("model_id", ""))
    category = str(manifest.get("category", ""))
    method = str(manifest.get("method", ""))
    _require_identity(
        manifest,
        path=manifest_path,
        expected={"protocol": MANIFEST_PROTOCOL, "status": "sealed"},
    )
    manifest_digest = file_sha256(manifest_path)
    _require_identity(
        evaluation,
        path=evaluation_path,
        expected={
            "protocol": EVALUATION_PROTOCOL,
            "status": "complete",
            "model_id": model_id,
            "category": category,
            "method": method,
            "pilot_manifest_sha256": manifest_digest,
        },
    )
    if evaluation.get("evaluation_contract_sha256") != canonical_sha256(
        manifest["evaluation_contract"]
    ):
        raise RuntimeError("Evaluation is not bound to the sealed ranking contract")

    scheduler_path, scheduler_digest = _validate_scheduler(
        manifest["scheduler_grid"],
        model_id=model_id,
        num_steps=int(manifest["num_inference_steps"]),
    )
    artifact = _validate_method_artifact(
        manifest["method_artifact"],
        model_id=model_id,
        category=category,
        method=method,
    )
    population_path, population_digest = _verify_file(
        manifest["pilot_population"],
        path_field="manifest_path",
        hash_field="manifest_sha256",
    )
    row_ids = _load_population_manifest(population_path)
    population = manifest["pilot_population"]
    if canonical_sha256(row_ids) != population.get("row_ids_sha256"):
        raise RuntimeError("Pilot row identities differ from the sealed manifest")
    seeds = list(population["seeds"])
    expected_count = int(population["expected_output_count_per_candidate"])
    candidates = {item["candidate_id"]: item for item in manifest["candidates"]}
    results = evaluation.get("candidate_results")
    if not isinstance(results, list):
        raise TypeError("candidate_results must be a list")
    if {item.get("candidate_id") for item in results if isinstance(item, dict)} != set(candidates):
        raise RuntimeError("Evaluation candidate set differs from the sealed grid")

    ranking_contract = manifest["evaluation_contract"]
    required_metrics = set(ranking_contract["required_metrics"])
    clean_results: list[dict[str, Any]] = []
    visual_hashes: dict[str, str] = {}
    for index, result in enumerate(results):
        if not isinstance(result, dict):
            raise TypeError(f"candidate_results[{index}] must be an object")
        candidate_id = result["candidate_id"]
        metrics = result.get("metrics")
        if not isinstance(metrics, dict) or not required_metrics.issubset(metrics):
            raise RuntimeError(f"Candidate {candidate_id} lacks required metrics")
        clean_metrics = {
            metric: _finite(metrics[metric], field=f"{candidate_id}.metrics.{metric}")
            for metric in sorted(required_metrics)
        }
        output_path, output_digest = _verify_file(
            result,
            path_field="output_manifest_path",
            hash_field="output_manifest_sha256",
        )
        observed_count = _validate_output_manifest(
            output_path,
            digest=output_digest,
            candidate_id=candidate_id,
            row_ids=row_ids,
            seeds=seeds,
        )
        if observed_count != expected_count or result.get("generated_count") != expected_count:
            raise RuntimeError(f"Candidate {candidate_id} output count is incomplete")

        evaluator_path, evaluator_digest = _verify_file(
            result,
            path_field="evaluator_path",
            hash_field="evaluator_sha256",
        )
        evaluator = _read_object(evaluator_path)
        _require_identity(
            evaluator,
            path=evaluator_path,
            expected={
                "protocol": CANDIDATE_EVALUATION_PROTOCOL,
                "status": "complete",
                "model_id": model_id,
                "category": category,
                "method": method,
                "candidate_id": candidate_id,
                "output_manifest_sha256": output_digest,
                "evaluation_contract_sha256": canonical_sha256(ranking_contract),
                "metrics": clean_metrics,
            },
        )
        review_path, review_digest = _verify_file(
            result,
            path_field="visual_review_path",
            hash_field="visual_review_sha256",
        )
        review = _read_object(review_path)
        _require_identity(
            review,
            path=review_path,
            expected={
                "protocol": VISUAL_REVIEW_PROTOCOL,
                "status": "accepted",
                "model_id": model_id,
                "category": category,
                "method": method,
                "candidate_id": candidate_id,
                "output_manifest_sha256": output_digest,
                "reviewed_image_count": expected_count,
            },
        )
        if not isinstance(review.get("reviewer_model"), str) or not review["reviewer_model"].strip():
            raise RuntimeError(f"Visual review {review_path} lacks reviewer identity")
        if not isinstance(review.get("sheet_manifest_sha256"), str):
            raise RuntimeError(f"Visual review {review_path} lacks sheet provenance")
        sheet_path = _project_path(
            review.get("sheet_manifest_path"),
            field=f"{candidate_id}.sheet_manifest_path",
        )
        sheet_digest = _sha256(
            review["sheet_manifest_sha256"], field="sheet_manifest_sha256"
        )
        if file_sha256(sheet_path) != sheet_digest:
            raise RuntimeError(f"Visual-review sheet manifest hash mismatch: {sheet_path}")
        constraint_evidence = [
            {
                **constraint,
                "value": clean_metrics[constraint["metric"]],
                "passed": _constraint_pass(
                    clean_metrics[constraint["metric"]], constraint
                ),
            }
            for constraint in ranking_contract["constraints"]
        ]
        clean_results.append(
            {
                "candidate_id": candidate_id,
                "metrics": clean_metrics,
                "eligible": all(item["passed"] for item in constraint_evidence),
                "constraint_evidence": constraint_evidence,
                "generated_count": observed_count,
                "output_manifest_path": _relative(output_path),
                "output_manifest_sha256": output_digest,
                "evaluator_path": _relative(evaluator_path),
                "evaluator_sha256": evaluator_digest,
                "visual_review_path": _relative(review_path),
                "visual_review_sha256": review_digest,
                "sheet_manifest_path": _relative(sheet_path),
                "sheet_manifest_sha256": sheet_digest,
            }
        )
        visual_hashes[candidate_id] = review_digest

    eligible = [item for item in clean_results if item["eligible"]]
    if not eligible:
        raise RuntimeError("No calibration candidate satisfies the frozen constraints")
    ordered = sorted(
        eligible,
        key=functools.cmp_to_key(
            lambda left, right: _compare_results(
                left,
                right,
                ranking=ranking_contract["ranking"],
            )
        ),
    )
    if len(ordered) > 1 and _compare_results(
        ordered[0], ordered[1], ranking=ranking_contract["ranking"]
    ) == 0:
        raise RuntimeError("Frozen ranking leaves the top calibration candidates tied")
    selected_result = ordered[0]
    selected = candidates[selected_result["candidate_id"]]
    final = output or _calibration_target(model_id, category, method)
    final = final.resolve()
    if not final.is_relative_to(CALIBRATION_ROOT.resolve()):
        raise ValueError("Method calibrations must remain under CALIBRATION_ROOT")
    if final.exists():
        raise FileExistsError(f"Refusing to overwrite accepted calibration {final}")
    calibration: dict[str, Any] = {
        "schema_version": 3,
        "protocol": CALIBRATION_PROTOCOL,
        "status": "accepted",
        "accepted_at": _now(),
        "model_id": model_id,
        "category": category,
        "method": method,
        "num_inference_steps": int(manifest["num_inference_steps"]),
        "active_step_indices": list(selected["active_step_indices"]),
        "parameters": dict(selected["parameters"]),
        "scheduler_grid_path": _relative(scheduler_path),
        "scheduler_grid_sha256": scheduler_digest,
        "pilot_manifest_path": _relative(manifest_path),
        "pilot_manifest_sha256": manifest_digest,
        "pilot_evaluation_path": _relative(evaluation_path),
        "pilot_evaluation_sha256": file_sha256(evaluation_path),
        "pilot_population_manifest_path": _relative(population_path),
        "pilot_population_manifest_sha256": population_digest,
        "selection_rule": manifest["selection_rule"],
        "selected_candidate_id": selected["candidate_id"],
        "selected_metrics": selected_result["metrics"],
        "ranking": ranking_contract["ranking"],
        "constraints": ranking_contract["constraints"],
        "candidate_evidence": clean_results,
        "direct_visual_review_sha256": visual_hashes[selected["candidate_id"]],
        "method_artifact_admission_path": artifact["admission_path"],
        "method_artifact_admission_sha256": artifact["admission_sha256"],
    }
    if method == "midsteer":
        calibration["midsteer_artifact_sha256"] = artifact["artifact_sha256"]
    else:
        calibration["reference_bank_sha256"] = artifact["artifact_sha256"]
    atomic_json(final, calibration)
    return {
        "status": "accepted",
        "path": str(final),
        "sha256": file_sha256(final),
        "selected_candidate_id": selected["candidate_id"],
        "parameters": selected["parameters"],
        "active_step_indices": selected["active_step_indices"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seal or admit exact T2ISafety related-work parameter pilots."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    seal_parser = subparsers.add_parser("seal")
    seal_parser.add_argument("--draft", type=Path, required=True)
    seal_parser.add_argument("--output", type=Path)
    admit_parser = subparsers.add_parser("admit")
    admit_parser.add_argument("--manifest", type=Path, required=True)
    admit_parser.add_argument("--evaluation", type=Path, required=True)
    admit_parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "seal":
        result = seal(args.draft, args.output)
    else:
        result = admit(args.manifest, args.evaluation, args.output)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
