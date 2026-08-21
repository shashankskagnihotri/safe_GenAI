"""Authenticate and target-blindly evaluate Flux calibration v2.

The semantic policy is intentionally inherited from the frozen v1 evaluator:
exact scale-1 inert-negative equality, objective non-collapse thresholds,
source-fidelity/path-parity hard gates, source-concept suppression only, and
one smallest globally passing scale across all three prompts.  This module
adds the selected-common-seed evidence chain, exact manifest-row binding, and
mandatory per-attempt environment-preflight authentication.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, __version__ as PILLOW_VERSION

from hierasafe_flow.benchmarks import finer_detailing_correction as finer
from hierasafe_flow.benchmarks.flux1_native_negative_calibration_v2 import (
    CALIBRATION_ID,
    CALIBRATION_STAGE as CALIBRATION_STAGE,
    ELIGIBLE_SCALES as ELIGIBLE_SCALES,
    EVIDENCE_ROLES as EVIDENCE_ROLES,
    EXPECTED_ROWS,
    MAX_SELECTED_COMMON_SEED,
    MODEL_ID,
    MODEL_NAME,
    MODEL_REVISION,
    PIPELINE_CLASS,
    PROMPT_IDS as PROMPT_IDS,
    ROLE_CONTROL as ROLE_CONTROL,
    ROLE_LADDER as ROLE_LADDER,
    ROLE_OFFICIAL as ROLE_OFFICIAL,
    SLURM_JOB_NAME,
    TRUE_CFG_SCALES as TRUE_CFG_SCALES,
    V1_OUTPUT_ROOT_RELATIVE,
    read_calibration_manifest_for_audit,
    sha256_file,
)
from hierasafe_flow.benchmarks.slurm_tracking import (
    read_environment_preflight,
    read_execution_identity,
    read_submission_registry,
)
from hierasafe_flow.evaluation import native_negative_calibration as v1
from hierasafe_flow.evaluation.flux1_dual_view_jobs_v3 import (
    ADAPTER_KEY as FLUX1_V3_ADAPTER_KEY,
    MODE_AUDIT as FLUX1_V3_MODE_AUDIT,
    NEGATIVE_MODE_EXPLICIT_NONE_CONTROL,
    NEGATIVE_MODE_NOT_APPLIED,
    NEGATIVE_MODE_PAIRED_REGISTERED,
    SOURCE_PROTOCOL_INPUT_ROLES as FLUX1_V3_SOURCE_INPUT_ROLES,
    validate_flux1_job_v3,
    validate_flux1_runtime_v3,
)
from hierasafe_flow.utils.immutable_publication import (
    cleanup_owned_staging,
    freeze_tree,
    publish_hardlink_tree_commit_last,
    require_nonwritable_directories,
)


SCHEMA_VERSION = 1
OBJECTIVE_EVALUATION = "flux1_native_negative_calibration_objective_v2"
REVIEW_PACKAGE = "flux1_native_negative_calibration_blinded_review_v2"
PARITY_REVIEW_PACKAGE = "flux1_native_negative_calibration_blinded_path_parity_review_v2"
PRIVATE_REVIEW_MANIFEST = "flux1_native_negative_calibration_private_review_map_v2"
UNBLINDING_MAP = "flux1_native_negative_calibration_unblinding_v2"
MANUAL_REVIEW = "flux1_native_negative_calibration_manual_review_v2"
SOURCE_MANUAL_REVIEW = "flux1_native_negative_calibration_source_manual_review_v2"
PARITY_MANUAL_REVIEW = "flux1_native_negative_calibration_path_parity_manual_review_v2"
SELECTION_REPORT = "flux1_native_negative_calibration_selection_v2"
MEDIA_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})
FINAL_EVIDENCE_ROOT_RELATIVE = Path(
    "debugging/audits/finer_detailing_20260720/"
    "flux1_native_negative_scale_calibration_v2_selection_attempt002"
)
FINAL_EVIDENCE_LEDGER_FILENAME = "manual_review_ledger.json"
FINAL_EVIDENCE_SELECTION_FILENAME = "selection_report.json"

# Re-export the exact frozen semantic vocabulary/measurements for callers and
# tests.  No v2 threshold or reviewer concept list is independently editable.
COMMON_HARD_GATES = v1.COMMON_HARD_GATES
PERSON_HARD_GATES = v1.PERSON_HARD_GATES
MALL_HARD_GATES = v1.MALL_HARD_GATES
PATH_PARITY_CATEGORIES = v1.PATH_PARITY_CATEGORIES
SOURCE_CONCEPTS = v1.SOURCE_CONCEPTS
ROW_VERDICTS = v1.ROW_VERDICTS
SOURCE_VERDICTS = v1.SOURCE_VERDICTS
PARITY_VERDICTS = v1.PARITY_VERDICTS
decoded_rgb = v1.decoded_rgb
image_statistics = v1.image_statistics
evaluate_noncollapse = v1.evaluate_noncollapse
canonical_sha256 = v1.canonical_sha256
document_digest = v1.document_digest


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot decode {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return payload


def _binding(path: str | Path) -> dict[str, Any]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"Cannot bind missing or empty evidence: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _require_access_disjoint_roots(*roots: Path, label: str) -> tuple[Path, ...]:
    """Reject equal or nested roots before publishing or joining review evidence."""

    resolved = tuple(root.expanduser().resolve() for root in roots)
    if len(set(resolved)) != len(resolved) or any(
        left in right.parents or right in left.parents
        for index, left in enumerate(resolved)
        for right in resolved[index + 1 :]
    ):
        raise ValueError(f"{label} must be separate, non-nested paths.")
    return resolved


def _finite_positive(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a JSON number, not a coercible value.")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a positive finite number.")
    return number


def _exact_selected_common_seed(value: Any, label: str) -> int:
    """Validate a seed independently of the source ladder's candidate count."""

    if (
        isinstance(value, bool)
        or not isinstance(value, int)
        or value < 0
        or value > MAX_SELECTED_COMMON_SEED
    ):
        raise ValueError(f"{label} must be one exact non-negative signed-64-bit integer.")
    return value


def _validate_manifest_seed_binding(binding: Any, selected_seed: int) -> None:
    """Require the authenticated manifest cohort's explicit seed-homogeneity proof."""

    if not isinstance(binding, Mapping) or (
        binding.get("selected_common_seed") != selected_seed
        or binding.get("seed_homogeneous_job_count") != EXPECTED_ROWS
        or binding.get("every_manifest_job_uses_selected_common_seed") is not True
    ):
        raise ValueError(
            "Calibration-v2 manifest cohort does not authenticate the selected common seed."
        )


def _parse_timing_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a timezone-aware ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not valid ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} has no UTC offset.")
    return parsed


def _expected_sample_paths(
    output_dir: Path, *, include_trace: bool = True
) -> dict[str, Path]:
    sample = output_dir / "sample_0000"
    paths = {
        "image_0": sample / "image_000.png",
        "report": sample / "report.json",
        "timing": sample / "timing.json",
    }
    if include_trace:
        paths["trace"] = sample / "steering_trace.json"
    return paths


def _decoded_rgb_bytes(
    payload: bytes, *, label: str, width: int, height: int
) -> np.ndarray:
    """Decode the exact media bytes authenticated by the central FLUX-v3 reader."""

    if not isinstance(payload, bytes) or not payload:
        raise ValueError(f"{label} must be non-empty authenticated bytes.")
    try:
        with Image.open(BytesIO(payload)) as probe:
            probe.verify()
        with Image.open(BytesIO(payload)) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError(
                    f"Calibration media must be an RGB PNG; got "
                    f"{image.format}/{image.mode}: {label}"
                )
            if image.size != (width, height):
                raise ValueError(
                    f"Calibration PNG dimensions differ from {(width, height)}: "
                    f"{image.size} at {label}"
                )
            rgb = np.asarray(image, dtype=np.uint8)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Calibration PNG is not decodable: {label}: {exc}") from exc
    if rgb.shape != (height, width, 3):
        raise ValueError(f"Decoded RGB shape drifted for {label}: {rgb.shape}.")
    return np.ascontiguousarray(rgb)


def _validate_timing(
    path: Path,
    *,
    label: str,
    job: Mapping[str, Any],
    manifest_sha256: str,
    selected_seed: int,
    authenticated_payload: Mapping[str, Any] | None = None,
) -> tuple[dict[str, Any], float]:
    payload = (
        deepcopy(dict(authenticated_payload))
        if authenticated_payload is not None
        else _load_json(path, label)
    )
    if payload.get("status") != "completed":
        raise ValueError(f"{label} is not complete: {path}")
    duration_key = "wall_seconds" if label == "experiment timing" else "total_seconds"
    duration = _finite_positive(payload.get(duration_key), f"{label}.{duration_key}")
    if label == "experiment timing":
        if set(payload) != {
            "status",
            "started_at_utc",
            "finished_at_utc",
            "wall_seconds",
        }:
            raise ValueError("Experiment timing schema drifted.")
        started = _parse_timing_timestamp(payload["started_at_utc"], "experiment started")
        ended = _parse_timing_timestamp(payload["finished_at_utc"], "experiment finished")
        elapsed = (ended - started).total_seconds()
        if elapsed <= 0 or abs(elapsed - duration) > 1.0:
            raise ValueError("Experiment timing timestamps and wall duration are inconsistent.")
    else:
        if payload.get("schema_version") != 1:
            raise ValueError(f"{label} schema version drifted: {path}")
        started = _parse_timing_timestamp(payload.get("started_at"), f"{label} started_at")
        ended = _parse_timing_timestamp(payload.get("ended_at"), f"{label} ended_at")
        elapsed = (ended - started).total_seconds()
        if elapsed <= 0 or abs(elapsed - duration) > 1.0:
            raise ValueError(f"{label} timestamps and total duration are inconsistent.")
        benchmark = payload.get("benchmark") or {}
        expected = {
            "attempt": job["attempt"],
            "condition_id": job["condition_id"],
            "manifest_sha256": manifest_sha256,
            "model_revision": MODEL_REVISION,
            "name": CALIBRATION_ID,
            "negative_prompt": job["negative_prompt"],
            "prompt_id": job["prompt_id"],
            "seed": selected_seed,
            "stage": CALIBRATION_STAGE,
            "variant": job["variant"],
        }
        if {key: benchmark.get(key) for key in expected} != expected:
            raise ValueError(f"{label} benchmark/selected-seed identity drifted: {path}")
        expected_options = job.get("native_negative_prompt_options") or {}
        if job["calibration_row"]["role"] == ROLE_OFFICIAL:
            if "native_negative_prompt_options" in benchmark:
                raise ValueError(f"{label} gives the official baseline native options.")
        elif benchmark.get("native_negative_prompt_options") != expected_options:
            raise ValueError(f"{label} native-negative options drifted: {path}")
        generation = payload.get("generation") or {}
        for key, value in job["generation"].items():
            if generation.get(key) != value:
                raise ValueError(f"{label} generation field {key!r} drifted: {path}")
        model = payload.get("model") or {}
        if {
            "adapter": model.get("adapter"),
            "model_id": model.get("model_id"),
            "revision": model.get("revision"),
            "pipeline_class": model.get("pipeline_class"),
        } != {
            "adapter": FLUX1_V3_ADAPTER_KEY,
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "pipeline_class": PIPELINE_CLASS,
        }:
            raise ValueError(f"{label} model identity drifted: {path}")
        trace_required = (
            job["calibration_row"]["role"] != ROLE_OFFICIAL
            or (job.get("output") or {}).get("save_traces") is True
        )
        expected_paths = _expected_sample_paths(
            Path(str(job["output_dir"])), include_trace=trace_required
        )
        if label == "sample timing":
            if (
                payload.get("prompt") != job["prompt"]
                or payload.get("sample_id") != "sample_0000"
                or payload.get("task") != "text_to_image"
                or payload.get("media") != {"present": True, "num_images": 1}
                or {
                    key: Path(str((payload.get("output_paths") or {}).get(key, "")))
                    for key in expected_paths
                }
                != expected_paths
            ):
                raise ValueError(f"Sample timing output/prompt identity drifted: {path}")
            phases = payload.get("phases_seconds")
            if (
                not isinstance(phases, Mapping)
                or not phases
                or any(
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                    or float(value) < 0
                    for value in phases.values()
                )
            ):
                raise ValueError(f"Sample timing phase accounting is invalid: {path}")
        else:
            records = payload.get("records")
            if (
                not isinstance(records, list)
                or len(records) != 1
                or records[0].get("prompt") != job["prompt"]
                or records[0].get("sample_id") != "sample_0000"
                or {
                    key: Path(str((records[0].get("output_paths") or {}).get(key, "")))
                    for key in expected_paths
                }
                != expected_paths
            ):
                raise ValueError(f"Run timing record/output identity drifted: {path}")
        if label == "run timing":
            validate_flux1_runtime_v3(job, run_timing=payload)
    return payload, duration


def _expected_native_call_provenance(job: Mapping[str, Any]) -> dict[str, Any]:
    row = job["calibration_row"]
    options = job["native_negative_prompt_options"]
    return {
        "schema_version": 1,
        "calibration_id": CALIBRATION_ID,
        "calibration_config_sha256": options["calibration_config_sha256"],
        "calibration_role": row["role"],
        "negative_prompt_mode": row["negative_prompt_mode"],
        "negative_prompt_is_none": row["role"] == ROLE_CONTROL,
        "true_cfg_scale": row["true_cfg_scale"],
        "only_controlled_call_difference": "paired_negative_prompt_values",
    }


def _validate_execution_report(
    *,
    job: Mapping[str, Any],
    manifest_sha256: str,
    image_path: Path,
    report: Mapping[str, Any],
    trace: Any,
    report_path: Path,
    trace_path: Path | None,
    audit_snapshot_root: Path | None = None,
    audit_snapshot_objects: Mapping[str, Path] | None = None,
    shared_job_already_authenticated: bool = False,
) -> tuple[Path, Path | None]:
    if not shared_job_already_authenticated:
        job_root = Path(str(job["base_config"])).resolve().parents[1]
        audit_kwargs: dict[str, Any] = {}
        if audit_snapshot_root is not None or audit_snapshot_objects is not None:
            if audit_snapshot_root is None or audit_snapshot_objects is None:
                raise ValueError("Flux-v3 evaluator audit snapshot context is incomplete.")
            archived_inputs: dict[str, Path] = {}
            for role in FLUX1_V3_SOURCE_INPUT_ROLES:
                record = (job.get("input_files") or {}).get(role)
                digest = str(record.get("sha256", "")) if isinstance(record, Mapping) else ""
                if digest not in audit_snapshot_objects:
                    raise ValueError(
                        f"Flux-v3 evaluator snapshot lacks static source role {role!r}."
                    )
                archived_inputs[role] = audit_snapshot_objects[digest]
            audit_kwargs = {
                "audit_snapshot_root": audit_snapshot_root,
                "audit_snapshot_input_paths": archived_inputs,
            }
        validate_flux1_job_v3(
            job,
            project_root=job_root,
            mode=FLUX1_V3_MODE_AUDIT,
            **audit_kwargs,
        )
    role = job["calibration_row"]["role"]
    expected_generation = {**dict(job["generation"]), "prompt": job["prompt"], "prompt_file": None}
    benchmark = report.get("benchmark") or {}
    expected_benchmark = {
        "attempt": job["attempt"],
        "condition_id": job["condition_id"],
        "manifest_sha256": manifest_sha256,
        "model_revision": MODEL_REVISION,
        "name": CALIBRATION_ID,
        "negative_prompt": job["negative_prompt"],
        "prompt_id": job["prompt_id"],
        "seed": job["seed"],
        "stage": CALIBRATION_STAGE,
        "variant": job["variant"],
    }
    model = report.get("model") or {}
    if (
        report.get("schema_version") != 1
        or report.get("prompt") != job["prompt"]
        or report.get("sample_id") != "sample_0000"
        or report.get("task") != "text_to_image"
        or {key: benchmark.get(key) for key in expected_benchmark} != expected_benchmark
        or report.get("generation") != expected_generation
        or {
            "adapter": model.get("adapter"),
            "model_id": model.get("model_id"),
            "revision": model.get("revision"),
            "pipeline_class": model.get("pipeline_class"),
        }
        != {
            "adapter": FLUX1_V3_ADAPTER_KEY,
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "pipeline_class": PIPELINE_CLASS,
        }
        or Path(str((report.get("output_paths") or {}).get("image_0", ""))) != image_path
        or (
            role != ROLE_OFFICIAL
            and (
                trace_path is None
                or Path(str((report.get("output_paths") or {}).get("trace", "")))
                != trace_path
            )
        )
    ):
        raise ValueError(f"Sample execution report identity drifted: {report_path}")

    negative_mode = job["calibration_row"]["negative_prompt_mode"]
    condition = report.get("condition")
    if not isinstance(trace, list) or (report.get("interpretability") or {}).get(
        "timesteps"
    ) != trace:
        raise ValueError("Execution trace and sample-report trace differ.")
    if role == ROLE_OFFICIAL:
        if (
            negative_mode != NEGATIVE_MODE_NOT_APPLIED
            or
            condition
            != {
                "steering_mode": "none",
                "decode_outputs": True,
                "is_native_negative_prompt": False,
                "negative_prompt": None,
            }
            or "native_negative_prompt_options" in benchmark
        ):
            raise ValueError("Official baseline execution report claims native conditioning.")
    else:
        options = job["native_negative_prompt_options"]
        actual_negative = None if role == ROLE_CONTROL else job["negative_prompt"]
        plan = job["generation"]["flux_dual_view_conditioning"]
        actual_negative_2 = (
            None
            if role == ROLE_CONTROL
            else plan["negative"]["t5_negative_prompt_2"]
        )
        provenance = _expected_native_call_provenance(job)
        if (
            negative_mode
            != (
                NEGATIVE_MODE_EXPLICIT_NONE_CONTROL
                if role == ROLE_CONTROL
                else NEGATIVE_MODE_PAIRED_REGISTERED
            )
            or
            benchmark.get("native_negative_prompt_options") != options
            or condition
            != {
                "steering_mode": "native_negative_prompt",
                "decode_outputs": True,
                "is_native_negative_prompt": True,
                "negative_prompt": actual_negative,
                "negative_prompt_2": actual_negative_2,
                "flux1_native_negative_calibration": provenance,
            }
        ):
            raise ValueError("Native execution report does not prove the exact Flux call.")
        if (
            len(trace) != 1
            or not isinstance(trace[0], Mapping)
            or trace[0].get("flux1_native_negative_calibration") != provenance
        ):
            raise ValueError("Native conditioning trace lost calibration provenance.")
        native_call = trace[0].get("flux_dual_view_native_call")
        expected_call_keys = {
            "prompt",
            "prompt_2",
            "negative_prompt",
            "negative_prompt_2",
            "guidance_scale",
            "true_cfg_scale",
        }
        if not isinstance(native_call, Mapping) or set(native_call) != expected_call_keys:
            raise ValueError("Native execution trace lacks the exact effective Flux call.")
        for key, expected in (
            ("guidance_scale", job["generation"].get("guidance_scale")),
            ("true_cfg_scale", options.get("true_cfg_scale")),
        ):
            actual = native_call.get(key)
            if (
                type(expected) is not float
                or not math.isfinite(expected)
                or type(actual) is not float
                or not math.isfinite(actual)
                or actual != expected
            ):
                raise ValueError(
                    "Native execution trace changed the sealed effective "
                    f"guidance argument {key!r}."
                )
    validate_flux1_runtime_v3(
        job,
        sample_report=report,
        conditioning_cache=(
            report.get("conditioning_cache")
            if isinstance(report.get("conditioning_cache"), Mapping)
            else None
        ),
        native_trace=trace if role != ROLE_OFFICIAL else None,
    )
    return report_path, trace_path


def _validate_result_row(
    *,
    job: Mapping[str, Any],
    job_index: int,
    manifest_sha256: str,
    registry_entry: Mapping[str, Any],
    selected_seed: int,
    reopened: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = Path(str(job["output_dir"]))
    bound = deepcopy(dict(job))
    bound["launch_manifest_sha256"] = manifest_sha256
    bound["launch_manifest_job_index"] = job_index
    if (
        reopened.get("schema_version") != 3
        or reopened.get("status") != "passed"
        or reopened.get("contract") != "finer_detailing_flux1_completed_output_reopen_v3"
        or reopened.get("manifest_sha256") != manifest_sha256
        or reopened.get("manifest_job_index") != job_index
    ):
        raise ValueError(f"Central FLUX-v3 reopening identity drifted at index {job_index}.")
    bindings = reopened.get("evidence_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError(f"Central FLUX-v3 evidence bindings are absent at index {job_index}.")
    role = job["calibration_row"]["role"]
    required_bindings = {
        "result",
        "resolved_config",
        "run_timing",
        "sample_report",
        "media",
    } | ({"steering_trace"} if role != ROLE_OFFICIAL else set())
    if set(bindings) != required_bindings:
        raise ValueError(f"Central FLUX-v3 evidence topology drifted at index {job_index}.")
    for key, binding in bindings.items():
        if (
            not isinstance(binding, Mapping)
            or not isinstance(binding.get("path"), str)
            or not isinstance(binding.get("sha256"), str)
            or len(binding["sha256"]) != 64
            or isinstance(binding.get("size_bytes"), bool)
            or not isinstance(binding.get("size_bytes"), int)
            or binding["size_bytes"] <= 0
        ):
            raise ValueError(
                f"Central FLUX-v3 evidence binding {key!r} is invalid at index {job_index}."
            )
    expected_binding_paths = {
        "result": output_dir / "benchmark_job_result.json",
        "resolved_config": output_dir / "resolved_config.yaml",
        "run_timing": output_dir / "run_timing.json",
        "sample_report": output_dir / "sample_0000" / "report.json",
        "media": output_dir / "sample_0000" / "image_000.png",
        **(
            {"steering_trace": output_dir / "sample_0000" / "steering_trace.json"}
            if role != ROLE_OFFICIAL
            else {}
        ),
    }
    if {
        key: Path(str(binding["path"])) for key, binding in bindings.items()
    } != expected_binding_paths:
        raise ValueError(f"Central FLUX-v3 evidence paths drifted at index {job_index}.")

    result = reopened.get("result")
    if not isinstance(result, Mapping):
        raise ValueError(f"Central FLUX-v3 result payload is absent at index {job_index}.")
    if (
        result.get("schema_version") != 2
        or result.get("status") != "completed"
        or result.get("job") != bound
    ):
        raise ValueError(f"Result does not bind exact manifest index {job_index}.")

    validation = reopened.get("media_validation")
    if not isinstance(validation, Mapping):
        raise ValueError(f"Central FLUX-v3 media validation is absent at index {job_index}.")
    image_path = Path(str(validation["path"]))
    media_binding = bindings["media"]
    media_bytes = reopened.get("media_bytes")
    if (
        Path(str(media_binding["path"])) != image_path
        or result.get("media_validation") != validation
        or result.get("validated_media_paths") != [str(image_path)]
        or not isinstance(media_bytes, bytes)
        or hashlib.sha256(media_bytes).hexdigest() != media_binding["sha256"]
        or validation.get("sha256") != media_binding["sha256"]
        or len(media_bytes) != media_binding["size_bytes"]
    ):
        raise ValueError(f"Result media binding drifted at index {job_index}.")
    rgb = _decoded_rgb_bytes(
        media_bytes,
        label=str(image_path),
        width=int(job["generation"]["width"]),
        height=int(job["generation"]["height"]),
    )
    report = reopened.get("sample_report")
    if not isinstance(report, Mapping):
        raise ValueError(f"Central FLUX-v3 sample report is absent at index {job_index}.")
    authenticated_run_timing = reopened.get("run_timing")
    if not isinstance(authenticated_run_timing, Mapping):
        raise ValueError(f"Central FLUX-v3 run timing is absent at index {job_index}.")
    report_path = Path(str(bindings["sample_report"]["path"]))
    trace_path = (
        Path(str(bindings["steering_trace"]["path"]))
        if role != ROLE_OFFICIAL
        else None
    )
    trace = (
        reopened.get("native_trace")
        if role != ROLE_OFFICIAL
        else (report.get("interpretability") or {}).get("timesteps")
    )
    report_path, trace_path = _validate_execution_report(
        job=bound,
        manifest_sha256=manifest_sha256,
        image_path=image_path,
        report=report,
        trace=trace,
        report_path=report_path,
        trace_path=trace_path,
        shared_job_already_authenticated=True,
    )

    preflight = read_environment_preflight(
        output_dir,
        expected_job=bound,
        expected_job_index=job_index,
    )
    if preflight.get("status") != "verified_before_generation":
        raise ValueError(f"Environment preflight is not verified at index {job_index}.")
    preflight_path = output_dir / "environment_preflight.json"
    preflight_sidecar = output_dir / "environment_preflight.json.sha256"

    identity_path = output_dir / "execution_identity.json"
    identity = read_execution_identity(identity_path)
    expected_identity = {
        "SLURM_ARRAY_JOB_ID": str(registry_entry["slurm_array_job_id"]),
        "SLURM_ARRAY_TASK_ID": str(job_index),
        "SLURM_JOB_NAME": SLURM_JOB_NAME,
        "slurm_task_id": str(registry_entry["slurm_task_id"]),
    }
    if {key: identity.get(key) for key in expected_identity} != expected_identity:
        raise ValueError(f"Slurm identity differs from registry at index {job_index}.")
    if registry_entry.get("slurm_array_task_id") != job_index:
        raise ValueError(f"Registry task ID differs from manifest index {job_index}.")

    experiment_path = output_dir / "experiment_timing.json"
    run_path = output_dir / "run_timing.json"
    sample_path = output_dir / "sample_0000" / "timing.json"
    _, experiment_wall = _validate_timing(
        experiment_path,
        label="experiment timing",
        job=job,
        manifest_sha256=manifest_sha256,
        selected_seed=selected_seed,
    )
    run, run_total = _validate_timing(
        run_path,
        label="run timing",
        job=job,
        manifest_sha256=manifest_sha256,
        selected_seed=selected_seed,
        authenticated_payload=authenticated_run_timing,
    )
    sample, sample_total = _validate_timing(
        sample_path,
        label="sample timing",
        job=job,
        manifest_sha256=manifest_sha256,
        selected_seed=selected_seed,
    )
    load_seconds = run.get("adapter_load_seconds", run.get("pipeline_load_seconds"))
    load_seconds = _finite_positive(load_seconds, "run timing model load")
    phases = sample["phases_seconds"]
    role = job["calibration_row"]["role"]
    if role == ROLE_OFFICIAL:
        if (
            set(phases)
            != {
                "prepare_latents_and_timesteps",
                "denoising",
                "decode",
                "save_media_trace_and_report",
            }
            or abs(sum(float(value) for value in phases.values()) - sample_total) > 1.0
        ):
            raise ValueError("Official sample timing phase accounting drifted.")
    else:
        if (
            set(phases)
            != {
                "pipeline_load",
                "native_pipeline_call",
                "save_media_trace_and_report",
            }
            or abs(float(phases["pipeline_load"]) - load_seconds) > 1.0
            or abs(
                float(phases["native_pipeline_call"])
                + float(phases["save_media_trace_and_report"])
                - sample_total
            )
            > 1.0
        ):
            raise ValueError("Native sample timing phase accounting drifted.")
    if run_total + 1.0 < load_seconds + sample_total or experiment_wall + 1.0 < run_total:
        raise ValueError("Experiment/run/sample timing hierarchy is inconsistent.")
    row = job["calibration_row"]
    return {
        "job_index": job_index,
        "condition_id": job["condition_id"],
        "prompt_id": job["prompt_id"],
        "prompt": job["prompt"],
        "selected_common_seed": selected_seed,
        "role": row["role"],
        "true_cfg_scale": row["true_cfg_scale"],
        "eligible_for_scale_selection": row["eligible_for_scale_selection"],
        "blind_id": row["blind_id"],
        "image_path": str(image_path),
        "image_sha256": validation["sha256"],
        "image_size_bytes": media_binding["size_bytes"],
        "width": validation["width"],
        "height": validation["height"],
        "rgb": rgb,
        "statistics": image_statistics(rgb),
        "runtime_seconds": {
            "experiment_wall": experiment_wall,
            "run_total": run_total,
            "sample_total": sample_total,
            "model_load": load_seconds,
        },
        "environment_status": preflight["status"],
        "manifest_job_index_authenticated": True,
        "artifact_bindings": {
            "benchmark_job_result": deepcopy(dict(bindings["result"])),
            "resolved_config": deepcopy(dict(bindings["resolved_config"])),
            "image": deepcopy(dict(media_binding)),
            "environment_preflight": _binding(preflight_path),
            "environment_preflight_sidecar": _binding(preflight_sidecar),
            "execution_identity": _binding(identity_path),
            "experiment_timing": _binding(experiment_path),
            "run_timing": deepcopy(dict(bindings["run_timing"])),
            "sample_timing": _binding(sample_path),
            "sample_execution_report": deepcopy(dict(bindings["sample_report"])),
            **(
                {
                    "conditioning_trace": deepcopy(
                        dict(bindings["steering_trace"])
                    )
                }
                if trace_path is not None
                else {}
            ),
        },
    }


def _bound_evidence_json(manifest: Mapping[str, Any], role: str) -> dict[str, Any]:
    record = manifest["selection_evidence"][role]
    path = Path(str(record["path"])).resolve()
    if path.is_file() and sha256_file(path) == record["sha256"]:
        return _load_json(path, role)
    snapshot = manifest.get("snapshot_bundle") or {}
    snapshot_root = Path(str(snapshot.get("root_path", ""))).resolve()
    index = _load_json(Path(str(snapshot.get("index_path", ""))).resolve(), "snapshot index")
    object_record = (index.get("objects") or {}).get(str(record["sha256"]))
    if not isinstance(object_record, Mapping):
        raise ValueError(f"Snapshot does not contain selected-seed evidence {role!r}.")
    archived = (snapshot_root / str(object_record.get("path", ""))).resolve()
    if (
        snapshot_root not in archived.parents
        or not archived.is_file()
        or sha256_file(archived) != record["sha256"]
    ):
        raise ValueError(f"Archived selected-seed evidence changed: {role!r}.")
    return _load_json(archived, f"archived {role}")


def _authenticate_no_prior_media_reuse(
    manifest: Mapping[str, Any], rows: list[Mapping[str, Any]], project_root: Path
) -> dict[str, Any]:
    authentication = _bound_evidence_json(manifest, "authentication_report")
    receipt = _bound_evidence_json(manifest, "selection_receipt")
    common_paths = {
        Path(str(row["media_path"])).resolve() for row in authentication.get("rows", ())
    } | {
        Path(str(row["path"])).resolve()
        for row in (receipt.get("reviewed_media") or {}).get("records", ())
    }
    if len(common_paths) < 24:
        raise ValueError("Common-seed prior-media path evidence is incomplete.")
    missing_common = sorted(str(path) for path in common_paths if not path.is_file())
    if missing_common:
        raise FileNotFoundError(
            "Cannot prove calibration-v2 media non-reuse because prior common-seed media "
            f"is missing: {missing_common[:3]}"
        )
    v1_root = (project_root / V1_OUTPUT_ROOT_RELATIVE).resolve()
    v1_paths = (
        {path.resolve() for path in v1_root.rglob("*.png") if path.is_file()}
        if v1_root.is_dir()
        else set()
    )
    if len(v1_paths) < EXPECTED_ROWS:
        raise FileNotFoundError(
            "Cannot prove calibration-v2 media non-reuse without the complete prior "
            f"v1 PNG cohort (found {len(v1_paths)}, require at least {EXPECTED_ROWS})."
        )
    prior_paths = common_paths | v1_paths
    prior_inodes = {(path.stat().st_dev, path.stat().st_ino) for path in prior_paths}
    prior_hashes = {sha256_file(path) for path in prior_paths}
    v2_paths = [Path(str(row["image_path"])).resolve() for row in rows]
    if (
        len(v2_paths) != EXPECTED_ROWS
        or len(set(v2_paths)) != EXPECTED_ROWS
        or any(not path.is_file() for path in v2_paths)
    ):
        raise ValueError("Calibration-v2 non-reuse requires 45 distinct live PNG paths.")
    if any(path in prior_paths for path in v2_paths):
        raise ValueError("Calibration-v2 media path aliases a prior cohort.")
    inode_aliases = [
        str(path)
        for path in v2_paths
        if (path.stat().st_dev, path.stat().st_ino) in prior_inodes or path.stat().st_nlink != 1
    ]
    if inode_aliases:
        raise ValueError(f"Calibration-v2 media reuses a prior/hardlinked inode: {inode_aliases}")
    return {
        "status": "authenticated_fresh_execution_paths_and_inodes",
        "v2_rows_checked": EXPECTED_ROWS,
        "common_seed_prior_paths_checked": len(common_paths),
        "v1_prior_paths_checked": len(v1_paths),
        "all_v2_paths_distinct_from_prior": True,
        "all_v2_inodes_distinct_from_prior": True,
        "all_v2_link_counts_equal_one": True,
        "byte_identical_prior_rows": sum(sha256_file(path) in prior_hashes for path in v2_paths),
        "byte_identity_interpretation": (
            "reported_not_rejected_because_fresh_deterministic_generation_can_be_byte_identical"
        ),
    }


def collect_calibration_artifacts(
    manifest_path: str | Path,
    registry_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    project_root = Path(root).expanduser().resolve()
    resolved_manifest = Path(manifest_path).expanduser().resolve()
    resolved_registry = Path(registry_path).expanduser().resolve()
    manifest = read_calibration_manifest_for_audit(resolved_manifest, project_root)
    registry = read_submission_registry(resolved_registry)
    manifest_sha256 = str(manifest["manifest_sha256"])
    expected_indices = list(range(EXPECTED_ROWS))
    if (
        registry.get("benchmark") != CALIBRATION_ID
        or Path(str(registry.get("manifest_path", ""))).resolve() != resolved_manifest
        or registry.get("manifest_sha256") != manifest_sha256
        or registry.get("slurm_job_name") != SLURM_JOB_NAME
        or registry.get("num_registered_tasks") != EXPECTED_ROWS
        or [entry.get("job_index") for entry in registry.get("submissions", ())] != expected_indices
    ):
        raise ValueError("Calibration-v2 registry must authenticate exact indices 0..44.")
    entries = {int(row["job_index"]): row for row in registry["submissions"]}
    selected_seed = _exact_selected_common_seed(
        manifest["selected_common_seed"], "Calibration-v2 manifest selected common seed"
    )
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != EXPECTED_ROWS:
        raise ValueError("Calibration-v2 manifest must contain exactly 45 jobs.")
    rows = []
    for index, job in enumerate(jobs):
        if not isinstance(job, Mapping):
            raise ValueError(f"Calibration-v2 manifest row {index} is not an object.")
        bound = {
            **deepcopy(dict(job)),
            "launch_manifest_sha256": manifest_sha256,
            "launch_manifest_job_index": index,
        }
        reopened = finer.reopen_completed_flux1_output_v3(
            bound,
            root=project_root,
            manifest_path=resolved_manifest,
            manifest_sha256=manifest_sha256,
            manifest_job_index=index,
        )
        rows.append(
            _validate_result_row(
                job=job,
                job_index=index,
                manifest_sha256=manifest_sha256,
                registry_entry=entries[index],
                selected_seed=selected_seed,
                reopened=reopened,
            )
        )
    if len({row["image_path"] for row in rows}) != EXPECTED_ROWS:
        raise ValueError("Calibration-v2 rows do not bind 45 fresh, distinct media paths.")
    reuse_authentication = _authenticate_no_prior_media_reuse(manifest, rows, project_root)
    return {
        "manifest": manifest,
        "manifest_path": resolved_manifest,
        "registry": registry,
        "registry_path": resolved_registry,
        "rows": rows,
        "reuse_authentication": reuse_authentication,
    }


def _v1_collected_view(collected: Mapping[str, Any]) -> dict[str, Any]:
    view = deepcopy(dict(collected))
    manifest = view["manifest"]
    manifest["calibration_inputs"] = {
        "calibration_config": manifest["protocol_inputs"]["calibration_v2_config"]
    }
    return view


def build_objective_report(
    collected: Mapping[str, Any], *, validate: bool = True
) -> dict[str, Any]:
    """Apply the byte-for-byte v1 numerical policy, then bind v2 provenance."""

    report = v1.build_objective_report(_v1_collected_view(collected), validate=False)
    manifest = collected["manifest"]
    selected_seed = _exact_selected_common_seed(
        manifest["selected_common_seed"], "Calibration-v2 manifest selected common seed"
    )
    report.update(
        {
            "evaluation": OBJECTIVE_EVALUATION,
            "calibration_id": CALIBRATION_ID,
            "selected_common_seed": selected_seed,
            "semantic_policy_origin": {
                "module": "hierasafe_flow.evaluation.native_negative_calibration",
                "policy": "v1_frozen_without_threshold_or_gate_changes",
            },
            "reuse_authentication": deepcopy(collected["reuse_authentication"]),
        }
    )
    report["source_bindings"]["calibration_config"] = deepcopy(
        manifest["protocol_inputs"]["calibration_v2_config"]
    )
    report["source_bindings"]["calibration_config"]["size_bytes"] = (
        Path(report["source_bindings"]["calibration_config"]["path"]).stat().st_size
    )
    report["source_bindings"]["calibration_v2_protocol"] = deepcopy(
        manifest["protocol_inputs"]["calibration_v2_protocol"]
    )
    report["source_bindings"]["selected_common_seed_evidence"] = deepcopy(
        manifest["selection_evidence"]
    )
    report["source_bindings"]["manifest"].update(
        {
            "selected_common_seed": selected_seed,
            "seed_homogeneous_job_count": EXPECTED_ROWS,
            "every_manifest_job_uses_selected_common_seed": True,
        }
    )
    report["evaluator_provenance"] = {
        "source": _binding(Path(__file__)),
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "pillow_version": PILLOW_VERSION,
        "v1_semantic_evaluator_source": _binding(Path(v1.__file__)),
    }
    for row in report["rows"]:
        row["selected_common_seed"] = selected_seed
        row["environment_status"] = "verified_before_generation"
        row["manifest_job_index_authenticated"] = True
    report["document_sha256"] = document_digest(report)
    if validate:
        validate_objective_report(report, verify_source_files=True)
    return report


def _v1_objective_view(payload: Mapping[str, Any]) -> dict[str, Any]:
    view = deepcopy(dict(payload))
    view["evaluation"] = v1.OBJECTIVE_EVALUATION
    view["calibration_id"] = v1.CALIBRATION_ID
    view.pop("selected_common_seed", None)
    view.pop("semantic_policy_origin", None)
    view.pop("reuse_authentication", None)
    view["source_bindings"] = {
        key: value
        for key, value in view["source_bindings"].items()
        if key not in {"calibration_v2_protocol", "selected_common_seed_evidence"}
    }
    manifest_binding = view["source_bindings"].get("manifest")
    if isinstance(manifest_binding, dict):
        manifest_binding.pop("selected_common_seed", None)
        manifest_binding.pop("seed_homogeneous_job_count", None)
        manifest_binding.pop("every_manifest_job_uses_selected_common_seed", None)
    for row in view["rows"]:
        row.pop("selected_common_seed", None)
        row.pop("environment_status", None)
        row.pop("manifest_job_index_authenticated", None)
    view["document_sha256"] = v1.document_digest(view)
    return view


def _verify_binding_tree(value: Any) -> None:
    if isinstance(value, Mapping):
        if {"path", "sha256", "size_bytes"} <= set(value):
            path = Path(str(value["path"])).resolve()
            if (
                not path.is_file()
                or path.stat().st_size != value["size_bytes"]
                or sha256_file(path) != value["sha256"]
            ):
                raise ValueError(f"Bound calibration-v2 evidence changed: {path}")
        for item in value.values():
            _verify_binding_tree(item)
    elif isinstance(value, list):
        for item in value:
            _verify_binding_tree(item)


def validate_objective_report(
    payload: Mapping[str, Any], *, verify_source_files: bool = True
) -> dict[str, Any]:
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("evaluation") != OBJECTIVE_EVALUATION
        or payload.get("calibration_id") != CALIBRATION_ID
        or payload.get("status") != "complete"
        or payload.get("model_name") != MODEL_NAME
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("document_sha256") != document_digest(payload)
    ):
        raise ValueError("Calibration-v2 objective identity, status, or digest is invalid.")
    selected_seed = _exact_selected_common_seed(
        payload.get("selected_common_seed"), "Calibration-v2 objective selected common seed"
    )
    _validate_manifest_seed_binding(
        (payload.get("source_bindings") or {}).get("manifest"), selected_seed
    )
    rows = payload.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != EXPECTED_ROWS
        or any(
            row.get("selected_common_seed") != selected_seed
            or row.get("environment_status") != "verified_before_generation"
            or row.get("manifest_job_index_authenticated") is not True
            for row in rows
        )
    ):
        raise ValueError(
            "Calibration-v2 objective row seed/preflight/index coverage is incomplete."
        )
    reuse = payload.get("reuse_authentication") or {}
    if reuse.get("status") != "authenticated_fresh_execution_paths_and_inodes" or any(
        reuse.get(key) is not True
        for key in (
            "all_v2_paths_distinct_from_prior",
            "all_v2_inodes_distinct_from_prior",
            "all_v2_link_counts_equal_one",
        )
    ):
        raise ValueError("Calibration-v2 prior-media non-reuse authentication is absent.")
    # This is the executable guarantee that all v1 semantic checks and their
    # exact topology still validate after stripping only v2 provenance fields.
    v1.validate_objective_report(_v1_objective_view(payload), verify_source_files=False)
    if verify_source_files:
        _verify_binding_tree(payload.get("source_bindings"))
        for row in rows:
            _verify_binding_tree(row.get("artifact_bindings"))
        bindings = payload["source_bindings"]
        recollected = collect_calibration_artifacts(
            bindings["manifest"]["path"],
            bindings["submission_registry"]["path"],
            root=Path(__file__).resolve().parents[3],
        )
        rebuilt = build_objective_report(recollected, validate=False)
        immutable = set(rebuilt) - {"created_at_utc", "document_sha256"}
        drift = [key for key in immutable if payload.get(key) != rebuilt.get(key)]
        if drift:
            raise ValueError(f"Calibration-v2 objective differs from source evidence: {drift}")
    return {
        "status": "complete",
        "row_count": EXPECTED_ROWS,
        "selected_common_seed": selected_seed,
        "document_sha256": payload["document_sha256"],
    }


def build_review_documents(
    objective: Mapping[str, Any],
    *,
    objective_path: Path,
    review_root: Path,
    final_review_root: Path,
    parity_review_root: Path,
    final_parity_review_root: Path,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
    dict[str, Any],
]:
    """Create disjoint source and parity packages plus one private join map."""

    _require_access_disjoint_roots(
        final_review_root,
        final_parity_review_root,
        label="Source and parity reviewer roots",
    )

    combined, unblinding, combined_template = v1.build_review_documents(
        objective,
        objective_path=objective_path,
        review_root=review_root,
        final_review_root=final_review_root,
    )
    private_reviewer = deepcopy(combined)
    private_reviewer.update(
        {
            "review_package": PRIVATE_REVIEW_MANIFEST,
            "calibration_id": CALIBRATION_ID,
        }
    )
    private_reviewer["document_sha256"] = document_digest(private_reviewer)

    source_reviewer = deepcopy(combined)
    source_reviewer.pop("path_parity_pairs", None)
    source_reviewer.pop("ladder_candidate_groups", None)
    source_reviewer.pop("contact_sheets", None)
    source_reviewer.update(
        {
            "review_package": REVIEW_PACKAGE,
            "calibration_id": CALIBRATION_ID,
            "blinding_policy": {
                "roles_hidden": True,
                "true_cfg_scales_hidden": True,
                "source_result_paths_hidden": True,
                "positive_target_concepts_absent": True,
                "role_revealing_groups_absent": True,
                "individual_pngs_require_original_resolution_review": True,
            },
        }
    )
    source_reviewer["document_sha256"] = document_digest(source_reviewer)

    if parity_review_root.exists():
        if any(parity_review_root.iterdir()):
            raise FileExistsError(f"Parity review staging root is not empty: {parity_review_root}")
    else:
        parity_review_root.mkdir(parents=True)
    parity_media_root = parity_review_root / "images"
    parity_sheets_root = parity_review_root / "contact_sheets"
    parity_media_root.mkdir()
    parity_sheets_root.mkdir()
    combined_by_blind = {row["blind_id"]: row for row in combined["rows"]}
    parity_ids = {
        blind_id for pair in combined["path_parity_pairs"] for blind_id in pair["blind_ids"]
    }
    for blind_id in sorted(parity_ids):
        shutil.copy2(
            review_root / "images" / f"{blind_id}.png",
            parity_media_root / f"{blind_id}.png",
        )
    parity_sheet_records = [
        deepcopy(record)
        for record in combined["contact_sheets"]
        if record["section"] == "path_parity"
    ]
    for record in parity_sheet_records:
        source = review_root / record["relative_path"]
        destination = parity_review_root / record["relative_path"]
        shutil.copy2(source, destination)
    parity_rows = [
        {
            key: combined_by_blind[blind_id][key]
            for key in (
                "blind_id",
                "prompt_id",
                "image_relative_path",
                "image_sha256",
                "width",
                "height",
            )
        }
        for blind_id in sorted(parity_ids)
    ]
    parity_reviewer: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "review_package": PARITY_REVIEW_PACKAGE,
        "status": "prepared_not_reviewed",
        "created_at_utc": combined["created_at_utc"],
        "calibration_id": CALIBRATION_ID,
        "objective_report_document_sha256": objective["document_sha256"],
        "blinding_policy": {
            "official_vs_native_identity_hidden_within_each_pair": True,
            "true_cfg_scales_absent": True,
            "source_45_row_candidate_package_absent": True,
            "source_result_paths_hidden": True,
            "positive_target_concepts_absent": True,
            "individual_pngs_require_original_resolution_review": True,
        },
        "row_count": 6,
        "source_prompts": deepcopy(combined["source_prompts"]),
        "rows": parity_rows,
        "path_parity_pairs": deepcopy(combined["path_parity_pairs"]),
        "contact_sheets": parity_sheet_records,
    }
    parity_reviewer["document_sha256"] = document_digest(parity_reviewer)

    # No pair/group surface remains in the 45-row source package.  Removing
    # every generated sheet also prevents filenames/titles from acting as a
    # covert role label.
    shutil.rmtree(review_root / "contact_sheets")
    source_reviewer_path = review_root / "reviewer_manifest.json"
    source_reviewer_path.write_text(
        json.dumps(source_reviewer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    parity_reviewer_path = parity_review_root / "reviewer_manifest.json"
    parity_reviewer_path.write_text(
        json.dumps(parity_reviewer, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    unblinding.update(
        {
            "unblinding_map": UNBLINDING_MAP,
            "calibration_id": CALIBRATION_ID,
            "selected_common_seed": objective["selected_common_seed"],
            "reviewer_manifest_document_sha256": private_reviewer["document_sha256"],
            "source_reviewer_manifest_document_sha256": source_reviewer["document_sha256"],
            "parity_reviewer_manifest_document_sha256": parity_reviewer["document_sha256"],
        }
    )
    for row in unblinding["rows"]:
        row["selected_common_seed"] = objective["selected_common_seed"]
    unblinding["document_sha256"] = document_digest(unblinding)
    source_template: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "review": SOURCE_MANUAL_REVIEW,
        "status": "draft_not_reviewed",
        "reviewer": None,
        "reviewed_at_utc": None,
        "source_bindings": {
            "source_reviewer_manifest": {"document_sha256": source_reviewer["document_sha256"]}
        },
        "viewing_attestation": {
            "all_45_blinded_images_opened_at_original_resolution": None,
            "roles_or_scales_revealed_during_review": None,
            "positive_target_concepts_scored": None,
            "parity_package_or_decisions_seen": None,
        },
        "row_reviews": deepcopy(combined_template["row_reviews"]),
        "review_instructions": {
            "row_hard_gate_values": sorted(ROW_VERDICTS),
            "source_concept_values": sorted(SOURCE_VERDICTS),
            "indeterminate_is_fail_closed": True,
            "minimum_suppressed_sources_per_prompt": 2,
            "positive_target_concepts_must_not_be_added_or_scored": True,
        },
    }
    parity_template: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "review": PARITY_MANUAL_REVIEW,
        "status": "draft_not_reviewed",
        "reviewer": None,
        "reviewed_at_utc": None,
        "source_bindings": {
            "parity_reviewer_manifest": {"document_sha256": parity_reviewer["document_sha256"]}
        },
        "viewing_attestation": {
            "all_6_blinded_images_opened_at_original_resolution": None,
            "all_3_pairs_compared_side_by_side": None,
            "official_vs_native_identity_revealed_within_pairs": None,
            "source_45_row_package_or_decisions_seen": None,
            "positive_target_concepts_scored": None,
        },
        "path_parity_reviews": deepcopy(combined_template["path_parity_reviews"]),
        "review_instructions": {
            "path_parity_values": sorted(PARITY_VERDICTS),
            "indeterminate_is_fail_closed": True,
            "positive_target_concepts_must_not_be_added_or_scored": True,
        },
    }
    return (
        source_reviewer,
        parity_reviewer,
        private_reviewer,
        unblinding,
        source_template,
        parity_template,
    )


def _validate_review_and_unblinding(
    reviewer: Mapping[str, Any], unblinding: Mapping[str, Any], selected_seed: int
) -> None:
    if (
        reviewer.get("review_package") != PRIVATE_REVIEW_MANIFEST
        or reviewer.get("calibration_id") != CALIBRATION_ID
        or reviewer.get("document_sha256") != document_digest(reviewer)
    ):
        raise ValueError("Calibration-v2 reviewer manifest identity/digest is invalid.")
    if (
        unblinding.get("unblinding_map") != UNBLINDING_MAP
        or unblinding.get("calibration_id") != CALIBRATION_ID
        or unblinding.get("selected_common_seed") != selected_seed
        or unblinding.get("reviewer_manifest_document_sha256") != reviewer.get("document_sha256")
        or unblinding.get("document_sha256") != document_digest(unblinding)
        or any(
            row.get("selected_common_seed") != selected_seed for row in unblinding.get("rows", ())
        )
    ):
        raise ValueError("Calibration-v2 unblinding identity, digest, or seed binding is invalid.")
    reviewer_v1 = deepcopy(dict(reviewer))
    reviewer_v1["review_package"] = v1.REVIEW_PACKAGE
    reviewer_v1["calibration_id"] = v1.CALIBRATION_ID
    reviewer_v1["document_sha256"] = v1.document_digest(reviewer_v1)
    unblinding_v1 = deepcopy(dict(unblinding))
    unblinding_v1["unblinding_map"] = v1.UNBLINDING_MAP
    unblinding_v1["calibration_id"] = v1.CALIBRATION_ID
    unblinding_v1.pop("selected_common_seed", None)
    for row in unblinding_v1["rows"]:
        row.pop("selected_common_seed", None)
    unblinding_v1["reviewer_manifest_document_sha256"] = reviewer_v1["document_sha256"]
    unblinding_v1["document_sha256"] = v1.document_digest(unblinding_v1)
    v1._validate_review_and_unblinding(reviewer_v1, unblinding_v1)


def _validate_review_evidence_topology(
    *,
    objective: Mapping[str, Any],
    private_reviewer: Mapping[str, Any],
    private_reviewer_path: Path,
    source_reviewer: Mapping[str, Any],
    source_reviewer_path: Path,
    parity_reviewer: Mapping[str, Any],
    parity_reviewer_path: Path,
    unblinding: Mapping[str, Any],
) -> None:
    """Authenticate two access-disjoint public reviews and their private join map."""

    selected_seed = _exact_selected_common_seed(
        objective["selected_common_seed"], "Calibration-v2 objective selected common seed"
    )
    _validate_review_and_unblinding(private_reviewer, unblinding, selected_seed)
    expected_source_keys = {
        "schema_version",
        "review_package",
        "status",
        "created_at_utc",
        "calibration_id",
        "objective_report_document_sha256",
        "blinding_policy",
        "row_count",
        "source_prompts",
        "source_concepts",
        "rows",
        "document_sha256",
    }
    expected_parity_keys = {
        "schema_version",
        "review_package",
        "status",
        "created_at_utc",
        "calibration_id",
        "objective_report_document_sha256",
        "blinding_policy",
        "row_count",
        "source_prompts",
        "rows",
        "path_parity_pairs",
        "contact_sheets",
        "document_sha256",
    }
    expected_private_keys = expected_parity_keys | {
        "source_concepts",
        "ladder_candidate_groups",
    }
    expected_unblinding_keys = {
        "schema_version",
        "unblinding_map",
        "status",
        "created_at_utc",
        "calibration_id",
        "objective_report_document_sha256",
        "reviewer_manifest_document_sha256",
        "source_reviewer_manifest_document_sha256",
        "parity_reviewer_manifest_document_sha256",
        "selected_common_seed",
        "rows",
        "document_sha256",
    }
    if (
        set(source_reviewer) != expected_source_keys
        or set(parity_reviewer) != expected_parity_keys
        or set(private_reviewer) != expected_private_keys
        or set(unblinding) != expected_unblinding_keys
    ):
        raise ValueError("Blinded reviewer schema contains missing or role-revealing fields.")
    expected_source_policy = {
        "roles_hidden": True,
        "true_cfg_scales_hidden": True,
        "source_result_paths_hidden": True,
        "positive_target_concepts_absent": True,
        "role_revealing_groups_absent": True,
        "individual_pngs_require_original_resolution_review": True,
    }
    expected_parity_policy = {
        "official_vs_native_identity_hidden_within_each_pair": True,
        "true_cfg_scales_absent": True,
        "source_45_row_candidate_package_absent": True,
        "source_result_paths_hidden": True,
        "positive_target_concepts_absent": True,
        "individual_pngs_require_original_resolution_review": True,
    }
    expected_private_policy = {
        "roles_hidden": True,
        "true_cfg_scales_hidden": True,
        "source_result_paths_hidden": True,
        "positive_target_concepts_absent": True,
        "contact_sheets_for_navigation_only": True,
        "individual_pngs_require_original_resolution_review": True,
    }
    if (
        source_reviewer.get("schema_version") != SCHEMA_VERSION
        or parity_reviewer.get("schema_version") != SCHEMA_VERSION
        or private_reviewer.get("schema_version") != SCHEMA_VERSION
        or source_reviewer.get("review_package") != REVIEW_PACKAGE
        or parity_reviewer.get("review_package") != PARITY_REVIEW_PACKAGE
        or private_reviewer.get("review_package") != PRIVATE_REVIEW_MANIFEST
        or source_reviewer.get("calibration_id") != CALIBRATION_ID
        or parity_reviewer.get("calibration_id") != CALIBRATION_ID
        or private_reviewer.get("calibration_id") != CALIBRATION_ID
        or source_reviewer.get("document_sha256") != document_digest(source_reviewer)
        or parity_reviewer.get("document_sha256") != document_digest(parity_reviewer)
        or private_reviewer.get("document_sha256") != document_digest(private_reviewer)
        or source_reviewer.get("status") != "prepared_not_reviewed"
        or source_reviewer.get("row_count") != EXPECTED_ROWS
        or source_reviewer.get("blinding_policy") != expected_source_policy
        or parity_reviewer.get("status") != "prepared_not_reviewed"
        or parity_reviewer.get("blinding_policy") != expected_parity_policy
        or private_reviewer.get("status") != "prepared_not_reviewed"
        or private_reviewer.get("row_count") != EXPECTED_ROWS
        or private_reviewer.get("blinding_policy") != expected_private_policy
        or source_reviewer.get("created_at_utc") != private_reviewer.get("created_at_utc")
        or parity_reviewer.get("created_at_utc") != private_reviewer.get("created_at_utc")
        or private_reviewer.get("objective_report_document_sha256")
        != objective.get("document_sha256")
        or source_reviewer.get("objective_report_document_sha256")
        != objective.get("document_sha256")
        or parity_reviewer.get("objective_report_document_sha256")
        != objective.get("document_sha256")
        or unblinding.get("objective_report_document_sha256") != objective.get("document_sha256")
        or unblinding.get("source_reviewer_manifest_document_sha256")
        != source_reviewer.get("document_sha256")
        or unblinding.get("parity_reviewer_manifest_document_sha256")
        != parity_reviewer.get("document_sha256")
    ):
        raise ValueError("Review evidence binds another calibration-v2 objective report.")
    objective_rows = objective.get("rows")
    if not isinstance(objective_rows, list) or len(objective_rows) != EXPECTED_ROWS:
        raise ValueError("Calibration-v2 objective review topology must contain 45 rows.")
    by_blind = {str(row.get("blind_id")): row for row in objective_rows}
    if len(by_blind) != EXPECTED_ROWS:
        raise ValueError("Calibration-v2 objective blind IDs are not unique.")

    manifest_sha256 = str(
        (objective.get("source_bindings") or {}).get("manifest", {}).get("manifest_sha256", "")
    )
    expected_review_rows = sorted(
        (v1._review_row(row) for row in objective_rows),
        key=lambda item: v1._blind_sort_key(manifest_sha256, item["blind_id"], "all_rows"),
    )
    if (
        source_reviewer.get("rows") != expected_review_rows
        or private_reviewer.get("rows") != expected_review_rows
    ):
        raise ValueError("Blinded source-review rows differ from the exact objective rows.")
    expected_prompts = {
        prompt_id: next(
            str(row["prompt"]) for row in objective_rows if row["prompt_id"] == prompt_id
        )
        for prompt_id in PROMPT_IDS
    }
    if (
        source_reviewer.get("source_prompts") != expected_prompts
        or parity_reviewer.get("source_prompts") != expected_prompts
        or private_reviewer.get("source_prompts") != expected_prompts
        or source_reviewer.get("source_concepts")
        != {prompt_id: list(SOURCE_CONCEPTS[prompt_id]) for prompt_id in PROMPT_IDS}
        or private_reviewer.get("source_concepts")
        != {prompt_id: list(SOURCE_CONCEPTS[prompt_id]) for prompt_id in PROMPT_IDS}
    ):
        raise ValueError("Blinded reviewer source-only prompt/concept policy drifted.")

    source_root, parity_root, private_root = _require_access_disjoint_roots(
        source_reviewer_path.resolve().parent,
        parity_reviewer_path.resolve().parent,
        private_reviewer_path.resolve().parent,
        label="Source, parity, and private review roots",
    )
    source_media_root = (source_root / "images").resolve()
    for row in expected_review_rows:
        expected_relative = f"images/{row['blind_id']}.png"
        path = (source_root / str(row["image_relative_path"])).resolve()
        if (
            row["image_relative_path"] != expected_relative
            or source_media_root not in path.parents
            or not path.is_file()
            or sha256_file(path) != row["image_sha256"]
        ):
            raise ValueError(f"Blinded review PNG changed for {row['blind_id']}.")
        source_media = Path(str(by_blind[row["blind_id"]]["image_path"])).resolve()
        copied_stat = path.stat()
        source_stat = source_media.stat()
        if copied_stat.st_nlink != 1 or (copied_stat.st_dev, copied_stat.st_ino) == (
            source_stat.st_dev,
            source_stat.st_ino,
        ):
            raise ValueError(f"Blinded review PNG is not an independent copy: {row['blind_id']}.")
        decoded_rgb(path, width=int(row["width"]), height=int(row["height"]))

    expected_pairs = []
    expected_groups = []
    expected_sheets = []
    expected_private_sheet_specs = []
    parity_rows_by_id: dict[str, dict[str, Any]] = {}
    parity_objective_rows: dict[str, Mapping[str, Any]] = {}
    for prompt_id in PROMPT_IDS:
        parity = [
            row
            for row in objective_rows
            if row["prompt_id"] == prompt_id and row["role"] in {ROLE_OFFICIAL, ROLE_CONTROL}
        ]
        parity.sort(
            key=lambda item: v1._blind_sort_key(
                manifest_sha256, item["blind_id"], f"parity|{prompt_id}"
            )
        )
        pair_id = (
            "parity_"
            + hashlib.sha256(f"{manifest_sha256}|{prompt_id}|path_parity".encode()).hexdigest()[:16]
        )
        expected_pairs.append(
            {
                "pair_id": pair_id,
                "prompt_id": prompt_id,
                "blind_ids": [row["blind_id"] for row in parity],
                "category_ids": list(PATH_PARITY_CATEGORIES),
            }
        )
        expected_sheets.append(("path_parity", prompt_id, f"contact_sheets/{pair_id}.png"))
        expected_private_sheet_specs.append(
            (
                "path_parity",
                prompt_id,
                f"contact_sheets/{pair_id}.png",
                parity,
                f"{prompt_id}: blinded path-parity controls (A/B order randomized)",
                2,
            )
        )
        for row in parity:
            parity_objective_rows[row["blind_id"]] = row
            review_row = next(
                item for item in expected_review_rows if item["blind_id"] == row["blind_id"]
            )
            parity_rows_by_id[row["blind_id"]] = {
                key: review_row[key]
                for key in (
                    "blind_id",
                    "prompt_id",
                    "image_relative_path",
                    "image_sha256",
                    "width",
                    "height",
                )
            }
        ladder = [
            row
            for row in objective_rows
            if row["prompt_id"] == prompt_id and row["role"] == ROLE_LADDER
        ]
        ladder.sort(
            key=lambda item: v1._blind_sort_key(
                manifest_sha256, item["blind_id"], f"ladder|{prompt_id}"
            )
        )
        group_id = (
            "candidate_"
            + hashlib.sha256(f"{manifest_sha256}|{prompt_id}|ladder".encode()).hexdigest()[:16]
        )
        expected_groups.append(
            {
                "group_id": group_id,
                "prompt_id": prompt_id,
                "blind_ids": [row["blind_id"] for row in ladder],
                "source_concept_ids": list(SOURCE_CONCEPTS[prompt_id]),
                "positive_target_concepts_scored": False,
            }
        )
        expected_private_sheet_specs.append(
            (
                "native_negative_ladder",
                prompt_id,
                f"contact_sheets/{group_id}.png",
                ladder,
                f"{prompt_id}: blinded native-negative ladder (order randomized)",
                4,
            )
        )

    expected_parity_rows = [parity_rows_by_id[key] for key in sorted(parity_rows_by_id)]
    if parity_reviewer.get("rows") != expected_parity_rows or parity_reviewer.get("row_count") != 6:
        raise ValueError("Parity reviewer rows differ from the exact six control rows.")
    parity_media_root = (parity_root / "images").resolve()
    for row in expected_parity_rows:
        path = (parity_root / row["image_relative_path"]).resolve()
        if (
            parity_media_root not in path.parents
            or not path.is_file()
            or sha256_file(path) != row["image_sha256"]
        ):
            raise ValueError(f"Blinded parity PNG changed for {row['blind_id']}.")
        source_copy = (source_media_root / f"{row['blind_id']}.png").stat()
        parity_copy = path.stat()
        if parity_copy.st_nlink != 1 or (parity_copy.st_dev, parity_copy.st_ino) == (
            source_copy.st_dev,
            source_copy.st_ino,
        ):
            raise ValueError(f"Blinded parity PNG is not an independent copy: {row['blind_id']}.")
        decoded_rgb(path, width=int(row["width"]), height=int(row["height"]))
    if (
        parity_reviewer.get("path_parity_pairs") != expected_pairs
        or private_reviewer.get("path_parity_pairs") != expected_pairs
        or private_reviewer.get("ladder_candidate_groups") != expected_groups
    ):
        raise ValueError("Blinded path-parity pair topology drifted.")
    sheets = parity_reviewer.get("contact_sheets")
    if not isinstance(sheets, list) or len(sheets) != len(expected_sheets):
        raise ValueError("Blinded contact-sheet coverage drifted.")
    for record, (section, prompt_id, relative) in zip(sheets, expected_sheets, strict=True):
        path = (parity_root / str(record.get("relative_path", ""))).resolve()
        sheet_root = (parity_root / "contact_sheets").resolve()
        pair = expected_pairs[PROMPT_IDS.index(prompt_id)]
        with tempfile.TemporaryDirectory(prefix="flux-v2-parity-auth-") as temporary:
            regenerated = Path(temporary) / "sheet.png"
            v1._contact_sheet(
                [parity_objective_rows[blind_id] for blind_id in pair["blind_ids"]],
                regenerated,
                title=(f"{prompt_id}: blinded path-parity controls (A/B order randomized)"),
                columns=2,
            )
            expected_sheet_sha256 = sha256_file(regenerated)
        if (
            record
            != {
                "section": section,
                "prompt_id": prompt_id,
                "relative_path": relative,
                "sha256": expected_sheet_sha256,
            }
            or sheet_root not in path.parents
            or not path.is_file()
            or sha256_file(path) != expected_sheet_sha256
        ):
            raise ValueError(f"Blinded contact sheet changed for {prompt_id}/{section}.")

    expected_private_sheets = []
    for section, prompt_id, relative, rows, title, columns in expected_private_sheet_specs:
        with tempfile.TemporaryDirectory(prefix="flux-v2-private-sheet-auth-") as temporary:
            regenerated = Path(temporary) / "sheet.png"
            v1._contact_sheet(rows, regenerated, title=title, columns=columns)
            expected_private_sheets.append(
                {
                    "section": section,
                    "prompt_id": prompt_id,
                    "relative_path": relative,
                    "sha256": sha256_file(regenerated),
                }
            )
    if private_reviewer.get("contact_sheets") != expected_private_sheets:
        raise ValueError("Private review join-map contact-sheet topology drifted.")

    expected_unblinding_rows = [
        {
            "blind_id": row["blind_id"],
            "job_index": row["job_index"],
            "condition_id": row["condition_id"],
            "prompt_id": row["prompt_id"],
            "role": row["role"],
            "true_cfg_scale": row["true_cfg_scale"],
            "eligible_for_scale_selection": row["eligible_for_scale_selection"],
            "source_image_path": row["image_path"],
            "source_image_sha256": row["image_sha256"],
            "review_image_path": str(source_media_root / f"{row['blind_id']}.png"),
            "selected_common_seed": selected_seed,
        }
        for row in sorted(objective_rows, key=lambda item: item["job_index"])
    ]
    if unblinding.get("rows") != expected_unblinding_rows:
        raise ValueError("Private unblinding rows differ from the exact objective cohort.")


def _validate_normalized_manual_ledger(
    ledger: Mapping[str, Any],
    *,
    objective: Mapping[str, Any],
    objective_path: Path,
    private_reviewer: Mapping[str, Any],
    private_reviewer_path: Path,
    source_reviewer: Mapping[str, Any],
    parity_reviewer: Mapping[str, Any],
    unblinding: Mapping[str, Any],
) -> None:
    """Re-normalize all manual decisions and demand exact canonical equality."""

    selected_seed = _exact_selected_common_seed(
        objective["selected_common_seed"], "Calibration-v2 objective selected common seed"
    )
    expected_receipt = objective["source_bindings"]["selected_common_seed_evidence"][
        "selection_receipt"
    ]["document_sha256"]
    if (
        ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("review") != MANUAL_REVIEW
        or ledger.get("status") != "complete"
        or ledger.get("selected_common_seed") != selected_seed
        or ledger.get("common_seed_selection_document_sha256") != expected_receipt
        or ledger.get("source_reviewer_manifest_document_sha256")
        != source_reviewer.get("document_sha256")
        or ledger.get("parity_reviewer_manifest_document_sha256")
        != parity_reviewer.get("document_sha256")
        or not str(ledger.get("source_reviewer_identity", "")).strip()
        or not str(ledger.get("parity_reviewer_identity", "")).strip()
        or ledger.get("source_reviewer_identity") == ledger.get("parity_reviewer_identity")
        or not str(ledger.get("source_reviewed_at_utc", "")).strip()
        or not str(ledger.get("parity_reviewed_at_utc", "")).strip()
        or ledger.get("document_sha256") != document_digest(ledger)
    ):
        raise ValueError("Normalized calibration-v2 manual ledger identity/digest is invalid.")
    source_time = _parse_timing_timestamp(
        ledger["source_reviewed_at_utc"], "source reviewed_at_utc"
    )
    parity_time = _parse_timing_timestamp(
        ledger["parity_reviewed_at_utc"], "parity reviewed_at_utc"
    )
    _parse_timing_timestamp(ledger.get("sealed_at_utc"), "manual ledger sealed_at_utc")
    if (
        ledger.get("reviewer")
        != (
            f"source:{ledger['source_reviewer_identity']}; "
            f"parity:{ledger['parity_reviewer_identity']}"
        )
        or ledger.get("reviewed_at_utc") != max(source_time, parity_time).isoformat()
    ):
        raise ValueError("Normalized calibration-v2 reviewer identity join is inconsistent.")
    translated = deepcopy(dict(ledger))
    translated["review"] = v1.MANUAL_REVIEW
    expected = v1.normalize_manual_ledger(
        translated,
        objective=objective,
        objective_path=objective_path,
        reviewer=private_reviewer,
        reviewer_path=private_reviewer_path,
        unblinding=unblinding,
    )
    expected["sealed_at_utc"] = ledger.get("sealed_at_utc")
    expected.update(
        {
            "review": MANUAL_REVIEW,
            "selected_common_seed": selected_seed,
            "common_seed_selection_document_sha256": expected_receipt,
            "source_reviewer_manifest_document_sha256": source_reviewer["document_sha256"],
            "parity_reviewer_manifest_document_sha256": parity_reviewer["document_sha256"],
            "source_reviewer_identity": ledger["source_reviewer_identity"],
            "parity_reviewer_identity": ledger["parity_reviewer_identity"],
            "source_reviewed_at_utc": ledger["source_reviewed_at_utc"],
            "parity_reviewed_at_utc": ledger["parity_reviewed_at_utc"],
        }
    )
    expected["document_sha256"] = document_digest(expected)
    if dict(ledger) != expected:
        raise ValueError("Calibration-v2 manual ledger differs after exact re-normalization.")


def normalize_manual_ledger(
    source_draft: Mapping[str, Any],
    parity_draft: Mapping[str, Any],
    *,
    objective: Mapping[str, Any],
    objective_path: Path,
    private_reviewer: Mapping[str, Any],
    private_reviewer_path: Path,
    source_reviewer: Mapping[str, Any],
    source_reviewer_path: Path,
    parity_reviewer: Mapping[str, Any],
    parity_reviewer_path: Path,
    unblinding: Mapping[str, Any],
) -> dict[str, Any]:
    selected_seed = _exact_selected_common_seed(
        objective["selected_common_seed"], "Calibration-v2 objective selected common seed"
    )
    reuse = objective.get("reuse_authentication") or {}
    if any(
        reuse.get(key) is not True
        for key in (
            "all_v2_paths_distinct_from_prior",
            "all_v2_inodes_distinct_from_prior",
            "all_v2_link_counts_equal_one",
        )
    ):
        raise ValueError("Selection cannot claim non-reuse without objective authentication.")
    _validate_review_evidence_topology(
        objective=objective,
        private_reviewer=private_reviewer,
        private_reviewer_path=private_reviewer_path,
        source_reviewer=source_reviewer,
        source_reviewer_path=source_reviewer_path,
        parity_reviewer=parity_reviewer,
        parity_reviewer_path=parity_reviewer_path,
        unblinding=unblinding,
    )
    expected_source_bindings = {
        "source_reviewer_manifest": {"document_sha256": source_reviewer["document_sha256"]}
    }
    expected_parity_bindings = {
        "parity_reviewer_manifest": {"document_sha256": parity_reviewer["document_sha256"]}
    }
    expected_source_attestation = {
        "all_45_blinded_images_opened_at_original_resolution": True,
        "roles_or_scales_revealed_during_review": False,
        "positive_target_concepts_scored": False,
        "parity_package_or_decisions_seen": False,
    }
    expected_parity_attestation = {
        "all_6_blinded_images_opened_at_original_resolution": True,
        "all_3_pairs_compared_side_by_side": True,
        "official_vs_native_identity_revealed_within_pairs": False,
        "source_45_row_package_or_decisions_seen": False,
        "positive_target_concepts_scored": False,
    }
    expected_source_instructions = {
        "row_hard_gate_values": sorted(ROW_VERDICTS),
        "source_concept_values": sorted(SOURCE_VERDICTS),
        "indeterminate_is_fail_closed": True,
        "minimum_suppressed_sources_per_prompt": 2,
        "positive_target_concepts_must_not_be_added_or_scored": True,
    }
    expected_parity_instructions = {
        "path_parity_values": sorted(PARITY_VERDICTS),
        "indeterminate_is_fail_closed": True,
        "positive_target_concepts_must_not_be_added_or_scored": True,
    }
    source_identity = str(source_draft.get("reviewer") or "").strip()
    parity_identity = str(parity_draft.get("reviewer") or "").strip()
    source_reviewed_at = str(source_draft.get("reviewed_at_utc") or "")
    parity_reviewed_at = str(parity_draft.get("reviewed_at_utc") or "")
    if (
        set(source_draft)
        != {
            "schema_version",
            "review",
            "status",
            "reviewer",
            "reviewed_at_utc",
            "source_bindings",
            "viewing_attestation",
            "row_reviews",
            "review_instructions",
        }
        or source_draft.get("schema_version") != SCHEMA_VERSION
        or source_draft.get("review") != SOURCE_MANUAL_REVIEW
        or source_draft.get("status") != "complete"
        or source_draft.get("source_bindings") != expected_source_bindings
        or source_draft.get("viewing_attestation") != expected_source_attestation
        or source_draft.get("review_instructions") != expected_source_instructions
        or set(parity_draft)
        != {
            "schema_version",
            "review",
            "status",
            "reviewer",
            "reviewed_at_utc",
            "source_bindings",
            "viewing_attestation",
            "path_parity_reviews",
            "review_instructions",
        }
        or parity_draft.get("schema_version") != SCHEMA_VERSION
        or parity_draft.get("review") != PARITY_MANUAL_REVIEW
        or parity_draft.get("status") != "complete"
        or parity_draft.get("source_bindings") != expected_parity_bindings
        or parity_draft.get("viewing_attestation") != expected_parity_attestation
        or parity_draft.get("review_instructions") != expected_parity_instructions
        or not source_identity
        or not parity_identity
        or source_identity == parity_identity
    ):
        raise ValueError("Independent source/parity manual review contracts are incomplete.")
    source_time = _parse_timing_timestamp(source_reviewed_at, "source reviewed_at_utc")
    parity_time = _parse_timing_timestamp(parity_reviewed_at, "parity reviewed_at_utc")
    translated = {
        "schema_version": SCHEMA_VERSION,
        "review": v1.MANUAL_REVIEW,
        "status": "complete",
        "reviewer": f"source:{source_identity}; parity:{parity_identity}",
        "reviewed_at_utc": max(source_time, parity_time).isoformat(),
        "source_bindings": {
            "objective_report": {
                **_binding(objective_path),
                "document_sha256": objective["document_sha256"],
            },
            "reviewer_manifest": {
                **_binding(private_reviewer_path),
                "document_sha256": private_reviewer["document_sha256"],
            },
        },
        "viewing_attestation": {
            "all_45_blinded_images_opened_at_original_resolution": True,
            "path_parity_pairs_compared_side_by_side": True,
            "contact_sheets_used_for_navigation_only": True,
            "roles_or_scales_revealed_during_review": False,
            "positive_target_concepts_scored": False,
        },
        "row_reviews": deepcopy(source_draft["row_reviews"]),
        "path_parity_reviews": deepcopy(parity_draft["path_parity_reviews"]),
    }
    ledger = v1.normalize_manual_ledger(
        translated,
        objective=objective,
        objective_path=objective_path,
        reviewer=private_reviewer,
        reviewer_path=private_reviewer_path,
        unblinding=unblinding,
    )
    ledger.update(
        {
            "review": MANUAL_REVIEW,
            "selected_common_seed": selected_seed,
            "common_seed_selection_document_sha256": objective["source_bindings"][
                "selected_common_seed_evidence"
            ]["selection_receipt"]["document_sha256"],
            "source_reviewer_manifest_document_sha256": source_reviewer["document_sha256"],
            "parity_reviewer_manifest_document_sha256": parity_reviewer["document_sha256"],
            "source_reviewer_identity": source_identity,
            "parity_reviewer_identity": parity_identity,
            "source_reviewed_at_utc": source_reviewed_at,
            "parity_reviewed_at_utc": parity_reviewed_at,
        }
    )
    ledger["document_sha256"] = document_digest(ledger)
    return ledger


def build_selection_report(
    *,
    objective: Mapping[str, Any],
    objective_path: Path,
    private_reviewer: Mapping[str, Any],
    private_reviewer_path: Path,
    source_reviewer: Mapping[str, Any],
    source_reviewer_path: Path,
    parity_reviewer: Mapping[str, Any],
    parity_reviewer_path: Path,
    unblinding: Mapping[str, Any],
    unblinding_path: Path,
    ledger: Mapping[str, Any],
    ledger_path: Path,
) -> dict[str, Any]:
    selected_seed = _exact_selected_common_seed(
        objective["selected_common_seed"], "Calibration-v2 objective selected common seed"
    )
    reuse = objective.get("reuse_authentication") or {}
    if any(
        reuse.get(key) is not True
        for key in (
            "all_v2_paths_distinct_from_prior",
            "all_v2_inodes_distinct_from_prior",
            "all_v2_link_counts_equal_one",
        )
    ):
        raise ValueError("Selection cannot claim non-reuse without objective authentication.")
    _validate_review_evidence_topology(
        objective=objective,
        private_reviewer=private_reviewer,
        private_reviewer_path=private_reviewer_path,
        source_reviewer=source_reviewer,
        source_reviewer_path=source_reviewer_path,
        parity_reviewer=parity_reviewer,
        parity_reviewer_path=parity_reviewer_path,
        unblinding=unblinding,
    )
    if ledger.get("selected_common_seed") != selected_seed:
        raise ValueError("Manual ledger selected seed differs from the objective cohort.")
    _validate_normalized_manual_ledger(
        ledger,
        objective=objective,
        objective_path=objective_path,
        private_reviewer=private_reviewer,
        private_reviewer_path=private_reviewer_path,
        source_reviewer=source_reviewer,
        parity_reviewer=parity_reviewer,
        unblinding=unblinding,
    )
    if any(row.get("selected_common_seed") != selected_seed for row in objective.get("rows", ())):
        raise ValueError("Objective rows are not homogeneous on the selected common seed.")
    report = v1.build_selection_report(
        objective=objective,
        objective_path=objective_path,
        reviewer=private_reviewer,
        reviewer_path=private_reviewer_path,
        unblinding=unblinding,
        unblinding_path=unblinding_path,
        ledger=ledger,
        ledger_path=ledger_path,
    )
    report.update(
        {
            "selection_report": SELECTION_REPORT,
            "calibration_id": CALIBRATION_ID,
            "selected_common_seed": selected_seed,
            "every_evaluated_row_used_selected_common_seed": True,
            "semantic_selection_rule_identical_to_v1": True,
        }
    )
    report["source_bindings"]["calibration_config"].pop("calibration_config_sha256", None)
    report["source_bindings"]["calibration_config"]["calibration_config_sha256"] = objective[
        "source_bindings"
    ]["calibration_config"]["sha256"]
    report["source_bindings"]["calibration_v2_protocol"] = objective["source_bindings"][
        "calibration_v2_protocol"
    ]
    report["source_bindings"]["selected_common_seed_evidence"] = objective["source_bindings"][
        "selected_common_seed_evidence"
    ]
    report["source_bindings"]["source_reviewer_manifest"] = {
        **_binding(source_reviewer_path),
        "document_sha256": source_reviewer["document_sha256"],
    }
    report["source_bindings"]["parity_reviewer_manifest"] = {
        **_binding(parity_reviewer_path),
        "document_sha256": parity_reviewer["document_sha256"],
    }
    report["qualified_setting_binding"] = {
        "selected_common_seed": selected_seed,
        "selected_true_cfg_scale": report["selected_true_cfg_scale"],
        "common_seed_selection_document_sha256": objective["source_bindings"][
            "selected_common_seed_evidence"
        ]["selection_receipt"]["document_sha256"],
        "all_45_rows_seed_homogeneous": True,
        "v1_media_reused": False,
        "common_seed_candidate_media_reused": False,
        "reuse_authentication_status": reuse.get("status"),
    }
    report["document_sha256"] = document_digest(report)
    validate_selection_report(report, verify_source_files=False)
    return report


def validate_selection_report_against_evidence(
    payload: Mapping[str, Any],
    *,
    objective: Mapping[str, Any],
    objective_path: Path,
    private_reviewer: Mapping[str, Any],
    private_reviewer_path: Path,
    source_reviewer: Mapping[str, Any],
    source_reviewer_path: Path,
    parity_reviewer: Mapping[str, Any],
    parity_reviewer_path: Path,
    unblinding: Mapping[str, Any],
    unblinding_path: Path,
    ledger: Mapping[str, Any],
    ledger_path: Path,
) -> None:
    """Recompute the entire scale report from its bound substantive evidence."""

    validate_selection_report(payload, verify_source_files=False)
    if payload.get("selected_common_seed") != objective.get("selected_common_seed"):
        raise ValueError("Selection report and objective bind different common seeds.")
    _validate_review_evidence_topology(
        objective=objective,
        private_reviewer=private_reviewer,
        private_reviewer_path=private_reviewer_path,
        source_reviewer=source_reviewer,
        source_reviewer_path=source_reviewer_path,
        parity_reviewer=parity_reviewer,
        parity_reviewer_path=parity_reviewer_path,
        unblinding=unblinding,
    )
    _validate_normalized_manual_ledger(
        ledger,
        objective=objective,
        objective_path=objective_path,
        private_reviewer=private_reviewer,
        private_reviewer_path=private_reviewer_path,
        source_reviewer=source_reviewer,
        parity_reviewer=parity_reviewer,
        unblinding=unblinding,
    )
    rebuilt = build_selection_report(
        objective=objective,
        objective_path=objective_path,
        private_reviewer=private_reviewer,
        private_reviewer_path=private_reviewer_path,
        source_reviewer=source_reviewer,
        source_reviewer_path=source_reviewer_path,
        parity_reviewer=parity_reviewer,
        parity_reviewer_path=parity_reviewer_path,
        unblinding=unblinding,
        unblinding_path=unblinding_path,
        ledger=ledger,
        ledger_path=ledger_path,
    )
    rebuilt["created_at_utc"] = payload.get("created_at_utc")
    rebuilt["document_sha256"] = document_digest(rebuilt)
    if dict(payload) != rebuilt:
        differing = sorted(
            key for key in set(payload) | set(rebuilt) if payload.get(key) != rebuilt.get(key)
        )
        raise ValueError(
            f"Calibration-v2 selection report differs from recomputed evidence: {differing}"
        )


def validate_selection_report(
    payload: Mapping[str, Any], *, verify_source_files: bool = False
) -> None:
    """Recompute the global minimum-scale decision from a v2 report."""

    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("selection_report") != SELECTION_REPORT
        or payload.get("calibration_id") != CALIBRATION_ID
        or payload.get("model_name") != MODEL_NAME
        or payload.get("model_revision") != MODEL_REVISION
        or payload.get("status") not in {"selected", "no_selection"}
        or payload.get("document_sha256") != document_digest(payload)
    ):
        raise ValueError("Calibration-v2 selection report identity/status/digest is invalid.")
    try:
        selected_seed = _exact_selected_common_seed(
            payload.get("selected_common_seed"),
            "Calibration-v2 selection selected common seed",
        )
    except ValueError as exc:
        raise ValueError(
            "Calibration-v2 selection report selected-seed binding is invalid."
        ) from exc
    if (
        payload.get("every_evaluated_row_used_selected_common_seed") is not True
        or payload.get("semantic_selection_rule_identical_to_v1") is not True
    ):
        raise ValueError("Calibration-v2 selection report selected-seed binding is invalid.")
    if payload.get("selection_rule") != {
        "one_global_scale_across_all_three_prompts": True,
        "smallest_globally_passing_scale": True,
        "per_prompt_tuning": False,
        "averaging_across_prompts": False,
        "threshold_relaxation_after_viewing": False,
        "least_bad_selection": False,
    }:
        raise ValueError("Calibration-v2 global scale-selection rule drifted.")
    scale_results = payload.get("scale_results")
    if not isinstance(scale_results, list) or [
        row.get("true_cfg_scale") for row in scale_results
    ] != list(ELIGIBLE_SCALES):
        raise ValueError("Calibration-v2 scale-result coverage/order drifted.")
    derived_passing: list[float] = []
    for expected_scale, row in zip(ELIGIBLE_SCALES, scale_results, strict=True):
        prompts = row.get("prompts")
        if not isinstance(prompts, Mapping) or set(prompts) != set(PROMPT_IDS):
            raise ValueError(f"Calibration-v2 prompt coverage drifted at scale {expected_scale}.")
        derived = all(prompt.get("pass") is True for prompt in prompts.values())
        if row.get("passes_all_preregistered_gates_on_all_prompts") is not derived:
            raise ValueError(f"Calibration-v2 scale arithmetic drifted at {expected_scale}.")
        if derived:
            derived_passing.append(expected_scale)
    selected_scale = min(derived_passing) if derived_passing else None
    if (
        payload.get("passing_scales") != derived_passing
        or payload.get("selected_true_cfg_scale") != selected_scale
        or (payload.get("status") == "selected") != (selected_scale is not None)
        or payload.get("normal_benchmark_update_authorized") is not (selected_scale is not None)
    ):
        raise ValueError("Calibration-v2 selected-scale arithmetic is inconsistent.")
    source = payload.get("source_bindings") or {}
    _validate_manifest_seed_binding(source.get("calibration_manifest"), selected_seed)
    common_evidence = source.get("selected_common_seed_evidence") or {}
    receipt = common_evidence.get("selection_receipt") or {}
    qualified = payload.get("qualified_setting_binding")
    if qualified != {
        "selected_common_seed": selected_seed,
        "selected_true_cfg_scale": selected_scale,
        "common_seed_selection_document_sha256": receipt.get("document_sha256"),
        "all_45_rows_seed_homogeneous": True,
        "v1_media_reused": False,
        "common_seed_candidate_media_reused": False,
        "reuse_authentication_status": "authenticated_fresh_execution_paths_and_inodes",
    }:
        raise ValueError("Calibration-v2 qualified-setting provenance is inconsistent.")
    prerequisites = payload.get("global_prerequisites") or {}
    if prerequisites.get("positive_target_concepts_scored") is not False or set(
        (prerequisites.get("path_parity_by_prompt") or {})
    ) != set(PROMPT_IDS):
        raise ValueError("Calibration-v2 global prerequisite coverage drifted.")
    if verify_source_files:
        required_documents = {
            "objective_report": ("evaluation", OBJECTIVE_EVALUATION),
            "reviewer_manifest": ("review_package", PRIVATE_REVIEW_MANIFEST),
            "source_reviewer_manifest": ("review_package", REVIEW_PACKAGE),
            "parity_reviewer_manifest": (
                "review_package",
                PARITY_REVIEW_PACKAGE,
            ),
            "unblinding_map": ("unblinding_map", UNBLINDING_MAP),
            "blinded_manual_review_ledger": ("review", MANUAL_REVIEW),
        }
        if not isinstance(source, Mapping) or not all(
            isinstance(source.get(role), Mapping) for role in required_documents
        ):
            raise ValueError("Calibration-v2 selection source-document bindings are incomplete.")
        _require_published_review_cohort(
            source_root=Path(str(source["source_reviewer_manifest"]["path"])).resolve().parent,
            parity_root=Path(str(source["parity_reviewer_manifest"]["path"])).resolve().parent,
            private_root=Path(str(source["objective_report"]["path"])).resolve().parent,
        )
        _verify_binding_tree(source)
        documents = {
            role: _validate_document(
                Path(str(source[role]["path"])).resolve(),
                field,
                expected,
            )
            for role, (field, expected) in required_documents.items()
        }
        objective = documents["objective_report"]
        private_reviewer = documents["reviewer_manifest"]
        source_reviewer = documents["source_reviewer_manifest"]
        parity_reviewer = documents["parity_reviewer_manifest"]
        unblinding = documents["unblinding_map"]
        ledger = documents["blinded_manual_review_ledger"]
        _require_published_review_cohort(
            source_root=Path(str(source["source_reviewer_manifest"]["path"])).resolve().parent,
            parity_root=Path(str(source["parity_reviewer_manifest"]["path"])).resolve().parent,
            private_root=Path(str(source["objective_report"]["path"])).resolve().parent,
            source_reviewer=source_reviewer,
            parity_reviewer=parity_reviewer,
        )
        validate_objective_report(objective, verify_source_files=True)
        validate_selection_report_against_evidence(
            payload,
            objective=objective,
            objective_path=Path(str(source["objective_report"]["path"])).resolve(),
            private_reviewer=private_reviewer,
            private_reviewer_path=Path(str(source["reviewer_manifest"]["path"])).resolve(),
            source_reviewer=source_reviewer,
            source_reviewer_path=Path(str(source["source_reviewer_manifest"]["path"])).resolve(),
            parity_reviewer=parity_reviewer,
            parity_reviewer_path=Path(str(source["parity_reviewer_manifest"]["path"])).resolve(),
            unblinding=unblinding,
            unblinding_path=Path(str(source["unblinding_map"]["path"])).resolve(),
            ledger=ledger,
            ledger_path=Path(str(source["blinded_manual_review_ledger"]["path"])).resolve(),
        )


def _write_document_unchecked(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_document_new(path: Path, payload: Mapping[str, Any]) -> tuple[Path, Path]:
    if payload.get("document_sha256") != document_digest(payload):
        raise ValueError("Cannot publish a calibration-v2 document with an invalid digest.")
    return v1._write_document_new(path, payload)


def _publish_package_commit_last(
    staging: Path, destination: Path, *, commit_name: str
) -> tuple[int, int]:
    """Publish one review tree without replacement and return its owned inode."""

    publish_hardlink_tree_commit_last(
        staging,
        destination,
        commit_relative_path=Path(commit_name),
    )
    observed = destination.lstat()
    return observed.st_dev, observed.st_ino


def _require_sealed_package(root: Path, *, commit_name: str, label: str) -> None:
    if root.is_symlink() or not root.is_dir():
        raise FileNotFoundError(f"{label} is absent: {root}")
    require_nonwritable_directories(root, label=label)
    for member in root.rglob("*"):
        observed = member.lstat()
        if member.is_symlink() or (not member.is_dir() and not member.is_file()):
            raise ValueError(f"{label} contains an aliased or unsupported member: {member}")
        if member.is_file() and observed.st_mode & 0o222:
            raise ValueError(f"{label} contains a writable file: {member}")
    commit = root / commit_name
    sidecar = commit.with_suffix(commit.suffix + ".sha256")
    if not commit.is_file() or not sidecar.is_file():
        raise ValueError(f"{label} lacks its commit-last document: {commit_name}")


def _require_published_review_cohort(
    *,
    source_root: Path,
    parity_root: Path,
    private_root: Path,
    source_reviewer: Mapping[str, Any] | None = None,
    parity_reviewer: Mapping[str, Any] | None = None,
) -> None:
    """Admit only all three sealed packages, with the private join map last."""

    source_root, parity_root, private_root = _require_access_disjoint_roots(
        source_root,
        parity_root,
        private_root,
        label="Published source, parity, and private review roots",
    )
    _require_sealed_package(
        source_root,
        commit_name="reviewer_manifest.json",
        label="Source-only calibration-v2 review package",
    )
    _require_sealed_package(
        parity_root,
        commit_name="reviewer_manifest.json",
        label="Parity-only calibration-v2 review package",
    )
    # The unblinding map authenticates both public manifest digests and is
    # published only after both complete public trees.
    _require_sealed_package(
        private_root,
        commit_name="unblinding_map.json",
        label="Private calibration-v2 join package",
    )
    if source_reviewer is None or parity_reviewer is None:
        return
    expected_source_root = {
        "README.md",
        "images",
        "manual_review_template.json",
        "reviewer_manifest.json",
        "reviewer_manifest.json.sha256",
    }
    expected_parity_root = expected_source_root | {"contact_sheets"}
    expected_private_root = {
        "objective_report.json",
        "objective_report.json.sha256",
        "private_reviewer_manifest.json",
        "private_reviewer_manifest.json.sha256",
        "unblinding_map.json",
        "unblinding_map.json.sha256",
    }
    if {path.name for path in source_root.iterdir()} != expected_source_root:
        raise ValueError("Source-only review package membership drifted.")
    if {path.name for path in parity_root.iterdir()} != expected_parity_root:
        raise ValueError("Parity-only review package membership drifted.")
    if {path.name for path in private_root.iterdir()} != expected_private_root:
        raise ValueError("Private review package membership drifted.")
    expected_source_images = {
        Path(str(row["image_relative_path"])).name for row in source_reviewer.get("rows", ())
    }
    expected_parity_images = {
        Path(str(row["image_relative_path"])).name for row in parity_reviewer.get("rows", ())
    }
    expected_parity_sheets = {
        Path(str(row["relative_path"])).name for row in parity_reviewer.get("contact_sheets", ())
    }
    if {path.name for path in (source_root / "images").iterdir()} != expected_source_images:
        raise ValueError("Source-only review image membership drifted.")
    if {path.name for path in (parity_root / "images").iterdir()} != expected_parity_images:
        raise ValueError("Parity-only review image membership drifted.")
    if {path.name for path in (parity_root / "contact_sheets").iterdir()} != expected_parity_sheets:
        raise ValueError("Parity-only review contact-sheet membership drifted.")


def _require_published_final_evidence(root: Path) -> None:
    _require_sealed_package(
        root,
        commit_name="selection_report.json",
        label="Final calibration-v2 evidence",
    )
    expected = {
        "manual_review_ledger.json",
        "manual_review_ledger.json.sha256",
        "selection_report.json",
        "selection_report.json.sha256",
    }
    if {path.name for path in root.iterdir()} != expected:
        raise ValueError("Final calibration-v2 evidence membership drifted.")


def canonical_final_evidence_root(project_root: str | Path | None = None) -> Path:
    """Return the sole production path for the calibration-v2 decision."""

    root = Path(project_root or Path(__file__).resolve().parents[3]).expanduser().resolve()
    return root / FINAL_EVIDENCE_ROOT_RELATIVE


def require_canonical_final_evidence_root(
    path: str | Path,
    *,
    project_root: str | Path | None = None,
) -> Path:
    """Reject a calibration-v2 final-evidence path outside the frozen audit root."""

    expected = canonical_final_evidence_root(project_root)
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = Path(project_root or Path(__file__).resolve().parents[3]) / candidate
    resolved = candidate.resolve()
    if resolved != expected:
        raise ValueError(
            "Calibration-v2 final evidence must use its exact canonical attempt-2 root: "
            f"expected={expected}, found={resolved}."
        )
    return expected


def read_final_evidence(
    path: str | Path | None = None,
    *,
    project_root: str | Path | None = None,
) -> dict[str, Any]:
    """Read one canonical, immutable, fully source-verified v2 scale decision.

    The selection report is the commit-last member of a four-file sealed
    publication.  Admission reopens every substantive source bound by the
    report and recomputes the scale decision; a locally self-consistent report
    is therefore insufficient.
    """

    evidence_root = (
        canonical_final_evidence_root(project_root)
        if path is None
        else require_canonical_final_evidence_root(path, project_root=project_root)
    )
    _require_published_final_evidence(evidence_root)
    ledger_path = evidence_root / FINAL_EVIDENCE_LEDGER_FILENAME
    selection_path = evidence_root / FINAL_EVIDENCE_SELECTION_FILENAME
    ledger = _validate_document(ledger_path, "review", MANUAL_REVIEW)
    selection = _validate_document(selection_path, "selection_report", SELECTION_REPORT)

    expected_ledger_binding = {
        **_binding(ledger_path),
        "document_sha256": ledger["document_sha256"],
    }
    source_bindings = selection.get("source_bindings")
    if (
        not isinstance(source_bindings, Mapping)
        or source_bindings.get("blinded_manual_review_ledger") != expected_ledger_binding
    ):
        raise ValueError(
            "Calibration-v2 selection report does not bind the canonical final manual ledger."
        )

    # This is intentionally always live and exhaustive.  There is no
    # audit-only switch on the public admission reader.
    validate_selection_report(selection, verify_source_files=True)

    # Detect replacement of the canonical publication while its deep source
    # graph was being authenticated.
    _require_published_final_evidence(evidence_root)
    if (
        _validate_document(ledger_path, "review", MANUAL_REVIEW) != ledger
        or _validate_document(selection_path, "selection_report", SELECTION_REPORT) != selection
    ):
        raise RuntimeError("Canonical calibration-v2 final evidence changed during admission.")
    return selection


def prepare_evaluation_package(
    *,
    manifest_path: str | Path,
    registry_path: str | Path,
    output_root: str | Path,
    parity_output_root: str | Path,
    private_output_root: str | Path,
    root: str | Path,
) -> dict[str, Any]:
    source_destination, parity_destination, private_destination = _require_access_disjoint_roots(
        Path(output_root),
        Path(parity_output_root),
        Path(private_output_root),
        label="Source-review, parity-review, and private evidence roots",
    )
    for destination in (source_destination, parity_destination, private_destination):
        if destination.exists():
            raise FileExistsError(f"Refusing to overwrite evaluation package: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
    source_temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{source_destination.name}.tmp-",
            dir=source_destination.parent,
        )
    )
    parity_temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{parity_destination.name}.tmp-",
            dir=parity_destination.parent,
        )
    )
    private_temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{private_destination.name}.tmp-",
            dir=private_destination.parent,
        )
    )
    staged_identities = {
        temporary: (temporary.lstat().st_dev, temporary.lstat().st_ino)
        for temporary in (source_temporary, parity_temporary, private_temporary)
    }
    published: dict[Path, tuple[int, int]] = {}
    try:
        collected = collect_calibration_artifacts(manifest_path, registry_path, root=root)
        objective = build_objective_report(collected)
        objective_path = private_temporary / "objective_report.json"
        _write_document_unchecked(objective_path, objective)
        objective_path.with_suffix(".json.sha256").write_text(
            f"{objective['document_sha256']}  {objective_path.name}\n", encoding="utf-8"
        )
        (
            source_reviewer,
            parity_reviewer,
            private_reviewer,
            unblinding,
            source_template,
            parity_template,
        ) = build_review_documents(
            objective,
            objective_path=objective_path,
            review_root=source_temporary,
            final_review_root=source_destination,
            parity_review_root=parity_temporary,
            final_parity_review_root=parity_destination,
        )
        _write_document_unchecked(source_temporary / "manual_review_template.json", source_template)
        _write_document_unchecked(parity_temporary / "manual_review_template.json", parity_template)
        _write_document_unchecked(
            private_temporary / "private_reviewer_manifest.json", private_reviewer
        )
        _write_document_unchecked(private_temporary / "unblinding_map.json", unblinding)
        for directory, name, payload in (
            (source_temporary, "reviewer_manifest.json", source_reviewer),
            (parity_temporary, "reviewer_manifest.json", parity_reviewer),
            (private_temporary, "objective_report.json", objective),
            (
                private_temporary,
                "private_reviewer_manifest.json",
                private_reviewer,
            ),
            (private_temporary, "unblinding_map.json", unblinding),
        ):
            (directory / f"{name}.sha256").write_text(
                f"{payload['document_sha256']}  {name}\n", encoding="utf-8"
            )
        (source_temporary / "README.md").write_text(
            "# Source-only blinded Flux.1 calibration-v2 review\n\n"
            "Open all 45 PNGs in `images/` at original resolution and score only the "
            "registered source concepts and hard gates in a writable copy of "
            "`manual_review_template.json`. Do not access the separate parity package. "
            "This package contains no role/scale groups, pair topology, contact sheets, "
            "selected seed, unblinding map, or positive target concepts.\n",
            encoding="utf-8",
        )
        (parity_temporary / "README.md").write_text(
            "# Path-parity-only blinded Flux.1 calibration-v2 review\n\n"
            "Open all six PNGs at original resolution and compare the three randomized "
            "pairs using the contact sheets for navigation. Complete a writable copy of "
            "`manual_review_template.json`. Do not access the separate 45-row source "
            "package. This package contains no ladder candidates, scales, role labels, "
            "selected seed, unblinding map, or positive target concepts.\n",
            encoding="utf-8",
        )
        for public_tree in (source_temporary, parity_temporary):
            freeze_tree(public_tree, label="Calibration-v2 blinded review staging")
        freeze_tree(private_temporary, label="Calibration-v2 private review staging")
        for path in private_temporary.rglob("*"):
            path.chmod(0o500 if path.is_dir() else 0o400)
        private_temporary.chmod(0o500)
        # Each public package is complete on its own.  The private unblinding
        # map, which binds both public manifest digests, is the cross-package
        # admission edge and is therefore published last.
        published[source_destination] = _publish_package_commit_last(
            source_temporary,
            source_destination,
            commit_name="reviewer_manifest.json",
        )
        published[parity_destination] = _publish_package_commit_last(
            parity_temporary,
            parity_destination,
            commit_name="reviewer_manifest.json",
        )
        published[private_destination] = _publish_package_commit_last(
            private_temporary,
            private_destination,
            commit_name="unblinding_map.json",
        )
        for temporary, identity in staged_identities.items():
            cleanup_owned_staging(temporary, identity)
        _require_published_review_cohort(
            source_root=source_destination,
            parity_root=parity_destination,
            private_root=private_destination,
            source_reviewer=source_reviewer,
            parity_reviewer=parity_reviewer,
        )
        _validate_review_evidence_topology(
            objective=objective,
            private_reviewer=private_reviewer,
            private_reviewer_path=private_destination / "private_reviewer_manifest.json",
            source_reviewer=source_reviewer,
            source_reviewer_path=source_destination / "reviewer_manifest.json",
            parity_reviewer=parity_reviewer,
            parity_reviewer_path=parity_destination / "reviewer_manifest.json",
            unblinding=unblinding,
        )
    except BaseException:
        for destination, identity in reversed(tuple(published.items())):
            cleanup_owned_staging(destination, identity)
        for temporary, identity in staged_identities.items():
            cleanup_owned_staging(temporary, identity)
        raise
    return {
        "status": "prepared_pending_target_blind_manual_review",
        "selected_common_seed": objective["selected_common_seed"],
        "source_reviewer_output_root": str(source_destination),
        "parity_reviewer_output_root": str(parity_destination),
        "private_output_root": str(private_destination),
        "objective_report": str(private_destination / "objective_report.json"),
        "objective_report_document_sha256": objective["document_sha256"],
        "private_reviewer_manifest": str(private_destination / "private_reviewer_manifest.json"),
        "private_reviewer_manifest_document_sha256": private_reviewer["document_sha256"],
        "source_reviewer_manifest": str(source_destination / "reviewer_manifest.json"),
        "source_reviewer_manifest_document_sha256": source_reviewer["document_sha256"],
        "source_manual_review_template": str(source_destination / "manual_review_template.json"),
        "parity_reviewer_manifest": str(parity_destination / "reviewer_manifest.json"),
        "parity_reviewer_manifest_document_sha256": parity_reviewer["document_sha256"],
        "parity_manual_review_template": str(parity_destination / "manual_review_template.json"),
        "unblinding_map": str(private_destination / "unblinding_map.json"),
        "unblinding_map_document_sha256": unblinding["document_sha256"],
        "inert_negative_integrity_pass": objective["inert_negative_integrity"]["pass"],
    }


def _validate_document(path: Path, field: str, expected: str) -> dict[str, Any]:
    payload = _load_json(path, expected)
    if payload.get(field) != expected or payload.get("document_sha256") != document_digest(payload):
        raise ValueError(f"Document identity or digest is invalid: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split() != [
        payload["document_sha256"],
        path.name,
    ]:
        raise ValueError(f"Document sidecar is absent or inconsistent: {path}")
    return payload


def finalize_evaluation(
    *,
    objective_path: str | Path,
    private_reviewer_manifest_path: str | Path,
    source_reviewer_manifest_path: str | Path,
    parity_reviewer_manifest_path: str | Path,
    unblinding_map_path: str | Path,
    source_manual_draft_path: str | Path,
    parity_manual_draft_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    objective_path = Path(objective_path).expanduser().resolve()
    private_reviewer_path = Path(private_reviewer_manifest_path).expanduser().resolve()
    source_reviewer_path = Path(source_reviewer_manifest_path).expanduser().resolve()
    parity_reviewer_path = Path(parity_reviewer_manifest_path).expanduser().resolve()
    unblinding_path = Path(unblinding_map_path).expanduser().resolve()
    source_draft_path = Path(source_manual_draft_path).expanduser().resolve()
    parity_draft_path = Path(parity_manual_draft_path).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite final calibration-v2 evidence: {destination}")
    private_root = objective_path.parent
    if private_reviewer_path.parent != private_root or unblinding_path.parent != private_root:
        raise ValueError(
            "Objective, private reviewer manifest, and unblinding map must share one private root."
        )
    source_root, parity_root, _ = _require_access_disjoint_roots(
        source_reviewer_path.parent,
        parity_reviewer_path.parent,
        private_root,
        label="Source-review, parity-review, and private evidence roots",
    )
    if (
        source_draft_path == parity_draft_path
        or private_root in source_draft_path.parents
        or private_root in parity_draft_path.parents
        or parity_root in source_draft_path.parents
        or source_root in parity_draft_path.parents
    ):
        raise ValueError("Manual drafts violate the access-disjoint reviewer topology.")
    if source_draft_path.parent == parity_draft_path.parent:
        raise ValueError("Source and parity manual drafts must come from separate access roots.")
    if source_root in destination.parents or parity_root in destination.parents:
        raise ValueError("Final joined evidence cannot be published inside a public review root.")
    _require_published_review_cohort(
        source_root=source_root,
        parity_root=parity_root,
        private_root=private_root,
    )
    objective = _validate_document(objective_path, "evaluation", OBJECTIVE_EVALUATION)
    validate_objective_report(objective, verify_source_files=True)
    private_reviewer = _validate_document(
        private_reviewer_path, "review_package", PRIVATE_REVIEW_MANIFEST
    )
    source_reviewer = _validate_document(source_reviewer_path, "review_package", REVIEW_PACKAGE)
    parity_reviewer = _validate_document(
        parity_reviewer_path, "review_package", PARITY_REVIEW_PACKAGE
    )
    unblinding = _validate_document(unblinding_path, "unblinding_map", UNBLINDING_MAP)
    _require_published_review_cohort(
        source_root=source_root,
        parity_root=parity_root,
        private_root=private_root,
        source_reviewer=source_reviewer,
        parity_reviewer=parity_reviewer,
    )
    source_draft = _load_json(source_draft_path, "source manual review draft")
    parity_draft = _load_json(parity_draft_path, "path-parity manual review draft")
    ledger = normalize_manual_ledger(
        source_draft,
        parity_draft,
        objective=objective,
        objective_path=objective_path,
        private_reviewer=private_reviewer,
        private_reviewer_path=private_reviewer_path,
        source_reviewer=source_reviewer,
        source_reviewer_path=source_reviewer_path,
        parity_reviewer=parity_reviewer,
        parity_reviewer_path=parity_reviewer_path,
        unblinding=unblinding,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent))
    staging_stat = staging.lstat()
    staging_identity = (staging_stat.st_dev, staging_stat.st_ino)
    published_identity: tuple[int, int] | None = None
    try:
        staged_ledger_path, _ = _write_document_new(staging / "manual_review_ledger.json", ledger)
        ledger_path = destination / "manual_review_ledger.json"
        selection = build_selection_report(
            objective=objective,
            objective_path=objective_path,
            private_reviewer=private_reviewer,
            private_reviewer_path=private_reviewer_path,
            source_reviewer=source_reviewer,
            source_reviewer_path=source_reviewer_path,
            parity_reviewer=parity_reviewer,
            parity_reviewer_path=parity_reviewer_path,
            unblinding=unblinding,
            unblinding_path=unblinding_path,
            ledger=ledger,
            ledger_path=staged_ledger_path,
        )
        ledger_binding = selection["source_bindings"]["blinded_manual_review_ledger"]
        if ledger_binding.get("path") != str(staged_ledger_path):
            raise RuntimeError("Staged calibration-v2 ledger binding was not reconstructed.")
        ledger_binding["path"] = str(ledger_path)
        selection["document_sha256"] = document_digest(selection)
        validate_selection_report(selection, verify_source_files=False)
        _write_document_new(staging / "selection_report.json", selection)
        freeze_tree(staging, label="Final calibration-v2 evidence staging")
        published_identity = _publish_package_commit_last(
            staging,
            destination,
            commit_name="selection_report.json",
        )
        cleanup_owned_staging(staging, staging_identity)
        _require_published_final_evidence(destination)
        reopened_ledger = _validate_document(ledger_path, "review", MANUAL_REVIEW)
        selection_path = destination / "selection_report.json"
        reopened_selection = _validate_document(
            selection_path, "selection_report", SELECTION_REPORT
        )
        if reopened_ledger != ledger or reopened_selection != selection:
            raise RuntimeError("Published calibration-v2 evidence changed during admission.")
        validate_selection_report(reopened_selection, verify_source_files=True)
    except BaseException:
        cleanup_owned_staging(staging, staging_identity)
        if published_identity is not None:
            cleanup_owned_staging(destination, published_identity)
        raise
    return {
        "status": selection["status"],
        "selected_common_seed": selection["selected_common_seed"],
        "selected_true_cfg_scale": selection["selected_true_cfg_scale"],
        "manual_review_ledger": str(ledger_path),
        "manual_review_ledger_document_sha256": ledger["document_sha256"],
        "selection_report": str(selection_path),
        "selection_report_document_sha256": selection["document_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--submission-registry", required=True)
    prepare.add_argument("--output-root", required=True, help="New reviewer-only package root.")
    prepare.add_argument(
        "--parity-output-root",
        required=True,
        help="Separate new package root for the independent path-parity reviewer.",
    )
    prepare.add_argument(
        "--private-output-root",
        required=True,
        help="Separate new root for objective and unblinding evidence.",
    )
    prepare.add_argument("--project-root", default=str(Path(__file__).resolve().parents[3]))
    finalize = commands.add_parser("finalize")
    finalize.add_argument("--objective-report", required=True)
    finalize.add_argument("--private-reviewer-manifest", required=True)
    finalize.add_argument("--source-reviewer-manifest", required=True)
    finalize.add_argument("--parity-reviewer-manifest", required=True)
    finalize.add_argument("--unblinding-map", required=True)
    finalize.add_argument("--source-manual-draft", required=True)
    finalize.add_argument("--parity-manual-draft", required=True)
    finalize.add_argument("--output-root", required=True)
    finalize.add_argument("--project-root", default=str(Path(__file__).resolve().parents[3]))
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_evaluation_package(
            manifest_path=args.manifest,
            registry_path=args.submission_registry,
            output_root=args.output_root,
            parity_output_root=args.parity_output_root,
            private_output_root=args.private_output_root,
            root=args.project_root,
        )
    else:
        final_root = require_canonical_final_evidence_root(
            args.output_root,
            project_root=args.project_root,
        )
        result = finalize_evaluation(
            objective_path=args.objective_report,
            private_reviewer_manifest_path=args.private_reviewer_manifest,
            source_reviewer_manifest_path=args.source_reviewer_manifest,
            parity_reviewer_manifest_path=args.parity_reviewer_manifest,
            unblinding_map_path=args.unblinding_map,
            source_manual_draft_path=args.source_manual_draft,
            parity_manual_draft_path=args.parity_manual_draft,
            output_root=final_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
