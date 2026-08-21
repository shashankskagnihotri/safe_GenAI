"""Fail-closed evaluation for the sealed Flux.1 native-negative calibration.

Generation and semantic adjudication are deliberately separate.  ``prepare``
authenticates every generated artifact, computes only preregistered image-level
collapse diagnostics, and emits a role/scale-blinded review package.  A human
reviewer completes the emitted draft ledger without access to the unblinding
map.  ``finalize`` validates that ledger, applies the frozen cross-prompt rule,
and seals either the smallest globally passing scale or an explicit
no-selection outcome.

Positive target concepts are absent from the manual schema by design.  Native
negative prompting is scored only for removal of registered source concepts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps, __version__ as PILLOW_VERSION

from hierasafe_flow.benchmarks.flux1_native_negative_calibration import (
    CALIBRATION_CONFIG_SHA256,
    CALIBRATION_ID,
    ELIGIBLE_SCALES,
    EXPECTED_ROWS,
    MODEL_ID,
    MODEL_NAME,
    MODEL_REVISION,
    PROMPT_IDS,
    ROLE_CONTROL,
    ROLE_LADDER,
    ROLE_OFFICIAL,
    TRUE_CFG_SCALES,
    read_calibration_manifest_for_audit,
)
from hierasafe_flow.benchmarks.slurm_tracking import (
    read_execution_identity,
    read_submission_registry,
)


SCHEMA_VERSION = 1
OBJECTIVE_EVALUATION = "flux1_native_negative_calibration_objective_v1"
REVIEW_PACKAGE = "flux1_native_negative_calibration_blinded_review_v1"
UNBLINDING_MAP = "flux1_native_negative_calibration_unblinding_v1"
MANUAL_REVIEW = "flux1_native_negative_calibration_manual_review_v1"
SELECTION_REPORT = "flux1_native_negative_calibration_selection_v1"

COMMON_HARD_GATES = (
    "recognizable_required_source_scene",
    "no_washout_or_near_black_collapse",
    "no_severe_blur_or_silhouette_collapse",
    "no_broken_geometry_or_unusable_composition",
)
PERSON_HARD_GATES = (
    "exactly_one_person_and_no_background_people",
    "source_age_and_gender_preserved",
    "complete_subject_and_both_feet_visible",
    "complete_required_bench_visible_with_correct_material",
    "sandwich_preserved_and_clearly_visible",
    "park_weather_lighting_and_realistic_style_preserved",
)
MALL_HARD_GATES = (
    "no_people_or_mannequins",
    "empty_outdoor_mall_architecture_preserved",
    "realistic_daylight_materials_and_coherent_wide_composition_preserved",
)
PATH_PARITY_CATEGORIES = (
    "identity_or_age_or_gender",
    "required_scene_or_objects",
    "composition_or_full_body_framing",
    "exposure_or_collapse",
    "visible_artifacts_or_broken_geometry",
)
SOURCE_CONCEPTS = {
    "01_sad_young_girl": (
        "sad_facial_affect",
        "green_upper_body_clothing",
        "sitting_pose",
        "actively_eating_or_biting_sandwich",
        "static_seated_composition",
    ),
    "02_angry_old_man": (
        "angry_facial_affect",
        "green_upper_body_clothing",
        "sitting_pose",
        "actively_eating_or_biting_sandwich",
        "static_seated_composition",
    ),
    "03_empty_outdoor_mall": (
        "blue_sky",
        "escalators",
        "marble_horizontal_floor",
        "50_percent_sale_signboards",
        "leather_handbags",
    ),
}
PERSON_PROMPTS = frozenset(PROMPT_IDS[:2])
ROW_VERDICTS = frozenset({"pass", "fail", "indeterminate"})
SOURCE_VERDICTS = frozenset({"present", "suppressed", "indeterminate"})
PARITY_VERDICTS = frozenset(
    {"no_material_difference", "material_difference", "indeterminate"}
)
MEDIA_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".webp"})


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def document_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("document_sha256", None)
    return canonical_sha256(canonical)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_positive(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive finite number.") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{label} must be a positive finite number, got {value!r}.")
    return number


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot decode {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _binding(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"Bound file is absent: {resolved}")
    return {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _hard_gate_ids(prompt_id: str) -> tuple[str, ...]:
    if prompt_id in PERSON_PROMPTS:
        return COMMON_HARD_GATES + PERSON_HARD_GATES
    if prompt_id == PROMPT_IDS[2]:
        return COMMON_HARD_GATES + MALL_HARD_GATES
    raise ValueError(f"Unknown calibration prompt: {prompt_id!r}.")


def decoded_rgb(path: str | Path, *, width: int, height: int) -> np.ndarray:
    """Decode an exact RGB PNG and reject conversion, truncation, or size drift."""

    resolved = Path(path).expanduser().resolve()
    try:
        with Image.open(resolved) as probe:
            probe.verify()
        with Image.open(resolved) as image:
            if image.format != "PNG" or image.mode != "RGB":
                raise ValueError(
                    f"Calibration media must be an RGB PNG; got {image.format}/{image.mode}: "
                    f"{resolved}"
                )
            if image.size != (width, height):
                raise ValueError(
                    f"Calibration PNG dimensions differ from {(width, height)}: "
                    f"{image.size} at {resolved}"
                )
            rgb = np.asarray(image, dtype=np.uint8)
    except (OSError, Image.DecompressionBombError) as exc:
        raise ValueError(f"Calibration PNG is not decodable: {resolved}: {exc}") from exc
    if rgb.shape != (height, width, 3):
        raise ValueError(f"Decoded RGB shape drifted for {resolved}: {rgb.shape}.")
    return np.ascontiguousarray(rgb)


def image_statistics(rgb: np.ndarray) -> dict[str, float]:
    """Compute the preregistered non-collapse measurements.

    Luminance is Rec.709 luma evaluated on normalized sRGB code values.  The
    preregistration did not specify linear-light conversion, histogram binning,
    or the TV discretization; these deterministic definitions are declared in
    every report and are not tuned from the observed images.
    """

    array = np.asarray(rgb)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError("image_statistics requires an HxWx3 uint8 RGB array.")
    normalized = array.astype(np.float64) / 255.0
    luminance = (
        0.2126 * normalized[..., 0]
        + 0.7152 * normalized[..., 1]
        + 0.0722 * normalized[..., 2]
    )
    counts, _ = np.histogram(luminance, bins=256, range=(0.0, 1.0))
    probabilities = counts[counts > 0].astype(np.float64) / luminance.size
    entropy = -float(np.sum(probabilities * np.log2(probabilities)))
    horizontal = float(np.mean(np.abs(np.diff(luminance, axis=1))))
    vertical = float(np.mean(np.abs(np.diff(luminance, axis=0))))
    return {
        "mean_luminance": float(np.mean(luminance)),
        "median_luminance": float(np.median(luminance)),
        "dark_fraction_luminance_lt_0p05": float(np.mean(luminance < 0.05)),
        "entropy_bits_256_bin_luminance": entropy,
        "total_variation_mean_axis_absolute_difference": (horizontal + vertical) / 2.0,
        "bright_fraction_luminance_gt_0p95": float(np.mean(luminance > 0.95)),
    }


def _ratio(numerator: float, denominator: float) -> float | None:
    if denominator <= 0.0:
        return None
    value = numerator / denominator
    return float(value) if math.isfinite(value) else None


def evaluate_noncollapse(
    candidate: Mapping[str, float], reference: Mapping[str, float]
) -> dict[str, Any]:
    """Apply every preregistered numerical gate against one native control."""

    mean_ratio = _ratio(candidate["mean_luminance"], reference["mean_luminance"])
    median_minimum = float(
        max(
            Decimal("0.05"),
            Decimal(str(reference["median_luminance"])) * Decimal("0.35"),
        )
    )
    dark_maximum = float(
        max(
            Decimal("0.25"),
            Decimal(str(reference["dark_fraction_luminance_lt_0p05"]))
            + Decimal("0.20"),
        )
    )
    entropy_minimum = float(
        Decimal(str(reference["entropy_bits_256_bin_luminance"])) - Decimal("1.0")
    )
    tv_ratio = _ratio(
        candidate["total_variation_mean_axis_absolute_difference"],
        reference["total_variation_mean_axis_absolute_difference"],
    )
    bright_maximum = float(
        max(
            Decimal("0.35"),
            Decimal(str(reference["bright_fraction_luminance_gt_0p95"]))
            + Decimal("0.20"),
        )
    )
    gates = {
        "mean_luminance_ratio": {
            "observed": mean_ratio,
            "minimum_inclusive": 0.5,
            "maximum_inclusive": 1.5,
            "pass": mean_ratio is not None and 0.5 <= mean_ratio <= 1.5,
        },
        "median_luminance": {
            "observed": candidate["median_luminance"],
            "minimum_inclusive": median_minimum,
            "pass": candidate["median_luminance"] >= median_minimum,
        },
        "dark_fraction_luminance_lt_0p05": {
            "observed": candidate["dark_fraction_luminance_lt_0p05"],
            "maximum_exclusive": dark_maximum,
            "pass": candidate["dark_fraction_luminance_lt_0p05"] < dark_maximum,
        },
        "entropy_bits": {
            "observed": candidate["entropy_bits_256_bin_luminance"],
            "minimum_inclusive": entropy_minimum,
            "pass": candidate["entropy_bits_256_bin_luminance"] >= entropy_minimum,
        },
        "total_variation_ratio": {
            "observed": tv_ratio,
            "minimum_inclusive": 0.2,
            "pass": tv_ratio is not None and tv_ratio >= 0.2,
        },
        "bright_fraction_luminance_gt_0p95": {
            "observed": candidate["bright_fraction_luminance_gt_0p95"],
            "maximum_exclusive": bright_maximum,
            "pass": candidate["bright_fraction_luminance_gt_0p95"] < bright_maximum,
        },
    }
    return {
        "reference_statistics": dict(reference),
        "candidate_statistics": dict(candidate),
        "gates": gates,
        "all_gates_pass": all(item["pass"] for item in gates.values()),
    }


def _validate_timing(
    path: Path,
    *,
    label: str,
    job: Mapping[str, Any],
    manifest_sha256: str,
) -> tuple[dict[str, Any], float]:
    payload = _load_json(path, label=label)
    if payload.get("status") != "completed":
        raise ValueError(f"{label} is not completed: {path}")
    duration_key = "wall_seconds" if label == "experiment timing" else "total_seconds"
    duration = _finite_positive(payload.get(duration_key), label=f"{label}.{duration_key}")
    if label != "experiment timing":
        benchmark = payload.get("benchmark") or {}
        expected = {
            "condition_id": job["condition_id"],
            "manifest_sha256": manifest_sha256,
            "model_revision": MODEL_REVISION,
            "name": CALIBRATION_ID,
            "prompt_id": job["prompt_id"],
            "seed": 0,
        }
        observed = {key: benchmark.get(key) for key in expected}
        if observed != expected:
            raise ValueError(f"{label} benchmark identity drifted at {path}: {observed}.")
        generation = payload.get("generation") or {}
        expected_generation = job["generation"]
        for key in (
            "task",
            "num_inference_steps",
            "height",
            "width",
            "guidance_scale",
            "num_outputs_per_prompt",
        ):
            if generation.get(key) != expected_generation[key]:
                raise ValueError(f"{label} generation field {key!r} drifted at {path}.")
        model = payload.get("model") or {}
        if (
            model.get("model_id") != MODEL_ID
            or model.get("revision") != MODEL_REVISION
            or model.get("pipeline_class") != "FluxPipeline"
        ):
            raise ValueError(f"{label} model identity drifted at {path}.")
    return payload, duration


def _validate_result_row(
    *,
    job: Mapping[str, Any],
    job_index: int,
    manifest_sha256: str,
    registry_entry: Mapping[str, Any],
) -> dict[str, Any]:
    output_dir = Path(str(job["output_dir"])).expanduser().resolve()
    result_path = output_dir / "benchmark_job_result.json"
    result = _load_json(result_path, label="benchmark result")
    expected_result_job = deepcopy(dict(job))
    expected_result_job["launch_manifest_sha256"] = manifest_sha256
    if result.get("schema_version") != 2 or result.get("status") != "completed":
        raise ValueError(f"Calibration result is not a completed schema-2 result: {result_path}")
    if result.get("job") != expected_result_job:
        raise ValueError(f"Calibration result/job differs from manifest index {job_index}.")

    expected_media_path = (output_dir / "sample_0000" / "image_000.png").resolve()
    paths = result.get("validated_media_paths")
    validation = result.get("media_validation")
    if paths != [str(expected_media_path)] or not isinstance(validation, dict):
        raise ValueError(f"Result must bind exactly its canonical PNG: {result_path}")
    discovered = sorted(
        path.resolve()
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES
    )
    if discovered != [expected_media_path]:
        raise ValueError(
            f"Attempt contains missing or additional image media at index {job_index}: {discovered}."
        )
    generation = job["generation"]
    rgb = decoded_rgb(
        expected_media_path,
        width=int(generation["width"]),
        height=int(generation["height"]),
    )
    media_sha256 = sha256_file(expected_media_path)
    expected_validation = {
        "decode_verified": True,
        "height": int(generation["height"]),
        "media_type": "image/png",
        "mode": "RGB",
        "path": str(expected_media_path),
        "sha256": media_sha256,
        "size_bytes": expected_media_path.stat().st_size,
        "width": int(generation["width"]),
    }
    if validation != expected_validation:
        raise ValueError(f"Result media validation drifted at index {job_index}.")

    identity_path = output_dir / "execution_identity.json"
    identity = read_execution_identity(identity_path)
    expected_task_id = str(registry_entry["slurm_task_id"])
    if {
        "SLURM_ARRAY_JOB_ID": identity.get("SLURM_ARRAY_JOB_ID"),
        "SLURM_ARRAY_TASK_ID": identity.get("SLURM_ARRAY_TASK_ID"),
        "SLURM_JOB_NAME": identity.get("SLURM_JOB_NAME"),
        "slurm_task_id": identity.get("slurm_task_id"),
    } != {
        "SLURM_ARRAY_JOB_ID": str(registry_entry["slurm_array_job_id"]),
        "SLURM_ARRAY_TASK_ID": str(job_index),
        "SLURM_JOB_NAME": "flux1-neg-cal",
        "slurm_task_id": expected_task_id,
    }:
        raise ValueError(f"Execution identity differs from registry at index {job_index}.")

    experiment_path = output_dir / "experiment_timing.json"
    run_path = output_dir / "run_timing.json"
    sample_path = output_dir / "sample_0000" / "timing.json"
    _, experiment_wall = _validate_timing(
        experiment_path,
        label="experiment timing",
        job=job,
        manifest_sha256=manifest_sha256,
    )
    run_timing, run_total = _validate_timing(
        run_path,
        label="run timing",
        job=job,
        manifest_sha256=manifest_sha256,
    )
    _, sample_total = _validate_timing(
        sample_path,
        label="sample timing",
        job=job,
        manifest_sha256=manifest_sha256,
    )
    load_seconds = run_timing.get("adapter_load_seconds")
    if load_seconds is None:
        load_seconds = run_timing.get("pipeline_load_seconds")
    load_seconds = _finite_positive(load_seconds, label="run timing load seconds")

    row = job["calibration_row"]
    return {
        "job_index": job_index,
        "condition_id": job["condition_id"],
        "prompt_id": job["prompt_id"],
        "prompt": job["prompt"],
        "role": row["role"],
        "true_cfg_scale": row["true_cfg_scale"],
        "eligible_for_scale_selection": row["eligible_for_scale_selection"],
        "blind_id": row["blind_id"],
        "image_path": str(expected_media_path),
        "image_sha256": media_sha256,
        "image_size_bytes": expected_media_path.stat().st_size,
        "width": int(generation["width"]),
        "height": int(generation["height"]),
        "rgb": rgb,
        "statistics": image_statistics(rgb),
        "runtime_seconds": {
            "experiment_wall": experiment_wall,
            "run_total": run_total,
            "sample_total": sample_total,
            "model_load": load_seconds,
        },
        "artifact_bindings": {
            "benchmark_job_result": _binding(result_path),
            "image": _binding(expected_media_path),
            "execution_identity": _binding(identity_path),
            "experiment_timing": _binding(experiment_path),
            "run_timing": _binding(run_path),
            "sample_timing": _binding(sample_path),
        },
    }


def collect_calibration_artifacts(
    manifest_path: str | Path,
    registry_path: str | Path,
    *,
    root: str | Path,
) -> dict[str, Any]:
    """Authenticate the immutable 45-row manifest, registry, results, and timings."""

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
        or registry.get("num_registered_tasks") != EXPECTED_ROWS
        or [entry.get("job_index") for entry in registry.get("submissions", ())]
        != expected_indices
    ):
        raise ValueError("Calibration submission registry does not cover exact indices 0..44.")
    registry_by_index = {
        int(entry["job_index"]): entry for entry in registry["submissions"]
    }
    rows = [
        _validate_result_row(
            job=job,
            job_index=index,
            manifest_sha256=manifest_sha256,
            registry_entry=registry_by_index[index],
        )
        for index, job in enumerate(manifest["jobs"])
    ]
    if len({row["image_path"] for row in rows}) != EXPECTED_ROWS:
        raise ValueError("Calibration rows do not bind 45 distinct media paths.")
    return {
        "manifest": manifest,
        "manifest_path": resolved_manifest,
        "registry": registry,
        "registry_path": resolved_registry,
        "rows": rows,
    }


def build_objective_report(
    collected: Mapping[str, Any], *, validate: bool = True
) -> dict[str, Any]:
    manifest = collected["manifest"]
    rows = list(collected["rows"])
    by_prompt_role_scale = {
        (row["prompt_id"], row["role"], row["true_cfg_scale"]): row for row in rows
    }
    sentinel_rows: list[dict[str, Any]] = []
    all_sentinels_pass = True
    for prompt_id in PROMPT_IDS:
        control = by_prompt_role_scale[(prompt_id, ROLE_CONTROL, 1.0)]
        sentinel = by_prompt_role_scale[(prompt_id, ROLE_LADDER, 1.0)]
        equal = bool(np.array_equal(control["rgb"], sentinel["rgb"]))
        difference = np.abs(control["rgb"].astype(np.int16) - sentinel["rgb"].astype(np.int16))
        record = {
            "prompt_id": prompt_id,
            "control_blind_id": control["blind_id"],
            "scale_1_string_blind_id": sentinel["blind_id"],
            "exact_decoded_rgb_equality": equal,
            "maximum_absolute_channel_difference": int(difference.max()),
            "differing_channel_value_count": int(np.count_nonzero(difference)),
            "control_image_sha256": control["image_sha256"],
            "scale_1_string_image_sha256": sentinel["image_sha256"],
        }
        sentinel_rows.append(record)
        all_sentinels_pass = all_sentinels_pass and equal

    public_rows: list[dict[str, Any]] = []
    for row in rows:
        reference = by_prompt_role_scale[(row["prompt_id"], ROLE_CONTROL, 1.0)]
        objective = None
        if row["role"] == ROLE_LADDER:
            objective = evaluate_noncollapse(row["statistics"], reference["statistics"])
        public_rows.append(
            {
                key: deepcopy(row[key])
                for key in (
                    "job_index",
                    "condition_id",
                    "prompt_id",
                    "prompt",
                    "role",
                    "true_cfg_scale",
                    "eligible_for_scale_selection",
                    "blind_id",
                    "image_path",
                    "image_sha256",
                    "image_size_bytes",
                    "width",
                    "height",
                    "statistics",
                    "runtime_seconds",
                    "artifact_bindings",
                )
            }
            | {"objective_noncollapse": objective}
        )
    objective_by_scale = []
    for scale in ELIGIBLE_SCALES:
        prompt_results = {
            prompt_id: bool(
                by_row["objective_noncollapse"]["all_gates_pass"]
            )
            for prompt_id in PROMPT_IDS
            for by_row in [
                next(
                    row
                    for row in public_rows
                    if row["prompt_id"] == prompt_id
                    and row["role"] == ROLE_LADDER
                    and row["true_cfg_scale"] == scale
                )
            ]
        }
        objective_by_scale.append(
            {
                "true_cfg_scale": scale,
                "pass_by_prompt": prompt_results,
                "passes_all_prompts": all(prompt_results.values()),
            }
        )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evaluation": OBJECTIVE_EVALUATION,
        "status": "complete",
        "created_at_utc": _utc_now(),
        "calibration_id": CALIBRATION_ID,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "source_bindings": {
            "calibration_config": _binding(
                Path(manifest["calibration_inputs"]["calibration_config"]["path"])
            ),
            "manifest": {
                **_binding(collected["manifest_path"]),
                "manifest_sha256": manifest["manifest_sha256"],
            },
            "manifest_sidecar": _binding(
                Path(collected["manifest_path"]).with_suffix(
                    Path(collected["manifest_path"]).suffix + ".sha256"
                )
            ),
            "submission_registry": {
                **_binding(collected["registry_path"]),
                "registry_sha256": collected["registry"]["registry_sha256"],
                "slurm_array_job_id": collected["registry"]["slurm_array_job_id"],
            },
        },
        "evaluator_provenance": {
            "source": _binding(Path(__file__)),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "pillow_version": PILLOW_VERSION,
        },
        "coverage": {
            "expected_rows": EXPECTED_ROWS,
            "authenticated_completed_rows": len(rows),
            "prompt_ids": list(PROMPT_IDS),
            "role_counts": {
                role: sum(row["role"] == role for row in rows)
                for role in (ROLE_OFFICIAL, ROLE_CONTROL, ROLE_LADDER)
            },
            "eligible_scales": list(ELIGIBLE_SCALES),
        },
        "measurement_definition": {
            "decoded_array": "exact uint8 RGB values from verified RGB PNG",
            "luminance": "Rec.709 coefficients on normalized sRGB code values",
            "entropy": "Shannon entropy of a fixed 256-bin [0,1] luminance histogram",
            "total_variation": (
                "mean of horizontal and vertical mean absolute Rec.709-luma differences"
            ),
            "threshold_boundary_policy": {
                "minimum_or_at_least": "inclusive",
                "ratio_interval": "inclusive",
                "below_or_fraction_maximum": "exclusive",
                "undefined_zero_reference_ratio": "fail_closed",
            },
            "ambiguity_disclosure": (
                "The preregistration freezes thresholds but does not state linear-light "
                "conversion, entropy binning, TV discretization, or equality at the words "
                "'below'. These definitions were fixed in evaluator code before selection; "
                "no observed threshold is relaxed."
            ),
        },
        "inert_negative_integrity": {
            "requirement": "exact_decoded_rgb_equality",
            "pass": all_sentinels_pass,
            "prompts": sentinel_rows,
        },
        "objective_gate_summary_by_scale": objective_by_scale,
        "rows": public_rows,
        "selection_status": "blocked_pending_blinded_manual_review",
    }
    report["document_sha256"] = document_digest(report)
    if validate:
        validate_objective_report(report, verify_source_files=True)
    return report


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
    ):
        raise ValueError("Objective calibration report identity/status is invalid.")
    if payload.get("document_sha256") != document_digest(payload):
        raise ValueError("Objective calibration report digest mismatch.")
    coverage = payload.get("coverage") or {}
    rows = payload.get("rows")
    if (
        coverage.get("expected_rows") != EXPECTED_ROWS
        or coverage.get("authenticated_completed_rows") != EXPECTED_ROWS
        or coverage.get("prompt_ids") != list(PROMPT_IDS)
        or coverage.get("eligible_scales") != list(ELIGIBLE_SCALES)
        or not isinstance(rows, list)
        or len(rows) != EXPECTED_ROWS
    ):
        raise ValueError("Objective calibration report coverage is incomplete.")
    identities = {
        (row.get("prompt_id"), row.get("role"), row.get("true_cfg_scale")) for row in rows
    }
    expected = {
        (prompt_id, role, scale)
        for prompt_id in PROMPT_IDS
        for role, scales in (
            (ROLE_OFFICIAL, (None,)),
            (ROLE_CONTROL, (1.0,)),
            (ROLE_LADDER, TRUE_CFG_SCALES),
        )
        for scale in scales
    }
    if identities != expected or len({row.get("blind_id") for row in rows}) != EXPECTED_ROWS:
        raise ValueError("Objective report row identity grid is not exact.")
    integrity = payload.get("inert_negative_integrity") or {}
    if (
        integrity.get("requirement") != "exact_decoded_rgb_equality"
        or not isinstance(integrity.get("pass"), bool)
        or len(integrity.get("prompts") or ()) != len(PROMPT_IDS)
    ):
        raise ValueError("Objective report scale-1 integrity coverage is malformed.")
    for row in rows:
        if row.get("role") == ROLE_LADDER:
            evaluation = row.get("objective_noncollapse") or {}
            gates = evaluation.get("gates") or {}
            if set(gates) != {
                "mean_luminance_ratio",
                "median_luminance",
                "dark_fraction_luminance_lt_0p05",
                "entropy_bits",
                "total_variation_ratio",
                "bright_fraction_luminance_gt_0p95",
            } or evaluation.get("all_gates_pass") != all(
                item.get("pass") is True for item in gates.values()
            ):
                raise ValueError("Objective non-collapse gate coverage/result is malformed.")
        elif row.get("objective_noncollapse") is not None:
            raise ValueError("Only native-negative ladder rows may carry objective gates.")
    if verify_source_files:
        for binding in (payload.get("source_bindings") or {}).values():
            path = Path(str(binding.get("path", ""))).expanduser().resolve()
            if (
                not path.is_file()
                or sha256_file(path) != binding.get("sha256")
                or path.stat().st_size != binding.get("size_bytes")
            ):
                raise ValueError(f"Objective source binding changed: {path}.")
        for row in rows:
            for binding in (row.get("artifact_bindings") or {}).values():
                path = Path(str(binding.get("path", ""))).expanduser().resolve()
                if (
                    not path.is_file()
                    or sha256_file(path) != binding.get("sha256")
                    or path.stat().st_size != binding.get("size_bytes")
                ):
                    raise ValueError(f"Objective row artifact changed: {path}.")
        # Re-run the complete collector from the bound immutable manifest and
        # submission registry, then independently regenerate every non-time
        # field.  This prevents a digest-recomputed report from changing a row
        # role/scale, runtime, statistic, sentinel, threshold, or gate verdict.
        bindings = payload["source_bindings"]
        collected = collect_calibration_artifacts(
            bindings["manifest"]["path"],
            bindings["submission_registry"]["path"],
            root=Path(__file__).resolve().parents[3],
        )
        expected = build_objective_report(collected, validate=False)
        immutable_fields = (
            "schema_version",
            "evaluation",
            "status",
            "calibration_id",
            "model_name",
            "model_revision",
            "source_bindings",
            "evaluator_provenance",
            "coverage",
            "measurement_definition",
            "inert_negative_integrity",
            "objective_gate_summary_by_scale",
            "rows",
            "selection_status",
        )
        drift = [field for field in immutable_fields if payload.get(field) != expected[field]]
        if drift:
            raise ValueError(
                "Objective calibration report differs from recomputed source evidence: "
                f"{drift}."
            )
    return {
        "status": "complete",
        "document_sha256": payload["document_sha256"],
        "row_count": len(rows),
        "integrity_pass": integrity["pass"],
    }


def _blind_sort_key(manifest_sha256: str, blind_id: str, section: str) -> str:
    return hashlib.sha256(
        f"{manifest_sha256}|{section}|{blind_id}|v1".encode("utf-8")
    ).hexdigest()


def _copy_blinded_images(rows: Sequence[Mapping[str, Any]], review_root: Path) -> None:
    image_root = review_root / "images"
    image_root.mkdir(parents=True, exist_ok=False)
    for row in rows:
        source = Path(str(row["image_path"])).resolve()
        destination = image_root / f"{row['blind_id']}.png"
        shutil.copyfile(source, destination)
        if sha256_file(destination) != row["image_sha256"]:
            raise RuntimeError(f"Blinded image copy changed decoded source bytes: {destination}")


def _contact_sheet(
    rows: Sequence[Mapping[str, Any]],
    destination: Path,
    *,
    title: str,
    columns: int = 4,
) -> None:
    thumb_size = (300, 300)
    label_height = 46
    margin = 20
    rows_count = math.ceil(len(rows) / columns)
    sheet = Image.new(
        "RGB",
        (
            margin + columns * (thumb_size[0] + margin),
            70 + rows_count * (thumb_size[1] + label_height + margin),
        ),
        "white",
    )
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    draw.text((margin, 20), title, fill="black", font=font)
    for index, row in enumerate(rows):
        x = margin + (index % columns) * (thumb_size[0] + margin)
        y = 70 + (index // columns) * (thumb_size[1] + label_height + margin)
        with Image.open(row["image_path"]) as source:
            thumbnail = ImageOps.contain(source.convert("RGB"), thumb_size)
        image_x = x + (thumb_size[0] - thumbnail.width) // 2
        image_y = y + (thumb_size[1] - thumbnail.height) // 2
        sheet.paste(thumbnail, (image_x, image_y))
        draw.text((x, y + thumb_size[1] + 8), str(row["blind_id"]), fill="black", font=font)
    destination.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(destination, format="PNG")


def _review_row(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "blind_id": row["blind_id"],
        "prompt_id": row["prompt_id"],
        "image_relative_path": f"images/{row['blind_id']}.png",
        "image_sha256": row["image_sha256"],
        "width": row["width"],
        "height": row["height"],
        "hard_gate_ids": list(_hard_gate_ids(str(row["prompt_id"]))),
        "source_concept_ids": list(SOURCE_CONCEPTS[str(row["prompt_id"])]),
        "positive_target_concepts_scored": False,
    }


def build_review_documents(
    objective: Mapping[str, Any],
    *,
    objective_path: Path,
    review_root: Path,
    final_review_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    rows = list(objective["rows"])
    manifest_sha256 = objective["source_bindings"]["manifest"]["manifest_sha256"]
    _copy_blinded_images(rows, review_root)
    review_rows = sorted(
        (_review_row(row) for row in rows),
        key=lambda item: _blind_sort_key(manifest_sha256, item["blind_id"], "all_rows"),
    )
    parity_pairs = []
    ladder_groups = []
    sheet_records = []
    for prompt_id in PROMPT_IDS:
        parity = [
            row
            for row in rows
            if row["prompt_id"] == prompt_id
            and row["role"] in {ROLE_OFFICIAL, ROLE_CONTROL}
        ]
        parity.sort(
            key=lambda item: _blind_sort_key(
                manifest_sha256, item["blind_id"], f"parity|{prompt_id}"
            )
        )
        pair_id = "parity_" + hashlib.sha256(
            f"{manifest_sha256}|{prompt_id}|path_parity".encode()
        ).hexdigest()[:16]
        parity_pairs.append(
            {
                "pair_id": pair_id,
                "prompt_id": prompt_id,
                "blind_ids": [row["blind_id"] for row in parity],
                "category_ids": list(PATH_PARITY_CATEGORIES),
            }
        )
        parity_sheet = review_root / "contact_sheets" / f"{pair_id}.png"
        _contact_sheet(
            parity,
            parity_sheet,
            title=f"{prompt_id}: blinded path-parity controls (A/B order randomized)",
            columns=2,
        )
        sheet_records.append(
            {
                "section": "path_parity",
                "prompt_id": prompt_id,
                "relative_path": str(parity_sheet.relative_to(review_root)),
                "sha256": sha256_file(parity_sheet),
            }
        )

        ladder = [
            row
            for row in rows
            if row["prompt_id"] == prompt_id and row["role"] == ROLE_LADDER
        ]
        ladder.sort(
            key=lambda item: _blind_sort_key(
                manifest_sha256, item["blind_id"], f"ladder|{prompt_id}"
            )
        )
        group_id = "candidate_" + hashlib.sha256(
            f"{manifest_sha256}|{prompt_id}|ladder".encode()
        ).hexdigest()[:16]
        ladder_groups.append(
            {
                "group_id": group_id,
                "prompt_id": prompt_id,
                "blind_ids": [row["blind_id"] for row in ladder],
                "source_concept_ids": list(SOURCE_CONCEPTS[prompt_id]),
                "positive_target_concepts_scored": False,
            }
        )
        candidate_sheet = review_root / "contact_sheets" / f"{group_id}.png"
        _contact_sheet(
            ladder,
            candidate_sheet,
            title=f"{prompt_id}: blinded native-negative ladder (order randomized)",
        )
        sheet_records.append(
            {
                "section": "native_negative_ladder",
                "prompt_id": prompt_id,
                "relative_path": str(candidate_sheet.relative_to(review_root)),
                "sha256": sha256_file(candidate_sheet),
            }
        )

    reviewer_manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "review_package": REVIEW_PACKAGE,
        "status": "prepared_not_reviewed",
        "created_at_utc": _utc_now(),
        "calibration_id": CALIBRATION_ID,
        "objective_report_document_sha256": objective["document_sha256"],
        "blinding_policy": {
            "roles_hidden": True,
            "true_cfg_scales_hidden": True,
            "source_result_paths_hidden": True,
            "positive_target_concepts_absent": True,
            "contact_sheets_for_navigation_only": True,
            "individual_pngs_require_original_resolution_review": True,
        },
        "row_count": EXPECTED_ROWS,
        "source_prompts": {
            prompt_id: next(
                str(row["prompt"]) for row in rows if row["prompt_id"] == prompt_id
            )
            for prompt_id in PROMPT_IDS
        },
        "source_concepts": {
            prompt_id: list(SOURCE_CONCEPTS[prompt_id]) for prompt_id in PROMPT_IDS
        },
        "rows": review_rows,
        "path_parity_pairs": parity_pairs,
        "ladder_candidate_groups": ladder_groups,
        "contact_sheets": sheet_records,
    }
    reviewer_manifest["document_sha256"] = document_digest(reviewer_manifest)

    unblinding: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "unblinding_map": UNBLINDING_MAP,
        "status": "sealed_for_post_review_use_only",
        "created_at_utc": _utc_now(),
        "calibration_id": CALIBRATION_ID,
        "objective_report_document_sha256": objective["document_sha256"],
        "reviewer_manifest_document_sha256": reviewer_manifest["document_sha256"],
        "rows": [
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
                "review_image_path": str(
                    final_review_root / "images" / f"{row['blind_id']}.png"
                ),
            }
            for row in sorted(rows, key=lambda item: item["job_index"])
        ],
    }
    unblinding["document_sha256"] = document_digest(unblinding)

    objective_binding = {
        **_binding(objective_path),
        "document_sha256": objective["document_sha256"],
    }
    reviewer_manifest_path = review_root / "reviewer_manifest.json"
    _write_document_unchecked(reviewer_manifest_path, reviewer_manifest)
    review_binding = {
        **_binding(reviewer_manifest_path),
        "path": str((final_review_root / "reviewer_manifest.json").resolve()),
        "document_sha256": reviewer_manifest["document_sha256"],
    }
    template: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "review": MANUAL_REVIEW,
        "status": "draft_not_reviewed",
        "reviewer": None,
        "reviewed_at_utc": None,
        "source_bindings": {
            "objective_report": objective_binding,
            "reviewer_manifest": review_binding,
        },
        "viewing_attestation": {
            "all_45_blinded_images_opened_at_original_resolution": None,
            "path_parity_pairs_compared_side_by_side": None,
            "contact_sheets_used_for_navigation_only": None,
            "roles_or_scales_revealed_during_review": None,
            "positive_target_concepts_scored": None,
        },
        "row_reviews": [
            {
                "blind_id": row["blind_id"],
                "prompt_id": row["prompt_id"],
                "hard_gates": {gate_id: None for gate_id in row["hard_gate_ids"]},
                "source_concepts": {
                    concept_id: None for concept_id in row["source_concept_ids"]
                },
                "notes": "",
            }
            for row in review_rows
        ],
        "path_parity_reviews": [
            {
                "pair_id": pair["pair_id"],
                "prompt_id": pair["prompt_id"],
                "blind_ids": pair["blind_ids"],
                "material_difference_categories": {
                    category: None for category in PATH_PARITY_CATEGORIES
                },
                "notes": "",
            }
            for pair in parity_pairs
        ],
        "review_instructions": {
            "row_hard_gate_values": sorted(ROW_VERDICTS),
            "source_concept_values": sorted(SOURCE_VERDICTS),
            "path_parity_values": sorted(PARITY_VERDICTS),
            "indeterminate_is_fail_closed": True,
            "minimum_suppressed_sources_per_prompt": 2,
            "positive_target_concepts_must_not_be_added_or_scored": True,
        },
    }
    return reviewer_manifest, unblinding, template


def _write_document_unchecked(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_document_new(path: Path, payload: Mapping[str, Any]) -> tuple[Path, Path]:
    if payload.get("document_sha256") != document_digest(payload):
        raise ValueError("Cannot write an immutable document with an invalid digest.")
    path = path.expanduser().resolve()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite immutable evidence: {path}")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        sidecar.write_text(f"{payload['document_sha256']}  {path.name}\n", encoding="utf-8")
        sidecar.chmod(0o444)
    except BaseException:
        path.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    return path, sidecar


def prepare_evaluation_package(
    *,
    manifest_path: str | Path,
    registry_path: str | Path,
    output_root: str | Path,
    root: str | Path,
) -> dict[str, Any]:
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation package: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.tmp-", dir=destination.parent)
    )
    try:
        collected = collect_calibration_artifacts(
            manifest_path,
            registry_path,
            root=root,
        )
        objective = build_objective_report(collected)
        objective_path = temporary / "objective_report.json"
        _write_document_unchecked(objective_path, objective)
        objective_path.with_suffix(".json.sha256").write_text(
            f"{objective['document_sha256']}  {objective_path.name}\n", encoding="utf-8"
        )
        review_root = temporary / "review"
        review_root.mkdir()
        reviewer, unblinding, template = build_review_documents(
            objective,
            objective_path=objective_path,
            review_root=review_root,
            final_review_root=destination / "review",
        )
        # Replace the temporary path in the template with the final immutable path.
        objective_binding = template["source_bindings"]["objective_report"]
        objective_binding["path"] = str((destination / "objective_report.json").resolve())
        template_path = temporary / "manual_review_template.json"
        _write_document_unchecked(template_path, template)
        unblinding_path = temporary / "unblinding_map.json"
        _write_document_unchecked(unblinding_path, unblinding)
        unblinding_path.with_suffix(".json.sha256").write_text(
            f"{unblinding['document_sha256']}  {unblinding_path.name}\n", encoding="utf-8"
        )
        readme = temporary / "README.md"
        readme.write_text(
            "# Blinded Flux.1 native-negative calibration review\n\n"
            "Open every PNG in `review/images/` at original resolution. Contact sheets are "
            "navigation aids only. Complete a writable copy of `manual_review_template.json`; "
            "do not inspect `unblinding_map.json` until all verdicts and the blinding "
            "attestation are final. Score only the listed source concepts. Positive target "
            "concepts are outside this native-negative calibration and must not be scored.\n",
            encoding="utf-8",
        )
        # The reviewer manifest was written before its sidecar was available.
        (review_root / "reviewer_manifest.json.sha256").write_text(
            f"{reviewer['document_sha256']}  reviewer_manifest.json\n", encoding="utf-8"
        )
        for file_path in temporary.rglob("*"):
            if file_path.is_file():
                file_path.chmod(0o444)
        for directory in sorted(
            (path for path in temporary.rglob("*") if path.is_dir()), reverse=True
        ):
            directory.chmod(0o555)
        temporary.chmod(0o555)
        os.replace(temporary, destination)
    except BaseException:
        try:
            temporary.chmod(0o755)
            for directory in temporary.rglob("*"):
                if directory.is_dir():
                    directory.chmod(0o755)
        except OSError:
            pass
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    return {
        "status": "prepared_pending_manual_review",
        "output_root": str(destination),
        "objective_report": str(destination / "objective_report.json"),
        "objective_report_document_sha256": objective["document_sha256"],
        "reviewer_manifest": str(destination / "review" / "reviewer_manifest.json"),
        "reviewer_manifest_document_sha256": reviewer["document_sha256"],
        "manual_review_template": str(destination / "manual_review_template.json"),
        "unblinding_map": str(destination / "unblinding_map.json"),
        "unblinding_map_document_sha256": unblinding["document_sha256"],
        "inert_negative_integrity_pass": objective["inert_negative_integrity"]["pass"],
    }


def _validate_document_file(
    path: Path,
    *,
    name_field: str,
    expected_name: str,
) -> dict[str, Any]:
    payload = _load_json(path, label=expected_name)
    if payload.get(name_field) != expected_name:
        raise ValueError(f"Document {path} is not {expected_name!r}.")
    if payload.get("document_sha256") != document_digest(payload):
        raise ValueError(f"Document digest mismatch: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    fields = sidecar.read_text(encoding="utf-8").split() if sidecar.is_file() else []
    if not fields or fields[0] != payload["document_sha256"]:
        raise ValueError(f"Document sidecar is missing or inconsistent: {path}")
    return payload


def _validate_review_and_unblinding(
    reviewer: Mapping[str, Any], unblinding: Mapping[str, Any]
) -> None:
    if (
        reviewer.get("schema_version") != SCHEMA_VERSION
        or reviewer.get("review_package") != REVIEW_PACKAGE
        or reviewer.get("status") != "prepared_not_reviewed"
        or reviewer.get("document_sha256") != document_digest(reviewer)
        or reviewer.get("row_count") != EXPECTED_ROWS
        or len(reviewer.get("rows") or ()) != EXPECTED_ROWS
    ):
        raise ValueError("Reviewer manifest identity, digest, or coverage is invalid.")
    policy = reviewer.get("blinding_policy") or {}
    if policy != {
        "roles_hidden": True,
        "true_cfg_scales_hidden": True,
        "source_result_paths_hidden": True,
        "positive_target_concepts_absent": True,
        "contact_sheets_for_navigation_only": True,
        "individual_pngs_require_original_resolution_review": True,
    }:
        raise ValueError("Reviewer manifest blinding policy drifted.")
    def recursive_keys(value: Any) -> set[str]:
        if isinstance(value, Mapping):
            return set(map(str, value)) | set().union(
                *(recursive_keys(item) for item in value.values())
            )
        if isinstance(value, list):
            return set().union(*(recursive_keys(item) for item in value))
        return set()

    reviewer_keys = recursive_keys(reviewer)
    for forbidden in ("true_cfg_scale", "eligible_for_scale_selection", "source_image_path"):
        if forbidden in reviewer_keys:
            raise ValueError(f"Reviewer manifest leaks unblinding field {forbidden!r}.")
    if (
        unblinding.get("schema_version") != SCHEMA_VERSION
        or unblinding.get("unblinding_map") != UNBLINDING_MAP
        or unblinding.get("status") != "sealed_for_post_review_use_only"
        or unblinding.get("document_sha256") != document_digest(unblinding)
        or unblinding.get("reviewer_manifest_document_sha256")
        != reviewer.get("document_sha256")
        or len(unblinding.get("rows") or ()) != EXPECTED_ROWS
    ):
        raise ValueError("Unblinding map identity, digest, or coverage is invalid.")
    review_ids = {row["blind_id"] for row in reviewer["rows"]}
    unblind_ids = {row["blind_id"] for row in unblinding["rows"]}
    if review_ids != unblind_ids or len(review_ids) != EXPECTED_ROWS:
        raise ValueError("Reviewer and unblinding blind-ID sets differ.")


def normalize_manual_ledger(
    draft: Mapping[str, Any],
    *,
    objective: Mapping[str, Any],
    objective_path: Path,
    reviewer: Mapping[str, Any],
    reviewer_path: Path,
    unblinding: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a completed blinded draft and return a digest-bearing ledger."""

    if draft.get("schema_version") != SCHEMA_VERSION or draft.get("review") != MANUAL_REVIEW:
        raise ValueError("Manual review draft identity is invalid.")
    if draft.get("status") not in {"draft_not_reviewed", "complete"}:
        raise ValueError("Manual review draft has an invalid status.")
    reviewer_name = str(draft.get("reviewer") or "").strip()
    reviewed_at = str(draft.get("reviewed_at_utc") or "").strip()
    if not reviewer_name or not reviewed_at:
        raise ValueError("Completed manual review requires reviewer and reviewed_at_utc.")
    expected_bindings = {
        "objective_report": {
            **_binding(objective_path),
            "document_sha256": objective["document_sha256"],
        },
        "reviewer_manifest": {
            **_binding(reviewer_path),
            "document_sha256": reviewer["document_sha256"],
        },
    }
    if draft.get("source_bindings") != expected_bindings:
        raise ValueError("Manual review source bindings differ from objective/review evidence.")
    expected_attestation = {
        "all_45_blinded_images_opened_at_original_resolution": True,
        "path_parity_pairs_compared_side_by_side": True,
        "contact_sheets_used_for_navigation_only": True,
        "roles_or_scales_revealed_during_review": False,
        "positive_target_concepts_scored": False,
    }
    if draft.get("viewing_attestation") != expected_attestation:
        raise ValueError("Manual review viewing/blinding attestation is incomplete.")

    review_rows = {row["blind_id"]: row for row in reviewer["rows"]}
    draft_rows = draft.get("row_reviews")
    if not isinstance(draft_rows, list) or len(draft_rows) != EXPECTED_ROWS:
        raise ValueError("Manual review must contain exactly 45 row reviews.")
    normalized_rows = []
    seen: set[str] = set()
    for raw in draft_rows:
        if not isinstance(raw, Mapping):
            raise ValueError("Manual row review must be an object.")
        allowed = {"blind_id", "prompt_id", "hard_gates", "source_concepts", "notes"}
        if set(raw) != allowed:
            raise ValueError("Manual row review fields drifted or include target scoring.")
        blind_id = str(raw.get("blind_id", ""))
        if blind_id in seen or blind_id not in review_rows:
            raise ValueError(f"Manual row blind ID is duplicate/unknown: {blind_id!r}.")
        seen.add(blind_id)
        review_row = review_rows[blind_id]
        if raw.get("prompt_id") != review_row["prompt_id"]:
            raise ValueError(f"Manual row prompt differs for {blind_id}.")
        hard = raw.get("hard_gates")
        sources = raw.get("source_concepts")
        if not isinstance(hard, Mapping) or set(hard) != set(review_row["hard_gate_ids"]):
            raise ValueError(f"Manual hard-gate coverage differs for {blind_id}.")
        if any(value not in ROW_VERDICTS for value in hard.values()):
            raise ValueError(f"Manual hard-gate verdict is invalid for {blind_id}.")
        if not isinstance(sources, Mapping) or set(sources) != set(
            review_row["source_concept_ids"]
        ):
            raise ValueError(f"Manual source-concept coverage differs for {blind_id}.")
        if any(value not in SOURCE_VERDICTS for value in sources.values()):
            raise ValueError(f"Manual source-concept verdict is invalid for {blind_id}.")
        normalized_rows.append(
            {
                "blind_id": blind_id,
                "prompt_id": raw["prompt_id"],
                "hard_gates": {key: str(hard[key]) for key in review_row["hard_gate_ids"]},
                "source_concepts": {
                    key: str(sources[key]) for key in review_row["source_concept_ids"]
                },
                "notes": str(raw.get("notes", "")),
            }
        )
    if seen != set(review_rows):
        raise ValueError("Manual row review blind-ID coverage is incomplete.")

    expected_pairs = {pair["pair_id"]: pair for pair in reviewer["path_parity_pairs"]}
    raw_pairs = draft.get("path_parity_reviews")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != len(PROMPT_IDS):
        raise ValueError("Manual review must contain exactly three path-parity reviews.")
    normalized_pairs = []
    seen_pairs: set[str] = set()
    for raw in raw_pairs:
        if not isinstance(raw, Mapping) or set(raw) != {
            "pair_id",
            "prompt_id",
            "blind_ids",
            "material_difference_categories",
            "notes",
        }:
            raise ValueError("Manual path-parity review fields drifted.")
        pair_id = str(raw.get("pair_id", ""))
        if pair_id in seen_pairs or pair_id not in expected_pairs:
            raise ValueError(f"Manual path-parity pair is duplicate/unknown: {pair_id!r}.")
        seen_pairs.add(pair_id)
        expected_pair = expected_pairs[pair_id]
        if (
            raw.get("prompt_id") != expected_pair["prompt_id"]
            or raw.get("blind_ids") != expected_pair["blind_ids"]
        ):
            raise ValueError(f"Manual path-parity identity differs for {pair_id}.")
        categories = raw.get("material_difference_categories")
        if not isinstance(categories, Mapping) or set(categories) != set(
            PATH_PARITY_CATEGORIES
        ):
            raise ValueError(f"Manual path-parity category coverage differs for {pair_id}.")
        if any(value not in PARITY_VERDICTS for value in categories.values()):
            raise ValueError(f"Manual path-parity verdict is invalid for {pair_id}.")
        normalized_pairs.append(
            {
                "pair_id": pair_id,
                "prompt_id": raw["prompt_id"],
                "blind_ids": list(raw["blind_ids"]),
                "material_difference_categories": {
                    key: str(categories[key]) for key in PATH_PARITY_CATEGORIES
                },
                "notes": str(raw.get("notes", "")),
            }
        )
    if seen_pairs != set(expected_pairs):
        raise ValueError("Manual path-parity review coverage is incomplete.")

    ledger: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "review": MANUAL_REVIEW,
        "status": "complete",
        "reviewer": reviewer_name,
        "reviewed_at_utc": reviewed_at,
        "sealed_at_utc": _utc_now(),
        "source_bindings": expected_bindings,
        "viewing_attestation": expected_attestation,
        "semantic_policy": {
            "positive_target_concepts_scored": False,
            "indeterminate_is_fail_closed": True,
            "minimum_suppressed_source_concepts_per_prompt": 2,
        },
        "row_reviews": sorted(normalized_rows, key=lambda row: row["blind_id"]),
        "path_parity_reviews": sorted(
            normalized_pairs, key=lambda row: row["prompt_id"]
        ),
        "unblinding_document_sha256_after_review": unblinding["document_sha256"],
    }
    ledger["document_sha256"] = document_digest(ledger)
    return ledger


def build_selection_report(
    *,
    objective: Mapping[str, Any],
    objective_path: Path,
    reviewer: Mapping[str, Any],
    reviewer_path: Path,
    unblinding: Mapping[str, Any],
    unblinding_path: Path,
    ledger: Mapping[str, Any],
    ledger_path: Path,
) -> dict[str, Any]:
    rows = {row["blind_id"]: row for row in objective["rows"]}
    manual_rows = {row["blind_id"]: row for row in ledger["row_reviews"]}
    unblind_rows = {row["blind_id"]: row for row in unblinding["rows"]}
    if set(rows) != set(manual_rows) or set(rows) != set(unblind_rows):
        raise ValueError("Objective/manual/unblinding blind-ID coverage differs.")

    parity_by_prompt: dict[str, dict[str, Any]] = {}
    for parity in ledger["path_parity_reviews"]:
        prompt_id = parity["prompt_id"]
        pair_ids = parity["blind_ids"]
        roles = {unblind_rows[blind_id]["role"] for blind_id in pair_ids}
        controls_pass_source = all(
            all(value == "pass" for value in manual_rows[blind_id]["hard_gates"].values())
            and all(
                value == "present"
                for value in manual_rows[blind_id]["source_concepts"].values()
            )
            for blind_id in pair_ids
        )
        no_material_difference = all(
            value == "no_material_difference"
            for value in parity["material_difference_categories"].values()
        )
        exact_roles = roles == {ROLE_OFFICIAL, ROLE_CONTROL}
        parity_by_prompt[prompt_id] = {
            "pair_id": parity["pair_id"],
            "blind_ids": pair_ids,
            "exact_official_and_native_control_roles": exact_roles,
            "both_controls_pass_full_source_fidelity": controls_pass_source,
            "no_material_manual_difference": no_material_difference,
            "pass": exact_roles and controls_pass_source and no_material_difference,
        }

    integrity_pass = objective["inert_negative_integrity"]["pass"] is True
    scale_results = []
    passing_scales = []
    for scale in ELIGIBLE_SCALES:
        prompt_results = {}
        for prompt_id in PROMPT_IDS:
            match = [
                row
                for row in unblind_rows.values()
                if row["prompt_id"] == prompt_id
                and row["role"] == ROLE_LADDER
                and row["true_cfg_scale"] == scale
            ]
            if len(match) != 1:
                raise ValueError(f"Unblinding map does not resolve scale {scale}/{prompt_id} once.")
            blind_id = match[0]["blind_id"]
            objective_row = rows[blind_id]
            manual_row = manual_rows[blind_id]
            objective_pass = objective_row["objective_noncollapse"]["all_gates_pass"] is True
            hard_pass = all(value == "pass" for value in manual_row["hard_gates"].values())
            suppressed = sorted(
                concept_id
                for concept_id, verdict in manual_row["source_concepts"].items()
                if verdict == "suppressed"
            )
            suppression_pass = len(suppressed) >= 2
            prompt_pass = (
                integrity_pass
                and parity_by_prompt[prompt_id]["pass"]
                and objective_pass
                and hard_pass
                and suppression_pass
            )
            failures = []
            if not integrity_pass:
                failures.append("inert_negative_integrity_failed")
            if not parity_by_prompt[prompt_id]["pass"]:
                failures.append("path_parity_or_control_source_fidelity_failed")
            if not objective_pass:
                failures.append("objective_noncollapse_failed")
            if not hard_pass:
                failures.append("manual_hard_gate_failed_or_indeterminate")
            if not suppression_pass:
                failures.append("fewer_than_two_unambiguously_suppressed_sources")
            prompt_results[prompt_id] = {
                "blind_id": blind_id,
                "objective_noncollapse_pass": objective_pass,
                "manual_hard_gates_pass": hard_pass,
                "suppressed_source_concepts": suppressed,
                "suppressed_source_concept_count": len(suppressed),
                "source_suppression_pass": suppression_pass,
                "pass": prompt_pass,
                "failure_reasons": failures,
            }
        all_prompts_pass = all(item["pass"] for item in prompt_results.values())
        scale_results.append(
            {
                "true_cfg_scale": scale,
                "prompts": prompt_results,
                "passes_all_preregistered_gates_on_all_prompts": all_prompts_pass,
            }
        )
        if all_prompts_pass:
            passing_scales.append(scale)

    selected = min(passing_scales) if passing_scales else None
    status = "selected" if selected is not None else "no_selection"
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "selection_report": SELECTION_REPORT,
        "status": status,
        "created_at_utc": _utc_now(),
        "calibration_id": CALIBRATION_ID,
        "model_name": MODEL_NAME,
        "model_revision": MODEL_REVISION,
        "source_bindings": {
            "calibration_config": {
                **objective["source_bindings"]["calibration_config"],
                "calibration_config_sha256": CALIBRATION_CONFIG_SHA256,
            },
            "calibration_manifest": objective["source_bindings"]["manifest"],
            "objective_report": {
                **_binding(objective_path),
                "document_sha256": objective["document_sha256"],
            },
            "reviewer_manifest": {
                **_binding(reviewer_path),
                "document_sha256": reviewer["document_sha256"],
            },
            "unblinding_map": {
                **_binding(unblinding_path),
                "document_sha256": unblinding["document_sha256"],
            },
            "blinded_manual_review_ledger": {
                **_binding(ledger_path),
                "document_sha256": ledger["document_sha256"],
            },
        },
        "global_prerequisites": {
            "inert_negative_integrity_pass": integrity_pass,
            "path_parity_by_prompt": parity_by_prompt,
            "all_path_parity_prompts_pass": all(
                item["pass"] for item in parity_by_prompt.values()
            ),
            "positive_target_concepts_scored": False,
        },
        "scale_results": scale_results,
        "passing_scales": passing_scales,
        "selected_true_cfg_scale": selected,
        "selection_rule": {
            "one_global_scale_across_all_three_prompts": True,
            "smallest_globally_passing_scale": True,
            "per_prompt_tuning": False,
            "averaging_across_prompts": False,
            "threshold_relaxation_after_viewing": False,
            "least_bad_selection": False,
        },
        "normal_benchmark_update_authorized": selected is not None,
    }
    report["document_sha256"] = document_digest(report)
    return report


def finalize_evaluation(
    *,
    objective_path: str | Path,
    reviewer_manifest_path: str | Path,
    unblinding_map_path: str | Path,
    manual_draft_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    objective_path = Path(objective_path).expanduser().resolve()
    reviewer_path = Path(reviewer_manifest_path).expanduser().resolve()
    unblinding_path = Path(unblinding_map_path).expanduser().resolve()
    draft_path = Path(manual_draft_path).expanduser().resolve()
    destination = Path(output_root).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite final calibration evidence: {destination}")
    objective = _validate_document_file(
        objective_path,
        name_field="evaluation",
        expected_name=OBJECTIVE_EVALUATION,
    )
    validate_objective_report(objective, verify_source_files=True)
    reviewer = _validate_document_file(
        reviewer_path,
        name_field="review_package",
        expected_name=REVIEW_PACKAGE,
    )
    unblinding = _validate_document_file(
        unblinding_path,
        name_field="unblinding_map",
        expected_name=UNBLINDING_MAP,
    )
    _validate_review_and_unblinding(reviewer, unblinding)
    draft = _load_json(draft_path, label="manual review draft")
    ledger = normalize_manual_ledger(
        draft,
        objective=objective,
        objective_path=objective_path,
        reviewer=reviewer,
        reviewer_path=reviewer_path,
        unblinding=unblinding,
    )
    destination.mkdir(parents=True, exist_ok=False)
    try:
        ledger_path, _ = _write_document_new(destination / "manual_review_ledger.json", ledger)
        selection = build_selection_report(
            objective=objective,
            objective_path=objective_path,
            reviewer=reviewer,
            reviewer_path=reviewer_path,
            unblinding=unblinding,
            unblinding_path=unblinding_path,
            ledger=ledger,
            ledger_path=ledger_path,
        )
        selection_path, _ = _write_document_new(
            destination / "selection_report.json", selection
        )
        for path in destination.iterdir():
            path.chmod(0o444)
        destination.chmod(0o555)
    except BaseException:
        destination.chmod(0o755)
        shutil.rmtree(destination, ignore_errors=True)
        raise
    return {
        "status": selection["status"],
        "selected_true_cfg_scale": selection["selected_true_cfg_scale"],
        "manual_review_ledger": str(ledger_path),
        "manual_review_ledger_document_sha256": ledger["document_sha256"],
        "selection_report": str(selection_path),
        "selection_report_document_sha256": selection["document_sha256"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Authenticate outputs and prepare review.")
    prepare.add_argument("--manifest", required=True)
    prepare.add_argument("--submission-registry", required=True)
    prepare.add_argument("--output-root", required=True)
    prepare.add_argument("--project-root", default=str(Path(__file__).resolve().parents[3]))
    finalize = commands.add_parser("finalize", help="Seal a completed blinded review.")
    finalize.add_argument("--objective-report", required=True)
    finalize.add_argument("--reviewer-manifest", required=True)
    finalize.add_argument("--unblinding-map", required=True)
    finalize.add_argument("--manual-draft", required=True)
    finalize.add_argument("--output-root", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare_evaluation_package(
            manifest_path=args.manifest,
            registry_path=args.submission_registry,
            output_root=args.output_root,
            root=args.project_root,
        )
    else:
        result = finalize_evaluation(
            objective_path=args.objective_report,
            reviewer_manifest_path=args.reviewer_manifest,
            unblinding_map_path=args.unblinding_map,
            manual_draft_path=args.manual_draft,
            output_root=args.output_root,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":  # pragma: no cover
    main()
