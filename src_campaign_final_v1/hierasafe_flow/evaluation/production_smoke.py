"""Evidence-derived gates for finer-detailing production-shaped smokes.

Gate decisions are never accepted as caller claims.  The evaluator reopens an
exact smoke plan, derives its required row set, authenticates every fixed
attempt artifact and external structured ledger, and computes pass/fail from
registered metrics/decisions.  The resulting report stores the complete
derived proof and is re-evaluated whenever a later plan reopens it.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from hierasafe_flow.benchmarks import finer_detailing_correction as finer
from hierasafe_flow.benchmarks.finer_detailing_correction import BENCHMARK_NAME
from hierasafe_flow.benchmarks.finer_detailing_production_smoke import (
    EXACT_ONE_PATH_GATE,
    FINAL_CUMULATIVE_ADMISSION_GATE,
    FULL_PAIR_NON_REGRESSION_GATE,
    GATE_CONTRACTS,
    IDEOGRAM_CONDITIONING_GATE,
    NATIVE_BASELINE_GATE,
    NO_GENERATION_GATE,
    ORDINARY_FULL,
    ORDINARY_EXACT,
    POST_EXACT_FULL_MODES,
    SHAPLEY_FULL,
    SHAPLEY_EXACT,
    STAGE_SLICES,
    WAN_TRANSITION_GATE,
    ValidatedSmokePlan,
    _sha256_file,
    canonical_sha256,
    _builder_contract_sha256,
    _builder_arguments,
    paths_overlap,
    project_root,
    read_authenticated_document,
    require_external_artifact_path,
    smoke_plan_binding,
    validate_smoke_plan,
    write_authenticated_document,
)
from hierasafe_flow.benchmarks.slurm_tracking import (
    EXECUTION_IDENTITY_FILENAME,
    read_environment_preflight,
)
from hierasafe_flow.evaluation.temporal_metrics import (
    validate_temporal_metric_runtime_receipt,
)


EVIDENCE_INDEX_SCHEMA_VERSION = 2
EVIDENCE_INDEX_CONTRACT = "finer_detailing_smoke_gate_evidence_index_v2"
GATE_EVALUATION_SCHEMA_VERSION = 2
GATE_EVALUATION_CONTRACT = "finer_detailing_smoke_evidence_derived_gate_evaluation_v2"
MANUAL_LEDGER_FILENAME = "smoke_manual_ledger.json"
NO_GENERATION_RAW_EVIDENCE_SCHEMA_VERSION = 1
NO_GENERATION_RAW_EVIDENCE_CONTRACT = "finer_detailing_no_generation_raw_evidence_v1"

_METHOD_KIND_ALIASES: dict[str, str] = {}


def _normalize_variant_kind(kind: str) -> str:
    return _METHOD_KIND_ALIASES.get(kind, kind)

NO_GENERATION_CHECK_IDS = (
    "main_environment_verified_install",
    "ltx_environment_verified_install",
    "wan_real_prompt_cleaner",
    "cog_artifacts_api_channels_precision",
    "hunyuan_artifacts_api_tokens_precision",
    "joy_source_api_memory_layout_precision",
    "ltx_source_api_native_shape_precision",
    "wan_t2v_i2v_artifacts_api_precision",
    "segmented_metric_dependencies_and_weights",
    "peak_load_feasibility",
)
NO_GENERATION_TEST_NODEIDS = {
    "wan_real_prompt_cleaner": (
        "tests/test_wan_adapter.py::test_wan_cleaner_preflight_executes_pinned_module_and_hashes_sentinels",
        "tests/test_native_temporal_cli.py::test_wan_dependency_preflight_is_ordered_before_native_t2v_pipeline_load",
    ),
    "cog_artifacts_api_channels_precision": (
        "tests/test_cogvideox_adapter.py::test_checkpoint_architecture_contracts_reject_channel_drift",
        "tests/test_cogvideox_adapter.py::test_artifact_manifest_digest_is_checked_before_snapshot_resolution",
        "tests/test_native_temporal_cli.py::test_cog_native_cli_invokes_first_call_and_owned_three_segment_completion",
    ),
    "hunyuan_artifacts_api_tokens_precision": (
        "tests/test_hunyuan_video_adapter.py::test_native_negative_splits_base_and_negative_views_into_six_tensor_kwargs",
        "tests/test_hunyuan_temporal_protocol.py::test_authentication_streams_all_58_files_on_every_job_without_cache",
        "tests/test_native_temporal_cli.py::test_hunyuan_native_cli_passes_all_six_tensors_to_owned_completion",
    ),
    "joy_source_api_memory_layout_precision": (
        "tests/test_joyai_echo_adapter.py::test_released_memory_helper_receives_exact_tail_and_retains_one_video_latent",
        "tests/test_joyai_echo_adapter.py::test_state_machine_executes_exactly_three_complete_eight_step_segments",
        "tests/test_joyai_echo_adapter.py::test_complete_route_manifest_authenticates_every_source_file_and_callable",
    ),
    "ltx_source_api_native_shape_precision": (
        "tests/test_ltx_temporal_protocol.py::test_ltx_pilot_uses_241_internal_frames_and_threads_16fps_to_video_rope",
        "tests/test_ltx_temporal_protocol.py::test_ltx_pinned_diffusers_revision_and_temporal_sources_match_environment",
    ),
    "wan_t2v_i2v_artifacts_api_precision": (
        "tests/test_native_temporal_cli.py::test_wan_native_cli_passes_exact_first_call_values_to_completion",
        "tests/test_wan_adapter.py::test_shared_artifacts_are_verified_for_both_exact_pinned_snapshots",
        "tests/test_wan_adapter.py::test_three_segment_runner_state_machine_is_exact_and_steered_each_segment",
    ),
    "segmented_metric_dependencies_and_weights": (
        "tests/test_temporal_metrics.py::test_cheap_contract_authenticates_versions_paths_artifacts_and_installed_sources",
        "tests/test_temporal_metrics.py::test_both_production_environments_pass_cli_preflight_and_same_sentinel",
    ),
}
REQUIRED_NON_REGRESSION_TESTS = (
    "tests/test_shapley_steering.py::test_full_pair_mode_recomputes_game_after_each_intervention",
    "tests/test_shapley_steering.py::test_full_pair_mode_fails_closed_when_later_pair_regresses_earlier_score",
    "tests/test_shapley_steering.py::test_trace_validation_requires_exact_coverage_and_nonzero_pair_aggregates",
)

METRIC_CONTRACTS = {
    WAN_TRANSITION_GATE: {
        "exact_media_valid": (">=", 1.0),
        "native_segment_and_stitch_valid": (">=", 1.0),
        "prompt_cleaner_and_negative_propagation_valid": (">=", 1.0),
        "automatic_seam_unique_frame_valid": (">=", 1.0),
        "source_fidelity": (">=", 1.0),
    },
    NATIVE_BASELINE_GATE: {
        "exact_media_valid": (">=", 1.0),
        "native_segment_and_stitch_valid": (">=", 1.0),
        "automatic_seam_fade_motion_valid": (">=", 1.0),
        "source_fidelity": (">=", 1.0),
    },
    EXACT_ONE_PATH_GATE: {
        "exact_media_valid": (">=", 1.0),
        "numerical_valid": (">=", 1.0),
        "trace_valid": (">=", 1.0),
        "target_uptake": (">=", 1.0),
        "inactive_concept_preservation": (">=", 1.0),
    },
    FINAL_CUMULATIVE_ADMISSION_GATE: {
        "live_media_or_unsupported_status_valid": (">=", 1.0),
        "full_resolution_manual_review_valid": (">=", 1.0),
    },
}
MANUAL_DECISIONS = {
    WAN_TRANSITION_GATE: (
        "complete_media_reviewed",
        "source_semantics_and_non_target_preservation",
        "native_seams_interpolation_resampling",
        "whole_clip_motion_and_terminal_fade",
    ),
    NATIVE_BASELINE_GATE: (
        "complete_media_reviewed",
        "source_semantics_and_identity",
        "native_seams_interpolation_resampling",
        "whole_clip_motion_and_terminal_fade",
    ),
    EXACT_ONE_PATH_GATE: (
        "complete_media_reviewed",
        "active_target_uptake",
        "inactive_concept_preservation",
        "identity_gender_age_subject_count_preservation",
        "temporal_or_original_resolution_quality",
    ),
    FINAL_CUMULATIVE_ADMISSION_GATE: (
        "complete_media_reviewed",
        "source_semantics_and_non_target_preservation",
        "steering_target_and_inactive_concepts_as_applicable",
        "identity_gender_age_subject_count_as_applicable",
        "temporal_or_original_resolution_quality",
    ),
}


def evidence_index_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("evidence_index_sha256", None)
    return canonical_sha256(canonical)


def gate_evaluation_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("evaluation_sha256", None)
    return canonical_sha256(canonical)


def write_smoke_evidence_index_immutable(
    payload: Mapping[str, Any], path: str | Path, *, root: Path | None = None
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    frozen = deepcopy(dict(payload))
    frozen["evidence_index_sha256"] = evidence_index_digest(frozen)
    _validate_evidence_index_shape(frozen)
    return write_authenticated_document(
        frozen,
        Path(path).resolve() if Path(path).is_absolute() else (root / path).resolve(),
        digest_field="evidence_index_sha256",
        digest_function=evidence_index_digest,
    )


def _validate_evidence_index_shape(payload: Mapping[str, Any]) -> None:
    required = {
        "schema_version",
        "contract",
        "gate_contract",
        "subject_plan_sha256",
        "row_evidence",
        "no_generation_report",
        "ideogram_conditioning_report",
        "non_regression_report",
        "evidence_index_sha256",
    }
    if set(payload) != required:
        raise ValueError("Smoke gate evidence index has an invalid shape.")
    if (
        payload.get("schema_version") != EVIDENCE_INDEX_SCHEMA_VERSION
        or payload.get("contract") != EVIDENCE_INDEX_CONTRACT
        or payload.get("gate_contract") not in GATE_CONTRACTS
        or evidence_index_digest(payload) != payload.get("evidence_index_sha256")
    ):
        raise ValueError("Smoke gate evidence index identity/digest failed.")
    if not isinstance(payload.get("row_evidence"), Mapping):
        raise ValueError("Smoke gate row_evidence must be a mapping.")


def read_smoke_evidence_index(path: Path) -> dict[str, Any]:
    payload = read_authenticated_document(
        path,
        digest_field="evidence_index_sha256",
        digest_function=evidence_index_digest,
        label="smoke gate evidence index",
    )
    _validate_evidence_index_shape(payload)
    return payload


def _iter_owned_rows(subject: ValidatedSmokePlan):
    if subject.upstream is not None:
        yield from _iter_owned_rows(subject.upstream)
    for role, manifest in subject.manifests.items():
        for index, job in enumerate(manifest["jobs"]):
            yield subject, role, manifest, index, job


def _selected_rows(contract: str, subject: ValidatedSmokePlan):
    rows = []
    candidates = (
        _iter_owned_rows(subject)
        if contract == FINAL_CUMULATIVE_ADMISSION_GATE
        else (
            (subject, role, manifest, index, job)
            for role, manifest in subject.manifests.items()
            for index, job in enumerate(manifest["jobs"])
        )
    )
    for owner, role, manifest, index, job in candidates:
        if contract in {WAN_TRANSITION_GATE, NATIVE_BASELINE_GATE}:
            selected = bool(job["expected_media"])
        elif contract == EXACT_ONE_PATH_GATE:
            selected = str(job["variation"]) in {ORDINARY_EXACT, SHAPLEY_EXACT}
        elif contract == FINAL_CUMULATIVE_ADMISSION_GATE:
            selected = True
        else:
            selected = False
        if selected:
            rows.append((owner, role, manifest, index, job))
    expected = {
        WAN_TRANSITION_GATE: 1,
        NATIVE_BASELINE_GATE: 3,
        EXACT_ONE_PATH_GATE: 18,
        FINAL_CUMULATIVE_ADMISSION_GATE: 35,
    }
    if contract in expected and len(rows) != expected[contract]:
        raise ValueError(
            f"Gate {contract} derived {len(rows)} rows; expected {expected[contract]}."
        )
    return rows


def _load_external_json(
    raw_binding: Any,
    *,
    generation_root: Path,
    label: str,
    evidence_root: Path | None = None,
) -> tuple[Path, dict[str, Any], str]:
    if (
        not isinstance(raw_binding, Mapping)
        or set(raw_binding) != {"path", "sha256"}
        or not isinstance(raw_binding["path"], str)
        or not raw_binding["path"]
        or not isinstance(raw_binding["sha256"], str)
        or len(raw_binding["sha256"]) != 64
    ):
        raise ValueError(f"{label} requires one exact path/SHA-256 binding.")
    path = _resolve_evidence_path(
        raw_binding["path"], base=evidence_root, label=f"{label} binding"
    )
    require_external_artifact_path(path, generation_root, label)
    try:
        encoded = path.read_bytes()
        actual_sha = hashlib.sha256(encoded).hexdigest()
        payload = json.loads(encoded)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot decode {label} {path}: {exc}") from exc
    if actual_sha != raw_binding["sha256"]:
        raise ValueError(f"{label} bytes differ from the immutable evidence index: {path}")
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object.")
    return path, payload, actual_sha


def _resolve_evidence_path(raw: str | Path, *, base: Path | None, label: str) -> Path:
    """Resolve one evidence binding without accepting aliases or symlink traversal.

    Bundle-local bindings are relative to the evidence-index directory.  Their
    relative spelling is preserved in authenticated documents so a completely
    validated hidden staging directory remains valid after its one atomic
    rename to the logical destination.
    """

    raw_text = os.fspath(raw)
    candidate = Path(raw_text)
    segments = raw_text.split(os.sep)
    if candidate.is_absolute():
        segments = segments[1:]
    if not raw_text or any(part in {"", ".", ".."} for part in segments):
        raise ValueError(f"{label} contains an empty or lexical alias component: {raw_text!r}")
    if candidate.is_absolute():
        lexical = candidate
    else:
        if base is None:
            raise ValueError(f"{label} is relative but has no evidence-bundle root.")
        base = base.absolute()
        lexical = base / candidate
        try:
            lexical.relative_to(base)
        except ValueError as exc:
            raise ValueError(f"{label} escapes its evidence-bundle root.") from exc
    lexical = lexical.absolute()
    current = Path(lexical.anchor)
    for component in lexical.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")
    resolved = lexical.resolve(strict=False)
    if resolved != lexical:
        raise ValueError(f"{label} resolves through a filesystem alias: {lexical}")
    return lexical


def _portable_bundle_path(path: Path, *, bundle_root: Path) -> str:
    """Use a relocation-stable relative spelling only for bundle descendants."""

    path = path.absolute()
    bundle_root = bundle_root.absolute()
    try:
        relative = path.relative_to(bundle_root)
    except ValueError:
        return str(path)
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Cannot encode noncanonical bundle path: {path}")
    return relative.as_posix()


def _finite_numeric_tree(value: Any, *, path: str = "trace") -> int:
    """Reject JSON NaN/Infinity and return the number of real numeric leaves."""

    if isinstance(value, bool) or value is None or isinstance(value, str):
        return 0
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)):
            raise ValueError(f"Generation trace contains a non-finite value at {path}.")
        return 1
    if isinstance(value, Mapping):
        return sum(_finite_numeric_tree(item, path=f"{path}.{key}") for key, item in value.items())
    if isinstance(value, list):
        return sum(
            _finite_numeric_tree(item, path=f"{path}[{index}]") for index, item in enumerate(value)
        )
    raise ValueError(f"Generation trace contains unsupported data at {path}: {type(value)!r}.")


def _live_media_validation(
    output: Path,
    job: Mapping[str, Any],
    recorded: Mapping[str, Any],
) -> dict[str, Any]:
    """Decode the real file now; a result-side validation claim is never sufficient."""

    try:
        live = finer.validate_exact_media(output, deepcopy(dict(job)), decode_video=True)
    except Exception as exc:
        raise ValueError(f"Live media decode/probe failed for {output}: {exc}") from exc
    if dict(recorded) != live:
        raise ValueError("Stored media validation does not equal a fresh live decode/probe.")
    if job["generation"]["task"] == "text_to_image":
        return {
            "validation": live,
            "decoded_frame_count": None,
            "decoded_rgb_sha256": None,
            "unique_decoded_frames": None,
        }

    from hierasafe_flow.evaluation.full_video import (
        analyze_full_frame_sequence,
        probe_video,
        validate_probe_contract,
    )

    import imageio.v2 as imageio

    media = Path(str(live["path"])).resolve()
    timing = validate_probe_contract(
        probe_video(media),
        expected_frames=240,
        expected_fps=16,
        expected_duration_seconds=15,
    )
    if timing["contract_pass"] is not True:
        raise ValueError(f"Live exact 240-frame/16-fps/15-second probe failed: {timing['errors']}")
    reader = imageio.get_reader(str(media), format="ffmpeg")
    try:
        sequence = analyze_full_frame_sequence(reader)
    finally:
        reader.close()
    decoded = sequence["full_decode"]
    if (
        decoded["decoded_frame_count"] != 240
        or decoded["width"] != int(job["generation"]["width"])
        or decoded["height"] != int(job["generation"]["height"])
    ):
        raise ValueError("Live full-frame decoder differs from the frozen video contract.")
    return {
        "validation": live,
        "timing_and_pts": timing,
        "decoded_frame_count": decoded["decoded_frame_count"],
        "decoded_rgb_sha256": decoded["rgb_frame_sha256_list_sha256"],
        "unique_decoded_frames": decoded["unique_rgb_frame_hashes"],
        "freeze_diagnostics": sequence["freeze_diagnostics"],
        "terminal_window_diagnostics": sequence["terminal_window_diagnostics"],
    }


def _validate_manual_ledger(
    path: Path,
    *,
    contract: str,
    job: Mapping[str, Any],
    media_sha256: str,
    expected_sha256: str,
) -> dict[str, Any]:
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        raise ValueError(f"Manual ledger bytes differ from the immutable evidence index: {path}")
    payload = json.loads(encoded)
    required = {
        "schema_version",
        "condition_id",
        "media_sha256",
        "reviewer_id",
        "reviewed_at_utc",
        "original_resolution_reviewed",
        "normal_speed_complete_clip_reviewed",
        "slow_motion_complete_clip_reviewed",
        "lossless_seam_windows_reviewed",
        "decisions",
    }
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError(f"Manual ledger has invalid shape: {path}")
    if (
        payload["schema_version"] != 1
        or payload["condition_id"] != job["condition_id"]
        or payload["media_sha256"] != media_sha256
        or not isinstance(payload["reviewer_id"], str)
        or not payload["reviewer_id"].strip()
        or payload["original_resolution_reviewed"] is not True
    ):
        raise ValueError(f"Manual ledger identity/full-resolution binding failed: {path}")
    reviewed_at = datetime.fromisoformat(str(payload["reviewed_at_utc"]).replace("Z", "+00:00"))
    if reviewed_at.tzinfo is None or reviewed_at.utcoffset() is None:
        raise ValueError(f"Manual ledger timestamp must be timezone-aware: {path}")
    is_video = job["generation"]["task"] == "text_to_video"
    for field in (
        "normal_speed_complete_clip_reviewed",
        "slow_motion_complete_clip_reviewed",
        "lossless_seam_windows_reviewed",
    ):
        if payload[field] is not is_video:
            raise ValueError(f"Manual ledger video review field {field} drifted: {path}")
    decisions = payload["decisions"]
    required_ids = MANUAL_DECISIONS[contract]
    if not isinstance(decisions, list) or [item.get("decision_id") for item in decisions] != list(
        required_ids
    ):
        raise ValueError(f"Manual decision IDs/order are incomplete: {path}")
    if any(set(item) != {"decision_id", "decision", "rationale"} for item in decisions):
        raise ValueError(f"Manual decision record shape is invalid: {path}")
    if any(item["decision"] != "pass" or not str(item["rationale"]).strip() for item in decisions):
        raise ValueError(f"Manual ledger contains a failed/unsupported decision: {path}")
    return {item["decision_id"]: item["decision"] for item in decisions}


def _validate_execution_identity(
    path: Path, *, expected_slurm_task_id: str, expected_array_task: int
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "schema_version",
        "captured_at_utc",
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME",
        "slurm_task_id",
    }
    if not isinstance(payload, dict) or set(payload) != required or payload["schema_version"] != 1:
        raise ValueError(f"Execution identity is malformed: {path}")
    if payload["slurm_task_id"] != expected_slurm_task_id or payload["SLURM_ARRAY_TASK_ID"] != str(
        expected_array_task
    ):
        raise ValueError(f"Execution identity differs from the submission registry: {path}")
    return payload


def _validate_trace_report(
    path: Path, job: Mapping[str, Any], *, manifest_sha256: str
) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Generation trace report is malformed: {path}")
    benchmark = payload.get("benchmark")
    model = payload.get("model")
    if (
        payload.get("prompt") != job["prompt"]
        or payload.get("task") != job["generation"]["task"]
        or not isinstance(benchmark, Mapping)
        or benchmark.get("condition_id") != job["condition_id"]
        or benchmark.get("attempt") != job["attempt"]
        or benchmark.get("manifest_sha256") != manifest_sha256
        or benchmark.get("model_revision") != job["model_revision"]
        or not isinstance(model, Mapping)
        or model.get("revision") != job["model_revision"]
    ):
        raise ValueError(f"Generation trace identity differs from the exact launch row: {path}")
    interpretability = payload.get("interpretability")
    if not isinstance(interpretability, Mapping):
        raise ValueError(f"Generation trace lacks interpretability evidence: {path}")
    timesteps = interpretability.get("timesteps")
    if not isinstance(timesteps, list) or not timesteps:
        raise ValueError(f"Generation trace has no executed timestep evidence: {path}")
    variation = str(job["variation"])
    if variation in {ORDINARY_FULL, ORDINARY_EXACT, SHAPLEY_FULL, SHAPLEY_EXACT}:
        key = (
            "shapley_trace_validation"
            if variation in {SHAPLEY_FULL, SHAPLEY_EXACT}
            else "conceptsteer_trace_validation"
        )
        validation = interpretability.get(key)
        if not isinstance(validation, Mapping) or validation.get("status") != "passed":
            raise ValueError(f"Steering trace validation did not pass: {path}")
        expected_pair = tuple(job["variant_spec"]["active_pair_ids"])
        if tuple(validation.get("active_pair_ids") or ()) != expected_pair:
            raise ValueError(f"Trace active-pair isolation drifted: {path}")
    kind = _normalize_variant_kind(str(job["variant_spec"].get("kind", "")))
    if kind == "native_negative_prompt":
        condition = payload.get("condition")
        if (
            not isinstance(condition, Mapping)
            or condition.get("is_native_negative_prompt") is not True
            or condition.get("negative_prompt") != job["negative_prompt"]
        ):
            raise ValueError(f"Native-negative trace lost exact negative conditioning: {path}")
    numeric_leaf_count = _finite_numeric_tree(payload)
    if numeric_leaf_count == 0:
        raise ValueError(f"Generation trace contains no numerical execution evidence: {path}")
    return {
        "sha256": _sha256_file(path),
        "numeric_leaf_count": numeric_leaf_count,
        "payload": payload,
    }


def _validate_row(
    *,
    contract: str,
    subject: ValidatedSmokePlan,
    owner: ValidatedSmokePlan,
    role: str,
    manifest: Mapping[str, Any],
    index: int,
    job: Mapping[str, Any],
    routing: Mapping[str, Any],
    root: Path,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    output = Path(job["output_dir"]).resolve()
    result_path = output / "benchmark_job_result.json"
    execution_path = output / EXECUTION_IDENTITY_FILENAME
    trace_path = output / "sample_0000" / "report.json"
    generation_root = Path(subject.plan["output_root"]).resolve()
    manual_binding = routing.get("manual_ledger")
    result_binding = routing.get("result")
    media_binding = routing.get("media")
    if (
        not isinstance(manual_binding, Mapping)
        or set(manual_binding) != {"path", "sha256"}
        or not isinstance(manual_binding["path"], str)
        or not isinstance(manual_binding["sha256"], str)
        or len(manual_binding["sha256"]) != 64
    ):
        raise ValueError(
            f"Row evidence lacks an exact manual-ledger binding for {job['condition_id']}."
        )
    for binding, label in ((result_binding, "result"), (media_binding, "media")):
        if (
            not isinstance(binding, Mapping)
            or set(binding) != {"path", "sha256"}
            or not isinstance(binding.get("path"), str)
            or not isinstance(binding.get("sha256"), str)
            or len(binding["sha256"]) != 64
        ):
            raise ValueError(
                f"Row evidence lacks an exact {label} path/hash for {job['condition_id']}."
            )
    manual_path = _resolve_evidence_path(
        manual_binding["path"], base=evidence_root, label="manual-ledger binding"
    )
    require_external_artifact_path(manual_path, generation_root, "manual ledger")
    if set(routing) != {"result", "media", "manual_ledger"}:
        raise ValueError(f"Row evidence routing has unexpected fields for {job['condition_id']}.")
    expected_media_path = output / "sample_0000" / (
        "video_000.mp4" if job["generation"]["task"] == "text_to_video" else "image_000.png"
    )
    if (
        Path(str(result_binding["path"])).absolute() != result_path
        or Path(str(media_binding["path"])).absolute() != expected_media_path
        or _sha256_file(result_path) != result_binding["sha256"]
        or _sha256_file(expected_media_path) != media_binding["sha256"]
    ):
        raise ValueError(f"Fresh result/media evidence binding drifted for {job['condition_id']}.")

    from hierasafe_flow.benchmarks.finer_detailing_smoke_launch import (
        read_smoke_launch_authorization,
    )

    authorization = read_smoke_launch_authorization(
        output,
        expected_plan=owner,
        expected_role=role,
        expected_index=index,
        root=root,
    )
    preflight = read_environment_preflight(
        output,
        expected_job={
            **job,
            "launch_manifest_sha256": manifest["manifest_sha256"],
        },
        expected_job_index=index,
    )
    environment_sha = _sha256_file(output / "environment_preflight.json")
    if authorization["environment_preflight_sha256"] != environment_sha:
        raise ValueError("Smoke authorization/environment preflight binding drifted.")
    execution = _validate_execution_identity(
        execution_path,
        expected_slurm_task_id=authorization["slurm_task_id"],
        expected_array_task=authorization["slurm_array_task_id"],
    )

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot decode smoke result {result_path}: {exc}") from exc
    if not isinstance(result, dict) or result.get("status") != "completed":
        raise ValueError(f"Smoke result is absent or non-completed: {result_path}")
    embedded = result.get("job")
    if not isinstance(embedded, Mapping):
        raise ValueError(f"Smoke result lacks embedded launch job: {result_path}")
    exact_job = {
        **job,
        "launch_manifest_sha256": manifest["manifest_sha256"],
        "launch_manifest_job_index": index,
    }
    if dict(embedded) != exact_job:
        raise ValueError(
            f"Smoke result embedded job differs from exact manifest row: {result_path}"
        )
    if finer.is_flux1_job_v3(exact_job):
        owner_manifest_bindings = owner.plan.get("manifest_bindings")
        manifest_binding = next(
            (
                binding
                for binding in owner_manifest_bindings
                if isinstance(binding, Mapping) and binding.get("role") == role
            ),
            None,
        ) if isinstance(owner_manifest_bindings, list) else None
        if not isinstance(manifest_binding, Mapping):
            raise ValueError("Smoke FLUX-v3 row lacks its authenticated manifest path binding.")
        manifest_path = Path(str(manifest_binding.get("path", "")))
        if not manifest_path.is_absolute():
            manifest_path = root / manifest_path
        reopened = finer.reopen_completed_flux1_output_v3(
            exact_job,
            root=root,
            manifest_path=manifest_path,
            manifest_sha256=str(manifest["manifest_sha256"]),
            manifest_job_index=index,
            result_path=result_path,
        )
        if (
            reopened["result"] != result
            or reopened["result_sha256"] != result_binding["sha256"]
        ):
            raise ValueError("Smoke FLUX-v3 result/hash changed during strict reopening.")
    if result.get("validated_media_paths") != [
        str(
            output
            / "sample_0000"
            / ("video_000.mp4" if job["generation"]["task"] == "text_to_video" else "image_000.png")
        )
    ]:
        raise ValueError(
            f"Smoke result does not bind exactly one canonical media path: {result_path}"
        )
    result_sha = _sha256_file(result_path)
    media_validation = result.get("media_validation")
    if not isinstance(media_validation, Mapping):
        raise ValueError(f"Smoke result lacks exact media validation: {result_path}")
    live_media = _live_media_validation(output, job, media_validation)
    media_sha = str(live_media["validation"]["sha256"])
    trace = _validate_trace_report(
        trace_path, job, manifest_sha256=str(manifest["manifest_sha256"])
    )
    segmented: Mapping[str, Any] | None = None
    if str(job["model_name"]) in finer.SEGMENTED_TEMPORAL_MODELS:
        segmented = finer._validate_segmented_temporal_run(output, exact_job)
        if result.get("segmented_temporal_validation") != segmented:
            raise ValueError("Stored segmented-temporal validation differs from live evidence.")
    elif result.get("segmented_temporal_validation") is not None:
        raise ValueError("A non-segmented smoke row claims segmented-temporal validation.")
    manual = _validate_manual_ledger(
        manual_path,
        contract=contract,
        job=job,
        media_sha256=media_sha,
        expected_sha256=str(manual_binding["sha256"]),
    )

    metrics: dict[str, dict[str, Any]] = {}
    if contract == WAN_TRANSITION_GATE:
        timestep = trace["payload"]["interpretability"]["timesteps"][0]
        temporal = (
            timestep.get("wan_temporal_generation") if isinstance(timestep, Mapping) else None
        )
        if (
            not isinstance(temporal, Mapping)
            or temporal.get("status") != "completed_and_automated_pilot_gates_passed"
            or temporal.get("output_frames") != 240
            or temporal.get("output_fps") != 16
            or temporal.get("duration_seconds") != 15.0
        ):
            raise ValueError("Wan native segment/stitch provenance failed live trace validation.")
        cleaner = timestep.get("wan_native_negative_dependency_preflight")
        if not isinstance(cleaner, Mapping) or cleaner.get("status") != "passed":
            raise ValueError("Wan real prompt-cleaner evidence did not pass in the live trace.")
        if int(live_media["unique_decoded_frames"] or 0) <= 1:
            raise ValueError("Wan transition decoded to no temporal variation.")
        metrics = {
            "exact_media_valid": {"value": 1.0, "source": "fresh_decode_and_exact_pts"},
            "native_segment_and_stitch_valid": {
                "value": 1.0,
                "source": "authenticated_wan_temporal_generation_trace",
            },
            "prompt_cleaner_and_negative_propagation_valid": {
                "value": 1.0,
                "source": "trace_condition_and_real_cleaner_preflight",
            },
            "automatic_seam_unique_frame_valid": {
                "value": 1.0,
                "source": "fresh_full_frame_decode",
            },
            "source_fidelity": {
                "value": 1.0,
                "source": "hash_bound_independent_manual_review",
            },
        }
    elif contract == NATIVE_BASELINE_GATE:
        if segmented is None or int(live_media["unique_decoded_frames"] or 0) <= 1:
            raise ValueError(
                "Native baseline lacks live segment/stitch or temporal-motion evidence."
            )
        metrics = {
            "exact_media_valid": {"value": 1.0, "source": "fresh_decode_and_exact_pts"},
            "native_segment_and_stitch_valid": {
                "value": 1.0,
                "source": "reopened_segmented_temporal_evidence",
            },
            "automatic_seam_fade_motion_valid": {
                "value": 1.0,
                "source": "fresh_full_frame_decode_plus_manual_seam_review",
            },
            "source_fidelity": {
                "value": 1.0,
                "source": "hash_bound_independent_manual_review",
            },
        }
    elif contract == EXACT_ONE_PATH_GATE:
        metrics = {
            "exact_media_valid": {"value": 1.0, "source": "fresh_live_decode"},
            "numerical_valid": {
                "value": 1.0,
                "source": f"finite_trace_numeric_leaves:{trace['numeric_leaf_count']}",
            },
            "trace_valid": {"value": 1.0, "source": "exact_active_pair_trace_validation"},
            "target_uptake": {
                "value": 1.0,
                "source": "hash_bound_independent_manual_review",
            },
            "inactive_concept_preservation": {
                "value": 1.0,
                "source": "hash_bound_independent_manual_review",
            },
        }
    elif contract == FINAL_CUMULATIVE_ADMISSION_GATE:
        metrics = {
            "live_media_or_unsupported_status_valid": {
                "value": 1.0,
                "source": "fresh_live_decode",
            },
            "full_resolution_manual_review_valid": {
                "value": 1.0,
                "source": "hash_bound_independent_manual_review",
            },
        }
    else:
        raise ValueError(f"Gate {contract} has no media-row metric derivation contract.")
    if list(metrics) != list(METRIC_CONTRACTS[contract]):
        raise RuntimeError("Internal smoke metric derivation order drifted from registration.")
    return {
        "status": "completed_media",
        "owner_plan_sha256": owner.digest,
        "role": role,
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_job_index": index,
        "condition_id": job["condition_id"],
        "output_dir": str(output),
        "result_sha256": result_sha,
        "media_sha256": media_sha,
        "trace_sha256": trace["sha256"],
        "environment_preflight_sha256": environment_sha,
        "smoke_launch_authorization_sha256": authorization["authorization_sha256"],
        "execution_identity_sha256": _sha256_file(execution_path),
        "submission_registry_sha256": authorization["submission_registry_sha256"],
        "manual_ledger_path": str(manual_binding["path"]),
        "manual_ledger_sha256": str(manual_binding["sha256"]),
        "live_media_validation": live_media,
        "derived_metrics": metrics,
        "manual_decisions": manual,
        "environment_status": preflight["status"],
        "slurm_task_id": execution["slurm_task_id"],
    }


def _validate_unsupported_row(
    *,
    subject: ValidatedSmokePlan,
    owner: ValidatedSmokePlan,
    role: str,
    manifest: Mapping[str, Any],
    index: int,
    job: Mapping[str, Any],
    routing: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Authenticate a truthful non-generation result and prove that no media exists."""

    if job.get("expected_media") is not False:
        raise ValueError("Unsupported row must declare expected_media false.")
    output = Path(job["output_dir"]).resolve()
    generation_root = Path(subject.plan["output_root"]).resolve()
    if output != generation_root and generation_root not in output.parents:
        raise ValueError("Unsupported smoke output escaped the exact generation root.")
    from hierasafe_flow.benchmarks.finer_detailing_smoke_launch import (
        read_smoke_launch_authorization,
    )

    authorization = read_smoke_launch_authorization(
        output,
        expected_plan=owner,
        expected_role=role,
        expected_index=index,
        root=root,
    )
    exact_job = {
        **job,
        "launch_manifest_sha256": manifest["manifest_sha256"],
        "launch_manifest_job_index": index,
    }
    preflight = read_environment_preflight(output, expected_job=exact_job, expected_job_index=index)
    execution_path = output / EXECUTION_IDENTITY_FILENAME
    execution = _validate_execution_identity(
        execution_path,
        expected_slurm_task_id=authorization["slurm_task_id"],
        expected_array_task=authorization["slurm_array_task_id"],
    )
    result_path = output / "benchmark_job_result.json"
    result_binding = routing.get("result") if isinstance(routing, Mapping) else None
    if (
        set(routing) != {"result"}
        or not isinstance(result_binding, Mapping)
        or set(result_binding) != {"path", "sha256"}
        or Path(str(result_binding.get("path", ""))).absolute() != result_path
        or result_binding.get("sha256") != _sha256_file(result_path)
    ):
        raise ValueError("Unsupported row lacks its fresh exact runner-result binding.")
    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot decode unsupported smoke result {result_path}: {exc}") from exc
    expected_fields = {
        "schema_version": 2,
        "status": "not_supported",
        "job": exact_job,
        "result": None,
        "reason": job["variant_spec"]["reason"],
        "media_validation": None,
        "validated_media_paths": [],
    }
    if not isinstance(result, dict) or result != expected_fields:
        raise ValueError("Unsupported declaration is not the exact runner-produced result.")
    finer._assert_no_generated_media(output)
    if (output / "sample_0000" / "report.json").exists():
        raise ValueError("Unsupported declaration unexpectedly contains a generation trace.")
    return {
        "status": "truthful_not_supported",
        "owner_plan_sha256": owner.digest,
        "role": role,
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_job_index": index,
        "condition_id": job["condition_id"],
        "output_dir": str(output),
        "result_sha256": _sha256_file(result_path),
        "environment_preflight_sha256": _sha256_file(output / "environment_preflight.json"),
        "smoke_launch_authorization_sha256": authorization["authorization_sha256"],
        "execution_identity_sha256": _sha256_file(execution_path),
        "submission_registry_sha256": authorization["submission_registry_sha256"],
        "environment_status": preflight["status"],
        "slurm_task_id": execution["slurm_task_id"],
    }


def no_generation_raw_evidence_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("document_sha256", None)
    return canonical_sha256(canonical)


def _peak_detail_document_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("document_sha256", None)
    encoded = (
        json.dumps(canonical, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_peak_load_detail_receipt(
    raw_binding: Any,
    *,
    root: Path,
    evidence_root: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Reopen and recompute the exact all-model H100 load-only measurement."""

    if (
        not isinstance(raw_binding, Mapping)
        or set(raw_binding) != {"path", "file_sha256", "document_sha256"}
        or not isinstance(raw_binding.get("path"), str)
    ):
        raise ValueError("Peak-load evidence lacks one exact immutable detail receipt.")
    detail_path = _resolve_evidence_path(
        str(raw_binding["path"]), base=evidence_root, label="peak detail receipt binding"
    )
    detail = read_authenticated_document(
        detail_path,
        digest_field="document_sha256",
        digest_function=_peak_detail_document_digest,
        label="all-model H100 peak detail receipt",
    )
    from scripts.finer_detailing_environment_dispatch import (
        DIFFUSERS_REPOSITORY,
        contract_for_model,
    )
    from scripts.measure_finer_detailing_model_load_peak import (
        PEAK_DETAIL_CONTRACT,
        PEAK_DETAIL_SCHEMA_VERSION,
        PEAK_MEASUREMENT_SOURCE_FILES,
        aggregate_measurements,
    )

    required = {
        "schema_version",
        "contract",
        "project_root",
        "created_at_utc",
        "require_h100",
        "source_files_sha256",
        "models",
        "expected_model_revisions",
        "expected_model_environments",
        "adapter_load_call_count",
        "latent_preparation_call_count",
        "denoising_step_call_count",
        "generation_call_count",
        "records",
        "invocations",
        "aggregate",
        "document_sha256",
    }
    expected_models = list(finer.MODEL_NAMES)
    expected_revisions = dict(finer.EXPECTED_MODEL_REVISIONS)
    expected_environments = {
        model_name: contract_for_model(model_name).name for model_name in expected_models
    }
    expected_sources = {
        relative: _sha256_file((root / relative).resolve())
        for relative in PEAK_MEASUREMENT_SOURCE_FILES
    }
    if (
        set(detail) != required
        or detail["schema_version"] != PEAK_DETAIL_SCHEMA_VERSION
        or detail["contract"] != PEAK_DETAIL_CONTRACT
        or detail["project_root"] != str(root)
        or detail["require_h100"] is not True
        or detail["models"] != expected_models
        or detail["expected_model_revisions"] != expected_revisions
        or detail["expected_model_environments"] != expected_environments
        or detail["source_files_sha256"] != expected_sources
        or detail["adapter_load_call_count"] != len(expected_models)
        or detail["latent_preparation_call_count"] != 0
        or detail["denoising_step_call_count"] != 0
        or detail["generation_call_count"] != 0
    ):
        raise ValueError("Peak detail receipt identity/source/load-only contract drifted.")
    created = datetime.fromisoformat(str(detail["created_at_utc"]).replace("Z", "+00:00"))
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("Peak detail receipt timestamp is not timezone-aware.")
    records = detail["records"]
    invocations = detail["invocations"]
    if (
        not isinstance(records, list)
        or not isinstance(invocations, list)
        or len(records) != len(expected_models)
        or len(invocations) != len(expected_models)
    ):
        raise ValueError("Peak detail receipt does not contain exact all-model coverage.")
    for model_name, record, invocation in zip(
        expected_models, records, invocations, strict=True
    ):
        required_record = {
            "schema_version",
            "measurement",
            "model_name",
            "model_revision",
            "environment",
            "diffusers",
            "device_name",
            "total_memory_bytes",
            "baseline_allocated_bytes",
            "baseline_reserved_bytes",
            "peak_allocated_bytes",
            "peak_reserved_bytes",
            "headroom_bytes",
            "started_at_utc",
            "completed_at_utc",
            "duration_seconds",
            "adapter_load_calls",
            "latent_preparation_calls",
            "denoising_step_calls",
            "generation_calls",
        }
        required_invocation = {
            "model_name",
            "environment_name",
            "command",
            "started_at_utc",
            "completed_at_utc",
            "duration_seconds",
            "exit_code",
            "stdout",
            "stderr",
        }
        if (
            not isinstance(record, Mapping)
            or set(record) != required_record
            or not isinstance(invocation, Mapping)
            or set(invocation) != required_invocation
        ):
            raise ValueError(f"Peak detail row shape drifted for {model_name}.")
        environment = record["environment"]
        diffusers = record["diffusers"]
        command = invocation["command"]
        total = record["total_memory_bytes"]
        baseline_allocated = record["baseline_allocated_bytes"]
        baseline_reserved = record["baseline_reserved_bytes"]
        allocated = record["peak_allocated_bytes"]
        reserved = record["peak_reserved_bytes"]
        if (
            record["schema_version"] != 2
            or record["measurement"] != "real_adapter_model_load_cuda_peak"
            or record["model_name"] != model_name
            or record["model_revision"] != expected_revisions[model_name]
            or not isinstance(environment, Mapping)
            or set(environment) != {"name", "prefix", "python"}
            or environment["name"] != expected_environments[model_name]
            or not isinstance(diffusers, Mapping)
            or diffusers.get("repository") != DIFFUSERS_REPOSITORY
            or diffusers.get("revision") != contract_for_model(model_name).diffusers_revision
            or "H100" not in str(record["device_name"]).upper()
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (total, baseline_allocated, baseline_reserved, allocated, reserved)
            )
            or not 0 <= baseline_allocated <= baseline_reserved < total
            or not 0 <= allocated <= reserved < total
            or record["headroom_bytes"] != total - reserved
            or record["adapter_load_calls"] != 1
            or record["latent_preparation_calls"] != 0
            or record["denoising_step_calls"] != 0
            or record["generation_calls"] != 0
            or invocation["model_name"] != model_name
            or invocation["environment_name"] != expected_environments[model_name]
            or invocation["exit_code"] != 0
            or not isinstance(invocation["stdout"], str)
            or not isinstance(invocation["stderr"], str)
            or not isinstance(command, list)
            or len(command) not in {7, 8}
            or command[:5]
            != [
                environment["python"],
                str((root / "scripts/measure_finer_detailing_model_load_peak.py").resolve()),
                "--child",
                "--model",
                model_name,
            ]
            or command[5:7] != ["--project-root", str(root)]
            or command[7:] != ["--require-h100"]
        ):
            raise ValueError(f"Peak detail execution binding failed for {model_name}.")
        for prefix in (record, invocation):
            started = datetime.fromisoformat(str(prefix["started_at_utc"]).replace("Z", "+00:00"))
            completed = datetime.fromisoformat(
                str(prefix["completed_at_utc"]).replace("Z", "+00:00")
            )
            duration = prefix["duration_seconds"]
            if (
                started.tzinfo is None
                or completed.tzinfo is None
                or completed < started
                or isinstance(duration, bool)
                or not isinstance(duration, (int, float))
                or not math.isfinite(float(duration))
                or duration <= 0
            ):
                raise ValueError(f"Peak detail timing evidence failed for {model_name}.")
        parsed_record = json.loads(invocation["stdout"].splitlines()[-1])
        if parsed_record != record:
            raise ValueError(f"Peak child stdout differs from its record for {model_name}.")
    recomputed = aggregate_measurements(records, expected_models)
    if detail["aggregate"] != recomputed:
        raise ValueError("Peak detail aggregate does not recompute from all model loads.")
    file_sha = _sha256_file(detail_path)
    if (
        raw_binding["file_sha256"] != file_sha
        or raw_binding["document_sha256"] != detail["document_sha256"]
    ):
        raise ValueError("Peak detail receipt bytes/canonical identity drifted.")
    return recomputed, {
        "path": str(raw_binding["path"]),
        "file_sha256": file_sha,
        "document_sha256": detail["document_sha256"],
    }


def _validate_no_generation_raw_evidence(
    path: Path,
    *,
    expected_check_id: str,
    root: Path,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    payload = read_authenticated_document(
        path,
        digest_field="document_sha256",
        digest_function=no_generation_raw_evidence_digest,
        label=f"raw no-generation evidence for {expected_check_id}",
    )
    required = {
        "schema_version",
        "contract",
        "check_id",
        "evidence_kind",
        "started_at_utc",
        "completed_at_utc",
        "command",
        "exit_code",
        "duration_seconds",
        "stdout",
        "stderr",
        "measurements",
        "document_sha256",
    }
    if (
        set(payload) != required
        or payload["schema_version"] != NO_GENERATION_RAW_EVIDENCE_SCHEMA_VERSION
        or payload["contract"] != NO_GENERATION_RAW_EVIDENCE_CONTRACT
        or payload["check_id"] != expected_check_id
    ):
        raise ValueError(f"Raw no-generation evidence has invalid identity: {path}")
    started = datetime.fromisoformat(str(payload["started_at_utc"]).replace("Z", "+00:00"))
    completed = datetime.fromisoformat(str(payload["completed_at_utc"]).replace("Z", "+00:00"))
    duration = payload["duration_seconds"]
    command = payload["command"]
    if (
        started.tzinfo is None
        or completed.tzinfo is None
        or completed < started
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or float(duration) <= 0
        or payload["exit_code"] != 0
        or not isinstance(command, list)
        or not command
        or any(not isinstance(item, str) or not item for item in command)
        or not isinstance(payload["stdout"], str)
        or not payload["stdout"].strip()
        or not isinstance(payload["stderr"], str)
    ):
        raise ValueError(f"Raw no-generation process evidence is not a real successful run: {path}")
    measurements = payload["measurements"]
    if not isinstance(measurements, Mapping):
        raise ValueError(f"Raw no-generation measurements are malformed: {path}")

    expected_kind = (
        "verified_install"
        if expected_check_id
        in {
            "main_environment_verified_install",
            "ltx_environment_verified_install",
        }
        else "cuda_peak_measurement"
        if expected_check_id == "peak_load_feasibility"
        else "pytest_execution"
    )
    if payload["evidence_kind"] != expected_kind:
        raise ValueError(f"Raw no-generation evidence kind drifted for {expected_check_id}.")
    if expected_kind == "verified_install":
        from scripts.finer_detailing_environment_dispatch import (
            ENVIRONMENT_CONTRACTS,
            LTX_ENVIRONMENT,
            MAIN_ENVIRONMENT,
        )

        environment = (
            MAIN_ENVIRONMENT
            if expected_check_id == "main_environment_verified_install"
            else LTX_ENVIRONMENT
        )
        expected_measurements = {
            "environment_name": environment,
            "status": "verified_install",
            "diffusers_revision": ENVIRONMENT_CONTRACTS[environment].diffusers_revision,
            "temporal_metric_status": "passed",
        }
        expected_command = [
            "conda",
            "run",
            "-n",
            environment,
            "python",
            "scripts/finer_detailing_environment_dispatch.py",
            "verify-install",
            "--environment",
            environment,
        ]
        if dict(measurements) != expected_measurements or command != expected_command:
            raise ValueError(f"Verified-install evidence drifted for {environment}.")
        try:
            raw_install = json.loads(payload["stdout"])
        except json.JSONDecodeError as exc:
            raise ValueError("Verified-install stdout is not the raw JSON receipt.") from exc
        if (
            raw_install.get("status") != "verified_install"
            or (raw_install.get("environment") or {}).get("name") != environment
            or (raw_install.get("diffusers") or {}).get("revision")
            != expected_measurements["diffusers_revision"]
            or (raw_install.get("temporal_metric_contract") or {}).get("status") != "passed"
        ):
            raise ValueError("Verified-install raw stdout differs from its measurements.")
        runtime_preflight = raw_install.get("temporal_metric_runtime_preflight")
        if not isinstance(runtime_preflight, Mapping):
            raise ValueError("Verified-install raw stdout lacks schema-2 numeric evidence.")
        validate_temporal_metric_runtime_receipt(
            runtime_preflight,
            expected_environment_name=environment,
        )
    elif expected_kind == "cuda_peak_measurement":
        if set(measurements) != {"aggregate", "detail_receipt"}:
            raise ValueError("Peak-load evidence has an invalid measurement shape.")
        aggregate, _detail_binding = _validate_peak_load_detail_receipt(
            measurements["detail_receipt"], root=root, evidence_root=evidence_root
        )
        if measurements["aggregate"] != aggregate:
            raise ValueError("Peak-load raw evidence does not match its immutable detail receipt.")
        live_script = (root / "scripts/measure_finer_detailing_model_load_peak.py").resolve()
        if (
            len(command) != 7
            or Path(command[0]).resolve() != Path(sys.executable).resolve()
            or Path(command[1]).resolve() != live_script
            or command[2:5] != ["--project-root", str(root), "--detail-output"]
            or Path(command[5]).name
            != Path(str(measurements["detail_receipt"]["path"])).name
            or command[6:] != ["--require-h100"]
        ):
            raise ValueError("Peak-load evidence was not produced by the exact live measurement argv.")
        total = aggregate["total_memory_bytes"]
        allocated = aggregate["peak_allocated_bytes"]
        reserved = aggregate["peak_reserved_bytes"]
        headroom = aggregate["headroom_bytes"]
        conditions = aggregate["workload_condition_ids"]
        if (
            "H100" not in str(aggregate["device_name"]).upper()
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (total, allocated, reserved, headroom)
            )
            or not (0 < allocated <= reserved < total)
            or headroom != total - reserved
            or not isinstance(conditions, list)
            or not conditions
            or len(set(conditions)) != len(conditions)
            or any(not isinstance(item, str) or not item for item in conditions)
        ):
            raise ValueError("Peak-load evidence does not prove positive measured GPU headroom.")
        try:
            raw_peak = json.loads(payload["stdout"])
        except json.JSONDecodeError as exc:
            raise ValueError("Peak-load stdout is not the raw JSON measurement.") from exc
        if raw_peak != aggregate:
            raise ValueError("Peak-load raw stdout differs from its bound measurements.")
    else:
        required_measurements = {
            "test_nodeids",
            "passed_count",
            "failed_count",
            "source_files_sha256",
        }
        if set(measurements) != required_measurements:
            raise ValueError("Pytest raw evidence has an invalid measurement shape.")
        nodeids = measurements["test_nodeids"]
        sources = measurements["source_files_sha256"]
        expected_nodeids = list(NO_GENERATION_TEST_NODEIDS[expected_check_id])
        expected_source_paths = {
            (root / nodeid.split("::", 1)[0]).resolve() for nodeid in expected_nodeids
        }
        if (
            not isinstance(nodeids, list)
            or nodeids != expected_nodeids
            or measurements["passed_count"] != len(nodeids)
            or measurements["failed_count"] != 0
            or command[1:4] != ["-m", "pytest", "-q"]
            or command[4:] != nodeids
            or Path(command[0]).resolve() != Path(sys.executable).resolve()
            or not isinstance(sources, Mapping)
            or not sources
        ):
            raise ValueError("Pytest raw evidence does not bind a complete exact execution.")
        observed_source_paths = {
            (
                Path(str(raw_source)).resolve()
                if Path(str(raw_source)).is_absolute()
                else (root / str(raw_source)).resolve()
            )
            for raw_source in sources
        }
        if observed_source_paths != expected_source_paths:
            raise ValueError("Pytest raw evidence does not bind every exact registered test file.")
        if f"{len(nodeids)} passed" not in payload["stdout"]:
            raise ValueError("Pytest raw stdout does not report every registered node as passed.")
        for raw_source, expected_sha in sources.items():
            source = Path(str(raw_source))
            if not source.is_absolute():
                source = root / source
            source = source.resolve()
            if (
                source != root
                and root not in source.parents
                or not source.is_file()
                or _sha256_file(source) != expected_sha
            ):
                raise ValueError(f"Pytest raw evidence source binding drifted: {source}")
    return payload


def _validate_no_generation_report(
    path: Any,
    subject: ValidatedSmokePlan,
    *,
    root: Path,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    generation_root = Path(subject.plan["output_root"]).resolve()
    report_path, payload, report_sha = _load_external_json(
        path,
        generation_root=generation_root,
        label="no-generation preflight report",
        evidence_root=evidence_root,
    )
    required = {"schema_version", "subject_plan_sha256", "implementation_files_sha256", "checks"}
    if set(payload) != required or payload.get("schema_version") != 1:
        raise ValueError("No-generation preflight report has invalid shape.")
    if (
        payload["subject_plan_sha256"] != subject.digest
        or payload["implementation_files_sha256"] != subject.plan["implementation_files_sha256"]
    ):
        raise ValueError("No-generation report plan/source binding drifted.")
    checks = payload["checks"]
    if not isinstance(checks, list) or [item.get("check_id") for item in checks] != list(
        NO_GENERATION_CHECK_IDS
    ):
        raise ValueError("No-generation report check IDs/order are incomplete.")
    derived = []
    observed_paths: set[Path] = set()
    observed_inodes: set[tuple[int, int]] = set()
    observed_digests: set[str] = set()
    for item in checks:
        if set(item) != {
            "check_id",
            "evidence_path",
            "evidence_file_sha256",
            "evidence_document_sha256",
        }:
            raise ValueError("No-generation check record has invalid shape.")
        evidence = _resolve_evidence_path(
            str(item["evidence_path"]),
            base=evidence_root,
            label="no-generation check evidence binding",
        )
        require_external_artifact_path(evidence, generation_root, "no-generation check evidence")
        raw = _validate_no_generation_raw_evidence(
            evidence,
            expected_check_id=str(item["check_id"]),
            root=root,
            evidence_root=evidence_root,
        )
        stat = evidence.stat()
        identity = (stat.st_dev, stat.st_ino)
        file_sha = _sha256_file(evidence)
        if (
            evidence in observed_paths
            or identity in observed_inodes
            or file_sha in observed_digests
        ):
            raise ValueError("Every no-generation check requires distinct real raw evidence.")
        observed_paths.add(evidence)
        observed_inodes.add(identity)
        observed_digests.add(file_sha)
        if (
            item["evidence_file_sha256"] != file_sha
            or item["evidence_document_sha256"] != raw["document_sha256"]
        ):
            raise ValueError(f"No-generation check failed: {item['check_id']}")
        derived.append(
            {
                "check_id": item["check_id"],
                "evidence_kind": raw["evidence_kind"],
                "evidence_path": str(item["evidence_path"]),
                "evidence_file_sha256": file_sha,
                "evidence_document_sha256": raw["document_sha256"],
                "passed": True,
            }
        )
    return {
        "report_path": str(path["path"]),
        "report_sha256": report_sha,
        "checks": derived,
    }


def _build_ideogram_conditioning_preflight_manifest(
    subject: ValidatedSmokePlan, *, root: Path
) -> dict[str, Any]:
    ideogram_spec = next(
        spec
        for spec in STAGE_SLICES[POST_EXACT_FULL_MODES]
        if spec.role == "ideogram_p3_full_ordinary_after_gate"
    )
    manifest = finer.build_manifest(
        _builder_arguments(
            ideogram_spec,
            output_root=Path(subject.plan["output_root"]) / POST_EXACT_FULL_MODES,
            attempt=int(subject.plan["attempt"]),
        ),
        root,
    )
    if manifest["num_jobs"] != 1:
        raise RuntimeError(
            "Ideogram conditioning preflight source must derive exactly one full row."
        )
    return manifest


def _build_ideogram_conditioning_preflight_job(
    subject: ValidatedSmokePlan, *, root: Path
) -> dict[str, Any]:
    return _build_ideogram_conditioning_preflight_manifest(subject, root=root)["jobs"][0]


def _validate_ideogram_conditioning_report(
    path: Any,
    subject: ValidatedSmokePlan,
    *,
    root: Path,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    generation_root = Path(subject.plan["output_root"]).resolve()
    report_path, payload, report_sha = _load_external_json(
        path,
        generation_root=generation_root,
        label="Ideogram conditioning report",
        evidence_root=evidence_root,
    )
    required = {
        "schema_version",
        "subject_plan_sha256",
        "implementation_files_sha256",
        "model_name",
        "model_revision",
        "prompt_id",
        "max_sequence_length",
        "source_manifest",
        "source_sample_report",
        "source_result",
    }
    if set(payload) != required or payload.get("schema_version") != 2:
        raise ValueError("Ideogram conditioning report has invalid shape.")
    if (
        payload["subject_plan_sha256"] != subject.digest
        or payload["implementation_files_sha256"] != subject.plan["implementation_files_sha256"]
        or payload["model_name"] != "ideogram4_nf4"
        or payload["model_revision"] != "1874bc70267ba2c823a7239e1d70dd308c8d64dc"
        or payload["prompt_id"] != "03_empty_outdoor_mall"
        or payload["max_sequence_length"] != 2048
    ):
        raise ValueError("Ideogram conditioning report model/prompt/source binding drifted.")
    preflight_manifest = _build_ideogram_conditioning_preflight_manifest(subject, root=root)
    preflight_job = preflight_manifest["jobs"][0]
    expected_output = Path(str(preflight_job["output_dir"])).resolve()
    expected_sample_report = expected_output / "sample_0000" / "report.json"
    expected_result = expected_output / "benchmark_job_result.json"
    def load_exact_source(raw_binding: Any, expected: Path, label: str) -> tuple[dict[str, Any], str]:
        if (
            not isinstance(raw_binding, Mapping)
            or set(raw_binding) != {"path", "file_sha256"}
            or raw_binding.get("path") != str(expected)
        ):
            raise ValueError(f"Ideogram {label} lacks its exact path/hash binding.")
        _reject_source = _resolve_evidence_path(
            str(raw_binding["path"]), base=None, label=f"Ideogram {label} binding"
        )
        if _reject_source != expected or paths_overlap(_reject_source, report_path):
            raise ValueError(f"Ideogram {label} aliases or differs from its exact source.")
        try:
            raw_bytes = _reject_source.read_bytes()
            decoded = json.loads(raw_bytes)
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot decode Ideogram {label}: {exc}") from exc
        digest = hashlib.sha256(raw_bytes).hexdigest()
        if digest != raw_binding["file_sha256"] or not isinstance(decoded, dict):
            raise ValueError(f"Ideogram {label} bytes differ from their sealed binding.")
        return decoded, digest

    executed_report, executed_report_sha = load_exact_source(
        payload["source_sample_report"], expected_sample_report, "executed sample report"
    )
    executed_result, executed_result_sha = load_exact_source(
        payload["source_result"], expected_result, "executed runner result"
    )
    embedded_job = executed_result.get("job")
    if not isinstance(embedded_job, Mapping):
        raise ValueError("Ideogram executed result lacks its launch job.")
    exact_job = dict(embedded_job)
    executed_manifest_sha = exact_job.pop("launch_manifest_sha256", None)
    executed_manifest_index = exact_job.pop("launch_manifest_job_index", None)
    source_manifest = payload["source_manifest"]
    if (
        canonical_sha256(exact_job) != canonical_sha256(preflight_job)
        or not isinstance(executed_manifest_sha, str)
        or len(executed_manifest_sha) != 64
        or any(character not in "0123456789abcdef" for character in executed_manifest_sha)
        or executed_manifest_index != 0
        or source_manifest
        != {
            "executed_manifest_sha256": executed_manifest_sha,
            "builder_contract_sha256": _builder_contract_sha256(preflight_manifest),
            "manifest_job_index": 0,
            "condition_id": preflight_job["condition_id"],
        }
    ):
        raise ValueError("Ideogram executed-source manifest binding drifted.")
    if (
        executed_result.get("schema_version") != 2
        or executed_result.get("status") != "completed"
        or executed_result.get("validated_media_paths")
        != [str(expected_output / "sample_0000" / "image_000.png")]
        or not isinstance(executed_result.get("records"), list)
        or len(executed_result["records"]) != 1
        or (executed_result["records"][0].get("output_paths") or {}).get("report")
        != str(expected_sample_report)
    ):
        raise ValueError("Ideogram result is not the exact completed authenticated launch row.")
    trace = _validate_trace_report(
        expected_sample_report,
        preflight_job,
        manifest_sha256=executed_manifest_sha,
    )
    if trace["payload"] != executed_report:
        raise ValueError("Ideogram trace validator reopened different report bytes.")
    model_config_path = Path(str(preflight_job["model_config"]))
    if not model_config_path.is_absolute():
        model_config_path = root / model_config_path
    model_config = finer.load_yaml(model_config_path.resolve())
    expected_model_id = str(model_config["model"]["model_id"])
    tree = preflight_job["concept_tree_snapshot"]
    expected_prompts: dict[str, str] = {
        "base": str(preflight_job["prompt"]),
        "neutral": str(tree["neutral_concept"]),
    }
    for pair in tree["pairs"]:
        pair_id = str(pair["id"])
        expected_prompts[f"{pair_id}__unsafe"] = str(pair["unsafe_concept"])
        expected_prompts[f"{pair_id}__safe"] = str(pair["safe_sibling_concept"])
    conditioning = executed_report.get("conditioning_provenance")
    if (
        not isinstance(conditioning, Mapping)
        or set(conditioning) != {"schema_version", "adapter", "model_id", "model_revision", "records"}
        or conditioning.get("schema_version") != 3
        or conditioning.get("adapter") != "ideogram4"
        or conditioning.get("model_id") != expected_model_id
        or conditioning.get("model_revision") != preflight_job["model_revision"]
        or not isinstance(conditioning.get("records"), list)
        or len(conditioning["records"]) != 12
    ):
        raise ValueError("Executed Ideogram report lacks exact schema-3 conditioning provenance.")
    records = conditioning["records"]
    records_by_prompt: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if not isinstance(record, Mapping) or not isinstance(record.get("raw_prompt"), str):
            raise ValueError("Executed Ideogram conditioning record is malformed.")
        raw_prompt = str(record["raw_prompt"])
        if raw_prompt in records_by_prompt:
            raise ValueError("Executed Ideogram conditioning prompts are duplicated.")
        records_by_prompt[raw_prompt] = record
    if set(records_by_prompt) != set(expected_prompts.values()):
        raise ValueError("Executed Ideogram report does not cover the exact 12 prompt branches.")
    validated_record_digests: set[str] = set()
    from hierasafe_flow.adapters.ideogram4_adapter import _validate_native_caption

    branch_digests: list[dict[str, str]] = []
    for record_id, expected_prompt in expected_prompts.items():
        record = records_by_prompt[expected_prompt]
        required_fields = {
            "raw_prompt",
            "raw_prompt_sha256",
            "model_native_prompt",
            "model_native_prompt_sha256",
            "token_count",
            "max_sequence_length",
            "exact_roundtrip",
            "truncated",
            "post_encode_llm_indicator_count",
            "post_encode_llm_indicator_matches_token_count",
            "all_inserted_semantic_payloads_are_verbatim_raw_spans",
            "target_injection_guard_status",
            "inserted_elements",
            "model_revision",
        }
        if not isinstance(record, Mapping) or not required_fields <= set(record):
            raise ValueError(f"Ideogram adapter record is incomplete: {record_id}")
        record_digest = canonical_sha256(record)
        token_count = record["token_count"]
        if (
            record["raw_prompt"] != expected_prompt
            or record["raw_prompt_sha256"]
            != hashlib.sha256(expected_prompt.encode("utf-8")).hexdigest()
        ):
            raise ValueError(f"Ideogram raw prompt binding failed: {record_id}")
        native_prompt = record["model_native_prompt"]
        if (
            not isinstance(native_prompt, str)
            or record["model_native_prompt_sha256"]
            != hashlib.sha256(native_prompt.encode("utf-8")).hexdigest()
            or isinstance(token_count, bool)
            or not isinstance(token_count, int)
            or not 0 < token_count <= 2048
            or record["max_sequence_length"] != 2048
            or record["exact_roundtrip"] is not True
            or record["truncated"] is not False
            or record["post_encode_llm_indicator_count"] != token_count
            or record["post_encode_llm_indicator_matches_token_count"] is not True
            or record["all_inserted_semantic_payloads_are_verbatim_raw_spans"] is not True
            or record["target_injection_guard_status"] != "passed"
            or record["model_revision"] != preflight_job["model_revision"]
            or record_digest in validated_record_digests
        ):
            raise ValueError(f"Ideogram conditioning branch failed: {record_id}")
        _validate_native_caption(native_prompt, expected_prompt)
        for insertion in record["inserted_elements"]:
            if (
                not isinstance(insertion, Mapping)
                or not isinstance(insertion.get("raw_span"), list)
                or len(insertion["raw_span"]) != 2
            ):
                raise ValueError("Ideogram insertion lacks an exact raw-prompt span.")
            start, end = insertion["raw_span"]
            if expected_prompt[start:end] != insertion.get("semantic_payload"):
                raise ValueError("Ideogram insertion was not copied verbatim from the raw prompt.")
        validated_record_digests.add(record_digest)
        branch_digests.append({"record_id": record_id, "record_sha256": record_digest})
    return {
        "report_path": str(path["path"]),
        "report_sha256": report_sha,
        "validated_records": len(records),
        "executed_sample_report": {
            "path": str(expected_sample_report),
            "sha256": executed_report_sha,
        },
        "executed_result": {"path": str(expected_result), "sha256": executed_result_sha},
        "conditioning_schema_version": 3,
        "branch_digests": branch_digests,
    }


def _validate_non_regression_report(
    path: Any,
    subject: ValidatedSmokePlan,
    *,
    root: Path,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    generation_root = Path(subject.plan["output_root"]).resolve()
    report_path, payload, report_sha = _load_external_json(
        path,
        generation_root=generation_root,
        label="structured non-regression test report",
        evidence_root=evidence_root,
    )
    required = {
        "schema_version",
        "subject_plan_sha256",
        "implementation_files_sha256",
        "command",
        "started_at_utc",
        "completed_at_utc",
        "duration_seconds",
        "exit_code",
        "stdout",
        "stderr",
        "test_results",
        "source_files_sha256",
        "junit_xml",
        "structured_output_sha256",
    }
    if set(payload) != required or payload.get("schema_version") != 2:
        raise ValueError("Non-regression structured report has invalid shape.")
    if (
        payload["subject_plan_sha256"] != subject.digest
        or payload["implementation_files_sha256"] != subject.plan["implementation_files_sha256"]
        or payload["exit_code"] != 0
    ):
        raise ValueError("Non-regression report source/exit status failed.")
    command = payload["command"]
    expected_command = [sys.executable, "-m", "pytest", "-q", *REQUIRED_NON_REGRESSION_TESTS]
    if command != expected_command:
        raise ValueError("Non-regression report command is not the exact registered argv.")
    started = datetime.fromisoformat(str(payload["started_at_utc"]).replace("Z", "+00:00"))
    completed_at = datetime.fromisoformat(
        str(payload["completed_at_utc"]).replace("Z", "+00:00")
    )
    duration = payload["duration_seconds"]
    if (
        started.tzinfo is None
        or completed_at.tzinfo is None
        or completed_at < started
        or isinstance(duration, bool)
        or not isinstance(duration, (int, float))
        or not math.isfinite(float(duration))
        or duration <= 0
        or not isinstance(payload["stdout"], str)
        or not isinstance(payload["stderr"], str)
    ):
        raise ValueError("Non-regression report lacks real process timing/stdout/stderr capture.")
    results = payload["test_results"]
    if not isinstance(results, list) or [item.get("nodeid") for item in results] != list(
        REQUIRED_NON_REGRESSION_TESTS
    ):
        raise ValueError("Non-regression report does not bind exact named test cases.")
    if any(set(item) != {"nodeid", "outcome", "duration_seconds"} for item in results):
        raise ValueError("Non-regression structured test record shape is invalid.")
    if any(
        item["outcome"] != "passed"
        or isinstance(item["duration_seconds"], bool)
        or not isinstance(item["duration_seconds"], (int, float))
        or item["duration_seconds"] < 0
        for item in results
    ):
        raise ValueError("A required non-regression test did not pass.")
    expected_sources = {
        str((root / nodeid.split("::", 1)[0]).resolve())
        for nodeid in REQUIRED_NON_REGRESSION_TESTS
    }
    sources = payload["source_files_sha256"]
    if not isinstance(sources, Mapping) or set(sources) != expected_sources:
        raise ValueError("Non-regression report source-file coverage drifted.")
    for raw_source, expected_sha in sources.items():
        source = Path(str(raw_source)).resolve()
        if source != root and root not in source.parents or _sha256_file(source) != expected_sha:
            raise ValueError(f"Non-regression source changed: {source}")
    junit_binding = payload["junit_xml"]
    if (
        not isinstance(junit_binding, Mapping)
        or set(junit_binding) != {"path", "file_sha256"}
    ):
        raise ValueError("Non-regression report lacks exact JUnit XML binding.")
    junit_path = _resolve_evidence_path(
        str(junit_binding.get("path", "")),
        base=evidence_root,
        label="non-regression JUnit XML binding",
    )
    require_external_artifact_path(junit_path, generation_root, "non-regression JUnit XML")
    if paths_overlap(junit_path, report_path) or _sha256_file(junit_path) != junit_binding.get(
        "file_sha256"
    ):
        raise ValueError("Non-regression JUnit XML bytes/path binding drifted.")
    structured_results = parse_non_regression_junit(junit_path)
    if structured_results != results:
        raise ValueError("Non-regression claimed results differ from structured JUnit capture.")
    canonical_output = {
        "command": command,
        "started_at_utc": payload["started_at_utc"],
        "completed_at_utc": payload["completed_at_utc"],
        "duration_seconds": duration,
        "exit_code": payload["exit_code"],
        "stdout": payload["stdout"],
        "stderr": payload["stderr"],
        "test_results": results,
        "source_files_sha256": sources,
        "junit_xml": dict(junit_binding),
    }
    if payload["structured_output_sha256"] != canonical_sha256(canonical_output):
        raise ValueError("Non-regression structured output digest drifted.")
    environment = dict(os.environ)
    environment["PYTEST_ADDOPTS"] = ""
    completed = subprocess.run(
        expected_command,
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if completed.returncode != 0:
        raise ValueError(
            "The evaluator independently executed the registered non-regression tests and "
            f"they failed (exit {completed.returncode}): {completed.stdout}\n{completed.stderr}"
        )
    return {
        "report_path": str(path["path"]),
        "report_sha256": report_sha,
        "validated_test_nodeids": list(REQUIRED_NON_REGRESSION_TESTS),
        "junit_xml": {
            "path": str(junit_binding["path"]),
            "sha256": junit_binding["file_sha256"],
        },
        "independent_live_execution": True,
    }


def parse_non_regression_junit(path: Path) -> list[dict[str, Any]]:
    """Derive the exact registered outcomes from pytest's JUnit XML."""

    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as exc:
        raise ValueError(f"Cannot parse non-regression JUnit XML {path}: {exc}") from exc
    cases = list(root.iter("testcase"))
    expected_by_name = {
        nodeid.split("::", 1)[1]: nodeid for nodeid in REQUIRED_NON_REGRESSION_TESTS
    }
    observed: dict[str, dict[str, Any]] = {}
    for case in cases:
        name = case.get("name")
        if name not in expected_by_name or name in observed:
            raise ValueError("JUnit XML contains an unexpected or duplicated test case.")
        if any(case.find(kind) is not None for kind in ("failure", "error", "skipped")):
            outcome = "failed"
        else:
            outcome = "passed"
        try:
            duration = float(case.get("time", ""))
        except ValueError as exc:
            raise ValueError("JUnit XML testcase has invalid duration.") from exc
        if not math.isfinite(duration) or duration < 0:
            raise ValueError("JUnit XML testcase has non-finite/negative duration.")
        observed[expected_by_name[name]] = {
            "nodeid": expected_by_name[name],
            "outcome": outcome,
            "duration_seconds": duration,
        }
    if set(observed) != set(REQUIRED_NON_REGRESSION_TESTS):
        raise ValueError("JUnit XML does not cover the exact registered non-regression tests.")
    return [observed[nodeid] for nodeid in REQUIRED_NON_REGRESSION_TESTS]


def _derive_evaluation(
    contract: str,
    *,
    subject: ValidatedSmokePlan,
    evidence_index: Mapping[str, Any],
    root: Path,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    if (
        evidence_index["gate_contract"] != contract
        or evidence_index["subject_plan_sha256"] != subject.digest
    ):
        raise ValueError("Gate evidence index contract/subject binding drifted.")
    rows = _selected_rows(contract, subject)
    routing = evidence_index["row_evidence"]
    expected_conditions = [str(job["condition_id"]) for _, _, _, _, job in rows]
    if list(routing) != expected_conditions:
        raise ValueError(
            "Gate row evidence keys/order must equal the exact plan-derived condition list."
        )
    special_fields = {
        "no_generation_report": evidence_index["no_generation_report"],
        "ideogram_conditioning_report": evidence_index["ideogram_conditioning_report"],
        "non_regression_report": evidence_index["non_regression_report"],
    }
    expected_special = {
        NO_GENERATION_GATE: "no_generation_report",
        IDEOGRAM_CONDITIONING_GATE: "ideogram_conditioning_report",
        FULL_PAIR_NON_REGRESSION_GATE: "non_regression_report",
    }.get(contract)
    for field, value in special_fields.items():
        if (field == expected_special) != (value is not None):
            raise ValueError(f"Gate evidence special field {field} is missing or unexpectedly set.")

    derived_rows = []
    for owner, role, manifest, index, job in rows:
        validator = _validate_row if bool(job["expected_media"]) else _validate_unsupported_row
        arguments = {
            "subject": subject,
            "owner": owner,
            "role": role,
            "manifest": manifest,
            "index": index,
            "job": job,
            "routing": routing[job["condition_id"]],
            "root": root,
        }
        if validator is _validate_row:
            arguments["contract"] = contract
            arguments["evidence_root"] = evidence_root
        derived_rows.append(validator(**arguments))
    special: dict[str, Any] | None = None
    if contract == NO_GENERATION_GATE:
        special = _validate_no_generation_report(
            evidence_index["no_generation_report"],
            subject,
            root=root,
            evidence_root=evidence_root,
        )
    elif contract == IDEOGRAM_CONDITIONING_GATE:
        special = _validate_ideogram_conditioning_report(
            evidence_index["ideogram_conditioning_report"],
            subject,
            root=root,
            evidence_root=evidence_root,
        )
    elif contract == FULL_PAIR_NON_REGRESSION_GATE:
        special = _validate_non_regression_report(
            evidence_index["non_regression_report"],
            subject,
            root=root,
            evidence_root=evidence_root,
        )
    exact_counts = {
        "required_rows": len(rows),
        "validated_rows": len(derived_rows),
        "failed_rows": 0,
    }
    if contract == EXACT_ONE_PATH_GATE:
        exact_counts.update(
            {
                "ordinary_exact_one_rows": sum(
                    job["variation"] == ORDINARY_EXACT for _, _, _, _, job in rows
                ),
                "shapley_exact_one_rows": sum(
                    job["variation"] == SHAPLEY_EXACT for _, _, _, _, job in rows
                ),
            }
        )
        if (
            exact_counts["ordinary_exact_one_rows"],
            exact_counts["shapley_exact_one_rows"],
        ) != (6, 12):
            raise ValueError("Derived exact-one gate is not the registered 6+12 cohort.")
    if contract == FINAL_CUMULATIVE_ADMISSION_GATE:
        exact_counts.update(
            {
                "media_rows": sum(row["status"] == "completed_media" for row in derived_rows),
                "unsupported_rows": sum(
                    row["status"] == "truthful_not_supported" for row in derived_rows
                ),
            }
        )
        if (exact_counts["media_rows"], exact_counts["unsupported_rows"]) != (33, 2):
            raise ValueError(
                "Final smoke admission did not validate exact 33 media + 2 unsupported."
            )
    return {
        "counts": exact_counts,
        "rows": derived_rows,
        "special_evidence": special,
        "decision": "pass",
    }


def evaluate_smoke_gate(
    contract: str,
    *,
    subject_plan_path: str | Path,
    evidence_index_path: str | Path,
    output_path: str | Path,
    root: Path | None = None,
    bundle_relative_bindings: bool = False,
) -> dict[str, Any]:
    """Derive, publish, and immediately reopen one immutable gate evaluation."""

    root = (root or project_root()).resolve()
    if contract not in GATE_CONTRACTS:
        raise ValueError(f"Unknown smoke gate contract {contract!r}.")
    subject = validate_smoke_plan(subject_plan_path, root=root)
    generation_root = Path(subject.plan["output_root"]).resolve()
    evidence_path = Path(evidence_index_path).resolve()
    report_path = Path(output_path).resolve()
    for path, label in ((evidence_path, "evidence index"), (report_path, "gate evaluation")):
        require_external_artifact_path(path, generation_root, label)
    if paths_overlap(evidence_path, report_path):
        raise ValueError("Evidence index and evaluator report may not alias or nest.")
    evidence = read_smoke_evidence_index(evidence_path)
    derived = _derive_evaluation(
        contract,
        subject=subject,
        evidence_index=evidence,
        root=root,
        evidence_root=evidence_path.parent,
    )
    evaluator_source = Path(__file__).resolve()
    report: dict[str, Any] = {
        "schema_version": GATE_EVALUATION_SCHEMA_VERSION,
        "contract": contract,
        "evaluation_contract": GATE_EVALUATION_CONTRACT,
        "benchmark": BENCHMARK_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subject_plan": smoke_plan_binding(subject),
        "implementation_files_sha256": subject.plan["implementation_files_sha256"],
        "evaluator_source": str(evaluator_source),
        "evaluator_source_sha256": _sha256_file(evaluator_source),
        "evidence_index": {
            "path": (
                _portable_bundle_path(evidence_path, bundle_root=report_path.parent)
                if bundle_relative_bindings
                else str(evidence_path)
            ),
            "file_sha256": _sha256_file(evidence_path),
            "evidence_index_sha256": evidence["evidence_index_sha256"],
        },
        "derived": derived,
        "decision": derived["decision"],
    }
    report["evaluation_sha256"] = gate_evaluation_digest(report)
    write_authenticated_document(
        report,
        report_path,
        digest_field="evaluation_sha256",
        digest_function=gate_evaluation_digest,
    )
    return read_smoke_gate_evaluation(
        report_path,
        subject=subject,
        expected_contract=contract,
        root=root,
        _allow_uncommitted_bundle=bundle_relative_bindings,
    )


def read_smoke_gate_evaluation(
    path: str | Path,
    *,
    subject: ValidatedSmokePlan,
    expected_contract: str,
    root: Path | None = None,
    _allow_uncommitted_bundle: bool = False,
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    report_path = Path(path).resolve()
    generation_root = Path(subject.plan["output_root"]).resolve()
    require_external_artifact_path(report_path, generation_root, "gate evaluation")
    report = read_authenticated_document(
        report_path,
        digest_field="evaluation_sha256",
        digest_function=gate_evaluation_digest,
        label="smoke gate evaluation",
    )
    required = {
        "schema_version",
        "contract",
        "evaluation_contract",
        "benchmark",
        "created_at_utc",
        "subject_plan",
        "implementation_files_sha256",
        "evaluator_source",
        "evaluator_source_sha256",
        "evidence_index",
        "derived",
        "decision",
        "evaluation_sha256",
    }
    if set(report) != required:
        raise ValueError("Smoke gate evaluation report has invalid shape.")
    if (
        report["schema_version"] != GATE_EVALUATION_SCHEMA_VERSION
        or report["evaluation_contract"] != GATE_EVALUATION_CONTRACT
        or report["contract"] != expected_contract
        or report["benchmark"] != BENCHMARK_NAME
        or report["subject_plan"] != smoke_plan_binding(subject)
        or report["implementation_files_sha256"] != subject.plan["implementation_files_sha256"]
        or report["decision"] != "pass"
    ):
        raise ValueError("Smoke gate evaluation identity/decision binding failed.")
    evaluator_source = Path(report["evaluator_source"]).resolve()
    if evaluator_source != Path(__file__).resolve() or report[
        "evaluator_source_sha256"
    ] != _sha256_file(evaluator_source):
        raise ValueError("Smoke gate evaluator implementation changed.")
    evidence_binding = report["evidence_index"]
    if not isinstance(evidence_binding, Mapping) or set(evidence_binding) != {
        "path",
        "file_sha256",
        "evidence_index_sha256",
    }:
        raise ValueError("Gate evidence-index binding has an invalid shape.")
    if not Path(str(evidence_binding["path"])).is_absolute() and not _allow_uncommitted_bundle:
        from hierasafe_flow.evaluation.production_smoke_collection import (
            validate_committed_smoke_evidence_bundle_root,
        )

        validate_committed_smoke_evidence_bundle_root(
            report_path.parent,
            gate_contract=expected_contract,
            subject_plan_sha256=subject.digest,
        )
    evidence_path = _resolve_evidence_path(
        str(evidence_binding["path"]),
        base=report_path.parent,
        label="gate evidence-index binding",
    )
    require_external_artifact_path(evidence_path, generation_root, "evidence index")
    if (
        paths_overlap(evidence_path, report_path)
        or _sha256_file(evidence_path) != evidence_binding["file_sha256"]
    ):
        raise ValueError("Gate evidence index path/hash binding failed.")
    evidence = read_smoke_evidence_index(evidence_path)
    if evidence["evidence_index_sha256"] != evidence_binding["evidence_index_sha256"]:
        raise ValueError("Gate evidence index canonical identity drifted.")
    derived = _derive_evaluation(
        expected_contract,
        subject=subject,
        evidence_index=evidence,
        root=root,
        evidence_root=evidence_path.parent,
    )
    if derived != report["derived"] or derived["decision"] != report["decision"]:
        raise ValueError("Gate evaluation no longer re-derives exactly from evidence.")
    return report
