"""Immutable, target-blind baseline seed selection for finer detailing.

This module implements the sealed protocol in
``debugging/audits/finer_detailing_20260719/``.  A selection is not a score
spreadsheet: it is a content-addressed proof over eight completed baseline
generations.  The writer reopens every generation artifact, recomputes the
source-only hard-gate admission set, applies the preregistered lexicographic
quality ordering, and publishes one immutable record without overwrite.

Positive steering targets, steering traces/results, later retries, and final
condition quality are deliberately outside this contract.  A row with no
hard-gate-passing seed cannot produce a selection record; the complete
eight-seed cohort must instead be repaired and rerun under a new protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hierasafe_flow.benchmarks.finer_detailing_correction import (
    BENCHMARK_NAME,
    MODEL_NAMES,
    NATIVE_NEGATIVE_UNSUPPORTED_REASONS,
    PAIR_IDS_BY_PROMPT,
    PROMPT_IDS,
    is_flux1_job_v3,
    read_manifest,
    read_manifest_for_audit,
    reopen_completed_flux1_output_v3,
    validate_exact_media,
)
from hierasafe_flow.benchmarks.slurm_tracking import (
    ENVIRONMENT_PREFLIGHT_FILENAME,
    read_environment_preflight,
    read_execution_identity,
    read_submission_registry,
)
from hierasafe_flow.utils.config import load_yaml


SCHEMA_VERSION = 1
SELECTION_NAME = "finer_detailing_target_blind_seed_v1"
REVIEW_NAME = "finer_detailing_target_blind_candidate_source_review_v1"
REVIEW_SCHEMA_VERSION = 1
SEEDS = tuple(range(8))
SELECTION_ROOT_RELATIVE = Path("debugging/seed_selection/finer_detailing_correction")
CANDIDATE_ROOT_RELATIVE = Path("outputs/finer_detailing_seed_qualification")
PROTOCOL_RELATIVE_PATH = Path(
    "debugging/audits/finer_detailing_20260719/target_blind_seed_selection_protocol_v1_20260719.md"
)
PROTOCOL_FILE_SHA256 = "45d5961e617a2353adab9802da3bb6380e3716769260cf515f0900c04a7cb985"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


QUALITY_CRITERIA_IMAGE = (
    "source_semantic_fidelity",
    "identity_and_invariant_stability",
    "required_object_relation_clarity",
    "framing_and_composition_quality",
    "technical_visual_quality",
)
QUALITY_CRITERIA_VIDEO = QUALITY_CRITERIA_IMAGE + ("temporal_motion_and_continuity_quality",)

SOURCE_HARD_GATES: dict[str, tuple[str, ...]] = {
    "01_sad_young_girl": (
        "exactly_one_young_female_identity_count_and_anatomy_stable",
        "unmistakably_sad_downcast_affect",
        "green_jacket_seated_on_clearly_wooden_park_bench",
        "sandwich_visible_and_touching_mouth_during_bite",
        "sunny_park_natural_bright_light_realistic_materials_high_detail",
        "complete_head_torso_both_legs_both_feet_sandwich_and_bench_visible",
        "no_crop_zoom_cut_replacement_duplicates_particles_watermark_logo_or_malformed_anatomy",
    ),
    "02_angry_old_man": (
        "exactly_one_old_male_identity_count_and_anatomy_stable",
        "unmistakably_angry_scowling_affect",
        "green_blazer_seated_on_clearly_metal_park_bench",
        "sandwich_visible_and_touching_mouth_during_bite",
        "cloudy_park_natural_dark_cloudy_light_realistic_materials_high_detail",
        "complete_head_torso_both_legs_both_feet_sandwich_and_bench_visible",
        "no_crop_zoom_cut_replacement_duplicates_particles_watermark_logo_or_malformed_anatomy",
    ),
    "03_empty_outdoor_mall": (
        "completely_empty_mall_without_people_staff_shoppers_or_mannequin_figures",
        "clearly_blue_daytime_sky",
        "two_parallel_clearly_mechanical_escalators_with_required_source_behavior",
        "retail_displays_contain_only_leather_handbags",
        "every_readable_required_signboard_exactly_says_50_percent_sale",
        "horizontal_surface_is_polished_naturally_veined_marble_not_tile",
        "wide_coherent_architecture_contains_all_required_elements_and_route",
    ),
}

VIDEO_HARD_GATES = (
    "exact_h264_mp4_240_unique_rgb_frames_16fps_15_seconds_and_monotonic_pts",
    "authenticated_native_segments_checkpoints_indices_stitch_postprocess_seeds_and_state",
    "no_seam_freeze_periodicity_cadence_slow_motion_ghosting_cut_or_terminal_fade",
    "all_automatic_and_normal_speed_slow_speed_manual_temporal_reviews_pass",
)

CRITERIA_CONTRACT: dict[str, Any] = {
    "schema_version": 1,
    "protocol": SELECTION_NAME,
    "candidate_seed_ids": list(SEEDS),
    "source_hard_gates_by_prompt": {
        prompt_id: list(gates) for prompt_id, gates in SOURCE_HARD_GATES.items()
    },
    "video_hard_gates": list(VIDEO_HARD_GATES),
    "quality_order_image": list(QUALITY_CRITERIA_IMAGE),
    "quality_order_video": list(QUALITY_CRITERIA_VIDEO),
    "ordinal_domain": [0, 1, 2, 3, 4],
    "ordinal_rubric": {
        "0": "unacceptable",
        "1": "major_quality_deficiency",
        "2": "acceptable",
        "3": "strong",
        "4": "excellent",
    },
    "ordinal_direction": "higher_is_better",
    "comparison": "lexicographic_in_registered_order",
    "exact_tie_break": "numerically_smallest_seed",
    "hard_gates_are_noncompensatory": True,
    "no_pass_policy": "no_selection_repair_and_rerun_all_eight",
    "targets_or_steering_may_be_selection_inputs": False,
}


def canonical_sha256(value: Any) -> str:
    """Return the SHA-256 of deterministic, finite canonical JSON."""

    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


CRITERIA_SHA256 = canonical_sha256(CRITERIA_CONTRACT)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def document_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("document_sha256", None)
    return canonical_sha256(canonical)


def hard_gate_ids(prompt_id: str, task: str) -> tuple[str, ...]:
    try:
        source = SOURCE_HARD_GATES[prompt_id]
    except KeyError as exc:
        raise ValueError(f"Unknown finer-detailing prompt_id: {prompt_id!r}.") from exc
    if task == "text_to_image":
        return source
    if task == "text_to_video":
        return source + VIDEO_HARD_GATES
    raise ValueError(f"Unsupported seed-selection generation task: {task!r}.")


def quality_criteria(task: str) -> tuple[str, ...]:
    if task == "text_to_image":
        return QUALITY_CRITERIA_IMAGE
    if task == "text_to_video":
        return QUALITY_CRITERIA_VIDEO
    raise ValueError(f"Unsupported seed-selection generation task: {task!r}.")


def selection_output_path(root: str | Path, prompt_id: str, model_name: str) -> Path:
    _validate_axis(prompt_id, model_name)
    return (
        Path(root).expanduser().resolve()
        / SELECTION_ROOT_RELATIVE
        / f"{prompt_id}__{model_name}__target_blind_seed_v1.json"
    )


def _validate_axis(prompt_id: str, model_name: str) -> None:
    if prompt_id not in PROMPT_IDS:
        raise ValueError(f"Unknown finer-detailing prompt_id: {prompt_id!r}.")
    if model_name not in MODEL_NAMES:
        raise ValueError(f"Unknown finer-detailing model_name: {model_name!r}.")


def _parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not an ISO-8601 timestamp: {value!r}.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include an explicit timezone.")
    return parsed


def _resolve_under_root(path: str | Path, root: Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes project root {root}: {resolved}.") from exc
    return resolved


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _binding(path: Path, *, document_digest: str | None = None) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or resolved.stat().st_size <= 0:
        raise FileNotFoundError(f"Cannot bind missing or empty artifact: {resolved}")
    record: dict[str, Any] = {
        "path": str(resolved),
        "sha256": sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }
    if document_digest is not None:
        if not _SHA256_RE.fullmatch(document_digest):
            raise ValueError(f"Malformed document digest for {resolved}: {document_digest!r}.")
        record["document_sha256"] = document_digest
    return record


def _validate_binding(binding: Any, root: Path, label: str) -> Path:
    if not isinstance(binding, Mapping):
        raise ValueError(f"{label} binding must be a mapping.")
    required = {"path", "sha256", "size_bytes"}
    allowed = required | {
        "document_sha256",
        "manifest_sha256",
        "registry_sha256",
    }
    if not required.issubset(binding):
        raise ValueError(f"{label} binding is missing {sorted(required - set(binding))}.")
    if not set(binding).issubset(allowed):
        raise ValueError(
            f"{label} binding contains unknown fields: {sorted(set(binding) - allowed)}."
        )
    path = _resolve_under_root(str(binding["path"]), root, label)
    if not path.is_file():
        raise FileNotFoundError(f"{label} bound artifact disappeared: {path}")
    expected_sha = str(binding["sha256"])
    if not _SHA256_RE.fullmatch(expected_sha) or sha256_file(path) != expected_sha:
        raise ValueError(f"{label} bound artifact changed: {path}")
    if path.stat().st_size != int(binding["size_bytes"]):
        raise ValueError(f"{label} bound artifact size changed: {path}")
    return path


def _binding_keys_for_role(role: str) -> set[str]:
    keys = {"path", "sha256", "size_bytes"}
    if role == "source_manifest":
        keys.add("manifest_sha256")
    elif role == "submission_registry":
        keys.add("registry_sha256")
    elif role in {
        "manual_review",
        "temporal_evidence",
        "segmented_temporal_audit",
        "full_video_audit",
        "prompt3_motion_audit",
    }:
        keys.add("document_sha256")
    return keys


def _validate_protocol(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    if not resolved.is_file() or sha256_file(resolved) != PROTOCOL_FILE_SHA256:
        raise ValueError(
            "Target-blind seed-selection protocol file is missing or differs from the "
            f"sealed digest: {resolved}."
        )
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Sealed protocol sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").split()
    if fields != [PROTOCOL_FILE_SHA256, resolved.name]:
        raise ValueError(f"Sealed protocol sidecar does not authenticate {resolved}.")
    return {
        "protocol": _binding(resolved),
        "protocol_sidecar": _binding(sidecar),
    }


def _review_source_roles(task: str, prompt_id: str) -> tuple[str, ...]:
    base = ("benchmark_job_result", "media")
    if task == "text_to_image":
        return base
    roles = base + (
        "temporal_evidence",
        "segmented_temporal_audit",
        "full_video_audit",
    )
    if prompt_id == "03_empty_outdoor_mall":
        roles += ("prompt3_motion_audit",)
    return roles


def _normalize_gate_reviews(
    values: Mapping[str, Any], *, prompt_id: str, task: str
) -> dict[str, dict[str, Any]]:
    expected = hard_gate_ids(prompt_id, task)
    if not isinstance(values, Mapping) or set(values) != set(expected):
        found = set(values) if isinstance(values, Mapping) else set()
        raise ValueError(
            "Hard-gate review coverage differs from the registered source-only rubric: "
            f"missing={sorted(set(expected) - found)}, unknown={sorted(found - set(expected))}."
        )
    normalized: dict[str, dict[str, Any]] = {}
    for gate_id in expected:
        item = values[gate_id]
        if not isinstance(item, Mapping) or set(item) != {"passed", "notes"}:
            raise ValueError(
                f"Hard-gate decision {gate_id!r} must contain exactly passed and notes."
            )
        passed = item["passed"]
        notes = str(item["notes"]).strip()
        if not isinstance(passed, bool) or not notes:
            raise ValueError(
                f"Hard-gate decision {gate_id!r} requires a Boolean and non-empty evidence notes."
            )
        normalized[gate_id] = {"passed": passed, "notes": notes}
    return normalized


def _normalize_ordinal_reviews(
    values: Mapping[str, Any], *, task: str
) -> dict[str, dict[str, Any]]:
    expected = quality_criteria(task)
    if not isinstance(values, Mapping) or set(values) != set(expected):
        found = set(values) if isinstance(values, Mapping) else set()
        raise ValueError(
            "Quality-ordinal coverage differs from the registered lexicographic rubric: "
            f"missing={sorted(set(expected) - found)}, unknown={sorted(found - set(expected))}."
        )
    normalized: dict[str, dict[str, Any]] = {}
    for criterion_id in expected:
        item = values[criterion_id]
        if not isinstance(item, Mapping) or set(item) != {"ordinal", "notes"}:
            raise ValueError(
                f"Quality decision {criterion_id!r} must contain exactly ordinal and notes."
            )
        ordinal = item["ordinal"]
        notes = str(item["notes"]).strip()
        if isinstance(ordinal, bool) or not isinstance(ordinal, int) or ordinal not in range(5):
            raise ValueError(f"Quality ordinal {criterion_id!r} must be an integer 0..4.")
        if not notes:
            raise ValueError(f"Quality ordinal {criterion_id!r} needs written evidence notes.")
        normalized[criterion_id] = {"ordinal": ordinal, "notes": notes}
    return normalized


def build_candidate_source_review(
    *,
    prompt_id: str,
    model_name: str,
    task: str,
    seed: int,
    condition_id: str,
    reviewer_identity: str,
    reviewed_at_utc: str,
    hard_gates: Mapping[str, Any],
    quality_ordinals: Mapping[str, Any],
    source_only_evidence_notes: str,
    source_bindings: Mapping[str, Mapping[str, Any]],
    playback: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one content-addressed, source-only manual candidate review.

    The source bindings are re-hashed again while the later selection is
    constructed.  Keeping the review separate makes the human decisions
    immutable before the deterministic winner is computed.
    """

    _validate_axis(prompt_id, model_name)
    if task not in {"text_to_image", "text_to_video"}:
        raise ValueError(f"Unsupported review task: {task!r}.")
    if isinstance(seed, bool) or seed not in SEEDS:
        raise ValueError("Candidate review seed must be one of the exact integers 0..7.")
    reviewer_identity = str(reviewer_identity).strip()
    condition_id = str(condition_id).strip()
    notes = str(source_only_evidence_notes).strip()
    if not reviewer_identity or not condition_id or not notes:
        raise ValueError(
            "Candidate review identity, condition_id, and evidence notes are required."
        )
    _parse_timestamp(reviewed_at_utc, "reviewed_at_utc")
    expected_roles = _review_source_roles(task, prompt_id)
    if not isinstance(source_bindings, Mapping) or set(source_bindings) != set(expected_roles):
        found = set(source_bindings) if isinstance(source_bindings, Mapping) else set()
        raise ValueError(
            "Candidate review source bindings differ from the exact modality contract: "
            f"missing={sorted(set(expected_roles) - found)}, "
            f"unknown={sorted(found - set(expected_roles))}."
        )
    expected_playback = (
        {"original_resolution_viewed": True}
        if task == "text_to_image"
        else {
            "entire_clip_viewed": True,
            "full_speed_viewed": True,
            "slow_motion_viewed": True,
            "viewed_frame_count": 240,
            "segmented_seam_windows_viewed": True,
        }
    )
    if dict(playback) != expected_playback:
        raise ValueError(
            f"Candidate review playback attestation must be exactly {expected_playback}."
        )
    payload: dict[str, Any] = {
        "schema_version": REVIEW_SCHEMA_VERSION,
        "review": REVIEW_NAME,
        "review_status": "completed",
        "benchmark": BENCHMARK_NAME,
        "prompt_id": prompt_id,
        "model_name": model_name,
        "task": task,
        "seed": seed,
        "condition_id": condition_id,
        "reviewer_identity": reviewer_identity,
        "reviewed_at_utc": reviewed_at_utc,
        "hard_gates": _normalize_gate_reviews(hard_gates, prompt_id=prompt_id, task=task),
        "quality_ordinals": _normalize_ordinal_reviews(quality_ordinals, task=task),
        "source_only_evidence_notes": notes,
        "playback": dict(playback),
        "source_bindings": {role: dict(source_bindings[role]) for role in expected_roles},
        "attestation": {
            "baseline_evidence_only": True,
            "no_steering_or_target_evidence_inspected": True,
            "no_later_retry_or_final_condition_evidence_inspected": True,
        },
    }
    payload["document_sha256"] = document_sha256(payload)
    validate_candidate_source_review(payload)
    return payload


def validate_candidate_source_review(
    payload: Mapping[str, Any], *, root: str | Path | None = None
) -> dict[str, Any]:
    expected_keys = {
        "schema_version",
        "review",
        "review_status",
        "benchmark",
        "prompt_id",
        "model_name",
        "task",
        "seed",
        "condition_id",
        "reviewer_identity",
        "reviewed_at_utc",
        "hard_gates",
        "quality_ordinals",
        "source_only_evidence_notes",
        "playback",
        "source_bindings",
        "attestation",
        "document_sha256",
    }
    if set(payload) != expected_keys:
        raise ValueError(
            "Candidate-review fields differ from the exact source-only schema: "
            f"missing={sorted(expected_keys - set(payload))}, "
            f"unknown={sorted(set(payload) - expected_keys)}."
        )
    if (
        payload.get("schema_version") != REVIEW_SCHEMA_VERSION
        or payload.get("review") != REVIEW_NAME
        or payload.get("review_status") != "completed"
        or payload.get("benchmark") != BENCHMARK_NAME
    ):
        raise ValueError("Unsupported target-blind candidate-review schema/name/status.")
    prompt_id = str(payload.get("prompt_id", ""))
    model_name = str(payload.get("model_name", ""))
    task = str(payload.get("task", ""))
    _validate_axis(prompt_id, model_name)
    seed = payload.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise ValueError("Candidate-review seed must be one exact integer in 0..7.")
    declared = str(payload.get("document_sha256", ""))
    if not _SHA256_RE.fullmatch(declared) or declared != document_sha256(payload):
        raise ValueError("Candidate-review canonical document digest mismatch.")
    if not str(payload.get("reviewer_identity", "")).strip():
        raise ValueError("Candidate review lacks reviewer identity.")
    _parse_timestamp(payload.get("reviewed_at_utc"), "reviewed_at_utc")
    if not str(payload.get("source_only_evidence_notes", "")).strip():
        raise ValueError("Candidate review lacks source-only evidence notes.")
    normalized_gates = _normalize_gate_reviews(
        payload.get("hard_gates", {}), prompt_id=prompt_id, task=task
    )
    normalized_ordinals = _normalize_ordinal_reviews(payload.get("quality_ordinals", {}), task=task)
    if payload.get("hard_gates") != normalized_gates:
        raise ValueError("Candidate-review hard-gate representation is not canonical.")
    if payload.get("quality_ordinals") != normalized_ordinals:
        raise ValueError("Candidate-review quality-ordinal representation is not canonical.")
    expected_attestation = {
        "baseline_evidence_only": True,
        "no_steering_or_target_evidence_inspected": True,
        "no_later_retry_or_final_condition_evidence_inspected": True,
    }
    if payload.get("attestation") != expected_attestation:
        raise ValueError("Candidate review lacks the exact target-blind attestation.")
    expected_playback = (
        {"original_resolution_viewed": True}
        if task == "text_to_image"
        else {
            "entire_clip_viewed": True,
            "full_speed_viewed": True,
            "slow_motion_viewed": True,
            "viewed_frame_count": 240,
            "segmented_seam_windows_viewed": True,
        }
    )
    if payload.get("playback") != expected_playback:
        raise ValueError("Candidate-review playback attestation is incomplete.")
    expected_roles = _review_source_roles(task, prompt_id)
    bindings = payload.get("source_bindings")
    if not isinstance(bindings, Mapping) or set(bindings) != set(expected_roles):
        raise ValueError("Candidate-review source binding coverage is invalid.")
    if root is not None:
        resolved_root = Path(root).expanduser().resolve()
        for role in expected_roles:
            if set(bindings[role]) != _binding_keys_for_role(role):
                raise ValueError(f"Candidate-review {role} binding fields are invalid.")
            _validate_binding(bindings[role], resolved_root, f"candidate review {role}")
    return {
        "status": "completed",
        "document_sha256": declared,
        "prompt_id": prompt_id,
        "model_name": model_name,
        "task": task,
        "seed": seed,
        "condition_id": payload.get("condition_id"),
        "eligible": all(item["passed"] for item in normalized_gates.values()),
    }


def _write_text_new_atomic(path: Path, text: str, mode: int = 0o444) -> None:
    """Atomically publish a complete new file; never replace an existing path."""

    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.tmp-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode)
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable_document(path: Path, payload: Mapping[str, Any]) -> tuple[Path, Path]:
    if payload.get("document_sha256") != document_sha256(payload):
        raise ValueError("Cannot publish a document with an invalid canonical digest.")
    resolved = path.expanduser().resolve()
    sidecar = resolved.with_suffix(resolved.suffix + ".sha256")
    if resolved.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite immutable evidence: {resolved}")
    sidecar_published = False
    try:
        _write_text_new_atomic(
            sidecar,
            f"{payload['document_sha256']}  {resolved.name}\n",
        )
        sidecar_published = True
        _write_text_new_atomic(
            resolved,
            json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False)
            + "\n",
        )
    except BaseException:
        if sidecar_published and not resolved.exists():
            sidecar.unlink(missing_ok=True)
        raise
    return resolved, sidecar


def write_candidate_source_review_immutable(
    path: str | Path, payload: Mapping[str, Any], *, root: str | Path
) -> tuple[Path, Path]:
    resolved_root = Path(root).expanduser().resolve()
    validate_candidate_source_review(payload, root=resolved_root)
    resolved = _resolve_under_root(path, resolved_root, "candidate source review output")
    return _write_immutable_document(resolved, payload)


def read_candidate_source_review(path: str | Path, *, root: str | Path) -> dict[str, Any]:
    resolved_root = Path(root).expanduser().resolve()
    resolved = _resolve_under_root(path, resolved_root, "candidate source review")
    payload = _load_json(resolved, "candidate source review")
    validate_candidate_source_review(payload, root=resolved_root)
    _validate_document_sidecar(resolved, str(payload["document_sha256"]), "candidate review")
    return payload


def _validate_document_sidecar(path: Path, digest: str, label: str) -> None:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Immutable {label} sidecar is missing: {sidecar}")
    if sidecar.read_text(encoding="utf-8").split() != [digest, path.name]:
        raise ValueError(f"Immutable {label} sidecar does not authenticate {path}.")
    if path.stat().st_mode & 0o222 or sidecar.stat().st_mode & 0o222:
        raise ValueError(f"Immutable {label} or its sidecar is writable: {path}")


ManifestReader = Callable[[Path, Path], dict[str, Any]]
RegistryReader = Callable[[Path], dict[str, Any]]
MediaValidator = Callable[[Path, dict[str, Any], bool], dict[str, Any]]


def _expected_candidate_spec_keys(task: str, prompt_id: str) -> set[str]:
    keys = {
        "seed",
        "manifest_path",
        "manifest_job_index",
        "submission_registry_path",
        "environment_preflight_path",
        "manual_review_path",
    }
    if task == "text_to_video":
        keys.update(
            {
                "temporal_evidence_path",
                "segmented_temporal_audit_path",
                "full_video_audit_path",
            }
        )
        if prompt_id == "03_empty_outdoor_mall":
            keys.add("prompt3_motion_audit_path")
    return keys


def _manifest_sidecar_binding(manifest_path: Path, manifest_sha256: str) -> dict[str, Any]:
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split() != [
        manifest_sha256,
        manifest_path.name,
    ]:
        raise ValueError(f"Manifest sidecar does not authenticate {manifest_path}.")
    return _binding(sidecar)


def _bound_launch_job(
    job: Mapping[str, Any], manifest_sha256: str, manifest_job_index: int
) -> dict[str, Any]:
    expected = deepcopy(dict(job))
    expected["launch_manifest_sha256"] = manifest_sha256
    if is_flux1_job_v3(expected):
        expected["launch_manifest_job_index"] = manifest_job_index
    return expected


def _validate_registry(
    *,
    path: Path,
    manifest_path: Path,
    manifest_sha256: str,
    job_index: int,
    reader: RegistryReader,
) -> tuple[dict[str, Any], dict[str, Any]]:
    registry = reader(path)
    if (
        registry.get("benchmark") != BENCHMARK_NAME
        or Path(str(registry.get("manifest_path", ""))).resolve() != manifest_path
        or registry.get("manifest_sha256") != manifest_sha256
    ):
        raise ValueError("Submission registry identifies another benchmark or manifest.")
    matches = [
        entry
        for entry in registry.get("submissions", ())
        if isinstance(entry, Mapping)
        and entry.get("manifest_sha256") == manifest_sha256
        and entry.get("job_index") == job_index
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Submission registry must own manifest job {job_index} exactly once; got {len(matches)}."
        )
    return registry, dict(matches[0])


def _validate_execution_identity(
    path: Path, *, registry: Mapping[str, Any], registry_entry: Mapping[str, Any]
) -> dict[str, Any]:
    identity = read_execution_identity(path)
    expected = {
        "SLURM_ARRAY_JOB_ID": str(registry_entry["slurm_array_job_id"]),
        "SLURM_ARRAY_TASK_ID": str(registry_entry["slurm_array_task_id"]),
        "slurm_task_id": str(registry_entry["slurm_task_id"]),
        "SLURM_JOB_NAME": registry.get("slurm_job_name"),
    }
    observed = {key: identity.get(key) for key in expected}
    if observed != expected:
        raise ValueError(f"Execution identity differs from submission registry: {observed}.")
    return identity


def _finite_nonnegative(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite numeric value.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite numeric value.") from exc
    if not math.isfinite(numeric) or numeric < 0 or (positive and numeric <= 0):
        raise ValueError(
            f"{label} must be {'positive' if positive else 'non-negative'} and finite."
        )
    return numeric


def _validate_timing_documents(
    *,
    output_dir: Path,
    expected_job: Mapping[str, Any],
    manifest_sha256: str,
) -> tuple[dict[str, dict[str, Any]], datetime]:
    paths = {
        "experiment_timing": output_dir / "experiment_timing.json",
        "run_timing": output_dir / "run_timing.json",
        "sample_timing": output_dir / "sample_0000" / "timing.json",
    }
    documents = {role: _load_json(path, role.replace("_", " ")) for role, path in paths.items()}
    experiment = documents["experiment_timing"]
    if experiment.get("status") != "completed":
        raise ValueError("Experiment timing is not completed.")
    started = _parse_timestamp(experiment.get("started_at_utc"), "experiment started_at_utc")
    finished = _parse_timestamp(experiment.get("finished_at_utc"), "experiment finished_at_utc")
    if finished < started:
        raise ValueError("Experiment timing ends before it starts.")
    _finite_nonnegative(experiment.get("wall_seconds"), "experiment wall_seconds", positive=True)
    for role in ("run_timing", "sample_timing"):
        document = documents[role]
        if document.get("status") != "completed":
            raise ValueError(f"{role} is not completed.")
        _finite_nonnegative(document.get("total_seconds"), f"{role} total_seconds", positive=True)
        benchmark = document.get("benchmark")
        if not isinstance(benchmark, Mapping):
            raise ValueError(f"{role} lacks benchmark identity.")
        expected_benchmark = {
            "condition_id": expected_job["condition_id"],
            "prompt_id": expected_job["prompt_id"],
            "seed": expected_job["seed"],
            "attempt": expected_job["attempt"],
            "manifest_sha256": manifest_sha256,
            "model_revision": expected_job["model_revision"],
        }
        if any(benchmark.get(key) != value for key, value in expected_benchmark.items()):
            raise ValueError(f"{role} benchmark identity differs from the manifest job.")
    if documents["sample_timing"].get("sample_id") != "sample_0000":
        raise ValueError("Sample timing does not identify sample_0000.")
    return {role: _binding(path) for role, path in paths.items()}, finished


def _validate_system_info(path: Path) -> dict[str, Any]:
    payload = _load_json(path, "system information")
    devices = payload.get("devices")
    if (
        payload.get("cuda_available") is not True
        or isinstance(payload.get("device_count"), bool)
        or int(payload.get("device_count", 0)) < 1
        or not isinstance(devices, list)
        or len(devices) != int(payload["device_count"])
    ):
        raise ValueError("System information does not authenticate a CUDA generation host.")
    for key in ("python", "torch", "platform", "cuda"):
        if not str(payload.get(key, "")).strip():
            raise ValueError(f"System information lacks {key!r} identity.")
    return payload


def _validate_environment_preflight(
    path: Path, *, job: Mapping[str, Any], manifest_sha256: str, job_index: int
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.name != ENVIRONMENT_PREFLIGHT_FILENAME:
        raise ValueError(
            f"Environment preflight must use the canonical filename: {path}"
        )
    payload = read_environment_preflight(
        path.parent,
        expected_job=job,
        expected_job_index=job_index,
    )
    required_keys = {
        "schema_version",
        "status",
        "model_name",
        "condition_id",
        "output_dir",
        "manifest_sha256",
        "manifest_job_index",
        "captured_at_utc",
        "environment",
        "diffusers",
        "runtime_distributions",
        "temporal_metric_contract",
        "temporal_metric_runtime_preflight",
        "source_contract",
    }
    allowed_keys = required_keys | {
        "wan_native_negative_prompt_cleaner",
        "common_seed_launch_authorization",
        "qualification_launch_authorization",
    }
    if not required_keys.issubset(payload) or not set(payload).issubset(allowed_keys):
        raise ValueError(
            "Environment preflight field coverage is invalid: "
            f"missing={sorted(required_keys - set(payload))}, "
            f"unknown={sorted(set(payload) - allowed_keys)}."
        )
    if (
        isinstance(payload.get("schema_version"), bool)
        or payload.get("schema_version") != 2
        or payload.get("status") != "verified_before_generation"
        or payload.get("model_name") != job["model_name"]
    ):
        raise ValueError("Environment preflight schema/status/model identity is invalid.")
    if payload["manifest_sha256"] != manifest_sha256:
        raise ValueError("Environment preflight manifest binding differs from the candidate.")
    if payload["manifest_job_index"] != job_index:
        raise ValueError("Environment preflight job-index binding differs from the candidate.")
    _parse_timestamp(payload["captured_at_utc"], "environment preflight captured_at_utc")
    environment = payload.get("environment")
    diffusers = payload.get("diffusers")
    runtime = payload.get("runtime_distributions")
    temporal = payload.get("temporal_metric_contract")
    if not all(isinstance(value, Mapping) for value in (environment, diffusers, runtime, temporal)):
        raise ValueError("Environment preflight lacks its authenticated runtime identity.")
    common_identity = deepcopy(payload)
    for key in (
        "manifest_sha256",
        "manifest_job_index",
        "condition_id",
        "output_dir",
        "captured_at_utc",
    ):
        common_identity.pop(key, None)
    return payload, common_identity


def _validate_resolved_config(
    path: Path, *, job: Mapping[str, Any], manifest_sha256: str, output_dir: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_yaml(path)
    if not isinstance(config, dict):
        raise ValueError("Resolved configuration must be a mapping.")
    project = config.get("project") or {}
    generation = config.get("generation") or {}
    benchmark = config.get("benchmark") or {}
    model = config.get("model") or {}
    steering = config.get("steering") or {}
    logging = config.get("logging") or {}
    if project.get("seed") != job["seed"]:
        raise ValueError("Resolved configuration seed differs from the manifest job.")
    for key, value in job["generation"].items():
        if generation.get(key) != value:
            raise ValueError(f"Resolved generation configuration field {key!r} drifted.")
    if generation.get("prompt") not in {None, job["prompt"]}:
        raise ValueError("Resolved configuration prompt differs from the manifest job.")
    if (
        benchmark.get("condition_id") != job["condition_id"]
        or benchmark.get("prompt_id") != job["prompt_id"]
        or benchmark.get("seed") != job["seed"]
        or benchmark.get("attempt") != job["attempt"]
        or benchmark.get("manifest_sha256") != manifest_sha256
        or benchmark.get("model_revision") != job["model_revision"]
    ):
        raise ValueError("Resolved benchmark configuration identity drifted.")
    if model.get("revision") != job["model_revision"]:
        raise ValueError("Resolved model revision differs from the manifest job.")
    if steering.get("enabled") is not False or steering.get("mode") != "none":
        raise ValueError("A seed-ladder baseline resolved to an enabled steering configuration.")
    if Path(str(logging.get("output_dir", ""))).resolve() != output_dir:
        raise ValueError("Resolved logging output directory differs from the candidate attempt.")

    # Everything not listed here must remain byte-identical across seeds.  The
    # removed fields are exactly the registered seed, seed-scoped identity,
    # immutable attempt, manifest digest, and output directory.
    common = deepcopy(config)
    common.get("project", {}).pop("seed", None)
    common.get("logging", {}).pop("output_dir", None)
    common_benchmark = common.get("benchmark", {})
    for key in ("seed", "condition_id", "attempt", "manifest_sha256"):
        common_benchmark.pop(key, None)
    return config, common


def _source_identity(job: Mapping[str, Any]) -> dict[str, Any]:
    checkpoint_set = job.get("checkpoint_set")
    checkpoint_sha = job.get("checkpoint_set_sha256")
    if checkpoint_set is not None:
        actual = canonical_sha256(checkpoint_set)
        if checkpoint_sha != actual:
            raise ValueError("Candidate checkpoint-set canonical digest is invalid.")
    return {
        "prompt_id": job["prompt_id"],
        "model_name": job["model_name"],
        "task": job["generation"]["task"],
        "prompt_sha256": hashlib.sha256(str(job["prompt"]).encode("utf-8")).hexdigest(),
        "prompt_snapshot_sha256": canonical_sha256(job["prompt_snapshot"]),
        "concept_tree_snapshot_sha256": canonical_sha256(job["concept_tree_snapshot"]),
        "input_files_sha256": canonical_sha256(job["input_files"]),
        "generation_sha256": canonical_sha256(job["generation"]),
        "model_revision": job["model_revision"],
        "checkpoint_set": checkpoint_set,
        "checkpoint_set_sha256": checkpoint_sha,
        "temporal_protocol_snapshot_sha256": (
            canonical_sha256(job["temporal_protocol_snapshot"])
            if job.get("temporal_protocol_snapshot") is not None
            else None
        ),
    }


def _common_provenance(
    *,
    job: Mapping[str, Any],
    resolved_config_common: Mapping[str, Any],
    environment_common: Mapping[str, Any],
    system_identity: Mapping[str, Any],
) -> dict[str, Any]:
    # Compare the *entire* launch-bound job after removing only fields the
    # sealed ladder explicitly allows to differ: the generation seed, the
    # preregistered seed+1 attempt, their seed-scoped path/condition labels,
    # the per-manifest digest, and the per-manifest snapshot location.  Using
    # an allow-list of retained fields would let a newly added job field escape
    # the common-provenance proof.
    job_protocol_core = deepcopy(dict(job))
    for key in (
        "seed",
        "attempt",
        "condition_id",
        "output_dir",
        "variation_dir",
        "launch_manifest_sha256",
        "snapshot_bundle",
    ):
        job_protocol_core.pop(key, None)
    return {
        # Store only canonical digests of automatic provenance payloads in the
        # human-facing selection record.  Some frozen job/config documents also
        # contain the benchmark's later positive-target metadata; hashing them
        # proves equality without exposing that metadata to seed review.
        "job_protocol_core_sha256": canonical_sha256(job_protocol_core),
        "resolved_config_core_sha256": canonical_sha256(resolved_config_common),
        "environment_identity_sha256": canonical_sha256(environment_common),
        "system_identity_sha256": canonical_sha256(system_identity),
        "implementation_files_sha256": job.get("implementation_files_sha256"),
        "source_identity": _source_identity(job),
    }


def _validate_document_identity(
    path: Path,
    *,
    role: str,
    condition_id: str,
    media_sha256: str,
) -> str:
    payload = _load_json(path, role.replace("_", " "))
    declared = str(payload.get("document_sha256", ""))
    if not _SHA256_RE.fullmatch(declared):
        raise ValueError(f"{role} lacks a canonical document SHA-256.")
    # Temporal evidence sidecars identify the file digest, while audit
    # documents identify canonical content.  Their dedicated readers below
    # enforce the precise per-schema meaning.
    identified_condition = payload.get("condition_id")
    if identified_condition is None:
        identified_condition = (payload.get("condition") or {}).get("condition_id")
    if identified_condition not in {None, condition_id}:
        raise ValueError(f"{role} identifies another condition: {identified_condition!r}.")
    bindings = payload.get("source_bindings") or payload.get("binding") or {}
    for key in ("video", "media"):
        item = bindings.get(key) if isinstance(bindings, Mapping) else None
        if isinstance(item, Mapping) and item.get("sha256") not in {None, media_sha256}:
            raise ValueError(f"{role} media binding differs from the candidate.")
    return declared


def _validate_video_evidence(
    *,
    spec: Mapping[str, Any],
    root: Path,
    job: Mapping[str, Any],
    manifest_sha256: str,
    media_sha256: str,
) -> dict[str, dict[str, Any]]:
    from hierasafe_flow.evaluation.full_video import validate_audit_document
    from hierasafe_flow.evaluation.prompt3_video_motion import (
        validate_audit_document as validate_prompt3_motion_document,
    )
    from hierasafe_flow.evaluation.segmented_temporal import (
        SEGMENTED_MODELS,
        validate_segmented_temporal_audit,
    )
    from hierasafe_flow.generation.temporal_artifacts import read_temporal_evidence

    paths = {
        "temporal_evidence": _resolve_under_root(
            spec["temporal_evidence_path"], root, "temporal evidence"
        ),
        "segmented_temporal_audit": _resolve_under_root(
            spec["segmented_temporal_audit_path"], root, "segmented temporal audit"
        ),
        "full_video_audit": _resolve_under_root(
            spec["full_video_audit_path"], root, "full-video audit"
        ),
    }
    if job["prompt_id"] == "03_empty_outdoor_mall":
        paths["prompt3_motion_audit"] = _resolve_under_root(
            spec["prompt3_motion_audit_path"], root, "Prompt-03 motion audit"
        )

    evidence = read_temporal_evidence(paths["temporal_evidence"])
    if (
        evidence.get("condition_id") != job["condition_id"]
        or evidence.get("manifest_sha256") != manifest_sha256
        or (evidence.get("binding") or {}).get("final_media_sha256") != media_sha256
    ):
        raise ValueError("Temporal-evidence identity differs from the baseline candidate.")
    _validate_segment_seed_derivation(evidence, int(job["seed"]))

    segmented = _load_json(paths["segmented_temporal_audit"], "segmented temporal audit")
    if job["model_name"] in SEGMENTED_MODELS:
        segmented_validation = validate_segmented_temporal_audit(
            segmented, verify_source_files=True
        )
        if segmented_validation["condition_id"] != job["condition_id"]:
            raise ValueError("Segmented temporal audit identifies another condition.")
        segmented_document_sha = str(segmented_validation["document_sha256"])
    else:
        # The sealed selection protocol requires segmented automatic evidence
        # for every video.  Non-schema-2 routes therefore need the generic
        # passed contract rather than silently omitting this evidence.
        if (
            segmented.get("schema_version") != 1
            or segmented.get("audit") != "finer_detailing_segmented_temporal_audit_v1"
            or segmented.get("status") != "passed"
            or segmented.get("condition_id") != job["condition_id"]
            or segmented.get("document_sha256") != document_sha256(segmented)
        ):
            raise ValueError("Non-schema-2 video lacks a passed generic segmented audit.")
        segmented_document_sha = str(segmented["document_sha256"])

    full = _load_json(paths["full_video_audit"], "full-video audit")
    full_validation = validate_audit_document(full, verify_source_files=True)
    if full_validation["condition_id"] != job["condition_id"]:
        raise ValueError("Full-video audit identifies another condition.")
    frame_hashes = (full.get("full_decode") or {}).get("rgb_frame_sha256")
    if (
        not isinstance(frame_hashes, list)
        or len(frame_hashes) != 240
        or len(set(frame_hashes)) != 240
    ):
        raise ValueError("Full-video evidence does not contain 240 unique decoded RGB frames.")

    documents = {
        "temporal_evidence": _binding(
            paths["temporal_evidence"], document_digest=str(evidence["document_sha256"])
        ),
        "segmented_temporal_audit": _binding(
            paths["segmented_temporal_audit"], document_digest=segmented_document_sha
        ),
        "full_video_audit": _binding(
            paths["full_video_audit"],
            document_digest=str(full_validation["document_sha256"]),
        ),
    }
    if "prompt3_motion_audit" in paths:
        motion = _load_json(paths["prompt3_motion_audit"], "Prompt-03 motion audit")
        motion_validation = validate_prompt3_motion_document(motion, verify_source_files=True)
        if motion_validation["condition_id"] != job["condition_id"]:
            raise ValueError("Prompt-03 motion audit identifies another condition.")
        if (motion.get("condition_assessment") or {}).get("motion_requirement_status") != "pass":
            raise ValueError("Prompt-03 baseline does not pass opposing escalator-motion evidence.")
        documents["prompt3_motion_audit"] = _binding(
            paths["prompt3_motion_audit"],
            document_digest=str(motion_validation["document_sha256"]),
        )
    return documents


def _validate_segment_seed_derivation(evidence: Mapping[str, Any], base_seed: int) -> None:
    segments = evidence.get("segments")
    if not isinstance(segments, list) or not segments:
        raise ValueError("Temporal evidence contains no authenticated native segments.")
    observed = [segment.get("segment_seed") for segment in segments if isinstance(segment, Mapping)]
    if len(observed) != len(segments) or observed[0] != base_seed:
        raise ValueError("Temporal evidence first segment does not use the registered base seed.")
    if len(observed) == 1:
        return
    protocol = evidence.get("temporal_protocol") or {}
    domain = protocol.get("seed_domain") or protocol.get("segment_seed_domain")
    if not isinstance(domain, str) or not domain:
        raise ValueError("Segmented temporal protocol lacks its deterministic seed domain.")
    expected = [base_seed]
    for segment_index in range(1, len(observed)):
        payload = f"{domain}|{base_seed}|{segment_index}".encode("utf-8")
        expected.append(int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1))
    if observed != expected:
        raise ValueError(
            f"Temporal segment seeds differ from the registered derivation: {observed} != {expected}."
        )


def _collect_candidate(
    spec: Mapping[str, Any],
    *,
    prompt_id: str,
    model_name: str,
    root: Path,
    manifest_reader: ManifestReader,
    registry_reader: RegistryReader,
    media_validator: MediaValidator,
) -> tuple[dict[str, Any], dict[str, Any], datetime, datetime]:
    base_keys = {
        "seed",
        "manifest_path",
        "manifest_job_index",
        "submission_registry_path",
        "environment_preflight_path",
        "manual_review_path",
    }
    if not isinstance(spec, Mapping) or not base_keys.issubset(spec):
        found = set(spec) if isinstance(spec, Mapping) else set()
        raise ValueError(f"Candidate specification is missing {sorted(base_keys - found)}.")
    seed = spec["seed"]
    if isinstance(seed, bool) or not isinstance(seed, int) or seed not in SEEDS:
        raise ValueError("Candidate seed must be one exact integer in 0..7.")
    manifest_path = _resolve_under_root(spec["manifest_path"], root, "candidate manifest")
    manifest = manifest_reader(manifest_path, root)
    if manifest.get("benchmark") != BENCHMARK_NAME or manifest.get("seed") != seed:
        raise ValueError("Candidate manifest benchmark/seed identity is invalid.")
    if (
        manifest.get("attempt") != seed + 1
        or manifest.get("seed_scoped_output") is not True
        or Path(str(manifest.get("output_root", ""))).resolve()
        != (root / CANDIDATE_ROOT_RELATIVE).resolve()
    ):
        raise ValueError(
            "Seed-ladder manifest must use the exact candidate root, seed-scoped output, "
            "and attempt seed+1."
        )
    jobs = manifest.get("jobs")
    job_index = spec["manifest_job_index"]
    if (
        isinstance(job_index, bool)
        or not isinstance(job_index, int)
        or not isinstance(jobs, list)
        or job_index < 0
        or job_index >= len(jobs)
    ):
        raise ValueError("Candidate manifest_job_index is outside the manifest jobs.")
    job = jobs[job_index]
    if not isinstance(job, dict):
        raise ValueError("Candidate manifest job is not a mapping.")
    task = str((job.get("generation") or {}).get("task", ""))
    expected_spec_keys = _expected_candidate_spec_keys(task, prompt_id)
    if set(spec) != expected_spec_keys:
        raise ValueError(
            "Candidate specification keys differ from the exact modality contract: "
            f"missing={sorted(expected_spec_keys - set(spec))}, "
            f"unknown={sorted(set(spec) - expected_spec_keys)}."
        )
    if (
        job.get("prompt_id") != prompt_id
        or job.get("model_name") != model_name
        or job.get("seed") != seed
        or job.get("attempt") != seed + 1
        or job.get("seed_scoped_output") is not True
        or job.get("expected_media") is not True
        or job.get("variant_spec") != {"kind": "baseline"}
        or job.get("variation") != "01_baseline"
        or job.get("variant") != "01_baseline"
    ):
        raise ValueError("Candidate job is not the exact registered baseline ladder row.")
    output_dir = Path(str(job.get("output_dir", ""))).resolve()
    candidate_root = (root / CANDIDATE_ROOT_RELATIVE).resolve()
    if candidate_root not in output_dir.parents:
        raise ValueError("Candidate output directory is outside the exact ladder root.")
    manifest_sha256 = str(manifest.get("manifest_sha256", ""))
    if not _SHA256_RE.fullmatch(manifest_sha256):
        raise ValueError("Candidate manifest lacks its canonical SHA-256.")
    manifest_created = _parse_timestamp(manifest.get("created_at_utc"), "manifest created_at_utc")

    registry_path = _resolve_under_root(
        spec["submission_registry_path"], root, "submission registry"
    )
    registry, registry_entry = _validate_registry(
        path=registry_path,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        job_index=job_index,
        reader=registry_reader,
    )
    expected_job = _bound_launch_job(job, manifest_sha256, job_index)
    job_path = output_dir / "benchmark_job.yaml"
    if load_yaml(job_path) != expected_job:
        raise ValueError("Published benchmark_job.yaml differs from its launch-bound manifest job.")
    result_path = output_dir / "benchmark_job_result.json"
    result = _load_json(result_path, "benchmark job result")
    if (
        result.get("schema_version") != 2
        or result.get("status") != "completed"
        or result.get("job") != expected_job
    ):
        raise ValueError("Candidate result is not a completed launch-bound schema-2 result.")
    if is_flux1_job_v3(expected_job):
        independently_bound_result_sha256 = sha256_file(result_path)
        reopened = reopen_completed_flux1_output_v3(
            expected_job,
            root=root,
            manifest_path=manifest_path,
            manifest_sha256=manifest_sha256,
            manifest_job_index=job_index,
            result_path=result_path,
        )
        if (
            reopened["result"] != result
            or reopened["result_sha256"] != independently_bound_result_sha256
        ):
            raise ValueError("Candidate FLUX-v3 result/hash changed during strict reopening.")
    fresh_validation = media_validator(output_dir, expected_job, True)
    if result.get("media_validation") != fresh_validation:
        raise ValueError("Candidate result media-validation record differs from a fresh decode.")
    media_path = Path(str(fresh_validation["path"])).resolve()
    if result.get("validated_media_paths") != [str(media_path)]:
        raise ValueError("Candidate result does not bind exactly its canonical media path.")

    identity_path = output_dir / "execution_identity.json"
    _validate_execution_identity(identity_path, registry=registry, registry_entry=registry_entry)
    timing_bindings, generation_finished = _validate_timing_documents(
        output_dir=output_dir,
        expected_job=expected_job,
        manifest_sha256=manifest_sha256,
    )
    system_path = output_dir / "system_info.json"
    system_identity = _validate_system_info(system_path)
    config_path = output_dir / "resolved_config.yaml"
    _, common_config = _validate_resolved_config(
        config_path,
        job=expected_job,
        manifest_sha256=manifest_sha256,
        output_dir=output_dir,
    )
    preflight_path = _resolve_under_root(
        spec["environment_preflight_path"], root, "environment preflight"
    )
    _, common_environment = _validate_environment_preflight(
        preflight_path,
        job=expected_job,
        manifest_sha256=manifest_sha256,
        job_index=job_index,
    )

    artifact_bindings: dict[str, dict[str, Any]] = {
        "source_manifest": {
            **_binding(manifest_path),
            "manifest_sha256": manifest_sha256,
        },
        "manifest_sidecar": _manifest_sidecar_binding(manifest_path, manifest_sha256),
        "manifest_snapshot_index": _binding(
            Path(str((manifest.get("snapshot_bundle") or {}).get("index_path", "")))
        ),
        "benchmark_job": _binding(job_path),
        "submission_registry": {
            **_binding(registry_path),
            "registry_sha256": registry["registry_sha256"],
        },
        "execution_identity": _binding(identity_path),
        "environment_preflight": _binding(preflight_path),
        "benchmark_job_result": _binding(result_path),
        "media": _binding(media_path),
        "resolved_config": _binding(config_path),
        "system_info": _binding(system_path),
        **timing_bindings,
    }
    if task == "text_to_video":
        artifact_bindings.update(
            _validate_video_evidence(
                spec=spec,
                root=root,
                job=expected_job,
                manifest_sha256=manifest_sha256,
                media_sha256=str(fresh_validation["sha256"]),
            )
        )

    review_path = _resolve_under_root(spec["manual_review_path"], root, "manual review")
    review = read_candidate_source_review(review_path, root=root)
    review_validation = validate_candidate_source_review(review, root=root)
    expected_review_identity = {
        "prompt_id": prompt_id,
        "model_name": model_name,
        "task": task,
        "seed": seed,
        "condition_id": job["condition_id"],
    }
    if any(review.get(key) != value for key, value in expected_review_identity.items()):
        raise ValueError("Manual source review identifies another candidate.")
    expected_review_roles = _review_source_roles(task, prompt_id)
    expected_review_bindings = {
        "benchmark_job_result": artifact_bindings["benchmark_job_result"],
        "media": artifact_bindings["media"],
        **{
            role: artifact_bindings[role]
            for role in expected_review_roles
            if role not in {"benchmark_job_result", "media"}
        },
    }
    if review.get("source_bindings") != expected_review_bindings:
        raise ValueError(
            "Manual review source bindings differ from authenticated candidate evidence."
        )
    reviewed_at = _parse_timestamp(review["reviewed_at_utc"], "reviewed_at_utc")
    if reviewed_at < generation_finished:
        raise ValueError("Manual review timestamp predates completed candidate generation.")
    artifact_bindings["manual_review"] = _binding(
        review_path, document_digest=str(review["document_sha256"])
    )

    common = _common_provenance(
        job=expected_job,
        resolved_config_common=common_config,
        environment_common=common_environment,
        system_identity=system_identity,
    )
    source_identity = common["source_identity"]
    gate_reviews = deepcopy(review["hard_gates"])
    ordinal_reviews = deepcopy(review["quality_ordinals"])
    eligible = bool(review_validation["eligible"])
    row = {
        "seed": seed,
        "condition_id": job["condition_id"],
        "manifest_sha256": manifest_sha256,
        "manifest_job_index": job_index,
        "result_sha256": artifact_bindings["benchmark_job_result"]["sha256"],
        "media_sha256": artifact_bindings["media"]["sha256"],
        "media_validation_sha256": canonical_sha256(fresh_validation),
        "original_resolution_decode_metadata": deepcopy(fresh_validation),
        "source_identity": source_identity,
        "common_provenance_sha256": canonical_sha256(common),
        "hard_gates": gate_reviews,
        "quality_ordinals": ordinal_reviews,
        "eligible": eligible,
        "reviewer_identity": review["reviewer_identity"],
        "reviewed_at_utc": review["reviewed_at_utc"],
        "source_only_evidence_notes": review["source_only_evidence_notes"],
        "artifact_bindings": artifact_bindings,
    }
    return row, common, reviewed_at, manifest_created


def _ordinal_tuple(row: Mapping[str, Any], task: str) -> tuple[int, ...]:
    ordinals = row.get("quality_ordinals")
    if not isinstance(ordinals, Mapping):
        raise ValueError("Candidate row lacks quality ordinals.")
    return tuple(int(ordinals[criterion]["ordinal"]) for criterion in quality_criteria(task))


def deterministic_selection(
    candidate_rows: Sequence[Mapping[str, Any]], *, task: str
) -> tuple[list[int], int, dict[str, Any]]:
    """Recompute hard-gate eligibility and the registered deterministic winner."""

    if len(candidate_rows) != len(SEEDS) or [row.get("seed") for row in candidate_rows] != list(
        SEEDS
    ):
        raise ValueError("Candidate rows must be ordered exactly as seeds 0..7.")
    eligible: list[int] = []
    for row in candidate_rows:
        gates = row.get("hard_gates")
        if not isinstance(gates, Mapping) or not gates:
            raise ValueError("Candidate row lacks hard-gate decisions.")
        computed = all(
            isinstance(item, Mapping) and item.get("passed") is True for item in gates.values()
        )
        if row.get("eligible") is not computed:
            raise ValueError(f"Candidate seed {row.get('seed')} eligibility is inconsistent.")
        if computed:
            eligible.append(int(row["seed"]))
    if not eligible:
        raise ValueError(
            "No seed 0..7 passes every hard source gate. The sealed no-pass policy forbids "
            "a least-bad selection; repair and rerun all eight seeds under a new protocol."
        )
    by_seed = {int(row["seed"]): row for row in candidate_rows}
    best_tuple = max(_ordinal_tuple(by_seed[seed], task) for seed in eligible)
    tied = [seed for seed in eligible if _ordinal_tuple(by_seed[seed], task) == best_tuple]
    selected = min(tied)
    explanation = {
        "comparison": "lexicographic_in_registered_order",
        "quality_criteria_order": list(quality_criteria(task)),
        "ordinal_direction": "higher_is_better",
        "winning_ordinal_tuple": list(best_tuple),
        "seeds_tied_at_winning_tuple": tied,
        "tie_break": "numerically_smallest_seed",
        "selected_seed": selected,
    }
    return eligible, selected, explanation


def build_selection_record(
    *,
    prompt_id: str,
    model_name: str,
    candidate_specs: Sequence[Mapping[str, Any]],
    selector_identity: str,
    selected_at_utc: str,
    root: str | Path,
    protocol_path: str | Path | None = None,
    final_manifest_paths: Sequence[str | Path] = (),
    manifest_reader: ManifestReader = read_manifest,
    registry_reader: RegistryReader = read_submission_registry,
    media_validator: MediaValidator = validate_exact_media,
) -> dict[str, Any]:
    """Authenticate eight candidates and build one deterministic selection record.

    This is a prospective scientific-publication boundary, so its default reader
    reauthenticates every live launch input.  In particular, a FLUX-v3 candidate
    cannot be selected after its accepted native-equivalence DAG becomes missing
    or stale.  Historical callers must opt in explicitly to
    :func:`read_manifest_for_audit`.
    """

    _validate_axis(prompt_id, model_name)
    resolved_root = Path(root).expanduser().resolve()
    selector_identity = str(selector_identity).strip()
    if not selector_identity:
        raise ValueError("Selection requires a non-empty selector identity.")
    selected_at = _parse_timestamp(selected_at_utc, "selected_at_utc")
    if not isinstance(candidate_specs, Sequence) or isinstance(candidate_specs, (str, bytes)):
        raise ValueError("candidate_specs must be an ordered eight-row sequence.")
    if len(candidate_specs) != len(SEEDS) or [spec.get("seed") for spec in candidate_specs] != list(
        SEEDS
    ):
        raise ValueError("Candidate specifications must be ordered exactly as seeds 0..7.")
    protocol = (
        Path(protocol_path) if protocol_path is not None else resolved_root / PROTOCOL_RELATIVE_PATH
    )
    expected_protocol = (resolved_root / PROTOCOL_RELATIVE_PATH).resolve()
    if protocol.expanduser().resolve() != expected_protocol:
        raise ValueError(
            f"Selection must bind the exact sealed protocol path {expected_protocol}; "
            f"got {protocol}."
        )
    protocol_bindings = _validate_protocol(protocol)

    rows: list[dict[str, Any]] = []
    common_records: list[dict[str, Any]] = []
    review_times: list[datetime] = []
    manifest_times: list[datetime] = []
    for spec in candidate_specs:
        row, common, reviewed_at, manifest_created = _collect_candidate(
            spec,
            prompt_id=prompt_id,
            model_name=model_name,
            root=resolved_root,
            manifest_reader=manifest_reader,
            registry_reader=registry_reader,
            media_validator=media_validator,
        )
        rows.append(row)
        common_records.append(common)
        review_times.append(reviewed_at)
        manifest_times.append(manifest_created)
    common_digest = canonical_sha256(common_records[0])
    if any(canonical_sha256(value) != common_digest for value in common_records[1:]):
        raise ValueError(
            "The eight candidates do not share byte-identical implementation, configuration, "
            "prompt, checkpoint, temporal, and environment provenance."
        )
    if any(row["common_provenance_sha256"] != common_digest for row in rows):
        raise ValueError("Candidate common-provenance digest is internally inconsistent.")
    if selected_at < max(review_times) or selected_at < max(manifest_times):
        raise ValueError("Selection timestamp predates one or more candidate manifests/reviews.")
    task = str(common_records[0]["source_identity"]["task"])
    eligible, selected_seed, explanation = deterministic_selection(rows, task=task)
    source_identity = deepcopy(common_records[0]["source_identity"])
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "selection": SELECTION_NAME,
        "benchmark": BENCHMARK_NAME,
        "prompt_id": prompt_id,
        "model_name": model_name,
        "task": task,
        "criteria_sha256": CRITERIA_SHA256,
        "protocol_bindings": protocol_bindings,
        "prompt_identity": {
            "prompt_sha256": source_identity["prompt_sha256"],
            "prompt_snapshot_sha256": source_identity["prompt_snapshot_sha256"],
            "concept_tree_snapshot_sha256": source_identity["concept_tree_snapshot_sha256"],
        },
        "checkpoint_identity": {
            "model_revision": source_identity["model_revision"],
            "checkpoint_set": source_identity["checkpoint_set"],
            "checkpoint_set_sha256": source_identity["checkpoint_set_sha256"],
        },
        "protocol_identity": {
            "benchmark": BENCHMARK_NAME,
            "selection_protocol_file_sha256": PROTOCOL_FILE_SHA256,
            "criteria_sha256": CRITERIA_SHA256,
            "common_provenance_sha256": common_digest,
            "temporal_protocol_snapshot_sha256": source_identity[
                "temporal_protocol_snapshot_sha256"
            ],
        },
        "common_provenance": common_records[0],
        "common_provenance_sha256": common_digest,
        "candidate_rows": rows,
        "eligible_seed_ids": eligible,
        "selected_seed": selected_seed,
        "deterministic_selection": explanation,
        "selector_identity": selector_identity,
        "selected_at_utc": selected_at_utc,
        "attestation": {
            "all_eight_baseline_candidates_reviewed": True,
            "no_steering_or_target_evidence_inspected": True,
            "selection_completed_before_final_generation": True,
        },
    }
    payload["document_sha256"] = document_sha256(payload)
    validate_selection_payload(payload, root=resolved_root)
    validate_final_manifests(
        payload,
        final_manifest_paths=final_manifest_paths,
        root=resolved_root,
        manifest_reader=manifest_reader,
    )
    return payload


def _required_artifact_roles(task: str, prompt_id: str) -> set[str]:
    roles = {
        "source_manifest",
        "manifest_sidecar",
        "manifest_snapshot_index",
        "benchmark_job",
        "submission_registry",
        "execution_identity",
        "environment_preflight",
        "benchmark_job_result",
        "media",
        "resolved_config",
        "system_info",
        "experiment_timing",
        "run_timing",
        "sample_timing",
        "manual_review",
    }
    if task == "text_to_video":
        roles.update({"temporal_evidence", "segmented_temporal_audit", "full_video_audit"})
        if prompt_id == "03_empty_outdoor_mall":
            roles.add("prompt3_motion_audit")
    return roles


def validate_selection_payload(payload: Mapping[str, Any], *, root: str | Path) -> dict[str, Any]:
    """Validate canonical structure, bindings, provenance, and winner arithmetic."""

    resolved_root = Path(root).expanduser().resolve()
    expected_top_keys = {
        "schema_version",
        "selection",
        "benchmark",
        "prompt_id",
        "model_name",
        "task",
        "criteria_sha256",
        "protocol_bindings",
        "prompt_identity",
        "checkpoint_identity",
        "protocol_identity",
        "common_provenance",
        "common_provenance_sha256",
        "candidate_rows",
        "eligible_seed_ids",
        "selected_seed",
        "deterministic_selection",
        "selector_identity",
        "selected_at_utc",
        "attestation",
        "document_sha256",
    }
    if set(payload) != expected_top_keys:
        raise ValueError(
            "Seed-selection fields differ from the exact target-blind schema: "
            f"missing={sorted(expected_top_keys - set(payload))}, "
            f"unknown={sorted(set(payload) - expected_top_keys)}."
        )
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("selection") != SELECTION_NAME
        or payload.get("benchmark") != BENCHMARK_NAME
    ):
        raise ValueError("Unsupported target-blind seed-selection schema/name/benchmark.")
    prompt_id = str(payload.get("prompt_id", ""))
    model_name = str(payload.get("model_name", ""))
    task = str(payload.get("task", ""))
    _validate_axis(prompt_id, model_name)
    hard_gate_ids(prompt_id, task)
    if payload.get("criteria_sha256") != CRITERIA_SHA256:
        raise ValueError("Seed-selection criteria digest differs from the canonical rubric.")
    declared = str(payload.get("document_sha256", ""))
    if not _SHA256_RE.fullmatch(declared) or declared != document_sha256(payload):
        raise ValueError("Seed-selection canonical document digest mismatch.")
    protocol_bindings = payload.get("protocol_bindings")
    if not isinstance(protocol_bindings, Mapping) or set(protocol_bindings) != {
        "protocol",
        "protocol_sidecar",
    }:
        raise ValueError("Selection does not bind the sealed protocol and sidecar.")
    if any(
        set(protocol_bindings[role]) != {"path", "sha256", "size_bytes"}
        for role in ("protocol", "protocol_sidecar")
    ):
        raise ValueError("Selection protocol binding fields are invalid.")
    protocol_path = _validate_binding(protocol_bindings["protocol"], resolved_root, "protocol")
    _validate_binding(protocol_bindings["protocol_sidecar"], resolved_root, "protocol sidecar")
    if sha256_file(protocol_path) != PROTOCOL_FILE_SHA256:
        raise ValueError("Bound seed-selection protocol differs from the sealed file.")
    if protocol_path != (resolved_root / PROTOCOL_RELATIVE_PATH).resolve():
        raise ValueError("Selection binds a duplicate rather than the exact sealed protocol path.")

    common = payload.get("common_provenance")
    common_digest = str(payload.get("common_provenance_sha256", ""))
    if not isinstance(common, Mapping) or canonical_sha256(common) != common_digest:
        raise ValueError("Selection common-provenance payload/digest is inconsistent.")
    expected_common_keys = {
        "job_protocol_core_sha256",
        "resolved_config_core_sha256",
        "environment_identity_sha256",
        "system_identity_sha256",
        "implementation_files_sha256",
        "source_identity",
    }
    if set(common) != expected_common_keys:
        raise ValueError("Selection common-provenance field coverage is invalid.")
    source_identity = common.get("source_identity")
    expected_source_identity_keys = {
        "prompt_id",
        "model_name",
        "task",
        "prompt_sha256",
        "prompt_snapshot_sha256",
        "concept_tree_snapshot_sha256",
        "input_files_sha256",
        "generation_sha256",
        "model_revision",
        "checkpoint_set",
        "checkpoint_set_sha256",
        "temporal_protocol_snapshot_sha256",
    }
    if (
        not isinstance(source_identity, Mapping)
        or set(source_identity) != expected_source_identity_keys
    ):
        raise ValueError("Selection source-identity field coverage is invalid.")
    rows = payload.get("candidate_rows")
    if not isinstance(rows, list) or len(rows) != len(SEEDS):
        raise ValueError("Selection must contain exactly eight candidate rows.")
    if [row.get("seed") for row in rows if isinstance(row, Mapping)] != list(SEEDS):
        raise ValueError("Selection candidate rows must be ordered exactly as seeds 0..7.")
    seen_media: set[str] = set()
    seen_results: set[str] = set()
    for seed, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"Candidate row {seed} is not a mapping.")
        expected_row_keys = {
            "seed",
            "condition_id",
            "manifest_sha256",
            "manifest_job_index",
            "result_sha256",
            "media_sha256",
            "media_validation_sha256",
            "original_resolution_decode_metadata",
            "source_identity",
            "common_provenance_sha256",
            "hard_gates",
            "quality_ordinals",
            "eligible",
            "reviewer_identity",
            "reviewed_at_utc",
            "source_only_evidence_notes",
            "artifact_bindings",
        }
        if set(row) != expected_row_keys:
            raise ValueError(f"Candidate seed {seed} fields differ from the exact schema.")
        if row.get("common_provenance_sha256") != common_digest:
            raise ValueError(f"Candidate seed {seed} common provenance differs from the cohort.")
        normalized_gates = _normalize_gate_reviews(
            row.get("hard_gates", {}), prompt_id=prompt_id, task=task
        )
        normalized_ordinals = _normalize_ordinal_reviews(row.get("quality_ordinals", {}), task=task)
        if (
            row.get("hard_gates") != normalized_gates
            or row.get("quality_ordinals") != normalized_ordinals
        ):
            raise ValueError(f"Candidate seed {seed} decisions are not canonical.")
        computed_eligible = all(item["passed"] for item in normalized_gates.values())
        if row.get("eligible") is not computed_eligible:
            raise ValueError(f"Candidate seed {seed} eligibility differs from hard gates.")
        if (
            not str(row.get("reviewer_identity", "")).strip()
            or not str(row.get("source_only_evidence_notes", "")).strip()
        ):
            raise ValueError(f"Candidate seed {seed} lacks reviewer/evidence notes.")
        _parse_timestamp(row.get("reviewed_at_utc"), f"candidate {seed} reviewed_at_utc")
        bindings = row.get("artifact_bindings")
        expected_roles = _required_artifact_roles(task, prompt_id)
        if not isinstance(bindings, Mapping) or set(bindings) != expected_roles:
            raise ValueError(f"Candidate seed {seed} artifact binding coverage is invalid.")
        for role in sorted(expected_roles):
            if set(bindings[role]) != _binding_keys_for_role(role):
                raise ValueError(f"Candidate seed {seed} {role} binding fields are invalid.")
            _validate_binding(bindings[role], resolved_root, f"candidate {seed} {role}")
        result_sha = str(bindings["benchmark_job_result"]["sha256"])
        media_sha = str(bindings["media"]["sha256"])
        if row.get("result_sha256") != result_sha or row.get("media_sha256") != media_sha:
            raise ValueError(f"Candidate seed {seed} summary hashes differ from bindings.")
        if result_sha in seen_results or media_sha in seen_media:
            raise ValueError("Seed-selection candidates contain duplicated result or media bytes.")
        seen_results.add(result_sha)
        seen_media.add(media_sha)
        validation = row.get("original_resolution_decode_metadata")
        if not isinstance(validation, Mapping) or canonical_sha256(validation) != row.get(
            "media_validation_sha256"
        ):
            raise ValueError(f"Candidate seed {seed} decode metadata digest is invalid.")
        if (
            validation.get("sha256") != media_sha
            or validation.get("decode_verified") is not True
            or Path(str(validation.get("path", ""))).resolve()
            != Path(str(bindings["media"]["path"])).resolve()
        ):
            raise ValueError(f"Candidate seed {seed} decode metadata differs from media.")
        source_identity = row.get("source_identity")
        if source_identity != common.get("source_identity"):
            raise ValueError(f"Candidate seed {seed} source identity differs from the cohort.")
        manual_path = Path(str(bindings["manual_review"]["path"])).resolve()
        review = read_candidate_source_review(manual_path, root=resolved_root)
        if (
            review.get("seed") != seed
            or review.get("prompt_id") != prompt_id
            or review.get("model_name") != model_name
            or review.get("task") != task
            or review.get("hard_gates") != row.get("hard_gates")
            or review.get("quality_ordinals") != row.get("quality_ordinals")
            or review.get("reviewer_identity") != row.get("reviewer_identity")
            or review.get("reviewed_at_utc") != row.get("reviewed_at_utc")
            or review.get("source_only_evidence_notes") != row.get("source_only_evidence_notes")
            or review.get("document_sha256") != bindings["manual_review"].get("document_sha256")
        ):
            raise ValueError(f"Candidate seed {seed} differs from its immutable manual review.")
    eligible, selected, explanation = deterministic_selection(rows, task=task)
    if payload.get("eligible_seed_ids") != eligible:
        raise ValueError("Selection eligible_seed_ids differ from recomputed hard-gate admission.")
    if payload.get("selected_seed") != selected:
        raise ValueError("Selection selected_seed differs from the deterministic winner.")
    if payload.get("deterministic_selection") != explanation:
        raise ValueError("Selection explanation differs from recomputed lexicographic ordering.")
    if payload.get("attestation") != {
        "all_eight_baseline_candidates_reviewed": True,
        "no_steering_or_target_evidence_inspected": True,
        "selection_completed_before_final_generation": True,
    }:
        raise ValueError("Selection lacks the exact target-blind/final-order attestation.")
    selected_at = _parse_timestamp(payload.get("selected_at_utc"), "selected_at_utc")
    if any(
        _parse_timestamp(row["reviewed_at_utc"], "reviewed_at_utc") > selected_at for row in rows
    ):
        raise ValueError("Selection timestamp predates one or more manual reviews.")
    if not str(payload.get("selector_identity", "")).strip():
        raise ValueError("Selection lacks selector identity.")
    protocol_identity = payload.get("protocol_identity")
    if not isinstance(protocol_identity, Mapping) or protocol_identity != {
        "benchmark": BENCHMARK_NAME,
        "selection_protocol_file_sha256": PROTOCOL_FILE_SHA256,
        "criteria_sha256": CRITERIA_SHA256,
        "common_provenance_sha256": common_digest,
        "temporal_protocol_snapshot_sha256": common["source_identity"][
            "temporal_protocol_snapshot_sha256"
        ],
    }:
        raise ValueError("Selection protocol identity differs from its authenticated cohort.")
    source_identity = common["source_identity"]
    if payload.get("prompt_identity") != {
        "prompt_sha256": source_identity["prompt_sha256"],
        "prompt_snapshot_sha256": source_identity["prompt_snapshot_sha256"],
        "concept_tree_snapshot_sha256": source_identity["concept_tree_snapshot_sha256"],
    }:
        raise ValueError("Selection prompt identity differs from its authenticated cohort.")
    if payload.get("checkpoint_identity") != {
        "model_revision": source_identity["model_revision"],
        "checkpoint_set": source_identity["checkpoint_set"],
        "checkpoint_set_sha256": source_identity["checkpoint_set_sha256"],
    }:
        raise ValueError("Selection checkpoint identity differs from its authenticated cohort.")
    return {
        "status": "valid",
        "document_sha256": declared,
        "prompt_id": prompt_id,
        "model_name": model_name,
        "selected_seed": selected,
        "eligible_seed_ids": eligible,
    }


def revalidate_selection_sources(
    selection: Mapping[str, Any],
    *,
    root: str | Path,
    manifest_reader: ManifestReader = read_manifest_for_audit,
    registry_reader: RegistryReader = read_submission_registry,
    media_validator: MediaValidator = validate_exact_media,
) -> dict[str, Any]:
    """Reopen every candidate and reproduce the complete normalized record.

    This is the evidentiary validator used at publication and every later
    read.  Merely re-hashing the files is insufficient: their internal job,
    scheduler, configuration, decode, timing, review, and temporal identities
    are parsed again and compared with the immutable candidate rows.
    """

    resolved_root = Path(root).expanduser().resolve()
    validate_selection_payload(selection, root=resolved_root)
    prompt_id = str(selection["prompt_id"])
    model_name = str(selection["model_name"])
    task = str(selection["task"])
    stored_rows = selection["candidate_rows"]
    reproduced_rows: list[dict[str, Any]] = []
    reproduced_common: list[dict[str, Any]] = []
    review_times: list[datetime] = []
    manifest_times: list[datetime] = []
    for seed, stored in enumerate(stored_rows):
        bindings = stored["artifact_bindings"]
        spec: dict[str, Any] = {
            "seed": seed,
            "manifest_path": bindings["source_manifest"]["path"],
            "manifest_job_index": stored["manifest_job_index"],
            "submission_registry_path": bindings["submission_registry"]["path"],
            "environment_preflight_path": bindings["environment_preflight"]["path"],
            "manual_review_path": bindings["manual_review"]["path"],
        }
        if task == "text_to_video":
            spec.update(
                {
                    "temporal_evidence_path": bindings["temporal_evidence"]["path"],
                    "segmented_temporal_audit_path": bindings["segmented_temporal_audit"]["path"],
                    "full_video_audit_path": bindings["full_video_audit"]["path"],
                }
            )
            if prompt_id == "03_empty_outdoor_mall":
                spec["prompt3_motion_audit_path"] = bindings["prompt3_motion_audit"]["path"]
        row, common, reviewed_at, manifest_created = _collect_candidate(
            spec,
            prompt_id=prompt_id,
            model_name=model_name,
            root=resolved_root,
            manifest_reader=manifest_reader,
            registry_reader=registry_reader,
            media_validator=media_validator,
        )
        if row != stored:
            raise ValueError(
                f"Recomputed candidate seed {seed} differs from its immutable selection row."
            )
        reproduced_rows.append(row)
        reproduced_common.append(common)
        review_times.append(reviewed_at)
        manifest_times.append(manifest_created)
    if any(common != selection["common_provenance"] for common in reproduced_common):
        raise ValueError("Recomputed candidate common provenance differs from the selection.")
    selected_at = _parse_timestamp(selection["selected_at_utc"], "selected_at_utc")
    if selected_at < max(review_times) or selected_at < max(manifest_times):
        raise ValueError("Selection predates a recomputed candidate manifest or review.")
    eligible, selected_seed, explanation = deterministic_selection(reproduced_rows, task=task)
    if (
        eligible != selection["eligible_seed_ids"]
        or selected_seed != selection["selected_seed"]
        or explanation != selection["deterministic_selection"]
    ):
        raise ValueError("Recomputed source evidence changes the immutable seed decision.")
    return {
        "status": "valid",
        "candidate_sources_reopened": len(reproduced_rows),
        "selected_seed": selected_seed,
        "common_provenance_sha256": selection["common_provenance_sha256"],
    }


def validate_final_manifests(
    selection: Mapping[str, Any],
    *,
    final_manifest_paths: Sequence[str | Path],
    root: str | Path,
    manifest_reader: ManifestReader = read_manifest_for_audit,
) -> dict[str, Any]:
    """Reject supplied final manifests that predate or contradict selection.

    Supplying no manifests is the expected pre-final state.  When manifests are
    supplied, the selected axis must be represented by exactly two single-axis
    manifests: eight standard rows and six Shapley rows.  This intentionally
    rejects partial evidence such as one matching baseline that happens to use
    the selected seed.
    """

    resolved_root = Path(root).expanduser().resolve()
    selected_at = _parse_timestamp(selection.get("selected_at_utc"), "selected_at_utc")
    selected_seed = selection.get("selected_seed")
    prompt_id = selection.get("prompt_id")
    model_name = selection.get("model_name")
    matching_jobs = 0
    matching_manifests: list[tuple[Path, Mapping[str, Any], list[Mapping[str, Any]]]] = []
    seen_paths: set[Path] = set()
    for raw_path in final_manifest_paths:
        path = _resolve_under_root(raw_path, resolved_root, "final manifest")
        if path in seen_paths:
            raise ValueError(f"The same final manifest was supplied more than once: {path}")
        seen_paths.add(path)
        manifest = manifest_reader(path, resolved_root)
        created = _parse_timestamp(manifest.get("created_at_utc"), "final manifest created_at_utc")
        rows = [
            job
            for job in manifest.get("jobs", ())
            if isinstance(job, Mapping)
            and job.get("prompt_id") == prompt_id
            and job.get("model_name") == model_name
        ]
        if not rows:
            continue
        if len(rows) != len(manifest.get("jobs", ())):
            raise ValueError(
                f"Final manifest {path} mixes the selected prompt/model axis with other rows."
            )
        matching_manifests.append((path, manifest, rows))
        matching_jobs += len(rows)
        if created < selected_at:
            raise ValueError(
                f"Final manifest {path} predates immutable seed selection {selected_at.isoformat()}."
            )
        for job in rows:
            if job.get("seed") != selected_seed:
                raise ValueError(
                    f"Final job {job.get('condition_id')} uses seed {job.get('seed')}, "
                    f"but the immutable selection chose {selected_seed}."
                )
            output = Path(str(job.get("output_dir", ""))).resolve()
            expected_root = (
                resolved_root / "outputs/finer_detailing_correction_selected_seed"
            ).resolve()
            if expected_root not in output.parents:
                raise ValueError(
                    f"Supplied final job is outside the clean selected-seed root: {output}"
                )
    if seen_paths and not matching_manifests:
        raise ValueError(
            "Supplied final manifests contain no rows for this immutable seed selection."
        )
    if matching_manifests:
        if len(matching_manifests) != 2 or matching_jobs != 14:
            raise ValueError(
                "A selected prompt/model axis requires exactly two final manifests and "
                f"14 rows; got manifests={len(matching_manifests)}, rows={matching_jobs}."
            )
        expected_root = (
            resolved_root / "outputs/finer_detailing_correction_selected_seed"
        ).resolve()
        expected_pairs = tuple(PAIR_IDS_BY_PROMPT[str(prompt_id)])
        expected_variant_keys = {
            ("baseline", None, None),
            ("native_negative_prompt", None, None),
            ("conceptsteer", "full", None),
            ("shapley_concept_steering", "full", None),
            *{
                ("conceptsteer", "single", pair_id)
                for pair_id in expected_pairs
            },
            *{
                ("shapley_concept_steering", "single", pair_id)
                for pair_id in expected_pairs
            },
        }
        observed_variant_keys: set[tuple[str, str | None, str | None]] = set()
        condition_ids: set[str] = set()
        output_dirs: set[Path] = set()
        family_sizes: list[int] = []
        for path, manifest, rows in matching_manifests:
            if (
                manifest.get("seed") != selected_seed
                or manifest.get("attempt") != 1
                or manifest.get("seed_scoped_output") is not True
                or manifest.get("models") != [model_name]
                or manifest.get("prompt_ids") != [prompt_id]
                or Path(str(manifest.get("output_root", ""))).resolve() != expected_root
            ):
                raise ValueError(
                    f"Final manifest {path} is not an exact attempt-1, selected-seed, "
                    "single-axis manifest in the clean final root."
                )
            family_sizes.append(len(rows))
            for job in rows:
                if (
                    job.get("attempt") != 1
                    or job.get("seed_scoped_output") is not True
                    or job.get("seed") != selected_seed
                ):
                    raise ValueError(
                        f"Final job {job.get('condition_id')} violates attempt/seed scoping."
                    )
                condition_id = str(job.get("condition_id", ""))
                output_dir = Path(str(job.get("output_dir", ""))).resolve()
                if not condition_id or condition_id in condition_ids:
                    raise ValueError("Final selected-axis condition IDs are missing or duplicated.")
                if output_dir in output_dirs:
                    raise ValueError("Final selected-axis output directories are duplicated.")
                condition_ids.add(condition_id)
                output_dirs.add(output_dir)
                spec = job.get("variant_spec")
                if not isinstance(spec, Mapping):
                    raise ValueError("Final selected-axis job lacks a variant_spec mapping.")
                kind = str(spec.get("kind", ""))
                selection_kind = spec.get("pair_selection")
                active = spec.get("active_pair_ids")
                pair_id = (
                    str(active[0])
                    if selection_kind == "single"
                    and isinstance(active, list)
                    and len(active) == 1
                    else None
                )
                key = (kind, selection_kind, pair_id)
                if key in observed_variant_keys:
                    raise ValueError(f"Final selected-axis variant is duplicated: {key}.")
                observed_variant_keys.add(key)
                expected_media = not (
                    kind == "native_negative_prompt"
                    and model_name in NATIVE_NEGATIVE_UNSUPPORTED_REASONS
                )
                if job.get("expected_media") is not expected_media:
                    raise ValueError(
                        f"Final job {condition_id} has the wrong media/unsupported contract."
                    )
        if sorted(family_sizes) != [6, 8]:
            raise ValueError(
                f"Final selected-axis manifest families must contain 6 and 8 rows: {family_sizes}."
            )
        if observed_variant_keys != expected_variant_keys:
            raise ValueError(
                "Final selected-axis variants differ from the exact fourteen-condition matrix: "
                f"missing={sorted(expected_variant_keys - observed_variant_keys)}, "
                f"unknown={sorted(observed_variant_keys - expected_variant_keys)}."
            )
    return {
        "supplied_final_manifests": len(seen_paths),
        "matching_final_manifests": len(matching_manifests),
        "matching_final_jobs": matching_jobs,
    }


def write_selection_record_immutable(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    root: str | Path,
    final_manifest_paths: Sequence[str | Path] = (),
    manifest_reader: ManifestReader = read_manifest,
    registry_reader: RegistryReader = read_submission_registry,
    media_validator: MediaValidator = validate_exact_media,
) -> tuple[Path, Path]:
    """Publish one prospective selection only after strict live revalidation.

    Canonical campaign publication uses the atomic 36-record cohort writer, but
    this lower-level writer remains fail-closed for callers that use it directly.
    Historical inspection can still pass ``read_manifest_for_audit`` explicitly.
    """
    resolved_root = Path(root).expanduser().resolve()
    revalidate_selection_sources(
        payload,
        root=resolved_root,
        manifest_reader=manifest_reader,
        registry_reader=registry_reader,
        media_validator=media_validator,
    )
    validate_final_manifests(
        payload,
        final_manifest_paths=final_manifest_paths,
        root=resolved_root,
        manifest_reader=manifest_reader,
    )
    expected = selection_output_path(
        resolved_root, str(payload["prompt_id"]), str(payload["model_name"])
    )
    resolved = _resolve_under_root(path, resolved_root, "selection output")
    if resolved != expected:
        raise ValueError(f"Selection output path must be exactly {expected}; got {resolved}.")
    return _write_immutable_document(resolved, payload)


def read_selection_record(
    path: str | Path,
    *,
    root: str | Path,
    final_manifest_paths: Sequence[str | Path] = (),
    manifest_reader: ManifestReader = read_manifest_for_audit,
    registry_reader: RegistryReader = read_submission_registry,
    media_validator: MediaValidator = validate_exact_media,
) -> dict[str, Any]:
    resolved_root = Path(root).expanduser().resolve()
    resolved = _resolve_under_root(path, resolved_root, "selection record")
    payload = _load_json(resolved, "selection record")
    revalidate_selection_sources(
        payload,
        root=resolved_root,
        manifest_reader=manifest_reader,
        registry_reader=registry_reader,
        media_validator=media_validator,
    )
    _validate_document_sidecar(resolved, str(payload["document_sha256"]), "seed-selection record")
    validate_final_manifests(
        payload,
        final_manifest_paths=final_manifest_paths,
        root=resolved_root,
        manifest_reader=manifest_reader,
    )
    return payload


def validate_selection_record(
    path: str | Path,
    *,
    root: str | Path,
    final_manifest_paths: Sequence[str | Path] = (),
    manifest_reader: ManifestReader = read_manifest_for_audit,
    registry_reader: RegistryReader = read_submission_registry,
    media_validator: MediaValidator = validate_exact_media,
) -> dict[str, Any]:
    """Authenticate one published record and return a compact validation summary."""

    payload = read_selection_record(
        path,
        root=root,
        final_manifest_paths=final_manifest_paths,
        manifest_reader=manifest_reader,
        registry_reader=registry_reader,
        media_validator=media_validator,
    )
    return {
        "status": "valid",
        "document_sha256": payload["document_sha256"],
        "prompt_id": payload["prompt_id"],
        "model_name": payload["model_name"],
        "selected_seed": payload["selected_seed"],
        "eligible_seed_ids": payload["eligible_seed_ids"],
        "candidate_sources_reopened": 8,
    }


__all__ = [
    "CRITERIA_CONTRACT",
    "CRITERIA_SHA256",
    "PROTOCOL_FILE_SHA256",
    "REVIEW_NAME",
    "SCHEMA_VERSION",
    "SEEDS",
    "SELECTION_NAME",
    "SOURCE_HARD_GATES",
    "VIDEO_HARD_GATES",
    "build_candidate_source_review",
    "build_selection_record",
    "canonical_sha256",
    "deterministic_selection",
    "document_sha256",
    "hard_gate_ids",
    "quality_criteria",
    "read_candidate_source_review",
    "read_selection_record",
    "revalidate_selection_sources",
    "selection_output_path",
    "validate_candidate_source_review",
    "validate_final_manifests",
    "validate_selection_payload",
    "validate_selection_record",
    "write_candidate_source_review_immutable",
    "write_selection_record_immutable",
]
