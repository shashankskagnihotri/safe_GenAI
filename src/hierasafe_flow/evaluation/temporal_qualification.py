"""Content-authenticated promotion evidence for all long-video adapters.

Production approval is deliberately not a collection of caller-supplied
booleans.  A qualifying document covers the exact three-prompt, three-seed,
six-condition pilot grid for one adapter (54 logical runs), re-hashes every
bound artifact, validates full-video and manual evidence, and proves that the
pilot protocol core is identical to the requested production protocol core.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from hierasafe_flow.evaluation.full_video import validate_audit_document
from hierasafe_flow.evaluation.manual_review import validate_manual_review_document
from hierasafe_flow.evaluation.prompt3_video_motion import (
    validate_audit_document as validate_prompt3_motion_document,
)
from hierasafe_flow.evaluation.segmented_temporal import (
    validate_segmented_temporal_audit,
)
from hierasafe_flow.generation.temporal_artifacts import read_temporal_evidence


SCHEMA_VERSION = 1
QUALIFICATION_NAME = "finer_detailing_temporal_qualification_v1"
BENCHMARK_NAME = "finer_detailing_correction_v1"
PROMPT_IDS = (
    "01_sad_young_girl",
    "02_angry_old_man",
    "03_empty_outdoor_mall",
)
SEEDS = (0, 1, 2)
PERSON_PAIR_IDS = (
    "facial_affect_negative_to_happy",
    "body_pose_sitting_to_walking",
    "clothing_color_green_to_red_blue",
    "sandwich_action_eating_to_holding",
    "composition_static_to_dynamic",
)
MALL_PAIR_IDS = (
    "sky_color_blue_to_pink",
    "vertical_circulation_escalators_to_marble_stairs",
    "horizontal_floor_marble_to_tile",
    "signage_sale_to_new_arrival",
    "merchandise_handbags_to_cars",
)
SINGLE_PAIR_BY_PROMPT = {
    "01_sad_young_girl": "facial_affect_negative_to_happy",
    "02_angry_old_man": "facial_affect_negative_to_happy",
    "03_empty_outdoor_mall": "vertical_circulation_escalators_to_marble_stairs",
}
PAIR_IDS_BY_PROMPT = {
    "01_sad_young_girl": PERSON_PAIR_IDS,
    "02_angry_old_man": PERSON_PAIR_IDS,
    "03_empty_outdoor_mall": MALL_PAIR_IDS,
}
VARIANT_GROUPS = (
    "01_baseline",
    "02_negative_prompt",
    "conceptsteer_full",
    "shapley_concept_steering_full",
    "conceptsteer_single",
    "shapley_concept_steering_single",
)
_VARIANT_GROUP_ALIASES = {
    "native_negative_prompt": "02_negative_prompt",
}
_METHOD_KIND_ALIASES: dict[str, str] = {}


def _normalize_variant_group(group: str) -> str:
    return _VARIANT_GROUP_ALIASES.get(group, group)


def _normalize_variant_kind(kind: str) -> str:
    return _METHOD_KIND_ALIASES.get(kind, kind)
NATIVE_NEGATIVE_UNSUPPORTED = {"joyai_echo", "ltx_23"}
TEMPORAL_CRITERIA_BY_MODEL = {
    "cogvideox_5b": (
        "exact_media_contract",
        "semantic_source_fidelity",
        "temporal_continuity",
        "no_freeze_or_repetition",
        "interpolation_quality",
        "steering_trace_nonzero",
        "native_negative_parity",
    ),
    "hunyuan_video": (
        "exact_240_frame_16_cfr",
        "exact_monotonic_pts",
        "no_temporal_repetition",
        "no_slow_motion",
        "temporal_coherence",
        "prompt_semantics",
        "steering_effect",
        "non_target_preservation",
        "native_negative_parity",
    ),
    "joyai_echo": (
        "exact_240_frame_16_cfr",
        "exact_monotonic_pts",
        "video_rope_uses_effective_16_fps",
        "audiovisual_latent_clock_alignment",
        "no_terminal_crop_artifact",
        "no_temporal_repetition",
        "no_slow_motion",
        "temporal_coherence",
        "prompt_semantics",
        "steering_effect",
        "non_target_preservation",
    ),
    "ltx_23": (
        "exact_240_frame_16_cfr",
        "exact_monotonic_pts",
        "video_rope_uses_requested_16_fps",
        "audiovisual_latent_clock_alignment",
        "no_terminal_crop_artifact",
        "no_temporal_repetition",
        "no_slow_motion",
        "temporal_coherence",
        "prompt_semantics",
        "steering_effect",
        "non_target_preservation",
    ),
    "wan22_t2v_a14b": (
        "exact_240_frame_16_cfr",
        "exact_monotonic_pts",
        "verified_primary_and_continuation_artifacts",
        "continuation_seam_quality",
        "no_temporal_repetition",
        "no_slow_motion",
        "temporal_coherence",
        "prompt_semantics",
        "steering_effect",
        "non_target_preservation",
        "native_negative_parity",
    ),
}

COMMON_COMPLETED_BINDINGS = frozenset(
    {
        "source_manifest",
        "benchmark_job_result",
        "benchmark_job",
        "video",
        "steering_trace",
        "sample_report",
        "sample_timing",
        "run_timing",
        "experiment_timing",
        "resolved_config",
        "system_info",
        "execution_identity",
        "full_video_audit",
        "manual_review",
    }
)
SCHEMA2_SEGMENTED_BINDINGS = frozenset(
    {
        "artifact_manifest",
        "temporal_evidence",
        "segmented_temporal_audit",
    }
)
UNSUPPORTED_BINDINGS = frozenset(
    {
        "source_manifest",
        "benchmark_job_result",
        "benchmark_job",
        "run_timing",
        "execution_identity",
    }
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _document_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("document_sha256", None)
    return canonical_sha256(canonical)


def normalized_temporal_protocol(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Remove only approval-phase fields, retaining every generation parameter."""

    normalized = dict(protocol)
    normalized.pop("execution_phase", None)
    normalized.pop("production_gate", None)
    # Historical Wan pilot configs used this name. It is excluded solely so a
    # sealed common gate can replace it without altering the generation core.
    normalized.pop("production_approval", None)
    return normalized


def _variant_spec(prompt_id: str, group: str) -> dict[str, Any]:
    group = _normalize_variant_group(group)
    all_pairs = PAIR_IDS_BY_PROMPT[prompt_id]
    single_pair = SINGLE_PAIR_BY_PROMPT[prompt_id]
    if group == "01_baseline":
        return {
            "variant": "01_baseline",
            "variation": "01_baseline",
            "kind": "baseline",
            "active_pair_ids": (),
        }
    if group == "02_negative_prompt":
        return {
            "variant": "02_negative_prompt",
            "variation": "02_negative_prompt",
            "kind": "native_negative_prompt",
            "active_pair_ids": (),
        }
    if group == "conceptsteer_full":
        return {
            "variant": "conceptsteer_full",
            "variation": "03_concept_steering",
            "kind": "conceptsteer",
            "active_pair_ids": all_pairs,
        }
    if group == "shapley_concept_steering_full":
        return {
            "variant": "shapley_concept_steering_full",
            "variation": "04_shapley_concept_steering",
            "kind": "shapley_concept_steering",
            "active_pair_ids": all_pairs,
        }
    if group == "conceptsteer_single":
        return {
            "variant": f"conceptsteer_single__{single_pair}",
            "variation": "05_concept_steering_single_pair",
            "kind": "conceptsteer",
            "active_pair_ids": (single_pair,),
        }
    if group == "shapley_concept_steering_single":
        return {
            "variant": f"shapley_concept_steering_single__{single_pair}",
            "variation": "06_shapley_concept_steering_single_pair",
            "kind": "shapley_concept_steering",
            "active_pair_ids": (single_pair,),
        }
    raise ValueError(f"Unknown qualification group: {group!r}.")


def expected_run_specs(model_name: str) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    for prompt_id in PROMPT_IDS:
        for seed in SEEDS:
            for group in VARIANT_GROUPS:
                variant = _variant_spec(prompt_id, group)
                run_key = f"{prompt_id}|seed={seed}|variant={variant['variant']}"
                status = (
                    "not_supported"
                    if group == "02_negative_prompt" and model_name in NATIVE_NEGATIVE_UNSUPPORTED
                    else "completed"
                )
                specs[run_key] = {
                    "run_key": run_key,
                    "prompt_id": prompt_id,
                    "seed": seed,
                    "group": group,
                    "status": status,
                    **variant,
                }
    if len(specs) != 54:
        raise AssertionError(f"Qualification contract produced {len(specs)} rather than 54 runs.")
    return specs


def _load_json_binding(
    binding: Mapping[str, Any], *, binding_name: str, verify_file: bool
) -> tuple[Path, dict[str, Any]]:
    path = Path(str(binding.get("path", ""))).expanduser().resolve()
    expected_sha = str(binding.get("sha256", ""))
    if not path.is_file():
        raise FileNotFoundError(f"Qualification binding {binding_name!r} is missing: {path}")
    if verify_file and (not expected_sha or sha256_file(path) != expected_sha):
        raise ValueError(f"Qualification binding changed: {binding_name}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Qualification JSON binding is not an object: {binding_name}.")
    return path, payload


def _validate_manifest_binding(
    manifest_path: Path,
    manifest: Mapping[str, Any],
    result_job: Mapping[str, Any],
) -> None:
    declared = str(manifest.get("manifest_sha256", ""))
    canonical = dict(manifest)
    canonical.pop("manifest_sha256", None)
    if not declared or canonical_sha256(canonical) != declared:
        raise ValueError("Qualification source manifest canonical digest is invalid.")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.read_text(encoding="utf-8").split()[:1] != [declared]:
        raise ValueError("Qualification source manifest sidecar is missing or inconsistent.")
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Qualification source manifest jobs are malformed.")
    condition_id = result_job.get("condition_id")
    matches = [job for job in jobs if job.get("condition_id") == condition_id]
    if len(matches) != 1 or matches[0] != result_job:
        raise ValueError("Qualification manifest does not contain the exact frozen result job.")


def _validate_prompt3_motion_audit(
    audit: Mapping[str, Any],
    *,
    row: Mapping[str, Any],
    bindings: Mapping[str, Any],
) -> None:
    validate_prompt3_motion_document(dict(audit), verify_source_files=True)
    if (
        audit.get("schema_version") != 1
        or audit.get("audit") != "prompt03_video_structural_motion_v1"
        or audit.get("prompt_id") != "03_empty_outdoor_mall"
        or audit.get("structural_contract_pass") is not True
    ):
        raise ValueError("Prompt-03 motion audit failed its structural identity contract.")
    semantics = audit.get("condition_semantics") or {}
    semantics_kind = _normalize_variant_kind(str(semantics.get("variant_kind", "")))
    row_kind = _normalize_variant_kind(str(row.get("variant_kind", "")))
    if semantics_kind != row_kind or list(
        semantics.get("active_pair_ids", ())
    ) != list(row.get("active_pair_ids", ())):
        raise ValueError("Prompt-03 motion audit condition semantics differ from the run.")
    condition = audit.get("condition") or {}
    if condition.get("condition_id") != row.get("condition_id"):
        raise ValueError("Prompt-03 motion audit identifies another qualification condition.")
    assessment = audit.get("condition_assessment") or {}
    if row_kind == "native_negative_prompt":
        if assessment.get("motion_requirement_status") != "not_applicable_to_target_achievement":
            raise ValueError("Prompt-03 native-negative audit incorrectly claims target evidence.")
    elif assessment.get("motion_requirement_status") != "pass":
        raise ValueError("Prompt-03 required circulation motion evidence did not pass.")
    source = audit.get("source_bindings") or {}
    expected = {
        "benchmark_job_result": bindings["benchmark_job_result"]["sha256"],
        "video": bindings["video"]["sha256"],
    }
    for binding_name, digest in expected.items():
        observed = source.get(binding_name) or {}
        if observed.get("sha256") != digest:
            raise ValueError(f"Prompt-03 audit source differs from {binding_name}.")
    decode = audit.get("full_decode") or {}
    if decode.get("decoded_frame_count") != 240 or len(decode.get("rgb_frame_sha256") or ()) != 240:
        raise ValueError("Prompt-03 motion audit lacks complete 240-frame decode evidence.")


def _validate_trace_report(
    row: Mapping[str, Any],
    report: Mapping[str, Any],
    *,
    require_segmented: bool = False,
    expected_segments: Sequence[Mapping[str, Any]] | None = None,
    expected_shapley_binding: Mapping[str, Any] | None = None,
    steering_trace: Sequence[Mapping[str, Any]] | None = None,
) -> None:
    kind = _normalize_variant_kind(str(row["variant_kind"]))
    interpretation = report.get("interpretability") or {}
    if kind == "conceptsteer":
        validation = interpretation.get("conceptsteer_trace_validation")
    elif kind == "shapley_concept_steering":
        validation = interpretation.get("shapley_trace_validation")
    else:
        validation = None

    per_pair: Mapping[str, Any] = {}
    if kind in {"conceptsteer", "shapley_concept_steering"}:
        if not isinstance(validation, Mapping) or validation.get("status") != "passed":
            raise ValueError(f"{kind} report is missing its passed trace validation.")
        if list(validation.get("active_pair_ids", ())) != list(row["active_pair_ids"]):
            raise ValueError(f"{kind} trace validation active pairs differ from the run.")
        observed_per_pair = validation.get("per_pair")
        if not isinstance(observed_per_pair, Mapping) or set(observed_per_pair) != set(
            row["active_pair_ids"]
        ):
            raise ValueError(f"{kind} trace validation pair coverage is incomplete.")
        per_pair = observed_per_pair
        for pair_id, summary in per_pair.items():
            if not isinstance(summary, Mapping) or int(
                summary.get("nonzero_delta_step_count", 0)
            ) <= 0:
                raise ValueError(f"{kind} trace for pair {pair_id!r} is inert.")
        if kind == "shapley_concept_steering":
            if validation.get("schema_version") != 2:
                raise ValueError("Shapley trace validation schema must be version 2.")
            if validation.get("intervention_identity") != (
                "shapley_selected_attribution_weighted_current_to_safe_prediction_rollback"
            ):
                raise ValueError("Shapley trace intervention identity is invalid.")
            positive_fields = (
                "nonzero_delta_step_count",
                "score_decrease_step_count",
                "accepted_token_count",
                "total_score_decrease",
            )
            zero_fields = (
                "post_quantization_trust_cap_violation_count",
                "unselected_bit_change_count",
                "nondecreasing_accepted_token_count",
                "final_active_pair_regression_count",
            )
            for scope, summary in (("run", validation), *per_pair.items()):
                if any(float(summary.get(field, 0.0)) <= 0.0 for field in positive_fields):
                    raise ValueError(
                        f"Shapley {scope} evidence is inert or lacks score descent."
                    )
                if any(int(summary.get(field, -1)) != 0 for field in zero_fields):
                    raise ValueError(f"Shapley {scope} evidence has a validation violation.")
            if expected_shapley_binding is not None:
                _validate_shapley_protocol_bindings(
                    report=report,
                    validation=validation,
                    steering_trace=steering_trace,
                    expected=expected_shapley_binding,
                )

    if not require_segmented:
        return
    if not expected_segments:
        raise ValueError("Segmented trace validation lacks its frozen route contract.")
    segment_validation = interpretation.get("segment_trace_validation")
    if (
        not isinstance(segment_validation, Mapping)
        or segment_validation.get("schema_version") != 1
        or segment_validation.get("status") != "passed"
    ):
        raise ValueError(f"{kind} report lacks passed segment-trace schema 1 evidence.")
    expected_segment_count = len(expected_segments)
    if segment_validation.get("segment_count") != expected_segment_count:
        raise ValueError(f"{kind} segment count differs from the frozen temporal route.")
    observed_segments = segment_validation.get("segments")
    if not isinstance(observed_segments, list) or len(observed_segments) != len(
        expected_segments
    ):
        raise ValueError(f"{kind} segment-trace route coverage is incomplete.")
    for observed, frozen in zip(observed_segments, expected_segments, strict=True):
        if not isinstance(observed, Mapping):
            raise ValueError(f"{kind} segment-trace entry is malformed.")
        if {
            "segment_index": observed.get("segment_index"),
            "model_role": observed.get("model_role"),
            "local_num_steps": observed.get("local_num_steps"),
        } != {
            "segment_index": frozen["segment_index"],
            "model_role": frozen["model_role"],
            "local_num_steps": frozen["local_num_steps"],
        }:
            raise ValueError(f"{kind} segment identity differs from the frozen route.")

    # Baseline and official native-negative runs have no intervention-pair
    # validation, but their exact three-segment execution trace remains
    # mandatory and was checked above.
    if validation is None:
        return
    per_segment = validation.get("per_segment")
    if not isinstance(per_segment, list) or len(per_segment) != expected_segment_count:
        raise ValueError(f"{kind} trace validation lacks every temporal segment.")
    if [entry.get("segment_index") for entry in per_segment] != list(
        range(expected_segment_count)
    ):
        raise ValueError(f"{kind} temporal segment ordering is invalid.")
    for entry in per_segment:
        if entry.get("status") != "passed":
            raise ValueError(f"{kind} per-segment validation is not passed.")
        segment_pairs = entry.get("per_pair")
        if not isinstance(segment_pairs, Mapping) or set(segment_pairs) != set(
            row["active_pair_ids"]
        ):
            raise ValueError(f"{kind} per-segment pair coverage is incomplete.")
        for pair_id, summary in segment_pairs.items():
            if int(summary.get("nonzero_delta_step_count", 0)) <= 0:
                raise ValueError(
                    f"{kind} segment {entry['segment_index']} pair {pair_id!r} is inert."
                )
            if kind == "shapley_concept_steering":
                for field in (
                    "score_decrease_step_count",
                    "accepted_token_count",
                    "total_score_decrease",
                ):
                    if float(summary.get(field, 0.0)) <= 0.0:
                        raise ValueError(
                            f"Shapley segment {entry['segment_index']} pair {pair_id!r} "
                            "lacks score-descent evidence."
                        )
                for field in (
                    "post_quantization_trust_cap_violation_count",
                    "unselected_bit_change_count",
                    "nondecreasing_accepted_token_count",
                    "final_active_pair_regression_count",
                ):
                    if int(summary.get(field, -1)) != 0:
                        raise ValueError(
                            f"Shapley segment {entry['segment_index']} pair {pair_id!r} "
                            "has a validation violation."
                        )


def _validate_shapley_protocol_bindings(
    *,
    report: Mapping[str, Any],
    validation: Mapping[str, Any],
    steering_trace: Sequence[Mapping[str, Any]] | None,
    expected: Mapping[str, Any],
) -> None:
    config = expected.get("shapley_config")
    provenance = expected.get("shapley_provenance")
    if not isinstance(config, Mapping) or not isinstance(provenance, Mapping):
        raise ValueError("Frozen Shapley manifest binding is malformed.")
    canonical_binding = {
        "schema_version": 1,
        "shapley_config": dict(config),
        "shapley_config_sha256": canonical_sha256(config),
        "shapley_provenance": dict(provenance),
        "shapley_provenance_sha256": canonical_sha256(provenance),
    }
    benchmark = report.get("benchmark")
    steering = report.get("steering")
    if not isinstance(benchmark, Mapping) or not isinstance(steering, Mapping):
        raise ValueError("Shapley report lacks runner protocol metadata.")
    if benchmark.get("shapley") != config or steering.get("shapley") != config:
        raise ValueError("Shapley report config differs from the frozen manifest/job.")
    if benchmark.get("shapley_provenance") != provenance:
        raise ValueError("Shapley report provenance differs from the frozen manifest/job.")
    if validation.get("protocol_binding") != canonical_binding:
        raise ValueError("Shapley trace validation protocol binding is not exact.")
    if not isinstance(steering_trace, Sequence) or isinstance(
        steering_trace, (str, bytes)
    ):
        raise ValueError("Shapley raw steering trace is missing or malformed.")
    observed_interventions = 0
    for step in steering_trace:
        if not isinstance(step, Mapping):
            raise ValueError("Shapley raw steering trace step is malformed.")
        concepts = step.get("concepts")
        if not isinstance(concepts, list):
            raise ValueError("Shapley raw steering trace concepts are malformed.")
        for concept in concepts:
            if not isinstance(concept, Mapping):
                raise ValueError("Shapley raw trace concept is malformed.")
            shapley = concept.get("shapley")
            if shapley is None:
                continue
            observed_interventions += 1
            if not isinstance(shapley, Mapping) or shapley.get(
                "protocol_binding"
            ) != canonical_binding:
                raise ValueError("Shapley raw trace protocol binding is not exact.")
            if shapley.get("schema_version") != 2 or shapley.get(
                "intervention_identity"
            ) != provenance.get("intervention_identity"):
                raise ValueError("Shapley raw trace intervention identity is invalid.")
    if observed_interventions == 0:
        raise ValueError("Shapley raw trace contains no bound intervention evidence.")


def _expected_segment_contract(protocol: Mapping[str, Any]) -> list[dict[str, Any]]:
    if protocol.get("schema_version") != 2:
        return []
    raw_segments = protocol.get("segments")
    if isinstance(raw_segments, list):
        expected: list[dict[str, Any]] = []
        for position, raw in enumerate(raw_segments):
            if not isinstance(raw, Mapping):
                raise ValueError("Temporal protocol segment definition is malformed.")
            steps = raw.get("steps", raw.get("denoising_steps"))
            expected.append(
                {
                    "segment_index": position,
                    "model_role": str(raw.get("role", "")),
                    "local_num_steps": int(steps),
                }
            )
            if raw.get("index") != position or not expected[-1]["model_role"]:
                raise ValueError("Temporal protocol segment identity is malformed.")
        if not expected:
            raise ValueError("Temporal protocol schema 2 has no segments.")
        return expected
    count = protocol.get("segment_count")
    roles = protocol.get("segment_roles")
    steps = protocol.get("num_inference_steps_per_segment")
    if (
        isinstance(count, bool)
        or not isinstance(count, int)
        or not isinstance(roles, list)
        or len(roles) != count
        or isinstance(steps, bool)
        or not isinstance(steps, int)
    ):
        raise ValueError("Temporal protocol schema 2 lacks a complete segment contract.")
    return [
        {
            "segment_index": index,
            "model_role": str(role),
            "local_num_steps": steps,
        }
        for index, role in enumerate(roles)
    ]


def _validate_row_files(
    row: Mapping[str, Any],
    *,
    expected: Mapping[str, Any],
    model_name: str,
    model_revision: str,
    protocol_core: Mapping[str, Any],
) -> None:
    bindings = row.get("artifact_bindings")
    if not isinstance(bindings, Mapping):
        raise ValueError(f"Qualification row {row.get('run_key')} lacks artifact_bindings.")
    required = set(
        UNSUPPORTED_BINDINGS if row["status"] == "not_supported" else COMMON_COMPLETED_BINDINGS
    )
    schema2_segmented = protocol_core.get("schema_version") == 2
    if schema2_segmented:
        required.add("artifact_manifest")
        if row["status"] == "completed":
            required.update({"temporal_evidence", "segmented_temporal_audit"})
    if row["status"] == "completed" and row["prompt_id"] == "03_empty_outdoor_mall":
        required.add("prompt3_motion_audit")
    if set(bindings) != required:
        raise ValueError(
            f"Qualification row bindings differ from the exact contract: "
            f"missing={sorted(required - set(bindings))}, "
            f"unknown={sorted(set(bindings) - required)}."
        )
    resolved: dict[str, Path] = {}
    for binding_name, binding in bindings.items():
        if not isinstance(binding, Mapping):
            raise ValueError(f"Qualification binding {binding_name!r} must be a mapping.")
        path = Path(str(binding.get("path", ""))).expanduser().resolve()
        if not path.is_file() or sha256_file(path) != binding.get("sha256"):
            raise ValueError(f"Qualification binding changed: {binding_name}.")
        resolved[binding_name] = path

    result = json.loads(resolved["benchmark_job_result"].read_text(encoding="utf-8"))
    if result.get("status") != row["status"]:
        raise ValueError("Qualification row status differs from benchmark result status.")
    job = result.get("job")
    if not isinstance(job, Mapping):
        raise ValueError("Qualification result is missing its frozen job.")
    manifest = json.loads(resolved["source_manifest"].read_text(encoding="utf-8"))
    _validate_manifest_binding(resolved["source_manifest"], manifest, job)
    job_variant = job.get("variant_spec") or {}
    observed_identity = {
        "prompt_id": job.get("prompt_id"),
        "seed": job.get("seed"),
        "variant": job.get("variant"),
        "variation": job.get("variation"),
        "variant_kind": _normalize_variant_kind(str(job_variant.get("kind", ""))),
        "active_pair_ids": list(job_variant.get("active_pair_ids", ())),
        "model_name": job.get("model_name"),
        "model_revision": job.get("model_revision"),
        "condition_id": job.get("condition_id"),
    }
    wanted_identity = {
        "prompt_id": expected["prompt_id"],
        "seed": expected["seed"],
        "variant": expected["variant"],
        "variation": expected["variation"],
        "variant_kind": expected["kind"],
        "active_pair_ids": list(expected["active_pair_ids"]),
        "model_name": model_name,
        "model_revision": model_revision,
        "condition_id": row.get("condition_id"),
    }
    if observed_identity != wanted_identity:
        raise ValueError(
            f"Qualification result identity differs from its run contract: "
            f"expected={wanted_identity}, observed={observed_identity}."
        )
    temporal = job.get("temporal_protocol_snapshot") or {}
    observed_core = normalized_temporal_protocol(temporal.get("protocol") or {})
    if observed_core != dict(protocol_core):
        raise ValueError("Qualification run temporal protocol core differs from approval core.")
    if schema2_segmented:
        checkpoint_set = job.get("checkpoint_set")
        checkpoint_sha = job.get("checkpoint_set_sha256")
        top_sets = manifest.get("checkpoint_sets_by_model") or {}
        top_shas = manifest.get("checkpoint_set_sha256_by_model") or {}
        if (
            not isinstance(checkpoint_set, list)
            or not isinstance(checkpoint_sha, str)
            or canonical_sha256(checkpoint_set) != checkpoint_sha
            or top_sets.get(model_name) != checkpoint_set
            or top_shas.get(model_name) != checkpoint_sha
            or temporal.get("checkpoint_set") != checkpoint_set
            or temporal.get("checkpoint_set_sha256") != checkpoint_sha
        ):
            raise ValueError("Schema-2 checkpoint-set binding is incomplete or inconsistent.")
        artifact_path = Path(str(job.get("artifact_manifest", ""))).expanduser().resolve()
        artifact_sha = str(job.get("artifact_manifest_sha256", ""))
        if (
            resolved["artifact_manifest"] != artifact_path
            or not artifact_sha
            or bindings["artifact_manifest"].get("sha256") != artifact_sha
        ):
            raise ValueError("Schema-2 artifact-manifest binding is inconsistent.")
    generation = job.get("generation") or {}
    if {
        "task": generation.get("task"),
        "num_frames": generation.get("num_frames"),
        "fps": generation.get("fps"),
        "duration_seconds": float(generation.get("duration_seconds", -1)),
    } != {
        "task": "text_to_video",
        "num_frames": 240,
        "fps": 16,
        "duration_seconds": 15.0,
    }:
        raise ValueError("Qualification result does not use exact 240/16/15 video contract.")

    output_dir = resolved["benchmark_job_result"].parent
    exact_paths = {
        "benchmark_job": output_dir / "benchmark_job.yaml",
        "run_timing": output_dir / "run_timing.json",
        "execution_identity": output_dir / "execution_identity.json",
    }
    if row["status"] == "completed":
        sample_dir = output_dir / "sample_0000"
        exact_paths.update(
            {
                "steering_trace": sample_dir / "steering_trace.json",
                "sample_report": sample_dir / "report.json",
                "sample_timing": sample_dir / "timing.json",
                "experiment_timing": output_dir / "experiment_timing.json",
                "resolved_config": output_dir / "resolved_config.yaml",
                "system_info": output_dir / "system_info.json",
            }
        )
        media_paths = result.get("validated_media_paths")
        if not isinstance(media_paths, list) or len(media_paths) != 1:
            raise ValueError("Completed qualification result must bind exactly one video.")
        media_path = Path(str(media_paths[0])).expanduser().resolve()
        if resolved["video"] != media_path:
            raise ValueError("Qualification video binding differs from result media path.")
        if (result.get("media_validation") or {}).get("sha256") != bindings["video"]["sha256"]:
            raise ValueError("Qualification result/video digest mismatch.")
        if schema2_segmented:
            temporal_evidence = read_temporal_evidence(resolved["temporal_evidence"])
            evidence_metadata = temporal_evidence.get("metadata") or {}
            expected_manifest_sha = str(manifest.get("manifest_sha256", ""))
            if {
                "condition_id": temporal_evidence.get("condition_id"),
                "attempt": temporal_evidence.get("attempt"),
                "manifest_sha256": temporal_evidence.get("manifest_sha256"),
                "temporal_protocol": temporal_evidence.get("temporal_protocol"),
            } != {
                "condition_id": row.get("condition_id"),
                "attempt": job.get("attempt"),
                "manifest_sha256": expected_manifest_sha,
                "temporal_protocol": temporal.get("protocol"),
            }:
                raise ValueError("Schema-2 temporal evidence execution/protocol identity drifted.")
            if (
                evidence_metadata.get("checkpoint_set") != job.get("checkpoint_set")
                or evidence_metadata.get("checkpoint_set_sha256")
                != job.get("checkpoint_set_sha256")
                or Path(str(evidence_metadata.get("artifact_manifest", ""))).resolve()
                != resolved["artifact_manifest"]
                or evidence_metadata.get("artifact_manifest_sha256")
                != job.get("artifact_manifest_sha256")
                or evidence_metadata.get("segmented_temporal_contract")
                != job.get("segmented_temporal_contract")
            ):
                raise ValueError("Schema-2 temporal evidence checkpoint/artifact metadata drifted.")
            segmented = json.loads(
                resolved["segmented_temporal_audit"].read_text(encoding="utf-8")
            )
            segmented_validation = validate_segmented_temporal_audit(
                segmented,
                verify_source_files=True,
            )
            if (
                segmented_validation["condition_id"] != row.get("condition_id")
                or segmented.get("model_name") != model_name
                or segmented.get("manifest_sha256") != expected_manifest_sha
                or segmented_validation.get("qualification_eligible") is not True
                or (segmented.get("source") or {}).get(
                    "temporal_evidence_document_sha256"
                )
                != temporal_evidence["document_sha256"]
            ):
                raise ValueError("Schema-2 segmented audit binding/eligibility is invalid.")
    for binding_name, expected_path in exact_paths.items():
        if resolved[binding_name] != expected_path.resolve():
            raise ValueError(f"Qualification binding {binding_name} has a noncanonical path.")

    if row["status"] == "completed":
        full_audit = json.loads(resolved["full_video_audit"].read_text(encoding="utf-8"))
        full_validation = validate_audit_document(full_audit, verify_source_files=True)
        if full_validation["condition_id"] != row["condition_id"]:
            raise ValueError("Full-video audit identifies another qualification condition.")
        manual = json.loads(resolved["manual_review"].read_text(encoding="utf-8"))
        manual_validation = validate_manual_review_document(manual, verify_source_files=True)
        if manual_validation["condition_id"] != row["condition_id"]:
            raise ValueError("Manual review identifies another qualification condition.")
        if schema2_segmented:
            manual_run = manual.get("run") or {}
            manual_sources = manual.get("source_bindings") or {}
            if manual_run.get("temporal_protocol_schema_version") != 2:
                raise ValueError("Schema-2 manual review uses a stale temporal rubric.")
            if (
                (manual_sources.get("temporal_evidence") or {}).get("sha256")
                != bindings["temporal_evidence"]["sha256"]
                or (manual_sources.get("segmented_temporal_audit") or {}).get("sha256")
                != bindings["segmented_temporal_audit"]["sha256"]
            ):
                raise ValueError("Schema-2 manual review binds different temporal evidence.")
        report = json.loads(resolved["sample_report"].read_text(encoding="utf-8"))
        shapley_binding = None
        steering_trace = None
        if _normalize_variant_kind(str(row["variant_kind"])) == "shapley_concept_steering":
            top_config = manifest.get("shapley_config")
            top_provenance = manifest.get("shapley_provenance")
            if not isinstance(top_config, Mapping) or not isinstance(
                top_provenance, Mapping
            ):
                raise ValueError("Schema-2 Shapley manifest lacks frozen protocol bindings.")
            if job_variant.get("shapley") != top_config or job_variant.get(
                "shapley_provenance"
            ) != top_provenance:
                raise ValueError("Shapley job config/provenance differs from manifest top level.")
            shapley_binding = {
                "shapley_config": dict(top_config),
                "shapley_provenance": dict(top_provenance),
            }
            raw_trace = json.loads(resolved["steering_trace"].read_text(encoding="utf-8"))
            if not isinstance(raw_trace, list):
                raise ValueError("Shapley steering trace must be a list.")
            steering_trace = raw_trace
        _validate_trace_report(
            row,
            report,
            require_segmented=protocol_core.get("schema_version") == 2,
            expected_segments=_expected_segment_contract(protocol_core),
            expected_shapley_binding=shapley_binding,
            steering_trace=steering_trace,
        )
        if row["prompt_id"] == "03_empty_outdoor_mall":
            motion = json.loads(resolved["prompt3_motion_audit"].read_text(encoding="utf-8"))
            _validate_prompt3_motion_audit(motion, row=row, bindings=bindings)


def collect_qualification_rows(
    *,
    manifest_paths: Sequence[str | Path],
    model_name: str,
    model_revision: str,
    temporal_protocol: Mapping[str, Any],
    evidence_root: str | Path,
) -> list[dict[str, Any]]:
    """Collect the exact 54 rows from frozen manifests and canonical evidence paths."""

    expected = expected_run_specs(model_name)
    root = Path(evidence_root).expanduser().resolve()
    collected: dict[str, dict[str, Any]] = {}
    for raw_manifest_path in manifest_paths:
        manifest_path = Path(raw_manifest_path).expanduser().resolve()
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Qualification source manifest is missing: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        jobs = manifest.get("jobs")
        if not isinstance(jobs, list):
            raise ValueError(f"Qualification manifest jobs are malformed: {manifest_path}")
        for job in jobs:
            if not isinstance(job, Mapping) or job.get("model_name") != model_name:
                continue
            prompt_id = str(job.get("prompt_id", ""))
            seed = job.get("seed")
            variant = str(job.get("variant", ""))
            run_key = f"{prompt_id}|seed={seed}|variant={variant}"
            if run_key not in expected:
                continue
            if run_key in collected:
                raise ValueError(f"Qualification manifests duplicate run {run_key!r}.")
            spec = expected[run_key]
            output_dir = Path(str(job.get("output_dir", ""))).expanduser().resolve()
            result_path = output_dir / "benchmark_job_result.json"
            if not result_path.is_file():
                raise FileNotFoundError(f"Qualification result is missing: {result_path}")
            result = json.loads(result_path.read_text(encoding="utf-8"))
            status = str(result.get("status", ""))
            if status != spec["status"]:
                raise ValueError(
                    f"Qualification result {run_key!r} has status {status!r}, "
                    f"expected {spec['status']!r}."
                )

            def bind(path: Path) -> dict[str, str]:
                resolved = path.expanduser().resolve()
                if not resolved.is_file():
                    raise FileNotFoundError(
                        f"Qualification artifact for {run_key!r} is missing: {resolved}"
                    )
                return {"path": str(resolved), "sha256": sha256_file(resolved)}

            bindings: dict[str, dict[str, str]] = {
                "source_manifest": bind(manifest_path),
                "benchmark_job_result": bind(result_path),
                "benchmark_job": bind(output_dir / "benchmark_job.yaml"),
                "run_timing": bind(output_dir / "run_timing.json"),
                "execution_identity": bind(output_dir / "execution_identity.json"),
            }
            schema2_segmented = temporal_protocol.get("schema_version") == 2
            if schema2_segmented:
                bindings["artifact_manifest"] = bind(
                    Path(str(job.get("artifact_manifest", "")))
                )
            if status == "completed":
                media_paths = result.get("validated_media_paths")
                if not isinstance(media_paths, list) or len(media_paths) != 1:
                    raise ValueError(
                        f"Qualification result {run_key!r} does not bind exactly one video."
                    )
                sample_dir = output_dir / "sample_0000"
                condition_id = str(job.get("condition_id", ""))
                run_evidence = root / model_name / condition_id
                bindings.update(
                    {
                        "video": bind(Path(str(media_paths[0]))),
                        "steering_trace": bind(sample_dir / "steering_trace.json"),
                        "sample_report": bind(sample_dir / "report.json"),
                        "sample_timing": bind(sample_dir / "timing.json"),
                        "experiment_timing": bind(output_dir / "experiment_timing.json"),
                        "resolved_config": bind(output_dir / "resolved_config.yaml"),
                        "system_info": bind(output_dir / "system_info.json"),
                        "full_video_audit": bind(run_evidence / "full_video_audit.json"),
                        "manual_review": bind(run_evidence / "manual_review.json"),
                    }
                )
                if schema2_segmented:
                    bindings.update(
                        {
                            "temporal_evidence": bind(
                                sample_dir / "temporal_evidence.json"
                            ),
                            "segmented_temporal_audit": bind(
                                run_evidence / "segmented_temporal_audit.json"
                            ),
                        }
                    )
                if prompt_id == "03_empty_outdoor_mall":
                    bindings["prompt3_motion_audit"] = bind(
                        run_evidence / "prompt3_motion_audit.json"
                    )
            row = {
                "run_key": run_key,
                "condition_id": str(job.get("condition_id", "")),
                "prompt_id": prompt_id,
                "seed": seed,
                "variant": variant,
                "variation": str(job.get("variation", "")),
                "variant_kind": _normalize_variant_kind(
                    str((job.get("variant_spec") or {}).get("kind", ""))
                ),
                "active_pair_ids": list((job.get("variant_spec") or {}).get("active_pair_ids", ())),
                "status": status,
                "artifact_bindings": bindings,
            }
            _validate_row_files(
                row,
                expected=spec,
                model_name=model_name,
                model_revision=model_revision,
                protocol_core=normalized_temporal_protocol(temporal_protocol),
            )
            collected[run_key] = row
    if set(collected) != set(expected):
        raise ValueError(
            "Qualification manifest union does not cover the exact 54-run grid: "
            f"missing={sorted(set(expected) - set(collected))}, "
            f"unknown={sorted(set(collected) - set(expected))}."
        )
    return [collected[key] for key in sorted(collected)]


def build_qualification_document(
    *,
    model_name: str,
    model_revision: str,
    temporal_protocol: Mapping[str, Any],
    criteria_names: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    reviewer: str,
    reviewed_at_utc: str,
) -> dict[str, Any]:
    expected = expected_run_specs(model_name)
    row_keys = [str(row.get("run_key", "")) for row in rows]
    if len(rows) != 54 or set(row_keys) != set(expected) or len(set(row_keys)) != 54:
        raise ValueError("Qualification rows must cover each exact 54-run key once.")
    ordered_rows = [
        dict(next(row for row in rows if row.get("run_key") == key)) for key in sorted(expected)
    ]
    completed_keys = [row["run_key"] for row in ordered_rows if row.get("status") == "completed"]
    criteria_names = tuple(str(value) for value in criteria_names)
    if not criteria_names or len(criteria_names) != len(set(criteria_names)):
        raise ValueError("Qualification criteria_names must be unique and non-empty.")
    protocol_core = normalized_temporal_protocol(temporal_protocol)
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "qualification": QUALIFICATION_NAME,
        "benchmark": BENCHMARK_NAME,
        "status": "passed",
        "model_name": model_name,
        "model_revision": model_revision,
        "reviewer": str(reviewer).strip(),
        "reviewed_at_utc": str(reviewed_at_utc).strip(),
        "temporal_protocol_core": protocol_core,
        "temporal_protocol_core_sha256": canonical_sha256(protocol_core),
        "coverage_contract": {
            "prompt_ids": list(PROMPT_IDS),
            "seed_ids": list(SEEDS),
            "variant_groups": list(VARIANT_GROUPS),
            "logical_run_count": 54,
            "completed_media_run_count": len(completed_keys),
            "not_supported_run_count": 54 - len(completed_keys),
        },
        "required_criteria": list(criteria_names),
        "criteria": {
            name: {
                "status": "passed",
                "evidence_kind": "machine_and_full_video_human_review",
                "run_keys": completed_keys,
            }
            for name in criteria_names
        },
        "runs": ordered_rows,
    }
    if not payload["reviewer"] or not payload["reviewed_at_utc"]:
        raise ValueError("Qualification requires non-empty reviewer and reviewed_at_utc.")
    payload["document_sha256"] = _document_digest(payload)
    validate_qualification_document(
        payload,
        expected_model_name=model_name,
        expected_model_revision=model_revision,
        expected_temporal_protocol=temporal_protocol,
        expected_criteria=criteria_names,
        verify_source_files=True,
    )
    return payload


def validate_qualification_document(
    payload: Mapping[str, Any],
    *,
    expected_model_name: str,
    expected_model_revision: str,
    expected_temporal_protocol: Mapping[str, Any],
    expected_criteria: Sequence[str],
    verify_source_files: bool = True,
) -> dict[str, Any]:
    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("qualification") != QUALIFICATION_NAME
        or payload.get("benchmark") != BENCHMARK_NAME
    ):
        raise ValueError("Unsupported temporal qualification schema/name/benchmark.")
    declared = str(payload.get("document_sha256", ""))
    actual = _document_digest(payload)
    if not declared or declared != actual:
        raise ValueError(
            f"Temporal qualification document digest mismatch: declared={declared!r}, "
            f"actual={actual}."
        )
    if payload.get("status") != "passed":
        raise ValueError("Temporal qualification status is not passed.")
    if (
        payload.get("model_name") != expected_model_name
        or payload.get("model_revision") != expected_model_revision
    ):
        raise ValueError("Temporal qualification model name/revision differs from adapter.")
    protocol_core = normalized_temporal_protocol(expected_temporal_protocol)
    if payload.get("temporal_protocol_core") != protocol_core or payload.get(
        "temporal_protocol_core_sha256"
    ) != canonical_sha256(protocol_core):
        raise ValueError("Temporal qualification protocol core differs from adapter protocol.")
    if (
        not str(payload.get("reviewer", "")).strip()
        or not str(payload.get("reviewed_at_utc", "")).strip()
    ):
        raise ValueError("Temporal qualification reviewer/time are missing.")

    expected_specs = expected_run_specs(expected_model_name)
    rows = payload.get("runs")
    if not isinstance(rows, list) or len(rows) != 54:
        raise ValueError("Temporal qualification must contain exactly 54 run rows.")
    by_key: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("Temporal qualification run rows must be mappings.")
        key = str(row.get("run_key", ""))
        if key in by_key:
            raise ValueError(f"Temporal qualification duplicates run key {key!r}.")
        by_key[key] = row
    if set(by_key) != set(expected_specs):
        raise ValueError("Temporal qualification run-key coverage differs from exact 54-run grid.")

    completed_keys: list[str] = []
    for key in sorted(expected_specs):
        row = by_key[key]
        expected = expected_specs[key]
        expected_fields = {
            "run_key": key,
            "prompt_id": expected["prompt_id"],
            "seed": expected["seed"],
            "variant": expected["variant"],
            "variation": expected["variation"],
            "variant_kind": expected["kind"],
            "active_pair_ids": list(expected["active_pair_ids"]),
            "status": expected["status"],
        }
        observed_fields = {field: row.get(field) for field in expected_fields}
        observed_fields["variant_kind"] = _normalize_variant_kind(
            str(observed_fields.get("variant_kind", ""))
        )
        if observed_fields != expected_fields:
            raise ValueError(
                f"Temporal qualification row {key!r} differs from contract: "
                f"expected={expected_fields}, observed={observed_fields}."
            )
        condition_id = row.get("condition_id")
        if not isinstance(condition_id, str) or not condition_id.strip():
            raise ValueError(f"Temporal qualification row {key!r} lacks condition_id.")
        if row["status"] == "completed":
            completed_keys.append(key)
        if verify_source_files:
            _validate_row_files(
                row,
                expected=expected,
                model_name=expected_model_name,
                model_revision=expected_model_revision,
                protocol_core=protocol_core,
            )

    expected_criteria = tuple(str(value) for value in expected_criteria)
    if payload.get("required_criteria") != list(expected_criteria):
        raise ValueError("Temporal qualification required criteria differ from adapter contract.")
    criteria = payload.get("criteria")
    if not isinstance(criteria, Mapping) or set(criteria) != set(expected_criteria):
        raise ValueError("Temporal qualification criteria coverage is incomplete or unknown.")
    for criterion_name in expected_criteria:
        criterion = criteria[criterion_name]
        if not isinstance(criterion, Mapping) or criterion != {
            "status": "passed",
            "evidence_kind": "machine_and_full_video_human_review",
            "run_keys": completed_keys,
        }:
            raise ValueError(
                f"Temporal qualification criterion {criterion_name!r} is not bound to every "
                "completed run."
            )
    coverage = payload.get("coverage_contract")
    if coverage != {
        "prompt_ids": list(PROMPT_IDS),
        "seed_ids": list(SEEDS),
        "variant_groups": list(VARIANT_GROUPS),
        "logical_run_count": 54,
        "completed_media_run_count": len(completed_keys),
        "not_supported_run_count": 54 - len(completed_keys),
    }:
        raise ValueError("Temporal qualification coverage summary differs from run evidence.")
    return {
        "status": "passed",
        "document_sha256": declared,
        "model_name": expected_model_name,
        "evaluated_seed_count": 3,
        "logical_run_count": 54,
        "completed_media_run_count": len(completed_keys),
        "source_files_rehashed": bool(verify_source_files),
    }


def validate_temporal_production_gate(
    protocol: Mapping[str, Any],
    *,
    model_name: str,
    model_revision: str,
    criteria_names: Sequence[str],
) -> dict[str, Any] | None:
    """Validate pilot separation or open/re-hash a production evidence manifest."""

    phase = protocol.get("execution_phase")
    gate = protocol.get("production_gate")
    if phase == "pilot":
        if gate is not None:
            raise ValueError(f"{model_name} pilot phase must not carry production_gate evidence.")
        return None
    if phase != "production":
        raise ValueError(f"{model_name} temporal execution_phase must be pilot or production.")
    required_gate_fields = {
        "status",
        "evidence_manifest_path",
        "evidence_manifest_sha256",
        "evidence_document_sha256",
    }
    if not isinstance(gate, Mapping) or set(gate) != required_gate_fields:
        raise RuntimeError(
            f"{model_name} production is blocked until production_gate contains exactly "
            f"{sorted(required_gate_fields)}."
        )
    if gate.get("status") != "passed":
        raise RuntimeError(f"{model_name} temporal production_gate.status must be 'passed'.")
    path = Path(str(gate.get("evidence_manifest_path", ""))).expanduser().resolve()
    expected_file_sha = str(gate.get("evidence_manifest_sha256", ""))
    if not path.is_file() or not expected_file_sha or sha256_file(path) != expected_file_sha:
        raise RuntimeError(f"{model_name} production evidence manifest is missing or changed.")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise RuntimeError(f"{model_name} production evidence digest sidecar is missing.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    validation = validate_qualification_document(
        payload,
        expected_model_name=model_name,
        expected_model_revision=model_revision,
        expected_temporal_protocol=protocol,
        expected_criteria=criteria_names,
        verify_source_files=True,
    )
    if validation["document_sha256"] != gate.get("evidence_document_sha256"):
        raise RuntimeError(f"{model_name} production evidence document digest changed.")
    if sidecar.read_text(encoding="utf-8").split()[:1] != [validation["document_sha256"]]:
        raise RuntimeError(f"{model_name} production evidence sidecar is inconsistent.")
    return validation


def write_immutable_qualification(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    expected_temporal_protocol: Mapping[str, Any],
    expected_criteria: Sequence[str],
) -> tuple[Path, Path]:
    validate_qualification_document(
        payload,
        expected_model_name=str(payload.get("model_name", "")),
        expected_model_revision=str(payload.get("model_revision", "")),
        expected_temporal_protocol=expected_temporal_protocol,
        expected_criteria=expected_criteria,
        verify_source_files=True,
    )
    output = Path(path).expanduser().resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite immutable qualification: {output}")
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        sidecar.write_text(f"{payload['document_sha256']}  {output.name}\n", encoding="utf-8")
        sidecar.chmod(0o444)
    except BaseException:
        output.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    return output, sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seal preassembled 54-run temporal qualification evidence."
    )
    rows = parser.add_mutually_exclusive_group(required=True)
    rows.add_argument("--rows-json")
    rows.add_argument("--manifests", nargs="+")
    parser.add_argument("--evidence-root")
    parser.add_argument("--protocol-json", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--criteria", required=True, help="Comma-separated exact criterion IDs.")
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed-at-utc", required=True)
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    protocol = json.loads(Path(args.protocol_json).read_text(encoding="utf-8"))
    if not isinstance(protocol, dict):
        raise ValueError("protocol-json must be an object.")
    if args.rows_json:
        rows_payload = json.loads(Path(args.rows_json).read_text(encoding="utf-8"))
        if not isinstance(rows_payload, list):
            raise ValueError("rows-json must be a list.")
        qualification_rows = rows_payload
    else:
        if not args.evidence_root:
            raise ValueError("--evidence-root is required with --manifests.")
        qualification_rows = collect_qualification_rows(
            manifest_paths=args.manifests,
            model_name=args.model_name,
            model_revision=args.model_revision,
            temporal_protocol=protocol,
            evidence_root=args.evidence_root,
        )
    criteria = tuple(value.strip() for value in args.criteria.split(",") if value.strip())
    payload = build_qualification_document(
        model_name=args.model_name,
        model_revision=args.model_revision,
        temporal_protocol=protocol,
        criteria_names=criteria,
        rows=qualification_rows,
        reviewer=args.reviewer,
        reviewed_at_utc=args.reviewed_at_utc,
    )
    output, sidecar = write_immutable_qualification(
        args.output_json,
        payload,
        expected_temporal_protocol=protocol,
        expected_criteria=criteria,
    )
    print(
        json.dumps(
            {
                "output_json": str(output),
                "sha256_sidecar": str(sidecar),
                "file_sha256": sha256_file(output),
                "document_sha256": payload["document_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
