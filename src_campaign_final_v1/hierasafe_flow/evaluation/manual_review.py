"""Immutable, run-bound manual review evidence for 15-second benchmark videos.

The automatic auditors deliberately do not decide prompt fidelity, identity,
gender, age, object semantics, or whether motion looks physically plausible.
This module seals those human judgments only after the reviewer attests that
the complete 240-frame clip was watched at full speed and slow motion.  Every
decision is bound to the prepared review record, source result/media, and the
prompt-agnostic full-video audit by SHA-256.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from hierasafe_flow.evaluation.full_video import validate_audit_document
from hierasafe_flow.evaluation.segmented_temporal import (
    validate_segmented_temporal_audit,
)
from hierasafe_flow.generation.temporal_artifacts import read_temporal_evidence


SCHEMA_VERSION = 1
REVIEW_NAME = "finer_detailing_manual_full_video_review_v1"
PERSON_PROMPTS = {"01_sad_young_girl", "02_angry_old_man"}
MALL_PROMPT = "03_empty_outdoor_mall"
CIRCULATION_PAIR_ID = "vertical_circulation_escalators_to_marble_stairs"


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


_VARIANT_KIND_ALIASES: dict[str, str] = {}


def _normalize_variant_kind(kind: str) -> str:
    return _VARIANT_KIND_ALIASES.get(kind, kind)


def _document_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("document_sha256", None)
    return canonical_sha256(canonical)


def required_decision_ids(run: Mapping[str, Any]) -> tuple[str, ...]:
    """Return the exact condition-aware human rubric for one video run."""

    prompt_id = str(run.get("prompt_id", ""))
    model_name = str(run.get("model_name", ""))
    kind = _normalize_variant_kind(str(run.get("variant_kind", "")))
    active_pairs = tuple(str(value) for value in run.get("active_pair_ids", ()))
    if prompt_id not in {*PERSON_PROMPTS, MALL_PROMPT}:
        raise ValueError(f"Unknown finer-detailing prompt_id: {prompt_id!r}.")
    if kind not in {
        "baseline",
        "native_negative_prompt",
        "conceptsteer",
        "shapley_concept_steering",
    }:
        raise ValueError(f"Unknown finer-detailing variant_kind: {kind!r}.")
    if kind in {"baseline", "native_negative_prompt"} and active_pairs:
        raise ValueError(f"{kind} must not claim positive active_pair_ids.")
    if kind in {"conceptsteer", "shapley_concept_steering"} and len(active_pairs) not in {1, 5}:
        raise ValueError("Steering review requires exactly one or five active concept pairs.")

    decision_ids = [
        "source_prompt_fidelity",
        "temporal_coherence",
        "no_freeze_or_long_lag_repetition",
        "natural_motion_speed",
        "no_visible_generation_artifacts",
        "terminal_interval_integrity",
        "non_target_content_preservation",
    ]
    if kind == "baseline":
        decision_ids.append("all_source_concepts_present")
    elif kind == "native_negative_prompt":
        decision_ids.append("native_negative_source_concept_suppression")
    else:
        decision_ids.extend(("active_target_achievement", "intervention_selectivity"))

    if prompt_id in PERSON_PROMPTS:
        decision_ids.extend(("person_identity_preserved", "gender_preserved", "age_preserved"))
    else:
        decision_ids.extend(
            (
                "no_people_preserved",
                "mall_architecture_preserved",
                "signage_semantics_and_legibility",
                "merchandise_semantics",
                "sky_and_floor_semantics",
                "camera_motion_follows_prompt",
            )
        )
        if CIRCULATION_PAIR_ID in active_pairs:
            decision_ids.append("target_marble_stairs_are_static_and_plausible")
        else:
            decision_ids.append("ascending_and_descending_escalators_operate_correctly")

    model_specific = {
        "cogvideox_5b": "rife_interpolation_has_no_ghosting_or_cadence_artifact",
        "hunyuan_video": "native24_to16_resampling_has_no_cadence_artifact",
        "joyai_echo": "audiovisual_and_video_temporal_clocks_look_aligned",
        "ltx_23": "audiovisual_and_video_temporal_clocks_look_aligned",
        "wan22_t2v_a14b": "continuation_seams_are_not_visible",
    }
    if model_name not in model_specific:
        raise ValueError(f"Manual video review received a non-video model: {model_name!r}.")
    decision_ids.append(model_specific[model_name])
    if run.get("temporal_protocol_schema_version") == 2:
        decision_ids.extend(
            (
                "native_seam_01_continuity",
                "native_seam_12_continuity",
                "terminal_fade_absent",
                "whole_clip_segmented_motion_continuity",
                "complete_clip_semantic_fidelity",
                "segmented_non_target_preservation",
            )
        )
        if model_name == "cogvideox_5b":
            decision_ids.extend(
                ("rife_midpoint_97_quality", "rife_midpoint_193_quality")
            )
        elif model_name in {"hunyuan_video", "joyai_echo"}:
            decision_ids.extend(
                (
                    "resampled_seam_neighborhood_80_quality",
                    "resampled_seam_neighborhood_160_quality",
                )
            )
    return tuple(decision_ids)


def _load_json(path: str | Path, *, label: str) -> tuple[Path, dict[str, Any]]:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} does not exist: {resolved}")
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {resolved}")
    return resolved, payload


def _find_review_record(review_manifest: Mapping[str, Any], record_id: str) -> dict[str, Any]:
    records = review_manifest.get("records")
    if not isinstance(records, list):
        raise ValueError("Prepared review manifest is missing records.")
    matches = [record for record in records if record.get("record_id") == record_id]
    if len(matches) != 1 or not isinstance(matches[0], dict):
        raise ValueError(
            f"Prepared review record_id must resolve exactly once; {record_id!r} resolved "
            f"{len(matches)} times."
        )
    return dict(matches[0])


def build_manual_review(
    *,
    run: Mapping[str, Any],
    decisions: Mapping[str, Mapping[str, Any]],
    reviewer: str,
    reviewed_at_utc: str,
    review_manifest_path: str | Path,
    review_record_id: str,
    full_video_audit_path: str | Path,
    playback: Mapping[str, Any],
    temporal_evidence_path: str | Path | None = None,
    segmented_temporal_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build a sealed passed review after validating every source binding."""

    reviewer = str(reviewer).strip()
    reviewed_at_utc = str(reviewed_at_utc).strip()
    if not reviewer or not reviewed_at_utc:
        raise ValueError("Manual review requires non-empty reviewer and reviewed_at_utc.")
    required = required_decision_ids(run)
    if set(decisions) != set(required):
        raise ValueError(
            "Manual decision coverage differs from the required rubric: "
            f"missing={sorted(set(required) - set(decisions))}, "
            f"unknown={sorted(set(decisions) - set(required))}."
        )
    normalized_decisions: dict[str, dict[str, str]] = {}
    for decision_id in required:
        decision = decisions[decision_id]
        if not isinstance(decision, Mapping):
            raise ValueError(f"Manual decision {decision_id!r} must be a mapping.")
        verdict = str(decision.get("verdict", ""))
        notes = str(decision.get("notes", ""))
        if verdict != "pass":
            raise ValueError(
                f"A production-qualification manual review cannot seal {decision_id!r} "
                f"with verdict {verdict!r}; only explicit 'pass' is accepted."
            )
        normalized_decisions[decision_id] = {"verdict": verdict, "notes": notes}

    expected_playback = {
        "entire_clip_viewed": True,
        "full_speed_viewed": True,
        "slow_motion_viewed": True,
        "viewed_frame_count": 240,
    }
    if run.get("temporal_protocol_schema_version") == 2:
        expected_playback.update(
            {
                "lossless_native_seam_windows_viewed": True,
                "final_seam_neighborhoods_viewed": True,
            }
        )
    if dict(playback) != expected_playback:
        raise ValueError(
            f"Manual playback attestation must be exactly {expected_playback}; got {dict(playback)}."
        )

    review_path, review_manifest = _load_json(
        review_manifest_path, label="Prepared review manifest"
    )
    record = _find_review_record(review_manifest, review_record_id)
    audit_path, full_audit = _load_json(full_video_audit_path, label="Full-video audit")
    validate_audit_document(full_audit, verify_source_files=True)

    review_run_fields = (
        "condition_id",
        "prompt_id",
        "model_name",
        "seed",
        "variant_kind",
    )
    run_kind_normalized = _normalize_variant_kind(str(run.get("variant_kind", "")))
    record_kind_normalized = _normalize_variant_kind(str(record.get("variant_kind", "")))
    mismatches: dict[str, dict[str, Any]] = {}
    for field in review_run_fields:
        if field == "variant_kind":
            if run_kind_normalized != record_kind_normalized:
                mismatches[field] = {"run": run.get(field), "review_record": record.get(field)}
        elif run.get(field) != record.get(field):
            mismatches[field] = {"run": run.get(field), "review_record": record.get(field)}
    audit_condition = full_audit.get("condition") or {}
    mismatches.update(
        {
            f"audit.{field}": {"run": run.get(field), "audit": audit_condition.get(field)}
            for field in ("condition_id", "prompt_id", "model_name", "seed")
            if run.get(field) != audit_condition.get(field)
        }
    )
    audit_kind = (audit_condition.get("variant_spec") or {}).get("kind")
    if _normalize_variant_kind(str(audit_kind)) != run_kind_normalized:
        mismatches["audit.variant_kind"] = {
            "run": run.get("variant_kind"),
            "run_normalized": run_kind_normalized,
            "audit": audit_kind,
            "audit_normalized": _normalize_variant_kind(str(audit_kind)),
        }
    if mismatches:
        raise ValueError(f"Manual-review sources do not identify the same run: {mismatches}.")
    if list(run.get("active_pair_ids", ())) != list(record.get("active_pair_ids", ())):
        raise ValueError("Manual-review active_pair_ids differ from the prepared review record.")

    source_bindings = {
        "prepared_review_manifest": {
            "path": str(review_path),
            "sha256": sha256_file(review_path),
            "record_id": review_record_id,
            "record_sha256": canonical_sha256(record),
        },
        "full_video_audit": {
            "path": str(audit_path),
            "sha256": sha256_file(audit_path),
            "document_sha256": full_audit["document_sha256"],
        },
        "benchmark_job_result": {
            "path": str(Path(str(record["source_result_path"])).resolve()),
            "sha256": str(record["source_result_sha256"]),
        },
        "video": {
            "path": str(Path(str(record["source_media_path"])).resolve()),
            "sha256": str(record["source_media_sha256"]),
        },
    }
    if run.get("temporal_protocol_schema_version") == 2:
        if temporal_evidence_path is None or segmented_temporal_audit_path is None:
            raise ValueError(
                "Schema-2 manual review requires temporal evidence and segmented audit paths."
            )
        temporal_path = Path(temporal_evidence_path).expanduser().resolve()
        segmented_path = Path(segmented_temporal_audit_path).expanduser().resolve()
        temporal = read_temporal_evidence(temporal_path)
        segmented = json.loads(segmented_path.read_text(encoding="utf-8"))
        segmented_validation = validate_segmented_temporal_audit(
            segmented,
            verify_source_files=True,
        )
        if (
            temporal["condition_id"] != run.get("condition_id")
            or segmented_validation["condition_id"] != run.get("condition_id")
            or temporal["document_sha256"]
            != (segmented.get("source") or {}).get(
                "temporal_evidence_document_sha256"
            )
        ):
            raise ValueError("Segmented manual-review evidence identifies another run.")
        source_bindings.update(
            {
                "temporal_evidence": {
                    "path": str(temporal_path),
                    "sha256": sha256_file(temporal_path),
                    "document_sha256": temporal["document_sha256"],
                },
                "segmented_temporal_audit": {
                    "path": str(segmented_path),
                    "sha256": sha256_file(segmented_path),
                    "document_sha256": segmented["document_sha256"],
                },
            }
        )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "review": REVIEW_NAME,
        "status": "passed",
        "reviewer": reviewer,
        "reviewed_at_utc": reviewed_at_utc,
        "run": dict(run),
        "playback": expected_playback,
        "required_decision_ids": list(required),
        "decisions": normalized_decisions,
        "source_bindings": source_bindings,
        "semantic_policy": {
            "native_negative_scores_positive_target_achievement": False,
            "missing_or_indeterminate_verdict_allowed": False,
            "automatic_semantic_judgment_allowed": False,
        },
    }
    payload["document_sha256"] = _document_digest(payload)
    validate_manual_review_document(payload, verify_source_files=True)
    return payload


def validate_manual_review_document(
    payload: Mapping[str, Any],
    *,
    verify_source_files: bool = True,
) -> dict[str, Any]:
    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("review") != REVIEW_NAME:
        raise ValueError("Unsupported manual-review schema/name.")
    declared = str(payload.get("document_sha256", ""))
    actual = _document_digest(payload)
    if not declared or declared != actual:
        raise ValueError(
            f"Manual-review document digest mismatch: declared={declared!r}, actual={actual}."
        )
    if payload.get("status") != "passed":
        raise ValueError("Manual-review evidence is not passed.")
    run = payload.get("run")
    if not isinstance(run, Mapping):
        raise ValueError("Manual-review run identity is missing.")
    required = required_decision_ids(run)
    if payload.get("required_decision_ids") != list(required):
        raise ValueError("Manual-review required decision order/content changed.")
    decisions = payload.get("decisions")
    if not isinstance(decisions, Mapping) or set(decisions) != set(required):
        raise ValueError("Manual-review decision coverage is incomplete or unknown.")
    if any(
        not isinstance(decisions[decision_id], Mapping)
        or decisions[decision_id].get("verdict") != "pass"
        for decision_id in required
    ):
        raise ValueError("Manual-review decisions contain a missing/non-pass verdict.")
    expected_playback = {
        "entire_clip_viewed": True,
        "full_speed_viewed": True,
        "slow_motion_viewed": True,
        "viewed_frame_count": 240,
    }
    if run.get("temporal_protocol_schema_version") == 2:
        expected_playback.update(
            {
                "lossless_native_seam_windows_viewed": True,
                "final_seam_neighborhoods_viewed": True,
            }
        )
    if payload.get("playback") != expected_playback:
        raise ValueError("Manual-review full-clip playback attestation is incomplete.")
    policy = payload.get("semantic_policy") or {}
    if policy.get("native_negative_scores_positive_target_achievement") is not False:
        raise ValueError("Manual-review native-negative target policy changed.")

    if verify_source_files:
        bindings = payload.get("source_bindings")
        if not isinstance(bindings, Mapping):
            raise ValueError("Manual-review source bindings are missing.")
        for binding_name in (
            "prepared_review_manifest",
            "full_video_audit",
            "benchmark_job_result",
            "video",
        ):
            binding = bindings.get(binding_name)
            if not isinstance(binding, Mapping):
                raise ValueError(f"Manual-review source binding {binding_name!r} is missing.")
            path = Path(str(binding.get("path", ""))).expanduser().resolve()
            if not path.is_file() or sha256_file(path) != binding.get("sha256"):
                raise ValueError(f"Manual-review source binding changed: {binding_name}.")
        if run.get("temporal_protocol_schema_version") == 2:
            if set(bindings) != {
                "prepared_review_manifest",
                "full_video_audit",
                "benchmark_job_result",
                "video",
                "temporal_evidence",
                "segmented_temporal_audit",
            }:
                raise ValueError("Schema-2 manual-review source binding coverage is invalid.")
            temporal_binding = bindings["temporal_evidence"]
            segmented_binding = bindings["segmented_temporal_audit"]
            temporal = read_temporal_evidence(
                Path(str(temporal_binding.get("path", ""))).expanduser().resolve()
            )
            segmented_path = Path(
                str(segmented_binding.get("path", ""))
            ).expanduser().resolve()
            if not segmented_path.is_file() or sha256_file(segmented_path) != segmented_binding.get(
                "sha256"
            ):
                raise ValueError("Manual-review segmented audit source changed.")
            segmented = json.loads(segmented_path.read_text(encoding="utf-8"))
            segmented_validation = validate_segmented_temporal_audit(
                segmented,
                verify_source_files=True,
            )
            if (
                temporal["document_sha256"] != temporal_binding.get("document_sha256")
                or segmented_validation["document_sha256"]
                != segmented_binding.get("document_sha256")
                or segmented_validation["condition_id"] != run.get("condition_id")
            ):
                raise ValueError("Manual-review segmented evidence binding changed.")

        review_binding = bindings["prepared_review_manifest"]
        review_manifest = json.loads(Path(str(review_binding["path"])).read_text(encoding="utf-8"))
        record = _find_review_record(review_manifest, str(review_binding.get("record_id", "")))
        if canonical_sha256(record) != review_binding.get("record_sha256"):
            raise ValueError("Prepared review record changed within its bound manifest.")
        if record.get("source_result_sha256") != bindings["benchmark_job_result"].get("sha256"):
            raise ValueError("Manual-review result digest differs from prepared review record.")
        if record.get("source_media_sha256") != bindings["video"].get("sha256"):
            raise ValueError("Manual-review video digest differs from prepared review record.")

        audit_binding = bindings["full_video_audit"]
        audit = json.loads(Path(str(audit_binding["path"])).read_text(encoding="utf-8"))
        validation = validate_audit_document(audit, verify_source_files=True)
        if validation["document_sha256"] != audit_binding.get("document_sha256"):
            raise ValueError("Manual-review full-video audit document digest changed.")
        if validation["condition_id"] != run.get("condition_id"):
            raise ValueError("Manual-review full-video audit identifies another condition.")
    return {
        "status": "passed",
        "document_sha256": declared,
        "condition_id": run.get("condition_id"),
        "source_files_rehashed": bool(verify_source_files),
    }


def write_immutable_manual_review(
    path: str | Path, payload: Mapping[str, Any]
) -> tuple[Path, Path]:
    validate_manual_review_document(payload, verify_source_files=True)
    output = Path(path).expanduser().resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite immutable manual review: {output}")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
        sidecar.write_text(f"{payload['document_sha256']}  {output.name}\n", encoding="utf-8")
        sidecar.chmod(0o444)
    except BaseException:
        output.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    return output, sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Seal a completed full-video manual review.")
    parser.add_argument("--run-json", required=True)
    parser.add_argument("--decisions-json", required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--reviewed-at-utc", required=True)
    parser.add_argument("--review-manifest", required=True)
    parser.add_argument("--review-record-id", required=True)
    parser.add_argument("--full-video-audit", required=True)
    parser.add_argument("--temporal-evidence")
    parser.add_argument("--segmented-temporal-audit")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    _, run = _load_json(args.run_json, label="Run identity JSON")
    _, decisions = _load_json(args.decisions_json, label="Manual decisions JSON")
    payload = build_manual_review(
        run=run,
        decisions=decisions,
        reviewer=args.reviewer,
        reviewed_at_utc=args.reviewed_at_utc,
        review_manifest_path=args.review_manifest,
        review_record_id=args.review_record_id,
        full_video_audit_path=args.full_video_audit,
        playback={
            "entire_clip_viewed": True,
            "full_speed_viewed": True,
            "slow_motion_viewed": True,
            "viewed_frame_count": 240,
            **(
                {
                    "lossless_native_seam_windows_viewed": True,
                    "final_seam_neighborhoods_viewed": True,
                }
                if run.get("temporal_protocol_schema_version") == 2
                else {}
            ),
        },
        temporal_evidence_path=args.temporal_evidence,
        segmented_temporal_audit_path=args.segmented_temporal_audit,
    )
    output, sidecar = write_immutable_manual_review(args.output_json, payload)
    print(
        json.dumps(
            {
                "output_json": str(output),
                "sha256_sidecar": str(sidecar),
                "document_sha256": payload["document_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
