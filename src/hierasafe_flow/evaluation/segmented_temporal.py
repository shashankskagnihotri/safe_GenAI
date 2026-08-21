"""Authenticated automatic audit for schema-2 segmented temporal generation.

This audit consumes the lossless native evidence transaction.  It never tries
to reconstruct segment boundaries from the final MP4 alone.  Hunyuan/Joy
qualification thresholds must come from a separately sealed target-blind
calibration; the built-in values for those routes are engineering diagnostics
and deliberately cannot promote a protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from hierasafe_flow.evaluation.temporal_metrics import (
    SEGMENTED_METRIC_PARAMETERS,
    SEGMENTED_METRIC_PARAMETERS_SHA256,
    FixedLpipsMetric,
    validate_temporal_metric_contract,
)
from hierasafe_flow.generation.temporal_artifacts import (
    decode_video_rgb_frames,
    frame_rgb_sha256,
    frame_sequence_sha256,
    read_temporal_evidence,
)


SCHEMA_VERSION = 1
AUDIT_NAME = "segmented_temporal_evidence_audit_v1"
SEGMENTED_MODELS = frozenset({"cogvideox_5b", "hunyuan_video", "joyai_echo"})

_ROUTE_CONTRACTS: Mapping[str, Mapping[str, Any]] = {
    "cogvideox_5b": {
        "native_lengths": [49, 49, 49],
        "native_fps": [8, 8, 8],
        "retained": [list(range(49)), list(range(1, 49)), list(range(1, 25))],
        "native_stitch_count": 121,
        "native_seams": [49, 97],
        "final_seams": [97, 193],
        "postprocess_method": "pinned_rife_midpoint_once_over_complete_stitch",
    },
    "hunyuan_video": {
        "native_lengths": [121, 121, 121],
        "native_fps": [24, 24, 24],
        "retained": [list(range(121)), list(range(1, 121)), list(range(1, 121))],
        "native_stitch_count": 361,
        "native_seams": [121, 241],
        "final_seams": [80, 160],
        "postprocess_method": "nearest_timestamp_decimation_round_half_up",
    },
    "joyai_echo": {
        "native_lengths": [121, 121, 121],
        "native_fps": [24, 24, 24],
        "retained": [list(range(120)), list(range(120)), list(range(120))],
        "native_stitch_count": 360,
        "native_seams": [120, 240],
        "final_seams": [80, 160],
        "postprocess_method": "nearest_timestamp_decimation_round_half_up",
    },
}

_BASE_THRESHOLDS: Mapping[str, Any] = {
    "schema_version": 1,
    "analysis_rgb_size": [256, 256],
    "flow_size": [128, 128],
    "histogram_bins_per_channel": 64,
    "low_change_mae": 0.002,
    "whole_clip_low_change_fraction_max": 0.95,
    "terminal_low_change_fraction_max": 0.95,
    "exact_unique_frame_ratio_min": 0.95,
    "cut_absolute_mae_max": 0.35,
    "cut_robust_mad_multiplier": 6.0,
    "cut_robust_minimum_allowance": 0.02,
    "near_periodic_match_fraction_max": 0.95,
    "alternating_cadence_relative_difference_max": 0.50,
    "terminal_luma_ratio_min": 0.60,
    "terminal_black_fraction_increase_max": 0.20,
    "terminal_monotone_spearman_max": -0.80,
    "terminal_monotone_pvalue_max": 0.01,
    "terminal_monotone_relative_drop_min": 0.20,
    "black_luma_threshold": 16.0 / 255.0,
    "farneback": {
        "pyr_scale": 0.5,
        "levels": 3,
        "winsize": 15,
        "iterations": 3,
        "poly_n": 5,
        "poly_sigma": 1.2,
        "flags": 0,
    },
}


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_segmented_temporal_audit(
    *,
    temporal_evidence_path: str | Path,
    model_name: str,
    project_root: str | Path,
    device: str | torch.device = "cpu",
    registered_thresholds: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if model_name not in SEGMENTED_MODELS:
        raise ValueError(f"No segmented temporal route is registered for {model_name!r}.")
    evidence_path = Path(temporal_evidence_path).expanduser().resolve()
    evidence = read_temporal_evidence(evidence_path)
    route = _validate_route_contract(evidence, model_name=model_name)
    thresholds, threshold_registration = _threshold_contract(
        model_name, registered_thresholds
    )
    metric_contract = validate_temporal_metric_contract(
        project_root=Path(project_root).resolve()
    )
    if metric_contract.get("status") != "passed":
        raise RuntimeError("Pinned temporal metric contract did not pass authentication.")

    segment_frames = [
        decode_video_rgb_frames(segment["lossless_artifacts"]["ffv1_mkv_path"])
        for segment in evidence["segments"]
    ]
    stitch_map = [tuple(int(value) for value in pair) for pair in evidence["stitch"]["map"]]
    stitched = [segment_frames[segment][frame] for segment, frame in stitch_map]
    if frame_sequence_sha256(stitched) != evidence["stitch"]["native_rgb_sha256"]:
        raise RuntimeError("Decoded native stitch differs from temporal evidence binding.")
    final_path = evidence_path.parent / "video_000.mp4"
    final_frames = decode_video_rgb_frames(final_path)
    if frame_sequence_sha256(final_frames) != evidence["binding"][
        "final_decoded_rgb_sha256"
    ]:
        raise RuntimeError("Decoded final MP4 differs from temporal evidence binding.")

    lpips_metric = FixedLpipsMetric.from_preregistered_artifacts(
        project_root=Path(project_root).resolve(), device=device
    )
    native_metrics = _timeline_metrics(
        stitched,
        lpips_metric=lpips_metric,
        thresholds=thresholds,
    )
    final_metrics = _timeline_metrics(
        final_frames,
        lpips_metric=lpips_metric,
        thresholds=thresholds,
    )
    segment_unique = [
        len({frame_rgb_sha256(frame) for frame in frames}) / len(frames)
        for frames in segment_frames
    ]
    anchor_records = _anchor_reconstruction_records(evidence, segment_frames)
    native_seams = _seam_records(
        native_metrics["adjacent"],
        seam_positions=route["native_seams"],
        thresholds=thresholds,
    )
    final_seams = _seam_records(
        final_metrics["adjacent"],
        seam_positions=route["final_seams"],
        thresholds=thresholds,
    )
    fade = _terminal_fade(final_frames, thresholds=thresholds)
    diagnostics = {
        "native": _timeline_diagnostics(native_metrics, thresholds=thresholds),
        "final": _timeline_diagnostics(final_metrics, thresholds=thresholds),
        "terminal_fade": fade,
    }

    gates: dict[str, bool] = {
        "complete_native_segments": True,
        "retained_discarded_partition": True,
        "exact_stitch_arithmetic": True,
        "exact_final_binding": True,
        "per_complete_segment_unique_ratio": all(
            ratio >= float(thresholds["exact_unique_frame_ratio_min"])
            for ratio in segment_unique
        ),
        "native_seams_not_abrupt": all(record["passed"] for record in native_seams),
        "final_seams_not_abrupt": all(record["passed"] for record in final_seams),
        "no_native_whole_or_terminal_freeze": not (
            diagnostics["native"]["whole_clip_freeze"]
            or diagnostics["native"]["terminal_freeze"]
        ),
        "no_final_whole_or_terminal_freeze": not (
            diagnostics["final"]["whole_clip_freeze"]
            or diagnostics["final"]["terminal_freeze"]
        ),
        "no_native_near_periodicity": not diagnostics["native"]["near_periodic"],
        "no_final_near_periodicity": not diagnostics["final"]["near_periodic"],
        "no_native_alternating_cadence": not diagnostics["native"][
            "alternating_cadence"
        ],
        "no_final_alternating_cadence": not diagnostics["final"][
            "alternating_cadence"
        ],
        "no_unintended_terminal_fade": fade["passed"],
    }
    if model_name in {"cogvideox_5b", "hunyuan_video"}:
        gates["anchor_reconstruction"] = all(
            record["anchor_sha256_matches_previous_terminal"]
            and record["reconstruction_mae"]
            <= float(thresholds["reconstruction_mae_max"])
            and record["reconstruction_psnr_db"]
            >= float(thresholds["reconstruction_psnr_min_db"])
            and record["anchor_to_first_motion_mae"]
            <= float(thresholds["anchor_to_first_motion_mae_max"])
            for record in anchor_records
        )
    status = "passed" if all(gates.values()) else "failed"
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit": AUDIT_NAME,
        "status": status,
        "qualification_eligible": bool(threshold_registration["qualification_eligible"]),
        "model_name": model_name,
        "condition_id": evidence["condition_id"],
        "attempt": evidence["attempt"],
        "job_id": evidence["job_id"],
        "manifest_sha256": evidence["manifest_sha256"],
        "source": {
            "temporal_evidence_path": str(evidence_path),
            "temporal_evidence_file_sha256": sha256_file(evidence_path),
            "temporal_evidence_document_sha256": evidence["document_sha256"],
            "temporal_protocol_sha256": evidence["temporal_protocol_sha256"],
            "final_media_path": str(final_path),
            "final_media_sha256": evidence["binding"]["final_media_sha256"],
            "final_decoded_rgb_sha256": evidence["binding"][
                "final_decoded_rgb_sha256"
            ],
            "final_frame_sha256": evidence["binding"]["final_frame_sha256"],
        },
        "route_contract": dict(route),
        "threshold_registration": threshold_registration,
        "thresholds": thresholds,
        "thresholds_sha256": canonical_sha256(thresholds),
        "metric_contract": metric_contract,
        "metric_parameters": _plain(SEGMENTED_METRIC_PARAMETERS),
        "metric_parameters_sha256": SEGMENTED_METRIC_PARAMETERS_SHA256,
        "segment_unique_frame_ratio": segment_unique,
        "anchor_reconstruction": anchor_records,
        "native_timeline": native_metrics,
        "final_timeline": final_metrics,
        "native_seams": native_seams,
        "final_seams": final_seams,
        "diagnostics": diagnostics,
        "gates": gates,
    }
    payload["document_sha256"] = _document_sha256(payload)
    validate_segmented_temporal_audit(
        payload,
        verify_source_files=True,
        require_pass=False,
    )
    return payload


def validate_segmented_temporal_audit(
    payload: Mapping[str, Any],
    *,
    verify_source_files: bool = True,
    require_pass: bool = True,
) -> dict[str, Any]:
    if payload.get("schema_version") != 1 or payload.get("audit") != AUDIT_NAME:
        raise ValueError("Unsupported segmented temporal audit schema/name.")
    if payload.get("document_sha256") != _document_sha256(payload):
        raise ValueError("Segmented temporal audit document digest mismatch.")
    if payload.get("model_name") not in SEGMENTED_MODELS:
        raise ValueError("Segmented temporal audit model identity is invalid.")
    if payload.get("metric_parameters_sha256") != SEGMENTED_METRIC_PARAMETERS_SHA256:
        raise ValueError("Segmented temporal metric parameter digest drifted.")
    thresholds = payload.get("thresholds")
    if not isinstance(thresholds, Mapping) or payload.get(
        "thresholds_sha256"
    ) != canonical_sha256(thresholds):
        raise ValueError("Segmented temporal threshold binding is invalid.")
    gates = payload.get("gates")
    if not isinstance(gates, Mapping) or not gates or any(
        not isinstance(value, bool) for value in gates.values()
    ):
        raise ValueError("Segmented temporal audit gates are malformed.")
    expected_status = "passed" if all(gates.values()) else "failed"
    if payload.get("status") != expected_status:
        raise ValueError("Segmented temporal audit status/gates are inconsistent.")
    if require_pass and expected_status != "passed":
        raise ValueError("Segmented temporal audit has one or more failed gates.")
    if verify_source_files:
        source = payload.get("source") or {}
        evidence_path = Path(str(source.get("temporal_evidence_path", ""))).resolve()
        if not evidence_path.is_file() or sha256_file(evidence_path) != source.get(
            "temporal_evidence_file_sha256"
        ):
            raise ValueError("Segmented temporal source evidence changed.")
        evidence = read_temporal_evidence(evidence_path)
        if evidence["document_sha256"] != source.get("temporal_evidence_document_sha256"):
            raise ValueError("Segmented temporal evidence document binding changed.")
        if evidence["binding"]["final_decoded_rgb_sha256"] != source.get(
            "final_decoded_rgb_sha256"
        ):
            raise ValueError("Segmented/full decoded frame binding changed.")
        if evidence["binding"]["final_frame_sha256"] != source.get(
            "final_frame_sha256"
        ):
            raise ValueError("Segmented decoded frame-list binding changed.")
    return {
        "status": expected_status,
        "document_sha256": payload["document_sha256"],
        "condition_id": payload.get("condition_id"),
        "qualification_eligible": bool(payload.get("qualification_eligible")),
        "final_decoded_rgb_sha256": (payload.get("source") or {}).get(
            "final_decoded_rgb_sha256"
        ),
    }


def write_segmented_temporal_audit(path: str | Path, payload: Mapping[str, Any]) -> Path:
    validate_segmented_temporal_audit(
        payload,
        verify_source_files=True,
        require_pass=False,
    )
    output = Path(path).expanduser().resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite segmented temporal audit: {output}")
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        sidecar.write_text(f"{hashlib.sha256(raw).hexdigest()}  {output.name}\n", encoding="utf-8")
        sidecar.chmod(0o444)
    except BaseException:
        output.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    return output


def _validate_route_contract(evidence: Mapping[str, Any], *, model_name: str) -> dict[str, Any]:
    contract = dict(_ROUTE_CONTRACTS[model_name])
    segments = evidence["segments"]
    if [segment["decoded_frame_count"] for segment in segments] != contract["native_lengths"]:
        raise RuntimeError("Temporal native segment lengths differ from the registered route.")
    if [segment["native_fps"] for segment in segments] != contract["native_fps"]:
        raise RuntimeError("Temporal native segment clocks differ from the registered route.")
    if [segment["retained_indices"] for segment in segments] != contract["retained"]:
        raise RuntimeError("Temporal retained indices differ from the registered route.")
    if evidence["stitch"]["native_frame_count"] != contract["native_stitch_count"]:
        raise RuntimeError("Temporal stitch arithmetic differs from the registered route.")
    if evidence["output_contract"] != {
        "frame_count": 240,
        "fps": 16,
        "duration_seconds": 15.0,
    }:
        raise RuntimeError("Temporal final output contract is not exact 240/16/15.")
    method = str((evidence.get("postprocess") or {}).get("method", ""))
    if method != contract["postprocess_method"]:
        raise RuntimeError(
            f"Temporal postprocess method drifted: expected {contract['postprocess_method']!r}, "
            f"observed {method!r}."
        )
    return contract


def _threshold_contract(
    model_name: str, registered: Mapping[str, Any] | None
) -> tuple[dict[str, Any], dict[str, Any]]:
    thresholds = json.loads(json.dumps(_BASE_THRESHOLDS))
    thresholds.update(
        {
            "reconstruction_mae_max": 0.20,
            "reconstruction_psnr_min_db": 12.0,
            "anchor_to_first_motion_mae_max": 0.35,
        }
    )
    if registered is not None:
        supplied = json.loads(json.dumps(dict(registered)))
        if supplied.get("schema_version") != 1:
            raise ValueError("Registered temporal thresholds must use schema version 1.")
        registration = supplied.pop("registration", None)
        if not isinstance(registration, Mapping):
            raise ValueError("Registered temporal thresholds lack calibration registration.")
        thresholds.update(supplied)
        if registration.get("model_name") != model_name or not registration.get(
            "target_blind_calibration_sha256"
        ):
            raise ValueError("Temporal threshold calibration identity is incomplete.")
        return thresholds, {
            **dict(registration),
            "qualification_eligible": True,
        }
    if model_name == "cogvideox_5b":
        return thresholds, {
            "source": "sealed_cogvideox_temporal_repair_design_20260719",
            "qualification_eligible": True,
        }
    return thresholds, {
        "source": "engineering_diagnostic_defaults_pending_target_blind_calibration",
        "qualification_eligible": False,
    }


def _timeline_metrics(
    frames: Sequence[np.ndarray], *, lpips_metric: Any, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    resized = [_resize_rgb(frame, thresholds["analysis_rgb_size"]) for frame in frames]
    adjacent: list[dict[str, Any]] = []
    lpips_values = _lpips_adjacent(resized, lpips_metric)
    for index in range(1, len(resized)):
        left = resized[index - 1].astype(np.float32) / 255.0
        right = resized[index].astype(np.float32) / 255.0
        mse = float(np.mean(np.square(right - left)))
        mae = float(np.mean(np.abs(right - left)))
        adjacent.append(
            {
                "left_index": index - 1,
                "right_index": index,
                "normalized_rgb_mae": mae,
                "psnr_db": (120.0 if mse == 0.0 else -10.0 * math.log10(mse)),
                "ssim": _global_ssim(left, right),
                "lpips": lpips_values[index - 1],
                "color_histogram_l1": _histogram_distance(left, right, thresholds),
                "rec709_luma_mean_abs": float(
                    np.mean(np.abs(_rec709_luma(left) - _rec709_luma(right)))
                ),
                "dense_flow": _flow_summary(left, right, thresholds),
                "exact_equal": frame_rgb_sha256(frames[index - 1])
                == frame_rgb_sha256(frames[index]),
            }
        )
    return {
        "frame_count": len(frames),
        "rgb_sequence_sha256": frame_sequence_sha256(frames),
        "adjacent": adjacent,
        "adjacent_metric_summary": {
            key: _finite_summary([row[key] for row in adjacent])
            for key in (
                "normalized_rgb_mae",
                "psnr_db",
                "ssim",
                "lpips",
                "color_histogram_l1",
                "rec709_luma_mean_abs",
            )
        },
        "long_lag_periodicity": _long_lag_periodicity(resized, thresholds=thresholds),
    }


def _lpips_adjacent(frames: Sequence[np.ndarray], metric: Any) -> list[float]:
    values: list[float] = []
    for start in range(0, max(0, len(frames) - 1), 8):
        pairs = list(range(start, min(len(frames) - 1, start + 8)))
        left = torch.from_numpy(np.stack([frames[index] for index in pairs])).permute(0, 3, 1, 2)
        right = torch.from_numpy(np.stack([frames[index + 1] for index in pairs])).permute(0, 3, 1, 2)
        with torch.inference_mode():
            observed = metric(left.float().div(255.0), right.float().div(255.0))
        flat = observed.detach().cpu().float().flatten()
        if flat.numel() != len(pairs) or not torch.isfinite(flat).all():
            raise RuntimeError("LPIPS metric returned malformed adjacent-frame values.")
        values.extend(float(value) for value in flat.tolist())
    return values


def _anchor_reconstruction_records(
    evidence: Mapping[str, Any], segment_frames: Sequence[Sequence[np.ndarray]]
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, segment in enumerate(evidence["segments"]):
        if index == 0 or segment.get("anchor_sha256") is None:
            continue
        anchor = segment_frames[index - 1][-1]
        reconstruction = segment_frames[index][int(segment["reconstruction_index"])]
        first_motion = segment_frames[index][int(segment["first_motion_index"])]
        reconstruction_metrics = _pair_basic(anchor, reconstruction)
        motion_metrics = _pair_basic(anchor, first_motion)
        records.append(
            {
                "segment_index": index,
                "anchor_sha256_matches_previous_terminal": frame_rgb_sha256(anchor)
                == segment["anchor_sha256"],
                "reconstruction_mae": reconstruction_metrics["normalized_rgb_mae"],
                "reconstruction_psnr_db": reconstruction_metrics["psnr_db"],
                "reconstruction_ssim": reconstruction_metrics["ssim"],
                "anchor_to_first_motion_mae": motion_metrics["normalized_rgb_mae"],
                "anchor_to_first_motion_psnr_db": motion_metrics["psnr_db"],
                "anchor_to_first_motion_ssim": motion_metrics["ssim"],
            }
        )
    return records


def _pair_basic(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    left_f = _resize_rgb(left, (256, 256)).astype(np.float32) / 255.0
    right_f = _resize_rgb(right, (256, 256)).astype(np.float32) / 255.0
    mse = float(np.mean(np.square(right_f - left_f)))
    return {
        "normalized_rgb_mae": float(np.mean(np.abs(right_f - left_f))),
        "psnr_db": 120.0 if mse == 0.0 else -10.0 * math.log10(mse),
        "ssim": _global_ssim(left_f, right_f),
    }


def _seam_records(
    adjacent: Sequence[Mapping[str, Any]],
    *,
    seam_positions: Sequence[int],
    thresholds: Mapping[str, Any],
) -> list[dict[str, Any]]:
    distances = np.asarray(
        [float(row["normalized_rgb_mae"]) for row in adjacent], dtype=np.float64
    )
    records = []
    for right_index in seam_positions:
        pair_index = right_index - 1
        if not 0 <= pair_index < len(adjacent):
            raise RuntimeError("Registered seam is outside the decoded timeline.")
        neighborhood_indices = [
            index
            for index in range(max(0, pair_index - 12), min(len(adjacent), pair_index + 13))
            if index != pair_index
        ]
        neighborhood = distances[neighborhood_indices]
        median = float(np.median(neighborhood))
        mad = float(np.median(np.abs(neighborhood - median)))
        robust_limit = median + max(
            float(thresholds["cut_robust_mad_multiplier"]) * mad,
            float(thresholds["cut_robust_minimum_allowance"]),
        )
        observed = float(distances[pair_index])
        passed = observed <= robust_limit and observed <= float(
            thresholds["cut_absolute_mae_max"]
        )
        records.append(
            {
                "left_index": pair_index,
                "right_index": right_index,
                "normalized_rgb_mae": observed,
                "neighborhood_median": median,
                "neighborhood_mad": mad,
                "robust_limit": robust_limit,
                "absolute_limit": float(thresholds["cut_absolute_mae_max"]),
                "passed": passed,
            }
        )
    return records


def _timeline_diagnostics(
    metrics: Mapping[str, Any], *, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    adjacent = metrics["adjacent"]
    changes = [float(row["normalized_rgb_mae"]) for row in adjacent]
    low = [value <= float(thresholds["low_change_mae"]) for value in changes]
    terminal = low[-32:] if len(low) >= 32 else low
    periodicity = metrics["long_lag_periodicity"]
    exact_adjacent_fraction = (
        float(np.mean([bool(row["exact_equal"]) for row in adjacent])) if adjacent else 1.0
    )
    even = changes[0::2]
    odd = changes[1::2]
    even_mean = float(np.mean(even)) if even else 0.0
    odd_mean = float(np.mean(odd)) if odd else 0.0
    denominator = max((even_mean + odd_mean) / 2.0, 1.0e-12)
    cadence_difference = abs(even_mean - odd_mean) / denominator
    return {
        "low_change_fraction": float(np.mean(low)) if low else 1.0,
        "terminal_low_change_fraction": float(np.mean(terminal)) if terminal else 1.0,
        "exact_adjacent_fraction": exact_adjacent_fraction,
        "whole_clip_freeze": bool(
            low
            and float(np.mean(low))
            >= float(thresholds["whole_clip_low_change_fraction_max"])
        ),
        "terminal_freeze": bool(
            terminal
            and float(np.mean(terminal))
            >= float(thresholds["terminal_low_change_fraction_max"])
        ),
        "near_periodic": bool(periodicity["near_periodic"]),
        "long_lag_periodicity": periodicity,
        "alternating_cadence_relative_difference": cadence_difference,
        "alternating_cadence": cadence_difference
        > float(thresholds["alternating_cadence_relative_difference_max"]),
    }


def _long_lag_periodicity(
    frames: Sequence[np.ndarray], *, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    if len(frames) < 4:
        return {
            "near_periodic": False,
            "maximum_match_fraction": 0.0,
            "maximum_match_lag": None,
            "per_lag": [],
        }
    # A true long-lag loop/repetition can have perfectly healthy adjacent
    # motion.  Compare registered lags independently at the fixed analysis
    # resolution instead of reusing adjacent exact-equality flags.
    maximum_lag = min(32, len(frames) // 2)
    low_change = float(thresholds["low_change_mae"])
    rows: list[dict[str, Any]] = []
    for lag in range(2, maximum_lag + 1):
        distances = []
        exact = 0
        for index in range(len(frames) - lag):
            left = frames[index]
            right = frames[index + lag]
            distances.append(
                float(
                    np.mean(
                        np.abs(
                            left.astype(np.float32) / 255.0
                            - right.astype(np.float32) / 255.0
                        )
                    )
                )
            )
            exact += int(np.array_equal(left, right))
        match_fraction = float(np.mean(np.asarray(distances) <= low_change))
        rows.append(
            {
                "lag": lag,
                "comparison_count": len(distances),
                "low_change_match_fraction": match_fraction,
                "exact_match_fraction": exact / len(distances),
                "median_normalized_rgb_mae": float(np.median(distances)),
            }
        )
    maximum = max(rows, key=lambda row: row["low_change_match_fraction"])
    near_periodic = maximum["low_change_match_fraction"] >= float(
        thresholds["near_periodic_match_fraction_max"]
    )
    return {
        "near_periodic": near_periodic,
        "maximum_match_fraction": maximum["low_change_match_fraction"],
        "maximum_match_lag": maximum["lag"],
        "per_lag": rows,
    }


def _terminal_fade(
    frames: Sequence[np.ndarray], *, thresholds: Mapping[str, Any]
) -> dict[str, Any]:
    if len(frames) < 64:
        raise RuntimeError("Terminal fade audit requires at least 64 decoded frames.")
    luma = np.asarray(
        [float(np.median(_rec709_luma(frame.astype(np.float32) / 255.0))) for frame in frames],
        dtype=np.float64,
    )
    prior = luma[-64:-32]
    last32 = luma[-32:]
    last8 = luma[-8:]
    denominator = max(float(np.median(prior)), 1.0e-6)
    ratio32 = float(np.median(last32)) / denominator
    ratio8 = float(np.median(last8)) / denominator
    black_threshold = float(thresholds["black_luma_threshold"])
    prior_black = float(np.mean(prior <= black_threshold))
    last_black = float(np.mean(last8 <= black_threshold))
    try:
        from scipy.stats import spearmanr
    except ModuleNotFoundError as exc:
        raise RuntimeError("Terminal fade audit requires the pinned SciPy runtime.") from exc
    correlation, pvalue = spearmanr(np.arange(32, dtype=np.float64), last32)
    relative_drop = max(0.0, float(last32[0] - last32[-1]) / max(float(last32[0]), 1e-6))
    monotone_fade = bool(
        np.isfinite(correlation)
        and np.isfinite(pvalue)
        and correlation <= float(thresholds["terminal_monotone_spearman_max"])
        and pvalue <= float(thresholds["terminal_monotone_pvalue_max"])
        and relative_drop >= float(thresholds["terminal_monotone_relative_drop_min"])
    )
    passed = (
        ratio32 >= float(thresholds["terminal_luma_ratio_min"])
        and ratio8 >= float(thresholds["terminal_luma_ratio_min"])
        and last_black - prior_black
        <= float(thresholds["terminal_black_fraction_increase_max"])
        and not monotone_fade
    )
    return {
        "median_luma_last32_to_previous32_ratio": ratio32,
        "median_luma_last8_to_previous32_ratio": ratio8,
        "black_fraction_increase": last_black - prior_black,
        "spearman_correlation_last32": float(correlation),
        "spearman_pvalue_last32": float(pvalue),
        "relative_endpoint_drop_last32": relative_drop,
        "monotone_fade": monotone_fade,
        "passed": passed,
    }


def _global_ssim(left: np.ndarray, right: np.ndarray) -> float:
    c1 = 0.01**2
    c2 = 0.03**2
    values = []
    for channel in range(3):
        x = left[..., channel].astype(np.float64)
        y = right[..., channel].astype(np.float64)
        mux, muy = float(x.mean()), float(y.mean())
        varx, vary = float(x.var()), float(y.var())
        covariance = float(np.mean((x - mux) * (y - muy)))
        values.append(
            ((2.0 * mux * muy + c1) * (2.0 * covariance + c2))
            / ((mux * mux + muy * muy + c1) * (varx + vary + c2))
        )
    return float(np.mean(values))


def _histogram_distance(
    left: np.ndarray, right: np.ndarray, thresholds: Mapping[str, Any]
) -> float:
    bins = int(thresholds["histogram_bins_per_channel"])
    distance = 0.0
    for channel in range(3):
        first, _ = np.histogram(left[..., channel], bins=bins, range=(0.0, 1.0), density=False)
        second, _ = np.histogram(right[..., channel], bins=bins, range=(0.0, 1.0), density=False)
        first = first.astype(np.float64) / max(int(first.sum()), 1)
        second = second.astype(np.float64) / max(int(second.sum()), 1)
        distance += float(np.abs(first - second).sum())
    return distance / 3.0


def _rec709_luma(rgb: np.ndarray) -> np.ndarray:
    return 0.2126 * rgb[..., 0] + 0.7152 * rgb[..., 1] + 0.0722 * rgb[..., 2]


def _flow_summary(
    left: np.ndarray, right: np.ndarray, thresholds: Mapping[str, Any]
) -> dict[str, float]:
    width, height = (int(value) for value in thresholds["flow_size"])
    first = cv2.resize(_rec709_luma(left), (width, height), interpolation=cv2.INTER_AREA)
    second = cv2.resize(_rec709_luma(right), (width, height), interpolation=cv2.INTER_AREA)
    parameters = thresholds["farneback"]
    flow = cv2.calcOpticalFlowFarneback(
        first.astype(np.float32),
        second.astype(np.float32),
        None,
        float(parameters["pyr_scale"]),
        int(parameters["levels"]),
        int(parameters["winsize"]),
        int(parameters["iterations"]),
        int(parameters["poly_n"]),
        float(parameters["poly_sigma"]),
        int(parameters["flags"]),
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1], angleInDegrees=False)
    return {
        "mean_magnitude": float(np.mean(magnitude)),
        "p90_magnitude": float(np.quantile(magnitude, 0.90)),
        "mean_direction_radians": float(
            math.atan2(float(np.mean(np.sin(angle))), float(np.mean(np.cos(angle))))
        ),
    }


def _resize_rgb(frame: np.ndarray, size: Sequence[int]) -> np.ndarray:
    width, height = (int(value) for value in size)
    return np.ascontiguousarray(
        cv2.resize(np.asarray(frame)[..., :3], (width, height), interpolation=cv2.INTER_AREA)
    )


def _finite_summary(values: Sequence[Any]) -> dict[str, float | None]:
    array = np.asarray([float(value) for value in values], dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"minimum": None, "median": None, "mean": None, "p90": None, "maximum": None}
    return {
        "minimum": float(np.min(array)),
        "median": float(np.median(array)),
        "mean": float(np.mean(array)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_plain(item) for item in value]
    return value


def _document_sha256(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("document_sha256", None)
    return canonical_sha256(canonical)
