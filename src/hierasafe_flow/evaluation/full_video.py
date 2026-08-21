"""Prompt-agnostic, fail-closed evidence audit for benchmark videos.

This module establishes structural and low-level temporal evidence only. It does
not infer prompt fidelity, object identity, camera intent, or steering success.
Prompt-specific audits (for example, Prompt 03 circulation ROIs) and finalized
manual review are additional layers, never replacements for this audit.

The finer-detailing benchmark contract is exactly 240 decoded RGB frames at 16
constant frames per second for 15 seconds. Every presentation timestamp and
decoded frame is checked, and the source manifest/result/media are hash-bound.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np

from hierasafe_flow.evaluation.segmented_temporal import (
    validate_segmented_temporal_audit,
)


SCHEMA_VERSION = 1
AUDIT_NAME = "prompt_agnostic_full_video_v1"
EXPECTED_FRAME_COUNT = 240
EXPECTED_FPS = 16
EXPECTED_DURATION_SECONDS = 15
ANALYSIS_SIZE = (32, 32)
LOW_CHANGE_THRESHOLD = 2.0e-3
CUT_ABSOLUTE_THRESHOLD = 0.12
MAX_PERCEPTUAL_LAG = 120
TERMINAL_WINDOW_FRAMES = 32
CADENCE_PARITY_IMBALANCE_THRESHOLD = 0.50


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _document_digest(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("document_sha256", None)
    return canonical_sha256(canonical)


def probe_video(path: str | Path) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-count_frames",
        "-show_entries",
        (
            "stream=codec_name,width,height,pix_fmt,time_base,start_time,duration,duration_ts,"
            "avg_frame_rate,r_frame_rate,nb_frames,nb_read_frames"
        ),
        "-show_entries",
        "frame=best_effort_timestamp,pkt_duration",
        "-show_entries",
        "format=format_name,start_time,duration",
        "-of",
        "json",
        str(Path(path).resolve()),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
    return json.loads(completed.stdout)


def _as_fraction(value: Any, *, label: str, errors: list[str]) -> Fraction | None:
    if value in (None, "", "N/A"):
        errors.append(f"missing {label}")
        return None
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        errors.append(f"invalid {label}: {value!r}")
        return None


def validate_probe_contract(
    probe: dict[str, Any],
    *,
    expected_frames: int = EXPECTED_FRAME_COUNT,
    expected_fps: int = EXPECTED_FPS,
    expected_duration_seconds: int = EXPECTED_DURATION_SECONDS,
) -> dict[str, Any]:
    """Validate exact CFR, full PTS sequence, and duration using rational arithmetic."""

    if expected_frames < 1 or expected_fps < 1 or expected_duration_seconds < 1:
        raise ValueError("Expected video contract values must be positive integers.")
    errors: list[str] = []
    streams = probe.get("streams") or []
    if len(streams) != 1:
        errors.append(f"expected exactly one selected video stream, found {len(streams)}")
        stream: dict[str, Any] = {}
    else:
        stream = streams[0]
    frames = probe.get("frames") or []
    time_base = _as_fraction(stream.get("time_base"), label="stream.time_base", errors=errors)
    average_rate = _as_fraction(
        stream.get("avg_frame_rate"), label="stream.avg_frame_rate", errors=errors
    )
    real_rate = _as_fraction(stream.get("r_frame_rate"), label="stream.r_frame_rate", errors=errors)
    expected_rate = Fraction(expected_fps, 1)
    if average_rate is not None and average_rate != expected_rate:
        errors.append(f"avg_frame_rate is {average_rate}, expected exactly {expected_rate}")
    if real_rate is not None and real_rate != expected_rate:
        errors.append(f"r_frame_rate is {real_rate}, expected exactly {expected_rate}")

    raw_count = stream.get("nb_read_frames")
    if raw_count in (None, "", "N/A"):
        raw_count = stream.get("nb_frames")
    try:
        reported_count = int(raw_count)
    except (TypeError, ValueError):
        reported_count = -1
        errors.append(f"invalid reported frame count: {raw_count!r}")
    if reported_count != expected_frames:
        errors.append(f"reported frame count is {reported_count}, expected {expected_frames}")
    if len(frames) != expected_frames:
        errors.append(f"PTS frame record count is {len(frames)}, expected {expected_frames}")

    pts_seconds: list[Fraction] = []
    packet_durations: list[Fraction] = []
    if time_base is not None:
        for index, frame in enumerate(frames):
            try:
                pts_seconds.append(int(frame.get("best_effort_timestamp")) * time_base)
            except (TypeError, ValueError):
                errors.append(
                    f"frame {index} has invalid best_effort_timestamp "
                    f"{frame.get('best_effort_timestamp')!r}"
                )
                break
            raw_duration = frame.get("pkt_duration")
            if raw_duration not in (None, "", "N/A"):
                try:
                    packet_durations.append(int(raw_duration) * time_base)
                except (TypeError, ValueError):
                    errors.append(f"frame {index} has invalid pkt_duration {raw_duration!r}")
                    break
        if len(pts_seconds) == len(frames):
            bad_pts = [
                index
                for index, value in enumerate(pts_seconds)
                if value != Fraction(index, expected_fps)
            ]
            if bad_pts:
                errors.append(
                    "PTS sequence differs from exact frame_index/fps at indices "
                    f"{bad_pts[:12]}{'...' if len(bad_pts) > 12 else ''}"
                )
        if packet_durations and len(packet_durations) != len(frames):
            errors.append(
                "pkt_duration is present for only a subset of frames: "
                f"{len(packet_durations)}/{len(frames)}"
            )
        bad_durations = [
            index
            for index, value in enumerate(packet_durations)
            if value != Fraction(1, expected_fps)
        ]
        if bad_durations:
            errors.append(
                "packet durations differ from exact 1/fps at indices "
                f"{bad_durations[:12]}{'...' if len(bad_durations) > 12 else ''}"
            )

    expected_duration = Fraction(expected_duration_seconds, 1)
    stream_duration: Fraction | None
    if time_base is not None and stream.get("duration_ts") not in (None, "", "N/A"):
        try:
            stream_duration = int(stream["duration_ts"]) * time_base
        except (TypeError, ValueError):
            stream_duration = None
            errors.append(f"invalid stream.duration_ts: {stream.get('duration_ts')!r}")
    else:
        stream_duration = _as_fraction(
            stream.get("duration"), label="stream.duration", errors=errors
        )
    format_duration = _as_fraction(
        (probe.get("format") or {}).get("duration"), label="format.duration", errors=errors
    )
    if stream_duration is not None and stream_duration != expected_duration:
        errors.append(
            f"stream duration is {stream_duration}, expected exactly {expected_duration} seconds"
        )
    if format_duration is not None and format_duration != expected_duration:
        errors.append(
            f"format duration is {format_duration}, expected exactly {expected_duration} seconds"
        )

    return {
        "contract_pass": not errors,
        "errors": errors,
        "expected": {
            "frame_count": expected_frames,
            "fps": expected_fps,
            "duration_seconds": expected_duration_seconds,
            "first_pts_seconds": "0",
            "last_pts_seconds": f"{expected_frames - 1}/{expected_fps}",
            "pts_delta_seconds": f"1/{expected_fps}",
        },
        "observed": {
            "frame_count_reported": reported_count,
            "pts_record_count": len(frames),
            "avg_frame_rate": None if average_rate is None else str(average_rate),
            "r_frame_rate": None if real_rate is None else str(real_rate),
            "time_base": None if time_base is None else str(time_base),
            "first_pts_seconds": None if not pts_seconds else str(pts_seconds[0]),
            "last_pts_seconds": None if not pts_seconds else str(pts_seconds[-1]),
            "stream_duration_seconds": None if stream_duration is None else str(stream_duration),
            "format_duration_seconds": None if format_duration is None else str(format_duration),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "codec_name": stream.get("codec_name"),
            "pixel_format": stream.get("pix_fmt"),
        },
    }


def _rgb_uint8(frame: np.ndarray, *, index: int) -> np.ndarray:
    rgb = np.asarray(frame)
    if rgb.ndim != 3 or rgb.shape[2] not in {3, 4}:
        raise ValueError(f"Decoded frame {index} must be RGB/RGBA; got shape {rgb.shape}.")
    rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating) and rgb.size and float(rgb.max()) <= 1.0:
            rgb = np.rint(rgb * 255.0)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(rgb)


def _maximum_true_run(values: Sequence[bool]) -> int:
    maximum = current = 0
    for value in values:
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return maximum


def _finite_summary(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
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


def _temporal_diagnostics(
    perceptual_frames: Sequence[np.ndarray],
    frame_hashes: Sequence[str],
) -> dict[str, Any]:
    frame_count = len(frame_hashes)
    pair_differences = [
        float(
            np.mean(
                np.abs(
                    perceptual_frames[index].astype(np.float32)
                    - perceptual_frames[index - 1].astype(np.float32)
                )
            )
            / 255.0
        )
        for index in range(1, frame_count)
    ]
    exact_equal_pairs = [
        frame_hashes[index] == frame_hashes[index - 1] for index in range(1, frame_count)
    ]
    low_change_pairs = [value <= LOW_CHANGE_THRESHOLD for value in pair_differences]
    low_change_fraction = float(np.mean(low_change_pairs)) if low_change_pairs else 1.0
    frozen_candidate = bool(
        frame_count > 1
        and (
            low_change_fraction >= 0.95
            or _maximum_true_run(exact_equal_pairs) >= max(2, frame_count - 2)
        )
    )

    if pair_differences:
        differences = np.asarray(pair_differences, dtype=np.float64)
        median = float(np.median(differences))
        mad = float(np.median(np.abs(differences - median)))
        cut_threshold = max(
            CUT_ABSOLUTE_THRESHOLD,
            median + 8.0 * 1.4826 * max(mad, 1.0e-6),
        )
        cut_indices = [
            index + 1 for index, value in enumerate(pair_differences) if value >= cut_threshold
        ]
    else:
        median = mad = 0.0
        cut_threshold = CUT_ABSOLUTE_THRESHOLD
        cut_indices = []

    hash_groups: dict[str, list[int]] = defaultdict(list)
    for index, frame_hash in enumerate(frame_hashes):
        hash_groups[frame_hash].append(index)
    repeated_groups = [indices for indices in hash_groups.values() if len(indices) > 1]
    repeated_groups.sort(key=lambda values: (-len(values), values[0]))
    exact_long_lag_pairs = sum(
        1
        for indices in repeated_groups
        for position, left in enumerate(indices)
        for right in indices[position + 1 :]
        if right - left > 1
    )
    lag_rows: list[dict[str, Any]] = []
    if frame_count >= 3:
        stack = np.stack(perceptual_frames).astype(np.float32)
        for lag in range(2, min(MAX_PERCEPTUAL_LAG, frame_count - 1) + 1):
            lag_rows.append(
                {
                    "lag_frames": lag,
                    "mean_abs_difference": float(
                        np.mean(np.abs(stack[lag:] - stack[:-lag])) / 255.0
                    ),
                }
            )
        lag_rows.sort(key=lambda row: (row["mean_abs_difference"], row["lag_frames"]))
    best_lags = lag_rows[:10]

    even = pair_differences[0::2]
    odd = pair_differences[1::2]
    even_median = float(np.median(even)) if even else None
    odd_median = float(np.median(odd)) if odd else None
    if even_median is None or odd_median is None:
        parity_imbalance = None
    else:
        parity_imbalance = abs(even_median - odd_median) / max(even_median + odd_median, 1.0e-12)
    cadence_anomaly = bool(
        parity_imbalance is not None
        and parity_imbalance >= CADENCE_PARITY_IMBALANCE_THRESHOLD
        and max(even_median or 0.0, odd_median or 0.0) > LOW_CHANGE_THRESHOLD
    )
    second_differences = [
        abs(pair_differences[index] - pair_differences[index - 1])
        for index in range(1, len(pair_differences))
    ]

    return {
        "freeze_diagnostics": {
            "frozen_candidate": frozen_candidate,
            "low_change_threshold_normalized_mae": LOW_CHANGE_THRESHOLD,
            "low_change_pair_fraction": low_change_fraction,
            "maximum_low_change_pair_run": _maximum_true_run(low_change_pairs),
            "exact_equal_pair_fraction": (
                float(np.mean(exact_equal_pairs)) if exact_equal_pairs else 1.0
            ),
            "maximum_exact_equal_pair_run": _maximum_true_run(exact_equal_pairs),
            "interpretation": "candidate diagnostic requiring full-video manual confirmation",
        },
        "cut_diagnostics": {
            "abrupt_cut_candidate_indices": cut_indices,
            "candidate_count": len(cut_indices),
            "normalized_mae_threshold": cut_threshold,
            "pair_difference_median": median,
            "pair_difference_mad": mad,
            "interpretation": "abrupt-discontinuity candidates, not semantic scene-cut claims",
        },
        "long_lag_repetition_diagnostics": {
            "repeated_exact_hash_groups": repeated_groups,
            "exact_long_lag_duplicate_pair_count": exact_long_lag_pairs,
            "best_perceptual_lags": best_lags,
            "near_periodic_candidate": bool(
                best_lags and best_lags[0]["mean_abs_difference"] <= LOW_CHANGE_THRESHOLD
            ),
            "maximum_tested_lag_frames": min(MAX_PERCEPTUAL_LAG, max(0, frame_count - 1)),
            "interpretation": "exact/perceptual recurrence evidence, not scene identity",
        },
        "cadence_diagnostics": {
            "adjacent_pair_count": len(pair_differences),
            "adjacent_normalized_mae": pair_differences,
            "adjacent_normalized_mae_sha256": canonical_sha256(pair_differences),
            "adjacent_change_summary": _finite_summary(pair_differences),
            "absolute_second_difference_summary": _finite_summary(second_differences),
            "even_pair_median": even_median,
            "odd_pair_median": odd_median,
            "parity_imbalance": parity_imbalance,
            "parity_imbalance_threshold": CADENCE_PARITY_IMBALANCE_THRESHOLD,
            "alternating_cadence_anomaly_candidate": cadence_anomaly,
            "interpretation": "image-change cadence evidence requiring visual confirmation",
        },
    }


def analyze_full_frame_sequence(frames: Iterable[np.ndarray]) -> dict[str, Any]:
    """Fully decode and hash a sequence, then compute prompt-agnostic diagnostics."""

    hashes: list[str] = []
    perceptual_frames: list[np.ndarray] = []
    dimensions: tuple[int, int] | None = None
    for index, frame in enumerate(frames):
        rgb = _rgb_uint8(np.asarray(frame), index=index)
        observed_dimensions = (int(rgb.shape[1]), int(rgb.shape[0]))
        if dimensions is None:
            dimensions = observed_dimensions
        elif dimensions != observed_dimensions:
            raise ValueError(
                f"Decoded frame {index} changed dimensions from {dimensions} "
                f"to {observed_dimensions}."
            )
        hashes.append(hashlib.sha256(rgb.tobytes()).hexdigest())
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        perceptual_frames.append(cv2.resize(gray, ANALYSIS_SIZE, interpolation=cv2.INTER_AREA))

    diagnostics = _temporal_diagnostics(perceptual_frames, hashes)
    terminal_start = max(0, len(hashes) - TERMINAL_WINDOW_FRAMES)
    terminal_diagnostics = _temporal_diagnostics(
        perceptual_frames[terminal_start:], hashes[terminal_start:]
    )
    return {
        "full_decode": {
            "decode_completed": True,
            "decoded_frame_count": len(hashes),
            "width": None if dimensions is None else dimensions[0],
            "height": None if dimensions is None else dimensions[1],
            "rgb_frame_sha256": hashes,
            "rgb_frame_sha256_list_sha256": canonical_sha256(hashes),
            "unique_rgb_frame_hashes": len(set(hashes)),
        },
        **diagnostics,
        "terminal_window_diagnostics": {
            "start_frame_inclusive": terminal_start,
            "end_frame_inclusive": max(-1, len(hashes) - 1),
            "frame_count": len(hashes) - terminal_start,
            **terminal_diagnostics,
        },
    }


def _manifest_digest(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    return canonical_sha256(canonical)


def _load_source_bindings(
    video_path: Path,
    result_path: Path,
    manifest_path: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    if result.get("status") != "completed":
        raise ValueError(f"Benchmark result is not completed: {result_path}")
    job = result.get("job")
    if not isinstance(job, dict):
        raise ValueError("Benchmark result is missing its frozen job mapping.")
    generation = job.get("generation") or {}
    if generation.get("task") != "text_to_video":
        raise ValueError("Full-video audit accepts only text_to_video benchmark results.")
    exact_contract = {
        "num_frames": EXPECTED_FRAME_COUNT,
        "fps": EXPECTED_FPS,
        "duration_seconds": float(EXPECTED_DURATION_SECONDS),
    }
    for key, expected in exact_contract.items():
        observed = generation.get(key)
        if isinstance(expected, float):
            matches = (
                isinstance(observed, (int, float))
                and not isinstance(observed, bool)
                and math.isclose(float(observed), expected, rel_tol=0.0, abs_tol=1.0e-12)
            )
        else:
            matches = observed == expected
        if not matches:
            raise ValueError(
                f"Frozen job generation.{key} is {observed!r}, expected exact {expected!r}."
            )

    media_paths = result.get("validated_media_paths")
    if not isinstance(media_paths, list) or len(media_paths) != 1:
        raise ValueError("Completed result must bind exactly one validated media path.")
    bound_media = Path(str(media_paths[0])).expanduser()
    if not bound_media.is_absolute():
        bound_media = (result_path.parent / bound_media).resolve()
    else:
        bound_media = bound_media.resolve()
    if bound_media != video_path:
        raise ValueError("Requested video differs from the result's validated media path.")
    media_validation = result.get("media_validation") or {}
    recorded_media_sha = str(media_validation.get("sha256", ""))
    actual_media_sha = sha256_file(video_path)
    if not recorded_media_sha or recorded_media_sha != actual_media_sha:
        raise ValueError("Video SHA-256 differs from benchmark result media validation.")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    declared_manifest_sha = str(manifest.get("manifest_sha256", ""))
    actual_manifest_digest = _manifest_digest(manifest)
    if not declared_manifest_sha or declared_manifest_sha != actual_manifest_digest:
        raise ValueError("Source manifest's canonical manifest_sha256 is missing or invalid.")
    sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
    if not sidecar.is_file() or not sidecar.read_text(encoding="utf-8").split():
        raise FileNotFoundError(f"Source manifest digest sidecar is missing: {sidecar}")
    if sidecar.read_text(encoding="utf-8").split()[0] != declared_manifest_sha:
        raise ValueError("Source manifest digest sidecar does not match manifest_sha256.")
    condition_id = str(job.get("condition_id", ""))
    manifest_jobs = manifest.get("jobs")
    if not isinstance(manifest_jobs, list):
        raise ValueError("Source manifest jobs field is malformed.")
    matches = [row for row in manifest_jobs if str(row.get("condition_id", "")) == condition_id]
    if len(matches) != 1 or matches[0] != job:
        raise ValueError(
            "Source manifest must contain exactly one byte-equivalent frozen job for condition_id."
        )

    bindings = {
        "source_manifest": {
            "path": str(manifest_path),
            "file_sha256": sha256_file(manifest_path),
            "manifest_sha256": declared_manifest_sha,
        },
        "benchmark_job_result": {
            "path": str(result_path),
            "sha256": sha256_file(result_path),
        },
        "video": {
            "path": str(video_path),
            "sha256": actual_media_sha,
            "size_bytes": video_path.stat().st_size,
        },
    }
    frozen_condition = {
        "condition_id": condition_id,
        "seed": job.get("seed"),
        "prompt_id": job.get("prompt_id"),
        "model_name": job.get("model_name"),
        "model_revision": job.get("model_revision"),
        "variation": job.get("variation"),
        "variant": job.get("variant"),
        "variant_spec": job.get("variant_spec"),
        "generation": generation,
        "temporal_protocol_snapshot": job.get("temporal_protocol_snapshot"),
    }
    return frozen_condition, bindings


def audit_full_video(
    video_path: str | Path,
    *,
    result_path: str | Path,
    manifest_path: str | Path,
    segmented_temporal_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    """Create sealed prompt-agnostic evidence for one immutable benchmark result."""

    video = Path(video_path).expanduser().resolve()
    result = Path(result_path).expanduser().resolve()
    manifest = Path(manifest_path).expanduser().resolve()
    for label, path in (("video", video), ("result", result), ("manifest", manifest)):
        if not path.is_file():
            raise FileNotFoundError(f"Full-video audit {label} does not exist: {path}")
    condition, bindings = _load_source_bindings(video, result, manifest)
    implementation = Path(__file__).resolve()
    bindings["audit_implementation"] = {
        "path": str(implementation),
        "sha256": sha256_file(implementation),
    }
    timing = validate_probe_contract(probe_video(video))
    reader = imageio.get_reader(str(video), format="ffmpeg")
    try:
        sequence = analyze_full_frame_sequence(reader)
    finally:
        reader.close()
    decode = sequence["full_decode"]
    decode_errors: list[str] = []
    if decode["decoded_frame_count"] != EXPECTED_FRAME_COUNT:
        decode_errors.append(
            f"decoded {decode['decoded_frame_count']} frames, expected {EXPECTED_FRAME_COUNT}"
        )
    if len(decode["rgb_frame_sha256"]) != decode["decoded_frame_count"]:
        decode_errors.append("decoded frame hash cardinality differs from decoded frame count")
    if (decode["width"], decode["height"]) != (
        timing["observed"]["width"],
        timing["observed"]["height"],
    ):
        decode_errors.append("full-decoder dimensions differ from ffprobe dimensions")
    decode["decode_contract_pass"] = not decode_errors
    decode["errors"] = decode_errors
    structural_pass = bool(timing["contract_pass"] and not decode_errors)
    temporal_snapshot = condition.get("temporal_protocol_snapshot") or {}
    temporal_protocol = temporal_snapshot.get("protocol") or {}
    if temporal_protocol.get("schema_version") == 2:
        segmented_path = (
            Path(segmented_temporal_audit_path).expanduser().resolve()
            if segmented_temporal_audit_path is not None
            else video.parent / "segmented_temporal_audit.json"
        )
        if not segmented_path.is_file():
            raise FileNotFoundError(
                f"Schema-2 full-video audit requires segmented evidence: {segmented_path}"
            )
        segmented = json.loads(segmented_path.read_text(encoding="utf-8"))
        segmented_validation = validate_segmented_temporal_audit(
            segmented,
            verify_source_files=True,
        )
        if segmented_validation["condition_id"] != condition["condition_id"]:
            raise ValueError("Full/segmented audits identify different conditions.")
        if segmented.get("model_name") != condition["model_name"]:
            raise ValueError("Full/segmented audits identify different models.")
        if segmented.get("manifest_sha256") != (
            bindings["source_manifest"]["manifest_sha256"]
        ):
            raise ValueError("Full/segmented audits bind different launch manifests.")
        if (segmented.get("source") or {}).get("final_frame_sha256") != decode.get(
            "rgb_frame_sha256_list_sha256"
        ):
            raise ValueError("Full-video decoded frames differ from segmented evidence.")
        bindings["segmented_temporal_audit"] = {
            "path": str(segmented_path),
            "sha256": sha256_file(segmented_path),
            "document_sha256": segmented["document_sha256"],
            "final_frame_sha256": (segmented.get("source") or {}).get(
                "final_frame_sha256"
            ),
        }
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "audit": AUDIT_NAME,
        "source_bindings": bindings,
        "condition": condition,
        "contract": {
            "frame_count": EXPECTED_FRAME_COUNT,
            "fps": EXPECTED_FPS,
            "duration_seconds": EXPECTED_DURATION_SECONDS,
        },
        "timing_and_pts_contract": timing,
        **sequence,
        "structural_contract_pass": structural_pass,
        "evidence_contract_pass": structural_pass,
        "semantic_success_assessed": False,
        "manual_full_video_review_required": True,
        "scope_limitations": [
            "Diagnostics do not identify prompt objects, people, identity, or camera intent.",
            "Cut, freeze, repetition, and cadence candidates require full-video visual review.",
            "Prompt-specific motion evidence is supplied only by separate specialized audits.",
        ],
    }
    payload["document_sha256"] = _document_digest(payload)
    return payload


def validate_audit_document(
    payload: dict[str, Any],
    *,
    verify_source_files: bool = True,
) -> dict[str, Any]:
    """Validate a sealed audit and optionally re-hash every bound source file."""

    if payload.get("schema_version") != SCHEMA_VERSION or payload.get("audit") != AUDIT_NAME:
        raise ValueError("Unsupported full-video audit schema/name.")
    declared = str(payload.get("document_sha256", ""))
    actual = _document_digest(payload)
    if not declared or declared != actual:
        raise ValueError(
            f"Full-video audit document digest mismatch: declared={declared!r}, actual={actual}."
        )
    if payload.get("evidence_contract_pass") is not True:
        raise ValueError("Full-video audit did not pass its exact evidence contract.")
    decode = payload.get("full_decode") or {}
    hashes = decode.get("rgb_frame_sha256")
    if (
        decode.get("decode_contract_pass") is not True
        or not isinstance(hashes, list)
        or len(hashes) != EXPECTED_FRAME_COUNT
        or any(not isinstance(value, str) or len(value) != 64 for value in hashes)
        or canonical_sha256(hashes) != decode.get("rgb_frame_sha256_list_sha256")
    ):
        raise ValueError("Full-video decoded-frame hash coverage is missing or inconsistent.")
    required_evidence = (
        "freeze_diagnostics",
        "cut_diagnostics",
        "long_lag_repetition_diagnostics",
        "cadence_diagnostics",
        "terminal_window_diagnostics",
    )
    if any(not isinstance(payload.get(key), dict) for key in required_evidence):
        raise ValueError("Full-video temporal diagnostic evidence is incomplete.")
    cadence = payload["cadence_diagnostics"]
    pair_differences = cadence.get("adjacent_normalized_mae")
    if (
        not isinstance(pair_differences, list)
        or len(pair_differences) != EXPECTED_FRAME_COUNT - 1
        or canonical_sha256(pair_differences) != cadence.get("adjacent_normalized_mae_sha256")
    ):
        raise ValueError("Full-video cadence series is incomplete or has been altered.")

    if verify_source_files:
        bindings = payload.get("source_bindings") or {}
        hash_fields = {
            "source_manifest": "file_sha256",
            "benchmark_job_result": "sha256",
            "video": "sha256",
            "audit_implementation": "sha256",
        }
        for binding_name, hash_field in hash_fields.items():
            binding = bindings.get(binding_name)
            if not isinstance(binding, dict):
                raise ValueError(f"Full-video audit is missing source binding {binding_name!r}.")
            path = Path(str(binding.get("path", ""))).expanduser().resolve()
            expected_sha = str(binding.get(hash_field, ""))
            if not path.is_file() or not expected_sha or sha256_file(path) != expected_sha:
                raise ValueError(f"Full-video source binding changed: {binding_name}.")
        temporal_snapshot = (payload.get("condition") or {}).get(
            "temporal_protocol_snapshot"
        ) or {}
        if (temporal_snapshot.get("protocol") or {}).get("schema_version") == 2:
            binding = bindings.get("segmented_temporal_audit")
            if not isinstance(binding, dict):
                raise ValueError("Schema-2 full-video audit lacks segmented-audit binding.")
            segmented_path = Path(str(binding.get("path", ""))).expanduser().resolve()
            if not segmented_path.is_file() or sha256_file(segmented_path) != binding.get(
                "sha256"
            ):
                raise ValueError("Full-video segmented-audit source changed.")
            segmented = json.loads(segmented_path.read_text(encoding="utf-8"))
            validation = validate_segmented_temporal_audit(
                segmented,
                verify_source_files=True,
            )
            if validation["document_sha256"] != binding.get("document_sha256"):
                raise ValueError("Full-video segmented-audit document binding changed.")
            if binding.get("final_frame_sha256") != decode.get(
                "rgb_frame_sha256_list_sha256"
            ):
                raise ValueError("Full-video decoded-frame/segmented binding changed.")
    return {
        "status": "passed",
        "document_sha256": declared,
        "condition_id": (payload.get("condition") or {}).get("condition_id"),
        "source_files_rehashed": bool(verify_source_files),
    }


def write_immutable_audit(path: str | Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    """Write a sealed audit and digest sidecar without ever overwriting evidence."""

    validate_audit_document(payload, verify_source_files=True)
    output = Path(path).expanduser().resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite immutable audit evidence: {output}")
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(encoded)
    except BaseException:
        output.unlink(missing_ok=True)
        raise
    try:
        sidecar.write_text(f"{payload['document_sha256']}  {output.name}\n", encoding="utf-8")
        sidecar.chmod(0o444)
    except BaseException:
        output.unlink(missing_ok=True)
        sidecar.unlink(missing_ok=True)
        raise
    return output, sidecar


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create prompt-agnostic exact full-video qualification evidence."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--segmented-temporal-audit")
    parser.add_argument("--output-json", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    payload = audit_full_video(
        args.video,
        result_path=args.result_json,
        manifest_path=args.manifest,
        segmented_temporal_audit_path=args.segmented_temporal_audit,
    )
    output, sidecar = write_immutable_audit(args.output_json, payload)
    print(
        json.dumps(
            {
                "output_json": str(output),
                "sha256_sidecar": str(sidecar),
                "document_sha256": payload["document_sha256"],
                "evidence_contract_pass": payload["evidence_contract_pass"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
