"""Fail-closed structural and motion audit for Prompt 03 benchmark videos.

The audit deliberately separates observations from semantic claims.  Dense optical
flow can provide evidence about image-plane motion inside manually bound
circulation ROIs; it cannot identify an escalator, prove that a stationary object
is a marble staircase, or establish that apparent global motion came from a
physical camera.  Those limitations are carried into every JSON result.

The production contract is exactly 240 frames at 16 CFR for 15 seconds.  Every
decoded RGB frame is SHA-256 hashed, and every presentation timestamp is checked
against the exact rational sequence ``frame_index / 16``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import defaultdict
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterable, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np


SCHEMA_VERSION = 1
PROMPT_ID = "03_empty_outdoor_mall"
CIRCULATION_PAIR_ID = "vertical_circulation_escalators_to_marble_stairs"
PROMPT3_PAIR_IDS = (
    "sky_color_blue_to_pink",
    CIRCULATION_PAIR_ID,
    "horizontal_floor_marble_to_tile",
    "signage_sale_to_new_arrival",
    "merchandise_handbags_to_cars",
)
EXPECTED_FRAME_COUNT = 240
EXPECTED_FPS = 16
EXPECTED_DURATION_SECONDS = 15
ANALYSIS_MAX_DIMENSION = 320
FLOW_PARAMETERS = {
    "pyr_scale": 0.5,
    "levels": 3,
    "winsize": 15,
    "iterations": 3,
    "poly_n": 5,
    "poly_sigma": 1.2,
    "flags": 0,
}
LOW_CHANGE_THRESHOLD = 2.0e-3
CUT_ABSOLUTE_THRESHOLD = 0.12
MOVING_THRESHOLD_PX_PER_FRAME = 0.20
STATIONARY_THRESHOLD_PX_PER_FRAME = 0.10
MIN_DIRECTIONAL_CONSISTENCY = 0.65
MIN_VALID_MOTION_PAIRS = 10


@dataclass(frozen=True)
class RoiSpec:
    """A manually bound circulation ROI and its positive tread-motion axis.

    Bounds are normalized ``(x0, y0, x1, y1)`` coordinates in ``[0, 1]``.
    ``axis_xy`` is an image-plane vector; only its direction is used. For each
    escalator, the positive axis must point from its lower landing toward its
    upper landing. Opposite signed projections then mean one tread moves toward
    its upper landing while the other moves toward its lower landing, even when
    perspective makes the two image-plane axes mirror each other. Visibility
    ranges restrict evidence to frame pairs where manual review confirms that
    both circulation structures still occupy the bound regions.
    """

    name: str
    bounds_normalized: tuple[float, float, float, float]
    axis_xy: tuple[float, float]
    active_frame_pair_ranges: tuple[tuple[int, int], ...] = ((0, EXPECTED_FRAME_COUNT - 2),)

    def validated(self) -> "RoiSpec":
        if not self.name.strip():
            raise ValueError("ROI names must be non-empty.")
        if len(self.bounds_normalized) != 4:
            raise ValueError(f"ROI {self.name!r} must have four normalized bounds.")
        x0, y0, x1, y1 = self.bounds_normalized
        if not all(math.isfinite(value) for value in self.bounds_normalized):
            raise ValueError(f"ROI {self.name!r} contains a non-finite bound.")
        if not (0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0):
            raise ValueError(
                f"ROI {self.name!r} bounds must satisfy 0 <= x0 < x1 <= 1 and "
                f"0 <= y0 < y1 <= 1; got {self.bounds_normalized}."
            )
        if len(self.axis_xy) != 2 or not all(math.isfinite(value) for value in self.axis_xy):
            raise ValueError(f"ROI {self.name!r} axis must be a finite two-vector.")
        if math.hypot(*self.axis_xy) <= 1.0e-12:
            raise ValueError(f"ROI {self.name!r} axis must be non-zero.")
        if not self.active_frame_pair_ranges:
            raise ValueError(f"ROI {self.name!r} must have at least one active frame-pair range.")
        previous_end = -1
        for raw_range in self.active_frame_pair_ranges:
            if len(raw_range) != 2 or any(
                isinstance(value, bool) or not isinstance(value, int) for value in raw_range
            ):
                raise ValueError(
                    f"ROI {self.name!r} active frame-pair ranges must be integer [start, end] pairs."
                )
            start, end = raw_range
            if not 0 <= start <= end <= EXPECTED_FRAME_COUNT - 2:
                raise ValueError(
                    f"ROI {self.name!r} active frame-pair range {raw_range} is outside "
                    f"0..{EXPECTED_FRAME_COUNT - 2}."
                )
            if start <= previous_end:
                raise ValueError(
                    f"ROI {self.name!r} active frame-pair ranges must be sorted and disjoint."
                )
            previous_end = end
        return self

    def is_active_for_pair(self, frame_pair_index: int) -> bool:
        return any(start <= frame_pair_index <= end for start, end in self.active_frame_pair_ranges)


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


def expected_condition_semantics(variant_spec: dict[str, Any]) -> dict[str, Any]:
    """Map benchmark variation metadata to its preregistered motion requirement."""

    kind = _normalize_variant_kind(str(variant_spec.get("kind", "")))
    active_pair_ids = tuple(str(value) for value in variant_spec.get("active_pair_ids", ()))
    if kind == "baseline":
        if active_pair_ids:
            raise ValueError("Baseline variation cannot have active steering pairs.")
        requirement = "opposing_escalator_motion"
        condition_class = "source_baseline"
    elif kind == "native_negative_prompt":
        if active_pair_ids:
            raise ValueError("Native-negative variation cannot have active steering pairs.")
        requirement = "source_suppression_only"
        condition_class = "native_negative_source_suppression"
    elif kind in {"conceptsteer", "shapley_concept_steering"}:
        if len(active_pair_ids) not in {1, 5} or len(set(active_pair_ids)) != len(active_pair_ids):
            raise ValueError(
                "Steering variations must carry exactly one or five unique active_pair_ids."
            )
        unknown = sorted(set(active_pair_ids) - set(PROMPT3_PAIR_IDS))
        if unknown or (len(active_pair_ids) == 5 and set(active_pair_ids) != set(PROMPT3_PAIR_IDS)):
            raise ValueError(
                f"Steering active_pair_ids differ from the Prompt 03 pair registry: {unknown}."
            )
        if CIRCULATION_PAIR_ID in active_pair_ids:
            requirement = "stationary_circulation_motion"
            condition_class = (
                "circulation_single_pair_target" if len(active_pair_ids) == 1 else "full_target"
            )
        else:
            requirement = "opposing_escalator_motion"
            condition_class = "non_circulation_single_pair_source_preservation"
    else:
        raise ValueError(f"Unsupported Prompt 03 variation kind {kind!r}.")
    return {
        "variant_kind": kind,
        "active_pair_ids": list(active_pair_ids),
        "condition_class": condition_class,
        "required_motion_evidence": requirement,
        "native_negative_target_achievement_allowed": False,
    }


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
        "frame=best_effort_timestamp,best_effort_timestamp_time,pkt_duration,pkt_duration_time",
        "-show_entries",
        "format=format_name,start_time,duration",
        "-of",
        "json",
        str(Path(path).resolve()),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
    return json.loads(result.stdout)


def _as_fraction(value: Any, *, field: str, errors: list[str]) -> Fraction | None:
    if value in (None, "", "N/A"):
        errors.append(f"missing {field}")
        return None
    try:
        return Fraction(str(value))
    except (ValueError, ZeroDivisionError):
        errors.append(f"invalid {field}: {value!r}")
        return None


def validate_probe_contract(
    probe: dict[str, Any],
    *,
    expected_frames: int = EXPECTED_FRAME_COUNT,
    expected_fps: int = EXPECTED_FPS,
    expected_duration_seconds: int = EXPECTED_DURATION_SECONDS,
) -> dict[str, Any]:
    """Validate one video stream and its exact rational CFR PTS sequence."""

    errors: list[str] = []
    streams = probe.get("streams") or []
    if len(streams) != 1:
        errors.append(f"expected exactly one selected video stream, found {len(streams)}")
        stream: dict[str, Any] = {}
    else:
        stream = streams[0]
    frames = probe.get("frames") or []
    time_base = _as_fraction(stream.get("time_base"), field="stream.time_base", errors=errors)
    avg_rate = _as_fraction(
        stream.get("avg_frame_rate"), field="stream.avg_frame_rate", errors=errors
    )
    real_rate = _as_fraction(stream.get("r_frame_rate"), field="stream.r_frame_rate", errors=errors)
    expected_rate = Fraction(expected_fps, 1)
    if avg_rate is not None and avg_rate != expected_rate:
        errors.append(f"avg_frame_rate is {avg_rate}, expected exactly {expected_rate}")
    if real_rate is not None and real_rate != expected_rate:
        errors.append(f"r_frame_rate is {real_rate}, expected exactly {expected_rate}")

    reported_count = stream.get("nb_read_frames")
    if reported_count in (None, "N/A"):
        reported_count = stream.get("nb_frames")
    try:
        parsed_count = int(reported_count)
    except (TypeError, ValueError):
        parsed_count = -1
        errors.append(f"invalid reported frame count: {reported_count!r}")
    if parsed_count != expected_frames:
        errors.append(f"reported frame count is {parsed_count}, expected {expected_frames}")
    if len(frames) != expected_frames:
        errors.append(f"PTS frame record count is {len(frames)}, expected {expected_frames}")

    expected_delta = Fraction(1, expected_fps)
    pts_seconds: list[Fraction] = []
    duration_seconds: list[Fraction] = []
    if time_base is not None:
        for index, frame in enumerate(frames):
            raw_pts = frame.get("best_effort_timestamp")
            try:
                pts_seconds.append(int(raw_pts) * time_base)
            except (TypeError, ValueError):
                errors.append(f"frame {index} has invalid best_effort_timestamp {raw_pts!r}")
                break
            raw_duration = frame.get("pkt_duration")
            if raw_duration not in (None, "", "N/A"):
                try:
                    duration_seconds.append(int(raw_duration) * time_base)
                except (TypeError, ValueError):
                    errors.append(f"frame {index} has invalid pkt_duration {raw_duration!r}")
                    break
        if len(pts_seconds) == len(frames):
            if pts_seconds and pts_seconds[0] != 0:
                errors.append(f"first PTS is {pts_seconds[0]}, expected exactly 0")
            bad_pts = [
                index
                for index, value in enumerate(pts_seconds)
                if value != Fraction(index, expected_fps)
            ]
            if bad_pts:
                errors.append(
                    "PTS sequence differs from exact frame_index/fps at frame indices "
                    f"{bad_pts[:12]}{'...' if len(bad_pts) > 12 else ''}"
                )
        if duration_seconds and len(duration_seconds) != len(frames):
            errors.append(
                "pkt_duration is present for only a subset of frames: "
                f"{len(duration_seconds)}/{len(frames)}"
            )
        if duration_seconds:
            bad_durations = [
                index for index, value in enumerate(duration_seconds) if value != expected_delta
            ]
            if bad_durations:
                errors.append(
                    "packet durations differ from exact 1/fps at frame indices "
                    f"{bad_durations[:12]}{'...' if len(bad_durations) > 12 else ''}"
                )

    expected_duration = Fraction(expected_duration_seconds, 1)
    stream_duration: Fraction | None = None
    if time_base is not None and stream.get("duration_ts") not in (None, "", "N/A"):
        try:
            stream_duration = int(stream["duration_ts"]) * time_base
        except (TypeError, ValueError):
            errors.append(f"invalid stream.duration_ts: {stream.get('duration_ts')!r}")
    else:
        stream_duration = _as_fraction(
            stream.get("duration"), field="stream.duration", errors=errors
        )
    format_payload = probe.get("format") or {}
    format_duration = _as_fraction(
        format_payload.get("duration"), field="format.duration", errors=errors
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
            "first_pts_seconds": 0,
            "last_pts_seconds": f"{expected_frames - 1}/{expected_fps}",
            "pts_delta_seconds": f"1/{expected_fps}",
        },
        "observed": {
            "frame_count_reported": parsed_count,
            "pts_record_count": len(frames),
            "avg_frame_rate": None if avg_rate is None else str(avg_rate),
            "r_frame_rate": None if real_rate is None else str(real_rate),
            "time_base": None if time_base is None else str(time_base),
            "first_pts_seconds": None if not pts_seconds else str(pts_seconds[0]),
            "last_pts_seconds": None if not pts_seconds else str(pts_seconds[-1]),
            "stream_duration_seconds": (None if stream_duration is None else str(stream_duration)),
            "format_duration_seconds": (None if format_duration is None else str(format_duration)),
            "width": stream.get("width"),
            "height": stream.get("height"),
            "codec_name": stream.get("codec_name"),
            "pixel_format": stream.get("pix_fmt"),
        },
    }


def _analysis_gray(frame: np.ndarray) -> np.ndarray:
    rgb = np.asarray(frame)
    if rgb.ndim != 3 or rgb.shape[2] not in {3, 4}:
        raise ValueError(f"Decoded frame must have RGB/RGBA shape, got {rgb.shape}.")
    if rgb.shape[2] == 4:
        rgb = rgb[..., :3]
    if rgb.dtype != np.uint8:
        if np.issubdtype(rgb.dtype, np.floating) and rgb.size and rgb.max() <= 1.0:
            rgb = np.rint(rgb * 255.0)
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    height, width = rgb.shape[:2]
    scale = min(1.0, ANALYSIS_MAX_DIMENSION / max(height, width))
    if scale < 1.0:
        rgb = cv2.resize(
            rgb,
            (max(1, round(width * scale)), max(1, round(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


def _roi_slice(spec: RoiSpec, shape: tuple[int, int]) -> tuple[slice, slice]:
    height, width = shape
    x0, y0, x1, y1 = spec.bounds_normalized
    left = min(width - 1, max(0, math.floor(x0 * width)))
    right = min(width, max(left + 1, math.ceil(x1 * width)))
    top = min(height - 1, max(0, math.floor(y0 * height)))
    bottom = min(height, max(top + 1, math.ceil(y1 * height)))
    if right - left < 8 or bottom - top < 8:
        raise ValueError(
            f"ROI {spec.name!r} is smaller than 8x8 at analysis resolution {width}x{height}."
        )
    return slice(top, bottom), slice(left, right)


def _flow_observation(
    previous: np.ndarray,
    current: np.ndarray,
    rois: Sequence[RoiSpec],
) -> dict[str, Any]:
    flow = cv2.calcOpticalFlowFarneback(previous, current, None, **FLOW_PARAMETERS)
    grad_x = cv2.Sobel(previous, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(previous, cv2.CV_32F, 0, 1, ksize=3)
    textured = np.hypot(grad_x, grad_y) >= 4.0
    background = np.ones(previous.shape, dtype=bool)
    roi_slices: dict[str, tuple[slice, slice]] = {}
    for spec in rois:
        rows, columns = _roi_slice(spec, previous.shape)
        roi_slices[spec.name] = (rows, columns)
        background[rows, columns] = False
    valid_background = background & textured & np.isfinite(flow).all(axis=2)
    if int(valid_background.sum()) >= 128:
        global_xy = np.median(flow[valid_background], axis=0)
        global_valid = True
    else:
        global_xy = np.array([0.0, 0.0], dtype=np.float32)
        global_valid = False

    roi_payload: dict[str, Any] = {}
    for spec in rois:
        rows, columns = roi_slices[spec.name]
        roi_flow = flow[rows, columns]
        roi_texture = textured[rows, columns]
        valid = roi_texture & np.isfinite(roi_flow).all(axis=2)
        if int(valid.sum()) >= 32 and global_valid:
            local_xy = np.median(roi_flow[valid], axis=0)
            residual_xy = local_xy - global_xy
            axis = np.asarray(spec.axis_xy, dtype=np.float64)
            axis /= np.linalg.norm(axis)
            projection = float(np.dot(residual_xy, axis))
            residual_magnitude = float(np.linalg.norm(residual_xy))
            valid_motion = True
        else:
            local_xy = np.array([0.0, 0.0], dtype=np.float32)
            residual_xy = np.array([0.0, 0.0], dtype=np.float32)
            projection = 0.0
            residual_magnitude = 0.0
            valid_motion = False
        roi_payload[spec.name] = {
            "valid": valid_motion,
            "textured_pixel_count": int(valid.sum()),
            "local_median_flow_xy": [float(local_xy[0]), float(local_xy[1])],
            "camera_subtracted_flow_xy": [
                float(residual_xy[0]),
                float(residual_xy[1]),
            ],
            "axis_projection_px_per_frame": projection,
            "residual_magnitude_px_per_frame": residual_magnitude,
        }
    return {
        "global_valid": global_valid,
        "global_background_median_flow_xy": [float(global_xy[0]), float(global_xy[1])],
        "global_textured_pixel_count": int(valid_background.sum()),
        "rois": roi_payload,
    }


def _maximum_true_run(values: Sequence[bool]) -> int:
    best = current = 0
    for value in values:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _finite_summary(values: Sequence[float]) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not len(array):
        return {"median": None, "p10": None, "p90": None, "maximum": None}
    return {
        "median": float(np.median(array)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "maximum": float(np.max(array)),
    }


def analyze_frame_sequence(
    frames: Iterable[np.ndarray],
    rois: Sequence[RoiSpec] = (),
) -> dict[str, Any]:
    """Fully consume decoded frames and return hash, discontinuity, and motion evidence."""

    validated_rois = tuple(spec.validated() for spec in rois)
    if len({spec.name for spec in validated_rois}) != len(validated_rois):
        raise ValueError("ROI names must be unique.")
    if len(validated_rois) not in {0, 2}:
        raise ValueError(
            "Prompt 03 motion classification requires either zero or exactly two ROIs."
        )
    cv2.setNumThreads(1)
    cv2.setRNGSeed(0)

    hashes: list[str] = []
    perceptual_frames: list[np.ndarray] = []
    pair_differences: list[float] = []
    pair_hash_equal: list[bool] = []
    flow_rows: list[dict[str, Any]] = []
    dimensions: tuple[int, int] | None = None
    previous_gray: np.ndarray | None = None
    previous_hash: str | None = None

    for index, frame in enumerate(frames):
        rgb = np.asarray(frame)
        if rgb.ndim != 3 or rgb.shape[2] not in {3, 4}:
            raise ValueError(f"Decoded frame {index} has invalid shape {rgb.shape}.")
        if dimensions is None:
            dimensions = (int(rgb.shape[1]), int(rgb.shape[0]))
        elif dimensions != (int(rgb.shape[1]), int(rgb.shape[0])):
            raise ValueError(
                f"Decoded frame {index} changed dimensions from {dimensions} to "
                f"{(rgb.shape[1], rgb.shape[0])}."
            )
        rgb_bytes = np.ascontiguousarray(rgb[..., :3]).tobytes()
        frame_hash = hashlib.sha256(rgb_bytes).hexdigest()
        hashes.append(frame_hash)
        gray = _analysis_gray(rgb)
        perceptual_frames.append(cv2.resize(gray, (16, 16), interpolation=cv2.INTER_AREA))
        if previous_gray is not None:
            difference = float(
                np.mean(np.abs(gray.astype(np.float32) - previous_gray.astype(np.float32))) / 255.0
            )
            pair_differences.append(difference)
            pair_hash_equal.append(frame_hash == previous_hash)
            flow_rows.append(_flow_observation(previous_gray, gray, validated_rois))
        previous_gray = gray
        previous_hash = frame_hash

    frame_count = len(hashes)
    hash_groups: dict[str, list[int]] = defaultdict(list)
    for index, value in enumerate(hashes):
        hash_groups[value].append(index)
    repeated_groups = [indices for indices in hash_groups.values() if len(indices) > 1]
    repeated_groups.sort(key=lambda values: (-len(values), values[0]))
    exact_long_lag_pairs = sum(
        1
        for indices in repeated_groups
        for position, left_index in enumerate(indices)
        for right_index in indices[position + 1 :]
        if right_index - left_index > 1
    )

    low_change = [value <= LOW_CHANGE_THRESHOLD for value in pair_differences]
    low_change_fraction = float(np.mean(low_change)) if low_change else 1.0
    exact_equal_fraction = float(np.mean(pair_hash_equal)) if pair_hash_equal else 1.0
    frozen_candidate = bool(
        frame_count > 1
        and (
            low_change_fraction >= 0.95
            or _maximum_true_run(pair_hash_equal) >= max(2, frame_count - 2)
        )
    )

    if pair_differences:
        differences = np.asarray(pair_differences, dtype=np.float64)
        median_difference = float(np.median(differences))
        mad = float(np.median(np.abs(differences - median_difference)))
        cut_threshold = max(
            CUT_ABSOLUTE_THRESHOLD,
            median_difference + 8.0 * 1.4826 * max(mad, 1.0e-6),
        )
        cut_indices = [
            index + 1 for index, value in enumerate(pair_differences) if value >= cut_threshold
        ]
    else:
        median_difference = 0.0
        mad = 0.0
        cut_threshold = CUT_ABSOLUTE_THRESHOLD
        cut_indices = []

    lag_diagnostics: list[dict[str, Any]] = []
    if len(perceptual_frames) >= 3:
        stack = np.stack(perceptual_frames).astype(np.float32)
        for lag in range(2, min(120, frame_count - 1) + 1):
            score = float(np.mean(np.abs(stack[lag:] - stack[:-lag])) / 255.0)
            lag_diagnostics.append({"lag_frames": lag, "mean_abs_difference": score})
        lag_diagnostics.sort(key=lambda row: (row["mean_abs_difference"], row["lag_frames"]))
    best_lags = lag_diagnostics[:5]

    excluded_pair_indices = {index - 1 for index in cut_indices}
    pairwise_flow_evidence = [
        {
            "frame_pair": [pair_index, pair_index + 1],
            "excluded_as_cut_candidate": pair_index in excluded_pair_indices,
            "active_circulation_rois": [
                spec.name for spec in validated_rois if spec.is_active_for_pair(pair_index)
            ],
            **row,
        }
        for pair_index, row in enumerate(flow_rows)
    ]
    global_vectors: list[list[float]] = []
    roi_aggregates: dict[str, dict[str, Any]] = {}
    for pair_index, row in enumerate(flow_rows):
        if pair_index in excluded_pair_indices or not row["global_valid"]:
            continue
        global_vectors.append(row["global_background_median_flow_xy"])
    if global_vectors:
        global_array = np.asarray(global_vectors, dtype=np.float64)
        cumulative = np.cumsum(global_array, axis=0)
        global_summary = {
            "valid_pair_count": len(global_vectors),
            "median_flow_xy_px_per_frame": [
                float(np.median(global_array[:, 0])),
                float(np.median(global_array[:, 1])),
            ],
            "cumulative_image_plane_displacement_xy_px": [
                float(cumulative[-1, 0]),
                float(cumulative[-1, 1]),
            ],
            "total_image_plane_path_length_px": float(np.linalg.norm(global_array, axis=1).sum()),
        }
    else:
        global_summary = {
            "valid_pair_count": 0,
            "median_flow_xy_px_per_frame": None,
            "cumulative_image_plane_displacement_xy_px": None,
            "total_image_plane_path_length_px": None,
        }

    for spec in validated_rois:
        projections: list[float] = []
        magnitudes: list[float] = []
        for pair_index, row in enumerate(flow_rows):
            if pair_index in excluded_pair_indices or not spec.is_active_for_pair(pair_index):
                continue
            observation = row["rois"][spec.name]
            if observation["valid"]:
                projections.append(float(observation["axis_projection_px_per_frame"]))
                magnitudes.append(float(observation["residual_magnitude_px_per_frame"]))
        projection_median = float(np.median(projections)) if projections else None
        magnitude_median = float(np.median(magnitudes)) if magnitudes else None
        if projections and projection_median is not None and abs(projection_median) > 1.0e-12:
            sign = math.copysign(1.0, projection_median)
            directional_consistency = float(
                np.mean([math.copysign(1.0, value) == sign for value in projections if value])
            )
        else:
            directional_consistency = None
        roi_aggregates[spec.name] = {
            "bounds_normalized": list(spec.bounds_normalized),
            "positive_axis_xy": list(spec.axis_xy),
            "active_frame_pair_ranges": [
                list(frame_range) for frame_range in spec.active_frame_pair_ranges
            ],
            "valid_pair_count": len(projections),
            "axis_projection_px_per_frame": _finite_summary(projections),
            "residual_magnitude_px_per_frame": _finite_summary(magnitudes),
            "median_directional_consistency": directional_consistency,
            "motion_state": _roi_motion_state(
                projection_median,
                magnitude_median,
                directional_consistency,
                len(projections),
            ),
        }

    circulation = _classify_circulation_motion(
        roi_aggregates,
        frozen_candidate=frozen_candidate,
    )
    return {
        "full_decode": {
            "decode_completed": True,
            "decoded_frame_count": frame_count,
            "width": None if dimensions is None else dimensions[0],
            "height": None if dimensions is None else dimensions[1],
            "rgb_frame_sha256": hashes,
            "unique_rgb_frame_hashes": len(hash_groups),
        },
        "freeze_diagnostics": {
            "frozen_candidate": frozen_candidate,
            "interpretation": "candidate diagnostic, not a semantic scene judgment",
            "low_change_threshold_normalized_mae": LOW_CHANGE_THRESHOLD,
            "low_change_pair_fraction": low_change_fraction,
            "maximum_low_change_pair_run": _maximum_true_run(low_change),
            "exact_equal_pair_fraction": exact_equal_fraction,
            "maximum_exact_equal_pair_run": _maximum_true_run(pair_hash_equal),
        },
        "cut_diagnostics": {
            "abrupt_cut_candidate_indices": cut_indices,
            "candidate_count": len(cut_indices),
            "normalized_mae_threshold": cut_threshold,
            "pair_difference_median": median_difference,
            "pair_difference_mad": mad,
            "interpretation": "abrupt-discontinuity candidates requiring visual confirmation",
        },
        "repetition_diagnostics": {
            "repeated_exact_hash_groups": repeated_groups,
            "exact_long_lag_duplicate_pair_count": exact_long_lag_pairs,
            "best_perceptual_lags": best_lags,
            "near_periodic_candidate": bool(
                best_lags and best_lags[0]["mean_abs_difference"] <= LOW_CHANGE_THRESHOLD
            ),
            "interpretation": "hash/low-resolution recurrence diagnostic, not scene identity",
        },
        "global_background_motion": {
            **global_summary,
            "method": "median dense Farneback image-plane flow outside manual ROIs",
            "physical_camera_motion_proven": False,
        },
        "pairwise_flow_evidence": pairwise_flow_evidence,
        "circulation_roi_motion": {
            "rois": roi_aggregates,
            **circulation,
            "method": "ROI median dense flow minus global background median flow",
            "structure_identity_proven": False,
            "marble_stairs_proven": False,
        },
    }


def _roi_motion_state(
    projection_median: float | None,
    magnitude_median: float | None,
    consistency: float | None,
    count: int,
) -> str:
    if count < MIN_VALID_MOTION_PAIRS or projection_median is None or magnitude_median is None:
        return "indeterminate"
    if magnitude_median <= STATIONARY_THRESHOLD_PX_PER_FRAME:
        return "stationary"
    if (
        abs(projection_median) >= MOVING_THRESHOLD_PX_PER_FRAME
        and consistency is not None
        and consistency >= MIN_DIRECTIONAL_CONSISTENCY
    ):
        return "positive_axis_motion" if projection_median > 0 else "negative_axis_motion"
    return "indeterminate"


def _classify_circulation_motion(
    roi_aggregates: dict[str, dict[str, Any]],
    *,
    frozen_candidate: bool,
) -> dict[str, Any]:
    if frozen_candidate:
        return {
            "observed_classification": "indeterminate",
            "reason": "clip-level freeze diagnostic prevents stationary-circulation inference",
        }
    if len(roi_aggregates) != 2:
        return {
            "observed_classification": "indeterminate",
            "reason": "exactly two manually bound circulation ROIs are required",
        }
    states = [row["motion_state"] for row in roi_aggregates.values()]
    if states.count("stationary") == 2:
        return {
            "observed_classification": "stationary",
            "reason": "both camera-subtracted ROI flows are below the stationary threshold",
        }
    directional = {"positive_axis_motion", "negative_axis_motion"}
    if all(state in directional for state in states):
        if states[0] != states[1]:
            return {
                "observed_classification": "opposing_motion",
                "reason": "the two ROI projections are reliable and have opposite signs",
            }
        return {
            "observed_classification": "same_direction_motion",
            "reason": "the two ROI projections are reliable and have the same sign",
        }
    return {
        "observed_classification": "indeterminate",
        "reason": f"ROI motion states do not support a binary conclusion: {states}",
    }


def assess_condition(
    semantics: dict[str, Any],
    motion: dict[str, Any],
    *,
    structural_contract_pass: bool,
) -> dict[str, Any]:
    observed = str(motion.get("observed_classification", "indeterminate"))
    required = str(semantics["required_motion_evidence"])
    if not structural_contract_pass:
        status = "indeterminate"
        reason = "structural timing/decode contract failed"
    elif required == "source_suppression_only":
        status = "not_applicable_to_target_achievement"
        reason = (
            "native-negative evidence can assess source-motion suppression only; it cannot "
            "establish the marble-stairs target"
        )
    elif observed == "indeterminate":
        status = "indeterminate"
        reason = str(motion.get("reason", "motion evidence is indeterminate"))
    elif required == "opposing_escalator_motion":
        status = "pass" if observed == "opposing_motion" else "fail"
        reason = f"required opposing_motion; observed {observed}"
    elif required == "stationary_circulation_motion":
        status = "pass" if observed == "stationary" else "fail"
        reason = f"required stationary; observed {observed}"
    else:  # pragma: no cover - expected_condition_semantics constrains this value.
        raise ValueError(f"Unknown motion requirement {required!r}.")

    native_suppression = "not_applicable"
    if required == "source_suppression_only":
        if observed == "stationary":
            native_suppression = "motion_consistent_with_source_suppression"
        elif observed in {"opposing_motion", "same_direction_motion"}:
            native_suppression = "source_motion_still_observed"
        else:
            native_suppression = "indeterminate"
    return {
        "motion_requirement_status": status,
        "reason": reason,
        "required_motion_evidence": required,
        "observed_motion_evidence": observed,
        "native_negative_source_suppression_evidence": native_suppression,
        "target_structure_achievement": "not_assessed_by_motion_signal",
        "target_material_achievement": "not_assessed_by_motion_signal",
        "manual_visual_review_still_required": True,
    }


def load_roi_specs(path: str | Path) -> tuple[RoiSpec, RoiSpec]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = payload.get("rois") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or len(rows) != 2:
        raise ValueError("ROI JSON must contain exactly two entries in a 'rois' list.")
    specs = tuple(
        RoiSpec(
            name=str(row["name"]),
            bounds_normalized=tuple(float(value) for value in row["bounds_normalized"]),
            axis_xy=tuple(float(value) for value in row["axis_xy"]),
            active_frame_pair_ranges=tuple(
                tuple(int(value) for value in frame_range)
                for frame_range in row.get(
                    "active_frame_pair_ranges",
                    ((0, EXPECTED_FRAME_COUNT - 2),),
                )
            ),
        ).validated()
        for row in rows
    )
    if len({spec.name for spec in specs}) != 2:
        raise ValueError("ROI JSON names must be unique.")
    return specs  # type: ignore[return-value]


def _load_source_binding(
    video_path: Path,
    result_path: Path | None,
    manifest_path: Path | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    binding: dict[str, Any] = {
        "video": {"path": str(video_path), "sha256": sha256_file(video_path)}
    }
    result: dict[str, Any] = {}
    if result_path is not None:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        binding["benchmark_job_result"] = {
            "path": str(result_path),
            "sha256": sha256_file(result_path),
        }
        media_paths = [
            Path(str(value)).resolve() for value in result.get("validated_media_paths", ())
        ]
        if video_path not in media_paths:
            raise ValueError(
                f"Video {video_path} is not one of the result's validated_media_paths."
            )
        recorded_sha = (result.get("media_validation") or {}).get("sha256")
        if recorded_sha is not None and str(recorded_sha) != binding["video"]["sha256"]:
            raise ValueError(
                "Video SHA-256 differs from benchmark_job_result.json media_validation."
            )
    if manifest_path is not None:
        binding["source_manifest"] = {
            "path": str(manifest_path),
            "sha256": sha256_file(manifest_path),
        }
    return result, binding


def audit_video(
    video_path: str | Path,
    *,
    variant_spec: dict[str, Any],
    rois: Sequence[RoiSpec] = (),
    result_path: str | Path | None = None,
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    path = Path(video_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Prompt 03 video does not exist: {path}")
    resolved_result = None if result_path is None else Path(result_path).resolve()
    resolved_manifest = None if manifest_path is None else Path(manifest_path).resolve()
    source_result, binding = _load_source_binding(path, resolved_result, resolved_manifest)
    frozen_condition: dict[str, Any] = {}
    if source_result:
        job = source_result.get("job") or {}
        if job.get("prompt_id") != PROMPT_ID:
            raise ValueError(
                f"Motion audit accepts only prompt_id={PROMPT_ID!r}; got {job.get('prompt_id')!r}."
            )
        if (job.get("generation") or {}).get("task") != "text_to_video":
            raise ValueError("Prompt 03 motion audit accepts only text_to_video results.")
        frozen_variant = job.get("variant_spec") or {}
        if frozen_variant != variant_spec:
            raise ValueError("Provided variant_spec differs from benchmark_job_result.json.")
        frozen_condition = {
            "condition_id": job.get("condition_id"),
            "seed": job.get("seed"),
            "prompt_id": job.get("prompt_id"),
            "model_name": job.get("model_name"),
            "model_revision": job.get("model_revision"),
            "variation": job.get("variation"),
            "variant": job.get("variant"),
            "variant_spec": frozen_variant,
        }

    semantics = expected_condition_semantics(variant_spec)
    probe = probe_video(path)
    timing = validate_probe_contract(probe)
    reader = imageio.get_reader(str(path), format="ffmpeg")
    try:
        sequence = analyze_frame_sequence(reader, rois)
    finally:
        reader.close()
    decoded = sequence["full_decode"]
    decode_errors: list[str] = []
    if decoded["decoded_frame_count"] != EXPECTED_FRAME_COUNT:
        decode_errors.append(
            f"decoded {decoded['decoded_frame_count']} frames, expected {EXPECTED_FRAME_COUNT}"
        )
    if len(decoded["rgb_frame_sha256"]) != decoded["decoded_frame_count"]:
        decode_errors.append("decoded frame hash cardinality differs from decoded frame count")
    probe_width = timing["observed"]["width"]
    probe_height = timing["observed"]["height"]
    if (decoded["width"], decoded["height"]) != (probe_width, probe_height):
        decode_errors.append(
            "decoder dimensions differ from ffprobe: "
            f"decoded={decoded['width']}x{decoded['height']}, "
            f"probe={probe_width}x{probe_height}"
        )
    decoded["decode_contract_pass"] = not decode_errors
    decoded["errors"] = decode_errors
    structural_contract_pass = bool(timing["contract_pass"] and not decode_errors)
    motion = sequence["circulation_roi_motion"]
    assessment = assess_condition(
        semantics,
        motion,
        structural_contract_pass=structural_contract_pass,
    )

    roi_payload = [asdict(spec.validated()) for spec in rois]
    audit_parameters = {
        "schema_version": SCHEMA_VERSION,
        "expected_frame_count": EXPECTED_FRAME_COUNT,
        "expected_fps": EXPECTED_FPS,
        "expected_duration_seconds": EXPECTED_DURATION_SECONDS,
        "analysis_max_dimension": ANALYSIS_MAX_DIMENSION,
        "flow_parameters": FLOW_PARAMETERS,
        "low_change_threshold": LOW_CHANGE_THRESHOLD,
        "cut_absolute_threshold": CUT_ABSOLUTE_THRESHOLD,
        "moving_threshold_px_per_frame": MOVING_THRESHOLD_PX_PER_FRAME,
        "stationary_threshold_px_per_frame": STATIONARY_THRESHOLD_PX_PER_FRAME,
        "minimum_directional_consistency": MIN_DIRECTIONAL_CONSISTENCY,
        "rois": roi_payload,
        "semantics": semantics,
    }
    implementation_path = Path(__file__).resolve()
    binding["audit_implementation"] = {
        "path": str(implementation_path),
        "sha256": sha256_file(implementation_path),
    }
    payload = {
        "schema_version": SCHEMA_VERSION,
        "audit": "prompt03_video_structural_motion_v1",
        "prompt_id": PROMPT_ID,
        "condition": frozen_condition,
        "source_bindings": binding,
        "audit_parameters_sha256": canonical_sha256(audit_parameters),
        "audit_parameters": audit_parameters,
        "timing_and_pts_contract": timing,
        **sequence,
        "structural_contract_pass": structural_contract_pass,
        "condition_semantics": semantics,
        "condition_assessment": assessment,
        "scope_limitations": [
            "No person or mannequin detector is run by this module.",
            "Global background image-plane flow does not prove physical camera motion.",
            "ROI flow does not identify escalators or prove marble-stair structure/material.",
            "Abrupt-cut, freeze, and repetition flags are diagnostics requiring visual review.",
        ],
    }
    payload["document_sha256"] = canonical_sha256(payload)
    return payload


def validate_audit_document(
    payload: dict[str, Any],
    *,
    verify_source_files: bool = True,
) -> dict[str, Any]:
    """Validate a sealed Prompt-03 audit and all of its bound source files."""

    if (
        payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("audit") != "prompt03_video_structural_motion_v1"
        or payload.get("prompt_id") != PROMPT_ID
    ):
        raise ValueError("Unsupported Prompt-03 motion-audit schema/name/prompt.")
    canonical = dict(payload)
    declared = str(canonical.pop("document_sha256", ""))
    actual = canonical_sha256(canonical)
    if not declared or declared != actual:
        raise ValueError(
            f"Prompt-03 motion-audit digest mismatch: declared={declared!r}, actual={actual}."
        )
    if payload.get("structural_contract_pass") is not True:
        raise ValueError("Prompt-03 motion audit did not pass its structural contract.")
    parameters = payload.get("audit_parameters")
    if not isinstance(parameters, dict) or canonical_sha256(parameters) != payload.get(
        "audit_parameters_sha256"
    ):
        raise ValueError("Prompt-03 motion-audit parameter binding is inconsistent.")
    decode = payload.get("full_decode") or {}
    hashes = decode.get("rgb_frame_sha256")
    if (
        decode.get("decode_contract_pass") is not True
        or decode.get("decoded_frame_count") != EXPECTED_FRAME_COUNT
        or not isinstance(hashes, list)
        or len(hashes) != EXPECTED_FRAME_COUNT
        or any(not isinstance(value, str) or len(value) != 64 for value in hashes)
    ):
        raise ValueError("Prompt-03 motion audit lacks an exact complete full decode.")
    if (
        not isinstance(payload.get("pairwise_flow_evidence"), list)
        or len(payload["pairwise_flow_evidence"]) != EXPECTED_FRAME_COUNT - 1
    ):
        raise ValueError("Prompt-03 motion audit lacks all 239 frame-pair flow records.")
    if verify_source_files:
        bindings = payload.get("source_bindings") or {}
        for binding_name in (
            "benchmark_job_result",
            "video",
            "source_manifest",
            "audit_implementation",
        ):
            binding = bindings.get(binding_name)
            if not isinstance(binding, dict):
                raise ValueError(f"Prompt-03 audit source binding {binding_name!r} is missing.")
            path = Path(str(binding.get("path", ""))).expanduser().resolve()
            if not path.is_file() or sha256_file(path) != binding.get("sha256"):
                raise ValueError(f"Prompt-03 audit source binding changed: {binding_name}.")
    return {
        "status": "passed",
        "document_sha256": declared,
        "condition_id": (payload.get("condition") or {}).get("condition_id"),
        "source_files_rehashed": bool(verify_source_files),
    }


def write_immutable_audit(path: str | Path, payload: dict[str, Any]) -> tuple[Path, Path]:
    """Publish a Prompt-03 motion audit and digest sidecar without overwrite."""

    validate_audit_document(payload, verify_source_files=True)
    output = Path(path).expanduser().resolve()
    sidecar = output.with_suffix(output.suffix + ".sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite immutable Prompt-03 audit: {output}")
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
        description="Audit exact Prompt 03 video structure and manually bound circulation motion."
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--result-json", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument(
        "--roi-json",
        help=(
            "JSON with exactly two rois; omit only when a structural-only audit with an "
            "explicitly indeterminate motion result is intended."
        ),
    )
    parser.add_argument(
        "--variant-kind",
        choices=(
            "baseline",
            "native_negative_prompt",
            "conceptsteer",
            "shapley_concept_steering",
            "conceptsteer_current_repaired",
            "hierasafe_chs_v2",
            "midsteer",
            "sgf_switch_adapted",
            "safe_denoiser_switch_adapted",
        ),
        help="Diagnostic override is retained for API compatibility; immutable CLI audits use the result job.",
    )
    parser.add_argument("--active-pair-id", action="append", default=[])
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result_path = Path(args.result_json).resolve()
    source_result = json.loads(result_path.read_text(encoding="utf-8"))
    variant_spec = dict((source_result.get("job") or {}).get("variant_spec") or {})
    rois: Sequence[RoiSpec] = () if args.roi_json is None else load_roi_specs(args.roi_json)
    payload = audit_video(
        args.video,
        variant_spec=variant_spec,
        rois=rois,
        result_path=result_path,
        manifest_path=args.manifest,
    )
    output_path, sidecar = write_immutable_audit(args.output_json, payload)
    print(
        json.dumps(
            {
                "output_json": str(output_path),
                "sha256_sidecar": str(sidecar),
                **payload["condition_assessment"],
            }
        )
    )


if __name__ == "__main__":  # pragma: no cover
    main()
