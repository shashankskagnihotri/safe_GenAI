"""Exact temporal arithmetic for the HunyuanVideo 15-second pilot.

HunyuanVideo's native temporal grid uses ``4*k+1`` decoded frames.  The
frozen long-video pilot therefore denoises 361 frames: 360 half-open samples
cover 15 seconds at the model's native 24 fps and the final frame is the
alignment endpoint at exactly 15 seconds.  Saving those frames directly at
16 fps would slow the model trajectory and fabricate duration.  Instead this
module drops only the terminal endpoint and performs deterministic nearest-
timestamp 24-to-16 fps decimation:

``361 -> [0:360] -> 240``.

No frame is duplicated or synthesized.  Pixel hashes and every selected
source index are recorded so the conversion is auditable.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from typing import Any, Sequence

import numpy as np


@dataclass(frozen=True)
class HunyuanTemporalResamplingProvenance:
    """JSON-serializable proof of the duration-preserving frame conversion."""

    schema_version: int
    method: str
    inclusive_source_frame_count: int
    half_open_source_frame_count: int
    source_fps: int
    terminal_alignment_frame_index: int
    terminal_alignment_frame_dropped: bool
    output_frame_count: int
    output_fps: int
    duration_seconds: float
    selected_source_indices: list[int]
    dropped_source_indices: list[int]
    source_frame_pixel_sha256: list[str]
    output_frame_pixel_sha256: list[str]
    duplicated_source_indices: list[int]
    synthesized_frame_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def resample_hunyuan_native_24_to_16_exact(
    source_frames: Sequence[Any],
    *,
    source_fps: int = 24,
    output_fps: int = 16,
    duration_seconds: float = 15.0,
    output_frame_count: int = 240,
) -> tuple[list[Any], HunyuanTemporalResamplingProvenance]:
    """Decimate an inclusive native-rate trajectory without changing duration.

    Output sample ``j`` is selected from the nearest native timestamp using an
    integer round-half-up rule.  For 24-to-16 fps this alternates source-index
    increments of one and two.  Because this is downsampling, every selected
    source index must be unique and no new image may be synthesized.
    """

    if isinstance(source_fps, bool) or not isinstance(source_fps, int) or source_fps <= 0:
        raise ValueError("Hunyuan temporal source_fps must be a positive integer.")
    if isinstance(output_fps, bool) or not isinstance(output_fps, int) or output_fps <= 0:
        raise ValueError("Hunyuan temporal output_fps must be a positive integer.")
    if output_fps >= source_fps:
        raise ValueError(
            "Hunyuan temporal protocol is strict decimation; output_fps must be lower "
            f"than source_fps, got {source_fps} -> {output_fps}."
        )
    if duration_seconds <= 0:
        raise ValueError("Hunyuan temporal duration_seconds must be positive.")
    source_intervals = duration_seconds * source_fps
    output_samples = duration_seconds * output_fps
    if int(source_intervals) != source_intervals or int(output_samples) != output_samples:
        raise ValueError("Duration must contain integral source and output frame counts.")
    expected_inclusive_count = int(source_intervals) + 1
    expected_output_count = int(output_samples)
    if len(source_frames) != expected_inclusive_count:
        raise ValueError(
            f"Expected {expected_inclusive_count} inclusive Hunyuan source frames for "
            f"{duration_seconds}s at {source_fps} fps, got {len(source_frames)}."
        )
    if output_frame_count != expected_output_count:
        raise ValueError(
            f"Expected output_frame_count={expected_output_count}, got {output_frame_count}."
        )

    source_arrays = [_frame_to_rgb_uint8(frame) for frame in source_frames]
    reference_shape = source_arrays[0].shape
    if any(array.shape != reference_shape for array in source_arrays):
        raise ValueError("All Hunyuan temporal source frames must have identical RGB dimensions.")

    # Nearest timestamp with round-half-up, expressed entirely in integers to
    # avoid platform-dependent Python/binary floating-point tie handling.
    selected_indices: list[int] = []
    for output_index in range(output_frame_count):
        numerator = output_index * source_fps
        quotient, remainder = divmod(numerator, output_fps)
        source_index = quotient + int(2 * remainder >= output_fps)
        selected_indices.append(source_index)

    half_open_source_count = int(source_intervals)
    if not selected_indices or selected_indices[0] != 0:
        raise RuntimeError("Hunyuan temporal resampling must preserve the t=0 source frame.")
    if selected_indices[-1] >= half_open_source_count:
        raise RuntimeError(
            "Hunyuan temporal output selected the terminal alignment endpoint or an "
            "out-of-range source frame."
        )
    if any(right <= left for left, right in zip(selected_indices, selected_indices[1:])):
        raise RuntimeError(
            "Hunyuan 24-to-16 fps conversion duplicated or reordered a source frame."
        )

    output = [source_frames[index] for index in selected_indices]
    if len(output) != output_frame_count:
        raise RuntimeError(
            f"Hunyuan temporal conversion produced {len(output)} frames; "
            f"expected {output_frame_count}."
        )

    source_hashes = [_pixel_sha256(array) for array in source_arrays]
    output_hashes = [_pixel_sha256(_frame_to_rgb_uint8(frame)) for frame in output]
    for output_index, source_index in enumerate(selected_indices):
        if output_hashes[output_index] != source_hashes[source_index]:
            raise RuntimeError(
                f"Hunyuan source frame {source_index} was not preserved byte-exactly at "
                f"output index {output_index}."
            )

    selected_set = set(selected_indices)
    dropped_indices = [
        index for index in range(expected_inclusive_count) if index not in selected_set
    ]
    provenance = HunyuanTemporalResamplingProvenance(
        schema_version=1,
        method="nearest_timestamp_decimation_round_half_up",
        inclusive_source_frame_count=expected_inclusive_count,
        half_open_source_frame_count=half_open_source_count,
        source_fps=source_fps,
        terminal_alignment_frame_index=expected_inclusive_count - 1,
        terminal_alignment_frame_dropped=True,
        output_frame_count=len(output),
        output_fps=output_fps,
        duration_seconds=float(duration_seconds),
        selected_source_indices=selected_indices,
        dropped_source_indices=dropped_indices,
        source_frame_pixel_sha256=source_hashes,
        output_frame_pixel_sha256=output_hashes,
        duplicated_source_indices=[],
        synthesized_frame_count=0,
    )
    return output, provenance


def _frame_to_rgb_uint8(frame: Any) -> np.ndarray:
    if hasattr(frame, "convert"):
        array = np.asarray(frame.convert("RGB"))
    else:
        array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(
            "Hunyuan temporal frames must be RGB uint8 images with shape "
            f"[height, width, 3]; got shape={array.shape}, dtype={array.dtype}."
        )
    return np.ascontiguousarray(array)


def _pixel_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
