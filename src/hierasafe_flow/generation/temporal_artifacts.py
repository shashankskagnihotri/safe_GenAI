from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from PIL import Image

from hierasafe_flow.generation.conditioning_cache import canonical_json_bytes


_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SLICE_RE = re.compile(r"^segment_(\d+)\[(\d+):(\d+)\]$")

_REGISTERED_ROUTE_CONTRACTS: Mapping[str, Mapping[str, Any]] = {
    "official_family_t2v_i2v_three_segment_composite": {
        "roles": ("t2v_primary", "i2v_continuation", "i2v_continuation"),
        "frame_counts": (49, 49, 49),
        "native_fps": (8, 8, 8),
        "local_steps": (50, 50, 50),
        "retained": (tuple(range(49)), tuple(range(1, 49)), tuple(range(1, 25))),
        "scheduler_classes": (
            "CogVideoXDDIMScheduler",
            "CogVideoXDDIMScheduler",
            "CogVideoXDDIMScheduler",
        ),
        "stitch_strategy": "segment_0[0:49]+segment_1[1:49]+segment_2[1:25]",
        "native_frame_count": 121,
        "postprocess_method": "pinned_rife_midpoint_once_over_complete_stitch",
    },
    "community_diffusers_t2v_i2v_three_segment_composite": {
        "roles": ("t2v_primary", "i2v_continuation", "i2v_continuation"),
        "frame_counts": (121, 121, 121),
        "native_fps": (24, 24, 24),
        "local_steps": (50, 50, 50),
        "retained": (tuple(range(121)), tuple(range(1, 121)), tuple(range(1, 121))),
        "scheduler_classes": (
            "FlowMatchEulerDiscreteScheduler",
            "FlowMatchEulerDiscreteScheduler",
            "FlowMatchEulerDiscreteScheduler",
        ),
        "scheduler_shifts": (7.0, 17.0, 17.0),
        "stitch_strategy": "segment_0[0:121]+segment_1[1:121]+segment_2[1:121]",
        "native_frame_count": 361,
        "postprocess_method": "nearest_timestamp_decimation_round_half_up",
    },
    "reference_conditioned_multishot": {
        "roles": ("base_t2av", "memory_t2av", "memory_t2av"),
        "frame_counts": (121, 121, 121),
        "native_fps": (24, 24, 24),
        "local_steps": (8, 8, 8),
        "retained": (tuple(range(120)), tuple(range(120)), tuple(range(120))),
        "stitch_strategy": "segment_0[0:120]+segment_1[0:120]+segment_2[0:120]",
        "native_frame_count": 360,
        "postprocess_method": "nearest_timestamp_decimation_round_half_up",
    },
}


@dataclass(frozen=True)
class TemporalSegmentEvidence:
    segment_index: int
    model_role: str
    model_id: str
    model_revision: str
    segment_seed: int
    native_fps: int
    frames: tuple[Any, ...] = field(repr=False, compare=False)
    retained_indices: tuple[int, ...] = ()
    discarded_indices: tuple[int, ...] = ()
    anchor_sha256: str | None = None
    reconstruction_index: int | None = None
    first_motion_index: int | None = None
    scheduler: Mapping[str, Any] = field(default_factory=dict)
    conditioning: Mapping[str, Any] = field(default_factory=dict)
    protected_state: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if isinstance(self.segment_index, bool) or self.segment_index < 0:
            raise ValueError("Temporal segment index must be a non-negative integer.")
        if not self.model_role or not self.model_id or not self.model_revision:
            raise ValueError("Temporal segment model role, ID, and revision must be non-empty.")
        if isinstance(self.segment_seed, bool) or self.segment_seed < 0:
            raise ValueError("Temporal segment seed must be a non-negative integer.")
        if isinstance(self.native_fps, bool) or self.native_fps <= 0:
            raise ValueError("Temporal segment native fps must be positive.")
        if not self.frames:
            raise ValueError("Temporal segment evidence requires complete decoded native frames.")
        frame_count = len(self.frames)
        retained = tuple(self.retained_indices or range(frame_count))
        discarded = tuple(self.discarded_indices)
        _validate_index_set(retained, frame_count, "retained_indices")
        _validate_index_set(discarded, frame_count, "discarded_indices")
        if set(retained) & set(discarded):
            raise ValueError("Retained and discarded segment indices must be disjoint.")
        if set(retained) | set(discarded) != set(range(frame_count)):
            raise ValueError("Retained/discarded indices must partition the complete segment.")
        object.__setattr__(self, "retained_indices", retained)
        object.__setattr__(self, "discarded_indices", discarded)
        if self.anchor_sha256 is not None and not _SHA256_RE.fullmatch(self.anchor_sha256):
            raise ValueError("Temporal anchor digest must be canonical lowercase SHA-256.")
        for label, index in {
            "reconstruction_index": self.reconstruction_index,
            "first_motion_index": self.first_motion_index,
        }.items():
            if index is not None and not 0 <= index < frame_count:
                raise ValueError(f"{label} is outside the native segment.")


@dataclass(frozen=True)
class TemporalEvidenceBundle:
    """In-memory handoff from a segmented adapter to transactional saving."""

    temporal_protocol: Mapping[str, Any]
    segments: tuple[TemporalSegmentEvidence, ...]
    stitch_map: tuple[tuple[int, int], ...]
    output_fps: int
    output_frame_count: int
    stitch_strategy: str
    postprocess: Mapping[str, Any]
    scientific_label: str
    trace_schema_version: int = 1
    condition_id: str | None = None
    attempt: int | None = None
    manifest_sha256: str | None = None
    job_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    binding: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError("Temporal evidence bundle schema_version must be 1.")
        protocol = dict(self.temporal_protocol)
        if protocol.get("schema_version") != 2:
            raise ValueError("Temporal evidence requires temporal protocol schema 2.")
        if not self.segments:
            raise ValueError("Temporal evidence bundle requires native segments.")
        indices = [segment.segment_index for segment in self.segments]
        if indices != list(range(len(self.segments))):
            raise ValueError("Temporal evidence segment indices must be contiguous and ordered.")
        if self.output_fps <= 0 or self.output_frame_count <= 0:
            raise ValueError("Temporal evidence output clock and frame count must be positive.")
        if not self.stitch_strategy or not self.scientific_label:
            raise ValueError("Temporal evidence strategy and scientific label must be non-empty.")
        route = _registered_route_contract(protocol)
        if len(self.segments) != len(route["roles"]):
            raise ValueError("Temporal evidence segment count differs from the registered route.")
        retained_refs = {
            (segment.segment_index, index)
            for segment in self.segments
            for index in segment.retained_indices
        }
        if len(self.stitch_map) != len(retained_refs) or set(self.stitch_map) != retained_refs:
            raise ValueError(
                "Temporal stitch map must reference every retained native frame exactly once."
            )
        if len(set(self.stitch_map)) != len(self.stitch_map):
            raise ValueError("Temporal stitch map may not duplicate native frames.")
        for segment_index, frame_index in self.stitch_map:
            if not 0 <= segment_index < len(self.segments):
                raise ValueError("Temporal stitch map references an unknown segment.")
            if frame_index not in self.segments[segment_index].retained_indices:
                raise ValueError("Temporal stitch map references a discarded frame.")
        expected_stitch_map = tuple(
            (segment_index, frame_index)
            for segment_index, indices in enumerate(route["retained"])
            for frame_index in indices
        )
        if self.stitch_map != expected_stitch_map:
            raise ValueError("Temporal stitch map order differs from the registered route.")
        if self.stitch_strategy != route["stitch_strategy"]:
            raise ValueError("Temporal stitch strategy differs from the registered route.")
        _validate_bundle_topology(self, route)
        if self.manifest_sha256 is not None and not _SHA256_RE.fullmatch(
            self.manifest_sha256
        ):
            raise ValueError("manifest_sha256 must be canonical lowercase SHA-256.")
        canonical_json_bytes(protocol)
        canonical_json_bytes(dict(self.postprocess))
        canonical_json_bytes(dict(self.metadata))
        canonical_json_bytes(dict(self.binding))

    @property
    def temporal_protocol_sha256(self) -> str:
        return hashlib.sha256(canonical_json_bytes(dict(self.temporal_protocol))).hexdigest()

    def bind(self, **binding: Any) -> "TemporalEvidenceBundle":
        merged = dict(self.binding)
        merged.update(binding)
        return replace(self, binding=merged)

    def with_execution_identity(
        self,
        *,
        condition_id: str,
        attempt: int,
        manifest_sha256: str,
        job_id: str,
        metadata: Mapping[str, Any] | None = None,
    ) -> "TemporalEvidenceBundle":
        if not condition_id or not job_id:
            raise ValueError("Temporal execution identity requires condition_id and job_id.")
        if isinstance(attempt, bool) or attempt <= 0:
            raise ValueError("Temporal execution attempt must be a positive integer.")
        if not _SHA256_RE.fullmatch(manifest_sha256):
            raise ValueError("Temporal execution manifest digest is malformed.")
        merged_metadata = dict(self.metadata)
        if metadata:
            merged_metadata.update(dict(metadata))
        return replace(
            self,
            condition_id=condition_id,
            attempt=attempt,
            manifest_sha256=manifest_sha256,
            job_id=job_id,
            metadata=merged_metadata,
        )

    def with_runner_trace_evidence(
        self, trace: Sequence[Mapping[str, Any]]
    ) -> "TemporalEvidenceBundle":
        """Bind ordered call-role and protected-state records to every segment.

        Adapter evidence describes model-native fixed state.  The runner is the
        authoritative observer of cache hits and every base/unsafe/safe call,
        so publication combines both views before the transaction is written.
        """

        grouped: dict[int, list[Mapping[str, Any]]] = {
            index: [] for index in range(len(self.segments))
        }
        local_steps_by_segment: dict[int, int] = {}
        for segment in self.segments:
            local_steps = segment.scheduler.get(
                "num_inference_steps", segment.scheduler.get("denoising_steps")
            )
            if isinstance(local_steps, bool) or not isinstance(local_steps, int):
                raise ValueError("Temporal segment scheduler lacks an integer step count.")
            local_steps_by_segment[segment.segment_index] = local_steps
        expected_global_steps = sum(local_steps_by_segment.values())
        if len(trace) != expected_global_steps:
            raise ValueError("Temporal runner trace length differs from native schedules.")
        for expected_global_index, step in enumerate(trace):
            if not isinstance(step, Mapping):
                raise ValueError("Temporal runner trace step is malformed.")
            segment = step.get("segment")
            if not isinstance(segment, Mapping):
                raise ValueError("Temporal runner trace lacks segment identity.")
            segment_index = segment.get("segment_index")
            if segment_index not in grouped:
                raise ValueError("Temporal runner trace references an unknown segment.")
            if segment.get("global_step_index") != expected_global_index:
                raise ValueError("Temporal runner trace global ordering is invalid.")
            adapter_segment = self.segments[int(segment_index)]
            expected_step_identity = {
                "schema_version": 1,
                "global_num_steps": expected_global_steps,
                "segment_index": adapter_segment.segment_index,
                "segment_count": len(self.segments),
                "local_num_steps": local_steps_by_segment[adapter_segment.segment_index],
                "model_role": adapter_segment.model_role,
                "model_id": adapter_segment.model_id,
                "model_revision": adapter_segment.model_revision,
                "condition_epoch": adapter_segment.segment_index,
                "anchor_sha256": adapter_segment.anchor_sha256,
                "segment_seed": adapter_segment.segment_seed,
            }
            if any(
                segment.get(key) != value
                for key, value in expected_step_identity.items()
            ):
                raise ValueError("Temporal runner/adapter step identity differs.")
            grouped[int(segment_index)].append(step)

        rebound: list[TemporalSegmentEvidence] = []
        for segment in self.segments:
            steps = grouped[segment.segment_index]
            if not steps:
                raise ValueError(
                    f"Temporal segment {segment.segment_index} has no runner trace evidence."
                )
            if [step["segment"].get("local_step_index") for step in steps] != list(
                range(local_steps_by_segment[segment.segment_index])
            ):
                raise ValueError(
                    f"Temporal segment {segment.segment_index} local trace coverage is invalid."
                )
            condition_records: dict[str, Any] = {}
            ordered_condition_digests: list[str] = []
            protected_records: dict[str, Any] = {}
            ordered_protected_digests: list[str] = []
            for step in steps:
                identity = step["segment"]
                calls = step.get("condition_calls")
                if not isinstance(calls, list) or not calls:
                    raise ValueError("Temporal runner step lacks condition-call evidence.")
                for call in calls:
                    if not isinstance(call, Mapping):
                        raise ValueError("Temporal runner condition-call record is malformed.")
                    call_identity = call.get("identity")
                    if not isinstance(call_identity, Mapping):
                        raise ValueError("Temporal condition call lacks canonical identity.")
                    expected_call_identity = {
                        "segment_index": segment.segment_index,
                        "model_role": segment.model_role,
                        "model_id": segment.model_id,
                        "model_revision": segment.model_revision,
                        "condition_epoch": segment.segment_index,
                        "segment_seed": segment.segment_seed,
                        "anchor_sha256": segment.anchor_sha256,
                    }
                    if any(
                        call_identity.get(key) != value
                        for key, value in expected_call_identity.items()
                    ):
                        raise ValueError(
                            "Temporal condition-call identity differs from adapter evidence."
                        )
                    optional_local_identity = {
                        "local_step_index": identity["local_step_index"],
                        "local_num_steps": identity["local_num_steps"],
                    }
                    if any(
                        key in call_identity and call_identity.get(key) != value
                        for key, value in optional_local_identity.items()
                    ):
                        raise ValueError(
                            "Temporal condition-call local context differs from its step."
                        )
                    digest = hashlib.sha256(canonical_json_bytes(dict(call))).hexdigest()
                    condition_records.setdefault(digest, dict(call))
                    ordered_condition_digests.append(digest)
                protected = step.get("protected_state")
                if not isinstance(protected, Mapping) or not protected:
                    raise ValueError("Temporal runner step lacks protected-state evidence.")
                protected_digest = hashlib.sha256(
                    canonical_json_bytes(dict(protected))
                ).hexdigest()
                protected_records.setdefault(protected_digest, dict(protected))
                ordered_protected_digests.append(protected_digest)

            conditioning = dict(segment.conditioning)
            conditioning["runner_condition_calls"] = {
                "schema_version": 1,
                "call_count": len(ordered_condition_digests),
                "call_roles": sorted(
                    {str(record["call_role"]) for record in condition_records.values()}
                ),
                "ordered_record_sha256": hashlib.sha256(
                    canonical_json_bytes(ordered_condition_digests)
                ).hexdigest(),
                "records": condition_records,
            }
            protected_state = dict(segment.protected_state)
            protected_state["runner_protected_state"] = {
                "schema_version": 1,
                "step_count": len(ordered_protected_digests),
                "ordered_record_sha256": hashlib.sha256(
                    canonical_json_bytes(ordered_protected_digests)
                ).hexdigest(),
                "records": protected_records,
            }
            rebound.append(
                replace(
                    segment,
                    conditioning=conditioning,
                    protected_state=protected_state,
                )
            )
        return replace(self, segments=tuple(rebound))


def temporal_evidence_to_dict(
    bundle: TemporalEvidenceBundle,
    *,
    segment_artifacts: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    artifacts = list(segment_artifacts or [{} for _ in bundle.segments])
    if len(artifacts) != len(bundle.segments):
        raise ValueError("Segment artifact records do not match the evidence segment count.")
    segments: list[dict[str, Any]] = []
    for segment, artifact in zip(bundle.segments, artifacts):
        frame_records = [_frame_record(frame) for frame in segment.frames]
        segments.append(
            {
                "segment_index": segment.segment_index,
                "model_role": segment.model_role,
                "model_id": segment.model_id,
                "model_revision": segment.model_revision,
                "segment_seed": segment.segment_seed,
                "native_fps": segment.native_fps,
                "decoded_frame_count": len(segment.frames),
                "decoded_shape": frame_records[0]["shape"],
                "decoded_dtype": frame_records[0]["dtype"],
                "color_interpretation": "RGB_uint8",
                "frame_sha256": [record["rgb_sha256"] for record in frame_records],
                "retained_indices": list(segment.retained_indices),
                "discarded_indices": list(segment.discarded_indices),
                "anchor_sha256": segment.anchor_sha256,
                "reconstruction_index": segment.reconstruction_index,
                "first_motion_index": segment.first_motion_index,
                "scheduler": dict(segment.scheduler),
                "conditioning": dict(segment.conditioning),
                "protected_state": dict(segment.protected_state),
                "lossless_artifacts": dict(artifact),
            }
        )
    stitched_frames = [
        bundle.segments[segment_index].frames[frame_index]
        for segment_index, frame_index in bundle.stitch_map
    ]
    return {
        "schema_version": 1,
        "evidence_schema_version": 1,
        "temporal_protocol_schema_version": 2,
        "temporal_protocol": dict(bundle.temporal_protocol),
        "temporal_protocol_sha256": bundle.temporal_protocol_sha256,
        "trace_schema_version": bundle.trace_schema_version,
        "scientific_label": bundle.scientific_label,
        "condition_id": bundle.condition_id,
        "attempt": bundle.attempt,
        "manifest_sha256": bundle.manifest_sha256,
        "job_id": bundle.job_id,
        "segments": segments,
        "stitch": {
            "strategy": bundle.stitch_strategy,
            "map": [list(item) for item in bundle.stitch_map],
            "native_frame_count": len(stitched_frames),
            "native_rgb_sha256": _frame_sequence_sha256(stitched_frames),
        },
        "postprocess": dict(bundle.postprocess),
        "output_contract": {
            "frame_count": bundle.output_frame_count,
            "fps": bundle.output_fps,
            "duration_seconds": bundle.output_frame_count / bundle.output_fps,
        },
        "metadata": dict(bundle.metadata),
        "binding": dict(bundle.binding),
    }


def validate_temporal_evidence_record(
    record: Mapping[str, Any],
    *,
    require_final_binding: bool = True,
) -> dict[str, Any]:
    if record.get("schema_version") != 1:
        raise RuntimeError("Temporal evidence record schema_version must be 1.")
    if record.get("evidence_schema_version") != 1:
        raise RuntimeError("Temporal evidence nested schema version must be 1.")
    if record.get("temporal_protocol_schema_version") != 2:
        raise RuntimeError("Temporal evidence must explicitly bind protocol schema 2.")
    if not isinstance(record.get("condition_id"), str) or not record["condition_id"]:
        raise RuntimeError("Temporal evidence condition_id is missing.")
    if (
        isinstance(record.get("attempt"), bool)
        or not isinstance(record.get("attempt"), int)
        or record["attempt"] <= 0
    ):
        raise RuntimeError("Temporal evidence attempt must be a positive integer.")
    if not isinstance(record.get("job_id"), str) or not record["job_id"]:
        raise RuntimeError("Temporal evidence job_id is missing.")
    if not isinstance(record.get("manifest_sha256"), str) or not _SHA256_RE.fullmatch(
        record["manifest_sha256"]
    ):
        raise RuntimeError("Temporal evidence manifest SHA-256 is missing or malformed.")
    protocol = record.get("temporal_protocol")
    if not isinstance(protocol, Mapping) or protocol.get("schema_version") != 2:
        raise RuntimeError("Temporal evidence record is not bound to protocol schema 2.")
    protocol_sha = hashlib.sha256(canonical_json_bytes(dict(protocol))).hexdigest()
    if record.get("temporal_protocol_sha256") != protocol_sha:
        raise RuntimeError("Temporal evidence protocol digest mismatch.")
    segments = record.get("segments")
    if not isinstance(segments, list) or not segments:
        raise RuntimeError("Temporal evidence record has no native segments.")
    route = _registered_route_contract(protocol)
    if len(segments) != len(route["roles"]):
        raise RuntimeError("Temporal evidence segment count differs from its frozen protocol.")
    for expected_index, segment in enumerate(segments):
        if not isinstance(segment, Mapping) or segment.get("segment_index") != expected_index:
            raise RuntimeError("Temporal evidence segment ordering is invalid.")
        frame_hashes = segment.get("frame_sha256")
        if not isinstance(frame_hashes, list) or len(frame_hashes) != segment.get(
            "decoded_frame_count"
        ):
            raise RuntimeError("Temporal evidence native frame hashes are incomplete.")
        if any(not isinstance(value, str) or not _SHA256_RE.fullmatch(value) for value in frame_hashes):
            raise RuntimeError("Temporal evidence contains a malformed native frame digest.")
        _validate_record_segment_topology(segment, expected_index, route)
        artifact = segment.get("lossless_artifacts")
        if not isinstance(artifact, Mapping):
            raise RuntimeError("Temporal evidence segment artifact record is missing.")
        if require_final_binding:
            for key in ("ffv1_mkv_sha256", "ffv1_mkv_path"):
                if not artifact.get(key):
                    raise RuntimeError(f"Temporal segment is missing lossless artifact field {key}.")
            windows = artifact.get("lossless_png_windows")
            if not isinstance(windows, list):
                raise RuntimeError("Temporal segment lossless PNG windows are missing.")
            observed_window_indices = [window.get("frame_index") for window in windows]
            expected_window_indices = _lossless_window_indices(segment)
            if observed_window_indices != expected_window_indices:
                raise RuntimeError(
                    "Temporal segment PNG windows differ from the exact registered indices."
                )
            for window in windows:
                if not isinstance(window, Mapping) or not window.get("path") or not _SHA256_RE.fullmatch(
                    str(window.get("sha256", ""))
                ):
                    raise RuntimeError("Temporal segment PNG window record is malformed.")
    stitch = record.get("stitch")
    expected_stitch_map = [
        [segment_index, frame_index]
        for segment_index, indices in enumerate(route["retained"])
        for frame_index in indices
    ]
    if not isinstance(stitch, Mapping) or stitch.get("map") != expected_stitch_map:
        raise RuntimeError("Temporal evidence stitch map/order differs from the frozen route.")
    if stitch.get("strategy") != route["stitch_strategy"] or stitch.get(
        "native_frame_count"
    ) != route["native_frame_count"]:
        raise RuntimeError("Temporal evidence stitch identity/arithmetic is invalid.")
    postprocess = record.get("postprocess")
    if not isinstance(postprocess, Mapping) or postprocess.get("method") != route[
        "postprocess_method"
    ]:
        raise RuntimeError("Temporal evidence postprocess route is invalid.")
    output = record.get("output_contract")
    if not isinstance(output, Mapping) or output.get("frame_count") != 240 or output.get("fps") != 16:
        raise RuntimeError("Temporal evidence final contract must be exactly 240 frames at 16 fps.")
    if require_final_binding:
        binding = record.get("binding")
        required = {
            "final_decoded_rgb_sha256",
            "final_frame_sha256",
            "final_media_sha256",
            "trace_sha256",
            "report_sha256",
        }
        if not isinstance(binding, Mapping) or not required.issubset(binding):
            raise RuntimeError("Temporal evidence final media/trace/report binding is incomplete.")
        for key in required:
            value = binding[key]
            if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
                raise RuntimeError(f"Temporal evidence binding {key} is not a SHA-256 digest.")
        if not record.get("document_sha256_sidecar_path"):
            raise RuntimeError("Temporal evidence document digest sidecar path is missing.")
    return dict(record)


def read_temporal_evidence(path: Path) -> dict[str, Any]:
    """Reopen and authenticate a published temporal evidence transaction."""

    if not path.is_file():
        raise FileNotFoundError(f"Temporal evidence document is missing: {path}")
    raw = path.read_bytes()
    record = json.loads(raw)
    validate_temporal_evidence_record(record, require_final_binding=True)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    declared_sidecar = Path(str(record["document_sha256_sidecar_path"]))
    if declared_sidecar.resolve() != sidecar.resolve():
        raise RuntimeError("Temporal evidence sidecar path does not match its publication path.")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Temporal evidence digest sidecar is missing: {sidecar}")
    expected = sidecar.read_text(encoding="utf-8").split()[0]
    actual = hashlib.sha256(raw).hexdigest()
    if expected != actual:
        raise RuntimeError("Temporal evidence document digest sidecar mismatch.")
    root = path.parent.resolve()
    for segment in record["segments"]:
        artifacts = segment["lossless_artifacts"]
        ffv1_path = Path(str(artifacts["ffv1_mkv_path"])).resolve()
        if root not in ffv1_path.parents or not ffv1_path.is_file():
            raise RuntimeError("Temporal FFV1 artifact is missing or escapes the sample root.")
        if _file_sha256(ffv1_path) != artifacts["ffv1_mkv_sha256"]:
            raise RuntimeError("Temporal FFV1 artifact digest mismatch.")
        decoded_segment = decode_video_rgb_frames(ffv1_path)
        decoded_hashes = [frame_rgb_sha256(frame) for frame in decoded_segment]
        if decoded_hashes != segment["frame_sha256"]:
            raise RuntimeError("Temporal FFV1 decoded frames differ from native evidence.")
        if any(list(frame.shape) != segment["decoded_shape"] for frame in decoded_segment):
            raise RuntimeError("Temporal FFV1 decoded shape differs from native evidence.")
        windows = artifacts.get("lossless_png_windows", [])
        if [window.get("frame_index") for window in windows] != _lossless_window_indices(
            segment
        ):
            raise RuntimeError("Temporal PNG window indices differ from the sealed record.")
        for window in windows:
            window_path = Path(str(window["path"])).resolve()
            if root not in window_path.parents or not window_path.is_file():
                raise RuntimeError("Temporal PNG window is missing or escapes the sample root.")
            if _file_sha256(window_path) != window["sha256"]:
                raise RuntimeError("Temporal PNG window digest mismatch.")
            frame_index = int(window["frame_index"])
            expected_name = f"frame_{frame_index:04d}.png"
            if window_path.name != expected_name:
                raise RuntimeError("Temporal PNG window path/index binding is invalid.")
            with Image.open(window_path) as image:
                decoded_png = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if frame_rgb_sha256(decoded_png) != segment["frame_sha256"][frame_index]:
                raise RuntimeError("Temporal PNG pixels differ from their native frame binding.")
    binding = record["binding"]
    expected_files = {
        "final_media_sha256": root / "video_000.mp4",
        "trace_sha256": root / "steering_trace.json",
        "report_sha256": root / "report.json",
    }
    for digest_key, bound_path in expected_files.items():
        if not bound_path.is_file() or _file_sha256(bound_path) != binding[digest_key]:
            raise RuntimeError(f"Temporal evidence binding mismatch for {digest_key}.")
    decoded_final = decode_video_rgb_frames(expected_files["final_media_sha256"])
    if len(decoded_final) != record["output_contract"]["frame_count"]:
        raise RuntimeError("Temporal final media decoded frame count is inconsistent.")
    decoded_final_hashes = [frame_rgb_sha256(frame) for frame in decoded_final]
    if binding["final_decoded_rgb_sha256"] != frame_sequence_sha256(decoded_final):
        raise RuntimeError("Temporal final decoded RGB sequence binding mismatch.")
    if binding["final_frame_sha256"] != hashlib.sha256(
        canonical_json_bytes(decoded_final_hashes)
    ).hexdigest():
        raise RuntimeError("Temporal final decoded frame-list binding mismatch.")
    result = dict(record)
    result["document_sha256"] = actual
    result["document_sha256_sidecar_verified"] = True
    return result


def save_lossless_temporal_segments(
    bundle: TemporalEvidenceBundle,
    directory: Path,
) -> list[dict[str, Any]]:
    directory.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    for segment in bundle.segments:
        path = directory / f"segment_{segment.segment_index:02d}_native_ffv1.mkv"
        _save_ffv1(segment.frames, path, fps=segment.native_fps)
        window_dir = directory / f"segment_{segment.segment_index:02d}_lossless_windows"
        window_dir.mkdir(parents=True, exist_ok=True)
        window_indices = sorted(
            set(range(min(3, len(segment.frames))))
            | set(range(max(0, len(segment.frames) - 3), len(segment.frames)))
            | set(segment.discarded_indices)
            | {
                index
                for index in (segment.reconstruction_index, segment.first_motion_index)
                if index is not None
            }
        )
        png_records: list[dict[str, Any]] = []
        for frame_index in window_indices:
            frame_path = window_dir / f"frame_{frame_index:04d}.png"
            Image.fromarray(_as_rgb_uint8(segment.frames[frame_index]), mode="RGB").save(
                frame_path, format="PNG", compress_level=0
            )
            png_records.append(
                {
                    "frame_index": frame_index,
                    "path": str(frame_path),
                    "sha256": _file_sha256(frame_path),
                }
            )
        records.append(
            {
                "codec": "FFV1",
                "container": "Matroska",
                "pixel_format": "rgb24",
                "ffv1_mkv_path": str(path),
                "ffv1_mkv_sha256": _file_sha256(path),
                "lossless_png_windows": png_records,
            }
        )
    return records


def validate_staged_lossless_temporal_segments(
    bundle: TemporalEvidenceBundle,
    records: Sequence[Mapping[str, Any]],
    directory: Path,
) -> dict[str, Any]:
    """Decode and authenticate every staged native artifact before publication."""

    root = directory.resolve()
    if len(records) != len(bundle.segments):
        raise RuntimeError("Staged lossless records do not cover every native segment.")
    expected_files: set[Path] = set()
    decoded_frame_count = 0
    png_count = 0
    for segment, record in zip(bundle.segments, records, strict=True):
        if not isinstance(record, Mapping):
            raise RuntimeError("Staged lossless segment record is malformed.")
        if {
            "codec": record.get("codec"),
            "container": record.get("container"),
            "pixel_format": record.get("pixel_format"),
        } != {
            "codec": "FFV1",
            "container": "Matroska",
            "pixel_format": "rgb24",
        }:
            raise RuntimeError("Staged lossless codec contract drifted.")
        ffv1_path = Path(str(record.get("ffv1_mkv_path", ""))).resolve()
        expected_ffv1 = root / f"segment_{segment.segment_index:02d}_native_ffv1.mkv"
        if ffv1_path != expected_ffv1 or not ffv1_path.is_file():
            raise RuntimeError("Staged FFV1 path is missing or outside its exact topology.")
        expected_files.add(ffv1_path)
        if _file_sha256(ffv1_path) != record.get("ffv1_mkv_sha256"):
            raise RuntimeError("Staged FFV1 byte digest differs from its record.")
        decoded = decode_video_rgb_frames(ffv1_path)
        expected_hashes = [frame_rgb_sha256(frame) for frame in segment.frames]
        decoded_hashes = [frame_rgb_sha256(frame) for frame in decoded]
        if decoded_hashes != expected_hashes:
            raise RuntimeError("Staged FFV1 decoded pixels differ from native frames.")
        expected_shape = list(_as_rgb_uint8(segment.frames[0]).shape)
        if any(list(frame.shape) != expected_shape for frame in decoded):
            raise RuntimeError("Staged FFV1 decoded frame shape differs from native frames.")
        decoded_frame_count += len(decoded)

        windows = record.get("lossless_png_windows")
        if not isinstance(windows, list):
            raise RuntimeError("Staged lossless PNG window records are missing.")
        expected_indices = sorted(
            set(range(min(3, len(segment.frames))))
            | set(range(max(0, len(segment.frames) - 3), len(segment.frames)))
            | set(segment.discarded_indices)
            | {
                index
                for index in (segment.reconstruction_index, segment.first_motion_index)
                if index is not None
            }
        )
        if [window.get("frame_index") for window in windows] != expected_indices:
            raise RuntimeError("Staged lossless PNG indices differ from the registered set.")
        window_root = root / f"segment_{segment.segment_index:02d}_lossless_windows"
        for window, frame_index in zip(windows, expected_indices, strict=True):
            if not isinstance(window, Mapping):
                raise RuntimeError("Staged lossless PNG window record is malformed.")
            png_path = Path(str(window.get("path", ""))).resolve()
            expected_png = window_root / f"frame_{frame_index:04d}.png"
            if png_path != expected_png or not png_path.is_file():
                raise RuntimeError("Staged lossless PNG path/index topology is invalid.")
            expected_files.add(png_path)
            if _file_sha256(png_path) != window.get("sha256"):
                raise RuntimeError("Staged lossless PNG byte digest differs from its record.")
            with Image.open(png_path) as image:
                decoded_png = np.asarray(image.convert("RGB"), dtype=np.uint8)
            if frame_rgb_sha256(decoded_png) != expected_hashes[frame_index]:
                raise RuntimeError("Staged lossless PNG pixels differ from native frames.")
            png_count += 1
    observed_files = {path.resolve() for path in directory.rglob("*") if path.is_file()}
    if observed_files != expected_files:
        raise RuntimeError("Staged native artifact file topology contains missing or extra files.")
    return {
        "schema_version": 1,
        "status": "passed",
        "segment_count": len(bundle.segments),
        "decoded_native_frame_count": decoded_frame_count,
        "lossless_png_count": png_count,
    }


def final_media_binding(
    *,
    final_frames: Sequence[Any],
    media_path: Path,
    trace_bytes: bytes,
    report_bytes: bytes,
) -> dict[str, Any]:
    if len(final_frames) != 240:
        raise RuntimeError(
            f"Temporal final binding requires exactly 240 frames; got {len(final_frames)}."
        )
    decoded_frames = decode_video_rgb_frames(media_path)
    if len(decoded_frames) != 240:
        raise RuntimeError(
            f"Encoded temporal media must decode to exactly 240 frames; got {len(decoded_frames)}."
        )
    frame_hashes = [_frame_record(frame)["rgb_sha256"] for frame in decoded_frames]
    return {
        "preencode_rgb_sha256": _frame_sequence_sha256(final_frames),
        "final_decoded_rgb_sha256": _frame_sequence_sha256(decoded_frames),
        "final_frame_sha256": hashlib.sha256(
            canonical_json_bytes(frame_hashes)
        ).hexdigest(),
        "final_media_sha256": _file_sha256(media_path),
        "trace_sha256": hashlib.sha256(trace_bytes).hexdigest(),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
    }


def frame_rgb_sha256(frame: Any) -> str:
    return hashlib.sha256(_as_rgb_uint8(frame).tobytes(order="C")).hexdigest()


def frame_sequence_sha256(frames: Sequence[Any]) -> str:
    return _frame_sequence_sha256(frames)


def decode_video_rgb_frames(path: str | Path) -> list[np.ndarray]:
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise RuntimeError("Temporal media verification requires imageio with FFmpeg.") from exc
    reader = imageio.get_reader(str(Path(path).resolve()), format="FFMPEG")
    try:
        frames = [_as_rgb_uint8(frame) for frame in reader]
    finally:
        reader.close()
    if not frames:
        raise RuntimeError(f"Temporal media decoded no RGB frames: {path}")
    return frames


def _frame_record(frame: Any) -> dict[str, Any]:
    array = _as_rgb_uint8(frame)
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "rgb_sha256": hashlib.sha256(array.tobytes(order="C")).hexdigest(),
    }


def _frame_sequence_sha256(frames: Sequence[Any]) -> str:
    digest = hashlib.sha256()
    digest.update(len(frames).to_bytes(8, "big"))
    for frame in frames:
        array = _as_rgb_uint8(frame)
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def _as_rgb_uint8(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        array = np.asarray(frame.convert("RGB"), dtype=np.uint8)
    elif isinstance(frame, torch.Tensor):
        tensor = frame.detach().cpu()
        if tensor.ndim == 3 and tensor.shape[0] in {1, 3, 4}:
            tensor = tensor.permute(1, 2, 0)
        array = tensor.numpy()
    else:
        array = np.asarray(frame)
    if array.ndim == 2:
        array = np.repeat(array[..., None], 3, axis=-1)
    if array.ndim != 3 or array.shape[-1] not in {1, 3, 4}:
        raise TypeError(f"Temporal frame is not an HxWxRGB-compatible image: {array.shape}.")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if array.shape[-1] == 4:
        array = array[..., :3]
    if array.dtype != np.uint8:
        array = array.astype(np.float32)
        minimum = float(np.nanmin(array))
        maximum = float(np.nanmax(array))
        if not np.isfinite(array).all():
            raise ValueError("Temporal frame contains non-finite pixels.")
        if minimum >= -1.0 and maximum <= 1.0 and minimum < 0.0:
            array = (array + 1.0) * 127.5
        elif minimum >= 0.0 and maximum <= 1.0:
            array = array * 255.0
        array = np.rint(np.clip(array, 0.0, 255.0)).astype(np.uint8)
    return np.ascontiguousarray(array)


def _save_ffv1(frames: Sequence[Any], path: Path, *, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except ModuleNotFoundError as exc:
        raise RuntimeError("Lossless temporal evidence requires imageio with FFmpeg.") from exc
    writer = imageio.get_writer(
        str(path),
        format="FFMPEG",
        mode="I",
        fps=fps,
        codec="ffv1",
        pixelformat="rgb24",
        macro_block_size=None,
        ffmpeg_log_level="error",
    )
    try:
        for frame in frames:
            writer.append_data(_as_rgb_uint8(frame))
    finally:
        writer.close()
    if not path.is_file() or path.stat().st_size <= 0:
        raise RuntimeError("FFV1 temporal segment encoding did not produce a file.")


def _validate_index_set(indices: tuple[int, ...], frame_count: int, label: str) -> None:
    if len(set(indices)) != len(indices):
        raise ValueError(f"Temporal {label} contains duplicates.")
    if any(isinstance(index, bool) or not isinstance(index, int) for index in indices):
        raise TypeError(f"Temporal {label} must contain integers.")
    if any(not 0 <= index < frame_count for index in indices):
        raise ValueError(f"Temporal {label} contains an out-of-range index.")


def _registered_route_contract(protocol: Mapping[str, Any]) -> Mapping[str, Any]:
    strategy = protocol.get("strategy")
    route = _REGISTERED_ROUTE_CONTRACTS.get(str(strategy))
    if route is None:
        raise ValueError(f"Temporal protocol strategy is not registered: {strategy!r}.")
    if protocol.get("schema_version") != 2:
        raise ValueError("Registered temporal route requires protocol schema version 2.")

    roles = tuple(route["roles"])
    frame_counts = tuple(route["frame_counts"])
    native_fps = tuple(route["native_fps"])
    local_steps = tuple(route["local_steps"])
    configured_segments = protocol.get("segments")
    if isinstance(configured_segments, list):
        normalized = tuple(
            (
                segment.get("index"),
                segment.get("role"),
                segment.get("frames"),
                segment.get("fps"),
                segment.get("steps", segment.get("denoising_steps")),
            )
            for segment in configured_segments
            if isinstance(segment, Mapping)
        )
        expected = tuple(
            (index, roles[index], frame_counts[index], native_fps[index], local_steps[index])
            for index in range(len(roles))
        )
        if normalized != expected:
            raise ValueError("Temporal protocol segment topology differs from its route.")
    else:
        if (
            protocol.get("segment_count") != len(roles)
            or tuple(protocol.get("segment_roles") or ()) != roles
            or tuple(protocol.get("native_segment_frames") or ()) != frame_counts
            or protocol.get("native_fps") != native_fps[0]
            or protocol.get("num_inference_steps_per_segment") != local_steps[0]
        ):
            raise ValueError("Temporal protocol scalar segment topology differs from its route.")

    registered_slices = [
        f"segment_{segment_index}[{indices[0]}:{indices[-1] + 1}]"
        for segment_index, indices in enumerate(route["retained"])
    ]
    stitch = protocol.get("stitch")
    if isinstance(stitch, Mapping):
        if stitch.get("retained") != registered_slices or stitch.get(
            "native_frame_count"
        ) != route["native_frame_count"]:
            raise ValueError("Temporal protocol retained-frame slices differ from its route.")
    else:
        if protocol.get("stitch_strategy") != route["stitch_strategy"] or protocol.get(
            "stitched_native_frames"
        ) != route["native_frame_count"]:
            raise ValueError("Temporal protocol stitch arithmetic differs from its route.")
    output_frames = protocol.get("output_frames")
    output_fps = protocol.get("output_fps")
    duration = protocol.get("duration_seconds")
    if str(strategy) == "official_family_t2v_i2v_three_segment_composite":
        postprocess = protocol.get("postprocess") or {}
        if postprocess.get("method") != route["postprocess_method"]:
            raise ValueError("Temporal protocol postprocess method differs from its route.")
        output_frames = postprocess.get("output_frames")
        output_fps = postprocess.get("output_fps")
        duration = postprocess.get("duration_seconds")
    elif str(strategy) == "community_diffusers_t2v_i2v_three_segment_composite":
        if protocol.get("resampling_method") != route["postprocess_method"]:
            raise ValueError("Temporal protocol resampling method differs from its route.")
    elif str(strategy) == "reference_conditioned_multishot":
        resampling = protocol.get("resampling") or {}
        if resampling.get("method") != route["postprocess_method"]:
            raise ValueError("Temporal protocol resampling method differs from its route.")
    if (output_frames, output_fps, float(duration or -1.0)) != (240, 16, 15.0):
        raise ValueError("Temporal protocol final media contract is not exact 240/16/15.")
    return route


def _validate_bundle_topology(
    bundle: TemporalEvidenceBundle, route: Mapping[str, Any]
) -> None:
    expected_discarded = tuple(
        tuple(sorted(set(range(frame_count)) - set(retained)))
        for frame_count, retained in zip(route["frame_counts"], route["retained"])
    )
    for index, segment in enumerate(bundle.segments):
        if (
            segment.model_role != route["roles"][index]
            or len(segment.frames) != route["frame_counts"][index]
            or segment.native_fps != route["native_fps"][index]
            or segment.retained_indices != route["retained"][index]
            or segment.discarded_indices != expected_discarded[index]
        ):
            raise ValueError(f"Temporal segment {index} differs from the registered topology.")
        _validate_scheduler_record(segment.scheduler, index, route, error_type=ValueError)
        if not _mapping_has_evidence(segment.conditioning):
            raise ValueError(f"Temporal segment {index} lacks conditioning evidence.")
        if not _mapping_has_evidence(segment.protected_state):
            raise ValueError(f"Temporal segment {index} lacks protected-state evidence.")
        if index == 0:
            if segment.anchor_sha256 is not None:
                raise ValueError("Temporal primary segment must not claim a continuation anchor.")
        elif (
            segment.anchor_sha256 is None
            or segment.reconstruction_index != 0
            or segment.first_motion_index != 1
        ) and bundle.temporal_protocol.get("strategy") != "reference_conditioned_multishot":
            raise ValueError("Temporal continuation segment lacks anchor/reconstruction evidence.")
    if bundle.output_frame_count != 240 or bundle.output_fps != 16:
        raise ValueError("Temporal bundle final contract must be exact 240/16/15.")
    if bundle.postprocess.get("method") != route["postprocess_method"]:
        raise ValueError("Temporal bundle postprocess method differs from its registered route.")


def _validate_record_segment_topology(
    segment: Mapping[str, Any], index: int, route: Mapping[str, Any]
) -> None:
    retained = list(route["retained"][index])
    discarded = sorted(set(range(route["frame_counts"][index])) - set(retained))
    if {
        "model_role": segment.get("model_role"),
        "decoded_frame_count": segment.get("decoded_frame_count"),
        "native_fps": segment.get("native_fps"),
        "retained_indices": segment.get("retained_indices"),
        "discarded_indices": segment.get("discarded_indices"),
    } != {
        "model_role": route["roles"][index],
        "decoded_frame_count": route["frame_counts"][index],
        "native_fps": route["native_fps"][index],
        "retained_indices": retained,
        "discarded_indices": discarded,
    }:
        raise RuntimeError(f"Temporal segment {index} topology differs from its frozen route.")
    if not segment.get("model_id") or not segment.get("model_revision"):
        raise RuntimeError(f"Temporal segment {index} lacks checkpoint identity.")
    if segment.get("decoded_dtype") != "uint8" or segment.get(
        "color_interpretation"
    ) != "RGB_uint8":
        raise RuntimeError("Temporal segment decoded pixel contract is invalid.")
    shape = segment.get("decoded_shape")
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or shape[-1] != 3
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
    ):
        raise RuntimeError("Temporal segment decoded shape is invalid.")
    _validate_scheduler_record(segment.get("scheduler"), index, route, error_type=RuntimeError)
    if not _mapping_has_evidence(segment.get("conditioning")):
        raise RuntimeError(f"Temporal segment {index} lacks conditioning evidence.")
    if not _mapping_has_evidence(segment.get("protected_state")):
        raise RuntimeError(f"Temporal segment {index} lacks protected-state evidence.")
    if index == 0:
        if segment.get("anchor_sha256") is not None:
            raise RuntimeError("Temporal primary segment unexpectedly claims an anchor.")
    elif route["roles"][index] == "i2v_continuation":
        if (
            not _SHA256_RE.fullmatch(str(segment.get("anchor_sha256", "")))
            or segment.get("reconstruction_index") != 0
            or segment.get("first_motion_index") != 1
        ):
            raise RuntimeError("Temporal I2V continuation anchor indices are invalid.")


def _validate_scheduler_record(
    scheduler: Any,
    index: int,
    route: Mapping[str, Any],
    *,
    error_type: type[Exception],
) -> None:
    if not isinstance(scheduler, Mapping) or not scheduler:
        raise error_type(f"Temporal segment {index} lacks scheduler-reset evidence.")
    expected_steps = route["local_steps"][index]
    observed_steps = scheduler.get(
        "num_inference_steps", scheduler.get("denoising_steps")
    )
    if observed_steps != expected_steps:
        raise error_type(f"Temporal segment {index} scheduler step count drifted.")
    classes = route.get("scheduler_classes")
    if classes is not None and scheduler.get("class") != classes[index]:
        raise error_type(f"Temporal segment {index} scheduler class drifted.")
    shifts = route.get("scheduler_shifts")
    if shifts is not None and float(scheduler.get("shift", float("nan"))) != shifts[index]:
        raise error_type(f"Temporal segment {index} scheduler shift drifted.")


def _lossless_window_indices(segment: Mapping[str, Any]) -> list[int]:
    frame_count = segment.get("decoded_frame_count")
    discarded = segment.get("discarded_indices")
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise RuntimeError("Temporal PNG window contract has an invalid frame count.")
    if not isinstance(discarded, list) or any(
        isinstance(index, bool) or not isinstance(index, int) for index in discarded
    ):
        raise RuntimeError("Temporal PNG window contract has malformed discarded indices.")
    special = {
        index
        for index in (
            segment.get("reconstruction_index"),
            segment.get("first_motion_index"),
        )
        if index is not None
    }
    return sorted(
        set(range(min(3, frame_count)))
        | set(range(max(0, frame_count - 3), frame_count))
        | set(discarded)
        | special
    )


def _mapping_has_evidence(value: Any) -> bool:
    if not isinstance(value, Mapping) or not value:
        return False
    for item in value.values():
        if isinstance(item, Mapping):
            if _mapping_has_evidence(item):
                return True
        elif isinstance(item, (list, tuple)):
            if item:
                return True
        elif item is not None and item != "":
            return True
    return False


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
