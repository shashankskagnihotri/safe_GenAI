from __future__ import annotations

import json
import hashlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

import torch

from hierasafe_flow.generation.video_utils import save_video_frames
from hierasafe_flow.generation.temporal_artifacts import (
    TemporalEvidenceBundle,
    final_media_binding,
    save_lossless_temporal_segments,
    temporal_evidence_to_dict,
    validate_staged_lossless_temporal_segments,
    validate_temporal_evidence_record,
)
from hierasafe_flow.utils.io import ensure_dir, write_json


def save_generation_output(
    media: Any,
    latents: torch.Tensor,
    trace: list[dict[str, Any]],
    output_dir: str | Path,
    sample_id: str,
    task: str,
    save_latents: bool = True,
    save_traces: bool = True,
    image_format: str = "png",
    video_format: str = "mp4",
    fps: int = 16,
    temporal_evidence: TemporalEvidenceBundle | None = None,
    report: dict[str, Any] | None = None,
) -> dict[str, str]:
    if temporal_evidence is not None:
        if report is None:
            raise RuntimeError(
                "Segmented temporal publication requires the sample report in the same "
                "transaction."
            )
        return _save_transactional_temporal_output(
            media=media,
            latents=latents,
            trace=trace,
            output_dir=Path(output_dir),
            sample_id=sample_id,
            task=task,
            save_latents=save_latents,
            save_traces=save_traces,
            image_format=image_format,
            video_format=video_format,
            fps=fps,
            temporal_evidence=temporal_evidence,
            report=report,
        )
    directory = ensure_dir(Path(output_dir) / sample_id)
    paths: dict[str, str] = {}

    if save_latents:
        latent_path = directory / "final_latents.pt"
        torch.save(latents.detach().cpu(), latent_path)
        paths["latents"] = str(latent_path)

    if save_traces:
        trace_path = directory / "steering_trace.json"
        write_json(trace_path, trace)
        paths["trace"] = str(trace_path)

    if media is None:
        return paths

    if isinstance(media, torch.Tensor):
        tensor_path = directory / "decoded_tensor.pt"
        torch.save(media.detach().cpu(), tensor_path)
        paths["media"] = str(tensor_path)
        return paths

    if task == "text_to_image":
        images = media if isinstance(media, list) else [media]
        for idx, image in enumerate(images):
            if not hasattr(image, "save"):
                continue
            path = directory / f"image_{idx:03d}.{image_format}"
            image.save(path)
            paths[f"image_{idx}"] = str(path)
        return paths

    if task == "text_to_video":
        videos = media if isinstance(media, list) else [media]
        for idx, frames in enumerate(videos):
            if not isinstance(frames, (list, tuple)):
                continue
            path = directory / f"video_{idx:03d}.{video_format}"
            save_video_frames(frames, path, fps=fps)
            paths[f"video_{idx}"] = str(path)
        return paths

    return paths


def _save_transactional_temporal_output(
    *,
    media: Any,
    latents: torch.Tensor,
    trace: list[dict[str, Any]],
    output_dir: Path,
    sample_id: str,
    task: str,
    save_latents: bool,
    save_traces: bool,
    image_format: str,
    video_format: str,
    fps: int,
    temporal_evidence: TemporalEvidenceBundle,
    report: dict[str, Any],
) -> dict[str, str]:
    del image_format
    if video_format != "mp4":
        raise RuntimeError(
            "Segmented temporal transactional publication requires video_format='mp4'."
        )
    if task != "text_to_video":
        raise RuntimeError("Temporal evidence may only be attached to text-to-video output.")
    if not save_traces:
        raise RuntimeError("Segmented temporal publication may not disable trace saving.")
    if media is None:
        raise RuntimeError("Segmented temporal publication requires decoded final media.")
    videos = media if isinstance(media, list) else [media]
    if len(videos) != 1 or not isinstance(videos[0], (list, tuple)):
        raise RuntimeError(
            "Segmented temporal transactional saving currently requires one decoded video."
        )
    final_frames = list(videos[0])
    if len(final_frames) != temporal_evidence.output_frame_count:
        raise RuntimeError(
            "Decoded final media does not match temporal evidence output frame count."
        )
    if fps != temporal_evidence.output_fps:
        raise RuntimeError("Requested saver fps does not match temporal evidence output clock.")

    output_dir.mkdir(parents=True, exist_ok=True)
    final_directory = output_dir / sample_id
    if final_directory.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing transactional sample directory {final_directory}."
        )
    staging_root = Path(tempfile.mkdtemp(prefix=f".{sample_id}.staging-", dir=output_dir))
    staging_directory = staging_root / sample_id
    staging_directory.mkdir()
    final_paths: dict[str, str] = {}
    published = False
    try:
        if save_latents:
            latent_path = staging_directory / "final_latents.pt"
            torch.save(latents.detach().cpu(), latent_path)
            final_paths["latents"] = str(final_directory / latent_path.name)

        trace_path = staging_directory / "steering_trace.json"
        trace_bytes = _json_bytes(trace)
        trace_path.write_bytes(trace_bytes)
        final_paths["trace"] = str(final_directory / trace_path.name)

        media_path = staging_directory / f"video_000.{video_format}"
        save_video_frames(final_frames, media_path, fps=fps)
        if not media_path.is_file() or media_path.stat().st_size <= 0:
            raise RuntimeError("Final temporal media encoding did not produce a file.")
        final_paths["video_0"] = str(final_directory / media_path.name)

        artifact_dir = staging_directory / "temporal_native_evidence"
        artifacts = save_lossless_temporal_segments(temporal_evidence, artifact_dir)
        validate_staged_lossless_temporal_segments(
            temporal_evidence,
            artifacts,
            artifact_dir,
        )
        artifacts = [
            _rewrite_artifact_paths(
                record,
                staging_directory=staging_directory,
                final_directory=final_directory,
            )
            for record in artifacts
        ]

        evidence_path = staging_directory / "temporal_evidence.json"
        final_paths["temporal_evidence"] = str(final_directory / evidence_path.name)
        evidence_sidecar_path = staging_directory / "temporal_evidence.json.sha256"
        final_paths["temporal_evidence_sha256"] = str(
            final_directory / evidence_sidecar_path.name
        )
        report_path = staging_directory / "report.json"
        final_paths["report"] = str(final_directory / report_path.name)
        bound_report = dict(report)
        bound_report["output_paths"] = dict(final_paths)
        bound_report["temporal_evidence"] = {
            "schema_version": 1,
            "temporal_protocol_sha256": temporal_evidence.temporal_protocol_sha256,
            "path": final_paths["temporal_evidence"],
        }
        report_bytes = _json_bytes(bound_report)
        report_path.write_bytes(report_bytes)

        binding = final_media_binding(
            final_frames=final_frames,
            media_path=media_path,
            trace_bytes=trace_bytes,
            report_bytes=report_bytes,
        )
        bound_bundle = temporal_evidence.bind(**binding)
        evidence_record = temporal_evidence_to_dict(
            bound_bundle,
            segment_artifacts=artifacts,
        )
        evidence_record["document_sha256_sidecar_path"] = final_paths[
            "temporal_evidence_sha256"
        ]
        validate_temporal_evidence_record(evidence_record, require_final_binding=True)
        evidence_bytes = _json_bytes(evidence_record)
        evidence_path.write_bytes(evidence_bytes)
        evidence_sha = hashlib.sha256(evidence_bytes).hexdigest()
        evidence_sidecar_path.write_text(
            f"{evidence_sha}  temporal_evidence.json\n", encoding="utf-8"
        )

        # Flush every file and directory entry before a single same-filesystem
        # rename makes the complete sample visible.
        for path in staging_directory.rglob("*"):
            if path.is_file():
                with path.open("rb") as handle:
                    os.fsync(handle.fileno())
        _fsync_directory_tree(staging_directory)
        _fsync_directory(staging_root)
        _fsync_directory(output_dir)
        os.replace(staging_directory, final_directory)
        published = True
        _fsync_directory(staging_root)
        _fsync_directory(output_dir)
        staging_root.rmdir()
        _fsync_directory(output_dir)
        return final_paths
    except Exception:
        if published and final_directory.exists():
            shutil.rmtree(final_directory, ignore_errors=True)
            try:
                _fsync_directory(output_dir)
            except OSError:
                pass
        shutil.rmtree(staging_root, ignore_errors=True)
        raise


def save_generation_report(
    report: dict[str, Any],
    output_dir: str | Path,
    sample_id: str,
) -> str:
    directory = ensure_dir(Path(output_dir) / sample_id)
    report_path = directory / "report.json"
    write_json(report_path, report)
    return str(report_path)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, allow_nan=False, sort_keys=True, indent=2)
        + "\n"
    ).encode("utf-8")


def _rewrite_artifact_paths(
    record: dict[str, Any],
    *,
    staging_directory: Path,
    final_directory: Path,
) -> dict[str, Any]:
    output = dict(record)
    for key in ("ffv1_mkv_path",):
        if output.get(key):
            relative = Path(output[key]).relative_to(staging_directory)
            output[key] = str(final_directory / relative)
    windows = []
    for item in output.get("lossless_png_windows", []):
        rewritten = dict(item)
        relative = Path(rewritten["path"]).relative_to(staging_directory)
        rewritten["path"] = str(final_directory / relative)
        windows.append(rewritten)
    output["lossless_png_windows"] = windows
    return output


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory_tree(root: Path) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        _fsync_directory(directory)
    _fsync_directory(root)
