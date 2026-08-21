#!/usr/bin/env python3
"""Prepare judgment-free visual review material for immutable runs.

Complete publication is the default and represents every expected-media row in
the supplied immutable manifests exactly once.  Missing or non-completed rows
fail before an output directory is published.  An explicitly requested partial
live-preview mode may inspect a still-running, noncanonical cohort, but records
its omissions and is forbidden for the canonical selected-seed final campaign.

The script validates and decodes source media, samples videos at seven fixed
temporal positions, renders deterministic labeled views, and emits JSON/CSV
review records whose semantic rubric fields are intentionally unset.  It does
not infer or mark semantic success.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import tempfile
import textwrap
from collections import Counter, defaultdict
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont, ImageOps


FRAME_PERCENTAGES = (0, 10, 25, 50, 75, 90, 100)
FONT_PATH = Path("/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf")
FONT_SIZE = 16
SMALL_FONT_SIZE = 13
IMAGE_ENTRY_SIZE = (420, 360)
IMAGE_VIEW_BOX = (380, 270)
VIDEO_FRAME_WIDTH = 220
VIDEO_FRAME_IMAGE_HEIGHT = 150
VIDEO_FRAME_LABEL_HEIGHT = 42
VIDEO_STRIP_PADDING = 12
SHEET_MAX_WIDTH = 2400
SHEET_MAX_HEIGHT = 3000
SHEET_HEADER_HEIGHT = 72
SHEET_GAP = 12
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".gif"}
COMPLETE_PUBLICATION_MODE = "complete"
PARTIAL_LIVE_PREVIEW_MODE = "partial_live_preview"
PUBLICATION_MODES = frozenset({COMPLETE_PUBLICATION_MODE, PARTIAL_LIVE_PREVIEW_MODE})

# These are security markers for the canonical paths owned by
# ``evaluation.finer_detailing_campaign``.  Keeping the guard local avoids
# importing the full generation/campaign stack into this rendering utility.
_CANONICAL_FINAL_MANIFEST_ROOT = Path(
    "debugging/manifests/finer_detailing_correction_selected_seed_v1"
)
_CANONICAL_FINAL_OUTPUT_ROOT = Path("outputs/finer_detailing_correction_selected_seed")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare exhaustive visual-review contact sheets from immutable manifests."
    )
    parser.add_argument("manifests", nargs="+", help="One or more immutable manifest JSON files.")
    parser.add_argument(
        "--review-root", required=True, help="New output directory; it must not exist."
    )
    parser.add_argument(
        "--partial-live-preview",
        action="store_true",
        help=(
            "Render only currently completed rows from a noncanonical live cohort and "
            "record every omission. This mode is forbidden for canonical selected-seed "
            "final manifests/outputs; omit it for complete publication."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    result = prepare_visual_review(
        [Path(path) for path in args.manifests],
        Path(args.review_root),
        publication_mode=(
            PARTIAL_LIVE_PREVIEW_MODE if args.partial_live_preview else COMPLETE_PUBLICATION_MODE
        ),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


def prepare_visual_review(
    manifest_paths: list[Path],
    review_root: Path,
    *,
    publication_mode: str = COMPLETE_PUBLICATION_MODE,
) -> dict[str, Any]:
    """Atomically create a complete package or an explicit noncanonical preview."""
    if publication_mode not in PUBLICATION_MODES:
        raise ValueError(
            f"Unknown visual-review publication mode {publication_mode!r}; "
            f"expected one of {sorted(PUBLICATION_MODES)}."
        )
    if not manifest_paths:
        raise ValueError("At least one immutable manifest is required.")
    resolved_manifests = [path.expanduser().resolve() for path in manifest_paths]
    if len(resolved_manifests) != len(set(resolved_manifests)):
        raise ValueError("The same manifest path was supplied more than once.")
    review_root = review_root.expanduser().resolve()
    if publication_mode == PARTIAL_LIVE_PREVIEW_MODE and _path_targets_canonical_final_tree(
        review_root
    ):
        raise ValueError(
            "Partial live-preview mode is forbidden from publishing inside the "
            "canonical selected-seed final manifest or output tree."
        )
    if review_root.exists():
        raise FileExistsError(
            f"Review root already exists; refusing to overwrite it: {review_root}"
        )
    review_root.parent.mkdir(parents=True, exist_ok=True)
    if not FONT_PATH.is_file():
        raise FileNotFoundError(
            f"Deterministic review font is unavailable: {FONT_PATH}. "
            "Install DejaVu Sans rather than silently changing layout."
        )
    _require_binary("ffprobe")
    _require_binary("ffmpeg")

    loaded = [_load_immutable_manifest(path) for path in resolved_manifests]
    expected_media = _expected_media_jobs(loaded)
    if not expected_media:
        raise ValueError(
            "Visual review refuses a zero-media campaign: supplied manifests contain "
            "zero expected-media rows."
        )
    if publication_mode == PARTIAL_LIVE_PREVIEW_MODE and (
        _contains_canonical_final_scope(loaded)
        or _expected_media_targets_canonical_final_scope(expected_media)
    ):
        raise ValueError(
            "Partial live-preview mode is forbidden for canonical selected-seed final "
            "manifests or output paths."
        )
    completed, incomplete = _collect_completed_media(
        loaded,
        expected_media,
        publication_mode=publication_mode,
    )
    if not completed:
        raise RuntimeError(
            "Visual review refuses to publish zero completed media records, including "
            "in partial live-preview mode."
        )
    temp_root = Path(
        tempfile.mkdtemp(prefix=f".{review_root.name}.tmp-", dir=str(review_root.parent))
    )
    try:
        records: list[dict[str, Any]] = []
        for item in completed:
            records.append(_prepare_item(item, temp_root))
        contact_sheets = _build_grouped_contact_sheets(records, temp_root)
        _assert_complete_one_to_one_coverage(
            expected_media,
            records,
            contact_sheets,
            publication_mode=publication_mode,
            incomplete=incomplete,
        )
        review_manifest = _build_review_manifest(
            loaded,
            expected_media,
            records,
            contact_sheets,
            publication_mode=publication_mode,
            incomplete=incomplete,
        )
        _write_json(temp_root / "review_manifest.json", review_manifest)
        _write_review_csv(temp_root / "review_manifest.csv", records)
        os.replace(temp_root, review_root)
    except BaseException:
        shutil.rmtree(temp_root, ignore_errors=True)
        raise

    return {
        "review_root": str(review_root),
        "review_manifest": str(review_root / "review_manifest.json"),
        "review_csv": str(review_root / "review_manifest.csv"),
        "publication_mode": publication_mode,
        "publication_complete": not incomplete,
        "expected_media_records": len(expected_media),
        "completed_media_records": len(completed),
        "omitted_media_records": len(incomplete),
        "contact_sheet_pages": len(contact_sheets),
    }


def _load_immutable_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or len(jobs) != int(payload.get("num_jobs", -1)):
        raise ValueError(f"Manifest job count is malformed: {path}")
    expected_digest = str(payload.get("manifest_sha256", ""))
    actual_digest = _manifest_digest(payload)
    if not expected_digest or expected_digest != actual_digest:
        raise ValueError(
            f"Manifest digest mismatch for {path}: expected={expected_digest!r}, actual={actual_digest}."
        )
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise FileNotFoundError(f"Immutable manifest digest sidecar is missing: {sidecar}")
    fields = sidecar.read_text(encoding="utf-8").split()
    if not fields or fields[0] != actual_digest:
        raise ValueError(f"Manifest digest sidecar does not match {path}: {sidecar}")
    condition_ids = [str(job.get("condition_id")) for job in jobs]
    if "None" in condition_ids or len(condition_ids) != len(set(condition_ids)):
        raise ValueError(f"Manifest contains missing or duplicate condition IDs: {path}")
    output_dirs = [str(job.get("output_dir")) for job in jobs]
    if "None" in output_dirs or len(output_dirs) != len(set(output_dirs)):
        raise ValueError(f"Manifest contains missing or duplicate output directories: {path}")
    return {
        "path": path,
        "file_sha256": _sha256_file(path),
        "manifest_sha256": actual_digest,
        "payload": payload,
    }


def _expected_media_jobs(manifests: list[dict[str, Any]]) -> list[dict[str, Any]]:
    expected: list[dict[str, Any]] = []
    condition_owners: dict[str, tuple[int, int]] = {}
    output_owners: dict[Path, tuple[int, int]] = {}
    for manifest_index, manifest in enumerate(manifests):
        for job_index, job in enumerate(manifest["payload"]["jobs"]):
            marker = job.get("expected_media")
            if marker is not True and marker is not False:
                raise ValueError(
                    "Every immutable manifest row must declare boolean expected_media; "
                    f"manifest={manifest['path']}, job_index={job_index}."
                )
            if marker is False:
                continue
            condition_id = str(job["condition_id"])
            output_dir = Path(str(job["output_dir"])).expanduser().resolve()
            identity = (manifest_index, job_index)
            if condition_id in condition_owners:
                raise ValueError(
                    "Expected-media condition_id is duplicated across supplied manifests: "
                    f"{condition_id!r}, owners={condition_owners[condition_id]} and {identity}."
                )
            if output_dir in output_owners:
                raise ValueError(
                    "Expected-media output directory is duplicated across supplied manifests: "
                    f"{output_dir}, owners={output_owners[output_dir]} and {identity}."
                )
            condition_owners[condition_id] = identity
            output_owners[output_dir] = identity
            expected.append(
                {
                    "manifest_input_index": manifest_index,
                    "manifest_job_index": job_index,
                    "manifest_path": manifest["path"],
                    "manifest_file_sha256": manifest["file_sha256"],
                    "manifest_sha256": manifest["manifest_sha256"],
                    "job": job,
                    "output_dir": output_dir,
                }
            )
    return expected


def _collect_completed_media(
    manifests: list[dict[str, Any]],
    expected_media: list[dict[str, Any]],
    *,
    publication_mode: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    completed: list[dict[str, Any]] = []
    incomplete: list[dict[str, Any]] = []
    seen_source_paths: dict[Path, str] = {}
    for manifest in manifests:
        for job in manifest["payload"]["jobs"]:
            if job.get("expected_media") is False:
                result_path = (
                    Path(str(job["output_dir"])).expanduser().resolve()
                    / "benchmark_job_result.json"
                )
                if not result_path.is_file():
                    continue
                result = json.loads(result_path.read_text(encoding="utf-8"))
                if not isinstance(result, dict):
                    raise ValueError(f"Benchmark result must contain an object: {result_path}")
                if str(result.get("status")) == "completed":
                    raise ValueError(
                        "Completed result unexpectedly contains a media contract for "
                        f"not-supported job: {result_path}"
                    )

    for item in expected_media:
        manifest_index = int(item["manifest_input_index"])
        job_index = int(item["manifest_job_index"])
        job = item["job"]
        output_dir: Path = item["output_dir"]
        result_path = output_dir / "benchmark_job_result.json"
        if not result_path.is_file():
            incomplete.append(
                _incomplete_media_row(item, result_path, reason="missing_result", status=None)
            )
            continue
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if not isinstance(result, dict):
            raise ValueError(f"Benchmark result must contain an object: {result_path}")
        status = str(result.get("status"))
        if status != "completed":
            incomplete.append(
                _incomplete_media_row(
                    item,
                    result_path,
                    reason="result_status_not_completed",
                    status=status,
                )
            )
            continue
        result_job = result.get("job")
        if not isinstance(result_job, dict):
            raise ValueError(f"Completed result has no job object: {result_path}")
        if str(result_job.get("condition_id")) != str(job.get("condition_id")):
            raise ValueError(f"Result/job condition mismatch: {result_path}")
        for field in ("prompt_id", "model_name", "variation", "seed", "output_dir"):
            if result_job.get(field) != job.get(field):
                raise ValueError(f"Result/job {field} mismatch in completed result: {result_path}")
        paths = result.get("validated_media_paths")
        validation = result.get("media_validation")
        if not isinstance(paths, list) or len(paths) != 1 or not isinstance(validation, dict):
            raise ValueError(
                f"Completed result must name exactly one validated media path: {result_path}"
            )
        source_path = Path(str(paths[0])).expanduser()
        if not source_path.is_absolute():
            source_path = (result_path.parent / source_path).resolve()
        else:
            source_path = source_path.resolve()
        validation_path = Path(str(validation.get("path", ""))).expanduser()
        if not validation_path.is_absolute():
            validation_path = (result_path.parent / validation_path).resolve()
        else:
            validation_path = validation_path.resolve()
        if validation_path != source_path:
            raise ValueError(f"Validated path disagreement in {result_path}")
        expected_source_path = _expected_media_path(output_dir, job)
        if source_path != expected_source_path:
            raise ValueError(
                f"Completed result media path is not canonical for its task: "
                f"expected={expected_source_path}, actual={source_path}."
            )
        if source_path in seen_source_paths:
            raise ValueError(
                f"Completed media source is duplicated by conditions "
                f"{seen_source_paths[source_path]!r} and {job['condition_id']!r}: {source_path}"
            )
        condition_id = str(job["condition_id"])
        discovered_media = sorted(
            path.resolve()
            for path in output_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES
        )
        if discovered_media != [source_path]:
            raise ValueError(
                f"Completed attempt must contain exactly its one validated media file; "
                f"expected={[source_path]}, discovered={discovered_media}"
            )
        seen_source_paths[source_path] = condition_id
        completed.append(
            {
                "manifest_input_index": manifest_index,
                "manifest_job_index": job_index,
                "manifest_path": item["manifest_path"],
                "manifest_file_sha256": item["manifest_file_sha256"],
                "manifest_sha256": item["manifest_sha256"],
                "result_path": result_path.resolve(),
                "result_file_sha256": _sha256_file(result_path),
                "job": job,
                "source_path": source_path,
                "source_validation": validation,
            }
        )
    if incomplete and publication_mode == COMPLETE_PUBLICATION_MODE:
        reasons = Counter(str(item["reason"]) for item in incomplete)
        examples = [
            {
                "condition_id": item["condition_id"],
                "reason": item["reason"],
                "result_status": item["result_status"],
                "result_path": item["result_path"],
            }
            for item in incomplete[:5]
        ]
        raise RuntimeError(
            "Complete visual review requires a completed structurally valid result for every "
            f"expected-media row: expected={len(expected_media)}, completed={len(completed)}, "
            f"incomplete={len(incomplete)}, reasons={dict(reasons)}, examples={examples}."
        )
    completed.sort(
        key=lambda item: (
            str(item["job"]["prompt_id"]),
            str(item["job"]["model_name"]),
            str(item["job"]["variation"]),
            str(item["job"]["condition_id"]),
            str(item["source_path"]),
        )
    )
    incomplete.sort(
        key=lambda item: (
            str(item["condition_id"]),
            int(item["manifest_input_index"]),
            int(item["manifest_job_index"]),
        )
    )
    return completed, incomplete


def _incomplete_media_row(
    item: dict[str, Any],
    result_path: Path,
    *,
    reason: str,
    status: str | None,
) -> dict[str, Any]:
    job = item["job"]
    return {
        "manifest_input_index": int(item["manifest_input_index"]),
        "manifest_job_index": int(item["manifest_job_index"]),
        "manifest_path": str(item["manifest_path"]),
        "manifest_sha256": str(item["manifest_sha256"]),
        "condition_id": str(job["condition_id"]),
        "output_dir": str(item["output_dir"]),
        "result_path": str(result_path.resolve()),
        "reason": reason,
        "result_status": status,
    }


def _expected_media_path(output_dir: Path, job: dict[str, Any]) -> Path:
    task = str(job.get("generation", {}).get("task", ""))
    if task == "text_to_image":
        return (output_dir / "sample_0000" / "image_000.png").resolve()
    if task == "text_to_video":
        return (output_dir / "sample_0000" / "video_000.mp4").resolve()
    raise ValueError(f"Unsupported expected-media generation task: {task!r}")


def _contains_canonical_final_scope(manifests: list[dict[str, Any]]) -> bool:
    manifest_marker = _CANONICAL_FINAL_MANIFEST_ROOT.parts
    output_marker = _CANONICAL_FINAL_OUTPUT_ROOT.parts
    for manifest in manifests:
        if _path_contains_parts(Path(manifest["path"]), manifest_marker):
            return True
        payload = manifest["payload"]
        output_root = payload.get("output_root")
        if output_root and _path_contains_parts(Path(str(output_root)), output_marker):
            return True
        if any(
            _path_contains_parts(Path(str(job.get("output_dir", ""))), output_marker)
            for job in payload["jobs"]
        ):
            return True
    return False


def _expected_media_targets_canonical_final_scope(
    expected_media: list[dict[str, Any]],
) -> bool:
    """Reject canonical descendants hidden below an otherwise benign output dir."""

    marker = _CANONICAL_FINAL_OUTPUT_ROOT.parts
    for item in expected_media:
        output_dir = Path(item["output_dir"])
        job = item["job"]
        candidate_paths = (
            output_dir,
            output_dir / "benchmark_job_result.json",
            _expected_media_path(output_dir, job),
        )
        if any(_path_contains_parts(path, marker) for path in candidate_paths):
            return True
    return False


def _path_targets_canonical_final_tree(path: Path) -> bool:
    return any(
        _path_contains_parts(path, marker.parts)
        for marker in (_CANONICAL_FINAL_MANIFEST_ROOT, _CANONICAL_FINAL_OUTPUT_ROOT)
    )


def _path_contains_parts(path: Path, marker: tuple[str, ...]) -> bool:
    width = len(marker)
    expanded = path.expanduser()
    # Check both the literal spelling and the same canonical path semantics used
    # later by ``_expected_media_jobs``.  Looking only at lexical components lets
    # a symlink (or a ``..`` alias) hide a canonical final output beneath an
    # innocuous-looking preview path.
    for candidate in (expanded, expanded.resolve(strict=False)):
        parts = candidate.parts
        if any(parts[index : index + width] == marker for index in range(len(parts) - width + 1)):
            return True
    return False


def _prepare_item(item: dict[str, Any], temp_root: Path) -> dict[str, Any]:
    job = item["job"]
    source_path: Path = item["source_path"]
    if not source_path.is_file() or source_path.stat().st_size <= 0:
        raise RuntimeError(f"Completed source media is missing or empty: {source_path}")
    actual_sha = _sha256_file(source_path)
    expected_sha = str(item["source_validation"].get("sha256", ""))
    if not expected_sha or actual_sha != expected_sha:
        raise ValueError(
            f"Source media SHA mismatch for {source_path}: expected={expected_sha}, actual={actual_sha}."
        )
    record_id = hashlib.sha256(
        (
            f"{item['manifest_sha256']}:{item['manifest_job_index']}:"
            f"{job['condition_id']}:{source_path}:{actual_sha}"
        ).encode("utf-8")
    ).hexdigest()[:20]
    item_dir = temp_root / "items" / record_id
    item_dir.mkdir(parents=True, exist_ok=False)
    task = str(job["generation"]["task"])
    label = _item_label(job)
    frame_records: list[dict[str, Any]] = []
    if task == "text_to_image":
        if source_path.suffix.lower() != ".png":
            raise ValueError(f"Image job did not produce PNG media: {source_path}")
        media_metadata = _validate_png(source_path, job)
        entry_path = item_dir / "image_entry.png"
        _render_image_entry(source_path, entry_path, label)
        media_kind = "image"
    elif task == "text_to_video":
        if source_path.suffix.lower() != ".mp4":
            raise ValueError(f"Video job did not produce MP4 media: {source_path}")
        media_metadata = _probe_video(source_path, job)
        frame_records = _extract_labeled_video_frames(
            source_path,
            item_dir,
            media_metadata,
        )
        entry_path = item_dir / "video_strip.png"
        _render_video_strip(frame_records, entry_path, label, temp_root)
        media_kind = "video"
    else:
        raise ValueError(f"Unsupported generation task for review: {task!r}")

    concept_pairs = _concept_pairs(job)
    variant_spec = job.get("variant_spec", {})
    variant_kind = str(variant_spec.get("kind", ""))
    active_pair_ids = tuple(str(value) for value in variant_spec.get("active_pair_ids", ()))
    legal_pair_ids = tuple(pair["id"] for pair in concept_pairs)
    native_negative_source_pair_ids: tuple[str, ...] = ()
    if variant_kind == "native_negative_prompt":
        if active_pair_ids:
            raise ValueError(
                "Native-negative jobs must not claim positive active_pair_ids; "
                f"got {list(active_pair_ids)} for {job['condition_id']}."
            )
        # Native negative prompting suppresses the five unwanted/source
        # concepts. It does not encode any of their positive steering targets.
        native_negative_source_pair_ids = legal_pair_ids
    illegal_active = sorted(set(active_pair_ids) - set(legal_pair_ids))
    if illegal_active:
        raise ValueError(
            f"Job {job['condition_id']} activates unknown concept pairs: {illegal_active}"
        )
    inactive_pair_ids = (
        ()
        if native_negative_source_pair_ids
        else tuple(pair_id for pair_id in legal_pair_ids if pair_id not in active_pair_ids)
    )
    rubric = _empty_rubric(
        job,
        concept_pairs,
        active_pair_ids=active_pair_ids,
        inactive_pair_ids=inactive_pair_ids,
        native_negative_source_pair_ids=native_negative_source_pair_ids,
    )
    return {
        "record_id": record_id,
        "review_status": "not_reviewed",
        "manifest_input_index": int(item["manifest_input_index"]),
        "manifest_job_index": int(item["manifest_job_index"]),
        "source_manifest_path": str(item["manifest_path"]),
        "source_manifest_file_sha256": item["manifest_file_sha256"],
        "source_manifest_sha256": item["manifest_sha256"],
        "source_result_path": str(item["result_path"]),
        "source_result_sha256": item["result_file_sha256"],
        "source_media_path": str(source_path),
        "source_media_sha256": actual_sha,
        "source_media_size_bytes": source_path.stat().st_size,
        "source_media_validation": media_metadata,
        "media_kind": media_kind,
        "prompt_id": str(job["prompt_id"]),
        "prompt": str(job.get("prompt", "")),
        "model_name": str(job["model_name"]),
        "seed": int(job["seed"]),
        "condition_id": str(job["condition_id"]),
        "condition_family": str(job["variation"]),
        "variant": str(job.get("variant", "")),
        "variant_kind": variant_kind,
        "active_pair_ids": list(active_pair_ids),
        "inactive_pair_ids": list(inactive_pair_ids),
        "native_negative_source_pair_ids": list(native_negative_source_pair_ids),
        "review_entry_path": _relative(entry_path, temp_root),
        "review_entry_sha256": _sha256_file(entry_path),
        "sampled_frames": frame_records,
        "contact_sheet_path": None,
        "contact_sheet_page": None,
        "contact_sheet_position": None,
        "rubric": rubric,
    }


def _validate_png(path: Path, job: dict[str, Any]) -> dict[str, Any]:
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        width, height = image.size
        mode = image.mode
        image_format = image.format
    if image_format != "PNG":
        raise ValueError(f"Expected PNG encoding at {path}; found {image_format!r}.")
    expected = (int(job["generation"]["width"]), int(job["generation"]["height"]))
    if (width, height) != expected:
        raise ValueError(f"PNG dimensions at {path} are {(width, height)}, expected {expected}.")
    return {
        "decode_verified": True,
        "format": image_format,
        "width": width,
        "height": height,
        "mode": mode,
    }


def _probe_video(path: Path, job: dict[str, Any]) -> dict[str, Any]:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-count_frames",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,pix_fmt,avg_frame_rate,nb_frames,nb_read_frames,duration",
        "-show_entries",
        "format=format_name,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
    payload = json.loads(result.stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise ValueError(f"Expected exactly one video stream in {path}; found {len(streams)}.")
    stream = streams[0]
    frame_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frame_value in {None, "N/A"}:
        raise ValueError(f"ffprobe did not provide a frame count for {path}.")
    frame_count = int(frame_value)
    if frame_count < len(FRAME_PERCENTAGES):
        raise ValueError(f"Video needs at least seven frames for fixed review sampling: {path}")
    width, height = int(stream["width"]), int(stream["height"])
    fps = float(Fraction(str(stream["avg_frame_rate"])))
    expected_size = (int(job["generation"]["width"]), int(job["generation"]["height"]))
    expected_frames = int(job["generation"]["num_frames"])
    expected_fps = float(job["generation"]["fps"])
    if (width, height) != expected_size:
        raise ValueError(
            f"Video dimensions at {path} are {(width, height)}, expected {expected_size}."
        )
    if frame_count != expected_frames:
        raise ValueError(
            f"Video frame count at {path} is {frame_count}, expected {expected_frames}."
        )
    if abs(fps - expected_fps) > 1.0e-3:
        raise ValueError(f"Video FPS at {path} is {fps}, expected {expected_fps}.")
    duration = float((payload.get("format") or {}).get("duration") or stream.get("duration"))
    return {
        "decode_verified": True,
        "codec_name": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
    }


def temporal_frame_indices(frame_count: int) -> tuple[int, ...]:
    if frame_count < len(FRAME_PERCENTAGES):
        raise ValueError("At least seven frames are required.")
    last = frame_count - 1
    indices = tuple(
        math.floor((percentage / 100.0) * last + 0.5) for percentage in FRAME_PERCENTAGES
    )
    if len(indices) != 7 or len(set(indices)) != 7 or indices[0] != 0 or indices[-1] != last:
        raise ValueError(f"Fixed temporal samples are not seven unique frames: {indices}")
    return indices


def _extract_labeled_video_frames(
    source_path: Path,
    item_dir: Path,
    metadata: dict[str, Any],
) -> list[dict[str, Any]]:
    indices = temporal_frame_indices(int(metadata["frame_count"]))
    raw_dir = item_dir / ".raw_frames"
    raw_dir.mkdir()
    expression = "+".join(f"eq(n\\,{index})" for index in indices)
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-i",
            str(source_path),
            "-map",
            "0:v:0",
            "-vf",
            f"select={expression}",
            # FFmpeg 4.4 (the frozen experiment environment) predates the
            # output-scoped ``-fps_mode`` spelling.  ``-vsync vfr`` is the
            # equivalent supported form and preserves exactly the selected
            # source frames without duplicating them.
            "-vsync",
            "vfr",
            "-start_number",
            "0",
            str(raw_dir / "frame_%02d.png"),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    raw_frames = sorted(raw_dir.glob("frame_*.png"))
    if len(raw_frames) != 7:
        raise RuntimeError(
            f"ffmpeg extracted {len(raw_frames)} frames rather than exactly seven from {source_path}."
        )
    font = _font(SMALL_FONT_SIZE)
    records: list[dict[str, Any]] = []
    for ordinal, (percentage, frame_index, raw_path) in enumerate(
        zip(FRAME_PERCENTAGES, indices, raw_frames, strict=True)
    ):
        with Image.open(raw_path) as raw:
            raw.load()
            tile = _fit_on_canvas(raw.convert("RGB"), (VIDEO_FRAME_WIDTH, VIDEO_FRAME_IMAGE_HEIGHT))
        canvas = Image.new(
            "RGB",
            (VIDEO_FRAME_WIDTH, VIDEO_FRAME_IMAGE_HEIGHT + VIDEO_FRAME_LABEL_HEIGHT),
            "#151515",
        )
        canvas.paste(tile, (0, 0))
        seconds = frame_index / float(metadata["fps"])
        label = f"{percentage}%  frame {frame_index}/{int(metadata['frame_count']) - 1}\n{seconds:.3f} s"
        ImageDraw.Draw(canvas).multiline_text(
            (6, VIDEO_FRAME_IMAGE_HEIGHT + 4), label, font=font, fill="white", spacing=1
        )
        labeled_path = (
            item_dir / f"frame_{ordinal:02d}_{percentage:03d}pct_index_{frame_index:06d}.png"
        )
        _save_png(canvas, labeled_path)
        records.append(
            {
                "ordinal": ordinal,
                "percentage": percentage,
                "frame_index": frame_index,
                "timestamp_seconds": seconds,
                "path": _relative(labeled_path, item_dir.parents[1]),
                "sha256": _sha256_file(labeled_path),
            }
        )
    shutil.rmtree(raw_dir)
    return records


def _render_image_entry(source_path: Path, output_path: Path, label: str) -> None:
    with Image.open(source_path) as source:
        source.load()
        fitted = _fit_on_canvas(source.convert("RGB"), IMAGE_VIEW_BOX)
    canvas = Image.new("RGB", IMAGE_ENTRY_SIZE, "#101010")
    x = (IMAGE_ENTRY_SIZE[0] - fitted.width) // 2
    canvas.paste(fitted, (x, 8))
    draw = ImageDraw.Draw(canvas)
    draw.multiline_text(
        (12, IMAGE_VIEW_BOX[1] + 20),
        _wrap_label(label, 49),
        font=_font(SMALL_FONT_SIZE),
        fill="white",
        spacing=2,
    )
    _save_png(canvas, output_path)


def _render_video_strip(
    frame_records: list[dict[str, Any]],
    output_path: Path,
    label: str,
    temp_root: Path,
) -> None:
    if len(frame_records) != 7:
        raise ValueError("A per-video strip requires exactly seven labeled frames.")
    frames: list[Image.Image] = []
    for record in frame_records:
        path = temp_root / str(record["path"])
        with Image.open(path) as image:
            image.load()
            frames.append(image.convert("RGB"))
    header_height = 54
    width = VIDEO_STRIP_PADDING + sum(frame.width + VIDEO_STRIP_PADDING for frame in frames)
    height = (
        header_height
        + VIDEO_STRIP_PADDING
        + max(frame.height for frame in frames)
        + VIDEO_STRIP_PADDING
    )
    canvas = Image.new("RGB", (width, height), "#0d0d0d")
    ImageDraw.Draw(canvas).multiline_text(
        (VIDEO_STRIP_PADDING, 7),
        _wrap_label(label, 150),
        font=_font(FONT_SIZE),
        fill="white",
        spacing=2,
    )
    x = VIDEO_STRIP_PADDING
    y = header_height + VIDEO_STRIP_PADDING
    for frame in frames:
        canvas.paste(frame, (x, y))
        x += frame.width + VIDEO_STRIP_PADDING
    _save_png(canvas, output_path)


def _build_grouped_contact_sheets(
    records: list[dict[str, Any]],
    temp_root: Path,
) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[(record["prompt_id"], record["model_name"], record["condition_family"])].append(
            record
        )
    pages: list[dict[str, Any]] = []
    for group_key in sorted(groups):
        group_records = sorted(
            groups[group_key], key=lambda record: (record["condition_id"], record["record_id"])
        )
        entry_images: list[tuple[dict[str, Any], Image.Image]] = []
        for record in group_records:
            path = temp_root / record["review_entry_path"]
            with Image.open(path) as image:
                image.load()
                entry_images.append((record, image.convert("RGB")))
        max_width = max(image.width for _, image in entry_images)
        max_height = max(image.height for _, image in entry_images)
        columns = max(1, (SHEET_MAX_WIDTH - SHEET_GAP) // (max_width + SHEET_GAP))
        rows_per_page = max(
            1,
            (SHEET_MAX_HEIGHT - SHEET_HEADER_HEIGHT - SHEET_GAP) // (max_height + SHEET_GAP),
        )
        per_page = columns * rows_per_page
        for page_index, start in enumerate(range(0, len(entry_images), per_page), start=1):
            page_entries = entry_images[start : start + per_page]
            rows = math.ceil(len(page_entries) / columns)
            width = min(
                SHEET_MAX_WIDTH,
                SHEET_GAP + columns * (max_width + SHEET_GAP),
            )
            height = min(
                SHEET_MAX_HEIGHT,
                SHEET_HEADER_HEIGHT + SHEET_GAP + rows * (max_height + SHEET_GAP),
            )
            canvas = Image.new("RGB", (width, height), "#090909")
            title = " | ".join(group_key) + f" | page {page_index}"
            ImageDraw.Draw(canvas).text((SHEET_GAP, 18), title, font=_font(FONT_SIZE), fill="white")
            page_record_ids: list[str] = []
            for offset, (record, entry_image) in enumerate(page_entries):
                row, column = divmod(offset, columns)
                x = SHEET_GAP + column * (max_width + SHEET_GAP)
                y = SHEET_HEADER_HEIGHT + SHEET_GAP + row * (max_height + SHEET_GAP)
                canvas.paste(entry_image, (x, y))
                page_record_ids.append(record["record_id"])
            relative = (
                Path("contact_sheets")
                / _slug(group_key[0])
                / _slug(group_key[1])
                / _slug(group_key[2])
                / f"page_{page_index:03d}.png"
            )
            output_path = temp_root / relative
            _save_png(canvas, output_path)
            page = {
                "prompt_id": group_key[0],
                "model_name": group_key[1],
                "condition_family": group_key[2],
                "page": page_index,
                "path": str(relative),
                "sha256": _sha256_file(output_path),
                "width": width,
                "height": height,
                "record_ids": page_record_ids,
            }
            pages.append(page)
            for position, (record, _) in enumerate(page_entries):
                record["contact_sheet_path"] = str(relative)
                record["contact_sheet_page"] = page_index
                record["contact_sheet_position"] = position
    return pages


def _empty_rubric(
    job: dict[str, Any],
    concept_pairs: list[dict[str, str]],
    *,
    active_pair_ids: tuple[str, ...],
    inactive_pair_ids: tuple[str, ...],
    native_negative_source_pair_ids: tuple[str, ...],
) -> dict[str, Any]:
    by_id = {pair["id"]: pair for pair in concept_pairs}
    prompt_id = str(job["prompt_id"])
    task = str(job["generation"]["task"])
    rubric: dict[str, Any] = {
        "overall_semantic_success": None,
        "source_fidelity": {"rating": None, "notes": ""},
        "active_targets": {
            pair_id: {
                "source_concept": by_id[pair_id]["unsafe_concept"],
                "target_concept": by_id[pair_id]["safe_sibling_concept"],
                "achieved": None,
                "selective_without_collateral_change": None,
                "notes": "",
            }
            for pair_id in active_pair_ids
        },
        "native_negative_source_suppression": {
            pair_id: {
                "source_concept": by_id[pair_id]["unsafe_concept"],
                "suppressed": None,
                "positive_target_achievement_not_applicable": True,
                "notes": "",
            }
            for pair_id in native_negative_source_pair_ids
        },
        "inactive_source_preservation": {
            pair_id: {
                "source_concept": by_id[pair_id]["unsafe_concept"],
                "preserved": None,
                "notes": "",
            }
            for pair_id in inactive_pair_ids
        },
        "identity": {
            "applicable": prompt_id in {"01_sad_young_girl", "02_angry_old_man"},
            "person_identity_preserved": None,
            "gender_preserved": None,
            "age_preserved": None,
            "notes": "",
        },
        "mall": {
            "applicable": prompt_id == "03_empty_outdoor_mall",
            "no_people_preserved": None,
            "architecture_and_layout_preserved": None,
            "camera_and_object_motion_correct": None,
            "escalator_direction_or_static_stair_behavior_correct": None,
            "signage_content_and_legibility_correct": None,
            "notes": "",
        },
        "artifacts": {"present": None, "severity": None, "notes": ""},
        "temporal_consistency": {
            "applicable": task == "text_to_video",
            "rating": None,
            "notes": "",
        },
        "reviewer_notes": "",
    }
    return rubric


def _concept_pairs(job: dict[str, Any]) -> list[dict[str, str]]:
    raw_pairs = job.get("concept_tree_snapshot", {}).get("pairs") or []
    pairs: list[dict[str, str]] = []
    for raw in raw_pairs:
        if not isinstance(raw, dict):
            raise ValueError(f"Malformed concept pair in job {job.get('condition_id')}: {raw!r}")
        required = {"id", "unsafe_concept", "safe_sibling_concept"}
        if not required.issubset(raw):
            raise ValueError(f"Concept pair is missing review fields: {raw!r}")
        pairs.append({key: str(raw[key]) for key in required})
    if len(pairs) != 5 or len({pair["id"] for pair in pairs}) != 5:
        raise ValueError("Review expects exactly five unique prompt-specific concept pairs.")
    return pairs


def _assert_complete_one_to_one_coverage(
    expected_media: list[dict[str, Any]],
    records: list[dict[str, Any]],
    contact_sheets: list[dict[str, Any]],
    *,
    publication_mode: str,
    incomplete: list[dict[str, Any]],
) -> None:
    expected = [
        (
            str(item["manifest_sha256"]),
            int(item["manifest_job_index"]),
            str(item["job"]["condition_id"]),
        )
        for item in expected_media
    ]
    actual = [
        (
            str(record["source_manifest_sha256"]),
            int(record["manifest_job_index"]),
            str(record["condition_id"]),
        )
        for record in records
    ]
    expected_counter = Counter(expected)
    actual_counter = Counter(actual)
    omitted = list((expected_counter - actual_counter).elements())
    unexpected = list((actual_counter - expected_counter).elements())
    if unexpected:
        raise RuntimeError(f"Review contains records outside expected media: {unexpected}")
    if publication_mode == COMPLETE_PUBLICATION_MODE and omitted:
        raise RuntimeError(f"Complete review coverage omitted expected media: {omitted}")
    declared_omitted = [
        (
            str(item["manifest_sha256"]),
            int(item["manifest_job_index"]),
            str(item["condition_id"]),
        )
        for item in incomplete
    ]
    if Counter(declared_omitted) != Counter(omitted):
        raise RuntimeError(
            "Review omission ledger differs from the expected-media minus review-record set."
        )
    if any(count != 1 for count in Counter(actual).values()):
        raise RuntimeError("An expected-media manifest row was duplicated in review records.")
    source_paths = [record["source_media_path"] for record in records]
    if len(source_paths) != len(set(source_paths)):
        raise RuntimeError("A completed source media path was duplicated in review records.")
    record_ids = [record["record_id"] for record in records]
    if len(record_ids) != len(set(record_ids)):
        raise RuntimeError("Review record IDs are duplicated.")
    entry_paths = [record["review_entry_path"] for record in records]
    if len(entry_paths) != len(set(entry_paths)):
        raise RuntimeError("Review entry paths are duplicated.")
    sheet_members = [record_id for page in contact_sheets for record_id in page["record_ids"]]
    if Counter(sheet_members) != Counter(record_ids):
        raise RuntimeError(
            "Contact sheets omitted or duplicated one or more completed review records."
        )
    for record in records:
        if record["media_kind"] == "video" and len(record["sampled_frames"]) != 7:
            raise RuntimeError(
                f"Video record does not contain exactly seven frames: {record['record_id']}"
            )
        if record["media_kind"] == "image" and record["sampled_frames"]:
            raise RuntimeError(
                f"Image record unexpectedly contains sampled video frames: {record['record_id']}"
            )
        if record["contact_sheet_path"] is None:
            raise RuntimeError(
                f"Review record was not assigned to a contact sheet: {record['record_id']}"
            )


def _build_review_manifest(
    manifests: list[dict[str, Any]],
    expected_media: list[dict[str, Any]],
    records: list[dict[str, Any]],
    contact_sheets: list[dict[str, Any]],
    *,
    publication_mode: str,
    incomplete: list[dict[str, Any]],
) -> dict[str, Any]:
    images = sum(record["media_kind"] == "image" for record in records)
    videos = sum(record["media_kind"] == "video" for record in records)
    publication_complete = not incomplete and len(records) == len(expected_media)
    return {
        "schema_version": 2,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "publication": {
            "mode": publication_mode,
            "complete": publication_complete,
            "canonical_final_partial_preview_forbidden": True,
        },
        "semantic_judgment_policy": {
            "automatic_semantic_success_forbidden": True,
            "all_semantic_fields_initialized_unreviewed": True,
            "review_status_initial_value": "not_reviewed",
        },
        "deterministic_rendering": {
            "font_path": str(FONT_PATH),
            "font_sha256": _sha256_file(FONT_PATH),
            "font_size": FONT_SIZE,
            "small_font_size": SMALL_FONT_SIZE,
            "sheet_max_width": SHEET_MAX_WIDTH,
            "sheet_max_height": SHEET_MAX_HEIGHT,
        },
        "video_sampling": {
            "percentages": list(FRAME_PERCENTAGES),
            "index_rule": "floor((percentage/100)*(frame_count-1)+0.5)",
            "frames_per_video": 7,
        },
        "source_manifests": [
            {
                "input_index": index,
                "path": str(manifest["path"]),
                "file_sha256": manifest["file_sha256"],
                "manifest_sha256": manifest["manifest_sha256"],
                "num_jobs": len(manifest["payload"]["jobs"]),
            }
            for index, manifest in enumerate(manifests)
        ],
        "counts": {
            "completed_media_records": len(records),
            "images": images,
            "videos": videos,
            "sampled_video_frames": videos * 7,
            "contact_sheet_pages": len(contact_sheets),
        },
        "coverage": {
            "expected_completed_media": len(expected_media),
            "expected_media_rows": len(expected_media),
            "completed_media_results": len(records),
            "review_records": len(records),
            "omitted": incomplete,
            "omitted_count": len(incomplete),
            "duplicated": [],
            "one_to_one_verified": publication_complete,
            "record_to_media_one_to_one_verified": True,
        },
        "contact_sheets": contact_sheets,
        "records": records,
    }


def _write_review_csv(path: Path, records: list[dict[str, Any]]) -> None:
    fields = [
        "record_id",
        "review_status",
        "manifest_input_index",
        "manifest_job_index",
        "source_manifest_path",
        "source_manifest_file_sha256",
        "source_manifest_sha256",
        "source_result_path",
        "source_result_sha256",
        "source_media_path",
        "source_media_sha256",
        "media_kind",
        "prompt_id",
        "model_name",
        "seed",
        "condition_id",
        "condition_family",
        "variant",
        "variant_kind",
        "active_pair_ids",
        "inactive_pair_ids",
        "native_negative_source_pair_ids",
        "sampled_frame_indices",
        "sampled_frames_json",
        "review_entry_path",
        "review_entry_sha256",
        "contact_sheet_path",
        "contact_sheet_page",
        "contact_sheet_position",
        "source_fidelity_rating",
        "overall_semantic_success",
        "person_identity_preserved",
        "gender_preserved",
        "age_preserved",
        "mall_no_people_preserved",
        "mall_architecture_preserved",
        "mall_motion_correct",
        "mall_signage_correct",
        "artifacts_present",
        "temporal_consistency_rating",
        "active_targets_json",
        "native_negative_source_suppression_json",
        "inactive_source_preservation_json",
        "reviewer_notes",
        "rubric_json",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            rubric = record["rubric"]
            writer.writerow(
                {
                    "record_id": record["record_id"],
                    "review_status": record["review_status"],
                    "manifest_input_index": record["manifest_input_index"],
                    "manifest_job_index": record["manifest_job_index"],
                    "source_manifest_path": record["source_manifest_path"],
                    "source_manifest_file_sha256": record["source_manifest_file_sha256"],
                    "source_manifest_sha256": record["source_manifest_sha256"],
                    "source_result_path": record["source_result_path"],
                    "source_result_sha256": record["source_result_sha256"],
                    "source_media_path": record["source_media_path"],
                    "source_media_sha256": record["source_media_sha256"],
                    "media_kind": record["media_kind"],
                    "prompt_id": record["prompt_id"],
                    "model_name": record["model_name"],
                    "seed": record["seed"],
                    "condition_id": record["condition_id"],
                    "condition_family": record["condition_family"],
                    "variant": record["variant"],
                    "variant_kind": record["variant_kind"],
                    "active_pair_ids": json.dumps(record["active_pair_ids"]),
                    "inactive_pair_ids": json.dumps(record["inactive_pair_ids"]),
                    "native_negative_source_pair_ids": json.dumps(
                        record["native_negative_source_pair_ids"]
                    ),
                    "sampled_frame_indices": json.dumps(
                        [frame["frame_index"] for frame in record["sampled_frames"]]
                    ),
                    "sampled_frames_json": json.dumps(record["sampled_frames"], sort_keys=True),
                    "review_entry_path": record["review_entry_path"],
                    "review_entry_sha256": record["review_entry_sha256"],
                    "contact_sheet_path": record["contact_sheet_path"],
                    "contact_sheet_page": record["contact_sheet_page"],
                    "contact_sheet_position": record["contact_sheet_position"],
                    "source_fidelity_rating": "",
                    "overall_semantic_success": "",
                    "person_identity_preserved": "",
                    "gender_preserved": "",
                    "age_preserved": "",
                    "mall_no_people_preserved": "",
                    "mall_architecture_preserved": "",
                    "mall_motion_correct": "",
                    "mall_signage_correct": "",
                    "artifacts_present": "",
                    "temporal_consistency_rating": "",
                    "active_targets_json": json.dumps(rubric["active_targets"], sort_keys=True),
                    "native_negative_source_suppression_json": json.dumps(
                        rubric["native_negative_source_suppression"], sort_keys=True
                    ),
                    "inactive_source_preservation_json": json.dumps(
                        rubric["inactive_source_preservation"], sort_keys=True
                    ),
                    "reviewer_notes": "",
                    "rubric_json": json.dumps(rubric, sort_keys=True),
                }
            )


def _item_label(job: dict[str, Any]) -> str:
    active = list(job.get("variant_spec", {}).get("active_pair_ids", ()))
    kind = str(job.get("variant_spec", {}).get("kind", ""))
    if kind == "native_negative_prompt":
        active_label = "all 5 source concepts (native negative)"
    elif len(active) == 5:
        active_label = "all 5 concept pairs"
    elif active:
        active_label = str(active[0])
    else:
        active_label = "none"
    return (
        f"{job['prompt_id']} | {job['model_name']}\n"
        f"family={job['variation']} | variant={job['variant']}\n"
        f"active={active_label}"
    )


def _fit_on_canvas(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    contained = ImageOps.contain(image, size, method=Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", size, "#222222")
    canvas.paste(contained, ((size[0] - contained.width) // 2, (size[1] - contained.height) // 2))
    return canvas


def _font(size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(FONT_PATH), size=size, layout_engine=ImageFont.Layout.BASIC)


def _wrap_label(label: str, width: int) -> str:
    return "\n".join(
        wrapped
        for line in label.splitlines()
        for wrapped in (textwrap.wrap(line, width=width, break_long_words=True) or [""])
    )


def _save_png(image: Image.Image, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path, format="PNG", optimize=False, compress_level=9)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _relative(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))


def _slug(value: str) -> str:
    normalized = "".join(
        character if character.isalnum() or character in "-_" else "_" for character in value
    )
    normalized = normalized.strip("_")
    if not normalized:
        raise ValueError(f"Cannot construct a review path from {value!r}.")
    return normalized


def _manifest_digest(payload: dict[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("manifest_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _require_binary(name: str) -> None:
    if shutil.which(name) is None:
        raise FileNotFoundError(f"Required executable is unavailable: {name}")


if __name__ == "__main__":
    main()
