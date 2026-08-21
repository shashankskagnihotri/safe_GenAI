#!/usr/bin/env python
"""Audit immutable finer-detailing manifests and their attempt artifacts.

This command is independent of generation but deliberately Slurm-aware.  It can
be run repeatedly while arrays are active, but it never mutates an attempt
directory.  The only optional mutation is creation of a *new* immutable retry
manifest at a caller-selected path and strictly higher attempt number.

Manifest parsing is delegated to the benchmark's historical audit reader so
the benchmark name, embedded SHA-256 digest, optional digest sidecar, immutable
snapshot bundle, job count, and output-directory uniqueness are verified
without requiring old live source files to remain unchanged. Artifact
validation is deliberately repeated here: a ``completed`` string alone is not
enough to count an experiment as complete.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import subprocess
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

from PIL import Image

from hierasafe_flow.benchmarks.slurm_tracking import (
    EXECUTION_IDENTITY_FILENAME,
    SlurmStateProvider,
    load_submission_lookup,
    normalize_slurm_state,
    read_execution_identity,
)


STATUS_ORDER = (
    "pending",
    "running",
    "completed",
    "not_supported",
    "failed",
    "incomplete",
)
MEDIA_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".gif"}
ATTEMPT_DIRECTORY_RE = re.compile(r"^attempt_[0-9]+$")
SLURM_QUEUED_STATES = {
    "CONFIGURING",
    "PENDING",
    "REQUEUED",
    "REQUEUE_FED",
    "REQUEUE_HOLD",
    "RESIZING",
}
SLURM_ACTIVE_STATES = {
    "COMPLETING",
    "RUNNING",
    "SIGNALING",
    "STAGE_OUT",
    "SUSPENDED",
}
SLURM_FAILURE_STATES = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OOM",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "SPECIAL_EXIT",
    "STOPPED",
    "TIMEOUT",
}


@dataclass
class JobAudit:
    """Internal result for one manifest job/attempt."""

    manifest_path: str
    manifest_sha256: str
    job_index: int
    job: dict[str, Any]
    output_dir: Path
    condition_id: str
    status: str
    expected_status: str
    result_status: str | None = None
    media_path: str | None = None
    slurm_task_id: str | None = None
    slurm_state: str | None = None
    submission_registered: bool = False
    submission_registry_path: str | None = None
    execution_identity_path: str | None = None
    started: bool = False
    retry_blocked_reason: str | None = None
    issues: list[str] = field(default_factory=list)

    @property
    def attempt(self) -> int:
        return int(self.job.get("attempt", 0))

    @property
    def runnable(self) -> bool:
        return self.expected_status == "completed"

    @property
    def logical_key(self) -> tuple[str, str, str, str, int]:
        return (
            str(self.job.get("prompt_id", "unknown_prompt")),
            str(self.job.get("model_name", "unknown_model")),
            self.condition_id,
            str((self.job.get("generation") or {}).get("task", "unknown_task")),
            int(self.job.get("seed", 0)),
        )

    def mark_integrity_failure(self, issue: str) -> None:
        if issue not in self.issues:
            self.issues.append(issue)
        self.status = "incomplete"

    def as_public_dict(self, *, latest: bool) -> dict[str, Any]:
        task = str((self.job.get("generation") or {}).get("task", "unknown_task"))
        return {
            "manifest_path": self.manifest_path,
            "manifest_sha256": self.manifest_sha256,
            "job_index": self.job_index,
            "prompt_id": str(self.job.get("prompt_id", "unknown_prompt")),
            "model_name": str(self.job.get("model_name", "unknown_model")),
            "condition_id": self.condition_id,
            "variation": str(self.job.get("variation", self.job.get("variant", "unknown"))),
            "task": task,
            "seed": int(self.job.get("seed", 0)),
            "attempt": self.attempt,
            "output_dir": str(self.output_dir),
            "status": self.status,
            "result_status": self.result_status,
            "expected_status": self.expected_status,
            "runnable": self.runnable,
            "is_latest_attempt": latest,
            "media_path": self.media_path,
            "slurm_task_id": self.slurm_task_id,
            "slurm_state": self.slurm_state,
            "submission_registered": self.submission_registered,
            "submission_registry_path": self.submission_registry_path,
            "execution_identity_path": self.execution_identity_path,
            "started": self.started,
            "retry_eligible": _is_retry_eligible(self),
            "retry_blocked_reason": self.retry_blocked_reason,
            "issues": list(self.issues),
        }


@dataclass
class AuditReport:
    summary: dict[str, Any]
    records: list[JobAudit]
    latest_records: list[JobAudit]


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _benchmark_api() -> tuple[
    Callable[[Path, Path], dict[str, Any]],
    Callable[[dict[str, Any]], str],
    Callable[[dict[str, Any], Path, Path], None],
]:
    """Load the benchmark API lazily to avoid import-time coupling."""

    from hierasafe_flow.benchmarks.finer_detailing_correction import (
        manifest_digest,
        read_manifest_for_audit,
        write_manifest_immutable,
    )

    return read_manifest_for_audit, manifest_digest, write_manifest_immutable


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit finer-detailing manifests, artifacts, completeness, and retry eligibility."
    )
    parser.add_argument("manifests", nargs="+", help="One or more immutable manifest JSON files.")
    parser.add_argument("--project-root", default=str(project_root()))
    parser.add_argument(
        "--output-json",
        default="debugging/finer_detailing_correction_audit.json",
        help="Atomic machine-readable audit output.",
    )
    parser.add_argument(
        "--output-markdown",
        default="debugging/finer_detailing_correction_audit.md",
        help="Atomic human-readable audit output.",
    )
    parser.add_argument(
        "--require-complete",
        action="store_true",
        help="Exit nonzero unless every latest logical job has its expected valid terminal result and no stale attempts exist.",
    )
    parser.add_argument(
        "--retry-manifest",
        default=None,
        help="Write a new immutable manifest containing eligible latest failed/incomplete/pending jobs only.",
    )
    parser.add_argument(
        "--retry-attempt",
        type=int,
        default=None,
        help="Strictly higher positive attempt number for --retry-manifest.",
    )
    parser.add_argument(
        "--submission-registry",
        action="append",
        default=[],
        help=(
            "Immutable submission registry JSON. Repeat for arrays from multiple manifests. "
            "Registered queued/running tasks are never selected for retry."
        ),
    )
    parser.add_argument(
        "--allow-unsubmitted-pending",
        action="store_true",
        help=(
            "Explicitly allow never-registered pending jobs into a retry manifest. "
            "They are excluded by default to prevent duplicate launches."
        ),
    )
    parser.add_argument(
        "--allow-slurm-terminal-unconfirmed",
        action="store_true",
        help=(
            "Allow retry for slurm_terminal_state_unconfirmed records. The default "
            "behavior treats these rows as blocked because the scheduler no longer "
            "exposes terminal state."
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if (args.retry_manifest is None) != (args.retry_attempt is None):
        raise SystemExit("--retry-manifest and --retry-attempt must be supplied together.")

    root = Path(args.project_root).resolve()
    read_manifest, digest_fn, write_manifest = _benchmark_api()
    report = audit_manifests(
        [Path(value) for value in args.manifests],
        root=root,
        manifest_reader=read_manifest,
        submission_registry_paths=[Path(value) for value in args.submission_registry],
    )

    if args.retry_manifest is not None:
        retry_path = _resolve_against_root(Path(args.retry_manifest), root)
        retry_manifest = build_retry_manifest(
            report,
            int(args.retry_attempt),
            digest_fn=digest_fn,
            root=root,
            allow_unsubmitted_pending=bool(args.allow_unsubmitted_pending),
            allow_slurm_terminal_unconfirmed=bool(args.allow_slurm_terminal_unconfirmed),
        )
        write_manifest(retry_manifest, retry_path, root)
        report.summary["retry_manifest"] = {
            "path": str(retry_path),
            "sha256_sidecar": str(retry_path.with_suffix(retry_path.suffix + ".sha256")),
            "manifest_sha256": retry_manifest["manifest_sha256"],
            "attempt": retry_manifest["attempt"],
            "num_jobs": retry_manifest["num_jobs"],
        }

    json_path = _resolve_against_root(Path(args.output_json), root)
    markdown_path = _resolve_against_root(Path(args.output_markdown), root)
    _atomic_write_text(json_path, json.dumps(report.summary, indent=2, sort_keys=True) + "\n")
    _atomic_write_text(markdown_path, render_markdown(report.summary))

    print(
        json.dumps(
            {
                "terminal_complete": report.summary["terminal_complete"],
                "latest_status_counts": report.summary["latest_status_counts"],
                "retry_eligible": report.summary["retry_eligible_count"],
                "output_json": str(json_path),
                "output_markdown": str(markdown_path),
                "retry_manifest": report.summary.get("retry_manifest"),
            },
            indent=2,
            sort_keys=True,
        )
    )
    if args.require_complete and not bool(report.summary["terminal_complete"]):
        return 2
    return 0


def audit_manifests(
    manifest_paths: Sequence[Path],
    *,
    root: Path,
    manifest_reader: Callable[[Path, Path], dict[str, Any]],
    submission_registry_paths: Sequence[Path] = (),
    state_provider: Callable[[str], str | None] | None = None,
) -> AuditReport:
    """Read, verify, and classify all jobs from the supplied manifests."""

    root = root.resolve()
    resolved_manifest_paths = [_resolve_against_root(Path(path), root) for path in manifest_paths]
    if not resolved_manifest_paths:
        raise ValueError("At least one manifest is required.")
    if len(set(resolved_manifest_paths)) != len(resolved_manifest_paths):
        raise ValueError("The same manifest path was supplied more than once.")

    resolved_registry_paths = [
        _resolve_against_root(Path(path), root) for path in submission_registry_paths
    ]
    if len(set(resolved_registry_paths)) != len(resolved_registry_paths):
        raise ValueError("The same submission registry path was supplied more than once.")
    submission_lookup, submission_rows = load_submission_lookup(resolved_registry_paths)
    scheduler_state = state_provider or SlurmStateProvider()

    records: list[JobAudit] = []
    manifest_rows: list[dict[str, Any]] = []
    output_roots: set[Path] = set()
    for manifest_path in resolved_manifest_paths:
        # The benchmark reader verifies the embedded digest and its sidecar when
        # present. Do not replace this call with an unchecked json.loads.
        manifest = manifest_reader(manifest_path, root)
        digest = str(manifest["manifest_sha256"])
        output_root = _resolve_against_root(Path(str(manifest["output_root"])), root)
        output_roots.add(output_root)
        jobs = manifest["jobs"]
        manifest_rows.append(
            {
                "path": str(manifest_path),
                "schema_version": int(manifest.get("schema_version", 0)),
                "manifest_sha256": digest,
                "attempt": int(manifest.get("attempt", 0)),
                "num_jobs": len(jobs),
                "output_root": str(output_root),
                "sidecar_present": manifest_path.with_suffix(
                    manifest_path.suffix + ".sha256"
                ).is_file(),
                "video_contract": copy.deepcopy(manifest.get("video_contract")),
                "shapley_config": copy.deepcopy(manifest.get("shapley_config")),
                "shapley_provenance": copy.deepcopy(manifest.get("shapley_provenance")),
            }
        )
        for job_index, job_value in enumerate(jobs):
            if not isinstance(job_value, dict):
                raise ValueError(f"Manifest job {job_index} in {manifest_path} is not a mapping.")
            job = dict(job_value)
            records.append(
                classify_job(
                    job,
                    manifest_path=manifest_path,
                    manifest_sha256=digest,
                    job_index=job_index,
                    root=root,
                    submission=submission_lookup.get((digest, job_index)),
                    state_provider=scheduler_state,
                )
            )

    supplied_manifest_digests = {row["manifest_sha256"] for row in manifest_rows}
    foreign_registry_digests = sorted(
        {digest for digest, _ in submission_lookup if digest not in supplied_manifest_digests}
    )
    if foreign_registry_digests:
        raise ValueError(
            "Submission registries refer to manifests not included in this audit: "
            f"{foreign_registry_digests}"
        )

    duplicate_output_dirs = _mark_duplicate_output_dirs(records)
    duplicate_logical_attempts = _mark_duplicate_logical_attempts(records)
    latest_records = _latest_records(records)
    latest_ids = {id(record) for record in latest_records}

    referenced_dirs = {record.output_dir.resolve(strict=False) for record in records}
    stale_attempts = _find_unmanifested_attempts(output_roots, referenced_dirs)
    stale_manifested = [
        str(record.output_dir)
        for record in records
        if any(issue.startswith("result_job_mismatch") for issue in record.issues)
    ]

    attempt_status_counts = _complete_status_counter(record.status for record in records)
    latest_status_counts = _complete_status_counter(record.status for record in latest_records)
    expected_counts = _expected_counts(latest_records)
    successful_latest = all(record.status == record.expected_status for record in latest_records)
    terminal_complete = (
        bool(latest_records)
        and successful_latest
        and not (
            duplicate_output_dirs
            or duplicate_logical_attempts
            or stale_attempts
            or stale_manifested
        )
    )

    public_records = [record.as_public_dict(latest=id(record) in latest_ids) for record in records]
    public_latest = [record.as_public_dict(latest=True) for record in latest_records]
    aggregates = {
        "by_prompt": _aggregate(latest_records, ("prompt_id",)),
        "by_model": _aggregate(latest_records, ("model_name",)),
        "by_condition": _aggregate(latest_records, ("condition_id",)),
        "by_task": _aggregate(latest_records, ("task",)),
        "by_prompt_model_condition_task": _aggregate(
            latest_records,
            ("prompt_id", "model_name", "condition_id", "task"),
        ),
    }
    retry_eligible = [record for record in latest_records if _is_retry_eligible(record)]
    summary = {
        "schema_version": 2,
        "benchmark": "finer_detailing_correction_v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(root),
        "manifests": manifest_rows,
        "submission_registries": submission_rows,
        "submission_registry_count": len(submission_rows),
        "registered_attempt_count": sum(record.submission_registered for record in records),
        "unsubmitted_pending_count": sum(
            record.status == "pending" and not record.submission_registered
            for record in latest_records
        ),
        "manifest_count": len(manifest_rows),
        "attempt_record_count": len(records),
        "logical_experiment_count": len(latest_records),
        "attempt_status_counts": attempt_status_counts,
        "latest_status_counts": latest_status_counts,
        "expected_counts": expected_counts,
        "retry_eligible_count": len(retry_eligible),
        "terminal_complete": terminal_complete,
        "integrity": {
            "duplicate_output_dirs": duplicate_output_dirs,
            "duplicate_logical_attempts": duplicate_logical_attempts,
            "unmanifested_attempts": stale_attempts,
            "stale_manifested_attempts": sorted(set(stale_manifested)),
            "clean": not (
                duplicate_output_dirs
                or duplicate_logical_attempts
                or stale_attempts
                or stale_manifested
            ),
        },
        "aggregates": aggregates,
        "jobs": public_records,
        "latest_jobs": public_latest,
    }
    return AuditReport(summary=summary, records=records, latest_records=latest_records)


def classify_job(
    job: dict[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    job_index: int,
    root: Path,
    submission: dict[str, Any] | None = None,
    state_provider: Callable[[str], str | None] | None = None,
) -> JobAudit:
    """Classify artifacts, then reconcile nonterminal state with Slurm."""

    record = _classify_job_artifacts(
        job,
        manifest_path=manifest_path,
        manifest_sha256=manifest_sha256,
        job_index=job_index,
        root=root,
    )
    return _reconcile_slurm_state(
        record,
        submission=submission,
        state_provider=state_provider or SlurmStateProvider(),
    )


def _classify_job_artifacts(
    job: dict[str, Any],
    *,
    manifest_path: Path,
    manifest_sha256: str,
    job_index: int,
    root: Path,
) -> JobAudit:
    output_dir_value = job.get("output_dir")
    if not output_dir_value:
        output_dir = root / "__missing_output_dir__"
        return JobAudit(
            manifest_path=str(manifest_path),
            manifest_sha256=manifest_sha256,
            job_index=job_index,
            job=job,
            output_dir=output_dir,
            condition_id=_condition_id(job),
            status="incomplete",
            expected_status=_expected_terminal_status(job),
            issues=["manifest_job_missing_output_dir"],
        )

    output_dir = _resolve_against_root(Path(str(output_dir_value)), root)
    record = JobAudit(
        manifest_path=str(manifest_path),
        manifest_sha256=manifest_sha256,
        job_index=job_index,
        job=job,
        output_dir=output_dir,
        condition_id=_condition_id(job),
        status="pending",
        expected_status=_expected_terminal_status(job),
    )
    if not output_dir.exists():
        return record
    if not output_dir.is_dir():
        record.status = "incomplete"
        record.issues.append("output_path_is_not_a_directory")
        return record
    files = [path for path in output_dir.rglob("*") if path.is_file()]
    if not files:
        return record
    record.started = True

    result_path = output_dir / "benchmark_job_result.json"
    if not result_path.is_file():
        timing_status = _read_timing_status(output_dir / "experiment_timing.json")
        if timing_status in {"completed", "not_supported", "failed"}:
            record.status = "incomplete"
            record.issues.append(f"terminal_timing_without_result:{timing_status}")
        elif (output_dir / "benchmark_job.yaml").is_file():
            record.status = "running"
        else:
            record.status = "incomplete"
            record.issues.append("artifacts_exist_without_result_or_start_record")
        return record

    try:
        result = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        record.status = "incomplete"
        record.issues.append(f"unreadable_result_json:{type(exc).__name__}:{exc}")
        return record
    if not isinstance(result, dict):
        record.status = "incomplete"
        record.issues.append("result_json_is_not_an_object")
        return record

    raw_status = result.get("status")
    record.result_status = str(raw_status) if raw_status is not None else None
    if result.get("job") != job:
        record.status = "incomplete"
        record.issues.append(
            "result_job_mismatch:embedded job does not equal immutable manifest job"
        )
        return record

    if raw_status == "completed":
        issues, media_path = _validate_completed_result(job, output_dir, result)
        record.media_path = media_path
        if issues:
            record.status = "incomplete"
            record.issues.extend(issues)
        else:
            record.status = "completed"
        return record

    if raw_status == "not_supported":
        issues = _validate_not_supported_result(job, output_dir, result)
        if issues:
            record.status = "incomplete"
            record.issues.extend(issues)
        else:
            record.status = "not_supported"
        return record

    if raw_status == "failed":
        record.status = "failed"
        if not result.get("error"):
            record.issues.append("failed_result_missing_error_message")
        return record

    if raw_status in {"running", "started"}:
        record.status = "running"
        return record

    record.status = "incomplete"
    record.issues.append(f"unknown_or_missing_result_status:{raw_status!r}")
    return record


def _reconcile_slurm_state(
    record: JobAudit,
    *,
    submission: dict[str, Any] | None,
    state_provider: Callable[[str], str | None],
) -> JobAudit:
    """Use durable identity plus scheduler truth for nonterminal attempts."""

    registry_task_id: str | None = None
    if submission is not None:
        record.submission_registered = True
        record.submission_registry_path = str(submission.get("registry_path") or "") or None
        registry_task_id = str(submission.get("slurm_task_id") or "") or None

    identity_path = record.output_dir / EXECUTION_IDENTITY_FILENAME
    identity_task_id: str | None = None
    if identity_path.is_file():
        record.started = True
        record.execution_identity_path = str(identity_path)
        try:
            identity = read_execution_identity(identity_path)
            identity_task_id = str(identity.get("slurm_task_id") or "") or None
        except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
            record.mark_integrity_failure(f"invalid_execution_identity:{type(exc).__name__}:{exc}")

    if registry_task_id and identity_task_id and registry_task_id != identity_task_id:
        record.slurm_task_id = registry_task_id
        record.mark_integrity_failure(
            "submission_execution_identity_mismatch:"
            f"registry={registry_task_id};execution={identity_task_id}"
        )
        return record
    record.slurm_task_id = registry_task_id or identity_task_id

    # Successful validated results are authoritative and cannot be selected for
    # retry. A failed result is still reconciled when scheduler identity exists,
    # preventing a retry while the failing task is RUNNING/COMPLETING.
    if record.status in {"completed", "not_supported"}:
        return record
    if record.slurm_task_id is None:
        return record

    artifact_status = record.status

    try:
        record.slurm_state = normalize_slurm_state(state_provider(record.slurm_task_id))
    except Exception as exc:  # Audits must survive a transient scheduler query failure.
        record.issues.append(f"slurm_state_query_failed:{type(exc).__name__}:{exc}")
        record.slurm_state = None

    state = record.slurm_state
    if state in SLURM_QUEUED_STATES:
        # Keep a never-started queued element visibly pending, but its durable
        # submission registration makes it ineligible for retries.
        record.status = "running" if record.started else "pending"
        return record
    if state in SLURM_ACTIVE_STATES:
        record.status = "running"
        return record
    if state == "COMPLETED":
        if artifact_status == "failed":
            record.status = "failed"
            record.issues.append("benchmark_failed_result_with_slurm_completed_state")
        else:
            record.status = "incomplete"
            record.issues.append("slurm_completed_without_valid_terminal_result")
        return record
    if state in SLURM_FAILURE_STATES:
        record.status = "failed"
        record.issues.append(f"slurm_terminal_failure:{state}")
        return record
    if state is None:
        record.status = "failed" if artifact_status == "failed" else "incomplete"
        record.retry_blocked_reason = "slurm_terminal_state_unconfirmed"
        record.issues.append(
            "submitted_slurm_task_not_found_in_squeue_or_sacct;retry_blocked_without_terminal_state"
        )
        return record

    # Unknown scheduler states must not become automatic retries.  Classifying
    # them as running is conservative until the state contract is extended.
    record.status = "running"
    record.issues.append(f"unrecognized_nonterminal_slurm_state:{state}")
    return record


def _validate_completed_result(
    job: dict[str, Any], output_dir: Path, result: dict[str, Any]
) -> tuple[list[str], str | None]:
    issues: list[str] = []
    task = str((job.get("generation") or {}).get("task", ""))
    if task == "text_to_image":
        expected_path = output_dir / "sample_0000" / "image_000.png"
    elif task == "text_to_video":
        expected_path = output_dir / "sample_0000" / "video_000.mp4"
    else:
        return [f"unsupported_generation_task:{task!r}"], None

    media = _media_files(output_dir)
    if [path.resolve(strict=False) for path in media] != [expected_path.resolve(strict=False)]:
        issues.append(
            "exact_media_set_mismatch:"
            f"expected={expected_path};found={[str(path) for path in media]}"
        )
    if not expected_path.is_file() or expected_path.stat().st_size <= 0:
        issues.append(f"missing_or_empty_expected_media:{expected_path}")
        return issues, str(expected_path)

    validation = result.get("media_validation")
    if not isinstance(validation, dict):
        issues.append("completed_result_missing_media_validation_object")
        return issues, str(expected_path)
    recorded_paths = result.get("validated_media_paths")
    if not isinstance(recorded_paths, list) or len(recorded_paths) != 1:
        issues.append("completed_result_requires_exactly_one_validated_media_path")
    else:
        if _canonical_recorded_path(recorded_paths[0], output_dir) != expected_path.resolve(
            strict=False
        ):
            issues.append("validated_media_paths_does_not_match_exact_expected_path")
    if _canonical_recorded_path(validation.get("path"), output_dir) != expected_path.resolve(
        strict=False
    ):
        issues.append("media_validation_path_does_not_match_exact_expected_path")

    try:
        fresh = _fresh_media_metadata(expected_path, job)
    except Exception as exc:  # The issue must be recorded, not crash a long audit.
        issues.append(f"media_probe_failed:{type(exc).__name__}:{exc}")
        return issues, str(expected_path)

    required_keys = set(fresh)
    missing_keys = sorted(required_keys - set(validation))
    if missing_keys:
        issues.append(f"media_validation_missing_keys:{missing_keys}")
    for key, fresh_value in fresh.items():
        if key == "path":
            continue
        stored_value = validation.get(key)
        if key in {"fps", "duration_seconds"}:
            try:
                if abs(float(stored_value) - float(fresh_value)) > 1.0e-6:
                    issues.append(
                        f"media_metadata_mismatch:{key}:{stored_value!r}!={fresh_value!r}"
                    )
            except (TypeError, ValueError):
                issues.append(f"media_metadata_invalid_numeric:{key}:{stored_value!r}")
        elif stored_value != fresh_value:
            issues.append(f"media_metadata_mismatch:{key}:{stored_value!r}!={fresh_value!r}")
    return issues, str(expected_path)


def _validate_not_supported_result(
    job: dict[str, Any], output_dir: Path, result: dict[str, Any]
) -> list[str]:
    issues: list[str] = []
    media = _media_files(output_dir)
    if media:
        issues.append(f"not_supported_contains_media:{[str(path) for path in media]}")
    if result.get("media_validation") is not None:
        issues.append("not_supported_media_validation_must_be_null")
    if result.get("validated_media_paths") != []:
        issues.append("not_supported_validated_media_paths_must_be_empty")
    if _expected_terminal_status(job) != "not_supported":
        issues.append("unexpected_not_supported_for_runnable_job")
    if not result.get("reason"):
        issues.append("not_supported_result_missing_reason")
    return issues


def _fresh_media_metadata(path: Path, job: dict[str, Any]) -> dict[str, Any]:
    task = str((job.get("generation") or {}).get("task", ""))
    generation = job.get("generation") or {}
    if task == "text_to_image":
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            image_format = image.format
        if image_format != "PNG":
            raise ValueError(f"expected PNG encoding, found {image_format!r}")
        expected_size = (int(generation["width"]), int(generation["height"]))
        if (width, height) != expected_size:
            raise ValueError(f"image dimensions {(width, height)} != {expected_size}")
        return {
            "path": str(path),
            "media_type": "image/png",
            "size_bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
            "width": width,
            "height": height,
            "mode": mode,
            "decode_verified": True,
        }

    if task != "text_to_video":
        raise ValueError(f"unsupported task {task!r}")
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
    probe = subprocess.run(command, check=True, capture_output=True, text=True, timeout=600)
    payload = json.loads(probe.stdout)
    streams = payload.get("streams") or []
    if len(streams) != 1:
        raise ValueError(f"expected one video stream, found {len(streams)}")
    stream = streams[0]
    width, height = int(stream["width"]), int(stream["height"])
    expected_size = (int(generation["width"]), int(generation["height"]))
    if (width, height) != expected_size:
        raise ValueError(f"video dimensions {(width, height)} != {expected_size}")
    frame_value = stream.get("nb_read_frames") or stream.get("nb_frames")
    if frame_value in (None, "N/A"):
        raise ValueError("ffprobe did not report frame count")
    frame_count = int(frame_value)
    expected_frames = int(generation["num_frames"])
    if frame_count != expected_frames:
        raise ValueError(f"video frame count {frame_count} != {expected_frames}")
    fps = float(Fraction(str(stream["avg_frame_rate"])))
    expected_fps = float(generation["fps"])
    if abs(fps - expected_fps) > 1.0e-3:
        raise ValueError(f"video fps {fps} != {expected_fps}")
    duration = float((payload.get("format") or {}).get("duration") or stream.get("duration"))
    expected_duration = expected_frames / expected_fps
    if abs(duration - expected_duration) > (1.0 / expected_fps + 1.0e-3):
        raise ValueError(f"video duration {duration} != approximately {expected_duration}")
    return {
        "path": str(path),
        "media_type": "video/mp4",
        "size_bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
        "codec_name": stream.get("codec_name"),
        "pixel_format": stream.get("pix_fmt"),
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        # Generation records must prove that the benchmark performed its full
        # decode check; the unchanged SHA-256 binds that proof to this file.
        "decode_verified": True,
    }


def build_retry_manifest(
    report: AuditReport,
    attempt: int,
    *,
    digest_fn: Callable[[dict[str, Any]], str],
    root: Path,
    allow_unsubmitted_pending: bool = False,
    allow_slurm_terminal_unconfirmed: bool = False,
) -> dict[str, Any]:
    """Create (but do not write) a retry manifest for eligible latest jobs."""

    if attempt <= 0:
        raise ValueError("Retry attempt must be positive.")
    max_attempt = max((record.attempt for record in report.records), default=0)
    if attempt <= max_attempt:
        raise ValueError(
            f"Retry attempt {attempt} must be higher than every supplied attempt ({max_attempt})."
        )
    eligible = [
        record
        for record in report.latest_records
        if _is_retry_eligible(
            record,
            allow_unsubmitted_pending=allow_unsubmitted_pending,
            allow_slurm_terminal_unconfirmed=allow_slurm_terminal_unconfirmed,
        )
    ]
    if not eligible:
        raise ValueError(
            "No latest failed or incomplete runnable jobs are eligible for retry. "
            "Never-submitted pending jobs require explicit allow_unsubmitted_pending=True."
        )

    # This builder is intentionally an operational same-implementation retry,
    # not a protocol migration tool.  In particular, a repaired Shapley cohort
    # must be rebuilt from ``build_manifest`` so completed historical rows are
    # included under one new config/provenance and implementation digest.
    frozen_implementation_digests = {
        str(record.job["implementation_files_sha256"])
        for record in eligible
        if isinstance(record.job.get("implementation_files_sha256"), str)
    }
    if len(frozen_implementation_digests) > 1:
        raise ValueError(
            "Operational retry jobs span multiple implementation digests; split same-code "
            "retries or rebuild a complete repaired protocol cohort with build_manifest."
        )
    if frozen_implementation_digests:
        from hierasafe_flow.benchmarks.finer_detailing_correction import (
            _implementation_provenance,
        )

        live_digest = str(_implementation_provenance(root)["implementation_files_sha256"])
        frozen_digest = next(iter(frozen_implementation_digests))
        if frozen_digest != live_digest:
            raise ValueError(
                "Operational retry source implementation differs from the live source. "
                "Do not use build_retry_manifest for a Shapley protocol repair; build the "
                "complete higher-attempt cohort with build_manifest so old completed rows "
                "are superseded consistently."
            )

    jobs: list[dict[str, Any]] = []
    target_dirs: set[Path] = set()
    for record in eligible:
        job = copy.deepcopy(record.job)
        logical_dir = _logical_variation_dir(job, record.output_dir, root)
        output_dir = logical_dir / "attempts" / f"attempt_{attempt:03d}"
        canonical = output_dir.resolve(strict=False)
        if canonical in target_dirs:
            raise ValueError(f"Retry jobs collide at {output_dir}.")
        if output_dir.exists():
            raise FileExistsError(f"Retry output directory already exists: {output_dir}")
        target_dirs.add(canonical)
        job["attempt"] = attempt
        job["variation_dir"] = str(logical_dir)
        job["output_dir"] = str(output_dir)
        job["retry_of_manifest_sha256"] = record.manifest_sha256
        job["retry_of_job_index"] = record.job_index
        job["retry_of_result_status"] = record.result_status or record.status
        job["retry_of_attempt"] = record.attempt
        job["retry_of_output_dir"] = str(record.output_dir)
        jobs.append(job)

    output_roots = {str(row["output_root"]) for row in report.summary["manifests"]}
    if len(output_roots) != 1:
        raise ValueError(
            f"Retry manifest requires one shared output root, found {sorted(output_roots)}"
        )
    seeds = {int(job.get("seed", -1)) for job in jobs}
    if len(seeds) != 1:
        raise ValueError(
            "One retry manifest must preserve one paired generation seed; "
            f"split mixed-seed jobs into separate retries, found {sorted(seeds)}."
        )
    seed = next(iter(seeds))
    if not 0 <= seed <= 2**32 - 1:
        raise ValueError(f"Retry job seed is outside 0..2**32-1: {seed}")
    manifest: dict[str, Any] = {
        "schema_version": max(
            (int(row.get("schema_version", 0)) for row in report.summary["manifests"]),
            default=3,
        ),
        "benchmark": "finer_detailing_correction_v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "attempt": attempt,
        "models": sorted({str(job.get("model_name")) for job in jobs}),
        "prompt_ids": sorted({str(job.get("prompt_id")) for job in jobs}),
        "variation_groups": sorted({str(job.get("variation", job.get("variant"))) for job in jobs}),
        "condition_ids": sorted({_condition_id(job) for job in jobs}),
        "output_root": next(iter(output_roots)),
        "num_jobs": len(jobs),
        "expected_media_jobs": sum(bool(job.get("expected_media", True)) for job in jobs),
        "expected_not_supported_jobs": sum(
            not bool(job.get("expected_media", True)) for job in jobs
        ),
        "jobs": jobs,
        "retry_source_manifests": [
            {
                "path": row["path"],
                "manifest_sha256": row["manifest_sha256"],
                "attempt": row["attempt"],
            }
            for row in report.summary["manifests"]
        ],
        "retry_selection_statuses": sorted({record.status for record in eligible}),
        "retry_excludes_running_jobs": True,
        "retry_excludes_registered_pending_jobs": True,
        "retry_allows_unsubmitted_pending_jobs": allow_unsubmitted_pending,
    }
    video_contracts = {
        json.dumps(row["video_contract"], sort_keys=True)
        for row in report.summary["manifests"]
        if row.get("video_contract") is not None
    }
    if len(video_contracts) == 1:
        manifest["video_contract"] = json.loads(next(iter(video_contracts)))
    shapley_configs = {
        json.dumps(row["shapley_config"], sort_keys=True)
        for row in report.summary["manifests"]
        if row.get("shapley_config") is not None
    }
    if len(shapley_configs) == 1:
        manifest["shapley_config"] = json.loads(next(iter(shapley_configs)))
    shapley_provenance = {
        json.dumps(row["shapley_provenance"], sort_keys=True)
        for row in report.summary["manifests"]
        if row.get("shapley_provenance") is not None
    }
    if len(shapley_provenance) == 1:
        parsed_shapley_provenance = json.loads(next(iter(shapley_provenance)))
        manifest["shapley_provenance"] = parsed_shapley_provenance
        if (
            parsed_shapley_provenance.get("protocol_version") == 2
            and len(shapley_configs) != 1
        ):
            raise ValueError(
                "Protocol-v2 same-code Shapley retries require one frozen top-level "
                "shapley_config; rebuild or split the retry inputs."
            )
    manifest["manifest_sha256"] = digest_fn(manifest)
    return manifest


def _is_retry_eligible(
    record: JobAudit,
    *,
    allow_unsubmitted_pending: bool = False,
    allow_slurm_terminal_unconfirmed: bool = False,
) -> bool:
    if not record.runnable:
        return False
    if record.retry_blocked_reason is not None:
        if (
            not allow_slurm_terminal_unconfirmed
            or record.retry_blocked_reason != "slurm_terminal_state_unconfirmed"
        ):
            return False
        if record.status != "incomplete":
            return False
        if record.started:
            return False
        if record.slurm_task_id is None:
            return False
    if record.status in {"failed", "incomplete"}:
        return True
    return bool(
        allow_unsubmitted_pending
        and record.status == "pending"
        and not record.submission_registered
        and record.slurm_task_id is None
    )


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Finer Detailing Correction — Completeness Audit",
        "",
        f"Generated: `{summary['generated_at_utc']}`  ",
        f"Terminal complete: **{str(summary['terminal_complete']).lower()}**  ",
        f"Immutable manifests: **{summary['manifest_count']}**  ",
        f"Logical experiments: **{summary['logical_experiment_count']}**  ",
        f"Recorded attempts: **{summary['attempt_record_count']}**  ",
        f"Retry-eligible latest jobs: **{summary['retry_eligible_count']}**",
        "",
        "## Expected and Observed Latest Results",
        "",
        "| Metric | Count |",
        "|---|---:|",
    ]
    for key, value in summary["expected_counts"].items():
        lines.append(f"| Expected {key.replace('_', ' ')} | {value} |")
    for status in STATUS_ORDER:
        lines.append(f"| Observed {status} | {summary['latest_status_counts'][status]} |")

    lines.extend(
        ["", "## Manifests", "", "| Attempt | Jobs | SHA-256 | Path |", "|---:|---:|---|---|"]
    )
    for row in summary["manifests"]:
        lines.append(
            f"| {row['attempt']} | {row['num_jobs']} | `{row['manifest_sha256']}` | `{row['path']}` |"
        )

    for title, key in (
        ("By Prompt", "by_prompt"),
        ("By Model", "by_model"),
        ("By Condition", "by_condition"),
        ("By Task", "by_task"),
    ):
        lines.extend(["", f"## {title}", "", _aggregate_markdown(summary["aggregates"][key])])

    integrity = summary["integrity"]
    lines.extend(["", "## Integrity", ""])
    lines.append(f"Integrity clean: **{str(integrity['clean']).lower()}**")
    for label in (
        "duplicate_output_dirs",
        "duplicate_logical_attempts",
        "unmanifested_attempts",
        "stale_manifested_attempts",
    ):
        values = integrity[label]
        lines.extend(["", f"### {label.replace('_', ' ').title()}", ""])
        if values:
            lines.extend(f"- `{value}`" for value in values)
        else:
            lines.append("None.")

    problematic = [
        row
        for row in summary["latest_jobs"]
        if row["status"] != row["expected_status"] or row["issues"]
    ]
    lines.extend(["", "## Latest Jobs Requiring Attention", ""])
    if not problematic:
        lines.append("None.")
    else:
        lines.extend(
            [
                "| Prompt | Model | Condition | Attempt | Status | Expected | Issues | Output |",
                "|---|---|---|---:|---|---|---|---|",
            ]
        )
        for row in problematic:
            issue_text = "; ".join(str(value).replace("|", "\\|") for value in row["issues"]) or "—"
            lines.append(
                f"| {row['prompt_id']} | {row['model_name']} | {row['condition_id']} | "
                f"{row['attempt']} | {row['status']} | {row['expected_status']} | {issue_text} | "
                f"`{row['output_dir']}` |"
            )
    retry = summary.get("retry_manifest")
    if retry:
        lines.extend(
            [
                "",
                "## Retry Manifest Written",
                "",
                f"- Path: `{retry['path']}`",
                f"- Attempt: `{retry['attempt']}`",
                f"- Jobs: `{retry['num_jobs']}`",
                f"- SHA-256: `{retry['manifest_sha256']}`",
            ]
        )
    return "\n".join(lines) + "\n"


def _aggregate(records: Sequence[JobAudit], fields: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[JobAudit]] = defaultdict(list)
    for record in records:
        values = {
            "prompt_id": str(record.job.get("prompt_id", "unknown_prompt")),
            "model_name": str(record.job.get("model_name", "unknown_model")),
            "condition_id": record.condition_id,
            "task": str((record.job.get("generation") or {}).get("task", "unknown_task")),
        }
        grouped[tuple(values[field] for field in fields)].append(record)
    rows: list[dict[str, Any]] = []
    for key_values, group in sorted(grouped.items()):
        row = {field: value for field, value in zip(fields, key_values)}
        row.update(
            {
                "expected_total": len(group),
                "expected_completed": sum(
                    record.expected_status == "completed" for record in group
                ),
                "expected_not_supported": sum(
                    record.expected_status == "not_supported" for record in group
                ),
                "expected_media": sum(record.expected_status == "completed" for record in group),
                "status_counts": _complete_status_counter(record.status for record in group),
                "terminal_complete": all(
                    record.status == record.expected_status for record in group
                ),
            }
        )
        rows.append(row)
    return rows


def _aggregate_markdown(rows: Sequence[dict[str, Any]]) -> str:
    if not rows:
        return "No records."
    dimension_keys = [
        key for key in ("prompt_id", "model_name", "condition_id", "task") if key in rows[0]
    ]
    headers = dimension_keys + [
        "Expected",
        "Completed",
        "Unsupported",
        "Failed",
        "Incomplete",
        "Pending",
        "Running",
        "OK",
    ]
    output = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        counts = row["status_counts"]
        values = [str(row[key]) for key in dimension_keys] + [
            str(row["expected_total"]),
            str(counts["completed"]),
            str(counts["not_supported"]),
            str(counts["failed"]),
            str(counts["incomplete"]),
            str(counts["pending"]),
            str(counts["running"]),
            "yes" if row["terminal_complete"] else "no",
        ]
        output.append("| " + " | ".join(values) + " |")
    return "\n".join(output)


def _condition_id(job: dict[str, Any]) -> str:
    condition = None
    for key in ("condition_id", "condition", "variant", "variation"):
        value = job.get(key)
        if value not in (None, ""):
            condition = str(value)
            break
    if condition is None:
        condition = "unknown_condition"
    pair_id = job.get("active_pair_id") or job.get("pair_id")
    if not pair_id:
        active = (job.get("variant_spec") or {}).get("active_pair_ids")
        if isinstance(active, (list, tuple)) and len(active) == 1:
            pair_id = active[0]
    if pair_id and str(pair_id) not in condition:
        condition = f"{condition}__{pair_id}"
    return condition


def _expected_terminal_status(job: dict[str, Any]) -> str:
    if job.get("expected_media") is False:
        return "not_supported"
    if job.get("expected_media") is True:
        return "completed"
    if job.get("runnable") is False:
        return "not_supported"
    variant_spec = job.get("variant_spec") or {}
    if (
        variant_spec.get("kind") == "native_negative_prompt"
        and variant_spec.get("capability") == "not_supported"
    ):
        return "not_supported"
    return "completed"


def _expected_counts(records: Sequence[JobAudit]) -> dict[str, int]:
    completed = sum(record.expected_status == "completed" for record in records)
    unsupported = sum(record.expected_status == "not_supported" for record in records)
    return {
        "logical_jobs": len(records),
        "completed": completed,
        "not_supported": unsupported,
        "media_files": completed,
    }


def _mark_duplicate_output_dirs(records: Sequence[JobAudit]) -> list[str]:
    grouped: dict[Path, list[JobAudit]] = defaultdict(list)
    for record in records:
        grouped[record.output_dir.resolve(strict=False)].append(record)
    duplicates: list[str] = []
    for path, group in grouped.items():
        if len(group) > 1:
            duplicates.append(str(path))
            for record in group:
                record.mark_integrity_failure(f"duplicate_output_dir_across_manifests:{path}")
    return sorted(duplicates)


def _mark_duplicate_logical_attempts(records: Sequence[JobAudit]) -> list[str]:
    grouped: dict[tuple[tuple[str, str, str, str, int], int], list[JobAudit]] = defaultdict(list)
    for record in records:
        grouped[(record.logical_key, record.attempt)].append(record)
    duplicates: list[str] = []
    for (logical_key, attempt), group in grouped.items():
        if len(group) > 1:
            label = f"{logical_key}:attempt_{attempt:03d}"
            duplicates.append(label)
            for record in group:
                record.mark_integrity_failure(f"duplicate_logical_attempt:{label}")
    return sorted(duplicates)


def _latest_records(records: Sequence[JobAudit]) -> list[JobAudit]:
    grouped: dict[tuple[str, str, str, str, int], list[JobAudit]] = defaultdict(list)
    for record in records:
        grouped[record.logical_key].append(record)
    latest: list[JobAudit] = []
    for key in sorted(grouped):
        group = sorted(
            grouped[key],
            key=lambda record: (record.attempt, record.manifest_path, record.job_index),
        )
        latest.append(group[-1])
    return latest


def _find_unmanifested_attempts(output_roots: Iterable[Path], referenced: set[Path]) -> list[str]:
    stale: set[str] = set()
    for output_root in sorted(set(output_roots)):
        if not output_root.is_dir():
            continue
        for path in output_root.rglob("attempt_*"):
            if path.is_dir() and ATTEMPT_DIRECTORY_RE.fullmatch(path.name):
                canonical = path.resolve(strict=False)
                if canonical not in referenced:
                    stale.add(str(canonical))
    return sorted(stale)


def _logical_variation_dir(job: dict[str, Any], old_output_dir: Path, root: Path) -> Path:
    value = job.get("variation_dir")
    if value:
        base = _resolve_against_root(Path(str(value)), root)
    elif old_output_dir.parent.name == "attempts":
        base = old_output_dir.parent.parent
    elif ATTEMPT_DIRECTORY_RE.fullmatch(old_output_dir.name):
        base = old_output_dir.parent
    else:
        raise ValueError(f"Cannot derive logical variation directory from {old_output_dir}")
    if base.name == "attempts":
        base = base.parent
    if ATTEMPT_DIRECTORY_RE.fullmatch(base.name):
        base = base.parent
    return base


def _read_timing_status(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        return None
    if isinstance(payload, dict) and payload.get("status") is not None:
        return str(payload["status"])
    return None


def _media_files(output_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in MEDIA_SUFFIXES
    )


def _canonical_recorded_path(value: Any, output_dir: Path) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    if not path.is_absolute():
        # Generation normally records an absolute path. Relative paths are
        # interpreted from the attempt directory, never from the auditor cwd.
        path = output_dir / path
    return path.resolve(strict=False)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _complete_status_counter(statuses: Iterable[str]) -> dict[str, int]:
    counts = Counter(statuses)
    return {status: int(counts.get(status, 0)) for status in STATUS_ORDER}


def _resolve_against_root(path: Path, root: Path) -> Path:
    if not path.is_absolute():
        path = root / path
    return path.resolve(strict=False)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


if __name__ == "__main__":
    raise SystemExit(main())
