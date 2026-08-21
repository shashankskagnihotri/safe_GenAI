"""Seal the fresh finer-detailing plotting evidence without self-attestation.

The plotter deliberately consumes a compact evidence bundle.  This module is
the only production writer for that bundle.  Callers supply paths, never file
hashes or campaign-row identities: manifests, Slurm registries, completed
attempt directories, selections, and three domain-separated raw review/metric
documents are reopened and hashed here.

Publication uses a Ceph-safe commit-last transaction.  The bundle and all
normalized ledgers are first written below a hidden sibling directory, reopened
through ``load_fresh_plot_data``, frozen, and hard-linked into an exclusively
claimed canonical directory.  The self-authenticating ``fresh_bundle.json`` is
linked last and the entire canonical tree is then made read-only.  Readers reject
both a missing commit and a writable claim, so no partial ledger set is admissible
and no racing writer can be replaced.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import stat
import tempfile
from typing import Any

from hierasafe_flow.benchmarks import finer_detailing_qualification as qualification
from hierasafe_flow.benchmarks.finer_detailing_correction import (
    BENCHMARK_NAME,
    is_flux1_job_v3,
    read_manifest_for_audit,
    reopen_completed_flux1_output_v3,
    validate_exact_media,
)
from hierasafe_flow.benchmarks.slurm_tracking import (
    EXECUTION_IDENTITY_FILENAME,
    read_execution_identity,
    read_submission_registry,
)
from hierasafe_flow.evaluation import finer_detailing_campaign as campaign
from hierasafe_flow.evaluation import finer_detailing_selection_cohort as selection_cohort
from hierasafe_flow.utils.immutable_publication import (
    cleanup_owned_staging,
    freeze_tree,
    publish_hardlink_tree_commit_last,
)


PUBLICATION_CONTRACT = "finer_detailing_fresh_plot_bundle_publication_v1"
RAW_LEDGER_INPUT_CONTRACTS = {
    "objective_temporal": "finer_detailing_objective_temporal_raw_v1",
    "manual_semantic": "finer_detailing_manual_semantic_raw_v1",
    "shapley_diagnostics": "finer_detailing_shapley_diagnostics_raw_v1",
}
OUTPUT_LEDGER_CONTRACTS = {
    "objective_temporal": "finer_detailing_objective_temporal_ledger_v1",
    "manual_semantic": "finer_detailing_manual_semantic_ledger_v1",
    "shapley_diagnostics": "finer_detailing_shapley_diagnostics_ledger_v1",
}
EXPECTED_MANIFEST_COUNTS = {"qualification": 8, "seed_ladder": 8, "final": 72}
EXPECTED_LOGICAL_COUNTS = {"qualification": 396, "seed_ladder": 288, "final": 504}
EXPECTED_REGISTRY_COUNT = sum(EXPECTED_MANIFEST_COUNTS.values())

_RAW_TOP_LEVEL_KEYS = {
    "schema_version",
    "evidence_input",
    "sealed_at_utc",
    "rows",
    "document_sha256",
}
_RAW_ROW_KEYS = {
    "objective_temporal": {
        "condition_id",
        "automatic_gate_pass",
        "freeze_candidate",
        "near_periodic_candidate",
        "cut_candidate",
        "cadence_candidate",
        "prompt3_motion_gate_pass",
    },
    "manual_semantic": {
        "condition_id",
        "semantic_success",
        "source_fidelity",
        "non_target_preservation",
        "gender_preserved",
        "native_negative_source_suppression",
        "active_pair_results",
    },
    "shapley_diagnostics": {
        "condition_id",
        "protocol_version",
        "trace_schema_version",
        "validation_pass",
        "accepted_step_fraction",
        "aggregate_score_decrease",
        "selected_coordinate_fraction",
        "max_ci_half_width",
        "non_target_violation_count",
    },
}


PlotLoader = Callable[[Path, Path], Any]
MediaValidator = Callable[[Path, dict[str, Any], bool], Mapping[str, Any]]
SelectionCohortReader = Callable[..., Mapping[str, Any]]


def _cohort_condition_id(cohort: str, condition_id: str) -> str:
    if cohort not in EXPECTED_MANIFEST_COUNTS or not condition_id:
        raise ValueError("Plot evidence row lacks a registered cohort/condition identity.")
    return f"{cohort}::{condition_id}"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
            raise ValueError(f"SHA-256 source is not a regular file: {path}.")
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any], field: str) -> str:
    canonical = dict(payload)
    canonical.pop(field, None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a timezone-aware ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not valid ISO-8601: {value!r}.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include an explicit timezone.")
    return parsed


def _ensure_below_root(path: Path, root: Path, label: str) -> None:
    if path != root and root not in path.parents:
        raise ValueError(f"{label} escapes the project root: {path}.")


def _source_path(value: str | Path, root: Path, label: str) -> Path:
    raw = Path(value).expanduser()
    lexical = raw if raw.is_absolute() else root / raw
    lexical = Path(os.path.abspath(lexical))
    _ensure_below_root(lexical, root, label)
    cursor = root
    for part in lexical.relative_to(root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} traverses a forbidden symlink: {cursor}.")
    try:
        resolved = lexical.resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {lexical}.") from exc
    _ensure_below_root(resolved, root, label)
    if not resolved.is_file():
        raise ValueError(f"{label} is not a regular file: {resolved}.")
    return resolved


def _destination_path(value: str | Path, root: Path) -> Path:
    raw = Path(value).expanduser()
    lexical = raw if raw.is_absolute() else root / raw
    destination = Path(os.path.abspath(lexical))
    _ensure_below_root(destination, root, "plot evidence destination")
    if destination == root:
        raise ValueError("Plot evidence destination cannot be the project root.")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Immutable plot evidence destination exists: {destination}.")
    parent = destination.parent.resolve(strict=True)
    _ensure_below_root(parent, root, "plot evidence destination parent")
    cursor = root
    for part in destination.parent.relative_to(root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"Destination parent traverses a symlink: {cursor}.")
    return destination


def _existing_directory(value: Any, root: Path, label: str) -> Path:
    raw = Path(str(value)).expanduser()
    lexical = raw if raw.is_absolute() else root / raw
    lexical = Path(os.path.abspath(lexical))
    _ensure_below_root(lexical, root, label)
    cursor = root
    for part in lexical.relative_to(root).parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ValueError(f"{label} traverses a forbidden symlink: {cursor}.")
    resolved = lexical.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} is not a directory: {resolved}.")
    return resolved


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read {label} {path}: {exc}.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}.")
    return payload


def _validate_canonical_document(
    path: Path,
    *,
    digest_field: str,
    label: str,
    require_read_only: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if path.is_symlink():
        raise ValueError(f"{label} cannot be a symlink: {path}.")
    payload = _read_json(path, label)
    declared = payload.get(digest_field)
    actual = _canonical_sha256(payload, digest_field)
    if declared != actual:
        raise ValueError(
            f"{label} canonical digest mismatch: declared={declared!r}, actual={actual}."
        )
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file() or sidecar.is_symlink():
        raise FileNotFoundError(f"{label} canonical sidecar is absent: {sidecar}.")
    if sidecar.read_text(encoding="utf-8").split() != [actual, path.name]:
        raise ValueError(f"{label} canonical sidecar does not authenticate {path}.")
    if require_read_only and (
        path.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
        or sidecar.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH)
    ):
        raise ValueError(f"{label} and its sidecar must be read-only before sealing.")
    return payload, {
        "path": str(path),
        "file_sha256": _sha256_file(path),
        digest_field: actual,
        "sidecar_path": str(sidecar),
        "sidecar_sha256": _sha256_file(sidecar),
    }


def _manifest_binding(path: Path, root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    manifest = read_manifest_for_audit(path, root)
    payload, binding = _validate_canonical_document(
        path, digest_field="manifest_sha256", label="campaign manifest"
    )
    if payload != manifest:
        raise ValueError(f"Manifest changed between authenticated reads: {path}.")
    if manifest.get("benchmark") != BENCHMARK_NAME:
        raise ValueError(f"Manifest has the wrong benchmark identity: {path}.")
    if (
        manifest.get("allows_unvalidated_temporal_pilot") is not False
        or manifest.get("unvalidated_temporal_pilot_jobs") != 0
    ):
        raise ValueError(f"Fresh plotting evidence rejects every pilot route: {path}.")
    return manifest, binding


def _selection_binding(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    return _validate_canonical_document(
        path, digest_field="document_sha256", label="target-blind selection"
    )


def _selection_cohort_snapshot(
    root: Path, *, reader: SelectionCohortReader
) -> tuple[dict[str, Any], tuple[Path, ...]]:
    reopened = reader(root=root)
    required = {
        "status",
        "contract",
        "commit_path",
        "commit_sha256",
        "commit_file_sha256",
        "selection_paths",
        "selection_count",
    }
    if not isinstance(reopened, Mapping) or not required <= set(reopened):
        raise ValueError("Committed selection-cohort reader returned an incomplete snapshot.")
    commit_path = _source_path(reopened["commit_path"], root, "selection-cohort commit")
    commit = _read_json(commit_path, "selection-cohort commit")
    commit_sha = str(reopened["commit_sha256"])
    commit_file_sha = str(reopened["commit_file_sha256"])
    raw_paths = reopened["selection_paths"]
    if not isinstance(raw_paths, list):
        raise ValueError("Committed selection cohort has no exact ordered selection paths.")
    paths = tuple(
        _source_path(path, root, f"selection-cohort member[{index}]")
        for index, path in enumerate(raw_paths)
    )
    if (
        reopened["status"] != "valid"
        or reopened["contract"] != selection_cohort.COHORT_CONTRACT
        or commit.get("commit_sha256") != commit_sha
        or _sha256_file(commit_path) != commit_file_sha
        or reopened["selection_count"] != selection_cohort.EXPECTED_RECORDS
        or len(paths) != selection_cohort.EXPECTED_RECORDS
        or len(set(paths)) != selection_cohort.EXPECTED_RECORDS
        or [str(path) for path in paths] != raw_paths
    ):
        raise ValueError("Committed selection cohort identity/count/path binding drifted.")
    return (
        {
            "contract": selection_cohort.COHORT_CONTRACT,
            "commit_path": str(commit_path),
            "commit_sha256": commit_sha,
            "commit_file_sha256": commit_file_sha,
            "selection_paths": [str(path) for path in paths],
            "selection_count": len(paths),
        },
        paths,
    )


def _validate_raw_ledger(
    name: str, path: Path
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, binding = _validate_canonical_document(
        path,
        digest_field="document_sha256",
        label=f"{name} raw ledger",
        require_read_only=True,
    )
    if set(payload) != _RAW_TOP_LEVEL_KEYS:
        raise ValueError(
            f"{name} raw ledger fields differ from the exact input schema: "
            f"missing={sorted(_RAW_TOP_LEVEL_KEYS - set(payload))}, "
            f"unknown={sorted(set(payload) - _RAW_TOP_LEVEL_KEYS)}."
        )
    if (
        payload.get("schema_version") != 1
        or payload.get("evidence_input") != RAW_LEDGER_INPUT_CONTRACTS[name]
    ):
        raise ValueError(f"{name} raw ledger schema/contract mismatch.")
    _parse_timestamp(payload.get("sealed_at_utc"), f"{name}.sealed_at_utc")
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError(f"{name} raw ledger rows must be a list.")
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _RAW_ROW_KEYS[name]:
            keys = set(row) if isinstance(row, Mapping) else set()
            raise ValueError(
                f"{name} raw row {index} has wrong fields: "
                f"missing={sorted(_RAW_ROW_KEYS[name] - keys)}, "
                f"unknown={sorted(keys - _RAW_ROW_KEYS[name])}."
            )
        condition_id = row.get("condition_id")
        if not isinstance(condition_id, str) or not condition_id or condition_id in seen:
            raise ValueError(f"{name} raw ledger has a missing/duplicate condition_id.")
        seen.add(condition_id)
    return payload, binding


def _registry_index(
    registry_paths: Sequence[Path],
    manifests: Mapping[Path, Mapping[str, Any]],
) -> tuple[dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]], list[dict[str, Any]]]:
    if len(registry_paths) != EXPECTED_REGISTRY_COUNT:
        raise ValueError(
            f"Fresh publication requires exactly {EXPECTED_REGISTRY_COUNT} registries; "
            f"got {len(registry_paths)}."
        )
    if len(set(registry_paths)) != len(registry_paths):
        raise ValueError("Submission registry paths contain a duplicate.")
    manifest_by_sha = {
        str(manifest["manifest_sha256"]): (path, manifest)
        for path, manifest in manifests.items()
    }
    if len(manifest_by_sha) != len(manifests):
        raise ValueError("Campaign manifests reuse a canonical digest.")
    owners: dict[str, Path] = {}
    lookup: dict[tuple[str, int], tuple[dict[str, Any], dict[str, Any]]] = {}
    provenance: list[dict[str, Any]] = []
    for path in registry_paths:
        if path.is_symlink():
            raise ValueError(f"Submission registry cannot be a symlink: {path}.")
        registry = read_submission_registry(path)
        manifest_sha = str(registry.get("manifest_sha256", ""))
        if manifest_sha not in manifest_by_sha or manifest_sha in owners:
            raise ValueError(
                "Every campaign manifest must have exactly one distinct submission registry."
            )
        manifest_path, manifest = manifest_by_sha[manifest_sha]
        if Path(str(registry.get("manifest_path", ""))).resolve() != manifest_path:
            raise ValueError(f"Registry {path} points to a stale/different manifest path.")
        jobs = manifest["jobs"]
        submissions = registry.get("submissions")
        expected_indices = list(range(len(jobs)))
        observed_indices = [entry.get("job_index") for entry in submissions]
        if observed_indices != expected_indices:
            raise ValueError(
                f"Registry {path} must cover every manifest index exactly once in order."
            )
        owners[manifest_sha] = path
        file_sha = _sha256_file(path)
        record = {
            "path": str(path),
            "file_sha256": file_sha,
            "registry_sha256": registry["registry_sha256"],
            "manifest_path": str(manifest_path),
            "manifest_sha256": manifest_sha,
            "slurm_array_job_id": registry["slurm_array_job_id"],
            "num_registered_tasks": len(expected_indices),
        }
        provenance.append(record)
        for entry in submissions:
            index = int(entry["job_index"])
            lookup[(manifest_sha, index)] = (dict(entry), record)
    if set(owners) != set(manifest_by_sha):
        raise ValueError("Submission registries omit one or more campaign manifests.")
    return lookup, sorted(provenance, key=lambda row: row["path"])


def _validate_execution_identity(
    output_dir: Path,
    *,
    registry_entry: Mapping[str, Any],
    registry_record: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    path = output_dir / EXECUTION_IDENTITY_FILENAME
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(f"Attempt execution identity is absent: {path}.")
    identity = read_execution_identity(path)
    expected_job_id = str(registry_entry["slurm_array_job_id"])
    expected_index = str(registry_entry["slurm_array_task_id"])
    if (
        identity.get("SLURM_ARRAY_JOB_ID") != expected_job_id
        or identity.get("SLURM_ARRAY_TASK_ID") != expected_index
        or identity.get("slurm_task_id") != registry_entry["slurm_task_id"]
    ):
        raise ValueError(f"Execution identity differs from registry task: {path}.")
    _parse_timestamp(identity.get("captured_at_utc"), f"execution identity {path}")
    return path, {
        "path": str(path),
        "sha256": _sha256_file(path),
        "slurm_task_id": registry_entry["slurm_task_id"],
        "registry_path": registry_record["path"],
        "registry_sha256": registry_record["registry_sha256"],
    }


def _stable_record(path: Path) -> tuple[int, int, int, int, str]:
    digest = hashlib.sha256()
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    with os.fdopen(descriptor, "rb") as handle:
        metadata = os.fstat(handle.fileno())
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"Stable evidence source is not a regular file: {path}.")
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return (
        metadata.st_dev,
        metadata.st_ino,
        metadata.st_size,
        metadata.st_mtime_ns,
        digest.hexdigest(),
    )


def _derive_cohorts(
    *,
    root: Path,
    manifest_paths_by_cohort: Mapping[str, Sequence[Path]],
    registry_lookup: Mapping[
        tuple[str, int], tuple[dict[str, Any], dict[str, Any]]
    ],
    manifests: Mapping[Path, Mapping[str, Any]],
    media_validator: MediaValidator,
    require_flux1_v3_completed_audit: bool,
) -> tuple[dict[str, Any], dict[str, dict[str, Any]], dict[Path, tuple[int, int, int, int, str]]]:
    from scripts import plot_finer_detailing_correction as plotter

    descriptors: dict[str, Any] = {}
    by_condition: dict[str, dict[str, Any]] = {}
    stable: dict[Path, tuple[int, int, int, int, str]] = {}
    flux1_reopened_by_cohort = {"qualification": 0, "seed_ladder": 0, "final": 0}
    for cohort in ("qualification", "seed_ladder", "final"):
        manifest_bindings: list[dict[str, Any]] = []
        row_bindings: list[dict[str, Any]] = []
        for manifest_path in manifest_paths_by_cohort[cohort]:
            manifest = manifests[manifest_path]
            _, manifest_binding = _validate_canonical_document(
                manifest_path,
                digest_field="manifest_sha256",
                label=f"{cohort} manifest",
            )
            manifest_bindings.append(manifest_binding)
            stable[manifest_path] = _stable_record(manifest_path)
            stable[manifest_path.with_suffix(manifest_path.suffix + ".sha256")] = (
                _stable_record(manifest_path.with_suffix(manifest_path.suffix + ".sha256"))
            )
            manifest_sha = str(manifest["manifest_sha256"])
            for index, job in enumerate(manifest["jobs"]):
                if not isinstance(job, Mapping):
                    raise ValueError(f"Manifest job {manifest_path}[{index}] is not a mapping.")
                condition_id = str(job.get("condition_id", ""))
                evidence_condition_id = _cohort_condition_id(cohort, condition_id)
                if evidence_condition_id in by_condition:
                    raise ValueError("Campaign condition IDs are missing or reused within a cohort.")
                output_dir = _existing_directory(
                    job.get("output_dir"), root, "campaign output directory"
                )
                entry, registry = registry_lookup[(manifest_sha, index)]
                identity_path, identity_binding = _validate_execution_identity(
                    output_dir, registry_entry=entry, registry_record=registry
                )
                result_path = output_dir / "benchmark_job_result.json"
                timing_path = output_dir / "experiment_timing.json"
                for path, label in (
                    (result_path, "benchmark result"),
                    (timing_path, "experiment timing"),
                ):
                    if not path.is_file() or path.is_symlink():
                        raise FileNotFoundError(f"{label} is absent: {path}.")
                result_sha = _sha256_file(result_path)
                timing_sha = _sha256_file(timing_path)
                result = _read_json(result_path, "benchmark result")
                timing = _read_json(timing_path, "experiment timing")
                finished_at = _parse_timestamp(
                    timing.get("finished_at_utc"), f"experiment timing {timing_path}"
                )
                expected_job = dict(job)
                expected_job["launch_manifest_sha256"] = manifest_sha
                expected_job["launch_manifest_job_index"] = index
                expected_status = (
                    "completed" if job.get("expected_media") is True else "not_supported"
                )
                if (
                    result.get("schema_version") != 2
                    or result.get("status") != expected_status
                    or result.get("job") != expected_job
                ):
                    raise ValueError(
                        "Result is not an exact launch-bound schema-2 terminal result: "
                        f"{result_path}."
                    )
                if job.get("expected_media") is not True:
                    expected_unsupported = {
                        "schema_version": 2,
                        "status": "not_supported",
                        "job": expected_job,
                        "result": None,
                        "reason": job["variant_spec"]["reason"],
                        "media_validation": None,
                        "validated_media_paths": [],
                    }
                    if result != expected_unsupported:
                        raise ValueError(
                            "Unsupported plot row is not the exact runner-produced result."
                        )
                if is_flux1_job_v3(expected_job):
                    reopened = reopen_completed_flux1_output_v3(
                        expected_job,
                        root=root,
                        manifest_path=manifest_path,
                        manifest_sha256=manifest_sha,
                        manifest_job_index=index,
                        result_path=result_path,
                    )
                    if (
                        reopened["result"] != result
                        or reopened["result_sha256"] != result_sha
                    ):
                        raise ValueError(
                            "Plot FLUX-v3 result/hash changed during strict reopening."
                        )
                    flux1_reopened_by_cohort[cohort] += 1
                if job.get("expected_media") is True:
                    fresh_media_validation = media_validator(output_dir, expected_job, True)
                    if result.get("media_validation") != fresh_media_validation:
                        raise ValueError(
                            "Result media receipt differs from a fresh decode/probe: "
                            f"{result_path}."
                        )
                row_binding = {
                    "manifest_sha256": manifest_sha,
                    "manifest_job_index": index,
                    "result": {"path": str(result_path), "sha256": result_sha},
                    "timing": {"path": str(timing_path), "sha256": timing_sha},
                    "submission_registry": deepcopy(registry),
                    "execution_identity": identity_binding,
                }
                row_bindings.append(row_binding)
                family, method, _ablation, _active = plotter._condition_family(job)
                by_condition[evidence_condition_id] = {
                    "evidence_condition_id": evidence_condition_id,
                    "condition_id": condition_id,
                    "cohort": cohort,
                    "manifest_sha256": manifest_sha,
                    "manifest_job_index": index,
                    "result_sha256": result_sha,
                    "expected_media": job.get("expected_media") is True,
                    "task": str((job.get("generation") or {}).get("task", "")),
                    "method": method,
                    "family": family,
                    "job": job,
                    "finished_at": finished_at,
                }
                stable[result_path] = _stable_record(result_path)
                stable[timing_path] = _stable_record(timing_path)
                stable[identity_path] = _stable_record(identity_path)
                if job.get("expected_media") is True:
                    media = output_dir / (
                        "sample_0000/image_000.png"
                        if by_condition[evidence_condition_id]["task"] == "text_to_image"
                        else "sample_0000/video_000.mp4"
                    )
                    media = _source_path(media, root, "generated media")
                    stable[media] = _stable_record(media)
        if len(row_bindings) != EXPECTED_LOGICAL_COUNTS[cohort]:
            raise ValueError(
                f"{cohort} contains {len(row_bindings)} logical rows; "
                f"expected {EXPECTED_LOGICAL_COUNTS[cohort]}."
            )
        descriptors[cohort] = {
            "expected_counts": plotter.EXPECTED_COHORT_COUNTS[cohort],
            "manifests": manifest_bindings,
            "rows": row_bindings,
        }
    expected_flux1 = {"qualification": 18, "seed_ladder": 24, "final": 42}
    if require_flux1_v3_completed_audit and (
        flux1_reopened_by_cohort != expected_flux1
        or sum(flux1_reopened_by_cohort.values()) != 84
    ):
        raise ValueError(
            "Plot completed-output audit must reopen qualification=18, ladder=24, "
            f"final=42 FLUX-v3 rows; found {flux1_reopened_by_cohort}."
        )
    return descriptors, by_condition, stable


def _expected_raw_conditions(name: str, rows: Mapping[str, Mapping[str, Any]]) -> set[str]:
    if name == "manual_semantic":
        return {condition for condition, row in rows.items() if row["expected_media"]}
    if name == "objective_temporal":
        return {
            condition
            for condition, row in rows.items()
            if row["expected_media"] and row["task"] == "text_to_video"
        }
    return {
        condition
        for condition, row in rows.items()
        if row["expected_media"] and row["method"] == "shapley"
    }


def _normalized_ledger(
    *,
    name: str,
    raw: Mapping[str, Any],
    raw_binding: Mapping[str, Any],
    campaign_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    raw_rows = {str(row["condition_id"]): row for row in raw["rows"]}
    expected = _expected_raw_conditions(name, campaign_rows)
    if set(raw_rows) != expected:
        raise ValueError(
            f"{name} raw coverage differs from completed campaign evidence: "
            f"missing={sorted(expected - set(raw_rows))[:8]}, "
            f"extra={sorted(set(raw_rows) - expected)[:8]}."
        )
    sealed_at = _parse_timestamp(raw.get("sealed_at_utc"), f"{name}.sealed_at_utc")
    latest_generation = max(campaign_rows[condition]["finished_at"] for condition in expected)
    if sealed_at < latest_generation:
        raise ValueError(
            f"{name} raw ledger predates campaign evidence: "
            f"sealed={sealed_at.isoformat()}, latest_generation={latest_generation.isoformat()}."
        )
    source_sha = str(raw_binding["file_sha256"])
    rows: list[dict[str, Any]] = []
    for condition_id in sorted(expected):
        source = campaign_rows[condition_id]
        raw_row = raw_rows[condition_id]
        normalized = {
            "cohort": source["cohort"],
            "manifest_sha256": source["manifest_sha256"],
            "manifest_job_index": source["manifest_job_index"],
            "result_sha256": source["result_sha256"],
            "source_sha256": source_sha,
        }
        normalized.update(
            {key: deepcopy(value) for key, value in raw_row.items() if key != "condition_id"}
        )
        rows.append(normalized)
    return {
        "schema_version": 1,
        "evidence": OUTPUT_LEDGER_CONTRACTS[name],
        "raw_source_document": deepcopy(dict(raw_binding)),
        "source_artifacts": [
            {"path": raw_binding["path"], "sha256": raw_binding["file_sha256"]}
        ],
        "rows": rows,
    }


def _write_document(
    stage_path: Path, payload: Mapping[str, Any], *, publication_root: Path
) -> dict[str, Any]:
    document = deepcopy(dict(payload))
    document["document_sha256"] = _canonical_sha256(document, "document_sha256")
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    stage_path.parent.mkdir(parents=True, exist_ok=True)
    stage_path.write_bytes(encoded)
    canonical_sidecar = stage_path.with_suffix(stage_path.suffix + ".sha256")
    canonical_sidecar.write_text(
        f"{document['document_sha256']}  {stage_path.name}\n", encoding="utf-8"
    )
    raw_sha = hashlib.sha256(encoded).hexdigest()
    raw_sidecar = stage_path.with_suffix(stage_path.suffix + ".raw.sha256")
    raw_sidecar.write_text(f"{raw_sha}  {stage_path.name}\n", encoding="utf-8")
    return {
        "path": str(stage_path.relative_to(publication_root)),
        "file_sha256": raw_sha,
        "document_sha256": document["document_sha256"],
        "raw_sidecar_path": str(raw_sidecar.relative_to(publication_root)),
        "raw_sidecar_sha256": _sha256_file(raw_sidecar),
    }


def _publish_directory_commit_last(staging: Path, destination: Path) -> None:
    try:
        publish_hardlink_tree_commit_last(
            staging,
            destination,
            commit_relative_path=Path("fresh_bundle.json"),
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"Immutable plot evidence bundle exists: {destination}."
        ) from exc


def _default_plot_loader(bundle_path: Path, root: Path) -> Any:
    from scripts.plot_finer_detailing_correction import load_fresh_plot_data

    return load_fresh_plot_data(bundle_path, root)


def _seal_authenticated_sources(
    *,
    root: Path,
    destination: Path,
    manifest_paths_by_cohort: Mapping[str, Sequence[Path]],
    registry_paths: Sequence[Path],
    raw_ledger_paths: Mapping[str, Path],
    selection_cohort_reader: SelectionCohortReader = selection_cohort.read_selection_cohort,
    plot_loader: PlotLoader = _default_plot_loader,
    media_validator: MediaValidator = validate_exact_media,
    require_flux1_v3_completed_audit: bool = False,
) -> dict[str, Any]:
    """Assemble already topology-authenticated sources and publish once.

    Production callers must enter through :func:`seal_fresh_plot_evidence`,
    which runs the qualification and campaign validators before this private
    assembly boundary.
    """

    if set(manifest_paths_by_cohort) != set(EXPECTED_MANIFEST_COUNTS):
        raise ValueError("Source cohorts must be exactly qualification, seed_ladder, and final.")
    for cohort, expected in EXPECTED_MANIFEST_COUNTS.items():
        paths = tuple(manifest_paths_by_cohort[cohort])
        if len(paths) != expected or len(set(paths)) != expected:
            raise ValueError(f"{cohort} must bind exactly {expected} distinct manifests.")
    all_manifest_paths = tuple(
        path
        for cohort in ("qualification", "seed_ladder", "final")
        for path in manifest_paths_by_cohort[cohort]
    )
    if len(set(all_manifest_paths)) != EXPECTED_REGISTRY_COUNT:
        raise ValueError("Campaign cohorts reuse one or more manifest paths.")

    manifests: dict[Path, Mapping[str, Any]] = {}
    for path in all_manifest_paths:
        manifest, _ = _manifest_binding(path, root)
        manifests[path] = manifest
    registry_lookup, registry_provenance = _registry_index(registry_paths, manifests)
    descriptors, campaign_rows, stable = _derive_cohorts(
        root=root,
        manifest_paths_by_cohort=manifest_paths_by_cohort,
        registry_lookup=registry_lookup,
        manifests=manifests,
        media_validator=media_validator,
        require_flux1_v3_completed_audit=require_flux1_v3_completed_audit,
    )
    for path in registry_paths:
        stable[path] = _stable_record(path)

    cohort_snapshot, selection_paths = _selection_cohort_snapshot(
        root, reader=selection_cohort_reader
    )
    if len(selection_paths) != campaign.EXPECTED_SELECTION_RECORDS:
        raise ValueError("Fresh publication requires the exact committed 36-selection cohort.")
    commit_path = Path(cohort_snapshot["commit_path"])
    stable[commit_path] = _stable_record(commit_path)
    selection_bindings: list[dict[str, Any]] = []
    for path in selection_paths:
        _, binding = _selection_binding(path)
        selection_bindings.append(binding)
        stable[path] = _stable_record(path)
        sidecar = path.with_suffix(path.suffix + ".sha256")
        stable[sidecar] = _stable_record(sidecar)

    if set(raw_ledger_paths) != set(RAW_LEDGER_INPUT_CONTRACTS):
        raise ValueError("Exactly three domain-separated raw ledgers are required.")
    raw_documents: dict[str, dict[str, Any]] = {}
    raw_bindings: dict[str, dict[str, Any]] = {}
    for name, path in raw_ledger_paths.items():
        raw_documents[name], raw_bindings[name] = _validate_raw_ledger(name, path)
        stable[path] = _stable_record(path)
        sidecar = path.with_suffix(path.suffix + ".sha256")
        stable[sidecar] = _stable_record(sidecar)

    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    stage_stat = staging.lstat()
    stage_identity = (stage_stat.st_dev, stage_stat.st_ino)
    try:
        ledger_bindings: dict[str, dict[str, Any]] = {}
        for name in RAW_LEDGER_INPUT_CONTRACTS:
            normalized = _normalized_ledger(
                name=name,
                raw=raw_documents[name],
                raw_binding=raw_bindings[name],
                campaign_rows=campaign_rows,
            )
            ledger_bindings[name] = _write_document(
                staging / "ledgers" / f"{name}.json",
                normalized,
                publication_root=staging,
            )
        bundle = {
            "schema_version": 1,
            "contract": "finer_detailing_plot_evidence_bundle_v1",
            "publication_contract": PUBLICATION_CONTRACT,
            "cohorts": descriptors,
            "selection_cohort_commit": cohort_snapshot,
            "seed_selections": sorted(selection_bindings, key=lambda row: row["path"]),
            "submission_registries": registry_provenance,
            "raw_ledger_sources": raw_bindings,
            "evidence_ledgers": ledger_bindings,
        }
        bundle_binding = _write_document(
            staging / "fresh_bundle.json", bundle, publication_root=staging
        )
        bundle_path = staging / "fresh_bundle.json"

        freeze_tree(staging, label="Fresh plot evidence staging")
        reconciled = plot_loader(bundle_path, staging)
        snapshot = reconciled.snapshot()
        if snapshot["cohorts"] != {
            "qualification": {"logical": 396, "media": 369, "unsupported": 27},
            "seed_ladder": {"logical": 288, "media": 288, "unsupported": 0},
            "final": {"logical": 504, "media": 489, "unsupported": 15},
        } or snapshot["seed_selection_count"] != 36:
            raise RuntimeError(f"Staged plot evidence snapshot is incomplete: {snapshot}.")

        changed = [str(path) for path, record in stable.items() if _stable_record(path) != record]
        if changed:
            raise ValueError(
                "Campaign evidence changed while the plotting bundle was assembled: "
                f"{changed[:8]}."
            )
        reopened_cohort, reopened_paths = _selection_cohort_snapshot(
            root, reader=selection_cohort_reader
        )
        if reopened_cohort != cohort_snapshot or reopened_paths != selection_paths:
            raise ValueError("Selection cohort changed while plot evidence was assembled.")
        _publish_directory_commit_last(staging, destination)
        return {
            "status": "published",
            "contract": PUBLICATION_CONTRACT,
            "destination": str(destination),
            "bundle_path": str(destination / "fresh_bundle.json"),
            "bundle_document_sha256": bundle_binding["document_sha256"],
            "bundle_file_sha256": bundle_binding["file_sha256"],
            "manifest_counts": dict(EXPECTED_MANIFEST_COUNTS),
            "registry_count": len(registry_paths),
            "selection_count": len(selection_paths),
            "selection_cohort_commit": cohort_snapshot,
            "plot_data_snapshot": snapshot,
        }
    finally:
        cleanup_owned_staging(staging, stage_identity)


def seal_fresh_plot_evidence(
    *,
    root: str | Path,
    qualification_q2_plan_path: str | Path,
    submission_registry_paths: Sequence[str | Path],
    objective_temporal_raw_path: str | Path,
    manual_semantic_raw_path: str | Path,
    shapley_diagnostics_raw_path: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    """Validate the complete production campaign and atomically seal plot inputs."""

    resolved_root = Path(root).expanduser().resolve(strict=True)
    destination_path = _destination_path(destination, resolved_root)
    q2_path = _source_path(
        qualification_q2_plan_path, resolved_root, "Q2 qualification plan"
    )
    # Plot sealing is necessarily downstream of the atomic five-model temporal
    # promotion.  The live model YAMLs therefore differ deliberately from the
    # pilot-era Q1/Q2 snapshots.  Reopen those historical cohorts through the
    # audit reader, which accepts that protocol-input promotion while still
    # requiring exact frozen implementation bytes and Git provenance.
    q2 = qualification.validate_qualification_plan_for_audit(
        q2_path, root=resolved_root, expected_phase=qualification.Q2
    )
    if q2.upstream_q1 is None:
        raise ValueError("Q2 qualification plan lacks its authenticated Q1 cohort.")
    qualification_flux1_receipt = (
        qualification.derive_completed_flux1_qualification_union_receipt_v3(
            q2, root=resolved_root
        )
    )
    if (
        qualification_flux1_receipt.get("expected_rows") != 18
        or qualification_flux1_receipt.get("phase_counts") != {"q1": 15, "q2": 3}
    ):
        raise ValueError("Plot sealing lacks the exact completed 18-row FLUX-v3 qualification union.")
    qualification_paths = tuple(
        _source_path(binding["path"], resolved_root, "qualification manifest")
        for validated in (q2.upstream_q1, q2)
        for binding in validated.plan["manifest_bindings"]
    )
    ladder_paths = tuple(
        _source_path(
            resolved_root
            / campaign.LADDER_MANIFEST_ROOT_RELATIVE
            / f"seed_{seed:08d}.json",
            resolved_root,
            "seed-ladder manifest",
        )
        for seed in campaign.SEEDS
    )
    final_paths = tuple(
        _source_path(
            resolved_root
            / campaign.FINAL_MANIFEST_ROOT_RELATIVE
            / f"{prompt_id}__{model_name}__{family}.json",
            resolved_root,
            "final manifest",
        )
        for prompt_id, model_name in campaign.AXES
        for family in campaign.FINAL_FAMILIES
    )
    _selection_snapshot, selection_paths = _selection_cohort_snapshot(
        resolved_root, reader=selection_cohort.read_selection_cohort
    )

    campaign_result = campaign.validate_production_campaign(
        ladder_paths,
        selection_paths,
        final_paths,
        root=resolved_root,
    )
    if campaign_result.get("status") != "production_validated":
        raise ValueError(f"Campaign is not production validated: {campaign_result!r}.")
    completed_flux1_campaign = campaign.validate_completed_flux1_production_campaign_v3(
        ladder_paths,
        final_paths,
        root=resolved_root,
    )
    if completed_flux1_campaign.get("counts") != {
        "seed_ladder": 24,
        "final": 42,
    }:
        raise ValueError("Plot sealing lacks the exact completed 66-row FLUX-v3 campaign.")

    registry_paths = tuple(
        _source_path(path, resolved_root, "submission registry")
        for path in submission_registry_paths
    )
    raw_paths = {
        "objective_temporal": _source_path(
            objective_temporal_raw_path, resolved_root, "objective temporal raw ledger"
        ),
        "manual_semantic": _source_path(
            manual_semantic_raw_path, resolved_root, "manual semantic raw ledger"
        ),
        "shapley_diagnostics": _source_path(
            shapley_diagnostics_raw_path, resolved_root, "Shapley diagnostics raw ledger"
        ),
    }
    return _seal_authenticated_sources(
        root=resolved_root,
        destination=destination_path,
        manifest_paths_by_cohort={
            "qualification": qualification_paths,
            "seed_ladder": ladder_paths,
            "final": final_paths,
        },
        registry_paths=registry_paths,
        raw_ledger_paths=raw_paths,
        require_flux1_v3_completed_audit=True,
    )


__all__ = [
    "EXPECTED_MANIFEST_COUNTS",
    "EXPECTED_REGISTRY_COUNT",
    "PUBLICATION_CONTRACT",
    "RAW_LEDGER_INPUT_CONTRACTS",
    "seal_fresh_plot_evidence",
]
