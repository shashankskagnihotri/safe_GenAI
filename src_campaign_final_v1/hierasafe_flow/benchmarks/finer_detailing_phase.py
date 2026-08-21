"""Fail-closed, resumable Slurm phase orchestration for finer detailing.

The phase plan is an immutable, content-addressed description of already-frozen
benchmark manifests.  This module validates the complete selected-job union
before the first submission, submits one array at a time, immediately creates
the existing immutable submission registry, and durably checkpoints every
state transition.  It never guesses whether an ambiguous ``sbatch`` call
succeeded: that state requires operator reconciliation and cannot be replayed.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import stat
import subprocess
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hierasafe_flow.benchmarks.finer_detailing_correction import (
    BENCHMARK_NAME,
    flux_common_seed_module_for_stage,
    flux_common_seed_stage_from_manifest,
    project_root,
    read_manifest,
)
from hierasafe_flow.benchmarks.slurm_tracking import (
    build_submission_registry,
    parse_array_spec,
    read_submission_registry,
    write_submission_registry,
)


PHASE_PLAN_SCHEMA_VERSION = 1
PHASE_STATE_SCHEMA_VERSION = 1
REQUIRED_PILOT_ENV_BY_MODEL: dict[str, dict[str, str]] = {
    "wan22_t2v_a14b": {"HIERASAFE_WAN_TEMPORAL_PILOT": "1"},
}
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ENV_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
_PHASE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_SUBMISSION_ID_RE = _PHASE_ID_RE
_SBATCH_RESULT_RE = re.compile(r"^(?P<job_id>[0-9]+)(?:;[A-Za-z0-9._-]+)?$")
_PLAN_KEYS = {
    "schema_version",
    "benchmark",
    "phase_id",
    "created_at_utc",
    "state_path",
    "submissions",
    "phase_plan_sha256",
}
_SUBMISSION_KEYS = {
    "submission_id",
    "manifest_path",
    "manifest_sha256",
    "launcher_path",
    "launcher_sha256",
    "array_spec",
    "registry_path",
    "slurm_job_name",
    "exported_environment",
    "pilot_environment",
}
_STATE_ENTRY_CORE_KEYS = (
    "submission_id",
    "manifest_path",
    "manifest_sha256",
    "array_spec",
    "registry_path",
)


class PhaseReplayError(RuntimeError):
    """Raised when a completed or potentially submitted phase would be replayed."""


class PhaseStateError(RuntimeError):
    """Raised when durable phase state cannot be safely resumed."""


@dataclass(frozen=True)
class ValidatedSubmission:
    submission_id: str
    manifest_path: Path
    manifest_sha256: str
    manifest: dict[str, Any]
    launcher_path: Path
    launcher_sha256: str
    array_spec: str
    indices: tuple[int, ...]
    registry_path: Path
    slurm_job_name: str
    exported_environment: dict[str, str]
    pilot_environment: dict[str, str]

    @property
    def merged_environment(self) -> dict[str, str]:
        return {**self.exported_environment, **self.pilot_environment}


@dataclass(frozen=True)
class ValidatedPhase:
    plan_path: Path
    plan: dict[str, Any]
    state_path: Path
    submissions: tuple[ValidatedSubmission, ...]
    union_proof: dict[str, Any]

    @property
    def phase_plan_sha256(self) -> str:
        return str(self.plan["phase_plan_sha256"])


ManifestReader = Callable[[Path, Path], dict[str, Any]]
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
RegistryWriter = Callable[[Mapping[str, Any], Path], None]


def phase_plan_digest(plan: Mapping[str, Any]) -> str:
    """Return the canonical digest of a phase plan, excluding its identity field."""

    canonical = dict(plan)
    canonical.pop("phase_plan_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def phase_state_digest(state: Mapping[str, Any]) -> str:
    """Return the canonical digest of a durable phase state/receipt."""

    canonical = dict(state)
    canonical.pop("phase_state_sha256", None)
    encoded = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_phase_plan_immutable(plan: Mapping[str, Any], path: Path) -> dict[str, Any]:
    """Publish a plan and SHA sidecar exactly once.

    ``plan`` must omit ``phase_plan_sha256`` or contain the correct value.  The
    sidecar is published first and the JSON plan is the final commit marker.
    Neither existing artifact is ever overwritten.
    """

    frozen = deepcopy(dict(plan))
    expected = phase_plan_digest(frozen)
    supplied = frozen.get("phase_plan_sha256")
    if supplied is not None and supplied != expected:
        raise ValueError("Phase plan digest is inconsistent before publication.")
    frozen["phase_plan_sha256"] = expected
    _validate_plan_shape(frozen)
    # Resolve only the parent.  Resolving the complete destination would follow
    # a dangling symlink at the commit-marker name and could publish somewhere
    # other than the path the caller asked us to freeze.
    requested_path = path.expanduser()
    if requested_path.is_symlink():
        raise ValueError(f"Refusing to publish a phase plan through a symlink: {requested_path}")
    path = requested_path.parent.resolve(strict=False) / requested_path.name
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.is_symlink() or sidecar.is_symlink():
        raise ValueError("Refusing to publish phase-plan artifacts through symlinks.")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite immutable phase plan: {path}")
    sidecar_published = False
    try:
        _atomic_write_new_text(sidecar, f"{expected}  {path.name}\n", mode=0o444)
        sidecar_published = True
        _atomic_write_new_text(
            path,
            json.dumps(frozen, indent=2, sort_keys=True) + "\n",
            mode=0o444,
        )
    except Exception:
        if sidecar_published and not path.exists():
            sidecar.unlink(missing_ok=True)
        raise
    return frozen


def read_phase_plan(path: Path, root: Path | None = None) -> dict[str, Any]:
    """Read and authenticate one immutable phase plan and its required sidecar."""

    root = (root or project_root()).resolve()
    requested_path = path.expanduser()
    requested_path = requested_path if requested_path.is_absolute() else root / requested_path
    if requested_path.is_symlink():
        raise ValueError(f"Immutable phase plan must not be a symlink: {requested_path}")
    path = _resolve_under_root(path, root, "phase plan")
    try:
        payload = json.loads(_read_readonly_regular_text(path, "phase plan"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Cannot read phase plan {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Phase plan {path} must contain a JSON object.")
    _validate_plan_shape(payload)
    actual = phase_plan_digest(payload)
    if payload.get("phase_plan_sha256") != actual:
        raise ValueError(
            f"Phase plan digest mismatch in {path}: "
            f"expected={payload.get('phase_plan_sha256')}, actual={actual}."
        )
    sidecar = path.with_suffix(path.suffix + ".sha256")
    fields = _read_readonly_regular_text(sidecar, "phase plan sidecar").split()
    if len(fields) != 2 or fields[0] != actual or fields[1] != path.name:
        raise ValueError(f"Phase plan sidecar does not authenticate {path}: {sidecar}")
    return payload


def validate_phase_plan(
    plan_path: Path,
    *,
    root: Path | None = None,
    manifest_reader: ManifestReader = read_manifest,
) -> ValidatedPhase:
    """Validate every launch input and prove the selected-job union is disjoint."""

    root = (root or project_root()).resolve()
    # Authenticate the caller-supplied leaf before canonicalization so a
    # symlink alias cannot be laundered into an apparently regular plan path.
    plan = read_phase_plan(plan_path, root)
    plan_path = _resolve_under_root(plan_path, root, "phase plan")
    state_path = _resolve_under_root(Path(str(plan["state_path"])), root, "phase state")
    if state_path in {plan_path, plan_path.with_suffix(plan_path.suffix + ".sha256")}:
        raise ValueError("Phase state path must differ from the immutable plan artifacts.")

    validated: list[ValidatedSubmission] = []
    submission_ids: set[str] = set()
    registry_paths: set[Path] = set()
    selected_jobs: set[tuple[str, int]] = set()
    condition_owners: dict[str, str] = {}
    output_owners: dict[str, str] = {}
    union_rows: list[dict[str, Any]] = []

    for position, raw in enumerate(plan["submissions"]):
        submission_id = str(raw["submission_id"])
        if submission_id in submission_ids:
            raise ValueError(f"Duplicate phase submission_id: {submission_id!r}.")
        submission_ids.add(submission_id)
        manifest_path = _resolve_under_root(
            Path(str(raw["manifest_path"])), root, f"submission {submission_id} manifest"
        )
        manifest = manifest_reader(manifest_path, root)
        expected_manifest_sha = str(raw["manifest_sha256"])
        if manifest.get("manifest_sha256") != expected_manifest_sha:
            raise ValueError(
                f"Submission {submission_id} manifest digest mismatch: "
                f"plan={expected_manifest_sha}, manifest={manifest.get('manifest_sha256')}."
            )
        jobs = manifest.get("jobs")
        if not isinstance(jobs, list):
            raise ValueError(f"Submission {submission_id} manifest jobs are not a list.")
        array_spec = str(raw["array_spec"])
        indices = tuple(parse_array_spec(array_spec))
        out_of_range = [index for index in indices if index >= len(jobs)]
        if out_of_range:
            raise ValueError(
                f"Submission {submission_id} array exceeds manifest range 0..{len(jobs) - 1}: "
                f"{out_of_range}."
            )

        launcher_path = _resolve_under_root(
            Path(str(raw["launcher_path"])), root, f"submission {submission_id} launcher"
        )
        if not launcher_path.is_file():
            raise FileNotFoundError(
                f"Submission {submission_id} launcher is missing: {launcher_path}"
            )
        launcher_sha = _sha256_file(launcher_path)
        if launcher_sha != raw["launcher_sha256"]:
            raise ValueError(
                f"Submission {submission_id} launcher digest mismatch: "
                f"plan={raw['launcher_sha256']}, actual={launcher_sha}."
            )
        registry_path = _resolve_under_root(
            Path(str(raw["registry_path"])), root, f"submission {submission_id} registry"
        )
        if registry_path in registry_paths:
            raise ValueError(f"Phase reuses submission registry path: {registry_path}")
        if registry_path in {plan_path, state_path}:
            raise ValueError(
                f"Submission registry path collides with a phase artifact: {registry_path}"
            )
        registry_paths.add(registry_path)

        exported_environment = _validate_environment_mapping(
            raw["exported_environment"], f"submission {submission_id} exported_environment"
        )
        pilot_environment = _validate_environment_mapping(
            raw["pilot_environment"], f"submission {submission_id} pilot_environment"
        )
        duplicate_env = sorted(set(exported_environment) & set(pilot_environment))
        if duplicate_env:
            raise ValueError(
                f"Submission {submission_id} repeats variables across environment mappings: "
                f"{duplicate_env}."
            )
        manifest_env = exported_environment.get("FINER_DETAILING_MANIFEST")
        if (
            manifest_env is None
            or _resolve_under_root(
                Path(manifest_env), root, f"submission {submission_id} FINER_DETAILING_MANIFEST"
            )
            != manifest_path
        ):
            raise ValueError(
                f"Submission {submission_id} must explicitly export "
                "FINER_DETAILING_MANIFEST resolving to its frozen manifest."
            )
        exported_environment["FINER_DETAILING_MANIFEST"] = str(manifest_path)
        if any("PILOT" in key.upper() for key in exported_environment):
            raise ValueError(
                f"Submission {submission_id} must place all pilot variables in pilot_environment."
            )
        if any("PILOT" not in key.upper() for key in pilot_environment):
            raise ValueError(
                f"Submission {submission_id} pilot_environment contains a non-pilot variable."
            )

        selected = [jobs[index] for index in indices]
        required_pilot_environment = _required_pilot_environment(selected)
        if pilot_environment != required_pilot_environment:
            raise ValueError(
                f"Submission {submission_id} pilot environment does not exactly match selected "
                f"pilot jobs: expected={required_pilot_environment}, got={pilot_environment}."
            )
        for index, job in zip(indices, selected, strict=True):
            key = (expected_manifest_sha, index)
            owner = f"{submission_id}[{index}]"
            if key in selected_jobs:
                raise ValueError(
                    f"Manifest job {expected_manifest_sha}:{index} appears in more than one array."
                )
            selected_jobs.add(key)
            condition_id = str(job.get("condition_id", "")).strip()
            if not condition_id:
                raise ValueError(f"Selected job {owner} has no condition_id.")
            if condition_id in condition_owners:
                raise ValueError(
                    f"Condition-id union overlap for {condition_id!r}: "
                    f"{condition_owners[condition_id]} and {owner}."
                )
            condition_owners[condition_id] = owner
            raw_output_dir = str(job.get("output_dir", "")).strip()
            if not raw_output_dir:
                raise ValueError(f"Selected job {owner} has no output_dir.")
            output_dir = _resolve_under_root(
                Path(raw_output_dir), root, f"selected job {owner} output_dir"
            )
            output_key = str(output_dir)
            if output_key in output_owners:
                raise ValueError(
                    f"Output-dir union overlap for {output_key}: "
                    f"{output_owners[output_key]} and {owner}."
                )
            output_owners[output_key] = owner
            union_rows.append(
                {
                    "submission_id": submission_id,
                    "manifest_sha256": expected_manifest_sha,
                    "job_index": index,
                    "condition_id": condition_id,
                    "output_dir": output_key,
                }
            )

        validated.append(
            ValidatedSubmission(
                submission_id=submission_id,
                manifest_path=manifest_path,
                manifest_sha256=expected_manifest_sha,
                manifest=manifest,
                launcher_path=launcher_path,
                launcher_sha256=launcher_sha,
                array_spec=array_spec,
                indices=indices,
                registry_path=registry_path,
                slurm_job_name=str(raw["slurm_job_name"]),
                exported_environment=exported_environment,
                pilot_environment=pilot_environment,
            )
        )

    common_seed_stage_by_submission = [
        flux_common_seed_stage_from_manifest(submission.manifest)
        for submission in validated
    ]
    common_seed_stages = {
        stage for stage in common_seed_stage_by_submission if stage is not None
    }
    if common_seed_stages:
        if len(common_seed_stages) != 1 or any(
            stage is None for stage in common_seed_stage_by_submission
        ):
            raise ValueError(
                "A common-seed phase cannot mix protocol versions or ordinary submissions."
            )
        common_seed_stage = next(iter(common_seed_stages))
        flux_common_seed_module_for_stage(
            common_seed_stage
        ).validate_common_seed_phase_contract(
            plan_path=plan_path,
            plan=plan,
            state_path=state_path,
            submissions=validated,
            root=root,
        )

    union_rows.sort(key=lambda row: (row["submission_id"], row["job_index"]))
    union_encoded = json.dumps(union_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
    union_proof = {
        "selected_job_count": len(union_rows),
        "unique_manifest_job_count": len(selected_jobs),
        "unique_condition_id_count": len(condition_owners),
        "unique_output_dir_count": len(output_owners),
        "selected_job_union_sha256": hashlib.sha256(union_encoded).hexdigest(),
    }
    return ValidatedPhase(
        plan_path=plan_path,
        plan=plan,
        state_path=state_path,
        submissions=tuple(validated),
        union_proof=union_proof,
    )


def build_sbatch_command(submission: ValidatedSubmission) -> list[str]:
    """Build the exact shell-free command for one validated submission."""

    environment = submission.merged_environment
    exported = ",".join(f"{key}={value}" for key, value in sorted(environment.items()))
    command = [
        "sbatch",
        "--parsable",
    ]
    if flux_common_seed_stage_from_manifest(submission.manifest) is not None:
        # No common-seed task may start until every versioned cohort array has immutable
        # registries and the complete held-phase launch commit exists.
        command.append("--hold")
    command.extend(
        [
        "--array",
        submission.array_spec,
        "--job-name",
        submission.slurm_job_name,
        "--export",
        f"ALL,{exported}",
        str(submission.launcher_path),
        ]
    )
    return command


def phase_preview(validated: ValidatedPhase) -> dict[str, Any]:
    """Return a non-mutating validation/dry-run receipt."""

    return {
        "schema_version": 1,
        "benchmark": BENCHMARK_NAME,
        "mode": "dry_run",
        "phase_id": validated.plan["phase_id"],
        "phase_plan_path": str(validated.plan_path),
        "phase_plan_sha256": validated.phase_plan_sha256,
        "state_path": str(validated.state_path),
        "union_proof": validated.union_proof,
        "commands": [build_sbatch_command(entry) for entry in validated.submissions],
    }


def submit_phase(
    validated: ValidatedPhase,
    *,
    root: Path | None = None,
    run: CommandRunner = subprocess.run,
    registry_writer: RegistryWriter = write_submission_registry,
    manifest_reader: ManifestReader = read_manifest,
) -> dict[str, Any]:
    """Submit or safely resume one validated phase.

    A completed phase is never treated as a no-op because that can conceal an
    accidental replay in automation.  It raises :class:`PhaseReplayError`.
    """

    root = (root or project_root()).resolve()
    lock_path = validated.state_path.with_suffix(validated.state_path.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise PhaseReplayError(
                f"Another orchestrator holds the phase lock: {lock_path}"
            ) from exc

        common_seed_stage_by_submission = [
            flux_common_seed_stage_from_manifest(submission.manifest)
            for submission in validated.submissions
        ]
        common_seed_phase = bool(common_seed_stage_by_submission) and all(
            stage is not None and stage == common_seed_stage_by_submission[0]
            for stage in common_seed_stage_by_submission
        )
        common_seed_module = (
            flux_common_seed_module_for_stage(common_seed_stage_by_submission[0])
            if common_seed_phase
            else None
        )
        state = _load_or_initialize_state(validated)
        state = _reconcile_state(validated, state, registry_writer)
        if state["status"] == "complete":
            if common_seed_phase:
                assert common_seed_module is not None
                common_seed_module.authorize_and_release_complete_common_seed_phase(
                    validated=validated,
                    state=state,
                    root=root,
                    run=run,
                )
                return state
            raise PhaseReplayError(
                f"Phase {validated.plan['phase_id']!r} is already complete; refusing replay."
            )
        if state["status"] == "submission_outcome_unknown":
            raise PhaseStateError(
                "A prior sbatch outcome is ambiguous; reconcile it manually and create the "
                "planned registry before attempting any continuation."
            )

        for position, submission in enumerate(validated.submissions):
            entry = state["submissions"][position]
            if entry["status"] == "registered":
                continue
            if entry["status"] == "registry_pending":
                state = _publish_planned_registry(
                    validated, state, position, submission, registry_writer
                )
                continue
            if entry["status"] != "pending":
                raise PhaseStateError(
                    f"Submission {submission.submission_id} cannot resume from "
                    f"status {entry['status']!r}."
                )

            # Close the validation-to-submit race for every still-pending array.
            # Already-submitted registry repair deliberately does not depend on
            # live implementation state, but no new job may launch after drift.
            try:
                current_plan = read_phase_plan(validated.plan_path, root)
            except ValueError as exc:
                raise PhaseStateError(
                    f"Phase plan is no longer immutable: {validated.plan_path}"
                ) from exc
            if current_plan != validated.plan:
                raise PhaseStateError(
                    f"Phase plan changed after validation: {validated.plan_path}"
                )
            if _sha256_file(submission.launcher_path) != submission.launcher_sha256:
                raise PhaseStateError(
                    f"Launcher changed after phase validation: {submission.launcher_path}"
                )
            current_manifest = manifest_reader(submission.manifest_path, root)
            if current_manifest != submission.manifest:
                raise PhaseStateError(
                    f"Manifest changed after phase validation: {submission.manifest_path}"
                )

            state = _transition_entry(
                validated,
                state,
                position,
                status="submitting",
                phase_status="active",
            )
            command = build_sbatch_command(submission)
            try:
                result = run(
                    command,
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                slurm_job_id = _parse_sbatch_result(result.stdout)
            except Exception as exc:
                state = _transition_entry(
                    validated,
                    state,
                    position,
                    status="submission_outcome_unknown",
                    phase_status="submission_outcome_unknown",
                    error={"type": type(exc).__name__, "message": str(exc)[:2000]},
                )
                raise PhaseStateError(
                    f"sbatch outcome for {submission.submission_id} is not safely replayable."
                ) from exc

            state = _transition_entry(
                validated,
                state,
                position,
                status="registry_pending",
                phase_status="active",
                slurm_array_job_id=slurm_job_id,
                sbatch_command=command,
            )
            state = _publish_planned_registry(
                validated, state, position, submission, registry_writer
            )

        final = deepcopy(state)
        final["status"] = "complete"
        final["completed_at_utc"] = _utc_now()
        state = _publish_state(validated.state_path, final, previous=state)
        if common_seed_phase:
            assert common_seed_module is not None
            common_seed_module.authorize_and_release_complete_common_seed_phase(
                validated=validated,
                state=state,
                root=root,
                run=run,
            )
        return state


def read_phase_state(path: Path) -> dict[str, Any]:
    """Read a content-authenticated phase state/receipt."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PhaseStateError(f"Cannot read phase state {path}: {exc}") from exc
    if not isinstance(payload, dict) or payload.get("schema_version") != PHASE_STATE_SCHEMA_VERSION:
        raise PhaseStateError(f"Unsupported phase state schema in {path}.")
    actual = phase_state_digest(payload)
    if payload.get("phase_state_sha256") != actual:
        raise PhaseStateError(f"Phase state digest mismatch: {path}")
    return payload


def _validate_plan_shape(plan: Mapping[str, Any]) -> None:
    if set(plan) != _PLAN_KEYS:
        raise ValueError(
            f"Phase plan keys must be exactly {sorted(_PLAN_KEYS)}; got {sorted(plan)}."
        )
    if plan.get("schema_version") != PHASE_PLAN_SCHEMA_VERSION:
        raise ValueError("Unsupported finer-detailing phase-plan schema.")
    if plan.get("benchmark") != BENCHMARK_NAME:
        raise ValueError("Phase plan benchmark identity mismatch.")
    if _PHASE_ID_RE.fullmatch(str(plan.get("phase_id", ""))) is None:
        raise ValueError("Phase plan phase_id is invalid.")
    if not str(plan.get("created_at_utc", "")).strip():
        raise ValueError("Phase plan created_at_utc must be explicit.")
    if not str(plan.get("state_path", "")).strip():
        raise ValueError("Phase plan state_path must be explicit.")
    digest = plan.get("phase_plan_sha256")
    if not isinstance(digest, str) or _SHA256_RE.fullmatch(digest) is None:
        raise ValueError("Phase plan must carry a lowercase SHA-256 identity.")
    submissions = plan.get("submissions")
    if not isinstance(submissions, list) or not submissions:
        raise ValueError("Phase plan submissions must be a non-empty list.")
    for position, raw in enumerate(submissions):
        if not isinstance(raw, dict) or set(raw) != _SUBMISSION_KEYS:
            keys = sorted(raw) if isinstance(raw, dict) else type(raw).__name__
            raise ValueError(
                f"Phase submission {position} keys must be exactly "
                f"{sorted(_SUBMISSION_KEYS)}; got {keys}."
            )
        if _SUBMISSION_ID_RE.fullmatch(str(raw.get("submission_id", ""))) is None:
            raise ValueError(f"Phase submission {position} has an invalid submission_id.")
        for field in (
            "manifest_path",
            "launcher_path",
            "array_spec",
            "registry_path",
            "slurm_job_name",
        ):
            if not isinstance(raw.get(field), str) or not str(raw[field]).strip():
                raise ValueError(f"Phase submission {position} requires non-empty {field}.")
        for field in ("manifest_sha256", "launcher_sha256"):
            if not isinstance(raw.get(field), str) or _SHA256_RE.fullmatch(raw[field]) is None:
                raise ValueError(f"Phase submission {position} requires lowercase SHA-256 {field}.")
        if not isinstance(raw.get("exported_environment"), dict) or not isinstance(
            raw.get("pilot_environment"), dict
        ):
            raise ValueError(f"Phase submission {position} environment mappings must be objects.")


def _validate_environment_mapping(raw: Any, field: str) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ValueError(f"{field} must be an object.")
    output: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(key, str) or _ENV_NAME_RE.fullmatch(key) is None:
            raise ValueError(f"{field} contains an invalid environment variable name: {key!r}.")
        if not isinstance(value, str) or any(
            character in value for character in (",", "\n", "\r", "\0")
        ):
            raise ValueError(
                f"{field}.{key} must be a string without commas, newlines, or NUL bytes."
            )
        output[key] = value
    return output


def _required_pilot_environment(jobs: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    required: dict[str, str] = {}
    for job in jobs:
        snapshot = job.get("temporal_protocol_snapshot")
        if not isinstance(snapshot, dict) or snapshot.get("qualification") != "pilot":
            continue
        model_name = str(job.get("model_name", ""))
        required.update(REQUIRED_PILOT_ENV_BY_MODEL.get(model_name, {}))
        for name in _collect_mapping_values(snapshot, "pilot_environment_variable"):
            required[name] = "1"
    return dict(sorted(required.items()))


def _collect_mapping_values(value: Any, key: str) -> set[str]:
    matches: set[str] = set()
    if isinstance(value, Mapping):
        for nested_key, nested_value in value.items():
            if nested_key == key and isinstance(nested_value, str) and nested_value:
                matches.add(nested_value)
            matches.update(_collect_mapping_values(nested_value, key))
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        for nested_value in value:
            matches.update(_collect_mapping_values(nested_value, key))
    return matches


def _initial_state(validated: ValidatedPhase) -> dict[str, Any]:
    state: dict[str, Any] = {
        "schema_version": PHASE_STATE_SCHEMA_VERSION,
        "benchmark": BENCHMARK_NAME,
        "phase_id": validated.plan["phase_id"],
        "phase_plan_path": str(validated.plan_path),
        "phase_plan_sha256": validated.phase_plan_sha256,
        "state_revision": 0,
        "previous_phase_state_sha256": None,
        "created_at_utc": _utc_now(),
        "updated_at_utc": _utc_now(),
        "completed_at_utc": None,
        "status": "active",
        "union_proof": validated.union_proof,
        "submissions": [
            {
                "submission_id": entry.submission_id,
                "manifest_path": str(entry.manifest_path),
                "manifest_sha256": entry.manifest_sha256,
                "array_spec": entry.array_spec,
                "registry_path": str(entry.registry_path),
                "status": "pending",
                "slurm_array_job_id": None,
                "registry_sha256": None,
                "sbatch_command": None,
                "error": None,
            }
            for entry in validated.submissions
        ],
    }
    state["phase_state_sha256"] = phase_state_digest(state)
    return state


def _load_or_initialize_state(validated: ValidatedPhase) -> dict[str, Any]:
    if not validated.state_path.exists():
        state = _initial_state(validated)
        _atomic_write_new_json(validated.state_path, state)
        return state
    state = read_phase_state(validated.state_path)
    _validate_state_binding(validated, state)
    return state


def _validate_state_binding(validated: ValidatedPhase, state: Mapping[str, Any]) -> None:
    expected = {
        "benchmark": BENCHMARK_NAME,
        "phase_id": validated.plan["phase_id"],
        "phase_plan_path": str(validated.plan_path),
        "phase_plan_sha256": validated.phase_plan_sha256,
        "union_proof": validated.union_proof,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise PhaseStateError(f"Phase state {key} does not match the immutable phase plan.")
    state_entries = state.get("submissions")
    if not isinstance(state_entries, list) or len(state_entries) != len(validated.submissions):
        raise PhaseStateError("Phase state submission list differs from the phase plan.")
    for saved, planned in zip(state_entries, validated.submissions, strict=True):
        core = {
            "submission_id": planned.submission_id,
            "manifest_path": str(planned.manifest_path),
            "manifest_sha256": planned.manifest_sha256,
            "array_spec": planned.array_spec,
            "registry_path": str(planned.registry_path),
        }
        if any(saved.get(key) != core[key] for key in _STATE_ENTRY_CORE_KEYS):
            raise PhaseStateError(
                f"Phase state entry differs from plan for {planned.submission_id}."
            )


def _reconcile_state(
    validated: ValidatedPhase,
    state: dict[str, Any],
    registry_writer: RegistryWriter,
) -> dict[str, Any]:
    _validate_state_binding(validated, state)
    for position, planned in enumerate(validated.submissions):
        entry = state["submissions"][position]
        status = entry.get("status")
        registry_exists = planned.registry_path.exists()
        if status == "registered":
            if not registry_exists:
                raise PhaseStateError(
                    f"Registered submission lost its immutable registry: {planned.registry_path}"
                )
            registry = _read_matching_registry(planned, entry.get("slurm_array_job_id"))
            if registry.get("registry_sha256") != entry.get("registry_sha256"):
                raise PhaseStateError(
                    f"Registry identity drift for submission {planned.submission_id}."
                )
        elif status == "registry_pending":
            if registry_exists:
                registry = _read_matching_registry(planned, entry.get("slurm_array_job_id"))
                state = _mark_registered(validated, state, position, registry)
            else:
                state = _publish_planned_registry(
                    validated, state, position, planned, registry_writer
                )
        elif status == "submitting":
            if not registry_exists:
                raise PhaseStateError(
                    f"Submission {planned.submission_id} stopped after durable submit intent "
                    "without a registry or recorded Slurm job ID; refusing replay."
                )
            registry = _read_matching_registry(planned, None)
            state = _mark_registered(validated, state, position, registry)
        elif status == "pending":
            if registry_exists:
                raise PhaseReplayError(
                    f"Pending submission already has an unbound registry; refusing to claim or "
                    f"replay it: {planned.registry_path}"
                )
        elif status == "submission_outcome_unknown":
            if registry_exists:
                registry = _read_matching_registry(planned, entry.get("slurm_array_job_id"))
                state = _mark_registered(validated, state, position, registry)
            else:
                return state
        else:
            raise PhaseStateError(
                f"Unknown phase submission state {status!r} for {planned.submission_id}."
            )
    return state


def _publish_planned_registry(
    validated: ValidatedPhase,
    state: dict[str, Any],
    position: int,
    submission: ValidatedSubmission,
    registry_writer: RegistryWriter,
) -> dict[str, Any]:
    entry = state["submissions"][position]
    slurm_job_id = entry.get("slurm_array_job_id")
    if not isinstance(slurm_job_id, str) or not slurm_job_id.isdigit():
        raise PhaseStateError(
            f"Submission {submission.submission_id} has no durable numeric Slurm job ID."
        )
    registry = build_submission_registry(
        manifest=submission.manifest,
        manifest_path=submission.manifest_path,
        slurm_array_job_id=slurm_job_id,
        array_spec=submission.array_spec,
        slurm_job_name=submission.slurm_job_name,
    )
    registry_writer(registry, submission.registry_path)
    verified = _read_matching_registry(submission, slurm_job_id)
    return _mark_registered(validated, state, position, verified)


def _read_matching_registry(submission: ValidatedSubmission, slurm_job_id: Any) -> dict[str, Any]:
    registry = read_submission_registry(submission.registry_path)
    expected = {
        "benchmark": BENCHMARK_NAME,
        "manifest_path": str(submission.manifest_path),
        "manifest_sha256": submission.manifest_sha256,
        "array_spec": submission.array_spec,
        "slurm_job_name": submission.slurm_job_name,
        "num_registered_tasks": len(submission.indices),
    }
    for key, value in expected.items():
        if registry.get(key) != value:
            raise PhaseStateError(
                f"Registry {submission.registry_path} {key} does not match its phase entry."
            )
    if slurm_job_id is not None and registry.get("slurm_array_job_id") != slurm_job_id:
        raise PhaseStateError(
            f"Registry {submission.registry_path} Slurm job ID differs from durable state."
        )
    if [row["job_index"] for row in registry["submissions"]] != list(submission.indices):
        raise PhaseStateError(
            f"Registry {submission.registry_path} task indices differ from its phase entry."
        )
    return registry


def _mark_registered(
    validated: ValidatedPhase,
    state: dict[str, Any],
    position: int,
    registry: Mapping[str, Any],
) -> dict[str, Any]:
    return _transition_entry(
        validated,
        state,
        position,
        status="registered",
        phase_status="active",
        slurm_array_job_id=str(registry["slurm_array_job_id"]),
        registry_sha256=str(registry["registry_sha256"]),
        error=None,
    )


def _transition_entry(
    validated: ValidatedPhase,
    state: dict[str, Any],
    position: int,
    *,
    status: str,
    phase_status: str,
    **updates: Any,
) -> dict[str, Any]:
    changed = deepcopy(state)
    changed["status"] = phase_status
    entry = changed["submissions"][position]
    entry["status"] = status
    entry.update(updates)
    return _publish_state(validated.state_path, changed, previous=state)


def _publish_state(
    path: Path, state: dict[str, Any], *, previous: Mapping[str, Any]
) -> dict[str, Any]:
    published = deepcopy(state)
    published["state_revision"] = int(previous["state_revision"]) + 1
    published["previous_phase_state_sha256"] = previous["phase_state_sha256"]
    published["updated_at_utc"] = _utc_now()
    published.pop("phase_state_sha256", None)
    published["phase_state_sha256"] = phase_state_digest(published)
    _atomic_replace_json(path, published)
    return published


def _parse_sbatch_result(stdout: str) -> str:
    value = stdout.strip()
    match = _SBATCH_RESULT_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"sbatch --parsable returned an invalid job identity: {value!r}.")
    return match.group("job_id")


def _resolve_under_root(path: Path, root: Path, field: str) -> Path:
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{field} escapes project root {root}: {resolved}") from exc
    return resolved


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_write_new_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write_new_text(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _atomic_write_new_text(path: Path, content: str, *, mode: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
            handle.flush()
            if mode is not None:
                os.fchmod(handle.fileno(), mode)
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"Refusing to overwrite immutable file: {path}") from exc
        temporary.unlink()
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_readonly_regular_text(path: Path, label: str) -> str:
    """Read one immutable artifact without following a symlink leaf.

    The descriptor and the post-read directory entry must name the same
    read-only regular inode.  This makes the read-only claim executable and
    prevents a check/read symlink substitution from weakening authentication.
    """

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Cannot open immutable {label} {path}: {exc}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"Immutable {label} is not a regular file: {path}")
        if opened.st_mode & 0o222:
            raise ValueError(f"Immutable {label} must be read-only (0444 or stricter): {path}")
        try:
            with os.fdopen(descriptor, mode="r", encoding="utf-8", closefd=False) as handle:
                content = handle.read()
        except UnicodeDecodeError as exc:
            raise ValueError(f"Cannot decode immutable {label} {path}: {exc}") from exc
    finally:
        os.close(descriptor)
    try:
        current = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"Immutable {label} disappeared during authentication: {path}") from exc
    if (
        stat.S_ISLNK(current.st_mode)
        or not stat.S_ISREG(current.st_mode)
        or (current.st_dev, current.st_ino) != (opened.st_dev, opened.st_ino)
        or current.st_mode & 0o222
    ):
        raise ValueError(f"Immutable {label} changed during authentication: {path}")
    return content


def _atomic_replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        temporary = None
        _fsync_directory(path.parent)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
