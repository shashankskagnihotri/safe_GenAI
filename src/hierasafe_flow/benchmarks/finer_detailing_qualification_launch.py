"""Held, cohort-bound Slurm launch for fresh Q1/Q2 qualification arrays.

All four arrays enter Slurm held.  Their exact registries and live held-job
records are committed in one immutable directory before one ``scontrol
release`` command is allowed.  At execution time the dedicated dispatcher
reopens that complete cohort and queries Slurm again before model loading.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import shlex
import subprocess
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from hierasafe_flow.benchmarks.finer_detailing_correction import project_root
from hierasafe_flow.benchmarks.finer_detailing_qualification import (
    MANIFEST_ROLE_ORDER,
    ValidatedQualificationPlan,
    _cleanup_owned_staging,
    _require_nonwritable_directories,
    canonical_sha256,
    canonical_qualification_cohort_root,
    canonical_qualification_manifest_paths,
    publish_hardlink_tree_commit_last,
    validate_qualification_plan,
)
from hierasafe_flow.benchmarks.slurm_tracking import (
    build_submission_registry,
    read_environment_preflight,
    read_submission_registry,
    write_submission_registry,
)


QUALIFICATION_SBATCH_LAUNCHER_RELATIVE = (
    "slurm/finer_detailing_qualification_h100.sbatch"
)
QUALIFICATION_SHELL_DISPATCHER_RELATIVE = (
    "scripts/run_finer_detailing_qualification_dispatched.sh"
)
QUALIFICATION_PYTHON_DISPATCHER_RELATIVE = (
    "scripts/finer_detailing_qualification_dispatch.py"
)
QUALIFICATION_LAUNCH_STATE_FILENAME = "held_submission_state.json"
QUALIFICATION_LAUNCH_COMMIT_FILENAME = "complete_held_launch_commit.json"
QUALIFICATION_RELEASE_AUTHORIZATION_FILENAME = "held_release_authorization.json"
QUALIFICATION_LAUNCH_STATE_CONTRACT = (
    "finer_detailing_qualification_held_submission_state_v1"
)
QUALIFICATION_LAUNCH_COMMIT_CONTRACT = (
    "finer_detailing_qualification_complete_held_launch_commit_v1"
)
QUALIFICATION_RELEASE_AUTHORIZATION_CONTRACT = (
    "finer_detailing_qualification_held_release_authorization_v1"
)
QUALIFICATION_RUNTIME_AUTHORIZATION_CONTRACT = (
    "finer_detailing_qualification_runtime_launch_authorization_v1"
)
QUALIFICATION_AUTHORIZATION_REQUIRED_ENV = (
    "HIERASAFE_REQUIRE_QUALIFICATION_LAUNCH_AUTHORIZATION"
)
QUALIFICATION_JOB_NAMES = {"q1": "finer-qual-q1", "q2": "finer-qual-q2"}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID_RE = re.compile(r"^[0-9]+$")
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


def _launch_publication_fault_hook(_step: str) -> None:
    """Test-only crash boundary; production deliberately performs no action."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("document_sha256", None)
    return canonical_sha256(canonical)


def _state_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("state_sha256", None)
    return canonical_sha256(canonical)


def _authorization_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("authorization_sha256", None)
    return canonical_sha256(canonical)


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "file_sha256": _sha256_file(path)}


def _parse_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a timezone-aware ISO-8601 timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not valid ISO-8601.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone offset.")
    return parsed


def _require_no_symlink_components(path: Path, *, root: Path, label: str) -> None:
    if not path.is_absolute() or (path != root and root not in path.parents):
        raise ValueError(f"{label} escaped the project root: {path}.")
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}.")
        if current == root:
            return
        current = current.parent


def canonical_qualification_launch_root(plan: ValidatedQualificationPlan) -> Path:
    cohort = canonical_qualification_cohort_root(plan.phase, plan.path.parents[3])
    return cohort.with_name(f"{cohort.name}.launch-{plan.digest[:16]}")


def canonical_qualification_registry_path(
    plan: ValidatedQualificationPlan, role: str
) -> Path:
    if role not in MANIFEST_ROLE_ORDER:
        raise ValueError(f"Unknown qualification manifest role {role!r}.")
    return canonical_qualification_launch_root(plan) / "registries" / f"{role}.json"


def canonical_qualification_launch_commit_path(
    plan: ValidatedQualificationPlan,
) -> Path:
    return canonical_qualification_launch_root(plan) / QUALIFICATION_LAUNCH_COMMIT_FILENAME


def canonical_qualification_release_authorization_path(
    plan: ValidatedQualificationPlan,
) -> Path:
    return (
        canonical_qualification_launch_root(plan)
        / QUALIFICATION_RELEASE_AUTHORIZATION_FILENAME
    )


def _launchers(root: Path) -> dict[str, dict[str, str]]:
    rows: dict[str, dict[str, str]] = {}
    for name, relative in (
        ("sbatch", QUALIFICATION_SBATCH_LAUNCHER_RELATIVE),
        ("shell_dispatcher", QUALIFICATION_SHELL_DISPATCHER_RELATIVE),
        ("python_dispatcher", QUALIFICATION_PYTHON_DISPATCHER_RELATIVE),
    ):
        path = (root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dedicated qualification launch file is absent: {path}.")
        rows[name] = _binding(path)
    return rows


def _array_spec(manifest: Mapping[str, Any]) -> str:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Qualification launch manifest has no jobs.")
    return f"0-{len(jobs) - 1}"


def _sbatch_command(
    plan: ValidatedQualificationPlan,
    *,
    role: str,
    manifest: Mapping[str, Any],
    root: Path,
) -> list[str]:
    exported = {
        "FINER_DETAILING_QUALIFICATION_PLAN": str(plan.path),
        "FINER_DETAILING_QUALIFICATION_REGISTRY": str(
            canonical_qualification_registry_path(plan, role)
        ),
        "FINER_DETAILING_QUALIFICATION_ROLE": role,
    }
    if any(
        any(character in value for character in (",", "\n", "\r", "\0"))
        for value in exported.values()
    ):
        raise ValueError("Qualification launch environment contains an unsafe character.")
    export_value = ",".join(f"{key}={value}" for key, value in sorted(exported.items()))
    return [
        "sbatch",
        "--parsable",
        "--hold",
        "--array",
        _array_spec(manifest),
        "--job-name",
        QUALIFICATION_JOB_NAMES[plan.phase],
        "--export",
        f"ALL,{export_value}",
        str((root / QUALIFICATION_SBATCH_LAUNCHER_RELATIVE).resolve()),
    ]


def qualification_launch_preview(
    plan_path: str | Path, *, root: Path | None = None
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    plan = validate_qualification_plan(plan_path, root=root)
    launchers = _launchers(root)
    manifests = canonical_qualification_manifest_paths(plan.phase, root)
    arrays = []
    for role in MANIFEST_ROLE_ORDER:
        manifest = plan.manifests[role]
        arrays.append(
            {
                "role": role,
                "manifest_path": str(manifests[role]),
                "manifest_sha256": manifest["manifest_sha256"],
                "array_spec": _array_spec(manifest),
                "indices": list(range(len(manifest["jobs"]))),
                "registry_path": str(canonical_qualification_registry_path(plan, role)),
                "slurm_job_name": QUALIFICATION_JOB_NAMES[plan.phase],
                "sbatch_command": _sbatch_command(
                    plan, role=role, manifest=manifest, root=root
                ),
            }
        )
    return {
        "schema_version": 1,
        "contract": "finer_detailing_qualification_held_launch_preview_v1",
        "phase": plan.phase,
        "plan_path": str(plan.path),
        "plan_sha256": plan.digest,
        "launch_root": str(canonical_qualification_launch_root(plan)),
        "launchers": launchers,
        "arrays": arrays,
        "array_count": 4,
        "logical_job_count": sum(len(plan.manifests[role]["jobs"]) for role in MANIFEST_ROLE_ORDER),
    }


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _temporary(path: Path, name: str, content: bytes) -> Path:
    descriptor, raw = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=path)
    temporary = Path(raw)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = _temporary(path.parent, path.name, encoded)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_files_new(files: Mapping[Path, bytes]) -> None:
    temporaries: dict[Path, Path] = {}
    published: list[tuple[Path, Path]] = []
    try:
        for path, content in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            temporaries[path] = _temporary(path.parent, path.name, content)
        for path, temporary in temporaries.items():
            try:
                os.link(temporary, path)
            except FileExistsError as exc:
                raise FileExistsError(f"Refusing to overwrite immutable file: {path}.") from exc
            published.append((path, temporary))
        for path in {item.parent for item in files}:
            _fsync_directory(path)
    except Exception:
        for path, temporary in reversed(published):
            try:
                if path.stat().st_ino == temporary.stat().st_ino:
                    path.unlink()
            except FileNotFoundError:
                pass
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)


def _write_document_pair(path: Path, payload: Mapping[str, Any]) -> None:
    frozen = deepcopy(dict(payload))
    frozen["document_sha256"] = _document_digest(frozen)
    encoded = (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    _publish_files_new(
        {
            sidecar: (
                f"{hashlib.sha256(encoded).hexdigest()}  {path.name}\n"
            ).encode(),
            path: encoded,
        }
    )


def _read_document_pair(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or path.with_suffix(path.suffix + ".sha256").is_symlink():
        raise ValueError(f"{label} cannot be a symlink.")
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
        sidecar = path.with_suffix(path.suffix + ".sha256").read_text(
            encoding="utf-8"
        ).split()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot authenticate {label} {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("document_sha256") != _document_digest(payload)
        or sidecar != [hashlib.sha256(encoded).hexdigest(), path.name]
    ):
        raise ValueError(f"{label} digest or sidecar is inconsistent.")
    return payload


def _state_payload(preview: Mapping[str, Any]) -> dict[str, Any]:
    now = _utc_now()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": QUALIFICATION_LAUNCH_STATE_CONTRACT,
        "status": "active",
        "created_at_utc": now,
        "updated_at_utc": now,
        "phase": preview["phase"],
        "plan_path": preview["plan_path"],
        "plan_sha256": preview["plan_sha256"],
        "launch_root": preview["launch_root"],
        "launchers": deepcopy(preview["launchers"]),
        "arrays": [
            {
                **deepcopy(row),
                "status": "pending",
                "slurm_array_job_id": None,
                "registry_sha256": None,
                "error": None,
            }
            for row in preview["arrays"]
        ],
    }
    payload["state_sha256"] = _state_digest(payload)
    return payload


def _validate_state(state: Mapping[str, Any], preview: Mapping[str, Any]) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "contract",
        "status",
        "created_at_utc",
        "updated_at_utc",
        "phase",
        "plan_path",
        "plan_sha256",
        "launch_root",
        "launchers",
        "arrays",
        "state_sha256",
    }
    if (
        set(state) != expected_fields
        or state.get("schema_version") != 1
        or state.get("contract") != QUALIFICATION_LAUNCH_STATE_CONTRACT
        or state.get("status")
        not in {"active", "submission_outcome_unknown", "complete_held_before_publish"}
        or state.get("state_sha256") != _state_digest(state)
    ):
        raise ValueError("Qualification held-submission state is malformed.")
    _parse_timestamp(state["created_at_utc"], "launch state created_at_utc")
    _parse_timestamp(state["updated_at_utc"], "launch state updated_at_utc")
    for field in ("phase", "plan_path", "plan_sha256", "launch_root", "launchers"):
        if state.get(field) != preview[field]:
            raise ValueError("Qualification launch state drifted from its preview.")
    arrays = state.get("arrays")
    if not isinstance(arrays, list) or len(arrays) != 4:
        raise ValueError("Qualification launch state must contain exactly four arrays.")
    dynamic = {"status", "slurm_array_job_id", "registry_sha256", "error"}
    allowed = {"pending", "submitting", "registry_pending", "registered", "unknown"}
    for saved, expected in zip(arrays, preview["arrays"], strict=True):
        if not isinstance(saved, Mapping) or set(saved) != set(expected) | dynamic:
            raise ValueError("Qualification launch state row schema drifted.")
        if any(saved.get(field) != value for field, value in expected.items()):
            raise ValueError("Qualification launch state row drifted from its preview.")
        if saved.get("status") not in allowed:
            raise ValueError("Qualification launch state row status is invalid.")
        job_id = saved.get("slurm_array_job_id")
        if job_id is not None and _JOB_ID_RE.fullmatch(str(job_id)) is None:
            raise ValueError("Qualification launch state has an invalid Slurm job ID.")
        registry_sha = saved.get("registry_sha256")
        if registry_sha is not None and _SHA256_RE.fullmatch(str(registry_sha)) is None:
            raise ValueError("Qualification launch state has an invalid registry digest.")
    return deepcopy(dict(state))


def _load_state(path: Path, preview: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot authenticate qualification launch state: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError("Qualification launch state must be a JSON object.")
    return _validate_state(payload, preview)


def _save_state(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    frozen = deepcopy(dict(state))
    frozen["updated_at_utc"] = _utc_now()
    frozen["state_sha256"] = _state_digest(frozen)
    _replace_json(path, frozen)
    reopened = json.loads(path.read_text(encoding="utf-8"))
    if reopened != frozen:
        raise RuntimeError("Qualification launch state failed durable reauthentication.")
    return frozen


def _parse_job_id(stdout: str) -> str:
    raw = stdout.strip()
    job_id = raw.split(";", 1)[0]
    if _JOB_ID_RE.fullmatch(job_id) is None or "\n" in raw:
        raise ValueError(f"Unexpected sbatch --parsable response: {stdout!r}.")
    return job_id


def _registry_schema_exact(registry: Mapping[str, Any]) -> bool:
    top = {
        "schema_version",
        "benchmark",
        "created_at_utc",
        "manifest_path",
        "manifest_sha256",
        "slurm_array_job_id",
        "slurm_job_name",
        "array_spec",
        "num_registered_tasks",
        "submissions",
        "registry_sha256",
    }
    row = {
        "manifest_sha256",
        "job_index",
        "slurm_array_job_id",
        "slurm_array_task_id",
        "slurm_task_id",
    }
    return set(registry) == top and all(
        isinstance(item, Mapping) and set(item) == row
        for item in registry.get("submissions", ())
    )


def _validate_registry(
    path: Path,
    *,
    plan: ValidatedQualificationPlan,
    role: str,
    expected_job_id: str | None = None,
) -> dict[str, Any]:
    registry = read_submission_registry(path)
    manifest = plan.manifests[role]
    manifest_path = canonical_qualification_manifest_paths(plan.phase, plan.path.parents[3])[role]
    indices = list(range(len(manifest["jobs"])))
    if (
        not _registry_schema_exact(registry)
        or registry.get("manifest_path") != str(manifest_path)
        or registry.get("manifest_sha256") != manifest["manifest_sha256"]
        or registry.get("array_spec") != _array_spec(manifest)
        or registry.get("slurm_job_name") != QUALIFICATION_JOB_NAMES[plan.phase]
        or registry.get("num_registered_tasks") != len(indices)
        or [item.get("job_index") for item in registry["submissions"]] != indices
        or [item.get("slurm_array_task_id") for item in registry["submissions"]]
        != indices
        or expected_job_id is not None
        and str(registry.get("slurm_array_job_id")) != expected_job_id
    ):
        raise ValueError(f"Qualification registry does not exactly cover role {role}.")
    job_id = str(registry["slurm_array_job_id"])
    if any(
        row.get("slurm_array_job_id") != job_id
        or row.get("slurm_task_id") != f"{job_id}_{index}"
        for index, row in enumerate(registry["submissions"])
    ):
        raise ValueError(f"Qualification registry task identity drifted for {role}.")
    return registry


def _parse_scontrol_record(stdout: str, *, label: str) -> dict[str, str]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"Slurm returned no unique {label} record.")
    try:
        tokens = shlex.split(lines[0])
    except ValueError as exc:
        raise RuntimeError(f"Slurm returned a malformed {label} record.") from exc
    return {
        key: value
        for token in tokens
        for key, separator, value in (token.partition("="),)
        if separator
    }


def validate_held_qualification_scheduler_identity(
    *,
    job_id: str,
    plan: ValidatedQualificationPlan,
    root: Path,
    run: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    if _JOB_ID_RE.fullmatch(job_id) is None:
        raise ValueError("Held qualification array job ID must be numeric.")
    try:
        result = run(
            ["scontrol", "show", "job", "-o", job_id],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except BaseException as exc:
        raise RuntimeError("Cannot authenticate a held qualification array in Slurm.") from exc
    record = _parse_scontrol_record(str(result.stdout), label="held qualification array")
    expected = {
        "JobId": job_id,
        "JobName": QUALIFICATION_JOB_NAMES[plan.phase],
        "JobState": "PENDING",
        "Reason": "JobHeldUser",
        "Command": str((root / QUALIFICATION_SBATCH_LAUNCHER_RELATIVE).resolve()),
        "WorkDir": str(root),
        "BatchFlag": "1",
    }
    drift = {
        field: {"expected": value, "actual": record.get(field)}
        for field, value in expected.items()
        if record.get(field) != value
    }
    if drift:
        raise RuntimeError(f"Held qualification Slurm identity drifted: {drift}.")
    return {
        "slurm_array_job_id": job_id,
        "slurm_job_name": expected["JobName"],
        "slurm_job_state": expected["JobState"],
        "reason": expected["Reason"],
        "launcher_path": expected["Command"],
        "work_dir": expected["WorkDir"],
        "batch_flag": 1,
    }


def validate_live_qualification_scheduler_identity(
    *,
    environ: Mapping[str, str],
    registry: Mapping[str, Any],
    index: int,
    plan: ValidatedQualificationPlan,
    root: Path,
    run: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    job_id = str(environ.get("SLURM_JOB_ID", ""))
    if _JOB_ID_RE.fullmatch(job_id) is None:
        raise RuntimeError("Live qualification SLURM_JOB_ID must be numeric.")
    try:
        result = run(
            ["scontrol", "show", "job", "-o", job_id],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except BaseException as exc:
        raise RuntimeError("Cannot authenticate the live qualification task in Slurm.") from exc
    record = _parse_scontrol_record(str(result.stdout), label="live qualification task")
    expected = {
        "JobId": job_id,
        "ArrayJobId": str(registry["slurm_array_job_id"]),
        "ArrayTaskId": str(index),
        "JobName": QUALIFICATION_JOB_NAMES[plan.phase],
        "JobState": "RUNNING",
        "Command": str((root / QUALIFICATION_SBATCH_LAUNCHER_RELATIVE).resolve()),
        "WorkDir": str(root),
        "BatchFlag": "1",
    }
    drift = {
        field: {"expected": value, "actual": record.get(field)}
        for field, value in expected.items()
        if record.get(field) != value
    }
    if drift:
        raise RuntimeError(f"Live qualification Slurm identity drifted: {drift}.")
    return {
        "slurm_job_id": job_id,
        "slurm_array_job_id": expected["ArrayJobId"],
        "slurm_array_task_id": index,
        "slurm_job_name": expected["JobName"],
        "slurm_job_state": expected["JobState"],
        "launcher_path": expected["Command"],
        "work_dir": expected["WorkDir"],
        "batch_flag": 1,
    }


def _staging_root(plan: ValidatedQualificationPlan) -> Path:
    target = canonical_qualification_launch_root(plan)
    return target.parent / f".{target.name}.held-staging"


def _lock_path(plan: ValidatedQualificationPlan) -> Path:
    target = canonical_qualification_launch_root(plan)
    return target.parent / f".{target.name}.orchestrator.lock"


def _ensure_staging(
    plan: ValidatedQualificationPlan, preview: Mapping[str, Any]
) -> Path:
    staging = _staging_root(plan)
    if staging.exists() or staging.is_symlink():
        if staging.is_symlink() or not staging.is_dir():
            raise ValueError("Qualification launch staging path is not a real directory.")
        _load_state(staging / QUALIFICATION_LAUNCH_STATE_FILENAME, preview)
        for path in staging.rglob("*"):
            if path.is_symlink():
                raise ValueError("Qualification launch staging contains a symlink.")
        staging.chmod(0o700)
        for directory in (path for path in staging.rglob("*") if path.is_dir()):
            directory.chmod(0o700)
        return staging
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise RuntimeError("A competing qualification launch publisher claimed staging.") from exc
    (staging / "registries").mkdir(mode=0o700)
    _replace_json(staging / QUALIFICATION_LAUNCH_STATE_FILENAME, _state_payload(preview))
    return staging


def _freeze_tree(root: Path) -> None:
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Qualification launch bundle contains a symlink: {path}.")
    for path in root.rglob("*"):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.chmod(0o555)
    root.chmod(0o555)


def _ensure_document_pair(path: Path, payload: Mapping[str, Any], *, label: str) -> None:
    """Create a document pair once, or authenticate the exact resumable pair."""

    frozen = deepcopy(dict(payload))
    frozen["document_sha256"] = _document_digest(frozen)
    encoded = (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode()
    sidecar = path.with_suffix(path.suffix + ".sha256")
    expected = {
        sidecar: f"{hashlib.sha256(encoded).hexdigest()}  {path.name}\n".encode(),
        path: encoded,
    }
    missing: dict[Path, bytes] = {}
    for member, content in expected.items():
        if member.is_symlink():
            raise ValueError(f"{label} cannot resume through a symlink.")
        if member.exists():
            if not member.is_file() or member.read_bytes() != content:
                raise ValueError(f"{label} differs from the recomputed launch document.")
        else:
            missing[member] = content
    if missing:
        _publish_files_new(missing)
    if _read_document_pair(path, label=label) != frozen:
        raise ValueError(f"{label} failed resumable pair authentication.")


def _load_resumable_document(path: Path, *, label: str) -> dict[str, Any] | None:
    """Read a staged document even if a crash preceded its sidecar link."""

    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.is_symlink() or sidecar.is_symlink():
        raise ValueError(f"{label} cannot resume through a symlink.")
    if not path.exists():
        if sidecar.exists():
            if not sidecar.is_file():
                raise ValueError(f"{label} has an invalid orphan sidecar.")
            sidecar.unlink()
            _fsync_directory(sidecar.parent)
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot decode {label}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("document_sha256") != _document_digest(payload)
    ):
        raise ValueError(f"{label} canonical digest drifted.")
    if sidecar.exists():
        return _read_document_pair(path, label=label)
    return payload


def _validate_staged_launch_tree(staging: Path) -> None:
    expected_entries = {
        "registries",
        QUALIFICATION_LAUNCH_STATE_FILENAME,
        QUALIFICATION_LAUNCH_COMMIT_FILENAME,
        f"{QUALIFICATION_LAUNCH_COMMIT_FILENAME}.sha256",
        QUALIFICATION_RELEASE_AUTHORIZATION_FILENAME,
        f"{QUALIFICATION_RELEASE_AUTHORIZATION_FILENAME}.sha256",
    }
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError("Qualification launch staging is not a real directory.")
    if {path.name for path in staging.iterdir()} != expected_entries:
        raise ValueError("Qualification launch staging artifact set is inexact.")
    registries = staging / "registries"
    if registries.is_symlink() or not registries.is_dir():
        raise ValueError("Qualification launch staging registry root is invalid.")
    if {path.name for path in registries.iterdir()} != {
        f"{role}.json" for role in MANIFEST_ROLE_ORDER
    }:
        raise ValueError("Qualification launch staging registry set is inexact.")
    for path in staging.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Qualification launch staging contains a symlink: {path}.")


def _build_launch_documents(
    *,
    plan: ValidatedQualificationPlan,
    state: Mapping[str, Any],
    staging: Path,
    root: Path,
    run: CommandRunner,
    commit_created_at_utc: str | None = None,
    authorization_created_at_utc: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preview = qualification_launch_preview(plan.path, root=root)
    state = _validate_state(state, preview)
    if state.get("status") != "complete_held_before_publish" or any(
        row.get("status") != "registered" for row in state["arrays"]
    ):
        raise ValueError("Qualification launch commit requires four registered held arrays.")
    roles = []
    job_ids: list[str] = []
    for row in state["arrays"]:
        role = str(row["role"])
        registry_path = staging / "registries" / f"{role}.json"
        registry = _validate_registry(
            registry_path,
            plan=plan,
            role=role,
            expected_job_id=str(row["slurm_array_job_id"]),
        )
        if registry["registry_sha256"] != row["registry_sha256"]:
            raise ValueError("Qualification held state and registry digest differ.")
        job_id = str(registry["slurm_array_job_id"])
        held_identity = validate_held_qualification_scheduler_identity(
            job_id=job_id, plan=plan, root=root, run=run
        )
        canonical_registry = canonical_qualification_registry_path(plan, role)
        job_ids.append(job_id)
        roles.append(
            {
                "role": role,
                "manifest_path": row["manifest_path"],
                "manifest_file_sha256": _sha256_file(Path(row["manifest_path"])),
                "manifest_sha256": row["manifest_sha256"],
                "array_spec": row["array_spec"],
                "indices": row["indices"],
                "num_jobs": len(row["indices"]),
                "slurm_job_name": row["slurm_job_name"],
                "slurm_array_job_id": job_id,
                "sbatch_command": row["sbatch_command"],
                "held_scheduler_identity": held_identity,
                "registry": {
                    "path": str(canonical_registry),
                    "file_sha256": _sha256_file(registry_path),
                    "registry_sha256": registry["registry_sha256"],
                },
            }
        )
    if len(job_ids) != 4 or len(set(job_ids)) != 4:
        raise ValueError("Qualification held arrays require four unique Slurm job IDs.")
    commit: dict[str, Any] = {
        "schema_version": 1,
        "contract": QUALIFICATION_LAUNCH_COMMIT_CONTRACT,
        "status": "complete_held_arrays_live_validated_before_release",
        "created_at_utc": commit_created_at_utc or _utc_now(),
        "phase": plan.phase,
        "plan": {
            **_binding(plan.path),
            "qualification_plan_sha256": plan.digest,
        },
        "launch_root": str(canonical_qualification_launch_root(plan)),
        "launchers": deepcopy(preview["launchers"]),
        "held_state": {
            "path": str(
                canonical_qualification_launch_root(plan)
                / QUALIFICATION_LAUNCH_STATE_FILENAME
            ),
            "file_sha256": _sha256_file(
                staging / QUALIFICATION_LAUNCH_STATE_FILENAME
            ),
            "state_sha256": state["state_sha256"],
        },
        "roles": roles,
        "array_count": 4,
        "logical_job_count": sum(row["num_jobs"] for row in roles),
        "release_command": ["scontrol", "release", ",".join(job_ids)],
    }
    commit["document_sha256"] = _document_digest(commit)
    authorization: dict[str, Any] = {
        "schema_version": 1,
        "contract": QUALIFICATION_RELEASE_AUTHORIZATION_CONTRACT,
        "status": "immutable_release_authorization_committed_before_scontrol",
        "created_at_utc": authorization_created_at_utc or _utc_now(),
        "phase": plan.phase,
        "plan_sha256": plan.digest,
        "launch_commit": {
            "path": str(canonical_qualification_launch_commit_path(plan)),
            "document_sha256": commit["document_sha256"],
        },
        "slurm_array_job_ids": job_ids,
        "release_command": deepcopy(commit["release_command"]),
    }
    authorization["document_sha256"] = _document_digest(authorization)
    return commit, authorization


def orchestrate_qualification_launch(
    plan_path: str | Path,
    *,
    root: Path | None = None,
    run: CommandRunner = subprocess.run,
    registry_writer: Callable[[Mapping[str, Any], Path], None] = write_submission_registry,
) -> dict[str, Any]:
    """Submit exactly four held arrays, atomically commit, and release once."""

    root = (root or project_root()).resolve()
    plan = validate_qualification_plan(plan_path, root=root)
    preview = qualification_launch_preview(plan.path, root=root)
    target = canonical_qualification_launch_root(plan)
    lock_path = _lock_path(plan)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another qualification launch orchestrator holds the lock.") from exc
        if target.exists() or target.is_symlink():
            read_qualification_launch_bundle(plan.path, root=root)
            raise RuntimeError("Qualification launch bundle exists; refusing scheduler replay.")
        staging = _ensure_staging(plan, preview)
        observed_staging = staging.lstat()
        staging_identity = (observed_staging.st_dev, observed_staging.st_ino)
        state_path = staging / QUALIFICATION_LAUNCH_STATE_FILENAME
        state = _load_state(state_path, preview)
        if state["status"] == "submission_outcome_unknown" or any(
            row["status"] in {"submitting", "unknown"} for row in state["arrays"]
        ):
            raise RuntimeError(
                "A qualification sbatch outcome is ambiguous; reconcile it without replay."
            )
        for position, role in enumerate(MANIFEST_ROLE_ORDER):
            row = state["arrays"][position]
            physical_registry = staging / "registries" / f"{role}.json"
            if row["status"] == "registered":
                registry = _validate_registry(
                    physical_registry,
                    plan=plan,
                    role=role,
                    expected_job_id=str(row["slurm_array_job_id"]),
                )
                if registry["registry_sha256"] != row["registry_sha256"]:
                    raise ValueError("Qualification registered state/registry drifted.")
                continue
            if row["status"] == "registry_pending":
                if physical_registry.exists():
                    registry = _validate_registry(
                        physical_registry,
                        plan=plan,
                        role=role,
                        expected_job_id=str(row["slurm_array_job_id"]),
                    )
                else:
                    registry = build_submission_registry(
                        manifest=plan.manifests[role],
                        manifest_path=Path(row["manifest_path"]),
                        slurm_array_job_id=str(row["slurm_array_job_id"]),
                        array_spec=row["array_spec"],
                        slurm_job_name=row["slurm_job_name"],
                    )
                    registry_writer(registry, physical_registry)
                    registry = _validate_registry(
                        physical_registry,
                        plan=plan,
                        role=role,
                        expected_job_id=str(row["slurm_array_job_id"]),
                    )
                row["status"] = "registered"
                row["registry_sha256"] = registry["registry_sha256"]
                state = _save_state(state_path, state)
                continue
            if row["status"] != "pending":
                raise RuntimeError(
                    f"Qualification role {role} cannot resume from {row['status']!r}."
                )
            current_preview = qualification_launch_preview(plan.path, root=root)
            if current_preview != preview:
                raise RuntimeError("Qualification plan/launcher changed before sbatch.")
            row["status"] = "submitting"
            state = _save_state(state_path, state)
            try:
                result = run(
                    row["sbatch_command"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                job_id = _parse_job_id(str(result.stdout))
            except BaseException as exc:
                row = state["arrays"][position]
                row["status"] = "unknown"
                row["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
                state["status"] = "submission_outcome_unknown"
                _save_state(state_path, state)
                raise RuntimeError(
                    f"Qualification sbatch outcome for {role} is ambiguous; refusing replay."
                ) from exc
            if job_id in {
                str(item["slurm_array_job_id"])
                for item in state["arrays"]
                if item["slurm_array_job_id"] is not None
            }:
                raise RuntimeError("Slurm reused one job ID for two qualification roles.")
            row = state["arrays"][position]
            row["status"] = "registry_pending"
            row["slurm_array_job_id"] = job_id
            state = _save_state(state_path, state)
            registry = build_submission_registry(
                manifest=plan.manifests[role],
                manifest_path=Path(row["manifest_path"]),
                slurm_array_job_id=job_id,
                array_spec=row["array_spec"],
                slurm_job_name=row["slurm_job_name"],
            )
            registry_writer(registry, physical_registry)
            registry = _validate_registry(
                physical_registry, plan=plan, role=role, expected_job_id=job_id
            )
            row = state["arrays"][position]
            row["status"] = "registered"
            row["registry_sha256"] = registry["registry_sha256"]
            state = _save_state(state_path, state)

        if state["status"] != "complete_held_before_publish":
            state["status"] = "complete_held_before_publish"
            state = _save_state(state_path, state)
        commit_path = staging / QUALIFICATION_LAUNCH_COMMIT_FILENAME
        authorization_path = staging / QUALIFICATION_RELEASE_AUTHORIZATION_FILENAME
        existing_commit = _load_resumable_document(
            commit_path, label="resumable staged qualification launch commit"
        )
        existing_authorization = _load_resumable_document(
            authorization_path,
            label="resumable staged qualification release authorization",
        )
        if existing_authorization is not None and existing_commit is None:
            raise ValueError(
                "Resumable release authorization exists without its launch commit."
            )
        if existing_commit is not None:
            _parse_timestamp(
                existing_commit.get("created_at_utc"),
                "resumable launch commit created_at_utc",
            )
        if existing_authorization is not None:
            _parse_timestamp(
                existing_authorization.get("created_at_utc"),
                "resumable release authorization created_at_utc",
            )
        commit, authorization = _build_launch_documents(
            plan=plan,
            state=state,
            staging=staging,
            root=root,
            run=run,
            commit_created_at_utc=(
                str(existing_commit["created_at_utc"])
                if existing_commit is not None
                else None
            ),
            authorization_created_at_utc=(
                str(existing_authorization["created_at_utc"])
                if existing_authorization is not None
                else None
            ),
        )
        _ensure_document_pair(
            commit_path, commit, label="staged qualification launch commit"
        )
        _ensure_document_pair(
            authorization_path,
            authorization,
            label="staged qualification release authorization",
        )
        if _read_document_pair(commit_path, label="staged qualification launch commit") != commit:
            raise RuntimeError("Staged qualification launch commit changed.")
        if (
            _read_document_pair(
                authorization_path, label="staged qualification release authorization"
            )
            != authorization
        ):
            raise RuntimeError("Staged qualification release authorization changed.")
        _validate_staged_launch_tree(staging)
        _freeze_tree(staging)
        publish_hardlink_tree_commit_last(
            staging,
            target,
            commit_relative_path=Path(QUALIFICATION_LAUNCH_COMMIT_FILENAME),
            fault_hook=_launch_publication_fault_hook,
        )
        reopened = read_qualification_launch_bundle(plan.path, root=root)
        if reopened["commit"] != commit or reopened["authorization"] != authorization:
            raise RuntimeError("Published qualification launch bundle failed authentication.")
        _cleanup_owned_staging(staging, staging_identity)
        try:
            result = run(
                authorization["release_command"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except BaseException as exc:
            raise RuntimeError(
                "Qualification release outcome is ambiguous after immutable authorization; "
                "never replay automatically."
            ) from exc
        return {
            "schema_version": 1,
            "status": "release_command_completed",
            "phase": plan.phase,
            "plan_sha256": plan.digest,
            "launch_root": str(target),
            "launch_commit_sha256": commit["document_sha256"],
            "release_authorization_sha256": authorization["document_sha256"],
            "release_command": authorization["release_command"],
            "scheduler_stdout": str(getattr(result, "stdout", "")).strip(),
            "scheduler_stderr": str(getattr(result, "stderr", "")).strip(),
        }


def read_qualification_launch_bundle(
    plan_path: str | Path, *, root: Path | None = None
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    plan = validate_qualification_plan(plan_path, root=root)
    preview = qualification_launch_preview(plan.path, root=root)
    launch_root = canonical_qualification_launch_root(plan)
    _require_no_symlink_components(launch_root, root=root, label="qualification launch root")
    if launch_root.is_symlink() or not launch_root.is_dir():
        raise FileNotFoundError(f"Qualification launch bundle is absent: {launch_root}.")
    _require_nonwritable_directories(
        launch_root, label="Canonical qualification launch bundle"
    )
    expected_entries = {
        "registries",
        QUALIFICATION_LAUNCH_STATE_FILENAME,
        QUALIFICATION_LAUNCH_COMMIT_FILENAME,
        f"{QUALIFICATION_LAUNCH_COMMIT_FILENAME}.sha256",
        QUALIFICATION_RELEASE_AUTHORIZATION_FILENAME,
        f"{QUALIFICATION_RELEASE_AUTHORIZATION_FILENAME}.sha256",
    }
    if {path.name for path in launch_root.iterdir()} != expected_entries:
        raise ValueError("Qualification launch bundle artifact set is inexact.")
    registries_root = launch_root / "registries"
    if registries_root.is_symlink() or not registries_root.is_dir():
        raise ValueError("Qualification registry root is not a real directory.")
    if {path.name for path in registries_root.iterdir()} != {
        f"{role}.json" for role in MANIFEST_ROLE_ORDER
    }:
        raise ValueError("Qualification launch registry set is inexact.")
    state = _load_state(launch_root / QUALIFICATION_LAUNCH_STATE_FILENAME, preview)
    if state["status"] != "complete_held_before_publish" or any(
        row["status"] != "registered" for row in state["arrays"]
    ):
        raise ValueError("Qualification launch bundle state is not complete-held.")
    registries = {
        role: _validate_registry(
            registries_root / f"{role}.json", plan=plan, role=role
        )
        for role in MANIFEST_ROLE_ORDER
    }
    commit = _read_document_pair(
        launch_root / QUALIFICATION_LAUNCH_COMMIT_FILENAME,
        label="qualification launch commit",
    )
    authorization = _read_document_pair(
        launch_root / QUALIFICATION_RELEASE_AUTHORIZATION_FILENAME,
        label="qualification release authorization",
    )
    expected_counts = {
        "q1": (330, [105, 75, 75, 75]),
        "q2": (66, [21, 15, 15, 15]),
    }[plan.phase]
    if (
        set(commit)
        != {
            "schema_version",
            "contract",
            "status",
            "created_at_utc",
            "phase",
            "plan",
            "launch_root",
            "launchers",
            "held_state",
            "roles",
            "array_count",
            "logical_job_count",
            "release_command",
            "document_sha256",
        }
        or commit.get("schema_version") != 1
        or commit.get("contract") != QUALIFICATION_LAUNCH_COMMIT_CONTRACT
        or commit.get("status")
        != "complete_held_arrays_live_validated_before_release"
        or commit.get("phase") != plan.phase
        or commit.get("plan")
        != {**_binding(plan.path), "qualification_plan_sha256": plan.digest}
        or commit.get("launch_root") != str(launch_root)
        or commit.get("launchers") != preview["launchers"]
        or commit.get("held_state")
        != {
            "path": str(launch_root / QUALIFICATION_LAUNCH_STATE_FILENAME),
            "file_sha256": _sha256_file(
                launch_root / QUALIFICATION_LAUNCH_STATE_FILENAME
            ),
            "state_sha256": state["state_sha256"],
        }
        or commit.get("array_count") != 4
        or commit.get("logical_job_count") != expected_counts[0]
        or [row.get("role") for row in commit.get("roles", ())]
        != list(MANIFEST_ROLE_ORDER)
        or [row.get("num_jobs") for row in commit.get("roles", ())]
        != expected_counts[1]
    ):
        raise ValueError("Qualification launch commit identity/topology drifted.")
    _parse_timestamp(commit["created_at_utc"], "qualification launch commit created_at_utc")
    expected_role_keys = {
        "role",
        "manifest_path",
        "manifest_file_sha256",
        "manifest_sha256",
        "array_spec",
        "indices",
        "num_jobs",
        "slurm_job_name",
        "slurm_array_job_id",
        "sbatch_command",
        "held_scheduler_identity",
        "registry",
    }
    job_ids = []
    for row in commit["roles"]:
        role = row["role"]
        registry = registries[role]
        canonical_registry = canonical_qualification_registry_path(plan, role)
        expected_held_identity = {
            "slurm_array_job_id": registry["slurm_array_job_id"],
            "slurm_job_name": QUALIFICATION_JOB_NAMES[plan.phase],
            "slurm_job_state": "PENDING",
            "reason": "JobHeldUser",
            "launcher_path": str(
                (root / QUALIFICATION_SBATCH_LAUNCHER_RELATIVE).resolve()
            ),
            "work_dir": str(root),
            "batch_flag": 1,
        }
        if (
            set(row) != expected_role_keys
            or row["manifest_path"] != preview["arrays"][MANIFEST_ROLE_ORDER.index(role)][
                "manifest_path"
            ]
            or row["manifest_file_sha256"]
            != _sha256_file(Path(row["manifest_path"]))
            or row["manifest_sha256"] != plan.manifests[role]["manifest_sha256"]
            or row["array_spec"] != _array_spec(plan.manifests[role])
            or row["indices"] != list(range(len(plan.manifests[role]["jobs"])))
            or row["slurm_job_name"] != QUALIFICATION_JOB_NAMES[plan.phase]
            or row["slurm_array_job_id"] != registry["slurm_array_job_id"]
            or row["registry"]
            != {
                "path": str(canonical_registry),
                "file_sha256": _sha256_file(canonical_registry),
                "registry_sha256": registry["registry_sha256"],
            }
            or row["sbatch_command"]
            != preview["arrays"][MANIFEST_ROLE_ORDER.index(role)]["sbatch_command"]
            or row["held_scheduler_identity"] != expected_held_identity
        ):
            raise ValueError(f"Qualification launch commit role binding drifted for {role}.")
        job_ids.append(registry["slurm_array_job_id"])
    release_command = ["scontrol", "release", ",".join(job_ids)]
    if commit.get("release_command") != release_command:
        raise ValueError("Qualification launch commit release command drifted.")
    if (
        set(authorization)
        != {
            "schema_version",
            "contract",
            "status",
            "created_at_utc",
            "phase",
            "plan_sha256",
            "launch_commit",
            "slurm_array_job_ids",
            "release_command",
            "document_sha256",
        }
        or authorization.get("schema_version") != 1
        or authorization.get("contract")
        != QUALIFICATION_RELEASE_AUTHORIZATION_CONTRACT
        or authorization.get("status")
        != "immutable_release_authorization_committed_before_scontrol"
        or authorization.get("phase") != plan.phase
        or authorization.get("plan_sha256") != plan.digest
        or authorization.get("launch_commit")
        != {
            "path": str(canonical_qualification_launch_commit_path(plan)),
            "document_sha256": commit["document_sha256"],
        }
        or authorization.get("slurm_array_job_ids") != job_ids
        or authorization.get("release_command") != release_command
    ):
        raise ValueError("Qualification release authorization drifted.")
    _parse_timestamp(
        authorization["created_at_utc"],
        "qualification release authorization created_at_utc",
    )
    return {
        "plan": plan,
        "preview": preview,
        "state": state,
        "registries": registries,
        "commit": commit,
        "authorization": authorization,
    }


def _registry_submission(
    registry: Mapping[str, Any], *, index: int
) -> Mapping[str, Any]:
    matches = [row for row in registry["submissions"] if row.get("job_index") == index]
    if len(matches) != 1:
        raise ValueError("Qualification registry lacks one exact task binding.")
    return matches[0]


def require_exact_qualification_registry_path(
    plan: ValidatedQualificationPlan, role: str, raw: str | Path, *, root: Path
) -> Path:
    expected = canonical_qualification_registry_path(plan, role).absolute()
    raw_text = os.fspath(raw)
    candidate = Path(raw).expanduser()
    if (
        any(part in {".", ".."} for part in candidate.parts)
        or not candidate.is_absolute()
        or str(candidate) != str(expected)
        or candidate.resolve(strict=False) != expected
    ):
        raise ValueError(
            f"Qualification registry path is noncanonical: expected={expected}, actual={raw_text}."
        )
    _require_no_symlink_components(candidate, root=root, label="qualification registry path")
    return candidate


def validate_live_qualification_dispatch_identity(
    *,
    plan: ValidatedQualificationPlan,
    role: str,
    index: int,
    registry_path: str | Path,
    root: Path,
    environ: Mapping[str, str] | None = None,
    slurm_run: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    bundle = read_qualification_launch_bundle(plan.path, root=root)
    if role not in MANIFEST_ROLE_ORDER:
        raise ValueError(f"Unknown qualification role {role!r}.")
    manifest = plan.manifests[role]
    if isinstance(index, bool) or index not in range(len(manifest["jobs"])):
        raise IndexError("Qualification job index is outside the exact role manifest.")
    canonical_registry = require_exact_qualification_registry_path(
        plan, role, registry_path, root=root
    )
    registry = _validate_registry(canonical_registry, plan=plan, role=role)
    submission = _registry_submission(registry, index=index)
    values = os.environ if environ is None else environ
    expected_live = {
        "SLURM_ARRAY_JOB_ID": str(submission["slurm_array_job_id"]),
        "SLURM_ARRAY_TASK_ID": str(index),
        "SLURM_JOB_NAME": QUALIFICATION_JOB_NAMES[plan.phase],
    }
    if any(values.get(field) != expected for field, expected in expected_live.items()):
        raise RuntimeError(
            f"Live Slurm identity differs from qualification registry: {expected_live}."
        )
    committed = [row for row in bundle["commit"]["roles"] if row["role"] == role]
    if len(committed) != 1 or committed[0]["registry"]["path"] != str(canonical_registry):
        raise ValueError("Qualification launch commit does not bind this role registry.")
    return validate_live_qualification_scheduler_identity(
        environ=values,
        registry=registry,
        index=index,
        plan=plan,
        root=root,
        run=slurm_run,
    )


def build_qualification_runtime_authorization(
    *,
    plan: ValidatedQualificationPlan,
    role: str,
    index: int,
    registry_path: str | Path,
    root: Path,
    environ: Mapping[str, str] | None = None,
    slurm_run: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    root = root.resolve()
    bundle = read_qualification_launch_bundle(plan.path, root=root)
    scheduler = validate_live_qualification_dispatch_identity(
        plan=plan,
        role=role,
        index=index,
        registry_path=registry_path,
        root=root,
        environ=environ,
        slurm_run=slurm_run,
    )
    manifest_path = canonical_qualification_manifest_paths(plan.phase, root)[role]
    manifest = plan.manifests[role]
    job = manifest["jobs"][index]
    canonical_registry = canonical_qualification_registry_path(plan, role)
    registry = bundle["registries"][role]
    submission = _registry_submission(registry, index=index)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": QUALIFICATION_RUNTIME_AUTHORIZATION_CONTRACT,
        "status": "authorized_before_model_load",
        "created_at_utc": _utc_now(),
        "phase": plan.phase,
        "plan_path": str(plan.path),
        "plan_file_sha256": _sha256_file(plan.path),
        "plan_sha256": plan.digest,
        "manifest_role": role,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": _sha256_file(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_job_index": index,
        "condition_id": job["condition_id"],
        "output_dir": str(Path(job["output_dir"]).resolve()),
        "implementation_files_sha256": plan.plan["implementation_files_sha256"],
        "submission_registry_path": str(canonical_registry),
        "submission_registry_file_sha256": _sha256_file(canonical_registry),
        "submission_registry_sha256": registry["registry_sha256"],
        "launch_commit_path": str(canonical_qualification_launch_commit_path(plan)),
        "launch_commit_file_sha256": _sha256_file(
            canonical_qualification_launch_commit_path(plan)
        ),
        "launch_commit_sha256": bundle["commit"]["document_sha256"],
        "release_authorization_path": str(
            canonical_qualification_release_authorization_path(plan)
        ),
        "release_authorization_file_sha256": _sha256_file(
            canonical_qualification_release_authorization_path(plan)
        ),
        "release_authorization_sha256": bundle["authorization"]["document_sha256"],
        "launcher_path": str((root / QUALIFICATION_SBATCH_LAUNCHER_RELATIVE).resolve()),
        "launcher_file_sha256": _sha256_file(
            (root / QUALIFICATION_SBATCH_LAUNCHER_RELATIVE).resolve()
        ),
        "slurm_array_job_id": str(submission["slurm_array_job_id"]),
        "slurm_array_task_id": int(submission["slurm_array_task_id"]),
        "slurm_task_id": submission["slurm_task_id"],
        "slurm_job_id": scheduler["slurm_job_id"],
        "slurm_job_name": scheduler["slurm_job_name"],
        "scheduler_identity": scheduler,
    }
    payload["authorization_sha256"] = _authorization_digest(payload)
    return payload


def _validate_runtime_authorization(
    payload: Mapping[str, Any],
    *,
    plan: ValidatedQualificationPlan,
    role: str,
    index: int,
    root: Path,
) -> None:
    expected_keys = {
        "schema_version",
        "contract",
        "status",
        "created_at_utc",
        "phase",
        "plan_path",
        "plan_file_sha256",
        "plan_sha256",
        "manifest_role",
        "manifest_path",
        "manifest_file_sha256",
        "manifest_sha256",
        "manifest_job_index",
        "condition_id",
        "output_dir",
        "implementation_files_sha256",
        "submission_registry_path",
        "submission_registry_file_sha256",
        "submission_registry_sha256",
        "launch_commit_path",
        "launch_commit_file_sha256",
        "launch_commit_sha256",
        "release_authorization_path",
        "release_authorization_file_sha256",
        "release_authorization_sha256",
        "launcher_path",
        "launcher_file_sha256",
        "slurm_array_job_id",
        "slurm_array_task_id",
        "slurm_task_id",
        "slurm_job_id",
        "slurm_job_name",
        "scheduler_identity",
        "authorization_sha256",
    }
    if (
        set(payload) != expected_keys
        or payload.get("schema_version") != 1
        or payload.get("contract") != QUALIFICATION_RUNTIME_AUTHORIZATION_CONTRACT
        or payload.get("status") != "authorized_before_model_load"
        or payload.get("authorization_sha256") != _authorization_digest(payload)
    ):
        raise ValueError("Qualification runtime authorization schema/digest drifted.")
    _parse_timestamp(payload["created_at_utc"], "runtime authorization created_at_utc")
    bundle = read_qualification_launch_bundle(plan.path, root=root)
    manifest_path = canonical_qualification_manifest_paths(plan.phase, root)[role]
    manifest = plan.manifests[role]
    job = manifest["jobs"][index]
    registry_path = canonical_qualification_registry_path(plan, role)
    registry = bundle["registries"][role]
    submission = _registry_submission(registry, index=index)
    fixed = {
        "phase": plan.phase,
        "plan_path": str(plan.path),
        "plan_file_sha256": _sha256_file(plan.path),
        "plan_sha256": plan.digest,
        "manifest_role": role,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": _sha256_file(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_job_index": index,
        "condition_id": job["condition_id"],
        "output_dir": str(Path(job["output_dir"]).resolve()),
        "implementation_files_sha256": plan.plan["implementation_files_sha256"],
        "submission_registry_path": str(registry_path),
        "submission_registry_file_sha256": _sha256_file(registry_path),
        "submission_registry_sha256": registry["registry_sha256"],
        "launch_commit_path": str(canonical_qualification_launch_commit_path(plan)),
        "launch_commit_file_sha256": _sha256_file(
            canonical_qualification_launch_commit_path(plan)
        ),
        "launch_commit_sha256": bundle["commit"]["document_sha256"],
        "release_authorization_path": str(
            canonical_qualification_release_authorization_path(plan)
        ),
        "release_authorization_file_sha256": _sha256_file(
            canonical_qualification_release_authorization_path(plan)
        ),
        "release_authorization_sha256": bundle["authorization"]["document_sha256"],
        "launcher_path": str((root / QUALIFICATION_SBATCH_LAUNCHER_RELATIVE).resolve()),
        "launcher_file_sha256": _sha256_file(
            (root / QUALIFICATION_SBATCH_LAUNCHER_RELATIVE).resolve()
        ),
        "slurm_array_job_id": str(submission["slurm_array_job_id"]),
        "slurm_array_task_id": int(submission["slurm_array_task_id"]),
        "slurm_task_id": submission["slurm_task_id"],
        "slurm_job_name": QUALIFICATION_JOB_NAMES[plan.phase],
    }
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in fixed.items()
        if payload.get(key) != value
    }
    scheduler = payload.get("scheduler_identity")
    expected_scheduler = {
        "slurm_job_id": payload.get("slurm_job_id"),
        "slurm_array_job_id": payload.get("slurm_array_job_id"),
        "slurm_array_task_id": index,
        "slurm_job_name": QUALIFICATION_JOB_NAMES[plan.phase],
        "slurm_job_state": "RUNNING",
        "launcher_path": str(
            (root / QUALIFICATION_SBATCH_LAUNCHER_RELATIVE).resolve()
        ),
        "work_dir": str(root),
        "batch_flag": 1,
    }
    if (
        drift
        or not isinstance(scheduler, Mapping)
        or dict(scheduler) != expected_scheduler
        or _JOB_ID_RE.fullmatch(str(payload.get("slurm_job_id", ""))) is None
    ):
        raise ValueError(f"Qualification runtime authorization binding drifted: {drift}.")


def read_qualification_launch_authorization(
    output_dir: str | Path,
    *,
    expected_plan: ValidatedQualificationPlan,
    expected_role: str,
    expected_index: int,
    root: Path | None = None,
) -> dict[str, Any]:
    """Reopen the runtime authorization embedded in the immutable preflight."""

    root = (root or project_root()).resolve()
    output = Path(output_dir).resolve()
    manifest = expected_plan.manifests[expected_role]
    job = manifest["jobs"][expected_index]
    preflight = read_environment_preflight(
        output,
        expected_job={
            **job,
            "launch_manifest_sha256": manifest["manifest_sha256"],
            "launch_manifest_job_index": expected_index,
        },
        expected_job_index=expected_index,
    )
    payload = preflight.get("qualification_launch_authorization")
    if not isinstance(payload, Mapping):
        raise ValueError("Qualification preflight lacks its runtime launch authorization.")
    _validate_runtime_authorization(
        payload,
        plan=expected_plan,
        role=expected_role,
        index=expected_index,
        root=root,
    )
    return deepcopy(dict(payload))


def reject_qualification_output_from_ordinary_dispatch(
    manifest: Mapping[str, Any], *, root: Path
) -> None:
    """Prevent copied/canonical Q1/Q2 rows from using the ordinary dispatcher."""

    qualification_root = (root / "outputs/finer_detailing_fresh_qualification_v1").resolve()
    for job in manifest.get("jobs", ()):
        if not isinstance(job, Mapping):
            continue
        output = Path(str(job.get("output_dir", ""))).resolve()
        if output == qualification_root or qualification_root in output.parents:
            raise ValueError(
                "Fresh qualification rows require the dedicated cohort-bound dispatcher."
            )


def is_fresh_qualification_job(job: Mapping[str, Any], root: Path) -> bool:
    qualification_root = (root / "outputs/finer_detailing_fresh_qualification_v1").resolve()
    output = Path(str(job.get("output_dir", ""))).resolve()
    return output != qualification_root and qualification_root in output.parents


def require_qualification_launch_authorization_for_job(
    job: Mapping[str, Any],
    *,
    root: Path,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Fail closed unless a canonical Q1/Q2 row has its exact preflight auth."""

    root = root.resolve()
    if not is_fresh_qualification_job(job, root):
        raise ValueError("Qualification authorization requested for a nonqualification job.")
    values = os.environ if environ is None else environ
    if values.get(QUALIFICATION_AUTHORIZATION_REQUIRED_ENV) != "1":
        raise RuntimeError(
            f"{QUALIFICATION_AUTHORIZATION_REQUIRED_ENV} must be exactly '1'."
        )
    raw_output = str(job.get("output_dir", ""))
    if not raw_output:
        raise ValueError("Qualification job lacks its exact output directory.")
    output = Path(raw_output).resolve()
    preflight = read_environment_preflight(output)
    authorization = preflight.get("qualification_launch_authorization")
    if not isinstance(authorization, Mapping):
        raise ValueError("Qualification preflight lacks runtime launch authorization.")
    role = authorization.get("manifest_role")
    index = authorization.get("manifest_job_index")
    plan_path = authorization.get("plan_path")
    if (
        role not in MANIFEST_ROLE_ORDER
        or isinstance(index, bool)
        or not isinstance(index, int)
        or not isinstance(plan_path, str)
        or not plan_path
    ):
        raise ValueError("Qualification preflight authorization routing is malformed.")
    plan = validate_qualification_plan(plan_path, root=root)
    manifest = plan.manifests[str(role)]
    if index not in range(len(manifest["jobs"])):
        raise ValueError("Qualification authorization job index is outside its manifest.")
    exact_job = {
        **manifest["jobs"][index],
        "launch_manifest_sha256": manifest["manifest_sha256"],
        "launch_manifest_job_index": index,
    }
    if dict(job) != exact_job:
        raise ValueError("Qualification runner job differs from its exact authorized row.")
    return read_qualification_launch_authorization(
        output,
        expected_plan=plan,
        expected_role=str(role),
        expected_index=index,
        root=root,
    )
