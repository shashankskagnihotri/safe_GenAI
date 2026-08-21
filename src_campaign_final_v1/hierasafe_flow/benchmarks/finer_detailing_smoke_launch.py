"""Dedicated launch authorization for canonical finer-detailing smoke jobs."""

from __future__ import annotations

import hashlib
import json
import os
import fcntl
import re
import shlex
import subprocess
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from hierasafe_flow.benchmarks.finer_detailing_correction import project_root
from hierasafe_flow.benchmarks.finer_detailing_production_smoke import (
    ValidatedSmokePlan,
    canonical_sha256,
    canonical_smoke_output_root,
    require_external_artifact_path,
    require_exact_canonical_path,
    validate_smoke_plan,
)
from hierasafe_flow.benchmarks.slurm_tracking import (
    ENVIRONMENT_PREFLIGHT_DIGEST_FILENAME,
    ENVIRONMENT_PREFLIGHT_FILENAME,
    build_submission_registry,
    read_environment_preflight,
    read_submission_registry,
    write_submission_registry,
)
from hierasafe_flow.utils.immutable_publication import (
    cleanup_owned_staging,
    freeze_tree,
    publish_hardlink_tree_commit_last,
    require_nonwritable_directories,
)


SMOKE_LAUNCH_AUTHORIZATION_FILENAME = "smoke_launch_authorization.json"
SMOKE_LAUNCH_AUTHORIZATION_DIGEST_FILENAME = f"{SMOKE_LAUNCH_AUTHORIZATION_FILENAME}.sha256"
SMOKE_LAUNCH_AUTHORIZATION_SCHEMA_VERSION = 2
SMOKE_LAUNCH_AUTHORIZATION_CONTRACT = "finer_detailing_smoke_launch_authorization_v2"
SMOKE_AUTHORIZATION_REQUIRED_ENV = "HIERASAFE_REQUIRE_SMOKE_LAUNCH_AUTHORIZATION"
SMOKE_SLURM_JOB_NAME = "finer-smoke"
SMOKE_SBATCH_LAUNCHER_RELATIVE = "slurm/finer_detailing_production_smoke_h100.sbatch"
SMOKE_SHELL_DISPATCHER_RELATIVE = "scripts/run_finer_detailing_smoke_dispatched.sh"
SMOKE_PYTHON_DISPATCHER_RELATIVE = "scripts/finer_detailing_smoke_dispatch.py"
SMOKE_LAUNCH_COMMIT_FILENAME = "complete_held_launch_commit.json"
SMOKE_HELD_RELEASE_RECEIPT_FILENAME = "held_release_receipt.json"
SMOKE_LAUNCH_STATE_FILENAME = "held_submission_state.json"
SMOKE_LAUNCH_COMMIT_CONTRACT = "finer_detailing_complete_held_smoke_launch_commit_v1"
SMOKE_HELD_RELEASE_RECEIPT_CONTRACT = "finer_detailing_smoke_held_release_receipt_v1"
SMOKE_LAUNCH_STATE_CONTRACT = "finer_detailing_smoke_held_submission_state_v1"
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SLURM_JOB_ID_RE = re.compile(r"^[0-9]+$")
_ROLE_RE = re.compile(r"^[a-z0-9][a-z0-9_]{0,127}$")
CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
_AUTHORIZATION_FIELDS = {
    "schema_version",
    "contract",
    "created_at_utc",
    "plan_path",
    "plan_file_sha256",
    "plan_sha256",
    "stage",
    "manifest_role",
    "manifest_path",
    "manifest_file_sha256",
    "manifest_sha256",
    "manifest_job_index",
    "condition_id",
    "output_dir",
    "implementation_files_sha256",
    "environment_preflight_sha256",
    "submission_registry_path",
    "submission_registry_file_sha256",
    "submission_registry_sha256",
    "launch_commit_path",
    "launch_commit_file_sha256",
    "launch_commit_sha256",
    "held_release_receipt_path",
    "held_release_receipt_file_sha256",
    "held_release_receipt_sha256",
    "launcher_path",
    "launcher_file_sha256",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "slurm_task_id",
    "slurm_job_id",
    "slurm_job_name",
    "scheduler_identity",
    "lineage",
    "authorization_sha256",
}
_AUTHORIZATION_HASH_FIELDS = {
    "plan_file_sha256",
    "plan_sha256",
    "manifest_file_sha256",
    "manifest_sha256",
    "implementation_files_sha256",
    "environment_preflight_sha256",
    "submission_registry_file_sha256",
    "submission_registry_sha256",
    "launch_commit_file_sha256",
    "launch_commit_sha256",
    "held_release_receipt_file_sha256",
    "held_release_receipt_sha256",
    "launcher_file_sha256",
    "authorization_sha256",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def authorization_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("authorization_sha256", None)
    return canonical_sha256(canonical)


def _document_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("document_sha256", None)
    return canonical_sha256(canonical)


def _state_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("state_sha256", None)
    return canonical_sha256(canonical)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


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


def _binding(path: Path) -> dict[str, str]:
    return {"path": str(path), "file_sha256": _sha256_file(path)}


def canonical_smoke_launch_root(plan: ValidatedSmokePlan) -> Path:
    """Return the plan-identity-scoped immutable launch-bundle path."""

    bundle = plan.path.parent
    return bundle.with_name(f"{bundle.name}.launch-{plan.digest[:16]}")


def canonical_smoke_registry_path(plan: ValidatedSmokePlan, role: str) -> Path:
    if _ROLE_RE.fullmatch(role) is None or role not in plan.manifests:
        raise ValueError(f"Unknown or unsafe smoke manifest role: {role!r}.")
    return canonical_smoke_launch_root(plan) / "registries" / f"{role}.json"


def require_exact_smoke_registry_path(plan: ValidatedSmokePlan, role: str, raw: str | Path) -> Path:
    expected = canonical_smoke_registry_path(plan, role).absolute()
    raw_text = os.fspath(raw)
    if any(segment in {".", ".."} for segment in raw_text.split(os.sep)):
        raise ValueError("Smoke registry path contains an explicit lexical alias.")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute() or str(candidate) != str(expected):
        raise ValueError(
            f"Smoke registry path is noncanonical: expected={expected}, actual={candidate}."
        )
    _require_no_symlink_components(candidate, "canonical smoke registry path")
    if candidate.resolve(strict=False) != expected.resolve(strict=False):
        raise ValueError("Smoke registry path resolves outside its canonical identity.")
    return candidate


def canonical_smoke_launch_commit_path(plan: ValidatedSmokePlan) -> Path:
    return canonical_smoke_launch_root(plan) / SMOKE_LAUNCH_COMMIT_FILENAME


def canonical_smoke_held_release_receipt_path(plan: ValidatedSmokePlan) -> Path:
    return canonical_smoke_launch_root(plan) / SMOKE_HELD_RELEASE_RECEIPT_FILENAME


def _staging_launch_root(plan: ValidatedSmokePlan) -> Path:
    target = canonical_smoke_launch_root(plan)
    return target.parent / f".{target.name}.held-staging"


def _lock_path(plan: ValidatedSmokePlan) -> Path:
    target = canonical_smoke_launch_root(plan)
    return target.parent / f".{target.name}.orchestrator.lock"


def _role_bindings(plan: ValidatedSmokePlan) -> list[tuple[str, Path, dict[str, Any]]]:
    bindings = plan.plan["manifest_bindings"]
    output: list[tuple[str, Path, dict[str, Any]]] = []
    for binding in bindings:
        role = str(binding["role"])
        if _ROLE_RE.fullmatch(role) is None or role not in plan.manifests:
            raise ValueError(f"Smoke launch plan contains an unsafe role {role!r}.")
        manifest_path = Path(str(binding["path"])).resolve()
        output.append((role, manifest_path, plan.manifests[role]))
    if not output:
        raise ValueError(f"Smoke stage {plan.stage!r} has no generation arrays to launch.")
    return output


def _full_array_spec(manifest: Mapping[str, Any]) -> str:
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list) or not jobs:
        raise ValueError("Every launched smoke role must contain at least one job.")
    return "0" if len(jobs) == 1 else f"0-{len(jobs) - 1}"


def _launcher_bindings(root: Path) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for name, relative in (
        ("sbatch", SMOKE_SBATCH_LAUNCHER_RELATIVE),
        ("shell_dispatcher", SMOKE_SHELL_DISPATCHER_RELATIVE),
        ("python_dispatcher", SMOKE_PYTHON_DISPATCHER_RELATIVE),
    ):
        path = (root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dedicated smoke launch file is absent: {path}")
        bindings[name] = _binding(path)
    return bindings


def _smoke_sbatch_command(
    plan: ValidatedSmokePlan,
    *,
    role: str,
    manifest: Mapping[str, Any],
    root: Path,
) -> list[str]:
    registry = canonical_smoke_registry_path(plan, role)
    exported = {
        "FINER_DETAILING_SMOKE_PLAN": str(plan.path),
        "FINER_DETAILING_SMOKE_REGISTRY": str(registry),
        "FINER_DETAILING_SMOKE_ROLE": role,
    }
    if any(
        any(character in value for character in (",", "\n", "\r", "\0"))
        for value in exported.values()
    ):
        raise ValueError("Smoke launch environment contains an unsafe character.")
    export_value = ",".join(f"{name}={value}" for name, value in sorted(exported.items()))
    return [
        "sbatch",
        "--parsable",
        "--hold",
        "--array",
        _full_array_spec(manifest),
        "--job-name",
        SMOKE_SLURM_JOB_NAME,
        "--export",
        f"ALL,{export_value}",
        str((root / SMOKE_SBATCH_LAUNCHER_RELATIVE).resolve()),
    ]


def smoke_launch_preview(plan_path: str | Path, *, root: Path | None = None) -> dict[str, Any]:
    """Validate and describe the complete held launch without mutating Slurm or disk."""

    root = (root or project_root()).resolve()
    plan = validate_smoke_plan(plan_path, root=root)
    launchers = _launcher_bindings(root)
    arrays = []
    for role, manifest_path, manifest in _role_bindings(plan):
        jobs = manifest["jobs"]
        arrays.append(
            {
                "role": role,
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest["manifest_sha256"],
                "array_spec": _full_array_spec(manifest),
                "indices": list(range(len(jobs))),
                "registry_path": str(canonical_smoke_registry_path(plan, role)),
                "slurm_job_name": SMOKE_SLURM_JOB_NAME,
                "sbatch_command": _smoke_sbatch_command(
                    plan, role=role, manifest=manifest, root=root
                ),
            }
        )
    return {
        "schema_version": 1,
        "contract": "finer_detailing_smoke_held_launch_preview_v1",
        "plan_path": str(plan.path),
        "plan_sha256": plan.digest,
        "stage": plan.stage,
        "launch_root": str(canonical_smoke_launch_root(plan)),
        "launchers": launchers,
        "arrays": arrays,
        "array_count": len(arrays),
        "logical_job_count": sum(len(item[2]["jobs"]) for item in _role_bindings(plan)),
    }


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = _temporary(path.parent, path.name, encoded)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _state_payload(preview: Mapping[str, Any]) -> dict[str, Any]:
    now = _utc_now()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": SMOKE_LAUNCH_STATE_CONTRACT,
        "status": "active",
        "created_at_utc": now,
        "updated_at_utc": now,
        "plan_path": preview["plan_path"],
        "plan_sha256": preview["plan_sha256"],
        "stage": preview["stage"],
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


def _validate_launch_state(state: Mapping[str, Any], preview: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version",
        "contract",
        "status",
        "created_at_utc",
        "updated_at_utc",
        "plan_path",
        "plan_sha256",
        "stage",
        "launch_root",
        "launchers",
        "arrays",
        "state_sha256",
    }
    if (
        not isinstance(state, Mapping)
        or set(state) != fields
        or state.get("schema_version") != 1
        or state.get("contract") != SMOKE_LAUNCH_STATE_CONTRACT
        or state.get("status")
        not in {"active", "submission_outcome_unknown", "complete_held_before_publish"}
        or state.get("state_sha256") != _state_digest(state)
    ):
        raise ValueError("Smoke held-submission state identity/digest is invalid.")
    _parse_timestamp(state["created_at_utc"], "smoke launch state created_at_utc")
    _parse_timestamp(state["updated_at_utc"], "smoke launch state updated_at_utc")
    fixed = {
        "plan_path": preview["plan_path"],
        "plan_sha256": preview["plan_sha256"],
        "stage": preview["stage"],
        "launch_root": preview["launch_root"],
        "launchers": preview["launchers"],
    }
    if any(state.get(key) != value for key, value in fixed.items()):
        raise ValueError("Smoke held-submission state drifted from the authenticated plan.")
    arrays = state.get("arrays")
    if not isinstance(arrays, list) or len(arrays) != len(preview["arrays"]):
        raise ValueError("Smoke held-submission state array count drifted.")
    allowed_status = {"pending", "submitting", "registry_pending", "registered", "unknown"}
    dynamic = {"status", "slurm_array_job_id", "registry_sha256", "error"}
    for saved, expected in zip(arrays, preview["arrays"], strict=True):
        if not isinstance(saved, Mapping) or set(saved) != set(expected) | dynamic:
            raise ValueError("Smoke held-submission state row schema drifted.")
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Smoke held-submission state row differs from its exact plan role.")
        if saved.get("status") not in allowed_status:
            raise ValueError("Smoke held-submission state row has an invalid status.")
        job_id = saved.get("slurm_array_job_id")
        if job_id is not None and _SLURM_JOB_ID_RE.fullmatch(str(job_id)) is None:
            raise ValueError("Smoke held-submission state contains an invalid Slurm job ID.")
        registry_sha = saved.get("registry_sha256")
        if registry_sha is not None and _SHA256_RE.fullmatch(str(registry_sha)) is None:
            raise ValueError("Smoke held-submission state contains an invalid registry digest.")
        if saved.get("error") is not None and not isinstance(saved.get("error"), Mapping):
            raise ValueError("Smoke held-submission state error must be an object or null.")
    return deepcopy(dict(state))


def _load_state(path: Path, preview: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot authenticate smoke held-submission state {path}: {exc}") from exc
    return _validate_launch_state(payload, preview)


def _publish_state(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    frozen = deepcopy(dict(state))
    frozen["updated_at_utc"] = _utc_now()
    frozen["state_sha256"] = _state_digest(frozen)
    _replace_json(path, frozen)
    reopened = json.loads(path.read_text(encoding="utf-8"))
    if reopened != frozen:
        raise RuntimeError("Smoke held-submission state failed durable reauthentication.")
    return frozen


def _parse_sbatch_job_id(stdout: str) -> str:
    raw = stdout.strip()
    job_id = raw.split(";", 1)[0]
    if _SLURM_JOB_ID_RE.fullmatch(job_id) is None or "\n" in raw:
        raise ValueError(f"Unexpected sbatch --parsable response: {stdout!r}.")
    return job_id


def validate_live_smoke_scheduler_identity(
    *,
    environ: Mapping[str, str],
    registry: Mapping[str, Any],
    index: int,
    launcher_path: Path,
    root: Path,
    run: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Query Slurm and bind the executing array task, command, and workdir."""

    job_id = str(environ.get("SLURM_JOB_ID", ""))
    if _SLURM_JOB_ID_RE.fullmatch(job_id) is None:
        raise RuntimeError("Live smoke SLURM_JOB_ID must be a numeric scheduler identity.")
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
        raise RuntimeError("Cannot authenticate the live smoke task through Slurm.") from exc
    lines = [line.strip() for line in str(result.stdout).splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError("Slurm returned no unique live smoke task record.")
    try:
        tokens = shlex.split(lines[0])
    except ValueError as exc:
        raise RuntimeError("Slurm returned a malformed live task record.") from exc
    record = {
        key: value
        for token in tokens
        for key, separator, value in (token.partition("="),)
        if separator
    }
    expected = {
        "JobId": job_id,
        "ArrayJobId": str(registry["slurm_array_job_id"]),
        "ArrayTaskId": str(index),
        "JobName": SMOKE_SLURM_JOB_NAME,
        "JobState": "RUNNING",
        "Command": str(launcher_path),
        "WorkDir": str(root),
        "BatchFlag": "1",
    }
    drift = {
        field: {"expected": value, "actual": record.get(field)}
        for field, value in expected.items()
        if record.get(field) != value
    }
    if drift:
        raise RuntimeError(f"Live Slurm smoke task record drifted: {drift}.")
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


def _registry_schema_is_exact(registry: Mapping[str, Any]) -> bool:
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
        isinstance(item, Mapping) and set(item) == row for item in registry.get("submissions", ())
    )


def _validate_role_registry(
    registry_path: Path,
    *,
    canonical_path: Path,
    manifest_path: Path,
    manifest: Mapping[str, Any],
    expected_job_id: str | None = None,
) -> dict[str, Any]:
    registry = read_submission_registry(registry_path)
    jobs = manifest["jobs"]
    indices = list(range(len(jobs)))
    if (
        not _registry_schema_is_exact(registry)
        or registry.get("manifest_path") != str(manifest_path)
        or registry.get("manifest_sha256") != manifest.get("manifest_sha256")
        or registry.get("array_spec") != _full_array_spec(manifest)
        or registry.get("num_registered_tasks") != len(jobs)
        or registry.get("slurm_job_name") != SMOKE_SLURM_JOB_NAME
        or [item.get("job_index") for item in registry.get("submissions", ())] != indices
        or [item.get("slurm_array_task_id") for item in registry.get("submissions", ())] != indices
        or expected_job_id is not None
        and str(registry.get("slurm_array_job_id")) != expected_job_id
    ):
        raise ValueError(
            f"Smoke registry does not cover the exact role-sized array: {canonical_path}"
        )
    job_id = str(registry["slurm_array_job_id"])
    if any(
        item.get("slurm_array_job_id") != job_id or item.get("slurm_task_id") != f"{job_id}_{index}"
        for index, item in enumerate(registry["submissions"])
    ):
        raise ValueError(f"Smoke registry task identity drifted: {canonical_path}")
    _parse_timestamp(registry["created_at_utc"], "smoke registry created_at_utc")
    return registry


def _write_document_pair(path: Path, payload: Mapping[str, Any]) -> None:
    frozen = deepcopy(dict(payload))
    frozen["document_sha256"] = _document_digest(frozen)
    encoded = (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode()
    raw_sha = hashlib.sha256(encoded).hexdigest()
    _publish_files_new(
        {
            path: encoded,
            path.with_suffix(path.suffix + ".sha256"): f"{raw_sha}  {path.name}\n".encode(),
        }
    )


def _read_document_pair(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or path.with_suffix(path.suffix + ".sha256").is_symlink():
        raise ValueError(f"{label} cannot be a symlink.")
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
        fields = path.with_suffix(path.suffix + ".sha256").read_text(encoding="utf-8").split()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot authenticate {label} {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("document_sha256") != _document_digest(payload)
        or fields != [hashlib.sha256(encoded).hexdigest(), path.name]
    ):
        raise ValueError(f"{label} digest or raw-file sidecar is inconsistent.")
    return payload


def _build_complete_launch_documents(
    *,
    plan: ValidatedSmokePlan,
    state: Mapping[str, Any],
    staging_root: Path,
    root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    preview = smoke_launch_preview(plan.path, root=root)
    state = _validate_launch_state(state, preview)
    if state.get("status") != "complete_held_before_publish" or any(
        row.get("status") != "registered" for row in state["arrays"]
    ):
        raise ValueError("Smoke launch commit requires every role array to be held and registered.")
    roles: list[dict[str, Any]] = []
    job_ids: list[str] = []
    for row, (role, manifest_path, manifest) in zip(
        state["arrays"], _role_bindings(plan), strict=True
    ):
        canonical_registry = canonical_smoke_registry_path(plan, role)
        physical_registry = staging_root / "registries" / f"{role}.json"
        registry = _validate_role_registry(
            physical_registry,
            canonical_path=canonical_registry,
            manifest_path=manifest_path,
            manifest=manifest,
            expected_job_id=str(row["slurm_array_job_id"]),
        )
        if row["registry_sha256"] != registry["registry_sha256"]:
            raise ValueError("Smoke held state and immutable registry digest differ.")
        job_id = str(registry["slurm_array_job_id"])
        job_ids.append(job_id)
        roles.append(
            {
                "role": role,
                "manifest_path": str(manifest_path),
                "manifest_file_sha256": _sha256_file(manifest_path),
                "manifest_sha256": manifest["manifest_sha256"],
                "num_jobs": len(manifest["jobs"]),
                "array_spec": _full_array_spec(manifest),
                "indices": list(range(len(manifest["jobs"]))),
                "slurm_job_name": SMOKE_SLURM_JOB_NAME,
                "slurm_array_job_id": job_id,
                "sbatch_command": deepcopy(row["sbatch_command"]),
                "registry": {
                    "path": str(canonical_registry),
                    "file_sha256": _sha256_file(physical_registry),
                    "registry_sha256": registry["registry_sha256"],
                },
            }
        )
    if len(set(job_ids)) != len(job_ids):
        raise ValueError("Every held smoke role array must have a unique Slurm array job ID.")
    state_path = staging_root / SMOKE_LAUNCH_STATE_FILENAME
    commit: dict[str, Any] = {
        "schema_version": 1,
        "contract": SMOKE_LAUNCH_COMMIT_CONTRACT,
        "status": "complete_held_arrays_committed_before_release",
        "created_at_utc": _utc_now(),
        "plan": {
            **_binding(plan.path),
            "smoke_plan_sha256": plan.digest,
            "stage": plan.stage,
        },
        "launch_root": str(canonical_smoke_launch_root(plan)),
        "launchers": deepcopy(preview["launchers"]),
        "held_state": {
            "path": str(canonical_smoke_launch_root(plan) / SMOKE_LAUNCH_STATE_FILENAME),
            "file_sha256": _sha256_file(state_path),
            "state_sha256": state["state_sha256"],
        },
        "roles": roles,
        "array_count": len(roles),
        "logical_job_count": sum(row["num_jobs"] for row in roles),
        "release_command": ["scontrol", "release", ",".join(job_ids)],
    }
    commit["document_sha256"] = _document_digest(commit)
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "contract": SMOKE_HELD_RELEASE_RECEIPT_CONTRACT,
        "status": "immutable_release_authorization_committed_before_scontrol",
        "created_at_utc": _utc_now(),
        "launch_commit": {
            "path": str(canonical_smoke_launch_commit_path(plan)),
            "document_sha256": commit["document_sha256"],
        },
        "plan_sha256": plan.digest,
        "stage": plan.stage,
        "slurm_array_job_ids": job_ids,
        "release_command": deepcopy(commit["release_command"]),
    }
    receipt["document_sha256"] = _document_digest(receipt)
    return commit, receipt


def _ensure_staging_root(plan: ValidatedSmokePlan, preview: Mapping[str, Any]) -> Path:
    staging = _staging_launch_root(plan)
    if staging.exists() or staging.is_symlink():
        if staging.is_symlink() or not staging.is_dir():
            raise ValueError(f"Smoke held-staging path is not a real directory: {staging}")
        _load_state(staging / SMOKE_LAUNCH_STATE_FILENAME, preview)
        return staging
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise RuntimeError(f"Competing smoke launch publisher claimed {staging}.") from exc
    (staging / "registries").mkdir(mode=0o700)
    state = _state_payload(preview)
    _replace_json(staging / SMOKE_LAUNCH_STATE_FILENAME, state)
    _fsync_directory(staging)
    return staging


def _freeze_tree(root: Path) -> None:
    freeze_tree(root, label="Immutable smoke launch bundle")
    _fsync_directory(root.parent)


def _publish_directory_commit_last(staging: Path, destination: Path) -> None:
    """Publish through the Ceph-safe commit-last admission edge."""

    try:
        publish_hardlink_tree_commit_last(
            staging,
            destination,
            commit_relative_path=Path(SMOKE_LAUNCH_COMMIT_FILENAME),
        )
    except FileExistsError as exc:
        raise FileExistsError(
            f"Immutable smoke launch bundle already exists: {destination}"
        ) from exc


def orchestrate_smoke_launch(
    plan_path: str | Path,
    *,
    root: Path | None = None,
    run: CommandRunner = subprocess.run,
    registry_writer: Callable[[Mapping[str, Any], Path], None] = write_submission_registry,
) -> dict[str, Any]:
    """Submit all exact role arrays held, commit every registry, then release once.

    A command failure after durable submit intent has an ambiguous scheduler outcome and
    is never automatically replayed.  The canonical launch root stays absent until all
    arrays, registries, the complete commit, and the release authorization are present.
    """

    root = (root or project_root()).resolve()
    plan = validate_smoke_plan(plan_path, root=root)
    preview = smoke_launch_preview(plan.path, root=root)
    canonical_root = canonical_smoke_launch_root(plan)
    lock_path = _lock_path(plan)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another smoke launch orchestrator holds {lock_path}.") from exc
        if canonical_root.exists() or canonical_root.is_symlink():
            read_smoke_launch_bundle(plan.path, root=root)
            raise RuntimeError(
                "The immutable smoke launch bundle already exists; refusing scheduler replay."
            )
        staging = _ensure_staging_root(plan, preview)
        state_path = staging / SMOKE_LAUNCH_STATE_FILENAME
        state = _load_state(state_path, preview)
        if state["status"] == "submission_outcome_unknown" or any(
            row["status"] in {"submitting", "unknown"} for row in state["arrays"]
        ):
            raise RuntimeError(
                "A smoke sbatch outcome is ambiguous; reconcile the held job manually and "
                "do not replay submission."
            )
        role_inputs = _role_bindings(plan)
        for position, (role, manifest_path, manifest) in enumerate(role_inputs):
            row = state["arrays"][position]
            physical_registry = staging / "registries" / f"{role}.json"
            canonical_registry = canonical_smoke_registry_path(plan, role)
            if row["status"] == "registered":
                registry = _validate_role_registry(
                    physical_registry,
                    canonical_path=canonical_registry,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    expected_job_id=str(row["slurm_array_job_id"]),
                )
                if registry["registry_sha256"] != row["registry_sha256"]:
                    raise ValueError("Registered smoke state differs from its registry.")
                continue
            if row["status"] == "registry_pending":
                if physical_registry.exists():
                    registry = _validate_role_registry(
                        physical_registry,
                        canonical_path=canonical_registry,
                        manifest_path=manifest_path,
                        manifest=manifest,
                        expected_job_id=str(row["slurm_array_job_id"]),
                    )
                else:
                    registry = build_submission_registry(
                        manifest=manifest,
                        manifest_path=manifest_path,
                        slurm_array_job_id=str(row["slurm_array_job_id"]),
                        array_spec=_full_array_spec(manifest),
                        slurm_job_name=SMOKE_SLURM_JOB_NAME,
                    )
                    registry_writer(registry, physical_registry)
                    registry = _validate_role_registry(
                        physical_registry,
                        canonical_path=canonical_registry,
                        manifest_path=manifest_path,
                        manifest=manifest,
                        expected_job_id=str(row["slurm_array_job_id"]),
                    )
                row["status"] = "registered"
                row["registry_sha256"] = registry["registry_sha256"]
                state = _publish_state(state_path, state)
                continue
            if row["status"] != "pending":
                raise RuntimeError(f"Smoke role {role} cannot resume from {row['status']!r}.")

            # Reopen all launch inputs immediately before durable submit intent.
            current = validate_smoke_plan(plan.path, root=root)
            current_preview = smoke_launch_preview(current.path, root=root)
            if current.digest != plan.digest or current_preview != preview:
                raise RuntimeError("Smoke plan or dedicated launcher changed before sbatch.")
            row["status"] = "submitting"
            state = _publish_state(state_path, state)
            try:
                result = run(
                    row["sbatch_command"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                job_id = _parse_sbatch_job_id(result.stdout)
            except BaseException as exc:
                row = state["arrays"][position]
                row["status"] = "unknown"
                row["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
                state["status"] = "submission_outcome_unknown"
                _publish_state(state_path, state)
                raise RuntimeError(
                    f"Smoke sbatch outcome for role {role!r} is ambiguous; refusing replay."
                ) from exc
            if job_id in {
                str(saved["slurm_array_job_id"])
                for saved in state["arrays"]
                if saved["slurm_array_job_id"] is not None
            }:
                raise RuntimeError("Slurm returned the same array job ID for two smoke roles.")
            row = state["arrays"][position]
            row["status"] = "registry_pending"
            row["slurm_array_job_id"] = job_id
            state = _publish_state(state_path, state)
            registry = build_submission_registry(
                manifest=manifest,
                manifest_path=manifest_path,
                slurm_array_job_id=job_id,
                array_spec=_full_array_spec(manifest),
                slurm_job_name=SMOKE_SLURM_JOB_NAME,
            )
            registry_writer(registry, physical_registry)
            registry = _validate_role_registry(
                physical_registry,
                canonical_path=canonical_registry,
                manifest_path=manifest_path,
                manifest=manifest,
                expected_job_id=job_id,
            )
            row = state["arrays"][position]
            row["status"] = "registered"
            row["registry_sha256"] = registry["registry_sha256"]
            state = _publish_state(state_path, state)

        state["status"] = "complete_held_before_publish"
        state = _publish_state(state_path, state)
        commit, receipt = _build_complete_launch_documents(
            plan=plan, state=state, staging_root=staging, root=root
        )
        commit_path = staging / SMOKE_LAUNCH_COMMIT_FILENAME
        receipt_path = staging / SMOKE_HELD_RELEASE_RECEIPT_FILENAME
        _write_document_pair(commit_path, commit)
        _write_document_pair(receipt_path, receipt)
        # Reopen bytes before the sole atomic visibility transition.
        if _read_document_pair(commit_path, label="staged smoke launch commit") != commit:
            raise RuntimeError("Staged smoke launch commit changed before publication.")
        if _read_document_pair(receipt_path, label="staged held-release receipt") != receipt:
            raise RuntimeError("Staged smoke release receipt changed before publication.")
        staging_stat = staging.lstat()
        staging_identity = (staging_stat.st_dev, staging_stat.st_ino)
        _freeze_tree(staging)
        try:
            _publish_directory_commit_last(staging, canonical_root)
        except BaseException:
            cleanup_owned_staging(staging, staging_identity)
            raise
        reopened = read_smoke_launch_bundle(plan.path, root=root)
        if reopened["commit"] != commit or reopened["receipt"] != receipt:
            raise RuntimeError("Published smoke launch bundle failed immediate reauthentication.")
        cleanup_owned_staging(staging, staging_identity)
        try:
            result = run(
                receipt["release_command"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except BaseException as exc:
            raise RuntimeError(
                "Smoke release outcome is ambiguous after immutable held-release receipt; "
                "never replay release automatically."
            ) from exc
        return {
            "schema_version": 1,
            "status": "release_command_completed",
            "plan_sha256": plan.digest,
            "launch_root": str(canonical_root),
            "launch_commit_sha256": commit["document_sha256"],
            "held_release_receipt_sha256": receipt["document_sha256"],
            "release_command": receipt["release_command"],
            "scheduler_stdout": str(getattr(result, "stdout", "")).strip(),
            "scheduler_stderr": str(getattr(result, "stderr", "")).strip(),
        }


def read_smoke_launch_bundle(plan_path: str | Path, *, root: Path | None = None) -> dict[str, Any]:
    """Authenticate the complete canonical all-registries-before-release bundle."""

    root = (root or project_root()).resolve()
    plan = validate_smoke_plan(plan_path, root=root)
    preview = smoke_launch_preview(plan.path, root=root)
    launch_root = canonical_smoke_launch_root(plan)
    _require_no_symlink_components(launch_root, "canonical smoke launch bundle")
    if launch_root.is_symlink() or not launch_root.is_dir():
        raise FileNotFoundError(f"Canonical smoke launch bundle is absent: {launch_root}")
    require_nonwritable_directories(
        launch_root, label="Canonical smoke launch bundle"
    )
    for member in launch_root.rglob("*"):
        observed = member.lstat()
        if member.is_symlink() or (member.is_file() and observed.st_mode & 0o222):
            raise ValueError(
                f"Canonical smoke launch bundle contains a mutable or aliased member: {member}"
            )
    expected_root_entries = {
        "registries",
        SMOKE_LAUNCH_STATE_FILENAME,
        SMOKE_LAUNCH_COMMIT_FILENAME,
        f"{SMOKE_LAUNCH_COMMIT_FILENAME}.sha256",
        SMOKE_HELD_RELEASE_RECEIPT_FILENAME,
        f"{SMOKE_HELD_RELEASE_RECEIPT_FILENAME}.sha256",
    }
    if {item.name for item in launch_root.iterdir()} != expected_root_entries:
        raise ValueError("Canonical smoke launch bundle has unexpected or missing artifacts.")
    registries_root = launch_root / "registries"
    if registries_root.is_symlink() or not registries_root.is_dir():
        raise ValueError("Canonical smoke registry root must be a real directory.")
    expected_registry_names = {f"{role}.json" for role in plan.manifests}
    if {item.name for item in registries_root.iterdir()} != expected_registry_names:
        raise ValueError("Canonical smoke registry set differs from the exact plan roles.")
    state = _load_state(launch_root / SMOKE_LAUNCH_STATE_FILENAME, preview)
    if state["status"] != "complete_held_before_publish" or any(
        row["status"] != "registered" for row in state["arrays"]
    ):
        raise ValueError("Published smoke launch state is not complete and held.")
    commit_path = canonical_smoke_launch_commit_path(plan)
    receipt_path = canonical_smoke_held_release_receipt_path(plan)
    commit = _read_document_pair(commit_path, label="smoke launch commit")
    receipt = _read_document_pair(receipt_path, label="smoke held-release receipt")
    commit_fields = {
        "schema_version",
        "contract",
        "status",
        "created_at_utc",
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
    if (
        set(commit) != commit_fields
        or commit.get("schema_version") != 1
        or commit.get("contract") != SMOKE_LAUNCH_COMMIT_CONTRACT
        or commit.get("status") != "complete_held_arrays_committed_before_release"
        or commit.get("launch_root") != str(launch_root)
        or commit.get("launchers") != preview["launchers"]
        or commit.get("array_count") != preview["array_count"]
        or commit.get("logical_job_count") != preview["logical_job_count"]
        or commit.get("plan")
        != {**_binding(plan.path), "smoke_plan_sha256": plan.digest, "stage": plan.stage}
        or commit.get("held_state")
        != {
            "path": str(launch_root / SMOKE_LAUNCH_STATE_FILENAME),
            "file_sha256": _sha256_file(launch_root / SMOKE_LAUNCH_STATE_FILENAME),
            "state_sha256": state["state_sha256"],
        }
    ):
        raise ValueError("Smoke launch commit identity, plan, launcher, or state binding drifted.")
    _parse_timestamp(commit["created_at_utc"], "smoke launch commit created_at_utc")
    role_fields = {
        "role",
        "manifest_path",
        "manifest_file_sha256",
        "manifest_sha256",
        "num_jobs",
        "array_spec",
        "indices",
        "slurm_job_name",
        "slurm_array_job_id",
        "sbatch_command",
        "registry",
    }
    roles = commit.get("roles")
    if not isinstance(roles, list) or len(roles) != len(plan.manifests):
        raise ValueError("Smoke launch commit role coverage drifted.")
    job_ids: list[str] = []
    for saved, preview_row, (role, manifest_path, manifest) in zip(
        roles, preview["arrays"], _role_bindings(plan), strict=True
    ):
        if not isinstance(saved, Mapping) or set(saved) != role_fields:
            raise ValueError("Smoke launch commit role schema drifted.")
        registry_path = canonical_smoke_registry_path(plan, role)
        if registry_path.is_symlink():
            raise ValueError("Canonical smoke registry cannot be a symlink.")
        registry = _validate_role_registry(
            registry_path,
            canonical_path=registry_path,
            manifest_path=manifest_path,
            manifest=manifest,
            expected_job_id=str(saved.get("slurm_array_job_id")),
        )
        exact = {
            "role": role,
            "manifest_path": str(manifest_path),
            "manifest_file_sha256": _sha256_file(manifest_path),
            "manifest_sha256": manifest["manifest_sha256"],
            "num_jobs": len(manifest["jobs"]),
            "array_spec": _full_array_spec(manifest),
            "indices": list(range(len(manifest["jobs"]))),
            "slurm_job_name": SMOKE_SLURM_JOB_NAME,
            "slurm_array_job_id": str(registry["slurm_array_job_id"]),
            "sbatch_command": preview_row["sbatch_command"],
            "registry": {
                **_binding(registry_path),
                "registry_sha256": registry["registry_sha256"],
            },
        }
        if dict(saved) != exact:
            raise ValueError("Smoke launch commit role/array/registry identity drifted.")
        job_ids.append(str(registry["slurm_array_job_id"]))
    release_command = ["scontrol", "release", ",".join(job_ids)]
    if len(set(job_ids)) != len(job_ids) or commit["release_command"] != release_command:
        raise ValueError("Smoke launch commit release identity drifted.")
    receipt_fields = {
        "schema_version",
        "contract",
        "status",
        "created_at_utc",
        "launch_commit",
        "plan_sha256",
        "stage",
        "slurm_array_job_ids",
        "release_command",
        "document_sha256",
    }
    if (
        set(receipt) != receipt_fields
        or receipt.get("schema_version") != 1
        or receipt.get("contract") != SMOKE_HELD_RELEASE_RECEIPT_CONTRACT
        or receipt.get("status") != "immutable_release_authorization_committed_before_scontrol"
        or receipt.get("launch_commit")
        != {"path": str(commit_path), "document_sha256": commit["document_sha256"]}
        or receipt.get("plan_sha256") != plan.digest
        or receipt.get("stage") != plan.stage
        or receipt.get("slurm_array_job_ids") != job_ids
        or receipt.get("release_command") != release_command
    ):
        raise ValueError("Smoke held-release receipt identity or scheduler coverage drifted.")
    _parse_timestamp(receipt["created_at_utc"], "smoke held-release receipt created_at_utc")
    return {"plan": plan, "state": state, "commit": commit, "receipt": receipt}


def is_canonical_smoke_job(job: Mapping[str, Any], root: Path | None = None) -> bool:
    root = (root or project_root()).resolve()
    raw = str(job.get("output_dir", "")).strip()
    if not raw:
        return False
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    lexical_output = candidate.absolute()
    smoke_root = canonical_smoke_output_root(root)
    if lexical_output == smoke_root or smoke_root in lexical_output.parents:
        return True
    output = lexical_output.resolve(strict=False)
    resolved_smoke_root = smoke_root.resolve(strict=False)
    return output == resolved_smoke_root or resolved_smoke_root in output.parents


def reject_canonical_smoke_manifest_from_ordinary_dispatch(
    manifest: Mapping[str, Any], root: Path | None = None
) -> None:
    """Integration hook: the ordinary dispatcher must call this after strict read."""

    jobs = manifest.get("jobs")
    if isinstance(jobs, list) and any(
        isinstance(job, Mapping) and is_canonical_smoke_job(job, root) for job in jobs
    ):
        raise RuntimeError(
            "Canonical engineering-smoke manifests require the dedicated smoke dispatcher, "
            "plan lineage, submission registry, and launch authorization."
        )


def smoke_plan_manifest_job(
    plan: ValidatedSmokePlan, role: str, index: int
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    if role not in plan.manifests:
        raise ValueError(f"Smoke plan {plan.stage} has no manifest role {role!r}.")
    binding = next(item for item in plan.plan["manifest_bindings"] if item["role"] == role)
    manifest = plan.manifests[role]
    jobs = manifest["jobs"]
    if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(jobs):
        raise IndexError(f"Smoke manifest index {index!r} is outside 0..{len(jobs) - 1}.")
    return Path(binding["path"]).resolve(), manifest, jobs[index]


def _lineage(plan: ValidatedSmokePlan) -> dict[str, list[dict[str, str]]]:
    plans: list[dict[str, str]] = []
    evaluations: list[dict[str, str]] = []

    def visit(current: ValidatedSmokePlan | None) -> None:
        if current is None:
            return
        visit(current.upstream)
        plans.append(
            {
                "stage": current.stage,
                "plan_sha256": current.digest,
                "file_sha256": _sha256_file(current.path),
            }
        )
        evaluations.extend(
            {
                "contract": report["contract"],
                "evaluation_sha256": report["evaluation_sha256"],
            }
            for report in current.gate_evaluations
        )

    visit(plan)
    return {"plans": plans, "gate_evaluations": evaluations}


def _registry_submission(
    registry: Mapping[str, Any], *, manifest_path: Path, manifest_sha256: str, index: int
) -> dict[str, Any]:
    if (
        Path(str(registry.get("manifest_path"))).resolve() != manifest_path
        or registry.get("manifest_sha256") != manifest_sha256
    ):
        raise ValueError("Smoke submission registry manifest binding drifted.")
    matches = [item for item in registry["submissions"] if item.get("job_index") == index]
    if len(matches) != 1:
        raise ValueError("Smoke registry must contain exactly one selected task binding.")
    return matches[0]


def validate_live_smoke_dispatch_identity(
    *,
    plan: ValidatedSmokePlan,
    role: str,
    index: int,
    submission_registry_path: str | Path,
    root: Path,
    environ: Mapping[str, str] | None = None,
    slurm_run: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Fail before preflight publication unless this is the committed live task."""

    root = root.resolve()
    manifest_path, manifest, _job = smoke_plan_manifest_job(plan, role, index)
    bundle = read_smoke_launch_bundle(plan.path, root=root)
    registry_path = require_exact_smoke_registry_path(plan, role, submission_registry_path)
    registry = _validate_role_registry(
        registry_path,
        canonical_path=registry_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    submission = _registry_submission(
        registry,
        manifest_path=manifest_path,
        manifest_sha256=manifest["manifest_sha256"],
        index=index,
    )
    values = os.environ if environ is None else environ
    expected_live = {
        "SLURM_ARRAY_JOB_ID": str(submission["slurm_array_job_id"]),
        "SLURM_ARRAY_TASK_ID": str(submission["slurm_array_task_id"]),
        "SLURM_JOB_NAME": SMOKE_SLURM_JOB_NAME,
    }
    if any(values.get(field) != expected for field, expected in expected_live.items()):
        raise RuntimeError(
            f"Live Slurm identity differs from smoke registry: expected={expected_live}."
        )
    committed = [item for item in bundle["commit"]["roles"] if item["role"] == role]
    if len(committed) != 1 or committed[0]["registry"]["path"] != str(registry_path):
        raise ValueError("Complete smoke launch commit does not bind this live role registry.")
    launcher_path = Path(bundle["commit"]["launchers"]["sbatch"]["path"])
    return validate_live_smoke_scheduler_identity(
        environ=values,
        registry=registry,
        index=index,
        launcher_path=launcher_path,
        root=root,
        run=slurm_run,
    )


def build_smoke_launch_authorization(
    *,
    plan_path: str | Path,
    role: str,
    index: int,
    submission_registry_path: str | Path,
    root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    slurm_run: CommandRunner = subprocess.run,
) -> tuple[dict[str, Any], Path]:
    root = (root or project_root()).resolve()
    plan = validate_smoke_plan(plan_path, root=root)
    manifest_path, manifest, job = smoke_plan_manifest_job(plan, role, index)
    launch_bundle = read_smoke_launch_bundle(plan.path, root=root)
    commit = launch_bundle["commit"]
    receipt = launch_bundle["receipt"]
    output = require_exact_canonical_path(
        job["output_dir"],
        expected=Path(job["output_dir"]).absolute(),
        descendant_root=canonical_smoke_output_root(root),
        label="smoke launch attempt output",
    )
    bound_job = {
        **job,
        "launch_manifest_sha256": manifest["manifest_sha256"],
        "launch_manifest_job_index": index,
    }
    read_environment_preflight(output, expected_job=bound_job, expected_job_index=index)
    preflight_path = output / ENVIRONMENT_PREFLIGHT_FILENAME
    registry_path = require_exact_smoke_registry_path(plan, role, submission_registry_path)
    require_external_artifact_path(
        registry_path,
        Path(plan.plan["output_root"]).resolve(),
        "smoke submission registry",
    )
    registry = _validate_role_registry(
        registry_path,
        canonical_path=registry_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    submission = _registry_submission(
        registry,
        manifest_path=manifest_path,
        manifest_sha256=manifest["manifest_sha256"],
        index=index,
    )
    values = os.environ if environ is None else environ
    expected_live = {
        "SLURM_ARRAY_JOB_ID": str(submission["slurm_array_job_id"]),
        "SLURM_ARRAY_TASK_ID": str(submission["slurm_array_task_id"]),
        "SLURM_JOB_NAME": SMOKE_SLURM_JOB_NAME,
    }
    if any(values.get(field) != expected for field, expected in expected_live.items()):
        raise RuntimeError(
            f"Live Slurm identity differs from smoke registry: expected={expected_live}."
        )
    slurm_job_id = str(values.get("SLURM_JOB_ID", ""))
    committed_role = [item for item in commit["roles"] if item["role"] == role]
    if len(committed_role) != 1 or committed_role[0]["registry"]["path"] != str(registry_path):
        raise ValueError("Complete smoke launch commit does not bind this role registry.")
    commit_path = canonical_smoke_launch_commit_path(plan)
    receipt_path = canonical_smoke_held_release_receipt_path(plan)
    launcher_path = Path(commit["launchers"]["sbatch"]["path"])
    scheduler_identity = validate_live_smoke_scheduler_identity(
        environ=values,
        registry=registry,
        index=index,
        launcher_path=launcher_path,
        root=root,
        run=slurm_run,
    )
    payload: dict[str, Any] = {
        "schema_version": SMOKE_LAUNCH_AUTHORIZATION_SCHEMA_VERSION,
        "contract": SMOKE_LAUNCH_AUTHORIZATION_CONTRACT,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "plan_path": str(plan.path),
        "plan_file_sha256": _sha256_file(plan.path),
        "plan_sha256": plan.digest,
        "stage": plan.stage,
        "manifest_role": role,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": _sha256_file(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "manifest_job_index": index,
        "condition_id": job["condition_id"],
        "output_dir": str(output),
        "implementation_files_sha256": job["implementation_files_sha256"],
        "environment_preflight_sha256": _sha256_file(preflight_path),
        "submission_registry_path": str(registry_path),
        "submission_registry_file_sha256": _sha256_file(registry_path),
        "submission_registry_sha256": registry["registry_sha256"],
        "launch_commit_path": str(commit_path),
        "launch_commit_file_sha256": _sha256_file(commit_path),
        "launch_commit_sha256": commit["document_sha256"],
        "held_release_receipt_path": str(receipt_path),
        "held_release_receipt_file_sha256": _sha256_file(receipt_path),
        "held_release_receipt_sha256": receipt["document_sha256"],
        "launcher_path": str(launcher_path),
        "launcher_file_sha256": _sha256_file(launcher_path),
        "slurm_array_job_id": str(submission["slurm_array_job_id"]),
        "slurm_array_task_id": int(submission["slurm_array_task_id"]),
        "slurm_task_id": submission["slurm_task_id"],
        "slurm_job_id": slurm_job_id,
        "slurm_job_name": SMOKE_SLURM_JOB_NAME,
        "scheduler_identity": scheduler_identity,
        "lineage": _lineage(plan),
    }
    payload["authorization_sha256"] = authorization_digest(payload)
    return payload, output


def _temporary(path: Path, name: str, content: bytes) -> Path:
    descriptor, raw = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=path)
    temporary = Path(raw)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return temporary


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _same_identity(path: Path, identity: tuple[int, int]) -> bool:
    try:
        observed = path.lstat()
    except FileNotFoundError:
        return False
    return not path.is_symlink() and (observed.st_dev, observed.st_ino) == identity


def _require_no_symlink_components(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute.")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}")


def _publish_files_new(documents: Mapping[Path, bytes]) -> None:
    """Publish files without replacement and clean up only inodes created here."""

    if not documents:
        return
    parents = {path.parent.resolve() for path in documents}
    if len(parents) != 1:
        raise ValueError("One immutable file transaction must use one parent directory.")
    parent = next(iter(parents))
    parent.mkdir(parents=True, exist_ok=True)
    temporaries: dict[Path, Path] = {}
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        for destination, content in documents.items():
            if destination.parent.resolve() != parent:
                raise ValueError("Immutable transaction destination escaped its parent.")
            temporary = _temporary(parent, destination.name, content)
            temporaries[destination] = temporary
        for destination, temporary in temporaries.items():
            source_stat = temporary.stat()
            identity = (source_stat.st_dev, source_stat.st_ino)
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite immutable smoke launch file: {destination}"
                ) from exc
            published.append((destination, identity))
            descriptor = os.open(destination, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                observed = os.fstat(descriptor)
                if (observed.st_dev, observed.st_ino) != identity:
                    raise RuntimeError(
                        f"Immutable smoke file was replaced during publication: {destination}"
                    )
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _fsync_directory(parent)
    except BaseException:
        for destination, identity in reversed(published):
            if _same_identity(destination, identity):
                destination.unlink()
        _fsync_directory(parent)
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)


def publish_smoke_launch_authorization(
    *,
    plan_path: str | Path,
    role: str,
    index: int,
    submission_registry_path: str | Path,
    root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    slurm_run: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    payload, output = build_smoke_launch_authorization(
        plan_path=plan_path,
        role=role,
        index=index,
        submission_registry_path=submission_registry_path,
        root=root,
        environ=environ,
        slurm_run=slurm_run,
    )
    expected_entries = {
        ENVIRONMENT_PREFLIGHT_FILENAME,
        ENVIRONMENT_PREFLIGHT_DIGEST_FILENAME,
    }
    if not output.is_dir() or {item.name for item in output.iterdir()} != expected_entries:
        raise FileExistsError(
            "Smoke authorization requires exactly the immutable environment-preflight pair."
        )
    target = output / SMOKE_LAUNCH_AUTHORIZATION_FILENAME
    sidecar = output / SMOKE_LAUNCH_AUTHORIZATION_DIGEST_FILENAME
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    file_sha = hashlib.sha256(encoded).hexdigest()
    _publish_files_new(
        {
            target: encoded,
            sidecar: f"{file_sha}  {target.name}\n".encode(),
        }
    )
    if read_smoke_launch_authorization(output, root=root) != payload:
        raise RuntimeError("Smoke launch authorization failed immediate reauthentication.")
    return payload


def read_smoke_launch_authorization(
    output_dir: Path,
    *,
    expected_plan: ValidatedSmokePlan | None = None,
    expected_role: str | None = None,
    expected_index: int | None = None,
    expected_job: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    raw_output = output_dir.expanduser()
    if not raw_output.is_absolute():
        raw_output = root / raw_output
    raw_output = raw_output.absolute()
    _require_no_symlink_components(raw_output, "smoke authorization attempt path")
    output = raw_output.resolve()
    target = output / SMOKE_LAUNCH_AUTHORIZATION_FILENAME
    sidecar = output / SMOKE_LAUNCH_AUTHORIZATION_DIGEST_FILENAME
    encoded = target.read_bytes()
    file_sha = hashlib.sha256(encoded).hexdigest()
    if sidecar.read_text(encoding="utf-8").split() != [file_sha, target.name]:
        raise ValueError("Smoke launch authorization file sidecar mismatch.")
    payload = json.loads(encoded)
    if (
        not isinstance(payload, dict)
        or set(payload) != _AUTHORIZATION_FIELDS
        or payload.get("schema_version") != SMOKE_LAUNCH_AUTHORIZATION_SCHEMA_VERSION
        or payload.get("contract") != SMOKE_LAUNCH_AUTHORIZATION_CONTRACT
        or payload.get("authorization_sha256") != authorization_digest(payload)
    ):
        raise ValueError("Smoke launch authorization identity/digest failed.")
    if any(
        not isinstance(payload.get(field), str)
        or len(payload[field]) != 64
        or any(character not in "0123456789abcdef" for character in payload[field])
        for field in _AUTHORIZATION_HASH_FIELDS
    ):
        raise ValueError("Smoke launch authorization contains an invalid SHA-256 binding.")
    _parse_timestamp(payload["created_at_utc"], "smoke launch authorization timestamp")
    for field in ("manifest_job_index", "slurm_array_task_id"):
        if (
            isinstance(payload[field], bool)
            or not isinstance(payload[field], int)
            or payload[field] < 0
        ):
            raise ValueError(f"Smoke launch authorization {field} must be a non-negative integer.")
    plan = validate_smoke_plan(payload["plan_path"], root=root)
    if expected_plan is not None and plan.digest != expected_plan.digest:
        raise ValueError("Smoke launch authorization binds another plan.")
    role = str(payload["manifest_role"])
    index = int(payload["manifest_job_index"])
    if expected_role is not None and role != expected_role:
        raise ValueError("Smoke launch authorization role drifted.")
    if expected_index is not None and index != expected_index:
        raise ValueError("Smoke launch authorization index drifted.")
    manifest_path, manifest, job = smoke_plan_manifest_job(plan, role, index)
    require_exact_canonical_path(
        raw_output,
        expected=Path(job["output_dir"]).absolute(),
        descendant_root=Path(plan.plan["output_root"]).absolute(),
        label="smoke authorization attempt path",
    )
    exact = {
        "plan_path": str(plan.path),
        "plan_file_sha256": _sha256_file(plan.path),
        "plan_sha256": plan.digest,
        "stage": plan.stage,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": _sha256_file(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "condition_id": job["condition_id"],
        "output_dir": str(Path(job["output_dir"]).resolve()),
        "implementation_files_sha256": job["implementation_files_sha256"],
        "environment_preflight_sha256": _sha256_file(output / ENVIRONMENT_PREFLIGHT_FILENAME),
        "lineage": _lineage(plan),
    }
    launch_bundle = read_smoke_launch_bundle(plan.path, root=root)
    commit = launch_bundle["commit"]
    receipt = launch_bundle["receipt"]
    commit_path = canonical_smoke_launch_commit_path(plan)
    receipt_path = canonical_smoke_held_release_receipt_path(plan)
    launcher_path = Path(commit["launchers"]["sbatch"]["path"])
    exact.update(
        {
            "launch_commit_path": str(commit_path),
            "launch_commit_file_sha256": _sha256_file(commit_path),
            "launch_commit_sha256": commit["document_sha256"],
            "held_release_receipt_path": str(receipt_path),
            "held_release_receipt_file_sha256": _sha256_file(receipt_path),
            "held_release_receipt_sha256": receipt["document_sha256"],
            "launcher_path": str(launcher_path),
            "launcher_file_sha256": _sha256_file(launcher_path),
            "slurm_job_name": SMOKE_SLURM_JOB_NAME,
        }
    )
    if any(payload.get(field) != value for field, value in exact.items()):
        raise ValueError(
            "Smoke launch authorization plan/job/environment/held-launch lineage drifted."
        )
    registry_path = require_exact_smoke_registry_path(
        plan, role, payload["submission_registry_path"]
    )
    require_external_artifact_path(
        registry_path,
        Path(plan.plan["output_root"]).resolve(),
        "smoke submission registry",
    )
    if _sha256_file(registry_path) != payload["submission_registry_file_sha256"]:
        raise ValueError("Smoke launch authorization registry file drifted.")
    registry = _validate_role_registry(
        registry_path,
        canonical_path=registry_path,
        manifest_path=manifest_path,
        manifest=manifest,
    )
    submission = _registry_submission(
        registry,
        manifest_path=manifest_path,
        manifest_sha256=manifest["manifest_sha256"],
        index=index,
    )
    scheduler_identity = {
        "slurm_job_id": str(payload["slurm_job_id"]),
        "slurm_array_job_id": str(registry["slurm_array_job_id"]),
        "slurm_array_task_id": index,
        "slurm_job_name": SMOKE_SLURM_JOB_NAME,
        "slurm_job_state": "RUNNING",
        "launcher_path": str(launcher_path),
        "work_dir": str(root),
        "batch_flag": 1,
    }
    if (
        registry["registry_sha256"] != payload["submission_registry_sha256"]
        or str(submission["slurm_task_id"]) != payload["slurm_task_id"]
        or str(submission["slurm_array_job_id"]) != payload["slurm_array_job_id"]
        or int(submission["slurm_array_task_id"]) != payload["slurm_array_task_id"]
        or payload["slurm_job_name"] != registry["slurm_job_name"]
        or _SLURM_JOB_ID_RE.fullmatch(str(payload["slurm_job_id"])) is None
        or payload["scheduler_identity"] != scheduler_identity
    ):
        raise ValueError("Smoke launch authorization registry task drifted.")
    bound_job = {
        **job,
        "launch_manifest_sha256": manifest["manifest_sha256"],
        "launch_manifest_job_index": index,
    }
    read_environment_preflight(output, expected_job=bound_job, expected_job_index=index)
    if expected_job is not None:
        if dict(expected_job) != bound_job:
            raise ValueError("Smoke authorization differs from the exact direct-runner job.")
    return payload


def require_smoke_launch_authorization_for_job(
    job: Mapping[str, Any],
    root: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    slurm_run: CommandRunner = subprocess.run,
) -> dict[str, Any] | None:
    """Integration hook called unconditionally before model load by ``run_job``."""

    root = (root or project_root()).resolve()
    if not is_canonical_smoke_job(job, root):
        return None
    output = Path(str(job["output_dir"])).resolve()
    exact_entries = {
        ENVIRONMENT_PREFLIGHT_FILENAME,
        ENVIRONMENT_PREFLIGHT_DIGEST_FILENAME,
        SMOKE_LAUNCH_AUTHORIZATION_FILENAME,
        SMOKE_LAUNCH_AUTHORIZATION_DIGEST_FILENAME,
    }
    if not output.is_dir() or {item.name for item in output.iterdir()} != exact_entries:
        raise FileNotFoundError(
            "Canonical smoke attempt requires exactly environment preflight and smoke launch "
            "authorization pairs before model load."
        )
    payload = read_smoke_launch_authorization(output, expected_job=job, root=root)
    registry = read_submission_registry(Path(payload["submission_registry_path"]))
    observed = validate_live_smoke_scheduler_identity(
        environ=os.environ if environ is None else environ,
        registry=registry,
        index=int(payload["manifest_job_index"]),
        launcher_path=Path(payload["launcher_path"]),
        root=root,
        run=slurm_run,
    )
    if observed != payload["scheduler_identity"]:
        raise RuntimeError("Live Slurm identity changed between smoke dispatch and model load.")
    return payload
