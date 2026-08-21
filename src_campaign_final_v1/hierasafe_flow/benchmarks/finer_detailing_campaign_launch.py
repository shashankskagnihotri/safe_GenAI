"""All-held launch authorization for the two canonical production campaigns.

The seed ladder is eight complete 0-35 arrays.  The selected-seed campaign is
72 complete arrays: 36 standard 0-7 arrays and 36 Shapley 0-5 arrays.  No
array is released until every held submission, immutable registry, live held
scheduler identity, task-union binding, launch commit, and release receipt is
atomically visible in one read-only directory.
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
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

from hierasafe_flow.benchmarks.finer_detailing_correction import (
    BENCHMARK_NAME,
    read_manifest_for_audit,
)
from hierasafe_flow.benchmarks.finer_detailing_correction import (
    project_root as benchmark_project_root,
)
from hierasafe_flow.benchmarks.slurm_tracking import (
    ENVIRONMENT_PREFLIGHT_DIGEST_FILENAME,
    ENVIRONMENT_PREFLIGHT_FILENAME,
    build_submission_registry,
    read_environment_preflight,
    read_submission_registry,
    write_submission_registry,
)
from hierasafe_flow.evaluation import finer_detailing_campaign as campaign


SEED_LADDER = "seed_ladder"
SELECTED_SEED_FINAL = "selected_seed_final"
COHORT_KINDS = (SEED_LADDER, SELECTED_SEED_FINAL)

CAMPAIGN_SBATCH_LAUNCHER_RELATIVE = "slurm/finer_detailing_campaign_h100.sbatch"
CAMPAIGN_SHELL_DISPATCHER_RELATIVE = (
    "scripts/run_finer_detailing_campaign_dispatched.sh"
)
CAMPAIGN_PYTHON_DISPATCHER_RELATIVE = "scripts/finer_detailing_campaign_dispatch.py"
CAMPAIGN_ORCHESTRATOR_RELATIVE = "scripts/orchestrate_finer_detailing_campaign_launch.py"
CAMPAIGN_RUNNER_RELATIVE = "scripts/run_finer_detailing_correction.py"
CAMPAIGN_MODULE_RELATIVE = (
    "src/hierasafe_flow/benchmarks/finer_detailing_campaign_launch.py"
)

LAUNCH_STATE_FILENAME = "held_submission_state.json"
LAUNCH_COMMIT_FILENAME = "complete_held_launch_commit.json"
RELEASE_RECEIPT_FILENAME = "held_release_receipt.json"
LAUNCH_COMMIT_CONTRACT = "finer_detailing_complete_held_campaign_launch_v1"
RELEASE_RECEIPT_CONTRACT = "finer_detailing_campaign_held_release_receipt_v1"
LAUNCH_STATE_CONTRACT = "finer_detailing_campaign_held_submission_state_v1"
LAUNCH_AUTHORIZATION_FILENAME = "campaign_launch_authorization.json"
LAUNCH_AUTHORIZATION_DIGEST_FILENAME = f"{LAUNCH_AUTHORIZATION_FILENAME}.sha256"
LAUNCH_AUTHORIZATION_CONTRACT = "finer_detailing_campaign_launch_authorization_v1"
LAUNCH_AUTHORIZATION_SCHEMA_VERSION = 1
CAMPAIGN_AUTHORIZATION_REQUIRED_ENV = (
    "HIERASAFE_REQUIRE_CAMPAIGN_LAUNCH_AUTHORIZATION"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_JOB_ID_RE = re.compile(r"^[0-9]+$")
_REGISTRY_TOP_FIELDS = {
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
_REGISTRY_ROW_FIELDS = {
    "manifest_sha256",
    "job_index",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "slurm_task_id",
}
_AUTHORIZATION_FIELDS = {
    "schema_version",
    "contract",
    "status",
    "created_at_utc",
    "cohort_kind",
    "cohort_root",
    "cohort_commit_path",
    "cohort_commit_file_sha256",
    "cohort_commit_sha256",
    "selection_cohort",
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
    "launchers",
    "slurm_array_job_id",
    "slurm_array_task_id",
    "slurm_task_id",
    "slurm_job_id",
    "slurm_job_name",
    "scheduler_identity",
    "authorization_sha256",
}
_AUTHORIZATION_HASH_FIELDS = {
    "cohort_commit_file_sha256",
    "cohort_commit_sha256",
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
    "authorization_sha256",
}

CommandRunner = Callable[..., subprocess.CompletedProcess[str]]


@dataclass(frozen=True)
class ValidatedCampaignCohort:
    """Authenticated immutable production cohort and canonical member order."""

    kind: str
    root: Path
    commit_path: Path
    commit: dict[str, Any]
    manifest_paths: tuple[Path, ...]
    manifests: tuple[dict[str, Any], ...]
    digest: str
    selection_cohort: dict[str, Any] | None = None


def project_root() -> Path:
    return benchmark_project_root().resolve()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _document_digest(payload: Mapping[str, Any], field: str) -> str:
    canonical = deepcopy(dict(payload))
    canonical.pop(field, None)
    return _canonical_sha256(canonical)


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


def _require_no_symlink_components(path: Path, label: str) -> None:
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute.")
    current = Path(path.anchor)
    for component in path.parts[1:]:
        current /= component
        if current.is_symlink():
            raise ValueError(f"{label} contains a symlink component: {current}.")


def _require_exact_path(raw: str | Path, expected: Path, label: str) -> Path:
    raw_text = os.fspath(raw)
    if raw_text != os.path.normpath(raw_text) or any(
        segment in {".", ".."} for segment in raw_text.split(os.sep)
    ):
        raise ValueError(f"{label} contains an explicit lexical alias.")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute() or str(candidate) != str(expected):
        raise ValueError(f"{label} is noncanonical: expected={expected}, actual={candidate}.")
    _require_no_symlink_components(candidate, label)
    if candidate.resolve(strict=False) != expected.resolve(strict=False):
        raise ValueError(f"{label} resolves outside its canonical identity.")
    return candidate


def _cohort_paths(kind: str, root: Path) -> tuple[Path, tuple[Path, ...]]:
    if kind == SEED_LADDER:
        paths = campaign.canonical_ladder_manifest_paths(root)
    elif kind == SELECTED_SEED_FINAL:
        paths = campaign.canonical_final_manifest_paths(root)
    else:
        raise ValueError(f"Unknown campaign cohort kind {kind!r}; expected {COHORT_KINDS}.")
    return paths[0].parent, paths


def _expected_campaign_counts(kind: str) -> dict[str, int]:
    if kind == SEED_LADDER:
        return {
            "manifest_count": campaign.EXPECTED_LADDER_MANIFESTS,
            "logical_rows": campaign.EXPECTED_LADDER_MEDIA,
            "media_rows": campaign.EXPECTED_LADDER_MEDIA,
            "unsupported_rows": 0,
            "exact_one_media_rows": 0,
        }
    return {
        "manifest_count": campaign.EXPECTED_FINAL_MANIFESTS,
        "logical_rows": campaign.EXPECTED_FINAL_LOGICAL,
        "media_rows": campaign.EXPECTED_FINAL_MEDIA,
        "unsupported_rows": campaign.EXPECTED_FINAL_UNSUPPORTED,
        "exact_one_media_rows": campaign.EXPECTED_FINAL_EXACT_ONE_MEDIA,
    }


def _read_campaign_commit(path: Path) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.is_symlink() or sidecar.is_symlink():
        raise ValueError("Campaign cohort commit and sidecar must not be symlinks.")
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
        sidecar_fields = sidecar.read_text(encoding="utf-8").split()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot authenticate campaign cohort commit {path}: {exc}") from exc
    fields = {
        "schema_version",
        "contract",
        "campaign_contract",
        "cohort_kind",
        "status",
        "created_at_utc",
        "cohort_root",
        "counts",
        "members",
        "members_sha256",
        "commit_sha256",
    }
    if (
        not isinstance(payload, dict)
        or set(payload) != fields
        or payload.get("schema_version") != 1
        or payload.get("contract") != campaign.COHORT_COMMIT_CONTRACT
        or payload.get("campaign_contract") != campaign.CAMPAIGN_CONTRACT
        or payload.get("status") != "complete_before_any_launch"
        or payload.get("commit_sha256") != _document_digest(payload, "commit_sha256")
        or payload.get("members_sha256") != _canonical_sha256(payload.get("members"))
        or sidecar_fields != [hashlib.sha256(encoded).hexdigest(), path.name]
    ):
        raise ValueError("Campaign cohort commit identity, schema, or digest is invalid.")
    _parse_timestamp(payload["created_at_utc"], "campaign cohort commit timestamp")
    return payload


def read_campaign_cohort(
    kind: str, *, root: Path | None = None
) -> ValidatedCampaignCohort:
    """Strictly authenticate one canonical atomic production campaign cohort."""

    root = (root or project_root()).resolve()
    cohort_root, expected_paths = _cohort_paths(kind, root)
    _require_no_symlink_components(cohort_root, "canonical campaign cohort")
    if cohort_root.is_symlink() or not cohort_root.is_dir():
        raise FileNotFoundError(f"Canonical campaign cohort is absent: {cohort_root}.")
    cohort_entries = list(cohort_root.rglob("*"))
    if any(entry.is_symlink() for entry in cohort_entries):
        raise ValueError("Canonical campaign cohort cannot contain symlinks.")
    if cohort_root.stat().st_mode & 0o222 or any(
        entry.stat().st_mode & 0o222 for entry in cohort_entries
    ):
        raise ValueError("Canonical campaign cohort must be recursively read-only.")
    commit_path = cohort_root / campaign.COHORT_COMMIT_FILENAME
    commit = _read_campaign_commit(commit_path)
    if (
        commit.get("cohort_kind") != kind
        or commit.get("cohort_root") != str(cohort_root)
        or commit.get("counts") != _expected_campaign_counts(kind)
    ):
        raise ValueError("Campaign cohort commit kind/root/counts drifted.")

    expected_entries = {
        campaign.COHORT_COMMIT_FILENAME,
        f"{campaign.COHORT_COMMIT_FILENAME}.sha256",
    }
    for path in expected_paths:
        expected_entries.update({path.name, f"{path.name}.sha256", f"{path.name}.snapshot"})
    if {entry.name for entry in cohort_root.iterdir()} != expected_entries:
        raise ValueError("Canonical campaign cohort has missing or unexpected top-level entries.")

    member_fields = {
        "index",
        "manifest_path",
        "manifest_sha256",
        "manifest_file_sha256",
        "manifest_sidecar_path",
        "manifest_sidecar_file_sha256",
        "snapshot_index_path",
        "snapshot_index_sha256",
        "snapshot_index_file_sha256",
    }
    members = commit.get("members")
    if not isinstance(members, list) or len(members) != len(expected_paths):
        raise ValueError("Campaign cohort commit member count drifted.")
    manifests: list[dict[str, Any]] = []
    for index, (path, member) in enumerate(zip(expected_paths, members, strict=True)):
        if not isinstance(member, Mapping) or set(member) != member_fields:
            raise ValueError("Campaign cohort member schema drifted.")
        sidecar = path.with_suffix(path.suffix + ".sha256")
        snapshot_index = Path(f"{path}.snapshot") / "index.json"
        if (
            member.get("index") != index
            or member.get("manifest_path") != str(path)
            or member.get("manifest_file_sha256") != _sha256_file(path)
            or member.get("manifest_sidecar_path") != str(sidecar)
            or member.get("manifest_sidecar_file_sha256") != _sha256_file(sidecar)
            or member.get("snapshot_index_path") != str(snapshot_index)
            or member.get("snapshot_index_file_sha256") != _sha256_file(snapshot_index)
        ):
            raise ValueError("Campaign cohort member file/path binding drifted.")
        manifest = read_manifest_for_audit(path, root)
        snapshot_bundle = manifest.get("snapshot_bundle")
        if (
            manifest.get("manifest_sha256") != member.get("manifest_sha256")
            or not isinstance(snapshot_bundle, Mapping)
            or snapshot_bundle.get("index_sha256")
            != member.get("snapshot_index_sha256")
        ):
            raise ValueError("Campaign cohort member semantic manifest binding drifted.")
        manifests.append(manifest)

    mapping = {path: manifest for path, manifest in zip(expected_paths, manifests, strict=True)}

    def reader(path: Path, _root: Path) -> Mapping[str, Any]:
        return deepcopy(mapping[path.resolve()])

    selection_binding: dict[str, Any] | None = None
    if kind == SEED_LADDER:
        result = campaign.validate_production_seed_ladder(
            expected_paths, root=root, manifest_reader=reader
        )
        if result.get("logical_rows") != campaign.EXPECTED_LADDER_MEDIA:
            raise ValueError("Seed-ladder production topology did not reauthenticate.")
    else:
        from hierasafe_flow.evaluation import finer_detailing_selection_cohort

        selection = finer_detailing_selection_cohort.read_selection_cohort(root=root)
        selection_paths = tuple(Path(path) for path in selection["selection_paths"])
        canonical_selections = finer_detailing_selection_cohort.canonical_selection_paths(
            root
        )
        if (
            selection_paths != canonical_selections
            or selection_paths != campaign.canonical_selection_paths(root)
            or selection.get("selection_count") != campaign.EXPECTED_SELECTION_RECORDS
        ):
            raise ValueError("Final launch requires the exact committed 36-selection cohort.")
        selection_binding = {
            "commit_path": selection["commit_path"],
            "commit_file_sha256": selection["commit_file_sha256"],
            "commit_sha256": selection["commit_sha256"],
            "selection_paths": list(selection["selection_paths"]),
        }
        result = campaign.validate_production_final_campaign(
            expected_paths,
            selection_paths,
            root=root,
            manifest_reader=reader,
        )
        if result.get("logical_rows") != campaign.EXPECTED_FINAL_LOGICAL:
            raise ValueError("Selected-seed production topology did not reauthenticate.")
    return ValidatedCampaignCohort(
        kind=kind,
        root=cohort_root,
        commit_path=commit_path,
        commit=commit,
        manifest_paths=tuple(expected_paths),
        manifests=tuple(manifests),
        digest=str(commit["commit_sha256"]),
        selection_cohort=selection_binding,
    )


def canonical_launch_root(cohort: ValidatedCampaignCohort) -> Path:
    return cohort.root.with_name(f"{cohort.root.name}.held-launch-{cohort.digest[:16]}")


def canonical_registry_path(cohort: ValidatedCampaignCohort, role: str) -> Path:
    expected_roles = {path.stem for path in cohort.manifest_paths}
    if role not in expected_roles or not re.fullmatch(r"[a-z0-9_]+", role):
        raise ValueError(f"Unknown or unsafe campaign manifest role {role!r}.")
    return canonical_launch_root(cohort) / "registries" / f"{role}.json"


def require_exact_registry_path(
    cohort: ValidatedCampaignCohort, role: str, raw: str | Path
) -> Path:
    return _require_exact_path(
        raw,
        canonical_registry_path(cohort, role),
        "canonical campaign registry path",
    )


def _launcher_bindings(root: Path) -> dict[str, dict[str, str]]:
    bindings: dict[str, dict[str, str]] = {}
    for name, relative in (
        ("sbatch", CAMPAIGN_SBATCH_LAUNCHER_RELATIVE),
        ("shell_dispatcher", CAMPAIGN_SHELL_DISPATCHER_RELATIVE),
        ("python_dispatcher", CAMPAIGN_PYTHON_DISPATCHER_RELATIVE),
        ("orchestrator", CAMPAIGN_ORCHESTRATOR_RELATIVE),
        ("runner", CAMPAIGN_RUNNER_RELATIVE),
        ("launch_module", CAMPAIGN_MODULE_RELATIVE),
    ):
        path = (root / relative).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Dedicated campaign launch file is absent: {path}.")
        bindings[name] = {"path": str(path), "file_sha256": _sha256_file(path)}
    return bindings


def _expected_array_spec(kind: str, path: Path) -> str:
    if kind == SEED_LADDER:
        return "0-35"
    return "0-7" if path.stem.endswith("__standard") else "0-5"


def _expected_job_name(kind: str, position: int, path: Path) -> str:
    if kind == SEED_LADDER:
        return f"fd-ladder-{position:02d}"
    family = "std" if path.stem.endswith("__standard") else "shp"
    return f"fd-final-{position:02d}-{family}"


def _sbatch_command(
    cohort: ValidatedCampaignCohort,
    *,
    role: str,
    path: Path,
    position: int,
    root: Path,
) -> list[str]:
    exported = {
        "FINER_DETAILING_CAMPAIGN_COHORT_KIND": cohort.kind,
        "FINER_DETAILING_CAMPAIGN_ROLE": role,
        "FINER_DETAILING_CAMPAIGN_REGISTRY": str(canonical_registry_path(cohort, role)),
    }
    if any(
        any(character in value for character in (",", "\n", "\r", "\0"))
        for value in exported.values()
    ):
        raise ValueError("Campaign launch environment contains an unsafe character.")
    export_value = ",".join(f"{name}={value}" for name, value in sorted(exported.items()))
    return [
        "sbatch",
        "--parsable",
        "--hold",
        "--array",
        _expected_array_spec(cohort.kind, path),
        "--job-name",
        _expected_job_name(cohort.kind, position, path),
        "--chdir",
        str(root),
        "--export",
        f"ALL,{export_value}",
        str((root / CAMPAIGN_SBATCH_LAUNCHER_RELATIVE).resolve()),
    ]


def _assert_fresh_outputs(cohort: ValidatedCampaignCohort) -> None:
    outputs: list[Path] = []
    for manifest in cohort.manifests:
        for job in manifest["jobs"]:
            raw = Path(str(job.get("output_dir", ""))).expanduser()
            if not raw.is_absolute() or raw.resolve(strict=False) != raw:
                raise ValueError(f"Campaign output attempt is noncanonical: {raw}.")
            outputs.append(raw)
    if len(outputs) != len(set(outputs)):
        raise ValueError("Campaign launch output-attempt union is not unique.")
    occupied = sorted(str(path) for path in outputs if path.exists() or path.is_symlink())
    if occupied:
        raise FileExistsError(f"Campaign launch output attempts already exist: {occupied}.")


def _campaign_launch_preview(
    kind: str,
    *,
    root: Path,
    cohort: ValidatedCampaignCohort | None = None,
    require_fresh_outputs: bool,
) -> dict[str, Any]:
    cohort = cohort or read_campaign_cohort(kind, root=root)
    launchers = _launcher_bindings(root)
    arrays: list[dict[str, Any]] = []
    for position, (path, manifest) in enumerate(
        zip(cohort.manifest_paths, cohort.manifests, strict=True)
    ):
        role = path.stem
        array_spec = _expected_array_spec(kind, path)
        expected_jobs = 36 if kind == SEED_LADDER else (8 if array_spec == "0-7" else 6)
        if len(manifest.get("jobs", ())) != expected_jobs:
            raise ValueError(f"Campaign role {role} has the wrong exact array size.")
        arrays.append(
            {
                "position": position,
                "role": role,
                "manifest_path": str(path),
                "manifest_file_sha256": _sha256_file(path),
                "manifest_sha256": manifest["manifest_sha256"],
                "array_spec": array_spec,
                "indices": list(range(expected_jobs)),
                "registry_path": str(canonical_registry_path(cohort, role)),
                "slurm_job_name": _expected_job_name(kind, position, path),
                "sbatch_command": _sbatch_command(
                    cohort, role=role, path=path, position=position, root=root
                ),
            }
        )
    expected_arrays = 8 if kind == SEED_LADDER else 72
    expected_jobs = 288 if kind == SEED_LADDER else 504
    if len(arrays) != expected_arrays or sum(len(row["indices"]) for row in arrays) != expected_jobs:
        raise ValueError("Campaign launch array/task topology is incomplete.")
    if require_fresh_outputs:
        _assert_fresh_outputs(cohort)
    return {
        "schema_version": 1,
        "contract": "finer_detailing_campaign_held_launch_preview_v1",
        "cohort_kind": kind,
        "cohort_root": str(cohort.root),
        "cohort_commit_path": str(cohort.commit_path),
        "cohort_commit_file_sha256": _sha256_file(cohort.commit_path),
        "cohort_commit_sha256": cohort.digest,
        "selection_cohort": deepcopy(cohort.selection_cohort),
        "launch_root": str(canonical_launch_root(cohort)),
        "launchers": launchers,
        "arrays": arrays,
        "array_count": expected_arrays,
        "logical_job_count": expected_jobs,
    }


def campaign_launch_preview(
    kind: str, *, root: Path | None = None
) -> dict[str, Any]:
    """Describe every exact held array without filesystem or scheduler mutation."""

    resolved_root = (root or project_root()).resolve()
    return _campaign_launch_preview(
        kind,
        root=resolved_root,
        require_fresh_outputs=True,
    )


def _state_digest(payload: Mapping[str, Any]) -> str:
    return _document_digest(payload, "state_sha256")


def _temporary(parent: Path, name: str, content: bytes) -> Path:
    descriptor, raw = tempfile.mkstemp(prefix=f".{name}.", suffix=".tmp", dir=parent)
    path = Path(raw)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(content)
        handle.flush()
        os.fsync(handle.fileno())
    return path


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    temporary = _temporary(path.parent, path.name, content)
    try:
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _publish_files_new(documents: Mapping[Path, bytes]) -> None:
    if not documents:
        raise ValueError("Immutable launch publication requires at least one file.")
    parents = {path.parent.resolve() for path in documents}
    if len(parents) != 1:
        raise ValueError("One immutable launch transaction must use one parent directory.")
    parent = next(iter(parents))
    parent.mkdir(parents=True, exist_ok=True)
    temporaries: dict[Path, Path] = {}
    published: list[tuple[Path, tuple[int, int]]] = []
    try:
        for destination, content in documents.items():
            if destination.parent.resolve() != parent:
                raise ValueError("Immutable launch destination escaped its parent.")
            temporaries[destination] = _temporary(parent, destination.name, content)
        for destination, temporary in temporaries.items():
            observed = temporary.stat()
            identity = (observed.st_dev, observed.st_ino)
            try:
                os.link(temporary, destination)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Refusing to overwrite immutable campaign launch file: {destination}."
                ) from exc
            published.append((destination, identity))
            descriptor = os.open(destination, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                linked = os.fstat(descriptor)
                if (linked.st_dev, linked.st_ino) != identity:
                    raise RuntimeError("Campaign launch file identity changed during publish.")
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        _fsync_directory(parent)
    except BaseException:
        for destination, identity in reversed(published):
            try:
                observed = destination.lstat()
            except FileNotFoundError:
                continue
            if not destination.is_symlink() and (observed.st_dev, observed.st_ino) == identity:
                destination.unlink()
        raise
    finally:
        for temporary in temporaries.values():
            temporary.unlink(missing_ok=True)


def _write_document_pair(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    frozen = deepcopy(dict(payload))
    frozen["document_sha256"] = _document_digest(frozen, "document_sha256")
    encoded = (json.dumps(frozen, indent=2, sort_keys=True) + "\n").encode()
    _publish_files_new(
        {
            path: encoded,
            path.with_suffix(path.suffix + ".sha256"): (
                f"{hashlib.sha256(encoded).hexdigest()}  {path.name}\n"
            ).encode(),
        }
    )
    return frozen


def _read_document_pair(path: Path, label: str) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.is_symlink() or sidecar.is_symlink():
        raise ValueError(f"{label} cannot be a symlink.")
    try:
        encoded = path.read_bytes()
        payload = json.loads(encoded)
        fields = sidecar.read_text(encoding="utf-8").split()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot authenticate {label} {path}: {exc}") from exc
    if (
        not isinstance(payload, dict)
        or payload.get("document_sha256") != _document_digest(payload, "document_sha256")
        or fields != [hashlib.sha256(encoded).hexdigest(), path.name]
    ):
        raise ValueError(f"{label} digest or raw sidecar is inconsistent.")
    return payload


def _state_payload(preview: Mapping[str, Any]) -> dict[str, Any]:
    now = _utc_now()
    payload: dict[str, Any] = {
        "schema_version": 1,
        "contract": LAUNCH_STATE_CONTRACT,
        "status": "active",
        "created_at_utc": now,
        "updated_at_utc": now,
        "cohort_kind": preview["cohort_kind"],
        "cohort_commit_sha256": preview["cohort_commit_sha256"],
        "selection_cohort": deepcopy(preview["selection_cohort"]),
        "launch_root": preview["launch_root"],
        "launchers": deepcopy(preview["launchers"]),
        "arrays": [
            {
                **deepcopy(row),
                "status": "pending",
                "slurm_array_job_id": None,
                "registry_sha256": None,
                "held_scheduler_identity": None,
                "error": None,
            }
            for row in preview["arrays"]
        ],
    }
    payload["state_sha256"] = _state_digest(payload)
    return payload


def _validate_state(state: Mapping[str, Any], preview: Mapping[str, Any]) -> dict[str, Any]:
    fields = {
        "schema_version",
        "contract",
        "status",
        "created_at_utc",
        "updated_at_utc",
        "cohort_kind",
        "cohort_commit_sha256",
        "selection_cohort",
        "launch_root",
        "launchers",
        "arrays",
        "state_sha256",
    }
    if (
        not isinstance(state, Mapping)
        or set(state) != fields
        or state.get("schema_version") != 1
        or state.get("contract") != LAUNCH_STATE_CONTRACT
        or state.get("status")
        not in {"active", "submission_outcome_unknown", "complete_held_before_publish"}
        or state.get("state_sha256") != _state_digest(state)
    ):
        raise ValueError("Campaign held-submission state schema/digest is invalid.")
    _parse_timestamp(state["created_at_utc"], "campaign launch state created_at")
    _parse_timestamp(state["updated_at_utc"], "campaign launch state updated_at")
    for key in (
        "cohort_kind",
        "cohort_commit_sha256",
        "selection_cohort",
        "launch_root",
        "launchers",
    ):
        if state.get(key) != preview.get(key):
            raise ValueError("Campaign launch state drifted from the authenticated preview.")
    arrays = state.get("arrays")
    dynamic = {
        "status",
        "slurm_array_job_id",
        "registry_sha256",
        "held_scheduler_identity",
        "error",
    }
    if not isinstance(arrays, list) or len(arrays) != len(preview["arrays"]):
        raise ValueError("Campaign launch state array count drifted.")
    for saved, expected in zip(arrays, preview["arrays"], strict=True):
        if not isinstance(saved, Mapping) or set(saved) != set(expected) | dynamic:
            raise ValueError("Campaign launch state row schema drifted.")
        if any(saved.get(key) != value for key, value in expected.items()):
            raise ValueError("Campaign launch state row differs from its exact array plan.")
        if saved.get("status") not in {
            "pending",
            "submitting",
            "registry_pending",
            "registered",
            "held_validated",
            "unknown",
        }:
            raise ValueError("Campaign launch state row status is invalid.")
        job_id = saved.get("slurm_array_job_id")
        if job_id is not None and _JOB_ID_RE.fullmatch(str(job_id)) is None:
            raise ValueError("Campaign launch state has an invalid Slurm array ID.")
        registry_sha = saved.get("registry_sha256")
        if registry_sha is not None and _SHA256_RE.fullmatch(str(registry_sha)) is None:
            raise ValueError("Campaign launch state has an invalid registry digest.")
        held = saved.get("held_scheduler_identity")
        if held is not None and not isinstance(held, Mapping):
            raise ValueError("Campaign launch state held identity must be an object or null.")
        if saved.get("error") is not None and not isinstance(saved.get("error"), Mapping):
            raise ValueError("Campaign launch state error must be an object or null.")
    return deepcopy(dict(state))


def _publish_state(path: Path, state: Mapping[str, Any]) -> dict[str, Any]:
    frozen = deepcopy(dict(state))
    frozen["updated_at_utc"] = _utc_now()
    frozen["state_sha256"] = _state_digest(frozen)
    _replace_json(path, frozen)
    reopened = json.loads(path.read_text(encoding="utf-8"))
    if reopened != frozen:
        raise RuntimeError("Campaign launch state failed durable reauthentication.")
    return frozen


def _load_state(path: Path, preview: Mapping[str, Any]) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot authenticate campaign launch state {path}: {exc}") from exc
    return _validate_state(payload, preview)


def _registry_schema_exact(registry: Mapping[str, Any]) -> bool:
    return set(registry) == _REGISTRY_TOP_FIELDS and all(
        isinstance(row, Mapping) and set(row) == _REGISTRY_ROW_FIELDS
        for row in registry.get("submissions", ())
    )


def _validate_registry(
    path: Path,
    *,
    canonical_path: Path,
    preview_row: Mapping[str, Any],
    manifest: Mapping[str, Any],
    expected_job_id: str | None = None,
) -> dict[str, Any]:
    registry = read_submission_registry(path)
    indices = list(preview_row["indices"])
    if (
        not _registry_schema_exact(registry)
        or registry.get("benchmark") != BENCHMARK_NAME
        or registry.get("manifest_path") != preview_row["manifest_path"]
        or registry.get("manifest_sha256") != preview_row["manifest_sha256"]
        or registry.get("array_spec") != preview_row["array_spec"]
        or registry.get("slurm_job_name") != preview_row["slurm_job_name"]
        or registry.get("num_registered_tasks") != len(indices)
        or [row.get("job_index") for row in registry.get("submissions", ())] != indices
        or [row.get("slurm_array_task_id") for row in registry.get("submissions", ())]
        != indices
        or expected_job_id is not None
        and str(registry.get("slurm_array_job_id")) != expected_job_id
        or len(manifest.get("jobs", ())) != len(indices)
    ):
        raise ValueError(f"Campaign registry does not cover its exact array: {canonical_path}.")
    job_id = str(registry["slurm_array_job_id"])
    if any(
        row.get("slurm_array_job_id") != job_id
        or row.get("slurm_task_id") != f"{job_id}_{index}"
        for index, row in enumerate(registry["submissions"])
    ):
        raise ValueError("Campaign registry composite task identity drifted.")
    _parse_timestamp(registry["created_at_utc"], "campaign registry created_at")
    return registry


def _parse_sbatch_job_id(stdout: str) -> str:
    raw = stdout.strip()
    job_id = raw.split(";", 1)[0]
    if _JOB_ID_RE.fullmatch(job_id) is None or "\n" in raw:
        raise ValueError(f"Unexpected sbatch --parsable response: {stdout!r}.")
    return job_id


def _parse_scontrol_record(stdout: str, label: str) -> dict[str, str]:
    lines = [line.strip() for line in str(stdout).splitlines() if line.strip()]
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


def _run_scontrol(
    job_id: str, *, root: Path, run: CommandRunner, label: str
) -> dict[str, str]:
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
        raise RuntimeError(f"Cannot authenticate {label} through Slurm.") from exc
    return _parse_scontrol_record(str(result.stdout), label)


def _validate_held_array(
    row: Mapping[str, Any], *, job_id: str, root: Path, run: CommandRunner
) -> dict[str, Any]:
    record = _run_scontrol(job_id, root=root, run=run, label="held campaign array")
    expected = {
        "JobId": job_id,
        "ArrayJobId": job_id,
        "ArrayTaskId": str(row["array_spec"]),
        "JobName": str(row["slurm_job_name"]),
        "JobState": "PENDING",
        "Reason": "JobHeldUser",
        "Command": str(row["sbatch_command"][-1]),
        "WorkDir": str(root),
        "BatchFlag": "1",
    }
    drift = {
        key: {"expected": value, "actual": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if drift:
        raise RuntimeError(f"Held campaign array scheduler identity drifted: {drift}.")
    return {
        "slurm_array_job_id": job_id,
        "array_spec": row["array_spec"],
        "slurm_job_name": row["slurm_job_name"],
        "slurm_job_state": "PENDING",
        "reason": "JobHeldUser",
        "launcher_path": row["sbatch_command"][-1],
        "work_dir": str(root),
        "batch_flag": 1,
    }


def _staging_root(cohort: ValidatedCampaignCohort) -> Path:
    target = canonical_launch_root(cohort)
    return target.parent / f".{target.name}.held-staging"


def _lock_path(cohort: ValidatedCampaignCohort) -> Path:
    target = canonical_launch_root(cohort)
    return target.parent / f".{target.name}.orchestrator.lock"


def _ensure_staging(
    cohort: ValidatedCampaignCohort, preview: Mapping[str, Any]
) -> Path:
    staging = _staging_root(cohort)
    if staging.exists() or staging.is_symlink():
        if staging.is_symlink() or not staging.is_dir():
            raise ValueError(f"Campaign held-staging path is not a real directory: {staging}.")
        _load_state(staging / LAUNCH_STATE_FILENAME, preview)
        return staging
    staging.parent.mkdir(parents=True, exist_ok=True)
    try:
        staging.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise RuntimeError(f"Competing campaign launch publisher claimed {staging}.") from exc
    (staging / "registries").mkdir(mode=0o700)
    _replace_json(staging / LAUNCH_STATE_FILENAME, _state_payload(preview))
    _fsync_directory(staging)
    return staging


def _freeze_tree(root: Path) -> None:
    paths = list(root.rglob("*"))
    if any(path.is_symlink() for path in paths):
        raise ValueError("Immutable campaign launch bundle cannot contain symlinks.")
    for path in paths:
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for path in sorted(
        (candidate for candidate in paths if candidate.is_dir()),
        key=lambda candidate: len(candidate.parts),
        reverse=True,
    ):
        path.chmod(0o555)
        _fsync_directory(path)
    root.chmod(0o555)
    _fsync_directory(root)


def _identity(path: Path) -> tuple[int, int]:
    observed = path.lstat()
    return observed.st_dev, observed.st_ino


def _same_identity(path: Path, expected: tuple[int, int]) -> bool:
    try:
        return not path.is_symlink() and _identity(path) == expected
    except FileNotFoundError:
        return False


def _link_new(source: Path, destination: Path) -> tuple[int, int]:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"Campaign launch hard-link source is not a physical file: {source}.")
    source_identity = _identity(source)
    try:
        os.link(source, destination, follow_symlinks=False)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Refusing to replace campaign launch destination: {destination}."
        ) from exc
    except BaseException:
        if _same_identity(destination, source_identity):
            destination.unlink()
        raise
    descriptor = os.open(destination, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        linked = os.fstat(descriptor)
        if (linked.st_dev, linked.st_ino) != source_identity:
            raise RuntimeError("Campaign launch hard-link identity changed during publication.")
        if linked.st_mode & 0o222:
            raise RuntimeError("Campaign launch member was writable during publication.")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    return source_identity


def _cleanup_uncommitted_launch_claim(
    root: Path,
    *,
    root_identity: tuple[int, int],
    registry_identity: tuple[int, int] | None,
    linked: Mapping[Path, tuple[int, int]],
) -> None:
    """Remove only our inodes from a caught, not-yet-admitted publication."""

    if _same_identity(root, root_identity):
        root.chmod(0o700)
    registries = root / "registries"
    if registry_identity is not None and _same_identity(registries, registry_identity):
        registries.chmod(0o700)
    for destination, expected in reversed(tuple(linked.items())):
        if _same_identity(destination, expected):
            destination.unlink()
    if registry_identity is not None and _same_identity(registries, registry_identity):
        try:
            registries.rmdir()
        except OSError:
            pass
    if _same_identity(root, root_identity):
        try:
            root.rmdir()
        except OSError:
            pass


def _remove_staging_links(staging: Path) -> None:
    """Unlink a successfully published hidden tree without chmodding shared files."""

    for directory in sorted(
        (path for path in staging.rglob("*") if path.is_dir()),
        key=lambda path: len(path.parts),
        reverse=True,
    ):
        directory.chmod(0o700)
    staging.chmod(0o700)
    for path in sorted(staging.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_file():
            path.unlink()
        elif path.is_dir():
            path.rmdir()
    staging.rmdir()


def _thaw_staging_directories(staging: Path) -> None:
    for path in staging.rglob("*"):
        if path.is_dir():
            path.chmod(0o700)
    staging.chmod(0o700)


def _reset_staged_launch_documents(staging: Path) -> None:
    """Keep held state/registries resumable while discarding an unadmitted commit."""

    for name in (
        LAUNCH_COMMIT_FILENAME,
        f"{LAUNCH_COMMIT_FILENAME}.sha256",
        RELEASE_RECEIPT_FILENAME,
        f"{RELEASE_RECEIPT_FILENAME}.sha256",
    ):
        path = staging / name
        if path.is_symlink():
            raise RuntimeError("Cannot reset a symlinked staged campaign launch document.")
        path.unlink(missing_ok=True)
    _fsync_directory(staging)


def _task_union(
    preview: Mapping[str, Any],
    registries: Mapping[str, Mapping[str, Any]],
    manifests: Mapping[str, Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    condition_ids: set[str] = set()
    output_dirs: set[str] = set()
    task_ids: set[str] = set()
    for array in preview["arrays"]:
        registry = registries[str(array["role"])]
        manifest = manifests[str(array["manifest_path"])]
        for index, (job, submission) in enumerate(
            zip(manifest["jobs"], registry["submissions"], strict=True)
        ):
            condition_id = str(job["condition_id"])
            output_dir = str(job["output_dir"])
            task_id = str(submission["slurm_task_id"])
            if (
                condition_id in condition_ids
                or output_dir in output_dirs
                or task_id in task_ids
                or submission["job_index"] != index
            ):
                raise ValueError("Campaign launch task union contains duplicate identities.")
            condition_ids.add(condition_id)
            output_dirs.add(output_dir)
            task_ids.add(task_id)
            rows.append(
                {
                    "manifest_role": array["role"],
                    "manifest_sha256": array["manifest_sha256"],
                    "manifest_job_index": index,
                    "condition_id": condition_id,
                    "output_dir": output_dir,
                    "slurm_array_job_id": registry["slurm_array_job_id"],
                    "slurm_array_task_id": index,
                    "slurm_task_id": task_id,
                }
            )
    if len(rows) != preview["logical_job_count"]:
        raise ValueError("Campaign launch task union is incomplete.")
    return rows


def _build_launch_documents(
    *,
    cohort: ValidatedCampaignCohort,
    preview: Mapping[str, Any],
    state: Mapping[str, Any],
    storage_root: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    state = _validate_state(state, preview)
    if state.get("status") != "complete_held_before_publish" or any(
        row.get("status") != "held_validated" for row in state["arrays"]
    ):
        raise ValueError("Campaign launch commit requires every exact array held and validated.")
    manifest_lookup = {
        str(path): manifest
        for path, manifest in zip(cohort.manifest_paths, cohort.manifests, strict=True)
    }
    registries: dict[str, dict[str, Any]] = {}
    arrays: list[dict[str, Any]] = []
    job_ids: list[str] = []
    for planned, saved in zip(preview["arrays"], state["arrays"], strict=True):
        role = str(planned["role"])
        physical_registry = storage_root / "registries" / f"{role}.json"
        registry = _validate_registry(
            physical_registry,
            canonical_path=Path(planned["registry_path"]),
            preview_row=planned,
            manifest=manifest_lookup[str(planned["manifest_path"])],
            expected_job_id=str(saved["slurm_array_job_id"]),
        )
        if registry["registry_sha256"] != saved["registry_sha256"]:
            raise ValueError("Campaign state and immutable registry digest differ.")
        registries[role] = registry
        job_id = str(registry["slurm_array_job_id"])
        job_ids.append(job_id)
        arrays.append(
            {
                **deepcopy(dict(planned)),
                "slurm_array_job_id": job_id,
                "held_scheduler_identity": deepcopy(saved["held_scheduler_identity"]),
                "registry": {
                    "path": planned["registry_path"],
                    "file_sha256": _sha256_file(physical_registry),
                    "registry_sha256": registry["registry_sha256"],
                },
            }
        )
    if len(job_ids) != preview["array_count"] or len(set(job_ids)) != len(job_ids):
        raise ValueError("Every campaign role must have one unique held Slurm array ID.")
    tasks = _task_union(preview, registries, manifest_lookup)
    state_path = storage_root / LAUNCH_STATE_FILENAME
    commit: dict[str, Any] = {
        "schema_version": 1,
        "contract": LAUNCH_COMMIT_CONTRACT,
        "status": "complete_held_arrays_committed_before_release",
        "created_at_utc": _utc_now(),
        "cohort": {
            "kind": cohort.kind,
            "root": str(cohort.root),
            "commit_path": str(cohort.commit_path),
            "commit_file_sha256": _sha256_file(cohort.commit_path),
            "commit_sha256": cohort.digest,
        },
        "selection_cohort": deepcopy(cohort.selection_cohort),
        "launch_root": str(canonical_launch_root(cohort)),
        "launchers": deepcopy(preview["launchers"]),
        "held_state": {
            "path": str(canonical_launch_root(cohort) / LAUNCH_STATE_FILENAME),
            "file_sha256": _sha256_file(state_path),
            "state_sha256": state["state_sha256"],
        },
        "arrays": arrays,
        "array_count": preview["array_count"],
        "logical_job_count": preview["logical_job_count"],
        "tasks": tasks,
        "task_union_sha256": _canonical_sha256(tasks),
        "release_command": ["scontrol", "release", ",".join(job_ids)],
    }
    commit["document_sha256"] = _document_digest(commit, "document_sha256")
    receipt: dict[str, Any] = {
        "schema_version": 1,
        "contract": RELEASE_RECEIPT_CONTRACT,
        "status": "immutable_release_authorization_committed_before_scontrol",
        "created_at_utc": _utc_now(),
        "cohort_kind": cohort.kind,
        "cohort_commit_sha256": cohort.digest,
        "selection_cohort": deepcopy(cohort.selection_cohort),
        "launch_commit": {
            "path": str(canonical_launch_root(cohort) / LAUNCH_COMMIT_FILENAME),
            "document_sha256": commit["document_sha256"],
        },
        "slurm_array_job_ids": job_ids,
        "release_command": deepcopy(commit["release_command"]),
    }
    receipt["document_sha256"] = _document_digest(receipt, "document_sha256")
    return commit, receipt


def _expected_launch_entries(preview: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    root_entries = {
        "registries",
        LAUNCH_STATE_FILENAME,
        LAUNCH_COMMIT_FILENAME,
        f"{LAUNCH_COMMIT_FILENAME}.sha256",
        RELEASE_RECEIPT_FILENAME,
        f"{RELEASE_RECEIPT_FILENAME}.sha256",
    }
    registry_entries = {f"{row['role']}.json" for row in preview["arrays"]}
    return root_entries, registry_entries


def canonical_launch_commit_path(cohort: ValidatedCampaignCohort) -> Path:
    return canonical_launch_root(cohort) / LAUNCH_COMMIT_FILENAME


def canonical_release_receipt_path(cohort: ValidatedCampaignCohort) -> Path:
    return canonical_launch_root(cohort) / RELEASE_RECEIPT_FILENAME


def _expected_held_identity(
    row: Mapping[str, Any], *, job_id: str, root: Path
) -> dict[str, Any]:
    return {
        "slurm_array_job_id": job_id,
        "array_spec": row["array_spec"],
        "slurm_job_name": row["slurm_job_name"],
        "slurm_job_state": "PENDING",
        "reason": "JobHeldUser",
        "launcher_path": row["sbatch_command"][-1],
        "work_dir": str(root),
        "batch_flag": 1,
    }


def read_campaign_launch_bundle(
    kind: str, *, root: Path | None = None
) -> dict[str, Any]:
    """Authenticate one complete all-held campaign launch publication."""

    root = (root or project_root()).resolve()
    cohort = read_campaign_cohort(kind, root=root)
    preview = _campaign_launch_preview(
        kind,
        root=root,
        cohort=cohort,
        require_fresh_outputs=False,
    )
    launch_root = canonical_launch_root(cohort)
    _require_no_symlink_components(launch_root, "canonical campaign launch bundle")
    if launch_root.is_symlink() or not launch_root.is_dir():
        raise FileNotFoundError(f"Canonical campaign launch bundle is absent: {launch_root}.")
    descendants = list(launch_root.rglob("*"))
    if any(path.is_symlink() for path in descendants):
        raise ValueError("Canonical campaign launch bundle cannot contain symlinks.")
    if launch_root.stat().st_mode & 0o222 or any(
        path.stat().st_mode & 0o222 for path in descendants
    ):
        raise ValueError("Canonical campaign launch bundle must be recursively read-only.")
    root_entries, registry_entries = _expected_launch_entries(preview)
    if {path.name for path in launch_root.iterdir()} != root_entries:
        raise ValueError("Canonical campaign launch bundle has missing or unexpected entries.")
    registries_root = launch_root / "registries"
    if not registries_root.is_dir() or {
        path.name for path in registries_root.iterdir()
    } != registry_entries:
        raise ValueError("Canonical campaign registry cohort is incomplete or contains extras.")

    state_path = launch_root / LAUNCH_STATE_FILENAME
    state = _load_state(state_path, preview)
    if state.get("status") != "complete_held_before_publish" or any(
        row.get("status") != "held_validated" for row in state["arrays"]
    ):
        raise ValueError("Published campaign launch state is not complete and held-validated.")
    commit_path = canonical_launch_commit_path(cohort)
    receipt_path = canonical_release_receipt_path(cohort)
    commit = _read_document_pair(commit_path, "campaign launch commit")
    receipt = _read_document_pair(receipt_path, "campaign held-release receipt")
    commit_fields = {
        "schema_version",
        "contract",
        "status",
        "created_at_utc",
        "cohort",
        "selection_cohort",
        "launch_root",
        "launchers",
        "held_state",
        "arrays",
        "array_count",
        "logical_job_count",
        "tasks",
        "task_union_sha256",
        "release_command",
        "document_sha256",
    }
    expected_cohort = {
        "kind": cohort.kind,
        "root": str(cohort.root),
        "commit_path": str(cohort.commit_path),
        "commit_file_sha256": _sha256_file(cohort.commit_path),
        "commit_sha256": cohort.digest,
    }
    if (
        set(commit) != commit_fields
        or commit.get("schema_version") != 1
        or commit.get("contract") != LAUNCH_COMMIT_CONTRACT
        or commit.get("status") != "complete_held_arrays_committed_before_release"
        or commit.get("cohort") != expected_cohort
        or commit.get("selection_cohort") != cohort.selection_cohort
        or commit.get("launch_root") != str(launch_root)
        or commit.get("launchers") != preview["launchers"]
        or commit.get("held_state")
        != {
            "path": str(state_path),
            "file_sha256": _sha256_file(state_path),
            "state_sha256": state["state_sha256"],
        }
        or commit.get("array_count") != preview["array_count"]
        or commit.get("logical_job_count") != preview["logical_job_count"]
    ):
        raise ValueError("Campaign launch commit cohort/source/state identity drifted.")
    _parse_timestamp(commit["created_at_utc"], "campaign launch commit timestamp")

    manifests = {
        str(path): manifest
        for path, manifest in zip(cohort.manifest_paths, cohort.manifests, strict=True)
    }
    array_fields = set(preview["arrays"][0]) | {
        "slurm_array_job_id",
        "held_scheduler_identity",
        "registry",
    }
    saved_arrays = commit.get("arrays")
    if not isinstance(saved_arrays, list) or len(saved_arrays) != preview["array_count"]:
        raise ValueError("Campaign launch commit array coverage drifted.")
    registries: dict[str, dict[str, Any]] = {}
    job_ids: list[str] = []
    for saved, planned, state_row in zip(
        saved_arrays, preview["arrays"], state["arrays"], strict=True
    ):
        if not isinstance(saved, Mapping) or set(saved) != array_fields:
            raise ValueError("Campaign launch commit array schema drifted.")
        role = str(planned["role"])
        registry_path = canonical_registry_path(cohort, role)
        if str(registry_path) != planned["registry_path"]:
            raise ValueError("Campaign launch preview registry routing drifted.")
        _require_no_symlink_components(registry_path, "campaign registry")
        job_id = str(saved.get("slurm_array_job_id", ""))
        registry = _validate_registry(
            registry_path,
            canonical_path=registry_path,
            preview_row=planned,
            manifest=manifests[str(planned["manifest_path"])],
            expected_job_id=job_id,
        )
        held = _expected_held_identity(planned, job_id=job_id, root=root)
        expected_saved = {
            **deepcopy(dict(planned)),
            "slurm_array_job_id": job_id,
            "held_scheduler_identity": held,
            "registry": {
                "path": str(registry_path),
                "file_sha256": _sha256_file(registry_path),
                "registry_sha256": registry["registry_sha256"],
            },
        }
        if (
            dict(saved) != expected_saved
            or state_row.get("slurm_array_job_id") != job_id
            or state_row.get("registry_sha256") != registry["registry_sha256"]
            or state_row.get("held_scheduler_identity") != held
        ):
            raise ValueError("Campaign launch array/registry/held identity drifted.")
        registries[role] = registry
        job_ids.append(job_id)
    if len(set(job_ids)) != len(job_ids) or any(
        _JOB_ID_RE.fullmatch(job_id) is None for job_id in job_ids
    ):
        raise ValueError("Campaign launch does not bind one unique numeric ID per array.")

    tasks = _task_union(preview, registries, manifests)
    release_command = ["scontrol", "release", ",".join(job_ids)]
    if (
        commit.get("tasks") != tasks
        or commit.get("task_union_sha256") != _canonical_sha256(tasks)
        or commit.get("release_command") != release_command
    ):
        raise ValueError("Campaign launch task union or sole release command drifted.")
    receipt_fields = {
        "schema_version",
        "contract",
        "status",
        "created_at_utc",
        "cohort_kind",
        "cohort_commit_sha256",
        "selection_cohort",
        "launch_commit",
        "slurm_array_job_ids",
        "release_command",
        "document_sha256",
    }
    if (
        set(receipt) != receipt_fields
        or receipt.get("schema_version") != 1
        or receipt.get("contract") != RELEASE_RECEIPT_CONTRACT
        or receipt.get("status")
        != "immutable_release_authorization_committed_before_scontrol"
        or receipt.get("cohort_kind") != kind
        or receipt.get("cohort_commit_sha256") != cohort.digest
        or receipt.get("selection_cohort") != cohort.selection_cohort
        or receipt.get("launch_commit")
        != {"path": str(commit_path), "document_sha256": commit["document_sha256"]}
        or receipt.get("slurm_array_job_ids") != job_ids
        or receipt.get("release_command") != release_command
    ):
        raise ValueError("Campaign held-release receipt identity/coverage drifted.")
    _parse_timestamp(receipt["created_at_utc"], "campaign release receipt timestamp")
    return {
        "cohort": cohort,
        "preview": preview,
        "state": state,
        "registries": registries,
        "commit": commit,
        "receipt": receipt,
    }


def _validate_precommit_launch_tree(
    *,
    cohort: ValidatedCampaignCohort,
    preview: Mapping[str, Any],
    destination: Path,
    staging: Path,
    root_identity: tuple[int, int],
    registry_identity: tuple[int, int],
    linked: Mapping[Path, tuple[int, int]],
    expected_state: Mapping[str, Any],
    expected_receipt: Mapping[str, Any],
) -> None:
    if not _same_identity(destination, root_identity) or not _same_identity(
        destination / "registries", registry_identity
    ):
        raise RuntimeError("Campaign launch directory identity changed before commit.")
    root_entries, registry_entries = _expected_launch_entries(preview)
    expected_precommit = root_entries - {LAUNCH_COMMIT_FILENAME}
    if {path.name for path in destination.iterdir()} != expected_precommit or {
        path.name for path in (destination / "registries").iterdir()
    } != registry_entries:
        raise RuntimeError("Campaign launch precommit tree is incomplete or contains extras.")
    for path, identity in linked.items():
        if not _same_identity(path, identity) or path.stat().st_mode & 0o222:
            raise RuntimeError("Campaign launch member changed before commit publication.")
    state = _load_state(destination / LAUNCH_STATE_FILENAME, preview)
    if state != expected_state:
        raise RuntimeError("Campaign launch state changed during commit-last publication.")
    receipt = _read_document_pair(
        destination / RELEASE_RECEIPT_FILENAME,
        "precommit campaign release receipt",
    )
    if receipt != expected_receipt:
        raise RuntimeError("Campaign release receipt changed before launch commit.")
    manifest_lookup = {
        str(path): manifest
        for path, manifest in zip(cohort.manifest_paths, cohort.manifests, strict=True)
    }
    for planned, saved in zip(preview["arrays"], state["arrays"], strict=True):
        role = str(planned["role"])
        registry = _validate_registry(
            destination / "registries" / f"{role}.json",
            canonical_path=Path(planned["registry_path"]),
            preview_row=planned,
            manifest=manifest_lookup[str(planned["manifest_path"])],
            expected_job_id=str(saved["slurm_array_job_id"]),
        )
        if registry["registry_sha256"] != saved["registry_sha256"]:
            raise RuntimeError("Campaign registry changed before launch commit.")
    sidecar_name = f"{LAUNCH_COMMIT_FILENAME}.sha256"
    if (destination / sidecar_name).read_bytes() != (staging / sidecar_name).read_bytes():
        raise RuntimeError("Campaign launch commit sidecar changed before commit.")


def _freeze_published_launch_directories(
    destination: Path,
    *,
    root_identity: tuple[int, int],
    registry_identity: tuple[int, int],
) -> None:
    registries = destination / "registries"
    if not _same_identity(destination, root_identity) or not _same_identity(
        registries, registry_identity
    ):
        raise RuntimeError("Campaign launch directories changed before final freeze.")
    registries.chmod(0o555)
    _fsync_directory(registries)
    destination.chmod(0o555)
    _fsync_directory(destination)


def _publish_launch_commit_last(
    *,
    cohort: ValidatedCampaignCohort,
    preview: Mapping[str, Any],
    staging: Path,
    destination: Path,
    state: Mapping[str, Any],
    commit: Mapping[str, Any],
    receipt: Mapping[str, Any],
    root: Path,
) -> dict[str, Any]:
    """Publish by O_EXCL hard links; the self-authenticating commit appears last."""

    root_entries, registry_entries = _expected_launch_entries(preview)
    if {path.name for path in staging.iterdir()} != root_entries or {
        path.name for path in (staging / "registries").iterdir()
    } != registry_entries:
        raise RuntimeError("Campaign launch hidden stage is incomplete before publication.")
    descendants = list(staging.rglob("*"))
    if any(path.is_symlink() for path in descendants) or any(
        path.is_file() and path.stat().st_mode & 0o222 for path in descendants
    ):
        raise ValueError("Campaign launch hidden stage is not immutable and physical.")
    if _load_state(staging / LAUNCH_STATE_FILENAME, preview) != state:
        raise RuntimeError("Campaign launch hidden state changed before publication.")
    if _read_document_pair(
        staging / LAUNCH_COMMIT_FILENAME, "staged campaign launch commit"
    ) != commit or _read_document_pair(
        staging / RELEASE_RECEIPT_FILENAME, "staged campaign release receipt"
    ) != receipt:
        raise RuntimeError("Campaign launch hidden documents changed before publication.")

    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Campaign launch destination already exists: {destination}.")
    _require_no_symlink_components(destination.parent, "campaign launch parent")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        destination.mkdir(mode=0o700)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Competing campaign launch publisher claimed {destination}."
        ) from exc
    root_identity = _identity(destination)
    registry_identity: tuple[int, int] | None = None
    linked: dict[Path, tuple[int, int]] = {}
    admitted = False
    try:
        registries = destination / "registries"
        registries.mkdir(mode=0o700)
        registry_identity = _identity(registries)

        commit_sidecar = f"{LAUNCH_COMMIT_FILENAME}.sha256"
        root_members = [
            commit_sidecar,
            LAUNCH_STATE_FILENAME,
            RELEASE_RECEIPT_FILENAME,
            f"{RELEASE_RECEIPT_FILENAME}.sha256",
        ]
        for name in root_members:
            if not _same_identity(destination, root_identity):
                raise RuntimeError("Campaign launch root changed during member linking.")
            target = destination / name
            linked[target] = _link_new(staging / name, target)
        for name in sorted(registry_entries):
            if not _same_identity(registries, registry_identity):
                raise RuntimeError("Campaign registry directory changed during linking.")
            target = registries / name
            linked[target] = _link_new(staging / "registries" / name, target)
        _fsync_directory(registries)
        _fsync_directory(destination)
        _validate_precommit_launch_tree(
            cohort=cohort,
            preview=preview,
            destination=destination,
            staging=staging,
            root_identity=root_identity,
            registry_identity=registry_identity,
            linked=linked,
            expected_state=state,
            expected_receipt=receipt,
        )
        commit_target = destination / LAUNCH_COMMIT_FILENAME
        commit_source = staging / LAUNCH_COMMIT_FILENAME
        commit_identity = _identity(commit_source)
        try:
            linked[commit_target] = _link_new(commit_source, commit_target)
        except BaseException:
            if _same_identity(commit_target, commit_identity):
                linked[commit_target] = commit_identity
            raise
        _fsync_directory(destination)
        _freeze_published_launch_directories(
            destination,
            root_identity=root_identity,
            registry_identity=registry_identity,
        )
        reopened = read_campaign_launch_bundle(cohort.kind, root=root)
        if reopened["commit"] != commit or reopened["receipt"] != receipt:
            raise RuntimeError("Commit-last campaign launch failed reauthentication.")
        admitted = True
        _remove_staging_links(staging)
        return reopened
    except BaseException:
        if not admitted:
            _cleanup_uncommitted_launch_claim(
                destination,
                root_identity=root_identity,
                registry_identity=registry_identity,
                linked=linked,
            )
            _thaw_staging_directories(staging)
            _reset_staged_launch_documents(staging)
        raise


def orchestrate_campaign_launch(
    kind: str,
    *,
    root: Path | None = None,
    run: CommandRunner = subprocess.run,
    registry_writer: Callable[[Mapping[str, Any], Path], None] = write_submission_registry,
) -> dict[str, Any]:
    """Submit every exact array held, atomically commit, then release exactly once."""

    root = (root or project_root()).resolve()
    cohort = read_campaign_cohort(kind, root=root)
    preview = campaign_launch_preview(kind, root=root)
    launch_root = canonical_launch_root(cohort)
    lock_path = _lock_path(cohort)
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as lock_handle:
        try:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another campaign launch orchestrator holds the cohort lock.") from exc
        if launch_root.exists() or launch_root.is_symlink():
            read_campaign_launch_bundle(kind, root=root)
            raise RuntimeError(
                "Immutable campaign launch bundle already exists; refusing scheduler replay."
            )
        staging = _ensure_staging(cohort, preview)
        state_path = staging / LAUNCH_STATE_FILENAME
        state = _load_state(state_path, preview)
        if state["status"] == "submission_outcome_unknown" or any(
            row["status"] in {"submitting", "unknown"} for row in state["arrays"]
        ):
            raise RuntimeError(
                "A campaign sbatch outcome is ambiguous; reconcile held jobs and never replay."
            )
        manifest_lookup = {
            str(path): manifest
            for path, manifest in zip(cohort.manifest_paths, cohort.manifests, strict=True)
        }
        for position, planned in enumerate(preview["arrays"]):
            saved = state["arrays"][position]
            role = str(planned["role"])
            physical_registry = staging / "registries" / f"{role}.json"
            manifest = manifest_lookup[str(planned["manifest_path"])]
            if saved["status"] in {"registered", "held_validated"}:
                registry = _validate_registry(
                    physical_registry,
                    canonical_path=Path(planned["registry_path"]),
                    preview_row=planned,
                    manifest=manifest,
                    expected_job_id=str(saved["slurm_array_job_id"]),
                )
                if registry["registry_sha256"] != saved["registry_sha256"]:
                    raise ValueError("Registered campaign state differs from its registry.")
                continue
            if saved["status"] == "registry_pending":
                if physical_registry.exists():
                    registry = _validate_registry(
                        physical_registry,
                        canonical_path=Path(planned["registry_path"]),
                        preview_row=planned,
                        manifest=manifest,
                        expected_job_id=str(saved["slurm_array_job_id"]),
                    )
                else:
                    registry = build_submission_registry(
                        manifest=manifest,
                        manifest_path=Path(planned["manifest_path"]),
                        slurm_array_job_id=str(saved["slurm_array_job_id"]),
                        array_spec=str(planned["array_spec"]),
                        slurm_job_name=str(planned["slurm_job_name"]),
                    )
                    registry_writer(registry, physical_registry)
                    registry = _validate_registry(
                        physical_registry,
                        canonical_path=Path(planned["registry_path"]),
                        preview_row=planned,
                        manifest=manifest,
                        expected_job_id=str(saved["slurm_array_job_id"]),
                    )
                saved["status"] = "registered"
                saved["registry_sha256"] = registry["registry_sha256"]
                state = _publish_state(state_path, state)
                continue
            if saved["status"] != "pending":
                raise RuntimeError(f"Campaign role {role} cannot resume from {saved['status']!r}.")

            current = read_campaign_cohort(kind, root=root)
            current_preview = campaign_launch_preview(kind, root=root)
            if current.digest != cohort.digest or current_preview != preview:
                raise RuntimeError("Campaign cohort or dedicated launch source changed before sbatch.")
            _assert_fresh_outputs(current)
            saved["status"] = "submitting"
            state = _publish_state(state_path, state)
            try:
                result = run(
                    planned["sbatch_command"],
                    cwd=root,
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=120,
                )
                job_id = _parse_sbatch_job_id(str(result.stdout))
            except BaseException as exc:
                saved = state["arrays"][position]
                saved["status"] = "unknown"
                saved["error"] = {"type": type(exc).__name__, "message": str(exc)[:2000]}
                state["status"] = "submission_outcome_unknown"
                _publish_state(state_path, state)
                raise RuntimeError(
                    f"Campaign sbatch outcome for role {role!r} is ambiguous; refusing replay."
                ) from exc
            existing_ids = {
                str(row["slurm_array_job_id"])
                for row in state["arrays"]
                if row["slurm_array_job_id"] is not None
            }
            if job_id in existing_ids:
                raise RuntimeError("Slurm returned one array ID for multiple campaign roles.")
            saved = state["arrays"][position]
            saved["status"] = "registry_pending"
            saved["slurm_array_job_id"] = job_id
            state = _publish_state(state_path, state)
            registry = build_submission_registry(
                manifest=manifest,
                manifest_path=Path(planned["manifest_path"]),
                slurm_array_job_id=job_id,
                array_spec=str(planned["array_spec"]),
                slurm_job_name=str(planned["slurm_job_name"]),
            )
            registry_writer(registry, physical_registry)
            registry = _validate_registry(
                physical_registry,
                canonical_path=Path(planned["registry_path"]),
                preview_row=planned,
                manifest=manifest,
                expected_job_id=job_id,
            )
            saved = state["arrays"][position]
            saved["status"] = "registered"
            saved["registry_sha256"] = registry["registry_sha256"]
            state = _publish_state(state_path, state)

        _assert_fresh_outputs(cohort)
        for position, planned in enumerate(preview["arrays"]):
            saved = state["arrays"][position]
            held = _validate_held_array(
                planned,
                job_id=str(saved["slurm_array_job_id"]),
                root=root,
                run=run,
            )
            saved["status"] = "held_validated"
            saved["held_scheduler_identity"] = held
            state = _publish_state(state_path, state)
        state["status"] = "complete_held_before_publish"
        state = _publish_state(state_path, state)
        commit, receipt = _build_launch_documents(
            cohort=cohort,
            preview=preview,
            state=state,
            storage_root=staging,
        )
        stored_commit = _write_document_pair(staging / LAUNCH_COMMIT_FILENAME, commit)
        stored_receipt = _write_document_pair(staging / RELEASE_RECEIPT_FILENAME, receipt)
        if stored_commit != commit or stored_receipt != receipt:
            raise RuntimeError("Campaign launch document write changed semantic bytes.")
        root_entries, registry_entries = _expected_launch_entries(preview)
        if {entry.name for entry in staging.iterdir()} != root_entries or {
            entry.name for entry in (staging / "registries").iterdir()
        } != registry_entries:
            raise RuntimeError("Campaign launch staging tree is incomplete or contains extras.")
        _assert_fresh_outputs(cohort)
        _freeze_tree(staging)
        reopened = _publish_launch_commit_last(
            cohort=cohort,
            preview=preview,
            staging=staging,
            destination=launch_root,
            state=state,
            commit=commit,
            receipt=receipt,
            root=root,
        )
        if reopened["commit"] != commit or reopened["receipt"] != receipt:
            raise RuntimeError("Published campaign launch bundle failed reauthentication.")
        _assert_fresh_outputs(read_campaign_cohort(kind, root=root))
        try:
            released = run(
                receipt["release_command"],
                cwd=root,
                check=True,
                capture_output=True,
                text=True,
                timeout=120,
            )
        except BaseException as exc:
            raise RuntimeError(
                "Campaign release outcome is ambiguous after immutable authorization; "
                "never replay release automatically."
            ) from exc
        return {
            "schema_version": 1,
            "status": "release_command_completed",
            "cohort_kind": kind,
            "cohort_commit_sha256": cohort.digest,
            "launch_root": str(launch_root),
            "launch_commit_sha256": commit["document_sha256"],
            "held_release_receipt_sha256": receipt["document_sha256"],
            "release_command": receipt["release_command"],
            "scheduler_stdout": str(getattr(released, "stdout", "")).strip(),
            "scheduler_stderr": str(getattr(released, "stderr", "")).strip(),
        }


def campaign_manifest_job(
    cohort: ValidatedCampaignCohort, role: str, index: int
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    matches = [
        (path, manifest)
        for path, manifest in zip(
            cohort.manifest_paths, cohort.manifests, strict=True
        )
        if path.stem == role
    ]
    if len(matches) != 1:
        raise ValueError(f"Campaign cohort has no unique manifest role {role!r}.")
    path, manifest = matches[0]
    jobs = manifest.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("Campaign manifest has no exact jobs list.")
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(jobs)
    ):
        raise IndexError(f"Campaign index {index!r} is outside 0..{len(jobs) - 1}.")
    job = jobs[index]
    if not isinstance(job, dict):
        raise ValueError("Campaign manifest job must be an object.")
    return path, manifest, job


def _registry_submission(
    registry: Mapping[str, Any], *, manifest_sha256: str, index: int
) -> dict[str, Any]:
    if registry.get("manifest_sha256") != manifest_sha256:
        raise ValueError("Campaign registry manifest binding drifted.")
    matches = [row for row in registry["submissions"] if row.get("job_index") == index]
    if len(matches) != 1:
        raise ValueError("Campaign registry must bind exactly one selected task.")
    return deepcopy(dict(matches[0]))


def validate_live_campaign_scheduler_identity(
    *,
    environ: Mapping[str, str],
    registry: Mapping[str, Any],
    index: int,
    slurm_job_name: str,
    launcher_path: Path,
    root: Path,
    run: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Authenticate the exact running Slurm task through the scheduler."""

    job_id = str(environ.get("SLURM_JOB_ID", ""))
    if _JOB_ID_RE.fullmatch(job_id) is None:
        raise RuntimeError("Live campaign SLURM_JOB_ID must be numeric.")
    record = _run_scontrol(job_id, root=root, run=run, label="live campaign task")
    expected = {
        "JobId": job_id,
        "ArrayJobId": str(registry["slurm_array_job_id"]),
        "ArrayTaskId": str(index),
        "JobName": slurm_job_name,
        "JobState": "RUNNING",
        "Command": str(launcher_path),
        "WorkDir": str(root),
        "BatchFlag": "1",
    }
    drift = {
        key: {"expected": value, "actual": record.get(key)}
        for key, value in expected.items()
        if record.get(key) != value
    }
    if drift:
        raise RuntimeError(f"Live campaign scheduler identity drifted: {drift}.")
    return {
        "slurm_job_id": job_id,
        "slurm_array_job_id": expected["ArrayJobId"],
        "slurm_array_task_id": index,
        "slurm_job_name": slurm_job_name,
        "slurm_job_state": "RUNNING",
        "launcher_path": str(launcher_path),
        "work_dir": str(root),
        "batch_flag": 1,
    }


def validate_live_campaign_dispatch_identity(
    *,
    kind: str,
    role: str,
    index: int,
    submission_registry_path: str | Path,
    root: Path,
    environ: Mapping[str, str] | None = None,
    slurm_run: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    """Fail before preflight unless this is one committed live campaign task."""

    root = root.resolve()
    cohort = read_campaign_cohort(kind, root=root)
    bundle = read_campaign_launch_bundle(kind, root=root)
    manifest_path, manifest, _job = campaign_manifest_job(cohort, role, index)
    registry_path = require_exact_registry_path(
        cohort, role, submission_registry_path
    )
    planned = [row for row in bundle["preview"]["arrays"] if row["role"] == role]
    if len(planned) != 1:
        raise ValueError("Campaign launch bundle has no unique role array.")
    registry = _validate_registry(
        registry_path,
        canonical_path=registry_path,
        preview_row=planned[0],
        manifest=manifest,
    )
    submission = _registry_submission(
        registry,
        manifest_sha256=manifest["manifest_sha256"],
        index=index,
    )
    values = os.environ if environ is None else environ
    expected_live = {
        "SLURM_ARRAY_JOB_ID": str(submission["slurm_array_job_id"]),
        "SLURM_ARRAY_TASK_ID": str(submission["slurm_array_task_id"]),
        "SLURM_JOB_NAME": str(planned[0]["slurm_job_name"]),
    }
    if any(values.get(key) != value for key, value in expected_live.items()):
        raise RuntimeError(
            f"Live Slurm identity differs from campaign registry: expected={expected_live}."
        )
    committed = [row for row in bundle["commit"]["arrays"] if row["role"] == role]
    if (
        len(committed) != 1
        or committed[0]["manifest_path"] != str(manifest_path)
        or committed[0]["registry"]["path"] != str(registry_path)
    ):
        raise ValueError("Complete campaign launch commit does not bind this live task.")
    return validate_live_campaign_scheduler_identity(
        environ=values,
        registry=registry,
        index=index,
        slurm_job_name=str(planned[0]["slurm_job_name"]),
        launcher_path=Path(bundle["commit"]["launchers"]["sbatch"]["path"]),
        root=root,
        run=slurm_run,
    )


def _campaign_output_roots(root: Path) -> tuple[Path, Path]:
    return (
        (root / campaign.LADDER_OUTPUT_ROOT_RELATIVE).resolve(strict=False),
        (root / campaign.FINAL_OUTPUT_ROOT_RELATIVE).resolve(strict=False),
    )


def is_canonical_production_campaign_job(
    job: Mapping[str, Any], root: Path | None = None
) -> bool:
    root = (root or project_root()).resolve()
    raw = str(job.get("output_dir", "")).strip()
    if not raw:
        return False
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    lexical = candidate.absolute()
    for output_root in _campaign_output_roots(root):
        if lexical == output_root or output_root in lexical.parents:
            return True
        resolved = lexical.resolve(strict=False)
        if resolved == output_root or output_root in resolved.parents:
            return True
    return False


def reject_canonical_campaign_manifest_from_ordinary_dispatch(
    manifest: Mapping[str, Any], root: Path | None = None
) -> None:
    jobs = manifest.get("jobs")
    if isinstance(jobs, list) and any(
        isinstance(job, Mapping)
        and is_canonical_production_campaign_job(job, root)
        for job in jobs
    ):
        raise RuntimeError(
            "Canonical seed-ladder/final campaign rows require the dedicated "
            "cohort-bound all-held campaign dispatcher."
        )


def _require_exact_campaign_output(
    raw: str | Path, *, expected: Path, root: Path
) -> Path:
    candidate = _require_exact_path(raw, expected, "campaign attempt output")
    if not any(
        candidate != output_root and output_root in candidate.parents
        for output_root in _campaign_output_roots(root)
    ):
        raise ValueError("Campaign attempt output is outside the two canonical roots.")
    _require_no_symlink_components(candidate, "campaign attempt output")
    return candidate


def _authorization_digest(payload: Mapping[str, Any]) -> str:
    return _document_digest(payload, "authorization_sha256")


def build_campaign_launch_authorization(
    *,
    kind: str,
    role: str,
    index: int,
    submission_registry_path: str | Path,
    root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    slurm_run: CommandRunner = subprocess.run,
) -> tuple[dict[str, Any], Path]:
    root = (root or project_root()).resolve()
    cohort = read_campaign_cohort(kind, root=root)
    manifest_path, manifest, job = campaign_manifest_job(cohort, role, index)
    bundle = read_campaign_launch_bundle(kind, root=root)
    scheduler = validate_live_campaign_dispatch_identity(
        kind=kind,
        role=role,
        index=index,
        submission_registry_path=submission_registry_path,
        root=root,
        environ=environ,
        slurm_run=slurm_run,
    )
    expected_output = Path(str(job["output_dir"]))
    output = _require_exact_campaign_output(
        job["output_dir"], expected=expected_output, root=root
    )
    bound_job = {
        **job,
        "launch_manifest_sha256": manifest["manifest_sha256"],
        "launch_manifest_job_index": index,
    }
    read_environment_preflight(
        output,
        expected_job=bound_job,
        expected_job_index=index,
    )
    preflight_path = output / ENVIRONMENT_PREFLIGHT_FILENAME
    registry_path = require_exact_registry_path(
        cohort, role, submission_registry_path
    )
    registry = bundle["registries"][role]
    submission = _registry_submission(
        registry,
        manifest_sha256=manifest["manifest_sha256"],
        index=index,
    )
    commit_path = canonical_launch_commit_path(cohort)
    receipt_path = canonical_release_receipt_path(cohort)
    payload: dict[str, Any] = {
        "schema_version": LAUNCH_AUTHORIZATION_SCHEMA_VERSION,
        "contract": LAUNCH_AUTHORIZATION_CONTRACT,
        "status": "authorized_before_model_load",
        "created_at_utc": _utc_now(),
        "cohort_kind": kind,
        "cohort_root": str(cohort.root),
        "cohort_commit_path": str(cohort.commit_path),
        "cohort_commit_file_sha256": _sha256_file(cohort.commit_path),
        "cohort_commit_sha256": cohort.digest,
        "selection_cohort": cohort.selection_cohort,
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
        "launch_commit_sha256": bundle["commit"]["document_sha256"],
        "held_release_receipt_path": str(receipt_path),
        "held_release_receipt_file_sha256": _sha256_file(receipt_path),
        "held_release_receipt_sha256": bundle["receipt"]["document_sha256"],
        "launchers": deepcopy(bundle["commit"]["launchers"]),
        "slurm_array_job_id": str(submission["slurm_array_job_id"]),
        "slurm_array_task_id": int(submission["slurm_array_task_id"]),
        "slurm_task_id": submission["slurm_task_id"],
        "slurm_job_id": scheduler["slurm_job_id"],
        "slurm_job_name": scheduler["slurm_job_name"],
        "scheduler_identity": scheduler,
    }
    payload["authorization_sha256"] = _authorization_digest(payload)
    return payload, output


def publish_campaign_launch_authorization(
    *,
    kind: str,
    role: str,
    index: int,
    submission_registry_path: str | Path,
    root: Path | None = None,
    environ: Mapping[str, str] | None = None,
    slurm_run: CommandRunner = subprocess.run,
) -> dict[str, Any]:
    payload, output = build_campaign_launch_authorization(
        kind=kind,
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
    if (
        output.is_symlink()
        or not output.is_dir()
        or {path.name for path in output.iterdir()} != expected_entries
    ):
        raise FileExistsError(
            "Campaign authorization requires exactly the immutable preflight pair."
        )
    target = output / LAUNCH_AUTHORIZATION_FILENAME
    encoded = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()
    _publish_files_new(
        {
            target: encoded,
            target.with_suffix(target.suffix + ".sha256"): (
                f"{hashlib.sha256(encoded).hexdigest()}  {target.name}\n"
            ).encode(),
        }
    )
    reopened = read_campaign_launch_authorization(
        output,
        expected_kind=kind,
        expected_role=role,
        expected_index=index,
        root=root,
    )
    if reopened != payload:
        raise RuntimeError("Campaign launch authorization failed reauthentication.")
    return payload


def read_campaign_launch_authorization(
    output_dir: str | Path,
    *,
    expected_kind: str | None = None,
    expected_role: str | None = None,
    expected_index: int | None = None,
    expected_job: Mapping[str, Any] | None = None,
    root: Path | None = None,
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    raw_text = os.fspath(output_dir)
    if any(segment in {".", ".."} for segment in raw_text.split(os.sep)):
        raise ValueError("Campaign authorization attempt contains a lexical alias.")
    raw_output = Path(raw_text).expanduser()
    if not raw_output.is_absolute():
        raw_output = root / raw_output
    output = raw_output.absolute()
    _require_no_symlink_components(output, "campaign authorization attempt")
    target = output / LAUNCH_AUTHORIZATION_FILENAME
    sidecar = output / LAUNCH_AUTHORIZATION_DIGEST_FILENAME
    if target.is_symlink() or sidecar.is_symlink():
        raise ValueError("Campaign authorization pair cannot contain symlinks.")
    try:
        encoded = target.read_bytes()
        payload = json.loads(encoded)
        sidecar_fields = sidecar.read_text(encoding="utf-8").split()
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot authenticate campaign authorization: {exc}") from exc
    if sidecar_fields != [hashlib.sha256(encoded).hexdigest(), target.name]:
        raise ValueError("Campaign authorization raw-file sidecar drifted.")
    if (
        not isinstance(payload, dict)
        or set(payload) != _AUTHORIZATION_FIELDS
        or payload.get("schema_version") != LAUNCH_AUTHORIZATION_SCHEMA_VERSION
        or payload.get("contract") != LAUNCH_AUTHORIZATION_CONTRACT
        or payload.get("status") != "authorized_before_model_load"
        or payload.get("authorization_sha256") != _authorization_digest(payload)
    ):
        raise ValueError("Campaign launch authorization schema/digest drifted.")
    if any(_SHA256_RE.fullmatch(str(payload.get(field, ""))) is None for field in _AUTHORIZATION_HASH_FIELDS):
        raise ValueError("Campaign launch authorization has an invalid SHA-256 binding.")
    _parse_timestamp(payload["created_at_utc"], "campaign authorization timestamp")
    kind = str(payload["cohort_kind"])
    role = str(payload["manifest_role"])
    index = payload["manifest_job_index"]
    if (
        kind not in COHORT_KINDS
        or isinstance(index, bool)
        or not isinstance(index, int)
        or index < 0
        or expected_kind is not None
        and kind != expected_kind
        or expected_role is not None
        and role != expected_role
        or expected_index is not None
        and index != expected_index
    ):
        raise ValueError("Campaign authorization routing identity drifted.")
    cohort = read_campaign_cohort(kind, root=root)
    manifest_path, manifest, job = campaign_manifest_job(cohort, role, index)
    _require_exact_campaign_output(
        output,
        expected=Path(str(job["output_dir"])),
        root=root,
    )
    bundle = read_campaign_launch_bundle(kind, root=root)
    commit_path = canonical_launch_commit_path(cohort)
    receipt_path = canonical_release_receipt_path(cohort)
    registry_path = require_exact_registry_path(
        cohort, role, payload["submission_registry_path"]
    )
    registry = bundle["registries"][role]
    submission = _registry_submission(
        registry,
        manifest_sha256=manifest["manifest_sha256"],
        index=index,
    )
    fixed = {
        "cohort_root": str(cohort.root),
        "cohort_commit_path": str(cohort.commit_path),
        "cohort_commit_file_sha256": _sha256_file(cohort.commit_path),
        "cohort_commit_sha256": cohort.digest,
        "selection_cohort": cohort.selection_cohort,
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": _sha256_file(manifest_path),
        "manifest_sha256": manifest["manifest_sha256"],
        "condition_id": job["condition_id"],
        "output_dir": str(output),
        "implementation_files_sha256": job["implementation_files_sha256"],
        "environment_preflight_sha256": _sha256_file(
            output / ENVIRONMENT_PREFLIGHT_FILENAME
        ),
        "submission_registry_path": str(registry_path),
        "submission_registry_file_sha256": _sha256_file(registry_path),
        "submission_registry_sha256": registry["registry_sha256"],
        "launch_commit_path": str(commit_path),
        "launch_commit_file_sha256": _sha256_file(commit_path),
        "launch_commit_sha256": bundle["commit"]["document_sha256"],
        "held_release_receipt_path": str(receipt_path),
        "held_release_receipt_file_sha256": _sha256_file(receipt_path),
        "held_release_receipt_sha256": bundle["receipt"]["document_sha256"],
        "launchers": bundle["commit"]["launchers"],
        "slurm_array_job_id": str(submission["slurm_array_job_id"]),
        "slurm_array_task_id": int(submission["slurm_array_task_id"]),
        "slurm_task_id": submission["slurm_task_id"],
        "slurm_job_name": next(
            row["slurm_job_name"]
            for row in bundle["preview"]["arrays"]
            if row["role"] == role
        ),
    }
    drift = {
        key: {"expected": value, "actual": payload.get(key)}
        for key, value in fixed.items()
        if payload.get(key) != value
    }
    scheduler = payload.get("scheduler_identity")
    expected_scheduler = {
        "slurm_job_id": str(payload.get("slurm_job_id")),
        "slurm_array_job_id": str(submission["slurm_array_job_id"]),
        "slurm_array_task_id": index,
        "slurm_job_name": fixed["slurm_job_name"],
        "slurm_job_state": "RUNNING",
        "launcher_path": bundle["commit"]["launchers"]["sbatch"]["path"],
        "work_dir": str(root),
        "batch_flag": 1,
    }
    if (
        drift
        or _JOB_ID_RE.fullmatch(str(payload.get("slurm_job_id", ""))) is None
        or scheduler != expected_scheduler
    ):
        raise ValueError(f"Campaign launch authorization binding drifted: {drift}.")
    bound_job = {
        **job,
        "launch_manifest_sha256": manifest["manifest_sha256"],
        "launch_manifest_job_index": index,
    }
    read_environment_preflight(
        output,
        expected_job=bound_job,
        expected_job_index=index,
    )
    if expected_job is not None and dict(expected_job) != bound_job:
        raise ValueError("Campaign authorization differs from the direct-runner job.")
    return deepcopy(payload)


def require_campaign_launch_authorization_for_job(
    job: Mapping[str, Any],
    root: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    slurm_run: CommandRunner = subprocess.run,
) -> dict[str, Any] | None:
    """Runner hook: reauthenticate authorization and live Slurm before model load."""

    root = (root or project_root()).resolve()
    if not is_canonical_production_campaign_job(job, root):
        return None
    values = os.environ if environ is None else environ
    if values.get(CAMPAIGN_AUTHORIZATION_REQUIRED_ENV) != "1":
        raise RuntimeError(f"{CAMPAIGN_AUTHORIZATION_REQUIRED_ENV} must be exactly '1'.")
    output = Path(str(job["output_dir"])).resolve()
    exact_entries = {
        ENVIRONMENT_PREFLIGHT_FILENAME,
        ENVIRONMENT_PREFLIGHT_DIGEST_FILENAME,
        LAUNCH_AUTHORIZATION_FILENAME,
        LAUNCH_AUTHORIZATION_DIGEST_FILENAME,
    }
    if not output.is_dir() or {path.name for path in output.iterdir()} != exact_entries:
        raise FileNotFoundError(
            "Canonical campaign attempt requires exactly preflight and campaign "
            "authorization pairs before model load."
        )
    payload = read_campaign_launch_authorization(
        output,
        expected_job=job,
        root=root,
    )
    registry = read_submission_registry(Path(payload["submission_registry_path"]))
    observed = validate_live_campaign_scheduler_identity(
        environ=values,
        registry=registry,
        index=int(payload["manifest_job_index"]),
        slurm_job_name=str(payload["slurm_job_name"]),
        launcher_path=Path(payload["launchers"]["sbatch"]["path"]),
        root=root,
        run=slurm_run,
    )
    if observed != payload["scheduler_identity"]:
        raise RuntimeError("Live campaign Slurm identity changed before model load.")
    return payload
