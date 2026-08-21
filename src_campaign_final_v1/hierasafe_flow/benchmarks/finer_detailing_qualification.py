"""Immutable Q1/Q2 planning for the finer-detailing qualification campaign.

The ordinary benchmark manifest builder remains the single source of job
semantics.  This module calls that builder four times per phase (image seed 0
and video seeds 0, 1, and 2), proves the exact registered topology, and only
then publishes the four manifests and a content-addressed phase specification.

Q2 is deliberately impossible to freeze from a command-line assertion alone.
It requires two authenticated receipts bound to the exact Q1 specification and
implementation: one proving all 66 exact-one Shapley rows passed every frozen
gate, and one proving the full sequential-pair non-regression guard and tests.
The gate reader reopens each runtime authorization, preflight, result, media,
trace, timing record, execution identity, and externally hash-bound manual
ledger, and independently reruns the exact registered non-regression tests.
This module plans qualification only; it does not submit Slurm jobs, select
seeds, build the final matrix, or mutate model configuration.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import secrets
import shutil
import stat
import subprocess
import sys
import tempfile
from collections import Counter
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from hierasafe_flow.benchmarks.finer_detailing_correction import (
    BENCHMARK_NAME,
    EXPECTED_NATIVE_NEGATIVE_SUPPORT,
    MODEL_NAMES,
    PAIR_IDS_BY_PROMPT,
    PROMPT_IDS,
    build_manifest,
    build_parser as build_benchmark_parser,
    is_flux1_job_v3,
    manifest_digest,
    project_root,
    read_manifest,
    read_manifest_for_audit,
    reopen_completed_flux1_output_v3,
    _verify_implementation_provenance,
    write_manifest_immutable,
)
from hierasafe_flow.evaluation import flux1_dual_view_jobs_v3 as flux_v3


QUALIFICATION_PLAN_SCHEMA_VERSION = 1
QUALIFICATION_PLAN_CONTRACT = "finer_detailing_fresh_qualification_phase_v1"
EXACT_ONE_GATE_CONTRACT = "finer_detailing_q1_exact_one_gate_v1"
NON_REGRESSION_GATE_CONTRACT = "finer_detailing_full_pair_non_regression_gate_v1"
GATE_RECEIPT_SCHEMA_VERSION = 1
EXACT_ONE_EVIDENCE_SCHEMA_VERSION = 1
EXACT_ONE_EVIDENCE_CONTRACT = "finer_detailing_q1_exact_one_runtime_evidence_v1"
NON_REGRESSION_EVIDENCE_SCHEMA_VERSION = 1
NON_REGRESSION_EVIDENCE_CONTRACT = (
    "finer_detailing_q1_full_pair_non_regression_runtime_evidence_v1"
)
FLUX1_COMPLETED_UNION_CONTRACT = (
    "finer_detailing_q1_q2_flux1_completed_union_receipt_v3"
)

QUALIFICATION_OUTPUT_ROOT_RELATIVE = Path(
    "outputs/finer_detailing_fresh_qualification_v1"
)
QUALIFICATION_COHORT_ROOT_RELATIVE = {
    "q1": Path("debugging/manifests/finer_detailing_fresh_qualification_q1_v1"),
    "q2": Path("debugging/manifests/finer_detailing_fresh_qualification_q2_v1"),
}
QUALIFICATION_PLAN_FILENAME = "qualification_plan.json"
QUALIFICATION_COHORT_COMMIT_FILENAME = "qualification_cohort_commit.json"
QUALIFICATION_COHORT_COMMIT_SCHEMA_VERSION = 1
QUALIFICATION_COHORT_COMMIT_CONTRACT = (
    "finer_detailing_atomic_four_manifest_qualification_cohort_v1"
)

Q1 = "q1"
Q2 = "q2"
PHASES = (Q1, Q2)

VIDEO_MODELS = (
    "cogvideox_5b",
    "hunyuan_video",
    "joyai_echo",
    "ltx_23",
    "wan22_t2v_a14b",
)
IMAGE_MODELS = tuple(model for model in MODEL_NAMES if model not in VIDEO_MODELS)
VIDEO_SEEDS = (0, 1, 2)
IMAGE_SEED = 0

QUALIFICATION_PAIR_BY_PROMPT = {
    "01_sad_young_girl": "facial_affect_negative_to_happy",
    "02_angry_old_man": "facial_affect_negative_to_happy",
    "03_empty_outdoor_mall": "vertical_circulation_escalators_to_marble_stairs",
}
QUALIFICATION_PAIR_IDS = tuple(dict.fromkeys(QUALIFICATION_PAIR_BY_PROMPT.values()))

Q1_VARIATIONS = (
    "01_baseline",
    "02_negative_prompt",
    "03_concept_steering",
    "05_concept_steering_single_pair",
    "06_shapley_concept_steering_single_pair",
)
Q2_VARIATIONS = ("04_shapley_concept_steering",)

_METHOD_KIND_ALIASES: dict[str, str] = {}


def _normalize_variant_kind(kind: str) -> str:
    return _METHOD_KIND_ALIASES.get(kind, kind)

MANIFEST_ROLE_ORDER = (
    "image_seed000",
    "video_seed000",
    "video_seed001",
    "video_seed002",
)
QUALIFICATION_MANIFEST_FILENAMES = {
    role: f"{role}.json" for role in MANIFEST_ROLE_ORDER
}
MANIFEST_ROLE_SPECS: dict[str, dict[str, Any]] = {
    "image_seed000": {
        "task": "text_to_image",
        "seed": 0,
        "models": IMAGE_MODELS,
    },
    "video_seed000": {
        "task": "text_to_video",
        "seed": 0,
        "models": VIDEO_MODELS,
    },
    "video_seed001": {
        "task": "text_to_video",
        "seed": 1,
        "models": VIDEO_MODELS,
    },
    "video_seed002": {
        "task": "text_to_video",
        "seed": 2,
        "models": VIDEO_MODELS,
    },
}

EXPECTED_PHASE_COUNTS = {
    Q1: {"logical": 330, "media": 303, "unsupported": 27},
    Q2: {"logical": 66, "media": 66, "unsupported": 0},
}
EXPECTED_ROLE_COUNTS = {
    Q1: {
        "image_seed000": {"logical": 105, "media": 96, "unsupported": 9},
        "video_seed000": {"logical": 75, "media": 69, "unsupported": 6},
        "video_seed001": {"logical": 75, "media": 69, "unsupported": 6},
        "video_seed002": {"logical": 75, "media": 69, "unsupported": 6},
    },
    Q2: {
        "image_seed000": {"logical": 21, "media": 21, "unsupported": 0},
        "video_seed000": {"logical": 15, "media": 15, "unsupported": 0},
        "video_seed001": {"logical": 15, "media": 15, "unsupported": 0},
        "video_seed002": {"logical": 15, "media": 15, "unsupported": 0},
    },
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PLAN_KEYS = {
    "schema_version",
    "contract",
    "benchmark",
    "phase",
    "created_at_utc",
    "attempt",
    "output_root",
    "implementation_files_sha256",
    "manifest_bindings",
    "topology_proof",
    "upstream_q1",
    "gate_receipts",
    "qualification_union_proof",
    "qualification_plan_sha256",
}
_MANIFEST_BINDING_KEYS = {
    "role",
    "task",
    "seed",
    "path",
    "file_sha256",
    "manifest_sha256",
    "logical_rows",
    "media_rows",
    "unsupported_rows",
}
_UPSTREAM_Q1_KEYS = {"path", "file_sha256", "qualification_plan_sha256"}
_GATE_BINDING_KEYS = {"contract", "path", "file_sha256", "receipt_sha256"}
_RECEIPT_KEYS = {
    "schema_version",
    "contract",
    "benchmark",
    "created_at_utc",
    "q1_plan",
    "implementation_files_sha256",
    "decision",
    "claims",
    "evidence",
    "receipt_sha256",
}
_EVIDENCE_KEYS = {"path", "sha256", "size_bytes"}
_EXACT_ONE_CLAIMS = {
    "expected_exact_one_shapley_rows": 66,
    "validated_exact_one_shapley_rows": 66,
    "numerical_pass_rows": 66,
    "trace_pass_rows": 66,
    "target_uptake_pass_rows": 66,
    "inactive_concept_preservation_pass_rows": 66,
    "manual_review_pass_rows": 66,
    "failed_rows": 0,
}
_NON_REGRESSION_REQUIRED_BOOLEANS = {
    "final_active_pair_non_regression_guard_present": True,
    "all_required_tests_passed": True,
}
_COHORT_COMMIT_KEYS = {
    "schema_version",
    "contract",
    "phase",
    "created_at_utc",
    "cohort_root",
    "output_root",
    "plan",
    "manifests",
    "counts",
    "document_sha256",
}
_COHORT_PLAN_KEYS = {
    "path",
    "file_sha256",
    "qualification_plan_sha256",
}
_COHORT_MANIFEST_KEYS = {
    "role",
    "path",
    "file_sha256",
    "manifest_sha256",
    "snapshot_index_path",
    "snapshot_index_sha256",
}


@dataclass(frozen=True)
class ValidatedQualificationPlan:
    """A fully reopened qualification specification and its manifest union."""

    path: Path
    plan: dict[str, Any]
    manifests: dict[str, dict[str, Any]]
    topology_proof: dict[str, Any]
    upstream_q1: "ValidatedQualificationPlan | None"
    gate_receipts: tuple[dict[str, Any], ...]

    @property
    def phase(self) -> str:
        return str(self.plan["phase"])

    @property
    def digest(self) -> str:
        return str(self.plan["qualification_plan_sha256"])


def canonical_sha256(payload: Any) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def qualification_plan_digest(plan: Mapping[str, Any]) -> str:
    canonical = dict(plan)
    canonical.pop("qualification_plan_sha256", None)
    return canonical_sha256(canonical)


def gate_receipt_digest(receipt: Mapping[str, Any]) -> str:
    canonical = dict(receipt)
    canonical.pop("receipt_sha256", None)
    return canonical_sha256(canonical)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _validate_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be an explicit timezone-aware timestamp.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} is not an ISO-8601 timestamp: {value!r}.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include a timezone offset.")
    return parsed


def _resolve(path: str | Path, root: Path) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    return candidate.resolve(strict=False)


def canonical_qualification_cohort_root(phase: str, root: Path | None = None) -> Path:
    """Return the only publication directory admitted for a Q1/Q2 cohort."""

    if phase not in PHASES:
        raise ValueError(f"Unknown qualification phase {phase!r}.")
    resolved_root = (root or project_root()).expanduser().resolve()
    return (resolved_root / QUALIFICATION_COHORT_ROOT_RELATIVE[phase]).absolute()


def canonical_qualification_plan_path(phase: str, root: Path | None = None) -> Path:
    return canonical_qualification_cohort_root(phase, root) / QUALIFICATION_PLAN_FILENAME


def canonical_qualification_manifest_paths(
    phase: str, root: Path | None = None
) -> dict[str, Path]:
    cohort = canonical_qualification_cohort_root(phase, root)
    return {
        role: cohort / QUALIFICATION_MANIFEST_FILENAMES[role]
        for role in MANIFEST_ROLE_ORDER
    }


def canonical_qualification_cohort_commit_path(
    phase: str, root: Path | None = None
) -> Path:
    return (
        canonical_qualification_cohort_root(phase, root)
        / QUALIFICATION_COHORT_COMMIT_FILENAME
    )


def canonical_qualification_output_root(root: Path | None = None) -> Path:
    resolved_root = (root or project_root()).expanduser().resolve()
    return (resolved_root / QUALIFICATION_OUTPUT_ROOT_RELATIVE).absolute()


def _existing_component_is_symlink(path: Path, *, root: Path) -> bool:
    if not path.is_absolute() or (path != root and root not in path.parents):
        return True
    current = path
    while True:
        if current.is_symlink():
            return True
        if current == root:
            return False
        current = current.parent


def _lexical_absolute(path: str | Path, root: Path, *, label: str) -> Path:
    raw = os.fspath(path)
    if not raw or "\0" in raw:
        raise ValueError(f"{label} is empty or contains NUL.")
    expanded = Path(path).expanduser()
    if any(part in {".", ".."} for part in expanded.parts):
        raise ValueError(f"{label} contains an explicit lexical alias: {path}.")
    candidate = expanded if expanded.is_absolute() else root / expanded
    candidate = candidate.absolute()
    if candidate.resolve(strict=False) != candidate:
        raise ValueError(f"{label} resolves through an alias or symlink: {candidate}.")
    if _existing_component_is_symlink(candidate.parent, root=root):
        raise ValueError(f"{label} has an existing symlink component: {candidate}.")
    return candidate


def _publication_fault_hook(_step: str) -> None:
    """Test-only crash boundary; production deliberately performs no action."""


def _tree_relative_entries(root: Path) -> tuple[set[Path], set[Path]]:
    directories: set[Path] = set()
    files: set[Path] = set()
    for path in root.rglob("*"):
        relative = path.relative_to(root)
        if path.is_symlink():
            raise ValueError(f"Immutable publication tree contains a symlink: {path}.")
        if path.is_dir():
            directories.add(relative)
        elif path.is_file():
            files.add(relative)
        else:
            raise ValueError(f"Immutable publication tree has an unsupported member: {path}.")
    return directories, files


def _require_nonwritable_directories(root: Path, *, label: str) -> None:
    """Require the publication root and every descendant directory to be sealed."""

    for directory in (root, *(path for path in root.rglob("*") if path.is_dir())):
        observed = directory.lstat()
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ValueError(f"{label} contains an invalid directory: {directory}.")
        if stat.S_IMODE(observed.st_mode) & 0o222:
            raise ValueError(f"{label} contains a writable directory: {directory}.")


def _remove_fd_tree(directory_fd: int) -> None:
    """Remove children through an already-open owned directory descriptor."""

    for name in os.listdir(directory_fd):
        observed = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
        if stat.S_ISDIR(observed.st_mode):
            child_fd = os.open(
                name,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
                dir_fd=directory_fd,
            )
            try:
                _remove_fd_tree(child_fd)
            finally:
                os.close(child_fd)
            os.rmdir(name, dir_fd=directory_fd)
        else:
            os.unlink(name, dir_fd=directory_fd)


def _cleanup_uncommitted_claim(
    destination: Path,
    *,
    destination_fd: int,
    identity: tuple[int, int],
    token_path: Path,
    token: str,
    commit_path: Path,
) -> None:
    """Delete only an inode/token-owned claim that has no published commit."""

    try:
        token_matches = token_path.read_text(encoding="utf-8") == token
        fd_stat = os.fstat(destination_fd)
        path_stat = destination.lstat()
    except (FileNotFoundError, OSError):
        return
    if (
        not token_matches
        or destination.is_symlink()
        or (fd_stat.st_dev, fd_stat.st_ino) != identity
        or (path_stat.st_dev, path_stat.st_ino) != identity
        or commit_path.exists()
        or commit_path.is_symlink()
    ):
        return
    _remove_fd_tree(destination_fd)
    try:
        path_stat = destination.lstat()
    except FileNotFoundError:
        return
    if (path_stat.st_dev, path_stat.st_ino) == identity and not destination.is_symlink():
        destination.rmdir()
    token_path.unlink(missing_ok=True)
    _fsync_parent(token_path.parent)


def _fsync_parent(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_hardlink_tree_commit_last(
    staging: Path,
    destination: Path,
    *,
    commit_relative_path: Path,
    fault_hook: Any = _publication_fault_hook,
) -> None:
    """Publish a Ceph-safe immutable tree with one commit-last admission edge.

    ``mkdir`` is the exclusive canonical-name claim. Every immutable file is
    then hard-linked with O_EXCL semantics. The commit sidecar is linked before
    the commit, while the self-authenticating commit JSON is the final file.
    Readers additionally require every directory to be non-writable, so a
    crash after commit but before chmod remains inadmissible.
    """

    staging = staging.absolute()
    destination = destination.absolute()
    if staging.is_symlink() or not staging.is_dir():
        raise ValueError("Commit-last staging must be a real directory.")
    if staging.parent != destination.parent:
        raise ValueError("Commit-last staging must be a hidden sibling of its destination.")
    if not staging.name.startswith(f".{destination.name}."):
        raise ValueError("Commit-last staging does not have the destination's hidden prefix.")
    directories, files = _tree_relative_entries(staging)
    commit_relative_path = Path(commit_relative_path)
    commit_sidecar = commit_relative_path.with_suffix(commit_relative_path.suffix + ".sha256")
    if commit_relative_path not in files or commit_sidecar not in files:
        raise ValueError("Commit-last tree lacks its commit JSON or sidecar.")
    _require_nonwritable_directories(staging, label="Commit-last staging tree")
    for relative in files:
        if stat.S_IMODE((staging / relative).lstat().st_mode) & 0o222:
            raise ValueError(f"Commit-last staging contains a writable file: {relative}.")
    fault_hook("preclaim")

    token = secrets.token_hex(32)
    token_path = destination.parent / f".{destination.name}.claim-{token}"
    token_fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(token_fd, "w", encoding="utf-8") as handle:
            handle.write(token)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_parent(destination.parent)
        try:
            destination.mkdir(mode=0o700)
        except FileExistsError as exc:
            token_path.unlink(missing_ok=True)
            _fsync_parent(destination.parent)
            raise FileExistsError(
                f"Immutable canonical publication root already exists: {destination}."
            ) from exc
        _fsync_parent(destination.parent)
        claimed = destination.lstat()
        claimed_identity = (claimed.st_dev, claimed.st_ino)
        try:
            destination_fd = os.open(
                destination,
                os.O_RDONLY
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0),
            )
        except BaseException:
            try:
                current = destination.lstat()
                if (
                    not destination.is_symlink()
                    and (current.st_dev, current.st_ino) == claimed_identity
                    and not any(destination.iterdir())
                ):
                    destination.rmdir()
            finally:
                token_path.unlink(missing_ok=True)
                _fsync_parent(destination.parent)
            raise
        observed = os.fstat(destination_fd)
        identity = (observed.st_dev, observed.st_ino)
        if identity != claimed_identity:
            os.close(destination_fd)
            token_path.unlink(missing_ok=True)
            raise RuntimeError("Canonical publication claim inode changed before authentication.")
        committed = False
        try:
            fault_hook("postclaim")
            for relative in sorted(directories, key=lambda item: (len(item.parts), str(item))):
                (destination / relative).mkdir(mode=0o700)
            ordinary_files = sorted(
                files - {commit_relative_path, commit_sidecar}, key=str
            )
            for position, relative in enumerate(ordinary_files):
                os.link(
                    staging / relative,
                    destination / relative,
                    follow_symlinks=False,
                )
                if position == 0:
                    fault_hook("midlink")
            os.link(
                staging / commit_sidecar,
                destination / commit_sidecar,
                follow_symlinks=False,
            )
            canonical_dirs, canonical_files = _tree_relative_entries(destination)
            if canonical_dirs != directories or canonical_files != files - {commit_relative_path}:
                raise RuntimeError("Canonical precommit tree differs from its complete stage.")
            for relative in canonical_files:
                staged_stat = (staging / relative).stat()
                canonical_stat = (destination / relative).stat()
                if (
                    (staged_stat.st_dev, staged_stat.st_ino)
                    != (canonical_stat.st_dev, canonical_stat.st_ino)
                    or _sha256_file(staging / relative)
                    != _sha256_file(destination / relative)
                ):
                    raise RuntimeError(f"Canonical hard-link binding drifted: {relative}.")
            for relative in sorted(directories, key=lambda item: len(item.parts), reverse=True):
                _fsync_parent(destination / relative)
            _fsync_parent(destination)
            fault_hook("precommit")
            os.link(
                staging / commit_relative_path,
                destination / commit_relative_path,
                follow_symlinks=False,
            )
            committed = True
            _fsync_parent(destination)
            fault_hook("postcommit_prechmod")
            for relative in sorted(
                directories, key=lambda item: len(item.parts), reverse=True
            ):
                (destination / relative).chmod(0o555)
                _fsync_parent(destination / relative)
            destination.chmod(0o555)
            _fsync_parent(destination)
            _fsync_parent(destination.parent)
        except BaseException:
            if not committed:
                _cleanup_uncommitted_claim(
                    destination,
                    destination_fd=destination_fd,
                    identity=identity,
                    token_path=token_path,
                    token=token,
                    commit_path=destination / commit_relative_path,
                )
            raise
        finally:
            os.close(destination_fd)
        token_path.unlink(missing_ok=True)
        _fsync_parent(destination.parent)
    except BaseException:
        if token_path.exists() and not destination.exists():
            token_path.unlink(missing_ok=True)
        raise


def _cleanup_owned_staging(staging: Path, identity: tuple[int, int]) -> None:
    """Remove only the private staging inode created by this publisher."""

    try:
        observed = staging.lstat()
    except FileNotFoundError:
        return
    if staging.is_symlink() or (observed.st_dev, observed.st_ino) != identity:
        return
    staging.chmod(0o700)
    for directory in sorted(
        (
            path for path in staging.rglob("*") if path.is_dir() and not path.is_symlink()
        ),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o700)
    shutil.rmtree(staging)


def _freeze_tree(directory: Path) -> None:
    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Qualification cohort cannot contain a symlink: {path}.")
    for path in directory.rglob("*"):
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for path in sorted(
        (item for item in directory.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        path.chmod(0o555)
    directory.chmod(0o555)


def _atomic_write_new_text(path: Path, text: str) -> None:
    """Atomically publish a new file and never replace an existing artifact."""

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
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"Refusing to overwrite immutable file: {path}") from exc
        temporary.unlink()
        temporary = None
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _write_authenticated_document(
    payload: Mapping[str, Any],
    path: Path,
    *,
    digest_field: str,
    digest_function: Any,
) -> dict[str, Any]:
    frozen = deepcopy(dict(payload))
    expected = digest_function(frozen)
    supplied = frozen.get(digest_field)
    if supplied is not None and supplied != expected:
        raise ValueError(f"{digest_field} is inconsistent before publication.")
    frozen[digest_field] = expected
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if path.exists() or sidecar.exists():
        raise FileExistsError(f"Refusing to overwrite immutable document: {path}")
    sidecar_published = False
    try:
        _atomic_write_new_text(sidecar, f"{expected}  {path.name}\n")
        sidecar_published = True
        _atomic_write_new_text(path, json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    except Exception:
        if sidecar_published and not path.exists():
            sidecar.unlink(missing_ok=True)
        raise
    return frozen


def _read_authenticated_document(
    path: Path,
    *,
    digest_field: str,
    digest_function: Any,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} is missing: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot decode {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    actual = digest_function(payload)
    if payload.get(digest_field) != actual:
        raise ValueError(f"{label} canonical digest mismatch: {path}")
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if not sidecar.is_file():
        raise ValueError(f"{label} sidecar is missing: {sidecar}")
    try:
        fields = sidecar.read_text(encoding="utf-8").split()
    except (OSError, UnicodeDecodeError) as exc:
        raise ValueError(f"Cannot read {label} sidecar {sidecar}: {exc}") from exc
    if fields != [actual, path.name]:
        raise ValueError(f"{label} sidecar does not authenticate {path}.")
    return payload


def _manifest_builder_arguments(
    *,
    phase: str,
    role: str,
    output_root: Path,
    attempt: int,
    allow_unvalidated_temporal_pilot: bool,
) -> Any:
    spec = MANIFEST_ROLE_SPECS[role]
    variations = Q1_VARIATIONS if phase == Q1 else Q2_VARIATIONS
    arguments = [
        "--output-root",
        str(output_root),
        "--attempt",
        str(attempt),
        "--models",
        ",".join(spec["models"]),
        "--prompt-ids",
        ",".join(PROMPT_IDS),
        "--variations",
        ",".join(variations),
        "--seed",
        str(spec["seed"]),
        "--seed-scoped-output",
    ]
    if phase == Q1:
        arguments.extend(["--pair-ids", ",".join(QUALIFICATION_PAIR_IDS)])
    if allow_unvalidated_temporal_pilot:
        arguments.append("--allow-unvalidated-temporal-pilot")
    return build_benchmark_parser().parse_args(arguments)


def build_qualification_manifests(
    phase: str,
    *,
    root: Path | None = None,
    output_root: str | Path,
    attempt: int,
    allow_unvalidated_temporal_pilot: bool,
    flux1_execution_protocol_inputs: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Build, but do not publish, the exact four manifests for Q1 or Q2."""

    root = (root or project_root()).resolve()
    if phase not in PHASES:
        raise ValueError(f"Unknown qualification phase {phase!r}; expected one of {PHASES}.")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt <= 0:
        raise ValueError("Qualification attempt must be a positive integer.")
    resolved_output_root = _resolve(output_root, root)
    manifests: dict[str, dict[str, Any]] = {}
    for role in MANIFEST_ROLE_ORDER:
        args = _manifest_builder_arguments(
            phase=phase,
            role=role,
            output_root=resolved_output_root,
            attempt=attempt,
            allow_unvalidated_temporal_pilot=allow_unvalidated_temporal_pilot,
        )
        flux_kwargs: dict[str, Any] = {}
        if "flux1_dev" in MANIFEST_ROLE_SPECS[role]["models"] and (
            flux1_execution_protocol_inputs is not None
        ):
            flux_kwargs = {
                "flux1_route_context": flux_v3.MODE_EXECUTION,
                "flux1_protocol_inputs": flux1_execution_protocol_inputs,
            }
        manifests[role] = build_manifest(args, root, **flux_kwargs)
    validate_qualification_manifests(
        phase,
        manifests,
        output_root=resolved_output_root,
        attempt=attempt,
        require_flux1_execution=flux1_execution_protocol_inputs is not None,
        root=root,
    )
    return manifests


def _expected_axis_rows(phase: str) -> set[tuple[Any, ...]]:
    rows: set[tuple[Any, ...]] = set()
    for role in MANIFEST_ROLE_ORDER:
        spec = MANIFEST_ROLE_SPECS[role]
        for prompt_id in PROMPT_IDS:
            exact_pair = QUALIFICATION_PAIR_BY_PROMPT[prompt_id]
            for model_name in spec["models"]:
                if phase == Q1:
                    variants: Sequence[tuple[str, str | None]] = (
                        ("01_baseline", None),
                        ("02_negative_prompt", None),
                        ("03_concept_steering", None),
                        ("05_concept_steering_single_pair", exact_pair),
                        ("06_shapley_concept_steering_single_pair", exact_pair),
                    )
                else:
                    variants = (("04_shapley_concept_steering", None),)
                for variation, pair_id in variants:
                    rows.add(
                        (
                            role,
                            spec["task"],
                            spec["seed"],
                            prompt_id,
                            model_name,
                            variation,
                            pair_id,
                        )
                    )
    return rows


def _job_pair_id(job: Mapping[str, Any]) -> str | None:
    spec = job.get("variant_spec")
    if not isinstance(spec, Mapping):
        raise ValueError(f"Qualification job {job.get('condition_id')} has no variant_spec.")
    if spec.get("pair_selection") != "single":
        return None
    active = spec.get("active_pair_ids")
    if (
        not isinstance(active, Sequence)
        or isinstance(active, (str, bytes, bytearray))
        or len(active) != 1
        or not isinstance(active[0], str)
    ):
        raise ValueError(
            f"Qualification exact-one job {job.get('condition_id')} must activate one pair."
        )
    return active[0]


def _expected_output_dir(
    *,
    output_root: Path,
    job: Mapping[str, Any],
    attempt: int,
) -> Path:
    variation = str(job["variation"])
    pair_id = _job_pair_id(job)
    if variation in {"03_concept_steering", "04_shapley_concept_steering"}:
        suffix = (variation, "full")
    elif variation in {
        "05_concept_steering_single_pair",
        "06_shapley_concept_steering_single_pair",
    }:
        if pair_id is None:
            raise ValueError("Single-pair qualification row is missing its active pair.")
        suffix = (variation, pair_id)
    else:
        suffix = (variation,)
    path = output_root / str(job["prompt_id"]) / str(job["model_name"])
    for part in suffix:
        path /= part
    path /= f"seed_{int(job['seed']):08d}"
    return path / "attempts" / f"attempt_{attempt:03d}"


def _validate_variant_contract(phase: str, job: Mapping[str, Any]) -> None:
    variation = str(job.get("variation", ""))
    spec = job.get("variant_spec")
    if not isinstance(spec, Mapping):
        raise ValueError("Qualification job variant_spec must be a mapping.")
    exact_pair = QUALIFICATION_PAIR_BY_PROMPT[str(job["prompt_id"])]
    kind = _normalize_variant_kind(str(spec.get("kind", "")))
    if phase == Q2:
        if variation != "04_shapley_concept_steering":
            raise ValueError("Q2 may contain only full Shapley jobs.")
        if kind != "shapley_concept_steering" or spec.get("pair_selection") != "full":
            raise ValueError("Q2 job is not full-five-pair Shapley steering.")
        if tuple(spec.get("active_pair_ids") or ()) != PAIR_IDS_BY_PROMPT[str(job["prompt_id"])]:
            raise ValueError("Q2 full Shapley active-pair coverage drifted.")
        return

    expected = {
        "01_baseline": ("baseline", None),
        "02_negative_prompt": ("native_negative_prompt", None),
        "03_concept_steering": ("conceptsteer", "full"),
        "05_concept_steering_single_pair": ("conceptsteer", "single"),
        "06_shapley_concept_steering_single_pair": (
            "shapley_concept_steering",
            "single",
        ),
    }
    if variation not in expected:
        raise ValueError(f"Q1 contains forbidden variation {variation!r}.")
    expected_kind, selection = expected[variation]
    if expected_kind != kind:
        raise ValueError(f"Q1 variation {variation} has the wrong steering kind.")
    if selection is not None and spec.get("pair_selection") != selection:
        raise ValueError(f"Q1 variation {variation} has the wrong pair selection.")
    if selection is None and (
        spec.get("pair_selection") is not None or spec.get("active_pair_ids") is not None
    ):
        raise ValueError(f"Q1 non-steering variation {variation} claims active pairs.")
    if variation == "02_negative_prompt":
        expected_capability = (
            "supported"
            if EXPECTED_NATIVE_NEGATIVE_SUPPORT[str(job["model_name"])]
            else "not_supported"
        )
        if spec.get("capability") != expected_capability:
            raise ValueError("Q1 native-negative capability declaration drifted.")
    if (
        selection == "full"
        and tuple(spec.get("active_pair_ids") or ()) != PAIR_IDS_BY_PROMPT[str(job["prompt_id"])]
    ):
        raise ValueError("Q1 ordinary full active-pair coverage drifted.")
    if selection == "single" and _job_pair_id(job) != exact_pair:
        raise ValueError(
            f"Q1 exact-one pair differs from the preregistered pair for {job['prompt_id']}."
        )


def validate_qualification_manifests(
    phase: str,
    manifests: Mapping[str, Mapping[str, Any]],
    *,
    output_root: str | Path,
    attempt: int,
    require_flux1_execution: bool = False,
    root: Path | None = None,
    reauthenticate_flux1_execution: bool | None = None,
) -> dict[str, Any]:
    """Prove exact axes, counts, media status, paths, and provenance for one phase."""

    if phase not in PHASES:
        raise ValueError(f"Unknown qualification phase {phase!r}.")
    if set(manifests) != set(MANIFEST_ROLE_ORDER):
        raise ValueError(
            "Qualification manifest roles must be exactly "
            f"{list(MANIFEST_ROLE_ORDER)}; got {sorted(manifests)}."
        )
    resolved_root = (root or project_root()).resolve()
    resolved_output_root = Path(output_root).expanduser().resolve()
    expected_axes = _expected_axis_rows(phase)
    actual_axes: set[tuple[Any, ...]] = set()
    union_rows: list[dict[str, Any]] = []
    condition_ids: set[str] = set()
    output_dirs: set[str] = set()
    implementation_digests: set[str] = set()
    git_provenance_digests: set[str] = set()
    total_media = 0
    total_unsupported = 0
    counts_by_role: dict[str, dict[str, int]] = {}
    task_counts: Counter[str] = Counter()
    seed_counts: Counter[int] = Counter()
    model_counts: Counter[str] = Counter()
    prompt_counts: Counter[str] = Counter()
    variation_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    flux_route_modes: set[str] = set()
    flux_protocol_inputs: dict[str, dict[str, str]] | None = None
    representative_flux_job: Mapping[str, Any] | None = None

    for role in MANIFEST_ROLE_ORDER:
        manifest = manifests[role]
        if manifest.get("benchmark") != BENCHMARK_NAME:
            raise ValueError(f"Qualification manifest {role} benchmark identity mismatch.")
        if manifest_digest(dict(manifest)) != manifest.get("manifest_sha256"):
            raise ValueError(f"Qualification manifest {role} canonical digest mismatch.")
        jobs = manifest.get("jobs")
        if not isinstance(jobs, list) or len(jobs) != manifest.get("num_jobs"):
            raise ValueError(f"Qualification manifest {role} has an invalid job list.")
        spec = MANIFEST_ROLE_SPECS[role]
        expected_variations = Q1_VARIATIONS if phase == Q1 else Q2_VARIATIONS
        expected_pair_map = (
            {prompt: [QUALIFICATION_PAIR_BY_PROMPT[prompt]] for prompt in PROMPT_IDS}
            if phase == Q1
            else {prompt: list(PAIR_IDS_BY_PROMPT[prompt]) for prompt in PROMPT_IDS}
        )
        exact_top_level = {
            "seed": spec["seed"],
            "seed_scoped_output": True,
            "attempt": attempt,
            "models": list(spec["models"]),
            "prompt_ids": list(PROMPT_IDS),
            "variation_groups": list(expected_variations),
            "selected_single_pair_ids_by_prompt": expected_pair_map,
            "output_root": str(resolved_output_root),
        }
        for field, expected in exact_top_level.items():
            actual = manifest.get(field)
            if actual != expected:
                raise ValueError(
                    f"Qualification manifest {role} {field} drifted: "
                    f"expected={expected!r}, actual={actual!r}."
                )
        implementation_digest = str(manifest.get("implementation_files_sha256", ""))
        if _SHA256_RE.fullmatch(implementation_digest) is None:
            raise ValueError(f"Qualification manifest {role} lacks source provenance.")
        implementation_digests.add(implementation_digest)
        git_provenance = manifest.get("git_provenance")
        if not isinstance(git_provenance, Mapping):
            raise ValueError(f"Qualification manifest {role} lacks Git provenance.")
        git_provenance_digests.add(canonical_sha256(git_provenance))

        role_media = 0
        role_unsupported = 0
        for job_index, job in enumerate(jobs):
            if not isinstance(job, Mapping):
                raise ValueError(f"Qualification manifest {role} job {job_index} is not a map.")
            task = str((job.get("generation") or {}).get("task", ""))
            pair_id = _job_pair_id(job)
            axis = (
                role,
                task,
                int(job.get("seed", -1)),
                str(job.get("prompt_id", "")),
                str(job.get("model_name", "")),
                str(job.get("variation", "")),
                pair_id,
            )
            if axis in actual_axes:
                raise ValueError(f"Qualification phase duplicates axis row {axis}.")
            actual_axes.add(axis)
            if task != spec["task"]:
                raise ValueError(f"Qualification role {role} contains wrong task {task!r}.")
            if job.get("attempt") != attempt or job.get("seed_scoped_output") is not True:
                raise ValueError(f"Qualification role {role} attempt/seed scoping drifted.")
            _validate_variant_contract(phase, job)

            if job.get("model_name") == "flux1_dev":
                if representative_flux_job is None:
                    representative_flux_job = job
                route = job.get("flux1_dual_view_route_v3")
                if not isinstance(route, Mapping):
                    raise ValueError("Qualification FLUX row lacks its shared-v3 route.")
                mode = str(route.get("mode", ""))
                if mode not in {flux_v3.MODE_PREVIEW, flux_v3.MODE_EXECUTION}:
                    raise ValueError("Qualification FLUX route mode is invalid.")
                if require_flux1_execution and mode != flux_v3.MODE_EXECUTION:
                    raise ValueError("Production qualification rejects a preview FLUX row.")
                inputs = route.get("protocol_inputs")
                expected_roles = flux_v3.expected_flux1_protocol_input_roles_v3(mode)
                if not isinstance(inputs, Mapping) or set(inputs) != set(expected_roles):
                    raise ValueError("Qualification FLUX protocol-input role set drifted.")
                normalized_inputs: dict[str, dict[str, str]] = {}
                for input_role, record in inputs.items():
                    if (
                        not isinstance(record, Mapping)
                        or set(record) != {"path", "sha256"}
                        or not isinstance(record.get("path"), str)
                        or not isinstance(record.get("sha256"), str)
                    ):
                        raise ValueError("Qualification FLUX protocol binding is malformed.")
                    normalized_inputs[str(input_role)] = {
                        "path": str(record["path"]),
                        "sha256": str(record["sha256"]),
                    }
                if flux_protocol_inputs is None:
                    flux_protocol_inputs = normalized_inputs
                elif flux_protocol_inputs != normalized_inputs:
                    raise ValueError(
                        "Qualification FLUX rows do not share one exact protocol binding."
                    )
                flux_route_modes.add(mode)

            condition_id = str(job.get("condition_id", ""))
            expected_condition = (
                f"{job['prompt_id']}__{job['model_name']}__{job['variant']}"
                f"__seed_{int(job['seed']):08d}"
            )
            if condition_id != expected_condition:
                raise ValueError(f"Qualification condition ID drifted: {condition_id!r}.")
            if condition_id in condition_ids:
                raise ValueError(f"Qualification condition ID overlaps: {condition_id}.")
            condition_ids.add(condition_id)

            expected_output = _expected_output_dir(
                output_root=resolved_output_root,
                job=job,
                attempt=attempt,
            ).resolve()
            actual_output = Path(str(job.get("output_dir", ""))).resolve()
            if actual_output != expected_output:
                raise ValueError(
                    f"Qualification output path drifted for {condition_id}: "
                    f"expected={expected_output}, actual={actual_output}."
                )
            output_key = str(actual_output)
            if output_key in output_dirs:
                raise ValueError(f"Qualification output directory overlaps: {output_key}.")
            output_dirs.add(output_key)

            expected_media = not (
                str(job["variation"]) == "02_negative_prompt"
                and not EXPECTED_NATIVE_NEGATIVE_SUPPORT[str(job["model_name"])]
            )
            if job.get("expected_media") is not expected_media:
                raise ValueError(f"Qualification media capability drifted for {condition_id}.")
            if expected_media:
                role_media += 1
            else:
                role_unsupported += 1

            task_counts[task] += 1
            seed_counts[int(job["seed"])] += 1
            model_counts[str(job["model_name"])] += 1
            prompt_counts[str(job["prompt_id"])] += 1
            variation_counts[str(job["variation"])] += 1
            if pair_id is not None:
                pair_counts[pair_id] += 1
            union_rows.append(
                {
                    "phase": phase,
                    "role": role,
                    "condition_id": condition_id,
                    "output_dir": output_key,
                    "task": task,
                    "seed": int(job["seed"]),
                    "prompt_id": str(job["prompt_id"]),
                    "model_name": str(job["model_name"]),
                    "variation": str(job["variation"]),
                    "pair_id": pair_id,
                    "expected_media": expected_media,
                }
            )

        counts = {
            "logical": len(jobs),
            "media": role_media,
            "unsupported": role_unsupported,
        }
        if counts != EXPECTED_ROLE_COUNTS[phase][role]:
            raise ValueError(
                f"Qualification manifest {role} count drift: "
                f"expected={EXPECTED_ROLE_COUNTS[phase][role]}, actual={counts}."
            )
        if (
            manifest.get("expected_media_jobs") != role_media
            or manifest.get("expected_not_supported_jobs") != role_unsupported
        ):
            raise ValueError(f"Qualification manifest {role} summary counts drifted.")
        counts_by_role[role] = counts
        total_media += role_media
        total_unsupported += role_unsupported

    if actual_axes != expected_axes:
        missing = sorted(expected_axes - actual_axes)
        extra = sorted(actual_axes - expected_axes)
        raise ValueError(
            f"Qualification axis coverage is not exact: missing={missing[:8]}, extra={extra[:8]}."
        )
    if len(implementation_digests) != 1:
        raise ValueError("Qualification manifests do not share one frozen implementation.")
    if len(git_provenance_digests) != 1:
        raise ValueError("Qualification manifests do not share one frozen Git state.")
    expected_counts = EXPECTED_PHASE_COUNTS[phase]
    observed_counts = {
        "logical": len(union_rows),
        "media": total_media,
        "unsupported": total_unsupported,
    }
    if observed_counts != expected_counts:
        raise ValueError(
            f"Qualification {phase} count drift: expected={expected_counts}, "
            f"actual={observed_counts}."
        )

    union_rows.sort(key=lambda row: (row["role"], row["condition_id"]))
    if require_flux1_execution and (
        flux_protocol_inputs is None or flux_route_modes != {flux_v3.MODE_EXECUTION}
    ):
        raise ValueError("Production qualification lacks one exact FLUX execution binding.")
    if flux_protocol_inputs is not None and len(flux_route_modes) != 1:
        raise ValueError("Qualification FLUX rows mix route modes.")
    should_reauthenticate = (
        require_flux1_execution
        if reauthenticate_flux1_execution is None
        else reauthenticate_flux1_execution
    )
    if should_reauthenticate:
        if not require_flux1_execution or representative_flux_job is None:
            raise ValueError(
                "Live FLUX qualification admission requires an execution-only cohort."
            )
        flux_v3.validate_flux1_job_v3(
            representative_flux_job,
            project_root=resolved_root,
            mode=flux_v3.MODE_EXECUTION,
        )
    flux_binding = (
        {
            "route_mode": next(iter(flux_route_modes)),
            "protocol_input_count": len(flux_protocol_inputs),
            "protocol_inputs_sha256": canonical_sha256(flux_protocol_inputs),
            "acceptance_receipt": deepcopy(
                flux_protocol_inputs.get("native_equivalence_receipt")
            ),
        }
        if flux_protocol_inputs is not None
        else None
    )
    return {
        "phase": phase,
        **observed_counts,
        "manifest_count": len(manifests),
        "unique_axis_rows": len(actual_axes),
        "unique_condition_ids": len(condition_ids),
        "unique_output_dirs": len(output_dirs),
        "implementation_files_sha256": next(iter(implementation_digests)),
        "git_provenance_sha256": next(iter(git_provenance_digests)),
        "counts_by_role": counts_by_role,
        "counts_by_task": dict(sorted(task_counts.items())),
        "counts_by_seed": {str(key): seed_counts[key] for key in sorted(seed_counts)},
        "counts_by_model": dict(sorted(model_counts.items())),
        "counts_by_prompt": dict(sorted(prompt_counts.items())),
        "counts_by_variation": dict(sorted(variation_counts.items())),
        "counts_by_exact_pair": dict(sorted(pair_counts.items())),
        "condition_union_sha256": canonical_sha256(
            sorted(row["condition_id"] for row in union_rows)
        ),
        "output_union_sha256": canonical_sha256(sorted(row["output_dir"] for row in union_rows)),
        "axis_union_sha256": canonical_sha256(union_rows),
        "flux1_execution_binding": flux_binding,
    }


def _qualification_union_proof(
    q1_manifests: Mapping[str, Mapping[str, Any]],
    q2_manifests: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    condition_ids: set[str] = set()
    output_dirs: set[str] = set()
    media = 0
    unsupported = 0
    for phase, manifests in ((Q1, q1_manifests), (Q2, q2_manifests)):
        for role in MANIFEST_ROLE_ORDER:
            for job in manifests[role]["jobs"]:
                condition_id = str(job["condition_id"])
                output_dir = str(Path(job["output_dir"]).resolve())
                if condition_id in condition_ids:
                    raise ValueError(f"Q1/Q2 condition overlap: {condition_id}.")
                if output_dir in output_dirs:
                    raise ValueError(f"Q1/Q2 output overlap: {output_dir}.")
                condition_ids.add(condition_id)
                output_dirs.add(output_dir)
                expected_media = bool(job["expected_media"])
                media += int(expected_media)
                unsupported += int(not expected_media)
                rows.append(
                    {
                        "phase": phase,
                        "role": role,
                        "condition_id": condition_id,
                        "output_dir": output_dir,
                        "expected_media": expected_media,
                    }
                )
    observed = {
        "logical": len(rows),
        "media": media,
        "unsupported": unsupported,
    }
    expected = {"logical": 396, "media": 369, "unsupported": 27}
    if observed != expected:
        raise ValueError(f"Q1/Q2 union count drift: expected={expected}, actual={observed}.")
    rows.sort(key=lambda row: (row["phase"], row["role"], row["condition_id"]))
    return {
        **observed,
        "unique_condition_ids": len(condition_ids),
        "unique_output_dirs": len(output_dirs),
        "q1_q2_union_sha256": canonical_sha256(rows),
    }


def _validate_plan_shape(plan: Mapping[str, Any]) -> None:
    if set(plan) != _PLAN_KEYS:
        raise ValueError(
            f"Qualification plan keys must be exactly {sorted(_PLAN_KEYS)}; got {sorted(plan)}."
        )
    if plan.get("schema_version") != QUALIFICATION_PLAN_SCHEMA_VERSION:
        raise ValueError("Unsupported qualification-plan schema.")
    if plan.get("contract") != QUALIFICATION_PLAN_CONTRACT:
        raise ValueError("Qualification-plan contract mismatch.")
    if plan.get("benchmark") != BENCHMARK_NAME or plan.get("phase") not in PHASES:
        raise ValueError("Qualification-plan benchmark or phase mismatch.")
    _validate_timestamp(plan.get("created_at_utc"), "qualification plan created_at_utc")
    if isinstance(plan.get("attempt"), bool) or not isinstance(plan.get("attempt"), int):
        raise ValueError("Qualification-plan attempt must be an integer.")
    if int(plan["attempt"]) <= 0:
        raise ValueError("Qualification-plan attempt must be positive.")
    if not isinstance(plan.get("output_root"), str) or not plan["output_root"]:
        raise ValueError("Qualification-plan output_root must be explicit.")
    if _SHA256_RE.fullmatch(str(plan.get("implementation_files_sha256", ""))) is None:
        raise ValueError("Qualification plan lacks implementation provenance.")
    if _SHA256_RE.fullmatch(str(plan.get("qualification_plan_sha256", ""))) is None:
        raise ValueError("Qualification plan lacks a canonical identity.")
    bindings = plan.get("manifest_bindings")
    if not isinstance(bindings, list) or len(bindings) != len(MANIFEST_ROLE_ORDER):
        raise ValueError("Qualification plan must bind exactly four manifests.")
    if [binding.get("role") for binding in bindings if isinstance(binding, Mapping)] != list(
        MANIFEST_ROLE_ORDER
    ):
        raise ValueError("Qualification manifest bindings are not in canonical role order.")
    for binding in bindings:
        if not isinstance(binding, Mapping) or set(binding) != _MANIFEST_BINDING_KEYS:
            raise ValueError("Qualification manifest binding has an invalid shape.")
        role = str(binding["role"])
        if (
            binding.get("task") != MANIFEST_ROLE_SPECS[role]["task"]
            or binding.get("seed") != MANIFEST_ROLE_SPECS[role]["seed"]
        ):
            raise ValueError(f"Qualification manifest binding axes drifted for {role}.")
        if not isinstance(binding.get("path"), str) or not binding["path"]:
            raise ValueError(f"Qualification manifest binding path is missing for {role}.")
        for field in ("file_sha256", "manifest_sha256"):
            if _SHA256_RE.fullmatch(str(binding.get(field, ""))) is None:
                raise ValueError(f"Qualification manifest binding lacks {field}.")
        for field in ("logical_rows", "media_rows", "unsupported_rows"):
            value = binding.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"Qualification manifest binding {field} is invalid for {role}.")
    if not isinstance(plan.get("topology_proof"), Mapping):
        raise ValueError("Qualification plan lacks its topology proof.")
    if plan["phase"] == Q1:
        if plan.get("upstream_q1") is not None or plan.get("gate_receipts") != []:
            raise ValueError("Q1 cannot claim upstream Q1 or Q2 gate receipts.")
        if plan.get("qualification_union_proof") is not None:
            raise ValueError("Q1 cannot claim a not-yet-built Q1/Q2 union.")
    else:
        if (
            not isinstance(plan.get("upstream_q1"), Mapping)
            or set(plan["upstream_q1"]) != _UPSTREAM_Q1_KEYS
        ):
            raise ValueError("Q2 must bind one exact upstream Q1 plan.")
        if not isinstance(plan["upstream_q1"].get("path"), str) or not plan["upstream_q1"]["path"]:
            raise ValueError("Q2 upstream Q1 path is missing.")
        for field in ("file_sha256", "qualification_plan_sha256"):
            if _SHA256_RE.fullmatch(str(plan["upstream_q1"].get(field, ""))) is None:
                raise ValueError(f"Q2 upstream Q1 binding lacks {field}.")
        gates = plan.get("gate_receipts")
        if not isinstance(gates, list) or len(gates) != 2:
            raise ValueError("Q2 must bind exactly two gate receipts.")
        if [gate.get("contract") for gate in gates if isinstance(gate, Mapping)] != [
            EXACT_ONE_GATE_CONTRACT,
            NON_REGRESSION_GATE_CONTRACT,
        ]:
            raise ValueError("Q2 gate receipts are missing or out of canonical order.")
        if any(not isinstance(gate, Mapping) or set(gate) != _GATE_BINDING_KEYS for gate in gates):
            raise ValueError("Q2 gate binding has an invalid shape.")
        for gate in gates:
            if not isinstance(gate.get("path"), str) or not gate["path"]:
                raise ValueError("Q2 gate binding path is missing.")
            for field in ("file_sha256", "receipt_sha256"):
                if _SHA256_RE.fullmatch(str(gate.get(field, ""))) is None:
                    raise ValueError(f"Q2 gate binding lacks {field}.")
        if not isinstance(plan.get("qualification_union_proof"), Mapping):
            raise ValueError("Q2 must carry the exact Q1/Q2 union proof.")


def write_qualification_plan_immutable(
    plan: Mapping[str, Any], path: str | Path, *, root: Path | None = None
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    resolved = _resolve(path, root)
    frozen = deepcopy(dict(plan))
    frozen["qualification_plan_sha256"] = qualification_plan_digest(frozen)
    _validate_plan_shape(frozen)
    return _write_authenticated_document(
        frozen,
        resolved,
        digest_field="qualification_plan_sha256",
        digest_function=qualification_plan_digest,
    )


def read_qualification_plan(path: str | Path, *, root: Path | None = None) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    resolved = _resolve(path, root)
    plan = _read_authenticated_document(
        resolved,
        digest_field="qualification_plan_sha256",
        digest_function=qualification_plan_digest,
        label="qualification plan",
    )
    _validate_plan_shape(plan)
    return plan


def qualification_cohort_commit_digest(payload: Mapping[str, Any]) -> str:
    canonical = dict(payload)
    canonical.pop("document_sha256", None)
    return canonical_sha256(canonical)


def _build_qualification_cohort_commit(
    *,
    phase: str,
    plan: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    physical_root: Path,
    root: Path,
) -> dict[str, Any]:
    logical_root = canonical_qualification_cohort_root(phase, root)
    logical_plan = canonical_qualification_plan_path(phase, root)
    physical_plan = physical_root / QUALIFICATION_PLAN_FILENAME
    rows: list[dict[str, Any]] = []
    for role in MANIFEST_ROLE_ORDER:
        filename = QUALIFICATION_MANIFEST_FILENAMES[role]
        physical_manifest = physical_root / filename
        logical_manifest = logical_root / filename
        snapshot_index = Path(f"{physical_manifest}.snapshot") / "index.json"
        logical_snapshot_index = Path(f"{logical_manifest}.snapshot") / "index.json"
        rows.append(
            {
                "role": role,
                "path": str(logical_manifest),
                "file_sha256": _sha256_file(physical_manifest),
                "manifest_sha256": manifests[role]["manifest_sha256"],
                "snapshot_index_path": str(logical_snapshot_index),
                "snapshot_index_sha256": _sha256_file(snapshot_index),
            }
        )
    payload: dict[str, Any] = {
        "schema_version": QUALIFICATION_COHORT_COMMIT_SCHEMA_VERSION,
        "contract": QUALIFICATION_COHORT_COMMIT_CONTRACT,
        "phase": phase,
        "created_at_utc": plan["created_at_utc"],
        "cohort_root": str(logical_root),
        "output_root": str(canonical_qualification_output_root(root)),
        "plan": {
            "path": str(logical_plan),
            "file_sha256": _sha256_file(physical_plan),
            "qualification_plan_sha256": plan["qualification_plan_sha256"],
        },
        "manifests": rows,
        "counts": {
            "manifest_count": len(rows),
            "logical": plan["topology_proof"]["logical"],
            "media": plan["topology_proof"]["media"],
            "unsupported": plan["topology_proof"]["unsupported"],
        },
    }
    payload["document_sha256"] = qualification_cohort_commit_digest(payload)
    return payload


def _validate_qualification_cohort_commit_payload(
    payload: Mapping[str, Any],
    *,
    phase: str,
    plan: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    root: Path,
) -> None:
    cohort_root = canonical_qualification_cohort_root(phase, root)
    if set(payload) != _COHORT_COMMIT_KEYS:
        raise ValueError("Qualification cohort commit has an invalid field set.")
    if (
        payload.get("schema_version") != QUALIFICATION_COHORT_COMMIT_SCHEMA_VERSION
        or payload.get("contract") != QUALIFICATION_COHORT_COMMIT_CONTRACT
        or payload.get("phase") != phase
        or payload.get("created_at_utc") != plan["created_at_utc"]
        or payload.get("cohort_root") != str(cohort_root)
        or payload.get("output_root") != str(canonical_qualification_output_root(root))
        or payload.get("document_sha256")
        != qualification_cohort_commit_digest(payload)
    ):
        raise ValueError("Qualification cohort commit identity or digest drifted.")
    _validate_timestamp(payload["created_at_utc"], "cohort commit created_at_utc")
    plan_binding = payload.get("plan")
    plan_path = canonical_qualification_plan_path(phase, root)
    if (
        not isinstance(plan_binding, Mapping)
        or set(plan_binding) != _COHORT_PLAN_KEYS
        or plan_binding.get("path") != str(plan_path)
        or plan_binding.get("file_sha256") != _sha256_file(plan_path)
        or plan_binding.get("qualification_plan_sha256")
        != plan["qualification_plan_sha256"]
    ):
        raise ValueError("Qualification cohort commit plan binding drifted.")
    rows = payload.get("manifests")
    if not isinstance(rows, list) or [row.get("role") for row in rows] != list(
        MANIFEST_ROLE_ORDER
    ):
        raise ValueError("Qualification cohort commit lacks exactly four ordered manifests.")
    for row in rows:
        if not isinstance(row, Mapping) or set(row) != _COHORT_MANIFEST_KEYS:
            raise ValueError("Qualification cohort manifest binding has an invalid field set.")
        role = str(row["role"])
        path = canonical_qualification_manifest_paths(phase, root)[role]
        index = Path(f"{path}.snapshot") / "index.json"
        manifest = manifests[role]
        descriptor = manifest.get("snapshot_bundle")
        if (
            row.get("path") != str(path)
            or row.get("file_sha256") != _sha256_file(path)
            or row.get("manifest_sha256") != manifest["manifest_sha256"]
            or row.get("snapshot_index_path") != str(index)
            or row.get("snapshot_index_sha256") != _sha256_file(index)
            or not isinstance(descriptor, Mapping)
            or descriptor.get("root_path") != str(Path(f"{path}.snapshot"))
            or descriptor.get("index_path") != str(index)
            or descriptor.get("index_sha256") != row["snapshot_index_sha256"]
        ):
            raise ValueError(f"Qualification cohort commit binding drifted for {role}.")
    expected_counts = {
        "manifest_count": 4,
        "logical": plan["topology_proof"]["logical"],
        "media": plan["topology_proof"]["media"],
        "unsupported": plan["topology_proof"]["unsupported"],
    }
    if payload.get("counts") != expected_counts:
        raise ValueError("Qualification cohort commit count proof drifted.")


def _validate_exact_snapshot_artifact_tree(snapshot_root: Path, *, label: str) -> None:
    index_path = snapshot_root / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    objects = index.get("objects")
    if not isinstance(objects, Mapping):
        raise ValueError(f"Qualification snapshot object index is invalid for {label}.")
    expected_files = {Path("index.json")}
    for record in objects.values():
        if not isinstance(record, Mapping):
            raise ValueError(f"Qualification snapshot object record is invalid for {label}.")
        relative = Path(str(record.get("path", "")))
        if (
            relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
            or relative.parts[:2] != ("objects", "sha256")
        ):
            raise ValueError(f"Qualification snapshot object path is invalid for {label}.")
        expected_files.add(relative)
    actual_files = {
        path.relative_to(snapshot_root)
        for path in snapshot_root.rglob("*")
        if path.is_file()
    }
    expected_directories = {
        parent
        for relative in expected_files
        for parent in relative.parents
        if parent != Path(".")
    }
    actual_directories = {
        path.relative_to(snapshot_root)
        for path in snapshot_root.rglob("*")
        if path.is_dir()
    }
    if actual_files != expected_files or actual_directories != expected_directories:
        raise ValueError(f"Qualification snapshot artifact tree is inexact for {label}.")


def read_qualification_cohort_commit(
    phase: str,
    *,
    plan: Mapping[str, Any],
    manifests: Mapping[str, Mapping[str, Any]],
    root: Path | None = None,
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    cohort_root = canonical_qualification_cohort_root(phase, root)
    expected_entries = {
        QUALIFICATION_PLAN_FILENAME,
        f"{QUALIFICATION_PLAN_FILENAME}.sha256",
        QUALIFICATION_COHORT_COMMIT_FILENAME,
        f"{QUALIFICATION_COHORT_COMMIT_FILENAME}.sha256",
    }
    for filename in QUALIFICATION_MANIFEST_FILENAMES.values():
        expected_entries.update({filename, f"{filename}.sha256", f"{filename}.snapshot"})
    if cohort_root.is_symlink() or not cohort_root.is_dir():
        raise FileNotFoundError(f"Canonical qualification cohort is absent: {cohort_root}.")
    if _existing_component_is_symlink(cohort_root, root=root):
        raise ValueError("Canonical qualification cohort path contains a symlink.")
    if {path.name for path in cohort_root.iterdir()} != expected_entries:
        raise ValueError("Qualification cohort has unexpected or missing artifacts.")
    for path in cohort_root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Qualification cohort contains a symlink: {path}.")
    for role, manifest_path in canonical_qualification_manifest_paths(
        phase, root
    ).items():
        _validate_exact_snapshot_artifact_tree(
            Path(f"{manifest_path}.snapshot"), label=role
        )
    _require_nonwritable_directories(
        cohort_root, label="Canonical qualification cohort"
    )
    commit_path = canonical_qualification_cohort_commit_path(phase, root)
    payload = _read_authenticated_document(
        commit_path,
        digest_field="document_sha256",
        digest_function=qualification_cohort_commit_digest,
        label="qualification cohort commit",
    )
    _validate_qualification_cohort_commit_payload(
        payload,
        phase=phase,
        plan=plan,
        manifests=manifests,
        root=root,
    )
    return payload


def _q1_binding(validated: ValidatedQualificationPlan) -> dict[str, Any]:
    return {
        "path": str(validated.path),
        "file_sha256": _sha256_file(validated.path),
        "qualification_plan_sha256": validated.digest,
    }


def _evidence_binding(path: str | Path, root: Path) -> dict[str, Any]:
    resolved = _resolve(path, root)
    if not resolved.is_file():
        raise FileNotFoundError(f"Gate evidence is missing: {resolved}")
    return {
        "path": str(resolved),
        "sha256": _sha256_file(resolved),
        "size_bytes": resolved.stat().st_size,
    }


def _external_evidence_path(path: Path, q1: ValidatedQualificationPlan, label: str) -> Path:
    generation_root = Path(q1.plan["output_root"]).resolve()
    resolved = path.resolve()
    if resolved == generation_root or generation_root in resolved.parents:
        raise ValueError(f"{label} must be stored outside the generation output root.")
    return resolved


def _exact_one_q1_rows(
    q1: ValidatedQualificationPlan,
) -> list[tuple[str, Mapping[str, Any], int, Mapping[str, Any]]]:
    rows: list[tuple[str, Mapping[str, Any], int, Mapping[str, Any]]] = []
    for role in MANIFEST_ROLE_ORDER:
        manifest = q1.manifests[role]
        for index, job in enumerate(manifest["jobs"]):
            if job["variation"] == "06_shapley_concept_steering_single_pair":
                rows.append((role, manifest, index, job))
    if len(rows) != 66 or len({row[3]["condition_id"] for row in rows}) != 66:
        raise ValueError("Q1 does not contain exactly 66 unique Shapley exact-one rows.")
    return rows


def _load_json_evidence(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot decode {label} {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}.")
    return payload


def _derive_exact_one_claims(
    path: Path, *, q1: ValidatedQualificationPlan, root: Path
) -> dict[str, Any]:
    evidence_path = _external_evidence_path(path, q1, "Exact-one evidence index")
    payload = _load_json_evidence(evidence_path, label="exact-one evidence index")
    required = {
        "schema_version",
        "contract",
        "q1_plan_sha256",
        "implementation_files_sha256",
        "row_evidence",
    }
    if (
        set(payload) != required
        or payload.get("schema_version") != EXACT_ONE_EVIDENCE_SCHEMA_VERSION
        or payload.get("contract") != EXACT_ONE_EVIDENCE_CONTRACT
        or payload.get("q1_plan_sha256") != q1.digest
        or payload.get("implementation_files_sha256")
        != q1.plan["implementation_files_sha256"]
    ):
        raise ValueError("Exact-one evidence index identity/source binding drifted.")
    rows = _exact_one_q1_rows(q1)
    routing = payload.get("row_evidence")
    expected_conditions = [str(job["condition_id"]) for _, _, _, job in rows]
    if not isinstance(routing, Mapping) or list(routing) != expected_conditions:
        raise ValueError(
            "Exact-one evidence routing must cover the exact 66 rows in plan order."
        )

    from hierasafe_flow.benchmarks.finer_detailing_qualification_launch import (
        read_qualification_launch_authorization,
    )
    from hierasafe_flow.benchmarks.slurm_tracking import read_environment_preflight
    from hierasafe_flow.benchmarks.slurm_tracking import EXECUTION_IDENTITY_FILENAME
    from hierasafe_flow.evaluation.production_smoke import (
        EXACT_ONE_PATH_GATE,
        _live_media_validation,
        _validate_manual_ledger,
        _validate_trace_report,
    )

    for role, manifest, index, job in rows:
        route = routing[job["condition_id"]]
        if (
            not isinstance(route, Mapping)
            or set(route) != {"manual_ledger_path", "manual_ledger_sha256"}
            or not isinstance(route["manual_ledger_path"], str)
            or _SHA256_RE.fullmatch(str(route["manual_ledger_sha256"])) is None
        ):
            raise ValueError(f"Exact-one manual evidence routing is invalid for {job['condition_id']}.")
        output = Path(str(job["output_dir"])).resolve()
        exact_job = {
            **job,
            "launch_manifest_sha256": manifest["manifest_sha256"],
            "launch_manifest_job_index": index,
        }
        authorization = read_qualification_launch_authorization(
            output,
            expected_plan=q1,
            expected_role=role,
            expected_index=index,
            root=root,
        )
        preflight = read_environment_preflight(
            output,
            expected_job=exact_job,
            expected_job_index=index,
        )
        if preflight.get("status") != "verified_before_generation":
            raise ValueError("Exact-one row lacks a successful runtime environment preflight.")
        result_path = output / "benchmark_job_result.json"
        result = _load_json_evidence(result_path, label="exact-one runner result")
        if (
            result.get("status") != "completed"
            or result.get("job") != exact_job
            or not isinstance(result.get("media_validation"), Mapping)
        ):
            raise ValueError(f"Exact-one runner result is not exact/completed: {result_path}.")
        if is_flux1_job_v3(exact_job):
            independently_bound_result_sha256 = _sha256_file(result_path)
            reopened = reopen_completed_flux1_output_v3(
                exact_job,
                root=root,
                manifest_path=canonical_qualification_manifest_paths(q1.phase, root)[role],
                manifest_sha256=str(manifest["manifest_sha256"]),
                manifest_job_index=index,
                result_path=result_path,
            )
            if (
                reopened["result"] != result
                or reopened["result_sha256"] != independently_bound_result_sha256
            ):
                raise ValueError(
                    "Exact-one FLUX-v3 result/hash changed during strict reopening."
                )
        live_media = _live_media_validation(output, job, result["media_validation"])
        expected_media = output / "sample_0000" / (
            "video_000.mp4"
            if job["generation"]["task"] == "text_to_video"
            else "image_000.png"
        )
        if result.get("validated_media_paths") != [str(expected_media)]:
            raise ValueError(f"Exact-one result does not bind one canonical media file: {output}.")
        media_sha = str(live_media["validation"]["sha256"])
        trace = _validate_trace_report(
            output / "sample_0000" / "report.json",
            job,
            manifest_sha256=str(manifest["manifest_sha256"]),
        )
        if trace["numeric_leaf_count"] <= 0:
            raise ValueError(f"Exact-one trace contains no finite runtime evidence: {output}.")
        manual_path = _external_evidence_path(
            Path(route["manual_ledger_path"]), q1, "Exact-one manual ledger"
        )
        _validate_manual_ledger(
            manual_path,
            contract=EXACT_ONE_PATH_GATE,
            job=job,
            media_sha256=media_sha,
            expected_sha256=str(route["manual_ledger_sha256"]),
        )
        if authorization.get("condition_id") != job["condition_id"]:
            raise ValueError("Exact-one qualification launch authorization row drifted.")
        execution_path = output / EXECUTION_IDENTITY_FILENAME
        execution = _load_json_evidence(
            execution_path, label="exact-one execution identity"
        )
        if (
            set(execution)
            != {
                "schema_version",
                "captured_at_utc",
                "SLURM_JOB_ID",
                "SLURM_ARRAY_JOB_ID",
                "SLURM_ARRAY_TASK_ID",
                "SLURM_JOB_NAME",
                "slurm_task_id",
            }
            or execution.get("schema_version") != 1
            or execution.get("SLURM_JOB_ID") != authorization["slurm_job_id"]
            or execution.get("SLURM_ARRAY_JOB_ID")
            != authorization["slurm_array_job_id"]
            or execution.get("SLURM_ARRAY_TASK_ID") != str(index)
            or execution.get("SLURM_JOB_NAME") != authorization["slurm_job_name"]
            or execution.get("slurm_task_id") != authorization["slurm_task_id"]
        ):
            raise ValueError(f"Exact-one execution identity drifted: {execution_path}.")
        _validate_timestamp(
            execution["captured_at_utc"], "exact-one execution captured_at_utc"
        )
        timing_path = output / "experiment_timing.json"
        timing = _load_json_evidence(timing_path, label="exact-one experiment timing")
        if (
            set(timing)
            != {"started_at_utc", "finished_at_utc", "wall_seconds", "status"}
            or timing.get("status") != "completed"
            or isinstance(timing.get("wall_seconds"), bool)
            or not isinstance(timing.get("wall_seconds"), (int, float))
            or not math.isfinite(float(timing["wall_seconds"]))
            or float(timing["wall_seconds"]) <= 0
        ):
            raise ValueError(f"Exact-one runtime timing is malformed: {timing_path}.")
        started = _validate_timestamp(
            timing["started_at_utc"], "exact-one timing started_at_utc"
        )
        finished = _validate_timestamp(
            timing["finished_at_utc"], "exact-one timing finished_at_utc"
        )
        if finished <= started:
            raise ValueError(f"Exact-one runtime timing is non-positive: {timing_path}.")
    return exact_one_gate_claims()


def _derive_non_regression_claims(
    path: Path, *, q1: ValidatedQualificationPlan, root: Path
) -> dict[str, Any]:
    evidence_path = _external_evidence_path(path, q1, "Non-regression evidence report")
    payload = _load_json_evidence(evidence_path, label="non-regression evidence report")
    required = {
        "schema_version",
        "contract",
        "q1_plan_sha256",
        "implementation_files_sha256",
        "started_at_utc",
        "completed_at_utc",
        "command",
        "exit_code",
        "test_results",
        "stdout",
        "stderr",
        "structured_output_sha256",
    }
    from hierasafe_flow.evaluation.production_smoke import REQUIRED_NON_REGRESSION_TESTS

    expected_command = [sys.executable, "-m", "pytest", "-q", *REQUIRED_NON_REGRESSION_TESTS]
    results = payload.get("test_results")
    if (
        set(payload) != required
        or payload.get("schema_version") != NON_REGRESSION_EVIDENCE_SCHEMA_VERSION
        or payload.get("contract") != NON_REGRESSION_EVIDENCE_CONTRACT
        or payload.get("q1_plan_sha256") != q1.digest
        or payload.get("implementation_files_sha256")
        != q1.plan["implementation_files_sha256"]
        or payload.get("command") != expected_command
        or payload.get("exit_code") != 0
        or not isinstance(payload.get("stdout"), str)
        or re.search(
            rf"(?:^|\s){len(REQUIRED_NON_REGRESSION_TESTS)} passed(?:\s|,|$)",
            payload["stdout"],
        )
        is None
        or not isinstance(payload.get("stderr"), str)
        or not isinstance(results, list)
        or [item.get("nodeid") for item in results]
        != list(REQUIRED_NON_REGRESSION_TESTS)
    ):
        raise ValueError("Non-regression evidence identity/source/test coverage drifted.")
    started = _validate_timestamp(payload["started_at_utc"], "non-regression started_at_utc")
    completed = _validate_timestamp(
        payload["completed_at_utc"], "non-regression completed_at_utc"
    )
    if completed < started:
        raise ValueError("Non-regression evidence completion predates its start.")
    if any(
        not isinstance(item, Mapping)
        or set(item) != {"nodeid", "outcome", "duration_seconds"}
        or item.get("outcome") != "passed"
        or isinstance(item.get("duration_seconds"), bool)
        or not isinstance(item.get("duration_seconds"), (int, float))
        or not math.isfinite(float(item["duration_seconds"]))
        or float(item["duration_seconds"]) < 0
        for item in results
    ):
        raise ValueError("Non-regression evidence contains a failed/malformed test record.")
    structured = {
        "command": expected_command,
        "exit_code": 0,
        "test_results": results,
    }
    if payload.get("structured_output_sha256") != canonical_sha256(structured):
        raise ValueError("Non-regression structured runtime output digest drifted.")
    environment = dict(os.environ)
    environment["PYTEST_ADDOPTS"] = ""
    completed_run = subprocess.run(
        expected_command,
        cwd=root,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=1800,
    )
    if completed_run.returncode != 0:
        raise ValueError(
            "Independent live non-regression rerun failed: "
            f"{completed_run.stdout}\n{completed_run.stderr}"
        )
    return non_regression_gate_claims(passing_test_cases=len(results))


def _derive_gate_claims_from_evidence(
    contract: str,
    *,
    evidence: Sequence[Mapping[str, Any]],
    q1: ValidatedQualificationPlan,
    root: Path,
) -> dict[str, Any]:
    if len(evidence) != 1:
        raise ValueError("Each qualification gate requires exactly one structured evidence file.")
    path = Path(str(evidence[0]["path"])).resolve()
    if contract == EXACT_ONE_GATE_CONTRACT:
        return _derive_exact_one_claims(path, q1=q1, root=root)
    if contract == NON_REGRESSION_GATE_CONTRACT:
        return _derive_non_regression_claims(path, q1=q1, root=root)
    raise ValueError(f"Unknown qualification gate contract {contract!r}.")


def build_gate_receipt(
    contract: str,
    *,
    q1_plan_path: str | Path,
    evidence_paths: Sequence[str | Path],
    claims: Mapping[str, Any],
    root: Path | None = None,
    created_at_utc: str | None = None,
) -> dict[str, Any]:
    """Derive gate claims from runtime evidence and serialize an exact receipt."""

    root = (root or project_root()).resolve()
    if contract not in {EXACT_ONE_GATE_CONTRACT, NON_REGRESSION_GATE_CONTRACT}:
        raise ValueError(f"Unknown qualification gate contract {contract!r}.")
    q1 = validate_qualification_plan(q1_plan_path, root=root, expected_phase=Q1)
    evidence = [_evidence_binding(path, root) for path in evidence_paths]
    if not evidence:
        raise ValueError("Qualification gate receipt requires at least one evidence file.")
    if len({item["path"] for item in evidence}) != len(evidence):
        raise ValueError("Qualification gate receipt evidence paths must be unique.")
    derived_claims = _derive_gate_claims_from_evidence(
        contract,
        evidence=evidence,
        q1=q1,
        root=root,
    )
    if dict(claims) != derived_claims:
        raise ValueError(
            "Qualification gate claims differ from independently derived runtime evidence."
        )
    receipt: dict[str, Any] = {
        "schema_version": GATE_RECEIPT_SCHEMA_VERSION,
        "contract": contract,
        "benchmark": BENCHMARK_NAME,
        "created_at_utc": created_at_utc or _utc_now(),
        "q1_plan": _q1_binding(q1),
        "implementation_files_sha256": q1.plan["implementation_files_sha256"],
        "decision": "pass",
        "claims": deepcopy(derived_claims),
        "evidence": evidence,
    }
    receipt["receipt_sha256"] = gate_receipt_digest(receipt)
    _validate_gate_receipt_payload(receipt, q1=q1, root=root, derive_evidence=False)
    return receipt


def _validate_gate_receipt_payload(
    receipt: Mapping[str, Any],
    *,
    q1: ValidatedQualificationPlan,
    root: Path,
    derive_evidence: bool = True,
) -> None:
    if set(receipt) != _RECEIPT_KEYS:
        raise ValueError(
            f"Gate receipt keys must be exactly {sorted(_RECEIPT_KEYS)}; got {sorted(receipt)}."
        )
    if receipt.get("schema_version") != GATE_RECEIPT_SCHEMA_VERSION:
        raise ValueError("Unsupported qualification gate-receipt schema.")
    contract = receipt.get("contract")
    if contract not in {EXACT_ONE_GATE_CONTRACT, NON_REGRESSION_GATE_CONTRACT}:
        raise ValueError("Qualification gate-receipt contract mismatch.")
    if receipt.get("benchmark") != BENCHMARK_NAME or receipt.get("decision") != "pass":
        raise ValueError("Qualification gate receipt is not an explicit benchmark pass.")
    created = _validate_timestamp(receipt.get("created_at_utc"), "gate receipt created_at_utc")
    q1_created = _validate_timestamp(q1.plan["created_at_utc"], "upstream Q1 created_at_utc")
    if created < q1_created:
        raise ValueError("Qualification gate receipt predates its upstream Q1 plan.")
    if receipt.get("q1_plan") != _q1_binding(q1):
        raise ValueError("Qualification gate receipt does not bind the exact Q1 plan.")
    if receipt.get("implementation_files_sha256") != q1.plan["implementation_files_sha256"]:
        raise ValueError("Qualification gate receipt implementation binding drifted.")
    if gate_receipt_digest(receipt) != receipt.get("receipt_sha256"):
        raise ValueError("Qualification gate receipt canonical digest mismatch.")

    claims = receipt.get("claims")
    if not isinstance(claims, Mapping):
        raise ValueError("Qualification gate receipt claims must be a mapping.")
    if contract == EXACT_ONE_GATE_CONTRACT:
        if dict(claims) != _EXACT_ONE_CLAIMS:
            raise ValueError(
                "Exact-one gate receipt must prove all 66 rows across every registered gate."
            )
    else:
        if set(claims) != {
            *_NON_REGRESSION_REQUIRED_BOOLEANS,
            "full_sequential_pair_test_cases_passed",
            "failed_test_cases",
        }:
            raise ValueError("Non-regression gate receipt claims have an invalid shape.")
        for field, expected in _NON_REGRESSION_REQUIRED_BOOLEANS.items():
            if claims.get(field) is not expected:
                raise ValueError(f"Non-regression gate claim {field} did not pass.")
        passed = claims.get("full_sequential_pair_test_cases_passed")
        if isinstance(passed, bool) or not isinstance(passed, int) or passed <= 0:
            raise ValueError("Non-regression receipt must bind at least one passing test case.")
        if claims.get("failed_test_cases") != 0:
            raise ValueError("Non-regression gate receipt reports failing test cases.")

    evidence = receipt.get("evidence")
    if not isinstance(evidence, list) or not evidence:
        raise ValueError("Qualification gate receipt requires evidence bindings.")
    paths: set[str] = set()
    for item in evidence:
        if not isinstance(item, Mapping) or set(item) != _EVIDENCE_KEYS:
            raise ValueError("Qualification gate evidence binding has an invalid shape.")
        if _SHA256_RE.fullmatch(str(item.get("sha256", ""))) is None:
            raise ValueError("Qualification gate evidence binding lacks a SHA-256 digest.")
        size = item.get("size_bytes")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ValueError("Qualification gate evidence binding size is invalid.")
        path = Path(str(item["path"])).resolve()
        path_key = str(path)
        if path_key in paths:
            raise ValueError("Qualification gate evidence paths are duplicated.")
        paths.add(path_key)
        if not path.is_file():
            raise FileNotFoundError(f"Qualification gate evidence disappeared: {path}")
        if (
            item.get("sha256") != _sha256_file(path)
            or item.get("size_bytes") != path.stat().st_size
        ):
            raise ValueError(f"Qualification gate evidence binding drifted: {path}")
    if derive_evidence:
        derived_claims = _derive_gate_claims_from_evidence(
            str(contract),
            evidence=evidence,
            q1=q1,
            root=root,
        )
        if dict(claims) != derived_claims:
            raise ValueError(
                "Qualification gate receipt claims no longer match live runtime evidence."
            )


def write_gate_receipt_immutable(
    receipt: Mapping[str, Any],
    path: str | Path,
    *,
    q1_plan_path: str | Path,
    root: Path | None = None,
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    q1 = validate_qualification_plan(q1_plan_path, root=root, expected_phase=Q1)
    frozen = deepcopy(dict(receipt))
    frozen["receipt_sha256"] = gate_receipt_digest(frozen)
    _validate_gate_receipt_payload(frozen, q1=q1, root=root)
    return _write_authenticated_document(
        frozen,
        _resolve(path, root),
        digest_field="receipt_sha256",
        digest_function=gate_receipt_digest,
    )


def read_gate_receipt(
    path: str | Path,
    *,
    q1: ValidatedQualificationPlan,
    expected_contract: str,
    root: Path | None = None,
) -> dict[str, Any]:
    root = (root or project_root()).resolve()
    resolved = _resolve(path, root)
    receipt = _read_authenticated_document(
        resolved,
        digest_field="receipt_sha256",
        digest_function=gate_receipt_digest,
        label="qualification gate receipt",
    )
    if receipt.get("contract") != expected_contract:
        raise ValueError(
            f"Qualification gate receipt contract mismatch: expected={expected_contract}, "
            f"actual={receipt.get('contract')}."
        )
    _validate_gate_receipt_payload(receipt, q1=q1, root=root)
    if resolved in {
        Path(item["path"]).resolve() for item in receipt["evidence"]
    } or resolved.with_suffix(resolved.suffix + ".sha256") in {
        Path(item["path"]).resolve() for item in receipt["evidence"]
    }:
        raise ValueError("Qualification gate receipt cannot cite itself as evidence.")
    return receipt


def _manifest_binding(
    role: str,
    path: Path,
    manifest: Mapping[str, Any],
    *,
    content_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "role": role,
        "task": MANIFEST_ROLE_SPECS[role]["task"],
        "seed": MANIFEST_ROLE_SPECS[role]["seed"],
        "path": str(path),
        "file_sha256": _sha256_file(content_path or path),
        "manifest_sha256": str(manifest["manifest_sha256"]),
        "logical_rows": int(manifest["num_jobs"]),
        "media_rows": int(manifest["expected_media_jobs"]),
        "unsupported_rows": int(manifest["expected_not_supported_jobs"]),
    }


def _gate_binding(path: Path, receipt: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "contract": str(receipt["contract"]),
        "path": str(path),
        "file_sha256": _sha256_file(path),
        "receipt_sha256": str(receipt["receipt_sha256"]),
    }


def _build_qualification_plan_payload(
    *,
    phase: str,
    manifest_paths: Mapping[str, Path],
    manifests: Mapping[str, Mapping[str, Any]],
    topology_proof: Mapping[str, Any],
    attempt: int,
    output_root: Path,
    upstream_q1: ValidatedQualificationPlan | None,
    gate_receipt_paths: Sequence[Path],
    gate_receipts: Sequence[Mapping[str, Any]],
    manifest_content_paths: Mapping[str, Path] | None = None,
) -> dict[str, Any]:
    union_proof = None
    upstream_binding = None
    if phase == Q2:
        if upstream_q1 is None:
            raise ValueError("Q2 plan construction requires validated upstream Q1.")
        union_proof = _qualification_union_proof(upstream_q1.manifests, manifests)
        upstream_binding = _q1_binding(upstream_q1)
    plan: dict[str, Any] = {
        "schema_version": QUALIFICATION_PLAN_SCHEMA_VERSION,
        "contract": QUALIFICATION_PLAN_CONTRACT,
        "benchmark": BENCHMARK_NAME,
        "phase": phase,
        "created_at_utc": _utc_now(),
        "attempt": attempt,
        "output_root": str(output_root),
        "implementation_files_sha256": topology_proof["implementation_files_sha256"],
        "manifest_bindings": [
            _manifest_binding(
                role,
                manifest_paths[role],
                manifests[role],
                content_path=(manifest_content_paths or manifest_paths)[role],
            )
            for role in MANIFEST_ROLE_ORDER
        ],
        "topology_proof": deepcopy(dict(topology_proof)),
        "upstream_q1": upstream_binding,
        "gate_receipts": [
            _gate_binding(path, receipt)
            for path, receipt in zip(gate_receipt_paths, gate_receipts, strict=True)
        ],
        "qualification_union_proof": union_proof,
    }
    plan["qualification_plan_sha256"] = qualification_plan_digest(plan)
    _validate_plan_shape(plan)
    return plan


def validate_qualification_plan_for_audit(
    path: str | Path,
    *,
    root: Path | None = None,
    expected_phase: str | None = None,
) -> ValidatedQualificationPlan:
    """Authenticate cohort inputs from snapshots while keeping live code exact.

    This reader is reserved for later promotion/audit after the five model
    configs have deliberately changed from pilot to production. Implementation
    source and Git provenance must still equal Q1/Q2 exactly. Generation and
    launch paths use :func:`validate_qualification_plan`, which additionally
    requires every live protocol input byte to remain equal.
    """

    root = (root or project_root()).resolve()
    resolved_path = _lexical_absolute(path, root, label="qualification audit plan path")
    plan = read_qualification_plan(resolved_path, root=root)
    phase = str(plan["phase"])
    if expected_phase is not None and phase != expected_phase:
        raise ValueError(f"Expected qualification phase {expected_phase}, found {phase}.")
    if resolved_path != canonical_qualification_plan_path(phase, root):
        raise ValueError("Qualification audit plan is outside its exact atomic cohort.")

    manifests: dict[str, dict[str, Any]] = {}
    expected_paths = canonical_qualification_manifest_paths(phase, root)
    for binding in plan["manifest_bindings"]:
        role = str(binding["role"])
        manifest_path = _lexical_absolute(
            binding["path"], root, label=f"qualification audit manifest for {role}"
        )
        if manifest_path != expected_paths[role]:
            raise ValueError(f"Qualification audit manifest path is noncanonical for {role}.")
        if _sha256_file(manifest_path) != binding["file_sha256"]:
            raise ValueError(f"Qualification audit manifest file binding drifted: {manifest_path}")
        manifest = read_manifest_for_audit(manifest_path, root)
        _verify_implementation_provenance(manifest, root)
        if manifest.get("manifest_sha256") != binding["manifest_sha256"]:
            raise ValueError(f"Qualification audit manifest identity drifted: {manifest_path}")
        if {
            "logical_rows": manifest["num_jobs"],
            "media_rows": manifest["expected_media_jobs"],
            "unsupported_rows": manifest["expected_not_supported_jobs"],
        } != {
            field: binding[field]
            for field in ("logical_rows", "media_rows", "unsupported_rows")
        }:
            raise ValueError(f"Qualification audit manifest counts drifted: {manifest_path}")
        manifests[role] = manifest

    topology = validate_qualification_manifests(
        phase,
        manifests,
        output_root=plan["output_root"],
        attempt=int(plan["attempt"]),
        require_flux1_execution=True,
        root=root,
        reauthenticate_flux1_execution=False,
    )
    if topology != plan["topology_proof"] or topology[
        "implementation_files_sha256"
    ] != plan["implementation_files_sha256"]:
        raise ValueError("Qualification audit topology/source proof drifted.")

    upstream: ValidatedQualificationPlan | None = None
    receipts: list[dict[str, Any]] = []
    if phase == Q2:
        q1_binding = plan["upstream_q1"]
        q1_path = _resolve(q1_binding["path"], root)
        if _sha256_file(q1_path) != q1_binding["file_sha256"]:
            raise ValueError("Qualification audit Q1 file binding drifted.")
        upstream = validate_qualification_plan_for_audit(
            q1_path, root=root, expected_phase=Q1
        )
        if upstream.digest != q1_binding["qualification_plan_sha256"]:
            raise ValueError("Qualification audit Q1 identity drifted.")
        if (
            upstream.plan["attempt"] != plan["attempt"]
            or upstream.plan["output_root"] != plan["output_root"]
            or upstream.plan["implementation_files_sha256"]
            != plan["implementation_files_sha256"]
            or upstream.topology_proof["git_provenance_sha256"]
            != topology["git_provenance_sha256"]
        ):
            raise ValueError("Qualification audit Q1/Q2 cohort binding drifted.")
        for binding, contract in zip(
            plan["gate_receipts"],
            (EXACT_ONE_GATE_CONTRACT, NON_REGRESSION_GATE_CONTRACT),
            strict=True,
        ):
            receipt_path = _resolve(binding["path"], root)
            if _sha256_file(receipt_path) != binding["file_sha256"]:
                raise ValueError(f"Qualification audit gate file drifted: {receipt_path}")
            receipt = _read_authenticated_document(
                receipt_path,
                digest_field="receipt_sha256",
                digest_function=gate_receipt_digest,
                label="qualification audit gate receipt",
            )
            if receipt.get("contract") != contract:
                raise ValueError("Qualification audit gate contract drifted.")
            _validate_gate_receipt_payload(
                receipt, q1=upstream, root=root, derive_evidence=False
            )
            if receipt["receipt_sha256"] != binding["receipt_sha256"]:
                raise ValueError("Qualification audit gate identity drifted.")
            receipts.append(receipt)
        plan_created = _validate_timestamp(
            plan["created_at_utc"], "Q2 qualification audit plan created_at_utc"
        )
        if any(
            _validate_timestamp(
                receipt["created_at_utc"], "Q2 qualification audit receipt created_at_utc"
            )
            > plan_created
            for receipt in receipts
        ):
            raise ValueError("Qualification audit Q2 plan predates a gate receipt.")
        if _qualification_union_proof(upstream.manifests, manifests) != plan[
            "qualification_union_proof"
        ]:
            raise ValueError("Qualification audit Q1/Q2 union proof drifted.")

    read_qualification_cohort_commit(
        phase,
        plan=plan,
        manifests=manifests,
        root=root,
    )
    return ValidatedQualificationPlan(
        path=resolved_path,
        plan=plan,
        manifests=manifests,
        topology_proof=topology,
        upstream_q1=upstream,
        gate_receipts=tuple(receipts),
    )


def validate_qualification_plan(
    path: str | Path,
    *,
    root: Path | None = None,
    expected_phase: str | None = None,
) -> ValidatedQualificationPlan:
    """Reopen every bound manifest/receipt and recompute the complete proof."""

    root = (root or project_root()).resolve()
    resolved_path = _lexical_absolute(path, root, label="qualification plan path")
    plan = read_qualification_plan(resolved_path, root=root)
    phase = str(plan["phase"])
    if expected_phase is not None and phase != expected_phase:
        raise ValueError(f"Expected qualification phase {expected_phase}, found {phase}.")
    expected_plan_path = canonical_qualification_plan_path(phase, root)
    if resolved_path != expected_plan_path:
        raise ValueError(
            "Qualification plan is outside its exact atomic cohort: "
            f"expected={expected_plan_path}, actual={resolved_path}."
        )

    manifests: dict[str, dict[str, Any]] = {}
    expected_manifest_paths = canonical_qualification_manifest_paths(phase, root)
    for binding in plan["manifest_bindings"]:
        role = str(binding["role"])
        manifest_path = _lexical_absolute(
            binding["path"], root, label=f"qualification manifest path for {role}"
        )
        if manifest_path != expected_manifest_paths[role]:
            raise ValueError(f"Qualification manifest path is noncanonical for {role}.")
        if _sha256_file(manifest_path) != binding["file_sha256"]:
            raise ValueError(f"Qualification manifest file binding drifted: {manifest_path}")
        manifest = read_manifest(manifest_path, root)
        if manifest.get("manifest_sha256") != binding["manifest_sha256"]:
            raise ValueError(f"Qualification manifest identity drifted: {manifest_path}")
        expected_binding_counts = {
            "logical_rows": manifest["num_jobs"],
            "media_rows": manifest["expected_media_jobs"],
            "unsupported_rows": manifest["expected_not_supported_jobs"],
        }
        if any(binding[field] != value for field, value in expected_binding_counts.items()):
            raise ValueError(f"Qualification manifest count binding drifted: {manifest_path}")
        manifests[role] = manifest

    topology = validate_qualification_manifests(
        phase,
        manifests,
        output_root=plan["output_root"],
        attempt=int(plan["attempt"]),
        require_flux1_execution=True,
        root=root,
    )
    if topology != plan["topology_proof"]:
        raise ValueError("Qualification plan topology proof does not recompute exactly.")
    if topology["implementation_files_sha256"] != plan["implementation_files_sha256"]:
        raise ValueError("Qualification plan implementation binding drifted.")

    upstream: ValidatedQualificationPlan | None = None
    receipts: list[dict[str, Any]] = []
    if phase == Q2:
        q1_binding = plan["upstream_q1"]
        q1_path = _resolve(q1_binding["path"], root)
        if _sha256_file(q1_path) != q1_binding["file_sha256"]:
            raise ValueError("Q2 upstream Q1 file binding drifted.")
        upstream = validate_qualification_plan(q1_path, root=root, expected_phase=Q1)
        if upstream.digest != q1_binding["qualification_plan_sha256"]:
            raise ValueError("Q2 upstream Q1 canonical identity drifted.")
        if (
            upstream.plan["attempt"] != plan["attempt"]
            or upstream.plan["output_root"] != plan["output_root"]
        ):
            raise ValueError("Q2 attempt/output root differs from its upstream Q1 cohort.")
        if upstream.plan["implementation_files_sha256"] != plan["implementation_files_sha256"]:
            raise ValueError("Q2 implementation differs from its upstream Q1 cohort.")
        if upstream.topology_proof["git_provenance_sha256"] != topology["git_provenance_sha256"]:
            raise ValueError("Q2 Git provenance differs from its upstream Q1 cohort.")
        expected_contracts = (EXACT_ONE_GATE_CONTRACT, NON_REGRESSION_GATE_CONTRACT)
        for binding, contract in zip(plan["gate_receipts"], expected_contracts, strict=True):
            receipt_path = _resolve(binding["path"], root)
            if _sha256_file(receipt_path) != binding["file_sha256"]:
                raise ValueError(f"Q2 gate receipt file binding drifted: {receipt_path}")
            receipt = read_gate_receipt(
                receipt_path,
                q1=upstream,
                expected_contract=contract,
                root=root,
            )
            if receipt["receipt_sha256"] != binding["receipt_sha256"]:
                raise ValueError(f"Q2 gate receipt identity drifted: {receipt_path}")
            receipts.append(receipt)
        plan_created = _validate_timestamp(
            plan["created_at_utc"], "Q2 qualification plan created_at_utc"
        )
        if any(
            _validate_timestamp(receipt["created_at_utc"], "Q2 gate receipt created_at_utc")
            > plan_created
            for receipt in receipts
        ):
            raise ValueError("Q2 qualification plan predates a required gate receipt.")
        union_proof = _qualification_union_proof(upstream.manifests, manifests)
        if union_proof != plan["qualification_union_proof"]:
            raise ValueError("Q2 qualification-union proof does not recompute exactly.")

    read_qualification_cohort_commit(
        phase,
        plan=plan,
        manifests=manifests,
        root=root,
    )

    return ValidatedQualificationPlan(
        path=resolved_path,
        plan=plan,
        manifests=manifests,
        topology_proof=topology,
        upstream_q1=upstream,
        gate_receipts=tuple(receipts),
    )


def _preflight_publication_paths(
    *,
    phase: str,
    plan_path: Path,
    manifest_paths: Mapping[str, Path],
    root: Path,
) -> Path:
    expected_plan = canonical_qualification_plan_path(phase, root)
    expected_manifests = canonical_qualification_manifest_paths(phase, root)
    if plan_path != expected_plan or manifest_paths != expected_manifests:
        raise ValueError(
            "Qualification publication paths must equal the exact atomic cohort layout."
        )
    cohort = canonical_qualification_cohort_root(phase, root)
    if _existing_component_is_symlink(cohort.parent, root=root):
        raise ValueError("Qualification cohort parent contains a symlink component.")
    if cohort.exists() or cohort.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite immutable qualification cohort: {cohort}."
        )
    return cohort


def _preflight_generation_outputs(manifests: Mapping[str, Mapping[str, Any]]) -> None:
    paths = [
        Path(str(job["output_dir"])).expanduser()
        for role in MANIFEST_ROLE_ORDER
        for job in manifests[role]["jobs"]
    ]
    if any(not path.is_absolute() or path.resolve(strict=False) != path for path in paths):
        raise ValueError("Qualification output attempt paths must be absolute and unaliased.")
    if len(paths) != len(set(paths)):
        raise ValueError("Qualification output attempt paths are not globally unique.")
    existing = sorted(str(path) for path in paths if path.exists() or path.is_symlink())
    if existing:
        raise FileExistsError(
            "Qualification attempt output directories must all be new; existing paths include "
            f"{existing[:8]}."
        )


def publish_qualification_phase(
    phase: str,
    *,
    plan_path: str | Path,
    manifest_paths: Mapping[str, str | Path],
    output_root: str | Path,
    attempt: int,
    allow_unvalidated_temporal_pilot: bool,
    flux1_native_equivalence_acceptance_receipt_path: str | Path,
    q1_plan_path: str | Path | None = None,
    exact_one_gate_receipt_path: str | Path | None = None,
    non_regression_gate_receipt_path: str | Path | None = None,
    root: Path | None = None,
) -> ValidatedQualificationPlan:
    """Build, validate, and immutably publish one Q1/Q2 specification.

    The plan JSON is the final commit marker.  Every destination must be a new,
    explicitly supplied path; this function never derives a debugging or
    manifest destination and never submits the resulting manifests.
    """

    root = (root or project_root()).resolve()
    if set(manifest_paths) != set(MANIFEST_ROLE_ORDER):
        raise ValueError(
            f"Caller must supply exactly these manifest paths: {list(MANIFEST_ROLE_ORDER)}."
        )
    if attempt != 1 or isinstance(attempt, bool):
        raise ValueError("Fresh qualification cohort attempt must be exactly 1.")
    resolved_plan_path = _lexical_absolute(
        plan_path, root, label="qualification plan publication path"
    )
    resolved_manifest_paths = {
        role: _lexical_absolute(
            manifest_paths[role],
            root,
            label=f"qualification manifest publication path for {role}",
        )
        for role in MANIFEST_ROLE_ORDER
    }
    resolved_output_root = _lexical_absolute(
        output_root, root, label="qualification generation output root"
    )
    if resolved_output_root != canonical_qualification_output_root(root):
        raise ValueError(
            "Qualification output root must equal the exact fresh qualification root."
        )
    cohort_root = _preflight_publication_paths(
        phase=phase,
        plan_path=resolved_plan_path,
        manifest_paths=resolved_manifest_paths,
        root=root,
    )

    upstream_q1: ValidatedQualificationPlan | None = None
    gate_paths: list[Path] = []
    gate_receipts: list[dict[str, Any]] = []
    supplied_gates = (
        q1_plan_path,
        exact_one_gate_receipt_path,
        non_regression_gate_receipt_path,
    )
    if phase == Q1:
        if any(value is not None for value in supplied_gates):
            raise ValueError("Q1 does not accept Q2 gate inputs.")
    elif phase == Q2:
        if any(value is None for value in supplied_gates):
            raise ValueError(
                "Q2 requires --q1-plan, --exact-one-gate-receipt, and "
                "--non-regression-gate-receipt before any manifest is built."
            )
        assert q1_plan_path is not None
        assert exact_one_gate_receipt_path is not None
        assert non_regression_gate_receipt_path is not None
        upstream_q1 = validate_qualification_plan(
            q1_plan_path,
            root=root,
            expected_phase=Q1,
        )
        if upstream_q1.plan["attempt"] != attempt or upstream_q1.plan["output_root"] != str(
            resolved_output_root
        ):
            raise ValueError("Q2 must use the exact upstream Q1 attempt and output root.")
        gate_paths = [
            _resolve(exact_one_gate_receipt_path, root),
            _resolve(non_regression_gate_receipt_path, root),
        ]
        gate_receipts = [
            read_gate_receipt(
                gate_paths[0],
                q1=upstream_q1,
                expected_contract=EXACT_ONE_GATE_CONTRACT,
                root=root,
            ),
            read_gate_receipt(
                gate_paths[1],
                q1=upstream_q1,
                expected_contract=NON_REGRESSION_GATE_CONTRACT,
                root=root,
            ),
        ]
    else:
        raise ValueError(f"Unknown qualification phase {phase!r}.")

    flux1_execution_protocol_inputs = flux_v3.load_flux1_execution_protocol_inputs_v3(
        flux1_native_equivalence_acceptance_receipt_path,
        project_root=root,
    )
    manifests = build_qualification_manifests(
        phase,
        root=root,
        output_root=resolved_output_root,
        attempt=attempt,
        allow_unvalidated_temporal_pilot=allow_unvalidated_temporal_pilot,
        flux1_execution_protocol_inputs=flux1_execution_protocol_inputs,
    )
    topology = validate_qualification_manifests(
        phase,
        manifests,
        output_root=resolved_output_root,
        attempt=attempt,
        require_flux1_execution=True,
        root=root,
    )
    _preflight_generation_outputs(manifests)
    if upstream_q1 is not None:
        if (
            topology["implementation_files_sha256"]
            != upstream_q1.plan["implementation_files_sha256"]
        ):
            raise ValueError("Q2 builder source differs from the authenticated Q1 source.")
        if topology["git_provenance_sha256"] != upstream_q1.topology_proof["git_provenance_sha256"]:
            raise ValueError("Q2 builder Git state differs from the authenticated Q1 source.")
        _qualification_union_proof(upstream_q1.manifests, manifests)

    cohort_root.parent.mkdir(parents=True, exist_ok=True)
    _preflight_publication_paths(
        phase=phase,
        plan_path=resolved_plan_path,
        manifest_paths=resolved_manifest_paths,
        root=root,
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{cohort_root.name}.staging-", dir=cohort_root.parent)
    )
    stat = staging.lstat()
    staging_identity = (stat.st_dev, stat.st_ino)
    try:
        physical_manifests = {
            role: staging / QUALIFICATION_MANIFEST_FILENAMES[role]
            for role in MANIFEST_ROLE_ORDER
        }
        for role in MANIFEST_ROLE_ORDER:
            write_manifest_immutable(
                manifests[role],
                physical_manifests[role],
                root,
                logical_publication_path=resolved_manifest_paths[role],
                allow_atomic_qualification_cohort_staging=True,
            )
        reopened: dict[str, dict[str, Any]] = {}
        for role in MANIFEST_ROLE_ORDER:
            physical = physical_manifests[role]
            payload = json.loads(physical.read_text(encoding="utf-8"))
            if not isinstance(payload, dict) or manifest_digest(payload) != payload.get(
                "manifest_sha256"
            ):
                raise RuntimeError(f"Staged qualification manifest is invalid for {role}.")
            sidecar = physical.with_suffix(physical.suffix + ".sha256")
            if sidecar.read_text(encoding="utf-8").split() != [
                payload["manifest_sha256"],
                physical.name,
            ]:
                raise RuntimeError(
                    f"Staged qualification manifest sidecar is invalid for {role}."
                )
            logical_snapshot = Path(f"{resolved_manifest_paths[role]}.snapshot")
            descriptor = payload.get("snapshot_bundle")
            if (
                not isinstance(descriptor, Mapping)
                or descriptor.get("root_path") != str(logical_snapshot)
                or descriptor.get("index_path") != str(logical_snapshot / "index.json")
                or descriptor.get("index_sha256")
                != _sha256_file(Path(f"{physical}.snapshot") / "index.json")
            ):
                raise RuntimeError(
                    f"Staged qualification snapshot binding is invalid for {role}."
                )
            reopened[role] = payload
        topology = validate_qualification_manifests(
            phase,
            reopened,
            output_root=resolved_output_root,
            attempt=attempt,
            require_flux1_execution=True,
            root=root,
        )
        plan = _build_qualification_plan_payload(
            phase=phase,
            manifest_paths=resolved_manifest_paths,
            manifests=reopened,
            topology_proof=topology,
            attempt=attempt,
            output_root=resolved_output_root,
            upstream_q1=upstream_q1,
            gate_receipt_paths=gate_paths,
            gate_receipts=gate_receipts,
            manifest_content_paths=physical_manifests,
        )
        physical_plan = staging / QUALIFICATION_PLAN_FILENAME
        frozen_plan = write_qualification_plan_immutable(plan, physical_plan, root=root)
        if read_qualification_plan(physical_plan, root=root) != frozen_plan:
            raise RuntimeError("Staged qualification plan failed immediate reauthentication.")
        commit = _build_qualification_cohort_commit(
            phase=phase,
            plan=frozen_plan,
            manifests=reopened,
            physical_root=staging,
            root=root,
        )
        physical_commit = staging / QUALIFICATION_COHORT_COMMIT_FILENAME
        frozen_commit = _write_authenticated_document(
            commit,
            physical_commit,
            digest_field="document_sha256",
            digest_function=qualification_cohort_commit_digest,
        )
        if (
            _read_authenticated_document(
                physical_commit,
                digest_field="document_sha256",
                digest_function=qualification_cohort_commit_digest,
                label="staged qualification cohort commit",
            )
            != frozen_commit
        ):
            raise RuntimeError("Staged qualification commit failed reauthentication.")
        expected_entries = {
            QUALIFICATION_PLAN_FILENAME,
            f"{QUALIFICATION_PLAN_FILENAME}.sha256",
            QUALIFICATION_COHORT_COMMIT_FILENAME,
            f"{QUALIFICATION_COHORT_COMMIT_FILENAME}.sha256",
        }
        for filename in QUALIFICATION_MANIFEST_FILENAMES.values():
            expected_entries.update(
                {filename, f"{filename}.sha256", f"{filename}.snapshot"}
            )
        if {path.name for path in staging.iterdir()} != expected_entries:
            raise RuntimeError("Staged qualification cohort has an inexact artifact set.")
        for role, physical_manifest in physical_manifests.items():
            _validate_exact_snapshot_artifact_tree(
                Path(f"{physical_manifest}.snapshot"), label=f"staged {role}"
            )
        _preflight_generation_outputs(reopened)
        _preflight_publication_paths(
            phase=phase,
            plan_path=resolved_plan_path,
            manifest_paths=resolved_manifest_paths,
            root=root,
        )
        _freeze_tree(staging)
        publish_hardlink_tree_commit_last(
            staging,
            cohort_root,
            commit_relative_path=Path(QUALIFICATION_COHORT_COMMIT_FILENAME),
            fault_hook=_publication_fault_hook,
        )
        validated = validate_qualification_plan(
            resolved_plan_path, root=root, expected_phase=phase
        )
        if validated.plan != frozen_plan:
            raise RuntimeError(
                "Atomically published qualification cohort failed immediate reauthentication."
            )
        return validated
    finally:
        _cleanup_owned_staging(staging, staging_identity)


def completed_flux1_union_receipt_digest(receipt: Mapping[str, Any]) -> str:
    canonical = deepcopy(dict(receipt))
    canonical.pop("receipt_sha256", None)
    return canonical_sha256(canonical)


def derive_completed_flux1_qualification_union_receipt_v3(
    q2: ValidatedQualificationPlan, *, root: Path
) -> dict[str, Any]:
    """Reopen all 18 completed FLUX-v3 Q1/Q2 rows from audit snapshots."""

    root = root.resolve()
    if q2.phase != Q2 or q2.upstream_q1 is None:
        raise ValueError("Completed FLUX-v3 union audit requires an authenticated Q2 plan.")
    phases = ((Q1, q2.upstream_q1), (Q2, q2))
    rows: list[dict[str, Any]] = []
    phase_counts = {Q1: 0, Q2: 0}
    for phase, validated in phases:
        manifest_paths = canonical_qualification_manifest_paths(phase, root)
        for role in MANIFEST_ROLE_ORDER:
            manifest = validated.manifests[role]
            manifest_sha256 = str(manifest["manifest_sha256"])
            for index, raw_job in enumerate(manifest["jobs"]):
                if not is_flux1_job_v3(raw_job):
                    continue
                if raw_job.get("expected_media") is not True:
                    raise ValueError("Qualification FLUX-v3 union contains a non-media row.")
                exact_job = {
                    **raw_job,
                    "launch_manifest_sha256": manifest_sha256,
                    "launch_manifest_job_index": index,
                }
                result_path = Path(str(raw_job["output_dir"])) / "benchmark_job_result.json"
                independently_bound_result_sha256 = _sha256_file(result_path)
                reopened = reopen_completed_flux1_output_v3(
                    exact_job,
                    root=root,
                    manifest_path=manifest_paths[role],
                    manifest_sha256=manifest_sha256,
                    manifest_job_index=index,
                    result_path=result_path,
                )
                if reopened["result_sha256"] != independently_bound_result_sha256:
                    raise ValueError(
                        "Qualification FLUX-v3 union result changed across its audit boundary."
                    )
                rows.append(
                    {
                        "phase": phase,
                        "role": role,
                        "manifest_path": str(manifest_paths[role]),
                        "manifest_sha256": manifest_sha256,
                        "manifest_job_index": index,
                        "condition_id": raw_job["condition_id"],
                        "job_sha256": reopened["job_sha256"],
                        "result_sha256": reopened["result_sha256"],
                        "runtime_validation_sha256": canonical_sha256(
                            reopened["runtime_validation"]
                        ),
                        "evidence_bindings": deepcopy(reopened["evidence_bindings"]),
                    }
                )
                phase_counts[phase] += 1
    if phase_counts != {Q1: 15, Q2: 3} or len(rows) != 18:
        raise ValueError(
            "Qualification completed FLUX-v3 union must contain Q1=15, Q2=3, total=18 rows."
        )
    identities = {
        (row["phase"], row["manifest_sha256"], row["manifest_job_index"])
        for row in rows
    }
    if len(identities) != 18 or len({row["condition_id"] for row in rows}) != 18:
        raise ValueError("Qualification completed FLUX-v3 union reuses a row identity.")
    receipt = {
        "schema_version": 3,
        "contract": FLUX1_COMPLETED_UNION_CONTRACT,
        "benchmark": BENCHMARK_NAME,
        "created_at_utc": _utc_now(),
        "q2_plan": {
            "path": str(q2.path),
            "file_sha256": _sha256_file(q2.path),
            "qualification_plan_sha256": q2.digest,
        },
        "expected_rows": 18,
        "phase_counts": phase_counts,
        "rows": rows,
    }
    receipt["receipt_sha256"] = completed_flux1_union_receipt_digest(receipt)
    return receipt


def write_completed_flux1_qualification_union_receipt_v3(
    *,
    q2_plan_path: str | Path,
    receipt_path: str | Path,
    root: Path | None = None,
) -> dict[str, Any]:
    """Derive and immutably publish the post-Q2 18-row FLUX-v3 receipt."""

    resolved_root = (root or project_root()).resolve()
    q2 = validate_qualification_plan_for_audit(
        q2_plan_path, root=resolved_root, expected_phase=Q2
    )
    destination = _resolve(receipt_path, resolved_root)
    generation_root = Path(str(q2.plan["output_root"])).resolve()
    if destination == generation_root or generation_root in destination.parents:
        raise ValueError("Completed FLUX-v3 union receipt must be outside generation outputs.")
    receipt = derive_completed_flux1_qualification_union_receipt_v3(
        q2, root=resolved_root
    )
    return _write_authenticated_document(
        receipt,
        destination,
        digest_field="receipt_sha256",
        digest_function=completed_flux1_union_receipt_digest,
    )


def exact_one_gate_claims() -> dict[str, Any]:
    """Return a fresh copy of the non-negotiable all-66 Q1 claim set."""

    return deepcopy(_EXACT_ONE_CLAIMS)


def non_regression_gate_claims(*, passing_test_cases: int) -> dict[str, Any]:
    """Return the required full-pair guard/test claim set."""

    if isinstance(passing_test_cases, bool) or not isinstance(passing_test_cases, int):
        raise ValueError("passing_test_cases must be a positive integer.")
    if passing_test_cases <= 0:
        raise ValueError("passing_test_cases must be a positive integer.")
    return {
        **_NON_REGRESSION_REQUIRED_BOOLEANS,
        "full_sequential_pair_test_cases_passed": passing_test_cases,
        "failed_test_cases": 0,
    }
