"""Atomic publication and authentication of all 36 seed selections.

The target-blind selector owns the scientific decision for one prompt/model
axis.  This module owns the *campaign boundary*: no canonical selection is
admissible until every one of the 36 independently validated decisions is
staged, bound to the same committed eight-seed ladder, re-opened, frozen, and
released by a commit-last transaction.

CephFS does not implement ``renameat2(RENAME_NOREPLACE)``.  Publication uses
only primitives that Ceph enforces without replacement: an atomic ``mkdir``
claim and ``link`` with ``EEXIST`` semantics.  Immutable members are linked
from the hidden stage, reauthenticated in the claimed tree, and the commit is
linked last.  Readers require that commit and a fully read-only exact tree; a
crash can leave an explicitly uncommitted claim, never an admissible cohort.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from hierasafe_flow.benchmarks.finer_detailing_correction import (
    BENCHMARK_NAME,
    MODEL_NAMES,
    PROMPT_IDS,
    read_manifest,
    read_manifest_for_audit,
)
from hierasafe_flow.evaluation import target_blind_seed_selection as target_blind


SCHEMA_VERSION = 1
COHORT_CONTRACT = "finer_detailing_atomic_target_blind_selection_cohort_v1"
COMMIT_FILENAME = "selection_cohort_commit.json"
EXPECTED_RECORDS = len(PROMPT_IDS) * len(MODEL_NAMES)
EXPECTED_CANDIDATE_BINDINGS = EXPECTED_RECORDS * len(target_blind.SEEDS)

VIDEO_MODELS = frozenset(
    {
        "cogvideox_5b",
        "hunyuan_video",
        "joyai_echo",
        "ltx_23",
        "wan22_t2v_a14b",
    }
)
AXES = tuple((prompt_id, model_name) for prompt_id in PROMPT_IDS for model_name in MODEL_NAMES)
TASK_BY_MODEL = {
    model_name: ("text_to_video" if model_name in VIDEO_MODELS else "text_to_image")
    for model_name in MODEL_NAMES
}

SelectionSourceValidator = Callable[[Mapping[str, Any], Path], Mapping[str, Any]]
SelectionReader = Callable[[Path, Path], Mapping[str, Any]]
SelectionCohortValidator = Callable[
    [Mapping[tuple[str, str], Mapping[str, Any]], Path], Mapping[str, Any]
]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(payload: Mapping[str, Any], *, excluded: str) -> str:
    value = deepcopy(dict(payload))
    value.pop(excluded, None)
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def canonical_selection_paths(root: str | Path) -> tuple[Path, ...]:
    resolved_root = Path(root).expanduser().resolve()
    return tuple(
        target_blind.selection_output_path(resolved_root, prompt_id, model_name)
        for prompt_id, model_name in AXES
    )


def selection_cohort_root(root: str | Path) -> Path:
    return Path(root).expanduser().resolve() / target_blind.SELECTION_ROOT_RELATIVE


def selection_cohort_commit_path(root: str | Path) -> Path:
    return selection_cohort_root(root) / COMMIT_FILENAME


def _task_for(model_name: str) -> str:
    try:
        return TASK_BY_MODEL[model_name]
    except KeyError as exc:  # pragma: no cover - guarded by the exact axis matrix
        raise ValueError(f"Unknown finer-detailing model: {model_name!r}.") from exc


def _resolve_under_root(path: str | Path, root: Path, *, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    resolved = candidate.resolve(strict=False)
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} escapes project root {root}: {resolved}.")
    return resolved


def _contains_symlink(path: Path, *, root: Path) -> bool:
    if path != root and root not in path.parents:
        return True
    current = path
    while True:
        if current.is_symlink():
            return True
        if current == root:
            return False
        current = current.parent


def _ensure_safe_parent(parent: Path, *, root: Path) -> None:
    relative = parent.relative_to(root)
    current = root
    for component in relative.parts:
        current = current / component
        if current.is_symlink():
            raise ValueError(f"Selection-cohort parent contains a symlink: {current}.")
        if current.exists():
            if not current.is_dir():
                raise NotADirectoryError(f"Selection-cohort parent is not a directory: {current}.")
            continue
        current.mkdir(mode=0o755)
        if current.is_symlink() or not current.is_dir():
            raise RuntimeError(f"Failed to create a physical selection-cohort parent: {current}.")


def _preflight_absent(path: Path, *, root: Path, label: str) -> None:
    if _contains_symlink(path, root=root):
        raise ValueError(f"{label} path contains a symlink: {path}.")
    if path.exists() or path.is_symlink():
        raise FileExistsError(f"Refusing to replace existing {label}: {path}.")


def _write_text_new(path: Path, text: str, *, mode: int = 0o444) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        descriptor = -1
        os.chmod(path, mode, follow_symlinks=False)
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _freeze_tree(path: Path) -> None:
    descendants = list(path.rglob("*"))
    if any(item.is_symlink() for item in descendants):
        raise ValueError("Selection cohort staging contains a symlink.")
    for item in descendants:
        if item.is_file():
            descriptor = os.open(item, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
            try:
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    for directory in sorted(
        (item for item in descendants if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o555)
        _fsync_directory(directory)
    path.chmod(0o555)
    _fsync_directory(path)


def _cleanup_owned_stage(path: Path, identity: tuple[int, int]) -> None:
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return
    if path.is_symlink() or (stat.st_dev, stat.st_ino) != identity:
        return
    for item in sorted(path.rglob("*"), key=lambda value: len(value.parts), reverse=True):
        if item.is_symlink():
            item.unlink(missing_ok=True)
        elif item.is_dir():
            item.chmod(0o700)
    path.chmod(0o700)
    shutil.rmtree(path)


def _tree_paths(root: Path) -> tuple[set[Path], set[Path]]:
    descendants = list(root.rglob("*"))
    if any(path.is_symlink() for path in descendants):
        raise ValueError(f"Publication tree contains a symlink: {root}.")
    files = {path.relative_to(root) for path in descendants if path.is_file()}
    directories = {path.relative_to(root) for path in descendants if path.is_dir()}
    return files, directories


def _cleanup_owned_claim(path: Path, identity: tuple[int, int], *, staging: Path) -> None:
    try:
        stat = path.lstat()
    except FileNotFoundError:
        return
    if path.is_symlink() or (stat.st_dev, stat.st_ino) != identity:
        return
    try:
        stage_files, stage_directories = _tree_paths(staging)
        claimed_files, claimed_directories = _tree_paths(path)
    except (FileNotFoundError, ValueError):
        return
    if not claimed_files.issubset(stage_files) or not claimed_directories.issubset(
        stage_directories
    ):
        return
    for relative in claimed_files:
        source = staging / relative
        target = path / relative
        if (
            not source.is_file()
            or not target.is_file()
            or source.stat().st_dev != target.stat().st_dev
            or source.stat().st_ino != target.stat().st_ino
        ):
            return
    for directory in sorted(
        (item for item in path.rglob("*") if item.is_dir() and not item.is_symlink()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        directory.chmod(0o700)
    path.chmod(0o700)
    shutil.rmtree(path)


def _claim_and_materialize_commit_last(
    *,
    staging: Path,
    destination: Path,
    commit_filename: str,
    root: Path,
    before_commit: Callable[[Path], None],
) -> None:
    """Claim a Ceph directory and admit it only after the last metadata step.

    The commit file is the final directory entry.  While the claimed root is
    writable, readers reject it; changing that root to 0555 is the admission
    point after all bytes and the commit have already been fsynced.
    """

    _preflight_absent(destination, root=root, label="selection cohort")
    try:
        os.mkdir(destination, 0o700)
    except FileExistsError as exc:
        raise FileExistsError(
            f"Selection cohort appeared before its atomic mkdir claim: {destination}."
        ) from exc
    claim_stat = destination.lstat()
    claim_identity = (claim_stat.st_dev, claim_stat.st_ino)
    _fsync_directory(destination.parent)
    admitted = False
    try:
        stage_files, stage_directories = _tree_paths(staging)
        commit_relative = Path(commit_filename)
        if commit_relative not in stage_files:
            raise RuntimeError("Hidden selection stage lacks its release commit.")
        for relative in sorted(stage_directories, key=lambda value: len(value.parts)):
            os.mkdir(destination / relative, 0o700)
        for relative in sorted(stage_files - {commit_relative}):
            source = staging / relative
            target = destination / relative
            try:
                os.link(source, target, follow_symlinks=False)
            except FileExistsError as exc:
                raise FileExistsError(
                    f"Claimed selection member already exists: {target}."
                ) from exc
        copied_files, copied_directories = _tree_paths(destination)
        if copied_directories != stage_directories or copied_files != stage_files - {
            commit_relative
        }:
            raise RuntimeError("Claimed selection tree differs before release commit.")
        for relative in copied_files:
            source = staging / relative
            target = destination / relative
            if (
                source.stat().st_ino != target.stat().st_ino
                or source.stat().st_dev != target.stat().st_dev
                or _sha256_file(source) != _sha256_file(target)
            ):
                raise RuntimeError(f"Claimed selection member is not staged inode: {relative}.")
        before_commit(destination)
        current_stat = destination.lstat()
        if (current_stat.st_dev, current_stat.st_ino) != claim_identity:
            raise RuntimeError("Selection-cohort claim identity changed before commit.")
        verified_files, verified_directories = _tree_paths(destination)
        if verified_directories != stage_directories or verified_files != stage_files - {
            commit_relative
        }:
            raise RuntimeError("Claimed selection tree changed immediately before commit.")
        for relative in verified_files:
            source = staging / relative
            target = destination / relative
            if (
                source.stat().st_ino != target.stat().st_ino
                or source.stat().st_dev != target.stat().st_dev
                or _sha256_file(source) != _sha256_file(target)
            ):
                raise RuntimeError(f"Claimed selection member changed before commit: {relative}.")
        try:
            os.link(
                staging / commit_relative,
                destination / commit_relative,
                follow_symlinks=False,
            )
        except FileExistsError as exc:
            raise FileExistsError(
                "Selection release commit already exists in claimed tree."
            ) from exc
        committed_files, committed_directories = _tree_paths(destination)
        if committed_files != stage_files or committed_directories != stage_directories:
            raise RuntimeError("Claimed selection tree changed after commit link.")
        for directory in sorted(
            (destination / relative for relative in stage_directories),
            key=lambda value: len(value.parts),
            reverse=True,
        ):
            directory.chmod(0o555)
            _fsync_directory(directory)
        _fsync_directory(destination)
        destination.chmod(0o555)
        _fsync_directory(destination)
        admitted = True
    finally:
        if not admitted:
            _cleanup_owned_claim(destination, claim_identity, staging=staging)


def _validate_payload_axis(payload: Mapping[str, Any], *, prompt_id: str, model_name: str) -> None:
    selected_seed = payload.get("selected_seed")
    if (
        payload.get("selection") != target_blind.SELECTION_NAME
        or payload.get("benchmark") != BENCHMARK_NAME
        or payload.get("prompt_id") != prompt_id
        or payload.get("model_name") != model_name
        or payload.get("task") != _task_for(model_name)
        or isinstance(selected_seed, bool)
        or not isinstance(selected_seed, int)
        or selected_seed not in target_blind.SEEDS
        or payload.get("document_sha256") != target_blind.document_sha256(payload)
    ):
        raise ValueError(
            f"Selection payload does not identify canonical axis {prompt_id}/{model_name}."
        )
    selected_at = payload.get("selected_at_utc")
    if not isinstance(selected_at, str):
        raise ValueError("Selection payload lacks a timezone-aware selected_at_utc.")
    parsed = datetime.fromisoformat(selected_at.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("Selection payload selected_at_utc must include a timezone.")


def _default_source_validator(payload: Mapping[str, Any], root: Path) -> Mapping[str, Any]:
    result = target_blind.revalidate_selection_sources(
        payload,
        root=root,
        manifest_reader=read_manifest,
    )
    target_blind.validate_final_manifests(
        payload,
        final_manifest_paths=(),
        root=root,
        manifest_reader=read_manifest,
    )
    return result


def _default_selection_reader(path: Path, root: Path) -> Mapping[str, Any]:
    return target_blind.read_selection_record(path, root=root)


def _default_cohort_validator(
    selections: Mapping[tuple[str, str], Mapping[str, Any]], root: Path
) -> Mapping[str, Any]:
    from hierasafe_flow.evaluation import finer_detailing_campaign as campaign

    ladder_paths = campaign.canonical_ladder_manifest_paths(root)
    ladder = campaign.validate_production_seed_ladder(
        ladder_paths,
        root=root,
        manifest_reader=read_manifest,
    )
    bindings = campaign._validate_selection_ladder_bindings(  # noqa: SLF001
        selections,
        root=root,
        manifest_reader=read_manifest,
    )
    return {"ladder": ladder, "selection_bindings": bindings}


def _audit_cohort_validator(
    selections: Mapping[tuple[str, str], Mapping[str, Any]], root: Path
) -> Mapping[str, Any]:
    """Authenticate a historical cohort without granting production authority."""

    from hierasafe_flow.evaluation import finer_detailing_campaign as campaign

    ladder_paths = campaign.canonical_ladder_manifest_paths(root)
    ladder = campaign.validate_production_seed_ladder(
        ladder_paths,
        root=root,
        manifest_reader=read_manifest_for_audit,
    )
    bindings = campaign._validate_selection_ladder_bindings(  # noqa: SLF001
        selections,
        root=root,
        manifest_reader=read_manifest_for_audit,
    )
    return {"ladder": ladder, "selection_bindings": bindings}


def _strict_payload_mapping(
    payloads: Mapping[str | Path, Mapping[str, Any]], *, root: Path
) -> dict[Path, Mapping[str, Any]]:
    if not isinstance(payloads, Mapping):
        raise TypeError("Selection cohort payloads must be a path-to-document mapping.")
    canonical = canonical_selection_paths(root)
    indexed: dict[Path, Mapping[str, Any]] = {}
    for raw_path, payload in payloads.items():
        supplied = Path(raw_path).expanduser()
        if not supplied.is_absolute():
            supplied = root / supplied
        path = _resolve_under_root(raw_path, root, label="selection cohort member")
        if Path(os.path.abspath(supplied)) != path or _contains_symlink(supplied.parent, root=root):
            raise ValueError(
                f"Selection cohort member must use its canonical physical path, not an alias: "
                f"{raw_path!r}."
            )
        if path in indexed:
            raise ValueError(f"Selection cohort aliases one member path twice: {path}.")
        indexed[path] = payload
    missing = set(canonical) - set(indexed)
    extra = set(indexed) - set(canonical)
    if missing or extra or len(indexed) != EXPECTED_RECORDS:
        raise ValueError(
            "Selection cohort differs from the exact 36 canonical paths: "
            f"missing={sorted(str(path) for path in missing)}, "
            f"extra={sorted(str(path) for path in extra)}."
        )
    return {path: indexed[path] for path in canonical}


def _validate_ladder_commit(root: Path) -> dict[str, Any]:
    # Lazy import avoids making the per-axis selector depend on campaign code.
    from hierasafe_flow.evaluation import finer_detailing_campaign as campaign

    cohort_root = root / campaign.LADDER_MANIFEST_ROOT_RELATIVE
    path = cohort_root / campaign.COHORT_COMMIT_FILENAME
    sidecar = path.with_suffix(path.suffix + ".sha256")
    if _contains_symlink(path, root=root) or path.is_symlink() or sidecar.is_symlink():
        raise ValueError("Seed-ladder cohort commit path contains a symlink.")
    if not path.is_file() or not sidecar.is_file():
        raise FileNotFoundError("The committed eight-seed ladder is required before selection.")
    if cohort_root.stat().st_mode & 0o222 or any(
        candidate.stat().st_mode & 0o222
        for candidate in cohort_root.rglob("*")
        if candidate.is_dir()
    ):
        raise ValueError("Seed-ladder cohort is an uncommitted writable claim.")
    if path.stat().st_mode & 0o222 or sidecar.stat().st_mode & 0o222:
        raise ValueError("Seed-ladder cohort commit or sidecar is writable.")
    file_sha = _sha256_file(path)
    if sidecar.read_text(encoding="utf-8").split() != [file_sha, path.name]:
        raise ValueError("Seed-ladder cohort commit sidecar is invalid.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_keys = {
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
    if not isinstance(payload, dict) or set(payload) != expected_keys:
        raise ValueError("Seed-ladder cohort commit fields are invalid.")
    members = payload.get("members")
    counts = payload.get("counts")
    if (
        payload.get("schema_version") != 1
        or payload.get("contract") != campaign.COHORT_COMMIT_CONTRACT
        or payload.get("campaign_contract") != campaign.CAMPAIGN_CONTRACT
        or payload.get("cohort_kind") != "seed_ladder"
        or payload.get("status") != "complete_before_any_launch"
        or str(payload.get("cohort_root", "")) != str(cohort_root)
        or counts
        != {
            "manifest_count": campaign.EXPECTED_LADDER_MANIFESTS,
            "logical_rows": campaign.EXPECTED_LADDER_MEDIA,
            "media_rows": campaign.EXPECTED_LADDER_MEDIA,
            "unsupported_rows": 0,
            "exact_one_media_rows": 0,
        }
        or not isinstance(members, list)
        or len(members) != campaign.EXPECTED_LADDER_MANIFESTS
        or payload.get("members_sha256")
        != hashlib.sha256(
            json.dumps(members, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        or payload.get("commit_sha256") != _canonical_sha256(payload, excluded="commit_sha256")
    ):
        raise ValueError("Seed-ladder cohort commit identity/count/digest is invalid.")
    expected_paths = campaign.canonical_ladder_manifest_paths(root)
    for index, (member, manifest_path) in enumerate(zip(members, expected_paths, strict=True)):
        expected_member_keys = {
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
        manifest_sidecar = manifest_path.with_suffix(manifest_path.suffix + ".sha256")
        snapshot_index = Path(f"{manifest_path}.snapshot") / "index.json"
        if (
            not isinstance(member, Mapping)
            or set(member) != expected_member_keys
            or member.get("index") != index
            or str(member.get("manifest_path", "")) != str(manifest_path)
            or str(member.get("manifest_sidecar_path", "")) != str(manifest_sidecar)
            or str(member.get("snapshot_index_path", "")) != str(snapshot_index)
            or any(
                candidate.is_symlink() or not candidate.is_file()
                for candidate in (manifest_path, manifest_sidecar, snapshot_index)
            )
            or any(
                candidate.stat().st_mode & 0o222
                for candidate in (manifest_path, manifest_sidecar, snapshot_index)
            )
            or member.get("manifest_file_sha256") != _sha256_file(manifest_path)
            or member.get("manifest_sidecar_file_sha256") != _sha256_file(manifest_sidecar)
            or member.get("snapshot_index_file_sha256") != _sha256_file(snapshot_index)
        ):
            raise ValueError(f"Seed-ladder cohort member {index} is invalid or changed.")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        snapshot = json.loads(snapshot_index.read_text(encoding="utf-8"))
        if manifest.get("manifest_sha256") != member.get("manifest_sha256") or snapshot.get(
            "index_sha256"
        ) != member.get("snapshot_index_sha256"):
            raise ValueError(f"Seed-ladder cohort member {index} semantic digest changed.")
    return {
        "path": str(path),
        "sha256": file_sha,
        "size_bytes": path.stat().st_size,
        "commit_sha256": payload["commit_sha256"],
    }


def _preflight_no_final_state(root: Path) -> None:
    from hierasafe_flow.evaluation import finer_detailing_campaign as campaign

    for label, relative in (
        ("selected-seed final output root", campaign.FINAL_OUTPUT_ROOT_RELATIVE),
        ("selected-seed final manifest root", campaign.FINAL_MANIFEST_ROOT_RELATIVE),
    ):
        _preflight_absent(root / relative, root=root, label=label)


def _member_record(
    path: Path,
    logical_path: Path,
    payload: Mapping[str, Any],
    *,
    axis_index: int,
) -> dict[str, Any]:
    sidecar = path.with_suffix(path.suffix + ".sha256")
    return {
        "axis_index": axis_index,
        "prompt_id": payload["prompt_id"],
        "model_name": payload["model_name"],
        "task": payload["task"],
        "selected_seed": payload["selected_seed"],
        "selection_path": str(logical_path),
        "document_sha256": payload["document_sha256"],
        "selection_file_sha256": _sha256_file(path),
        "selection_size_bytes": path.stat().st_size,
        "sidecar_path": str(logical_path.with_suffix(logical_path.suffix + ".sha256")),
        "sidecar_file_sha256": _sha256_file(sidecar),
    }


def _build_commit(
    *,
    root: Path,
    staging: Path,
    logical_paths: Sequence[Path],
    payloads: Mapping[Path, Mapping[str, Any]],
    ladder_binding: Mapping[str, Any],
) -> dict[str, Any]:
    members: list[dict[str, Any]] = []
    for axis_index, logical_path in enumerate(logical_paths):
        record = _member_record(
            staging / logical_path.name,
            logical_path,
            payloads[logical_path],
            axis_index=axis_index,
        )
        members.append(record)
    commit: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "contract": COHORT_CONTRACT,
        "status": "complete_before_final_manifest_or_generation",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort_root": str(selection_cohort_root(root)),
        "counts": {
            "selection_records": EXPECTED_RECORDS,
            "prompt_model_axes": EXPECTED_RECORDS,
            "candidate_bindings": EXPECTED_CANDIDATE_BINDINGS,
        },
        "ladder_cohort_commit": dict(ladder_binding),
        "members": members,
        "members_sha256": hashlib.sha256(
            json.dumps(members, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
    }
    commit["commit_sha256"] = _canonical_sha256(commit, excluded="commit_sha256")
    return commit


def _write_commit(staging: Path, commit: Mapping[str, Any]) -> None:
    path = staging / COMMIT_FILENAME
    _write_text_new(
        path,
        json.dumps(commit, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
    )
    _write_text_new(
        path.with_suffix(path.suffix + ".sha256"),
        f"{_sha256_file(path)}  {path.name}\n",
    )


def _expected_top_level(paths: Sequence[Path]) -> set[str]:
    expected = {COMMIT_FILENAME, f"{COMMIT_FILENAME}.sha256"}
    for path in paths:
        expected.update({path.name, f"{path.name}.sha256"})
    return expected


def _validate_commit_document(
    commit: Mapping[str, Any], *, root: Path, commit_path: Path, ladder_binding: Mapping[str, Any]
) -> None:
    expected_keys = {
        "schema_version",
        "contract",
        "status",
        "created_at_utc",
        "cohort_root",
        "counts",
        "ladder_cohort_commit",
        "members",
        "members_sha256",
        "commit_sha256",
    }
    if not isinstance(commit, Mapping):
        raise ValueError("Selection-cohort commit must be a JSON object.")
    members = commit.get("members")
    if (
        set(commit) != expected_keys
        or commit.get("schema_version") != SCHEMA_VERSION
        or commit.get("contract") != COHORT_CONTRACT
        or commit.get("status") != "complete_before_final_manifest_or_generation"
        or str(commit.get("cohort_root", "")) != str(selection_cohort_root(root))
        or commit.get("counts")
        != {
            "selection_records": EXPECTED_RECORDS,
            "prompt_model_axes": EXPECTED_RECORDS,
            "candidate_bindings": EXPECTED_CANDIDATE_BINDINGS,
        }
        or commit.get("ladder_cohort_commit") != dict(ladder_binding)
        or not isinstance(members, list)
        or len(members) != EXPECTED_RECORDS
        or commit.get("members_sha256")
        != hashlib.sha256(
            json.dumps(members, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        or commit.get("commit_sha256") != _canonical_sha256(commit, excluded="commit_sha256")
    ):
        raise ValueError("Selection-cohort commit schema, count, or digest is invalid.")
    created = datetime.fromisoformat(str(commit.get("created_at_utc", "")).replace("Z", "+00:00"))
    if created.tzinfo is None or created.utcoffset() is None:
        raise ValueError("Selection-cohort commit timestamp must include a timezone.")
    sidecar = commit_path.with_suffix(commit_path.suffix + ".sha256")
    if (
        commit_path.is_symlink()
        or sidecar.is_symlink()
        or not commit_path.is_file()
        or not sidecar.is_file()
        or bool(commit_path.stat().st_mode & 0o222)
        or bool(sidecar.stat().st_mode & 0o222)
        or sidecar.read_text(encoding="utf-8").split()
        != [_sha256_file(commit_path), commit_path.name]
    ):
        raise ValueError("Selection-cohort commit file/sidecar authentication failed.")


def _validate_members(
    *,
    cohort_root: Path,
    commit: Mapping[str, Any],
    root: Path,
    selection_reader: SelectionReader,
) -> dict[tuple[str, str], Mapping[str, Any]]:
    expected_paths = canonical_selection_paths(root)
    expected_member_keys = {
        "axis_index",
        "prompt_id",
        "model_name",
        "task",
        "selected_seed",
        "selection_path",
        "document_sha256",
        "selection_file_sha256",
        "selection_size_bytes",
        "sidecar_path",
        "sidecar_file_sha256",
    }
    loaded: dict[tuple[str, str], Mapping[str, Any]] = {}
    for axis_index, (axis, logical_path, member) in enumerate(
        zip(AXES, expected_paths, commit["members"], strict=True)
    ):
        prompt_id, model_name = axis
        physical = cohort_root / logical_path.name
        sidecar = physical.with_suffix(physical.suffix + ".sha256")
        if (
            not isinstance(member, Mapping)
            or set(member) != expected_member_keys
            or member.get("axis_index") != axis_index
            or member.get("prompt_id") != prompt_id
            or member.get("model_name") != model_name
            or member.get("task") != _task_for(model_name)
            or str(member.get("selection_path", "")) != str(logical_path)
            or str(member.get("sidecar_path", ""))
            != str(logical_path.with_suffix(logical_path.suffix + ".sha256"))
            or physical.is_symlink()
            or sidecar.is_symlink()
            or not physical.is_file()
            or not sidecar.is_file()
            or bool(physical.stat().st_mode & 0o222)
            or bool(sidecar.stat().st_mode & 0o222)
            or member.get("selection_file_sha256") != _sha256_file(physical)
            or member.get("selection_size_bytes") != physical.stat().st_size
            or member.get("sidecar_file_sha256") != _sha256_file(sidecar)
        ):
            raise ValueError(f"Selection-cohort member {axis_index} is invalid or changed.")
        payload = selection_reader(physical, root)
        _validate_payload_axis(payload, prompt_id=prompt_id, model_name=model_name)
        if (
            member.get("document_sha256") != payload.get("document_sha256")
            or member.get("selected_seed") != payload.get("selected_seed")
            or sidecar.read_text(encoding="utf-8").split()
            != [payload.get("document_sha256"), physical.name]
        ):
            raise ValueError(f"Selection-cohort member {axis_index} semantic binding changed.")
        loaded[axis] = payload
    return loaded


def publish_selection_cohort(
    payloads: Mapping[str | Path, Mapping[str, Any]],
    *,
    root: str | Path,
    source_validator: SelectionSourceValidator = _default_source_validator,
    selection_reader: SelectionReader = _default_selection_reader,
    cohort_validator: SelectionCohortValidator = _default_cohort_validator,
) -> tuple[Path, ...]:
    """Publish all 36 decisions through a Ceph-safe commit-last transaction."""

    resolved_root = Path(root).expanduser().resolve()
    indexed = _strict_payload_mapping(payloads, root=resolved_root)
    paths = tuple(indexed)
    cohort_root = selection_cohort_root(resolved_root)
    _preflight_absent(cohort_root, root=resolved_root, label="selection cohort")
    _preflight_no_final_state(resolved_root)
    ladder_binding = _validate_ladder_commit(resolved_root)

    for (prompt_id, model_name), path in zip(AXES, paths, strict=True):
        payload = indexed[path]
        _validate_payload_axis(payload, prompt_id=prompt_id, model_name=model_name)
        source_validator(payload, resolved_root)

    _ensure_safe_parent(cohort_root.parent, root=resolved_root)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{cohort_root.name}.selection-cohort-", dir=cohort_root.parent)
    )
    stage_stat = staging.lstat()
    stage_identity = (stage_stat.st_dev, stage_stat.st_ino)
    try:
        for logical_path in paths:
            payload = indexed[logical_path]
            physical = staging / logical_path.name
            _write_text_new(
                physical,
                json.dumps(
                    payload,
                    indent=2,
                    sort_keys=True,
                    ensure_ascii=False,
                    allow_nan=False,
                )
                + "\n",
            )
            _write_text_new(
                physical.with_suffix(physical.suffix + ".sha256"),
                f"{payload['document_sha256']}  {physical.name}\n",
            )
        commit = _build_commit(
            root=resolved_root,
            staging=staging,
            logical_paths=paths,
            payloads=indexed,
            ladder_binding=ladder_binding,
        )
        _write_commit(staging, commit)
        if {item.name for item in staging.iterdir()} != _expected_top_level(paths):
            raise RuntimeError("Selection cohort staging contains missing or extra entries.")
        reopened_commit = json.loads((staging / COMMIT_FILENAME).read_text(encoding="utf-8"))
        _validate_commit_document(
            reopened_commit,
            root=resolved_root,
            commit_path=staging / COMMIT_FILENAME,
            ladder_binding=ladder_binding,
        )
        loaded = _validate_members(
            cohort_root=staging,
            commit=reopened_commit,
            root=resolved_root,
            selection_reader=selection_reader,
        )
        cohort_validator(loaded, resolved_root)
        _preflight_no_final_state(resolved_root)
        _preflight_absent(cohort_root, root=resolved_root, label="selection cohort")
        _freeze_tree(staging)

        def before_commit(claimed_root: Path) -> None:
            expected_without_commit = _expected_top_level(paths) - {COMMIT_FILENAME}
            actual = {item.name for item in claimed_root.iterdir()}
            if actual != expected_without_commit:
                raise RuntimeError(
                    "Claimed selection tree differs before its release commit: "
                    f"missing={sorted(expected_without_commit - actual)}, "
                    f"extra={sorted(actual - expected_without_commit)}."
                )
            rebound = _validate_members(
                cohort_root=claimed_root,
                commit=reopened_commit,
                root=resolved_root,
                selection_reader=selection_reader,
            )
            cohort_validator(rebound, resolved_root)
            if _validate_ladder_commit(resolved_root) != ladder_binding:
                raise RuntimeError("Seed-ladder commit changed before selection release.")
            _preflight_no_final_state(resolved_root)

        _claim_and_materialize_commit_last(
            staging=staging,
            destination=cohort_root,
            commit_filename=COMMIT_FILENAME,
            root=resolved_root,
            before_commit=before_commit,
        )
    finally:
        _cleanup_owned_stage(staging, stage_identity)
    read_selection_cohort(
        root=resolved_root,
        selection_reader=selection_reader,
        cohort_validator=cohort_validator,
    )
    return paths


def read_selection_cohort(
    *,
    root: str | Path,
    selection_reader: SelectionReader = _default_selection_reader,
    cohort_validator: SelectionCohortValidator = _default_cohort_validator,
) -> dict[str, Any]:
    """Reopen the complete cohort and strictly revalidate live launch inputs."""

    resolved_root = Path(root).expanduser().resolve()
    cohort_root = selection_cohort_root(resolved_root)
    if (
        _contains_symlink(cohort_root, root=resolved_root)
        or cohort_root.is_symlink()
        or not cohort_root.is_dir()
    ):
        raise FileNotFoundError("A physical committed 36-selection cohort is required.")
    if cohort_root.stat().st_mode & 0o222 or any(
        path.stat().st_mode & 0o222 for path in cohort_root.rglob("*") if path.is_dir()
    ):
        raise ValueError("Selection cohort is an uncommitted writable claim.")
    paths = canonical_selection_paths(resolved_root)
    actual = {item.name for item in cohort_root.iterdir()}
    expected = _expected_top_level(paths)
    if actual != expected or any(item.is_symlink() for item in cohort_root.iterdir()):
        raise ValueError(
            "Published selection cohort top-level differs from the exact contract: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}."
        )
    ladder_binding = _validate_ladder_commit(resolved_root)
    commit_path = cohort_root / COMMIT_FILENAME
    commit = json.loads(commit_path.read_text(encoding="utf-8"))
    if not isinstance(commit, dict):
        raise ValueError("Selection-cohort commit must be a JSON object.")
    _validate_commit_document(
        commit,
        root=resolved_root,
        commit_path=commit_path,
        ladder_binding=ladder_binding,
    )
    selections = _validate_members(
        cohort_root=cohort_root,
        commit=commit,
        root=resolved_root,
        selection_reader=selection_reader,
    )
    topology = cohort_validator(selections, resolved_root)
    return {
        "status": "valid",
        "contract": COHORT_CONTRACT,
        "commit_path": str(commit_path),
        "commit_sha256": commit["commit_sha256"],
        "commit_file_sha256": _sha256_file(commit_path),
        "selection_paths": [str(path) for path in paths],
        "selection_count": len(selections),
        "candidate_bindings": EXPECTED_CANDIDATE_BINDINGS,
        "selections": selections,
        "topology": topology,
    }


def read_selection_cohort_for_audit(
    *,
    root: str | Path,
    selection_reader: SelectionReader = _default_selection_reader,
) -> dict[str, Any]:
    """Reopen historical selection evidence without making it launch-authoritative.

    The ordinary :func:`read_selection_cohort` is deliberately strict because its
    result feeds selected-seed production construction.  This named audit-only
    route preserves later evidence review after live protocol inputs have changed.
    """

    return read_selection_cohort(
        root=root,
        selection_reader=selection_reader,
        cohort_validator=_audit_cohort_validator,
    )


__all__ = [
    "AXES",
    "COHORT_CONTRACT",
    "COMMIT_FILENAME",
    "EXPECTED_CANDIDATE_BINDINGS",
    "EXPECTED_RECORDS",
    "TASK_BY_MODEL",
    "canonical_selection_paths",
    "publish_selection_cohort",
    "read_selection_cohort",
    "read_selection_cohort_for_audit",
    "selection_cohort_commit_path",
    "selection_cohort_root",
]
