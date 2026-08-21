"""Crash-safe application of the five-model temporal promotion cohort.

CephFS cannot atomically replace five independent YAML files.  The safe
alternative is an explicitly journalled transaction whose admission edge is a
final, self-authenticating apply record.  A live update is allowed only after
an immutable promotion cohort has committed the exact pilot and candidate
hashes.  Production readers require both that cohort and this completed apply
transaction.

The transaction deliberately does not roll back.  After a crash it accepts
only a candidate-prefix/pilot-suffix state owned by the durable intent.  Any
other live bytes are treated as competitor drift and are preserved.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import stat
import tempfile
from contextlib import contextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

from hierasafe_flow.evaluation import temporal_promotion
from hierasafe_flow.evaluation.temporal_qualification import canonical_sha256
from hierasafe_flow.utils.immutable_publication import (
    cleanup_owned_staging,
    freeze_tree,
    require_nonwritable_directories,
)


APPLY_SCHEMA_VERSION = 1
APPLY_ROOT_RELATIVE = Path(
    "debugging/temporal_qualification/finer_detailing_temporal_production_apply_v1"
)
APPLY_LOCK_FILENAME = ".finer_detailing_temporal_production_apply_v1.lock"
APPLY_INTENT_CONTRACT = "finer_detailing_temporal_production_apply_intent_v1"
APPLY_STEP_CONTRACT = "finer_detailing_temporal_production_apply_step_v1"
APPLY_COMMIT_CONTRACT = "finer_detailing_temporal_production_apply_commit_v1"
APPLY_INTENT_FILENAME = "apply_intent.json"
APPLY_STEP_FILENAME = "apply_step.json"
APPLY_COMMIT_FILENAME = "apply_commit.json"
_HEX64 = frozenset("0123456789abcdef")


def canonical_temporal_promotion_apply_root(root: Path) -> Path:
    return (root.resolve() / APPLY_ROOT_RELATIVE).absolute()


def canonical_temporal_promotion_apply_commit_path(root: Path) -> Path:
    return (
        canonical_temporal_promotion_apply_root(root)
        / "completion"
        / APPLY_COMMIT_FILENAME
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _document(payload: Mapping[str, Any]) -> dict[str, Any]:
    document = deepcopy(dict(payload))
    document["document_sha256"] = canonical_sha256(document)
    return document


def _json_bytes(payload: Mapping[str, Any]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _require_real_directory(path: Path, *, label: str) -> None:
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{label} is not a real directory: {path}.")


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


def _apply_fault_hook(_step: str) -> None:
    """Test-only crash boundary; production deliberately performs no action."""


@contextmanager
def _promotion_apply_lock(
    root: Path, *, exclusive: bool, blocking: bool, create: bool
) -> Iterator[None]:
    """Hold the process-wide promotion lock and authenticate its pathname."""

    project_root = root.resolve()
    apply_root = canonical_temporal_promotion_apply_root(project_root)
    lock_parent = apply_root.parent
    _require_no_symlink_components(
        lock_parent, root=project_root, label="Temporal apply lock parent"
    )
    if create:
        lock_parent.mkdir(parents=True, exist_ok=True)
        _fsync_directory(lock_parent.parent)
    elif lock_parent.is_symlink() or not lock_parent.is_dir():
        raise FileNotFoundError("Temporal promotion apply lock parent is absent.")
    lock_path = lock_parent / APPLY_LOCK_FILENAME
    flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    if create:
        flags |= os.O_CREAT
    try:
        descriptor = os.open(lock_path, flags, 0o600)
    except FileNotFoundError as exc:
        raise FileNotFoundError("Temporal promotion apply transaction is absent.") from exc
    try:
        opened = os.fstat(descriptor)
        observed = lock_path.lstat()
        if (
            not stat.S_ISREG(opened.st_mode)
            or lock_path.is_symlink()
            or (opened.st_dev, opened.st_ino) != (observed.st_dev, observed.st_ino)
        ):
            raise ValueError("Temporal promotion apply lock pathname is aliased or invalid.")
        operation = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        if not blocking:
            operation |= fcntl.LOCK_NB
        try:
            fcntl.flock(descriptor, operation)
        except BlockingIOError as exc:
            raise RuntimeError("Temporal promotion apply transaction is already active.") from exc
        if create:
            os.fsync(descriptor)
            _fsync_directory(lock_parent)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _stable_regular_file_bytes(path: Path, *, label: str) -> tuple[bytes, tuple[int, int]]:
    if path.is_symlink():
        raise RuntimeError(f"{label} is a symlink: {path}.")
    try:
        descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} is absent: {path}.") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise RuntimeError(f"{label} is not a regular file: {path}.")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        observed = path.lstat()
        identity = (opened.st_dev, opened.st_ino)
        if path.is_symlink() or identity != (observed.st_dev, observed.st_ino):
            raise RuntimeError(f"{label} changed inode while being read: {path}.")
        return b"".join(chunks), identity
    finally:
        os.close(descriptor)


def _record_expected_bytes(
    filename: str, payload: Mapping[str, Any]
) -> tuple[bytes, bytes]:
    document_bytes = _json_bytes(payload)
    sidecar_bytes = f"{payload['document_sha256']}  {filename}\n".encode("utf-8")
    return document_bytes, sidecar_bytes


def _record_state(
    destination: Path, *, filename: str, payload: Mapping[str, Any]
) -> str:
    """Return absent/partial/complete after authenticating every existing byte."""

    if not destination.exists() and not destination.is_symlink():
        return "absent"
    _require_real_directory(destination, label="Temporal apply record")
    expected_names = {filename, f"{filename}.sha256"}
    children = {path.name: path for path in destination.iterdir()}
    if not set(children).issubset(expected_names):
        raise RuntimeError(f"Temporal apply record has unexpected members: {destination}.")
    document_bytes, sidecar_bytes = _record_expected_bytes(filename, payload)
    expected = {filename: document_bytes, f"{filename}.sha256": sidecar_bytes}
    for name, path in children.items():
        data, _identity = _stable_regular_file_bytes(
            path, label=f"Temporal apply record member {name}"
        )
        if data != expected[name]:
            raise RuntimeError(
                f"Temporal apply record contains unrelated or changed bytes: {path}."
            )
        if stat.S_IMODE(path.lstat().st_mode) & 0o222:
            raise RuntimeError(f"Temporal apply record member is writable: {path}.")
    if filename in children and f"{filename}.sha256" not in children:
        raise RuntimeError("Temporal apply commit appeared before its sidecar.")
    return "complete" if set(children) == expected_names else "partial"


def _write_new_file(path: Path, payload: bytes, *, mode: int) -> tuple[int, int]:
    descriptor = os.open(
        path,
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0),
        mode,
    )
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fchmod(descriptor, mode)
        os.fsync(descriptor)
        observed = os.fstat(descriptor)
        return observed.st_dev, observed.st_ino
    finally:
        os.close(descriptor)


def _publish_record(
    destination: Path,
    *,
    filename: str,
    payload: Mapping[str, Any],
    fault_label: str,
    staging_parent: Path,
) -> None:
    """Recoverably publish a two-file record, with its JSON linked last.

    Staging lives outside the canonical apply tree.  Consequently, a machine
    crash can leave only an irrelevant hidden sibling plus an exact subset of
    the two canonical members.  A later invocation can finish that subset
    without deleting or replacing any pathname.
    """

    state = _record_state(destination, filename=filename, payload=payload)
    if state == "complete":
        if stat.S_IMODE(destination.lstat().st_mode) & 0o222:
            destination.chmod(0o555)
            _fsync_directory(destination)
            _fsync_directory(destination.parent)
        return
    if state == "absent":
        try:
            destination.mkdir(mode=0o700)
            _fsync_directory(destination.parent)
        except FileExistsError:
            pass
    _require_real_directory(destination, label="Temporal apply record destination")
    if not stat.S_IMODE(destination.lstat().st_mode) & 0o222:
        raise RuntimeError("Incomplete temporal apply record directory is not writable.")

    stage = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.{fault_label}.stage-", dir=staging_parent)
    )
    observed_stage = stage.lstat()
    stage_identity = (observed_stage.st_dev, observed_stage.st_ino)
    document_bytes, sidecar_bytes = _record_expected_bytes(filename, payload)
    try:
        _write_new_file(stage / filename, document_bytes, mode=0o444)
        _write_new_file(stage / f"{filename}.sha256", sidecar_bytes, mode=0o444)
        freeze_tree(stage, label=f"Temporal apply {fault_label} stage")
        _apply_fault_hook(f"{fault_label}:prelink")
        sidecar = destination / f"{filename}.sha256"
        if not sidecar.exists() and not sidecar.is_symlink():
            os.link(stage / sidecar.name, sidecar, follow_symlinks=False)
            _fsync_directory(destination)
        _record_state(destination, filename=filename, payload=payload)
        _apply_fault_hook(f"{fault_label}:precommit")
        commit_path = destination / filename
        if not commit_path.exists() and not commit_path.is_symlink():
            os.link(stage / filename, commit_path, follow_symlinks=False)
            _fsync_directory(destination)
        if _record_state(destination, filename=filename, payload=payload) != "complete":
            raise RuntimeError("Temporal apply record publication did not complete.")
        _apply_fault_hook(f"{fault_label}:postcommit_preseal")
        destination.chmod(0o555)
        _fsync_directory(destination)
        _fsync_directory(destination.parent)
    finally:
        cleanup_owned_staging(stage, stage_identity)


def _cohort_rows(
    root: Path, cohort: Mapping[str, Any]
) -> list[dict[str, Any]]:
    promotion_root = temporal_promotion.canonical_promotion_root(root)
    rows = cohort.get("models")
    if not isinstance(rows, list) or [row.get("model_name") for row in rows] != list(
        temporal_promotion.VIDEO_MODEL_ORDER
    ):
        raise ValueError("Temporal apply requires the exact ordered five-model cohort.")
    result: list[dict[str, Any]] = []
    for index, source in enumerate(rows):
        model_name = temporal_promotion.VIDEO_MODEL_ORDER[index]
        binding = source.get("post_promotion_config")
        if not isinstance(binding, Mapping):
            raise ValueError(f"Temporal apply cohort lacks a config binding for {model_name}.")
        candidate_path = (
            promotion_root
            / "post_promotion_configs"
            / temporal_promotion.MODEL_CONFIG_RELATIVE[model_name].name
        )
        live_path = temporal_promotion.canonical_live_model_config_path(root, model_name)
        if (
            binding.get("bundled_path") != str(candidate_path)
            or binding.get("live_path") != str(live_path)
        ):
            raise ValueError(f"Temporal apply cohort paths drifted for {model_name}.")
        candidate_bytes, _identity = _stable_regular_file_bytes(
            candidate_path, label=f"Committed candidate config for {model_name}"
        )
        candidate_sha = _sha256_bytes(candidate_bytes)
        pilot_sha = str(binding.get("pre_promotion_file_sha256", ""))
        if (
            candidate_sha != binding.get("file_sha256")
            or candidate_sha != binding.get("bundled_file_sha256")
            or len(pilot_sha) != 64
            or any(character not in _HEX64 for character in pilot_sha)
            or pilot_sha == candidate_sha
        ):
            raise ValueError(f"Temporal apply candidate hash binding drifted for {model_name}.")
        result.append(
            {
                "step_index": index,
                "model_name": model_name,
                "live_path": live_path,
                "pilot_sha256": pilot_sha,
                "candidate_path": candidate_path,
                "candidate_sha256": candidate_sha,
                "candidate_bytes": candidate_bytes,
            }
        )
    return result


def _transaction_id(root: Path, cohort: Mapping[str, Any]) -> str:
    return canonical_sha256(
        {
            "contract": APPLY_INTENT_CONTRACT,
            "project_root": str(root),
            "promotion_document_sha256": cohort["document_sha256"],
        }
    )


def _intent_document(
    root: Path, cohort: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    transaction_id = _transaction_id(root, cohort)
    promotion_commit_path = (
        temporal_promotion.canonical_promotion_root(root)
        / temporal_promotion.PROMOTION_COMMIT_FILENAME
    )
    return _document(
        {
            "schema_version": APPLY_SCHEMA_VERSION,
            "contract": APPLY_INTENT_CONTRACT,
            "status": "durable_intent_before_live_mutation",
            "transaction_id": transaction_id,
            "promotion": {
                "root": str(temporal_promotion.canonical_promotion_root(root)),
                "commit_path": str(promotion_commit_path),
                "commit_file_sha256": _sha256_file(promotion_commit_path),
                "commit_document_sha256": cohort["document_sha256"],
            },
            "models": [
                {
                    "step_index": row["step_index"],
                    "model_name": row["model_name"],
                    "live_path": str(row["live_path"]),
                    "pilot_sha256": row["pilot_sha256"],
                    "candidate_path": str(row["candidate_path"]),
                    "candidate_sha256": row["candidate_sha256"],
                    "stage_path": str(
                        Path(row["live_path"]).parent
                        / (
                            f".{Path(row['live_path']).name}.temporal-apply-"
                            f"{transaction_id}"
                        )
                    ),
                }
                for row in rows
            ],
            "model_count": len(rows),
        }
    )


def _step_document(
    intent: Mapping[str, Any], row: Mapping[str, Any]
) -> dict[str, Any]:
    return _document(
        {
            "schema_version": APPLY_SCHEMA_VERSION,
            "contract": APPLY_STEP_CONTRACT,
            "status": "candidate_installed_and_read_back",
            "transaction_id": intent["transaction_id"],
            "intent_document_sha256": intent["document_sha256"],
            "promotion_document_sha256": intent["promotion"][
                "commit_document_sha256"
            ],
            "step_index": row["step_index"],
            "model_name": row["model_name"],
            "live_path": str(row["live_path"]),
            "pilot_sha256": row["pilot_sha256"],
            "candidate_path": str(row["candidate_path"]),
            "candidate_sha256": row["candidate_sha256"],
            "readback_sha256": row["candidate_sha256"],
        }
    )


def _journal_name(row: Mapping[str, Any]) -> str:
    return f"{int(row['step_index']):03d}_{row['model_name']}"


def _completion_document(
    root: Path,
    cohort: Mapping[str, Any],
    intent: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    journal_root = canonical_temporal_promotion_apply_root(root) / "journal"
    model_records = []
    for row in rows:
        step_path = journal_root / _journal_name(row) / APPLY_STEP_FILENAME
        step = _step_document(intent, row)
        model_records.append(
            {
                "step_index": row["step_index"],
                "model_name": row["model_name"],
                "live_path": str(row["live_path"]),
                "pilot_sha256": row["pilot_sha256"],
                "candidate_sha256": row["candidate_sha256"],
                "readback_sha256": row["candidate_sha256"],
                "journal_path": str(step_path),
                "journal_file_sha256": _sha256_file(step_path),
                "journal_document_sha256": step["document_sha256"],
            }
        )
    return _document(
        {
            "schema_version": APPLY_SCHEMA_VERSION,
            "contract": APPLY_COMMIT_CONTRACT,
            "status": "complete_five_model_live_apply",
            "transaction_id": intent["transaction_id"],
            "promotion_root": str(temporal_promotion.canonical_promotion_root(root)),
            "promotion_commit_document_sha256": cohort["document_sha256"],
            "intent_path": str(
                canonical_temporal_promotion_apply_root(root)
                / "intent"
                / APPLY_INTENT_FILENAME
            ),
            "intent_document_sha256": intent["document_sha256"],
            "models": model_records,
            "model_count": len(rows),
            "all_live_candidate_bytes_read_back": True,
        }
    )


def _ensure_apply_root(root: Path) -> Path:
    apply_root = canonical_temporal_promotion_apply_root(root)
    _require_no_symlink_components(
        apply_root.parent, root=root, label="Temporal apply transaction parent"
    )
    if not apply_root.exists() and not apply_root.is_symlink():
        try:
            apply_root.mkdir(mode=0o700)
            _fsync_directory(apply_root.parent)
        except FileExistsError:
            pass
    _require_real_directory(apply_root, label="Temporal apply transaction root")
    children = {path.name for path in apply_root.iterdir()}
    if not children.issubset({"intent", "journal", "completion"}):
        raise RuntimeError("Temporal apply transaction root has unrelated members.")
    journal_root = apply_root / "journal"
    if not journal_root.exists() and not journal_root.is_symlink():
        if children - {"intent"}:
            raise RuntimeError("Temporal apply journal is missing from a nonempty transaction.")
        journal_root.mkdir(mode=0o700)
        _fsync_directory(apply_root)
    _require_real_directory(journal_root, label="Temporal apply journal root")
    return apply_root


def _live_prefix(rows: Sequence[Mapping[str, Any]]) -> tuple[int, list[str]]:
    states: list[str] = []
    hashes: list[str] = []
    for row in rows:
        live_bytes, _identity = _stable_regular_file_bytes(
            Path(row["live_path"]), label=f"Live temporal config for {row['model_name']}"
        )
        observed = _sha256_bytes(live_bytes)
        hashes.append(observed)
        if observed == row["candidate_sha256"] and live_bytes == row["candidate_bytes"]:
            states.append("candidate")
        elif observed == row["pilot_sha256"]:
            states.append("pilot")
        else:
            raise RuntimeError(
                "Temporal apply found unrelated or changed live bytes and will not overwrite "
                f"them: {row['model_name']} sha256={observed}."
            )
    prefix = 0
    while prefix < len(states) and states[prefix] == "candidate":
        prefix += 1
    if states != ["candidate"] * prefix + ["pilot"] * (len(states) - prefix):
        raise RuntimeError(
            "Temporal apply live state is not the transaction-owned candidate prefix."
        )
    return prefix, hashes


def _journal_layout(
    journal_root: Path,
    *,
    intent: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
) -> tuple[int, int | None]:
    expected_names = [_journal_name(row) for row in rows]
    observed_names = {path.name for path in journal_root.iterdir()}
    if not observed_names.issubset(set(expected_names)):
        raise RuntimeError("Temporal apply journal contains an unrelated entry.")
    present_indices = [index for index, name in enumerate(expected_names) if name in observed_names]
    if present_indices and present_indices != list(range(max(present_indices) + 1)):
        raise RuntimeError("Temporal apply journal is not a contiguous prefix.")
    complete = 0
    partial: int | None = None
    for index in present_indices:
        row = rows[index]
        state = _record_state(
            journal_root / expected_names[index],
            filename=APPLY_STEP_FILENAME,
            payload=_step_document(intent, row),
        )
        if state == "complete" and partial is None:
            complete += 1
        elif state == "partial" and partial is None and index == complete:
            partial = index
        else:
            raise RuntimeError("Temporal apply journal has data after an incomplete record.")
    return complete, partial


def _assert_stage_layout(
    intent: Mapping[str, Any], rows: Sequence[Mapping[str, Any]], *, live_prefix: int
) -> None:
    for index, (intent_row, row) in enumerate(zip(intent["models"], rows, strict=True)):
        stage_path = Path(intent_row["stage_path"])
        if index != live_prefix and (stage_path.exists() or stage_path.is_symlink()):
            raise RuntimeError(
                f"Temporal apply found an unexpected staged config and preserved it: {stage_path}."
            )
        if index == live_prefix and (stage_path.exists() or stage_path.is_symlink()):
            stage_bytes, _identity = _stable_regular_file_bytes(
                stage_path, label=f"Staged temporal config for {row['model_name']}"
            )
            if (
                stage_bytes != row["candidate_bytes"]
                or _sha256_bytes(stage_bytes) != row["candidate_sha256"]
            ):
                raise RuntimeError(
                    "Temporal apply staged config contains unrelated bytes and was preserved: "
                    f"{stage_path}."
                )


def _install_stage(row: Mapping[str, Any], stage_path: Path) -> None:
    if stage_path.exists() or stage_path.is_symlink():
        stage_bytes, _identity = _stable_regular_file_bytes(
            stage_path, label=f"Staged temporal config for {row['model_name']}"
        )
        if stage_bytes != row["candidate_bytes"]:
            raise RuntimeError(
                f"Temporal apply will not replace changed staged bytes: {stage_path}."
            )
        return
    parent = stage_path.parent
    build_path = parent / (
        f".{stage_path.name}.build-{os.getpid()}-{next(tempfile._get_candidate_names())}"
    )
    identity: tuple[int, int] | None = None
    try:
        identity = _write_new_file(build_path, row["candidate_bytes"], mode=0o644)
        _fsync_directory(parent)
        try:
            os.link(build_path, stage_path, follow_symlinks=False)
        except FileExistsError:
            pass
        _fsync_directory(parent)
        staged, _staged_identity = _stable_regular_file_bytes(
            stage_path, label=f"Staged temporal config for {row['model_name']}"
        )
        if staged != row["candidate_bytes"]:
            raise RuntimeError(
                f"Temporal apply staged pathname was claimed by changed bytes: {stage_path}."
            )
    finally:
        if identity is not None:
            try:
                observed = build_path.lstat()
                if (
                    not build_path.is_symlink()
                    and (observed.st_dev, observed.st_ino) == identity
                ):
                    build_path.unlink()
                    _fsync_directory(parent)
            except FileNotFoundError:
                pass


def _replace_one_live_config(row: Mapping[str, Any], stage_path: Path) -> None:
    live_path = Path(row["live_path"])
    parent = live_path.parent
    _install_stage(row, stage_path)
    _apply_fault_hook(f"before_replace:{row['model_name']}")

    live_bytes, live_identity = _stable_regular_file_bytes(
        live_path, label=f"Live pilot config for {row['model_name']}"
    )
    stage_bytes, stage_identity = _stable_regular_file_bytes(
        stage_path, label=f"Staged candidate config for {row['model_name']}"
    )
    if _sha256_bytes(live_bytes) != row["pilot_sha256"]:
        raise RuntimeError(
            f"Temporal apply refuses to overwrite changed live bytes: {row['model_name']}."
        )
    if stage_bytes != row["candidate_bytes"]:
        raise RuntimeError(
            f"Temporal apply refuses to install changed staged bytes: {row['model_name']}."
        )
    if (live_path.lstat().st_dev, live_path.lstat().st_ino) != live_identity:
        raise RuntimeError(f"Live config changed at replacement boundary: {row['model_name']}.")
    if (stage_path.lstat().st_dev, stage_path.lstat().st_ino) != stage_identity:
        raise RuntimeError(f"Staged config changed at replacement boundary: {row['model_name']}.")
    os.replace(stage_path, live_path)
    _fsync_directory(parent)
    readback, _identity = _stable_regular_file_bytes(
        live_path, label=f"Installed temporal config for {row['model_name']}"
    )
    if readback != row["candidate_bytes"] or _sha256_bytes(readback) != row[
        "candidate_sha256"
    ]:
        raise RuntimeError(
            f"Temporal apply exact readback failed after replacement: {row['model_name']}."
        )
    _apply_fault_hook(f"after_replace:{row['model_name']}")


def _validate_apply_tree(
    root: Path,
    *,
    cohort: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    require_sealed: bool,
) -> dict[str, Any]:
    apply_root = canonical_temporal_promotion_apply_root(root)
    _require_real_directory(apply_root, label="Temporal apply transaction root")
    if {path.name for path in apply_root.iterdir()} != {"intent", "journal", "completion"}:
        raise RuntimeError("Temporal apply transaction artifact set is incomplete or inexact.")
    intent = _intent_document(root, cohort, rows)
    if _record_state(
        apply_root / "intent", filename=APPLY_INTENT_FILENAME, payload=intent
    ) != "complete":
        raise RuntimeError("Temporal apply intent is incomplete.")
    journal_root = apply_root / "journal"
    _require_real_directory(journal_root, label="Temporal apply journal root")
    complete, partial = _journal_layout(journal_root, intent=intent, rows=rows)
    if complete != len(rows) or partial is not None:
        raise RuntimeError("Temporal apply journal does not commit all five models.")
    completion = _completion_document(root, cohort, intent, rows)
    if _record_state(
        apply_root / "completion", filename=APPLY_COMMIT_FILENAME, payload=completion
    ) != "complete":
        raise RuntimeError("Temporal promotion final apply commit is absent or incomplete.")
    live_prefix, _hashes = _live_prefix(rows)
    if live_prefix != len(rows):
        raise RuntimeError("Temporal promotion final apply commit has incomplete live bytes.")
    _assert_stage_layout(intent, rows, live_prefix=live_prefix)
    if require_sealed:
        require_nonwritable_directories(
            apply_root, label="Temporal production apply transaction"
        )
    return completion


def apply_temporal_production_promotion(
    *, root: Path, wait_for_lock: bool = True
) -> dict[str, Any]:
    """Apply or resume the exact committed five-model promotion transaction."""

    project_root = root.resolve()
    with _promotion_apply_lock(
        project_root, exclusive=True, blocking=wait_for_lock, create=True
    ):
        cohort = temporal_promotion.read_temporal_promotion_cohort(
            root=project_root, require_live_configs=False
        )
        rows = _cohort_rows(project_root, cohort)
        apply_root = canonical_temporal_promotion_apply_root(project_root)

        if apply_root.exists() and not apply_root.is_symlink():
            try:
                completed = _validate_apply_tree(
                    project_root,
                    cohort=cohort,
                    rows=rows,
                    require_sealed=True,
                )
            except (FileNotFoundError, RuntimeError, ValueError):
                completed = None
            if completed is not None:
                return deepcopy(completed)

        apply_root = _ensure_apply_root(project_root)
        intent = _intent_document(project_root, cohort, rows)
        intent_path = apply_root / "intent"
        intent_state = _record_state(
            intent_path, filename=APPLY_INTENT_FILENAME, payload=intent
        )
        live_prefix, _hashes = _live_prefix(rows)
        journal_root = apply_root / "journal"
        if intent_state != "complete":
            if live_prefix != 0 or any(journal_root.iterdir()) or (
                (apply_root / "completion").exists()
                or (apply_root / "completion").is_symlink()
            ):
                raise RuntimeError(
                    "Temporal apply can create its durable intent only from the exact all-pilot "
                    "state."
                )
            _publish_record(
                intent_path,
                filename=APPLY_INTENT_FILENAME,
                payload=intent,
                fault_label="intent",
                staging_parent=apply_root.parent,
            )
            _apply_fault_hook("after_intent")

        complete_steps, partial_step = _journal_layout(
            journal_root, intent=intent, rows=rows
        )
        live_prefix, _hashes = _live_prefix(rows)
        if partial_step is not None:
            if partial_step != complete_steps or live_prefix != complete_steps + 1:
                raise RuntimeError(
                    "Temporal apply partial journal does not match the owned live prefix."
                )
            row = rows[partial_step]
            _publish_record(
                journal_root / _journal_name(row),
                filename=APPLY_STEP_FILENAME,
                payload=_step_document(intent, row),
                fault_label=f"step-{partial_step:03d}",
                staging_parent=apply_root.parent,
            )
            complete_steps += 1
        elif live_prefix == complete_steps + 1:
            row = rows[complete_steps]
            _publish_record(
                journal_root / _journal_name(row),
                filename=APPLY_STEP_FILENAME,
                payload=_step_document(intent, row),
                fault_label=f"step-{complete_steps:03d}",
                staging_parent=apply_root.parent,
            )
            complete_steps += 1
        elif live_prefix != complete_steps:
            raise RuntimeError(
                "Temporal apply live prefix is not owned by the durable journal state."
            )

        while complete_steps < len(rows):
            live_prefix, _hashes = _live_prefix(rows)
            if live_prefix != complete_steps:
                raise RuntimeError("Temporal apply state drifted before the next replacement.")
            _assert_stage_layout(intent, rows, live_prefix=live_prefix)
            row = rows[complete_steps]
            stage_path = Path(intent["models"][complete_steps]["stage_path"])
            _replace_one_live_config(row, stage_path)
            _publish_record(
                journal_root / _journal_name(row),
                filename=APPLY_STEP_FILENAME,
                payload=_step_document(intent, row),
                fault_label=f"step-{complete_steps:03d}",
                staging_parent=apply_root.parent,
            )
            complete_steps += 1
            _apply_fault_hook(f"after_journal:{row['model_name']}")

        final_prefix, _hashes = _live_prefix(rows)
        if final_prefix != len(rows):
            raise RuntimeError("Temporal apply cannot commit before all exact readbacks.")
        _assert_stage_layout(intent, rows, live_prefix=final_prefix)
        completion = _completion_document(project_root, cohort, intent, rows)
        _publish_record(
            apply_root / "completion",
            filename=APPLY_COMMIT_FILENAME,
            payload=completion,
            fault_label="completion",
            staging_parent=apply_root.parent,
        )
        _apply_fault_hook("post_apply_commit_preseal")
        if {path.name for path in apply_root.iterdir()} != {
            "intent",
            "journal",
            "completion",
        }:
            raise RuntimeError("Temporal apply tree changed before its final seal.")
        freeze_tree(apply_root, label="Temporal production apply transaction")
        _fsync_directory(apply_root)
        _fsync_directory(apply_root.parent)
        reopened = _validate_apply_tree(
            project_root,
            cohort=cohort,
            rows=rows,
            require_sealed=True,
        )
        if reopened != completion:
            raise RuntimeError("Temporal apply transaction failed exact reopen validation.")
        return deepcopy(reopened)


def require_temporal_promotion_apply_commit(
    *, root: Path, promotion_commit: Mapping[str, Any]
) -> dict[str, Any]:
    """Production-reader hook for an already-authenticated promotion cohort."""

    project_root = root.resolve()
    with _promotion_apply_lock(
        project_root, exclusive=False, blocking=False, create=False
    ):
        rows = _cohort_rows(project_root, promotion_commit)
        return deepcopy(
            _validate_apply_tree(
                project_root,
                cohort=promotion_commit,
                rows=rows,
                require_sealed=True,
            )
        )


def read_temporal_promotion_apply(*, root: Path) -> dict[str, Any]:
    """Reopen and authenticate the immutable completed apply transaction."""

    project_root = root.resolve()
    with _promotion_apply_lock(
        project_root, exclusive=False, blocking=False, create=False
    ):
        cohort = temporal_promotion.read_temporal_promotion_cohort(
            root=project_root, require_live_configs=False
        )
        rows = _cohort_rows(project_root, cohort)
        return deepcopy(
            _validate_apply_tree(
                project_root, cohort=cohort, rows=rows, require_sealed=True
            )
        )
