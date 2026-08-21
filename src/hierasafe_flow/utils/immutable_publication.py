"""Ceph-safe immutable-directory publication with a commit-last admission edge."""

from __future__ import annotations

import hashlib
import os
import secrets
import stat
from pathlib import Path
from typing import Callable


FaultHook = Callable[[str], None]


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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


def require_nonwritable_directories(root: Path, *, label: str) -> None:
    """Require the publication root and every descendant directory to be sealed."""

    for directory in (root, *(path for path in root.rglob("*") if path.is_dir())):
        observed = directory.lstat()
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ValueError(f"{label} contains an invalid directory: {directory}.")
        if stat.S_IMODE(observed.st_mode) & 0o222:
            raise ValueError(f"{label} contains a writable directory: {directory}.")


def _remove_fd_tree(directory_fd: int) -> None:
    # Operate through the already-authenticated directory descriptor.  This
    # prevents a concurrent pathname replacement from redirecting cleanup at a
    # competitor's tree.
    os.fchmod(directory_fd, 0o700)
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


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


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
    _fsync_directory(token_path.parent)


def freeze_tree(directory: Path, *, label: str = "Immutable publication") -> None:
    """Flush and remove write bits from every regular member and directory."""

    for path in directory.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"{label} cannot contain a symlink: {path}.")
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


def cleanup_owned_staging(staging: Path, identity: tuple[int, int]) -> None:
    """Remove only the private staging inode created by this publisher.

    All recursive deletion is descriptor-relative.  A pathname race can at
    worst leave the displaced owned inode behind; it cannot redirect deletion
    into a replacement tree at ``staging``.
    """

    try:
        observed = staging.lstat()
    except FileNotFoundError:
        return
    if staging.is_symlink() or (observed.st_dev, observed.st_ino) != identity:
        return
    descriptor = os.open(
        staging,
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0),
    )
    try:
        opened = os.fstat(descriptor)
        if (opened.st_dev, opened.st_ino) != identity:
            return
        _remove_fd_tree(descriptor)
        try:
            current = staging.lstat()
        except FileNotFoundError:
            return
        if not staging.is_symlink() and (current.st_dev, current.st_ino) == identity:
            try:
                staging.rmdir()
            except OSError:
                # A concurrent writer may have inserted a new member into the
                # owned inode.  Preserve it instead of broadening cleanup.
                return
            _fsync_directory(staging.parent)
    finally:
        os.close(descriptor)


def publish_hardlink_tree_commit_last(
    staging: Path,
    destination: Path,
    *,
    commit_relative_path: Path,
    fault_hook: FaultHook | None = None,
) -> None:
    """Publish an immutable sibling tree using hard links and a final commit file.

    ``mkdir`` exclusively claims the canonical name. All immutable files are
    hard-linked without replacement, the commit sidecar is linked before the
    self-authenticating commit JSON, and the commit JSON is linked last. A
    reader must also call :func:`require_nonwritable_directories`, which makes
    a crash after commit but before the final directory seal inadmissible.
    """

    hook = fault_hook or (lambda _step: None)
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
    require_nonwritable_directories(staging, label="Commit-last staging tree")
    for relative in files:
        if stat.S_IMODE((staging / relative).lstat().st_mode) & 0o222:
            raise ValueError(f"Commit-last staging contains a writable file: {relative}.")
    hook("preclaim")

    token = secrets.token_hex(32)
    token_path = destination.parent / f".{destination.name}.claim-{token}"
    token_fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    token_stat = os.fstat(token_fd)
    token_identity = (token_stat.st_dev, token_stat.st_ino)
    try:
        with os.fdopen(token_fd, "w", encoding="utf-8") as handle:
            handle.write(token)
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_directory(destination.parent)
        try:
            destination.mkdir(mode=0o700)
        except FileExistsError as exc:
            token_path.unlink(missing_ok=True)
            _fsync_directory(destination.parent)
            raise FileExistsError(
                f"Immutable canonical publication root already exists: {destination}."
            ) from exc
        _fsync_directory(destination.parent)
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
                _fsync_directory(destination.parent)
            raise
        observed = os.fstat(destination_fd)
        identity = (observed.st_dev, observed.st_ino)
        if identity != claimed_identity:
            os.close(destination_fd)
            token_path.unlink(missing_ok=True)
            raise RuntimeError("Canonical publication claim inode changed before authentication.")
        committed = False
        try:
            hook("postclaim")
            for relative in sorted(directories, key=lambda item: (len(item.parts), str(item))):
                (destination / relative).mkdir(mode=0o700)
            ordinary_files = sorted(
                files - {commit_relative_path, commit_sidecar}, key=str
            )
            for position, relative in enumerate(ordinary_files):
                os.link(staging / relative, destination / relative, follow_symlinks=False)
                if position == 0:
                    hook("midlink")
            os.link(staging / commit_sidecar, destination / commit_sidecar, follow_symlinks=False)
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
                _fsync_directory(destination / relative)
            _fsync_directory(destination)
            hook("precommit")
            os.link(
                staging / commit_relative_path,
                destination / commit_relative_path,
                follow_symlinks=False,
            )
            committed = True
            _fsync_directory(destination)
            hook("postcommit_prechmod")
            for relative in sorted(directories, key=lambda item: len(item.parts), reverse=True):
                (destination / relative).chmod(0o555)
                _fsync_directory(destination / relative)
            destination.chmod(0o555)
            _fsync_directory(destination)
            _fsync_directory(destination.parent)
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
        _fsync_directory(destination.parent)
    finally:
        # The claim marker is never the admission edge.  Remove it after both
        # successful publication and every fault point, but only when its
        # inode and unguessable content still prove that it is ours.
        try:
            observed_token = token_path.lstat()
            if (
                not token_path.is_symlink()
                and (observed_token.st_dev, observed_token.st_ino) == token_identity
                and token_path.read_text(encoding="utf-8") == token
            ):
                token_path.unlink()
                _fsync_directory(destination.parent)
        except (FileNotFoundError, OSError):
            pass
