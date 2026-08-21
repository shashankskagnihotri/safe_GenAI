#!/usr/bin/env python3
"""Prospectively authorize the one FLUX-v3 source-003 quarantine repair.

This publisher is intentionally standalone.  It freezes its own source bytes,
the already immutable source-003 preregistration/result/tree, and the terminal
Slurm identity before any quarantine receipt or source-004 preregistration can
exist.  It does not import the admission validator that will be repaired in a
later stage, thereby avoiding an authorization/validator self-hash cycle.

The authorization is not an acceptance receipt.  It permanently classifies
source-003 as passed-but-unaccepted, forbids media reuse, and authorizes only a
fresh source-004 H100 equivalence execution after the three named source roles
have changed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from PIL import Image


SCHEMA_VERSION = 1
PROTOCOL_ID = "flux1_dual_view_adapter_native_equivalence_v3"
AUTHORIZATION_NAME = "flux1_source003_quarantine_authorization_v1"
AUTHORIZATION_STATUS = "authorized_before_quarantine_terminal_publication"
SOURCE_ATTEMPT = "003"
EXECUTION_ATTEMPT = "001"
SUCCESSOR_SOURCE_ATTEMPT = "004"
SLURM_JOB_ID = "291999"
SLURM_JOB_NAME = "flux1-dv-eq-v3"
SLURM_NODE_LIST = "dws-12"
SLURM_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
SACCT_FIELDS = ("JobIDRaw", "JobName", "State", "ExitCode", "Submit", "Start", "End")

AUDIT_ROOT_RELATIVE = Path("debugging/audits/finer_detailing_20260720")
PREREGISTRATION_FILENAME = (
    "flux1_dual_view_adapter_native_equivalence_v3_source_attempt_003_PREREGISTERED.json"
)
EXECUTION_DIRECTORY_NAME = (
    "flux1_dual_view_adapter_native_equivalence_v3_source_attempt_003_execution_attempt_001"
)
RESULT_FILENAME = "equivalence_result.json"
AUTHORIZATION_FILENAME = (
    "flux1_dual_view_adapter_native_equivalence_v3_source_attempt_003_QUARANTINE_AUTHORIZATION.json"
)
PUBLISHER_SOURCE_RELATIVE = Path("scripts/publish_flux1_source003_quarantine_authorization.py")

SOURCE003_PREREGISTRATION_SHA256 = (
    "6bc6597703d998e38a49c4229bae7c5501809eb57ce6d5e62f379557a6626314"
)
SOURCE003_PREREGISTRATION_SIDECAR_SHA256 = (
    "c10136fec71026091d59e0a6d1bf1d65d05bcb636f6b2d9338113275c0333dea"
)
SOURCE003_RESULT_SHA256 = "f4dda44dc0ac5c3dbe59a29d613819e1131d6464a61bc2eeef2ad24c65a59156"
SOURCE003_RESULT_SIDECAR_SHA256 = "994320c0ee730f54136dcb052fcfc12b680cefcf296a1d88698129538ead78d6"
SOURCE003_TREE_ENTRIES_SHA256 = "20ea106d3e8ab5bbf579c35b73f8963b6c5e74497c5569e0a231ad99a053e8ce"

# This complete ledger is itself contained in the exact preregistration bytes.
# Keeping the explicit mapping here prevents the repair publisher from silently
# accepting a different source-003 history that happens to use the same names.
SOURCE003_PREREGISTERED_FILE_SHA256 = {
    "adapter_base": "2fbd1962ffba6792c7ab6f771e47dd9fb0126dfe55621fad3d5acdd9cfd90f90",
    "adapter_registry": "af78e54628a547d9e2894aef0139fa68bf1c66325ff28727a2caadc7e27d552d",
    "base_config": "02531e45f5cbe844c4a13339f98f4ce7f33373a02e5cab88df95f8e84d16cde2",
    "benchmark_entrypoint": "65c8646754a48f20f9a64ae6cae05fa041fb9c9e45f69b99628d4b3cf84f910e",
    "benchmark_runner": "6ed9a48da298c617ae4c16701488ff585524f606218dacb5c41d3f19117809ef",
    "config_loader": "eb11393b0f7508460807bc10b0708030e67ba688622833f0f3ef9839c924d589",
    "diagnostic_script": "4c6670c59a5e30194fdb1201695d7bcd645449b4d6b7e7a8f01f6687688d6f03",
    "dual_view_adapter": "eb3a08ac40f09a2416d1920720bb113d7bde3c4f99fe385b2dbefb34a005ebf2",
    "environment": "0bb35ce6d3a8cf3ad5840f83e5d860df4275d4b05d7d0a8c79e5d3265538942c",
    "equivalence_admission": "c523578089fd6e4d949759556dc4aaf451768b5ada686e19128c7b44b12bf3d4",
    "flux_dual_view_job_projector": (
        "7d5831a41c6f7bd96a2b24471639d4b6b488915639ad21bb96a924583ab57c2b"
    ),
    "generation_runner": "1e6049f0c1201941bf5724d93db6bc9801c304b822d86fc23a742724147048e6",
    "legacy_flux_adapter": "114f76c09b76198f254bf5a7d5cf4a5b2cef5e12256835fd7b71fc035e19e439",
    "model_config": "624cc2f53e293102e6e31f016c307d30f5108a916afdbdc06d6e6365fb6bbd9d",
    "model_config_sidecar": ("2d0fd3ab0f048b8db1b9e175916cafa69bbe3f0b70f5c5fa9a98cf9f8252b9dd"),
    "prompt1_concept_hierarchy": (
        "7d05d6b1ca8afaaf9d48e9fc58b3b8e09cc066070ab949de2a3c5be8edcd57d0"
    ),
    "prompt2_concept_hierarchy": (
        "a66d5ef5a356e1d4d085558d18dbc34893d756246f4ba78e27675bccb5508ce1"
    ),
    "prompt3_concept_hierarchy": (
        "a87e9a692b83282278544e0e8522b6665a6b19155134d03e103a88aa94e61558"
    ),
    "prompt_contract": "1d061d331293980023599445f680f865b4d4af41f91144e9d6a3b3345ffe9b87",
    "prompt_contract_sidecar": ("1982188382fc546721ce45687cc7ecf92dd04423e148eafc4a2a002c6b4bd296"),
    "prompt_protocol": "c79c7ed65dac46a844be0022b2a81001536e10ad96ef00c7afecbddc6063e1ec",
    "prompt_protocol_sidecar": ("c9af58669b51afc1ced085b6b533befa6dfa3c7cdb5c53140a8c7f2b386ca458"),
    "pyproject": "257de9abc72c7923c1cadbfdce81d8157f574a31436bfdd9b621806807bd7ebd",
    "requirements": "d9b90a2af396893a7b92070031dcaf04236aa4f80bdc14fc1e3492a7d19829b2",
    "shared_benchmark_runner": ("4b0637163c4a23ed130631ea743a1b5079aa5ef33f676e67d226f2984d31026e"),
    "slurm_launcher": "a75f88316889622779bbefa7fe89a2dc738adff2ba21116505ce1e04a1956431",
}

SOURCE003_EXPECTED_TREE_FILES = {
    "01_sad_young_girl__flux_dual_view_adapter.png": {
        "sha256": "7275e4fcf5576c27b6be660bcf6c1dc9e1d31a0db9d02f9fa4403f6d5f672949",
        "size_bytes": 1_169_386,
    },
    "01_sad_young_girl__native_pipeline.png": {
        "sha256": "7275e4fcf5576c27b6be660bcf6c1dc9e1d31a0db9d02f9fa4403f6d5f672949",
        "size_bytes": 1_169_386,
    },
    "02_angry_old_man__flux_dual_view_adapter.png": {
        "sha256": "d4abba8ea8c87488ff1e9bf1e4704d10b99b3361f2b2a33264841878cec4277e",
        "size_bytes": 1_176_999,
    },
    "02_angry_old_man__native_pipeline.png": {
        "sha256": "d4abba8ea8c87488ff1e9bf1e4704d10b99b3361f2b2a33264841878cec4277e",
        "size_bytes": 1_176_999,
    },
    "03_empty_outdoor_mall__flux_dual_view_adapter.png": {
        "sha256": "040ea9961ba6d2910d7ef9940f70c71190ee5f0e5916aee9c0bd9d3bbd59d043",
        "size_bytes": 1_375_295,
    },
    "03_empty_outdoor_mall__native_pipeline.png": {
        "sha256": "040ea9961ba6d2910d7ef9940f70c71190ee5f0e5916aee9c0bd9d3bbd59d043",
        "size_bytes": 1_375_295,
    },
    RESULT_FILENAME: {"sha256": SOURCE003_RESULT_SHA256, "size_bytes": 381_161},
    f"{RESULT_FILENAME}.sha256": {
        "sha256": SOURCE003_RESULT_SIDECAR_SHA256,
        "size_bytes": 90,
    },
}

MEDIA_LAYOUTS = {
    "01_sad_young_girl": {"width": 832, "height": 1216},
    "02_angry_old_man": {"width": 832, "height": 1216},
    "03_empty_outdoor_mall": {"width": 1216, "height": 832},
}
MEDIA_ROUTES = ("native_pipeline", "flux_dual_view_adapter")
INTENDED_CHANGED_ROLES = (
    "diagnostic_script",
    "equivalence_admission",
    "flux_dual_view_job_projector",
)
REPAIR_REASON = (
    "Source-003 passed the H100 tensor/media comparisons, but its diagnostic hashed a "
    "preview projection containing a random TemporaryDirectory output_dir and the "
    "preregistered admission validator did not recognize the projector/validator route "
    "or its four projection receipt fields. The passed result is therefore quarantined "
    "without acceptance; source-004 must rerun under a versioned, recomputable contract."
)
LEGACY_RUNNER_ROUTE = (
    "flux1_dual_view_jobs_v3.project_flux1_job_v3(preview_v3)->"
    "flux1_dual_view_jobs_v3.validate_flux1_job_v3(preview_v3)->"
    "finer_detailing_correction._runner_config->benign_park._runner_config->"
    "load_config->GenerationRunner.__init__->_bind_flux_dual_view_conditioning->"
    "registry.create_adapter"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_AWARE_TIME_RE = re.compile(
    r"[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}"
    r"(?:\.[0-9]{1,6})?(?:Z|[+-](?:[01][0-9]|2[0-3]):?[0-5][0-9])"
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(raw)


def document_sha256(payload: Mapping[str, Any]) -> str:
    unsigned = deepcopy(dict(payload))
    unsigned.pop("document_sha256", None)
    return canonical_sha256(unsigned)


def _lexical_absolute(path: str | Path) -> Path:
    return Path(os.path.abspath(os.fspath(Path(path).expanduser())))


def _assert_below(path: Path, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} escapes its trusted root: {path}.") from exc


def _assert_no_symlink_components(path: Path, root: Path, label: str) -> None:
    _assert_below(path, root, label)
    root_metadata = os.lstat(root)
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError(f"{label} trusted root is not one real directory.")
    current = root
    for component in path.relative_to(root).parts:
        current = current / component
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode):
            raise ValueError(f"{label} contains a symlink component: {current}.")


def _read_regular_bytes(
    path: str | Path,
    *,
    trusted_root: Path,
    label: str,
    require_readonly: bool,
) -> bytes:
    target = _lexical_absolute(path)
    trusted_root = _lexical_absolute(trusted_root)
    _assert_no_symlink_components(target, trusted_root, label)
    before = os.lstat(target)
    if not stat.S_ISREG(before.st_mode) or before.st_nlink != 1:
        raise ValueError(f"{label} is not one singly linked regular file.")
    if require_readonly and before.st_mode & 0o222:
        raise ValueError(f"{label} must be read-only before authorization.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(target, flags)
    try:
        opened = os.fstat(descriptor)
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 16 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
    finally:
        os.close(descriptor)
    after = os.lstat(target)
    before_identity = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
        before.st_ctime_ns,
    )
    opened_identity = (
        opened.st_dev,
        opened.st_ino,
        opened.st_size,
        opened.st_mtime_ns,
        opened.st_ctime_ns,
    )
    after_identity = (
        after.st_dev,
        after.st_ino,
        after.st_size,
        after.st_mtime_ns,
        after.st_ctime_ns,
    )
    if before_identity != opened_identity or opened_identity != after_identity:
        raise ValueError(f"{label} changed during authenticated read.")
    raw = b"".join(chunks)
    if len(raw) != opened.st_size:
        raise ValueError(f"{label} size changed during authenticated read.")
    return raw


def _read_json_pair(
    payload_path: Path,
    *,
    trusted_root: Path,
    label: str,
    expected_payload_sha256: str,
    expected_sidecar_sha256: str,
) -> tuple[dict[str, Any], bytes, bytes]:
    raw = _read_regular_bytes(
        payload_path,
        trusted_root=trusted_root,
        label=label,
        require_readonly=True,
    )
    sidecar_path = Path(f"{payload_path}.sha256")
    sidecar_raw = _read_regular_bytes(
        sidecar_path,
        trusted_root=trusted_root,
        label=f"{label} sidecar",
        require_readonly=True,
    )
    if sha256_bytes(raw) != expected_payload_sha256:
        raise ValueError(f"{label} exact bytes drifted.")
    if sha256_bytes(sidecar_raw) != expected_sidecar_sha256:
        raise ValueError(f"{label} sidecar exact bytes drifted.")
    expected_sidecar = f"{expected_payload_sha256}  {payload_path.name}\n".encode("utf-8")
    if sidecar_raw != expected_sidecar:
        raise ValueError(f"{label} sidecar is semantically false.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} is not UTF-8 JSON.") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object.")
    return payload, raw, sidecar_raw


def _parse_aware(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or _AWARE_TIME_RE.fullmatch(value) is None:
        raise ValueError(f"{label} lacks a canonical explicit timezone.")
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    if re.search(r"[+-][0-9]{4}$", normalized):
        normalized = f"{normalized[:-2]}:{normalized[-2:]}"
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(f"{label} is not a valid timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} is timezone-naive.")
    return parsed


def _source_preregistration_attempts(audit_root: Path) -> list[str]:
    pattern = re.compile(
        r"flux1_dual_view_adapter_native_equivalence_v3_"
        r"source_attempt_([0-9]{3})_PREREGISTERED\.json"
    )
    attempts: list[str] = []
    for entry in os.scandir(audit_root):
        match = pattern.fullmatch(entry.name)
        if match is None:
            continue
        if entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise ValueError("Source-attempt namespace contains a non-regular preregistration.")
        attempts.append(match.group(1))
    return sorted(attempts)


def _source003_execution_attempts(audit_root: Path) -> list[str]:
    pattern = re.compile(
        r"flux1_dual_view_adapter_native_equivalence_v3_"
        r"source_attempt_003_execution_attempt_([0-9]{3})"
    )
    attempts: list[str] = []
    for entry in os.scandir(audit_root):
        match = pattern.fullmatch(entry.name)
        if match is None:
            continue
        if entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise ValueError("Source-003 execution namespace contains a non-directory attempt.")
        attempts.append(match.group(1))
    return sorted(attempts)


def _validate_preregistration(
    *, project_root: Path, audit_root: Path, require_live_sources: bool = True
) -> tuple[dict[str, Any], dict[str, Any]]:
    path = audit_root / PREREGISTRATION_FILENAME
    payload, raw, sidecar_raw = _read_json_pair(
        path,
        trusted_root=audit_root,
        label="source-003 preregistration",
        expected_payload_sha256=SOURCE003_PREREGISTRATION_SHA256,
        expected_sidecar_sha256=SOURCE003_PREREGISTRATION_SIDECAR_SHA256,
    )
    expected_top = {
        "schema_version",
        "protocol_id",
        "status",
        "created_at_utc",
        "source_attempt",
        "lineage",
        "execution",
        "files",
    }
    files = payload.get("files")
    if (
        set(payload) != expected_top
        or payload.get("schema_version") != 1
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "preregistered_before_execution"
        or payload.get("source_attempt") != SOURCE_ATTEMPT
        or not isinstance(files, Mapping)
        or set(files) != set(SOURCE003_PREREGISTERED_FILE_SHA256)
    ):
        raise ValueError("Source-003 preregistration identity/schema drifted.")
    normalized_files: dict[str, dict[str, str]] = {}
    for role, expected_sha256 in SOURCE003_PREREGISTERED_FILE_SHA256.items():
        record = files.get(role)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or not isinstance(record.get("path"), str)
            or not record["path"]
            or record.get("sha256") != expected_sha256
        ):
            raise ValueError(f"Source-003 preregistered role {role!r} drifted.")
        normalized_files[role] = {"path": str(record["path"]), "sha256": expected_sha256}
        if require_live_sources:
            live_path = project_root / str(record["path"])
            live_raw = _read_regular_bytes(
                live_path,
                trusted_root=project_root,
                label=f"live source-003 preregistered role {role}",
                require_readonly=False,
            )
            if sha256_bytes(live_raw) != expected_sha256:
                raise ValueError(f"Live source-003 preregistered role {role!r} drifted.")
    if normalized_files["diagnostic_script"]["sha256"] != (
        "4c6670c59a5e30194fdb1201695d7bcd645449b4d6b7e7a8f01f6687688d6f03"
    ) or normalized_files["equivalence_admission"]["sha256"] != (
        "c523578089fd6e4d949759556dc4aaf451768b5ada686e19128c7b44b12bf3d4"
    ):
        raise ValueError("Source-003 old diagnostic/admission hashes drifted.")
    _parse_aware(payload.get("created_at_utc"), "source-003 preregistration created_at_utc")
    return payload, {
        "path": str(path),
        "sha256": sha256_bytes(raw),
        "size_bytes": len(raw),
        "document_sha256": canonical_sha256(payload),
        "sidecar_path": str(Path(f"{path}.sha256")),
        "sidecar_sha256": sha256_bytes(sidecar_raw),
        "sidecar_size_bytes": len(sidecar_raw),
        "complete_file_ledger": normalized_files,
        "complete_file_ledger_sha256": canonical_sha256(normalized_files),
    }


def _media_geometry(path: Path, output_directory: Path) -> dict[str, Any]:
    raw = _read_regular_bytes(
        path,
        trusted_root=output_directory,
        label=f"source-003 media {path.name}",
        require_readonly=True,
    )
    try:
        with Image.open(io.BytesIO(raw)) as image:
            image.load()
            observed = {
                "format": image.format,
                "mode": image.mode,
                "width": image.width,
                "height": image.height,
            }
    except OSError as exc:
        raise ValueError(f"Source-003 media is not a decodable PNG: {path}.") from exc
    return {**observed, "sha256": sha256_bytes(raw), "size_bytes": len(raw)}


def _validate_execution_tree(output_directory: Path) -> dict[str, Any]:
    metadata = os.lstat(output_directory)
    if (
        stat.S_ISLNK(metadata.st_mode)
        or not stat.S_ISDIR(metadata.st_mode)
        or stat.S_IMODE(metadata.st_mode) != 0o755
    ):
        raise ValueError("Source-003 execution root must be one real exact-0755 directory.")
    entries = sorted(os.scandir(output_directory), key=lambda item: item.name)
    observed_names = {entry.name for entry in entries}
    if observed_names != set(SOURCE003_EXPECTED_TREE_FILES) or len(entries) != 8:
        raise ValueError("Source-003 prequarantine tree must contain exactly its eight files.")
    manifest_entries: list[dict[str, Any]] = []
    media_observed: dict[str, dict[str, Any]] = {}
    for entry in entries:
        entry_metadata = entry.stat(follow_symlinks=False)
        if (
            entry.is_symlink()
            or not stat.S_ISREG(entry_metadata.st_mode)
            or entry_metadata.st_nlink != 1
            or entry_metadata.st_mode & 0o222
        ):
            raise ValueError(
                f"Source-003 tree member is not an immutable regular file: {entry.path}."
            )
        path = Path(entry.path)
        raw = _read_regular_bytes(
            path,
            trusted_root=output_directory,
            label=f"source-003 tree member {entry.name}",
            require_readonly=True,
        )
        expected = SOURCE003_EXPECTED_TREE_FILES[entry.name]
        observed = {"sha256": sha256_bytes(raw), "size_bytes": len(raw)}
        if observed != expected:
            raise ValueError(f"Source-003 exact tree member drifted: {entry.name}.")
        manifest_entries.append(
            {
                "relative_path": entry.name,
                "kind": "file",
                "sha256": observed["sha256"],
                "size_bytes": observed["size_bytes"],
            }
        )
        if entry.name.endswith(".png"):
            media_observed[entry.name] = _media_geometry(path, output_directory)
    entries_receipt = {"schema_version": 1, "entries": manifest_entries}
    if canonical_sha256(entries_receipt) != SOURCE003_TREE_ENTRIES_SHA256:
        raise ValueError("Source-003 canonical eight-file tree digest drifted.")
    pairs: list[dict[str, Any]] = []
    pair_hashes: set[str] = set()
    for prompt_id, layout in MEDIA_LAYOUTS.items():
        route_records: dict[str, Any] = {}
        for route in MEDIA_ROUTES:
            filename = f"{prompt_id}__{route}.png"
            observed = media_observed[filename]
            if observed != {
                "format": "PNG",
                "mode": "RGB",
                "width": layout["width"],
                "height": layout["height"],
                **SOURCE003_EXPECTED_TREE_FILES[filename],
            }:
                raise ValueError(f"Source-003 PNG geometry/identity drifted: {filename}.")
            route_records[route] = {
                "path": str(output_directory / filename),
                **observed,
            }
        native = route_records["native_pipeline"]
        adapter = route_records["flux_dual_view_adapter"]
        if native["sha256"] != adapter["sha256"] or native["size_bytes"] != adapter["size_bytes"]:
            raise ValueError(f"Source-003 route media are not byte-identical for {prompt_id}.")
        pair_hashes.add(native["sha256"])
        pairs.append(
            {
                "prompt_id": prompt_id,
                "layout": deepcopy(layout),
                "native_pipeline": native,
                "flux_dual_view_adapter": adapter,
                "byte_identical": True,
            }
        )
    if len(pair_hashes) != len(MEDIA_LAYOUTS):
        raise ValueError("Source-003 prompts do not have three distinct media identities.")
    return {
        "schema_version": 1,
        "root": str(output_directory),
        "root_mode_before_quarantine": f"{stat.S_IMODE(metadata.st_mode):04o}",
        "entry_count": len(manifest_entries),
        "entries": manifest_entries,
        "entries_sha256": canonical_sha256(entries_receipt),
        "media_pair_count": len(pairs),
        "media_pairs": pairs,
    }


def _validate_result(
    *,
    project_root: Path,
    audit_root: Path,
    preregistration_binding: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    output_directory = audit_root / EXECUTION_DIRECTORY_NAME
    path = output_directory / RESULT_FILENAME
    payload, raw, sidecar_raw = _read_json_pair(
        path,
        trusted_root=audit_root,
        label="source-003 equivalence result",
        expected_payload_sha256=SOURCE003_RESULT_SHA256,
        expected_sidecar_sha256=SOURCE003_RESULT_SIDECAR_SHA256,
    )
    attempt = payload.get("attempt_lineage")
    environment = payload.get("environment_preflight")
    slurm = environment.get("slurm") if isinstance(environment, Mapping) else None
    preflight = payload.get("runner_registry_construction_preflight")
    rows = preflight.get("rows") if isinstance(preflight, Mapping) else None
    expected_row_fields = {
        "prompt_id",
        "generation_prompt_sha256",
        "dual_view_plan_sha256",
        "registered_generation_sha256",
        "projected_job_sha256",
        "projection_validation_sha256",
        "projection_mode",
        "projection_status",
        "production_runner_generation_sha256",
        "bound_model_config_sha256",
        "adapter_class",
        "adapter_name",
        "model_id",
        "revision",
        "exact_plan_bound",
        "generation_runner_constructor_checked",
    }
    if (
        payload.get("schema_version") != 1
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "passed"
        or not isinstance(attempt, Mapping)
        or attempt.get("source_attempt") != SOURCE_ATTEMPT
        or attempt.get("execution_attempt") != EXECUTION_ATTEMPT
        or attempt.get("preregistration_path") != preregistration_binding["path"]
        or attempt.get("output_directory") != str(output_directory)
        or slurm
        != {"job_id": SLURM_JOB_ID, "job_name": SLURM_JOB_NAME, "node_list": SLURM_NODE_LIST}
        or payload.get("comparison_count") != 282
        or payload.get("passed_comparison_count") != 282
        or payload.get("failed_comparison_ids") != []
        or payload.get("media_count") != 6
        or not isinstance(payload.get("media_paths"), list)
        or len(payload["media_paths"]) != 6
        or not isinstance(preflight, Mapping)
        or preflight.get("schema_version") != 1
        or preflight.get("status") != "passed"
        or preflight.get("route") != LEGACY_RUNNER_ROUTE
        or preflight.get("prompt_count") != 3
        or not isinstance(rows, list)
        or len(rows) != 3
    ):
        raise ValueError("Source-003 result quarantine identity drifted.")
    for row in rows:
        if (
            not isinstance(row, Mapping)
            or set(row) != expected_row_fields
            or _SHA256_RE.fullmatch(str(row.get("projected_job_sha256", ""))) is None
            or _SHA256_RE.fullmatch(str(row.get("projection_validation_sha256", ""))) is None
            or row.get("projection_mode") != "preview_v3"
            or row.get("projection_status")
            != "implementation_preview_not_launchable_pending_native_equivalence"
            or row.get("exact_plan_bound") is not True
            or row.get("generation_runner_constructor_checked") is not True
        ):
            raise ValueError("Source-003 legacy runner-preflight row drifted.")
    started = _parse_aware(payload.get("started_at"), "source-003 result started_at")
    ended = _parse_aware(payload.get("ended_at"), "source-003 result ended_at")
    if ended < started:
        raise ValueError("Source-003 result ended before it started.")
    tree = _validate_execution_tree(output_directory)
    return payload, {
        "path": str(path),
        "sha256": sha256_bytes(raw),
        "size_bytes": len(raw),
        "document_sha256": canonical_sha256(payload),
        "sidecar_path": str(Path(f"{path}.sha256")),
        "sidecar_sha256": sha256_bytes(sidecar_raw),
        "sidecar_size_bytes": len(sidecar_raw),
        "status": "passed",
        "comparison_count": 282,
        "passed_comparison_count": 282,
        "media_count": 6,
        "runtime_slurm": deepcopy(dict(slurm)),
        "started_at": payload["started_at"],
        "ended_at": payload["ended_at"],
        "legacy_runner_preflight": {
            "schema_version": preflight["schema_version"],
            "route": preflight["route"],
            "rows_sha256": canonical_sha256(rows),
            "raw_projected_job_hash_recomputable": False,
            "reason": "random TemporaryDirectory output_dir was not persisted",
        },
        "prequarantine_execution_tree": tree,
    }


def _sacct_command() -> list[str]:
    return [
        "sacct",
        "--noheader",
        "--parsable2",
        "--jobs",
        SLURM_JOB_ID,
        "--format",
        "JobIDRaw,JobName%128,State,ExitCode,Submit,Start,End",
    ]


def _parse_sacct_stdout(stdout: str) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for line in stdout.splitlines():
        if not line.strip():
            continue
        fields = line.split("|")
        if fields and fields[-1] == "":
            fields.pop()
        if len(fields) != len(SACCT_FIELDS):
            raise ValueError("Source-003 sacct output field coverage drifted.")
        rows.append(dict(zip(SACCT_FIELDS, fields, strict=True)))
    expected_ids = [SLURM_JOB_ID, f"{SLURM_JOB_ID}.batch", f"{SLURM_JOB_ID}.extern"]
    if [row["JobIDRaw"] for row in rows] != expected_ids:
        raise ValueError("Source-003 sacct row coverage/order drifted.")
    expected_names = [SLURM_JOB_NAME, "batch", "extern"]
    for row, expected_name in zip(rows, expected_names, strict=True):
        if (
            row["JobName"] != expected_name
            or row["State"] != "COMPLETED"
            or row["ExitCode"] != "0:0"
        ):
            raise ValueError("Source-003 sacct terminal identity drifted.")
        for field in ("Submit", "Start", "End"):
            _parse_aware(row[field], f"source-003 sacct {row['JobIDRaw']} {field}")
    if any(
        row[field] != rows[0][field] for row in rows[1:] for field in ("Submit", "Start", "End")
    ):
        raise ValueError("Source-003 sacct step timestamps differ from the allocation.")
    return rows


def query_source003_terminal_slurm(
    *,
    run: Callable[..., Any] = subprocess.run,
    queried_at_utc: str | None = None,
) -> dict[str, Any]:
    command = _sacct_command()
    environment = {
        **os.environ,
        "SLURM_TIME_FORMAT": SLURM_TIME_FORMAT,
        "TZ": "UTC",
    }
    completed = run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    if completed.returncode != 0 or completed.stderr:
        raise ValueError("Source-003 sacct query did not complete cleanly.")
    rows = _parse_sacct_stdout(completed.stdout)
    queried_at_utc = queried_at_utc or datetime.now(timezone.utc).isoformat()
    queried = _parse_aware(queried_at_utc, "source-003 sacct queried_at_utc")
    terminal_end = _parse_aware(rows[0]["End"], "source-003 sacct terminal End")
    if terminal_end > queried or queried > datetime.now(timezone.utc):
        raise ValueError("Source-003 sacct query timestamp order drifted.")
    return {
        "schema_version": 1,
        "command": command,
        "environment": {"SLURM_TIME_FORMAT": SLURM_TIME_FORMAT, "TZ": "UTC"},
        "queried_at_utc": queried_at_utc,
        "returncode": completed.returncode,
        "stdout": completed.stdout,
        "stdout_sha256": sha256_bytes(completed.stdout.encode("utf-8")),
        "stderr": completed.stderr,
        "rows": rows,
        "terminal": {
            "job_id": rows[0]["JobIDRaw"],
            "job_name": rows[0]["JobName"],
            "state": rows[0]["State"],
            "exit_code": rows[0]["ExitCode"],
            "submitted_at": rows[0]["Submit"],
            "started_at": rows[0]["Start"],
            "ended_at": rows[0]["End"],
        },
    }


def build_quarantine_authorization(
    *,
    project_root: str | Path,
    run: Callable[..., Any] = subprocess.run,
    queried_at_utc: str | None = None,
    authorized_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build, but do not publish, the exact source-003 authorization."""

    project_root = _lexical_absolute(project_root)
    audit_root = project_root / AUDIT_ROOT_RELATIVE
    _assert_no_symlink_components(audit_root, project_root, "source-003 audit root")
    if _source_preregistration_attempts(audit_root) != ["001", "002", "003"]:
        raise ValueError("Quarantine authorization requires exactly source attempts 001-003.")
    if _source003_execution_attempts(audit_root) != [EXECUTION_ATTEMPT]:
        raise ValueError("Quarantine authorization requires exactly source-003 execution 001.")
    authorization_path = audit_root / AUTHORIZATION_FILENAME
    if os.path.lexists(authorization_path) or os.path.lexists(Path(f"{authorization_path}.sha256")):
        raise FileExistsError("Source-003 quarantine authorization destination is occupied.")
    publisher_path = project_root / PUBLISHER_SOURCE_RELATIVE
    publisher_raw = _read_regular_bytes(
        publisher_path,
        trusted_root=project_root,
        label="quarantine authorization publisher source",
        require_readonly=False,
    )
    preregistration, preregistration_binding = _validate_preregistration(
        project_root=project_root,
        audit_root=audit_root,
    )
    result, result_binding = _validate_result(
        project_root=project_root,
        audit_root=audit_root,
        preregistration_binding=preregistration_binding,
    )
    scheduler = query_source003_terminal_slurm(run=run, queried_at_utc=queried_at_utc)
    preregistered = _parse_aware(
        preregistration["created_at_utc"], "source-003 preregistration created_at_utc"
    )
    submitted = _parse_aware(scheduler["terminal"]["submitted_at"], "source-003 Submit")
    scheduler_started = _parse_aware(
        scheduler["terminal"]["started_at"], "source-003 scheduler Start"
    )
    result_started = _parse_aware(result["started_at"], "source-003 result Start")
    result_ended = _parse_aware(result["ended_at"], "source-003 result End")
    scheduler_ended = _parse_aware(scheduler["terminal"]["ended_at"], "source-003 scheduler End")
    queried = _parse_aware(scheduler["queried_at_utc"], "source-003 query time")
    authorized_at_utc = authorized_at_utc or datetime.now(timezone.utc).isoformat()
    authorized = _parse_aware(authorized_at_utc, "quarantine authorized_at_utc")
    if authorized > datetime.now(timezone.utc) or not (
        preregistered
        <= submitted
        <= scheduler_started
        <= result_started
        <= result_ended
        <= scheduler_ended
        <= queried
        <= authorized
    ):
        raise ValueError("Source-003 quarantine authorization timestamp ordering drifted.")
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "authorization": AUTHORIZATION_NAME,
        "protocol_id": PROTOCOL_ID,
        "status": AUTHORIZATION_STATUS,
        "authorized_at_utc": authorized_at_utc,
        "scope": {
            "source_attempt": SOURCE_ATTEMPT,
            "execution_attempt": EXECUTION_ATTEMPT,
            "successor_source_attempt": SUCCESSOR_SOURCE_ATTEMPT,
            "slurm_job_id": SLURM_JOB_ID,
            "execution_directory": str(audit_root / EXECUTION_DIRECTORY_NAME),
        },
        "disposition": {
            "classification": "passed_but_unaccepted",
            "acceptance_forbidden": True,
            "ordinary_terminal_receipt_forbidden": True,
            "media_reuse": False,
            "scientific_media_reuse": False,
            "scientific_media_reuse_forbidden": True,
            "same_source_retry": False,
            "same_source_retry_forbidden": True,
            "fresh_h100_execution_required": True,
            "authorized_successor_source_attempt": SUCCESSOR_SOURCE_ATTEMPT,
        },
        "repair_intent": {
            "reason": REPAIR_REASON,
            "intended_changed_roles": list(INTENDED_CHANGED_ROLES),
            "future_source_hashes_committed": False,
            "future_source_hashes": None,
            "required_contract": (
                "versioned shared runner-preflight receipt with exact top-level output_dir "
                "sentinel canonicalization and strict projector/validator recomputation"
            ),
        },
        "publisher_source": {
            "path": str(publisher_path),
            "relative_path": str(PUBLISHER_SOURCE_RELATIVE),
            "sha256": sha256_bytes(publisher_raw),
            "size_bytes": len(publisher_raw),
        },
        "source003_preregistration": preregistration_binding,
        "source003_old_role_hashes": {
            role: deepcopy(preregistration_binding["complete_file_ledger"][role])
            for role in INTENDED_CHANGED_ROLES
        },
        "source003_result": {
            key: deepcopy(value)
            for key, value in result_binding.items()
            if key != "prequarantine_execution_tree"
        },
        "source003_prequarantine_execution_tree": result_binding["prequarantine_execution_tree"],
        "source003_scheduler_authentication": scheduler,
    }
    payload["document_sha256"] = document_sha256(payload)
    return payload


def validate_authorization_payload(
    payload: Any,
    *,
    project_root: str | Path,
    require_live_source003_roles: bool = True,
    requery_slurm_with: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """Validate the closed authorization schema before immutable publication."""

    project_root = _lexical_absolute(project_root)
    audit_root = project_root / AUDIT_ROOT_RELATIVE
    expected_top = {
        "schema_version",
        "authorization",
        "protocol_id",
        "status",
        "authorized_at_utc",
        "scope",
        "disposition",
        "repair_intent",
        "publisher_source",
        "source003_preregistration",
        "source003_old_role_hashes",
        "source003_result",
        "source003_prequarantine_execution_tree",
        "source003_scheduler_authentication",
        "document_sha256",
    }
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_top
        or payload.get("schema_version") != SCHEMA_VERSION
        or payload.get("authorization") != AUTHORIZATION_NAME
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != AUTHORIZATION_STATUS
        or payload.get("scope")
        != {
            "source_attempt": SOURCE_ATTEMPT,
            "execution_attempt": EXECUTION_ATTEMPT,
            "successor_source_attempt": SUCCESSOR_SOURCE_ATTEMPT,
            "slurm_job_id": SLURM_JOB_ID,
            "execution_directory": str(audit_root / EXECUTION_DIRECTORY_NAME),
        }
        or payload.get("disposition")
        != {
            "classification": "passed_but_unaccepted",
            "acceptance_forbidden": True,
            "ordinary_terminal_receipt_forbidden": True,
            "media_reuse": False,
            "scientific_media_reuse": False,
            "scientific_media_reuse_forbidden": True,
            "same_source_retry": False,
            "same_source_retry_forbidden": True,
            "fresh_h100_execution_required": True,
            "authorized_successor_source_attempt": SUCCESSOR_SOURCE_ATTEMPT,
        }
        or payload.get("repair_intent")
        != {
            "reason": REPAIR_REASON,
            "intended_changed_roles": list(INTENDED_CHANGED_ROLES),
            "future_source_hashes_committed": False,
            "future_source_hashes": None,
            "required_contract": (
                "versioned shared runner-preflight receipt with exact top-level output_dir "
                "sentinel canonicalization and strict projector/validator recomputation"
            ),
        }
        or payload.get("document_sha256") != document_sha256(payload)
    ):
        raise ValueError("Source-003 quarantine authorization schema/identity drifted.")
    _parse_aware(payload.get("authorized_at_utc"), "quarantine authorized_at_utc")
    publisher = payload.get("publisher_source")
    publisher_path = project_root / PUBLISHER_SOURCE_RELATIVE
    publisher_raw = _read_regular_bytes(
        publisher_path,
        trusted_root=project_root,
        label="quarantine authorization publisher source",
        require_readonly=False,
    )
    if publisher != {
        "path": str(publisher_path),
        "relative_path": str(PUBLISHER_SOURCE_RELATIVE),
        "sha256": sha256_bytes(publisher_raw),
        "size_bytes": len(publisher_raw),
    }:
        raise ValueError("Quarantine authorization publisher source binding drifted.")
    preregistration = payload.get("source003_preregistration")
    old_roles = payload.get("source003_old_role_hashes")
    ledger = (
        preregistration.get("complete_file_ledger")
        if isinstance(preregistration, Mapping)
        else None
    )
    if not isinstance(ledger, Mapping) or old_roles != {
        role: ledger.get(role) for role in INTENDED_CHANGED_ROLES
    }:
        raise ValueError("Quarantine authorization old source-role binding drifted.")
    expected_old_roles: dict[str, dict[str, str]] = {}
    for role in INTENDED_CHANGED_ROLES:
        record = ledger.get(role)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or not isinstance(record.get("path"), str)
        ):
            raise ValueError("Quarantine authorization old source-role binding drifted.")
        expected_old_roles[role] = {
            "path": str(record["path"]),
            "sha256": SOURCE003_PREREGISTERED_FILE_SHA256[role],
        }
    if old_roles != expected_old_roles:
        raise ValueError("Quarantine authorization old diagnostic/admission hash drifted.")
    tree = payload.get("source003_prequarantine_execution_tree")
    if (
        not isinstance(tree, Mapping)
        or tree.get("entry_count") != 8
        or tree.get("entries_sha256") != SOURCE003_TREE_ENTRIES_SHA256
        or tree.get("media_pair_count") != 3
    ):
        raise ValueError("Quarantine authorization execution-tree binding drifted.")
    result = payload.get("source003_result")
    scheduler = payload.get("source003_scheduler_authentication")
    terminal = scheduler.get("terminal") if isinstance(scheduler, Mapping) else None
    scheduler_fields = {
        "schema_version",
        "command",
        "environment",
        "queried_at_utc",
        "returncode",
        "stdout",
        "stdout_sha256",
        "stderr",
        "rows",
        "terminal",
    }
    parsed_scheduler_rows = (
        _parse_sacct_stdout(str(scheduler.get("stdout", "")))
        if isinstance(scheduler, Mapping)
        else []
    )
    expected_terminal = (
        {
            "job_id": parsed_scheduler_rows[0]["JobIDRaw"],
            "job_name": parsed_scheduler_rows[0]["JobName"],
            "state": parsed_scheduler_rows[0]["State"],
            "exit_code": parsed_scheduler_rows[0]["ExitCode"],
            "submitted_at": parsed_scheduler_rows[0]["Submit"],
            "started_at": parsed_scheduler_rows[0]["Start"],
            "ended_at": parsed_scheduler_rows[0]["End"],
        }
        if parsed_scheduler_rows
        else None
    )
    if (
        not isinstance(result, Mapping)
        or result.get("sha256") != SOURCE003_RESULT_SHA256
        or result.get("sidecar_sha256") != SOURCE003_RESULT_SIDECAR_SHA256
        or result.get("status") != "passed"
        or result.get("comparison_count") != 282
        or result.get("passed_comparison_count") != 282
        or result.get("media_count") != 6
        or not isinstance(scheduler, Mapping)
        or set(scheduler) != scheduler_fields
        or scheduler.get("schema_version") != 1
        or scheduler.get("command") != _sacct_command()
        or scheduler.get("environment") != {"SLURM_TIME_FORMAT": SLURM_TIME_FORMAT, "TZ": "UTC"}
        or scheduler.get("returncode") != 0
        or scheduler.get("stderr") != ""
        or not isinstance(scheduler.get("stdout"), str)
        or scheduler.get("stdout_sha256")
        != sha256_bytes(str(scheduler.get("stdout", "")).encode("utf-8"))
        or scheduler.get("rows") != parsed_scheduler_rows
        or not isinstance(terminal, Mapping)
        or terminal != expected_terminal
        or terminal.get("job_id") != SLURM_JOB_ID
        or terminal.get("job_name") != SLURM_JOB_NAME
        or terminal.get("state") != "COMPLETED"
        or terminal.get("exit_code") != "0:0"
    ):
        raise ValueError("Quarantine authorization result/scheduler binding drifted.")
    _parse_aware(scheduler.get("queried_at_utc"), "quarantine scheduler queried_at_utc")

    if _source_preregistration_attempts(audit_root) != ["001", "002", "003"]:
        raise ValueError("Quarantine authorization source namespace changed before commit.")
    if _source003_execution_attempts(audit_root) != [EXECUTION_ATTEMPT]:
        raise ValueError("Quarantine authorization execution namespace changed before commit.")

    # Reopen the complete source-003 evidence immediately before publication.
    # Requiring the live preregistered roles here is stage-1-only: after this
    # authorization is frozen, the three intended roles are expected to change.
    historical_preregistration, live_preregistration = _validate_preregistration(
        project_root=project_root,
        audit_root=audit_root,
        require_live_sources=require_live_source003_roles,
    )
    historical_result, live_result = _validate_result(
        project_root=project_root,
        audit_root=audit_root,
        preregistration_binding=live_preregistration,
    )
    expected_result_without_tree = {
        key: deepcopy(value)
        for key, value in live_result.items()
        if key != "prequarantine_execution_tree"
    }
    if (
        payload.get("source003_preregistration") != live_preregistration
        or payload.get("source003_result") != expected_result_without_tree
        or payload.get("source003_prequarantine_execution_tree")
        != live_result["prequarantine_execution_tree"]
        or historical_preregistration.get("source_attempt") != SOURCE_ATTEMPT
        or historical_result.get("status") != "passed"
    ):
        raise ValueError("Quarantine authorization no longer matches live source-003 evidence.")
    preregistered = _parse_aware(
        historical_preregistration.get("created_at_utc"),
        "source-003 preregistration created_at_utc",
    )
    submitted = _parse_aware(terminal["submitted_at"], "source-003 scheduler Submit")
    scheduler_started = _parse_aware(terminal["started_at"], "source-003 scheduler Start")
    result_started = _parse_aware(historical_result.get("started_at"), "source-003 result Start")
    result_ended = _parse_aware(historical_result.get("ended_at"), "source-003 result End")
    scheduler_ended = _parse_aware(terminal["ended_at"], "source-003 scheduler End")
    queried = _parse_aware(scheduler.get("queried_at_utc"), "source-003 scheduler query")
    authorized = _parse_aware(payload.get("authorized_at_utc"), "quarantine authorization time")
    if authorized > datetime.now(timezone.utc) or not (
        preregistered
        <= submitted
        <= scheduler_started
        <= result_started
        <= result_ended
        <= scheduler_ended
        <= queried
        <= authorized
    ):
        raise ValueError("Quarantine authorization timestamp ordering drifted.")
    if requery_slurm_with is not None:
        live_scheduler = query_source003_terminal_slurm(run=requery_slurm_with)
        for field in ("command", "environment", "stdout", "stdout_sha256", "rows", "terminal"):
            if live_scheduler[field] != scheduler[field]:
                raise ValueError(
                    "Quarantine authorization no longer matches live source-003 sacct evidence."
                )
    return deepcopy(dict(payload))


def _write_staged_readonly_at(directory_fd: int, name: str, raw: bytes) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, 0o600, dir_fd=directory_fd)
    try:
        view = memoryview(raw)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
    finally:
        os.close(descriptor)


def _publish_readonly_pair_no_overwrite(
    *, payload_path: Path, payload_raw: bytes, sidecar_raw: bytes
) -> None:
    parent = payload_path.parent
    sidecar_path = Path(f"{payload_path}.sha256")
    parent_before = os.lstat(parent)
    if stat.S_ISLNK(parent_before.st_mode) or not stat.S_ISDIR(parent_before.st_mode):
        raise ValueError("Quarantine authorization parent must be one real directory.")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    parent_fd = os.open(parent, directory_flags)
    staging: Path | None = None
    staging_fd: int | None = None
    try:
        opened_parent = os.fstat(parent_fd)
        if (parent_before.st_dev, parent_before.st_ino) != (
            opened_parent.st_dev,
            opened_parent.st_ino,
        ):
            raise ValueError("Quarantine authorization parent changed while opening it.")
        for name in (payload_path.name, sidecar_path.name):
            try:
                os.stat(name, dir_fd=parent_fd, follow_symlinks=False)
            except FileNotFoundError:
                pass
            else:
                raise FileExistsError(
                    "Quarantine authorization destination is occupied; refusing replay."
                )
        staging = Path(tempfile.mkdtemp(prefix=".source003-quarantine-auth-", dir=parent))
        staging_fd = os.open(staging.name, directory_flags, dir_fd=parent_fd)
        staging_metadata = os.fstat(staging_fd)
        if not stat.S_ISDIR(staging_metadata.st_mode):
            raise ValueError("Quarantine authorization staging inode is not a directory.")
        _write_staged_readonly_at(staging_fd, payload_path.name, payload_raw)
        _write_staged_readonly_at(staging_fd, sidecar_path.name, sidecar_raw)
        os.fsync(staging_fd)
        os.link(
            payload_path.name,
            payload_path.name,
            src_dir_fd=staging_fd,
            dst_dir_fd=parent_fd,
            follow_symlinks=False,
        )
        os.fsync(parent_fd)
        try:
            os.link(
                sidecar_path.name,
                sidecar_path.name,
                src_dir_fd=staging_fd,
                dst_dir_fd=parent_fd,
                follow_symlinks=False,
            )
        except Exception as exc:
            raise RuntimeError(
                "Quarantine authorization publication is ambiguous after payload commit; "
                "do not overwrite or replay it."
            ) from exc
        os.fsync(parent_fd)
        parent_after = os.lstat(parent)
        if (parent_after.st_dev, parent_after.st_ino) != (
            opened_parent.st_dev,
            opened_parent.st_ino,
        ):
            raise RuntimeError(
                "Quarantine authorization parent changed during commit; evidence is ambiguous."
            )
    finally:
        if staging_fd is not None:
            os.close(staging_fd)
        if staging is not None:
            shutil.rmtree(staging, ignore_errors=True)
        os.close(parent_fd)


def publish_quarantine_authorization(
    *,
    project_root: str | Path,
    run: Callable[..., Any] = subprocess.run,
    queried_at_utc: str | None = None,
    authorized_at_utc: str | None = None,
) -> dict[str, Any]:
    """Build and commit the authorization pair without overwrite."""

    project_root = _lexical_absolute(project_root)
    audit_root = project_root / AUDIT_ROOT_RELATIVE
    payload = build_quarantine_authorization(
        project_root=project_root,
        run=run,
        queried_at_utc=queried_at_utc,
        authorized_at_utc=authorized_at_utc,
    )
    validate_authorization_payload(
        payload,
        project_root=project_root,
        requery_slurm_with=run,
    )
    destination = audit_root / AUTHORIZATION_FILENAME
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    sidecar_raw = f"{sha256_bytes(raw)}  {destination.name}\n".encode("utf-8")
    _publish_readonly_pair_no_overwrite(
        payload_path=destination,
        payload_raw=raw,
        sidecar_raw=sidecar_raw,
    )
    published, published_raw, sidecar = _read_json_pair(
        destination,
        trusted_root=audit_root,
        label="source-003 quarantine authorization",
        expected_payload_sha256=sha256_bytes(raw),
        expected_sidecar_sha256=sha256_bytes(sidecar_raw),
    )
    validate_authorization_payload(
        published,
        project_root=project_root,
        requery_slurm_with=run,
    )
    return {
        "authorization": published,
        "binding": {
            "path": str(destination),
            "sha256": sha256_bytes(published_raw),
            "document_sha256": published["document_sha256"],
            "sidecar_path": str(Path(f"{destination}.sha256")),
            "sidecar_sha256": sha256_bytes(sidecar),
        },
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    output = publish_quarantine_authorization(project_root=args.project_root)
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
