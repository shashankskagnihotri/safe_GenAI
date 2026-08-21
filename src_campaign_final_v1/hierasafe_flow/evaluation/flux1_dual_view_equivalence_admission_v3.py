"""Fail-closed admission for the FLUX.1 dual-view native-equivalence gate.

This module is deliberately separate from the common-seed v3 protocol.  Its
bytes can be frozen before the H100 equivalence run, while the later common-
seed config and protocol remain free to bind the resulting evidence without a
self-hash cycle.  Admission requires the attempt-aware preregistration pair,
status-specific result pair, status-neutral terminal-receipt pair, and (for a
passing run) acceptance-receipt pair.  A passing result must contain the exact
282-comparison three-prompt result, six authenticated PNGs, and a successful
terminal Slurm identity.

The CLI publishes the acceptance receipt commit-last.  It never overwrites an
existing receipt or sidecar: a partially occupied destination is an ambiguous
publication that must be reconciled rather than replayed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import re
import secrets
import stat
import subprocess
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from PIL import Image
import yaml

SCHEMA_VERSION = 1
PROTOCOL_ID = "flux1_dual_view_adapter_native_equivalence_v3"
GATE = "three_prompt_bitwise_native_flux_dual_view_adapter_equivalence"
RECEIPT_NAME = "flux1_dual_view_adapter_native_equivalence_acceptance_v3"
ACCEPTED_STATUS = "accepted_for_flux1_common_seed_source_ladder_v3"
TERMINAL_RECEIPT_NAME = "flux1_dual_view_adapter_execution_terminal_v3"
TERMINAL_STATUS = "terminal_execution_authenticated"

MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
PIPELINE_CLASS = "FluxPipeline"
ADAPTER_KEY = "flux_dual_view"
DIFFUSERS_COMMIT = "577b28f8f5d30eabdd357d74944cd76568292faf"
DIFFUSERS_VERSION = "0.39.0.dev0"
EXPECTED_CONDA_PREFIX = "/ceph/sagnihot/miniconda3/envs/safe_genai_conceptsteer"
EXPECTED_DIFFUSERS_PIPELINE_SOURCE = Path(
    f"{EXPECTED_CONDA_PREFIX}/lib/python3.10/site-packages/"
    "diffusers/pipelines/flux/pipeline_flux.py"
)
EXPECTED_DIFFUSERS_SCHEDULER_SOURCE = Path(
    f"{EXPECTED_CONDA_PREFIX}/lib/python3.10/site-packages/"
    "diffusers/schedulers/scheduling_flow_match_euler_discrete.py"
)
EXPECTED_DIFFUSERS_CONFIGURATION_SOURCE = Path(
    f"{EXPECTED_CONDA_PREFIX}/lib/python3.10/site-packages/diffusers/configuration_utils.py"
)
EXPECTED_MODEL_SNAPSHOT_PATH = Path(
    "/ceph/sagnihot/cache/huggingface/hub/"
    "models--black-forest-labs--FLUX.1-dev/snapshots/"
    f"{MODEL_REVISION}"
)
EQUIVALENCE_SLURM_JOB_NAME = "flux1-dv-eq-v3"
SACCT_TIME_FORMAT = "%Y-%m-%dT%H:%M:%S%z"
SACCT_FIELDS = ("JobIDRaw", "JobName", "State", "ExitCode", "Submit", "Start", "End")

PROMPT_IDS = (
    "01_sad_young_girl",
    "02_angry_old_man",
    "03_empty_outdoor_mall",
)
IMAGE_LAYOUTS = {
    "01_sad_young_girl": {"width": 832, "height": 1216},
    "02_angry_old_man": {"width": 832, "height": 1216},
    "03_empty_outdoor_mall": {"width": 1216, "height": 832},
}
SEED = 0
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5
TRUE_CFG_SCALE = 1.0
COMPARISONS_PER_PROMPT = 94
COMPARISON_COUNT = len(PROMPT_IDS) * COMPARISONS_PER_PROMPT
MEDIA_COUNT = 2 * len(PROMPT_IDS)
MEDIA_ROUTES = ("native_pipeline", "flux_dual_view_adapter")

AUDIT_ROOT_RELATIVE = Path("debugging/audits/finer_detailing_20260720")
RESULT_FILENAME = "equivalence_result.json"
RESULT_SIDECAR_FILENAME = f"{RESULT_FILENAME}.sha256"
PREREGISTRATION_FILENAME_TEMPLATE = (
    "flux1_dual_view_adapter_native_equivalence_v3_"
    "source_attempt_{source_attempt}_PREREGISTERED.json"
)
RESULT_DIRECTORY_TEMPLATE = (
    "flux1_dual_view_adapter_native_equivalence_v3_"
    "source_attempt_{source_attempt}_execution_attempt_{execution_attempt}"
)
RECEIPT_FILENAME = "acceptance_receipt.json"
RECEIPT_SIDECAR_FILENAME = f"{RECEIPT_FILENAME}.sha256"
TERMINAL_RECEIPT_FILENAME = "terminal_receipt.json"
TERMINAL_RECEIPT_SIDECAR_FILENAME = f"{TERMINAL_RECEIPT_FILENAME}.sha256"
QUARANTINE_AUTHORIZATION_FILENAME = (
    "flux1_dual_view_adapter_native_equivalence_v3_source_attempt_003_QUARANTINE_AUTHORIZATION.json"
)
QUARANTINE_AUTHORIZATION_SIDECAR_FILENAME = f"{QUARANTINE_AUTHORIZATION_FILENAME}.sha256"
QUARANTINE_TERMINAL_RECEIPT_FILENAME = "quarantine_terminal_receipt.json"
QUARANTINE_TERMINAL_RECEIPT_SIDECAR_FILENAME = f"{QUARANTINE_TERMINAL_RECEIPT_FILENAME}.sha256"
QUARANTINE_TERMINAL_RECEIPT_NAME = "flux1_source003_quarantine_terminal_v1"
QUARANTINE_TERMINAL_STATUS = "quarantined_unaccepted_schema_mismatch"
QUARANTINE_SOURCE_ATTEMPT = "003"
QUARANTINE_EXECUTION_ATTEMPT = "001"
QUARANTINE_SUCCESSOR_SOURCE_ATTEMPT = "004"
QUARANTINE_SLURM_JOB_ID = "291999"
QUARANTINE_AUTHORIZATION_SHA256 = "fe8910b968637c15d6c01087f21dbff50e30f39145ef4471d3ac19c00ef39f80"
QUARANTINE_AUTHORIZATION_SIDECAR_SHA256 = (
    "208945c2422a602f467f0b88e43b9c8135bf530e4e9730b8c290497dad9c3a1e"
)
QUARANTINE_AUTHORIZATION_DOCUMENT_SHA256 = (
    "cee93d03871c7391b6416b32bd2e1745f62448ce5ac3ea3a9ddc6b600de67a50"
)
QUARANTINE_PUBLISHER_RELATIVE = Path("scripts/publish_flux1_source003_quarantine_authorization.py")
QUARANTINE_PUBLISHER_SHA256 = "f92b611528e2bb7c4d4f282c6ed5cbbfcd21bef1280fa977e8cb8efa228c7f67"
QUARANTINE_SOURCE003_PREREGISTRATION_SHA256 = (
    "6bc6597703d998e38a49c4229bae7c5501809eb57ce6d5e62f379557a6626314"
)
QUARANTINE_SOURCE003_RESULT_SHA256 = (
    "f4dda44dc0ac5c3dbe59a29d613819e1131d6464a61bc2eeef2ad24c65a59156"
)
QUARANTINE_SOURCE003_TREE_ENTRIES_SHA256 = (
    "20ea106d3e8ab5bbf579c35b73f8963b6c5e74497c5569e0a231ad99a053e8ce"
)
QUARANTINE_CHANGED_ROLES = (
    "diagnostic_script",
    "equivalence_admission",
    "flux_dual_view_job_projector",
)
SOURCE004_QUARANTINE_REPAIR_REASON = (
    "Prospective source-004 repairs the source-003 unrecomputable temporary-output projection "
    "digest and strict runner-preflight schema under the immutable passed-but-unaccepted "
    "quarantine authorization and terminal; it requires a fresh H100 execution."
)
PROMPT_CONTRACT_RELATIVE = Path("configs/experiments/flux1_common_seed_source_prompts_v3.yaml")
PROMPT_CONTRACT_SIDECAR_RELATIVE = Path(f"{PROMPT_CONTRACT_RELATIVE}.sha256")
PROMPT_PROTOCOL_RELATIVE = Path(
    "debugging/audits/finer_detailing_20260720/"
    "flux1_common_seed_source_prompts_v3_PROTOCOL_PREREGISTERED.md"
)
PROMPT_PROTOCOL_SIDECAR_RELATIVE = Path(f"{PROMPT_PROTOCOL_RELATIVE}.sha256")
MODEL_CONFIG_RELATIVE = Path("configs/models/t2i_flux1_dev_dual_view_v3.yaml")
MODEL_CONFIG_SIDECAR_RELATIVE = Path(f"{MODEL_CONFIG_RELATIVE}.sha256")
SLURM_LAUNCHER_RELATIVE = Path("slurm/flux1_dual_view_adapter_equivalence_h100.sbatch")
ADMISSION_SOURCE_RELATIVE = Path(
    "src/hierasafe_flow/evaluation/flux1_dual_view_equivalence_admission_v3.py"
)

PROMPT_CONTRACT_SHA256 = "1d061d331293980023599445f680f865b4d4af41f91144e9d6a3b3345ffe9b87"
PROMPT_PROTOCOL_SHA256 = "c79c7ed65dac46a844be0022b2a81001536e10ad96ef00c7afecbddc6063e1ec"
MODEL_CONFIG_SHA256 = "624cc2f53e293102e6e31f016c307d30f5108a916afdbdc06d6e6365fb6bbd9d"
PIPELINE_SOURCE_SHA256 = "ac0613aca45759fc2a5f4b5310ac9f2d8ea7f4cc34f04adada03b80896eaec92"
SCHEDULER_SOURCE_SHA256 = "8b96f25a6170480fb02decb58904e026250d6d68be5de8a7b8785b4c721e9d80"
CONFIGURATION_SOURCE_SHA256 = "4a7af9be48913edfa77f3d32c375c997fa15b53c3f04efd5c371b36c5c6c1960"
MODEL_INDEX_SHA256 = "24946df21ff25e210486b5f6b14208983a90c9c73f8d48cfa724c0e4e03f7201"
LEGACY_FLUX_ADAPTER_SHA256 = "114f76c09b76198f254bf5a7d5cf4a5b2cef5e12256835fd7b71fc035e19e439"
SNAPSHOT_SCHEDULER_CONFIG_RELATIVE = "scheduler/scheduler_config.json"
SNAPSHOT_SCHEDULER_CONFIG_SIZE_BYTES = 273
SNAPSHOT_SCHEDULER_CONFIG_SHA256 = (
    "63b6d2fa93579e383a308009bfd5ae12e61f8950017dd749302f8d54bb3520b6"
)
SNAPSHOT_SCHEDULER_CONFIG_PARSED_SHA256 = (
    "b2c89c753d9bca5ff22a22d33e9887ea45bf79b93053815efb60a8514d51e1fe"
)
SNAPSHOT_SCHEDULER_CONFIG_BLOB_NAME = "a64d1e8763bef5828984682680673fb97829734b"
EXPECTED_SNAPSHOT_SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.30.0.dev0",
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "use_dynamic_shifting": True,
}
EXPECTED_SCHEDULER_DEFAULT_VALUE_KEYS = (
    "invert_sigmas",
    "shift_terminal",
    "stochastic_sampling",
    "time_shift_type",
    "use_beta_sigmas",
    "use_exponential_sigmas",
    "use_karras_sigmas",
)
EXPECTED_SCHEDULER_CONFIG = {
    "_class_name": "FlowMatchEulerDiscreteScheduler",
    "_diffusers_version": "0.30.0.dev0",
    "_use_default_values": list(EXPECTED_SCHEDULER_DEFAULT_VALUE_KEYS),
    "base_image_seq_len": 256,
    "base_shift": 0.5,
    "invert_sigmas": False,
    "max_image_seq_len": 4096,
    "max_shift": 1.15,
    "num_train_timesteps": 1000,
    "shift": 3.0,
    "shift_terminal": None,
    "stochastic_sampling": False,
    "time_shift_type": "exponential",
    "use_beta_sigmas": False,
    "use_dynamic_shifting": True,
    "use_exponential_sigmas": False,
    "use_karras_sigmas": False,
}
EXPECTED_SCHEDULER_CONFIG_SHA256 = (
    "d32e7bf63b561cb6aa9101116b0bd3e4946d17af13a0dd1b674f24ca7dae509b"
)

# This is the exact preregistration surface.  It contains the code actually
# exercised by the diagnostic plus the sealed prompt/model inputs and this
# validator.  The later common-seed module and dispatch wrappers are excluded:
# they are not executed by this gate and must change after it to bind the real
# result and receipt.
REQUIRED_PREREGISTRATION_FILE_PATHS = {
    "diagnostic_script": Path("scripts/validate_flux1_dual_view_adapter_native_equivalence.py"),
    "slurm_launcher": SLURM_LAUNCHER_RELATIVE,
    "dual_view_adapter": Path("src/hierasafe_flow/adapters/flux_dual_view_adapter.py"),
    "legacy_flux_adapter": Path("src/hierasafe_flow/adapters/flux_adapter.py"),
    "adapter_base": Path("src/hierasafe_flow/adapters/base.py"),
    "adapter_registry": Path("src/hierasafe_flow/adapters/registry.py"),
    "generation_runner": Path("src/hierasafe_flow/generation/runner.py"),
    "flux_dual_view_job_projector": Path(
        "src/hierasafe_flow/evaluation/flux1_dual_view_jobs_v3.py"
    ),
    "config_loader": Path("src/hierasafe_flow/utils/config.py"),
    "benchmark_runner": Path("src/hierasafe_flow/benchmarks/finer_detailing_correction.py"),
    "shared_benchmark_runner": Path("src/hierasafe_flow/benchmarks/benign_park.py"),
    "benchmark_entrypoint": Path("scripts/run_finer_detailing_correction.py"),
    "base_config": Path("configs/default.yaml"),
    "prompt1_concept_hierarchy": Path("configs/concepts/finer_detailing_01_sad_young_girl.yaml"),
    "prompt2_concept_hierarchy": Path("configs/concepts/finer_detailing_02_angry_old_man.yaml"),
    "prompt3_concept_hierarchy": Path(
        "configs/concepts/finer_detailing_03_empty_outdoor_mall_t2i.yaml"
    ),
    "equivalence_admission": ADMISSION_SOURCE_RELATIVE,
    "model_config": MODEL_CONFIG_RELATIVE,
    "model_config_sidecar": MODEL_CONFIG_SIDECAR_RELATIVE,
    "prompt_contract": PROMPT_CONTRACT_RELATIVE,
    "prompt_contract_sidecar": PROMPT_CONTRACT_SIDECAR_RELATIVE,
    "prompt_protocol": PROMPT_PROTOCOL_RELATIVE,
    "prompt_protocol_sidecar": PROMPT_PROTOCOL_SIDECAR_RELATIVE,
    "requirements": Path("requirements.txt"),
    "environment": Path("environment.yml"),
    "pyproject": Path("pyproject.toml"),
}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_ATTEMPT_RE = re.compile(r"^[0-9]{3}$")
_PREREGISTRATION_NAME_RE = re.compile(
    r"^flux1_dual_view_adapter_native_equivalence_v3_"
    r"source_attempt_(?P<source>[0-9]{3})_PREREGISTERED\.json$"
)
_RESULT_DIRECTORY_RE = re.compile(
    r"^flux1_dual_view_adapter_native_equivalence_v3_"
    r"source_attempt_(?P<source>[0-9]{3})_"
    r"execution_attempt_(?P<execution>[0-9]{3})$"
)
_CANONICAL_AWARE_TIMESTAMP_RE = re.compile(
    r"(?P<calendar_time>[0-9]{4}-[0-9]{2}-[0-9]{2}T"
    r"[0-9]{2}:[0-9]{2}:[0-9]{2})"
    r"(?:\.(?P<fraction>[0-9]{1,6}))?"
    r"(?P<timezone>Z|[+-](?:[01][0-9]|2[0-3]):[0-5][0-9]|"
    r"(?P<basic_offset>[+-](?:[01][0-9]|2[0-3])[0-5][0-9]))"
)

PROTOCOL_INPUT_BINDING_ROLES = frozenset(
    {
        "native_equivalence_admission_validator",
        "native_equivalence_preregistration",
        "native_equivalence_preregistration_sidecar",
        "native_equivalence_result",
        "native_equivalence_result_sidecar",
        "native_equivalence_terminal_receipt",
        "native_equivalence_terminal_receipt_sidecar",
        "native_equivalence_receipt",
        "native_equivalence_receipt_sidecar",
    }
)


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256_bytes(encoded)


def parse_and_validate_snapshot_scheduler_config(raw: Any) -> dict[str, Any]:
    """Authenticate and parse the exact FLUX snapshot scheduler JSON bytes."""

    if (
        type(raw) is not bytes
        or len(raw) != SNAPSHOT_SCHEDULER_CONFIG_SIZE_BYTES
        or sha256_bytes(raw) != SNAPSHOT_SCHEDULER_CONFIG_SHA256
    ):
        raise ValueError("Pinned snapshot scheduler_config.json bytes drifted.")
    try:
        parsed = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("Pinned snapshot scheduler_config.json is not exact UTF-8 JSON.") from exc
    if (
        not isinstance(parsed, dict)
        or len(parsed) != 9
        or parsed != EXPECTED_SNAPSHOT_SCHEDULER_CONFIG
        or canonical_sha256(parsed) != SNAPSHOT_SCHEDULER_CONFIG_PARSED_SHA256
    ):
        raise ValueError("Pinned snapshot scheduler_config.json semantic mapping drifted.")
    return parsed


def normalize_and_validate_scheduler_config(raw: Any) -> dict[str, Any]:
    """Validate the pinned scheduler semantics and return one canonical record.

    At the pinned Diffusers commit, ``register_to_config`` constructs
    ``_use_default_values`` with ``list(set(...))`` and ``extract_init_dict``
    consumes that list only through membership checks. Its order is therefore
    process/hash-seed dependent and has no scheduler semantics. This is the
    only field normalized here; its type, unique membership, and cardinality
    remain exact, and every other field remains byte-semantically pinned by
    canonical SHA-256 after normalization.
    """

    if not isinstance(raw, Mapping) or set(raw) != set(EXPECTED_SCHEDULER_CONFIG):
        raise ValueError("Pinned FLUX.1 scheduler config field coverage drifted.")
    defaults = raw.get("_use_default_values")
    expected_defaults = EXPECTED_SCHEDULER_DEFAULT_VALUE_KEYS
    if (
        type(defaults) is not list
        or len(defaults) != len(expected_defaults)
        or any(type(key) is not str for key in defaults)
        or len(set(defaults)) != len(defaults)
        or set(defaults) != set(expected_defaults)
    ):
        raise ValueError("Pinned FLUX.1 scheduler default-value membership drifted.")
    normalized = deepcopy(dict(raw))
    normalized["_use_default_values"] = list(expected_defaults)
    try:
        digest = canonical_sha256(normalized)
    except (TypeError, ValueError) as exc:
        raise ValueError("Pinned FLUX.1 scheduler config is not canonical JSON.") from exc
    if normalized != EXPECTED_SCHEDULER_CONFIG or digest != EXPECTED_SCHEDULER_CONFIG_SHA256:
        raise ValueError("Pinned FLUX.1 scheduler semantic config drifted.")
    return normalized


def document_sha256(payload: Mapping[str, Any]) -> str:
    normalized = dict(payload)
    normalized.pop("document_sha256", None)
    return canonical_sha256(normalized)


def _aware_timestamp(value: Any, label: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty ISO-8601 timestamp.")
    match = _CANONICAL_AWARE_TIMESTAMP_RE.fullmatch(value)
    if match is None:
        raise ValueError(f"{label} must be a valid canonical ISO-8601 timestamp.")
    timezone_text = match.group("timezone")
    if timezone_text == "Z":
        normalized_timezone = "+00:00"
    elif (basic_offset := match.group("basic_offset")) is not None:
        normalized_timezone = f"{basic_offset[:3]}:{basic_offset[3:]}"
    else:
        normalized_timezone = timezone_text
    try:
        parsed = datetime.fromisoformat(f"{match.group('calendar_time')}{normalized_timezone}")
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid ISO-8601 timestamp.") from exc
    fraction = match.group("fraction")
    if fraction is not None:
        parsed = parsed.replace(microsecond=int(fraction.ljust(6, "0")))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include an explicit timezone.")
    return parsed


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be finite and nonnegative.")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be finite and nonnegative.") from exc
    if not math.isfinite(numeric) or numeric < 0:
        raise ValueError(f"{label} must be finite and nonnegative.")
    return numeric


def _read_regular_bytes(
    path: Path,
    label: str,
    *,
    require_readonly: bool,
    trusted_root: Path | None = None,
) -> bytes:
    """Read one canonical inode with no-follow and complete race checks."""

    expanded = path.expanduser()
    if not expanded.is_absolute() or ".." in expanded.parts:
        raise ValueError(f"{label} path must be absolute without parent traversal: {path}")
    path = Path(os.path.abspath(os.fspath(expanded)))
    if trusted_root is None:
        root = Path(path.anchor)
    else:
        root = trusted_root.expanduser().resolve(strict=True)
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} path escapes its trusted root: {path}") from exc
    if not relative.parts:
        raise ValueError(f"{label} must name a file below its trusted root.")

    component_snapshots: list[tuple[Path, int, int, int]] = []
    current_path = root
    for index, part in enumerate(relative.parts):
        current_path = current_path / part
        try:
            component = os.lstat(current_path)
        except OSError as exc:
            raise ValueError(f"Cannot lstat {label} component {current_path}: {exc}") from exc
        if stat.S_ISLNK(component.st_mode):
            raise ValueError(f"{label} path contains a symlink component: {current_path}")
        is_leaf = index == len(relative.parts) - 1
        if is_leaf and not stat.S_ISREG(component.st_mode):
            raise ValueError(f"{label} must be a regular file: {current_path}")
        if not is_leaf and not stat.S_ISDIR(component.st_mode):
            raise ValueError(f"{label} has a non-directory parent: {current_path}")
        component_snapshots.append(
            (current_path, component.st_dev, component.st_ino, component.st_mode)
        )
    leaf_before = os.lstat(path)
    if require_readonly and (leaf_before.st_mode & 0o222 or leaf_before.st_nlink != 1):
        raise ValueError(f"{label} must be read-only and singly linked: {path}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Cannot open {label} {path}: {exc}") from exc
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode) or (
            opened_before.st_dev,
            opened_before.st_ino,
        ) != (leaf_before.st_dev, leaf_before.st_ino):
            raise ValueError(f"{label} must be a regular file: {path}")
        if require_readonly and (opened_before.st_mode & 0o222 or opened_before.st_nlink != 1):
            raise ValueError(f"{label} must be read-only and singly linked: {path}")
        if opened_before.st_size <= 0:
            raise ValueError(f"{label} must not be empty: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_nlink",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if any(
        getattr(opened_before, field) != getattr(opened_after, field) for field in stable_fields
    ):
        raise ValueError(f"{label} inode changed during authentication: {path}")
    try:
        leaf_after = os.lstat(path)
    except OSError as exc:
        raise ValueError(f"{label} disappeared while authenticating: {path}") from exc
    if (
        any(getattr(opened_after, field) != getattr(leaf_after, field) for field in stable_fields)
        or len(raw) != opened_after.st_size
        or (require_readonly and (leaf_after.st_mode & 0o222 or leaf_after.st_nlink != 1))
    ):
        raise ValueError(f"{label} inode changed while authenticating: {path}")
    for component_path, expected_dev, expected_ino, expected_mode in component_snapshots:
        try:
            current = os.lstat(component_path)
        except OSError as exc:
            raise ValueError(
                f"{label} path component disappeared while authenticating: {component_path}"
            ) from exc
        if stat.S_ISLNK(current.st_mode) or (current.st_dev, current.st_ino, current.st_mode) != (
            expected_dev,
            expected_ino,
            expected_mode,
        ):
            raise ValueError(
                f"{label} path component changed while authenticating: {component_path}"
            )
    return raw


def read_authenticated_hf_snapshot_file(
    snapshot_root: str | Path,
    relative: str | Path,
    *,
    expected_blob_name: str | None = None,
) -> bytes:
    """Read one pinned Hugging Face snapshot symlink without trusting it.

    Hugging Face snapshots intentionally expose metadata as relative symlinks
    into the repository-local blobs directory. This narrow reader retains the
    canonical lexical snapshot path for the evidence receipt, permits only
    that exact topology, and authenticates the symlink, every parent directory,
    and the opened blob inode before and after the read. Callers may additionally
    pin the exact repository-local blob identity.
    """

    root = Path(snapshot_root).expanduser()
    if (
        not root.is_absolute()
        or ".." in root.parts
        or Path(os.path.abspath(os.fspath(root))) != EXPECTED_MODEL_SNAPSHOT_PATH
    ):
        raise ValueError("FLUX snapshot root is not the pinned canonical path.")
    relative_path = Path(relative)
    if (
        relative_path.is_absolute()
        or not relative_path.parts
        or ".." in relative_path.parts
        or "." in relative_path.parts
    ):
        raise ValueError("Hugging Face snapshot member must be a canonical relative path.")
    logical_path = root / relative_path
    repo_root = root.parent.parent
    expected_repo_name = "models--black-forest-labs--FLUX.1-dev"
    if (
        root.name != MODEL_REVISION
        or root.parent.name != "snapshots"
        or repo_root.name != expected_repo_name
    ):
        raise ValueError("FLUX snapshot repository/revision topology drifted.")

    parent_snapshots: dict[Path, tuple[int, int, int]] = {}

    def snapshot_directory_chain(directory: Path) -> None:
        current = Path(directory.anchor)
        for part in directory.parts[1:]:
            current = current / part
            if current in parent_snapshots:
                continue
            try:
                observed = os.lstat(current)
            except OSError as exc:
                raise ValueError(
                    f"Cannot authenticate Hugging Face cache parent: {current}."
                ) from exc
            if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
                raise ValueError(f"Hugging Face cache parent is not a real directory: {current}.")
            parent_snapshots[current] = (
                observed.st_dev,
                observed.st_ino,
                observed.st_mode,
            )

    snapshot_directory_chain(logical_path.parent)
    try:
        link_before = os.lstat(logical_path)
    except OSError as exc:
        raise ValueError(f"Pinned snapshot member is unavailable: {logical_path}.") from exc
    if not stat.S_ISLNK(link_before.st_mode):
        raise ValueError(f"Pinned snapshot member must be an HF blob symlink: {logical_path}.")
    link_target = os.readlink(logical_path)
    if Path(link_target).is_absolute():
        raise ValueError("Pinned snapshot member uses an absolute symlink target.")
    blob_path = Path(os.path.abspath(os.path.join(os.fspath(logical_path.parent), link_target)))
    blob_name = blob_path.name
    if (
        blob_path.parent != repo_root / "blobs"
        or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", blob_name) is None
    ):
        raise ValueError("Pinned snapshot member does not target one repository-local blob.")
    if expected_blob_name is not None and (
        type(expected_blob_name) is not str
        or re.fullmatch(r"(?:[0-9a-f]{40}|[0-9a-f]{64})", expected_blob_name) is None
        or blob_name != expected_blob_name
    ):
        raise ValueError("Pinned snapshot member blob identity drifted.")
    canonical_target = Path(os.path.relpath(blob_path, start=logical_path.parent))
    if Path(link_target) != canonical_target:
        raise ValueError("Pinned snapshot member symlink target is not canonical.")
    snapshot_directory_chain(blob_path.parent)
    try:
        blob_before = os.lstat(blob_path)
    except OSError as exc:
        raise ValueError(f"Pinned Hugging Face blob is unavailable: {blob_path}.") from exc
    if stat.S_ISLNK(blob_before.st_mode) or not stat.S_ISREG(blob_before.st_mode):
        raise ValueError("Pinned Hugging Face blob must be one real regular file.")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(blob_path, flags)
    except OSError as exc:
        raise ValueError(f"Cannot open pinned Hugging Face blob: {blob_path}.") from exc
    try:
        opened_before = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened_before.st_mode)
            or (opened_before.st_dev, opened_before.st_ino)
            != (blob_before.st_dev, blob_before.st_ino)
            or opened_before.st_size <= 0
        ):
            raise ValueError("Pinned Hugging Face blob changed before its read.")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 16 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(
            getattr(opened_before, field) != getattr(opened_after, field) for field in stable_fields
        )
        or len(raw) != opened_after.st_size
    ):
        raise ValueError("Pinned Hugging Face blob changed during its read.")
    blob_after = os.lstat(blob_path)
    if any(getattr(opened_after, field) != getattr(blob_after, field) for field in stable_fields):
        raise ValueError("Pinned Hugging Face blob path changed during its read.")
    link_after = os.lstat(logical_path)
    link_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(getattr(link_before, field) != getattr(link_after, field) for field in link_fields)
        or not stat.S_ISLNK(link_after.st_mode)
        or os.readlink(logical_path) != link_target
    ):
        raise ValueError("Pinned snapshot symlink changed during its authenticated read.")
    for parent, expected in parent_snapshots.items():
        try:
            current = os.lstat(parent)
        except OSError as exc:
            raise ValueError(
                f"Hugging Face cache parent disappeared during read: {parent}."
            ) from exc
        if (
            stat.S_ISLNK(current.st_mode)
            or not stat.S_ISDIR(current.st_mode)
            or (current.st_dev, current.st_ino, current.st_mode) != expected
        ):
            raise ValueError(f"Hugging Face cache parent changed during read: {parent}.")
    return raw


def _read_json(
    path: Path,
    label: str,
    *,
    require_readonly: bool,
    trusted_root: Path | None = None,
) -> tuple[dict[str, Any], bytes]:
    raw = _read_regular_bytes(
        path,
        label,
        require_readonly=require_readonly,
        trusted_root=trusted_root,
    )
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must contain one UTF-8 JSON object: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain one JSON object: {path}")
    return payload, raw


def _validate_sidecar(
    *,
    payload_path: Path,
    payload_raw: bytes,
    sidecar_path: Path,
    label: str,
    trusted_root: Path | None = None,
) -> dict[str, Any]:
    raw = _read_regular_bytes(
        sidecar_path,
        label,
        require_readonly=True,
        trusted_root=trusted_root,
    )
    expected_path = Path(f"{payload_path}.sha256")
    expected = f"{sha256_bytes(payload_raw)}  {payload_path.name}\n".encode("utf-8")
    if sidecar_path != expected_path or raw != expected:
        raise ValueError(f"{label} does not exactly authenticate {payload_path.name}.")
    return {
        "path": str(sidecar_path),
        "sha256": sha256_bytes(raw),
        "size_bytes": len(raw),
    }


def _artifact_binding(path: Path, raw: bytes) -> dict[str, Any]:
    return {"path": str(path), "sha256": sha256_bytes(raw), "size_bytes": len(raw)}


def expected_execution_contract() -> dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pipeline_class": PIPELINE_CLASS,
        "adapter_key": ADAPTER_KEY,
        "prompt_ids": list(PROMPT_IDS),
        "image_layouts": IMAGE_LAYOUTS,
        "seed": SEED,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "guidance_scale": GUIDANCE_SCALE,
        "true_cfg_scale": TRUE_CFG_SCALE,
        "num_images_per_prompt": 1,
        "dtype": "torch.bfloat16",
        "required_relation": "bitwise_identical",
        "comparisons_per_prompt": COMPARISONS_PER_PROMPT,
        "expected_total_comparisons": COMPARISON_COUNT,
        "expected_png_count": MEDIA_COUNT,
        "runner_registry_preflight_required": True,
        "independent_t5_sentinel_count": len(PROMPT_IDS),
    }


def normalize_attempt(value: Any, label: str) -> str:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive three-digit attempt.")
    if isinstance(value, int):
        normalized = f"{value:03d}"
    elif isinstance(value, str):
        normalized = value
    else:
        raise ValueError(f"{label} must be a positive three-digit attempt.")
    if _ATTEMPT_RE.fullmatch(normalized) is None or int(normalized) < 1:
        raise ValueError(f"{label} must be a positive three-digit attempt.")
    return normalized


def preregistration_filename(source_attempt: Any) -> str:
    return PREREGISTRATION_FILENAME_TEMPLATE.format(
        source_attempt=normalize_attempt(source_attempt, "source_attempt")
    )


def result_directory_name(source_attempt: Any, execution_attempt: Any) -> str:
    return RESULT_DIRECTORY_TEMPLATE.format(
        source_attempt=normalize_attempt(source_attempt, "source_attempt"),
        execution_attempt=normalize_attempt(execution_attempt, "execution_attempt"),
    )


def terminal_receipt_path(output_directory: str | Path) -> Path:
    """Return the one terminal scheduler receipt path for an execution attempt."""

    directory = Path(output_directory)
    parse_result_attempts(directory)
    return directory / TERMINAL_RECEIPT_FILENAME


def parse_preregistration_source_attempt(path: Path) -> str:
    match = _PREREGISTRATION_NAME_RE.fullmatch(path.name)
    if match is None or int(match.group("source")) < 1:
        raise ValueError("Equivalence preregistration filename lacks a valid source attempt.")
    return match.group("source")


def parse_result_attempts(path: Path) -> tuple[str, str]:
    match = _RESULT_DIRECTORY_RE.fullmatch(path.name)
    if match is None or int(match.group("source")) < 1 or int(match.group("execution")) < 1:
        raise ValueError("Equivalence result directory lacks canonical source/execution attempts.")
    return match.group("source"), match.group("execution")


def expected_attempt_lineage(
    *,
    source_attempt: Any,
    execution_attempt: Any,
    preregistration_path: Path,
    output_directory: Path,
) -> dict[str, Any]:
    source = normalize_attempt(source_attempt, "source_attempt")
    execution = normalize_attempt(execution_attempt, "execution_attempt")
    return {
        "schema_version": 1,
        "source_attempt": source,
        "execution_attempt": execution,
        "preregistration_path": str(preregistration_path),
        "output_directory": str(output_directory),
    }


def _validate_sealed_project_file(
    project_root: Path,
    relative: Path,
    expected_sha256: str,
    *,
    sidecar_relative: Path | None = None,
) -> tuple[Path, bytes]:
    path = project_root / relative
    raw = _read_regular_bytes(
        path,
        str(relative),
        require_readonly=True,
        trusted_root=project_root,
    )
    if sha256_bytes(raw) != expected_sha256:
        raise ValueError(f"Sealed project file digest drifted: {relative}")
    if sidecar_relative is not None:
        sidecar = project_root / sidecar_relative
        _validate_sidecar(
            payload_path=path,
            payload_raw=raw,
            sidecar_path=sidecar,
            label=f"{relative} sidecar",
            trusted_root=project_root,
        )
    return path, raw


def _load_prompt_contract(
    project_root: Path,
) -> tuple[Path, dict[str, dict[str, Any]], dict[str, str]]:
    path, raw = _validate_sealed_project_file(
        project_root,
        PROMPT_CONTRACT_RELATIVE,
        PROMPT_CONTRACT_SHA256,
        sidecar_relative=PROMPT_CONTRACT_SIDECAR_RELATIVE,
    )
    payload = yaml.safe_load(raw.decode("utf-8"))
    rows = payload.get("prompts") if isinstance(payload, Mapping) else None
    if (
        not isinstance(payload, Mapping)
        or payload.get("schema_version") != 1
        or payload.get("contract_id") != "flux1_common_seed_source_prompts_v3"
        or payload.get("status") != "preregistered_before_generation"
        or not isinstance(rows, list)
        or [row.get("prompt_id") for row in rows if isinstance(row, Mapping)] != list(PROMPT_IDS)
    ):
        raise ValueError("Sealed v3 source-prompt contract identity/order drifted.")
    tokenizer_files = payload.get("tokenizer_file_sha256")
    if not isinstance(tokenizer_files, Mapping) or not tokenizer_files:
        raise ValueError("Sealed v3 source-prompt contract lacks tokenizer fingerprints.")
    normalized_tokenizer_files: dict[str, str] = {}
    for relative, digest in tokenizer_files.items():
        if (
            not isinstance(relative, str)
            or not relative.startswith(("tokenizer/", "tokenizer_2/"))
            or Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or _SHA256_RE.fullmatch(str(digest)) is None
        ):
            raise ValueError("Sealed v3 tokenizer fingerprint schema drifted.")
        normalized_tokenizer_files[relative] = str(digest)
    normalized: dict[str, dict[str, Any]] = {}
    for prompt_id, row in zip(PROMPT_IDS, rows, strict=True):
        if not isinstance(row, Mapping):
            raise ValueError(f"Prompt contract row {prompt_id} is not a mapping.")
        required = {
            "clip_prompt",
            "clip_string_sha256",
            "clip_token_count",
            "clip_token_ids_sha256",
            "t5_prompt_2",
            "t5_string_sha256",
            "t5_token_count",
            "t5_token_ids_sha256",
        }
        if not required.issubset(row):
            raise ValueError(f"Prompt contract row {prompt_id} lacks positive-view fields.")
        clip = str(row["clip_prompt"])
        t5 = str(row["t5_prompt_2"])
        if (
            not clip.strip()
            or not t5.strip()
            or clip == t5
            or row["clip_string_sha256"] != sha256_bytes(clip.encode("utf-8"))
            or row["t5_string_sha256"] != sha256_bytes(t5.encode("utf-8"))
            or isinstance(row["clip_token_count"], bool)
            or not isinstance(row["clip_token_count"], int)
            or not 1 <= row["clip_token_count"] <= 77
            or isinstance(row["t5_token_count"], bool)
            or not isinstance(row["t5_token_count"], int)
            or not 1 <= row["t5_token_count"] <= 512
            or _SHA256_RE.fullmatch(str(row["clip_token_ids_sha256"])) is None
            or _SHA256_RE.fullmatch(str(row["t5_token_ids_sha256"])) is None
        ):
            raise ValueError(f"Prompt contract row {prompt_id} fingerprints drifted.")
        normalized[prompt_id] = deepcopy(dict(row))
    return path, normalized, normalized_tokenizer_files


def _positive_plan(row: Mapping[str, Any]) -> dict[str, Any]:
    positive_keys = (
        "clip_prompt",
        "t5_prompt_2",
        "clip_string_sha256",
        "clip_token_count",
        "clip_token_ids_sha256",
        "t5_string_sha256",
        "t5_token_count",
        "t5_token_ids_sha256",
    )
    return {
        "schema_version": 1,
        "positive": {key: deepcopy(row[key]) for key in positive_keys},
        "negative": None,
    }


def _validate_live_file_binding(
    record: Any,
    *,
    expected_path: Path,
    label: str,
    require_readonly: bool = False,
    trusted_root: Path | None = None,
) -> dict[str, str]:
    if not isinstance(record, Mapping) or set(record) != {"path", "sha256"}:
        raise ValueError(f"{label} file binding schema drifted.")
    path = Path(str(record.get("path", "")))
    if path != expected_path:
        raise ValueError(f"{label} file path drifted.")
    raw = _read_regular_bytes(
        path,
        label,
        require_readonly=require_readonly,
        trusted_root=trusted_root,
    )
    actual = sha256_bytes(raw)
    if record.get("sha256") != actual:
        raise ValueError(f"{label} file digest drifted.")
    return {"path": str(path), "sha256": actual}


def _immutable_json_binding(
    path: Path, *, audit_root: Path, label: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    payload, raw = _read_json(
        path,
        label,
        require_readonly=True,
        trusted_root=audit_root,
    )
    sidecar_path = Path(f"{path}.sha256")
    sidecar = _validate_sidecar(
        payload_path=path,
        payload_raw=raw,
        sidecar_path=sidecar_path,
        label=f"{label} sidecar",
        trusted_root=audit_root,
    )
    return payload, {
        "path": str(path),
        "sha256": sha256_bytes(raw),
        "document_sha256": canonical_sha256(payload),
        "sidecar_path": sidecar["path"],
        "sidecar_sha256": sidecar["sha256"],
    }


def _validate_preregistration_payload_schema(
    payload: Any, *, source_attempt: str
) -> tuple[dict[str, dict[str, str]], datetime]:
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
    if (
        not isinstance(payload, Mapping)
        or set(payload) != expected_top
        or payload.get("schema_version") != 1
        or payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "preregistered_before_execution"
        or payload.get("source_attempt") != source_attempt
        or payload.get("execution") != expected_execution_contract()
    ):
        raise ValueError("Equivalence preregistration identity/execution drifted.")
    created_at = _aware_timestamp(payload.get("created_at_utc"), "preregistration created_at_utc")
    raw_files = payload.get("files")
    if not isinstance(raw_files, Mapping) or set(raw_files) != set(
        REQUIRED_PREREGISTRATION_FILE_PATHS
    ):
        raise ValueError("Equivalence preregistration file-role coverage drifted.")
    files: dict[str, dict[str, str]] = {}
    for role, relative in REQUIRED_PREREGISTRATION_FILE_PATHS.items():
        record = raw_files.get(role)
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or Path(str(record.get("path", ""))) != relative
            or _SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None
        ):
            raise ValueError(f"Preregistered file role {role!r} path/schema drifted.")
        files[role] = {
            "path": str(record["path"]),
            "sha256": str(record["sha256"]),
        }
    return files, created_at


def _validate_source004_quarantine_lineage(
    raw: Any,
    *,
    current_files: Mapping[str, Any],
    current_created_at: datetime,
    project_root: Path,
) -> dict[str, Any]:
    """Validate the only passed-but-unaccepted predecessor exception."""

    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "recovery",
        "predecessor",
        "repair_reason",
        "changed_roles",
    }:
        raise ValueError("Source-004 quarantine lineage schema drifted.")
    predecessor = raw.get("predecessor")
    if (
        raw.get("schema_version") != 2
        or raw.get("recovery") != "source003_quarantine_to_fresh_source004_v1"
        or raw.get("repair_reason") != SOURCE004_QUARANTINE_REPAIR_REASON
        or not isinstance(predecessor, Mapping)
        or set(predecessor)
        != {
            "source_attempt",
            "preregistration",
            "result",
            "terminal_receipt",
            "acceptance_receipt",
            "quarantine_authorization",
            "quarantine_terminal_receipt",
        }
        or predecessor.get("source_attempt") != QUARANTINE_SOURCE_ATTEMPT
        or predecessor.get("terminal_receipt") is not None
        or predecessor.get("acceptance_receipt") is not None
    ):
        raise ValueError("Source-004 quarantine lineage identity/disposition drifted.")
    quarantine = validate_source003_quarantine_terminal(
        project_root=project_root,
        require_successor_absent=False,
        require_live_repair_sources=False,
    )
    preregistration_binding = {
        key: quarantine["authorization"]["preregistration"][key]
        for key in ("path", "sha256", "document_sha256", "sidecar_path", "sidecar_sha256")
    }
    result_binding = deepcopy(dict(quarantine["authorization"]["result"]))
    expected_predecessor = {
        "source_attempt": QUARANTINE_SOURCE_ATTEMPT,
        "preregistration": preregistration_binding,
        "result": result_binding,
        "terminal_receipt": None,
        "acceptance_receipt": None,
        "quarantine_authorization": deepcopy(dict(quarantine["authorization"]["authorization"])),
        "quarantine_terminal_receipt": deepcopy(dict(quarantine["receipt"])),
    }
    if predecessor != expected_predecessor:
        raise ValueError("Source-004 quarantine predecessor binding drifted.")
    if quarantine["prospective_file_ledger"] != current_files:
        raise ValueError("Source-004 files differ from the quarantine terminal commitment.")
    expected_changes = quarantine["payload"]["source_transition"]["changed_roles"]
    if raw.get("changed_roles") != expected_changes:
        raise ValueError("Source-004 quarantine changed-role ledger drifted.")

    predecessor_payload = quarantine["authorization"]["preregistration_payload"]
    predecessor_files, predecessor_created = _validate_preregistration_payload_schema(
        predecessor_payload,
        source_attempt=QUARANTINE_SOURCE_ATTEMPT,
    )
    _validate_source_attempt_lineage(
        predecessor_payload.get("lineage"),
        source_attempt=QUARANTINE_SOURCE_ATTEMPT,
        current_files=predecessor_files,
        current_created_at=predecessor_created,
        project_root=project_root,
    )
    result_payload = quarantine["authorization"]["result_payload"]
    result_started = _aware_timestamp(result_payload.get("started_at"), "source-003 result start")
    result_ended = _aware_timestamp(result_payload.get("ended_at"), "source-003 result end")
    authorized = _aware_timestamp(
        quarantine["authorization"]["payload"]["authorized_at_utc"],
        "source-003 quarantine authorization time",
    )
    quarantined = _aware_timestamp(
        quarantine["payload"]["recorded_at_utc"],
        "source-003 quarantine terminal time",
    )
    if not (
        predecessor_created
        <= result_started
        <= result_ended
        <= authorized
        <= quarantined
        <= current_created_at
    ):
        raise ValueError("Source-004 quarantine timestamp lineage drifted.")
    return deepcopy(dict(raw))


def _validate_source_attempt_lineage(
    raw: Any,
    *,
    source_attempt: str,
    current_files: Mapping[str, Any],
    current_created_at: datetime,
    project_root: Path,
) -> dict[str, Any]:
    audit_root = (project_root / AUDIT_ROOT_RELATIVE).resolve()
    if source_attempt == "001":
        expected = {
            "schema_version": 1,
            "predecessor": None,
            "repair_reason": None,
            "changed_roles": [],
        }
        if raw != expected:
            raise ValueError("Initial source-attempt lineage must have no predecessor/repair.")
        return deepcopy(expected)
    if source_attempt == QUARANTINE_SUCCESSOR_SOURCE_ATTEMPT:
        return _validate_source004_quarantine_lineage(
            raw,
            current_files=current_files,
            current_created_at=current_created_at,
            project_root=project_root,
        )
    if not isinstance(raw, Mapping) or set(raw) != {
        "schema_version",
        "predecessor",
        "repair_reason",
        "changed_roles",
    }:
        raise ValueError("Repaired source-attempt lineage schema drifted.")
    previous_attempt = f"{int(source_attempt) - 1:03d}"
    predecessor = raw.get("predecessor")
    repair_reason = raw.get("repair_reason")
    changed_roles = raw.get("changed_roles")
    if (
        raw.get("schema_version") != 1
        or not isinstance(predecessor, Mapping)
        or set(predecessor)
        != {
            "source_attempt",
            "preregistration",
            "result",
            "terminal_receipt",
            "acceptance_receipt",
        }
        or predecessor.get("source_attempt") != previous_attempt
        or not isinstance(repair_reason, str)
        or not repair_reason.strip()
        or not isinstance(changed_roles, list)
        or not changed_roles
    ):
        raise ValueError("Repaired source attempt lacks exact predecessor/repair evidence.")

    predecessor_prereg_path = audit_root / preregistration_filename(previous_attempt)
    predecessor_payload, predecessor_prereg_binding = _immutable_json_binding(
        predecessor_prereg_path,
        audit_root=audit_root,
        label="predecessor equivalence preregistration",
    )
    previous_files, predecessor_created_at = _validate_preregistration_payload_schema(
        predecessor_payload, source_attempt=previous_attempt
    )
    if predecessor.get("preregistration") != predecessor_prereg_binding:
        raise ValueError("Predecessor equivalence preregistration binding drifted.")
    _validate_source_attempt_lineage(
        predecessor_payload.get("lineage"),
        source_attempt=previous_attempt,
        current_files=previous_files,
        current_created_at=predecessor_created_at,
        project_root=project_root,
    )

    result_binding = predecessor.get("result")
    if not isinstance(result_binding, Mapping) or set(result_binding) != {
        "path",
        "sha256",
        "sidecar_path",
        "sidecar_sha256",
        "status",
        "source_attempt",
        "execution_attempt",
    }:
        raise ValueError("Predecessor equivalence result binding schema drifted.")
    predecessor_result_path = Path(str(result_binding.get("path", "")))
    result_source, result_execution = parse_result_attempts(predecessor_result_path.parent)
    predecessor_result, immutable_result = _immutable_json_binding(
        predecessor_result_path,
        audit_root=audit_root,
        label="predecessor equivalence result",
    )
    expected_result_binding = {
        "path": immutable_result["path"],
        "sha256": immutable_result["sha256"],
        "sidecar_path": immutable_result["sidecar_path"],
        "sidecar_sha256": immutable_result["sidecar_sha256"],
        "status": predecessor_result.get("status"),
        "source_attempt": result_source,
        "execution_attempt": result_execution,
    }
    expected_result_path = (
        audit_root / result_directory_name(previous_attempt, result_execution) / RESULT_FILENAME
    )
    expected_attempt = expected_attempt_lineage(
        source_attempt=previous_attempt,
        execution_attempt=result_execution,
        preregistration_path=predecessor_prereg_path,
        output_directory=predecessor_result_path.parent,
    )
    expected_preregistration_binding = {
        "path": predecessor_prereg_binding["path"],
        "raw_sha256": predecessor_prereg_binding["sha256"],
        "document_sha256": predecessor_prereg_binding["document_sha256"],
        "sidecar_path": predecessor_prereg_binding["sidecar_path"],
        "sidecar_sha256": predecessor_prereg_binding["sidecar_sha256"],
    }
    if (
        predecessor_result_path != expected_result_path
        or result_source != previous_attempt
        or predecessor_result.get("protocol_id") != PROTOCOL_ID
        or predecessor_result.get("status") not in {"error", "failed", "passed"}
        or predecessor_result.get("gate") != GATE
        or predecessor_result.get("schema_version") != SCHEMA_VERSION
        or predecessor_result.get("attempt_lineage") != expected_attempt
        or predecessor_result.get("preregistration_binding") != expected_preregistration_binding
        or result_binding != expected_result_binding
    ):
        raise ValueError("Predecessor equivalence result identity/digest drifted.")
    exact_execution = validate_terminal_execution_result(
        result_path=predecessor_result_path,
        project_root=project_root,
    )
    exact_result_binding = {
        "path": exact_execution["result"]["path"],
        "sha256": exact_execution["result"]["sha256"],
        "sidecar_path": exact_execution["result"]["sidecar_path"],
        "sidecar_sha256": exact_execution["result"]["sidecar_sha256"],
        "status": exact_execution["result"]["status"],
        "source_attempt": result_source,
        "execution_attempt": result_execution,
    }
    attempts = _attempt_directories_for_source(
        audit_root=audit_root, source_attempt=previous_attempt
    )
    required_attempts = {f"{index:03d}" for index in range(1, int(result_execution) + 1)}
    if result_binding != exact_result_binding or set(attempts) != required_attempts:
        raise ValueError(
            "Source repair does not cite the latest contiguous prior-source execution."
        )
    terminal_admission = validate_terminal_receipt(
        result_path=predecessor_result_path,
        project_root=project_root,
        require_latest=False,
    )
    terminal_binding = _terminal_receipt_binding(terminal_admission)
    if predecessor.get("terminal_receipt") != terminal_binding:
        raise ValueError("Source repair terminal-receipt binding drifted.")
    terminal_recorded = _aware_timestamp(
        terminal_admission["payload"]["recorded_at_utc"],
        "predecessor terminal receipt recorded_at_utc",
    )
    if predecessor_result.get("status") == "passed":
        acceptance_admission = validate_acceptance_receipt(
            result_path=predecessor_result_path,
            project_root=project_root,
            require_latest=False,
            require_live_preregistration_sources=False,
        )
        acceptance_binding: dict[str, Any] | None = _acceptance_receipt_binding(
            acceptance_admission
        )
        acceptance_recorded = _aware_timestamp(
            acceptance_admission["payload"]["accepted_at_utc"],
            "predecessor acceptance receipt accepted_at_utc",
        )
    else:
        acceptance_binding = None
        acceptance_recorded = terminal_recorded
    if predecessor.get("acceptance_receipt") != acceptance_binding:
        raise ValueError("Source repair acceptance-receipt binding drifted.")
    predecessor_started = _aware_timestamp(
        predecessor_result.get("started_at"), "predecessor result started_at"
    )
    predecessor_ended = _aware_timestamp(
        predecessor_result.get("ended_at"), "predecessor result ended_at"
    )
    if not (
        predecessor_created_at
        <= predecessor_started
        <= predecessor_ended
        <= terminal_recorded
        <= acceptance_recorded
        <= current_created_at
    ):
        raise ValueError("Source-attempt repair timestamp lineage is invalid.")

    expected_changes = [
        {
            "role": role,
            "previous_sha256": previous_files[role]["sha256"],
            "current_sha256": current_files[role]["sha256"],
        }
        for role in sorted(REQUIRED_PREREGISTRATION_FILE_PATHS)
        if previous_files[role]["sha256"] != current_files[role]["sha256"]
    ]
    if changed_roles != expected_changes:
        raise ValueError("Source-attempt changed-role digest ledger is incomplete or inaccurate.")
    return deepcopy(dict(raw))


def _validate_preregistration(
    raw: Any,
    *,
    project_root: Path,
    require_live_sources: bool = True,
) -> dict[str, Any]:
    expected_top = {
        "path",
        "raw_sha256",
        "document_sha256",
        "sidecar_path",
        "sidecar_sha256",
        "source_attempt",
        "lineage",
        "payload",
        "authenticated_files",
        "sealed_sources",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_top:
        raise ValueError("Equivalence environment preregistration schema drifted.")
    path = Path(str(raw.get("path", "")))
    audit_root = (project_root / AUDIT_ROOT_RELATIVE).resolve()
    source_attempt = parse_preregistration_source_attempt(path)
    if path != audit_root / preregistration_filename(source_attempt):
        raise ValueError("Equivalence preregistration path is noncanonical.")
    payload, payload_raw = _read_json(
        path,
        "equivalence preregistration",
        require_readonly=True,
        trusted_root=audit_root,
    )
    preregistration_sidecar_path = Path(str(raw.get("sidecar_path", "")))
    preregistration_sidecar = _validate_sidecar(
        payload_path=path,
        payload_raw=payload_raw,
        sidecar_path=preregistration_sidecar_path,
        label="equivalence preregistration sidecar",
        trusted_root=audit_root,
    )
    if (
        raw.get("raw_sha256") != sha256_bytes(payload_raw)
        or raw.get("document_sha256") != canonical_sha256(payload)
        or raw.get("sidecar_sha256") != preregistration_sidecar["sha256"]
        or raw.get("payload") != payload
    ):
        raise ValueError("Equivalence preregistration byte/document binding drifted.")
    files, created_at = _validate_preregistration_payload_schema(
        payload, source_attempt=source_attempt
    )
    if created_at > datetime.now(timezone.utc):
        raise ValueError("Equivalence preregistration timestamp is in the future.")
    authenticated = raw.get("authenticated_files")
    if not isinstance(authenticated, Mapping) or set(authenticated) != set(
        REQUIRED_PREREGISTRATION_FILE_PATHS
    ):
        raise ValueError("Equivalence preregistration file-role coverage drifted.")
    normalized_files: dict[str, dict[str, str]] = {}
    for role, relative in REQUIRED_PREREGISTRATION_FILE_PATHS.items():
        declaration = files[role]
        live = project_root / relative
        if require_live_sources:
            normalized = _validate_live_file_binding(
                authenticated[role],
                expected_path=live,
                label=f"preregistered {role}",
                trusted_root=project_root,
            )
        else:
            historical = authenticated[role]
            if (
                not isinstance(historical, Mapping)
                or set(historical) != {"path", "sha256"}
                or Path(str(historical.get("path", ""))) != live
                or historical.get("sha256") != declaration["sha256"]
            ):
                raise ValueError(f"Historical preregistered file role {role!r} binding drifted.")
            normalized = {
                "path": str(live),
                "sha256": str(historical["sha256"]),
            }
        if declaration.get("sha256") != normalized["sha256"]:
            raise ValueError(f"Preregistered file role {role!r} declaration drifted.")
        normalized_files[role] = normalized
    if normalized_files["legacy_flux_adapter"]["sha256"] != LEGACY_FLUX_ADAPTER_SHA256:
        raise ValueError("The byte-sealed legacy FLUX adapter drifted.")
    lineage = _validate_source_attempt_lineage(
        payload.get("lineage"),
        source_attempt=source_attempt,
        current_files=files,
        current_created_at=created_at,
        project_root=project_root,
    )
    if raw.get("source_attempt") != source_attempt or raw.get("lineage") != lineage:
        raise ValueError("Embedded preregistration attempt/lineage binding drifted.")

    sealed = raw.get("sealed_sources")
    expected_sealed: dict[str, dict[str, str]] = {}
    for artifact_role, sidecar_role, expected_sha in (
        ("prompt_contract", "prompt_contract_sidecar", PROMPT_CONTRACT_SHA256),
        ("prompt_protocol", "prompt_protocol_sidecar", PROMPT_PROTOCOL_SHA256),
        ("model_config", "model_config_sidecar", MODEL_CONFIG_SHA256),
    ):
        artifact = normalized_files[artifact_role]
        sidecar = normalized_files[sidecar_role]
        if artifact["sha256"] != expected_sha:
            raise ValueError(f"Sealed preregistration source {artifact_role!r} drifted.")
        if require_live_sources:
            sidecar_path = Path(sidecar["path"])
            sidecar_raw = _read_regular_bytes(
                sidecar_path,
                f"sealed {sidecar_role}",
                require_readonly=True,
                trusted_root=project_root,
            )
            if sidecar_raw != (f"{expected_sha}  {Path(artifact['path']).name}\n".encode("utf-8")):
                raise ValueError(f"Sealed source sidecar {sidecar_role!r} drifted.")
        expected_sealed[artifact_role] = {
            "path": artifact["path"],
            "sha256": expected_sha,
            "sidecar_path": sidecar["path"],
            "sidecar_sha256": sidecar["sha256"],
        }
    if sealed != expected_sealed:
        raise ValueError("Equivalence preregistration sealed-source receipts drifted.")
    return {
        "path": str(path),
        "raw_sha256": sha256_bytes(payload_raw),
        "document_sha256": canonical_sha256(payload),
        "sidecar_path": preregistration_sidecar["path"],
        "sidecar_sha256": preregistration_sidecar["sha256"],
        "source_attempt": source_attempt,
        "lineage": lineage,
        "payload": payload,
        "authenticated_files": normalized_files,
        "sealed_sources": expected_sealed,
    }


def _validate_tensor_summary(raw: Any, label: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "shape",
        "dtype",
        "sha256",
        "finite",
        "min",
        "max",
        "mean",
    }:
        raise ValueError(f"{label} tensor-summary schema drifted.")
    shape = raw.get("shape")
    if (
        not isinstance(shape, list)
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in shape
        )
        or not isinstance(raw.get("dtype"), str)
        or not raw.get("dtype")
        or _SHA256_RE.fullmatch(str(raw.get("sha256", ""))) is None
        or raw.get("finite") is not True
    ):
        raise ValueError(f"{label} tensor summary is malformed or non-finite.")
    for field in ("min", "max", "mean"):
        value = raw.get(field)
        if value is not None and (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(f"{label}.{field} must be finite or null.")
    return deepcopy(dict(raw))


def expected_comparison_ids(prompt_id: str) -> list[str]:
    identifiers = [
        f"{prompt_id}.conditioning.prompt_embeds",
        f"{prompt_id}.conditioning.pooled_prompt_embeds",
        f"{prompt_id}.conditioning.text_ids",
        f"{prompt_id}.latents.initial_packed",
        f"{prompt_id}.latents.image_ids",
        f"{prompt_id}.schedule.sigmas",
        f"{prompt_id}.schedule.timesteps",
        f"{prompt_id}.transformer.embedded_guidance",
    ]
    for step in range(NUM_INFERENCE_STEPS):
        identifiers.extend(
            [
                f"{prompt_id}.step.{step:02d}.normalized_timestep",
                f"{prompt_id}.step.{step:02d}.prediction",
                f"{prompt_id}.step.{step:02d}.post_scheduler_latents",
            ]
        )
    identifiers.extend([f"{prompt_id}.decode.vae_output", f"{prompt_id}.decode.rgb_uint8"])
    if len(identifiers) != COMPARISONS_PER_PROMPT:
        raise AssertionError("Internal equivalence comparison topology drifted.")
    return identifiers


def independent_t5_sentinel_comparison_id(prompt_id: str) -> str:
    if prompt_id not in PROMPT_IDS:
        raise ValueError(f"Unknown equivalence prompt ID: {prompt_id!r}.")
    return f"{prompt_id}.sentinel.independent_t5_vs_mirrored_clip_text.prompt_embeds"


def _validate_comparisons(raw: Any, prompt_id: str) -> list[dict[str, Any]]:
    if not isinstance(raw, list) or len(raw) != COMPARISONS_PER_PROMPT:
        raise ValueError(f"{prompt_id} comparison coverage is not exactly 94.")
    if [row.get("comparison_id") for row in raw if isinstance(row, Mapping)] != (
        expected_comparison_ids(prompt_id)
    ):
        raise ValueError(f"{prompt_id} comparison IDs/order/uniqueness drifted.")
    expected_fields = {
        "comparison_id",
        "passed",
        "required_relation",
        "same_shape",
        "same_dtype",
        "max_abs_difference",
        "mean_abs_difference",
        "native",
        "adapter",
    }
    normalized: list[dict[str, Any]] = []
    for row in raw:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError(f"{prompt_id} comparison schema drifted.")
        native = _validate_tensor_summary(row["native"], f"{row['comparison_id']}.native")
        adapter = _validate_tensor_summary(row["adapter"], f"{row['comparison_id']}.adapter")
        if (
            row.get("passed") is not True
            or row.get("required_relation") != "bitwise_identical"
            or row.get("same_shape") is not True
            or row.get("same_dtype") is not True
            or row.get("max_abs_difference") != 0.0
            or row.get("mean_abs_difference") != 0.0
            or native["shape"] != adapter["shape"]
            or native["dtype"] != adapter["dtype"]
            or native["sha256"] != adapter["sha256"]
        ):
            raise ValueError(f"Bitwise equivalence failed for {row.get('comparison_id')!r}.")
        normalized.append(deepcopy(dict(row)))
    return normalized


def _expected_conditioning_preflight(plan: Mapping[str, Any]) -> dict[str, Any]:
    positive = plan["positive"]
    return {
        "schema_version": 1,
        "status": "passed",
        "plan_sha256": canonical_sha256(plan),
        "require_no_primary_or_secondary_truncation": True,
        "token_ids_hash_encoding": "compact_json_integer_array_utf8",
        "views": [
            {
                "role": "positive.clip",
                "call_key": "prompt",
                "encoder": "CLIPTextModel",
                "text_utf8_sha256": positive["clip_string_sha256"],
                "token_count": positive["clip_token_count"],
                "token_ids_sha256": positive["clip_token_ids_sha256"],
                "maximum_tokens_including_special_tokens": 77,
                "truncated": False,
            },
            {
                "role": "positive.t5",
                "call_key": "prompt_2",
                "encoder": "T5EncoderModel",
                "text_utf8_sha256": positive["t5_string_sha256"],
                "token_count": positive["t5_token_count"],
                "token_ids_sha256": positive["t5_token_ids_sha256"],
                "maximum_tokens_including_special_tokens": 512,
                "truncated": False,
            },
        ],
    }


def _expected_conditioning_provenance(plan: Mapping[str, Any]) -> dict[str, Any]:
    positive = plan["positive"]
    clip = {
        "call_key": "prompt",
        "encoder": "CLIPTextModel",
        "text": positive["clip_prompt"],
        "utf8_sha256": positive["clip_string_sha256"],
    }
    t5 = {
        "call_key": "prompt_2",
        "encoder": "T5EncoderModel",
        "text": positive["t5_prompt_2"],
        "utf8_sha256": positive["t5_string_sha256"],
    }
    return {
        "schema_version": 1,
        "method": "flux1_registered_dual_prompt_views",
        "plan_sha256": canonical_sha256(plan),
        "positive": {
            "clip": {
                **clip,
                "frozen_token_count": positive["clip_token_count"],
                "frozen_token_ids_sha256": positive["clip_token_ids_sha256"],
            },
            "t5": {
                **t5,
                "frozen_token_count": positive["t5_token_count"],
                "frozen_token_ids_sha256": positive["t5_token_ids_sha256"],
            },
        },
        "negative": None,
        "runtime_preflight": _expected_conditioning_preflight(plan),
        "encode_calls": [
            {
                "sequence_index": 0,
                "role": "registered_positive",
                "registered_dual_view_applied": True,
                "clip": clip,
                "t5": t5,
            }
        ],
        "native_paired_calls": [],
    }


def _validate_token_id_record(
    raw: Any,
    *,
    expected_text_sha256: str | None,
    expected_count: int | None,
    expected_token_sha256: str | None,
    label: str,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "text_utf8_sha256",
        "token_count",
        "token_ids_sha256",
        "token_ids",
    }:
        raise ValueError(f"{label} token record schema drifted.")
    token_ids = raw.get("token_ids")
    if (
        not isinstance(token_ids, list)
        or not token_ids
        or any(isinstance(value, bool) or not isinstance(value, int) for value in token_ids)
        or raw.get("token_count") != len(token_ids)
        or raw.get("token_ids_sha256") != canonical_sha256(token_ids)
        or _SHA256_RE.fullmatch(str(raw.get("text_utf8_sha256", ""))) is None
    ):
        raise ValueError(f"{label} token record is internally inconsistent.")
    if (
        (expected_text_sha256 is not None and raw["text_utf8_sha256"] != expected_text_sha256)
        or (expected_count is not None and len(token_ids) != expected_count)
        or (expected_token_sha256 is not None and raw["token_ids_sha256"] != expected_token_sha256)
    ):
        raise ValueError(f"{label} token record differs from the sealed prompt contract.")
    return deepcopy(dict(raw))


def _validate_independent_t5_sentinel(
    raw: Any,
    *,
    prompt_id: str,
    row: Mapping[str, Any],
) -> dict[str, Any]:
    expected_fields = {
        "schema_version",
        "status",
        "full_t5_prompt_2",
        "mirrored_clip_text_as_t5_prompt_2",
        "token_ids_differ",
        "full_vs_mirrored_prompt_embeds",
        "native_full_vs_adapter_full_comparison_id",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != expected_fields
        or raw.get("schema_version") != 1
        or raw.get("status") != "passed"
    ):
        raise ValueError(f"{prompt_id} independent-T5 sentinel identity drifted.")
    full = _validate_token_id_record(
        raw["full_t5_prompt_2"],
        expected_text_sha256=row["t5_string_sha256"],
        expected_count=row["t5_token_count"],
        expected_token_sha256=row["t5_token_ids_sha256"],
        label=f"{prompt_id}.full_t5_prompt_2",
    )
    mirrored = _validate_token_id_record(
        raw["mirrored_clip_text_as_t5_prompt_2"],
        expected_text_sha256=row["clip_string_sha256"],
        expected_count=None,
        expected_token_sha256=None,
        label=f"{prompt_id}.mirrored_clip_as_t5",
    )
    if raw.get("token_ids_differ") is not True or full["token_ids"] == mirrored["token_ids"]:
        raise ValueError(f"{prompt_id} independent-T5 token sentinel did not distinguish views.")
    comparison = raw.get("full_vs_mirrored_prompt_embeds")
    expected_comparison_fields = {
        "comparison_id",
        "passed",
        "required_relation",
        "same_shape",
        "same_dtype",
        "bitwise_equal",
        "max_abs_difference",
        "mean_abs_difference",
        "full_independent_t5",
        "mirrored_clip_text_as_t5",
    }
    if not isinstance(comparison, Mapping) or set(comparison) != expected_comparison_fields:
        raise ValueError(f"{prompt_id} independent-T5 embedding comparison schema drifted.")
    full_summary = _validate_tensor_summary(
        comparison["full_independent_t5"], f"{prompt_id}.sentinel.full"
    )
    mirrored_summary = _validate_tensor_summary(
        comparison["mirrored_clip_text_as_t5"], f"{prompt_id}.sentinel.mirrored"
    )
    if (
        comparison.get("comparison_id") != independent_t5_sentinel_comparison_id(prompt_id)
        or comparison.get("passed") is not True
        or comparison.get("required_relation") != "not_bitwise_identical"
        or comparison.get("same_shape") is not True
        or comparison.get("same_dtype") is not True
        or comparison.get("bitwise_equal") is not False
        or _finite_nonnegative(
            comparison.get("max_abs_difference"), f"{prompt_id} sentinel max delta"
        )
        <= 0
        or full_summary["shape"] != mirrored_summary["shape"]
        or full_summary["dtype"] != mirrored_summary["dtype"]
        or full_summary["sha256"] == mirrored_summary["sha256"]
        or raw.get("native_full_vs_adapter_full_comparison_id")
        != f"{prompt_id}.conditioning.prompt_embeds"
    ):
        raise ValueError(f"{prompt_id} independent-T5 embedding sentinel failed.")
    _finite_nonnegative(comparison.get("mean_abs_difference"), f"{prompt_id} sentinel mean delta")
    return deepcopy(dict(raw))


def _recompute_runner_registry_preflight_row(
    *,
    project_root: Path,
    prompt_id: str,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Independently replay the producer's projector/validator/runner bridge."""

    from hierasafe_flow.benchmarks.finer_detailing_correction import (
        BENCHMARK_NAME as FINER_BENCHMARK_NAME,
        _runner_config as production_runner_config,
    )
    from hierasafe_flow.evaluation.flux1_dual_view_jobs_v3 import (
        MODE_PREVIEW,
        RUNNER_PREFLIGHT_PROJECTED_JOB_HASH_CONTRACT,
        _load_source_views,
        project_flux1_job_v3,
        runner_preflight_projected_job_sha256_v3,
        validate_flux1_job_v3,
    )
    from hierasafe_flow.generation.runner import _bind_flux_dual_view_conditioning

    sealed_source = _load_source_views(project_root)[prompt_id]
    for identity_field in (
        "clip_prompt",
        "t5_prompt_2",
        "negative_clip_prompt",
        "clip_string_sha256",
        "t5_string_sha256",
    ):
        if source.get(identity_field) != sealed_source.get(identity_field):
            raise ValueError(
                f"Runner-preflight prompt source drifted for {prompt_id}.{identity_field}."
            )
    source = sealed_source
    generation = {
        "task": "text_to_image",
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "height": IMAGE_LAYOUTS[prompt_id]["height"],
        "width": IMAGE_LAYOUTS[prompt_id]["width"],
        "guidance_scale": GUIDANCE_SCALE,
        "num_outputs_per_prompt": 1,
    }
    concept_role = {
        "01_sad_young_girl": "prompt1_concept_hierarchy",
        "02_angry_old_man": "prompt2_concept_hierarchy",
        "03_empty_outdoor_mall": "prompt3_concept_hierarchy",
    }[prompt_id]
    ephemeral_output = str(
        project_root / AUDIT_ROOT_RELATIVE / ".flux1-runner-preflight-recomputation" / prompt_id
    )
    synthetic_job = {
        "benchmark": FINER_BENCHMARK_NAME,
        "stage": "flux1_common_seed_source_ladder_v3",
        "variant": "01_baseline",
        "variant_spec": {"kind": "baseline"},
        "condition_id": f"{prompt_id}__flux1_dev__01_baseline__seed_00000000",
        "attempt": 1,
        "seed": SEED,
        "prompt_id": prompt_id,
        "prompt": str(source["clip_prompt"]),
        "negative_prompt": str(source["negative_clip_prompt"]),
        "model_name": "flux1_dev",
        "model_revision": MODEL_REVISION,
        "base_config": str(
            (project_root / REQUIRED_PREREGISTRATION_FILE_PATHS["base_config"]).resolve()
        ),
        "model_config": str((project_root / MODEL_CONFIG_RELATIVE).resolve()),
        "concept_tree": str(
            (project_root / REQUIRED_PREREGISTRATION_FILE_PATHS[concept_role]).resolve()
        ),
        "generation": generation,
        "runtime": {"device": "cuda", "dtype": "bfloat16"},
        "logging": {"tensorboard": False},
        "output": {
            "decode": True,
            "save_latents": False,
            "save_traces": False,
            "image_format": "png",
        },
        "output_dir": ephemeral_output,
        "launch_manifest_sha256": "0" * 64,
    }
    projected_job = project_flux1_job_v3(
        synthetic_job,
        project_root=project_root,
        mode=MODE_PREVIEW,
    )
    projection_validation = validate_flux1_job_v3(
        projected_job,
        project_root=project_root,
        mode=MODE_PREVIEW,
    )
    plan = _positive_plan(source)
    if (
        projected_job.get("prompt") != source["clip_prompt"]
        or projected_job.get("generation", {}).get("flux_dual_view_conditioning") != plan
        or projection_validation.get("mode") != MODE_PREVIEW
    ):
        raise ValueError(f"Runner-preflight projector recomputation drifted for {prompt_id}.")
    runner_config = production_runner_config(projected_job, project_root)
    if (
        runner_config.get("generation", {}).get("prompt") != source["clip_prompt"]
        or runner_config.get("generation", {}).get("flux_dual_view_conditioning") != plan
    ):
        raise ValueError(f"Runner-preflight production config drifted for {prompt_id}.")
    bound_model = _bind_flux_dual_view_conditioning(
        model_config=deepcopy(dict(runner_config["model"])),
        generation_config=deepcopy(dict(runner_config["generation"])),
    )
    return {
        "prompt_id": prompt_id,
        "generation_prompt_sha256": source["clip_string_sha256"],
        "dual_view_plan_sha256": canonical_sha256(plan),
        "registered_generation_sha256": canonical_sha256(generation),
        "projected_job_sha256": runner_preflight_projected_job_sha256_v3(
            projected_job,
            expected_output_dir=ephemeral_output,
        ),
        "projection_validation_sha256": canonical_sha256(projection_validation),
        "projection_mode": projection_validation["mode"],
        "projection_status": projection_validation["status"],
        "production_runner_generation_sha256": canonical_sha256(runner_config["generation"]),
        "bound_model_config_sha256": canonical_sha256(bound_model),
        "adapter_class": "FluxDualViewAdapter",
        "adapter_name": ADAPTER_KEY,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "exact_plan_bound": True,
        "generation_runner_constructor_checked": True,
        # Requiring the shared constant here makes import drift fail before a
        # result row can be admitted under a locally invented hash meaning.
        "_projected_job_hash_contract": RUNNER_PREFLIGHT_PROJECTED_JOB_HASH_CONTRACT,
    }


def _validate_runner_registry_preflight(
    raw: Any,
    *,
    project_root: Path,
    prompt_rows: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    expected_top = {
        "schema_version",
        "status",
        "route",
        "projected_job_hash_contract",
        "model_loading_performed",
        "config_source",
        "prompt_count",
        "rows",
    }
    from hierasafe_flow.evaluation.flux1_dual_view_jobs_v3 import (
        RUNNER_PREFLIGHT_PROJECTED_JOB_HASH_CONTRACT,
        RUNNER_PREFLIGHT_RECEIPT_SCHEMA_VERSION,
        RUNNER_PREFLIGHT_ROUTE,
    )

    model_config_path = (project_root / MODEL_CONFIG_RELATIVE).resolve()
    if (
        not isinstance(raw, Mapping)
        or set(raw) != expected_top
        or raw.get("schema_version") != RUNNER_PREFLIGHT_RECEIPT_SCHEMA_VERSION
        or raw.get("status") != "passed"
        or raw.get("route") != RUNNER_PREFLIGHT_ROUTE
        or raw.get("projected_job_hash_contract") != RUNNER_PREFLIGHT_PROJECTED_JOB_HASH_CONTRACT
        or raw.get("model_loading_performed") is not False
        or raw.get("config_source")
        != {"path": str(model_config_path), "sha256": MODEL_CONFIG_SHA256}
        or raw.get("prompt_count") != len(PROMPT_IDS)
    ):
        raise ValueError("Runner/registry construction preflight identity drifted.")
    rows = raw.get("rows")
    if (
        not isinstance(rows, list)
        or len(rows) != len(PROMPT_IDS)
        or [row.get("prompt_id") for row in rows if isinstance(row, Mapping)] != list(PROMPT_IDS)
    ):
        raise ValueError("Runner/registry preflight prompt coverage/order drifted.")
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
    for prompt_id, observed in zip(PROMPT_IDS, rows, strict=True):
        if not isinstance(observed, Mapping) or set(observed) != expected_row_fields:
            raise ValueError(f"Runner/registry preflight row {prompt_id} schema drifted.")
        expected = _recompute_runner_registry_preflight_row(
            project_root=project_root,
            prompt_id=prompt_id,
            source=prompt_rows[prompt_id],
        )
        contract = expected.pop("_projected_job_hash_contract")
        if contract != raw["projected_job_hash_contract"]:
            raise ValueError("Runner/registry projected-job hash contract drifted.")
        if observed != expected:
            raise ValueError(f"Runner/registry preflight row {prompt_id} drifted.")
    return deepcopy(dict(raw))


def _validate_environment_preflight(
    raw: Any,
    *,
    project_root: Path,
    preregistration_binding: Any,
    tokenizer_file_sha256: Mapping[str, str],
    require_live_preregistration_sources: bool = True,
) -> dict[str, Any]:
    if not isinstance(raw, Mapping) or set(raw) != {
        "gpu",
        "environment",
        "preregistration",
        "package_sources",
        "model_snapshot",
        "slurm",
    }:
        raise ValueError("Equivalence environment-preflight schema drifted.")
    gpu = raw.get("gpu")
    if (
        not isinstance(gpu, Mapping)
        or set(gpu) != {"name", "count", "compute_capability", "bfloat16_supported"}
        or not isinstance(gpu.get("name"), str)
        or "H100" not in gpu["name"].upper()
        or gpu.get("count") != 1
        or gpu.get("bfloat16_supported") is not True
        or not isinstance(gpu.get("compute_capability"), list)
        or len(gpu["compute_capability"]) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in gpu["compute_capability"])
    ):
        raise ValueError("Equivalence result was not produced on exactly one bfloat16 H100.")
    environment = raw.get("environment")
    if (
        not isinstance(environment, Mapping)
        or set(environment)
        != {"conda_prefix", "python", "torch", "diffusers", "diffusers_commit", "transformers"}
        or environment.get("conda_prefix") != EXPECTED_CONDA_PREFIX
        or environment.get("diffusers") != DIFFUSERS_VERSION
        or environment.get("diffusers_commit") != DIFFUSERS_COMMIT
        or any(
            not isinstance(environment.get(key), str) or not environment.get(key)
            for key in ("python", "torch", "transformers")
        )
    ):
        raise ValueError("Equivalence production-environment identity drifted.")
    preregistration = _validate_preregistration(
        raw.get("preregistration"),
        project_root=project_root,
        require_live_sources=require_live_preregistration_sources,
    )
    expected_binding = {
        key: preregistration[key]
        for key in (
            "path",
            "raw_sha256",
            "document_sha256",
            "sidecar_path",
            "sidecar_sha256",
        )
    }
    if preregistration_binding != expected_binding:
        raise ValueError("Result preregistration binding differs from environment preflight.")

    package_sources = raw.get("package_sources")
    expected_package_sources = {
        "diffusers_configuration_utils": (
            EXPECTED_DIFFUSERS_CONFIGURATION_SOURCE,
            CONFIGURATION_SOURCE_SHA256,
        ),
        "diffusers_flux_pipeline": (
            EXPECTED_DIFFUSERS_PIPELINE_SOURCE,
            PIPELINE_SOURCE_SHA256,
        ),
        "diffusers_flow_match_scheduler": (
            EXPECTED_DIFFUSERS_SCHEDULER_SOURCE,
            SCHEDULER_SOURCE_SHA256,
        ),
    }
    if not isinstance(package_sources, Mapping) or set(package_sources) != set(
        expected_package_sources
    ):
        raise ValueError("Equivalence package-source coverage drifted.")
    for role, (expected_path, expected_sha) in expected_package_sources.items():
        record = package_sources[role]
        if (
            not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or Path(str(record.get("path", ""))) != expected_path
            or record.get("sha256") != expected_sha
            or _SHA256_RE.fullmatch(str(record.get("sha256", ""))) is None
        ):
            raise ValueError(f"Equivalence package-source receipt {role!r} drifted.")
        package_path = expected_path
        package_raw = _read_regular_bytes(
            package_path,
            f"equivalence package source {role}",
            require_readonly=False,
            trusted_root=Path(EXPECTED_CONDA_PREFIX),
        )
        if sha256_bytes(package_raw) != expected_sha:
            raise ValueError(f"Live equivalence package source {role!r} drifted.")

    snapshot = raw.get("model_snapshot")
    if (
        not isinstance(snapshot, Mapping)
        or set(snapshot)
        != {
            "model_id",
            "revision",
            "path",
            "model_index_sha256",
            "scheduler_config_file",
            "tokenizer_files",
        }
        or snapshot.get("model_id") != MODEL_ID
        or snapshot.get("revision") != MODEL_REVISION
        or snapshot.get("model_index_sha256") != MODEL_INDEX_SHA256
        or snapshot.get("path") != str(EXPECTED_MODEL_SNAPSHOT_PATH)
        or not isinstance(snapshot.get("tokenizer_files"), Mapping)
        or set(snapshot["tokenizer_files"]) != set(tokenizer_file_sha256)
    ):
        raise ValueError("Equivalence model-snapshot identity drifted.")
    scheduler_record = snapshot.get("scheduler_config_file")
    expected_scheduler_path = EXPECTED_MODEL_SNAPSHOT_PATH / SNAPSHOT_SCHEDULER_CONFIG_RELATIVE
    try:
        scheduler_record_parsed_sha256 = (
            canonical_sha256(scheduler_record.get("parsed_config"))
            if isinstance(scheduler_record, Mapping)
            else None
        )
    except (TypeError, ValueError):
        scheduler_record_parsed_sha256 = None
    if (
        not isinstance(scheduler_record, Mapping)
        or set(scheduler_record)
        != {
            "path",
            "sha256",
            "size_bytes",
            "blob_name",
            "parsed_sha256",
            "parsed_config",
        }
        or scheduler_record.get("path") != str(expected_scheduler_path)
        or scheduler_record.get("sha256") != SNAPSHOT_SCHEDULER_CONFIG_SHA256
        or type(scheduler_record.get("size_bytes")) is not int
        or scheduler_record.get("size_bytes") != SNAPSHOT_SCHEDULER_CONFIG_SIZE_BYTES
        or scheduler_record.get("blob_name") != SNAPSHOT_SCHEDULER_CONFIG_BLOB_NAME
        or scheduler_record.get("parsed_sha256") != SNAPSHOT_SCHEDULER_CONFIG_PARSED_SHA256
        or scheduler_record_parsed_sha256 != SNAPSHOT_SCHEDULER_CONFIG_PARSED_SHA256
        or scheduler_record.get("parsed_config") != EXPECTED_SNAPSHOT_SCHEDULER_CONFIG
    ):
        raise ValueError("Equivalence snapshot scheduler-config receipt drifted.")
    for role, record in snapshot["tokenizer_files"].items():
        if (
            not isinstance(role, str)
            or not isinstance(record, Mapping)
            or set(record) != {"path", "sha256"}
            or record.get("sha256") != tokenizer_file_sha256[role]
        ):
            raise ValueError("Equivalence tokenizer-file receipt drifted.")
    snapshot_root = Path(str(snapshot["path"]))
    model_index_raw = read_authenticated_hf_snapshot_file(snapshot_root, "model_index.json")
    if sha256_bytes(model_index_raw) != MODEL_INDEX_SHA256:
        raise ValueError("Live equivalence model_index.json digest drifted.")
    scheduler_config_raw = read_authenticated_hf_snapshot_file(
        snapshot_root,
        SNAPSHOT_SCHEDULER_CONFIG_RELATIVE,
        expected_blob_name=SNAPSHOT_SCHEDULER_CONFIG_BLOB_NAME,
    )
    scheduler_config_parsed = parse_and_validate_snapshot_scheduler_config(scheduler_config_raw)
    if (
        scheduler_config_parsed != scheduler_record["parsed_config"]
        or canonical_sha256(scheduler_config_parsed) != scheduler_record["parsed_sha256"]
    ):
        raise ValueError("Live equivalence snapshot scheduler config drifted.")
    for relative, record in snapshot["tokenizer_files"].items():
        expected_path = snapshot_root / relative
        if Path(str(record["path"])) != expected_path:
            raise ValueError(f"Equivalence tokenizer path drifted for {relative!r}.")
        tokenizer_raw = read_authenticated_hf_snapshot_file(snapshot_root, relative)
        if sha256_bytes(tokenizer_raw) != record["sha256"]:
            raise ValueError(f"Live equivalence tokenizer digest drifted for {relative!r}.")
    slurm = raw.get("slurm")
    if (
        not isinstance(slurm, Mapping)
        or set(slurm) != {"job_id", "job_name", "node_list"}
        or not str(slurm.get("job_id", "")).isdigit()
        or slurm.get("job_name") != EQUIVALENCE_SLURM_JOB_NAME
        or not isinstance(slurm.get("node_list"), str)
        or not slurm.get("node_list")
    ):
        raise ValueError("Equivalence runtime Slurm identity drifted.")
    return {
        "gpu": deepcopy(dict(gpu)),
        "environment": deepcopy(dict(environment)),
        "preregistration": preregistration,
        "package_sources": deepcopy(dict(package_sources)),
        "model_snapshot": deepcopy(dict(snapshot)),
        "slurm": deepcopy(dict(slurm)),
    }


def _validate_media_binding(
    raw: Any,
    *,
    result_directory: Path,
    prompt_id: str,
    route: str,
) -> dict[str, Any]:
    expected_fields = {
        "prompt_id",
        "route",
        "path",
        "sha256",
        "size_bytes",
        "width",
        "height",
    }
    expected_path = result_directory / f"{prompt_id}__{route}.png"
    layout = IMAGE_LAYOUTS[prompt_id]
    if (
        not isinstance(raw, Mapping)
        or set(raw) != expected_fields
        or raw.get("prompt_id") != prompt_id
        or raw.get("route") != route
        or Path(str(raw.get("path", ""))) != expected_path
        or raw.get("width") != layout["width"]
        or raw.get("height") != layout["height"]
    ):
        raise ValueError(f"Equivalence media binding drifted for {prompt_id}/{route}.")
    media_raw = _read_regular_bytes(
        expected_path,
        f"equivalence media {prompt_id}/{route}",
        require_readonly=True,
        trusted_root=result_directory,
    )
    if raw.get("sha256") != sha256_bytes(media_raw) or raw.get("size_bytes") != len(media_raw):
        raise ValueError(f"Equivalence media bytes drifted for {prompt_id}/{route}.")
    try:
        with Image.open(io.BytesIO(media_raw)) as image:
            image.load()
            observed = (image.format, image.mode, image.size)
    except Exception as exc:
        raise ValueError(f"Equivalence media is not a decodable PNG: {expected_path}") from exc
    if observed != ("PNG", "RGB", (layout["width"], layout["height"])):
        raise ValueError(f"Equivalence media geometry/mode drifted: {expected_path}")
    return deepcopy(dict(raw))


def _validate_prompt_result(
    raw: Any,
    *,
    prompt_id: str,
    source: Mapping[str, Any],
    result_directory: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    expected_fields = {
        "prompt_id",
        "status",
        "prompt_views",
        "layout",
        "conditioning_preflight",
        "conditioning_provenance",
        "independent_t5_sentinel",
        "timing_seconds",
        "comparisons",
        "failed_comparison_ids",
        "media",
    }
    if (
        not isinstance(raw, Mapping)
        or set(raw) != expected_fields
        or raw.get("prompt_id") != prompt_id
        or raw.get("status") != "passed"
        or raw.get("layout") != IMAGE_LAYOUTS[prompt_id]
    ):
        raise ValueError(f"Equivalence prompt result identity drifted for {prompt_id}.")
    expected_views = {
        "clip_prompt": source["clip_prompt"],
        "clip_prompt_sha256": source["clip_string_sha256"],
        "t5_prompt_2": source["t5_prompt_2"],
        "t5_prompt_2_sha256": source["t5_string_sha256"],
        "negative": None,
    }
    plan = _positive_plan(source)
    if (
        raw.get("prompt_views") != expected_views
        or raw.get("conditioning_preflight") != _expected_conditioning_preflight(plan)
        or raw.get("conditioning_provenance") != _expected_conditioning_provenance(plan)
        or raw.get("failed_comparison_ids") != []
    ):
        raise ValueError(f"Equivalence prompt-view/provenance drifted for {prompt_id}.")
    _validate_independent_t5_sentinel(
        raw.get("independent_t5_sentinel"), prompt_id=prompt_id, row=source
    )
    timing = raw.get("timing_seconds")
    if not isinstance(timing, Mapping) or set(timing) != {
        "native_pipeline",
        "dual_view_adapter",
    }:
        raise ValueError(f"Equivalence prompt timing schema drifted for {prompt_id}.")
    for role, value in timing.items():
        _finite_nonnegative(value, f"{prompt_id} {role} timing")
    _validate_comparisons(raw.get("comparisons"), prompt_id)
    prompt_media = raw.get("media")
    if not isinstance(prompt_media, Mapping) or set(prompt_media) != set(MEDIA_ROUTES):
        raise ValueError(f"Equivalence prompt media coverage drifted for {prompt_id}.")
    normalized_media: list[dict[str, Any]] = []
    for route in MEDIA_ROUTES:
        compact = prompt_media[route]
        if not isinstance(compact, Mapping) or set(compact) != {"path", "sha256"}:
            raise ValueError(f"Equivalence prompt media schema drifted for {prompt_id}/{route}.")
        full = _validate_media_binding(
            {
                "prompt_id": prompt_id,
                "route": route,
                "path": compact["path"],
                "sha256": compact["sha256"],
                "size_bytes": Path(str(compact["path"])).stat().st_size,
                "width": IMAGE_LAYOUTS[prompt_id]["width"],
                "height": IMAGE_LAYOUTS[prompt_id]["height"],
            },
            result_directory=result_directory,
            prompt_id=prompt_id,
            route=route,
        )
        normalized_media.append(full)
    if normalized_media[0]["sha256"] != normalized_media[1]["sha256"]:
        raise ValueError(f"Equivalent {prompt_id} routes serialized different PNG bytes.")
    return deepcopy(dict(raw)), normalized_media


def _validate_acceptance_summary(raw: Any) -> dict[str, Any]:
    expected = {
        "schema_version": 1,
        "status": "passed",
        "required_comparison_count": COMPARISON_COUNT,
        "observed_comparison_count": COMPARISON_COUNT,
        "passed_comparison_count": COMPARISON_COUNT,
        "failed_comparison_count": 0,
        "runner_registry_construction_preflight_status": "passed",
        "required_independent_t5_sentinel_count": len(PROMPT_IDS),
        "observed_independent_t5_sentinel_count": len(PROMPT_IDS),
        "passed_independent_t5_sentinel_count": len(PROMPT_IDS),
        "required_media_count": MEDIA_COUNT,
        "observed_media_count": MEDIA_COUNT,
    }
    if raw != expected:
        raise ValueError("Equivalence acceptance-summary arithmetic/status drifted.")
    return deepcopy(expected)


def _validate_lineage_comparisons(
    raw: Any, prompt_id: str
) -> tuple[list[dict[str, Any]], list[str]]:
    """Validate exact diagnostic comparison schema for a nonpassing execution."""

    if not isinstance(raw, list) or len(raw) != COMPARISONS_PER_PROMPT:
        raise ValueError(f"{prompt_id} lineage comparison coverage is not exactly 94.")
    if [row.get("comparison_id") for row in raw if isinstance(row, Mapping)] != (
        expected_comparison_ids(prompt_id)
    ):
        raise ValueError(f"{prompt_id} lineage comparison IDs/order drifted.")
    expected_fields = {
        "comparison_id",
        "passed",
        "required_relation",
        "same_shape",
        "same_dtype",
        "max_abs_difference",
        "mean_abs_difference",
        "native",
        "adapter",
    }
    normalized: list[dict[str, Any]] = []
    failed: list[str] = []
    for row in raw:
        if not isinstance(row, Mapping) or set(row) != expected_fields:
            raise ValueError(f"{prompt_id} lineage comparison schema drifted.")
        native = _validate_tensor_summary(row["native"], f"{row['comparison_id']}.native")
        adapter = _validate_tensor_summary(row["adapter"], f"{row['comparison_id']}.adapter")
        same_shape = native["shape"] == adapter["shape"]
        same_dtype = native["dtype"] == adapter["dtype"]
        exact = same_shape and same_dtype and native["sha256"] == adapter["sha256"]
        max_delta = row.get("max_abs_difference")
        mean_delta = row.get("mean_abs_difference")
        for value, label in (
            (max_delta, "max_abs_difference"),
            (mean_delta, "mean_abs_difference"),
        ):
            if value is not None:
                _finite_nonnegative(value, f"{row['comparison_id']}.{label}")
        if (
            row.get("required_relation") != "bitwise_identical"
            or row.get("same_shape") is not same_shape
            or row.get("same_dtype") is not same_dtype
            or row.get("passed") is not exact
            or (exact and max_delta not in (None, 0, 0.0))
            or (exact and mean_delta not in (None, 0, 0.0))
        ):
            raise ValueError(
                f"{prompt_id} lineage comparison arithmetic drifted for "
                f"{row.get('comparison_id')!r}."
            )
        if not exact:
            failed.append(str(row["comparison_id"]))
        normalized.append(deepcopy(dict(row)))
    return normalized, failed


def validate_equivalence_result(
    *,
    result_path: str | Path,
    result_sidecar_path: str | Path | None = None,
    project_root: str | Path,
    require_live_preregistration_sources: bool = True,
) -> dict[str, Any]:
    """Authenticate the immutable result, preregistration, comparisons and media."""

    project_root = Path(project_root).expanduser().resolve()
    result_path = Path(result_path).expanduser()
    if not result_path.is_absolute():
        result_path = project_root / result_path
    audit_root = (project_root / AUDIT_ROOT_RELATIVE).resolve()
    source_attempt, execution_attempt = parse_result_attempts(result_path.parent)
    if source_attempt == QUARANTINE_SOURCE_ATTEMPT:
        raise ValueError(
            "Source-003 is permanently quarantined passed-but-unaccepted and cannot enter "
            "ordinary equivalence admission."
        )
    if (
        audit_root not in result_path.parents
        or result_path.name != RESULT_FILENAME
        or result_path.parent
        != audit_root / result_directory_name(source_attempt, execution_attempt)
    ):
        raise ValueError("Equivalence result path is outside its canonical audit namespace.")
    sidecar_path = (
        Path(result_sidecar_path).expanduser()
        if result_sidecar_path is not None
        else Path(f"{result_path}.sha256")
    )
    if not sidecar_path.is_absolute():
        sidecar_path = project_root / sidecar_path
    result, result_raw = _read_json(
        result_path,
        "equivalence result",
        require_readonly=True,
        trusted_root=audit_root,
    )
    result_sidecar = _validate_sidecar(
        payload_path=result_path,
        payload_raw=result_raw,
        sidecar_path=sidecar_path,
        label="equivalence result sidecar",
        trusted_root=audit_root,
    )
    expected_top = {
        "schema_version",
        "status",
        "gate",
        "protocol_id",
        "attempt_lineage",
        "started_at",
        "ended_at",
        "environment_preflight",
        "preregistration_binding",
        "runner_registry_construction_preflight",
        "execution_contract",
        "prompt_contract",
        "scheduler_config",
        "scheduler_config_sha256",
        "timing_seconds",
        "prompt_results",
        "comparison_count",
        "passed_comparison_count",
        "failed_comparison_ids",
        "acceptance_summary",
        "media_count",
        "media_bindings",
        "media_paths",
    }
    if (
        set(result) != expected_top
        or result.get("schema_version") != SCHEMA_VERSION
        or result.get("status") != "passed"
        or result.get("gate") != GATE
        or result.get("protocol_id") != PROTOCOL_ID
        or result.get("execution_contract") != expected_execution_contract()
    ):
        raise ValueError("Equivalence result schema/identity/execution contract drifted.")
    started = _aware_timestamp(result.get("started_at"), "result started_at")
    ended = _aware_timestamp(result.get("ended_at"), "result ended_at")
    if ended < started:
        raise ValueError("Equivalence result ended before it started.")

    (
        prompt_contract_path,
        prompt_rows,
        tokenizer_file_sha256,
    ) = _load_prompt_contract(project_root)
    _validate_sealed_project_file(
        project_root,
        PROMPT_PROTOCOL_RELATIVE,
        PROMPT_PROTOCOL_SHA256,
        sidecar_relative=PROMPT_PROTOCOL_SIDECAR_RELATIVE,
    )
    _validate_sealed_project_file(
        project_root,
        MODEL_CONFIG_RELATIVE,
        MODEL_CONFIG_SHA256,
        sidecar_relative=MODEL_CONFIG_SIDECAR_RELATIVE,
    )
    if result.get("prompt_contract") != {
        "path": str(prompt_contract_path),
        "sha256": PROMPT_CONTRACT_SHA256,
    }:
        raise ValueError("Equivalence result prompt-contract binding drifted.")
    environment = _validate_environment_preflight(
        result.get("environment_preflight"),
        project_root=project_root,
        preregistration_binding=result.get("preregistration_binding"),
        tokenizer_file_sha256=tokenizer_file_sha256,
        require_live_preregistration_sources=require_live_preregistration_sources,
    )
    preregistration_path = Path(environment["preregistration"]["path"])
    expected_attempt = expected_attempt_lineage(
        source_attempt=source_attempt,
        execution_attempt=execution_attempt,
        preregistration_path=preregistration_path,
        output_directory=result_path.parent,
    )
    if (
        environment["preregistration"]["source_attempt"] != source_attempt
        or result.get("attempt_lineage") != expected_attempt
    ):
        raise ValueError("Equivalence result source/execution attempt lineage drifted.")
    preregistered_at = _aware_timestamp(
        environment["preregistration"]["payload"]["created_at_utc"],
        "preregistration created_at_utc",
    )
    if preregistered_at > started:
        raise ValueError("Equivalence preregistration was not completed before execution.")
    runner_preflight = _validate_runner_registry_preflight(
        result.get("runner_registry_construction_preflight"),
        project_root=project_root,
        prompt_rows=prompt_rows,
    )
    scheduler = result.get("scheduler_config")
    if (
        not isinstance(scheduler, Mapping)
        or dict(scheduler) != EXPECTED_SCHEDULER_CONFIG
        or canonical_sha256(scheduler) != EXPECTED_SCHEDULER_CONFIG_SHA256
        or result.get("scheduler_config_sha256") != EXPECTED_SCHEDULER_CONFIG_SHA256
    ):
        raise ValueError("Equivalence scheduler contract drifted.")
    timing = result.get("timing_seconds")
    if not isinstance(timing, Mapping) or set(timing) != {"model_load", "total"}:
        raise ValueError("Equivalence result timing schema drifted.")
    _finite_nonnegative(timing["model_load"], "equivalence model_load timing")
    _finite_nonnegative(timing["total"], "equivalence total timing")

    prompt_results = result.get("prompt_results")
    if (
        not isinstance(prompt_results, list)
        or len(prompt_results) != len(PROMPT_IDS)
        or [row.get("prompt_id") for row in prompt_results if isinstance(row, Mapping)]
        != list(PROMPT_IDS)
    ):
        raise ValueError("Equivalence result prompt coverage/order drifted.")
    derived_media: list[dict[str, Any]] = []
    for prompt_id, prompt_result in zip(PROMPT_IDS, prompt_results, strict=True):
        _, media = _validate_prompt_result(
            prompt_result,
            prompt_id=prompt_id,
            source=prompt_rows[prompt_id],
            result_directory=result_path.parent,
        )
        derived_media.extend(media)

    if (
        result.get("comparison_count") != COMPARISON_COUNT
        or result.get("passed_comparison_count") != COMPARISON_COUNT
        or result.get("failed_comparison_ids") != []
        or result.get("media_count") != MEDIA_COUNT
    ):
        raise ValueError("Equivalence top-level comparison/media arithmetic drifted.")
    _validate_acceptance_summary(result.get("acceptance_summary"))
    observed_media = result.get("media_bindings")
    if not isinstance(observed_media, list) or len(observed_media) != MEDIA_COUNT:
        raise ValueError("Equivalence top-level media-binding coverage drifted.")
    normalized_media = [
        _validate_media_binding(
            observed,
            result_directory=result_path.parent,
            prompt_id=prompt_id,
            route=route,
        )
        for observed, (prompt_id, route) in zip(
            observed_media,
            ((prompt_id, route) for prompt_id in PROMPT_IDS for route in MEDIA_ROUTES),
            strict=True,
        )
    ]
    if normalized_media != derived_media:
        raise ValueError("Prompt and top-level equivalence media bindings disagree.")
    expected_media_paths = sorted(row["path"] for row in normalized_media)
    if (
        result.get("media_paths") != expected_media_paths
        or len(set(expected_media_paths)) != MEDIA_COUNT
    ):
        raise ValueError("Equivalence media path bijection/uniqueness drifted.")
    for offset in range(0, MEDIA_COUNT, 2):
        if normalized_media[offset]["sha256"] != normalized_media[offset + 1]["sha256"]:
            raise ValueError("Native and adapter media hashes differ despite passed comparisons.")
    prompt_media_hashes = {
        normalized_media[offset]["sha256"] for offset in range(0, MEDIA_COUNT, 2)
    }
    if len(prompt_media_hashes) != len(PROMPT_IDS):
        raise ValueError("Distinct equivalence prompts serialized duplicate PNG evidence.")

    preregistration = environment["preregistration"]
    launcher = preregistration["authenticated_files"]["slurm_launcher"]
    return {
        "result": {
            **_artifact_binding(result_path, result_raw),
            "sidecar_path": result_sidecar["path"],
            "sidecar_sha256": result_sidecar["sha256"],
            "document_sha256": canonical_sha256(result),
            "status": "passed",
        },
        "payload": deepcopy(result),
        "preregistration": {
            key: preregistration[key]
            for key in (
                "path",
                "raw_sha256",
                "document_sha256",
                "sidecar_path",
                "sidecar_sha256",
                "source_attempt",
                "lineage",
            )
        }
        | {"created_at_utc": preregistration["payload"]["created_at_utc"]},
        "slurm_launcher": deepcopy(launcher),
        "runtime_slurm": deepcopy(environment["slurm"]),
        "gpu_name": environment["gpu"]["name"],
        "media": normalized_media,
        "runner_preflight_status": runner_preflight["status"],
        "started_at": result["started_at"],
        "ended_at": result["ended_at"],
        "attempt_lineage": expected_attempt,
    }


def _load_execution_result_immutable(
    *, result_path: Path, project_root: Path
) -> tuple[dict[str, Any], bytes, dict[str, Any], str, str]:
    audit_root = (project_root / AUDIT_ROOT_RELATIVE).resolve()
    source_attempt, execution_attempt = parse_result_attempts(result_path.parent)
    expected_path = (
        audit_root / result_directory_name(source_attempt, execution_attempt) / RESULT_FILENAME
    )
    if result_path != expected_path:
        raise ValueError("Execution result path is outside the canonical attempt namespace.")
    payload, raw = _read_json(
        result_path,
        "equivalence execution result",
        require_readonly=True,
        trusted_root=audit_root,
    )
    sidecar = _validate_sidecar(
        payload_path=result_path,
        payload_raw=raw,
        sidecar_path=Path(f"{result_path}.sha256"),
        label="equivalence execution result sidecar",
        trusted_root=audit_root,
    )
    return payload, raw, sidecar, source_attempt, execution_attempt


def _validate_error_execution_result(*, result_path: Path, project_root: Path) -> dict[str, Any]:
    result, raw, sidecar, source_attempt, execution_attempt = _load_execution_result_immutable(
        result_path=result_path, project_root=project_root
    )
    expected_fields = {
        "schema_version",
        "status",
        "gate",
        "protocol_id",
        "attempt_lineage",
        "preregistration_binding",
        "runtime_slurm",
        "started_at",
        "ended_at",
        "error_type",
        "error",
        "traceback",
    }
    preregistration_path = (
        project_root / AUDIT_ROOT_RELATIVE / preregistration_filename(source_attempt)
    ).resolve()
    preregistration = _validate_preregistration(
        _historical_preregistration_environment_record(
            project_root=project_root,
            preregistration_path=preregistration_path,
        ),
        project_root=project_root,
        require_live_sources=False,
    )
    expected_attempt = expected_attempt_lineage(
        source_attempt=source_attempt,
        execution_attempt=execution_attempt,
        preregistration_path=preregistration_path,
        output_directory=result_path.parent,
    )
    expected_preregistration = {
        key: preregistration[key]
        for key in (
            "path",
            "raw_sha256",
            "document_sha256",
            "sidecar_path",
            "sidecar_sha256",
        )
    }
    runtime_slurm = result.get("runtime_slurm")
    if (
        set(result) != expected_fields
        or result.get("schema_version") != SCHEMA_VERSION
        or result.get("status") != "error"
        or result.get("gate") != GATE
        or result.get("protocol_id") != PROTOCOL_ID
        or result.get("attempt_lineage") != expected_attempt
        or result.get("preregistration_binding") != expected_preregistration
        or not isinstance(runtime_slurm, Mapping)
        or set(runtime_slurm) != {"job_id", "job_name", "node_list"}
        or not str(runtime_slurm.get("job_id", "")).isdigit()
        or runtime_slurm.get("job_name") != EQUIVALENCE_SLURM_JOB_NAME
        or not isinstance(runtime_slurm.get("node_list"), str)
        or not runtime_slurm.get("node_list")
        or any(
            not isinstance(result.get(field), str) or not result[field]
            for field in ("error_type", "error", "traceback")
        )
    ):
        raise ValueError("Error execution result schema/identity binding drifted.")
    started = _aware_timestamp(result.get("started_at"), "error result started_at")
    ended = _aware_timestamp(result.get("ended_at"), "error result ended_at")
    preregistered = _aware_timestamp(
        preregistration["payload"]["created_at_utc"],
        "error result preregistration created_at_utc",
    )
    if not preregistered <= started <= ended:
        raise ValueError("Error execution result timestamp order drifted.")
    return {
        "result": {
            **_artifact_binding(result_path, raw),
            "sidecar_path": sidecar["path"],
            "sidecar_sha256": sidecar["sha256"],
            "document_sha256": canonical_sha256(result),
            "status": "error",
        },
        "payload": result,
        "preregistration": preregistration,
        "runtime_slurm": deepcopy(dict(runtime_slurm)),
        "started_at": result["started_at"],
        "ended_at": result["ended_at"],
        "attempt_lineage": expected_attempt,
    }


def _validate_failed_execution_result(*, result_path: Path, project_root: Path) -> dict[str, Any]:
    result, raw, sidecar, source_attempt, execution_attempt = _load_execution_result_immutable(
        result_path=result_path, project_root=project_root
    )
    expected_top = {
        "schema_version",
        "status",
        "gate",
        "protocol_id",
        "attempt_lineage",
        "started_at",
        "ended_at",
        "environment_preflight",
        "preregistration_binding",
        "runner_registry_construction_preflight",
        "execution_contract",
        "prompt_contract",
        "scheduler_config",
        "scheduler_config_sha256",
        "timing_seconds",
        "prompt_results",
        "comparison_count",
        "passed_comparison_count",
        "failed_comparison_ids",
        "acceptance_summary",
        "media_count",
        "media_bindings",
        "media_paths",
    }
    if (
        set(result) != expected_top
        or result.get("schema_version") != SCHEMA_VERSION
        or result.get("status") != "failed"
        or result.get("gate") != GATE
        or result.get("protocol_id") != PROTOCOL_ID
        or result.get("execution_contract") != expected_execution_contract()
        or result.get("scheduler_config") != EXPECTED_SCHEDULER_CONFIG
        or canonical_sha256(result.get("scheduler_config")) != EXPECTED_SCHEDULER_CONFIG_SHA256
        or result.get("scheduler_config_sha256") != EXPECTED_SCHEDULER_CONFIG_SHA256
    ):
        raise ValueError("Failed execution result schema/identity drifted.")
    started = _aware_timestamp(result.get("started_at"), "failed result started_at")
    ended = _aware_timestamp(result.get("ended_at"), "failed result ended_at")
    if ended < started:
        raise ValueError("Failed execution result ended before it started.")
    prompt_path, prompt_rows, tokenizer_files = _load_prompt_contract(project_root)
    if result.get("prompt_contract") != {
        "path": str(prompt_path),
        "sha256": PROMPT_CONTRACT_SHA256,
    }:
        raise ValueError("Failed execution prompt-contract binding drifted.")
    environment = _validate_environment_preflight(
        result.get("environment_preflight"),
        project_root=project_root,
        preregistration_binding=result.get("preregistration_binding"),
        tokenizer_file_sha256=tokenizer_files,
        require_live_preregistration_sources=False,
    )
    preregistration = environment["preregistration"]
    expected_attempt = expected_attempt_lineage(
        source_attempt=source_attempt,
        execution_attempt=execution_attempt,
        preregistration_path=Path(preregistration["path"]),
        output_directory=result_path.parent,
    )
    if (
        preregistration["source_attempt"] != source_attempt
        or result.get("attempt_lineage") != expected_attempt
    ):
        raise ValueError("Failed execution attempt lineage drifted.")
    preregistered = _aware_timestamp(
        preregistration["payload"]["created_at_utc"],
        "failed result preregistration created_at_utc",
    )
    if preregistered > started:
        raise ValueError("Failed execution predates its preregistration.")
    _validate_runner_registry_preflight(
        result.get("runner_registry_construction_preflight"),
        project_root=project_root,
        prompt_rows=prompt_rows,
    )
    timing = result.get("timing_seconds")
    if not isinstance(timing, Mapping) or set(timing) != {"model_load", "total"}:
        raise ValueError("Failed execution timing schema drifted.")
    _finite_nonnegative(timing["model_load"], "failed model_load timing")
    _finite_nonnegative(timing["total"], "failed total timing")
    prompt_results = result.get("prompt_results")
    if (
        not isinstance(prompt_results, list)
        or len(prompt_results) != len(PROMPT_IDS)
        or [row.get("prompt_id") for row in prompt_results if isinstance(row, Mapping)]
        != list(PROMPT_IDS)
    ):
        raise ValueError("Failed execution prompt coverage/order drifted.")
    failed_ids: list[str] = []
    derived_media: list[dict[str, Any]] = []
    prompt_fields = {
        "prompt_id",
        "status",
        "prompt_views",
        "layout",
        "conditioning_preflight",
        "conditioning_provenance",
        "independent_t5_sentinel",
        "timing_seconds",
        "comparisons",
        "failed_comparison_ids",
        "media",
    }
    for prompt_id, prompt_result in zip(PROMPT_IDS, prompt_results, strict=True):
        source = prompt_rows[prompt_id]
        plan = _positive_plan(source)
        if (
            not isinstance(prompt_result, Mapping)
            or set(prompt_result) != prompt_fields
            or prompt_result.get("prompt_id") != prompt_id
            or prompt_result.get("layout") != IMAGE_LAYOUTS[prompt_id]
            or prompt_result.get("prompt_views")
            != {
                "clip_prompt": source["clip_prompt"],
                "clip_prompt_sha256": source["clip_string_sha256"],
                "t5_prompt_2": source["t5_prompt_2"],
                "t5_prompt_2_sha256": source["t5_string_sha256"],
                "negative": None,
            }
            or prompt_result.get("conditioning_preflight") != _expected_conditioning_preflight(plan)
            or prompt_result.get("conditioning_provenance")
            != _expected_conditioning_provenance(plan)
        ):
            raise ValueError(f"Failed execution prompt schema drifted for {prompt_id}.")
        _validate_independent_t5_sentinel(
            prompt_result.get("independent_t5_sentinel"),
            prompt_id=prompt_id,
            row=source,
        )
        prompt_timing = prompt_result.get("timing_seconds")
        if not isinstance(prompt_timing, Mapping) or set(prompt_timing) != {
            "native_pipeline",
            "dual_view_adapter",
        }:
            raise ValueError(f"Failed execution prompt timing drifted for {prompt_id}.")
        for role, value in prompt_timing.items():
            _finite_nonnegative(value, f"{prompt_id} failed {role} timing")
        _, prompt_failed = _validate_lineage_comparisons(
            prompt_result.get("comparisons"), prompt_id
        )
        expected_prompt_status = "failed" if prompt_failed else "passed"
        if (
            prompt_result.get("status") != expected_prompt_status
            or prompt_result.get("failed_comparison_ids") != prompt_failed
        ):
            raise ValueError(f"Failed execution prompt arithmetic drifted for {prompt_id}.")
        failed_ids.extend(prompt_failed)
        prompt_media = prompt_result.get("media")
        if not isinstance(prompt_media, Mapping) or set(prompt_media) != set(MEDIA_ROUTES):
            raise ValueError(f"Failed execution media coverage drifted for {prompt_id}.")
        for route in MEDIA_ROUTES:
            compact = prompt_media[route]
            if not isinstance(compact, Mapping) or set(compact) != {"path", "sha256"}:
                raise ValueError(f"Failed execution media schema drifted for {prompt_id}.")
            path = Path(str(compact["path"]))
            derived_media.append(
                _validate_media_binding(
                    {
                        "prompt_id": prompt_id,
                        "route": route,
                        "path": str(path),
                        "sha256": compact["sha256"],
                        "size_bytes": path.stat().st_size,
                        "width": IMAGE_LAYOUTS[prompt_id]["width"],
                        "height": IMAGE_LAYOUTS[prompt_id]["height"],
                    },
                    result_directory=result_path.parent,
                    prompt_id=prompt_id,
                    route=route,
                )
            )
    if not failed_ids:
        raise ValueError("Failed execution contains no failed comparison.")
    passed_count = COMPARISON_COUNT - len(failed_ids)
    expected_summary = {
        "schema_version": 1,
        "status": "failed",
        "required_comparison_count": COMPARISON_COUNT,
        "observed_comparison_count": COMPARISON_COUNT,
        "passed_comparison_count": passed_count,
        "failed_comparison_count": len(failed_ids),
        "runner_registry_construction_preflight_status": "passed",
        "required_independent_t5_sentinel_count": len(PROMPT_IDS),
        "observed_independent_t5_sentinel_count": len(PROMPT_IDS),
        "passed_independent_t5_sentinel_count": len(PROMPT_IDS),
        "required_media_count": MEDIA_COUNT,
        "observed_media_count": MEDIA_COUNT,
    }
    if (
        result.get("comparison_count") != COMPARISON_COUNT
        or result.get("passed_comparison_count") != passed_count
        or result.get("failed_comparison_ids") != failed_ids
        or result.get("acceptance_summary") != expected_summary
        or result.get("media_count") != MEDIA_COUNT
        or result.get("media_bindings") != derived_media
        or result.get("media_paths") != sorted(row["path"] for row in derived_media)
        or len({row["path"] for row in derived_media}) != MEDIA_COUNT
    ):
        raise ValueError("Failed execution aggregate arithmetic/media binding drifted.")
    return {
        "result": {
            **_artifact_binding(result_path, raw),
            "sidecar_path": sidecar["path"],
            "sidecar_sha256": sidecar["sha256"],
            "document_sha256": canonical_sha256(result),
            "status": "failed",
        },
        "payload": result,
        "preregistration": preregistration,
        "runtime_slurm": deepcopy(environment["slurm"]),
        "started_at": result["started_at"],
        "ended_at": result["ended_at"],
        "attempt_lineage": expected_attempt,
    }


def validate_terminal_execution_result(
    *, result_path: str | Path, project_root: str | Path
) -> dict[str, Any]:
    """Validate a passed, failed, or caught-error diagnostic result exactly."""

    project_root = Path(project_root).expanduser().resolve()
    path = Path(result_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    source_attempt, _ = parse_result_attempts(path.parent)
    if source_attempt == QUARANTINE_SOURCE_ATTEMPT:
        raise ValueError(
            "Every source-003 result is excluded from ordinary terminal execution validation."
        )
    result, _, _, source_attempt, _ = _load_execution_result_immutable(
        result_path=path, project_root=project_root
    )
    status = result.get("status")
    if status == "passed":
        evidence = validate_equivalence_result(
            result_path=path,
            project_root=project_root,
            require_live_preregistration_sources=False,
        )
        preregistration_path = Path(evidence["preregistration"]["path"])
        evidence["preregistration"] = _validate_preregistration(
            _historical_preregistration_environment_record(
                project_root=project_root,
                preregistration_path=preregistration_path,
            ),
            project_root=project_root,
            require_live_sources=False,
        )
        return evidence
    if status == "failed":
        return _validate_failed_execution_result(result_path=path, project_root=project_root)
    if status == "error":
        return _validate_error_execution_result(result_path=path, project_root=project_root)
    raise ValueError(f"Unsupported equivalence execution result status: {status!r}.")


def _attempt_directories_for_source(*, audit_root: Path, source_attempt: str) -> dict[str, Path]:
    attempts: dict[str, Path] = {}
    try:
        entries = list(os.scandir(audit_root))
    except OSError as exc:
        raise ValueError(f"Cannot enumerate equivalence audit root: {audit_root}.") from exc
    for entry in entries:
        match = _RESULT_DIRECTORY_RE.fullmatch(entry.name)
        if match is None or match.group("source") != source_attempt:
            continue
        execution = match.group("execution")
        if int(execution) < 1 or entry.is_symlink() or not entry.is_dir(follow_symlinks=False):
            raise ValueError(f"Execution-attempt namespace is not a real directory: {entry.path}.")
        attempts[execution] = Path(entry.path)
    return attempts


def _source_preregistration_attempts(audit_root: Path) -> dict[str, Path]:
    attempts: dict[str, Path] = {}
    try:
        entries = list(os.scandir(audit_root))
    except OSError as exc:
        raise ValueError(f"Cannot enumerate equivalence audit root: {audit_root}.") from exc
    for entry in entries:
        match = _PREREGISTRATION_NAME_RE.fullmatch(entry.name)
        if match is None:
            continue
        source = match.group("source")
        if int(source) < 1 or entry.is_symlink() or not entry.is_file(follow_symlinks=False):
            raise ValueError(f"Source-attempt preregistration is not a real file: {entry.path}.")
        attempts[source] = Path(entry.path)
    return attempts


_EXECUTION_COMMIT_ARTIFACT_NAMES = frozenset(
    {
        TERMINAL_RECEIPT_FILENAME,
        TERMINAL_RECEIPT_SIDECAR_FILENAME,
        RECEIPT_FILENAME,
        RECEIPT_SIDECAR_FILENAME,
        QUARANTINE_TERMINAL_RECEIPT_FILENAME,
        QUARANTINE_TERMINAL_RECEIPT_SIDECAR_FILENAME,
    }
)


def _execution_tree_manifest(
    output_directory: Path,
    *,
    exclude_commit_artifacts: bool,
    require_readonly_files: bool,
) -> dict[str, Any]:
    """Content-bind a symlink-free execution tree in canonical path order."""

    root = Path(output_directory)
    try:
        root_metadata = os.lstat(root)
    except OSError as exc:
        raise ValueError(f"Cannot inspect equivalence execution directory: {root}.") from exc
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("Equivalence execution root must be one real directory.")
    entries: list[dict[str, Any]] = []

    def visit(directory: Path) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ValueError(f"Cannot enumerate equivalence execution tree: {directory}.") from exc
        for child in children:
            path = Path(child.path)
            relative = path.relative_to(root)
            if (
                exclude_commit_artifacts
                and len(relative.parts) == 1
                and child.name in _EXECUTION_COMMIT_ARTIFACT_NAMES
            ):
                continue
            try:
                metadata = child.stat(follow_symlinks=False)
            except OSError as exc:
                raise ValueError(f"Cannot inspect equivalence tree member: {path}.") from exc
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"Equivalence execution tree contains a symlink: {path}.")
            relative_text = relative.as_posix()
            if stat.S_ISDIR(metadata.st_mode):
                entries.append({"relative_path": relative_text, "kind": "directory"})
                visit(path)
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError(f"Equivalence tree member is special or multiply linked: {path}.")
            raw = _read_regular_bytes(
                path,
                f"equivalence execution tree member {relative_text}",
                require_readonly=require_readonly_files,
                trusted_root=root,
            )
            entries.append(
                {
                    "relative_path": relative_text,
                    "kind": "file",
                    "sha256": sha256_bytes(raw),
                    "size_bytes": len(raw),
                }
            )

    visit(root)
    return {
        "schema_version": 1,
        "root": str(root),
        "entries": entries,
    }


def _passed_preterminal_names() -> frozenset[str]:
    return frozenset(
        {
            RESULT_FILENAME,
            RESULT_SIDECAR_FILENAME,
            *(f"{prompt_id}__{route}.png" for prompt_id in PROMPT_IDS for route in MEDIA_ROUTES),
        }
    )


def _validate_passed_preterminal_tree(tree: Mapping[str, Any]) -> None:
    entries = tree.get("entries") if isinstance(tree, Mapping) else None
    observed = {
        str(entry.get("relative_path"))
        for entry in entries or []
        if isinstance(entry, Mapping) and entry.get("kind") == "file"
    }
    if (
        not isinstance(entries, list)
        or any(not isinstance(entry, Mapping) or entry.get("kind") != "file" for entry in entries)
        or observed != _passed_preterminal_names()
        or len(entries) != len(observed)
    ):
        raise ValueError(
            "Passed equivalence preterminal tree does not contain exactly six PNGs "
            "and the result pair."
        )


def _quarantine_disposition() -> dict[str, Any]:
    return {
        "classification": "passed_but_unaccepted",
        "acceptance_forbidden": True,
        "ordinary_terminal_receipt_forbidden": True,
        "media_reuse": False,
        "scientific_media_reuse": False,
        "scientific_media_reuse_forbidden": True,
        "same_source_retry": False,
        "same_source_retry_forbidden": True,
        "fresh_h100_execution_required": True,
        "authorized_successor_source_attempt": QUARANTINE_SUCCESSOR_SOURCE_ATTEMPT,
    }


def _source003_quarantine_paths(project_root: Path) -> dict[str, Path]:
    audit_root = (project_root / AUDIT_ROOT_RELATIVE).resolve()
    output = audit_root / result_directory_name(
        QUARANTINE_SOURCE_ATTEMPT,
        QUARANTINE_EXECUTION_ATTEMPT,
    )
    return {
        "audit_root": audit_root,
        "authorization": audit_root / QUARANTINE_AUTHORIZATION_FILENAME,
        "preregistration": audit_root / preregistration_filename(QUARANTINE_SOURCE_ATTEMPT),
        "output": output,
        "result": output / RESULT_FILENAME,
        "quarantine_terminal": output / QUARANTINE_TERMINAL_RECEIPT_FILENAME,
    }


def _require_source004_execution_namespace_absent(audit_root: Path) -> None:
    attempts = _attempt_directories_for_source(
        audit_root=audit_root,
        source_attempt=QUARANTINE_SUCCESSOR_SOURCE_ATTEMPT,
    )
    if attempts:
        raise ValueError("Fresh source-004 recovery forbids pre-existing execution attempts.")


def _require_source004_namespace_absent(audit_root: Path) -> None:
    preregistration = audit_root / preregistration_filename(QUARANTINE_SUCCESSOR_SOURCE_ATTEMPT)
    for candidate in (preregistration, Path(f"{preregistration}.sha256")):
        if os.path.lexists(candidate):
            raise ValueError("Source-003 quarantine requires the source-004 pair to be absent.")
    _require_source004_execution_namespace_absent(audit_root)


def _live_preregistration_file_ledger(project_root: Path) -> dict[str, dict[str, str]]:
    records: dict[str, dict[str, str]] = {}
    for role, relative in REQUIRED_PREREGISTRATION_FILE_PATHS.items():
        path = project_root / relative
        raw = _read_regular_bytes(
            path,
            f"prospective source-004 role {role}",
            require_readonly=False,
            trusted_root=project_root,
        )
        records[role] = {"path": str(relative), "sha256": sha256_bytes(raw)}
    return records


def validate_source003_quarantine_authorization(
    *,
    project_root: str | Path,
    require_successor_absent: bool,
) -> dict[str, Any]:
    """Authenticate the exact one-off authorization and source-003 history."""

    project_root = Path(project_root).expanduser().resolve()
    paths = _source003_quarantine_paths(project_root)
    audit_root = paths["audit_root"]
    publisher_path = project_root / QUARANTINE_PUBLISHER_RELATIVE
    publisher_raw = _read_regular_bytes(
        publisher_path,
        "source-003 quarantine authorization publisher",
        require_readonly=False,
        trusted_root=project_root,
    )
    if sha256_bytes(publisher_raw) != QUARANTINE_PUBLISHER_SHA256:
        raise ValueError("Source-003 quarantine publisher bytes drifted.")

    authorization, authorization_raw = _read_json(
        paths["authorization"],
        "source-003 quarantine authorization",
        require_readonly=True,
        trusted_root=audit_root,
    )
    authorization_sidecar = _validate_sidecar(
        payload_path=paths["authorization"],
        payload_raw=authorization_raw,
        sidecar_path=Path(f"{paths['authorization']}.sha256"),
        label="source-003 quarantine authorization sidecar",
        trusted_root=audit_root,
    )
    if (
        sha256_bytes(authorization_raw) != QUARANTINE_AUTHORIZATION_SHA256
        or authorization_sidecar["sha256"] != QUARANTINE_AUTHORIZATION_SIDECAR_SHA256
        or authorization.get("document_sha256") != QUARANTINE_AUTHORIZATION_DOCUMENT_SHA256
        or document_sha256(authorization) != QUARANTINE_AUTHORIZATION_DOCUMENT_SHA256
        or authorization.get("scope")
        != {
            "source_attempt": QUARANTINE_SOURCE_ATTEMPT,
            "execution_attempt": QUARANTINE_EXECUTION_ATTEMPT,
            "successor_source_attempt": QUARANTINE_SUCCESSOR_SOURCE_ATTEMPT,
            "slurm_job_id": QUARANTINE_SLURM_JOB_ID,
            "execution_directory": str(paths["output"]),
        }
        or authorization.get("disposition") != _quarantine_disposition()
        or authorization.get("repair_intent", {}).get("intended_changed_roles")
        != list(QUARANTINE_CHANGED_ROLES)
        or authorization.get("repair_intent", {}).get("future_source_hashes_committed") is not False
        or authorization.get("repair_intent", {}).get("future_source_hashes") is not None
        or authorization.get("publisher_source")
        != {
            "path": str(publisher_path),
            "relative_path": str(QUARANTINE_PUBLISHER_RELATIVE),
            "sha256": QUARANTINE_PUBLISHER_SHA256,
            "size_bytes": len(publisher_raw),
        }
    ):
        raise ValueError("Source-003 quarantine authorization identity drifted.")

    preregistration, preregistration_binding = _immutable_json_binding(
        paths["preregistration"],
        audit_root=audit_root,
        label="source-003 quarantine preregistration",
    )
    result, result_binding = _immutable_json_binding(
        paths["result"],
        audit_root=audit_root,
        label="source-003 quarantined result",
    )
    declared_preregistration = authorization.get("source003_preregistration")
    declared_result = authorization.get("source003_result")
    old_ledger = (
        declared_preregistration.get("complete_file_ledger")
        if isinstance(declared_preregistration, Mapping)
        else None
    )
    if (
        preregistration_binding["sha256"] != QUARANTINE_SOURCE003_PREREGISTRATION_SHA256
        or not isinstance(old_ledger, Mapping)
        or set(old_ledger) != set(REQUIRED_PREREGISTRATION_FILE_PATHS)
        or preregistration.get("files") != old_ledger
        or preregistration.get("source_attempt") != QUARANTINE_SOURCE_ATTEMPT
        or not isinstance(declared_result, Mapping)
        or result_binding["sha256"] != QUARANTINE_SOURCE003_RESULT_SHA256
        or declared_result.get("sha256") != result_binding["sha256"]
        or declared_result.get("sidecar_sha256") != result_binding["sidecar_sha256"]
        or result.get("status") != "passed"
        or result.get("comparison_count") != COMPARISON_COUNT
        or result.get("passed_comparison_count") != COMPARISON_COUNT
        or result.get("failed_comparison_ids") != []
        or result.get("media_count") != MEDIA_COUNT
        or result.get("attempt_lineage", {}).get("source_attempt") != QUARANTINE_SOURCE_ATTEMPT
        or result.get("attempt_lineage", {}).get("execution_attempt")
        != QUARANTINE_EXECUTION_ATTEMPT
        or result.get("environment_preflight", {}).get("slurm", {}).get("job_id")
        != QUARANTINE_SLURM_JOB_ID
    ):
        raise ValueError("Source-003 quarantine preregistration/result binding drifted.")

    for name in (
        TERMINAL_RECEIPT_FILENAME,
        TERMINAL_RECEIPT_SIDECAR_FILENAME,
        RECEIPT_FILENAME,
        RECEIPT_SIDECAR_FILENAME,
    ):
        if os.path.lexists(paths["output"] / name):
            raise ValueError("Source-003 quarantine contains a forbidden ordinary receipt.")
    preterminal_tree = _execution_tree_manifest(
        paths["output"],
        exclude_commit_artifacts=True,
        require_readonly_files=True,
    )
    _validate_passed_preterminal_tree(preterminal_tree)
    declared_tree = authorization.get("source003_prequarantine_execution_tree")
    tree_entries_receipt = {"schema_version": 1, "entries": preterminal_tree["entries"]}
    if (
        not isinstance(declared_tree, Mapping)
        or declared_tree.get("entries") != preterminal_tree["entries"]
        or declared_tree.get("entries_sha256") != QUARANTINE_SOURCE003_TREE_ENTRIES_SHA256
        or canonical_sha256(tree_entries_receipt) != QUARANTINE_SOURCE003_TREE_ENTRIES_SHA256
        or declared_tree.get("media_pair_count") != len(PROMPT_IDS)
    ):
        raise ValueError("Source-003 quarantine preterminal tree drifted.")

    scheduler = authorization.get("source003_scheduler_authentication")
    if (
        not isinstance(scheduler, Mapping)
        or scheduler.get("terminal", {}).get("job_id") != QUARANTINE_SLURM_JOB_ID
        or scheduler.get("terminal", {}).get("job_name") != EQUIVALENCE_SLURM_JOB_NAME
        or scheduler.get("terminal", {}).get("state") != "COMPLETED"
        or scheduler.get("terminal", {}).get("exit_code") != "0:0"
    ):
        raise ValueError("Source-003 quarantine scheduler authentication drifted.")

    sources = _source_preregistration_attempts(audit_root)
    attempts = _attempt_directories_for_source(
        audit_root=audit_root,
        source_attempt=QUARANTINE_SOURCE_ATTEMPT,
    )
    if set(attempts) != {QUARANTINE_EXECUTION_ATTEMPT}:
        raise ValueError("Source-003 quarantine execution namespace drifted.")
    expected_sources = {"001", "002", "003"}
    if require_successor_absent and set(sources) != expected_sources:
        raise ValueError("Source-003 quarantine authorization requires source-004 to be absent.")
    if not require_successor_absent and not expected_sources.issubset(sources):
        raise ValueError("Source-003 quarantine source history is incomplete.")
    if require_successor_absent:
        _require_source004_namespace_absent(audit_root)

    return {
        "authorization": {
            **_artifact_binding(paths["authorization"], authorization_raw),
            "document_sha256": authorization["document_sha256"],
            "sidecar_path": authorization_sidecar["path"],
            "sidecar_sha256": authorization_sidecar["sha256"],
        },
        "payload": deepcopy(authorization),
        "publisher_source": {
            "path": str(publisher_path),
            "sha256": QUARANTINE_PUBLISHER_SHA256,
        },
        "preregistration": {
            **preregistration_binding,
            "source_attempt": QUARANTINE_SOURCE_ATTEMPT,
        },
        "preregistration_payload": preregistration,
        "result": {
            **result_binding,
            "status": "passed",
            "source_attempt": QUARANTINE_SOURCE_ATTEMPT,
            "execution_attempt": QUARANTINE_EXECUTION_ATTEMPT,
        },
        "result_payload": result,
        "preterminal_execution_tree": preterminal_tree,
        "old_file_ledger": deepcopy(dict(old_ledger)),
        "scheduler": deepcopy(dict(scheduler)),
    }


def _validate_quarantine_source_transition(
    *,
    old_files: Mapping[str, Any],
    prospective_files: Mapping[str, Any],
) -> list[dict[str, str]]:
    expected_roles = set(REQUIRED_PREREGISTRATION_FILE_PATHS)
    if set(old_files) != expected_roles or set(prospective_files) != expected_roles:
        raise ValueError("Source-003 quarantine transition role coverage drifted.")
    changes: list[dict[str, str]] = []
    for role in sorted(expected_roles):
        relative = REQUIRED_PREREGISTRATION_FILE_PATHS[role]
        old = old_files.get(role)
        current = prospective_files.get(role)
        if (
            not isinstance(old, Mapping)
            or set(old) != {"path", "sha256"}
            or old.get("path") != str(relative)
            or _SHA256_RE.fullmatch(str(old.get("sha256", ""))) is None
            or not isinstance(current, Mapping)
            or set(current) != {"path", "sha256"}
            or current.get("path") != str(relative)
            or _SHA256_RE.fullmatch(str(current.get("sha256", ""))) is None
        ):
            raise ValueError(f"Source-003 quarantine transition role {role!r} drifted.")
        if old["sha256"] != current["sha256"]:
            changes.append(
                {
                    "role": role,
                    "previous_sha256": str(old["sha256"]),
                    "current_sha256": str(current["sha256"]),
                }
            )
    if [row["role"] for row in changes] != list(QUARANTINE_CHANGED_ROLES):
        raise ValueError(
            "Source-003 quarantine transition must change exactly the three authorized roles."
        )
    return changes


def _build_source003_quarantine_terminal_payload(
    *,
    authorization: Mapping[str, Any],
    prospective_files: Mapping[str, Any],
    scheduler_reauthentication: Mapping[str, Any],
    recorded_at_utc: str,
) -> dict[str, Any]:
    authorized = _aware_timestamp(
        authorization["payload"]["authorized_at_utc"],
        "source-003 quarantine authorization time",
    )
    queried = _aware_timestamp(
        scheduler_reauthentication.get("queried_at_utc"),
        "source-003 quarantine terminal scheduler query",
    )
    recorded = _aware_timestamp(recorded_at_utc, "source-003 quarantine terminal time")
    if recorded > datetime.now(timezone.utc) or not authorized <= queried <= recorded:
        raise ValueError("Source-003 quarantine terminal timestamp ordering drifted.")
    authorized_scheduler = authorization["scheduler"]
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
    if (
        set(authorized_scheduler) != scheduler_fields
        or set(scheduler_reauthentication) != scheduler_fields
        or scheduler_reauthentication.get("schema_version") != 1
        or scheduler_reauthentication.get("returncode") != 0
        or scheduler_reauthentication.get("stderr") != ""
    ):
        raise ValueError("Source-003 quarantine terminal live sacct schema drifted.")
    for field in scheduler_fields - {"queried_at_utc"}:
        if scheduler_reauthentication.get(field) != authorized_scheduler.get(field):
            raise ValueError("Source-003 quarantine terminal live sacct identity drifted.")
    transition = _validate_quarantine_source_transition(
        old_files=authorization["old_file_ledger"],
        prospective_files=prospective_files,
    )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt": QUARANTINE_TERMINAL_RECEIPT_NAME,
        "status": QUARANTINE_TERMINAL_STATUS,
        "recorded_at_utc": recorded_at_utc,
        "protocol_id": PROTOCOL_ID,
        "gate": GATE,
        "source_attempt": QUARANTINE_SOURCE_ATTEMPT,
        "execution_attempt": QUARANTINE_EXECUTION_ATTEMPT,
        "successor_source_attempt": QUARANTINE_SUCCESSOR_SOURCE_ATTEMPT,
        "disposition": _quarantine_disposition(),
        "authorization": deepcopy(dict(authorization["authorization"])),
        "authorization_publisher_source": deepcopy(dict(authorization["publisher_source"])),
        "preregistration": deepcopy(dict(authorization["preregistration"])),
        "result": deepcopy(dict(authorization["result"])),
        "prequarantine_execution_tree": {
            "root": authorization["preterminal_execution_tree"]["root"],
            "entries_sha256": QUARANTINE_SOURCE003_TREE_ENTRIES_SHA256,
            "entry_count": len(authorization["preterminal_execution_tree"]["entries"]),
        },
        "slurm": {
            "runtime": deepcopy(
                dict(authorization["result_payload"]["environment_preflight"]["slurm"])
            ),
            "authorization_scheduler_authentication_sha256": canonical_sha256(authorized_scheduler),
            "terminal_reauthentication": deepcopy(dict(scheduler_reauthentication)),
        },
        "source_transition": {
            "old_source_attempt": QUARANTINE_SOURCE_ATTEMPT,
            "prospective_source_attempt": QUARANTINE_SUCCESSOR_SOURCE_ATTEMPT,
            "changed_roles": transition,
            "invariant_role_count": len(REQUIRED_PREREGISTRATION_FILE_PATHS)
            - len(QUARANTINE_CHANGED_ROLES),
            "prospective_file_ledger": deepcopy(dict(prospective_files)),
            "prospective_file_ledger_sha256": canonical_sha256(prospective_files),
        },
    }
    payload["document_sha256"] = document_sha256(payload)
    return payload


def _validate_quarantine_closed_tree(output_directory: Path) -> None:
    tree = _execution_tree_manifest(
        output_directory,
        exclude_commit_artifacts=False,
        require_readonly_files=True,
    )
    expected = set(_passed_preterminal_names()) | {
        QUARANTINE_TERMINAL_RECEIPT_FILENAME,
        QUARANTINE_TERMINAL_RECEIPT_SIDECAR_FILENAME,
    }
    entries = tree["entries"]
    observed = {
        str(entry.get("relative_path"))
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("kind") == "file"
    }
    if (
        any(not isinstance(entry, Mapping) or entry.get("kind") != "file" for entry in entries)
        or observed != expected
        or len(entries) != len(expected)
    ):
        raise ValueError("Source-003 quarantined directory must contain exactly ten files.")
    _validate_frozen_execution_tree(output_directory)


def _freeze_source003_quarantine_tree_at(
    output_directory_fd: int,
    *,
    expected_sha256: Mapping[str, str],
) -> None:
    """Authenticate and seal the exact flat ten-file tree through one pinned FD."""

    expected_names = set(_passed_preterminal_names()) | {
        QUARANTINE_TERMINAL_RECEIPT_FILENAME,
        QUARANTINE_TERMINAL_RECEIPT_SIDECAR_FILENAME,
    }
    if (
        set(expected_sha256) != expected_names
        or set(os.listdir(output_directory_fd)) != expected_names
    ):
        raise ValueError("Source-003 quarantine commit tree membership drifted before sealing.")
    for name in sorted(expected_names):
        raw, metadata = _read_regular_at(
            output_directory_fd,
            name,
            label=f"source-003 quarantine member {name}",
        )
        if (
            sha256_bytes(raw) != expected_sha256[name]
            or stat.S_IMODE(metadata.st_mode) != 0o444
            or metadata.st_nlink != 1
        ):
            raise ValueError(f"Source-003 quarantine member {name} drifted before sealing.")
        descriptor = os.open(
            name,
            os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0),
            dir_fd=output_directory_fd,
        )
        try:
            opened = os.fstat(descriptor)
            if (
                not stat.S_ISREG(opened.st_mode)
                or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
                or opened.st_nlink != 1
            ):
                raise ValueError(f"Source-003 quarantine member {name} changed before sealing.")
            os.fchmod(descriptor, 0o444)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    os.fchmod(output_directory_fd, 0o555)
    os.fsync(output_directory_fd)
    sealed = os.fstat(output_directory_fd)
    if not stat.S_ISDIR(sealed.st_mode) or stat.S_IMODE(sealed.st_mode) != 0o555:
        raise RuntimeError("Source-003 quarantine directory did not seal to exact mode 0555.")


def validate_source003_quarantine_terminal(
    *,
    project_root: str | Path,
    require_successor_absent: bool,
    require_live_repair_sources: bool,
) -> dict[str, Any]:
    """Validate the distinct nonaccepting source-003 quarantine terminal."""

    project_root = Path(project_root).expanduser().resolve()
    authorization = validate_source003_quarantine_authorization(
        project_root=project_root,
        require_successor_absent=require_successor_absent,
    )
    paths = _source003_quarantine_paths(project_root)
    receipt, receipt_raw = _read_json(
        paths["quarantine_terminal"],
        "source-003 quarantine terminal receipt",
        require_readonly=True,
        trusted_root=paths["audit_root"],
    )
    sidecar = _validate_sidecar(
        payload_path=paths["quarantine_terminal"],
        payload_raw=receipt_raw,
        sidecar_path=Path(f"{paths['quarantine_terminal']}.sha256"),
        label="source-003 quarantine terminal sidecar",
        trusted_root=paths["audit_root"],
    )
    prospective_files = receipt.get("source_transition", {}).get("prospective_file_ledger")
    if not isinstance(prospective_files, Mapping):
        raise ValueError("Source-003 quarantine terminal prospective ledger is malformed.")
    if require_live_repair_sources:
        live_files = _live_preregistration_file_ledger(project_root)
        if prospective_files != live_files:
            raise ValueError("Source-003 quarantine terminal prospective live sources drifted.")
    scheduler = receipt.get("slurm")
    scheduler_reauthentication = (
        scheduler.get("terminal_reauthentication") if isinstance(scheduler, Mapping) else None
    )
    if not isinstance(scheduler_reauthentication, Mapping):
        raise ValueError("Source-003 quarantine terminal scheduler receipt is malformed.")
    expected = _build_source003_quarantine_terminal_payload(
        authorization=authorization,
        prospective_files=prospective_files,
        scheduler_reauthentication=scheduler_reauthentication,
        recorded_at_utc=str(receipt.get("recorded_at_utc", "")),
    )
    if receipt != expected:
        raise ValueError("Source-003 quarantine terminal receipt drifted.")
    _validate_quarantine_closed_tree(paths["output"])
    return {
        "receipt": {
            **_artifact_binding(paths["quarantine_terminal"], receipt_raw),
            "document_sha256": receipt["document_sha256"],
            "sidecar_path": sidecar["path"],
            "sidecar_sha256": sidecar["sha256"],
        },
        "payload": deepcopy(receipt),
        "authorization": deepcopy(authorization),
        "source_attempt": QUARANTINE_SOURCE_ATTEMPT,
        "execution_attempt": QUARANTINE_EXECUTION_ATTEMPT,
        "prospective_file_ledger": deepcopy(dict(prospective_files)),
    }


def publish_source003_quarantine_terminal(
    *,
    project_root: str | Path,
    recorded_at_utc: str | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Publish, once, the nonaccepting quarantine terminal and freeze source-003."""

    project_root = Path(project_root).expanduser().resolve()
    authorization = validate_source003_quarantine_authorization(
        project_root=project_root,
        require_successor_absent=True,
    )
    paths = _source003_quarantine_paths(project_root)
    _require_source004_namespace_absent(paths["audit_root"])
    if stat.S_IMODE(os.lstat(paths["output"]).st_mode) != 0o755:
        raise ValueError("Source-003 must remain exact 0755 before quarantine closure.")
    for candidate in (
        paths["quarantine_terminal"],
        Path(f"{paths['quarantine_terminal']}.sha256"),
    ):
        if os.path.lexists(candidate):
            raise FileExistsError("Source-003 quarantine terminal destination is occupied.")
    publisher = authorization["publisher_source"]["path"]
    if publisher != str(project_root / QUARANTINE_PUBLISHER_RELATIVE):
        raise ValueError("Source-003 quarantine publisher path drifted.")
    from scripts import publish_flux1_source003_quarantine_authorization as publisher_module

    scheduler_reauthentication = publisher_module.query_source003_terminal_slurm(run=run)
    prospective_files = _live_preregistration_file_ledger(project_root)
    recorded_at_utc = recorded_at_utc or datetime.now(timezone.utc).isoformat()
    payload = _build_source003_quarantine_terminal_payload(
        authorization=authorization,
        prospective_files=prospective_files,
        scheduler_reauthentication=scheduler_reauthentication,
        recorded_at_utc=recorded_at_utc,
    )
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    sidecar_raw = f"{sha256_bytes(raw)}  {QUARANTINE_TERMINAL_RECEIPT_FILENAME}\n".encode("utf-8")
    expected_tree_sha256 = {
        str(entry["relative_path"]): str(entry["sha256"])
        for entry in authorization["preterminal_execution_tree"]["entries"]
    }
    expected_tree_sha256[QUARANTINE_TERMINAL_RECEIPT_FILENAME] = sha256_bytes(raw)
    expected_tree_sha256[QUARANTINE_TERMINAL_RECEIPT_SIDECAR_FILENAME] = sha256_bytes(sidecar_raw)

    def require_quiescent_quarantine_inputs() -> None:
        if (
            validate_source003_quarantine_authorization(
                project_root=project_root,
                require_successor_absent=True,
            )
            != authorization
        ):
            raise ValueError("Source-003 authenticated evidence drifted before quarantine commit.")
        if _live_preregistration_file_ledger(project_root) != prospective_files:
            raise ValueError("Source-004 repair sources drifted before quarantine commit.")
        _require_source004_namespace_absent(paths["audit_root"])
        fresh_scheduler = publisher_module.query_source003_terminal_slurm(run=run)
        stable_scheduler_fields = {
            "schema_version",
            "command",
            "environment",
            "returncode",
            "stdout",
            "stdout_sha256",
            "stderr",
            "rows",
            "terminal",
        }
        if set(fresh_scheduler) != stable_scheduler_fields | {"queried_at_utc"} or any(
            fresh_scheduler.get(field) != scheduler_reauthentication.get(field)
            for field in stable_scheduler_fields
        ):
            raise ValueError("Source-003 live sacct state drifted before quarantine commit.")

    def seal_committed_quarantine(output_directory_fd: int) -> None:
        if _live_preregistration_file_ledger(project_root) != prospective_files:
            raise ValueError("Source-004 repair sources drifted during quarantine commit.")
        _require_source004_namespace_absent(paths["audit_root"])
        _freeze_source003_quarantine_tree_at(
            output_directory_fd,
            expected_sha256=expected_tree_sha256,
        )

    _publish_readonly_pair_no_overwrite(
        payload_path=paths["quarantine_terminal"],
        payload_raw=raw,
        sidecar_raw=sidecar_raw,
        label="Source-003 quarantine terminal",
        staging_parent=paths["audit_root"],
        pre_commit=require_quiescent_quarantine_inputs,
        post_commit=seal_committed_quarantine,
    )
    try:
        _require_source004_namespace_absent(paths["audit_root"])
        return validate_source003_quarantine_terminal(
            project_root=project_root,
            require_successor_absent=True,
            require_live_repair_sources=True,
        )
    except BaseException as exc:
        raise RuntimeError(
            "Source-003 quarantine terminal committed but closure validation is ambiguous; "
            "do not replay it."
        ) from exc


def _validate_passed_execution_members(output_directory: Path, *, accepted: bool) -> None:
    tree = _execution_tree_manifest(
        output_directory,
        exclude_commit_artifacts=False,
        require_readonly_files=True,
    )
    expected = set(_passed_preterminal_names()) | {
        TERMINAL_RECEIPT_FILENAME,
        TERMINAL_RECEIPT_SIDECAR_FILENAME,
    }
    if accepted:
        expected.update({RECEIPT_FILENAME, RECEIPT_SIDECAR_FILENAME})
    entries = tree["entries"]
    observed = {
        str(entry.get("relative_path"))
        for entry in entries
        if isinstance(entry, Mapping) and entry.get("kind") == "file"
    }
    if (
        any(entry.get("kind") != "file" for entry in entries)
        or observed != expected
        or len(entries) != len(observed)
    ):
        phase = "accepted" if accepted else "terminal pre-acceptance"
        raise ValueError(f"Passed equivalence {phase} directory membership drifted.")


def _freeze_execution_tree(output_directory: Path) -> None:
    """Commit one existing tree by sealing files 0444 and directories 0555."""

    root = Path(output_directory)

    def freeze(directory: Path) -> None:
        try:
            children = sorted(os.scandir(directory), key=lambda item: item.name)
        except OSError as exc:
            raise ValueError(f"Cannot enumerate execution tree for freezing: {directory}.") from exc
        directories: list[Path] = []
        for child in children:
            path = Path(child.path)
            metadata = child.stat(follow_symlinks=False)
            if stat.S_ISLNK(metadata.st_mode):
                raise ValueError(f"Cannot freeze symlinked execution evidence: {path}.")
            if stat.S_ISDIR(metadata.st_mode):
                directories.append(path)
                continue
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError(f"Cannot freeze special or multiply linked evidence: {path}.")
            flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
            descriptor = os.open(path, flags)
            try:
                opened = os.fstat(descriptor)
                if (
                    not stat.S_ISREG(opened.st_mode)
                    or (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
                    or opened.st_nlink != 1
                ):
                    raise ValueError(f"Execution evidence changed before freezing: {path}.")
                os.fchmod(descriptor, 0o444)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        for child_directory in directories:
            freeze(child_directory)
        descriptor = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fchmod(descriptor, 0o555)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    freeze(root)
    _validate_frozen_execution_tree(root)


def _validate_frozen_execution_tree(output_directory: Path) -> None:
    root = Path(output_directory)
    # The authenticated walk rejects symlinks, special files, hard links, and
    # writable files before exact directory modes are checked below.
    _execution_tree_manifest(
        root,
        exclude_commit_artifacts=False,
        require_readonly_files=True,
    )

    def validate_directory(directory: Path) -> None:
        metadata = os.lstat(directory)
        if (
            stat.S_ISLNK(metadata.st_mode)
            or not stat.S_ISDIR(metadata.st_mode)
            or stat.S_IMODE(metadata.st_mode) != 0o555
        ):
            raise ValueError(f"Committed execution directory is not mode 0555: {directory}.")
        for child in os.scandir(directory):
            child_metadata = child.stat(follow_symlinks=False)
            path = Path(child.path)
            if stat.S_ISDIR(child_metadata.st_mode):
                validate_directory(path)
            elif (
                not stat.S_ISREG(child_metadata.st_mode)
                or child_metadata.st_nlink != 1
                or stat.S_IMODE(child_metadata.st_mode) != 0o444
            ):
                raise ValueError(f"Committed execution file is not exact mode 0444: {path}.")

    validate_directory(root)


def _terminal_receipt_binding(admission: Mapping[str, Any]) -> dict[str, Any]:
    receipt = admission["receipt"]
    return {
        "path": receipt["path"],
        "sha256": receipt["sha256"],
        "document_sha256": receipt["document_sha256"],
        "sidecar_path": receipt["sidecar_path"],
        "sidecar_sha256": receipt["sidecar_sha256"],
        "source_attempt": admission["source_attempt"],
        "execution_attempt": admission["execution_attempt"],
        "result_status": admission["result"]["status"],
        "recorded_at_utc": admission["payload"]["recorded_at_utc"],
    }


def _acceptance_receipt_binding(admission: Mapping[str, Any]) -> dict[str, Any]:
    receipt = admission["receipt"]
    attempt = admission["payload"]["attempt_lineage"]
    return {
        "path": receipt["path"],
        "sha256": receipt["sha256"],
        "document_sha256": receipt["document_sha256"],
        "sidecar_path": receipt["sidecar_path"],
        "sidecar_sha256": receipt["sidecar_sha256"],
        "source_attempt": attempt["source_attempt"],
        "execution_attempt": attempt["execution_attempt"],
        "status": admission["payload"]["status"],
    }


def _build_terminal_receipt_payload(
    *,
    evidence: Mapping[str, Any],
    scheduler_query: Mapping[str, Any],
    predecessor_terminal_receipt: Mapping[str, Any] | None,
    preterminal_execution_tree: Mapping[str, Any],
    recorded_at_utc: str,
) -> dict[str, Any]:
    result_status = str(evidence["result"]["status"])
    scheduler_query = _validate_scheduler_query_evidence(
        scheduler_query,
        expected_job_id=str(evidence["runtime_slurm"]["job_id"]),
        result_status=result_status,
    )
    terminal = scheduler_query["terminal"]
    if (
        terminal["job_id"] != evidence["runtime_slurm"]["job_id"]
        or terminal["job_name"] != evidence["runtime_slurm"]["job_name"]
    ):
        raise ValueError("Terminal scheduler identity differs from the execution result.")
    preregistered = _aware_timestamp(
        evidence["preregistration"]["payload"]["created_at_utc"],
        "terminal receipt preregistration created_at_utc",
    )
    submitted = _aware_timestamp(terminal["submitted_at"], "terminal receipt Submit")
    scheduler_started = _aware_timestamp(terminal["started_at"], "terminal receipt Start")
    result_started = _aware_timestamp(evidence["started_at"], "terminal receipt result started_at")
    result_ended = _aware_timestamp(evidence["ended_at"], "terminal receipt result ended_at")
    scheduler_ended = _aware_timestamp(terminal["ended_at"], "terminal receipt End")
    queried = _aware_timestamp(scheduler_query["queried_at_utc"], "terminal receipt queried_at_utc")
    recorded = _aware_timestamp(recorded_at_utc, "terminal receipt recorded_at_utc")
    if recorded > datetime.now(timezone.utc):
        raise ValueError("Terminal receipt recorded_at_utc is in the future.")
    if predecessor_terminal_receipt is not None:
        if predecessor_terminal_receipt.get("result_status") not in {
            "failed",
            "error",
        }:
            raise ValueError("A passed equivalence execution cannot be retried.")
        predecessor_recorded = _aware_timestamp(
            predecessor_terminal_receipt.get("recorded_at_utc"),
            "predecessor terminal receipt recorded_at_utc",
        )
        if predecessor_recorded > submitted:
            raise ValueError("Retry was submitted before its predecessor became terminal.")
    if not (
        preregistered
        <= submitted
        <= scheduler_started
        <= result_started
        <= result_ended
        <= scheduler_ended
        <= queried
        <= recorded
    ):
        raise ValueError("Terminal receipt timestamp ordering is invalid.")
    attempt = evidence["attempt_lineage"]
    preregistration = evidence["preregistration"]
    launcher = preregistration["authenticated_files"]["slurm_launcher"]
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt": TERMINAL_RECEIPT_NAME,
        "status": TERMINAL_STATUS,
        "recorded_at_utc": recorded_at_utc,
        "protocol_id": PROTOCOL_ID,
        "gate": GATE,
        "source_attempt": attempt["source_attempt"],
        "execution_attempt": attempt["execution_attempt"],
        "attempt_lineage": deepcopy(dict(attempt)),
        "result": deepcopy(dict(evidence["result"])),
        "preregistration": {
            key: preregistration[key]
            for key in (
                "path",
                "raw_sha256",
                "document_sha256",
                "sidecar_path",
                "sidecar_sha256",
                "source_attempt",
                "lineage",
            )
        }
        | {"created_at_utc": preregistration["payload"]["created_at_utc"]},
        "slurm_launcher": deepcopy(dict(launcher)),
        "slurm": {
            "runtime": deepcopy(dict(evidence["runtime_slurm"])),
            "scheduler_query": deepcopy(dict(scheduler_query)),
        },
        "predecessor_terminal_receipt": (
            deepcopy(dict(predecessor_terminal_receipt))
            if predecessor_terminal_receipt is not None
            else None
        ),
        "preterminal_execution_tree": deepcopy(dict(preterminal_execution_tree)),
    }
    payload["document_sha256"] = document_sha256(payload)
    return payload


def validate_terminal_receipt(
    *,
    result_path: str | Path,
    project_root: str | Path,
    receipt_path: str | Path | None = None,
    receipt_sidecar_path: str | Path | None = None,
    require_latest: bool = True,
) -> dict[str, Any]:
    """Authenticate one terminal execution and its contiguous retry chain."""

    project_root = Path(project_root).expanduser().resolve()
    audit_root = (project_root / AUDIT_ROOT_RELATIVE).resolve()
    path = Path(result_path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    source_attempt, execution_attempt = parse_result_attempts(path.parent)
    if source_attempt == QUARANTINE_SOURCE_ATTEMPT:
        raise ValueError("Source-003 forbids an ordinary terminal receipt.")
    evidence = validate_terminal_execution_result(result_path=path, project_root=project_root)
    if evidence["result"]["status"] in {"failed", "error"} and any(
        os.path.lexists(candidate)
        for candidate in (
            path.parent / RECEIPT_FILENAME,
            path.parent / RECEIPT_SIDECAR_FILENAME,
        )
    ):
        raise ValueError("A non-passing execution has an ambiguous acceptance artifact.")
    expected_receipt_path = terminal_receipt_path(path.parent)
    receipt_path = (
        Path(receipt_path).expanduser() if receipt_path is not None else expected_receipt_path
    )
    if not receipt_path.is_absolute():
        receipt_path = project_root / receipt_path
    if receipt_path != expected_receipt_path:
        raise ValueError("Terminal receipt path is noncanonical for its execution.")
    sidecar_path = (
        Path(receipt_sidecar_path).expanduser()
        if receipt_sidecar_path is not None
        else Path(f"{receipt_path}.sha256")
    )
    if not sidecar_path.is_absolute():
        sidecar_path = project_root / sidecar_path
    receipt, raw = _read_json(
        receipt_path,
        "execution terminal receipt",
        require_readonly=True,
        trusted_root=audit_root,
    )
    sidecar = _validate_sidecar(
        payload_path=receipt_path,
        payload_raw=raw,
        sidecar_path=sidecar_path,
        label="execution terminal receipt sidecar",
        trusted_root=audit_root,
    )
    if execution_attempt == "001":
        predecessor = None
    else:
        previous_execution = f"{int(execution_attempt) - 1:03d}"
        previous_result = (
            audit_root / result_directory_name(source_attempt, previous_execution) / RESULT_FILENAME
        )
        predecessor_admission = validate_terminal_receipt(
            result_path=previous_result,
            project_root=project_root,
            require_latest=False,
        )
        predecessor = _terminal_receipt_binding(predecessor_admission)
    scheduler_query_raw = receipt.get("slurm")
    scheduler_query = (
        scheduler_query_raw.get("scheduler_query")
        if isinstance(scheduler_query_raw, Mapping)
        else None
    )
    preterminal_tree = _execution_tree_manifest(
        path.parent,
        exclude_commit_artifacts=True,
        require_readonly_files=True,
    )
    if evidence["result"]["status"] == "passed":
        _validate_passed_preterminal_tree(preterminal_tree)
    expected = _build_terminal_receipt_payload(
        evidence=evidence,
        scheduler_query=scheduler_query,
        predecessor_terminal_receipt=predecessor,
        preterminal_execution_tree=preterminal_tree,
        recorded_at_utc=str(receipt.get("recorded_at_utc", "")),
    )
    if receipt != expected:
        raise ValueError("Terminal receipt differs from its immutable execution chain.")
    attempts = _attempt_directories_for_source(audit_root=audit_root, source_attempt=source_attempt)
    required_attempts = {f"{index:03d}" for index in range(1, int(execution_attempt) + 1)}
    if not required_attempts.issubset(attempts):
        raise ValueError("Execution-attempt chain contains a gap.")
    if require_latest:
        if set(attempts) != required_attempts:
            raise ValueError("Terminal receipt is not the latest occupied execution attempt.")
        sources = _source_preregistration_attempts(audit_root)
        if not sources or max(sources) != source_attempt:
            raise ValueError("Terminal receipt is not from the latest source attempt.")
    if evidence["result"]["status"] in {"failed", "error"}:
        _validate_frozen_execution_tree(path.parent)
    return {
        "receipt": {
            **_artifact_binding(receipt_path, raw),
            "sidecar_path": sidecar["path"],
            "sidecar_sha256": sidecar["sha256"],
            "document_sha256": receipt["document_sha256"],
        },
        "payload": deepcopy(receipt),
        "result": deepcopy(evidence["result"]),
        "preregistration": deepcopy(evidence["preregistration"]),
        "source_attempt": source_attempt,
        "execution_attempt": execution_attempt,
        "slurm": deepcopy(dict(receipt["slurm"])),
        "predecessor_terminal_receipt": deepcopy(predecessor),
    }


def publish_terminal_receipt(
    *,
    result_path: str | Path,
    project_root: str | Path,
    receipt_path: str | Path | None = None,
    recorded_at_utc: str | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Query sacct and commit one no-overwrite terminal retry-chain receipt."""

    project_root = Path(project_root).expanduser().resolve()
    audit_root = (project_root / AUDIT_ROOT_RELATIVE).resolve()
    result_path = Path(result_path).expanduser()
    if not result_path.is_absolute():
        result_path = project_root / result_path
    source_attempt, execution_attempt = parse_result_attempts(result_path.parent)
    if source_attempt == QUARANTINE_SOURCE_ATTEMPT:
        raise ValueError("Source-003 forbids ordinary terminal-receipt publication.")
    attempts = _attempt_directories_for_source(audit_root=audit_root, source_attempt=source_attempt)
    expected_attempts = {f"{index:03d}" for index in range(1, int(execution_attempt) + 1)}
    if set(attempts) != expected_attempts:
        raise ValueError("Cannot terminalize a gapped or nonlatest occupied execution attempt.")
    sources = _source_preregistration_attempts(audit_root)
    if not sources or max(sources) != source_attempt:
        raise ValueError("Cannot terminalize an execution from a superseded source attempt.")
    evidence = validate_terminal_execution_result(
        result_path=result_path, project_root=project_root
    )
    if execution_attempt == "001":
        predecessor = None
    else:
        previous_execution = f"{int(execution_attempt) - 1:03d}"
        predecessor = _terminal_receipt_binding(
            validate_terminal_receipt(
                result_path=(
                    audit_root
                    / result_directory_name(source_attempt, previous_execution)
                    / RESULT_FILENAME
                ),
                project_root=project_root,
                require_latest=False,
            )
        )
    scheduler_query = query_terminal_slurm_evidence(
        job_id=str(evidence["runtime_slurm"]["job_id"]),
        result_status=str(evidence["result"]["status"]),
        run=run,
    )
    preterminal_tree = _execution_tree_manifest(
        result_path.parent,
        exclude_commit_artifacts=True,
        require_readonly_files=evidence["result"]["status"] == "passed",
    )
    if evidence["result"]["status"] == "passed":
        _validate_passed_preterminal_tree(preterminal_tree)
    receipt_path = (
        Path(receipt_path).expanduser()
        if receipt_path is not None
        else terminal_receipt_path(result_path.parent)
    )
    if not receipt_path.is_absolute():
        receipt_path = project_root / receipt_path
    if receipt_path != terminal_receipt_path(result_path.parent):
        raise ValueError("Terminal receipt must use its canonical result-local filename.")
    recorded_at_utc = recorded_at_utc or datetime.now(timezone.utc).isoformat()
    payload = _build_terminal_receipt_payload(
        evidence=evidence,
        scheduler_query=scheduler_query,
        predecessor_terminal_receipt=predecessor,
        preterminal_execution_tree=preterminal_tree,
        recorded_at_utc=recorded_at_utc,
    )
    payload_raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    sidecar_raw = f"{sha256_bytes(payload_raw)}  {receipt_path.name}\n".encode("utf-8")
    _publish_readonly_pair_no_overwrite(
        payload_path=receipt_path,
        payload_raw=payload_raw,
        sidecar_raw=sidecar_raw,
        label="Execution terminal receipt",
        staging_parent=audit_root,
    )
    if evidence["result"]["status"] in {"failed", "error"}:
        _freeze_execution_tree(result_path.parent)
    return validate_terminal_receipt(
        result_path=result_path,
        receipt_path=receipt_path,
        project_root=project_root,
        require_latest=True,
    )


def validate_prelaunch_attempt(
    *,
    preregistration_path: str | Path,
    output_directory: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Authorize exactly the next execution attempt before output-dir creation."""

    project_root = Path(project_root).expanduser().resolve()
    audit_root = (project_root / AUDIT_ROOT_RELATIVE).resolve()
    preregistration_path = Path(preregistration_path).expanduser()
    output_directory = Path(output_directory).expanduser()
    if not preregistration_path.is_absolute():
        preregistration_path = project_root / preregistration_path
    if not output_directory.is_absolute():
        output_directory = project_root / output_directory
    source_attempt = parse_preregistration_source_attempt(preregistration_path)
    output_source, execution_attempt = parse_result_attempts(output_directory)
    if source_attempt == QUARANTINE_SOURCE_ATTEMPT:
        raise ValueError("Source-003 quarantine forbids every new prelaunch or same-source retry.")
    if (
        source_attempt != output_source
        or preregistration_path != audit_root / preregistration_filename(source_attempt)
        or output_directory != audit_root / result_directory_name(source_attempt, execution_attempt)
        or os.path.lexists(output_directory)
    ):
        raise ValueError(
            "Prelaunch source/execution paths are mismatched, occupied, or noncanonical."
        )
    sources = _source_preregistration_attempts(audit_root)
    expected_sources = {f"{index:03d}" for index in range(1, int(source_attempt) + 1)}
    if set(sources) != expected_sources:
        raise ValueError("Prelaunch source-attempt namespace is gapped or superseded.")
    preregistration = load_preregistration(preregistration_path, project_root)
    attempts = _attempt_directories_for_source(audit_root=audit_root, source_attempt=source_attempt)
    expected_prior_attempts = {f"{index:03d}" for index in range(1, int(execution_attempt))}
    if set(attempts) != expected_prior_attempts:
        raise ValueError(
            "Prelaunch execution attempts are gapped, nonterminal, or not the next attempt."
        )
    if execution_attempt == "001":
        predecessor_terminal = None
    else:
        previous_execution = f"{int(execution_attempt) - 1:03d}"
        predecessor_terminal = _terminal_receipt_binding(
            validate_terminal_receipt(
                result_path=(
                    audit_root
                    / result_directory_name(source_attempt, previous_execution)
                    / RESULT_FILENAME
                ),
                project_root=project_root,
                require_latest=True,
            )
        )
        if predecessor_terminal["result_status"] == "passed":
            raise ValueError("A passed equivalence execution cannot be retried.")
    return {
        "schema_version": 1,
        "status": "authorized_before_output_creation",
        "protocol_id": PROTOCOL_ID,
        "source_attempt": source_attempt,
        "execution_attempt": execution_attempt,
        "preregistration": {
            "path": preregistration["path"],
            "raw_sha256": preregistration["raw_sha256"],
            "sidecar_path": preregistration["sidecar_path"],
            "sidecar_sha256": preregistration["sidecar_sha256"],
        },
        "output_directory": str(output_directory),
        "predecessor_terminal_receipt": predecessor_terminal,
    }


def _sacct_command(job_id: str) -> list[str]:
    return [
        "sacct",
        "--noheader",
        "--parsable2",
        "--jobs",
        job_id,
        "--format",
        "JobIDRaw,JobName%128,State,ExitCode,Submit,Start,End",
    ]


def _parse_sacct_stdout(
    stdout: str, *, expected_job_id: str, result_status: str = "passed"
) -> tuple[int, dict[str, str]]:
    rows: list[tuple[int, dict[str, str]]] = []
    for index, line in enumerate(stdout.splitlines()):
        if not line.strip():
            continue
        parts = line.split("|")
        if len(parts) == len(SACCT_FIELDS) + 1 and parts[-1] == "":
            parts.pop()
        if len(parts) != len(SACCT_FIELDS):
            raise ValueError(f"sacct row {index} does not contain exactly seven fields.")
        row = {key: value.strip() for key, value in zip(SACCT_FIELDS, parts, strict=True)}
        if row["JobIDRaw"] == expected_job_id:
            rows.append((index, row))
    if len(rows) != 1:
        raise ValueError("sacct output must contain exactly one row for the parent job ID.")
    index, row = rows[0]
    expected_terminal = {
        "passed": ("COMPLETED", "0:0"),
        "failed": ("FAILED", "1:0"),
        "error": ("FAILED", "1:0"),
    }.get(result_status)
    if expected_terminal is None:
        raise ValueError(f"Unsupported equivalence result status: {result_status!r}.")
    if (
        row["JobName"] != EQUIVALENCE_SLURM_JOB_NAME
        or (row["State"], row["ExitCode"]) != expected_terminal
    ):
        raise ValueError("sacct parent row does not exactly match the diagnostic result status.")
    submitted = _aware_timestamp(row["Submit"], "sacct Submit")
    started = _aware_timestamp(row["Start"], "sacct Start")
    ended = _aware_timestamp(row["End"], "sacct End")
    if not submitted <= started <= ended:
        raise ValueError("sacct terminal timestamps are not nondecreasing.")
    return index, {
        "job_id": row["JobIDRaw"],
        "job_name": row["JobName"],
        "state": row["State"],
        "exit_code": row["ExitCode"],
        "submitted_at": row["Submit"],
        "started_at": row["Start"],
        "ended_at": row["End"],
    }


def _validate_scheduler_query_evidence(
    raw: Any, *, expected_job_id: str, result_status: str = "passed"
) -> dict[str, Any]:
    expected_fields = {
        "command",
        "environment",
        "queried_at_utc",
        "returncode",
        "stdout",
        "stdout_sha256",
        "stderr",
        "stderr_sha256",
        "selected_row_index",
        "terminal",
    }
    if not isinstance(raw, Mapping) or set(raw) != expected_fields:
        raise ValueError("Scheduler-query evidence schema drifted.")
    stdout = raw.get("stdout")
    stderr = raw.get("stderr")
    if (
        raw.get("command") != _sacct_command(expected_job_id)
        or raw.get("environment") != {"SLURM_TIME_FORMAT": SACCT_TIME_FORMAT, "TZ": "UTC"}
        or raw.get("returncode") != 0
        or not isinstance(stdout, str)
        or not isinstance(stderr, str)
        or raw.get("stdout_sha256") != sha256_bytes(stdout.encode("utf-8"))
        or raw.get("stderr_sha256") != sha256_bytes(stderr.encode("utf-8"))
    ):
        raise ValueError("Scheduler-query command/log binding drifted.")
    _aware_timestamp(raw.get("queried_at_utc"), "scheduler queried_at_utc")
    selected_index, terminal = _parse_sacct_stdout(
        stdout,
        expected_job_id=expected_job_id,
        result_status=result_status,
    )
    if raw.get("selected_row_index") != selected_index or raw.get("terminal") != terminal:
        raise ValueError("Scheduler selected-row evidence drifted from preserved stdout.")
    return deepcopy(dict(raw))


def query_terminal_slurm_evidence(
    *,
    job_id: str,
    result_status: str = "passed",
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Query sacct directly; callers cannot assert terminal scheduler booleans."""

    if not str(job_id).isdigit():
        raise ValueError("Equivalence result Slurm job ID must be numeric before querying sacct.")
    command = _sacct_command(str(job_id))
    environment = {**os.environ, "SLURM_TIME_FORMAT": SACCT_TIME_FORMAT, "TZ": "UTC"}
    completed = run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
        env=environment,
    )
    stdout = str(getattr(completed, "stdout", ""))
    stderr = str(getattr(completed, "stderr", ""))
    returncode = int(getattr(completed, "returncode", -1))
    selected_index, terminal = _parse_sacct_stdout(
        stdout,
        expected_job_id=str(job_id),
        result_status=result_status,
    )
    evidence = {
        "command": command,
        "environment": {"SLURM_TIME_FORMAT": SACCT_TIME_FORMAT, "TZ": "UTC"},
        "queried_at_utc": datetime.now(timezone.utc).isoformat(),
        "returncode": returncode,
        "stdout": stdout,
        "stdout_sha256": sha256_bytes(stdout.encode("utf-8")),
        "stderr": stderr,
        "stderr_sha256": sha256_bytes(stderr.encode("utf-8")),
        "selected_row_index": selected_index,
        "terminal": terminal,
    }
    return _validate_scheduler_query_evidence(
        evidence,
        expected_job_id=str(job_id),
        result_status=result_status,
    )


def _build_receipt_payload(
    *,
    evidence: Mapping[str, Any],
    scheduler_query: Mapping[str, Any],
    terminal_execution_receipt: Mapping[str, Any],
    accepted_at_utc: str,
) -> dict[str, Any]:
    scheduler_query = _validate_scheduler_query_evidence(
        scheduler_query, expected_job_id=str(evidence["runtime_slurm"]["job_id"])
    )
    terminal_slurm = scheduler_query["terminal"]
    accepted_at = _aware_timestamp(accepted_at_utc, "receipt accepted_at_utc")
    terminal_recorded_at = _aware_timestamp(
        terminal_execution_receipt.get("recorded_at_utc"),
        "terminal execution receipt recorded_at_utc",
    )
    if accepted_at > datetime.now(timezone.utc):
        raise ValueError("Acceptance receipt accepted_at_utc is in the future.")
    if terminal_slurm["job_id"] != evidence["runtime_slurm"]["job_id"]:
        raise ValueError("Terminal Slurm job ID differs from the H100 result identity.")
    if terminal_slurm["job_name"] != evidence["runtime_slurm"]["job_name"]:
        raise ValueError("Terminal Slurm job name differs from the H100 result identity.")
    scheduler_submitted = _aware_timestamp(terminal_slurm["submitted_at"], "Slurm submitted_at")
    preregistered_at = _aware_timestamp(
        evidence["preregistration"]["created_at_utc"],
        "preregistration created_at_utc",
    )
    scheduler_started = _aware_timestamp(terminal_slurm["started_at"], "Slurm started_at")
    result_started = _aware_timestamp(evidence["started_at"], "result started_at")
    result_ended = _aware_timestamp(evidence["ended_at"], "result ended_at")
    scheduler_ended = _aware_timestamp(terminal_slurm["ended_at"], "Slurm ended_at")
    queried_at = _aware_timestamp(scheduler_query["queried_at_utc"], "scheduler queried_at_utc")
    if not (
        preregistered_at
        <= scheduler_submitted
        <= scheduler_started
        <= result_started
        <= result_ended
        <= scheduler_ended
        <= queried_at
        <= terminal_recorded_at
        <= accepted_at
    ):
        raise ValueError("Receipt/Slurm/result timestamp ordering is invalid.")
    payload: dict[str, Any] = {
        "schema_version": 1,
        "receipt": RECEIPT_NAME,
        "status": ACCEPTED_STATUS,
        "accepted_at_utc": accepted_at_utc,
        "protocol_id": PROTOCOL_ID,
        "gate": GATE,
        "attempt_lineage": deepcopy(evidence["attempt_lineage"]),
        "result": deepcopy(evidence["result"]),
        "preregistration": deepcopy(evidence["preregistration"]),
        "terminal_execution_receipt": deepcopy(dict(terminal_execution_receipt)),
        "slurm_launcher": deepcopy(evidence["slurm_launcher"]),
        "slurm": {
            "gpu_name": evidence["gpu_name"],
            "scheduler_query": deepcopy(dict(scheduler_query)),
        },
        "acceptance": {
            "result_status": "passed",
            "prompt_ids": list(PROMPT_IDS),
            "comparisons_total": COMPARISON_COUNT,
            "comparisons_passed": COMPARISON_COUNT,
            "failed_comparison_count": 0,
            "failed_comparison_ids": [],
            "media_count": MEDIA_COUNT,
            "runner_registry_construction_preflight_status": "passed",
            "independent_t5_sentinels_passed": len(PROMPT_IDS),
        },
        "media": deepcopy(evidence["media"]),
    }
    payload["document_sha256"] = document_sha256(payload)
    return payload


def validate_acceptance_receipt(
    *,
    result_path: str | Path,
    result_sidecar_path: str | Path | None = None,
    receipt_path: str | Path | None = None,
    receipt_sidecar_path: str | Path | None = None,
    project_root: str | Path,
    require_latest: bool = True,
    require_live_preregistration_sources: bool = True,
) -> dict[str, Any]:
    """Reauthenticate an immutable post-Slurm receipt against all live evidence."""

    candidate_result = Path(result_path).expanduser()
    if not candidate_result.is_absolute():
        candidate_result = Path(project_root).resolve() / candidate_result
    source_attempt, execution_attempt = parse_result_attempts(candidate_result.parent)
    if source_attempt == QUARANTINE_SOURCE_ATTEMPT:
        raise ValueError("Source-003 quarantine permanently forbids acceptance validation.")
    evidence = validate_equivalence_result(
        result_path=result_path,
        result_sidecar_path=result_sidecar_path,
        project_root=project_root,
        require_live_preregistration_sources=require_live_preregistration_sources,
    )
    result_path = Path(evidence["result"]["path"])
    receipt_path = (
        Path(receipt_path).expanduser()
        if receipt_path is not None
        else result_path.parent / RECEIPT_FILENAME
    )
    if not receipt_path.is_absolute():
        receipt_path = Path(project_root).resolve() / receipt_path
    expected_receipt_path = result_path.parent / RECEIPT_FILENAME
    if receipt_path != expected_receipt_path:
        raise ValueError("Acceptance receipt path is noncanonical for this result.")
    sidecar_path = (
        Path(receipt_sidecar_path).expanduser()
        if receipt_sidecar_path is not None
        else Path(f"{receipt_path}.sha256")
    )
    if not sidecar_path.is_absolute():
        sidecar_path = Path(project_root).resolve() / sidecar_path
    receipt, receipt_raw = _read_json(
        receipt_path,
        "equivalence acceptance receipt",
        require_readonly=True,
        trusted_root=result_path.parent,
    )
    receipt_sidecar = _validate_sidecar(
        payload_path=receipt_path,
        payload_raw=receipt_raw,
        sidecar_path=sidecar_path,
        label="equivalence acceptance receipt sidecar",
        trusted_root=result_path.parent,
    )
    expected_fields = {
        "schema_version",
        "receipt",
        "status",
        "accepted_at_utc",
        "protocol_id",
        "gate",
        "attempt_lineage",
        "result",
        "preregistration",
        "terminal_execution_receipt",
        "slurm_launcher",
        "slurm",
        "acceptance",
        "media",
        "document_sha256",
    }
    if (
        set(receipt) != expected_fields
        or receipt.get("schema_version") != 1
        or receipt.get("receipt") != RECEIPT_NAME
        or receipt.get("status") != ACCEPTED_STATUS
        or receipt.get("protocol_id") != PROTOCOL_ID
        or receipt.get("gate") != GATE
        or receipt.get("document_sha256") != document_sha256(receipt)
    ):
        raise ValueError("Equivalence acceptance receipt identity/document digest drifted.")
    slurm = receipt.get("slurm")
    if (
        not isinstance(slurm, Mapping)
        or set(slurm) != {"gpu_name", "scheduler_query"}
        or slurm.get("gpu_name") != evidence["gpu_name"]
    ):
        raise ValueError("Receipt Slurm schema drifted.")
    scheduler_query = _validate_scheduler_query_evidence(
        slurm.get("scheduler_query"),
        expected_job_id=str(evidence["runtime_slurm"]["job_id"]),
    )
    terminal_admission = validate_terminal_receipt(
        result_path=result_path,
        project_root=project_root,
        require_latest=require_latest,
    )
    terminal_binding = _terminal_receipt_binding(terminal_admission)
    if (
        receipt.get("terminal_execution_receipt") != terminal_binding
        or scheduler_query != terminal_admission["slurm"]["scheduler_query"]
    ):
        raise ValueError("Acceptance receipt terminal-execution binding drifted.")
    expected = _build_receipt_payload(
        evidence=evidence,
        scheduler_query=scheduler_query,
        terminal_execution_receipt=terminal_binding,
        accepted_at_utc=str(receipt.get("accepted_at_utc", "")),
    )
    if receipt != expected:
        raise ValueError("Acceptance receipt differs from the authenticated live result/state.")
    _validate_passed_execution_members(result_path.parent, accepted=True)
    _validate_frozen_execution_tree(result_path.parent)
    return {
        "receipt": {
            **_artifact_binding(receipt_path, receipt_raw),
            "sidecar_path": receipt_sidecar["path"],
            "sidecar_sha256": receipt_sidecar["sha256"],
            "document_sha256": receipt["document_sha256"],
        },
        "payload": deepcopy(receipt),
        "result": deepcopy(evidence["result"]),
        "preregistration": deepcopy(evidence["preregistration"]),
        "terminal_execution_receipt": terminal_binding,
        "slurm": deepcopy(dict(receipt["slurm"])),
        "acceptance": deepcopy(dict(receipt["acceptance"])),
        "media": deepcopy(evidence["media"]),
    }


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
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _directory_component_identities(path: Path, *, label: str) -> tuple[tuple[Path, int, int], ...]:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} must be lexically absolute without parent traversal.")
    identities: list[tuple[Path, int, int]] = []
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current /= part
        observed = os.lstat(current)
        if stat.S_ISLNK(observed.st_mode) or not stat.S_ISDIR(observed.st_mode):
            raise ValueError(f"{label} contains a non-directory or symlink: {current}.")
        identities.append((current, observed.st_dev, observed.st_ino))
    return tuple(identities)


def _open_authenticated_directory(
    path: Path, *, label: str
) -> tuple[int, tuple[int, int], tuple[tuple[Path, int, int], ...]]:
    before_components = _directory_component_identities(path, label=label)
    before = os.lstat(path)
    flags = (
        os.O_RDONLY
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags)
    opened = os.fstat(descriptor)
    after = os.lstat(path)
    after_components = _directory_component_identities(path, label=label)
    identity = (opened.st_dev, opened.st_ino)
    if (
        before_components != after_components
        or (before.st_dev, before.st_ino) != identity
        or (after.st_dev, after.st_ino) != identity
        or stat.S_ISLNK(after.st_mode)
        or not stat.S_ISDIR(after.st_mode)
    ):
        os.close(descriptor)
        raise ValueError(f"{label} changed while it was being authenticated.")
    return descriptor, identity, before_components


def _require_directory_identity(
    path: Path,
    identity: tuple[int, int],
    component_identities: tuple[tuple[Path, int, int], ...],
    *,
    label: str,
) -> None:
    observed = os.lstat(path)
    if (
        stat.S_ISLNK(observed.st_mode)
        or not stat.S_ISDIR(observed.st_mode)
        or (observed.st_dev, observed.st_ino) != identity
        or _directory_component_identities(path, label=label) != component_identities
    ):
        raise RuntimeError(f"{label} changed during immutable pair publication.")


def _entry_exists_at(directory_fd: int, name: str) -> bool:
    try:
        os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    except FileNotFoundError:
        return False
    return True


def _read_regular_at(directory_fd: int, name: str, *, label: str) -> tuple[bytes, os.stat_result]:
    before = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise RuntimeError(f"{label} is not a regular file after publication.")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(name, flags, dir_fd=directory_fd)
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode) or (
            opened_before.st_dev,
            opened_before.st_ino,
        ) != (before.st_dev, before.st_ino):
            raise RuntimeError(f"{label} is not a regular file after publication.")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        raw = b"".join(chunks)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    after = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
    stable_fields = (
        "st_dev",
        "st_ino",
        "st_mode",
        "st_size",
        "st_nlink",
        "st_mtime_ns",
        "st_ctime_ns",
    )
    if (
        any(getattr(before, field) != getattr(opened_before, field) for field in stable_fields)
        or any(
            getattr(opened_before, field) != getattr(opened_after, field) for field in stable_fields
        )
        or any(getattr(opened_after, field) != getattr(after, field) for field in stable_fields)
        or len(raw) != opened_after.st_size
    ):
        raise RuntimeError(f"{label} inode changed during post-publication authentication.")
    return raw, opened_after


def _publish_readonly_pair_no_overwrite(
    *,
    payload_path: Path,
    payload_raw: bytes,
    sidecar_raw: bytes,
    label: str,
    staging_parent: Path,
    pre_commit: Callable[[], None] | None = None,
    post_commit: Callable[[int], None] | None = None,
) -> None:
    """Commit a payload then sidecar through pinned directory descriptors.

    The sidecar is the commit marker.  Staging is explicitly separate from an
    execution evidence directory, and every operation after directory
    authentication is descriptor-relative.  Any failure after the first hard
    link is an ambiguous, non-replayable commit.
    """

    payload_path = payload_path.absolute()
    staging_parent = staging_parent.absolute()
    if not payload_path.is_absolute() or not staging_parent.is_absolute():  # pragma: no cover
        raise ValueError(f"{label} paths must be absolute.")
    sidecar_path = Path(f"{payload_path}.sha256")
    destination_fd, destination_identity, destination_components = _open_authenticated_directory(
        payload_path.parent,
        label=f"{label} destination parent",
    )
    staging_parent_fd: int | None = None
    staging_fd: int | None = None
    staging_name: str | None = None
    payload_linked = False
    try:
        (
            staging_parent_fd,
            staging_parent_identity,
            staging_parent_components,
        ) = _open_authenticated_directory(
            staging_parent,
            label=f"{label} staging parent",
        )
        for name in (payload_path.name, sidecar_path.name):
            if _entry_exists_at(destination_fd, name):
                raise FileExistsError(
                    f"{label} destination is occupied; refusing overwrite or replay."
                )
        for _ in range(128):
            candidate = f".immutable-pair-{secrets.token_hex(16)}"
            try:
                os.mkdir(candidate, mode=0o700, dir_fd=staging_parent_fd)
            except FileExistsError:
                continue
            staging_name = candidate
            break
        if staging_name is None:  # pragma: no cover - cryptographically implausible.
            raise RuntimeError(f"{label} could not claim a private staging directory.")
        directory_flags = (
            os.O_RDONLY
            | getattr(os, "O_DIRECTORY", 0)
            | getattr(os, "O_NOFOLLOW", 0)
            | getattr(os, "O_CLOEXEC", 0)
        )
        staging_fd = os.open(staging_name, directory_flags, dir_fd=staging_parent_fd)
        staged_metadata = os.fstat(staging_fd)
        staged_path_metadata = os.stat(
            staging_name,
            dir_fd=staging_parent_fd,
            follow_symlinks=False,
        )
        if not stat.S_ISDIR(staged_metadata.st_mode) or (
            staged_metadata.st_dev,
            staged_metadata.st_ino,
        ) != (staged_path_metadata.st_dev, staged_path_metadata.st_ino):
            raise ValueError(f"{label} staging directory identity drifted.")
        _write_staged_readonly_at(staging_fd, payload_path.name, payload_raw)
        _write_staged_readonly_at(staging_fd, sidecar_path.name, sidecar_raw)
        os.fsync(staging_fd)
        _require_directory_identity(
            payload_path.parent,
            destination_identity,
            destination_components,
            label=f"{label} destination parent",
        )
        _require_directory_identity(
            staging_parent,
            staging_parent_identity,
            staging_parent_components,
            label=f"{label} staging parent",
        )
        # Publication assumes the protocol's single-writer/quiescent-source
        # discipline.  This hook narrows the final source/namespace check to
        # the point immediately before the irreversible first hard link.
        if pre_commit is not None:
            pre_commit()
        _require_directory_identity(
            payload_path.parent,
            destination_identity,
            destination_components,
            label=f"{label} destination parent",
        )
        _require_directory_identity(
            staging_parent,
            staging_parent_identity,
            staging_parent_components,
            label=f"{label} staging parent",
        )
        try:
            os.link(
                payload_path.name,
                payload_path.name,
                src_dir_fd=staging_fd,
                dst_dir_fd=destination_fd,
                follow_symlinks=False,
            )
            payload_linked = True
            os.fsync(destination_fd)
            os.link(
                sidecar_path.name,
                sidecar_path.name,
                src_dir_fd=staging_fd,
                dst_dir_fd=destination_fd,
                follow_symlinks=False,
            )
            os.fsync(destination_fd)
            os.unlink(payload_path.name, dir_fd=staging_fd)
            os.unlink(sidecar_path.name, dir_fd=staging_fd)
            os.fsync(staging_fd)
            os.close(staging_fd)
            staging_fd = None
            os.rmdir(staging_name, dir_fd=staging_parent_fd)
            staging_name = None
            os.fsync(staging_parent_fd)

            for name, expected_raw in (
                (payload_path.name, payload_raw),
                (sidecar_path.name, sidecar_raw),
            ):
                observed_raw, metadata = _read_regular_at(
                    destination_fd,
                    name,
                    label=f"{label} published member {name}",
                )
                if (
                    observed_raw != expected_raw
                    or stat.S_IMODE(metadata.st_mode) != 0o444
                    or metadata.st_nlink != 1
                ):
                    raise RuntimeError(f"{label} published member {name} drifted.")
            if post_commit is not None:
                post_commit(destination_fd)
            _require_directory_identity(
                payload_path.parent,
                destination_identity,
                destination_components,
                label=f"{label} destination parent",
            )
            _require_directory_identity(
                staging_parent,
                staging_parent_identity,
                staging_parent_components,
                label=f"{label} staging parent",
            )
        except BaseException as exc:
            if not payload_linked:
                raise
            raise RuntimeError(
                f"{label} publication became ambiguous after the payload link; "
                "do not replay or overwrite it."
            ) from exc
    finally:
        cleanup_error: BaseException | None = None
        if staging_fd is not None:
            try:
                for name in (payload_path.name, sidecar_path.name):
                    try:
                        os.unlink(name, dir_fd=staging_fd)
                    except FileNotFoundError:
                        pass
                os.fsync(staging_fd)
            except BaseException as exc:  # pragma: no cover - injected I/O fault only.
                cleanup_error = exc
            finally:
                os.close(staging_fd)
        if staging_name is not None and staging_parent_fd is not None:
            try:
                os.rmdir(staging_name, dir_fd=staging_parent_fd)
                os.fsync(staging_parent_fd)
            except FileNotFoundError:
                pass
            except BaseException as exc:  # pragma: no cover - injected I/O fault only.
                cleanup_error = cleanup_error or exc
        if staging_parent_fd is not None:
            os.close(staging_parent_fd)
        os.close(destination_fd)
        if cleanup_error is not None:
            phase = "after payload commit" if payload_linked else "before payload commit"
            raise RuntimeError(f"{label} staging cleanup failed {phase}.") from cleanup_error


def _preregistration_environment_record(
    *, project_root: Path, preregistration_path: Path
) -> dict[str, Any]:
    """Build the diagnostic's embedded preregistration receipt from live bytes."""

    audit_root = (project_root / AUDIT_ROOT_RELATIVE).resolve()
    payload, raw = _read_json(
        preregistration_path,
        "equivalence preregistration",
        require_readonly=True,
        trusted_root=audit_root,
    )
    sidecar_path = Path(f"{preregistration_path}.sha256")
    sidecar = _validate_sidecar(
        payload_path=preregistration_path,
        payload_raw=raw,
        sidecar_path=sidecar_path,
        label="equivalence preregistration sidecar",
        trusted_root=audit_root,
    )
    authenticated: dict[str, dict[str, str]] = {}
    for role, relative in REQUIRED_PREREGISTRATION_FILE_PATHS.items():
        path = project_root / relative
        file_raw = _read_regular_bytes(
            path,
            f"preregistered {role}",
            require_readonly=False,
            trusted_root=project_root,
        )
        authenticated[role] = {"path": str(path), "sha256": sha256_bytes(file_raw)}
    sealed_sources: dict[str, dict[str, str]] = {}
    for artifact_role, sidecar_role, expected_sha in (
        ("prompt_contract", "prompt_contract_sidecar", PROMPT_CONTRACT_SHA256),
        ("prompt_protocol", "prompt_protocol_sidecar", PROMPT_PROTOCOL_SHA256),
        ("model_config", "model_config_sidecar", MODEL_CONFIG_SHA256),
    ):
        artifact = authenticated[artifact_role]
        sealed_sidecar = authenticated[sidecar_role]
        if artifact["sha256"] != expected_sha:
            raise ValueError(f"Cannot preregister drifted sealed source {artifact_role!r}.")
        sealed_sources[artifact_role] = {
            "path": artifact["path"],
            "sha256": expected_sha,
            "sidecar_path": sealed_sidecar["path"],
            "sidecar_sha256": sealed_sidecar["sha256"],
        }
    return {
        "path": str(preregistration_path),
        "raw_sha256": sha256_bytes(raw),
        "document_sha256": canonical_sha256(payload),
        "sidecar_path": sidecar["path"],
        "sidecar_sha256": sidecar["sha256"],
        "source_attempt": payload.get("source_attempt"),
        "lineage": deepcopy(payload.get("lineage")),
        "payload": payload,
        "authenticated_files": authenticated,
        "sealed_sources": sealed_sources,
    }


def _historical_preregistration_environment_record(
    *, project_root: Path, preregistration_path: Path
) -> dict[str, Any]:
    """Rebuild a preregistration embedding from immutable bytes, not live sources."""

    audit_root = (project_root / AUDIT_ROOT_RELATIVE).resolve()
    payload, raw = _read_json(
        preregistration_path,
        "historical equivalence preregistration",
        require_readonly=True,
        trusted_root=audit_root,
    )
    source_attempt = parse_preregistration_source_attempt(preregistration_path)
    files, _ = _validate_preregistration_payload_schema(payload, source_attempt=source_attempt)
    sidecar_path = Path(f"{preregistration_path}.sha256")
    sidecar = _validate_sidecar(
        payload_path=preregistration_path,
        payload_raw=raw,
        sidecar_path=sidecar_path,
        label="historical equivalence preregistration sidecar",
        trusted_root=audit_root,
    )
    authenticated = {
        role: {
            "path": str(project_root / relative),
            "sha256": files[role]["sha256"],
        }
        for role, relative in REQUIRED_PREREGISTRATION_FILE_PATHS.items()
    }
    sealed_sources: dict[str, dict[str, str]] = {}
    for artifact_role, sidecar_role, expected_sha in (
        ("prompt_contract", "prompt_contract_sidecar", PROMPT_CONTRACT_SHA256),
        ("prompt_protocol", "prompt_protocol_sidecar", PROMPT_PROTOCOL_SHA256),
        ("model_config", "model_config_sidecar", MODEL_CONFIG_SHA256),
    ):
        artifact = authenticated[artifact_role]
        sealed_sidecar = authenticated[sidecar_role]
        sealed_sources[artifact_role] = {
            "path": artifact["path"],
            "sha256": expected_sha,
            "sidecar_path": sealed_sidecar["path"],
            "sidecar_sha256": sealed_sidecar["sha256"],
        }
    return {
        "path": str(preregistration_path),
        "raw_sha256": sha256_bytes(raw),
        "document_sha256": canonical_sha256(payload),
        "sidecar_path": sidecar["path"],
        "sidecar_sha256": sidecar["sha256"],
        "source_attempt": source_attempt,
        "lineage": deepcopy(payload["lineage"]),
        "payload": payload,
        "authenticated_files": authenticated,
        "sealed_sources": sealed_sources,
    }


def load_preregistration(path: str | Path, project_root: str | Path) -> dict[str, Any]:
    """Authenticate one attempt-aware preregistration for the diagnostic."""

    project_root = Path(project_root).expanduser().resolve()
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = project_root / path
    embedded = _preregistration_environment_record(
        project_root=project_root,
        preregistration_path=path,
    )
    return _validate_preregistration(embedded, project_root=project_root)


def publish_preregistration(
    *,
    project_root: str | Path,
    source_attempt: Any = "001",
    preregistration_path: str | Path | None = None,
    created_at_utc: str | None = None,
    predecessor_result_path: str | Path | None = None,
    repair_reason: str | None = None,
) -> dict[str, Any]:
    """Publish the exact pre-execution commitment without overwrite."""

    project_root = Path(project_root).expanduser().resolve()
    audit_root = (project_root / AUDIT_ROOT_RELATIVE).resolve()
    source_attempt = normalize_attempt(source_attempt, "source_attempt")
    preregistration_path = (
        Path(preregistration_path).expanduser()
        if preregistration_path is not None
        else audit_root / preregistration_filename(source_attempt)
    )
    if not preregistration_path.is_absolute():
        preregistration_path = project_root / preregistration_path
    if preregistration_path != audit_root / preregistration_filename(source_attempt):
        raise ValueError("Equivalence preregistration must use its one canonical path.")
    created_at_utc = created_at_utc or datetime.now(timezone.utc).isoformat()
    created_at = _aware_timestamp(created_at_utc, "preregistration created_at_utc")
    if created_at > datetime.now(timezone.utc):
        raise ValueError("Equivalence preregistration timestamp is in the future.")
    existing_sources = _source_preregistration_attempts(audit_root)
    required_predecessor_sources = {f"{index:03d}" for index in range(1, int(source_attempt))}
    if set(existing_sources) != required_predecessor_sources:
        raise ValueError(
            "Source-attempt publication is gapped, concurrent, or not the next attempt."
        )
    if source_attempt == QUARANTINE_SUCCESSOR_SOURCE_ATTEMPT:
        _require_source004_execution_namespace_absent(audit_root)

    file_records: dict[str, dict[str, str]] = {}
    source004_quarantine_snapshot: dict[str, Any] | None = None
    for role, relative in REQUIRED_PREREGISTRATION_FILE_PATHS.items():
        path = project_root / relative
        raw = _read_regular_bytes(
            path,
            f"preregistration source {role}",
            require_readonly=False,
            trusted_root=project_root,
        )
        file_records[role] = {"path": str(relative), "sha256": sha256_bytes(raw)}
    if source_attempt == "001":
        if predecessor_result_path is not None or repair_reason is not None:
            raise ValueError("Initial source attempt cannot claim predecessor repair evidence.")
        lineage: dict[str, Any] = {
            "schema_version": 1,
            "predecessor": None,
            "repair_reason": None,
            "changed_roles": [],
        }
    else:
        if not isinstance(repair_reason, str) or not repair_reason.strip():
            raise ValueError("A repaired source attempt requires a non-empty repair_reason.")
        if predecessor_result_path is None:
            raise ValueError("A repaired source attempt requires its predecessor result.")
        previous_attempt = f"{int(source_attempt) - 1:03d}"
        predecessor_prereg_path = audit_root / preregistration_filename(previous_attempt)
        predecessor_payload, predecessor_prereg = _immutable_json_binding(
            predecessor_prereg_path,
            audit_root=audit_root,
            label="predecessor equivalence preregistration",
        )
        previous_files, _ = _validate_preregistration_payload_schema(
            predecessor_payload, source_attempt=previous_attempt
        )
        _validate_source_attempt_lineage(
            predecessor_payload.get("lineage"),
            source_attempt=previous_attempt,
            current_files=previous_files,
            current_created_at=_aware_timestamp(
                predecessor_payload["created_at_utc"],
                "predecessor preregistration created_at_utc",
            ),
            project_root=project_root,
        )
        changes = [
            {
                "role": role,
                "previous_sha256": previous_files[role]["sha256"],
                "current_sha256": file_records[role]["sha256"],
            }
            for role in sorted(file_records)
            if previous_files[role]["sha256"] != file_records[role]["sha256"]
        ]
        if not changes:
            raise ValueError(
                "Identical source bytes must reuse the existing preregistration and increment "
                "only execution_attempt."
            )
        predecessor_result_path = Path(predecessor_result_path).expanduser()
        if not predecessor_result_path.is_absolute():
            predecessor_result_path = project_root / predecessor_result_path
        result_source, result_execution = parse_result_attempts(predecessor_result_path.parent)
        predecessor_result, predecessor_result_binding = _immutable_json_binding(
            predecessor_result_path,
            audit_root=audit_root,
            label="predecessor equivalence result",
        )
        if (
            predecessor_result_path.name != RESULT_FILENAME
            or result_source != previous_attempt
            or predecessor_result.get("protocol_id") != PROTOCOL_ID
            or predecessor_result.get("status") not in {"error", "failed", "passed"}
        ):
            raise ValueError("Predecessor result is not the immediately prior source attempt.")
        if source_attempt == QUARANTINE_SUCCESSOR_SOURCE_ATTEMPT:
            if (
                previous_attempt != QUARANTINE_SOURCE_ATTEMPT
                or result_source != QUARANTINE_SOURCE_ATTEMPT
                or result_execution != QUARANTINE_EXECUTION_ATTEMPT
                or predecessor_result["status"] != "passed"
                or repair_reason != SOURCE004_QUARANTINE_REPAIR_REASON
            ):
                raise ValueError("Source-004 may cite only the exact quarantined source-003 run.")
            quarantine = validate_source003_quarantine_terminal(
                project_root=project_root,
                require_successor_absent=True,
                require_live_repair_sources=True,
            )
            source004_quarantine_snapshot = deepcopy(quarantine)
            expected_result = quarantine["authorization"]["result"]
            if (
                predecessor_result_binding["sha256"] != expected_result["sha256"]
                or file_records != quarantine["prospective_file_ledger"]
                or changes != quarantine["payload"]["source_transition"]["changed_roles"]
            ):
                raise ValueError("Source-004 quarantine result/source transition drifted.")
            lineage = {
                "schema_version": 2,
                "recovery": "source003_quarantine_to_fresh_source004_v1",
                "predecessor": {
                    "source_attempt": QUARANTINE_SOURCE_ATTEMPT,
                    "preregistration": {
                        key: quarantine["authorization"]["preregistration"][key]
                        for key in (
                            "path",
                            "sha256",
                            "document_sha256",
                            "sidecar_path",
                            "sidecar_sha256",
                        )
                    },
                    "result": deepcopy(dict(expected_result)),
                    "terminal_receipt": None,
                    "acceptance_receipt": None,
                    "quarantine_authorization": deepcopy(
                        dict(quarantine["authorization"]["authorization"])
                    ),
                    "quarantine_terminal_receipt": deepcopy(dict(quarantine["receipt"])),
                },
                "repair_reason": SOURCE004_QUARANTINE_REPAIR_REASON,
                "changed_roles": deepcopy(changes),
            }
        else:
            exact_execution = validate_terminal_execution_result(
                result_path=predecessor_result_path,
                project_root=project_root,
            )
            attempts = _attempt_directories_for_source(
                audit_root=audit_root, source_attempt=previous_attempt
            )
            required_attempts = {f"{index:03d}" for index in range(1, int(result_execution) + 1)}
            if (
                exact_execution["result"]["sha256"] != predecessor_result_binding["sha256"]
                or set(attempts) != required_attempts
            ):
                raise ValueError(
                    "Predecessor result is not the latest contiguous prior-source execution."
                )
            terminal_admission = validate_terminal_receipt(
                result_path=predecessor_result_path,
                project_root=project_root,
                require_latest=True,
            )
            terminal_binding = _terminal_receipt_binding(terminal_admission)
            if predecessor_result["status"] == "passed":
                acceptance_binding: dict[str, Any] | None = _acceptance_receipt_binding(
                    validate_acceptance_receipt(
                        result_path=predecessor_result_path,
                        project_root=project_root,
                        require_latest=True,
                        require_live_preregistration_sources=False,
                    )
                )
            else:
                acceptance_binding = None
            lineage = {
                "schema_version": 1,
                "predecessor": {
                    "source_attempt": previous_attempt,
                    "preregistration": predecessor_prereg,
                    "result": {
                        "path": predecessor_result_binding["path"],
                        "sha256": predecessor_result_binding["sha256"],
                        "sidecar_path": predecessor_result_binding["sidecar_path"],
                        "sidecar_sha256": predecessor_result_binding["sidecar_sha256"],
                        "status": predecessor_result["status"],
                        "source_attempt": result_source,
                        "execution_attempt": result_execution,
                    },
                    "terminal_receipt": terminal_binding,
                    "acceptance_receipt": acceptance_binding,
                },
                "repair_reason": repair_reason.strip(),
                "changed_roles": changes,
            }
    _validate_source_attempt_lineage(
        lineage,
        source_attempt=source_attempt,
        current_files=file_records,
        current_created_at=created_at,
        project_root=project_root,
    )
    payload = {
        "schema_version": 1,
        "protocol_id": PROTOCOL_ID,
        "status": "preregistered_before_execution",
        "created_at_utc": created_at_utc,
        "source_attempt": source_attempt,
        "lineage": lineage,
        "execution": expected_execution_contract(),
        "files": file_records,
    }
    payload_raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    sidecar_raw = f"{sha256_bytes(payload_raw)}  {preregistration_path.name}\n".encode("utf-8")

    def require_quiescent_source004_inputs() -> None:
        if source004_quarantine_snapshot is None:
            raise ValueError("Source-004 quarantine predecessor snapshot is missing.")
        fresh_quarantine = validate_source003_quarantine_terminal(
            project_root=project_root,
            require_successor_absent=not os.path.lexists(preregistration_path),
            require_live_repair_sources=True,
        )
        if fresh_quarantine != source004_quarantine_snapshot:
            raise ValueError("Source-003 quarantine evidence drifted during source-004 commit.")
        if _live_preregistration_file_ledger(project_root) != file_records:
            raise ValueError("Source-004 repair sources drifted during preregistration commit.")
        _require_source004_execution_namespace_absent(audit_root)

    source004_hook = (
        require_quiescent_source004_inputs
        if source_attempt == QUARANTINE_SUCCESSOR_SOURCE_ATTEMPT
        else None
    )
    _publish_readonly_pair_no_overwrite(
        payload_path=preregistration_path,
        payload_raw=payload_raw,
        sidecar_raw=sidecar_raw,
        label="Equivalence preregistration",
        staging_parent=audit_root,
        pre_commit=source004_hook,
        post_commit=(
            (lambda _directory_fd: require_quiescent_source004_inputs())
            if source004_hook is not None
            else None
        ),
    )
    try:
        return load_preregistration(preregistration_path, project_root)
    except BaseException as exc:
        raise RuntimeError(
            "Equivalence preregistration committed but postcommit validation is ambiguous; "
            "do not replay it."
        ) from exc


def publish_acceptance_receipt(
    *,
    result_path: str | Path,
    project_root: str | Path,
    result_sidecar_path: str | Path | None = None,
    receipt_path: str | Path | None = None,
    accepted_at_utc: str | None = None,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Validate a passed run, publish a JSON receipt, then reopen it strictly."""

    project_root = Path(project_root).expanduser().resolve()
    audit_root = (project_root / AUDIT_ROOT_RELATIVE).resolve()
    candidate_result = Path(result_path).expanduser()
    if not candidate_result.is_absolute():
        candidate_result = project_root / candidate_result
    source_attempt, execution_attempt = parse_result_attempts(candidate_result.parent)
    if source_attempt == QUARANTINE_SOURCE_ATTEMPT:
        raise ValueError("Source-003 quarantine permanently forbids acceptance publication.")
    evidence = validate_equivalence_result(
        result_path=result_path,
        result_sidecar_path=result_sidecar_path,
        project_root=project_root,
    )
    result_path = Path(evidence["result"]["path"])
    terminal_path = terminal_receipt_path(result_path.parent)
    if terminal_path.exists() or Path(f"{terminal_path}.sha256").exists():
        terminal_admission = validate_terminal_receipt(
            result_path=result_path,
            project_root=project_root,
            require_latest=True,
        )
    else:
        terminal_admission = publish_terminal_receipt(
            result_path=result_path,
            project_root=project_root,
            run=run,
        )
    terminal_binding = _terminal_receipt_binding(terminal_admission)
    scheduler_query = terminal_admission["slurm"]["scheduler_query"]
    receipt_path = (
        Path(receipt_path).expanduser()
        if receipt_path is not None
        else result_path.parent / RECEIPT_FILENAME
    )
    if not receipt_path.is_absolute():
        receipt_path = Path(project_root).resolve() / receipt_path
    if receipt_path != result_path.parent / RECEIPT_FILENAME:
        raise ValueError("Acceptance receipt must use the canonical result-local filename.")
    receipt_sidecar_path = Path(f"{receipt_path}.sha256")
    if os.path.lexists(receipt_path) or os.path.lexists(receipt_sidecar_path):
        raise FileExistsError(
            "Acceptance receipt destination is occupied; refusing overwrite or replay."
        )
    _validate_passed_execution_members(result_path.parent, accepted=False)
    accepted_at_utc = accepted_at_utc or datetime.now(timezone.utc).isoformat()
    receipt = _build_receipt_payload(
        evidence=evidence,
        scheduler_query=scheduler_query,
        terminal_execution_receipt=terminal_binding,
        accepted_at_utc=accepted_at_utc,
    )
    receipt_raw = (json.dumps(receipt, indent=2, sort_keys=True, allow_nan=False) + "\n").encode(
        "utf-8"
    )
    sidecar_raw = f"{sha256_bytes(receipt_raw)}  {receipt_path.name}\n".encode("utf-8")
    _publish_readonly_pair_no_overwrite(
        payload_path=receipt_path,
        payload_raw=receipt_raw,
        sidecar_raw=sidecar_raw,
        label="Acceptance receipt",
        staging_parent=audit_root,
    )
    _validate_passed_execution_members(result_path.parent, accepted=True)
    _freeze_execution_tree(result_path.parent)
    return validate_acceptance_receipt(
        result_path=result_path,
        result_sidecar_path=result_sidecar_path,
        receipt_path=receipt_path,
        project_root=project_root,
    )


def validate_protocol_input_bindings(
    *, protocol_inputs: Mapping[str, Any], project_root: str | Path
) -> dict[str, Any]:
    """Validate the complete equivalence DAG embedded in a main-v3 config."""

    project_root = Path(project_root).resolve()
    role_names = PROTOCOL_INPUT_BINDING_ROLES
    missing = sorted(role_names - set(protocol_inputs))
    if missing:
        raise ValueError(f"Main v3 protocol omits equivalence admission roles: {missing}.")
    admission_path = project_root / ADMISSION_SOURCE_RELATIVE
    _validate_live_file_binding(
        protocol_inputs["native_equivalence_admission_validator"],
        expected_path=admission_path,
        label="native-equivalence admission validator",
        require_readonly=True,
        trusted_root=project_root,
    )
    result_path = Path(str(protocol_inputs["native_equivalence_result"]["path"]))
    result_sidecar_path = Path(str(protocol_inputs["native_equivalence_result_sidecar"]["path"]))
    receipt_path = Path(str(protocol_inputs["native_equivalence_receipt"]["path"]))
    receipt_sidecar_path = Path(str(protocol_inputs["native_equivalence_receipt_sidecar"]["path"]))
    admission = validate_acceptance_receipt(
        result_path=result_path,
        result_sidecar_path=result_sidecar_path,
        receipt_path=receipt_path,
        receipt_sidecar_path=receipt_sidecar_path,
        project_root=project_root,
    )
    live_records = {
        "native_equivalence_result": admission["result"],
        "native_equivalence_result_sidecar": {
            "path": admission["result"]["sidecar_path"],
            "sha256": admission["result"]["sidecar_sha256"],
        },
        "native_equivalence_terminal_receipt": {
            "path": admission["terminal_execution_receipt"]["path"],
            "sha256": admission["terminal_execution_receipt"]["sha256"],
        },
        "native_equivalence_terminal_receipt_sidecar": {
            "path": admission["terminal_execution_receipt"]["sidecar_path"],
            "sha256": admission["terminal_execution_receipt"]["sidecar_sha256"],
        },
        "native_equivalence_receipt": admission["receipt"],
        "native_equivalence_receipt_sidecar": {
            "path": admission["receipt"]["sidecar_path"],
            "sha256": admission["receipt"]["sidecar_sha256"],
        },
        "native_equivalence_preregistration": {
            "path": admission["preregistration"]["path"],
            "sha256": admission["preregistration"]["raw_sha256"],
        },
        "native_equivalence_preregistration_sidecar": {
            "path": admission["preregistration"]["sidecar_path"],
            "sha256": admission["preregistration"]["sidecar_sha256"],
        },
    }
    for role, live in live_records.items():
        configured = protocol_inputs[role]
        if (
            not isinstance(configured, Mapping)
            or set(configured) != {"path", "sha256"}
            or configured.get("path") != live["path"]
            or configured.get("sha256") != live["sha256"]
        ):
            raise ValueError(f"Main v3 immutable equivalence binding drifted for {role}.")
    return admission


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preregister = commands.add_parser(
        "publish-preregistration", help="Publish the immutable pre-H100 commitment."
    )
    preregister.add_argument("--project-root", type=Path, required=True)
    preregister.add_argument("--source-attempt", default="001")
    preregister.add_argument("--preregistration", type=Path)
    preregister.add_argument("--created-at-utc")
    preregister.add_argument("--predecessor-result", type=Path)
    preregister.add_argument("--repair-reason")

    quarantine_terminal = commands.add_parser(
        "publish-source003-quarantine-terminal",
        help="Publish the one-off nonaccepting source-003 quarantine terminal.",
    )
    quarantine_terminal.add_argument("--project-root", type=Path, required=True)
    quarantine_terminal.add_argument("--recorded-at-utc")

    validate_quarantine = commands.add_parser(
        "validate-source003-quarantine-terminal",
        help="Reauthenticate the one-off source-003 quarantine closure.",
    )
    validate_quarantine.add_argument("--project-root", type=Path, required=True)
    validate_quarantine.add_argument(
        "--allow-source004-present",
        action="store_true",
        help="Validate historical closure after source-004 preregistration exists.",
    )

    terminal = commands.add_parser(
        "publish-terminal-receipt",
        help="Query sacct and publish one immutable execution-terminal receipt.",
    )
    terminal.add_argument("--project-root", type=Path, required=True)
    terminal.add_argument("--result", type=Path, required=True)
    terminal.add_argument("--receipt", type=Path)

    validate_terminal = commands.add_parser(
        "validate-terminal-receipt",
        help="Reauthenticate one contiguous terminal execution chain.",
    )
    validate_terminal.add_argument("--project-root", type=Path, required=True)
    validate_terminal.add_argument("--result", type=Path, required=True)
    validate_terminal.add_argument("--receipt", type=Path)
    validate_terminal.add_argument("--receipt-sidecar", type=Path)

    prelaunch = commands.add_parser(
        "validate-prelaunch",
        help="Authorize the next execution attempt before output creation.",
    )
    prelaunch.add_argument("--project-root", type=Path, required=True)
    prelaunch.add_argument("--preregistration", type=Path, required=True)
    prelaunch.add_argument("--output-dir", type=Path, required=True)

    receipt = commands.add_parser(
        "publish-receipt", help="Query sacct and publish post-H100 acceptance."
    )
    receipt.add_argument("--project-root", type=Path, required=True)
    receipt.add_argument("--result", type=Path, required=True)
    receipt.add_argument("--result-sidecar", type=Path)
    receipt.add_argument("--receipt", type=Path)

    validate = commands.add_parser(
        "validate-receipt", help="Reauthenticate an existing result/receipt DAG."
    )
    validate.add_argument("--project-root", type=Path, required=True)
    validate.add_argument("--result", type=Path, required=True)
    validate.add_argument("--result-sidecar", type=Path)
    validate.add_argument("--receipt", type=Path)
    validate.add_argument("--receipt-sidecar", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "publish-preregistration":
        output = publish_preregistration(
            project_root=args.project_root,
            source_attempt=args.source_attempt,
            preregistration_path=args.preregistration,
            created_at_utc=args.created_at_utc,
            predecessor_result_path=args.predecessor_result,
            repair_reason=args.repair_reason,
        )
    elif args.command == "publish-source003-quarantine-terminal":
        output = publish_source003_quarantine_terminal(
            project_root=args.project_root,
            recorded_at_utc=args.recorded_at_utc,
        )
    elif args.command == "validate-source003-quarantine-terminal":
        output = validate_source003_quarantine_terminal(
            project_root=args.project_root,
            require_successor_absent=not args.allow_source004_present,
            require_live_repair_sources=not args.allow_source004_present,
        )
    elif args.command == "publish-terminal-receipt":
        output = publish_terminal_receipt(
            project_root=args.project_root,
            result_path=args.result,
            receipt_path=args.receipt,
        )
    elif args.command == "validate-terminal-receipt":
        output = validate_terminal_receipt(
            project_root=args.project_root,
            result_path=args.result,
            receipt_path=args.receipt,
            receipt_sidecar_path=args.receipt_sidecar,
        )
    elif args.command == "validate-prelaunch":
        output = validate_prelaunch_attempt(
            project_root=args.project_root,
            preregistration_path=args.preregistration,
            output_directory=args.output_dir,
        )
    elif args.command == "publish-receipt":
        output = publish_acceptance_receipt(
            project_root=args.project_root,
            result_path=args.result,
            result_sidecar_path=args.result_sidecar,
            receipt_path=args.receipt,
        )
    elif args.command == "validate-receipt":
        output = validate_acceptance_receipt(
            project_root=args.project_root,
            result_path=args.result,
            result_sidecar_path=args.result_sidecar,
            receipt_path=args.receipt,
            receipt_sidecar_path=args.receipt_sidecar,
        )
    else:  # pragma: no cover - argparse enforces the closed command set.
        raise AssertionError(f"Unknown admission command: {args.command!r}")
    print(json.dumps(output, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
