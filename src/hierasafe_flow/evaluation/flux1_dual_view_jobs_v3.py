"""Shared FLUX.1-dev dual-view job projection and admission.

This module is intentionally independent of the common-seed protocol module.
It is the single campaign-wide authority for translating the public
``flux1_dev`` model name into the versioned ``flux_dual_view`` execution route.
The immutable source-prompt contract owns the independent CLIP/T5 strings and
token fingerprints.  The native-equivalence admission module owns the separate
nine-role evidence DAG required before any projected job can be launched.

Preview projection is useful for inspecting planned campaign matrices, but is
deliberately nonlaunchable.  Execution projection requires an authenticated
equivalence DAG; there is no legacy adapter fallback.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from hierasafe_flow.adapters.flux_dual_view_adapter import (
    FLUX_DUAL_VIEW_CONFIG_KEY,
    validate_flux_dual_view_conditioning,
)


SCHEMA_VERSION = 1
CONTRACT_ID = "finer_detailing_flux1_campaign_dual_view_v3"
PUBLIC_MODEL_NAME = "flux1_dev"
MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
PIPELINE_CLASS = "FluxPipeline"
ADAPTER_KEY = "flux_dual_view"

MODE_PREVIEW = "preview_v3"
MODE_EXECUTION = "execution_v3"
MODE_AUDIT = "audit_v3"
VALIDATION_MODES = frozenset({MODE_PREVIEW, MODE_EXECUTION, MODE_AUDIT})

PREVIEW_STATUS = "implementation_preview_not_launchable_pending_native_equivalence"
EXECUTION_STATUS = "authenticated_for_execution_after_native_equivalence"

RUNNER_PREFLIGHT_RECEIPT_SCHEMA_VERSION = 2
RUNNER_PREFLIGHT_PROJECTED_JOB_HASH_CONTRACT = (
    "flux1_dual_view_runner_preflight_top_level_output_dir_sentinel_v1"
)
RUNNER_PREFLIGHT_OUTPUT_DIR_SENTINEL = (
    "__HIERASAFE_FLUX1_DUAL_VIEW_RUNNER_PREFLIGHT_OUTPUT_DIR_V1__"
)
RUNNER_PREFLIGHT_ROUTE = (
    "flux1_dual_view_jobs_v3.project_flux1_job_v3(preview_v3)->"
    "flux1_dual_view_jobs_v3.validate_flux1_job_v3(preview_v3)->"
    "finer_detailing_correction._runner_config->benign_park._runner_config->"
    "load_config->GenerationRunner.__init__->_bind_flux_dual_view_conditioning->"
    "registry.create_adapter"
)
RUNNER_PREFLIGHT_OUTPUT_DIR_ALIASES = frozenset(
    {"output_directory", "output_path", "output_root", "results_dir"}
)

NEGATIVE_MODE_NOT_APPLIED = "not_applied"
NEGATIVE_MODE_EXPLICIT_NONE_CONTROL = "explicit_none_control"
NEGATIVE_MODE_PAIRED_REGISTERED = "paired_registered_negative"
NEGATIVE_MODES = frozenset(
    {
        NEGATIVE_MODE_NOT_APPLIED,
        NEGATIVE_MODE_EXPLICIT_NONE_CONTROL,
        NEGATIVE_MODE_PAIRED_REGISTERED,
    }
)

MODEL_CONFIG_RELATIVE = Path("configs/models/t2i_flux1_dev_dual_view_v3.yaml")
MODEL_CONFIG_SIDECAR_RELATIVE = Path(f"{MODEL_CONFIG_RELATIVE}.sha256")
SOURCE_PROMPTS_RELATIVE = Path(
    "configs/experiments/flux1_common_seed_source_prompts_v3.yaml"
)
SOURCE_PROMPTS_SIDECAR_RELATIVE = Path(f"{SOURCE_PROMPTS_RELATIVE}.sha256")
SOURCE_PROMPTS_PROTOCOL_RELATIVE = Path(
    "debugging/audits/finer_detailing_20260720/"
    "flux1_common_seed_source_prompts_v3_PROTOCOL_PREREGISTERED.md"
)
SOURCE_PROMPTS_PROTOCOL_SIDECAR_RELATIVE = Path(
    f"{SOURCE_PROMPTS_PROTOCOL_RELATIVE}.sha256"
)
SOURCE_PROMPTS_CONTRACT_ID = "flux1_common_seed_source_prompts_v3"

MODEL_CONFIG_SHA256 = "624cc2f53e293102e6e31f016c307d30f5108a916afdbdc06d6e6365fb6bbd9d"
SOURCE_PROMPTS_SHA256 = "1d061d331293980023599445f680f865b4d4af41f91144e9d6a3b3345ffe9b87"
SOURCE_PROMPTS_PROTOCOL_SHA256 = (
    "c79c7ed65dac46a844be0022b2a81001536e10ad96ef00c7afecbddc6063e1ec"
)

PROMPT_IDS = (
    "01_sad_young_girl",
    "02_angry_old_man",
    "03_empty_outdoor_mall",
)

# ``model_config`` replaces the ordinary five-role job binding.  The other
# five records are additional immutable source/model seals.
SOURCE_PROTOCOL_INPUT_ROLES = frozenset(
    {
        "model_config",
        "model_config_sidecar",
        "common_seed_source_prompts",
        "common_seed_source_prompts_sidecar",
        "common_seed_source_prompts_protocol",
        "common_seed_source_prompts_protocol_sidecar",
    }
)
POSITIVE_FIELDS = frozenset(
    {
        "clip_prompt",
        "t5_prompt_2",
        "clip_string_sha256",
        "clip_token_count",
        "clip_token_ids_sha256",
        "t5_string_sha256",
        "t5_token_count",
        "t5_token_ids_sha256",
    }
)
NEGATIVE_FIELDS = frozenset(
    {
        "clip_negative_prompt",
        "t5_negative_prompt_2",
        "clip_negative_string_sha256",
        "clip_negative_token_count",
        "clip_negative_token_ids_sha256",
        "t5_negative_string_sha256",
        "t5_negative_token_count",
        "t5_negative_token_ids_sha256",
    }
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def canonicalize_runner_preflight_projected_job_v3(
    projected_job: Mapping[str, Any],
    *,
    expected_output_dir: str,
) -> dict[str, Any]:
    """Replace only the exact top-level ephemeral output directory.

    The equivalence preflight executes through a fresh ``TemporaryDirectory``.
    That path is operationally irrelevant to conditioning but made the legacy
    source-003 projected-job digest unrecomputable.  This versioned contract
    requires one nonempty absolute top-level string, rejects competing aliases,
    and deliberately leaves every nested mapping untouched.
    """

    if not isinstance(projected_job, Mapping):
        raise TypeError("Runner-preflight projected job must be a mapping.")
    aliases = RUNNER_PREFLIGHT_OUTPUT_DIR_ALIASES.intersection(projected_job)
    if aliases:
        raise ValueError(
            f"Runner-preflight projected job has forbidden output-dir aliases: {sorted(aliases)}."
        )
    if "output_dir" not in projected_job:
        raise ValueError("Runner-preflight projected job lacks top-level output_dir.")
    output_dir = projected_job["output_dir"]
    if (
        type(output_dir) is not str
        or not output_dir
        or "\x00" in output_dir
        or output_dir == RUNNER_PREFLIGHT_OUTPUT_DIR_SENTINEL
    ):
        raise ValueError(
            "Runner-preflight top-level output_dir must be one real non-sentinel string."
        )
    if type(expected_output_dir) is not str or output_dir != expected_output_dir:
        raise ValueError(
            "Runner-preflight projected output_dir differs from the caller's ephemeral path."
        )
    output_path = Path(output_dir)
    if not output_path.is_absolute() or any(part in {".", ".."} for part in output_path.parts):
        raise ValueError("Runner-preflight top-level output_dir must be lexically absolute.")

    forbidden_nested_keys = {"output_dir", *RUNNER_PREFLIGHT_OUTPUT_DIR_ALIASES}

    def audit_nested(value: Any, location: str) -> None:
        if isinstance(value, Mapping):
            for key, nested in value.items():
                if type(key) is not str:
                    raise ValueError(
                        f"Runner-preflight projected job has a non-string key below {location}."
                    )
                nested_location = f"{location}.{key}"
                if key in forbidden_nested_keys:
                    raise ValueError(
                        "Runner-preflight projected job repeats an output-dir key below "
                        f"the top level at {nested_location}."
                    )
                if type(nested) is str and nested == expected_output_dir:
                    raise ValueError(
                        "Runner-preflight ephemeral output path occurs outside $.output_dir "
                        f"at {nested_location}."
                    )
                audit_nested(nested, nested_location)
        elif isinstance(value, (list, tuple)):
            for index, nested in enumerate(value):
                nested_location = f"{location}[{index}]"
                if type(nested) is str and nested == expected_output_dir:
                    raise ValueError(
                        "Runner-preflight ephemeral output path occurs outside $.output_dir "
                        f"at {nested_location}."
                    )
                audit_nested(nested, nested_location)

    for top_key, top_value in projected_job.items():
        if type(top_key) is not str:
            raise ValueError("Runner-preflight projected job has a non-string top-level key.")
        if top_key == "output_dir":
            continue
        if type(top_value) is str and top_value == expected_output_dir:
            raise ValueError(
                "Runner-preflight ephemeral output path occurs outside $.output_dir "
                f"at $.{top_key}."
            )
        audit_nested(top_value, f"$.{top_key}")
    canonical = deepcopy(dict(projected_job))
    canonical["output_dir"] = RUNNER_PREFLIGHT_OUTPUT_DIR_SENTINEL
    return canonical


def runner_preflight_projected_job_sha256_v3(
    projected_job: Mapping[str, Any],
    *,
    expected_output_dir: str,
) -> str:
    """Hash the strict versioned runner-preflight projection representation."""

    return canonical_sha256(
        canonicalize_runner_preflight_projected_job_v3(
            projected_job,
            expected_output_dir=expected_output_dir,
        )
    )


def _text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _lexical_absolute(path: Path) -> Path:
    return path.expanduser() if path.is_absolute() else (Path.cwd() / path).absolute()


def _authenticated_readonly_file(
    path: Path,
    *,
    trusted_root: Path,
    require_read_only: bool = True,
) -> bytes:
    """Read one file with lexical-path, no-follow, and inode stability checks.

    Live protocol inputs must be permanently read-only.  Content-addressed
    snapshot objects are authenticated by their frozen digest and may retain
    the archive creator's ordinary file mode, so snapshot audit explicitly
    opts out of only the mode-bit check.
    """

    trusted_root = _lexical_absolute(trusted_root)
    path = _lexical_absolute(path)
    try:
        relative = path.relative_to(trusted_root)
    except ValueError as exc:
        raise ValueError(f"Flux-v3 sealed path escapes its trusted root: {path}") from exc
    if not relative.parts or any(part in {"", ".", ".."} for part in relative.parts):
        raise ValueError(f"Flux-v3 sealed path is not lexically canonical: {path}")
    current = trusted_root
    for component in relative.parts[:-1]:
        current /= component
        metadata = os.lstat(current)
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(f"Flux-v3 sealed path has an aliased/non-directory parent: {current}")
    before = os.lstat(path)
    if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"Flux-v3 sealed input is symlinked or non-regular: {path}")
    if require_read_only and before.st_mode & 0o222:
        raise ValueError(f"Flux-v3 sealed input is writable: {path}")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        opened = os.fstat(descriptor)
        if (
            not stat.S_ISREG(opened.st_mode)
            or (opened.st_dev, opened.st_ino) != (before.st_dev, before.st_ino)
        ):
            raise ValueError(f"Flux-v3 sealed input changed during no-follow open: {path}")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 16 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        after = os.fstat(descriptor)
        if (
            (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
            != (opened.st_dev, opened.st_ino, opened.st_size, opened.st_mtime_ns)
        ):
            raise ValueError(f"Flux-v3 sealed input changed while being read: {path}")
        return b"".join(chunks)
    finally:
        os.close(descriptor)


def _file_record(path: Path, *, trusted_root: Path) -> dict[str, str]:
    raw = _authenticated_readonly_file(path, trusted_root=trusted_root)
    return {"path": str(_lexical_absolute(path)), "sha256": hashlib.sha256(raw).hexdigest()}


def _validate_sidecar(
    payload: Path,
    sidecar: Path,
    *,
    expected_sha256: str,
    trusted_root: Path,
) -> None:
    payload_raw = _authenticated_readonly_file(payload, trusted_root=trusted_root)
    sidecar_raw = _authenticated_readonly_file(sidecar, trusted_root=trusted_root)
    _validate_sidecar_payloads(
        payload_raw,
        payload_name=payload.name,
        sidecar_raw=sidecar_raw,
        expected_sha256=expected_sha256,
        label=str(sidecar),
    )


def _validate_sidecar_payloads(
    payload_raw: bytes,
    *,
    payload_name: str,
    sidecar_raw: bytes,
    expected_sha256: str,
    label: str,
) -> None:
    if hashlib.sha256(payload_raw).hexdigest() != expected_sha256:
        raise ValueError(f"Flux-v3 pinned SHA-256 drifted for {payload_name}.")
    expected_words = [expected_sha256, payload_name]
    if sidecar_raw.decode("utf-8").split() != expected_words:
        raise ValueError(f"Flux-v3 SHA-256 sidecar is inconsistent: {label}.")


def build_flux1_static_protocol_inputs_v3(
    project_root: str | Path,
) -> dict[str, dict[str, str]]:
    """Authenticate and return the sealed source/model binding subset."""

    root = _lexical_absolute(Path(project_root))
    model = root / MODEL_CONFIG_RELATIVE
    model_sidecar = root / MODEL_CONFIG_SIDECAR_RELATIVE
    source = root / SOURCE_PROMPTS_RELATIVE
    source_sidecar = root / SOURCE_PROMPTS_SIDECAR_RELATIVE
    protocol = root / SOURCE_PROMPTS_PROTOCOL_RELATIVE
    protocol_sidecar = root / SOURCE_PROMPTS_PROTOCOL_SIDECAR_RELATIVE
    _validate_sidecar(
        model,
        model_sidecar,
        expected_sha256=MODEL_CONFIG_SHA256,
        trusted_root=root,
    )
    _validate_sidecar(
        source,
        source_sidecar,
        expected_sha256=SOURCE_PROMPTS_SHA256,
        trusted_root=root,
    )
    _validate_sidecar(
        protocol,
        protocol_sidecar,
        expected_sha256=SOURCE_PROMPTS_PROTOCOL_SHA256,
        trusted_root=root,
    )
    _validate_model_config(model)
    protocol_text = _authenticated_readonly_file(protocol, trusted_root=root).decode("utf-8")
    _validate_source_protocol_text(protocol_text)
    _load_source_views(root)
    records = {
        "model_config": _file_record(model, trusted_root=root),
        "model_config_sidecar": _file_record(model_sidecar, trusted_root=root),
        "common_seed_source_prompts": _file_record(source, trusted_root=root),
        "common_seed_source_prompts_sidecar": _file_record(source_sidecar, trusted_root=root),
        "common_seed_source_prompts_protocol": _file_record(protocol, trusted_root=root),
        "common_seed_source_prompts_protocol_sidecar": _file_record(
            protocol_sidecar, trusted_root=root
        ),
    }
    if set(records) != SOURCE_PROTOCOL_INPUT_ROLES:
        raise AssertionError("Flux-v3 static protocol role construction drifted.")
    return records


def _validate_source_protocol_text(protocol_text: str) -> None:
    if (
        SOURCE_PROMPTS_CONTRACT_ID not in protocol_text
        or "preregistered" not in protocol_text.lower()
        or re.search(r"\bdraft\b", protocol_text, flags=re.IGNORECASE)
    ):
        raise ValueError("Flux-v3 source prompt protocol is not a final preregistration.")


def _validate_model_config(path: Path) -> None:
    root = _lexical_absolute(path).parents[2]
    _validate_model_config_payload(
        _authenticated_readonly_file(path, trusted_root=root)
    )


def _validate_model_config_payload(payload: bytes) -> None:
    raw = yaml.safe_load(payload.decode("utf-8"))
    expected = {
        "model": {
            "adapter": ADAPTER_KEY,
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "diffusers_pipeline_class": PIPELINE_CLASS,
            "verified_model_index": True,
            "components": [
                "scheduler",
                "text_encoder",
                "text_encoder_2",
                "tokenizer",
                "tokenizer_2",
                "transformer",
                "vae",
            ],
        },
        "generation": {
            "task": "text_to_image",
            "height": 1024,
            "width": 1024,
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
        },
    }
    if raw != expected:
        raise ValueError("Dedicated Flux-v3 model config semantics drifted.")


def _load_source_views(project_root: str | Path) -> dict[str, dict[str, Any]]:
    root = _lexical_absolute(Path(project_root))
    path = root / SOURCE_PROMPTS_RELATIVE
    source_raw = _authenticated_readonly_file(path, trusted_root=root)
    if hashlib.sha256(source_raw).hexdigest() != SOURCE_PROMPTS_SHA256:
        raise ValueError("Flux-v3 source prompt contract digest drifted.")
    return _load_source_views_payload(source_raw)


def _load_source_views_payload(source_raw: bytes) -> dict[str, dict[str, Any]]:
    raw = yaml.safe_load(source_raw.decode("utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("Flux-v3 source prompt contract must contain one mapping.")
    expected_model = {
        "model_name": PUBLIC_MODEL_NAME,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "pipeline_class": PIPELINE_CLASS,
    }
    expected_conditioning = {
        "schema_version": 1,
        "primary_argument": "prompt",
        "primary_encoder": "CLIPTextModel",
        "primary_tokenizer_subfolder": "tokenizer",
        "primary_max_tokens_including_special_tokens": 77,
        "secondary_argument": "prompt_2",
        "secondary_encoder": "T5EncoderModel",
        "secondary_tokenizer_subfolder": "tokenizer_2",
        "secondary_max_tokens_including_special_tokens": 512,
        "require_independent_prompt_views": True,
        "require_no_primary_or_secondary_truncation": True,
        "token_ids_hash_encoding": "compact_json_integer_array_utf8",
        "native_equivalence_required_before_manifest_publication": True,
        "steering_concept_prompt_behavior": "legacy_same_text_for_prompt_and_prompt_2",
        "native_negative_conditioning_requires_paired_negative_prompt_views": True,
        "negative_views_must_not_contain_target_positive_terms": True,
    }
    if (
        raw.get("schema_version") != 1
        or raw.get("contract_id") != SOURCE_PROMPTS_CONTRACT_ID
        or raw.get("status") != "preregistered_before_generation"
        or raw.get("model") != expected_model
        or raw.get("conditioning_contract") != expected_conditioning
    ):
        raise ValueError("Flux-v3 source prompt contract identity/semantics drifted.")
    rows = raw.get("prompts")
    if not isinstance(rows, list) or [row.get("prompt_id") for row in rows] != list(PROMPT_IDS):
        raise ValueError("Flux-v3 source prompt order/coverage drifted.")
    normalized: dict[str, dict[str, Any]] = {}
    for raw_row in rows:
        if not isinstance(raw_row, Mapping):
            raise ValueError("Flux-v3 source prompt row is malformed.")
        row = deepcopy(dict(raw_row))
        prompt_id = str(row.get("prompt_id"))
        text_hashes = (
            ("clip_prompt", "clip_string_sha256"),
            ("t5_prompt_2", "t5_string_sha256"),
            ("negative_clip_prompt", "negative_clip_string_sha256"),
            ("negative_t5_prompt_2", "negative_t5_string_sha256"),
        )
        for text_key, digest_key in text_hashes:
            text = row.get(text_key)
            if (
                not isinstance(text, str)
                or not text.strip()
                or row.get(digest_key) != _text_sha256(text)
            ):
                raise ValueError(f"Flux-v3 {prompt_id} {text_key} bytes drifted.")
        counts = (
            ("clip_token_count", "clip_token_ids_sha256", 77),
            ("t5_token_count", "t5_token_ids_sha256", 512),
            ("negative_clip_token_count", "negative_clip_token_ids_sha256", 77),
            ("negative_t5_token_count", "negative_t5_token_ids_sha256", 512),
        )
        for count_key, digest_key, maximum in counts:
            count = row.get(count_key)
            if (
                isinstance(count, bool)
                or not isinstance(count, int)
                or not 1 <= count <= maximum
                or _SHA256_RE.fullmatch(str(row.get(digest_key))) is None
            ):
                raise ValueError(f"Flux-v3 {prompt_id} {count_key} fingerprint drifted.")
        if row["clip_prompt"] == row["t5_prompt_2"]:
            raise ValueError(f"Flux-v3 {prompt_id} positive views are mirrored.")
        if row["negative_clip_prompt"] == row["negative_t5_prompt_2"]:
            raise ValueError(f"Flux-v3 {prompt_id} negative views are mirrored.")
        normalized[prompt_id] = row
    return normalized


def _positive_plan(source: Mapping[str, Any]) -> dict[str, Any]:
    return {key: deepcopy(source[key]) for key in POSITIVE_FIELDS}


def _negative_plan(source: Mapping[str, Any]) -> dict[str, Any]:
    source_to_plan = {
        "negative_clip_prompt": "clip_negative_prompt",
        "negative_t5_prompt_2": "t5_negative_prompt_2",
        "negative_clip_string_sha256": "clip_negative_string_sha256",
        "negative_clip_token_count": "clip_negative_token_count",
        "negative_clip_token_ids_sha256": "clip_negative_token_ids_sha256",
        "negative_t5_string_sha256": "t5_negative_string_sha256",
        "negative_t5_token_count": "t5_negative_token_count",
        "negative_t5_token_ids_sha256": "t5_negative_token_ids_sha256",
    }
    return {target: deepcopy(source[source_key]) for source_key, target in source_to_plan.items()}


def _validate_record_shape(role: str, value: Any) -> dict[str, str]:
    if (
        not isinstance(value, Mapping)
        or set(value) != {"path", "sha256"}
        or not isinstance(value.get("path"), str)
        or not Path(str(value["path"])).is_absolute()
        or _SHA256_RE.fullmatch(str(value.get("sha256"))) is None
    ):
        raise ValueError(f"Flux-v3 protocol input {role!r} is malformed.")
    return {"path": str(value["path"]), "sha256": str(value["sha256"])}


def _normalize_protocol_inputs(
    protocol_inputs: Mapping[str, Any], *, expected_roles: frozenset[str]
) -> dict[str, dict[str, str]]:
    if not isinstance(protocol_inputs, Mapping) or set(protocol_inputs) != set(expected_roles):
        found = sorted(protocol_inputs) if isinstance(protocol_inputs, Mapping) else protocol_inputs
        raise ValueError(
            "Flux-v3 protocol input roles are not exact: "
            f"expected={sorted(expected_roles)}, found={found}."
        )
    return {
        role: _validate_record_shape(role, protocol_inputs[role])
        for role in sorted(expected_roles)
    }


def _validate_static_bindings(
    protocol_inputs: Mapping[str, Any], *, project_root: Path
) -> dict[str, dict[str, str]]:
    supplied = {
        role: protocol_inputs[role]
        for role in SOURCE_PROTOCOL_INPUT_ROLES
        if role in protocol_inputs
    }
    expected = build_flux1_static_protocol_inputs_v3(project_root)
    normalized = _normalize_protocol_inputs(supplied, expected_roles=SOURCE_PROTOCOL_INPUT_ROLES)
    if normalized != expected:
        raise ValueError("Flux-v3 source/model protocol bindings differ from live sealed bytes.")
    return normalized


def _validate_archived_static_bindings(
    protocol_inputs: Mapping[str, Any],
    *,
    snapshot_root: Path,
    archived_input_paths: Mapping[str, str | Path],
) -> dict[str, dict[str, Any]]:
    """Authenticate source/model semantics from content-addressed audit objects."""

    normalized = _normalize_protocol_inputs(
        {
            role: protocol_inputs[role]
            for role in SOURCE_PROTOCOL_INPUT_ROLES
            if role in protocol_inputs
        },
        expected_roles=SOURCE_PROTOCOL_INPUT_ROLES,
    )
    if not isinstance(archived_input_paths, Mapping) or set(archived_input_paths) != set(
        SOURCE_PROTOCOL_INPUT_ROLES
    ):
        raise ValueError("Flux-v3 snapshot audit lacks the exact static input role set.")
    trusted_root = _lexical_absolute(snapshot_root)
    root_metadata = os.lstat(trusted_root)
    if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
        raise ValueError("Flux-v3 snapshot root is symlinked or non-directory.")
    payloads: dict[str, bytes] = {}
    for role in sorted(SOURCE_PROTOCOL_INPUT_ROLES):
        path = _lexical_absolute(Path(archived_input_paths[role]))
        payload = _authenticated_readonly_file(
            path,
            trusted_root=trusted_root,
            require_read_only=False,
        )
        actual = hashlib.sha256(payload).hexdigest()
        if actual != normalized[role]["sha256"]:
            raise ValueError(
                f"Flux-v3 snapshot object differs from the frozen {role!r} binding."
            )
        payloads[role] = payload

    _validate_sidecar_payloads(
        payloads["model_config"],
        payload_name=MODEL_CONFIG_RELATIVE.name,
        sidecar_raw=payloads["model_config_sidecar"],
        expected_sha256=MODEL_CONFIG_SHA256,
        label="archived model_config_sidecar",
    )
    _validate_sidecar_payloads(
        payloads["common_seed_source_prompts"],
        payload_name=SOURCE_PROMPTS_RELATIVE.name,
        sidecar_raw=payloads["common_seed_source_prompts_sidecar"],
        expected_sha256=SOURCE_PROMPTS_SHA256,
        label="archived common_seed_source_prompts_sidecar",
    )
    _validate_sidecar_payloads(
        payloads["common_seed_source_prompts_protocol"],
        payload_name=SOURCE_PROMPTS_PROTOCOL_RELATIVE.name,
        sidecar_raw=payloads["common_seed_source_prompts_protocol_sidecar"],
        expected_sha256=SOURCE_PROMPTS_PROTOCOL_SHA256,
        label="archived common_seed_source_prompts_protocol_sidecar",
    )
    _validate_model_config_payload(payloads["model_config"])
    _validate_source_protocol_text(
        payloads["common_seed_source_prompts_protocol"].decode("utf-8")
    )
    return _load_source_views_payload(payloads["common_seed_source_prompts"])


def expected_flux1_protocol_input_roles_v3(mode: str) -> frozenset[str]:
    if mode == MODE_PREVIEW:
        return SOURCE_PROTOCOL_INPUT_ROLES
    if mode in {MODE_EXECUTION, MODE_AUDIT}:
        return execution_protocol_input_roles_v3()
    raise ValueError(f"Unknown Flux-v3 validation mode {mode!r}.")


def equivalence_protocol_input_roles_v3() -> frozenset[str]:
    """Lazily consume the admission module's single-sourced exact role set."""

    from hierasafe_flow.evaluation import (
        flux1_dual_view_equivalence_admission_v3 as equivalence_admission,
    )

    roles = equivalence_admission.PROTOCOL_INPUT_BINDING_ROLES
    if not isinstance(roles, frozenset) or len(roles) != 9:
        raise RuntimeError("Native-equivalence admission role export drifted.")
    return roles


def execution_protocol_input_roles_v3() -> frozenset[str]:
    return frozenset(SOURCE_PROTOCOL_INPUT_ROLES | equivalence_protocol_input_roles_v3())


def load_flux1_execution_protocol_inputs_v3(
    acceptance_receipt_path: str | Path,
    *,
    project_root: str | Path,
) -> dict[str, dict[str, str]]:
    """Load the exact 6+9 execution DAG from one explicit accepted receipt.

    The caller must name the canonical ``acceptance_receipt.json`` itself.  We
    deliberately do not scan an audit directory, choose a newest attempt, or
    accept a caller-assembled mapping.  The admission validator reopens the
    canonical sibling result, terminal receipt, preregistration, and every
    sidecar before the resulting fifteen-role union is normalized and checked
    a second time through the shared main-v3 admission boundary.
    """

    from hierasafe_flow.evaluation import (
        flux1_dual_view_equivalence_admission_v3 as equivalence_admission,
    )

    root = _lexical_absolute(Path(project_root))
    supplied_receipt = Path(acceptance_receipt_path).expanduser()
    if not supplied_receipt.is_absolute():
        raise ValueError("Flux-v3 execution admission receipt path must be absolute.")
    receipt_path = _lexical_absolute(supplied_receipt)
    if (
        receipt_path.name != equivalence_admission.RECEIPT_FILENAME
        or receipt_path != receipt_path.resolve(strict=False)
    ):
        raise ValueError(
            "Flux-v3 execution admission must name the canonical, unaliased "
            "acceptance_receipt.json path."
        )
    result_path = receipt_path.parent / equivalence_admission.RESULT_FILENAME
    result_sidecar_path = Path(f"{result_path}.sha256")
    receipt_sidecar_path = Path(f"{receipt_path}.sha256")
    admitted = equivalence_admission.validate_acceptance_receipt(
        result_path=result_path,
        result_sidecar_path=result_sidecar_path,
        receipt_path=receipt_path,
        receipt_sidecar_path=receipt_sidecar_path,
        project_root=root,
        require_latest=True,
        require_live_preregistration_sources=True,
    )

    validator_path = root / equivalence_admission.ADMISSION_SOURCE_RELATIVE
    validator_raw = _authenticated_readonly_file(
        validator_path,
        trusted_root=root,
        require_read_only=True,
    )
    admission_inputs = {
        "native_equivalence_admission_validator": {
            "path": str(validator_path),
            "sha256": hashlib.sha256(validator_raw).hexdigest(),
        },
        "native_equivalence_preregistration": {
            "path": str(admitted["preregistration"]["path"]),
            "sha256": str(admitted["preregistration"]["raw_sha256"]),
        },
        "native_equivalence_preregistration_sidecar": {
            "path": str(admitted["preregistration"]["sidecar_path"]),
            "sha256": str(admitted["preregistration"]["sidecar_sha256"]),
        },
        "native_equivalence_result": {
            "path": str(admitted["result"]["path"]),
            "sha256": str(admitted["result"]["sha256"]),
        },
        "native_equivalence_result_sidecar": {
            "path": str(admitted["result"]["sidecar_path"]),
            "sha256": str(admitted["result"]["sidecar_sha256"]),
        },
        "native_equivalence_terminal_receipt": {
            "path": str(admitted["terminal_execution_receipt"]["path"]),
            "sha256": str(admitted["terminal_execution_receipt"]["sha256"]),
        },
        "native_equivalence_terminal_receipt_sidecar": {
            "path": str(admitted["terminal_execution_receipt"]["sidecar_path"]),
            "sha256": str(admitted["terminal_execution_receipt"]["sidecar_sha256"]),
        },
        "native_equivalence_receipt": {
            "path": str(admitted["receipt"]["path"]),
            "sha256": str(admitted["receipt"]["sha256"]),
        },
        "native_equivalence_receipt_sidecar": {
            "path": str(admitted["receipt"]["sidecar_path"]),
            "sha256": str(admitted["receipt"]["sidecar_sha256"]),
        },
    }
    expected_roles = execution_protocol_input_roles_v3()
    combined = {
        **build_flux1_static_protocol_inputs_v3(root),
        **admission_inputs,
    }
    if (
        len(SOURCE_PROTOCOL_INPUT_ROLES) != 6
        or len(equivalence_protocol_input_roles_v3()) != 9
        or len(expected_roles) != 15
        or set(combined) != set(expected_roles)
    ):
        raise RuntimeError("Flux-v3 execution protocol role union is not exact 6+9=15.")
    normalized = _normalize_protocol_inputs(combined, expected_roles=expected_roles)
    equivalence_admission.validate_protocol_input_bindings(
        protocol_inputs=normalized,
        project_root=root,
    )
    return normalized


def project_flux1_job_v3(
    job: Mapping[str, Any],
    *,
    project_root: str | Path,
    protocol_inputs: Mapping[str, Any] | None = None,
    mode: str = MODE_PREVIEW,
    negative_mode: str | None = None,
    native_pipeline_execution: bool | None = None,
) -> dict[str, Any]:
    """Project one public ``flux1_dev`` job onto the dual-view v3 route."""

    if mode not in {MODE_PREVIEW, MODE_EXECUTION}:
        raise ValueError("Flux-v3 projection mode must be preview_v3 or execution_v3.")
    root = _lexical_absolute(Path(project_root))
    if job.get("model_name") != PUBLIC_MODEL_NAME:
        raise ValueError("Flux-v3 projector accepts only public model_name='flux1_dev'.")
    if (job.get("generation") or {}).get("task") != "text_to_image":
        raise ValueError("Flux-v3 campaign projection is versioned only for text-to-image jobs.")
    prompt_id = str(job.get("prompt_id"))
    if prompt_id not in PROMPT_IDS:
        raise ValueError(f"Flux-v3 projector rejects unknown prompt ID {prompt_id!r}.")

    static_inputs = build_flux1_static_protocol_inputs_v3(root)
    if protocol_inputs is None:
        supplied: dict[str, Any] = deepcopy(static_inputs)
    else:
        supplied = deepcopy(dict(protocol_inputs))
        for role, record in static_inputs.items():
            supplied.setdefault(role, deepcopy(record))
    expected_roles = expected_flux1_protocol_input_roles_v3(mode)
    normalized_inputs = _normalize_protocol_inputs(supplied, expected_roles=expected_roles)
    if {role: normalized_inputs[role] for role in SOURCE_PROTOCOL_INPUT_ROLES} != static_inputs:
        raise ValueError("Flux-v3 projection source/model bindings drifted.")
    if mode == MODE_EXECUTION:
        from hierasafe_flow.evaluation import (
            flux1_dual_view_equivalence_admission_v3 as equivalence_admission,
        )

        equivalence_admission.validate_protocol_input_bindings(
            protocol_inputs=normalized_inputs,
            project_root=root,
        )

    projected = deepcopy(dict(job))
    source = _load_source_views(root)[prompt_id]
    variant_spec = projected.get("variant_spec")
    if not isinstance(variant_spec, Mapping):
        raise ValueError("Flux-v3 job has no variation specification.")
    native_variant = variant_spec.get("kind") == "native_negative_prompt"
    if native_variant and variant_spec.get("capability") != "supported":
        raise ValueError("Flux-v3 native-negative projection requires native support.")
    if negative_mode is None:
        negative_mode = (
            NEGATIVE_MODE_PAIRED_REGISTERED
            if native_variant
            else NEGATIVE_MODE_NOT_APPLIED
        )
    if negative_mode not in NEGATIVE_MODES:
        raise ValueError(f"Unknown Flux-v3 negative mode {negative_mode!r}.")
    if native_pipeline_execution is None:
        native_pipeline_execution = negative_mode != NEGATIVE_MODE_NOT_APPLIED
    if not isinstance(native_pipeline_execution, bool):
        raise TypeError("Flux-v3 native_pipeline_execution must be a boolean.")
    if (negative_mode == NEGATIVE_MODE_NOT_APPLIED) == native_pipeline_execution:
        raise ValueError(
            "Flux-v3 not_applied requires the GenerationRunner route; explicit-none and "
            "paired modes require the native pipeline route."
        )
    if native_variant != native_pipeline_execution:
        raise ValueError(
            "Flux-v3 variant kind and native_pipeline_execution disagree; calibration "
            "controls must use a native-negative variant specification."
        )
    positive = _positive_plan(source)
    negative = (
        _negative_plan(source)
        if negative_mode == NEGATIVE_MODE_PAIRED_REGISTERED
        else None
    )
    plan = validate_flux_dual_view_conditioning(
        {"schema_version": 1, "positive": positive, "negative": negative}
    )

    projected["prompt"] = positive["clip_prompt"]
    prompt_snapshot = deepcopy(dict(projected.get("prompt_snapshot") or {}))
    prompt_snapshot["prompt"] = positive["clip_prompt"]
    prompt_snapshot["selected_prompt_field"] = "v3_source_prompt_clip_prompt"
    projected["prompt_snapshot"] = prompt_snapshot
    generation = deepcopy(dict(projected.get("generation") or {}))
    generation[FLUX_DUAL_VIEW_CONFIG_KEY] = deepcopy(plan)
    projected["generation"] = generation
    projected["model_config"] = static_inputs["model_config"]["path"]
    projected["model_revision"] = MODEL_REVISION
    if negative_mode == NEGATIVE_MODE_PAIRED_REGISTERED:
        assert negative is not None
        projected["negative_prompt"] = negative["clip_negative_prompt"]
    elif negative_mode == NEGATIVE_MODE_EXPLICIT_NONE_CONTROL:
        projected["negative_prompt"] = None

    base_inputs = deepcopy(dict(projected.get("input_files") or {}))
    base_inputs["model_config"] = deepcopy(static_inputs["model_config"])
    for role, record in normalized_inputs.items():
        base_inputs[role] = deepcopy(record)
    projected["input_files"] = base_inputs
    projected["flux1_dual_view_source_contract"] = {
        "schema_version": 1,
        "contract_id": SOURCE_PROMPTS_CONTRACT_ID,
        "prompt_id": prompt_id,
        "source_file": deepcopy(normalized_inputs["common_seed_source_prompts"]),
        "positive": deepcopy(positive),
        "registered_negative": _negative_plan(source),
        "negative_mode": negative_mode,
        "native_pipeline_execution": native_pipeline_execution,
    }
    status = PREVIEW_STATUS if mode == MODE_PREVIEW else EXECUTION_STATUS
    projected["flux1_dual_view_route_v3"] = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "mode": mode,
        "status": status,
        "public_model_name": PUBLIC_MODEL_NAME,
        "adapter": ADAPTER_KEY,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pipeline_class": PIPELINE_CLASS,
        "prompt_id": prompt_id,
        "negative_mode": negative_mode,
        "native_pipeline_execution": native_pipeline_execution,
        "internal_steering_prompt_views": "legacy_mirrored_internal_prompt",
        "protocol_inputs": deepcopy(normalized_inputs),
    }
    validate_flux1_job_v3(projected, project_root=root, mode=mode)
    return projected


def is_flux1_job_v3(job: Mapping[str, Any]) -> bool:
    return isinstance(job.get("flux1_dual_view_route_v3"), Mapping)


def _validate_job_plan_against_source(
    job: Mapping[str, Any], *, source: Mapping[str, Any] | None
) -> dict[str, Any]:
    generation = job.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("Flux-v3 generation mapping is absent.")
    plan = validate_flux_dual_view_conditioning(generation.get(FLUX_DUAL_VIEW_CONFIG_KEY))
    positive = plan["positive"]
    if job.get("prompt") != positive["clip_prompt"]:
        raise ValueError("Flux-v3 job.prompt is not the registered compact CLIP view.")
    if positive["clip_prompt"] == positive["t5_prompt_2"]:
        raise ValueError("Flux-v3 positive CLIP/T5 source views are mirrored.")
    prompt_snapshot = job.get("prompt_snapshot")
    if (
        not isinstance(prompt_snapshot, Mapping)
        or prompt_snapshot.get("prompt") != positive["clip_prompt"]
        or prompt_snapshot.get("selected_prompt_field") != "v3_source_prompt_clip_prompt"
    ):
        raise ValueError("Flux-v3 prompt semantic snapshot does not bind the compact CLIP view.")
    variant = job.get("variant_spec")
    native_variant = isinstance(variant, Mapping) and variant.get("kind") == "native_negative_prompt"
    route = job.get("flux1_dual_view_route_v3")
    negative_mode = route.get("negative_mode") if isinstance(route, Mapping) else None
    native_pipeline = (
        route.get("native_pipeline_execution") if isinstance(route, Mapping) else None
    )
    if negative_mode not in NEGATIVE_MODES or not isinstance(native_pipeline, bool):
        raise ValueError("Flux-v3 negative-mode/native-pipeline role metadata is malformed.")
    if native_variant != native_pipeline:
        raise ValueError("Flux-v3 native variant and native-pipeline role disagree.")
    if negative_mode == NEGATIVE_MODE_PAIRED_REGISTERED:
        if (
            not native_pipeline
            or not isinstance(variant, Mapping)
            or variant.get("capability") != "supported"
            or plan["negative"] is None
        ):
            raise ValueError("Flux-v3 paired native job lacks its registered negative plan.")
        if job.get("negative_prompt") != plan["negative"]["clip_negative_prompt"]:
            raise ValueError("Flux-v3 native negative prompt is not the registered CLIP view.")
    elif plan["negative"] is not None:
        raise ValueError("Only paired_registered_negative may carry negative conditioning.")
    elif negative_mode == NEGATIVE_MODE_EXPLICIT_NONE_CONTROL:
        if not native_pipeline or job.get("negative_prompt") is not None:
            raise ValueError("Flux-v3 explicit-none control must carry an exact null negative.")
    elif native_pipeline or native_variant:
        raise ValueError("Flux-v3 not_applied mode cannot execute through the native pipeline.")
    if source is not None:
        if positive != _positive_plan(source):
            raise ValueError("Flux-v3 positive plan differs from the sealed source contract.")
        expected_negative = (
            _negative_plan(source)
            if negative_mode == NEGATIVE_MODE_PAIRED_REGISTERED
            else None
        )
        if plan["negative"] != expected_negative:
            raise ValueError("Flux-v3 negative plan differs from the sealed source contract.")
    source_contract = job.get("flux1_dual_view_source_contract")
    if not isinstance(source_contract, Mapping):
        raise ValueError("Flux-v3 source-contract projection is absent.")
    registered_negative = source_contract.get("registered_negative")
    if (
        set(source_contract)
        != {
            "schema_version",
            "contract_id",
            "prompt_id",
            "source_file",
            "positive",
            "registered_negative",
            "negative_mode",
            "native_pipeline_execution",
        }
        or source_contract.get("schema_version") != 1
        or source_contract.get("contract_id") != SOURCE_PROMPTS_CONTRACT_ID
        or source_contract.get("prompt_id") != job.get("prompt_id")
        or source_contract.get("positive") != positive
        or not isinstance(registered_negative, Mapping)
        or set(registered_negative) != NEGATIVE_FIELDS
        or source_contract.get("negative_mode") != negative_mode
        or source_contract.get("native_pipeline_execution") is not native_pipeline
    ):
        raise ValueError("Flux-v3 embedded source-contract projection drifted.")
    if (
        negative_mode == NEGATIVE_MODE_PAIRED_REGISTERED
        and dict(registered_negative) != plan["negative"]
    ):
        raise ValueError("Flux-v3 applied negative differs from the registered negative views.")
    if source is not None and dict(registered_negative) != _negative_plan(source):
        raise ValueError("Flux-v3 embedded registered-negative views differ from sealed source.")
    return plan


def validate_flux1_job_v3(
    job: Mapping[str, Any],
    *,
    project_root: str | Path,
    mode: str,
    audit_snapshot_root: str | Path | None = None,
    audit_snapshot_input_paths: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Validate one projected job; execution mode reauthenticates the evidence DAG."""

    if mode not in VALIDATION_MODES:
        raise ValueError(f"Unknown Flux-v3 validation mode {mode!r}.")
    root = _lexical_absolute(Path(project_root))
    route = job.get("flux1_dual_view_route_v3")
    if not isinstance(route, Mapping):
        raise ValueError("A generic Flux.1 job is missing its dual-view v3 route marker.")
    route_mode = route.get("mode")
    if route_mode not in {MODE_PREVIEW, MODE_EXECUTION}:
        raise ValueError("Flux-v3 route mode is invalid.")
    expected_status = PREVIEW_STATUS if route_mode == MODE_PREVIEW else EXECUTION_STATUS
    expected_identity = {
        "schema_version": SCHEMA_VERSION,
        "contract_id": CONTRACT_ID,
        "status": expected_status,
        "public_model_name": PUBLIC_MODEL_NAME,
        "adapter": ADAPTER_KEY,
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "pipeline_class": PIPELINE_CLASS,
        "prompt_id": job.get("prompt_id"),
        "internal_steering_prompt_views": "legacy_mirrored_internal_prompt",
    }
    expected_route_keys = {
        *expected_identity,
        "mode",
        "negative_mode",
        "native_pipeline_execution",
        "protocol_inputs",
    }
    if set(route) != expected_route_keys:
        raise ValueError("Flux-v3 route marker schema drifted.")
    for key, expected in expected_identity.items():
        if route.get(key) != expected:
            raise ValueError(f"Flux-v3 route identity drifted for {key!r}.")
    if route.get("negative_mode") not in NEGATIVE_MODES or not isinstance(
        route.get("native_pipeline_execution"), bool
    ):
        raise ValueError("Flux-v3 route negative role metadata drifted.")
    if job.get("model_name") != PUBLIC_MODEL_NAME or job.get("model_revision") != MODEL_REVISION:
        raise ValueError("Flux-v3 public model identity/revision drifted.")
    expected_model_path = root / MODEL_CONFIG_RELATIVE
    if _lexical_absolute(Path(str(job.get("model_config", "")))) != expected_model_path:
        raise ValueError("Flux-v3 job regressed to a non-v3 model config.")

    inputs = route.get("protocol_inputs")
    expected_roles = (
        SOURCE_PROTOCOL_INPUT_ROLES
        if route_mode == MODE_PREVIEW
        else execution_protocol_input_roles_v3()
    )
    normalized = _normalize_protocol_inputs(inputs, expected_roles=expected_roles)
    records = job.get("input_files")
    if not isinstance(records, Mapping):
        raise ValueError("Flux-v3 job input_files mapping is absent.")
    for role, record in normalized.items():
        if records.get(role) != record:
            raise ValueError(f"Flux-v3 input_files binding drifted for {role!r}.")

    # Reauthenticate the permanent live seals whenever they remain available.
    # Historical audit may instead use the exact content-addressed objects
    # bound by the immutable manifest; execution never accepts that fallback.
    try:
        _validate_static_bindings(normalized, project_root=root)
        source_views = _load_source_views(root)
        source_origin = "live_permanent_seals"
    except (OSError, ValueError):
        if (
            mode != MODE_AUDIT
            or audit_snapshot_root is None
            or audit_snapshot_input_paths is None
        ):
            raise
        try:
            source_views = _validate_archived_static_bindings(
                normalized,
                snapshot_root=Path(audit_snapshot_root),
                archived_input_paths=audit_snapshot_input_paths,
            )
        except (OSError, ValueError) as snapshot_error:
            raise ValueError(
                "Flux-v3 live sealed inputs and content-addressed audit objects both failed "
                "authentication."
            ) from snapshot_error
        source_origin = "content_addressed_snapshot"
    source: Mapping[str, Any] | None = source_views[str(job.get("prompt_id"))]
    plan = _validate_job_plan_against_source(job, source=source)
    source_contract = job["flux1_dual_view_source_contract"]
    if source_contract.get("source_file") != normalized["common_seed_source_prompts"]:
        raise ValueError("Flux-v3 source-file binding drifted.")

    if mode == MODE_EXECUTION:
        if route_mode != MODE_EXECUTION or route.get("status") != EXECUTION_STATUS:
            raise ValueError(
                "Flux-v3 preview jobs are nonlaunchable until native equivalence is admitted."
            )
        from hierasafe_flow.evaluation import (
            flux1_dual_view_equivalence_admission_v3 as equivalence_admission,
        )

        admission = equivalence_admission.validate_protocol_input_bindings(
            protocol_inputs=normalized,
            project_root=root,
        )
    else:
        admission = None
    return {
        "mode": route_mode,
        "status": route["status"],
        "plan_sha256": canonical_sha256(plan),
        "source_authentication": source_origin,
        "protocol_input_roles": sorted(normalized),
        "equivalence_admission": admission,
    }


def _view_record(*, call_key: str, encoder: str, text: str) -> dict[str, Any]:
    return {
        "call_key": call_key,
        "encoder": encoder,
        "text": text,
        "utf8_sha256": _text_sha256(text),
    }


def expected_flux1_conditioning_preflight_v3(plan: Mapping[str, Any]) -> dict[str, Any]:
    normalized = validate_flux_dual_view_conditioning(plan)
    positive = normalized["positive"]
    views = [
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
    ]
    negative = normalized["negative"]
    if negative is not None:
        views.extend(
            [
                {
                    "role": "negative.clip",
                    "call_key": "negative_prompt",
                    "encoder": "CLIPTextModel",
                    "text_utf8_sha256": negative["clip_negative_string_sha256"],
                    "token_count": negative["clip_negative_token_count"],
                    "token_ids_sha256": negative["clip_negative_token_ids_sha256"],
                    "maximum_tokens_including_special_tokens": 77,
                    "truncated": False,
                },
                {
                    "role": "negative.t5",
                    "call_key": "negative_prompt_2",
                    "encoder": "T5EncoderModel",
                    "text_utf8_sha256": negative["t5_negative_string_sha256"],
                    "token_count": negative["t5_negative_token_count"],
                    "token_ids_sha256": negative["t5_negative_token_ids_sha256"],
                    "maximum_tokens_including_special_tokens": 512,
                    "truncated": False,
                },
            ]
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "plan_sha256": canonical_sha256(normalized),
        "require_no_primary_or_secondary_truncation": True,
        "token_ids_hash_encoding": "compact_json_integer_array_utf8",
        "views": views,
    }


def _validate_runtime_provenance(
    provenance: Any, *, plan: Mapping[str, Any], negative_mode: str
) -> None:
    if not isinstance(provenance, Mapping):
        raise ValueError("Flux-v3 runtime conditioning provenance is absent.")
    normalized = validate_flux_dual_view_conditioning(plan)
    positive = normalized["positive"]
    negative = normalized["negative"]
    expected_positive = {
        "clip": _view_record(
            call_key="prompt", encoder="CLIPTextModel", text=positive["clip_prompt"]
        )
        | {
            "frozen_token_count": positive["clip_token_count"],
            "frozen_token_ids_sha256": positive["clip_token_ids_sha256"],
        },
        "t5": _view_record(
            call_key="prompt_2", encoder="T5EncoderModel", text=positive["t5_prompt_2"]
        )
        | {
            "frozen_token_count": positive["t5_token_count"],
            "frozen_token_ids_sha256": positive["t5_token_ids_sha256"],
        },
    }
    expected_negative = None
    if negative is not None:
        expected_negative = {
            "clip": _view_record(
                call_key="negative_prompt",
                encoder="CLIPTextModel",
                text=negative["clip_negative_prompt"],
            )
            | {
                "frozen_token_count": negative["clip_negative_token_count"],
                "frozen_token_ids_sha256": negative["clip_negative_token_ids_sha256"],
            },
            "t5": _view_record(
                call_key="negative_prompt_2",
                encoder="T5EncoderModel",
                text=negative["t5_negative_prompt_2"],
            )
            | {
                "frozen_token_count": negative["t5_negative_token_count"],
                "frozen_token_ids_sha256": negative["t5_negative_token_ids_sha256"],
            },
        }
    fixed = {
        "schema_version": 1,
        "method": "flux1_registered_dual_prompt_views",
        "plan_sha256": canonical_sha256(normalized),
        "positive": expected_positive,
        "negative": expected_negative,
        "runtime_preflight": expected_flux1_conditioning_preflight_v3(normalized),
    }
    if set(provenance) != {
        *fixed,
        "encode_calls",
        "native_paired_calls",
    }:
        raise ValueError("Flux-v3 runtime provenance schema drifted.")
    for key, expected in fixed.items():
        if provenance.get(key) != expected:
            raise ValueError(f"Flux-v3 runtime provenance drifted for {key!r}.")
    encode_calls = provenance.get("encode_calls")
    native_calls = provenance.get("native_paired_calls")
    if not isinstance(encode_calls, list) or not isinstance(native_calls, list):
        raise ValueError("Flux-v3 runtime provenance call ledgers are malformed.")
    exact_positive_call = {
        "sequence_index": 0,
        "role": "registered_positive",
        "registered_dual_view_applied": True,
        "clip": _view_record(
            call_key="prompt", encoder="CLIPTextModel", text=positive["clip_prompt"]
        ),
        "t5": _view_record(
            call_key="prompt_2", encoder="T5EncoderModel", text=positive["t5_prompt_2"]
        ),
    }
    if negative_mode == NEGATIVE_MODE_PAIRED_REGISTERED:
        if encode_calls != [] or len(native_calls) != 1 or negative is None:
            raise ValueError("Flux-v3 native-negative runtime lacks one paired native call.")
        expected_native_call = {
            "sequence_index": 0,
            "method": "native_flux_pipeline_paired_prompt_views",
            "positive": {
                "clip": exact_positive_call["clip"],
                "t5": exact_positive_call["t5"],
            },
            "negative": {
                "clip": _view_record(
                    call_key="negative_prompt",
                    encoder="CLIPTextModel",
                    text=negative["clip_negative_prompt"],
                ),
                "t5": _view_record(
                    call_key="negative_prompt_2",
                    encoder="T5EncoderModel",
                    text=negative["t5_negative_prompt_2"],
                ),
            },
        }
        if native_calls != [expected_native_call]:
            raise ValueError("Flux-v3 native-negative call did not use paired prompt views.")
    elif negative_mode == NEGATIVE_MODE_EXPLICIT_NONE_CONTROL:
        if encode_calls != [] or negative is not None:
            raise ValueError("Flux-v3 explicit-none control encoded an unintended negative view.")
        expected_explicit_none_call = {
            "sequence_index": 0,
            "method": "native_flux_pipeline_dual_positive_explicit_none_negative_views",
            "positive": {
                "clip": exact_positive_call["clip"],
                "t5": exact_positive_call["t5"],
            },
            "negative": None,
            "explicit_none_arguments": ["negative_prompt", "negative_prompt_2"],
        }
        if native_calls != [expected_explicit_none_call]:
            raise ValueError("Flux-v3 explicit-none native-call ledger drifted.")
    else:
        if negative_mode != NEGATIVE_MODE_NOT_APPLIED:
            raise ValueError(f"Unknown Flux-v3 runtime negative mode {negative_mode!r}.")
        if native_calls:
            raise ValueError("Non-native Flux-v3 generation unexpectedly made a native paired call.")
        if not encode_calls:
            raise ValueError("Flux-v3 GenerationRunner provenance has an empty encode ledger.")
        registered_positive_count = 0
        for index, call in enumerate(encode_calls):
            if not isinstance(call, Mapping) or call.get("sequence_index") != index:
                raise ValueError("Flux-v3 encode-call sequence is malformed or noncontiguous.")
            if call.get("role") == "registered_positive":
                expected = deepcopy(exact_positive_call)
                expected["sequence_index"] = index
                if dict(call) != expected:
                    raise ValueError("Flux-v3 registered-positive encode call drifted.")
                registered_positive_count += 1
                continue
            if (
                set(call)
                != {
                    "sequence_index",
                    "role",
                    "registered_dual_view_applied",
                    "clip",
                    "t5",
                }
                or call.get("role") != "legacy_mirrored_internal_prompt"
                or call.get("registered_dual_view_applied") is not False
            ):
                raise ValueError("Flux-v3 internal encode call has an unregistered role/schema.")
            clip = call.get("clip")
            t5 = call.get("t5")
            if (
                not isinstance(clip, Mapping)
                or not isinstance(t5, Mapping)
                or set(clip) != {"call_key", "encoder", "text", "utf8_sha256"}
                or set(t5) != {"call_key", "encoder", "text", "utf8_sha256"}
                or clip.get("call_key") != "prompt"
                or clip.get("encoder") != "CLIPTextModel"
                or t5.get("call_key") != "prompt_2"
                or t5.get("encoder") != "T5EncoderModel"
                or not isinstance(clip.get("text"), str)
                or not clip["text"].strip()
                or t5.get("text") != clip.get("text")
                or clip.get("utf8_sha256") != _text_sha256(str(clip["text"]))
                or t5.get("utf8_sha256") != clip.get("utf8_sha256")
            ):
                raise ValueError("Flux-v3 internal steering prompt was not exactly mirrored.")
        if registered_positive_count != 1:
            raise ValueError("Flux-v3 GenerationRunner must encode its registered positive once.")


def _validate_conditioning_cache(
    cache: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    sample_report: Mapping[str, Any] | None,
) -> None:
    records = cache.get("records")
    if (
        set(cache) != {"schema_version", "namespace", "records"}
        or cache.get("schema_version") != 1
        or cache.get("namespace") != "generation_runner"
        or not isinstance(records, list)
        or not records
    ):
        raise ValueError("Flux-v3 conditioning-cache evidence is absent or malformed.")
    normalized = validate_flux_dual_view_conditioning(plan)
    expected_plan_sha = canonical_sha256(normalized)
    positive = normalized["positive"]
    first_status: dict[str, str] = {}
    fingerprints: dict[str, str] = {}
    expected_record_keys = {
        "sequence_index",
        "namespace",
        "status",
        "call_role",
        "prompt_view",
        "prompt_sha256",
        "identity_sha256",
        "encoding_fingerprint",
        "identity",
    }
    for index, record in enumerate(records):
        if (
            not isinstance(record, Mapping)
            or set(record) != expected_record_keys
            or record.get("sequence_index") != index
            or record.get("namespace") != "generation_runner"
            or record.get("status") not in {"miss", "hit"}
            or not isinstance(record.get("call_role"), str)
            or not record["call_role"]
            or not isinstance(record.get("prompt_view"), str)
            or not record["prompt_view"]
            or _SHA256_RE.fullmatch(str(record.get("prompt_sha256"))) is None
            or _SHA256_RE.fullmatch(str(record.get("identity_sha256"))) is None
            or _SHA256_RE.fullmatch(str(record.get("encoding_fingerprint"))) is None
            or not isinstance(record.get("identity"), Mapping)
        ):
            raise ValueError(f"Flux-v3 conditioning-cache record {index} is malformed.")
        identity = record["identity"]
        dual = identity.get("flux_dual_view")
        if (
            record["identity_sha256"] != canonical_sha256(identity)
            or identity.get("adapter") != ADAPTER_KEY
            or identity.get("model_id") != MODEL_ID
            or identity.get("model_revision") != MODEL_REVISION
            or identity.get("prompt_sha256") != record["prompt_sha256"]
            or not isinstance(dual, Mapping)
            or set(dual)
            != {
                "schema_version",
                "plan_sha256",
                "role",
                "clip_prompt_sha256",
                "t5_prompt_2_sha256",
            }
            or dual.get("schema_version") != 1
            or dual.get("plan_sha256") != expected_plan_sha
            or dual.get("clip_prompt_sha256") != record["prompt_sha256"]
        ):
            raise ValueError(f"Flux-v3 conditioning-cache identity {index} drifted.")
        role = dual.get("role")
        if role == "registered_positive":
            if (
                dual.get("clip_prompt_sha256") != positive["clip_string_sha256"]
                or dual.get("t5_prompt_2_sha256") != positive["t5_string_sha256"]
                or record.get("prompt_view") != "registered"
                or record.get("call_role") != "base_current"
            ):
                raise ValueError("Flux-v3 registered-positive cache identity drifted.")
        elif role == "legacy_mirrored_internal_prompt":
            if dual.get("t5_prompt_2_sha256") != dual.get("clip_prompt_sha256"):
                raise ValueError("Flux-v3 internal cache identity is not exactly mirrored.")
        else:
            raise ValueError(f"Flux-v3 cache admitted forbidden prompt role {role!r}.")
        digest = str(record["identity_sha256"])
        prior_status = first_status.get(digest)
        if prior_status is None:
            if record["status"] != "miss":
                raise ValueError("Flux-v3 cache identity first appears as a hit.")
            first_status[digest] = "miss"
            fingerprints[digest] = str(record["encoding_fingerprint"])
        elif (
            record["status"] != "hit"
            or fingerprints[digest] != record["encoding_fingerprint"]
        ):
            raise ValueError("Flux-v3 cache hit/miss or encoding fingerprint drifted.")
    if not any(
        record["identity"]["flux_dual_view"]["role"] == "registered_positive"
        for record in records
    ):
        raise ValueError("Flux-v3 cache never used the registered positive source view.")
    if sample_report is not None:
        interpretability = sample_report.get("interpretability")
        timesteps = (
            interpretability.get("timesteps")
            if isinstance(interpretability, Mapping)
            else None
        )
        if not isinstance(timesteps, list):
            raise ValueError("Flux-v3 report omits timestep/cache linkage.")
        report_generation = sample_report.get("generation")
        expected_steps = (
            report_generation.get("num_inference_steps")
            if isinstance(report_generation, Mapping)
            else None
        )
        if (
            isinstance(expected_steps, bool)
            or not isinstance(expected_steps, int)
            or expected_steps <= 0
            or len(timesteps) != expected_steps
        ):
            raise ValueError("Flux-v3 report timestep count differs from its generation contract.")
        flattened: list[Any] = []
        for index, step in enumerate(timesteps):
            if not isinstance(step, Mapping) or step.get("step_index") != index:
                raise ValueError("Flux-v3 report timestep ordering drifted.")
            calls = step.get("condition_calls")
            if not isinstance(calls, list):
                raise ValueError("Flux-v3 report timestep omits condition_calls.")
            flattened.extend(calls)
        if flattened != records:
            raise ValueError("Flux-v3 report timestep condition_calls differ from cache evidence.")


def validate_flux1_runtime_v3(
    job: Mapping[str, Any],
    *,
    runner_config: Mapping[str, Any] | None = None,
    run_timing: Mapping[str, Any] | None = None,
    sample_report: Mapping[str, Any] | None = None,
    conditioning_cache: Mapping[str, Any] | None = None,
    native_trace: list[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate resolved config and, when supplied, actual runtime evidence.

    Collectors may call this incrementally: a resolved runner config can be
    checked before model load, then the same function can authenticate timing,
    report, cache, and native-call trace after generation.
    """

    generation = job.get("generation")
    if not isinstance(generation, Mapping):
        raise ValueError("Flux-v3 runtime validation requires job generation metadata.")
    plan = validate_flux_dual_view_conditioning(generation.get(FLUX_DUAL_VIEW_CONFIG_KEY))
    route = job.get("flux1_dual_view_route_v3")
    if not isinstance(route, Mapping):
        raise ValueError("Flux-v3 runtime validation requires route metadata.")
    negative_mode = str(route.get("negative_mode"))
    native_pipeline = route.get("native_pipeline_execution")
    if negative_mode not in NEGATIVE_MODES or not isinstance(native_pipeline, bool):
        raise ValueError("Flux-v3 runtime negative role metadata drifted.")
    expected_model = {
        "adapter": ADAPTER_KEY,
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
    }
    expected_runtime_model = expected_model | {"pipeline_class": PIPELINE_CLASS}
    if runner_config is not None:
        model = runner_config.get("model")
        resolved_generation = runner_config.get("generation")
        if not isinstance(model, Mapping) or any(
            model.get(key) != value for key, value in expected_model.items()
        ):
            raise ValueError("Resolved Flux-v3 runner config regressed to the legacy route.")
        if model.get(FLUX_DUAL_VIEW_CONFIG_KEY) != plan:
            raise ValueError("Resolved Flux-v3 model plan differs from the job plan.")
        if model.get("diffusers_pipeline_class") != PIPELINE_CLASS:
            raise ValueError("Resolved Flux-v3 runner config lost the FluxPipeline route.")
        if (
            not isinstance(resolved_generation, Mapping)
            or resolved_generation.get(FLUX_DUAL_VIEW_CONFIG_KEY) != plan
            or resolved_generation.get("prompt") != plan["positive"]["clip_prompt"]
        ):
            raise ValueError("Resolved Flux-v3 generation config lost its dual prompt plan.")
        native = runner_config.get("native_negative_prompt")
        if native_pipeline:
            if not isinstance(native, Mapping) or native.get("prompt") != job.get("negative_prompt"):
                raise ValueError("Resolved Flux-v3 native config lost the registered negative CLIP view.")
        elif isinstance(native, Mapping) and native:
            raise ValueError("Non-native Flux-v3 config unexpectedly applies negative guidance.")

    expected_preflight = expected_flux1_conditioning_preflight_v3(plan)
    if run_timing is not None:
        model = run_timing.get("model")
        timing_generation = run_timing.get("generation")
        if (
            run_timing.get("status") != "completed"
            or not isinstance(model, Mapping)
            or any(model.get(key) != value for key, value in expected_runtime_model.items())
            or not isinstance(timing_generation, Mapping)
            or timing_generation.get(FLUX_DUAL_VIEW_CONFIG_KEY) != plan
            or timing_generation.get("prompt") != plan["positive"]["clip_prompt"]
            or (
                not native_pipeline
                and run_timing.get("conditioning_preflight") != expected_preflight
            )
        ):
            raise ValueError("Flux-v3 run timing does not prove the exact dual-view route/preflight.")
    provenance = None
    if sample_report is not None:
        if sample_report.get("prompt") != plan["positive"]["clip_prompt"]:
            raise ValueError("Flux-v3 sample report prompt differs from the compact CLIP view.")
        report_generation = sample_report.get("generation")
        if (
            not isinstance(report_generation, Mapping)
            or report_generation.get(FLUX_DUAL_VIEW_CONFIG_KEY) != plan
            or report_generation.get("prompt") != plan["positive"]["clip_prompt"]
        ):
            raise ValueError("Flux-v3 sample report generation plan drifted.")
        model = sample_report.get("model")
        if not isinstance(model, Mapping) or any(
            model.get(key) != value for key, value in expected_runtime_model.items()
        ):
            raise ValueError("Flux-v3 sample report model route drifted.")
        provenance = sample_report.get("conditioning_provenance")
        _validate_runtime_provenance(
            provenance,
            plan=plan,
            negative_mode=negative_mode,
        )
        report_cache = sample_report.get("conditioning_cache")
        if conditioning_cache is None and isinstance(report_cache, Mapping):
            conditioning_cache = report_cache
        if native_pipeline and native_trace is None:
            interpretability = sample_report.get("interpretability")
            candidate_trace = (
                interpretability.get("timesteps")
                if isinstance(interpretability, Mapping)
                else None
            )
            if isinstance(candidate_trace, list):
                native_trace = candidate_trace
        if not native_pipeline and not isinstance(conditioning_cache, Mapping):
            raise ValueError("Flux-v3 GenerationRunner report omitted conditioning-cache evidence.")
    if conditioning_cache is not None:
        if native_pipeline:
            raise ValueError("Flux-v3 native-negative reports must not synthesize cache evidence.")
        _validate_conditioning_cache(
            conditioning_cache,
            plan=plan,
            sample_report=sample_report,
        )
    if native_pipeline and sample_report is not None and native_trace is None:
        raise ValueError("Flux-v3 native-negative report omitted its paired native-call trace.")
    if native_trace is not None:
        if not native_pipeline or len(native_trace) != 1:
            raise ValueError("Flux-v3 native trace cardinality/variation is invalid.")
        trace = native_trace[0]
        expected_conditioning_mode = (
            "flux_dual_view_paired_native_prompt_arguments"
            if negative_mode == NEGATIVE_MODE_PAIRED_REGISTERED
            else "flux_dual_view_explicit_none_native_prompt_arguments"
        )
        if (
            trace.get("conditioning_mode") != expected_conditioning_mode
            or trace.get("flux_dual_view_plan_preflight") != expected_preflight
        ):
            raise ValueError("Flux-v3 native trace does not prove paired prompt routing.")
        call = trace.get("flux_dual_view_native_call")
        negative = plan["negative"]
        expected_call_keys = {
            "prompt",
            "prompt_2",
            "negative_prompt",
            "negative_prompt_2",
            "guidance_scale",
            "true_cfg_scale",
        }
        native_options = job.get("native_negative_prompt_options")
        expected_guidance_scale = generation.get("guidance_scale")
        expected_true_cfg_scale = (
            native_options.get("true_cfg_scale")
            if isinstance(native_options, Mapping)
            else None
        )
        if (
            not isinstance(call, Mapping)
            or set(call) != expected_call_keys
            or call.get("prompt") != plan["positive"]["clip_prompt"]
            or call.get("prompt_2") != plan["positive"]["t5_prompt_2"]
        ):
            raise ValueError("Flux-v3 native trace raw prompt arguments drifted.")
        for key, expected in (
            ("guidance_scale", expected_guidance_scale),
            ("true_cfg_scale", expected_true_cfg_scale),
        ):
            actual = call.get(key)
            if (
                type(expected) is not float
                or not math.isfinite(expected)
                or type(actual) is not float
                or not math.isfinite(actual)
                or actual != expected
            ):
                raise ValueError(
                    "Flux-v3 native trace effective guidance argument drifted: "
                    f"{key}."
                )
        if negative_mode == NEGATIVE_MODE_PAIRED_REGISTERED:
            if (
                negative is None
                or call.get("negative_prompt") != negative["clip_negative_prompt"]
                or call.get("negative_prompt_2") != negative["t5_negative_prompt_2"]
            ):
                raise ValueError("Flux-v3 paired native trace negative views drifted.")
        elif (
            negative_mode != NEGATIVE_MODE_EXPLICIT_NONE_CONTROL
            or negative is not None
            or call.get("negative_prompt") is not None
            or call.get("negative_prompt_2") is not None
        ):
            raise ValueError("Flux-v3 explicit-none trace did not pass two exact null arguments.")
    return {
        "status": "passed",
        "plan_sha256": canonical_sha256(plan),
        "runner_config_checked": runner_config is not None,
        "run_timing_checked": run_timing is not None,
        "sample_report_checked": sample_report is not None,
        "conditioning_cache_checked": conditioning_cache is not None,
        "native_trace_checked": native_trace is not None,
    }
