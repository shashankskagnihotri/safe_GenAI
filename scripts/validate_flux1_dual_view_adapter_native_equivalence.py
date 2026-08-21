#!/usr/bin/env python3
"""Three-prompt real-H100 FLUX.1 dual-view/native bitwise equivalence gate.

The gate is intentionally impossible to execute without a separately sealed,
read-only preregistration JSON. That record freezes this diagnostic, its Slurm
launcher, every local implementation/config input, and the complete execution
contract before any GPU result exists.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import stat
import sys
import tempfile
import time
import traceback
from copy import deepcopy
from datetime import datetime, timezone
from importlib.metadata import distribution
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml

from hierasafe_flow.adapters.flux_dual_view_adapter import (
    FLUX_DUAL_VIEW_CONFIG_KEY,
    FluxDualViewAdapter,
    validate_flux_dual_view_conditioning,
)
from hierasafe_flow.adapters.registry import create_adapter
from hierasafe_flow.benchmarks.finer_detailing_correction import (
    BENCHMARK_NAME as FINER_BENCHMARK_NAME,
    _runner_config as _production_runner_config,
)
from hierasafe_flow.evaluation.flux1_dual_view_equivalence_admission_v3 import (
    ADAPTER_KEY,
    COMPARISON_COUNT as EXPECTED_COMPARISON_COUNT,
    COMPARISONS_PER_PROMPT as EXPECTED_COMPARISONS_PER_PROMPT,
    CONFIGURATION_SOURCE_SHA256,
    DIFFUSERS_COMMIT,
    DIFFUSERS_VERSION,
    EXPECTED_CONDA_PREFIX,
    EXPECTED_DIFFUSERS_CONFIGURATION_SOURCE,
    EXPECTED_DIFFUSERS_PIPELINE_SOURCE,
    EXPECTED_DIFFUSERS_SCHEDULER_SOURCE,
    EXPECTED_MODEL_SNAPSHOT_PATH,
    GATE,
    GUIDANCE_SCALE,
    IMAGE_LAYOUTS,
    MEDIA_COUNT as EXPECTED_PNG_COUNT,
    MODEL_ID,
    MODEL_INDEX_SHA256,
    MODEL_REVISION,
    NUM_INFERENCE_STEPS,
    PIPELINE_CLASS,
    PIPELINE_SOURCE_SHA256,
    PROMPT_IDS,
    PROTOCOL_ID,
    REQUIRED_PREREGISTRATION_FILE_PATHS as REQUIRED_PROTOCOL_FILE_PATHS,
    SCHEDULER_SOURCE_SHA256,
    SCHEMA_VERSION,
    SEED,
    SNAPSHOT_SCHEDULER_CONFIG_BLOB_NAME,
    SNAPSHOT_SCHEDULER_CONFIG_PARSED_SHA256,
    SNAPSHOT_SCHEDULER_CONFIG_RELATIVE,
    SNAPSHOT_SCHEDULER_CONFIG_SHA256,
    SNAPSHOT_SCHEDULER_CONFIG_SIZE_BYTES,
    TRUE_CFG_SCALE,
    expected_execution_contract as _expected_execution_contract,
    expected_attempt_lineage,
    independent_t5_sentinel_comparison_id,
    load_preregistration,
    normalize_and_validate_scheduler_config,
    parse_and_validate_snapshot_scheduler_config,
    parse_preregistration_source_attempt,
    parse_result_attempts,
    read_authenticated_hf_snapshot_file,
    validate_prelaunch_attempt,
)
from hierasafe_flow.evaluation.flux1_dual_view_jobs_v3 import (
    MODE_PREVIEW as FLUX1_V3_MODE_PREVIEW,
    RUNNER_PREFLIGHT_PROJECTED_JOB_HASH_CONTRACT,
    RUNNER_PREFLIGHT_RECEIPT_SCHEMA_VERSION,
    RUNNER_PREFLIGHT_ROUTE,
    project_flux1_job_v3,
    runner_preflight_projected_job_sha256_v3,
    validate_flux1_job_v3,
)
from hierasafe_flow.generation.runner import (
    GenerationRunner,
    _bind_flux_dual_view_conditioning,
)
from hierasafe_flow.utils.seed import make_generator, seed_everything


CONCEPT_HIERARCHIES = {
    "01_sad_young_girl": REQUIRED_PROTOCOL_FILE_PATHS["prompt1_concept_hierarchy"],
    "02_angry_old_man": REQUIRED_PROTOCOL_FILE_PATHS["prompt2_concept_hierarchy"],
    "03_empty_outdoor_mall": REQUIRED_PROTOCOL_FILE_PATHS["prompt3_concept_hierarchy"],
}
EXPECTED_LATENT_SHAPE = [1, 3952, 64]
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return _sha256_bytes(encoded)


def _read_authenticated_file_below_root(
    path: Path,
    trusted_root: Path,
    label: str,
    *,
    require_read_only: bool,
) -> tuple[Path, bytes, str]:
    """Read one stable regular inode without accepting any symlink component."""

    try:
        root = trusted_root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise ValueError(f"Trusted root for {label} is unavailable: {trusted_root}.") from exc
    expanded = path.expanduser()
    if ".." in expanded.parts:
        raise ValueError(f"{label} path must not contain parent traversal.")
    if not expanded.is_absolute():
        expanded = Path.cwd() / expanded
    candidate = Path(os.path.abspath(os.fspath(expanded)))
    try:
        relative = candidate.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be below its trusted root: {root}.") from exc
    if not relative.parts:
        raise ValueError(f"{label} must name a file below its trusted root.")

    component_snapshots: list[tuple[Path, int, int, int]] = []
    current = root
    for index, component in enumerate(relative.parts):
        current = current / component
        try:
            observed = os.lstat(current)
        except OSError as exc:
            raise ValueError(f"Cannot authenticate {label} path component: {current}.") from exc
        if stat.S_ISLNK(observed.st_mode):
            raise ValueError(f"{label} path contains a symlink component: {current}.")
        is_leaf = index == len(relative.parts) - 1
        if is_leaf and not stat.S_ISREG(observed.st_mode):
            raise ValueError(f"{label} must be a regular file: {current}.")
        if not is_leaf and not stat.S_ISDIR(observed.st_mode):
            raise ValueError(f"{label} has a non-directory parent component: {current}.")
        component_snapshots.append((current, observed.st_dev, observed.st_ino, observed.st_mode))
    leaf_before = os.lstat(candidate)
    if require_read_only and leaf_before.st_mode & 0o222:
        raise ValueError(f"{label} must be read-only: {candidate}.")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(candidate, flags)
    except OSError as exc:
        raise ValueError(f"Cannot open authenticated {label}: {candidate}.") from exc
    chunks: list[bytes] = []
    try:
        opened_before = os.fstat(descriptor)
        if not stat.S_ISREG(opened_before.st_mode) or (
            opened_before.st_dev,
            opened_before.st_ino,
        ) != (leaf_before.st_dev, leaf_before.st_ino):
            raise ValueError(f"{label} inode changed before authenticated read.")
        while True:
            chunk = os.read(descriptor, 16 * 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        opened_after = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    raw = b"".join(chunks)
    stable_fields = ("st_dev", "st_ino", "st_mode", "st_size", "st_mtime_ns", "st_ctime_ns")
    if any(
        getattr(opened_before, field) != getattr(opened_after, field) for field in stable_fields
    ):
        raise ValueError(f"{label} inode changed during authenticated read.")
    try:
        leaf_after = os.lstat(candidate)
    except OSError as exc:
        raise ValueError(f"{label} disappeared after authenticated read.") from exc
    if any(getattr(opened_after, field) != getattr(leaf_after, field) for field in stable_fields):
        raise ValueError(f"{label} path changed during authenticated read.")
    if len(raw) != opened_after.st_size:
        raise ValueError(f"{label} byte count changed during authenticated read.")
    for component_path, expected_dev, expected_ino, expected_mode in component_snapshots:
        try:
            current_stat = os.lstat(component_path)
        except OSError as exc:
            raise ValueError(
                f"{label} path component disappeared during authenticated read: {component_path}."
            ) from exc
        if stat.S_ISLNK(current_stat.st_mode) or (
            current_stat.st_dev,
            current_stat.st_ino,
            current_stat.st_mode,
        ) != (expected_dev, expected_ino, expected_mode):
            raise ValueError(
                f"{label} path component changed during authenticated read: {component_path}."
            )
    resolved = candidate.resolve(strict=True)
    if resolved != candidate or root not in resolved.parents:
        raise ValueError(f"{label} did not retain its canonical trusted-root path.")
    return resolved, raw, _sha256_bytes(raw)


def _require_aware_timestamp(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be an ISO-8601 timestamp string.")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{label} must be a valid ISO-8601 timestamp.") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{label} must include an explicit UTC offset.")
    return value


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(
        path,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
    )
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _create_immutable_file_exclusive(path: Path, raw: bytes, label: str) -> None:
    """Create one 0444 artifact without overwriting any existing directory entry."""

    if os.path.lexists(path):
        raise FileExistsError(f"Refusing to overwrite existing {label}: {path}")
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_EXCL
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_CLOEXEC", 0)
    )
    descriptor = os.open(path, flags, 0o400)
    try:
        remaining = memoryview(raw)
        while remaining:
            written = os.write(descriptor, remaining)
            if written <= 0:
                raise OSError(f"Short write while publishing {label}: {path}")
            remaining = remaining[written:]
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)


def _publish_immutable_json_with_sidecar(
    path: Path,
    payload: dict[str, Any],
) -> Path:
    """Publish JSON first and its exact SHA sidecar as the commit-last marker."""

    sidecar = path.with_name(f"{path.name}.sha256")
    if os.path.lexists(path) or os.path.lexists(sidecar):
        raise FileExistsError(
            f"Refusing to overwrite an existing result or sidecar: {path}, {sidecar}"
        )
    raw = (json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")
    sidecar_raw = f"{_sha256_bytes(raw)}  {path.name}\n".encode("utf-8")
    _create_immutable_file_exclusive(path, raw, "equivalence JSON")
    _create_immutable_file_exclusive(sidecar, sidecar_raw, "equivalence JSON sidecar")
    authenticated_path, authenticated_raw, authenticated_sha = _read_authenticated_file_below_root(
        path,
        path.parent,
        "published equivalence JSON",
        require_read_only=True,
    )
    authenticated_sidecar, authenticated_sidecar_raw, _ = _read_authenticated_file_below_root(
        sidecar,
        path.parent,
        "published equivalence JSON sidecar",
        require_read_only=True,
    )
    if (
        authenticated_path != path
        or authenticated_sidecar != sidecar
        or authenticated_raw != raw
        or authenticated_sha != _sha256_bytes(raw)
        or authenticated_sidecar_raw != sidecar_raw
    ):
        raise RuntimeError("Immutable equivalence JSON publication did not reauthenticate.")
    return sidecar


def _seal_existing_output_file(
    path: Path,
    output_root: Path,
    label: str,
) -> tuple[Path, bytes, str]:
    """Durably seal one existing output at 0444, then reopen and authenticate it."""

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"Cannot open {label} for immutable sealing: {path}.") from exc
    try:
        observed = os.fstat(descriptor)
        if not stat.S_ISREG(observed.st_mode):
            raise ValueError(f"{label} must be a regular file before sealing: {path}.")
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o444)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _fsync_directory(path.parent)
    return _read_authenticated_file_below_root(
        path,
        output_root,
        label,
        require_read_only=True,
    )


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise ValueError(f"{label} must be a lowercase SHA-256 digest.")
    return value


def _validate_prompt_row(row: Mapping[str, Any], expected_prompt_id: str) -> dict[str, Any]:
    expected_keys = {
        "prompt_id",
        "clip_prompt",
        "t5_prompt_2",
        "clip_string_sha256",
        "clip_token_count",
        "clip_token_ids_sha256",
        "t5_string_sha256",
        "t5_token_count",
        "t5_token_ids_sha256",
        "t5_source_contract",
        "negative_clip_prompt",
        "negative_t5_prompt_2",
        "negative_clip_string_sha256",
        "negative_clip_token_count",
        "negative_clip_token_ids_sha256",
        "negative_t5_string_sha256",
        "negative_t5_token_count",
        "negative_t5_token_ids_sha256",
        "negative_t5_source_contract",
    }
    if set(row) != expected_keys:
        raise ValueError(f"Prompt {expected_prompt_id} has an unexpected field set.")
    if row.get("prompt_id") != expected_prompt_id:
        raise ValueError("Prompt-contract order or prompt ID drifted.")
    text_roles = {
        "clip_prompt": "clip_string_sha256",
        "t5_prompt_2": "t5_string_sha256",
        "negative_clip_prompt": "negative_clip_string_sha256",
        "negative_t5_prompt_2": "negative_t5_string_sha256",
    }
    for text_key, sha_key in text_roles.items():
        text = row.get(text_key)
        if not isinstance(text, str) or not text.strip():
            raise ValueError(f"Prompt {expected_prompt_id} lacks {text_key}.")
        declared = _require_sha256(row.get(sha_key), f"{expected_prompt_id}.{sha_key}")
        if _sha256_bytes(text.encode("utf-8")) != declared:
            raise ValueError(f"Prompt {expected_prompt_id} {sha_key} does not bind its text.")
    fingerprint_roles = (
        ("clip_token_count", "clip_token_ids_sha256", 77),
        ("t5_token_count", "t5_token_ids_sha256", 512),
        ("negative_clip_token_count", "negative_clip_token_ids_sha256", 77),
        ("negative_t5_token_count", "negative_t5_token_ids_sha256", 512),
    )
    for count_key, sha_key, maximum in fingerprint_roles:
        count = row.get(count_key)
        if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= maximum:
            raise ValueError(f"Prompt {expected_prompt_id} {count_key} is invalid.")
        _require_sha256(row.get(sha_key), f"{expected_prompt_id}.{sha_key}")
    return dict(row)


def load_prompt_contract(path: Path) -> dict[str, Any]:
    """Validate the registered source file without importing v3 campaign code."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Cannot read Flux dual-view prompt contract: {path}.") from exc
    expected_top = {
        "schema_version",
        "contract_id",
        "status",
        "preregistered_at",
        "model",
        "conditioning_contract",
        "tokenizer_file_sha256",
        "prompts",
        "static_prompt3_claim_boundary",
    }
    if not isinstance(payload, Mapping) or set(payload) != expected_top:
        raise ValueError("Flux dual-view prompt contract schema is invalid.")
    if (
        payload.get("schema_version") != 1
        or payload.get("contract_id") != "flux1_common_seed_source_prompts_v3"
        or payload.get("status") != "preregistered_before_generation"
    ):
        raise ValueError("Flux dual-view prompt contract identity/status drifted.")
    _require_aware_timestamp(payload.get("preregistered_at"), "preregistered_at")
    model = payload.get("model")
    if not isinstance(model, Mapping) or dict(model) != {
        "model_name": "flux1_dev",
        "model_id": MODEL_ID,
        "revision": MODEL_REVISION,
        "pipeline_class": PIPELINE_CLASS,
    }:
        raise ValueError("Flux dual-view prompt contract model identity drifted.")
    conditioning = payload.get("conditioning_contract")
    required_conditioning = {
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
    if not isinstance(conditioning, Mapping) or dict(conditioning) != required_conditioning:
        raise ValueError("Flux dual-view conditioning contract drifted.")
    tokenizer_files = payload.get("tokenizer_file_sha256")
    if not isinstance(tokenizer_files, Mapping) or not tokenizer_files:
        raise ValueError("Flux dual-view prompt contract lacks tokenizer-file fingerprints.")
    for relative, digest in tokenizer_files.items():
        if not isinstance(relative, str) or not relative.startswith(("tokenizer/", "tokenizer_2/")):
            raise ValueError("Prompt contract contains an invalid tokenizer file path.")
        _require_sha256(digest, f"tokenizer_file_sha256.{relative}")
    rows = payload.get("prompts")
    if not isinstance(rows, list) or len(rows) != len(PROMPT_IDS):
        raise ValueError("Flux dual-view prompt contract must contain exactly three rows.")
    normalized = [
        _validate_prompt_row(row, prompt_id)
        for row, prompt_id in zip(rows, PROMPT_IDS, strict=True)
        if isinstance(row, Mapping)
    ]
    if len(normalized) != len(PROMPT_IDS):
        raise ValueError("Flux dual-view prompt rows must all be mappings.")
    return {
        "payload": dict(payload),
        "rows": normalized,
        "tokenizer_file_sha256": dict(tokenizer_files),
    }


def baseline_dual_view_plan(row: Mapping[str, Any]) -> dict[str, Any]:
    """Project one source row into the exact baseline-only adapter plan."""

    return validate_flux_dual_view_conditioning(
        {
            "schema_version": 1,
            "positive": {
                key: row[key]
                for key in (
                    "clip_prompt",
                    "t5_prompt_2",
                    "clip_string_sha256",
                    "clip_token_count",
                    "clip_token_ids_sha256",
                    "t5_string_sha256",
                    "t5_token_count",
                    "t5_token_ids_sha256",
                )
            },
            "negative": None,
        }
    )


def _validate_model_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError("Dual-view model config must be a mapping.")
    model = payload.get("model")
    if not isinstance(model, Mapping) or any(
        model.get(key) != value
        for key, value in {
            "adapter": ADAPTER_KEY,
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "diffusers_pipeline_class": PIPELINE_CLASS,
        }.items()
    ):
        raise ValueError("Dual-view model config identity drifted.")
    generation = payload.get("generation")
    if not isinstance(generation, Mapping) or any(
        generation.get(key) != value
        for key, value in {
            "task": "text_to_image",
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
        }.items()
    ):
        raise ValueError("Dual-view model config generation contract drifted.")
    return dict(payload)


def _runner_registry_construction_preflight(
    project_root: Path,
    prompt_contract: Mapping[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    """Exercise the production config-loader -> runner bridge -> registry path."""

    config_path = project_root / REQUIRED_PROTOCOL_FILE_PATHS["model_config"]
    _validate_model_config(config_path)
    rows: list[dict[str, Any]] = []
    for row in prompt_contract["rows"]:
        prompt_id = str(row["prompt_id"])
        prompt = str(row["clip_prompt"])
        plan = baseline_dual_view_plan(row)
        with tempfile.TemporaryDirectory(
            prefix=f"flux-dual-view-route-{prompt_id}-"
        ) as temporary_output:
            generation = {
                "task": "text_to_image",
                "num_inference_steps": NUM_INFERENCE_STEPS,
                "height": IMAGE_LAYOUTS[prompt_id]["height"],
                "width": IMAGE_LAYOUTS[prompt_id]["width"],
                "guidance_scale": GUIDANCE_SCALE,
                "num_outputs_per_prompt": 1,
            }
            synthetic_job = {
                "benchmark": FINER_BENCHMARK_NAME,
                "stage": "flux1_common_seed_source_ladder_v3",
                "variant": "01_baseline",
                "variant_spec": {"kind": "baseline"},
                "condition_id": f"{prompt_id}__flux1_dev__01_baseline__seed_00000000",
                "attempt": 1,
                "seed": SEED,
                "prompt_id": prompt_id,
                "prompt": prompt,
                "negative_prompt": str(row["negative_clip_prompt"]),
                "model_name": "flux1_dev",
                "model_revision": MODEL_REVISION,
                "base_config": str(
                    (project_root / REQUIRED_PROTOCOL_FILE_PATHS["base_config"]).resolve()
                ),
                "model_config": str(config_path.resolve()),
                "concept_tree": str((project_root / CONCEPT_HIERARCHIES[prompt_id]).resolve()),
                "generation": generation,
                "runtime": {
                    "device": str(device),
                    "dtype": "bfloat16" if dtype is torch.bfloat16 else "float32",
                },
                "logging": {"tensorboard": False},
                "output": {
                    "decode": True,
                    "save_latents": False,
                    "save_traces": False,
                    "image_format": "png",
                },
                "output_dir": temporary_output,
                "launch_manifest_sha256": "0" * 64,
            }
            projected_job = project_flux1_job_v3(
                synthetic_job,
                project_root=project_root,
                mode=FLUX1_V3_MODE_PREVIEW,
            )
            projection_validation = validate_flux1_job_v3(
                projected_job,
                project_root=project_root,
                mode=FLUX1_V3_MODE_PREVIEW,
            )
            if (
                projected_job["generation"].get(FLUX_DUAL_VIEW_CONFIG_KEY) != plan
                or projected_job.get("prompt") != prompt
                or projection_validation.get("mode") != FLUX1_V3_MODE_PREVIEW
            ):
                raise RuntimeError(
                    f"Shared campaign projector changed {prompt_id}'s registered prompt views."
                )
            runner_config = _production_runner_config(projected_job, project_root)
            if (
                runner_config.get("generation", {}).get("prompt") != prompt
                or runner_config.get("generation", {}).get(FLUX_DUAL_VIEW_CONFIG_KEY) != plan
            ):
                raise RuntimeError(
                    f"Production benchmark config builder changed {prompt_id}'s prompt/plan."
                )
            bound_model = _bind_flux_dual_view_conditioning(
                model_config=deepcopy(dict(runner_config["model"])),
                generation_config=deepcopy(dict(runner_config["generation"])),
            )
            adapter = create_adapter(
                bound_model,
                device=device,
                dtype=dtype,
            )
            exact_type = type(adapter) is FluxDualViewAdapter
            exact_plan = adapter.config.get(FLUX_DUAL_VIEW_CONFIG_KEY) == plan
            if not exact_type or not exact_plan:
                raise RuntimeError(
                    f"Runner/registry construction did not preserve {prompt_id}'s exact dual view."
                )
            adapter.validate_primary_prompts([prompt])
            provenance = adapter.conditioning_provenance()
            if (
                provenance["positive"]["clip"]["text"] != prompt
                or provenance["positive"]["t5"]["text"] != row["t5_prompt_2"]
                or provenance["negative"] is not None
            ):
                raise RuntimeError(
                    f"Runner/registry construction changed {prompt_id}'s prompt views."
                )
            runner = GenerationRunner(runner_config)
            try:
                runner_prompt = runner._collect_prompts(None)
                runner_exact = (
                    type(runner.adapter) is FluxDualViewAdapter
                    and runner.config["model"].get(FLUX_DUAL_VIEW_CONFIG_KEY) == plan
                    and runner_prompt == [prompt]
                    and runner.adapter.config.get(FLUX_DUAL_VIEW_CONFIG_KEY) == plan
                )
            finally:
                runner.tensorboard.close()
                for handler in runner.logger.handlers:
                    handler.close()
                runner.logger.handlers.clear()
        if not runner_exact:
            raise RuntimeError(
                f"GenerationRunner constructor did not preserve {prompt_id}'s exact route."
            )
        rows.append(
            {
                "prompt_id": prompt_id,
                "generation_prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                "dual_view_plan_sha256": _canonical_sha256(plan),
                "registered_generation_sha256": _canonical_sha256(generation),
                "projected_job_sha256": runner_preflight_projected_job_sha256_v3(
                    projected_job,
                    expected_output_dir=temporary_output,
                ),
                "projection_validation_sha256": _canonical_sha256(projection_validation),
                "projection_mode": projection_validation["mode"],
                "projection_status": projection_validation["status"],
                "production_runner_generation_sha256": _canonical_sha256(
                    runner_config["generation"]
                ),
                "bound_model_config_sha256": _canonical_sha256(bound_model),
                "adapter_class": type(adapter).__name__,
                "adapter_name": adapter.adapter_name,
                "model_id": adapter.model_id,
                "revision": adapter.config.get("revision"),
                "exact_plan_bound": exact_plan,
                "generation_runner_constructor_checked": True,
            }
        )
    if len(rows) != len(PROMPT_IDS):
        raise RuntimeError("Runner/registry construction preflight lacks three prompts.")
    return {
        "schema_version": RUNNER_PREFLIGHT_RECEIPT_SCHEMA_VERSION,
        "status": "passed",
        "route": RUNNER_PREFLIGHT_ROUTE,
        "projected_job_hash_contract": RUNNER_PREFLIGHT_PROJECTED_JOB_HASH_CONTRACT,
        "model_loading_performed": False,
        "config_source": {
            "path": str(config_path.resolve()),
            "sha256": _sha256_file(config_path),
        },
        "prompt_count": len(rows),
        "rows": rows,
    }


def _pinned_model_cache_dir() -> Path:
    """Derive the exact HF cache root encoded by the pinned snapshot path."""

    expected_repo_folder = f"models--{MODEL_ID.replace('/', '--')}"
    snapshot_parent = EXPECTED_MODEL_SNAPSHOT_PATH.parent
    repo_parent = snapshot_parent.parent
    if (
        not EXPECTED_MODEL_SNAPSHOT_PATH.is_absolute()
        or EXPECTED_MODEL_SNAPSHOT_PATH.name != MODEL_REVISION
        or snapshot_parent.name != "snapshots"
        or repo_parent.name != expected_repo_folder
    ):
        raise RuntimeError("Pinned FLUX.1-dev snapshot/cache layout drifted.")
    return repo_parent.parent


def _download_pinned_model_snapshot(snapshot_download: Any) -> Path:
    """Load only the exact lexical snapshot beneath the explicitly pinned cache."""

    cache_dir = _pinned_model_cache_dir()
    snapshot_raw = snapshot_download(
        repo_id=MODEL_ID,
        revision=MODEL_REVISION,
        cache_dir=os.fspath(cache_dir),
        local_files_only=True,
    )
    expected_raw = os.fspath(EXPECTED_MODEL_SNAPSHOT_PATH)
    if not isinstance(snapshot_raw, str) or snapshot_raw != expected_raw:
        raise RuntimeError("Pinned FLUX.1-dev snapshot path drifted.")
    return Path(snapshot_raw)


def _load_flux_pipeline_from_authenticated_preflight(
    pipeline_class: Any,
    *,
    environment_preflight: Mapping[str, Any],
    torch_dtype: torch.dtype,
) -> Any:
    """Load from the exact snapshot path already authenticated by preflight."""

    snapshot = environment_preflight.get("model_snapshot")
    expected_path = os.fspath(EXPECTED_MODEL_SNAPSHOT_PATH)
    if (
        not isinstance(snapshot, Mapping)
        or snapshot.get("model_id") != MODEL_ID
        or snapshot.get("revision") != MODEL_REVISION
        or not isinstance(snapshot.get("path"), str)
        or snapshot.get("path") != expected_path
    ):
        raise RuntimeError("Authenticated FLUX.1-dev snapshot binding drifted before load.")
    snapshot_path = snapshot["path"]
    return pipeline_class.from_pretrained(
        snapshot_path,
        torch_dtype=torch_dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )


def _environment_preflight(
    project_root: Path,
    preregistration: dict[str, Any],
    prompt_contract: dict[str, Any],
) -> dict[str, Any]:
    import diffusers
    import transformers
    from diffusers import configuration_utils as diffusers_configuration_utils
    from diffusers import FluxPipeline
    from diffusers.schedulers import scheduling_flow_match_euler_discrete
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Equivalence gate requires exactly one visible CUDA GPU.")
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" not in gpu_name.upper() or not torch.cuda.is_bf16_supported():
        raise RuntimeError(f"Equivalence gate requires one bfloat16 H100; observed {gpu_name!r}.")
    if os.environ.get("CONDA_PREFIX") != EXPECTED_CONDA_PREFIX:
        raise RuntimeError("Equivalence gate is outside the authenticated production environment.")
    if diffusers.__version__ != DIFFUSERS_VERSION:
        raise RuntimeError("Pinned Diffusers version drifted.")
    direct_url_raw = distribution("diffusers").read_text("direct_url.json")
    direct_url = json.loads(direct_url_raw) if direct_url_raw else {}
    installed_commit = direct_url.get("vcs_info", {}).get("commit_id")
    if installed_commit != DIFFUSERS_COMMIT:
        raise RuntimeError("Pinned Diffusers VCS commit drifted.")
    package_sources = {
        "diffusers_configuration_utils": (
            Path(inspect.getsourcefile(diffusers_configuration_utils) or "").resolve(),
            EXPECTED_DIFFUSERS_CONFIGURATION_SOURCE,
            CONFIGURATION_SOURCE_SHA256,
        ),
        "diffusers_flux_pipeline": (
            Path(inspect.getsourcefile(FluxPipeline) or "").resolve(),
            EXPECTED_DIFFUSERS_PIPELINE_SOURCE,
            PIPELINE_SOURCE_SHA256,
        ),
        "diffusers_flow_match_scheduler": (
            Path(inspect.getsourcefile(scheduling_flow_match_euler_discrete) or "").resolve(),
            EXPECTED_DIFFUSERS_SCHEDULER_SOURCE,
            SCHEDULER_SOURCE_SHA256,
        ),
    }
    source_receipts: dict[str, dict[str, str]] = {}
    for role, (path, expected_path, expected_sha256) in package_sources.items():
        actual = _sha256_file(path)
        if path != expected_path or actual != expected_sha256:
            raise RuntimeError(f"Pinned package source {role!r} drifted.")
        source_receipts[role] = {"path": str(path), "sha256": actual}

    snapshot = _download_pinned_model_snapshot(snapshot_download)
    model_index_raw = read_authenticated_hf_snapshot_file(snapshot, "model_index.json")
    if _sha256_bytes(model_index_raw) != MODEL_INDEX_SHA256:
        raise RuntimeError("Pinned FLUX.1-dev model_index.json digest drifted.")
    scheduler_config_raw = read_authenticated_hf_snapshot_file(
        snapshot,
        SNAPSHOT_SCHEDULER_CONFIG_RELATIVE,
        expected_blob_name=SNAPSHOT_SCHEDULER_CONFIG_BLOB_NAME,
    )
    scheduler_config_parsed = parse_and_validate_snapshot_scheduler_config(scheduler_config_raw)
    tokenizer_receipts: dict[str, dict[str, str]] = {}
    for relative, expected in prompt_contract["tokenizer_file_sha256"].items():
        path = snapshot / relative
        raw = read_authenticated_hf_snapshot_file(snapshot, relative)
        actual = _sha256_bytes(raw)
        if actual != expected:
            raise RuntimeError(f"Pinned tokenizer file drifted: {relative}.")
        tokenizer_receipts[relative] = {"path": str(path), "sha256": actual}
    return {
        "gpu": {
            "name": gpu_name,
            "count": torch.cuda.device_count(),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
            "bfloat16_supported": torch.cuda.is_bf16_supported(),
        },
        "environment": {
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "python": sys.executable,
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "diffusers_commit": installed_commit,
            "transformers": transformers.__version__,
        },
        "preregistration": preregistration,
        "package_sources": source_receipts,
        "model_snapshot": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "path": str(snapshot),
            "model_index_sha256": MODEL_INDEX_SHA256,
            "scheduler_config_file": {
                "path": str(snapshot / SNAPSHOT_SCHEDULER_CONFIG_RELATIVE),
                "sha256": SNAPSHOT_SCHEDULER_CONFIG_SHA256,
                "size_bytes": SNAPSHOT_SCHEDULER_CONFIG_SIZE_BYTES,
                "blob_name": SNAPSHOT_SCHEDULER_CONFIG_BLOB_NAME,
                "parsed_sha256": SNAPSHOT_SCHEDULER_CONFIG_PARSED_SHA256,
                "parsed_config": scheduler_config_parsed,
            },
            "tokenizer_files": tokenizer_receipts,
        },
        "slurm": _runtime_slurm_identity(),
    }


def _clone_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().cpu().clone()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().contiguous().cpu().view(torch.uint8).numpy().tobytes()


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous().cpu()
    finite = torch.isfinite(value.float())
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": _sha256_bytes(_tensor_bytes(value)),
        "finite": bool(finite.all().item()),
        "min": float(value.float().min().item()) if value.numel() else None,
        "max": float(value.float().max().item()) if value.numel() else None,
        "mean": float(value.float().mean().item()) if value.numel() else None,
    }


def _compare_tensors(
    comparison_id: str,
    native: torch.Tensor,
    adapter: torch.Tensor,
) -> dict[str, Any]:
    native_cpu = native.detach().contiguous().cpu()
    adapter_cpu = adapter.detach().contiguous().cpu()
    same_shape = native_cpu.shape == adapter_cpu.shape
    same_dtype = native_cpu.dtype == adapter_cpu.dtype
    exact = bool(same_shape and same_dtype and torch.equal(native_cpu, adapter_cpu))
    max_abs = None
    mean_abs = None
    if same_shape and native_cpu.numel():
        delta = (native_cpu.float() - adapter_cpu.float()).abs()
        max_abs = float(delta.max().item())
        mean_abs = float(delta.mean().item())
    return {
        "comparison_id": comparison_id,
        "passed": exact,
        "required_relation": "bitwise_identical",
        "same_shape": same_shape,
        "same_dtype": same_dtype,
        "max_abs_difference": max_abs,
        "mean_abs_difference": mean_abs,
        "native": _tensor_summary(native_cpu),
        "adapter": _tensor_summary(adapter_cpu),
    }


def _compare_tensors_not_equal(
    comparison_id: str,
    full: torch.Tensor,
    mirrored: torch.Tensor,
) -> dict[str, Any]:
    full_cpu = full.detach().contiguous().cpu()
    mirrored_cpu = mirrored.detach().contiguous().cpu()
    same_shape = full_cpu.shape == mirrored_cpu.shape
    same_dtype = full_cpu.dtype == mirrored_cpu.dtype
    bitwise_equal = bool(same_shape and same_dtype and torch.equal(full_cpu, mirrored_cpu))
    max_abs = None
    mean_abs = None
    if same_shape and full_cpu.numel():
        delta = (full_cpu.float() - mirrored_cpu.float()).abs()
        max_abs = float(delta.max().item())
        mean_abs = float(delta.mean().item())
    passed = bool(same_shape and same_dtype and not bitwise_equal and max_abs and max_abs > 0)
    return {
        "comparison_id": comparison_id,
        "passed": passed,
        "required_relation": "not_bitwise_identical",
        "same_shape": same_shape,
        "same_dtype": same_dtype,
        "bitwise_equal": bitwise_equal,
        "max_abs_difference": max_abs,
        "mean_abs_difference": mean_abs,
        "full_independent_t5": _tensor_summary(full_cpu),
        "mirrored_clip_text_as_t5": _tensor_summary(mirrored_cpu),
    }


def _tokenizer_id_record(tokenizer: Any, text: str) -> dict[str, Any]:
    encoded = tokenizer(
        text,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        add_special_tokens=True,
    )
    token_ids = (
        encoded.get("input_ids")
        if isinstance(encoded, Mapping)
        else getattr(encoded, "input_ids", None)
    )
    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.detach().cpu().tolist()
    if (
        isinstance(token_ids, (list, tuple))
        and len(token_ids) == 1
        and isinstance(token_ids[0], (list, tuple))
    ):
        token_ids = token_ids[0]
    if not isinstance(token_ids, (list, tuple)) or not all(
        isinstance(token, int) and not isinstance(token, bool) for token in token_ids
    ):
        raise RuntimeError("T5 sentinel tokenizer did not return one integer sequence.")
    normalized = [int(token) for token in token_ids]
    return {
        "text_utf8_sha256": _sha256_bytes(text.encode("utf-8")),
        "token_count": len(normalized),
        "token_ids_sha256": _canonical_sha256(normalized),
        "token_ids": normalized,
    }


def _extract_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    sample = getattr(output, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample
    raise TypeError(f"Cannot extract tensor from {type(output).__name__}.")


def _run_prompt_equivalence(
    *,
    pipe: Any,
    row: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    prompt_id = str(row["prompt_id"])
    layout = IMAGE_LAYOUTS[prompt_id]
    clip_prompt = str(row["clip_prompt"])
    t5_prompt = str(row["t5_prompt_2"])
    plan = baseline_dual_view_plan(row)
    adapter = FluxDualViewAdapter(
        model_id=MODEL_ID,
        device=device,
        dtype=dtype,
        config={
            "revision": MODEL_REVISION,
            "guidance_scale": GUIDANCE_SCALE,
            "flux_dual_view_conditioning": plan,
        },
    )
    adapter.pipeline = pipe
    adapter.loaded = True
    adapter.validate_primary_prompts([clip_prompt])
    conditioning_preflight = adapter.preflight_conditioning_plan()

    native_first_inputs: dict[str, torch.Tensor] = {}
    native_predictions: list[torch.Tensor] = []
    native_normalized_timesteps: list[torch.Tensor] = []
    native_post_latents: list[torch.Tensor] = []
    native_callback_timesteps: list[torch.Tensor] = []
    native_vae_outputs: list[torch.Tensor] = []

    def native_transformer_hook(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        required = {
            "hidden_states",
            "timestep",
            "guidance",
            "pooled_projections",
            "encoder_hidden_states",
            "txt_ids",
            "img_ids",
        }
        missing = sorted(required - set(kwargs))
        if missing:
            raise RuntimeError(f"Native transformer hook lacks required kwargs: {missing}.")
        if not native_first_inputs:
            native_first_inputs.update({key: _clone_cpu(kwargs[key]) for key in sorted(required)})
        native_normalized_timesteps.append(_clone_cpu(kwargs["timestep"]))
        native_predictions.append(_clone_cpu(_extract_tensor(output)))

    def native_vae_hook(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        native_vae_outputs.append(_clone_cpu(_extract_tensor(output)))

    def native_callback(
        _pipeline: Any,
        _step_index: int,
        timestep: torch.Tensor,
        callback_kwargs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        native_callback_timesteps.append(_clone_cpu(timestep))
        native_post_latents.append(_clone_cpu(callback_kwargs["latents"]))
        return callback_kwargs

    transformer_handle = pipe.transformer.register_forward_hook(
        native_transformer_hook,
        with_kwargs=True,
    )
    vae_handle = pipe.vae.decoder.register_forward_hook(native_vae_hook)
    native_started = time.perf_counter()
    try:
        seed_everything(SEED)
        with torch.inference_mode():
            native_output = pipe(
                prompt=clip_prompt,
                prompt_2=t5_prompt,
                negative_prompt=None,
                negative_prompt_2=None,
                true_cfg_scale=TRUE_CFG_SCALE,
                height=layout["height"],
                width=layout["width"],
                num_inference_steps=NUM_INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                num_images_per_prompt=1,
                generator=make_generator(SEED, device),
                output_type="pil",
                return_dict=True,
                callback_on_step_end=native_callback,
                callback_on_step_end_tensor_inputs=["latents"],
            )
        torch.cuda.synchronize()
    finally:
        transformer_handle.remove()
        vae_handle.remove()
    native_seconds = time.perf_counter() - native_started
    native_sigmas = _clone_cpu(pipe.scheduler.sigmas)
    if (
        len(native_predictions) != NUM_INFERENCE_STEPS
        or len(native_normalized_timesteps) != NUM_INFERENCE_STEPS
        or len(native_post_latents) != NUM_INFERENCE_STEPS
        or len(native_callback_timesteps) != NUM_INFERENCE_STEPS
        or len(native_vae_outputs) != 1
    ):
        raise RuntimeError(f"Native {prompt_id} instrumentation coverage is incomplete.")
    native_image = native_output.images[0]
    native_path = output_dir / f"{prompt_id}__native_pipeline.png"
    native_image.save(native_path)

    adapter_started = time.perf_counter()
    seed_everything(SEED)
    with torch.inference_mode():
        condition = adapter.prepare_prompt(clip_prompt)
        full_t5_tokens = _tokenizer_id_record(pipe.tokenizer_2, t5_prompt)
        mirrored_t5_tokens = _tokenizer_id_record(pipe.tokenizer_2, clip_prompt)
        if (
            full_t5_tokens["token_count"] != row["t5_token_count"]
            or full_t5_tokens["token_ids_sha256"] != row["t5_token_ids_sha256"]
            or full_t5_tokens["token_ids"] == mirrored_t5_tokens["token_ids"]
        ):
            raise RuntimeError(f"Independent T5 token sentinel failed for {prompt_id}.")
        mirrored_encoding = pipe.encode_prompt(
            prompt=clip_prompt,
            prompt_2=clip_prompt,
            device=device,
            num_images_per_prompt=1,
            max_sequence_length=512,
        )
        if not isinstance(mirrored_encoding, tuple) or not isinstance(
            mirrored_encoding[0], torch.Tensor
        ):
            raise RuntimeError("Pinned Flux encode_prompt output schema drifted.")
        full_vs_mirrored = _compare_tensors_not_equal(
            independent_t5_sentinel_comparison_id(prompt_id),
            condition.data["prompt_embeds"],
            mirrored_encoding[0],
        )
        if not full_vs_mirrored["passed"]:
            raise RuntimeError(
                f"Full and mirrored T5 embeddings are not distinguishable for {prompt_id}."
            )
        independent_t5_sentinel = {
            "schema_version": 1,
            "status": "passed",
            "full_t5_prompt_2": full_t5_tokens,
            "mirrored_clip_text_as_t5_prompt_2": mirrored_t5_tokens,
            "token_ids_differ": True,
            "full_vs_mirrored_prompt_embeds": full_vs_mirrored,
            "native_full_vs_adapter_full_comparison_id": (
                f"{prompt_id}.conditioning.prompt_embeds"
            ),
        }
        del mirrored_encoding
        adapter_generator = make_generator(SEED, device)
        latents, state = adapter.prepare_initial_latents(
            prompt=clip_prompt,
            batch_size=1,
            generator=adapter_generator,
            height=layout["height"],
            width=layout["width"],
        )
        if list(latents.shape) != EXPECTED_LATENT_SHAPE:
            raise RuntimeError(f"Packed latent shape drift for {prompt_id}: {list(latents.shape)}.")
        initial_adapter_latents = _clone_cpu(latents)
        state.extra["num_steps"] = NUM_INFERENCE_STEPS
        adapter_timesteps = adapter.set_timesteps(
            NUM_INFERENCE_STEPS,
            latents=latents,
            state=state,
        )
        adapter_sigmas = _clone_cpu(pipe.scheduler.sigmas)
        adapter_predictions: list[torch.Tensor] = []
        adapter_post_latents: list[torch.Tensor] = []
        for timestep in adapter_timesteps:
            prediction = adapter.predict_vector_field(latents, timestep, condition, state)
            adapter_predictions.append(_clone_cpu(prediction))
            step = adapter.scheduler_step(
                model_prediction=prediction,
                timestep=timestep,
                latents=latents,
                state=state,
                generator=adapter_generator,
            )
            latents, state = step.latents, step.state
            adapter_post_latents.append(_clone_cpu(latents))
        adapter_vae_outputs: list[torch.Tensor] = []

        def adapter_vae_hook(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            output: Any,
        ) -> None:
            adapter_vae_outputs.append(_clone_cpu(_extract_tensor(output)))

        adapter_vae_handle = pipe.vae.decoder.register_forward_hook(adapter_vae_hook)
        try:
            adapter_media = adapter.decode_latents(latents, state)
        finally:
            adapter_vae_handle.remove()
    torch.cuda.synchronize()
    adapter_seconds = time.perf_counter() - adapter_started
    if len(adapter_vae_outputs) != 1:
        raise RuntimeError(f"Adapter {prompt_id} VAE instrumentation coverage is incomplete.")
    adapter_image = adapter_media[0]
    adapter_path = output_dir / f"{prompt_id}__flux_dual_view_adapter.png"
    adapter_image.save(adapter_path)

    comparisons = [
        _compare_tensors(
            f"{prompt_id}.conditioning.prompt_embeds",
            native_first_inputs["encoder_hidden_states"],
            condition.data["prompt_embeds"],
        ),
        _compare_tensors(
            f"{prompt_id}.conditioning.pooled_prompt_embeds",
            native_first_inputs["pooled_projections"],
            condition.data["pooled_prompt_embeds"],
        ),
        _compare_tensors(
            f"{prompt_id}.conditioning.text_ids",
            native_first_inputs["txt_ids"],
            condition.data["text_ids"],
        ),
        _compare_tensors(
            f"{prompt_id}.latents.initial_packed",
            native_first_inputs["hidden_states"],
            initial_adapter_latents,
        ),
        _compare_tensors(
            f"{prompt_id}.latents.image_ids",
            native_first_inputs["img_ids"],
            state.extra["latent_image_ids"],
        ),
        _compare_tensors(f"{prompt_id}.schedule.sigmas", native_sigmas, adapter_sigmas),
        _compare_tensors(
            f"{prompt_id}.schedule.timesteps",
            torch.stack(native_callback_timesteps),
            torch.stack([_clone_cpu(value) for value in adapter_timesteps]),
        ),
        _compare_tensors(
            f"{prompt_id}.transformer.embedded_guidance",
            native_first_inputs["guidance"],
            torch.full((1,), GUIDANCE_SCALE, dtype=torch.float32),
        ),
    ]
    for step_index in range(NUM_INFERENCE_STEPS):
        comparisons.extend(
            [
                _compare_tensors(
                    f"{prompt_id}.step.{step_index:02d}.normalized_timestep",
                    native_normalized_timesteps[step_index],
                    torch.as_tensor(adapter_timesteps[step_index])
                    .reshape(1)
                    .to(dtype=torch.bfloat16)
                    / 1000,
                ),
                _compare_tensors(
                    f"{prompt_id}.step.{step_index:02d}.prediction",
                    native_predictions[step_index],
                    adapter_predictions[step_index],
                ),
                _compare_tensors(
                    f"{prompt_id}.step.{step_index:02d}.post_scheduler_latents",
                    native_post_latents[step_index],
                    adapter_post_latents[step_index],
                ),
            ]
        )
    comparisons.append(
        _compare_tensors(
            f"{prompt_id}.decode.vae_output",
            native_vae_outputs[0],
            adapter_vae_outputs[0],
        )
    )
    native_rgb = torch.from_numpy(np.asarray(native_image).copy())
    adapter_rgb = torch.from_numpy(np.asarray(adapter_image).copy())
    comparisons.append(_compare_tensors(f"{prompt_id}.decode.rgb_uint8", native_rgb, adapter_rgb))
    if len(comparisons) != EXPECTED_COMPARISONS_PER_PROMPT:
        raise RuntimeError(f"Comparison coverage drifted for {prompt_id}.")
    failures = [row["comparison_id"] for row in comparisons if not row["passed"]]
    _, _, native_png_sha = _seal_existing_output_file(
        native_path,
        output_dir,
        f"{prompt_id} native-pipeline PNG",
    )
    _, _, adapter_png_sha = _seal_existing_output_file(
        adapter_path,
        output_dir,
        f"{prompt_id} dual-view-adapter PNG",
    )
    if not failures and native_png_sha != adapter_png_sha:
        raise RuntimeError(
            f"Pixel-identical {prompt_id} outputs serialized to different PNG bytes."
        )
    return {
        "prompt_id": prompt_id,
        "status": "passed" if not failures else "failed",
        "prompt_views": {
            "clip_prompt": clip_prompt,
            "clip_prompt_sha256": _sha256_bytes(clip_prompt.encode("utf-8")),
            "t5_prompt_2": t5_prompt,
            "t5_prompt_2_sha256": _sha256_bytes(t5_prompt.encode("utf-8")),
            "negative": None,
        },
        "layout": layout,
        "conditioning_preflight": conditioning_preflight,
        "conditioning_provenance": adapter.conditioning_provenance(),
        "independent_t5_sentinel": independent_t5_sentinel,
        "timing_seconds": {"native_pipeline": native_seconds, "dual_view_adapter": adapter_seconds},
        "comparisons": comparisons,
        "failed_comparison_ids": failures,
        "media": {
            "native_pipeline": {"path": str(native_path), "sha256": native_png_sha},
            "flux_dual_view_adapter": {
                "path": str(adapter_path),
                "sha256": adapter_png_sha,
            },
        },
    }


def _preregistration_result_binding(
    preregistration: Mapping[str, Any],
) -> dict[str, str]:
    return {
        key: str(preregistration[key])
        for key in (
            "path",
            "raw_sha256",
            "document_sha256",
            "sidecar_path",
            "sidecar_sha256",
        )
    }


def _run_equivalence(
    output_dir: Path,
    project_root: Path,
    preregistration: Mapping[str, Any],
    *,
    attempt_lineage: Mapping[str, Any],
) -> dict[str, Any]:
    from diffusers import FluxPipeline

    started_at = _utc_now()
    started = time.perf_counter()
    prompt_path = project_root / REQUIRED_PROTOCOL_FILE_PATHS["prompt_contract"]
    prompt_contract = load_prompt_contract(prompt_path)
    _validate_model_config(project_root / REQUIRED_PROTOCOL_FILE_PATHS["model_config"])
    preflight = _environment_preflight(project_root, preregistration, prompt_contract)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    dtype = torch.bfloat16
    runner_registry_preflight = _runner_registry_construction_preflight(
        project_root,
        prompt_contract,
        device=device,
        dtype=dtype,
    )
    load_started = time.perf_counter()
    pipe = _load_flux_pipeline_from_authenticated_preflight(
        FluxPipeline,
        environment_preflight=preflight,
        torch_dtype=dtype,
    )
    pipe.to(device)
    load_seconds = time.perf_counter() - load_started
    scheduler_config = normalize_and_validate_scheduler_config(dict(pipe.scheduler.config))
    prompt_results = [
        _run_prompt_equivalence(
            pipe=pipe,
            row=row,
            output_dir=output_dir,
            device=device,
            dtype=dtype,
        )
        for row in prompt_contract["rows"]
    ]
    failures = [
        comparison_id
        for result in prompt_results
        for comparison_id in result["failed_comparison_ids"]
    ]
    pngs = sorted(output_dir.glob("*.png"))
    if len(pngs) != EXPECTED_PNG_COUNT:
        raise RuntimeError(f"Expected exactly six PNGs, found {len(pngs)}.")
    comparison_count = sum(len(result["comparisons"]) for result in prompt_results)
    passed_comparison_count = sum(
        int(comparison["passed"])
        for result in prompt_results
        for comparison in result["comparisons"]
    )
    if comparison_count != EXPECTED_COMPARISON_COUNT:
        raise RuntimeError(
            f"Expected {EXPECTED_COMPARISON_COUNT} comparisons, found {comparison_count}."
        )
    sentinel_pass_count = sum(
        result["independent_t5_sentinel"]["status"] == "passed" for result in prompt_results
    )
    if sentinel_pass_count != len(PROMPT_IDS):
        raise RuntimeError("Independent T5 sentinel coverage is incomplete.")
    media_bindings: list[dict[str, Any]] = []
    for result in prompt_results:
        for route in ("native_pipeline", "flux_dual_view_adapter"):
            media = result["media"][route]
            path = Path(media["path"])
            authenticated_path, media_raw, media_sha256 = _read_authenticated_file_below_root(
                path,
                output_dir,
                f"{result['prompt_id']} {route} PNG",
                require_read_only=True,
            )
            if media_sha256 != media["sha256"]:
                raise RuntimeError(f"Media binding drifted before result publication: {path}.")
            media_bindings.append(
                {
                    "prompt_id": result["prompt_id"],
                    "route": route,
                    "path": str(authenticated_path),
                    "sha256": media["sha256"],
                    "size_bytes": len(media_raw),
                    "width": result["layout"]["width"],
                    "height": result["layout"]["height"],
                }
            )
    preregistration_binding = _preregistration_result_binding(preregistration)
    acceptance_summary = {
        "schema_version": 1,
        "status": "passed" if not failures else "failed",
        "required_comparison_count": EXPECTED_COMPARISON_COUNT,
        "observed_comparison_count": comparison_count,
        "passed_comparison_count": passed_comparison_count,
        "failed_comparison_count": len(failures),
        "runner_registry_construction_preflight_status": runner_registry_preflight["status"],
        "required_independent_t5_sentinel_count": len(PROMPT_IDS),
        "observed_independent_t5_sentinel_count": len(prompt_results),
        "passed_independent_t5_sentinel_count": sentinel_pass_count,
        "required_media_count": EXPECTED_PNG_COUNT,
        "observed_media_count": len(media_bindings),
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not failures else "failed",
        "gate": GATE,
        "protocol_id": PROTOCOL_ID,
        "attempt_lineage": dict(attempt_lineage),
        "started_at": started_at,
        "ended_at": _utc_now(),
        "environment_preflight": preflight,
        "preregistration_binding": preregistration_binding,
        "runner_registry_construction_preflight": runner_registry_preflight,
        "execution_contract": _expected_execution_contract(),
        "prompt_contract": {
            "path": str(prompt_path.resolve()),
            "sha256": _sha256_file(prompt_path),
        },
        "scheduler_config": scheduler_config,
        "scheduler_config_sha256": _canonical_sha256(scheduler_config),
        "timing_seconds": {
            "model_load": load_seconds,
            "total": time.perf_counter() - started,
        },
        "prompt_results": prompt_results,
        "comparison_count": comparison_count,
        "passed_comparison_count": passed_comparison_count,
        "failed_comparison_ids": failures,
        "acceptance_summary": acceptance_summary,
        "media_count": len(media_bindings),
        "media_bindings": media_bindings,
        "media_paths": [str(path.resolve()) for path in pngs],
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New, nonexistent directory for authenticated equivalence evidence.",
    )
    parser.add_argument(
        "--preregistration",
        type=Path,
        required=True,
        help="Read-only external preregistration JSON frozen before sbatch.",
    )
    return parser


def _attempt_lineage_from_paths(
    *,
    preregistration_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_attempt = parse_preregistration_source_attempt(preregistration_path)
    result_source_attempt, execution_attempt = parse_result_attempts(output_dir)
    if result_source_attempt != source_attempt:
        raise ValueError("Equivalence output source attempt does not match its preregistration.")
    return expected_attempt_lineage(
        source_attempt=source_attempt,
        execution_attempt=execution_attempt,
        preregistration_path=preregistration_path,
        output_directory=output_dir,
    )


def _runtime_slurm_identity() -> dict[str, str | None]:
    return {
        "job_id": os.environ.get("SLURM_JOB_ID"),
        "job_name": os.environ.get("SLURM_JOB_NAME"),
        "node_list": os.environ.get("SLURM_JOB_NODELIST"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    preregistration_path = args.preregistration.expanduser()
    if not preregistration_path.is_absolute():
        preregistration_path = project_root / preregistration_path
    preregistration_path = Path(os.path.abspath(os.fspath(preregistration_path)))
    output_dir = args.output_dir.expanduser()
    if not output_dir.is_absolute():
        output_dir = project_root / output_dir
    output_dir = Path(os.path.abspath(os.fspath(output_dir)))
    audit_root = (project_root / "debugging/audits/finer_detailing_20260720").resolve()
    if output_dir.parent != audit_root or preregistration_path.parent != audit_root:
        raise ValueError(
            "Equivalence preregistration and output must be direct children of the fixed audit root."
        )
    attempt_lineage = _attempt_lineage_from_paths(
        preregistration_path=preregistration_path,
        output_dir=output_dir,
    )
    exported_preregistration = os.environ.get("FLUX1_DUAL_VIEW_EQUIVALENCE_PREREGISTRATION")
    exported_output = os.environ.get("FLUX1_DUAL_VIEW_EQUIVALENCE_OUTPUT_DIR")
    exported_preregistration_sha256 = os.environ.get(
        "FLUX1_DUAL_VIEW_EQUIVALENCE_PREREGISTRATION_SHA256"
    )
    if (
        exported_preregistration != str(preregistration_path)
        or exported_output != str(output_dir)
        or _SHA256_RE.fullmatch(str(exported_preregistration_sha256 or "")) is None
    ):
        raise ValueError(
            "Equivalence paths and preregistration digest must match the explicit launcher exports."
        )
    preregistration = load_preregistration(preregistration_path, project_root)
    if (
        preregistration["source_attempt"] != attempt_lineage["source_attempt"]
        or preregistration["raw_sha256"] != exported_preregistration_sha256
    ):
        raise RuntimeError(
            "Authenticated preregistration differs from the launcher attempt/digest binding."
        )
    preregistration_binding = _preregistration_result_binding(preregistration)
    run_started_at = _utc_now()
    prelaunch = validate_prelaunch_attempt(
        preregistration_path=preregistration_path,
        output_directory=output_dir,
        project_root=project_root,
    )
    if (
        prelaunch.get("source_attempt") != attempt_lineage["source_attempt"]
        or prelaunch.get("execution_attempt") != attempt_lineage["execution_attempt"]
        or prelaunch.get("output_directory") != str(output_dir)
        or prelaunch.get("preregistration")
        != {
            "path": str(preregistration_path),
            "raw_sha256": exported_preregistration_sha256,
            "sidecar_path": preregistration["sidecar_path"],
            "sidecar_sha256": preregistration["sidecar_sha256"],
        }
    ):
        raise RuntimeError(
            "Prelaunch authorization differs from the exported attempt/path/digest binding."
        )
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "equivalence_result.json"
    try:
        result = _run_equivalence(
            output_dir,
            project_root,
            preregistration,
            attempt_lineage=attempt_lineage,
        )
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "gate": GATE,
            "protocol_id": PROTOCOL_ID,
            "attempt_lineage": attempt_lineage,
            "started_at": run_started_at,
            "runtime_slurm": _runtime_slurm_identity(),
            "preregistration_binding": preregistration_binding,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "ended_at": _utc_now(),
        }
        _publish_immutable_json_with_sidecar(result_path, failure)
        raise
    result_sidecar = _publish_immutable_json_with_sidecar(result_path, result)
    print(
        json.dumps(
            {
                "result": str(result_path),
                "result_sidecar": str(result_sidecar),
                "status": result["status"],
            },
            sort_keys=True,
        )
    )
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
