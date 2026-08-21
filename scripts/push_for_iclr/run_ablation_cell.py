#!/usr/bin/env python3
"""Execute one immutable Stage-2 ablation cell."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import yaml
from PIL import Image


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hierasafe_flow.adapters.registry import create_adapter  # noqa: E402
from hierasafe_flow.campaigns.push_for_iclr.ablation_controller import (  # noqa: E402
    AblationContractError,
    AblationController,
    AblationMode,
    FrozenDirectionState,
    apply_relative_direction,
    build_unified_time_map,
    compute_no_neutral_direction,
    conceptsteer_direction,
    intervention_energy,
    mask_config_from_mapping,
    normalize_direction_rms,
    tensor_fingerprint,
    tensor_rms,
    tensor_stats,
    weighted_prediction_mean,
)
from hierasafe_flow.campaigns.push_for_iclr.ablation_manifests import (  # noqa: E402
    AblationManifestError,
    canonical_json,
    sha256_file,
    validate_sealed_job_manifest,
)


class CellExecutionError(RuntimeError):
    pass


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise CellExecutionError(message)


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _scheduler_sigma(adapter: Any, step_index: int) -> tuple[float | None, str]:
    for attribute in ("pipeline", "_pipeline", "pipe", "_pipe"):
        pipeline = getattr(adapter, attribute, None)
        scheduler = getattr(pipeline, "scheduler", None)
        sigmas = getattr(scheduler, "sigmas", None)
        if sigmas is None or step_index >= len(sigmas):
            continue
        value = sigmas[step_index]
        if torch.is_tensor(value):
            if value.numel() != 1:
                continue
            value = value.detach().float().cpu().item()
        result = float(value)
        if math.isfinite(result):
            return result, f"{attribute}.scheduler.sigmas"
    return None, "adapter_scheduler_sigma_not_exposed"


def _atomic_bytes(path: Path, payload: bytes) -> None:
    _require(not path.exists(), f"Refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    _require(not partial.exists(), f"Stale partial artifact: {partial}")
    try:
        with partial.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    _atomic_bytes(path, (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8"))


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    _require(isinstance(value, dict), f"Expected YAML mapping: {path}")
    return value


def _torch_dtype(name: str) -> torch.dtype:
    choices = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    _require(name in choices, f"Unsupported dtype: {name}")
    return choices[name]


def _tensor_to_pil(value: torch.Tensor) -> Image.Image:
    tensor = value.detach().float().cpu()
    if tensor.ndim == 4:
        _require(tensor.shape[0] == 1, f"Expected one decoded image, got {tensor.shape[0]}")
        tensor = tensor[0]
    if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
        tensor = tensor.permute(1, 2, 0)
    _require(tensor.ndim in (2, 3), f"Unsupported decoded tensor shape: {tuple(tensor.shape)}")
    if float(tensor.min()) < 0.0:
        tensor = tensor.add(1.0).div(2.0)
    array = tensor.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8).numpy()
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    return Image.fromarray(array)


def _decoded_to_pil(value: Any) -> Image.Image:
    if isinstance(value, Image.Image):
        return value
    if hasattr(value, "images"):
        value = value.images
    if isinstance(value, Mapping) and "images" in value:
        value = value["images"]
    if isinstance(value, (list, tuple)):
        _require(len(value) == 1, f"Expected one decoded image, got {len(value)}")
        return _decoded_to_pil(value[0])
    if torch.is_tensor(value):
        return _tensor_to_pil(value)
    if isinstance(value, np.ndarray):
        array = value
        if array.ndim == 4:
            _require(array.shape[0] == 1, f"Expected one image, got {array.shape[0]}")
            array = array[0]
        if np.issubdtype(array.dtype, np.floating):
            if float(array.min()) < 0.0:
                array = (array + 1.0) / 2.0
            array = np.clip(array, 0.0, 1.0) * 255.0
        return Image.fromarray(array.astype(np.uint8))
    raise CellExecutionError(f"Unsupported decoded image type: {type(value).__name__}")


def _prepare_probe_group(adapter: Any, state: Any, probes: Sequence[Mapping[str, Any]], role: str):
    conditions = []
    weights = []
    for probe in probes:
        conditions.append(
            adapter.prepare_prompt_for_state(
                probe["text"],
                state,
                prompt_view=f"global_ontology_{role}",
                call_role=f"{role}_probe:{probe['id']}",
            )
        )
        weights.append(float(probe["weight"]))
    return conditions, weights


def _compose_prompt(original: str, mode: AblationMode, ontology: Mapping[str, Any]):
    if mode is AblationMode.GENERIC_SUFFIX:
        suffix = ontology["generic_positive_suffix"]
        source = "generic_positive_suffix"
    elif mode is AblationMode.CONFIG_SUFFIX:
        suffix = ontology["config_positive_suffix"]
        source = "config_positive_suffix"
    else:
        return original, None, None
    _require(isinstance(suffix, str) and bool(suffix), f"Empty {source}")
    return original + "\n\n" + suffix, suffix, source


def _save_image_atomic(path: Path, image: Image.Image) -> None:
    _require(not path.exists(), f"Refusing to overwrite image: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f".{path.name}.partial.{os.getpid()}")
    _require(not partial.exists(), f"Stale partial image: {partial}")
    try:
        image.save(partial, format="PNG")
        with partial.open("rb") as handle:
            os.fsync(handle.fileno())
        os.replace(partial, path)
    except BaseException:
        if partial.exists():
            partial.unlink()
        raise


def _failure_path(output_dir: Path) -> Path:
    primary = output_dir / "_FAILURE.json"
    if not primary.exists():
        return primary
    token = os.environ.get("SLURM_JOB_ID", str(os.getpid()))
    return output_dir / f"_FAILURE_{token}.json"


def execute_cell(row: Mapping[str, Any], manifest_path: Path, manifest_file_sha256: str) -> Path:
    output_root = REPOSITORY_ROOT / "outputs/PUSH_FOR_ICLR"
    output_dir = (output_root / row["expected_output_relative_path"]).resolve()
    _require(output_dir.is_relative_to(output_root.resolve()), "Output escaped campaign root")
    if (output_dir / "_SUCCESS").exists():
        print(f"ALREADY_COMPLETE {output_dir}")
        return output_dir
    _require(not (output_dir / "_FAILURE.json").exists(), "Cell has a preserved failure; use retry manifest")
    output_dir.mkdir(parents=True, exist_ok=True)

    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    _require(current_commit == row["code_commit"], "Working-tree HEAD does not match job commit")
    for relative, expected in row["runtime_dependency_sha256"].items():
        _require(sha256_file(REPOSITORY_ROOT / relative) == expected, f"Dependency changed: {relative}")
    _require(sha256_file(manifest_path) == manifest_file_sha256, "Physical job manifest changed")
    model_config_path = REPOSITORY_ROOT / row["model_config_path"]
    ontology_path = REPOSITORY_ROOT / row["ontology_path"]
    _require(sha256_file(model_config_path) == row["model_config_sha256"], "Model config changed")
    _require(sha256_file(ontology_path) == row["ontology_sha256"], "Ontology changed")
    _require(
        hashlib.sha256(row["original_prompt"].encode("utf-8")).hexdigest()
        == row["original_prompt_sha256"],
        "Original prompt changed",
    )
    _require(row["prompt_specific_ontology_used"] is False, "Prompt-specific ontology forbidden")

    model_config = _load_yaml(model_config_path)
    ontology = _load_yaml(ontology_path)
    _require(model_config["model"]["revision"] == row["model_revision"], "Model revision mismatch")
    _require(ontology["ontology_id"] == row["ontology_id"], "Ontology ID mismatch")
    _require(ontology["category"] == row["category"], "Ontology/category mismatch")
    _require(ontology["prompt_independent"] is True, "Ontology must be prompt-independent")

    mode = AblationMode(row["ablation_id"])
    execution_prompt, exact_suffix, suffix_source = _compose_prompt(
        row["original_prompt"], mode, ontology
    )
    dtype = _torch_dtype("bfloat16")
    device = torch.device("cuda")
    _require(torch.cuda.is_available(), "CUDA is required for production generation")
    torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()

    _sync()
    load_start = time.perf_counter()
    adapter = create_adapter(model_config["model"], device=device, dtype=dtype)
    adapter.load()
    _sync()
    load_seconds = time.perf_counter() - load_start

    generator = torch.Generator(device=device).manual_seed(int(row["seed"]))
    generation = dict(model_config["generation"])
    num_steps = int(generation.pop("num_inference_steps"))
    generation.pop("task", None)

    _sync()
    conditioning_start = time.perf_counter()
    adapter.begin_conditioning_provenance_scope()
    latents, state = adapter.prepare_initial_latents(
        row["original_prompt"], batch_size=1, generator=generator, **generation
    )
    initial_latent_fingerprint = tensor_fingerprint(latents)
    timesteps = adapter.set_timesteps(num_steps, latents=latents, state=state)
    time_map = build_unified_time_map(timesteps)
    controller = AblationController(
        mode,
        time_map.unified_times,
        early_window=tuple(float(value) for value in row["early_window_unified_diffusion_time"]),
    )
    base_condition = adapter.prepare_prompt_for_state(
        execution_prompt, state, prompt_view="execution_prompt", call_role="base"
    )
    negative_conditions: Sequence[Any] = []
    negative_weights: Sequence[float] = []
    positive_conditions: Sequence[Any] = []
    positive_weights: Sequence[float] = []
    neutral_conditions: Sequence[Any] = []
    neutral_weights: Sequence[float] = []
    prepared_counts = {"source": 0, "target": 0, "neutral": 0}
    if controller.requires_concept_fields:
        groups = ontology["runtime_probe_groups"]
        negative_conditions, negative_weights = _prepare_probe_group(
            adapter, state, groups["negative"], "source"
        )
        positive_conditions, positive_weights = _prepare_probe_group(
            adapter, state, groups["positive"], "target"
        )
        prepared_counts["source"] = len(negative_conditions)
        prepared_counts["target"] = len(positive_conditions)
        if controller.requires_neutral:
            neutral_conditions, neutral_weights = _prepare_probe_group(
                adapter, state, groups["neutral"], "neutral"
            )
            prepared_counts["neutral"] = len(neutral_conditions)
    _sync()
    conditioning_seconds = time.perf_counter() - conditioning_start

    mask_config = mask_config_from_mapping(row["mask"])
    counters = {
        "base_denoiser_forwards": 0,
        "source_probe_forwards": 0,
        "target_probe_forwards": 0,
        "neutral_probe_forwards": 0,
        "direction_recomputations": 0,
        "active_intervention_steps": 0,
    }
    frozen = FrozenDirectionState()
    traces: list[dict[str, Any]] = []
    cumulative_energy = 0.0

    _sync()
    denoising_start = time.perf_counter()
    with torch.inference_mode():
        for step_index, (timestep, unified_time) in enumerate(
            zip(timesteps, time_map.unified_times)
        ):
            native_sigma, sigma_source = _scheduler_sigma(adapter, step_index)
            base_prediction = adapter.predict_vector_field(latents, timestep, base_condition, state)
            counters["base_denoiser_forwards"] += 1
            _require(bool(torch.isfinite(base_prediction).all()), "Non-finite base prediction")
            layout = adapter.latent_layout(base_prediction)
            direction: torch.Tensor | None = None
            direction_meta: dict[str, Any] | None = None
            unit_direction: torch.Tensor | None = None
            if controller.should_compute_direction(step_index):
                source_prediction = weighted_prediction_mean(
                    negative_conditions,
                    negative_weights,
                    lambda condition: adapter.predict_vector_field(
                        latents, timestep, condition, state
                    ),
                )
                counters["source_probe_forwards"] += len(negative_conditions)
                target_prediction = weighted_prediction_mean(
                    positive_conditions,
                    positive_weights,
                    lambda condition: adapter.predict_vector_field(
                        latents, timestep, condition, state
                    ),
                )
                counters["target_probe_forwards"] += len(positive_conditions)
                if controller.requires_neutral:
                    neutral_prediction = weighted_prediction_mean(
                        neutral_conditions,
                        neutral_weights,
                        lambda condition: adapter.predict_vector_field(
                            latents, timestep, condition, state
                        ),
                    )
                    counters["neutral_probe_forwards"] += len(neutral_conditions)
                    direction, direction_meta = conceptsteer_direction(
                        v_base=base_prediction,
                        v_unsafe=source_prediction,
                        v_safe=target_prediction,
                        v_neutral=neutral_prediction,
                        feature_dim=layout.feature_dim,
                        margin=float(row["margin"]),
                        mask_config=mask_config,
                    )
                else:
                    direction, direction_meta = compute_no_neutral_direction(
                        base_prediction,
                        source_prediction,
                        target_prediction,
                        feature_dim=layout.feature_dim,
                        margin=float(row["margin"]),
                        mask_config=mask_config,
                    )
                counters["direction_recomputations"] += 1
                _require(bool(torch.isfinite(direction).all()), "Non-finite direction")
                unit_direction = normalize_direction_rms(direction)
                if controller.should_capture_frozen_direction(step_index):
                    frozen = FrozenDirectionState(
                        direction_unit_rms=unit_direction.detach().clone(),
                        acquired_step=step_index,
                        acquired_timestep=time_map.raw_timesteps[step_index],
                        acquired_unified_time=float(unified_time),
                        acquired_sigma=native_sigma,
                        fingerprint=tensor_fingerprint(unit_direction),
                        direction_rms=float(tensor_rms(direction).detach().cpu()),
                    )

            direction_source = None
            if controller.should_use_frozen_direction(step_index):
                _require(frozen.direction_unit_rms is not None, "Frozen direction was not captured")
                unit_direction = frozen.direction_unit_rms
                direction_source = "frozen"
            elif unit_direction is not None:
                direction_source = "recomputed"

            schedule_weight = controller.current_schedule_weight(step_index)
            application_weight = controller.current_application_weight(step_index)
            if controller.should_apply_direction(step_index):
                _require(unit_direction is not None, "Application requested without direction")
                model_prediction = apply_relative_direction(
                    base_prediction,
                    unit_direction,
                    relative_strength=float(row["relative_rms_strength"]) * application_weight,
                )
                counters["active_intervention_steps"] += 1
            else:
                model_prediction = base_prediction
            delta = model_prediction.float() - base_prediction.float()
            step_energy = intervention_energy(base_prediction, delta)
            cumulative_energy += step_energy
            base_rms = float(tensor_rms(base_prediction).detach().cpu())
            delta_rms = float(tensor_rms(delta, eps=0.0).detach().cpu())
            relative_delta_rms = delta_rms / max(base_rms, 1.0e-12)
            traces.append(
                {
                    "step_index": step_index,
                    "native_timestep": time_map.raw_timesteps[step_index],
                    "native_sigma": native_sigma,
                    "native_sigma_source": sigma_source,
                    "unified_diffusion_time": float(unified_time),
                    "time_conversion_policy": time_map.conversion_policy,
                    "compute_direction": controller.should_compute_direction(step_index),
                    "apply_direction": controller.should_apply_direction(step_index),
                    "capture_frozen_direction": controller.should_capture_frozen_direction(step_index),
                    "use_frozen_direction": controller.should_use_frozen_direction(step_index),
                    "direction_source": direction_source,
                    "schedule_weight": schedule_weight,
                    "application_weight": application_weight,
                    "base_rms": base_rms,
                    "delta_rms": delta_rms,
                    "relative_delta_rms": relative_delta_rms,
                    "step_intervention_energy": step_energy,
                    "cumulative_intervention_energy": cumulative_energy,
                    "direction_fingerprint": (
                        tensor_fingerprint(unit_direction) if unit_direction is not None else None
                    ),
                    "direction_metadata": direction_meta,
                    "counters_after_step": dict(counters),
                }
            )
            result = adapter.scheduler_step(
                model_prediction, timestep, latents, state, generator=generator
            )
            latents, state = result.latents, result.state
    _sync()
    denoising_seconds = time.perf_counter() - denoising_start

    _sync()
    decode_start = time.perf_counter()
    with torch.inference_mode():
        decoded = adapter.decode_latents(latents, state)
    image = _decoded_to_pil(decoded)
    _sync()
    decode_seconds = time.perf_counter() - decode_start

    recomputations = sum(controller.should_compute_direction(i) for i in range(num_steps))
    active_steps = sum(controller.should_apply_direction(i) for i in range(num_steps))
    expected_counts = {
        "base_denoiser_forwards": num_steps,
        "source_probe_forwards": recomputations * prepared_counts["source"],
        "target_probe_forwards": recomputations * prepared_counts["target"],
        "neutral_probe_forwards": recomputations * prepared_counts["neutral"],
        "direction_recomputations": recomputations,
        "active_intervention_steps": active_steps,
    }
    _require(counters == expected_counts, f"Forward-count mismatch: {counters} != {expected_counts}")
    if mode is AblationMode.NO_NEUTRAL_AR:
        _require(prepared_counts["neutral"] == 0, "A03 encoded a neutral prompt")
        _require(counters["neutral_probe_forwards"] == 0, "A03 evaluated a neutral field")
    if not controller.requires_concept_fields:
        _require(sum(prepared_counts.values()) == 0, "Non-steering ablation encoded concept probes")

    adapter_metadata = adapter.inspect()
    conditioning_provenance = adapter.conditioning_provenance()
    timing = {
        "model_calibration_loading_seconds": load_seconds,
        "text_conditioning_preparation_seconds": conditioning_seconds,
        "denoising_seconds": denoising_seconds,
        "decode_seconds": decode_seconds,
        "end_to_end_seconds": time.perf_counter() - total_start,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "denoiser_forward_count": counters["base_denoiser_forwards"],
        "extra_method_forward_count": (
            counters["source_probe_forwards"]
            + counters["target_probe_forwards"]
            + counters["neutral_probe_forwards"]
        ),
    }
    metadata = {
        "schema_version": "push-for-iclr.ablation-cell-metadata.v1",
        "job": dict(row),
        "job_manifest_path": str(manifest_path),
        "job_manifest_file_sha256": manifest_file_sha256,
        "original_prompt": row["original_prompt"],
        "execution_prompt": execution_prompt,
        "execution_prompt_sha256": hashlib.sha256(execution_prompt.encode("utf-8")).hexdigest(),
        "suffix_source_path": row["ontology_path"] if suffix_source else None,
        "suffix_source_sha256": row["ontology_sha256"] if suffix_source else None,
        "suffix_field": suffix_source,
        "exact_suffix_text": exact_suffix,
        "initial_latent_fingerprint": initial_latent_fingerprint,
        "adapter": adapter_metadata,
        "conditioning_provenance": conditioning_provenance,
        "prepared_probe_counts": prepared_counts,
        "unified_time_map": {
            "raw_timesteps": list(time_map.raw_timesteps),
            "unified_times": list(time_map.unified_times),
            "conversion_policy": time_map.conversion_policy,
            "scale": time_map.scale,
        },
        "frozen_direction_state": {
            "acquisition_step": frozen.acquired_step,
            "acquisition_timestep": frozen.acquired_timestep,
            "acquisition_unified_diffusion_time": frozen.acquired_unified_time,
            "acquisition_sigma": frozen.acquired_sigma,
            "direction_fingerprint": frozen.fingerprint,
            "direction_rms": frozen.direction_rms,
        },
        "last_nonzero_schedule_step": controller.last_nonzero_schedule_step,
        "tail_relative_strength": (
            float(row["relative_rms_strength"])
            if mode is AblationMode.EARLY_FROZEN
            else None
        ),
        "tail_step_count": sum(
            controller.should_use_frozen_direction(i) for i in range(num_steps)
        ),
        "intervention_energy": cumulative_energy,
        "forward_counts": counters,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
        },
    }
    trace_validation = {
        "schema_version": "push-for-iclr.ablation-trace-validation.v1",
        "status": "PASS",
        "expected_counts": expected_counts,
        "observed_counts": counters,
        "neutral_encoded": prepared_counts["neutral"] > 0,
        "neutral_evaluated": counters["neutral_probe_forwards"] > 0,
        "initial_latent_fingerprint": initial_latent_fingerprint,
        "all_trace_values_finite": all(
            math.isfinite(math_value)
            for trace in traces
            for math_value in [
                trace["base_rms"],
                trace["delta_rms"],
                trace["relative_delta_rms"],
                trace["step_intervention_energy"],
            ]
        ),
    }
    _require(trace_validation["all_trace_values_finite"], "Non-finite intervention trace")

    _save_image_atomic(output_dir / "image.png", image)
    _atomic_json(output_dir / "metadata.json", metadata)
    _atomic_json(output_dir / "timing.json", timing)
    trace_payload = "".join(f"{canonical_json(trace)}\n" for trace in traces).encode("utf-8")
    _atomic_bytes(output_dir / "intervention_trace.jsonl", trace_payload)
    _atomic_json(output_dir / "trace_validation.json", trace_validation)
    _atomic_bytes(output_dir / "_SUCCESS", b"")
    print(f"SUCCESS {output_dir}")
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--index", type=int, default=None)
    args = parser.parse_args()
    index = args.index
    if index is None:
        value = os.environ.get("SLURM_ARRAY_TASK_ID")
        if value is None:
            parser.error("--index or SLURM_ARRAY_TASK_ID is required")
        index = int(value)
    output_dir: Path | None = None
    row: Mapping[str, Any] | None = None
    try:
        rows, _ = validate_sealed_job_manifest(args.manifest.resolve())
        _require(0 <= index < len(rows), f"Index out of range: {index}")
        row = rows[index]
        _require(row["job_index"] == index, "Manifest index mismatch")
        output_dir = (
            REPOSITORY_ROOT
            / "outputs/PUSH_FOR_ICLR"
            / row["expected_output_relative_path"]
        ).resolve()
        execute_cell(row, args.manifest.resolve(), args.manifest_file_sha256)
        return 0
    except BaseException as exc:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "schema_version": "push-for-iclr.cell-failure.v1",
                "status": (
                    "FAILED_NUMERICAL_CONTRACT"
                    if isinstance(exc, (AblationContractError, AblationManifestError))
                    else "FAILED_GENERATION"
                ),
                "exception_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(),
                "job_index": index,
                "job": dict(row) if row is not None else None,
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            }
            try:
                _atomic_json(_failure_path(output_dir), failure)
            except BaseException:
                traceback.print_exc()
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
