#!/usr/bin/env python3
"""Execute one immutable trust-region development cell."""

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

from hierasafe_flow.adapters.base import PromptCondition  # noqa: E402
from hierasafe_flow.adapters.registry import create_adapter  # noqa: E402
from hierasafe_flow.campaigns.push_for_iclr.ablation_controller import (  # noqa: E402
    build_unified_time_map,
    mask_config_from_mapping,
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
from hierasafe_flow.campaigns.push_for_iclr.trust_region_controller import (  # noqa: E402
    TrustRegionArm,
    TrustRegionContractError,
    bounded_trust_region_delta,
    calibrate_relative_rms_direction,
    contextualize_probe,
    matched_pair_direction,
    project_semantic_component,
    raised_cosine_window,
    route_pair_directions,
)
from hierasafe_flow.steering.canonical import (  # noqa: E402
    canonicalize_prediction,
    prediction_from_x0,
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
        scheduler = getattr(getattr(adapter, attribute, None), "scheduler", None)
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
        tensor = value.detach().float().cpu()
        if tensor.ndim == 4:
            _require(tensor.shape[0] == 1, f"Expected one image, got {tensor.shape[0]}")
            tensor = tensor[0]
        if tensor.ndim == 3 and tensor.shape[0] in (1, 3, 4):
            tensor = tensor.permute(1, 2, 0)
        if float(tensor.min()) < 0.0:
            tensor = tensor.add(1.0).div(2.0)
        array = tensor.clamp(0.0, 1.0).mul(255).round().to(torch.uint8).numpy()
        if array.ndim == 3 and array.shape[-1] == 1:
            array = array[..., 0]
        return Image.fromarray(array)
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


def _move_condition_value(value: Any, device: torch.device | str) -> Any:
    """Copy every tensor in immutable prompt-conditioning data to ``device``."""

    if torch.is_tensor(value):
        return value.detach().to(device=device)
    if isinstance(value, Mapping):
        return {key: _move_condition_value(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_condition_value(item, device) for item in value)
    if isinstance(value, list):
        return [_move_condition_value(item, device) for item in value]
    return value


def _condition_on_device(condition: PromptCondition, device: torch.device | str) -> PromptCondition:
    _require(isinstance(condition, PromptCondition), "Adapter returned a non-PromptCondition value")
    return PromptCondition(
        prompt=condition.prompt,
        data=_move_condition_value(condition.data, device),
    )


def _persistent_cpu_condition(condition: PromptCondition) -> PromptCondition:
    return _condition_on_device(condition, torch.device("cpu"))


def _predict_with_materialized_condition(
    adapter: Any,
    latents: torch.Tensor,
    timestep: Any,
    condition: PromptCondition,
    state: Any,
) -> torch.Tensor:
    """Materialize one condition on the execution device for exactly one forward."""

    materialized = _condition_on_device(condition, adapter.device)
    return adapter.predict_vector_field(latents, timestep, materialized, state)


def _condition_storage_stats(conditions: Sequence[PromptCondition]) -> dict[str, Any]:
    tensors: list[torch.Tensor] = []

    def collect(value: Any) -> None:
        if torch.is_tensor(value):
            tensors.append(value)
        elif isinstance(value, Mapping):
            for item in value.values():
                collect(item)
        elif isinstance(value, (tuple, list)):
            for item in value:
                collect(item)

    for condition in conditions:
        collect(condition.data)
    devices = sorted({tensor.device.type for tensor in tensors})
    return {
        "policy": "persistent_cpu_per_call_gpu_materialization_v1",
        "condition_count": len(conditions),
        "tensor_count": len(tensors),
        "persistent_devices": devices,
        "persistent_bytes": sum(tensor.numel() * tensor.element_size() for tensor in tensors),
        "execution_device": "cuda",
        "dtype_preserving": True,
    }


def _prepare_group(
    adapter: Any,
    state: Any,
    original_prompt: str,
    probes: Sequence[Mapping[str, Any]],
    role: str,
) -> tuple[list[Any], list[float], list[str]]:
    conditions: list[Any] = []
    weights: list[float] = []
    prompts: list[str] = []
    for probe in probes:
        prompt = contextualize_probe(original_prompt, str(probe["text"]))
        prompts.append(prompt)
        conditions.append(
            _persistent_cpu_condition(
                adapter.prepare_prompt_for_state(
                prompt,
                state,
                prompt_view="trust_region_context_probe",
                call_role=f"{role}:{probe['id']}",
                )
            )
        )
        weights.append(float(probe["weight"]))
    return conditions, weights, prompts


def execute_cell(row: Mapping[str, Any], manifest_path: Path, manifest_file_sha256: str) -> Path:
    output_root = REPOSITORY_ROOT / "outputs/PUSH_FOR_ICLR"
    output_dir = (output_root / row["expected_output_relative_path"]).resolve()
    _require(output_dir.is_relative_to(output_root.resolve()), "Output escaped campaign root")
    if (output_dir / "_SUCCESS").exists():
        print(f"ALREADY_COMPLETE {output_dir}")
        return output_dir
    _require(not (output_dir / "_FAILURE.json").exists(), "Preserved failure requires retry manifest")
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
    _require(sha256_file(manifest_path) == manifest_file_sha256, "Physical manifest changed")
    model_path = REPOSITORY_ROOT / row["model_config_path"]
    ontology_path = REPOSITORY_ROOT / row["ontology_path"]
    _require(sha256_file(model_path) == row["model_config_sha256"], "Model config changed")
    _require(sha256_file(ontology_path) == row["ontology_sha256"], "Ontology changed")
    _require(
        hashlib.sha256(row["original_prompt"].encode("utf-8")).hexdigest() == row["original_prompt_sha256"],
        "Original prompt changed",
    )
    _require(row["ablation_split"] == "development", "Locked validation is forbidden")
    _require(row["prompt_specific_ontology_used"] is False, "Prompt-specific ontology forbidden")
    _require(
        row["conditioning_memory_policy"]
        == "persistent_cpu_per_call_gpu_materialization_v1",
        "Unsealed conditioning memory policy",
    )

    model_config = _load_yaml(model_path)
    ontology = _load_yaml(ontology_path)
    _require(model_config["model"]["revision"] == row["model_revision"], "Revision mismatch")
    _require(ontology["ontology_id"] == row["ontology_id"], "Ontology ID mismatch")
    _require(ontology["category"] == row["category"], "Ontology category mismatch")
    _require(ontology["prompt_independent"] is True, "Ontology must be prompt-independent")
    arm = TrustRegionArm.from_mapping(row["arm"])
    _require(arm.arm_id == row["arm_id"], "Arm ID mismatch")

    device = torch.device("cuda")
    _require(torch.cuda.is_available(), "CUDA is required")
    torch.cuda.reset_peak_memory_stats()
    total_start = time.perf_counter()
    _sync()
    load_start = time.perf_counter()
    adapter = create_adapter(model_config["model"], device=device, dtype=torch.bfloat16)
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
    base_condition = _persistent_cpu_condition(
        adapter.prepare_prompt_for_state(
            row["original_prompt"], state, prompt_view="exact_original_prompt", call_role="base"
        )
    )

    prepared_pairs: list[dict[str, Any]] = []
    neutral_conditions: list[Any] = []
    neutral_weights: list[float] = []
    composed_probe_prompts: dict[str, Any] = {}
    if arm.enabled:
        neutral_conditions, neutral_weights, neutral_prompts = _prepare_group(
            adapter, state, row["original_prompt"], ontology["neutral_probes"], "neutral"
        )
        composed_probe_prompts["neutral"] = neutral_prompts
        for pair in ontology["runtime_probe_pairs"]:
            source_conditions, source_weights, source_prompts = _prepare_group(
                adapter, state, row["original_prompt"], pair["source"], f"source:{pair['id']}"
            )
            target_conditions, target_weights, target_prompts = _prepare_group(
                adapter, state, row["original_prompt"], pair["target"], f"target:{pair['id']}"
            )
            prepared_pairs.append(
                {
                    "id": pair["id"],
                    "parent": pair["parent"],
                    "source_conditions": source_conditions,
                    "source_weights": source_weights,
                    "target_conditions": target_conditions,
                    "target_weights": target_weights,
                }
            )
            composed_probe_prompts[pair["id"]] = {"source": source_prompts, "target": target_prompts}
    all_conditions = [base_condition, *neutral_conditions]
    for pair in prepared_pairs:
        all_conditions.extend(pair["source_conditions"])
        all_conditions.extend(pair["target_conditions"])
    condition_storage = _condition_storage_stats(all_conditions)
    _require(condition_storage["persistent_devices"] in ([], ["cpu"]), "Condition remained on GPU")
    torch.cuda.empty_cache()
    _sync()
    conditioning_seconds = time.perf_counter() - conditioning_start

    mask_config = mask_config_from_mapping(row["mask"])
    source_per_step = sum(len(pair["source_conditions"]) for pair in prepared_pairs)
    target_per_step = sum(len(pair["target_conditions"]) for pair in prepared_pairs)
    counters = {
        "base_denoiser_forwards": 0,
        "source_probe_forwards": 0,
        "target_probe_forwards": 0,
        "neutral_probe_forwards": 0,
        "pair_direction_evaluations": 0,
        "routing_evaluations": 0,
        "active_intervention_steps": 0,
    }
    traces: list[dict[str, Any]] = []
    cumulative_energy = 0.0

    _sync()
    denoising_start = time.perf_counter()
    with torch.inference_mode():
        for step_index, (timestep, unified_time) in enumerate(zip(timesteps, time_map.unified_times)):
            native_sigma, sigma_source = _scheduler_sigma(adapter, step_index)
            base_prediction = _predict_with_materialized_condition(
                adapter, latents, timestep, base_condition, state
            )
            counters["base_denoiser_forwards"] += 1
            _require(bool(torch.isfinite(base_prediction).all()), "Non-finite base prediction")
            guidance_scale = float(row["generation"].get("guidance_scale", 1.0))
            base_canonical = canonicalize_prediction(
                adapter=adapter,
                model_id=row["model_id"],
                latents=latents,
                native=base_prediction,
                timestep=timestep,
                step_index=step_index,
                guidance_scale=guidance_scale,
                branch="base",
            )
            base_x0 = base_canonical.predicted_x0
            _require(bool(torch.isfinite(base_x0).all()), "Non-finite canonical base x0")
            layout = adapter.latent_layout(base_x0)

            def canonical_x0(native_prediction: torch.Tensor, branch: str) -> torch.Tensor:
                canonical = canonicalize_prediction(
                    adapter=adapter,
                    model_id=row["model_id"],
                    latents=latents,
                    native=native_prediction,
                    timestep=timestep,
                    step_index=step_index,
                    guidance_scale=guidance_scale,
                    branch=branch,
                )
                _require(
                    canonical.parameterization == base_canonical.parameterization,
                    "Probe parameterization differs from base parameterization",
                )
                _require(
                    canonical.schedule == base_canonical.schedule,
                    "Probe schedule point differs from base schedule point",
                )
                _require(
                    bool(torch.isfinite(canonical.predicted_x0).all()),
                    f"Non-finite canonical x0 for {branch}",
                )
                return canonical.predicted_x0

            schedule_weight = (
                raised_cosine_window(
                    step_index,
                    num_steps,
                    start_fraction=arm.start_fraction,
                    end_fraction=arm.end_fraction,
                )
                if arm.enabled
                else 0.0
            )
            pair_records: list[Any] = []
            routing: dict[str, Any] = {
                "selected_pairs": [],
                "all_pair_scores": {},
                "weights": {},
                "reason": "arm_disabled_or_schedule_zero",
            }
            projection_meta = {
                "coefficient": arm.semantic_projection,
                "mean_abs_cosine_before": 0.0,
                "mean_abs_cosine_after": 0.0,
            }
            calibration_meta = {
                "policy": "normalize_projected_direction_then_scale_to_base_rms_v1",
                "reason": "arm_disabled_or_schedule_zero",
                "requested_relative_rms": arm.target_relative_rms,
                "base_rms": float(tensor_rms(base_x0, eps=0.0).cpu()),
                "input_direction_rms": 0.0,
                "unit_direction_rms": 0.0,
                "calibration_scale": 0.0,
                "calibrated_direction_rms": 0.0,
                "achieved_pre_trust_relative_rms": 0.0,
            }
            trust_meta = {
                "schedule_weight": schedule_weight,
                "effective_local_cap": 0.0,
                "maximum_observed_local_relative": 0.0,
                "energy_before_budget_scale": 0.0,
                "energy_after_budget_scale": 0.0,
                "budget_scale": 0.0,
            }
            if arm.enabled and schedule_weight > 0.0:
                neutral_prediction = weighted_prediction_mean(
                    neutral_conditions,
                    neutral_weights,
                    lambda condition: _predict_with_materialized_condition(
                        adapter, latents, timestep, condition, state
                    ),
                )
                counters["neutral_probe_forwards"] += len(neutral_conditions)
                neutral_x0 = canonical_x0(neutral_prediction, "neutral")
                pair_directions = []
                for pair in prepared_pairs:
                    source_prediction = weighted_prediction_mean(
                        pair["source_conditions"],
                        pair["source_weights"],
                        lambda condition: _predict_with_materialized_condition(
                            adapter, latents, timestep, condition, state
                        ),
                    )
                    target_prediction = weighted_prediction_mean(
                        pair["target_conditions"],
                        pair["target_weights"],
                        lambda condition: _predict_with_materialized_condition(
                            adapter, latents, timestep, condition, state
                        ),
                    )
                    counters["source_probe_forwards"] += len(pair["source_conditions"])
                    counters["target_probe_forwards"] += len(pair["target_conditions"])
                    source_x0 = canonical_x0(
                        source_prediction, f"source:{pair['id']}"
                    )
                    target_x0 = canonical_x0(
                        target_prediction, f"target:{pair['id']}"
                    )
                    pair_direction = matched_pair_direction(
                        pair_id=pair["id"],
                        parent=pair["parent"],
                        v_base=base_x0,
                        v_source=source_x0,
                        v_target=target_x0,
                        v_neutral=neutral_x0,
                        feature_dim=layout.feature_dim,
                        margin=float(row["margin"]),
                        mask_config=mask_config,
                        activation_top_fraction=arm.activation_top_fraction,
                    )
                    pair_directions.append(pair_direction)
                    counters["pair_direction_evaluations"] += 1
                routed_direction, routing = route_pair_directions(
                    pair_directions,
                    base=base_x0,
                    top_k_pairs=arm.top_k_pairs,
                    minimum_pair_score=arm.minimum_pair_score,
                    routing_temperature=arm.routing_temperature,
                )
                counters["routing_evaluations"] += 1
                projected_direction, projection_meta = project_semantic_component(
                    routed_direction,
                    base_x0 - neutral_x0,
                    feature_dim=layout.feature_dim,
                    coefficient=arm.semantic_projection,
                    eps=mask_config.eps,
                )
                calibrated_direction, calibration_meta = calibrate_relative_rms_direction(
                    base=base_x0,
                    direction=projected_direction,
                    target_relative_rms=arm.target_relative_rms,
                    eps=mask_config.eps,
                )
                delta, trust_meta = bounded_trust_region_delta(
                    base=base_x0,
                    direction=calibrated_direction,
                    feature_dim=layout.feature_dim,
                    schedule_weight=schedule_weight,
                    max_local_relative=arm.max_local_relative,
                    remaining_energy=max(0.0, arm.cumulative_energy_budget - cumulative_energy),
                    eps=mask_config.eps,
                )
                pair_records = [
                    {
                        "pair_id": pair.pair_id,
                        "parent": pair.parent,
                        "score": pair.score,
                        "metadata": pair.metadata,
                    }
                    for pair in pair_directions
                ]
            else:
                delta = torch.zeros_like(base_x0)

            delta_f = delta.float()
            if float(tensor_rms(delta_f, eps=0.0).cpu()) > 0.0:
                counters["active_intervention_steps"] += 1
            step_energy = float(trust_meta["energy_after_budget_scale"])
            cumulative_energy += step_energy
            _require(
                cumulative_energy <= arm.cumulative_energy_budget + 5.0e-7,
                "Cumulative intervention budget exceeded",
            )
            native_delta_f = torch.zeros_like(base_prediction, dtype=torch.float32)
            roundtrip_relative_rms_error = 0.0
            if float(tensor_rms(delta_f, eps=0.0).cpu()) > 0.0:
                steered_x0 = (base_x0.float() + delta_f).to(base_x0.dtype)
                model_prediction = prediction_from_x0(
                    latents,
                    steered_x0,
                    base_canonical.parameterization,
                    base_canonical.schedule,
                ).to(base_prediction.dtype)
                _require(
                    bool(torch.isfinite(model_prediction).all()),
                    "Non-finite native prediction converted from steered x0",
                )
                native_delta_f = model_prediction.float() - base_prediction.float()
                roundtrip_x0 = canonicalize_prediction(
                    adapter=adapter,
                    model_id=row["model_id"],
                    latents=latents,
                    native=model_prediction,
                    timestep=timestep,
                    step_index=step_index,
                    guidance_scale=guidance_scale,
                    branch="steered_roundtrip",
                ).predicted_x0
                roundtrip_relative_rms_error = float(
                    tensor_rms(roundtrip_x0.float() - steered_x0.float(), eps=0.0).cpu()
                    / tensor_rms(steered_x0.float()).cpu()
                )
                _require(
                    roundtrip_relative_rms_error <= 1.0e-2,
                    "Canonical x0 roundtrip error exceeded one percent",
                )
            else:
                steered_x0 = base_x0
                model_prediction = base_prediction
            calibration_meta["direction_space"] = "canonical_predicted_x0"
            traces.append(
                {
                    "step_index": step_index,
                    "native_timestep": time_map.raw_timesteps[step_index],
                    "native_sigma": native_sigma,
                    "native_sigma_source": sigma_source,
                    "unified_diffusion_time": float(unified_time),
                    "schedule_weight": schedule_weight,
                    "direction_space": "canonical_predicted_x0",
                    "canonical_prediction": base_canonical.metadata(),
                    "base_stats": tensor_stats(base_x0),
                    "native_base_stats": tensor_stats(base_prediction),
                    "delta_stats": tensor_stats(delta_f),
                    "native_delta_stats": tensor_stats(native_delta_f),
                    "steered_x0_stats": tensor_stats(steered_x0),
                    "canonical_roundtrip_relative_rms_error": roundtrip_relative_rms_error,
                    "step_intervention_energy": step_energy,
                    "cumulative_intervention_energy": cumulative_energy,
                    "remaining_energy": max(0.0, arm.cumulative_energy_budget - cumulative_energy),
                    "routing": routing,
                    "pair_records": pair_records,
                    "semantic_projection": projection_meta,
                    "relative_rms_calibration": calibration_meta,
                    "trust_region": trust_meta,
                    "delta_fingerprint": tensor_fingerprint(delta_f),
                    "counters_after_step": dict(counters),
                }
            )
            result = adapter.scheduler_step(model_prediction, timestep, latents, state, generator=generator)
            latents, state = result.latents, result.state
    _sync()
    denoising_seconds = time.perf_counter() - denoising_start
    _sync()
    decode_start = time.perf_counter()
    with torch.inference_mode():
        image = _decoded_to_pil(adapter.decode_latents(latents, state))
    _sync()
    decode_seconds = time.perf_counter() - decode_start

    scheduled_steps = (
        sum(
            raised_cosine_window(
                index,
                num_steps,
                start_fraction=arm.start_fraction,
                end_fraction=arm.end_fraction,
            )
            > 0.0
            for index in range(num_steps)
        )
        if arm.enabled
        else 0
    )
    expected_counts = {
        "base_denoiser_forwards": num_steps,
        "source_probe_forwards": scheduled_steps * source_per_step,
        "target_probe_forwards": scheduled_steps * target_per_step,
        "neutral_probe_forwards": scheduled_steps * len(neutral_conditions),
        "pair_direction_evaluations": scheduled_steps * len(prepared_pairs),
        "routing_evaluations": scheduled_steps,
    }
    for key, expected in expected_counts.items():
        _require(counters[key] == expected, f"Forward-count mismatch for {key}")
    if not arm.enabled:
        _require(sum(counters.values()) == num_steps, "Baseline evaluated concept probes")

    timing = {
        "model_loading_seconds": load_seconds,
        "conditioning_preparation_seconds": conditioning_seconds,
        "denoising_seconds": denoising_seconds,
        "decode_seconds": decode_seconds,
        "end_to_end_seconds": time.perf_counter() - total_start,
        "peak_gpu_memory_bytes": int(torch.cuda.max_memory_allocated()),
        "base_denoiser_forward_count": counters["base_denoiser_forwards"],
        "extra_method_forward_count": (
            counters["source_probe_forwards"]
            + counters["target_probe_forwards"]
            + counters["neutral_probe_forwards"]
        ),
    }
    metadata = {
        "schema_version": "push-for-iclr.trust-region-cell-metadata.v2",
        "job": dict(row),
        "job_manifest_path": str(manifest_path),
        "job_manifest_file_sha256": manifest_file_sha256,
        "original_prompt": row["original_prompt"],
        "initial_latent_fingerprint": initial_latent_fingerprint,
        "adapter": adapter.inspect(),
        "conditioning_provenance": adapter.conditioning_provenance(),
        "conditioning_memory": condition_storage,
        "probe_context_conditioning": row["probe_context_conditioning"],
        "direction_space": "canonical_predicted_x0",
        "native_conversion_policy": "canonical_x0_to_audited_model_parameterization_v1",
        "relative_rms_calibration_policy": "normalize_projected_direction_then_scale_to_canonical_x0_rms_v2",
        "composed_probe_prompts": composed_probe_prompts,
        "ontology_pair_count": len(prepared_pairs),
        "unified_time_map": {
            "raw_timesteps": list(time_map.raw_timesteps),
            "unified_times": list(time_map.unified_times),
            "conversion_policy": time_map.conversion_policy,
            "scale": time_map.scale,
        },
        "scheduled_step_count": scheduled_steps,
        "intervention_energy": cumulative_energy,
        "forward_counts": counters,
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "node": os.environ.get("SLURMD_NODENAME"),
        },
    }
    validation = {
        "schema_version": "push-for-iclr.trust-region-trace-validation.v1",
        "status": "PASS",
        "direction_space": "canonical_predicted_x0",
        "development_only": row["ablation_split"] == "development",
        "expected_counts": expected_counts,
        "observed_counts": counters,
        "cumulative_energy": cumulative_energy,
        "cumulative_energy_budget": arm.cumulative_energy_budget,
        "energy_budget_respected": cumulative_energy <= arm.cumulative_energy_budget + 5.0e-7,
        "all_trace_values_finite": all(
            math.isfinite(float(trace["step_intervention_energy"]))
            and math.isfinite(float(trace["cumulative_intervention_energy"]))
            and math.isfinite(float(trace["canonical_roundtrip_relative_rms_error"]))
            for trace in traces
        ),
    }
    _require(validation["energy_budget_respected"], "Energy validation failed")
    _require(validation["all_trace_values_finite"], "Non-finite trace")
    _save_image_atomic(output_dir / "image.png", image)
    _atomic_json(output_dir / "metadata.json", metadata)
    _atomic_json(output_dir / "timing.json", timing)
    _atomic_bytes(
        output_dir / "intervention_trace.jsonl",
        "".join(f"{canonical_json(trace)}\n" for trace in traces).encode("utf-8"),
    )
    _atomic_json(output_dir / "trace_validation.json", validation)
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
        _require(0 <= index < len(rows), "Index out of range")
        row = rows[index]
        _require(row["job_index"] == index, "Manifest index mismatch")
        output_dir = (
            REPOSITORY_ROOT / "outputs/PUSH_FOR_ICLR" / row["expected_output_relative_path"]
        ).resolve()
        execute_cell(row, args.manifest.resolve(), args.manifest_file_sha256)
        return 0
    except BaseException as exc:
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            failure = {
                "schema_version": "push-for-iclr.trust-region-cell-failure.v1",
                "status": (
                    "FAILED_NUMERICAL_CONTRACT"
                    if isinstance(exc, (TrustRegionContractError, AblationManifestError))
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
                _atomic_json(output_dir / "_FAILURE.json", failure)
            except BaseException:
                traceback.print_exc()
        traceback.print_exc()
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
