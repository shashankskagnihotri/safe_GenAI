#!/usr/bin/env python3
"""Build disjoint clean-space and MidSteer artifacts for one frozen model."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any
import json
import os
import traceback

import torch

from hierasafe_flow.campaigns.chatgpt_steering import (
    CampaignGenerationRunner,
    atomic_json,
    build_final_matrix,
    build_generation_config,
    load_campaign_spec,
    load_ontology,
    sha_file,
)
from hierasafe_flow.steering.canonical import canonicalize_prediction, channel_tokens
from hierasafe_flow.steering.related_work.midsteer import (
    choose_attention_module,
    fit_midsteer_artifact,
    resolve_module,
    resolve_transformer_root,
)


def _first_tensor(value: Any) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        return next((item for item in value if torch.is_tensor(item)), None)
    sample = getattr(value, "sample", None)
    return sample if torch.is_tensor(sample) else None


@contextmanager
def capture_activation(module: torch.nn.Module):
    captured: list[torch.Tensor] = []

    def hook(_module: Any, _inputs: Any, output: Any) -> None:
        tensor = _first_tensor(output)
        if tensor is not None:
            reduce_dims = tuple(range(max(tensor.ndim - 1, 0)))
            vector = tensor.detach().float().mean(dim=reduce_dims) if reduce_dims else tensor.detach().float()
            captured.append(vector.flatten().cpu())

    handle = module.register_forward_hook(hook)
    try:
        yield captured
    finally:
        handle.remove()


def _generator(adapter: Any, seed: int) -> torch.Generator:
    device = getattr(adapter, "device", None)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        return torch.Generator(device=device).manual_seed(seed)
    except Exception:
        return torch.Generator().manual_seed(seed)


def _initial_state(runner: CampaignGenerationRunner, prompt: str, seed: int) -> tuple[torch.Tensor, Any, list[Any]]:
    generation = runner.config["generation"]
    kwargs = {
        key: generation[key]
        for key in ("height", "width", "num_frames", "fps", "duration_seconds")
        if key in generation and generation[key] is not None
    }
    latents, state = runner.adapter.prepare_initial_latents(
        prompt,
        batch_size=1,
        generator=_generator(runner.adapter, seed),
        **kwargs,
    )
    timesteps = runner.adapter.set_timesteps(
        int(generation["num_inference_steps"]),
        latents=latents,
        state=state,
    )
    return latents, state, timesteps


def calibrate(model_index: int, config_path: str) -> dict[str, Any]:
    spec = load_campaign_spec(config_path)
    models = spec["models"]
    if not 0 <= model_index < len(models):
        raise IndexError(model_index)
    model = models[model_index]
    model_id = model["id"]
    rows = [row for row in build_final_matrix(spec) if row["model_id"] == model_id]
    row = dict(rows[0])
    row["variant"] = "baseline"
    row["active_pair_ids"] = list(load_ontology(spec))
    calibration_seed = int(spec["calibration"]["seed"])
    ontology = load_ontology(spec)
    first_prompt = next(iter(ontology.values()))["calibration"]["neutral"][0]
    row["prompt"] = first_prompt
    config = build_generation_config(row, spec)
    runner = CampaignGenerationRunner(config)
    runner.adapter.load()
    latents, state, timesteps = _initial_state(runner, first_prompt, calibration_seed)
    step_index = int(spec["calibration"]["denoising_step_index"])
    timestep = timesteps[step_index]
    root = resolve_transformer_root(runner.adapter)
    module_name = choose_attention_module(root)
    module = resolve_module(root, module_name)
    activations: dict[str, list[torch.Tensor]] = {"neutral": [], "source": [], "target": []}
    unsafe_prototypes: list[torch.Tensor] = []
    pair_order: list[str] = []

    for pair_id, pair in ontology.items():
        pair_order.append(pair_id)
        for group in ("neutral", "source", "target"):
            for prompt_index, text in enumerate(pair["calibration"][group]):
                role = f"calibration__{pair_id}__{group}__{prompt_index}"
                condition = runner.adapter.prepare_prompt_for_state(
                    text,
                    state,
                    prompt_view="global_v2_calibration",
                    call_role=role,
                )
                with torch.inference_mode(), capture_activation(module) as captured:
                    native = runner.adapter.predict_vector_field(latents, timestep, condition, state)
                if not captured:
                    raise RuntimeError(f"MidSteer hook captured no tensor at {module_name}")
                activations[group].append(torch.stack(captured).mean(dim=0))
                if group == "source" and prompt_index == 0:
                    canonical = canonicalize_prediction(
                        adapter=runner.adapter,
                        model_id=model_id,
                        latents=latents,
                        native=native,
                        timestep=timestep,
                        step_index=step_index,
                        guidance_scale=float(config["generation"].get("guidance_scale", 1.0)),
                        branch="calibration_unsafe",
                    )
                    tokens, _ = channel_tokens(canonical.predicted_x0, model_id, canonical.layout)
                    unsafe_prototypes.append(tokens.float().mean(dim=0).cpu())

    artifact_dir = Path(spec["_root"]) / spec["campaign"]["calibration_root"] / model_id
    artifact_dir.mkdir(parents=True, exist_ok=True)
    midsteer = fit_midsteer_artifact(
        module_name=module_name,
        neutral=torch.stack(activations["neutral"]),
        source=torch.stack(activations["source"]),
        target=torch.stack(activations["target"]),
    )
    midsteer_path = artifact_dir / "midsteer.json"
    midsteer.save(midsteer_path)
    prototype_path = artifact_dir / "clean_prototypes.pt"
    temporary = prototype_path.with_suffix(f".tmp.{os.getpid()}.pt")
    torch.save(
        {
            "unsafe_prototypes": torch.stack(unsafe_prototypes).to(dtype=torch.float32),
            "pair_ids": pair_order,
            "reduction": "channel_centroid",
            "calibration_seed": calibration_seed,
            "final_prompt_access": False,
        },
        temporary,
    )
    temporary.replace(prototype_path)
    result = {
        "status": "completed",
        "model_id": model_id,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "calibration_seed": calibration_seed,
        "final_prompt_access": False,
        "pair_count": len(pair_order),
        "midsteer_module": module_name,
        "midsteer_sha256": sha_file(midsteer_path),
        "prototype_sha256": sha_file(prototype_path),
        "prototype_shape": list(torch.stack(unsafe_prototypes).shape),
        "source_revisions": {
            "midsteer": "0f3b31e15cdda6ad0d46167e10319e896d6f1541",
            "sgf": "4bdd287475672c608eadfe32d466cf62b3a527fe",
            "safe_denoiser": "223415b2739de049969c8188e8c6c331ae3531b3",
        },
    }
    atomic_json(artifact_dir / "calibration_status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/chatgpt_steering_21_july.yaml")
    parser.add_argument("--model-index", type=int, default=None)
    args = parser.parse_args()
    index = args.model_index
    if index is None:
        index = int(os.environ["SLURM_ARRAY_TASK_ID"])
    spec = load_campaign_spec(args.config)
    model_id = spec["models"][index]["id"]
    status_path = Path(spec["_root"]) / spec["campaign"]["calibration_root"] / model_id / "calibration_status.json"
    try:
        result = calibrate(index, args.config)
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        atomic_json(
            status_path,
            {
                "status": "failed",
                "model_id": model_id,
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            },
        )
        raise


if __name__ == "__main__":
    main()
