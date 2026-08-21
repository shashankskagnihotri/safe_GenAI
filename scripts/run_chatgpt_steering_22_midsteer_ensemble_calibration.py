#!/usr/bin/env python3
"""Fit a diagnostic MidSteer artifact from independent prompt ensembles."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import traceback
from typing import Any

import torch

from hierasafe_flow.campaigns.chatgpt_steering_22_july import (
    artifact_model_role,
    atomic_json,
    build_generation_config,
    load_campaign_spec,
)
from hierasafe_flow.generation.runner import GenerationRunner
from hierasafe_flow.steering.related_work.midsteer_attn_output import (
    AttentionOutputMomentCapture,
    MIDSTEER_CONTROL_MODE,
    MIDSTEER_STEP_POLICY,
    MIDSTEER_TOKEN_SCOPE,
    MidSteerArtifact,
    fit_midsteer_site_transforms,
    merge_midsteer_moments,
    resolve_transformer_root,
)
from hierasafe_flow.utils.seed import make_generator


def read_row(path: Path, index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"Manifest {path} has no row {index}")


def release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def latent_kwargs(generation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in generation.items()
        if key not in {"prompt", "prompt_file", "flux_dual_view_conditioning"}
    }


def condition(adapter: Any, text: str, state: Any, role: str) -> Any:
    return adapter.prepare_prompt_for_state(
        text,
        state,
        prompt_view="midsteer_independent_prompt_ensemble_22_july",
        call_role=role,
    )


def merge_sites(
    aggregate: dict[str, dict[str, Any]] | None,
    observed: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if aggregate is None:
        return {
            site: merge_midsteer_moments(None, moments)
            for site, moments in observed.items()
        }
    if set(aggregate) != set(observed):
        raise RuntimeError("MidSteer attention-site topology changed across prompts")
    return {
        site: merge_midsteer_moments(aggregate[site], observed[site])
        for site in sorted(aggregate)
    }


@torch.inference_mode()
def capture_prompt(
    runner: GenerationRunner,
    generation: dict[str, Any],
    row: dict[str, Any],
    prompt: str,
    *,
    include_covariance: bool,
    token_aggregation: str,
    call_role: str,
) -> tuple[dict[str, dict[str, Any]], str, int]:
    generator = make_generator(0, runner.device)
    latents, state = runner.adapter.prepare_initial_latents(
        prompt=prompt,
        batch_size=1,
        generator=generator,
        **latent_kwargs(generation),
    )
    timesteps = runner.adapter.set_timesteps(
        int(generation["num_inference_steps"]), latents=latents, state=state
    )
    if len(timesteps) < 1:
        raise RuntimeError("MidSteer calibration received no diffusion timesteps")
    context = runner.adapter.denoising_step_context(0, len(timesteps), state)
    if int(context.local_step_index) != 0:
        raise RuntimeError("MidSteer ensemble calibration did not start at local step zero")
    state.extra["_active_denoising_step_context"] = context
    model_role = artifact_model_role(context, state)
    image_token_count: int | None = None
    if row["model_id"] in {"flux1_dev", "flux2_dev"}:
        if latents.ndim != 3:
            raise ValueError(
                "MidSteer FLUX calibration requires packed [batch,tokens,features] "
                f"latents, observed {tuple(latents.shape)}"
            )
        image_token_count = int(latents.shape[-2])
    prepared = condition(runner.adapter, prompt, state, call_role)
    with AttentionOutputMomentCapture(
        resolve_transformer_root(runner.adapter),
        include_covariance=include_covariance,
        image_token_count=image_token_count,
        token_aggregation=token_aggregation,
    ) as capture:
        runner.adapter.predict_vector_field(
            latents, timesteps[0], prepared, state
        )
    return capture.moments(), model_role, len(timesteps)


def calibrate(
    row: dict[str, Any],
    spec: dict[str, Any],
    ensemble_path: Path,
) -> dict[str, Any]:
    encoded = ensemble_path.read_bytes()
    ensemble = json.loads(encoded)
    if ensemble.get("schema_version") != 1:
        raise ValueError("Unsupported MidSteer ensemble schema")
    if ensemble.get("purpose") != "mechanism_diagnostic_not_production":
        raise ValueError("This runner accepts only explicitly diagnostic ensembles")
    for key in ("model_id", "prompt_id", "concept_manifest_sha256"):
        if ensemble.get(key) != row.get(key):
            raise ValueError(f"MidSteer ensemble/row mismatch for {key}")
    neutral_prompts = [str(value) for value in ensemble.get("neutral", [])]
    pairs = ensemble.get("pairs", {})
    if len(neutral_prompts) < 2 or not isinstance(pairs, dict) or not pairs:
        raise ValueError("MidSteer ensemble has insufficient populations")
    for pair_id, sides in pairs.items():
        if set(sides) != {"source", "target"}:
            raise ValueError(f"MidSteer pair {pair_id} lacks source/target prompts")
        if len(sides["source"]) < 2 or len(sides["source"]) != len(sides["target"]):
            raise ValueError(f"MidSteer pair {pair_id} has unbalanced populations")

    output = Path(spec["_root"]) / row["calibration_dir"]
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "midsteer_attn_output.pt"
    if artifact_path.exists():
        raise FileExistsError(f"Refusing to overwrite artifact: {artifact_path}")
    calibration_row = {**row, "variant": "baseline", "seed": 0}
    config = build_generation_config(calibration_row, spec)
    config["logging"]["output_dir"] = str(output / "runner")
    config["output"].update({"decode": False, "save_latents": False})
    runner = GenerationRunner(config)
    runner.adapter.load()
    generation = runner.config["generation"]

    neutral: dict[str, dict[str, Any]] | None = None
    model_role: str | None = None
    total_steps: int | None = None
    for index, prompt in enumerate(neutral_prompts):
        observed, observed_role, observed_steps = capture_prompt(
            runner,
            generation,
            row,
            prompt,
            include_covariance=True,
            token_aggregation="all",
            call_role=f"midsteer_neutral_{index:05d}",
        )
        if model_role is None:
            model_role, total_steps = observed_role, observed_steps
        elif observed_role != model_role or observed_steps != total_steps:
            raise RuntimeError("MidSteer model role/topology changed across neutral prompts")
        neutral = merge_sites(neutral, observed)
        del observed
        release_memory()
    assert neutral is not None and model_role is not None and total_steps is not None

    pair_moments: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for pair_id, sides in pairs.items():
        pair_moments[str(pair_id)] = {}
        for side in ("source", "target"):
            aggregate: dict[str, dict[str, Any]] | None = None
            for index, prompt in enumerate(sides[side]):
                observed, observed_role, observed_steps = capture_prompt(
                    runner,
                    generation,
                    row,
                    str(prompt),
                    include_covariance=False,
                    token_aggregation="average",
                    call_role=f"midsteer_{pair_id}_{side}_{index:05d}",
                )
                if observed_role != model_role or observed_steps != total_steps:
                    raise RuntimeError("MidSteer model role/topology changed across concept prompts")
                aggregate = merge_sites(aggregate, observed)
                del observed
            assert aggregate is not None
            pair_moments[str(pair_id)][side] = aggregate
            release_memory()

    transforms: list[dict[str, Any]] = []
    for site in sorted(neutral):
        fitted = fit_midsteer_site_transforms(
            neutral[site],
            {
                pair_id: {
                    side: pair_moments[pair_id][side][site]
                    for side in ("source", "target")
                }
                for pair_id in pair_moments
            },
        )
        for pair_id, values in sorted(fitted.items()):
            transforms.append(
                {
                    "model_role": model_role,
                    "calibration_step_index": 0,
                    "site": site,
                    "pair_id": pair_id,
                    **values,
                }
            )
        del fitted
        release_memory()

    metadata = {
        "model_id": row["model_id"],
        "prompt_id": row["prompt_id"],
        "model_config_sha256": row["model_config_sha256"],
        "concept_manifest": row["concept_manifest"],
        "concept_manifest_sha256": row["concept_manifest_sha256"],
        "ensemble_path": str(ensemble_path),
        "ensemble_sha256": hashlib.sha256(encoded).hexdigest(),
        "ensemble_protocol": ensemble["protocol"],
        "purpose": ensemble["purpose"],
        "neutral_prompt_population": len(neutral_prompts),
        "neutral_observation_policy": "all_image_tokens",
        "concept_prompt_population_per_side": {
            pair_id: len(sides["source"]) for pair_id, sides in pairs.items()
        },
        "concept_observation_policy": "one_token_average_per_independent_prompt",
        "moment_accumulation": "post_projection_exact_cpu_float64",
        "feature_group_count": 1,
        "token_scope": MIDSTEER_TOKEN_SCOPE,
        "calibration_steps": [
            {"global_step_index": 0, "model_role": model_role}
        ],
        "total_generation_steps": total_steps,
    }
    MidSteerArtifact.save(artifact_path, metadata=metadata, transforms=transforms)
    result = {
        "status": "completed",
        **metadata,
        "control_mode": MIDSTEER_CONTROL_MODE,
        "step_policy": MIDSTEER_STEP_POLICY,
        "midsteer_transform_count": len(transforms),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_json(output / "midsteer_attn_output_status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/chatgpt_steering_22_july.yaml",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--ensemble", type=Path, required=True)
    args = parser.parse_args()
    spec = load_campaign_spec(args.config)
    row = read_row(args.manifest, args.index)
    output = Path(spec["_root"]) / row["calibration_dir"]
    output.mkdir(parents=True, exist_ok=True)
    try:
        print(json.dumps(calibrate(row, spec, args.ensemble), sort_keys=True))
    except Exception as exc:
        atomic_json(
            output / "midsteer_attn_output_status.json",
            {
                "status": "failed",
                "model_id": row["model_id"],
                "prompt_id": row["prompt_id"],
                "failed_at": datetime.now(timezone.utc).isoformat(),
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            },
        )
        raise


if __name__ == "__main__":
    main()
