#!/usr/bin/env python3
"""Calibrate schema-v2 post-output MidSteer for one prompt/model cell."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import traceback
from typing import Any

import torch

from hierasafe_flow.campaigns.chatgpt_steering_22_july import (
    artifact_model_role,
    atomic_json,
    build_generation_config,
    concept_tree,
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


def _release_host_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def read_row(path: str | Path, index: int) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"Manifest {path} has no row {index}")


def _condition(adapter: Any, text: str, state: Any, role: str) -> Any:
    return adapter.prepare_prompt_for_state(
        text,
        state,
        prompt_view="midsteer_attn_output_calibration_22_july",
        call_role=role,
    )


def _predict_capture(
    runner: GenerationRunner,
    latents: torch.Tensor,
    timestep: Any,
    state: Any,
    text: str,
    role: str,
    *,
    include_covariance: bool,
    image_token_count: int | None,
) -> dict[str, dict[str, Any]]:
    condition = _condition(runner.adapter, text, state, role)
    with AttentionOutputMomentCapture(
        resolve_transformer_root(runner.adapter),
        include_covariance=include_covariance,
        image_token_count=image_token_count,
    ) as capture:
        runner.adapter.predict_vector_field(latents, timestep, condition, state)
    return capture.moments()


def _latent_kwargs(generation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in generation.items()
        if key not in {"prompt", "prompt_file", "flux_dual_view_conditioning"}
    }


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _merge_checkpoint(
    path: Path,
    *,
    seed: int,
    model_role: str,
    global_step_index: int,
    neutral: dict[str, dict[str, Any]],
    branches: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> None:
    site_names = set(neutral)
    if not site_names:
        raise RuntimeError(f"MidSteer captured no sites at step {global_step_index}")
    for pair_id, sides in branches.items():
        if set(sides) != {"source", "target"}:
            raise RuntimeError(f"Incomplete MidSteer sides for {pair_id}")
        for side, values in sides.items():
            if set(values) != site_names:
                raise RuntimeError(
                    f"MidSteer site mismatch for {pair_id}/{side} at "
                    f"step {global_step_index}"
                )

    if path.exists():
        aggregate = torch.load(path, map_location="cpu", weights_only=False)
        if aggregate.get("schema_version") != 2:
            raise RuntimeError(f"Unknown MidSteer checkpoint schema: {path}")
        if (
            aggregate.get("model_role") != model_role
            or int(aggregate.get("global_step_index", -1)) != global_step_index
        ):
            raise RuntimeError(f"MidSteer checkpoint identity mismatch: {path}")
        if seed in aggregate.get("seeds", []):
            raise RuntimeError(f"Duplicate MidSteer seed {seed} in {path}")
        if set(aggregate.get("sites", {})) != site_names:
            raise RuntimeError(f"MidSteer aggregate site mismatch: {path}")
        for site in sorted(site_names):
            destination = aggregate["sites"][site]
            destination["neutral"] = merge_midsteer_moments(
                destination["neutral"], neutral[site]
            )
            if set(destination.get("pairs", {})) != set(branches):
                raise RuntimeError(f"MidSteer aggregate pair mismatch: {path}")
            for pair_id, sides in branches.items():
                for side in ("source", "target"):
                    destination["pairs"][pair_id][side] = merge_midsteer_moments(
                        destination["pairs"][pair_id][side], sides[side][site]
                    )
        aggregate["seeds"].append(seed)
    else:
        aggregate = {
            "schema_version": 2,
            "control_mode": MIDSTEER_CONTROL_MODE,
            "step_policy": MIDSTEER_STEP_POLICY,
            "model_role": model_role,
            "global_step_index": global_step_index,
            "seeds": [seed],
            "sites": {
                site: {
                    "neutral": neutral[site],
                    "pairs": {
                        pair_id: {
                            "source": sides["source"][site],
                            "target": sides["target"][site],
                        }
                        for pair_id, sides in branches.items()
                    },
                }
                for site in sorted(site_names)
            },
        }
    _atomic_torch_save(path, aggregate)


@torch.inference_mode()
def calibrate_seed(
    row: dict[str, Any],
    spec: dict[str, Any],
    *,
    seed: int,
    moment_root: Path,
) -> tuple[list[tuple[int, str]], int]:
    calibration_row = {**row, "variant": "baseline", "seed": seed}
    config = build_generation_config(calibration_row, spec)
    config["logging"]["output_dir"] = str(
        Path(spec["_root"])
        / row["calibration_dir"]
        / "midsteer_attn_output_work"
        / f"seed_{seed:08d}"
    )
    config["output"].update({"decode": False, "save_latents": False})
    runner = GenerationRunner(config)
    runner.adapter.load()
    generation = runner.config["generation"]
    generator = make_generator(seed, runner.device)
    latents, state = runner.adapter.prepare_initial_latents(
        prompt=row["prompt"],
        batch_size=1,
        generator=generator,
        **_latent_kwargs(generation),
    )
    image_token_count: int | None = None
    if row["model_id"] in {"flux1_dev", "flux2_dev"}:
        if latents.ndim != 3:
            raise ValueError(
                "MidSteer FLUX calibration requires packed [batch,tokens,features] "
                f"latents, observed {tuple(latents.shape)}"
            )
        image_token_count = int(latents.shape[-2])
    timesteps = runner.adapter.set_timesteps(
        int(generation["num_inference_steps"]), latents=latents, state=state
    )
    tree = concept_tree(row)
    pairs = [dict(pair) for pair in tree["pairs"]]
    captured: list[tuple[int, str]] = []

    for step_index, timestep in enumerate(timesteps):
        context = runner.adapter.denoising_step_context(step_index, len(timesteps), state)
        state.extra["_active_denoising_step_context"] = context
        model_role = artifact_model_role(context, state)
        if int(context.local_step_index) == 0:
            neutral = _predict_capture(
                runner,
                latents,
                timestep,
                state,
                str(tree["neutral_concept"]),
                "midsteer_neutral",
                include_covariance=True,
                image_token_count=image_token_count,
            )
            branches: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
            for pair in pairs:
                pair_id = str(pair["id"])
                source = _predict_capture(
                    runner,
                    latents,
                    timestep,
                    state,
                    str(pair["unsafe_concept"]),
                    f"midsteer_{pair_id}__source",
                    include_covariance=False,
                    image_token_count=image_token_count,
                )
                target = _predict_capture(
                    runner,
                    latents,
                    timestep,
                    state,
                    str(pair.get("target_concept", pair["safe_sibling_concept"])),
                    f"midsteer_{pair_id}__target",
                    include_covariance=False,
                    image_token_count=image_token_count,
                )
                branches[pair_id] = {"source": source, "target": target}
            _merge_checkpoint(
                moment_root / f"step_{step_index:05d}.pt",
                seed=seed,
                model_role=model_role,
                global_step_index=step_index,
                neutral=neutral,
                branches=branches,
            )
            captured.append((step_index, model_role))
            del neutral, branches, source, target
            _release_host_memory()

        current = runner.adapter.predict_vector_field(
            latents,
            timestep,
            _condition(runner.adapter, row["prompt"], state, "calibration_current"),
            state,
        )
        result = runner.adapter.scheduler_step(
            current, timestep, latents, state, generator=generator
        )
        latents, state = result.latents, result.state
    return captured, len(timesteps)


def calibrate(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    output = Path(spec["_root"]) / row["calibration_dir"]
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / "midsteer_attn_output.pt"
    if artifact_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing artifact: {artifact_path}")
    seeds = [int(value) for value in spec["calibration"]["reference_seeds"]]
    scratch_id = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}"
    moment_root = output / "work" / f"midsteer_attn_output_moments_{scratch_id}"
    moment_root.mkdir(parents=True, exist_ok=False)
    expected_capture_keys: list[tuple[int, str]] | None = None
    expected_total_steps: int | None = None
    transforms: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            capture_keys, total_steps = calibrate_seed(
                row,
                spec,
                seed=seed,
                moment_root=moment_root,
            )
            if expected_capture_keys is None:
                expected_capture_keys = capture_keys
                expected_total_steps = total_steps
            elif (
                capture_keys != expected_capture_keys
                or total_steps != expected_total_steps
            ):
                raise RuntimeError("MidSteer role/step topology changed across seeds")
            _release_host_memory()

        checkpoints = sorted(moment_root.glob("step_*.pt"))
        if expected_capture_keys is None or len(checkpoints) != len(expected_capture_keys):
            raise RuntimeError(
                f"MidSteer checkpoint count mismatch: {len(checkpoints)} != "
                f"{0 if expected_capture_keys is None else len(expected_capture_keys)}"
            )
        for checkpoint in checkpoints:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload.get("seeds") != seeds:
                raise RuntimeError(
                    f"MidSteer seed coverage mismatch in {checkpoint}: "
                    f"{payload.get('seeds')}"
                )
            model_role = str(payload["model_role"])
            calibration_step_index = int(payload["global_step_index"])
            for site, site_moments in sorted(payload["sites"].items()):
                fitted = fit_midsteer_site_transforms(
                    site_moments["neutral"], site_moments["pairs"]
                )
                for pair_id, values in sorted(fitted.items()):
                    transforms.append(
                        {
                            "model_role": model_role,
                            "calibration_step_index": calibration_step_index,
                            "site": site,
                            "pair_id": pair_id,
                            **values,
                        }
                    )
                del fitted
            del payload
            _release_host_memory()
    finally:
        shutil.rmtree(moment_root, ignore_errors=True)

    metadata = {
        "model_id": row["model_id"],
        "prompt_id": row["prompt_id"],
        "model_config_sha256": row["model_config_sha256"],
        "concept_manifest": row["concept_manifest"],
        "concept_manifest_sha256": hashlib.sha256(
            (Path(spec["_root"]) / row["concept_manifest"]).read_bytes()
        ).hexdigest(),
        "reference_seeds": seeds,
        "moment_accumulation": "post_projection_exact_cpu_float64",
        "feature_group_count": 1,
        "token_scope": MIDSTEER_TOKEN_SCOPE,
        "calibration_steps": [
            {"global_step_index": step, "model_role": role}
            for step, role in (expected_capture_keys or [])
        ],
        "total_generation_steps": expected_total_steps,
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
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    atomic_json(output / "midsteer_attn_output_status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/chatgpt_steering_22_july.yaml",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int)
    args = parser.parse_args()
    index = args.index if args.index is not None else int(os.environ["SLURM_ARRAY_TASK_ID"])
    spec = load_campaign_spec(args.config)
    row = read_row(args.manifest, index)
    output = Path(spec["_root"]) / row["calibration_dir"]
    output.mkdir(parents=True, exist_ok=True)
    try:
        print(json.dumps(calibrate(row, spec), sort_keys=True))
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
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            },
        )
        raise


if __name__ == "__main__":
    main()
