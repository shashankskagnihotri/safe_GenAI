#!/usr/bin/env python3
"""Calibrate one prompt/model for MidSteer, SGF, and Safe Denoiser."""

from __future__ import annotations

import argparse
from collections import defaultdict
import ctypes
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
from hierasafe_flow.steering.canonical import canonicalize_prediction
from hierasafe_flow.steering.related_work.distribution_switch import (
    SwitchReferenceArtifact,
)
from hierasafe_flow.steering.related_work.midsteer_full import (
    AttentionHeadMomentCapture,
    MidSteerArtifact,
    fit_midsteer_transform_from_moments,
    merge_midsteer_moments,
    resolve_transformer_root,
)
from hierasafe_flow.utils.seed import make_generator


_LIBC = ctypes.CDLL(None)


def _release_host_memory() -> None:
    gc.collect()
    malloc_trim = getattr(_LIBC, "malloc_trim", None)
    if malloc_trim is not None:
        malloc_trim(0)


def read_row(path: str | Path, index: int) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"Manifest {path} has no row {index}")


def _condition(adapter: Any, text: str, state: Any, role: str) -> Any:
    return adapter.prepare_prompt_for_state(
        text, state, prompt_view="calibration_22_july", call_role=role
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
) -> tuple[torch.Tensor, dict[str, dict[str, Any]]]:
    condition = _condition(runner.adapter, text, state, role)
    with AttentionHeadMomentCapture(
        resolve_transformer_root(runner.adapter),
        include_covariance=include_covariance,
    ) as capture:
        prediction = runner.adapter.predict_vector_field(
            latents, timestep, condition, state
        )
    return prediction, capture.moments()


def _canonical(
    runner: GenerationRunner,
    row: dict[str, Any],
    latents: torch.Tensor,
    native: torch.Tensor,
    timestep: Any,
    step_index: int,
    branch: str,
) -> torch.Tensor:
    return canonicalize_prediction(
        adapter=runner.adapter,
        model_id=row["model_id"],
        latents=latents,
        native=native,
        timestep=timestep,
        step_index=step_index,
        guidance_scale=float(
            runner.config.get("generation", {}).get("guidance_scale", 1.0)
        ),
        branch=branch,
    ).predicted_x0


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


def _merge_step_checkpoint(
    path: Path,
    *,
    seed: int,
    model_role: str,
    step_index: int,
    neutral: dict[str, dict[str, Any]],
    branches: dict[str, dict[str, dict[str, dict[str, Any]]]],
) -> None:
    site_names = set(neutral)
    if not site_names:
        raise RuntimeError(f"MidSteer captured no neutral sites at step {step_index}")
    for pair_id, sides in branches.items():
        if set(sides) != {"source", "target"}:
            raise RuntimeError(f"Incomplete MidSteer sides for {pair_id} at step {step_index}")
        for side, values in sides.items():
            if set(values) != site_names:
                raise RuntimeError(
                    f"MidSteer site mismatch for {pair_id}/{side} at step {step_index}"
                )

    if path.exists():
        aggregate = torch.load(path, map_location="cpu", weights_only=False)
        if aggregate.get("schema_version") != 1:
            raise RuntimeError(f"Unknown MidSteer checkpoint schema: {path}")
        if aggregate.get("model_role") != model_role or int(
            aggregate.get("step_index", -1)
        ) != step_index:
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
            "schema_version": 1,
            "model_role": model_role,
            "step_index": step_index,
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
) -> tuple[
    dict[str, dict[str, dict[str, list[torch.Tensor]]]],
    int,
]:
    calibration_row = {**row, "variant": "baseline", "seed": seed}
    config = build_generation_config(calibration_row, spec)
    config["logging"]["output_dir"] = str(
        Path(spec["_root"]) / row["calibration_dir"] / "work" / f"seed_{seed:08d}"
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
    timesteps = runner.adapter.set_timesteps(
        int(generation["num_inference_steps"]), latents=latents, state=state
    )
    tree = concept_tree(row)
    pairs = [dict(pair) for pair in tree["pairs"]]
    refs: dict[str, dict[str, dict[str, list[torch.Tensor]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    for step_index, timestep in enumerate(timesteps):
        context = runner.adapter.denoising_step_context(step_index, len(timesteps), state)
        state.extra["_active_denoising_step_context"] = context
        model_role = artifact_model_role(context, state)
        at_segment_end = context.local_step_index == context.local_num_steps - 1
        branch_x0: dict[tuple[str, str], torch.Tensor] = {}
        _, neutral_capture = _predict_capture(
            runner,
            latents,
            timestep,
            state,
            str(tree["neutral_concept"]),
            "neutral",
            include_covariance=True,
        )
        step_branches: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
        for pair in pairs:
            pair_id = str(pair["id"])
            source_native, source_capture = _predict_capture(
                runner,
                latents,
                timestep,
                state,
                str(pair["unsafe_concept"]),
                f"{pair_id}__unsafe",
                include_covariance=False,
            )
            target_native, target_capture = _predict_capture(
                runner,
                latents,
                timestep,
                state,
                str(pair.get("target_concept", pair["safe_sibling_concept"])),
                f"{pair_id}__safe",
                include_covariance=False,
            )
            common = sorted(
                set(neutral_capture) & set(source_capture) & set(target_capture)
            )
            if set(common) != set(neutral_capture):
                raise RuntimeError(f"MidSteer site coverage changed for pair {pair_id}")
            step_branches[pair_id] = {
                "source": source_capture,
                "target": target_capture,
            }
            if at_segment_end:
                branch_x0[(pair_id, "source")] = _canonical(
                    runner,
                    row,
                    latents,
                    source_native,
                    timestep,
                    step_index,
                    f"{pair_id}:source_reference",
                )
                branch_x0[(pair_id, "target")] = _canonical(
                    runner,
                    row,
                    latents,
                    target_native,
                    timestep,
                    step_index,
                    f"{pair_id}:target_reference",
                )
        if at_segment_end:
            for (pair_id, side), value in branch_x0.items():
                refs[model_role][pair_id][side].append(
                    value.detach().to(device="cpu", dtype=torch.float16)
                )
        _merge_step_checkpoint(
            moment_root / f"step_{step_index:05d}.pt",
            seed=seed,
            model_role=model_role,
            step_index=step_index,
            neutral=neutral_capture,
            branches=step_branches,
        )
        del neutral_capture, step_branches
        if pairs:
            del source_capture, target_capture
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
    return refs, len(timesteps)


def _merge_refs(destination: dict, source: dict) -> None:
    for role, pairs in source.items():
        for pair_id, sides in pairs.items():
            for side, values in sides.items():
                destination[role][pair_id][side].extend(values)


def calibrate(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    output = Path(spec["_root"]) / row["calibration_dir"]
    output.mkdir(parents=True, exist_ok=True)
    seeds = [int(value) for value in spec["calibration"]["reference_seeds"]]
    all_refs: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    scratch_id = f"{os.environ.get('SLURM_JOB_ID', 'local')}_{os.getpid()}"
    moment_root = output / "work" / f"midsteer_moments_{scratch_id}"
    moment_root.mkdir(parents=True, exist_ok=False)
    expected_steps: int | None = None
    all_transforms: list[dict[str, Any]] = []
    try:
        for seed in seeds:
            refs, step_count = calibrate_seed(
                row,
                spec,
                seed=seed,
                moment_root=moment_root,
            )
            if expected_steps is None:
                expected_steps = step_count
            elif step_count != expected_steps:
                raise RuntimeError(
                    f"Calibration step count changed across seeds: {step_count} != {expected_steps}"
                )
            _merge_refs(all_refs, refs)
            _release_host_memory()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        checkpoints = sorted(moment_root.glob("step_*.pt"))
        if expected_steps is None or len(checkpoints) != expected_steps:
            raise RuntimeError(
                f"MidSteer checkpoint count mismatch: {len(checkpoints)} != {expected_steps}"
            )
        for checkpoint in checkpoints:
            payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if payload.get("seeds") != seeds:
                raise RuntimeError(
                    f"MidSteer seed coverage mismatch in {checkpoint}: {payload.get('seeds')}"
                )
            model_role = str(payload["model_role"])
            step_index = int(payload["step_index"])
            for site, site_moments in sorted(payload["sites"].items()):
                for pair_id, sides in sorted(site_moments["pairs"].items()):
                    fitted = fit_midsteer_transform_from_moments(
                        site_moments["neutral"],
                        sides["source"],
                        sides["target"],
                    )
                    all_transforms.append(
                        {
                            "model_role": model_role,
                            "step_index": step_index,
                            "site": site,
                            "pair_id": pair_id,
                            **fitted,
                        }
                    )
            del payload
            _release_host_memory()
    finally:
        shutil.rmtree(moment_root, ignore_errors=True)
    packed_refs: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for role, pairs in all_refs.items():
        packed_refs[role] = {}
        for pair_id, sides in pairs.items():
            packed_refs[role][pair_id] = {}
            for side, values in sides.items():
                if len(values) != len(seeds):
                    raise RuntimeError(
                        f"Reference count mismatch {role}/{pair_id}/{side}: "
                        f"{len(values)} != {len(seeds)}"
                    )
                packed_refs[role][pair_id][side] = torch.cat(values, dim=0)
    metadata = {
        "schema_version": 1,
        "model_id": row["model_id"],
        "prompt_id": row["prompt_id"],
        "model_config_sha256": row["model_config_sha256"],
        "concept_manifest": row["concept_manifest"],
        "concept_manifest_sha256": hashlib.sha256(
            (Path(spec["_root"]) / row["concept_manifest"]).read_bytes()
        ).hexdigest(),
        "reference_seeds": seeds,
        "moment_accumulation": "exact_out_of_core_float64",
    }
    MidSteerArtifact.save(
        output / "midsteer_full.pt",
        metadata=metadata,
        transforms=all_transforms,
    )
    SwitchReferenceArtifact.save(
        output / "switch_references.pt",
        metadata=metadata,
        references=packed_refs,
    )
    result = {
        "status": "completed",
        **metadata,
        "midsteer_transform_count": len(all_transforms),
        "reference_roles": sorted(packed_refs),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    atomic_json(output / "calibration_status.json", result)
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
            output / "calibration_status.json",
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
