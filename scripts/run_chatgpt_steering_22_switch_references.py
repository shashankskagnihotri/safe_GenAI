#!/usr/bin/env python3
"""Build strict paired SGF/Safe banks from complete conditioned trajectories."""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import re
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
from hierasafe_flow.utils.seed import make_generator


PRODUCTION_REFERENCE_POPULATION_PER_SIDE = 515
CONSTRUCTION = "full_paired_source_target_conditioned_trajectory"
SCHEMA_VERSION = 2
SIDES = ("source", "target")
GLOBAL_CALIBRATION_SCOPE = "global_prompt_agnostic"
GLOBAL_CALIBRATION_PROMPT = "A neutral prompt-agnostic calibration scene."


def read_row(path: str | Path, index: int) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"Manifest {path} has no row {index}")


def _latent_kwargs(generation: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in generation.items()
        if key not in {"prompt", "prompt_file", "flux_dual_view_conditioning"}
    }


def _condition(adapter: Any, text: str, state: Any, role: str) -> Any:
    return adapter.prepare_prompt_for_state(
        text,
        state,
        prompt_view="switch_full_paired_trajectory_22_july",
        call_role=role,
    )


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


def _atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def _status_filename(
    artifact_name: str,
    *,
    shard_index: int | None,
    shard_count: int | None,
) -> str:
    if shard_index is None or shard_count is None:
        return f"{artifact_name}.status.json"
    return (
        f"{artifact_name}.shard_{shard_index:04d}_of_{shard_count:04d}"
        ".status.json"
    )


def _release_memory() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _reference_identity(row: dict[str, Any]) -> tuple[str, str]:
    if row.get("calibration_scope") != GLOBAL_CALIBRATION_SCOPE:
        return "prompt_id", str(row["prompt_id"])
    profile_id = row.get("ontology_profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("Global switch calibration requires ontology_profile_id")
    if row.get("evaluation_prompt_used") is not False:
        raise ValueError("Global switch calibration must declare evaluation_prompt_used=false")
    if row.get("prompt") != GLOBAL_CALIBRATION_PROMPT:
        raise ValueError("Global switch calibration row contains a non-canonical prompt")
    calibration_dir = str(row.get("calibration_dir", ""))
    if "PROMPT_" in calibration_dir or "/GLOBAL_V2/" not in f"/{calibration_dir}/":
        raise ValueError("Global switch calibration must live under GLOBAL_V2")
    concept_manifest = str(row.get("concept_manifest", ""))
    if not concept_manifest.startswith("configs/concepts/global_v2/compiled/"):
        raise ValueError("Global switch calibration requires a compiled global-v2 manifest")
    return "ontology_profile_id", profile_id


def _sample_plan(
    pairs: list[dict[str, Any]], population: int, seed_start: int
) -> list[dict[str, Any]]:
    if population < len(pairs):
        raise ValueError(
            f"Reference population {population} is smaller than pair count {len(pairs)}"
        )
    quotient, remainder = divmod(population, len(pairs))
    plan: list[dict[str, Any]] = []
    ordinal = 0
    for pair_index, pair in enumerate(pairs):
        count = quotient + int(pair_index < remainder)
        for local_index in range(count):
            plan.append(
                {
                    "pair_id": str(pair["id"]),
                    "source_prompt": str(pair["unsafe_concept"]),
                    "target_prompt": str(
                        pair.get("target_concept", pair["safe_sibling_concept"])
                    ),
                    "pair_sample_index": local_index,
                    "seed": int(seed_start + ordinal),
                }
            )
            ordinal += 1
    if len(plan) != population:
        raise AssertionError(f"Reference plan size {len(plan)} != {population}")
    return plan


@torch.inference_mode()
def generate_side_trajectory(
    runner: GenerationRunner,
    row: dict[str, Any],
    *,
    prompt: str,
    pair_id: str,
    side: str,
    seed: int,
) -> dict[str, torch.Tensor]:
    if side not in SIDES:
        raise ValueError(f"Unknown paired-reference side {side!r}")
    generation = runner.config["generation"]
    generator = make_generator(seed, runner.device)
    latents, state = runner.adapter.prepare_initial_latents(
        prompt=prompt,
        batch_size=1,
        generator=generator,
        **_latent_kwargs(generation),
    )
    timesteps = runner.adapter.set_timesteps(
        int(generation["num_inference_steps"]), latents=latents, state=state
    )
    references: dict[str, torch.Tensor] = {}
    for step_index, timestep in enumerate(timesteps):
        context = runner.adapter.denoising_step_context(
            step_index, len(timesteps), state
        )
        state.extra["_active_denoising_step_context"] = context
        condition = _condition(
            runner.adapter,
            prompt,
            state,
            f"switch_{side}__{pair_id}",
        )
        native = runner.adapter.predict_vector_field(
            latents, timestep, condition, state
        )
        if context.local_step_index == context.local_num_steps - 1:
            model_role = artifact_model_role(context, state)
            if model_role in references:
                raise RuntimeError(
                    f"Duplicate terminal {side} reference for role {model_role!r}"
                )
            references[model_role] = (
                _canonical(
                    runner,
                    row,
                    latents,
                    native,
                    timestep,
                    step_index,
                    f"{pair_id}:full_{side}_trajectory",
                )
                .detach()
                .to(device="cpu", dtype=torch.float32)
            )
        result = runner.adapter.scheduler_step(
            native, timestep, latents, state, generator=generator
        )
        latents, state = result.latents, result.state
    if not references:
        raise RuntimeError(
            f"{side.title()} trajectory {pair_id}/{seed} produced no references"
        )
    return references


def _validate_shard(
    payload: dict[str, Any],
    *,
    row: dict[str, Any],
    item: dict[str, Any],
) -> dict[str, torch.Tensor]:
    identity_key, identity_value = _reference_identity(row)
    required = {
        "schema_version": SCHEMA_VERSION,
        "construction": CONSTRUCTION,
        "model_id": row["model_id"],
        identity_key: identity_value,
        "pair_id": item["pair_id"],
        "seed": item["seed"],
        "source_prompt_sha256": hashlib.sha256(
            item["source_prompt"].encode("utf-8")
        ).hexdigest(),
        "target_prompt_sha256": hashlib.sha256(
            item["target_prompt"].encode("utf-8")
        ).hexdigest(),
    }
    mismatches = {
        key: (payload.get(key), expected)
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Switch-reference shard identity mismatch: {mismatches}")
    references = payload.get("references")
    if not isinstance(references, dict) or set(references) != set(SIDES):
        raise RuntimeError("Switch-reference shard must contain source and target sides")
    expected_roles: set[str] | None = None
    for side in SIDES:
        side_references = references[side]
        if not isinstance(side_references, dict) or not side_references:
            raise RuntimeError(f"Switch-reference shard has no {side} role tensors")
        roles = set(side_references)
        if expected_roles is None:
            expected_roles = roles
        elif roles != expected_roles:
            raise RuntimeError("Paired switch-reference role topologies differ")
        for role, value in side_references.items():
            if (
                not isinstance(role, str)
                or not torch.is_tensor(value)
                or value.ndim < 2
            ):
                raise RuntimeError("Switch-reference shard contains malformed tensors")
            if int(value.shape[0]) != 1 or not torch.isfinite(value).all():
                raise RuntimeError(
                    "Switch-reference shard is non-finite or not singleton"
                )
            other = references["target" if side == "source" else "source"][role]
            if tuple(value.shape) != tuple(other.shape):
                raise RuntimeError("Paired switch-reference tensor shapes differ")
    return references


def build_references(
    row: dict[str, Any],
    spec: dict[str, Any],
    *,
    population: int,
    purpose: str,
    seed_start: int,
    artifact_name: str,
    shard_index: int | None = None,
    shard_count: int | None = None,
    pack_only: bool = False,
) -> dict[str, Any]:
    if (
        purpose == "production"
        and population != PRODUCTION_REFERENCE_POPULATION_PER_SIDE
    ):
        raise ValueError(
            "Production switch calibration requires exactly "
            f"{PRODUCTION_REFERENCE_POPULATION_PER_SIDE} samples per side"
        )
    if purpose not in {"diagnostic", "production"}:
        raise ValueError("purpose must be diagnostic or production")
    identity_key, identity_value = _reference_identity(row)
    if purpose == "production" and identity_key != "ontology_profile_id":
        raise ValueError(
            "Production switch calibration requires a global prompt-agnostic row"
        )
    if Path(artifact_name).name != artifact_name or not artifact_name.endswith(".pt"):
        raise ValueError("artifact-name must be a plain .pt filename")
    if (shard_index is None) != (shard_count is None):
        raise ValueError("shard-index and shard-count must be provided together")
    if shard_count is not None:
        if shard_count <= 0:
            raise ValueError("shard-count must be positive")
        if shard_index is None or not 0 <= shard_index < shard_count:
            raise ValueError("shard-index must lie in [0, shard-count)")
        if pack_only:
            raise ValueError("pack-only cannot be combined with shard arguments")

    tree = concept_tree(row)
    pairs = [dict(pair) for pair in tree["pairs"]]
    plan = _sample_plan(pairs, int(population), int(seed_start))
    output = Path(spec["_root"]) / row["calibration_dir"]
    output.mkdir(parents=True, exist_ok=True)
    artifact_path = output / artifact_name
    if artifact_path.exists():
        raise FileExistsError(f"Refusing to overwrite switch artifact: {artifact_path}")
    work_root = output / "work" / f"switch_{purpose}_{population}_{seed_start}"
    work_root.mkdir(parents=True, exist_ok=True)

    indexed_plan = list(enumerate(plan))
    if shard_count is not None:
        indexed_plan = [
            (plan_index, item)
            for plan_index, item in indexed_plan
            if plan_index % shard_count == shard_index
        ]
        if not indexed_plan:
            raise ValueError("Selected reference shard is empty")

    if not pack_only:
        calibration_row = {**row, "variant": "baseline", "seed": seed_start}
        config = build_generation_config(calibration_row, spec)
        config["logging"]["output_dir"] = str(work_root / "runner")
        config["output"].update({"decode": False, "save_latents": False})
        runner = GenerationRunner(config)
        runner.adapter.load()

        for plan_index, item in indexed_plan:
            pair_dir = work_root / _safe_component(item["pair_id"])
            shard_path = pair_dir / f"seed_{item['seed']:010d}.pt"
            if shard_path.exists():
                payload = torch.load(
                    shard_path, map_location="cpu", weights_only=False
                )
                _validate_shard(payload, row=row, item=item)
                print(
                    json.dumps(
                        {
                            "event": "switch_reference_shard_reused",
                            "index": plan_index,
                            "population": population,
                            "pair_id": item["pair_id"],
                            "seed": item["seed"],
                            "path": str(shard_path),
                        },
                        sort_keys=True,
                    ),
                    flush=True,
                )
                continue
            references = {
                side: generate_side_trajectory(
                    runner,
                    row,
                    prompt=item[f"{side}_prompt"],
                    pair_id=item["pair_id"],
                    side=side,
                    seed=item["seed"],
                )
                for side in SIDES
            }
            payload = {
                "schema_version": SCHEMA_VERSION,
                "construction": CONSTRUCTION,
                "model_id": row["model_id"],
                identity_key: identity_value,
                "pair_id": item["pair_id"],
                "pair_sample_index": item["pair_sample_index"],
                "seed": item["seed"],
                "source_prompt_sha256": hashlib.sha256(
                    item["source_prompt"].encode("utf-8")
                ).hexdigest(),
                "target_prompt_sha256": hashlib.sha256(
                    item["target_prompt"].encode("utf-8")
                ).hexdigest(),
                "references": references,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
            _atomic_torch_save(shard_path, payload)
            print(
                json.dumps(
                    {
                        "event": "switch_reference_shard_completed",
                        "index": plan_index,
                        "population": population,
                        "pair_id": item["pair_id"],
                        "seed": item["seed"],
                        "roles_by_side": {
                            side: sorted(references[side]) for side in SIDES
                        },
                        "path": str(shard_path),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            del references, payload
            _release_memory()

    if shard_count is not None:
        selected_pair_counts: dict[str, int] = defaultdict(int)
        selected_seeds: list[int] = []
        for _, item in indexed_plan:
            selected_pair_counts[item["pair_id"]] += 1
            selected_seeds.append(int(item["seed"]))
        result = {
            "status": "shard_completed",
            "schema_version": SCHEMA_VERSION,
            "model_id": row["model_id"],
            identity_key: identity_value,
            "construction": CONSTRUCTION,
            "purpose": purpose,
            "reference_population_per_side": int(population),
            "reference_population_total": int(2 * population),
            "shard_index": int(shard_index),
            "shard_count": int(shard_count),
            "shard_population": len(indexed_plan),
            "pair_reference_counts": dict(sorted(selected_pair_counts.items())),
            "reference_seeds": selected_seeds,
            "work_root": str(work_root),
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        }
        atomic_json(
            output
            / _status_filename(
                artifact_name,
                shard_index=shard_index,
                shard_count=shard_count,
            ),
            result,
        )
        return result

    packed_lists: dict[
        str, dict[str, dict[str, list[torch.Tensor]]]
    ] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    expected_roles: set[str] | None = None
    pair_counts: dict[str, int] = defaultdict(int)
    seeds: list[int] = []
    for item in plan:
        shard_path = (
            work_root
            / _safe_component(item["pair_id"])
            / f"seed_{item['seed']:010d}.pt"
        )
        payload = torch.load(shard_path, map_location="cpu", weights_only=False)
        references = _validate_shard(payload, row=row, item=item)
        roles = set(references["source"])
        if expected_roles is None:
            expected_roles = roles
        elif roles != expected_roles:
            raise RuntimeError(
                f"Switch reference role topology changed: {roles} != {expected_roles}"
            )
        for side in SIDES:
            if set(references[side]) != roles:
                raise RuntimeError("Packed source/target role topologies differ")
            for role, value in references[side].items():
                packed_lists[role][item["pair_id"]][side].append(value)
        pair_counts[item["pair_id"]] += 1
        seeds.append(int(item["seed"]))

    packed: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    role_populations: dict[str, dict[str, int]] = {}
    for role, role_pairs in packed_lists.items():
        packed[role] = {}
        role_population = {side: 0 for side in SIDES}
        for pair_id, side_values in role_pairs.items():
            packed[role][pair_id] = {}
            for side in SIDES:
                value = torch.cat(side_values[side], dim=0).contiguous()
                if int(value.shape[0]) != pair_counts[pair_id]:
                    raise RuntimeError(
                        f"Packed {side} count mismatch for {role}/{pair_id}"
                    )
                packed[role][pair_id][side] = value
                role_population[side] += int(value.shape[0])
        if any(role_population[side] != population for side in SIDES):
            raise RuntimeError(
                f"Packed role populations {role}/{role_population} != {population}"
            )
        role_populations[role] = role_population

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "model_id": row["model_id"],
        identity_key: identity_value,
        "calibration_scope": row.get("calibration_scope"),
        "evaluation_prompt_used": row.get("evaluation_prompt_used"),
        "model_config_sha256": row["model_config_sha256"],
        "concept_manifest": row["concept_manifest"],
        "concept_manifest_sha256": hashlib.sha256(
            (Path(spec["_root"]) / row["concept_manifest"]).read_bytes()
        ).hexdigest(),
        "construction": CONSTRUCTION,
        "purpose": purpose,
        "reference_population_per_side": int(population),
        "reference_population_total": int(2 * population),
        "reference_population_by_role": role_populations,
        "pair_reference_counts_per_side": dict(sorted(pair_counts.items())),
        "reference_seeds": seeds,
        "reference_dtype": "float32",
        "trajectory_conditioning": (
            "paired_source_or_target_prompt_at_every_denoising_step"
        ),
        "terminal_sample": "canonical_predicted_x0_at_each_model_role_segment_end",
    }
    temporary_artifact = artifact_path.with_name(
        f".{artifact_path.name}.{os.getpid()}.tmp"
    )
    SwitchReferenceArtifact.save(
        temporary_artifact, metadata=metadata, references=packed
    )
    os.replace(temporary_artifact, artifact_path)
    artifact_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    result = {
        "status": "completed",
        **metadata,
        "artifact_path": str(artifact_path),
        "artifact_sha256": artifact_sha256,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    atomic_json(output / f"{artifact_name}.status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/chatgpt_steering_22_july.yaml",
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--population", type=int, required=True)
    parser.add_argument(
        "--purpose", choices=("diagnostic", "production"), required=True
    )
    parser.add_argument("--seed-start", type=int, required=True)
    parser.add_argument("--artifact-name", required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int)
    parser.add_argument("--pack-only", action="store_true")
    args = parser.parse_args()
    spec = load_campaign_spec(args.config)
    row = read_row(args.manifest, args.index)
    output = Path(spec["_root"]) / row["calibration_dir"]
    output.mkdir(parents=True, exist_ok=True)
    try:
        print(
            json.dumps(
                build_references(
                    row,
                    spec,
                    population=args.population,
                    purpose=args.purpose,
                    seed_start=args.seed_start,
                    artifact_name=args.artifact_name,
                    shard_index=args.shard_index,
                    shard_count=args.shard_count,
                    pack_only=args.pack_only,
                ),
                sort_keys=True,
            )
        )
    except Exception as exc:
        atomic_json(
            output
            / _status_filename(
                args.artifact_name,
                shard_index=args.shard_index,
                shard_count=args.shard_count,
            ),
            {
                "status": "failed",
                "model_id": row["model_id"],
                "prompt_id": row["prompt_id"],
                "population": args.population,
                "purpose": args.purpose,
                "artifact_name": args.artifact_name,
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
