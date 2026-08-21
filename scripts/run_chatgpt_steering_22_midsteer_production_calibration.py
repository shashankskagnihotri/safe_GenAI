#!/usr/bin/env python3
"""Shard and pack production MidSteer first-step attention statistics."""

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
    atomic_json,
    build_generation_config,
    load_campaign_spec,
)
from hierasafe_flow.generation.runner import GenerationRunner
from hierasafe_flow.steering.related_work.midsteer_attn_output import (
    MIDSTEER_CONTROL_MODE,
    MIDSTEER_STEP_POLICY,
    MIDSTEER_TOKEN_SCOPE,
    MidSteerArtifact,
    fit_midsteer_site_transforms,
)
from scripts.run_chatgpt_steering_22_midsteer_ensemble_calibration import (
    capture_prompt,
    merge_sites,
    read_row,
    release_memory,
)


SCHEMA_VERSION = 1
PRODUCTION_PROTOCOL = "midsteer_official_population_custom_paired_prompt_ensemble_v1"
OFFICIAL_NEUTRAL_POPULATION = 50_000
OFFICIAL_CONCEPT_POPULATION = 1_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_torch_save(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_ensemble(path: Path, row: dict[str, Any]) -> tuple[dict[str, Any], str]:
    encoded = path.read_bytes()
    ensemble = json.loads(encoded)
    required = {
        "schema_version": 1,
        "protocol": PRODUCTION_PROTOCOL,
        "purpose": "production",
        "model_id": row["model_id"],
        "prompt_id": row["prompt_id"],
        "concept_manifest_sha256": row["concept_manifest_sha256"],
        "neutral_aggregation": "all_image_tokens",
        "concept_aggregation": "one_token_average_per_independent_prompt",
    }
    mismatches = {
        key: (ensemble.get(key), expected)
        for key, expected in required.items()
        if ensemble.get(key) != expected
    }
    if mismatches:
        raise ValueError(f"Production MidSteer ensemble mismatch: {mismatches}")
    neutral = ensemble.get("neutral")
    pairs = ensemble.get("pairs")
    if not isinstance(neutral, list) or len(neutral) != OFFICIAL_NEUTRAL_POPULATION:
        raise ValueError("Production MidSteer requires exactly 50,000 neutral prompts")
    if not isinstance(pairs, dict) or not pairs:
        raise ValueError("Production MidSteer ensemble has no concept pairs")
    for pair_id, sides in pairs.items():
        if set(sides) != {"source", "target"}:
            raise ValueError(f"Production MidSteer pair {pair_id} lacks both sides")
        for side in ("source", "target"):
            if len(sides[side]) != OFFICIAL_CONCEPT_POPULATION:
                raise ValueError(
                    f"Production MidSteer {pair_id}/{side} requires 1,000 prompts"
                )
    return ensemble, hashlib.sha256(encoded).hexdigest()


def shard_path(output: Path, ensemble_sha: str, index: int, count: int) -> Path:
    return (
        output
        / "work"
        / f"midsteer_production_{ensemble_sha[:16]}"
        / "moments"
        / f"shard_{index:04d}_of_{count:04d}.pt"
    )


def shard_status_path(output: Path, index: int, count: int) -> Path:
    return output / f"midsteer_production_shard_{index:04d}_of_{count:04d}.json"


def selected_indices(population: int, index: int, count: int) -> list[int]:
    return list(range(index, population, count))


def validate_shard_arguments(index: int, count: int) -> None:
    if count <= 0:
        raise ValueError("MidSteer shard-count must be positive")
    if not 0 <= index < count:
        raise ValueError("MidSteer shard-index must lie in [0, shard-count)")
    if count > OFFICIAL_CONCEPT_POPULATION:
        raise ValueError("MidSteer shard-count cannot exceed concept population")


def merge_observed(
    aggregate: dict[str, dict[str, Any]] | None,
    observed: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    merged = merge_sites(aggregate, observed)
    if aggregate is not None:
        aggregate.clear()
    return merged


@torch.inference_mode()
def generate_shard(
    row: dict[str, Any],
    spec: dict[str, Any],
    ensemble_path: Path,
    *,
    shard_index: int,
    shard_count: int,
) -> dict[str, Any]:
    validate_shard_arguments(shard_index, shard_count)
    ensemble, ensemble_sha = load_ensemble(ensemble_path, row)
    output = Path(spec["_root"]) / row["calibration_dir"]
    artifact_path = output / "midsteer_attn_output.pt"
    if artifact_path.exists():
        raise FileExistsError(f"Refusing shard generation after artifact exists: {artifact_path}")
    target = shard_path(output, ensemble_sha, shard_index, shard_count)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite MidSteer moment shard: {target}")

    calibration_row = {**row, "variant": "baseline", "seed": 0}
    config = build_generation_config(calibration_row, spec)
    config["logging"]["output_dir"] = str(
        target.parent.parent / "runner" / f"shard_{shard_index:04d}_of_{shard_count:04d}"
    )
    config["output"].update({"decode": False, "save_latents": False})
    runner = GenerationRunner(config)
    runner.adapter.load()
    generation = runner.config["generation"]

    neutral_indices = selected_indices(
        OFFICIAL_NEUTRAL_POPULATION, shard_index, shard_count
    )
    concept_indices = selected_indices(
        OFFICIAL_CONCEPT_POPULATION, shard_index, shard_count
    )
    neutral: dict[str, dict[str, Any]] | None = None
    model_role: str | None = None
    total_steps: int | None = None
    for ordinal, prompt_index in enumerate(neutral_indices):
        observed, observed_role, observed_steps = capture_prompt(
            runner,
            generation,
            row,
            str(ensemble["neutral"][prompt_index]),
            include_covariance=True,
            token_aggregation="all",
            call_role=f"midsteer_prod_neutral_{prompt_index:05d}",
        )
        if model_role is None:
            model_role, total_steps = observed_role, observed_steps
        elif observed_role != model_role or observed_steps != total_steps:
            raise RuntimeError("MidSteer model role/topology changed in neutral shard")
        neutral = merge_observed(neutral, observed)
        del observed
        if (ordinal + 1) % 50 == 0 or ordinal + 1 == len(neutral_indices):
            print(
                json.dumps(
                    {
                        "event": "midsteer_neutral_progress",
                        "shard_index": shard_index,
                        "completed": ordinal + 1,
                        "total": len(neutral_indices),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            release_memory()
    assert neutral is not None and model_role is not None and total_steps is not None

    pair_moments: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    for pair_id, sides in ensemble["pairs"].items():
        pair_moments[str(pair_id)] = {}
        for side in ("source", "target"):
            aggregate: dict[str, dict[str, Any]] | None = None
            for ordinal, prompt_index in enumerate(concept_indices):
                observed, observed_role, observed_steps = capture_prompt(
                    runner,
                    generation,
                    row,
                    str(sides[side][prompt_index]),
                    include_covariance=False,
                    token_aggregation="average",
                    call_role=(
                        f"midsteer_prod_{pair_id}_{side}_{prompt_index:04d}"
                    ),
                )
                if observed_role != model_role or observed_steps != total_steps:
                    raise RuntimeError("MidSteer model role/topology changed in concept shard")
                aggregate = merge_observed(aggregate, observed)
                del observed
                if (ordinal + 1) % 50 == 0 or ordinal + 1 == len(concept_indices):
                    print(
                        json.dumps(
                            {
                                "event": "midsteer_concept_progress",
                                "shard_index": shard_index,
                                "pair_id": pair_id,
                                "side": side,
                                "completed": ordinal + 1,
                                "total": len(concept_indices),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    release_memory()
            assert aggregate is not None
            pair_moments[str(pair_id)][side] = aggregate

    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PRODUCTION_PROTOCOL,
        "model_id": row["model_id"],
        "prompt_id": row["prompt_id"],
        "model_config_sha256": row["model_config_sha256"],
        "concept_manifest_sha256": row["concept_manifest_sha256"],
        "ensemble_sha256": ensemble_sha,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "neutral_indices": neutral_indices,
        "concept_indices": concept_indices,
        "model_role": model_role,
        "total_generation_steps": total_steps,
        "neutral": neutral,
        "pairs": pair_moments,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    atomic_torch_save(target, payload)
    result = {
        "status": "shard_completed",
        "model_id": row["model_id"],
        "prompt_id": row["prompt_id"],
        "ensemble_sha256": ensemble_sha,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "neutral_prompt_count": len(neutral_indices),
        "concept_prompt_count_per_side": len(concept_indices),
        "pair_count": len(pair_moments),
        "moment_shard": str(target),
        "moment_shard_sha256": sha256(target),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    atomic_json(shard_status_path(output, shard_index, shard_count), result)
    return result


def validate_payload(
    payload: dict[str, Any],
    row: dict[str, Any],
    *,
    ensemble_sha: str,
    shard_index: int,
    shard_count: int,
) -> None:
    required = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PRODUCTION_PROTOCOL,
        "model_id": row["model_id"],
        "prompt_id": row["prompt_id"],
        "model_config_sha256": row["model_config_sha256"],
        "concept_manifest_sha256": row["concept_manifest_sha256"],
        "ensemble_sha256": ensemble_sha,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "neutral_indices": selected_indices(
            OFFICIAL_NEUTRAL_POPULATION, shard_index, shard_count
        ),
        "concept_indices": selected_indices(
            OFFICIAL_CONCEPT_POPULATION, shard_index, shard_count
        ),
    }
    mismatches = {
        key: (payload.get(key), expected)
        for key, expected in required.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"MidSteer moment shard identity mismatch: {mismatches}")


def pack_shards(
    row: dict[str, Any],
    spec: dict[str, Any],
    ensemble_path: Path,
    *,
    shard_count: int,
) -> dict[str, Any]:
    validate_shard_arguments(0, shard_count)
    ensemble, ensemble_sha = load_ensemble(ensemble_path, row)
    output = Path(spec["_root"]) / row["calibration_dir"]
    artifact_path = output / "midsteer_attn_output.pt"
    neutral_path = output / "midsteer_neutral_moments.pt"
    if artifact_path.exists() or neutral_path.exists():
        raise FileExistsError("Refusing to overwrite production MidSteer outputs")

    neutral: dict[str, dict[str, Any]] | None = None
    pairs: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    model_role: str | None = None
    total_steps: int | None = None
    for index in range(shard_count):
        path = shard_path(output, ensemble_sha, index, shard_count)
        if not path.exists():
            raise FileNotFoundError(f"Missing production MidSteer shard: {path}")
        payload = torch.load(path, map_location="cpu", weights_only=False)
        validate_payload(
            payload,
            row,
            ensemble_sha=ensemble_sha,
            shard_index=index,
            shard_count=shard_count,
        )
        if model_role is None:
            model_role = str(payload["model_role"])
            total_steps = int(payload["total_generation_steps"])
        elif (
            payload["model_role"] != model_role
            or int(payload["total_generation_steps"]) != total_steps
        ):
            raise RuntimeError("MidSteer role or step topology differs across shards")
        neutral = merge_observed(neutral, payload["neutral"])
        if not pairs:
            pairs = payload["pairs"]
        else:
            if set(pairs) != set(payload["pairs"]):
                raise RuntimeError("MidSteer pair topology differs across shards")
            for pair_id in pairs:
                for side in ("source", "target"):
                    pairs[pair_id][side] = merge_observed(
                        pairs[pair_id][side], payload["pairs"][pair_id][side]
                    )
        del payload
        gc.collect()
    assert neutral is not None and model_role is not None and total_steps is not None

    transforms: list[dict[str, Any]] = []
    for site in sorted(neutral):
        fitted = fit_midsteer_site_transforms(
            neutral[site],
            {
                pair_id: {
                    side: pairs[pair_id][side][site]
                    for side in ("source", "target")
                }
                for pair_id in pairs
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

    metadata = {
        "model_id": row["model_id"],
        "prompt_id": row["prompt_id"],
        "model_config_sha256": row["model_config_sha256"],
        "concept_manifest": row["concept_manifest"],
        "concept_manifest_sha256": row["concept_manifest_sha256"],
        "ensemble_path": str(ensemble_path),
        "ensemble_sha256": ensemble_sha,
        "ensemble_protocol": PRODUCTION_PROTOCOL,
        "purpose": "production",
        "neutral_prompt_population": OFFICIAL_NEUTRAL_POPULATION,
        "neutral_observation_policy": "all_image_tokens",
        "concept_prompt_population_per_side": {
            pair_id: OFFICIAL_CONCEPT_POPULATION for pair_id in pairs
        },
        "concept_observation_policy": "one_token_average_per_independent_prompt",
        "moment_accumulation": "sharded_exact_post_projection_cpu_float64",
        "moment_shard_count": shard_count,
        "feature_group_count": 1,
        "token_scope": MIDSTEER_TOKEN_SCOPE,
        "calibration_steps": [
            {"global_step_index": 0, "model_role": model_role}
        ],
        "total_generation_steps": total_steps,
    }
    atomic_torch_save(
        neutral_path,
        {
            "schema_version": SCHEMA_VERSION,
            "metadata": metadata,
            "neutral": neutral,
        },
    )
    MidSteerArtifact.save(artifact_path, metadata=metadata, transforms=transforms)
    result = {
        "status": "completed",
        **metadata,
        "control_mode": MIDSTEER_CONTROL_MODE,
        "step_policy": MIDSTEER_STEP_POLICY,
        "midsteer_transform_count": len(transforms),
        "artifact_path": str(artifact_path),
        "artifact_sha256": sha256(artifact_path),
        "neutral_moments_path": str(neutral_path),
        "neutral_moments_sha256": sha256(neutral_path),
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    }
    atomic_json(output / "midsteer_attn_output_status.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("shard", "pack"), required=True)
    parser.add_argument(
        "--config", default="configs/experiments/chatgpt_steering_22_july.yaml"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--ensemble", type=Path, required=True)
    parser.add_argument("--shard-index", type=int)
    parser.add_argument("--shard-count", type=int, required=True)
    args = parser.parse_args()
    if args.mode == "shard" and args.shard_index is None:
        raise ValueError("Shard mode requires --shard-index")
    if args.mode == "pack" and args.shard_index is not None:
        raise ValueError("Pack mode does not accept --shard-index")

    spec = load_campaign_spec(args.config)
    row = read_row(args.manifest, args.index)
    output = Path(spec["_root"]) / row["calibration_dir"]
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.mode == "shard":
            result = generate_shard(
                row,
                spec,
                args.ensemble,
                shard_index=int(args.shard_index),
                shard_count=args.shard_count,
            )
        else:
            result = pack_shards(
                row,
                spec,
                args.ensemble,
                shard_count=args.shard_count,
            )
        print(json.dumps(result, sort_keys=True))
    except Exception as exc:
        suffix = (
            f"shard_{args.shard_index:04d}_of_{args.shard_count:04d}"
            if args.mode == "shard" and args.shard_index is not None
            else "pack"
        )
        atomic_json(
            output / f"midsteer_production_{suffix}_failed.json",
            {
                "status": "failed",
                "mode": args.mode,
                "model_id": row["model_id"],
                "prompt_id": row["prompt_id"],
                "shard_index": args.shard_index,
                "shard_count": args.shard_count,
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
