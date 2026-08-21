#!/usr/bin/env python3
"""Prepare a production MidSteer neutral and paired-concept ensemble."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
from pathlib import Path
import random
from typing import Any

import yaml
from datasets import load_dataset


SCHEMA_VERSION = 1
PROTOCOL = "midsteer_official_population_global_paired_prompt_ensemble_v2"
PURPOSE = "production"
RELAION_REPOSITORY = "laion/relaion2B-en-research"
RELAION_DATA_FILE = (
    "part-00000-b31ba513-fc6b-4450-9ba4-a1bba183f408-c000.snappy.parquet"
)
OFFICIAL_NEUTRAL_POPULATION = 50_000
OFFICIAL_CONCEPT_POPULATION = 1_000
GLOBAL_CALIBRATION_SCOPE = "global_prompt_agnostic"


def read_row(path: Path, index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"Manifest {path} has no row {index}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, path)


def neutral_captions(*, population: int, seed: int, cache_dir: Path) -> list[str]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise RuntimeError("HF_TOKEN is required for the official re-LAION dataset")
    dataset = load_dataset(
        RELAION_REPOSITORY,
        cache_dir=str(cache_dir),
        data_files=[RELAION_DATA_FILE],
        columns=["caption"],
        token=token,
    )
    captions: list[str] = []
    for value in dataset["train"]["caption"]:
        if value is None:
            continue
        caption = str(value).strip()
        if not caption:
            continue
        captions.append(caption)
        if len(captions) == population:
            break
    if len(captions) != population:
        raise RuntimeError(
            f"re-LAION supplied {len(captions)} captions, expected {population}"
        )
    random.Random(seed).shuffle(captions)
    if len(set(captions)) != len(captions):
        raise RuntimeError("Production MidSteer neutral captions are not unique")
    return captions


def global_contexts(*, population: int, seed: int) -> list[str]:
    media = [
        "documentary photograph",
        "editorial photograph",
        "catalog photograph",
        "cinematic still",
        "architectural photograph",
        "street photograph",
        "studio photograph",
        "fine-art photograph",
        "photojournalistic image",
        "high-resolution reference image",
    ]
    settings = [
        "a plain neutral studio",
        "an unspecified public interior",
        "an unspecified public exterior",
        "a minimal constructed set",
        "a generic urban setting",
        "a generic rural setting",
        "a simple indoor space",
        "a simple outdoor space",
        "an abstracted real-world setting",
        "a seamless neutral backdrop",
    ]
    lighting = [
        "soft diffuse light",
        "clear frontal light",
        "balanced side light",
        "gentle backlight",
        "neutral overhead light",
        "high-key illumination",
        "low-key illumination",
        "warm natural light",
        "cool natural light",
        "even shadow-free light",
    ]
    viewpoints = [
        "an eye-level viewpoint",
        "a low viewpoint",
        "a high viewpoint",
        "a three-quarter viewpoint",
        "a frontal viewpoint",
        "a profile viewpoint",
        "a wide contextual view",
        "a medium-distance view",
        "a symmetrical view",
        "an asymmetrical view",
    ]
    details = list(itertools.product(media, settings, lighting, viewpoints))
    random.Random(seed).shuffle(details)
    if population > len(details):
        raise ValueError(
            f"Requested {population} contexts but only {len(details)} are available"
        )
    contexts: list[str] = []
    for medium, setting, light, viewpoint in details[:population]:
        contexts.append(
            f"A {medium} in {setting}, under {light}, using {viewpoint}. "
            "Preserve every attribute not explicitly defined below"
        )
    if len(set(contexts)) != population:
        raise RuntimeError("Production MidSteer paired contexts are not unique")
    return contexts


def paired_prompts(
    concept_manifest: dict[str, Any], *, population: int, seed: int
) -> dict[str, dict[str, list[str]]]:
    contexts = global_contexts(population=population, seed=seed)
    result: dict[str, dict[str, list[str]]] = {}
    for pair in concept_manifest.get("pairs", []):
        pair_id = str(pair["id"])
        source = str(pair["unsafe_concept"]).strip()
        target = str(pair.get("target_concept", pair["safe_sibling_concept"])).strip()
        result[pair_id] = {
            "source": [
                f"{context}. Defining source attribute: {source}." for context in contexts
            ],
            "target": [
                f"{context}. Defining target attribute: {target}." for context in contexts
            ],
        }
    if not result:
        raise RuntimeError("Concept manifest contains no MidSteer pairs")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument(
        "--neutral-population", type=int, default=OFFICIAL_NEUTRAL_POPULATION
    )
    parser.add_argument(
        "--concept-population", type=int, default=OFFICIAL_CONCEPT_POPULATION
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.neutral_population != OFFICIAL_NEUTRAL_POPULATION:
        raise ValueError("Production MidSteer requires exactly 50,000 neutral prompts")
    if args.concept_population != OFFICIAL_CONCEPT_POPULATION:
        raise ValueError("Production MidSteer requires exactly 1,000 prompts per side")
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite ensemble: {args.output}")

    row = read_row(args.manifest, args.index)
    if row.get("calibration_scope") != GLOBAL_CALIBRATION_SCOPE:
        raise ValueError("Production MidSteer requires global_prompt_agnostic scope")
    if row.get("evaluation_prompt_used") is not False:
        raise ValueError("Production MidSteer must declare evaluation_prompt_used=false")
    profile_id = row.get("ontology_profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise ValueError("Production MidSteer requires ontology_profile_id")
    if "PROMPT_" in str(args.output) or "/GLOBAL_V2/" not in f"/{args.output}/":
        raise ValueError("Production MidSteer output must live under GLOBAL_V2")
    concept_path = Path(row["concept_manifest"])
    if not str(concept_path).startswith("configs/concepts/global_v2/compiled/"):
        raise ValueError("Production MidSteer requires a compiled global-v2 manifest")
    if sha256(concept_path) != row["concept_manifest_sha256"]:
        raise RuntimeError("Concept manifest hash differs from the sealed matrix row")
    concept_manifest = yaml.safe_load(concept_path.read_text(encoding="utf-8"))
    neutral = neutral_captions(
        population=args.neutral_population,
        seed=args.seed,
        cache_dir=args.cache_dir,
    )
    pairs = paired_prompts(
        concept_manifest,
        population=args.concept_population,
        seed=args.seed,
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "purpose": PURPOSE,
        "model_id": row["model_id"],
        "ontology_profile_id": profile_id,
        "calibration_scope": row["calibration_scope"],
        "evaluation_prompt_used": row["evaluation_prompt_used"],
        "concept_manifest": row["concept_manifest"],
        "concept_manifest_sha256": row["concept_manifest_sha256"],
        "neutral_source": {
            "repository": RELAION_REPOSITORY,
            "data_file": RELAION_DATA_FILE,
            "selection": "first_50000_nonempty_then_python_random_seed_42_shuffle",
        },
        "neutral_aggregation": "all_image_tokens",
        "concept_aggregation": "one_token_average_per_independent_prompt",
        "neutral": neutral,
        "pairs": pairs,
    }
    atomic_json(args.output, payload)
    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(args.output),
                "sha256": sha256(args.output),
                "neutral_population": len(neutral),
                "pair_count": len(pairs),
                "concept_population_per_side": args.concept_population,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
