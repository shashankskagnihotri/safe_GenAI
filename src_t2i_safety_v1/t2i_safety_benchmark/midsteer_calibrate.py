from __future__ import annotations

import argparse
import gc
import json
import os
import re
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import yaml

from hierasafe_flow.adapters.registry import create_adapter
from hierasafe_flow.steering.related_work.midsteer_attn_output import (
    MIDSTEER_CONTROL_MODE,
    MIDSTEER_REVISION,
    MIDSTEER_STEP_POLICY,
    MIDSTEER_TOKEN_SCOPE,
    AttentionOutputMomentCapture,
    MidSteerArtifact,
    fit_midsteer_site_transforms,
    merge_midsteer_moments,
    resolve_transformer_root,
)
from hierasafe_flow.utils.config import load_config
from hierasafe_flow.utils.device import configure_cuda, resolve_device, resolve_dtype
from hierasafe_flow.utils.seed import make_generator, seed_everything

from .contracts import (
    CALIBRATION_ROOT,
    PROJECT_ROOT,
    WORK_ROOT,
    BenchmarkContract,
    atomic_json,
    canonical_sha256,
    file_sha256,
)
from .midsteer_topology import (
    load_midsteer_topology_binding,
    midsteer_image_token_count,
    validate_midsteer_topology,
)


RELAION_REPO_ID = "laion/relaion2B-en-research"
RELAION_REVISION = "cb2173cfd818b41c8370b287dabf93ae85231c42"
RELAION_DATA_FILE = (
    "part-00000-b31ba513-fc6b-4450-9ba4-a1bba183f408-c000.snappy.parquet"
)
NEUTRAL_POPULATION = 50_000
CONCEPT_POPULATION = 1_000
PROMPT_MANIFEST = CALIBRATION_ROOT / "midsteer_prompt_manifest.json"
CONCEPT_SPEC = WORK_ROOT / "method_concepts.yaml"
WORK_MOMENTS_ROOT = CALIBRATION_ROOT / "midsteer_work"
ARTIFACT_ROOT = CALIBRATION_ROOT / "midsteer"


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _atomic_torch_save(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _load_concept_spec() -> dict[str, Any]:
    with CONCEPT_SPEC.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    neutral = value.get("neutral_corpus", {})
    required = {
        "repo_id": RELAION_REPO_ID,
        "revision": RELAION_REVISION,
        "samples": NEUTRAL_POPULATION,
        "aggregation": "all",
    }
    mismatches = {
        key: (neutral.get(key), expected)
        for key, expected in required.items()
        if neutral.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"MidSteer concept specification mismatch: {mismatches}")
    categories = value.get("categories")
    if not isinstance(categories, dict) or len(categories) != 10:
        raise RuntimeError("MidSteer requires exactly ten benchmark categories.")
    for category, sides in categories.items():
        if set(sides) != {"source_terms", "target_terms"}:
            raise RuntimeError(f"MidSteer category {category} has malformed sides.")
        for side in ("source_terms", "target_terms"):
            terms = sides[side]
            if not isinstance(terms, list) or not terms:
                raise RuntimeError(f"MidSteer category {category}/{side} is empty.")
    return value


def _term_pattern(terms: list[str]) -> re.Pattern[str]:
    alternatives = "|".join(
        re.escape(str(term).strip()) for term in terms if str(term).strip()
    )
    if not alternatives:
        raise ValueError("Cannot build an empty ReLAION concept matcher.")
    return re.compile(
        rf"(^|[\s.,\-:;])(?:{alternatives})($|[\s.,\-:;])",
        flags=re.IGNORECASE,
    )


def build_prompt_manifest() -> dict[str, Any]:
    if PROMPT_MANIFEST.exists():
        raise FileExistsError(
            f"Refusing to overwrite sealed MidSteer prompt manifest {PROMPT_MANIFEST}"
        )
    concept_spec = _load_concept_spec()
    try:
        import pyarrow.parquet as parquet
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise RuntimeError(
            "MidSteer prompt sealing requires pinned huggingface_hub and pyarrow."
        ) from exc

    parquet_path = Path(
        hf_hub_download(
            repo_id=RELAION_REPO_ID,
            filename=RELAION_DATA_FILE,
            repo_type="dataset",
            revision=RELAION_REVISION,
            cache_dir=str(WORK_ROOT / "upstream" / "cache" / "relaion"),
            token=os.environ.get("HF_TOKEN"),
        )
    )
    parquet_file = parquet.ParquetFile(parquet_path)
    category_values = concept_spec["categories"]
    patterns = {
        category: {
            "source": _term_pattern(list(values["source_terms"])),
            "target": _term_pattern(list(values["target_terms"])),
        }
        for category, values in category_values.items()
    }
    neutral: list[str] = []
    concepts = {
        category: {"source": [], "target": []}
        for category in category_values
    }
    seen = {
        category: {"source": set(), "target": set()}
        for category in category_values
    }

    complete = False
    for batch in parquet_file.iter_batches(batch_size=65_536, columns=["caption"]):
        for raw_caption in batch.column(0).to_pylist():
            if raw_caption is None:
                continue
            caption = str(raw_caption).strip()
            if not caption:
                continue
            if len(neutral) < NEUTRAL_POPULATION:
                neutral.append(caption)
            for category, sides in patterns.items():
                for side, pattern in sides.items():
                    selected = concepts[category][side]
                    if len(selected) >= CONCEPT_POPULATION:
                        continue
                    if (
                        caption in seen[category][side]
                        or pattern.search(caption) is None
                    ):
                        continue
                    seen[category][side].add(caption)
                    selected.append(caption)
            complete = len(neutral) == NEUTRAL_POPULATION and all(
                len(sides[side]) == CONCEPT_POPULATION
                for sides in concepts.values()
                for side in ("source", "target")
            )
            if complete:
                break
        if complete:
            break

    shortages = {
        f"{category}/{side}": len(sides[side])
        for category, sides in concepts.items()
        for side in ("source", "target")
        if len(sides[side]) != CONCEPT_POPULATION
    }
    if len(neutral) != NEUTRAL_POPULATION:
        shortages["neutral"] = len(neutral)
    if shortages:
        raise RuntimeError(f"ReLAION prompt populations are incomplete: {shortages}")

    payload: dict[str, Any] = {
        "schema_version": 1,
        "protocol": "midsteer_official_relaion_first_file_v1",
        "repo_id": RELAION_REPO_ID,
        "revision": RELAION_REVISION,
        "data_file": RELAION_DATA_FILE,
        "parquet_path": str(parquet_path),
        "parquet_sha256": file_sha256(parquet_path),
        "neutral_population": NEUTRAL_POPULATION,
        "concept_population_per_side": CONCEPT_POPULATION,
        "neutral_aggregation": "all",
        "concept_aggregation": "average",
        "selection": "upstream_order_first_matching_unique_caption",
        "concept_spec_path": str(CONCEPT_SPEC),
        "concept_spec_sha256": file_sha256(CONCEPT_SPEC),
        "neutral": neutral,
        "categories": concepts,
        "created_at": _utc(),
    }
    payload["content_sha256"] = canonical_sha256(
        {key: value for key, value in payload.items() if key != "created_at"}
    )
    atomic_json(PROMPT_MANIFEST, payload)
    return {
        "status": "completed",
        "path": str(PROMPT_MANIFEST),
        "sha256": file_sha256(PROMPT_MANIFEST),
        "content_sha256": payload["content_sha256"],
    }


def _load_prompt_manifest() -> tuple[dict[str, Any], str]:
    encoded = PROMPT_MANIFEST.read_bytes()
    value = json.loads(encoded)
    required = {
        "schema_version": 1,
        "protocol": "midsteer_official_relaion_first_file_v1",
        "repo_id": RELAION_REPO_ID,
        "revision": RELAION_REVISION,
        "data_file": RELAION_DATA_FILE,
        "neutral_population": NEUTRAL_POPULATION,
        "concept_population_per_side": CONCEPT_POPULATION,
        "neutral_aggregation": "all",
        "concept_aggregation": "average",
    }
    mismatches = {
        key: (value.get(key), expected)
        for key, expected in required.items()
        if value.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"MidSteer prompt manifest mismatch: {mismatches}")
    if len(value.get("neutral", [])) != NEUTRAL_POPULATION:
        raise RuntimeError("MidSteer neutral manifest population changed.")
    categories = value.get("categories", {})
    if len(categories) != 10:
        raise RuntimeError("MidSteer category manifest count changed.")
    for category, sides in categories.items():
        if any(len(sides.get(side, [])) != CONCEPT_POPULATION for side in ("source", "target")):
            raise RuntimeError(f"MidSteer category population changed for {category}.")
    return value, file_sha256(PROMPT_MANIFEST)


def _selected_indices(population: int, shard_index: int, shard_count: int) -> list[int]:
    return list(range(shard_index, population, shard_count))


def _validate_shards(shard_index: int, shard_count: int) -> None:
    if not 1 <= shard_count <= CONCEPT_POPULATION:
        raise ValueError(
            f"MidSteer shard count must be in [1,{CONCEPT_POPULATION}]."
        )
    if not 0 <= shard_index < shard_count:
        raise ValueError("MidSteer shard index is outside the shard count.")


def _merge_sites(
    aggregate: dict[str, dict[str, Any]] | None,
    observed: dict[str, dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    if aggregate is None:
        return {
            site: merge_midsteer_moments(None, moments)
            for site, moments in observed.items()
        }
    if set(aggregate) != set(observed):
        raise RuntimeError("MidSteer attention-site topology changed across prompts.")
    return {
        site: merge_midsteer_moments(aggregate[site], observed[site])
        for site in sorted(aggregate)
    }


class CalibrationRuntime:
    def __init__(self, model_id: str) -> None:
        contract = BenchmarkContract()
        model_spec = contract.model(model_id)
        seed_everything(0)
        configure_cuda(True)
        self.device = resolve_device("auto")
        self.dtype = resolve_dtype("bfloat16")
        model_config = load_config(model_spec["config"], project_root=PROJECT_ROOT)
        self.generation = {
            **dict(model_config["generation"]),
            "height": int(model_spec["height"]),
            "width": int(model_spec["width"]),
            "num_inference_steps": int(model_spec["steps"]),
            "num_outputs_per_prompt": 1,
            "guidance_scale": 1.0,
        }
        model_values = dict(model_config["model"])
        model_values["height"] = int(model_spec["height"])
        model_values["width"] = int(model_spec["width"])
        model_values["guidance_scale"] = 1.0
        self.model_id = model_id
        self.model_spec = model_spec
        self.adapter = create_adapter(model_values, self.device, self.dtype)
        self.adapter.load()
        self.root = resolve_transformer_root(self.adapter)
        self.topology = validate_midsteer_topology(
            model_id=model_id,
            adapter=self.adapter,
            root=self.root,
        )

    @torch.inference_mode()
    def capture(
        self,
        prompt: str,
        *,
        include_covariance: bool,
        token_aggregation: str,
        call_role: str,
    ) -> dict[str, dict[str, Any]]:
        generator = make_generator(0, self.device)
        latent_kwargs = {
            key: value
            for key, value in self.generation.items()
            if key not in {"prompt", "prompt_file"}
        }
        latents, state = self.adapter.prepare_initial_latents(
            prompt=prompt,
            batch_size=1,
            generator=generator,
            **latent_kwargs,
        )
        num_steps = int(self.model_spec["steps"])
        state.extra["num_steps"] = num_steps
        state.extra["base_seed"] = 0
        timesteps = self.adapter.set_timesteps(
            num_steps,
            latents=latents,
            state=state,
        )
        if len(timesteps) != num_steps:
            raise RuntimeError("MidSteer calibration received a non-native schedule.")
        context = self.adapter.denoising_step_context(0, len(timesteps), state)
        if int(context.local_step_index) != 0:
            raise RuntimeError("MidSteer calibration did not capture local step zero.")
        state.extra["_active_denoising_step_context"] = context
        prepared = self.adapter.prepare_prompt_for_state(
            prompt,
            state,
            prompt_view="t2isafety_midsteer_independent_prompt",
            call_role=call_role,
        )
        image_token_count = midsteer_image_token_count(self.topology, latents)
        with AttentionOutputMomentCapture(
            self.root,
            include_covariance=include_covariance,
            conditional_only=True,
            image_token_count=image_token_count,
            token_aggregation=token_aggregation,
        ) as capture:
            self.adapter.predict_vector_field(
                latents,
                timesteps[0],
                prepared,
                state,
            )
        return capture.moments()


def _shard_path(model_id: str, shard_index: int, shard_count: int) -> Path:
    return (
        WORK_MOMENTS_ROOT
        / model_id
        / f"shard_{shard_index:04d}_of_{shard_count:04d}.pt"
    )


def generate_shard(model_id: str, shard_index: int, shard_count: int) -> dict[str, Any]:
    _validate_shards(shard_index, shard_count)
    manifest, manifest_sha = _load_prompt_manifest()
    target = _shard_path(model_id, shard_index, shard_count)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite MidSteer moment shard {target}")
    runtime = CalibrationRuntime(model_id)

    neutral: dict[str, dict[str, Any]] | None = None
    neutral_indices = _selected_indices(
        NEUTRAL_POPULATION,
        shard_index,
        shard_count,
    )
    for ordinal, prompt_index in enumerate(neutral_indices):
        observed = runtime.capture(
            str(manifest["neutral"][prompt_index]),
            include_covariance=True,
            token_aggregation="all",
            call_role=f"neutral_{prompt_index:05d}",
        )
        neutral = _merge_sites(neutral, observed)
        del observed
        if (ordinal + 1) % 25 == 0:
            print(
                json.dumps(
                    {
                        "event": "midsteer_neutral_progress",
                        "model_id": model_id,
                        "shard_index": shard_index,
                        "completed": ordinal + 1,
                        "total": len(neutral_indices),
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        gc.collect()

    pairs: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    concept_indices = _selected_indices(
        CONCEPT_POPULATION,
        shard_index,
        shard_count,
    )
    for category, sides in manifest["categories"].items():
        pairs[category] = {}
        for side in ("source", "target"):
            aggregate: dict[str, dict[str, Any]] | None = None
            for ordinal, prompt_index in enumerate(concept_indices):
                observed = runtime.capture(
                    str(sides[side][prompt_index]),
                    include_covariance=False,
                    token_aggregation="average",
                    call_role=f"{category}_{side}_{prompt_index:04d}",
                )
                aggregate = _merge_sites(aggregate, observed)
                del observed
                if (ordinal + 1) % 25 == 0:
                    print(
                        json.dumps(
                            {
                                "event": "midsteer_concept_progress",
                                "model_id": model_id,
                                "category": category,
                                "side": side,
                                "shard_index": shard_index,
                                "completed": ordinal + 1,
                                "total": len(concept_indices),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                gc.collect()
            if aggregate is None:
                raise RuntimeError("MidSteer concept shard selected no prompts.")
            pairs[category][side] = aggregate

    if neutral is None:
        raise RuntimeError("MidSteer neutral shard selected no prompts.")
    payload = {
        "metadata": {
            "schema_version": 2,
            "protocol": "t2isafety_midsteer_exact_sufficient_statistics_v2",
            "model_id": model_id,
            "model_hf_id": runtime.adapter.model_id,
            "model_revision": runtime.adapter.config.get("revision"),
            "adapter": runtime.adapter.adapter_name,
            "prompt_manifest_sha256": manifest_sha,
            "shard_index": shard_index,
            "shard_count": shard_count,
            "neutral_indices": neutral_indices,
            "concept_indices": concept_indices,
            "neutral_aggregation": "all",
            "concept_aggregation": "average",
            "calibration_diffusion_step": 0,
            "calibration_guidance_scale": 1.0,
            "topology_sha256": runtime.topology["topology_sha256"],
            "site_contract_sha256": runtime.topology["site_contract_sha256"],
            "topology_admission_path": runtime.topology[
                "topology_admission_path"
            ],
            "topology_admission_sha256": runtime.topology[
                "topology_admission_sha256"
            ],
            "created_at": _utc(),
        },
        "neutral": neutral,
        "pairs": pairs,
    }
    _atomic_torch_save(target, payload)
    status = {
        "status": "completed",
        **payload["metadata"],
        "path": str(target),
        "sha256": file_sha256(target),
        "site_count": len(neutral),
    }
    atomic_json(target.with_suffix(".json"), status)
    return status


def _load_shard(
    model_id: str,
    shard_index: int,
    shard_count: int,
    manifest_sha: str,
) -> dict[str, Any]:
    path = _shard_path(model_id, shard_index, shard_count)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    metadata = payload.get("metadata", {})
    topology = load_midsteer_topology_binding(model_id)
    required = {
        "schema_version": 2,
        "protocol": "t2isafety_midsteer_exact_sufficient_statistics_v2",
        "model_id": model_id,
        "prompt_manifest_sha256": manifest_sha,
        "shard_index": shard_index,
        "shard_count": shard_count,
        "neutral_indices": _selected_indices(
            NEUTRAL_POPULATION,
            shard_index,
            shard_count,
        ),
        "concept_indices": _selected_indices(
            CONCEPT_POPULATION,
            shard_index,
            shard_count,
        ),
        "neutral_aggregation": "all",
        "concept_aggregation": "average",
        "calibration_diffusion_step": 0,
        "calibration_guidance_scale": 1.0,
        "topology_sha256": topology["topology_sha256"],
        "site_contract_sha256": topology["site_contract_sha256"],
        "topology_admission_path": topology["topology_admission_path"],
        "topology_admission_sha256": topology["topology_admission_sha256"],
    }
    mismatches = {
        key: (metadata.get(key), expected)
        for key, expected in required.items()
        if metadata.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"MidSteer shard identity mismatch in {path}: {mismatches}")
    if not isinstance(payload.get("neutral"), dict) or not isinstance(
        payload.get("pairs"), dict
    ):
        raise RuntimeError(f"Malformed MidSteer shard {path}")
    return payload


def pack_model(model_id: str, shard_count: int) -> dict[str, Any]:
    _validate_shards(0, shard_count)
    manifest, manifest_sha = _load_prompt_manifest()
    final_root = ARTIFACT_ROOT / model_id
    if final_root.exists():
        raise FileExistsError(f"Refusing to overwrite MidSteer model artifacts {final_root}")

    neutral: dict[str, dict[str, Any]] | None = None
    pairs: dict[str, dict[str, dict[str, dict[str, Any]]]] = {}
    shard_hashes: list[dict[str, Any]] = []
    model_identity: dict[str, Any] | None = None
    for shard_index in range(shard_count):
        path = _shard_path(model_id, shard_index, shard_count)
        payload = _load_shard(
            model_id,
            shard_index,
            shard_count,
            manifest_sha,
        )
        metadata = payload["metadata"]
        identity = {
            key: metadata.get(key)
            for key in (
                "model_id",
                "model_hf_id",
                "model_revision",
                "adapter",
                "topology_sha256",
                "site_contract_sha256",
                "topology_admission_path",
                "topology_admission_sha256",
            )
        }
        if model_identity is None:
            model_identity = identity
        elif identity != model_identity:
            raise RuntimeError("MidSteer model identity changed across shards.")
        neutral = _merge_sites(neutral, payload["neutral"])
        for category, sides in payload["pairs"].items():
            destination = pairs.setdefault(category, {})
            for side in ("source", "target"):
                destination[side] = _merge_sites(
                    destination.get(side),
                    sides[side],
                )
        shard_hashes.append(
            {
                "index": shard_index,
                "path": str(path),
                "sha256": file_sha256(path),
            }
        )
    if neutral is None or model_identity is None:
        raise RuntimeError("No MidSteer shards were packed.")
    if set(pairs) != set(manifest["categories"]):
        raise RuntimeError("MidSteer packed category set changed.")
    if any(set(sides) != {"source", "target"} for sides in pairs.values()):
        raise RuntimeError("MidSteer packed source/target topology changed.")
    sites = set(neutral)
    if any(
        set(site_moments) != sites
        for sides in pairs.values()
        for site_moments in sides.values()
    ):
        raise RuntimeError("MidSteer packed attention-site topology changed.")

    all_fitted: dict[str, dict[str, dict[str, torch.Tensor]]] = {}
    for site in sorted(sites):
        all_fitted[site] = fit_midsteer_site_transforms(
            neutral[site],
            {
                category: {
                    side: pairs[category][side][site]
                    for side in ("source", "target")
                }
                for category in sorted(pairs)
            },
        )

    final_root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{model_id}.pack-staging-",
            dir=final_root.parent,
        )
    )
    try:
        metadata = {
            "protocol": "t2isafety_midsteer_exact_affine_v3",
            **model_identity,
            "prompt_manifest_path": str(PROMPT_MANIFEST),
            "prompt_manifest_sha256": manifest_sha,
            "concept_spec_path": str(CONCEPT_SPEC),
            "concept_spec_sha256": file_sha256(CONCEPT_SPEC),
            "neutral_prompt_count": NEUTRAL_POPULATION,
            "concept_prompt_count_per_side": CONCEPT_POPULATION,
            "neutral_aggregation": "all",
            "concept_aggregation": "average",
            "calibration_diffusion_step": 0,
            "calibration_guidance_scale": 1.0,
            "first_diffusion_step_reused": True,
            "intermediate_clipping": False,
            "strength_grid": [1.0, 2.0, 3.0, 4.0, 5.0],
            "shard_count": shard_count,
            "shards": shard_hashes,
            "attention_sites": sorted(sites),
            "created_at": _utc(),
        }
        _atomic_torch_save(
            staging / "neutral_moments.pt",
            {"metadata": metadata, "neutral": neutral},
        )
        artifact_records: list[dict[str, Any]] = []
        for category in sorted(pairs):
            transforms: list[dict[str, Any]] = []
            for site in sorted(sites):
                transforms.append(
                    {
                        "model_role": "default",
                        "site": site,
                        "pair_id": category,
                        **all_fitted[site][category],
                    }
                )
            artifact_path = staging / category / "artifact.pt"
            MidSteerArtifact.save(
                artifact_path,
                metadata={**metadata, "category": category},
                transforms=transforms,
            )
            artifact_sha256 = file_sha256(artifact_path)
            admission_path = staging / category / "ADMISSION.json"
            atomic_json(
                admission_path,
                {
                    "schema_version": 3,
                    "protocol": "t2i_safety_midsteer_artifact_admission_v3",
                    "status": "accepted",
                    "model_id": model_id,
                    "category": category,
                    "variant": "midsteer",
                    "artifact_path": str(
                        final_root / category / "artifact.pt"
                    ),
                    "artifact_sha256": artifact_sha256,
                    "prompt_manifest_sha256": manifest_sha,
                    "concept_spec_sha256": file_sha256(CONCEPT_SPEC),
                    "upstream_revision": MIDSTEER_REVISION,
                    "control_mode": MIDSTEER_CONTROL_MODE,
                    "step_policy": MIDSTEER_STEP_POLICY,
                    "token_scope": MIDSTEER_TOKEN_SCOPE,
                    "topology_sha256": metadata["topology_sha256"],
                    "site_contract_sha256": metadata["site_contract_sha256"],
                    "topology_admission_path": metadata[
                        "topology_admission_path"
                    ],
                    "topology_admission_sha256": metadata[
                        "topology_admission_sha256"
                    ],
                    "intermediate_clipping": False,
                    "transform_count": len(transforms),
                    "attention_sites": sorted(sites),
                    "created_at": _utc(),
                },
            )
            artifact_records.append(
                {
                    "category": category,
                    "path": f"{category}/artifact.pt",
                    "sha256": artifact_sha256,
                    "admission_path": f"{category}/ADMISSION.json",
                    "admission_sha256": file_sha256(admission_path),
                    "transform_count": len(transforms),
                }
            )
        atomic_json(
            staging / "status.json",
            {
                "status": "completed",
                **metadata,
                "artifacts": artifact_records,
                "control_mode": MIDSTEER_CONTROL_MODE,
                "step_policy": MIDSTEER_STEP_POLICY,
                "token_scope": MIDSTEER_TOKEN_SCOPE,
                "upstream_revision": MIDSTEER_REVISION,
            },
        )
        os.replace(staging, final_root)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {
        "status": "completed",
        "model_id": model_id,
        "root": str(final_root),
        "status_sha256": file_sha256(final_root / "status.json"),
        "categories": sorted(pairs),
        "site_count": len(sites),
    }


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Exact category-level MidSteer ReLAION calibration."
    )
    sub = value.add_subparsers(dest="command", required=True)
    sub.add_parser("build-prompts")
    shard = sub.add_parser("shard")
    shard.add_argument("--model", required=True)
    shard.add_argument("--shard-index", required=True, type=int)
    shard.add_argument("--shard-count", required=True, type=int)
    pack = sub.add_parser("pack")
    pack.add_argument("--model", required=True)
    pack.add_argument("--shard-count", required=True, type=int)
    return value


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "build-prompts":
        result = build_prompt_manifest()
    elif args.command == "shard":
        result = generate_shard(args.model, args.shard_index, args.shard_count)
    elif args.command == "pack":
        result = pack_model(args.model, args.shard_count)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
