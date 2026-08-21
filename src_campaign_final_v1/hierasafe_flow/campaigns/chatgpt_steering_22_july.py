"""Corrected seven-variant campaign for the 22 July steering matrix."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any
import hashlib
import json
import math

import torch
import yaml

from ..benchmarks.finer_detailing_correction import _build_hunyuan_conditioning_plan
from ..generation.runner import GenerationRunner
from ..steering.bottleneck import BottleneckTrace, ConceptTrace
from ..steering.canonical import canonicalize_prediction, prediction_from_x0
from ..steering.chs_v2 import chs_v2_x0
from ..steering.conceptsteer_repaired import window_weight
from ..steering.canonical.contract import infer_channel_dim
from ..steering.related_work.distribution_switch_v3 import (
    SwitchReferenceArtifact,
    safe_denoiser_switch_x0,
    sgf_switch_x0,
)
from ..steering.related_work.midsteer_attn_output import (
    MidSteerArtifact,
    MidSteerIntervention,
    resolve_transformer_root,
)
from ..utils.config import deep_merge, load_config
from .chatgpt_steering import (
    atomic_json,
    canonical_json,
    load_negative_prompts,
    media_files,
    negative_prompt_for,
    serialize_result,
    sha_file,
)


VARIANTS = (
    "baseline",
    "native_negative_prompt",
    "current_conceptsteer",
    "hierasafe_chs_v2",
    "midsteer",
    "sgf_switch_adaptation",
    "safe_denoiser_switch_adaptation",
)


# Reproduce the generation trajectories used by the July 17
# finer-detailing pilots.  These values are model-specific: treating every
# image model as a 28-step model changes the scheduler trajectory and is not
# an exact historical control.
HISTORICAL_STEPS_BY_MODEL = {
    "cosmos3_super_text2image": 28,
    "flux1_dev": 28,
    "flux2_dev": 50,
    "ideogram4_nf4": 48,
    "qwen_image": 50,
    "qwen_image_2512": 50,
    "sd35_large": 28,
    "cogvideox_5b": 50,
    "hunyuan_video": 50,
    "joyai_echo": 8,
    "wan22_t2v_a14b": 50,
}


def primary_media_files(directory: str | Path, modality: str) -> list[Path]:
    """Return only final sample media, excluding lossless evidence frames."""

    pattern = "image_*.png" if modality == "t2i" else "video_*.mp4"
    if modality not in {"t2i", "t2v"}:
        raise ValueError(f"Unknown media modality {modality!r}")
    return sorted(Path(directory).glob(f"sample_*/{pattern}"))


def artifact_model_role(context: Any | None, state: Any | None = None) -> str:
    """Disambiguate repeated temporal roles by their segment index."""

    if context is not None:
        role = str(context.model_role)
        segment_index = int(context.segment_index)
    else:
        extra = getattr(state, "extra", {}) if state is not None else {}
        role = str(extra.get("model_role", "primary"))
        segment_index = int(extra.get("segment_index", 0))
    return f"{role}::segment_{segment_index:03d}"

PROMPT_DIRS = {
    "01_sad_young_girl": "PROMPT_01_SAD_YOUNG_GIRL",
    "02_angry_old_man": "PROMPT_02_ANGRY_OLD_MAN",
    "03_empty_outdoor_mall": "PROMPT_03_EMPTY_OUTDOOR_MALL",
}

GLOBAL_MANIFEST_BY_PROMPT = {
    "01_sad_young_girl": "configs/concepts/global_v2/compiled/human_attribute_set.yaml",
    "02_angry_old_man": "configs/concepts/global_v2/compiled/human_attribute_set.yaml",
    "03_empty_outdoor_mall": "configs/concepts/global_v2/compiled/mall_attribute_set.yaml",
}

GLOBAL_CALIBRATION_PROFILE_BY_PROMPT = {
    "01_sad_young_girl": "HUMAN_ATTRIBUTE_SET",
    "02_angry_old_man": "HUMAN_ATTRIBUTE_SET",
    "03_empty_outdoor_mall": "MALL_ATTRIBUTE_SET",
}

GLOBAL_RUNTIME_VARIANTS = {
    "hierasafe_chs_v2",
    "midsteer",
    "sgf_switch_adaptation",
    "safe_denoiser_switch_adaptation",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def resolve_path(path: str | Path, root: Path | None = None) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (root or project_root()) / value


def load_campaign_spec(
    path: str | Path = "configs/experiments/chatgpt_steering_22_july.yaml",
) -> dict[str, Any]:
    target = resolve_path(path)
    value = yaml.safe_load(target.read_text(encoding="utf-8"))
    if tuple(value["campaign"]["variants"]) != VARIANTS:
        raise ValueError("22 July campaign variants differ from the seven-variant contract")
    if int(value["campaign"]["seed"]) != 0:
        raise ValueError("22 July final matrix seed must be zero")
    registry_path = resolve_path(value["campaign"]["ontology_registry"])
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if int(registry.get("schema_version", -1)) != 2:
        raise ValueError("22 July campaign requires global ontology registry schema 2")
    entries = registry.get("pairs")
    if not isinstance(entries, list) or not entries:
        raise ValueError("Global ontology registry has no concept pairs")
    priorities: dict[str, float] = {}
    pair_hashes: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Global ontology registry entries must be mappings")
        pair_id = str(entry.get("id", ""))
        if not pair_id or pair_id in priorities:
            raise ValueError(f"Invalid or duplicate global ontology pair id {pair_id!r}")
        pair_path = resolve_path(str(entry.get("path", "")))
        pair = yaml.safe_load(pair_path.read_text(encoding="utf-8"))
        if str(pair.get("id", "")) != pair_id:
            raise ValueError(
                f"Global ontology registry id/path mismatch for {pair_id!r}"
            )
        priority = float(pair.get("priority"))
        if not math.isfinite(priority) or not 0.0 < priority <= 1.0:
            raise ValueError(
                f"Canonical priority for {pair_id!r} must be finite in (0, 1]"
            )
        priorities[pair_id] = priority
        pair_hashes[pair_id] = sha_file(pair_path)
    active_pairs = value["campaign"].get("active_pairs_by_prompt")
    if not isinstance(active_pairs, dict):
        raise ValueError("22 July campaign lacks active_pairs_by_prompt")
    for prompt_id in PROMPT_DIRS:
        selected = active_pairs.get(prompt_id)
        if (
            not isinstance(selected, list)
            or len(selected) != 5
            or len(set(map(str, selected))) != 5
        ):
            raise ValueError(f"{prompt_id} must declare five unique active pairs")
        unknown = sorted(set(map(str, selected)) - set(priorities))
        if unknown:
            raise ValueError(
                f"{prompt_id} declares pairs absent from the canonical registry: {unknown}"
            )
    value["_canonical_priorities"] = priorities
    value["_ontology_registry_sha256"] = sha_file(registry_path)
    value["_ontology_pair_sha256"] = pair_hashes
    value["_spec_path"] = str(target)
    value["_root"] = str(project_root())
    return value


def load_prompts(spec: dict[str, Any]) -> list[dict[str, Any]]:
    source = yaml.safe_load(
        resolve_path(spec["campaign"]["sealed_prompt_source"]).read_text(encoding="utf-8")
    )
    historical = spec["campaign"]["historical_concepts"]
    pilot = spec["campaign"]["pilot_concepts"]
    result: list[dict[str, Any]] = []
    for item in source["prompts"]:
        prompt_id = str(item["prompt_id"])
        if prompt_id not in PROMPT_DIRS:
            continue
        entry = deepcopy(item)
        entry["prompt_dir"] = PROMPT_DIRS[prompt_id]
        entry["concept_manifest_t2i"] = historical[prompt_id]["t2i"]
        entry["concept_manifest_t2v"] = historical[prompt_id]["t2v"]
        entry["pilot_concept_manifest_t2i"] = pilot[prompt_id]["t2i"]
        entry["pilot_concept_manifest_t2v"] = pilot[prompt_id]["t2v"]
        result.append(entry)
    if [item["prompt_id"] for item in result] != list(PROMPT_DIRS):
        raise ValueError("Prompt source does not contain the three ordered frozen prompts")
    return result


def selected_prompt(prompt: dict[str, Any], model_id: str, modality: str) -> str:
    if modality == "t2i":
        field = "image_prompt"
    elif model_id == "cogvideox_5b":
        field = "video_prompt_cogvideox_5b"
    elif model_id == "hunyuan_video":
        field = "video_prompt_hunyuan_video"
    else:
        field = "video_prompt"
    value = prompt.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Prompt {prompt['prompt_id']} is missing {field}")
    return value


def concept_manifest(prompt: dict[str, Any], modality: str, variant: str) -> str:
    if variant == "current_conceptsteer":
        key = "pilot_concept_manifest_t2v" if modality == "t2v" else "pilot_concept_manifest_t2i"
        return str(prompt[key])
    try:
        return GLOBAL_MANIFEST_BY_PROMPT[str(prompt["prompt_id"])]
    except KeyError as exc:
        raise KeyError(
            f"No global-v2 manifest for prompt {prompt.get('prompt_id')!r}"
        ) from exc


def calibration_dir(
    spec: dict[str, Any],
    prompt: dict[str, Any],
    model_id: str,
    variant: str,
) -> str:
    root = Path(spec["campaign"]["calibration_root"])
    if variant in GLOBAL_RUNTIME_VARIANTS:
        profile = GLOBAL_CALIBRATION_PROFILE_BY_PROMPT[str(prompt["prompt_id"])]
        return str(root / "GLOBAL_V2" / profile / model_id)
    return str(root / prompt["prompt_dir"] / model_id)


def concept_tree(row: dict[str, Any]) -> dict[str, Any]:
    value = yaml.safe_load(resolve_path(row["concept_manifest"]).read_text(encoding="utf-8"))
    pairs = value.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 5:
        raise ValueError(f"{row['concept_manifest']} must contain five historical pairs")
    return value


def build_matrix(spec: dict[str, Any], *, attempt: int = 1) -> list[dict[str, Any]]:
    prompts = load_prompts(spec)
    negatives = load_negative_prompts(spec)
    skeletons: list[dict[str, Any]] = []
    for prompt in prompts:
        for model in spec["models"]:
            for variant in VARIANTS:
                supported = not (
                    variant == "native_negative_prompt"
                    and not bool(model["native_negative_supported"])
                )
                text = selected_prompt(prompt, model["id"], model["modality"])
                manifest = concept_manifest(prompt, model["modality"], variant)
                output = (
                    Path(spec["campaign"]["output_root"])
                    / prompt["prompt_dir"]
                    / model["id"]
                    / variant
                    / f"attempt_{attempt:03d}"
                )
                row = {
                    "campaign_id": spec["campaign"]["id"],
                    "cell_id": (
                        f"{prompt['prompt_id']}__{model['id']}__{variant}"
                    ),
                    "prompt_id": prompt["prompt_id"],
                    "prompt_dir": prompt["prompt_dir"],
                    "model_id": model["id"],
                    "model_config": model["config"],
                    "model_config_sha256": sha_file(resolve_path(model["config"])),
                    "modality": model["modality"],
                    "variant": variant,
                    "execution_status": "runnable" if supported else "unsupported_by_model",
                    "unsupported_reason": (
                        None
                        if supported
                        else "installed native pipeline signature has no negative_prompt"
                    ),
                    "prompt": text,
                    "prompt_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "concept_manifest": manifest,
                    "concept_manifest_sha256": sha_file(resolve_path(manifest)),
                    "ontology_scope": (
                        "prompt_local_historical_control"
                        if variant == "current_conceptsteer"
                        else "global_prompt_agnostic"
                    ),
                    "native_negative_prompt": (
                        negative_prompt_for(
                            prompt["prompt_id"], model["modality"], negatives
                        )
                        if variant == "native_negative_prompt" and supported
                        else None
                    ),
                    "seed": 0,
                    "attempt": attempt,
                    "output_dir": str(output),
                    "calibration_dir": calibration_dir(
                        spec, prompt, str(model["id"]), variant
                    ),
                }
                if variant == "midsteer":
                    artifact_path = resolve_path(
                        Path(row["calibration_dir"]) / "midsteer_attn_output.pt"
                    )
                    if not artifact_path.is_file():
                        raise FileNotFoundError(
                            f"Missing production MidSteer artifact: {artifact_path}"
                        )
                    row["calibration_artifact_sha256"] = sha_file(artifact_path)
                elif variant in {
                    "sgf_switch_adaptation",
                    "safe_denoiser_switch_adaptation",
                }:
                    artifact_name = str(
                        spec["methods"][variant]["reference_artifact"]
                    )
                    if Path(artifact_name).name != artifact_name:
                        raise ValueError(
                            f"{variant} reference_artifact must be a plain filename"
                        )
                    artifact_path = resolve_path(
                        Path(row["calibration_dir"]) / artifact_name
                    )
                    if not artifact_path.is_file():
                        raise FileNotFoundError(
                            f"Missing production switch artifact: {artifact_path}"
                        )
                    row["calibration_artifact_sha256"] = sha_file(artifact_path)
                if model["id"] == "hunyuan_video":
                    row["prompt_entry"] = prompt
                skeletons.append(row)
    matrix_sha = hashlib.sha256(canonical_json(skeletons).encode("utf-8")).hexdigest()
    rows: list[dict[str, Any]] = []
    for row in skeletons:
        row["matrix_sha256"] = matrix_sha
        row["row_sha256"] = hashlib.sha256(
            canonical_json(row).encode("utf-8")
        ).hexdigest()
        rows.append(row)
    if len(rows) != 252:
        raise ValueError(f"Expected 252 matrix cells, built {len(rows)}")
    unsupported = [r for r in rows if r["execution_status"] == "unsupported_by_model"]
    if len(unsupported) != 12:
        raise ValueError(f"Expected 12 unsupported native-negative cells, got {len(unsupported)}")
    return rows


def write_matrix(rows: list[dict[str, Any]], path: str | Path) -> dict[str, Any]:
    target = resolve_path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "rows": len(rows),
        "runnable": sum(r["execution_status"] == "runnable" for r in rows),
        "unsupported_by_model": sum(
            r["execution_status"] == "unsupported_by_model" for r in rows
        ),
        "matrix_sha256": rows[0]["matrix_sha256"],
        "manifest_path": str(target),
        "manifest_file_sha256": sha_file(target),
    }
    atomic_json(target.with_suffix(".summary.json"), summary)
    return summary


def _historical_steering(
    pair_ids: list[str], *, modality: str, model_id: str, lambda_value: float = 1.0
) -> dict[str, Any]:
    if modality not in {"t2i", "t2v"}:
        raise ValueError(f"Unsupported historical-control modality {modality!r}")
    lambda_value = float(lambda_value)
    if not 0.0 < lambda_value < float("inf"):
        raise ValueError(
            f"Historical ConceptSteer lambda must be finite and positive, got {lambda_value!r}"
        )
    return {
        "enabled": True,
        "mode": "bottleneck",
        "start_step": 0,
        "end_step": None,
        "start_fraction": 0.0,
        "end_fraction": 1.0,
        "lambda_schedule": {
            "kind": "constant",
            "max_value": lambda_value,
            "min_value": lambda_value,
        },
        "margin": 0.05,
        "feature_dim": 1,
        "mask": {
            "mode": "max_normalized",
            "threshold": 0.1,
            "percentile": 0.85,
            "eps": 1.0e-6,
            "enabled": False,
        },
        "calibration": {"enabled": False},
        "active_pair_ids": pair_ids,
        "normalize_directions": False,
        # CogVideoX alone used append composition in the July 17 pilots.
        # Every other piloted image/video model used concept_only.
        "prompt_composition": "append" if model_id == "cogvideox_5b" else "concept_only",
        "step_stride": 1,
        # Exact historical control: the successful 17 July pilot scheduled
        # every active pair at every denoising step with no pair windows.
        "pair_overrides": {},
    }


def build_generation_config(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    root = Path(spec["_root"])
    declared_concept_sha = row.get("concept_manifest_sha256")
    if declared_concept_sha is not None:
        observed_concept_sha = sha_file(resolve_path(row["concept_manifest"], root))
        if observed_concept_sha != declared_concept_sha:
            raise RuntimeError(
                f"Concept manifest hash mismatch for {row['cell_id']}: "
                f"{observed_concept_sha} != {declared_concept_sha}"
            )
    smoke = (
        "configs/experiments/smoke_t2v.yaml"
        if row["modality"] == "t2v"
        else "configs/experiments/smoke_t2i.yaml"
    )
    config = deep_merge(
        load_config(resolve_path(smoke, root), project_root=root),
        load_config(resolve_path(row["model_config"], root), project_root=root),
    )
    config.setdefault("_meta", {})["project_root"] = str(root)
    config.setdefault("project", {})["seed"] = int(row["seed"])
    dtype = config.get("model", {}).get("torch_dtype")
    if dtype:
        config.setdefault("runtime", {})["dtype"] = dtype
    generation = config.setdefault("generation", {})
    model_default_num_inference_steps = int(
        generation.get(
            "num_inference_steps",
            spec["generation"]["steps_by_modality"][row["modality"]],
        )
    )
    generation.update(
        {
            "prompt": row["prompt"],
            "prompt_file": None,
            "num_outputs_per_prompt": 1,
            "num_inference_steps": int(
                spec["generation"]["steps_by_modality"][row["modality"]]
            ),
        }
    )
    if row["modality"] == "t2i":
        generation.update(
            {
                "height": int(spec["generation"]["image_height"]),
                "width": int(spec["generation"]["image_width"]),
            }
        )
    else:
        generation.update({"duration_seconds": 15, "num_frames": 240, "fps": 16})
    config.setdefault("logging", {})["output_dir"] = str(resolve_path(row["output_dir"], root))
    config.setdefault("output", {}).update(
        {"save_traces": True, "save_latents": False, "decode": True}
    )
    config["concepts"] = {"hierarchy_path": row["concept_manifest"]}
    tree = concept_tree(row)
    pair_ids = [str(pair["id"]) for pair in tree["pairs"]]
    expected_pair_ids = [
        str(pair_id)
        for pair_id in spec["campaign"]["active_pairs_by_prompt"][row["prompt_id"]]
    ]
    if pair_ids != expected_pair_ids:
        raise ValueError(
            f"Prompt-local pair order differs from the canonical campaign order for "
            f"{row['prompt_id']}: {pair_ids!r} != {expected_pair_ids!r}"
        )
    priority_map = {
        pair_id: float(spec["_canonical_priorities"][pair_id])
        for pair_id in pair_ids
    }
    if row["variant"] == "current_conceptsteer":
        config["steering"] = _historical_steering(
            pair_ids,
            modality=str(row["modality"]),
            model_id=str(row["model_id"]),
            lambda_value=float(row.get("current_conceptsteer_lambda", 1.0)),
        )
        # Preserve each piloted model's exact scheduler length.  LTX-2.3 was
        # not part of the July 17 pilots, so retain its model-native default
        # instead of inventing a historical value.
        generation["num_inference_steps"] = HISTORICAL_STEPS_BY_MODEL.get(
            str(row["model_id"]), model_default_num_inference_steps
        )
    else:
        config["steering"] = {"enabled": False, "mode": "none"}
    config["campaign_22"] = {
        "variant": row["variant"],
        "model_id": row["model_id"],
        "prompt_id": row["prompt_id"],
        "pair_ids": pair_ids,
        "pair_priorities": priority_map,
        "calibration_dir": row["calibration_dir"],
        "concept_manifest": row["concept_manifest"],
        "concept_manifest_sha256": row["concept_manifest_sha256"],
        "ontology_registry": spec["campaign"]["ontology_registry"],
        "ontology_registry_sha256": spec["_ontology_registry_sha256"],
        "ontology_pair_sha256": {
            pair_id: spec["_ontology_pair_sha256"][pair_id]
            for pair_id in pair_ids
        },
        "calibration_artifact_sha256": row.get(
            "calibration_artifact_sha256"
        ),
        "methods": deepcopy(spec["methods"]),
        "spec_path": spec["_spec_path"],
    }
    config["benchmark"] = {
        "condition_id": row["cell_id"],
        "manifest_sha256": row["matrix_sha256"],
        "attempt": int(row["attempt"]),
        "checkpoint_set": config.get("model", {}).get("checkpoint_set"),
        "checkpoint_set_sha256": config.get("model", {}).get("checkpoint_set_sha256"),
        "artifact_manifest": config.get("model", {}).get("artifact_manifest"),
        "artifact_manifest_sha256": config.get("model", {}).get(
            "artifact_manifest_sha256"
        ),
        "segmented_temporal_contract": {
            key: value
            for key, value in config.get("model", {}).items()
            if key.endswith("_temporal_protocol")
        },
    }
    if row["model_id"] == "hunyuan_video":
        negative = negative_prompt_for(
            row["prompt_id"], row["modality"], load_negative_prompts(spec)
        )
        config["model"]["hunyuan_dual_view_conditioning"] = (
            _build_hunyuan_conditioning_plan(
                prompt_id=row["prompt_id"],
                base_prompt=row["prompt"],
                negative_prompt=negative,
                concept_tree_snapshot=tree,
                prompt_entry=row["prompt_entry"],
            )
        )
    if row["variant"] == "native_negative_prompt":
        native = {"prompt": row["native_negative_prompt"]}
        options = spec.get("native_negative_options", {}).get(row["model_id"], {})
        native.update(deepcopy(options))
        requested_scale = row.get("native_negative_guidance_scale")
        if requested_scale is not None:
            if row["model_id"] != "cogvideox_5b":
                raise ValueError(
                    "Per-row native-negative guidance qualification is currently "
                    "defined only for CogVideoX-5b"
                )
            requested_scale = float(requested_scale)
            if not 1.0 < requested_scale <= 20.0:
                raise ValueError(
                    "CogVideoX native-negative guidance scale must lie in (1, 20]"
                )
            config["generation"]["guidance_scale"] = requested_scale
            native["guidance_scale"] = requested_scale
        config["native_negative_prompt"] = native
        config["generation"]["negative_prompt"] = row["native_negative_prompt"]
    return config


def _tensor_stats(value: torch.Tensor) -> dict[str, float]:
    work = value.detach().float()
    return {
        "mean": float(work.mean().item()),
        "std": float(work.std(unbiased=False).item()),
        "min": float(work.min().item()),
        "max": float(work.max().item()),
    }


def _json_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().flatten()[0].item() if value.numel() else None
    if isinstance(value, dict):
        return {str(k): _json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _scalar_timestep(value: Any) -> float:
    if torch.is_tensor(value):
        flattened = value.detach().float().flatten()
        if flattened.numel() == 0:
            raise ValueError("Denoising timestep tensor is empty")
        if not torch.allclose(flattened, flattened[:1]):
            raise ValueError("Campaign switch adaptation requires one shared batch timestep")
        result = float(flattened[0].item())
    else:
        result = float(value)
    if not torch.isfinite(torch.tensor(result)):
        raise ValueError(f"Denoising timestep is non-finite: {result!r}")
    return result


class Campaign22Runner(GenerationRunner):
    """Canonical runner for CHS v2 and independently calibrated related work."""

    def __init__(self, config: dict[str, Any]) -> None:
        safe = deepcopy(config)
        safe["steering"] = {"enabled": False, "mode": "none"}
        super().__init__(safe)
        self.campaign = deepcopy(config["campaign_22"])
        self.variant = str(self.campaign["variant"])
        if self.variant not in {
            "hierasafe_chs_v2",
            "midsteer",
            "sgf_switch_adaptation",
            "safe_denoiser_switch_adaptation",
        }:
            raise ValueError(f"Campaign22Runner cannot execute {self.variant!r}")
        tree = yaml.safe_load(
            resolve_path(config["concepts"]["hierarchy_path"]).read_text(encoding="utf-8")
        )
        all_pairs = [dict(pair) for pair in tree["pairs"]]
        all_pair_ids = [str(pair["id"]) for pair in all_pairs]
        priority_map = self.campaign.get("pair_priorities")
        if not isinstance(priority_map, dict) or set(map(str, priority_map)) != set(
            all_pair_ids
        ):
            raise ValueError(
                "Campaign canonical priority mapping must exactly cover prompt-local pairs"
            )
        self.priority_by_pair = {
            pair_id: float(priority_map[pair_id]) for pair_id in all_pair_ids
        }
        if any(
            not math.isfinite(priority) or not 0.0 < priority <= 1.0
            for priority in self.priority_by_pair.values()
        ):
            raise ValueError("Campaign canonical priorities must be finite in (0, 1]")
        method = self.campaign["methods"][self.variant]
        diagnostic_pair_ids = method.get("diagnostic_pair_ids")
        if diagnostic_pair_ids is None:
            self.pairs = all_pairs
        else:
            diagnostic_variants = {"midsteer", "hierasafe_chs_v2"}
            if (
                self.variant not in diagnostic_variants
                or method.get("diagnostic_only") is not True
            ):
                raise ValueError(
                    "A pair subset is allowed only for an explicitly diagnostic-only "
                    "MidSteer or HieraSafe CHS v2 run"
                )
            if (
                not isinstance(diagnostic_pair_ids, list)
                or not diagnostic_pair_ids
                or len(set(map(str, diagnostic_pair_ids))) != len(diagnostic_pair_ids)
            ):
                raise ValueError(
                    "Diagnostic pair ids must be a non-empty unique list"
                )
            requested = {str(value) for value in diagnostic_pair_ids}
            known = {str(pair["id"]) for pair in all_pairs}
            unknown = sorted(requested - known)
            if unknown:
                raise ValueError(f"Unknown diagnostic pair ids: {unknown}")
            self.pairs = [
                pair for pair in all_pairs if str(pair["id"]) in requested
            ]
            if len(self.pairs) != len(requested):
                raise RuntimeError("Diagnostic pair selection was incomplete")
        self.priorities = [
            self.priority_by_pair[str(pair["id"])] for pair in self.pairs
        ]
        self.neutral = str(tree["neutral_concept"])
        self._conditions: dict[tuple[Any, ...], Any] = {}
        self._midsteer_artifact: MidSteerArtifact | None = None
        self._reference_artifact: SwitchReferenceArtifact | None = None
        self._active_steps = 0
        self._pair_nonzero = {str(pair["id"]): 0 for pair in self.pairs}
        self._source_timestep_by_role: dict[str, float] = {}

    def _condition(self, text: str, state: Any, role: str) -> Any:
        key = (
            id(state),
            state.extra.get("condition_epoch"),
            state.extra.get("model_role"),
            text,
            role,
        )
        if key not in self._conditions:
            self._conditions[key] = self.adapter.prepare_prompt_for_state(
                text,
                state,
                prompt_view="global_ontology_v2",
                call_role=role,
            )
        return self._conditions[key]

    def _native(
        self,
        latents: torch.Tensor,
        timestep: Any,
        state: Any,
        text: str,
        role: str,
    ) -> torch.Tensor:
        return self.adapter.predict_vector_field(
            latents, timestep, self._condition(text, state, role), state
        )

    def _canonical(
        self,
        latents: torch.Tensor,
        native: torch.Tensor,
        timestep: Any,
        step_index: int,
        branch: str,
    ) -> Any:
        return canonicalize_prediction(
            adapter=self.adapter,
            model_id=str(self.campaign["model_id"]),
            latents=latents,
            native=native,
            timestep=timestep,
            step_index=step_index,
            guidance_scale=float(
                self.config.get("generation", {}).get("guidance_scale", 1.0)
            ),
            branch=branch,
        )

    def _midsteer(self) -> MidSteerArtifact:
        if self._midsteer_artifact is None:
            path = (
                resolve_path(self.campaign["calibration_dir"])
                / "midsteer_attn_output.pt"
            )
            if not path.is_file():
                raise FileNotFoundError(f"Missing full MidSteer artifact: {path}")
            self._midsteer_artifact = MidSteerArtifact.load(path)
        return self._midsteer_artifact

    def _references(self) -> SwitchReferenceArtifact:
        if self._reference_artifact is None:
            method = self.campaign["methods"][self.variant]
            artifact_name = str(method["reference_artifact"])
            if Path(artifact_name).name != artifact_name:
                raise ValueError("Switch reference_artifact must be a plain filename")
            path = resolve_path(self.campaign["calibration_dir"]) / artifact_name
            if not path.is_file():
                raise FileNotFoundError(f"Missing SGF/Safe reference artifact: {path}")
            expected_sha256 = self.campaign.get("calibration_artifact_sha256")
            if (
                not isinstance(expected_sha256, str)
                or len(expected_sha256) != 64
                or any(character not in "0123456789abcdef" for character in expected_sha256)
            ):
                raise ValueError(
                    "Switch generation requires a lowercase calibration_artifact_sha256"
                )
            observed_sha256 = sha_file(path)
            if observed_sha256 != expected_sha256:
                raise ValueError(
                    "Switch reference artifact hash mismatch: "
                    f"{observed_sha256} != {expected_sha256}"
                )
            artifact = SwitchReferenceArtifact.load(path)
            if int(artifact.metadata.get("schema_version", -1)) != 2:
                raise ValueError("Switch reference artifact must use paired schema 2")
            expected_construction = str(method["reference_construction"])
            if artifact.metadata.get("construction") != expected_construction:
                raise ValueError(
                    "Switch reference construction mismatch: "
                    f"{artifact.metadata.get('construction')!r} != "
                    f"{expected_construction!r}"
                )
            expected_population = int(method["reference_population_per_side"])
            expected_pair_ids = {str(pair["id"]) for pair in self.pairs}
            observed_populations: dict[str, dict[str, int]] = {}
            observed_pair_counts: dict[str, int] = {}
            for role, pairs in artifact.references.items():
                if set(map(str, pairs)) != expected_pair_ids:
                    raise ValueError(
                        f"Switch reference pair mismatch for role {role!r}"
                    )
                observed = {"source": 0, "target": 0}
                for pair_id, sides in pairs.items():
                    if not isinstance(sides, dict) or set(sides) != {
                        "source",
                        "target",
                    }:
                        raise ValueError(
                            f"Malformed paired switch bank {role}/{pair_id}"
                        )
                    source, target = sides["source"], sides["target"]
                    if (
                        not torch.is_tensor(source)
                        or not torch.is_tensor(target)
                        or source.ndim < 2
                        or target.shape != source.shape
                        or int(source.shape[0]) < 2
                        or not torch.isfinite(source).all()
                        or not torch.isfinite(target).all()
                    ):
                        raise ValueError(
                            f"Malformed paired switch tensors {role}/{pair_id}"
                        )
                    count = int(source.shape[0])
                    previous_count = observed_pair_counts.setdefault(
                        str(pair_id), count
                    )
                    if previous_count != count:
                        raise ValueError(
                            f"Switch pair population differs across roles for {pair_id}"
                        )
                    observed["source"] += count
                    observed["target"] += count
                observed_populations[str(role)] = observed
            if not observed_populations or any(
                side_population != expected_population
                for role_population in observed_populations.values()
                for side_population in role_population.values()
            ):
                raise ValueError(
                    "Switch paired reference population mismatch: "
                    f"observed={observed_populations!r}, "
                    f"expected_each_role_and_side={expected_population}"
                )
            if (
                int(
                    artifact.metadata.get("reference_population_per_side", -1)
                )
                != expected_population
                or int(artifact.metadata.get("reference_population_total", -1))
                != 2 * expected_population
            ):
                raise ValueError("Switch paired reference metadata population mismatch")
            if artifact.metadata.get("pair_reference_counts_per_side") != dict(
                sorted(observed_pair_counts.items())
            ):
                raise ValueError("Switch paired reference metadata pair counts mismatch")
            identity = {
                "model_id": self.campaign["model_id"],
                "prompt_id": self.campaign["prompt_id"],
                "concept_manifest_sha256": self.campaign[
                    "concept_manifest_sha256"
                ],
            }
            mismatches = {
                key: (artifact.metadata.get(key), expected)
                for key, expected in identity.items()
                if artifact.metadata.get(key) != expected
            }
            if mismatches:
                raise ValueError(
                    f"Switch paired reference identity mismatch: {mismatches}"
                )
            self._reference_artifact = artifact
        return self._reference_artifact

    def _trace(
        self,
        *,
        timestep: Any,
        step_index: int,
        enabled: bool,
        base: torch.Tensor,
        steered: torch.Tensor,
        weight: float,
        pair_details: dict[str, dict[str, Any]],
        method_details: dict[str, Any],
    ) -> BottleneckTrace:
        concepts: list[ConceptTrace] = []
        for pair in self.pairs:
            pair_id = str(pair["id"])
            details = _json_value(pair_details.get(pair_id, {}))
            concepts.append(
                ConceptTrace(
                    concept_id=pair_id,
                    parent=str(pair.get("parent", "prompt_local")),
                    lambda_t=float(weight),
                    activation={"enabled": float(enabled)},
                    mask={"coverage": 1.0},
                    unsafe_concept=str(pair["unsafe_concept"]),
                    target_concept=str(
                        pair.get("target_concept", pair["safe_sibling_concept"])
                    ),
                    steering_delta_stats=details,
                )
            )
        return BottleneckTrace(
            step_index=step_index,
            timestep=_json_value(timestep),
            enabled=enabled,
            concepts=concepts,
            base_stats=_tensor_stats(base),
            steered_stats=_tensor_stats(steered),
            segment={
                "campaign_variant": self.variant,
                "method_details": _json_value(method_details),
            },
            condition_calls=[],
            protected_state={"scheduler_step_unchanged": True, "seed": 0, "campaign_variant": self.variant, "method_details": _json_value(method_details)},
        )

    def _record_and_validate(
        self,
        pair_details: dict[str, dict[str, Any]],
        *,
        step_index: int,
        num_steps: int,
    ) -> None:
        self._active_steps += 1
        for pair_id in self._pair_nonzero:
            details = pair_details.get(pair_id)
            if not isinstance(details, dict):
                raise RuntimeError(f"{self.variant} omitted pair evidence for {pair_id}")
            norm = float(
                details.get(
                    "delta_norm",
                    details.get("direction_norm", details.get("intervention_norm", 0.0)),
                )
            )
            if not torch.isfinite(torch.tensor(norm)) or norm <= 0.0:
                raise RuntimeError(
                    f"{self.variant} produced zero/non-finite evidence for {pair_id}"
                )
            self._pair_nonzero[pair_id] += 1
        if step_index == num_steps - 1:
            if self._active_steps <= 0 or any(v <= 0 for v in self._pair_nonzero.values()):
                raise RuntimeError(f"{self.variant} failed complete-run intervention coverage")

    def _validate_complete_if_final(self, step_index: int, num_steps: int) -> None:
        if step_index != num_steps - 1:
            return
        if self._active_steps <= 0 or any(v <= 0 for v in self._pair_nonzero.values()):
            raise RuntimeError(f"{self.variant} failed complete-run intervention coverage")

    def _predict_step(
        self,
        latents: torch.Tensor,
        timestep: Any,
        state: Any,
        prompt: str,
        step_index: int,
        num_steps: int,
    ) -> tuple[torch.Tensor, Any]:
        method = self.campaign["methods"][self.variant]
        progress = step_index / max(num_steps - 1, 1)
        temporal_details: dict[str, Any] = {}
        switch_model_role: str | None = None
        if self.variant == "midsteer":
            # Upstream's default diffusion contract reuses key-0 calibration at
            # constant beta on every denoising step. It does not taper beta.
            weight = 1.0
        elif self.variant in {
            "sgf_switch_adaptation",
            "safe_denoiser_switch_adaptation",
        }:
            context = state.extra.get("_active_denoising_step_context")
            switch_model_role = artifact_model_role(context, state)
            native_timestep = _scalar_timestep(timestep)
            if switch_model_role not in self._source_timestep_by_role:
                if native_timestep <= 0.0:
                    raise ValueError(
                        f"Switch adaptation source timestep must be positive, got {native_timestep}"
                    )
                self._source_timestep_by_role[switch_model_role] = native_timestep
            source_timestep = self._source_timestep_by_role[switch_model_role]
            unified_timestep = 1000.0 * native_timestep / source_timestep
            window_start, window_end = [
                float(item) for item in method["unified_timestep_window"]
            ]
            if window_start < window_end:
                raise ValueError(
                    f"Invalid descending unified timestep window {method['unified_timestep_window']!r}"
                )
            weight = float(
                window_end <= unified_timestep <= window_start
            )
            temporal_details = {
                "temporal_policy": "constant_rectangular_unified_diffusion_time",
                "native_timestep": native_timestep,
                "source_native_timestep": source_timestep,
                "unified_timestep": unified_timestep,
                "unified_timestep_window": [window_start, window_end],
            }
        else:
            window_start, window_end = [float(item) for item in method["window"]]
            if not 0.0 <= window_start <= window_end <= 1.0:
                raise ValueError(f"Invalid normalized CHS window {method['window']!r}")
            temporal_profile = str(method.get("temporal_profile", "sine"))
            if temporal_profile == "sine":
                weight = window_weight(progress, window_start, window_end)
            elif temporal_profile == "rectangular":
                weight = float(window_start <= progress <= window_end)
            else:
                raise ValueError(
                    f"Unsupported CHS temporal profile {temporal_profile!r}"
                )
            temporal_details = {
                "temporal_policy": f"{temporal_profile}_normalized_progress",
                "normalized_progress": progress,
                "normalized_window": [window_start, window_end],
            }
        current_native = self._native(
            latents, timestep, state, prompt, "base_current"
        )
        if weight <= 0.0:
            self._validate_complete_if_final(step_index, num_steps)
            return current_native, self._trace(
                timestep=timestep,
                step_index=step_index,
                enabled=False,
                base=current_native,
                steered=current_native,
                weight=0.0,
                pair_details={},
                method_details={**temporal_details, "window_weight": 0.0},
            )
        pair_ids = [str(pair["id"]) for pair in self.pairs]
        if self.variant == "midsteer":
            context = state.extra.get("_active_denoising_step_context")
            model_role = artifact_model_role(context, state)
            image_token_count: int | None = None
            if self.campaign["model_id"] in {"flux1_dev", "flux2_dev"}:
                if latents.ndim != 3:
                    raise ValueError(
                        "MidSteer FLUX runtime requires packed "
                        "[batch,tokens,features] latents, observed "
                        f"{tuple(latents.shape)}"
                    )
                image_token_count = int(latents.shape[-2])
            with MidSteerIntervention(
                resolve_transformer_root(self.adapter),
                self._midsteer(),
                model_role=model_role,
                step_index=step_index,
                strength=float(method["strength"]) * weight,
                intermediate_clipping=bool(
                    method.get("intermediate_clipping", False)
                ),
                image_token_count=image_token_count,
            ) as intervention:
                steered_native = self._native(
                    latents, timestep, state, prompt, "midsteer_current"
                )
            evidence = intervention.evidence()
            pair_details = {
                pair_id: {
                    "delta_norm": float(
                        evidence["pair_delta_norms"].get(pair_id, 0.0)
                    )
                }
                for pair_id in pair_ids
            }
            self._record_and_validate(
                pair_details, step_index=step_index, num_steps=num_steps
            )
            return steered_native, self._trace(
                timestep=timestep,
                step_index=step_index,
                enabled=True,
                base=current_native,
                steered=steered_native,
                weight=weight,
                pair_details=pair_details,
                method_details={
                    **evidence,
                    "window_weight": weight,
                    "covariance": "full",
                },
            )
        current = self._canonical(
            latents, current_native, timestep, step_index, "current"
        )
        if self.variant == "hierasafe_chs_v2":
            unsafe_x0: list[torch.Tensor] = []
            safe_x0: list[torch.Tensor] = []
            for pair in self.pairs:
                pair_id = str(pair["id"])
                unsafe_native = self._native(
                    latents,
                    timestep,
                    state,
                    str(pair["unsafe_concept"]),
                    f"{pair_id}__unsafe",
                )
                safe_native = self._native(
                    latents,
                    timestep,
                    state,
                    str(pair.get("target_concept", pair["safe_sibling_concept"])),
                    f"{pair_id}__safe",
                )
                unsafe_x0.append(
                    self._canonical(
                        latents, unsafe_native, timestep, step_index, f"{pair_id}:unsafe"
                    ).predicted_x0
                )
                safe_x0.append(
                    self._canonical(
                        latents, safe_native, timestep, step_index, f"{pair_id}:safe"
                    ).predicted_x0
                )
            steered_x0, aggregate = chs_v2_x0(
                current.predicted_x0,
                unsafe_x0,
                safe_x0,
                priorities=self.priorities,
                strength=float(method["strength"]) * weight,
                max_relative_norm=float(method["max_relative_norm"]),
            )
            pair_details = {
                pair_id: {
                    "direction_norm": float(
                        (safe - unsafe).detach().float().norm().item()
                    ),
                    "projected_direction_norm": aggregate[
                        "projected_direction_norms"
                    ][index],
                    "applied_contribution_norm": aggregate[
                        "applied_contribution_norms"
                    ][index],
                    "canonical_priority": self.priorities[index],
                }
                for index, (pair_id, unsafe, safe) in enumerate(
                    zip(pair_ids, unsafe_x0, safe_x0, strict=True)
                )
            }
            method_details = {
                **temporal_details,
                **aggregate,
                "window_weight": weight,
            }
        else:
            context = state.extra.get("_active_denoising_step_context")
            model_role = switch_model_role or artifact_model_role(context, state)
            safe_gate_applied = True
            if self.variant == "sgf_switch_adaptation":
                steered_x0, pair_details = sgf_switch_x0(
                    current.predicted_x0,
                    self._references(),
                    model_role=model_role,
                    pair_ids=pair_ids,
                    strength=float(method["strength"]) * weight,
                    force_policy=str(method["force_policy"]),
                    top_k=int(method["top_k"]),
                    epsilon=float(method["epsilon"]),
                )
            elif self.variant == "safe_denoiser_switch_adaptation":
                steered_x0, pair_details = safe_denoiser_switch_x0(
                    current.predicted_x0,
                    self._references(),
                    model_role=model_role,
                    pair_ids=pair_ids,
                    sigma=float(method["sigma"]),
                    scale=float(method["scale"]) * weight,
                    feature_dim=infer_channel_dim(
                        current.predicted_x0,
                        str(self.campaign["model_id"]),
                        current.layout,
                    ),
                    kernel_distance_policy=str(method["kernel_distance_policy"]),
                    correction_policy=str(method["correction_policy"]),
                    beta_threshold_margin=float(method["beta_threshold_margin"]),
                    official_reference_population=int(
                        method["official_reference_population"]
                    ),
                    threshold_quantile=float(method["threshold_quantile"]),
                )
                safe_gate_applied = any(
                    bool(
                        details.get(
                            "intervention_applied",
                            details.get("is_negation", False),
                        )
                    )
                    for details in pair_details.values()
                )
            else:
                raise AssertionError(self.variant)
            method_details = {
                **temporal_details,
                "window_weight": weight,
                "model_role": model_role,
                "intervention_applied": safe_gate_applied,
                "beta_gate_applied": (
                    None
                    if self.variant == "safe_denoiser_switch_adaptation"
                    else safe_gate_applied
                ),
                "beta_gate_policy": (
                    "disabled_released_sd3_beta_threshold_false"
                    if self.variant == "safe_denoiser_switch_adaptation"
                    else "not_applicable"
                ),
            }
            if not safe_gate_applied:
                return current_native, self._trace(
                    timestep=timestep,
                    step_index=step_index,
                    enabled=False,
                    base=current_native,
                    steered=current_native,
                    weight=0.0,
                    pair_details=pair_details,
                    method_details=method_details,
                )
        baseline_roundtrip_native = prediction_from_x0(
            latents,
            current.predicted_x0,
            current.parameterization,
            current.schedule,
        )
        steered_roundtrip_native = prediction_from_x0(
            latents,
            steered_x0,
            current.parameterization,
            current.schedule,
        )
        current_native_float = current_native.detach().float()
        canonical_roundtrip_residual = (
            baseline_roundtrip_native.detach().float() - current_native_float
        )
        native_delta = (
            steered_roundtrip_native.detach().float()
            - baseline_roundtrip_native.detach().float()
        )
        # Preserve the model's exact native prediction and add only the
        # intervention-induced difference. Replacing the native prediction
        # with a canonical round trip injects BF16 quantization error even when
        # steering strength is zero, and that residual can dominate small
        # paper-scale updates.
        steered_native = current_native_float + native_delta
        native_delta_norm = float(native_delta.norm().item())
        native_reference_norm = float(current_native_float.norm().item())
        canonical_roundtrip_residual_norm = float(
            canonical_roundtrip_residual.norm().item()
        )
        model_dtype_native = steered_native.to(current_native.dtype)
        model_dtype_delta_norm = float(
            (model_dtype_native.detach().float() - current_native_float)
            .norm()
            .item()
        )
        if not torch.isfinite(torch.tensor(native_delta_norm)) or native_delta_norm <= 0.0:
            raise RuntimeError(
                f"{self.variant} produced a zero/non-finite native update after "
                "canonical schedule conversion"
            )
        for details in pair_details.values():
            details["float32_native_total_delta_norm"] = native_delta_norm
            details["model_dtype_cast_native_total_delta_norm"] = model_dtype_delta_norm
            details["native_relative_delta_norm"] = native_delta_norm / max(
                native_reference_norm, 1.0e-12
            )
            details["canonical_roundtrip_residual_norm"] = (
                canonical_roundtrip_residual_norm
            )
            details["native_delta_isolated_from_roundtrip"] = True
        method_details.update(
            {
                "float32_native_total_delta_norm": native_delta_norm,
                "model_dtype_cast_native_total_delta_norm": model_dtype_delta_norm,
                "native_relative_delta_norm": native_delta_norm
                / max(native_reference_norm, 1.0e-12),
                "canonical_roundtrip_residual_norm": (
                    canonical_roundtrip_residual_norm
                ),
                "native_delta_isolated_from_roundtrip": True,
                "native_output_dtype": str(steered_native.dtype),
                "model_native_dtype": str(current_native.dtype),
            }
        )
        self._record_and_validate(
            pair_details, step_index=step_index, num_steps=num_steps
        )
        return steered_native, self._trace(
            timestep=timestep,
            step_index=step_index,
            enabled=True,
            base=current_native,
            steered=steered_native,
            weight=weight,
            pair_details=pair_details,
            method_details=method_details,
        )


def row_modality(config: dict[str, Any]) -> str:
    task = str(config.get("generation", {}).get("task", "text_to_image"))
    return "t2v" if task == "text_to_video" else "t2i"
