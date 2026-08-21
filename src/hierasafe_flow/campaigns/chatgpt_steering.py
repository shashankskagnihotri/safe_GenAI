"""Sealed seven-variant steering campaign for the 21 July runbook."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable
import json
import os

import torch
import yaml

from ..generation.runner import GenerationRunner
from ..steering.bottleneck import BottleneckTrace, ConceptTrace
from ..steering.canonical import canonicalize_prediction, prediction_from_x0
from ..steering.chs_v2 import chs_v2_x0
from ..steering.conceptsteer_repaired import repaired_conceptsteer_x0, window_weight
from ..steering.related_work import (
    MidSteerArtifact,
    MidSteerHook,
    safe_denoiser_switch_x0,
    sgf_switch_x0,
)
from ..steering.related_work.midsteer import resolve_transformer_root
from ..utils.config import deep_merge, load_config


CAMPAIGN_VARIANTS = (
    "baseline",
    "native_negative_prompt",
    "conceptsteer_current_repaired",
    "hierasafe_chs_v2",
    "midsteer",
    "sgf_switch_adapted",
    "safe_denoiser_switch_adapted",
)

STEERING_VARIANTS = CAMPAIGN_VARIANTS[2:]


def _root() -> Path:
    return Path(__file__).resolve().parents[3]


def _resolve(path: str | Path, root: Path | None = None) -> Path:
    value = Path(path)
    return value if value.is_absolute() else (root or _root()) / value


def _sha_bytes(value: bytes) -> str:
    return sha256(value).hexdigest()


def sha_file(path: str | Path) -> str:
    with Path(path).open("rb") as handle:
        digest = sha256()
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def atomic_json(path: str | Path, value: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(target)


def atomic_jsonl(path: str | Path, rows: Iterable[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + f".tmp.{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(canonical_json(row) + "\n")
    temporary.replace(target)


def load_campaign_spec(path: str | Path = "configs/experiments/chatgpt_steering_21_july.yaml") -> dict[str, Any]:
    target = _resolve(path)
    value = yaml.safe_load(target.read_text(encoding="utf-8"))
    if value["campaign"]["seed"] != 0:
        raise ValueError("The sealed campaign seed must remain the repository's canonical seed 0")
    if tuple(value["campaign"]["variants"]) != CAMPAIGN_VARIANTS:
        raise ValueError("Campaign variants differ from the sealed seven-variant contract")
    value["_spec_path"] = str(target)
    value["_root"] = str(_root())
    return value


def load_sealed_prompts(spec: dict[str, Any]) -> list[dict[str, Any]]:
    source_path = _resolve(spec["campaign"]["sealed_prompt_source"], Path(spec["_root"]))
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    pair_map = spec["campaign"]["active_pairs_by_prompt"]
    prompts: list[dict[str, Any]] = []
    for item in source.get("prompts", []):
        prompt_id = item["prompt_id"]
        if prompt_id not in pair_map:
            continue
        prompt = item["prompt"]
        prompts.append(
            {
                "prompt_id": prompt_id,
                "prompt": prompt,
                "prompt_sha256": _sha_bytes(prompt.encode("utf-8")),
                "active_pair_ids": list(pair_map[prompt_id]),
                "concept_manifest": item.get("concept_tree"),
            }
        )
    if [item["prompt_id"] for item in prompts] != list(pair_map):
        raise ValueError("The sealed prompt source does not contain the expected ordered prompt IDs")
    if len(prompts) != 3 or len({item["prompt_sha256"] for item in prompts}) != 3:
        raise ValueError("Exactly three distinct sealed prompts are required")
    return prompts


def load_negative_prompts(spec: dict[str, Any]) -> dict[str, Any]:
    source_path = _resolve(spec["campaign"]["negative_prompt_source"], Path(spec["_root"]))
    source = yaml.safe_load(source_path.read_text(encoding="utf-8"))
    return dict(source["negative_prompts"])


def negative_prompt_for(prompt_id: str, modality: str, negatives: dict[str, Any]) -> str:
    value = negatives[prompt_id]
    if isinstance(value, str):
        return value
    return value["video" if modality == "t2v" else "image"]


def load_ontology(spec: dict[str, Any]) -> dict[str, dict[str, Any]]:
    root = Path(spec["_root"])
    registry_path = _resolve(spec["campaign"]["ontology_registry"], root)
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    pairs: dict[str, dict[str, Any]] = {}
    for item in registry["pairs"]:
        pair_path = _resolve(item["path"], root)
        value = yaml.safe_load(pair_path.read_text(encoding="utf-8"))
        if value["id"] != item["id"]:
            raise ValueError(f"Ontology ID mismatch in {pair_path}")
        value["_path"] = str(pair_path)
        value["_sha256"] = sha_file(pair_path)
        pairs[value["id"]] = value
    expected = {pair for values in spec["campaign"]["active_pairs_by_prompt"].values() for pair in values}
    if set(pairs) != expected or len(pairs) != 10:
        raise ValueError("The global ontology must contain exactly the ten active pair IDs")
    return pairs


def _cell_id(model_id: str, prompt_id: str, variant: str) -> str:
    return f"{model_id}__{prompt_id}__{variant}"


def build_final_matrix(spec: dict[str, Any]) -> list[dict[str, Any]]:
    prompts = load_sealed_prompts(spec)
    negatives = load_negative_prompts(spec)
    rows: list[dict[str, Any]] = []
    for model in spec["models"]:
        config_path = _resolve(model["config"], Path(spec["_root"]))
        for prompt in prompts:
            for variant in CAMPAIGN_VARIANTS:
                if variant == "native_negative_prompt" and not model["native_negative_supported"]:
                    continue
                cell_id = _cell_id(model["id"], prompt["prompt_id"], variant)
                row = {
                    "cell_id": cell_id,
                    "campaign_id": spec["campaign"]["id"],
                    "model_id": model["id"],
                    "model_config": str(Path(model["config"])),
                    "model_config_sha256": sha_file(config_path),
                    "modality": model["modality"],
                    "prompt_id": prompt["prompt_id"],
                    "prompt": prompt["prompt"],
                    "prompt_sha256": prompt["prompt_sha256"],
                    "active_pair_ids": prompt["active_pair_ids"],
                    "concept_manifest": prompt["concept_manifest"],
                    "variant": variant,
                    "seed": 0,
                    "native_negative_prompt": (
                        negative_prompt_for(prompt["prompt_id"], model["modality"], negatives)
                        if variant == "native_negative_prompt"
                        else None
                    ),
                    "native_video": deepcopy(model.get("native_video")),
                    "output_dir": str(Path(spec["campaign"]["output_root"]) / cell_id),
                }
                row["row_sha256"] = _sha_bytes(canonical_json(row).encode("utf-8"))
                rows.append(row)
    if len(rows) != 237:
        raise ValueError(f"Expected 237 supported final cells, constructed {len(rows)}")
    if sum(row["variant"] == "native_negative_prompt" for row in rows) != 21:
        raise ValueError("Expected 21 supported native-negative controls")
    return rows


def freeze_campaign(spec: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(spec["_root"])
    state_dir = _resolve(output_dir or spec["campaign"]["state_root"], root)
    prompts = load_sealed_prompts(spec)
    ontology = load_ontology(spec)
    rows = build_final_matrix(spec)
    frozen = {
        "schema_version": 2,
        "campaign_id": spec["campaign"]["id"],
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "seed": 0,
        "sealed_prompt_hashes": {item["prompt_id"]: item["prompt_sha256"] for item in prompts},
        "spec_sha256": sha_file(spec["_spec_path"]),
        "ontology_hashes": {key: value["_sha256"] for key, value in ontology.items()},
        "model_config_hashes": {row["model_id"]: row["model_config_sha256"] for row in rows},
        "methods": deepcopy(spec["methods"]),
        "upstreams": deepcopy(spec["upstreams"]),
        "matrix_rows": len(rows),
        "matrix_sha256": _sha_bytes("\n".join(canonical_json(row) for row in rows).encode("utf-8")),
        "final_prompts_used_for_tuning": False,
    }
    frozen["freeze_sha256"] = _sha_bytes(canonical_json(frozen).encode("utf-8"))
    state_dir.mkdir(parents=True, exist_ok=True)
    (state_dir / "frozen_hyperparameters.yaml").write_text(
        yaml.safe_dump(frozen, sort_keys=False), encoding="utf-8"
    )
    atomic_jsonl(state_dir / "final_matrix.jsonl", rows)
    atomic_jsonl(state_dir / "final_matrix_baselines.jsonl", [row for row in rows if row["variant"] in CAMPAIGN_VARIANTS[:2]])
    atomic_jsonl(state_dir / "final_matrix_steering.jsonl", [row for row in rows if row["variant"] in STEERING_VARIANTS])
    atomic_json(
        state_dir / "concept_registry_manifest.json",
        {
            "registry": spec["campaign"]["ontology_registry"],
            "pairs": [{"id": key, "path": value["_path"], "sha256": value["_sha256"]} for key, value in ontology.items()],
        },
    )
    atomic_json(
        state_dir / "matrix_summary.json",
        {
            "expected": 237,
            "baseline_stage": 57,
            "steering_stage": 180,
            "unsupported_native_negative_omissions": 15,
            "freeze_sha256": frozen["freeze_sha256"],
        },
    )
    return frozen


def _strip_temporal_protocol(model: dict[str, Any]) -> None:
    for key in list(model):
        if key.endswith("_temporal_protocol"):
            del model[key]


def build_generation_config(row: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    root = Path(spec["_root"])
    smoke_name = "configs/experiments/smoke_t2v.yaml" if row["modality"] == "t2v" else "configs/experiments/smoke_t2i.yaml"
    base = load_config(_resolve(smoke_name, root), project_root=root)
    model = load_config(_resolve(row["model_config"], root), project_root=root)
    config = deep_merge(base, model)
    config.setdefault("project", {})["seed"] = 0
    model_dtype = config.get("model", {}).get("torch_dtype")
    if model_dtype:
        config.setdefault("runtime", {})["dtype"] = model_dtype
    generation = config.setdefault("generation", {})
    generation["prompt"] = row["prompt"]
    generation["prompt_file"] = None
    generation["num_outputs_per_prompt"] = 1
    if row.get("native_video"):
        native = row["native_video"]
        _strip_temporal_protocol(config["model"])
        generation["num_frames"] = int(native["num_frames"])
        generation["fps"] = int(native["fps"])
        generation["duration_seconds"] = float(native["num_frames"]) / float(native["fps"])
    output_dir = _resolve(row["output_dir"], root)
    config.setdefault("logging", {})["output_dir"] = str(output_dir)
    config.setdefault("output", {})["save_traces"] = True
    config["output"]["save_latents"] = False
    config["concepts"] = {"hierarchy_path": spec["campaign"]["compatibility_hierarchy"]}
    config["steering"] = {"enabled": False, "mode": "none"}
    config["campaign"] = {
        "variant": row["variant"],
        "model_id": row["model_id"],
        "prompt_id": row["prompt_id"],
        "active_pair_ids": row["active_pair_ids"],
        "ontology_registry": spec["campaign"]["ontology_registry"],
        "calibration_root": spec["campaign"]["calibration_root"],
        "methods": deepcopy(spec["methods"]),
    }
    return config


def _tensor_stats(value: torch.Tensor) -> dict[str, float]:
    work = value.detach().float()
    return {
        "mean": float(work.mean().cpu()),
        "std": float(work.std(unbiased=False).cpu()),
        "norm": float(work.norm().cpu()),
        "abs_max": float(work.abs().max().cpu()),
    }


def _trace_json_value(value: Any) -> Any:
    """Convert campaign trace metadata without retaining live tensors."""
    if is_dataclass(value):
        return _trace_json_value(asdict(value))
    if isinstance(value, dict):
        return {str(key): _trace_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trace_json_value(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        work = value.detach()
        if work.numel() == 1:
            return work.item()
        return {
            "shape": list(work.shape),
            "dtype": str(work.dtype),
            "stats": _tensor_stats(work),
        }
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            scalar = item()
        except (TypeError, ValueError, RuntimeError):
            pass
        else:
            if isinstance(scalar, (str, int, float, bool)) or scalar is None:
                return scalar
    return str(value)


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


class CampaignGenerationRunner(GenerationRunner):
    """One scheduler-preserving runner for all non-native-negative variants."""

    def __init__(self, config: dict[str, Any]) -> None:
        campaign = deepcopy(config["campaign"])
        safe_config = deepcopy(config)
        safe_config["steering"] = {"enabled": False, "mode": "none"}
        super().__init__(safe_config)
        self.campaign = campaign
        self.campaign_variant = campaign["variant"]
        if self.campaign_variant not in CAMPAIGN_VARIANTS or self.campaign_variant == "native_negative_prompt":
            raise ValueError(f"Campaign runner cannot execute variant {self.campaign_variant!r}")
        spec = load_campaign_spec()
        all_pairs = load_ontology(spec)
        self.active_pairs = [all_pairs[pair_id] for pair_id in campaign["active_pair_ids"]]
        self._campaign_conditions: dict[tuple[int, str, str], Any] = {}
        self._midsteer_artifact: MidSteerArtifact | None = None
        self._prototype_artifact: dict[str, Any] | None = None

    @property
    def _model_id(self) -> str:
        return str(self.campaign["model_id"])

    def _condition(self, text: str, state: Any, role: str) -> Any:
        key = (id(state), text, role)
        if key not in self._campaign_conditions:
            self._campaign_conditions[key] = self.adapter.prepare_prompt_for_state(
                text,
                state,
                prompt_view="campaign_global_v2",
                call_role=role,
            )
        return self._campaign_conditions[key]

    def _native(self, latents: torch.Tensor, timestep: Any, state: Any, text: str, role: str) -> torch.Tensor:
        return self.adapter.predict_vector_field(latents, timestep, self._condition(text, state, role), state)

    def _canonical(
        self,
        latents: torch.Tensor,
        native: torch.Tensor,
        timestep: Any,
        step_index: int,
        branch: str,
    ) -> Any:
        guidance = float(self.config.get("generation", {}).get("guidance_scale", 1.0))
        return canonicalize_prediction(
            adapter=self.adapter,
            model_id=self._model_id,
            latents=latents,
            native=native,
            timestep=timestep,
            step_index=step_index,
            guidance_scale=guidance,
            branch=branch,
        )

    def _artifact_dir(self) -> Path:
        return _resolve(self.campaign["calibration_root"]) / self._model_id

    def _prototypes(self) -> torch.Tensor:
        if self._prototype_artifact is None:
            path = self._artifact_dir() / "clean_prototypes.pt"
            if not path.is_file():
                raise FileNotFoundError(f"Frozen clean-space calibration is missing: {path}")
            self._prototype_artifact = _torch_load(path)
        return self._prototype_artifact["unsafe_prototypes"]

    def _midsteer(self) -> MidSteerArtifact:
        if self._midsteer_artifact is None:
            path = self._artifact_dir() / "midsteer.json"
            if not path.is_file():
                raise FileNotFoundError(f"Frozen MidSteer calibration is missing: {path}")
            self._midsteer_artifact = MidSteerArtifact.load(path)
        return self._midsteer_artifact

    def _trace(
        self,
        *,
        timestep: Any,
        step_index: int,
        enabled: bool,
        base: torch.Tensor,
        steered: torch.Tensor,
        weight: float,
        details: dict[str, Any],
    ) -> BottleneckTrace:
        trace_details = _trace_json_value(details)
        if not isinstance(trace_details, dict):
            raise TypeError("Campaign method details must serialize to a JSON object")
        concepts = [
            ConceptTrace(
                concept_id=pair["id"],
                parent=str(pair.get("parent", "global_v2")),
                lambda_t=float(weight),
                activation={"enabled": float(enabled)},
                mask={"coverage": 1.0},
                unsafe_concept=pair["source"]["label"],
                target_concept=pair["target"]["label"],
                steering_delta_stats=trace_details,
            )
            for pair in self.active_pairs
        ]
        return BottleneckTrace(
            step_index=step_index,
            timestep=_trace_json_value(timestep),
            enabled=enabled,
            concepts=concepts,
            base_stats=_tensor_stats(base),
            steered_stats=_tensor_stats(steered),
            segment={"campaign_variant": self.campaign_variant, "method_details": trace_details},
            condition_calls=[],
            protected_state={"scheduler_step_unchanged": True, "seed": 0},
        )

    def _predict_step(
        self,
        latents: torch.Tensor,
        timestep: Any,
        state: Any,
        prompt: str,
        step_index: int,
        num_steps: int,
    ) -> tuple[torch.Tensor, Any]:
        if self.campaign_variant == "baseline":
            return super()._predict_step(latents, timestep, state, prompt, step_index, num_steps)
        method = self.campaign["methods"][self.campaign_variant]
        progress = step_index / max(num_steps - 1, 1)
        weight = window_weight(progress, float(method["window"][0]), float(method["window"][1]))
        if self.campaign_variant == "midsteer" and weight > 0.0:
            artifact = self._midsteer()
            root = resolve_transformer_root(self.adapter)
            with MidSteerHook(root, artifact, float(method["strength"]) * weight):
                native = self._native(latents, timestep, state, prompt, "current")
            details = {"hook_module": artifact.module_name, "window_weight": weight, "covariance": "diagonal"}
            return native, self._trace(
                timestep=timestep,
                step_index=step_index,
                enabled=True,
                base=native,
                steered=native,
                weight=weight,
                details=details,
            )
        current_native = self._native(latents, timestep, state, prompt, "current")
        if weight <= 0.0:
            return current_native, self._trace(
                timestep=timestep,
                step_index=step_index,
                enabled=False,
                base=current_native,
                steered=current_native,
                weight=0.0,
                details={"window_weight": 0.0},
            )
        current = self._canonical(latents, current_native, timestep, step_index, "current")
        details: dict[str, Any]
        if self.campaign_variant in {"conceptsteer_current_repaired", "hierasafe_chs_v2"}:
            unsafe_x0: list[torch.Tensor] = []
            safe_x0: list[torch.Tensor] = []
            for pair in self.active_pairs:
                unsafe_native = self._native(latents, timestep, state, pair["source"]["prompt"], f"{pair['id']}__unsafe")
                safe_native = self._native(latents, timestep, state, pair["target"]["prompt"], f"{pair['id']}__safe")
                unsafe_x0.append(self._canonical(latents, unsafe_native, timestep, step_index, "unsafe").predicted_x0)
                safe_x0.append(self._canonical(latents, safe_native, timestep, step_index, "safe").predicted_x0)
            if self.campaign_variant == "conceptsteer_current_repaired":
                steered_x0, details = repaired_conceptsteer_x0(
                    current.predicted_x0,
                    unsafe_x0,
                    safe_x0,
                    strength=float(method["strength"]) * weight,
                    max_relative_norm=float(method["max_relative_norm"]),
                )
            else:
                steered_x0, details = chs_v2_x0(
                    current.predicted_x0,
                    unsafe_x0,
                    safe_x0,
                    priorities=[float(pair["priority"]) for pair in self.active_pairs],
                    strength=float(method["strength"]) * weight,
                    max_relative_norm=float(method["max_relative_norm"]),
                )
        elif self.campaign_variant == "sgf_switch_adapted":
            steered_x0, details = sgf_switch_x0(
                current.predicted_x0,
                self._prototypes(),
                model_id=self._model_id,
                layout=current.layout,
                bandwidth=float(method["bandwidth"]),
                strength=float(method["strength"]) * weight,
                max_relative_norm=float(method["max_relative_norm"]),
            )
        elif self.campaign_variant == "safe_denoiser_switch_adapted":
            steered_x0, details = safe_denoiser_switch_x0(
                current.predicted_x0,
                self._prototypes(),
                model_id=self._model_id,
                layout=current.layout,
                bandwidth=float(method["bandwidth"]),
                eta=float(method["eta"]) * weight,
                max_relative_norm=float(method["max_relative_norm"]),
            )
        else:
            raise AssertionError(self.campaign_variant)
        details["window_weight"] = weight
        steered_native = prediction_from_x0(
            latents,
            steered_x0,
            current.parameterization,
            current.schedule,
        )
        return steered_native, self._trace(
            timestep=timestep,
            step_index=step_index,
            enabled=True,
            base=current_native,
            steered=steered_native,
            weight=weight,
            details=details,
        )


def serialize_result(value: Any) -> Any:
    if is_dataclass(value):
        return serialize_result(asdict(value))
    if isinstance(value, dict):
        return {str(key): serialize_result(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [serialize_result(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return {"shape": list(value.shape), "dtype": str(value.dtype)}
    if hasattr(value, "__dict__"):
        return serialize_result(vars(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def media_files(directory: str | Path) -> list[Path]:
    extensions = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".webm", ".gif"}
    return sorted(path for path in Path(directory).rglob("*") if path.is_file() and path.suffix.lower() in extensions)
