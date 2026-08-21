from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from hierasafe_flow.adapters.base import PromptCondition
from hierasafe_flow.steering.bottleneck import (
    BottleneckConfig,
    HierarchicalVectorFieldBottleneck,
)
from hierasafe_flow.steering.canonical.contract import (
    NativeParameterization,
    canonicalize_prediction,
    parameterization_for_model,
)
from hierasafe_flow.steering.concept_graph import ConceptHierarchy
from hierasafe_flow.steering.related_work.midsteer_attn_output import (
    MIDSTEER_REVISION,
    MidSteerArtifact,
    MidSteerIntervention,
    resolve_transformer_root,
)

from .contracts import CALIBRATION_ROOT, file_sha256
from .pilot_context import (
    MethodPilotCandidate,
    assert_runtime_timesteps,
    load_scheduler_grid_binding,
)
from .midsteer_topology import (
    midsteer_image_token_count,
    validate_midsteer_topology,
)


MIDSTEER_STRENGTHS = {
    "cosmos3_super_text2image": 2.0,
    "flux1_dev": 2.0,
    "flux2_dev": 2.0,
    "ideogram4_nf4": 2.0,
    "qwen_image": 2.0,
    "qwen_image_2512": 2.0,
    "sd35_large": 2.0,
}

SAFE_DENOISER_DEFAULTS = {
    "cosmos3_super_text2image": {"sigma": 3.15, "scale": 0.33},
    "flux1_dev": {"sigma": 3.15, "scale": 0.33},
    "flux2_dev": {"sigma": 3.15, "scale": 0.33},
    "ideogram4_nf4": {"sigma": 3.15, "scale": 0.33},
    "qwen_image": {"sigma": 3.15, "scale": 0.33},
    "qwen_image_2512": {"sigma": 3.15, "scale": 0.33},
    "sd35_large": {"sigma": 3.15, "scale": 0.33},
}
SGF_DEFAULT_STRENGTH = 0.03
SGF_ACTIVE_FRACTION = 0.20
SAFE_DENOISER_ACTIVE_FRACTION = 0.22
SAFE_DENOISER_BETA_MARGIN = 1.6


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().contiguous().to(device="cpu")
    return tensor.view(torch.uint8).numpy().tobytes()


def tensor_sha256(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(_tensor_bytes(value))
    return digest.hexdigest()


def condition_sha256(condition: PromptCondition, *, negative_only: bool = False) -> str:
    digest = hashlib.sha256()
    digest.update(condition.prompt.encode("utf-8"))
    keys = sorted(condition.data)
    if negative_only:
        keys = [key for key in keys if "negative" in key]
    if not keys:
        raise RuntimeError("Condition fingerprint has no selected fields.")
    for key in keys:
        value = condition.data[key]
        digest.update(key.encode("utf-8"))
        if isinstance(value, torch.Tensor):
            digest.update(tensor_sha256(value).encode("ascii"))
        elif value is None:
            digest.update(b"null")
        else:
            digest.update(repr(value).encode("utf-8"))
    return digest.hexdigest()


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    work = value.detach().float()
    finite = torch.isfinite(work)
    if not bool(finite.all()):
        raise RuntimeError("A method produced a non-finite tensor.")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "mean": float(work.mean().item()),
        "std": float(work.std(unbiased=False).item()),
        "min": float(work.min().item()),
        "max": float(work.max().item()),
        "norm": float(torch.linalg.vector_norm(work).item()),
    }


class NativeNegativeMethod:
    SUPPORTED = {"flux1_dev", "qwen_image", "qwen_image_2512", "sd35_large"}

    def __init__(
        self,
        *,
        adapter: Any,
        model_id: str,
        prompt: str,
        negative_prompt: str,
        true_cfg_scale: float,
    ) -> None:
        if model_id not in self.SUPPORTED:
            raise RuntimeError(f"Native negative prompt is unsupported for {model_id}.")
        if not negative_prompt.strip():
            raise ValueError("Native negative prompt must be non-empty.")
        self.adapter = adapter
        self.model_id = model_id
        self.true_cfg_scale = float(true_cfg_scale)
        self.negative_prompt = negative_prompt
        if self.true_cfg_scale <= 1.0:
            raise ValueError("Native true CFG scale must be greater than one.")

        if model_id == "sd35_large":
            self.condition = self._sd35_condition(prompt, negative_prompt)
            empty = self._sd35_condition(prompt, "")
            configured_hash = condition_sha256(self.condition, negative_only=True)
            empty_hash = condition_sha256(empty, negative_only=True)
            self.positive = None
            self.negative = None
        else:
            self.condition = None
            self.positive = adapter.prepare_prompt(prompt)
            self.negative = adapter.prepare_prompt(negative_prompt)
            empty = adapter.prepare_prompt("")
            configured_hash = condition_sha256(self.negative)
            empty_hash = condition_sha256(empty)
        if configured_hash == empty_hash:
            raise RuntimeError(
                f"{model_id} native-negative conditioning equals empty conditioning."
            )
        self.evidence = {
            "implementation": "installed_diffusers_native_true_cfg",
            "model_id": model_id,
            "negative_prompt": negative_prompt,
            "true_cfg_scale": self.true_cfg_scale,
            "configured_negative_condition_sha256": configured_hash,
            "empty_condition_sha256": empty_hash,
            "conditioning_difference_proved": True,
            "qwen_native_norm_rescale": model_id.startswith("qwen_image"),
        }

    def _sd35_condition(self, prompt: str, negative_prompt: str) -> PromptCondition:
        keys = ("negative_prompt", "negative_prompt_2", "negative_prompt_3")
        previous = {key: self.adapter.config.get(key) for key in keys}
        try:
            for key in keys:
                self.adapter.config[key] = negative_prompt
            return self.adapter.prepare_prompt(prompt)
        finally:
            for key, value in previous.items():
                if value is None:
                    self.adapter.config.pop(key, None)
                else:
                    self.adapter.config[key] = value

    def predict(
        self,
        latents: torch.Tensor,
        timestep: Any,
        state: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if self.model_id == "sd35_large":
            prediction = self.adapter.predict_vector_field(
                latents, timestep, self.condition, state
            )
            return prediction, {"combined_prediction": tensor_stats(prediction)}
        positive = self.adapter.predict_vector_field(
            latents, timestep, self.positive, state
        )
        negative = self.adapter.predict_vector_field(
            latents, timestep, self.negative, state
        )
        combined = negative + self.true_cfg_scale * (positive - negative)
        if self.model_id.startswith("qwen_image"):
            conditional_norm = torch.norm(positive, dim=-1, keepdim=True)
            combined_norm = torch.norm(combined, dim=-1, keepdim=True)
            if bool((combined_norm == 0).any()):
                raise RuntimeError("Qwen native CFG produced a zero combined norm.")
            combined = combined * (conditional_norm / combined_norm)
        if not bool(torch.isfinite(combined).all()):
            raise RuntimeError("Native negative CFG produced non-finite values.")
        delta = combined - positive
        if float(delta.float().norm().item()) <= 0.0:
            raise RuntimeError("Native negative CFG produced a zero prediction delta.")
        return combined, {
            "positive_prediction": tensor_stats(positive),
            "negative_prediction": tensor_stats(negative),
            "cfg_delta": tensor_stats(delta),
            "combined_prediction": tensor_stats(combined),
        }


def historical_category_contrast_hierarchy(
    *,
    category: str,
    unsafe_concept: str,
    safe_concept: str,
) -> ConceptHierarchy:
    category_label = category.replace("_", " ")
    shared_context = (
        "a coherent visual scene with subject identity and count, pose, scene, "
        "composition, camera, setting, lighting, medium, style, and every attribute "
        f"unrelated to {category_label} held fixed"
    )
    neutral = f"{shared_context}; the {category_label} safety attribute is neutral and unspecified"
    unsafe = f"{shared_context}; {unsafe_concept}"
    safe = f"{shared_context}; {safe_concept}"
    return ConceptHierarchy.from_dict(
        {
            "name": f"t2isafety_historical_category_contrast_v2_{category}",
            "neutral_concept": neutral,
            "pairs": [
                {
                    "id": category,
                    "parent": "t2isafety_category_control",
                    "unsafe_concept": unsafe,
                    "safe_sibling_concept": safe,
                }
            ],
        }
    )


def historical_conceptsteer(
    *,
    category: str,
    unsafe_concept: str,
    safe_concept: str,
    strength: float,
) -> HierarchicalVectorFieldBottleneck:
    hierarchy = historical_category_contrast_hierarchy(
        category=category,
        unsafe_concept=unsafe_concept,
        safe_concept=safe_concept,
    )
    endpoints = [
        hierarchy.neutral_concept,
        *[
            endpoint
            for pair in hierarchy.pairs
            for endpoint in (pair.unsafe_concept, pair.safe_sibling_concept)
        ],
    ]
    if any("source prompt" in endpoint.lower() for endpoint in endpoints):
        raise RuntimeError("Historical ConceptSteer endpoints must not reference the source prompt.")
    config = BottleneckConfig.from_dict(
        {
            "enabled": True,
            "start_fraction": 0.0,
            "end_fraction": 1.0,
            "step_stride": 1,
            "lambda_schedule": {
                "kind": "constant",
                "max_value": float(strength),
                "min_value": float(strength),
            },
            "margin": 0.05,
            "feature_dim": 1,
            "mask": {"enabled": False},
            "prompt_composition": "concept_only",
            "normalize_directions": False,
            "active_pair_ids": [category],
        }
    )
    return HierarchicalVectorFieldBottleneck(hierarchy=hierarchy, config=config)


def validate_conceptsteer_trace(trace: list[dict[str, Any]], category: str) -> dict[str, Any]:
    deltas: list[float] = []
    activations: list[float] = []
    for step in trace:
        for concept in step.get("concepts", []):
            if concept.get("concept_id") != category:
                continue
            stats = concept["steering_delta_stats"]
            deltas.append(
                max(
                    abs(float(stats["min"])),
                    abs(float(stats["max"])),
                    abs(float(stats["mean"])),
                    float(stats["std"]),
                )
            )
            activations.append(float(concept["activation"]["max"]))
    if len(deltas) != len(trace):
        raise RuntimeError(
            f"ConceptSteer expected {len(trace)} {category} interventions, got {len(deltas)}."
        )
    nonzero = sum(value > 0.0 for value in deltas)
    if nonzero == 0:
        raise RuntimeError("ConceptSteer produced zero intervention across the run.")
    return {
        "status": "passed",
        "steps": len(trace),
        "nonzero_delta_steps": nonzero,
        "maximum_delta_stat": max(deltas),
        "maximum_activation": max(activations),
    }


class MidSteerRuntime:
    def __init__(
        self,
        *,
        adapter: Any,
        model_id: str,
        category: str,
        pilot_candidate: MethodPilotCandidate | None = None,
    ) -> None:
        root = resolve_transformer_root(adapter)
        topology = validate_midsteer_topology(
            model_id=model_id,
            adapter=adapter,
            root=root,
        )
        if pilot_candidate is None:
            path = (
                CALIBRATION_ROOT
                / "midsteer"
                / model_id
                / category
                / "artifact.pt"
            )
            admission_path = path.parent / "ADMISSION.json"
        else:
            pilot_candidate.require_identity(
                model_id=model_id,
                category=category,
                method="midsteer",
            )
            path = pilot_candidate.method_artifact_path
            admission_path = pilot_candidate.method_admission_path
        if not path.is_file():
            raise FileNotFoundError(f"Missing admitted MidSteer artifact {path}")
        if not admission_path.is_file():
            raise FileNotFoundError(
                f"Missing MidSteer artifact admission {admission_path}"
            )
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        required_admission = {
            "protocol": "t2i_safety_midsteer_artifact_admission_v3",
            "status": "accepted",
            "model_id": model_id,
            "category": category,
            "variant": "midsteer",
            "intermediate_clipping": False,
            "topology_sha256": topology["topology_sha256"],
            "site_contract_sha256": topology["site_contract_sha256"],
            "topology_admission_sha256": topology[
                "topology_admission_sha256"
            ],
        }
        mismatches = {
            key: (admission.get(key), expected)
            for key, expected in required_admission.items()
            if admission.get(key) != expected
        }
        if mismatches:
            raise RuntimeError(
                f"MidSteer artifact admission mismatch: {mismatches}"
            )
        artifact_sha256 = file_sha256(path)
        if admission.get("artifact_sha256") != artifact_sha256:
            raise RuntimeError(
                f"MidSteer artifact/admission hash mismatch for {path}"
            )
        calibration = (
            load_method_calibration(model_id, category, "midsteer")
            if pilot_candidate is None
            else pilot_candidate.calibration_payload()
        )
        if calibration.get("midsteer_artifact_sha256") != artifact_sha256:
            raise RuntimeError(
                f"MidSteer calibration/artifact hash mismatch for {path}"
            )
        strength = float(calibration["parameters"]["strength"])
        if not math.isfinite(strength) or strength <= 0.0:
            raise RuntimeError(
                f"MidSteer calibration strength is invalid for {path}"
            )
        self.adapter = adapter
        self.model_id = model_id
        self.category = category
        self.strength = strength
        self.num_inference_steps = int(calibration["num_inference_steps"])
        self.active_step_indices = tuple(calibration["active_step_indices"])
        self._active_steps = set(self.active_step_indices)
        self.scheduler_grid_path, self.expected_timesteps = load_scheduler_grid_binding(
            model_id=model_id,
            num_steps=self.num_inference_steps,
            path_value=calibration["scheduler_grid_path"],
            sha256_value=calibration["scheduler_grid_sha256"],
        )
        self.path = path
        self.artifact = MidSteerArtifact.load(path)
        artifact_topology = {
            key: self.artifact.metadata.get(key)
            for key in (
                "topology_sha256",
                "site_contract_sha256",
                "topology_admission_sha256",
            )
        }
        expected_topology = {
            key: topology[key]
            for key in artifact_topology
        }
        if artifact_topology != expected_topology:
            raise RuntimeError(
                f"MidSteer artifact topology mismatch: "
                f"{artifact_topology} != {expected_topology}"
            )
        self.root = root
        self.topology = topology
        self.metadata = {
            "artifact_path": str(path),
            "artifact_sha256": artifact_sha256,
            "artifact_admission_path": str(admission_path),
            "artifact_admission_sha256": file_sha256(admission_path),
            "calibration_path": calibration["artifact_path"],
            "calibration_sha256": calibration["artifact_sha256"],
            "upstream_revision": MIDSTEER_REVISION,
            "strength": self.strength,
            "num_inference_steps": self.num_inference_steps,
            "active_step_indices": list(self.active_step_indices),
            "scheduler_grid_path": str(self.scheduler_grid_path),
            "scheduler_grid_sha256": calibration["scheduler_grid_sha256"],
            "calibration_source": "accepted_production_calibration"
            if pilot_candidate is None
            else "sealed_pilot_candidate",
            "control_mode": "attn_output_post_projection",
            "intermediate_clipping": False,
            "first_diffusion_step_reused": True,
            "topology_sha256": topology["topology_sha256"],
            "site_contract_sha256": topology["site_contract_sha256"],
            "topology_admission_path": topology["topology_admission_path"],
            "topology_admission_sha256": topology[
                "topology_admission_sha256"
            ],
        }

    def validate_runtime_timesteps(self, timesteps: Any) -> tuple[float, ...]:
        return assert_runtime_timesteps(
            expected=self.expected_timesteps,
            observed=timesteps,
            source=self.scheduler_grid_path,
        )

    def predict(
        self,
        *,
        latents: torch.Tensor,
        timestep: Any,
        condition: PromptCondition,
        state: Any,
        step_index: int,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        if step_index < 0 or step_index >= self.num_inference_steps:
            raise RuntimeError(
                f"MidSteer step {step_index} is outside its calibrated grid "
                f"of {self.num_inference_steps} steps."
            )
        if step_index not in self._active_steps:
            prediction = self.adapter.predict_vector_field(
                latents, timestep, condition, state
            )
            return prediction, {
                "active": False,
                "step_index": step_index,
                "reason": "outside_calibrated_active_step_set",
                "prediction": tensor_stats(prediction),
            }
        image_token_count = midsteer_image_token_count(self.topology, latents)
        intervention = MidSteerIntervention(
            self.root,
            self.artifact,
            model_role="default",
            step_index=step_index,
            strength=self.strength,
            intermediate_clipping=False,
            image_token_count=image_token_count,
        )
        with intervention:
            prediction = self.adapter.predict_vector_field(
                latents, timestep, condition, state
            )
        evidence = intervention.evidence()
        evidence["active"] = True
        evidence["step_index"] = step_index
        evidence["calibrated_strength"] = self.strength
        evidence["prediction"] = tensor_stats(prediction)
        return prediction, evidence


class UnsafeReferenceBank:
    def __init__(
        self,
        *,
        model_id: str,
        category: str,
        method: str,
        device: torch.device,
    ) -> None:
        path = (
            CALIBRATION_ROOT
            / "unsafe_references_v2"
            / model_id
            / category
            / "unsafe_latents.pt"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing unsafe-reference artifact {path}")
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        metadata = payload.get("metadata")
        references = payload.get("references")
        safe_references = payload.get("safe_references")
        if (
            not isinstance(metadata, dict)
            or not isinstance(references, torch.Tensor)
            or not isinstance(safe_references, torch.Tensor)
        ):
            raise RuntimeError(f"Malformed unsafe-reference artifact {path}")
        if metadata.get("protocol") != "author_switch_reference_bank_v2":
            raise RuntimeError(f"Unsafe-reference protocol mismatch in {path}")
        if metadata.get("model_id") != model_id or metadata.get("category") != category:
            raise RuntimeError(f"Unsafe-reference identity mismatch in {path}")
        reference_count = int(metadata.get("reference_count", -1))
        effective_population = int(
            metadata.get("effective_reference_population", -1)
        )
        if (
            not 2 <= reference_count <= 515
            or references.shape[0] != reference_count
            or safe_references.shape != references.shape
            or effective_population != 515
        ):
            raise RuntimeError(
                f"Unsafe-reference artifact {path} has an invalid empirical population."
            )
        observed = tensor_sha256(references)
        if observed != metadata.get("tensor_sha256"):
            raise RuntimeError(f"Unsafe-reference tensor hash mismatch in {path}")
        observed_safe = tensor_sha256(safe_references)
        if observed_safe != metadata.get("safe_tensor_sha256"):
            raise RuntimeError(f"Safe-reference tensor hash mismatch in {path}")
        if method == "sgf" and reference_count < 6:
            raise RuntimeError(
                f"SGF author top-k bandwidth requires at least six references in {path}."
            )
        beta_threshold = metadata.get("safe_beta_threshold")
        if not isinstance(beta_threshold, (int, float)) or not math.isfinite(
            float(beta_threshold)
        ):
            raise RuntimeError(f"Safe Denoiser beta threshold is invalid in {path}")
        self.path = path
        self.metadata = {
            **metadata,
            "artifact_path": str(path),
            "artifact_sha256": file_sha256(path),
        }
        self.reference_count = reference_count
        self.effective_reference_population = effective_population
        self.references = references.to(device=device, dtype=torch.float32)
        self.safe_references = safe_references.to(
            device=device,
            dtype=torch.float32,
        )
        self.safe_beta_threshold = float(beta_threshold)
        self.safe_beta_margin = float(
            metadata.get("safe_beta_threshold_margin", SAFE_DENOISER_BETA_MARGIN)
        )

    def validate_shape(
        self,
        current: torch.Tensor,
        *,
        safe_projection: bool = False,
    ) -> None:
        references = self.safe_references if safe_projection else self.references
        if tuple(references.shape[1:]) != tuple(current.shape[1:]):
            raise RuntimeError(
                f"Reference shape {tuple(references.shape[1:])} does not match "
                f"predicted clean shape {tuple(current.shape[1:])}."
            )


def _pairwise_squared_distance(x: torch.Tensor, references: torch.Tensor) -> torch.Tensor:
    x_flat = x.reshape(x.shape[0], -1).float()
    refs_flat = references.reshape(references.shape[0], -1).float()
    distances = (
        x_flat.square().sum(dim=1, keepdim=True)
        + refs_flat.square().sum(dim=1).unsqueeze(0)
        - 2.0 * (x_flat @ refs_flat.mT)
    )
    return distances.clamp_min_(0.0)


def sgf_correct_x0(
    current_x0: torch.Tensor,
    bank: UnsafeReferenceBank,
    *,
    strength: float = SGF_DEFAULT_STRENGTH,
    top_k: int = 3,
    epsilon: float = 0.05,
) -> tuple[torch.Tensor, dict[str, Any]]:
    bank.validate_shape(current_x0)
    if not 0.0 < epsilon < 1.0:
        raise ValueError("SGF epsilon must be in (0, 1).")
    references = bank.references
    distances = _pairwise_squared_distance(current_x0, references)
    if int(top_k) != 3:
        raise ValueError("The admitted SGF GitHub contract fixes top_k=3.")
    if references.shape[0] < 6:
        raise RuntimeError("The SGF GitHub bandwidth slice requires six references.")
    sorted_distances = torch.sort(distances, dim=1).values
    bandwidth_distance = sorted_distances[:, 3:6].mean().clamp_min(1e-12)
    gamma = -math.log(float(epsilon)) / float(bandwidth_distance.item())
    kernels = torch.exp(-gamma * distances)
    x_flat = current_x0.reshape(current_x0.shape[0], -1).float()
    refs_flat = references.reshape(references.shape[0], -1).float()
    field = -2.0 * gamma * (
        kernels.sum(dim=1, keepdim=True) * x_flat - kernels @ refs_flat
    )
    corrected = x_flat + float(strength) * field
    corrected = corrected.reshape_as(current_x0).to(dtype=current_x0.dtype)
    delta = corrected - current_x0
    if not bool(torch.isfinite(corrected).all()) or float(delta.float().norm()) <= 0.0:
        raise RuntimeError("SGF produced a zero or non-finite correction.")
    return corrected, {
        "method": "safety_guided_flow",
        "implementation_contract": "SGF_4bdd287_nudity_grad_mmd",
        "equation": "x0_plus_scale_times_author_dK_dX",
        "strength": float(strength),
        "top_k": int(top_k),
        "bandwidth_sorted_distance_slice": [3, 6],
        "epsilon": float(epsilon),
        "gamma": float(gamma),
        "bandwidth_distance": float(bandwidth_distance.item()),
        "kernel_mean": float(kernels.mean().item()),
        "kernel_max": float(kernels.max().item()),
        "unique_reference_count": bank.reference_count,
        "effective_reference_population": bank.effective_reference_population,
        "empirical_mass_multiplier": 1.0,
        "delta": tensor_stats(delta),
    }


def safe_kernel_statistics(
    current: torch.Tensor,
    references: torch.Tensor,
    *,
    sigma: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if sigma <= 0.0:
        raise ValueError("Safe Denoiser sigma must be positive.")
    current_flat = current.reshape(current.shape[0], -1).float()
    references_flat = references.reshape(references.shape[0], -1).float()
    distances = torch.cdist(current_flat, references_flat, p=2.0)
    kernels = torch.exp(-distances / (2.0 * float(sigma) ** 2))
    denominator = kernels.sum(dim=1, keepdim=True) + 1.0e-8
    return distances, kernels, denominator


def safe_denoiser_correct_x0(
    current_x0: torch.Tensor,
    bank: UnsafeReferenceBank,
    *,
    sigma: float,
    scale: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    bank.validate_shape(current_x0, safe_projection=True)
    if scale < 0.0:
        raise ValueError("Safe Denoiser scale must be non-negative.")
    references = bank.safe_references
    distances, kernels, denominator = safe_kernel_statistics(
        current_x0,
        references,
        sigma=sigma,
    )
    refs_flat = references.reshape(references.shape[0], -1).float()
    unsafe_denoiser = (kernels @ refs_flat) / denominator
    unsafe_denoiser = unsafe_denoiser.reshape_as(current_x0)
    threshold = bank.safe_beta_threshold - bank.safe_beta_margin
    if current_x0.shape[0] != 1:
        raise RuntimeError("Safe Denoiser switch adaptation requires batch size one.")
    is_negation = bool(float(denominator.item()) > threshold)
    correction = float(scale) * unsafe_denoiser
    corrected = (
        current_x0.float() - correction
        if is_negation
        else current_x0.float()
    ).to(dtype=current_x0.dtype)
    if not bool(torch.isfinite(corrected).all()):
        raise RuntimeError("Safe Denoiser produced a non-finite correction.")
    if is_negation and float(correction.norm()) <= 0.0:
        raise RuntimeError("Safe Denoiser admitted a zero correction.")
    return corrected, {
        "method": "training_free_safe_denoiser",
        "implementation_contract": (
            "Safe_Denoiser_223415b_kernel_fast_conditioning_threshold"
        ),
        "equation": "predicted_x0_minus_scale_times_Eunsafe",
        "sigma": float(sigma),
        "scale": float(scale),
        "kernel_distance_policy": "released_unsquared_l2",
        "kernel_space": (
            "unprojected_predicted_x0_vs_author_channel_normalized_references"
        ),
        "beta_denominator": float(denominator.item()),
        "beta_threshold_raw": bank.safe_beta_threshold,
        "beta_threshold_margin": bank.safe_beta_margin,
        "beta_threshold_effective": threshold,
        "is_negation": is_negation,
        "projected_l2_distance_mean": float(distances.mean().item()),
        "projected_l2_distance_min": float(distances.min().item()),
        "kernel_mean": float(kernels.mean().item()),
        "kernel_max": float(kernels.max().item()),
        "unsafe_denoiser": tensor_stats(unsafe_denoiser),
        "delta": tensor_stats(corrected - current_x0),
    }


def _native_scheduler(adapter: Any) -> Any:
    for owner in (
        adapter,
        getattr(adapter, "pipeline", None),
        getattr(adapter, "pipe", None),
    ):
        scheduler = getattr(owner, "scheduler", None) if owner is not None else None
        if scheduler is not None:
            return scheduler
    raise RuntimeError("The adapter exposes no native scheduler.")


def native_renoise(
    *,
    adapter: Any,
    model_id: str,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: Any,
) -> tuple[torch.Tensor, dict[str, Any]]:
    scheduler = _native_scheduler(adapter)
    parameterization = parameterization_for_model(model_id)
    if clean.ndim < 1 or clean.shape[0] < 1:
        raise RuntimeError("Native re-noising requires a non-empty batch dimension.")
    if noise.shape != clean.shape:
        raise RuntimeError(
            "Native re-noising requires noise with exactly the clean latent shape."
        )
    if noise.device != clean.device:
        raise RuntimeError("Native re-noising requires clean and noise on one device.")

    scheduler_timestep = timestep
    timestep_component = "adapter_native_timestep"
    if model_id == "ideogram4_nf4":
        extractor = getattr(adapter, "_timestep_values", None)
        if not callable(extractor):
            raise RuntimeError(
                "Ideogram4 native re-noising requires the adapter composite-timestep contract."
            )
        scheduler_timestep, _ = extractor(timestep)
        timestep_component = "ideogram4_composite_scheduler_component"

    schedule = getattr(scheduler, "timesteps", None)
    if schedule is None:
        raise RuntimeError(f"{model_id} scheduler exposes no initialized native schedule.")
    schedule_tensor = torch.as_tensor(schedule, device=clean.device)
    if schedule_tensor.ndim != 1 or schedule_tensor.numel() < 1:
        raise RuntimeError(f"{model_id} scheduler exposes an invalid native schedule.")
    scalar_timestep = torch.as_tensor(
        scheduler_timestep,
        device=clean.device,
        dtype=schedule_tensor.dtype,
    ).reshape(-1)
    if scalar_timestep.numel() != 1:
        raise RuntimeError(
            f"{model_id} native scheduler timestep must contain exactly one value."
        )
    scalar_timestep = scalar_timestep[0]
    exact_matches = (schedule_tensor == scalar_timestep).nonzero(as_tuple=False).flatten()
    if exact_matches.numel() < 1:
        raise RuntimeError(
            f"{model_id} re-noising timestep is not an exact member of its native schedule."
        )
    timestep_batch = scalar_timestep.reshape(1).expand(clean.shape[0])

    if parameterization in {
        NativeParameterization.FLOW,
        NativeParameterization.PHYSICAL_VELOCITY,
    }:
        api = getattr(scheduler, "scale_noise", None)
        api_name = "scale_noise"
    elif parameterization in {
        NativeParameterization.EPSILON,
        NativeParameterization.V_PREDICTION,
    }:
        api = getattr(scheduler, "add_noise", None)
        api_name = "add_noise"
    else:
        raise RuntimeError(
            f"No author-faithful re-noising contract for {parameterization.value}."
        )
    if not callable(api):
        raise RuntimeError(
            f"{model_id} scheduler {type(scheduler).__name__} lacks required "
            f"native {api_name}."
        )

    had_begin_index = hasattr(scheduler, "_begin_index")
    original_begin_index = getattr(scheduler, "_begin_index", None)
    if had_begin_index:
        scheduler._begin_index = None
    try:
        if api_name == "scale_noise":
            renoised = api(clean, timestep_batch, noise)
            argument_order = "clean_timestep_noise"
        else:
            renoised = api(clean, noise, timestep_batch)
            argument_order = "clean_noise_timestep"
    finally:
        if had_begin_index:
            scheduler._begin_index = original_begin_index
    if (
        not isinstance(renoised, torch.Tensor)
        or renoised.shape != clean.shape
        or not bool(torch.isfinite(renoised).all())
    ):
        raise RuntimeError(f"{model_id} native {api_name} returned invalid latents.")
    return renoised, {
        "scheduler_class": type(scheduler).__name__,
        "scheduler_api": api_name,
        "parameterization": parameterization.value,
        "argument_order": argument_order,
        "timestep_component": timestep_component,
        "timestep_batch_matches_latent_batch": True,
        "exact_native_schedule_membership": True,
        "scheduler_begin_index_isolated_and_restored": had_begin_index,
    }


def related_work_transition(
    *,
    method: str,
    adapter: Any,
    model_id: str,
    latents: torch.Tensor,
    native_prediction: torch.Tensor,
    timestep: Any,
    step_index: int,
    num_steps: int,
    guidance_scale: float,
    bank: UnsafeReferenceBank,
    generator: torch.Generator,
    sgf_strength: float = SGF_DEFAULT_STRENGTH,
    safe_sigma: float | None = None,
    safe_scale: float | None = None,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    canonical = canonicalize_prediction(
        adapter=adapter,
        model_id=model_id,
        latents=latents,
        native=native_prediction,
        timestep=timestep,
        step_index=step_index,
        guidance_scale=guidance_scale,
        branch="registered_positive",
    )
    if method == "sgf":
        eligible_steps = int(math.ceil(SGF_ACTIVE_FRACTION * num_steps))
        eligible = step_index < eligible_steps
        is_negation = eligible
        if eligible:
            corrected_x0, detail = sgf_correct_x0(
                canonical.predicted_x0,
                bank,
                strength=sgf_strength,
            )
        else:
            corrected_x0 = canonical.predicted_x0
            detail = {"method": "safety_guided_flow"}
    elif method == "safe_denoiser":
        eligible_steps = int(math.ceil(SAFE_DENOISER_ACTIVE_FRACTION * num_steps))
        eligible = step_index < eligible_steps
        defaults = SAFE_DENOISER_DEFAULTS[model_id]
        sigma = float(safe_sigma if safe_sigma is not None else defaults["sigma"])
        scale = float(
            safe_scale if safe_scale is not None else defaults["scale"]
        )
        if eligible:
            corrected_x0, detail = safe_denoiser_correct_x0(
                canonical.predicted_x0,
                bank,
                sigma=sigma,
                scale=scale,
            )
            is_negation = bool(detail["is_negation"])
        else:
            corrected_x0 = canonical.predicted_x0
            is_negation = False
            detail = {"method": "training_free_safe_denoiser"}
    else:
        raise ValueError(f"Unknown related-work method {method!r}")
    detail["eligible"] = eligible
    detail["eligible_steps"] = eligible_steps
    detail["active"] = bool(is_negation)
    detail["canonical"] = canonical.metadata()
    if is_negation:
        noise = torch.randn(
            corrected_x0.shape,
            generator=generator,
            device=corrected_x0.device,
            dtype=torch.float32,
        )
        renoised, renoise_metadata = native_renoise(
            adapter=adapter,
            model_id=model_id,
            clean=corrected_x0,
            noise=noise,
            timestep=timestep,
        )
        detail["fresh_noise"] = tensor_stats(noise)
        detail["renoise"] = renoise_metadata
        detail["renoised_latent_delta"] = tensor_stats(renoised - latents)
        return renoised, detail
    return None, detail


# EXACT_RELATED_WORK_V3_OVERRIDE
# These late bindings preserve all caller signatures while replacing the earlier
# release-code adaptations with the paper-exact, fail-closed implementations.
from .exact_related_work import (
    UnsafeReferenceBank as UnsafeReferenceBank,
    _load_method_calibration as load_method_calibration,
    related_work_transition as related_work_transition,
    safe_denoiser_correct_x0 as safe_denoiser_correct_x0,
    sgf_correct_x0 as sgf_correct_x0,
    validate_related_work_runtime_grid as validate_related_work_runtime_grid,
)
