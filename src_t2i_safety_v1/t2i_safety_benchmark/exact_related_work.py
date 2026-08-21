from __future__ import annotations

import hashlib
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from hierasafe_flow.steering.canonical.contract import (
    canonicalize_prediction,
    prediction_from_x0,
)

from .contracts import CALIBRATION_ROOT, file_sha256
from .pilot_context import (
    MethodPilotCandidate,
    assert_runtime_timesteps,
    load_scheduler_grid_binding,
)


REFERENCE_BANK_PROTOCOL = "paper_exact_switch_reference_bank_v3"
METHOD_CALIBRATION_PROTOCOL = "t2i_safety_method_calibration_v3"
REFERENCE_CHUNK_SIZE = 4
REFERENCE_BANK_METHODS = frozenset({"sgf", "safe_denoiser"})
SUPPORTED_METHODS = frozenset({"midsteer", *REFERENCE_BANK_METHODS})


def _tensor_bytes(value: torch.Tensor) -> bytes:
    tensor = value.detach().contiguous().to(device="cpu")
    return tensor.view(torch.uint8).numpy().tobytes()


def tensor_sha256(value: torch.Tensor) -> str:
    digest = hashlib.sha256()
    digest.update(str(tuple(value.shape)).encode("ascii"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(_tensor_bytes(value))
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


def _load_torch(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _require_sha256(value: Any, *, field: str, path: Path) -> str:
    text = str(value)
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise RuntimeError(f"{field} is not a lowercase SHA-256 digest in {path}")
    return text


class UnsafeReferenceBank:
    """Strict empirical unsafe-reference bank with no synthetic mass."""

    def __init__(
        self,
        *,
        model_id: str,
        category: str,
        method: str,
        device: torch.device,
    ) -> None:
        if method not in REFERENCE_BANK_METHODS:
            raise ValueError(f"Unsupported reference-bank method {method!r}")

        path = (
            CALIBRATION_ROOT
            / "unsafe_references_v3"
            / model_id
            / category
            / "unsafe_latents.pt"
        )
        if not path.is_file():
            raise FileNotFoundError(f"Missing unsafe-reference artifact {path}")
        admission_path = path.parent / "ADMISSION.json"
        if not admission_path.is_file():
            raise FileNotFoundError(
                f"Missing unsafe-reference admission {admission_path}"
            )
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        required_admission = {
            "protocol": "t2i_safety_reference_bank_admission_v3",
            "status": "accepted",
            "model_id": model_id,
            "category": category,
        }
        admission_mismatches = {
            key: (admission.get(key), expected)
            for key, expected in required_admission.items()
            if admission.get(key) != expected
        }
        if admission_mismatches:
            raise RuntimeError(
                f"Unsafe-reference admission mismatch: {admission_mismatches}"
            )
        artifact_sha256 = file_sha256(path)
        if admission.get("artifact_sha256") != artifact_sha256:
            raise RuntimeError(
                f"Unsafe-reference artifact/admission hash mismatch in {path}"
            )

        payload = _load_torch(path)
        if not isinstance(payload, dict):
            raise RuntimeError(f"Malformed unsafe-reference artifact {path}")
        metadata = payload.get("metadata")
        references = payload.get("references")
        kernel_references = payload.get("kernel_references")
        kernel_pairwise_squared_distances = payload.get(
            "kernel_pairwise_squared_distances"
        )
        if (
            not isinstance(metadata, dict)
            or not isinstance(references, torch.Tensor)
            or not isinstance(kernel_references, torch.Tensor)
            or not isinstance(kernel_pairwise_squared_distances, torch.Tensor)
        ):
            raise RuntimeError(f"Malformed unsafe-reference artifact {path}")
        if metadata.get("protocol") != REFERENCE_BANK_PROTOCOL:
            raise RuntimeError(f"Unsafe-reference protocol mismatch in {path}")
        if metadata.get("model_id") != model_id or metadata.get("category") != category:
            raise RuntimeError(f"Unsafe-reference identity mismatch in {path}")
        if references.ndim < 2 or kernel_references.shape != references.shape:
            raise RuntimeError(f"Unsafe-reference tensor shape mismatch in {path}")

        reference_count = int(references.shape[0])
        declared_count = int(metadata.get("reference_count", -1))
        declared_population = int(
            metadata.get("effective_reference_population", -1)
        )
        minimum_count = 4 if method == "sgf" else 2
        if reference_count < minimum_count:
            raise RuntimeError(
                f"{method} requires at least {minimum_count} unique references in {path}"
            )
        if declared_count != reference_count:
            raise RuntimeError(
                f"Reference count {declared_count} does not match tensor count "
                f"{reference_count} in {path}"
            )
        if declared_population != reference_count:
            raise RuntimeError(
                f"Empirical population must equal the {reference_count} unique "
                f"references in {path}; synthetic mass is forbidden"
            )
        if float(metadata.get("empirical_mass_multiplier", 1.0)) != 1.0:
            raise RuntimeError(f"Empirical mass inflation is forbidden in {path}")
        admission_population = {
            "reference_count": reference_count,
            "effective_reference_population": reference_count,
            "empirical_mass_multiplier": 1.0,
            "tensor_sha256": metadata.get("tensor_sha256"),
            "kernel_tensor_sha256": metadata.get("kernel_tensor_sha256"),
            "kernel_pairwise_squared_distances_sha256": metadata.get(
                "kernel_pairwise_squared_distances_sha256"
            ),
        }
        if any(
            admission.get(key) != expected
            for key, expected in admission_population.items()
        ):
            raise RuntimeError(
                f"Unsafe-reference admission population mismatch in {path}"
            )

        if tensor_sha256(references) != str(metadata.get("tensor_sha256")):
            raise RuntimeError(f"Unsafe-reference tensor hash mismatch in {path}")
        if tensor_sha256(kernel_references) != str(
            metadata.get("kernel_tensor_sha256")
        ):
            raise RuntimeError(f"Kernel-reference tensor hash mismatch in {path}")
        if tensor_sha256(kernel_pairwise_squared_distances) != str(
            metadata.get("kernel_pairwise_squared_distances_sha256")
        ):
            raise RuntimeError(f"Kernel-distance tensor hash mismatch in {path}")
        if tuple(kernel_pairwise_squared_distances.shape) != (
            reference_count,
            reference_count,
        ):
            raise RuntimeError(f"Kernel-distance matrix shape mismatch in {path}")
        if not bool(torch.isfinite(kernel_pairwise_squared_distances).all()):
            raise RuntimeError(f"Kernel-distance matrix is non-finite in {path}")
        if bool((kernel_pairwise_squared_distances < 0.0).any()):
            raise RuntimeError(f"Kernel-distance matrix is negative in {path}")
        if not torch.allclose(
            kernel_pairwise_squared_distances,
            kernel_pairwise_squared_distances.T,
            rtol=1.0e-5,
            atol=1.0e-5,
        ):
            raise RuntimeError(f"Kernel-distance matrix is not symmetric in {path}")
        if not torch.allclose(
            torch.diagonal(kernel_pairwise_squared_distances),
            torch.zeros(reference_count, dtype=kernel_pairwise_squared_distances.dtype),
            rtol=0.0,
            atol=1.0e-5,
        ):
            raise RuntimeError(f"Kernel-distance matrix diagonal is not zero in {path}")
        kernel_contract = metadata.get("safe_kernel")
        if not isinstance(kernel_contract, dict):
            raise RuntimeError(f"Safe Denoiser kernel contract is missing in {path}")
        feature_dim = kernel_contract.get("feature_dim")
        if (
            isinstance(feature_dim, bool)
            or not isinstance(feature_dim, int)
            or feature_dim <= 0
            or feature_dim >= references.ndim
        ):
            raise RuntimeError(f"Safe Denoiser feature dimension is invalid in {path}")
        expected_kernel_contract = {
            "feature_policy": "channel_l2_normalize_query_and_references_v3",
            "distance": "squared_euclidean_in_normalized_feature_space",
            "kernel": "exp(-squared_distance/(2*sigma^2))",
            "eq6_value_space": "raw_native_predicted_x0_latents",
            "beta_units": "mean_kernel_mass_over_unique_empirical_references",
            "threshold_policy": "leave_one_out_reference_beta_quantile",
        }
        contract_mismatches = {
            key: (kernel_contract.get(key), expected)
            for key, expected in expected_kernel_contract.items()
            if kernel_contract.get(key) != expected
        }
        if contract_mismatches:
            raise RuntimeError(
                f"Safe Denoiser kernel contract mismatch in {path}: "
                f"{contract_mismatches}"
            )

        self.model_id = model_id
        self.category = category
        self.method = method
        self.path = path
        self.reference_count = reference_count
        self.effective_reference_population = reference_count
        self.references = references.to(device=device, dtype=torch.float32)
        self.kernel_references = kernel_references.to(
            device=device, dtype=torch.float32
        )
        self.kernel_pairwise_squared_distances = (
            kernel_pairwise_squared_distances.to(device=device, dtype=torch.float32)
        )
        self.safe_feature_dim = feature_dim
        self._safe_threshold_cache: dict[
            tuple[float, float], tuple[float, dict[str, float]]
        ] = {}
        self.metadata = {
            **metadata,
            "reference_count": reference_count,
            "effective_reference_population": reference_count,
            "empirical_mass_multiplier": 1.0,
            "artifact_path": str(path),
            "artifact_sha256": artifact_sha256,
            "admission_path": str(admission_path),
            "admission_sha256": file_sha256(admission_path),
        }

    def validate_shape(self, current_x0: torch.Tensor) -> None:
        if current_x0.ndim != self.references.ndim:
            raise ValueError(
                "Unsafe-reference rank does not match the current clean estimate"
            )
        if tuple(current_x0.shape[1:]) != tuple(self.references.shape[1:]):
            raise ValueError(
                "Unsafe-reference latent shape does not match the current clean "
                f"estimate: {tuple(self.references.shape[1:])} != "
                f"{tuple(current_x0.shape[1:])}"
            )

    def safe_kernel_features(self, value: torch.Tensor) -> torch.Tensor:
        self.validate_shape(value)
        return F.normalize(
            value.to(dtype=torch.float32),
            p=2.0,
            dim=self.safe_feature_dim,
            eps=1.0e-12,
        )

    def safe_beta_threshold(
        self,
        *,
        sigma: float,
        beta_quantile: float,
    ) -> tuple[float, dict[str, float]]:
        if not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
            raise ValueError("Safe Denoiser sigma must be finite and positive")
        if (
            not math.isfinite(float(beta_quantile))
            or float(beta_quantile) < 0.0
            or float(beta_quantile) > 1.0
        ):
            raise ValueError("Safe Denoiser beta_quantile must be in [0, 1]")
        key = (float(sigma), float(beta_quantile))
        cached = self._safe_threshold_cache.get(key)
        if cached is not None:
            return cached
        denominator_scale = 2.0 * float(sigma) * float(sigma)
        kernels = torch.exp(
            -self.kernel_pairwise_squared_distances / denominator_scale
        )
        kernels.fill_diagonal_(0.0)
        reference_beta = kernels.sum(dim=1) / float(self.reference_count - 1)
        threshold = float(
            torch.quantile(reference_beta, float(beta_quantile)).item()
        )
        stats = {
            "minimum": float(reference_beta.min().item()),
            "maximum": float(reference_beta.max().item()),
            "mean": float(reference_beta.mean().item()),
            "quantile": float(beta_quantile),
            "threshold": threshold,
        }
        result = (threshold, stats)
        self._safe_threshold_cache[key] = result
        return result


def _pairwise_squared_distance(
    queries: torch.Tensor,
    references: torch.Tensor,
) -> torch.Tensor:
    query_norm = queries.square().sum(dim=1, keepdim=True)
    reference_norm = references.square().sum(dim=1).unsqueeze(0)
    distances = query_norm + reference_norm - 2.0 * queries @ references.T
    return distances.clamp_min_(0.0)


def _flatten_bank(
    current_x0: torch.Tensor,
    references: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    query = current_x0.to(dtype=torch.float32).reshape(current_x0.shape[0], -1)
    bank = references.to(
        device=current_x0.device,
        dtype=torch.float32,
    ).reshape(references.shape[0], -1)
    if query.shape[1] != bank.shape[1]:
        raise ValueError(
            f"Query/reference feature mismatch: {query.shape[1]} != {bank.shape[1]}"
        )
    return query, bank


def sgf_correct_x0(
    current_x0: torch.Tensor,
    bank: UnsafeReferenceBank,
    *,
    strength: float = 0.03,
    top_k: int = 3,
    epsilon: float = 0.05,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply the positive, normalized MMD repulsive field from SGF Eq. 7."""

    bank.validate_shape(current_x0)
    if not math.isfinite(float(strength)) or float(strength) < 0.0:
        raise ValueError("SGF strength must be finite and non-negative")
    if top_k <= 0 or bank.reference_count < top_k + 1:
        raise ValueError(
            f"SGF top_k={top_k} requires at least {top_k + 1} real references"
        )
    if not 0.0 < float(epsilon) < 1.0:
        raise ValueError("SGF epsilon must be in (0, 1)")

    query, references = _flatten_bank(current_x0, bank.references)
    nearest = torch.full(
        (query.shape[0], top_k + 1),
        float("inf"),
        device=query.device,
        dtype=torch.float32,
    )
    for start in range(0, bank.reference_count, REFERENCE_CHUNK_SIZE):
        chunk = references[start : start + REFERENCE_CHUNK_SIZE]
        distances = _pairwise_squared_distance(query, chunk)
        nearest = torch.topk(
            torch.cat((nearest, distances), dim=1),
            k=top_k + 1,
            dim=1,
            largest=False,
            sorted=True,
        ).values

    bandwidth_neighbors = nearest[:, 1 : top_k + 1]
    bandwidth_squared_distance = bandwidth_neighbors.mean().clamp_min(
        torch.finfo(torch.float32).tiny
    )
    gamma = -math.log(float(epsilon)) / bandwidth_squared_distance
    kernel_mass = torch.zeros(
        query.shape[0],
        1,
        device=query.device,
        dtype=torch.float32,
    )
    kernel_weighted_references = torch.zeros_like(query)
    for start in range(0, bank.reference_count, REFERENCE_CHUNK_SIZE):
        chunk = references[start : start + REFERENCE_CHUNK_SIZE]
        distances = _pairwise_squared_distance(query, chunk)
        kernels = torch.exp(-gamma * distances)
        kernel_mass.add_(kernels.sum(dim=1, keepdim=True))
        kernel_weighted_references.add_(kernels @ chunk)

    # grad_x[-2/N sum_i k(x,y_i)] = 4 gamma/N sum_i k(x,y_i)(x-y_i).
    summed_repulsion = query * kernel_mass - kernel_weighted_references
    field = (
        4.0
        * gamma
        * summed_repulsion
        / float(bank.reference_count)
    )
    correction = float(strength) * field
    corrected = (
        query + correction
    ).reshape_as(current_x0).to(dtype=current_x0.dtype)
    if not bool(torch.isfinite(corrected).all()):
        raise RuntimeError("SGF produced non-finite corrected clean estimates")

    detail = {
        "protocol": "sgf_paper_eq7_exact_chunked_v2",
        "paper_equation": "SGF Eq. 7 normalized MMD repulsive gradient",
        "repulsive_sign": "positive_away_from_unsafe_references",
        "reference_count": bank.reference_count,
        "effective_reference_population": bank.reference_count,
        "bandwidth_source": "published_appendix_d1_estimate_rbf_gamma",
        "self_distance_skipped": True,
        "skipped_sorted_neighbor_count": 1,
        "top_k_real_references": int(top_k),
        "bandwidth_neighbor_ranks_one_based": [2, int(top_k + 1)],
        "global_bandwidth": True,
        "epsilon": float(epsilon),
        "strength": float(strength),
        "reference_chunk_size": REFERENCE_CHUNK_SIZE,
        "bandwidth_squared_distance": float(
            bandwidth_squared_distance.detach().cpu()
        ),
        "gamma": float(gamma.detach().cpu()),
        "kernel_mass": [
            float(value) for value in kernel_mass[:, 0].detach().cpu()
        ],
        "correction_l2": [
            float(value)
            for value in torch.linalg.vector_norm(correction, dim=1).detach().cpu()
        ],
        "corrected_x0_stats": tensor_stats(corrected),
    }
    return corrected, detail


def safe_denoiser_correct_x0(
    current_x0: torch.Tensor,
    bank: UnsafeReferenceBank,
    *,
    sigma: float,
    scale: float,
    beta_quantile: float,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Apply Safe Denoiser Eq. 7 with the empirical beta switch."""

    bank.validate_shape(current_x0)
    if not math.isfinite(float(sigma)) or float(sigma) <= 0.0:
        raise ValueError("Safe Denoiser sigma must be finite and positive")
    if not math.isfinite(float(scale)) or float(scale) < 0.0:
        raise ValueError("Safe Denoiser eta must be finite and non-negative")
    if (
        not math.isfinite(float(beta_quantile))
        or float(beta_quantile) < 0.0
        or float(beta_quantile) > 1.0
    ):
        raise ValueError("Safe Denoiser beta_quantile must be in [0, 1]")

    query, raw_references = _flatten_bank(current_x0, bank.references)
    kernel_query, kernel_references = _flatten_bank(
        bank.safe_kernel_features(current_x0),
        bank.kernel_references,
    )
    kernel_mass = torch.zeros(
        query.shape[0],
        1,
        device=query.device,
        dtype=torch.float32,
    )
    kernel_weighted_references = torch.zeros_like(query)
    denominator_scale = 2.0 * float(sigma) * float(sigma)
    for start in range(0, bank.reference_count, REFERENCE_CHUNK_SIZE):
        kernel_chunk = kernel_references[start : start + REFERENCE_CHUNK_SIZE]
        raw_chunk = raw_references[start : start + REFERENCE_CHUNK_SIZE]
        distances = _pairwise_squared_distance(kernel_query, kernel_chunk)
        kernels = torch.exp(-distances / denominator_scale)
        kernel_mass.add_(kernels.sum(dim=1, keepdim=True))
        kernel_weighted_references.add_(kernels @ raw_chunk)

    beta = kernel_mass / float(bank.reference_count)
    threshold, threshold_stats = bank.safe_beta_threshold(
        sigma=float(sigma),
        beta_quantile=float(beta_quantile),
    )
    active = beta > threshold
    unsafe_denoiser = kernel_weighted_references / kernel_mass.clamp_min(
        torch.finfo(torch.float32).tiny
    )
    effective_beta = torch.where(active, beta, torch.zeros_like(beta))

    # E_safe = E_data + eta * beta(x_t) * (E_data - E_unsafe).
    correction = (
        float(scale)
        * effective_beta
        * (query - unsafe_denoiser)
    )
    corrected = (
        query + correction
    ).reshape_as(current_x0).to(dtype=current_x0.dtype)
    if not bool(torch.isfinite(corrected).all()):
        raise RuntimeError(
            "Safe Denoiser produced non-finite corrected clean estimates"
        )

    detail = {
        "protocol": "safe_denoiser_paper_eq6_eq7_switch_exact_chunked_v2",
        "paper_equation": (
            "E_safe = E_data + eta * beta(x_t) * "
            "(E_data - E_unsafe)"
        ),
        "reference_count": bank.reference_count,
        "effective_reference_population": bank.reference_count,
        "sigma": float(sigma),
        "eta": float(scale),
        "kernel_feature_policy": (
            "channel_l2_normalize_query_and_references_v3"
        ),
        "unsafe_denoiser_value_space": "raw_native_predicted_x0_latents",
        "reference_chunk_size": REFERENCE_CHUNK_SIZE,
        "beta": [float(value) for value in beta[:, 0].detach().cpu()],
        "beta_threshold": float(threshold),
        "beta_quantile": float(beta_quantile),
        "reference_beta_stats": threshold_stats,
        "effective_beta": [
            float(value) for value in effective_beta[:, 0].detach().cpu()
        ],
        "is_negation": bool(active.any().item()),
        "correction_l2": [
            float(value)
            for value in torch.linalg.vector_norm(correction, dim=1).detach().cpu()
        ],
        "corrected_x0_stats": tensor_stats(corrected),
    }
    return corrected, detail


@lru_cache(maxsize=None)
def _load_method_calibration(
    model_id: str,
    category: str,
    method: str,
) -> dict[str, Any]:
    if method not in SUPPORTED_METHODS:
        raise ValueError(f"Unsupported calibrated method {method!r}")
    path = (
        CALIBRATION_ROOT
        / "method_parameters_v3"
        / model_id
        / category
        / f"{method}.json"
    )
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing sealed per-model/category method calibration {path}"
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"Malformed method calibration {path}")
    if payload.get("protocol") != METHOD_CALIBRATION_PROTOCOL:
        raise RuntimeError(f"Method calibration protocol mismatch in {path}")
    if (
        payload.get("model_id") != model_id
        or payload.get("category") != category
        or payload.get("method") != method
    ):
        raise RuntimeError(f"Method calibration identity mismatch in {path}")
    if payload.get("status") != "accepted":
        raise RuntimeError(f"Method calibration was not admitted in {path}")

    num_steps = int(payload.get("num_inference_steps", 0))
    active_steps = payload.get("active_step_indices")
    parameters = payload.get("parameters")
    if num_steps <= 0 or not isinstance(active_steps, list) or not active_steps:
        raise RuntimeError(f"Method calibration has no native active steps in {path}")
    if not isinstance(parameters, dict):
        raise RuntimeError(f"Method calibration parameters are malformed in {path}")
    if (
        any(not isinstance(step, int) for step in active_steps)
        or active_steps != sorted(set(active_steps))
        or active_steps[0] < 0
        or active_steps[-1] >= num_steps
    ):
        raise RuntimeError(f"Method calibration active steps are invalid in {path}")

    _require_sha256(
        payload.get("scheduler_grid_sha256"),
        field="scheduler_grid_sha256",
        path=path,
    )
    if method == "midsteer":
        _require_sha256(
            payload.get("midsteer_artifact_sha256"),
            field="midsteer_artifact_sha256",
            path=path,
        )
    else:
        _require_sha256(
            payload.get("reference_bank_sha256"),
            field="reference_bank_sha256",
            path=path,
        )
    _require_sha256(
        payload.get("pilot_manifest_sha256"),
        field="pilot_manifest_sha256",
        path=path,
    )
    if not str(payload.get("selection_rule", "")).strip():
        raise RuntimeError(f"Method calibration selection rule is missing in {path}")

    if method == "midsteer":
        required = {"strength"}
    elif method == "sgf":
        required = {"strength", "top_k", "epsilon"}
    else:
        required = {"sigma", "eta", "beta_quantile"}
    if not required.issubset(parameters):
        raise RuntimeError(
            f"Method calibration lacks {sorted(required - set(parameters))} in {path}"
        )

    return {
        **payload,
        "active_step_indices": tuple(active_steps),
        "parameters": dict(parameters),
        "artifact_path": str(path),
        "artifact_sha256": file_sha256(path),
    }


def _runtime_calibration(
    *,
    model_id: str,
    category: str,
    method: str,
    pilot_candidate: MethodPilotCandidate | None,
) -> dict[str, Any]:
    if pilot_candidate is None:
        return _load_method_calibration(model_id, category, method)
    pilot_candidate.require_identity(
        model_id=model_id,
        category=category,
        method=method,
    )
    return pilot_candidate.calibration_payload()


def validate_related_work_runtime_grid(
    *,
    model_id: str,
    method: str,
    bank: UnsafeReferenceBank,
    timesteps: Any,
    pilot_candidate: MethodPilotCandidate | None = None,
) -> dict[str, Any]:
    calibration = _runtime_calibration(
        model_id=model_id,
        category=bank.category,
        method=method,
        pilot_candidate=pilot_candidate,
    )
    if calibration.get("reference_bank_sha256") != bank.metadata.get("artifact_sha256"):
        raise RuntimeError("Method calibration is not bound to the loaded reference bank")
    num_steps = int(calibration["num_inference_steps"])
    scheduler_path, expected = load_scheduler_grid_binding(
        model_id=model_id,
        num_steps=num_steps,
        path_value=calibration["scheduler_grid_path"],
        sha256_value=calibration["scheduler_grid_sha256"],
    )
    observed = assert_runtime_timesteps(
        expected=expected,
        observed=timesteps,
        source=scheduler_path,
    )
    return {
        "scheduler_grid_path": str(scheduler_path),
        "scheduler_grid_sha256": calibration["scheduler_grid_sha256"],
        "native_timesteps": list(observed),
        "calibration_sha256": calibration["artifact_sha256"],
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
    sgf_strength: float = 0.03,
    safe_sigma: float | None = None,
    safe_scale: float | None = None,
    pilot_candidate: MethodPilotCandidate | None = None,
) -> tuple[torch.Tensor | None, dict[str, Any]]:
    """Replace the native prediction in place; never re-noise corrected x0."""

    del generator
    calibration = _runtime_calibration(
        model_id=model_id,
        category=bank.category,
        method=method,
        pilot_candidate=pilot_candidate,
    )
    if int(calibration["num_inference_steps"]) != int(num_steps):
        raise RuntimeError(
            "Method calibration denoising grid does not match this run: "
            f"{calibration['num_inference_steps']} != {num_steps}"
        )
    if (
        str(calibration["reference_bank_sha256"])
        != str(bank.metadata["artifact_sha256"])
    ):
        raise RuntimeError(
            "Method calibration is not bound to the loaded reference bank"
        )

    canonical = canonicalize_prediction(
        adapter=adapter,
        model_id=model_id,
        latents=latents,
        native=native_prediction,
        timestep=timestep,
        step_index=step_index,
        guidance_scale=guidance_scale,
        branch="guided",
    )
    eligible = int(step_index) in calibration["active_step_indices"]
    metadata: dict[str, Any] = {
        "protocol": "paper_exact_related_work_transition_v3",
        "method": method,
        "eligible": eligible,
        "step_index": int(step_index),
        "num_inference_steps": int(num_steps),
        "native_timestep": canonical.schedule.timestep,
        "native_sigma": canonical.schedule.sigma,
        "native_alpha": canonical.schedule.alpha,
        "native_schedule_source": canonical.schedule.source,
        "solver_target": "corrected_x0_converted_to_native_prediction",
        "fresh_renoising": False,
        "old_prediction_reused": False,
        "calibration_path": calibration["artifact_path"],
        "calibration_sha256": calibration["artifact_sha256"],
        "scheduler_grid_sha256": calibration["scheduler_grid_sha256"],
        "reference_bank_sha256": calibration["reference_bank_sha256"],
        "active_step_indices": list(calibration["active_step_indices"]),
        "calibration_source": "accepted_production_calibration"
        if pilot_candidate is None
        else "sealed_pilot_candidate",
        "requested_legacy_values": {
            "sgf_strength": float(sgf_strength),
            "safe_sigma": None if safe_sigma is None else float(safe_sigma),
            "safe_scale": None if safe_scale is None else float(safe_scale),
        },
    }
    if not eligible:
        metadata["active"] = False
        return None, metadata

    parameters = calibration["parameters"]
    if method == "sgf":
        corrected_x0, detail = sgf_correct_x0(
            canonical.predicted_x0,
            bank,
            strength=float(parameters["strength"]),
            top_k=int(parameters["top_k"]),
            epsilon=float(parameters["epsilon"]),
        )
    elif method == "safe_denoiser":
        corrected_x0, detail = safe_denoiser_correct_x0(
            canonical.predicted_x0,
            bank,
            sigma=float(parameters["sigma"]),
            scale=float(parameters["eta"]),
            beta_quantile=float(parameters["beta_quantile"]),
        )
    else:
        raise ValueError(f"Unsupported related-work method {method!r}")

    replacement = prediction_from_x0(
        latents,
        corrected_x0,
        canonical.parameterization,
        canonical.schedule,
    )
    if replacement.shape != native_prediction.shape:
        raise RuntimeError(
            "Corrected native prediction shape changed: "
            f"{tuple(replacement.shape)} != {tuple(native_prediction.shape)}"
        )
    if not bool(torch.isfinite(replacement).all()):
        raise RuntimeError("Corrected native prediction contains non-finite values")
    native_prediction.copy_(
        replacement.to(
            device=native_prediction.device,
            dtype=native_prediction.dtype,
        )
    )
    metadata["native_prediction_replaced_in_place"] = True
    metadata["active"] = True if method == "sgf" else bool(detail["is_negation"])
    metadata["detail"] = detail
    metadata["replacement_stats"] = tensor_stats(native_prediction)
    return None, metadata


__all__ = [
    "UnsafeReferenceBank",
    "related_work_transition",
    "safe_denoiser_correct_x0",
    "sgf_correct_x0",
    "validate_related_work_runtime_grid",
]
