"""Unified state machine and vector-field math for the 13 ConceptSteer ablations."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Mapping, Optional, Sequence

import torch

from hierasafe_flow.steering.local_masks import (
    MaskConfig,
    activation_to_mask,
    broadcast_mask_to_vector_field,
)
from hierasafe_flow.steering.vector_fields import local_unsafe_activation


class AblationContractError(RuntimeError):
    """Raised instead of silently changing an ablation or timestep contract."""


class AblationMode(str, Enum):
    BASELINE = "A00_BASELINE"
    GENERIC_SUFFIX = "A01_GENERIC_POSITIVE_SUFFIX"
    CONFIG_SUFFIX = "A02_CONFIG_POSITIVE_SUFFIX"
    NO_NEUTRAL_AR = "A03_NO_NEUTRAL_ALL_STEP_RECOMPUTE"
    PULSE_K0 = "A04_SINGLE_PULSE_K0"
    PULSE_K2 = "A05_SINGLE_PULSE_K2"
    PULSE_K5 = "A06_SINGLE_PULSE_K5"
    FROZEN_K0 = "A07_FROZEN_TAIL_K0"
    FROZEN_K2 = "A08_FROZEN_TAIL_K2"
    FROZEN_K5 = "A09_FROZEN_TAIL_K5"
    EARLY_STOP = "A10_EARLY_SCHEDULE_STOP"
    EARLY_FROZEN = "A11_EARLY_SCHEDULE_FROZEN_TAIL"
    ALL_RECOMPUTE = "A12_ALL_STEP_RECOMPUTE"


ABLATION_IDS = tuple(mode.value for mode in AblationMode)


@dataclass
class FrozenDirectionState:
    direction_unit_rms: Optional[torch.Tensor] = None
    acquired_step: Optional[int] = None
    acquired_timestep: Optional[float] = None
    acquired_unified_time: Optional[float] = None
    acquired_sigma: Optional[float] = None
    fingerprint: Optional[str] = None
    direction_rms: Optional[float] = None


@dataclass(frozen=True)
class UnifiedTimeMap:
    raw_timesteps: tuple[float, ...]
    unified_times: tuple[float, ...]
    conversion_policy: str
    scale: float


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AblationContractError(message)


def scalar_timestep(value: Any) -> float:
    if torch.is_tensor(value):
        _require(value.numel() == 1, f"Timestep must be scalar, got shape {tuple(value.shape)}")
        value = value.detach().float().cpu().item()
    result = float(value)
    _require(math.isfinite(result), f"Non-finite timestep: {result}")
    return result


def build_unified_time_map(timesteps: Sequence[Any]) -> UnifiedTimeMap:
    """Map a native scalar time coordinate to the declared 0..1000 diffusion axis.

    The mapping accepts only a native 0..1000 coordinate or an explicitly
    normalized 0..1 coordinate. It never substitutes denoising-step fraction.
    """
    raw = tuple(scalar_timestep(value) for value in timesteps)
    _require(bool(raw), "Scheduler returned no timesteps")
    _require(
        all(left >= right for left, right in zip(raw, raw[1:])),
        f"Timesteps are not monotonically non-increasing: {raw}",
    )
    minimum, maximum = min(raw), max(raw)
    _require(minimum >= -1.0e-6, f"Negative native timestep is unsupported: {minimum}")
    if maximum >= 800.0 and maximum <= 1000.0 + 1.0e-3:
        scale = 1.0
        policy = "native_scalar_0_to_1000"
    elif maximum <= 1.0 + 1.0e-6 and maximum >= 0.8:
        scale = 1000.0
        policy = "native_normalized_scalar_0_to_1_scaled_to_1000"
    else:
        raise AblationContractError(
            "Cannot establish an exact unified diffusion-time map from native "
            f"timesteps (min={minimum}, max={maximum}); step-fraction fallback is forbidden"
        )
    return UnifiedTimeMap(
        raw_timesteps=raw,
        unified_times=tuple(value * scale for value in raw),
        conversion_policy=policy,
        scale=scale,
    )


class AblationController:
    """Separates direction computation, application, capture, and reuse."""

    _PULSE_STEPS = {
        AblationMode.PULSE_K0: 0,
        AblationMode.PULSE_K2: 2,
        AblationMode.PULSE_K5: 5,
    }
    _FROZEN_STEPS = {
        AblationMode.FROZEN_K0: 0,
        AblationMode.FROZEN_K2: 2,
        AblationMode.FROZEN_K5: 5,
    }

    def __init__(
        self,
        mode: AblationMode | str,
        unified_times: Sequence[float],
        *,
        early_window: tuple[float, float] = (1000.0, 800.0),
    ) -> None:
        self.mode = mode if isinstance(mode, AblationMode) else AblationMode(mode)
        self.unified_times = tuple(float(value) for value in unified_times)
        _require(bool(self.unified_times), "Controller requires at least one denoising step")
        high, low = map(float, early_window)
        _require(high >= low, f"Invalid early window: {early_window}")
        self.early_window = (high, low)
        self._early_active = tuple(
            index for index, value in enumerate(self.unified_times) if low <= value <= high
        )
        if self.mode in {AblationMode.EARLY_STOP, AblationMode.EARLY_FROZEN}:
            _require(bool(self._early_active), "Early schedule has no active native timesteps")
            _require(
                self._early_active == tuple(range(self._early_active[-1] + 1)),
                "Early window must be a contiguous prefix on a descending native schedule",
            )
        acquisition = self.acquisition_step
        if acquisition is not None:
            _require(acquisition < len(self.unified_times), f"Acquisition step {acquisition} absent")

    @property
    def acquisition_step(self) -> int | None:
        return self._PULSE_STEPS.get(self.mode, self._FROZEN_STEPS.get(self.mode))

    @property
    def requires_concept_fields(self) -> bool:
        return self.mode not in {
            AblationMode.BASELINE,
            AblationMode.GENERIC_SUFFIX,
            AblationMode.CONFIG_SUFFIX,
        }

    @property
    def requires_neutral(self) -> bool:
        return self.requires_concept_fields and self.mode is not AblationMode.NO_NEUTRAL_AR

    @property
    def last_nonzero_schedule_step(self) -> int | None:
        if self.mode in {AblationMode.EARLY_STOP, AblationMode.EARLY_FROZEN}:
            return self._early_active[-1]
        return None

    def _check_index(self, step: int) -> None:
        _require(0 <= step < len(self.unified_times), f"Step index out of range: {step}")

    def should_compute_direction(self, step: int) -> bool:
        self._check_index(step)
        if self.mode in {AblationMode.NO_NEUTRAL_AR, AblationMode.ALL_RECOMPUTE}:
            return True
        if self.mode in self._PULSE_STEPS:
            return step == self._PULSE_STEPS[self.mode]
        if self.mode in self._FROZEN_STEPS:
            return step == self._FROZEN_STEPS[self.mode]
        if self.mode in {AblationMode.EARLY_STOP, AblationMode.EARLY_FROZEN}:
            return step in self._early_active
        return False

    def should_apply_direction(self, step: int) -> bool:
        self._check_index(step)
        if self.mode in {AblationMode.NO_NEUTRAL_AR, AblationMode.ALL_RECOMPUTE}:
            return True
        if self.mode in self._PULSE_STEPS:
            return step == self._PULSE_STEPS[self.mode]
        if self.mode in self._FROZEN_STEPS:
            return step >= self._FROZEN_STEPS[self.mode]
        if self.mode is AblationMode.EARLY_STOP:
            return step in self._early_active
        if self.mode is AblationMode.EARLY_FROZEN:
            return step >= self._early_active[0]
        return False

    def current_schedule_weight(self, step: int) -> float:
        self._check_index(step)
        if self.mode in {AblationMode.EARLY_STOP, AblationMode.EARLY_FROZEN}:
            return 1.0 if step in self._early_active else 0.0
        return 1.0 if self.should_apply_direction(step) else 0.0

    def current_application_weight(self, step: int) -> float:
        return 1.0 if self.should_apply_direction(step) else 0.0

    def should_capture_frozen_direction(self, step: int) -> bool:
        self._check_index(step)
        if self.mode in self._FROZEN_STEPS:
            return step == self._FROZEN_STEPS[self.mode]
        if self.mode is AblationMode.EARLY_FROZEN:
            return step == self._early_active[-1]
        return False

    def should_use_frozen_direction(self, step: int) -> bool:
        self._check_index(step)
        if self.mode in self._FROZEN_STEPS:
            return step > self._FROZEN_STEPS[self.mode]
        if self.mode is AblationMode.EARLY_FROZEN:
            return step > self._early_active[-1]
        return False


def tensor_rms(value: torch.Tensor, eps: float = 1.0e-12) -> torch.Tensor:
    return value.float().square().mean().add(float(eps)).sqrt()


def normalize_direction_rms(
    direction: torch.Tensor, eps: float = 1.0e-12
) -> torch.Tensor:
    return direction.float() / tensor_rms(direction, eps)


def apply_relative_direction(
    base: torch.Tensor,
    unit_rms_direction: torch.Tensor,
    *,
    relative_strength: float,
) -> torch.Tensor:
    if float(relative_strength) == 0.0:
        return base
    delta = (
        float(relative_strength)
        * tensor_rms(base)
        * unit_rms_direction.to(device=base.device, dtype=torch.float32)
    )
    return (base.float() + delta).to(dtype=base.dtype)


def tensor_fingerprint(value: torch.Tensor) -> str:
    payload = value.detach().to(device="cpu", dtype=torch.float32).contiguous().numpy().tobytes()
    return hashlib.sha256(payload).hexdigest()


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    finite = torch.isfinite(value)
    cast = value.detach().float()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "finite_fraction": float(finite.float().mean().cpu()),
        "minimum": float(cast.min().cpu()),
        "maximum": float(cast.max().cpu()),
        "mean": float(cast.mean().cpu()),
        "rms": float(tensor_rms(cast).cpu()),
    }


def conceptsteer_direction(
    *,
    v_base: torch.Tensor,
    v_unsafe: torch.Tensor,
    v_safe: torch.Tensor,
    v_neutral: torch.Tensor,
    feature_dim: int,
    margin: float,
    mask_config: MaskConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    unsafe_basis = v_unsafe - v_neutral
    safe_basis = v_safe - v_neutral
    activation = local_unsafe_activation(
        v_base,
        unsafe_basis,
        safe_basis,
        feature_dim=feature_dim,
        margin=float(margin),
        eps=mask_config.eps,
    )
    mask = activation_to_mask(activation, mask_config)
    mask = broadcast_mask_to_vector_field(mask, v_base)
    direction = mask.to(v_base.dtype) * (safe_basis - unsafe_basis)
    return direction, {
        "neutral_used": True,
        "activation_stats": tensor_stats(activation),
        "mask_stats": tensor_stats(mask),
        "unsafe_basis_stats": tensor_stats(unsafe_basis),
        "safe_basis_stats": tensor_stats(safe_basis),
        "direction_stats": tensor_stats(direction),
    }


def compute_no_neutral_direction(
    v_base: torch.Tensor,
    v_unsafe: torch.Tensor,
    v_safe: torch.Tensor,
    *,
    feature_dim: int,
    margin: float,
    mask_config: MaskConfig,
) -> tuple[torch.Tensor, dict[str, Any]]:
    midpoint = 0.5 * (v_unsafe + v_safe)
    unsafe_basis = v_unsafe - midpoint
    safe_basis = v_safe - midpoint
    residual = v_base - midpoint
    activation = local_unsafe_activation(
        residual,
        unsafe_basis,
        safe_basis,
        feature_dim=feature_dim,
        margin=float(margin),
        eps=mask_config.eps,
    )
    mask = activation_to_mask(activation, mask_config)
    mask = broadcast_mask_to_vector_field(mask, v_base)
    direction = mask.to(v_base.dtype) * (safe_basis - unsafe_basis)
    return direction, {
        "neutral_used": False,
        "midpoint_stats": tensor_stats(midpoint),
        "activation_stats": tensor_stats(activation),
        "mask_stats": tensor_stats(mask),
        "direction_stats": tensor_stats(direction),
    }


def weighted_prediction_mean(
    conditions: Sequence[Any],
    weights: Sequence[float],
    predict: Callable[[Any], torch.Tensor],
) -> torch.Tensor:
    _require(len(conditions) == len(weights) and bool(conditions), "Invalid weighted probes")
    total_weight = float(sum(float(weight) for weight in weights))
    _require(total_weight > 0.0, "Probe weights must have a positive sum")
    accumulator: torch.Tensor | None = None
    output_dtype: torch.dtype | None = None
    for condition, weight in zip(conditions, weights):
        prediction = predict(condition)
        _require(bool(torch.isfinite(prediction).all()), "Non-finite concept prediction")
        output_dtype = prediction.dtype
        contribution = prediction.float() * float(weight)
        accumulator = contribution if accumulator is None else accumulator + contribution
    _require(accumulator is not None and output_dtype is not None, "No predictions accumulated")
    return (accumulator / total_weight).to(dtype=output_dtype)


def intervention_energy(base: torch.Tensor, delta: torch.Tensor, eps: float = 1.0e-12) -> float:
    numerator = delta.float().square().sum()
    denominator = base.float().square().sum().add(float(eps))
    return float((numerator / denominator).detach().cpu())


def mask_config_from_mapping(value: Mapping[str, Any]) -> MaskConfig:
    return MaskConfig(
        enabled=bool(value["enabled"]),
        mode=str(value["mode"]),
        threshold=float(value["threshold"]),
        percentile=float(value["percentile"]),
        eps=float(value["eps"]),
    )
