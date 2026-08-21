"""A fail-closed prediction contract shared by the July steering campaign.

The contract never changes scheduler stepping. It exposes every model's native
prediction together with an analytically derived clean sample and a canonical
``dx/dsigma`` direction. Controllers edit the clean sample and convert it back
to the model's native parameterization before the frozen scheduler sees it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any

import torch


class NativeParameterization(str, Enum):
    FLOW = "flow_velocity"
    PHYSICAL_VELOCITY = "physical_velocity"
    EPSILON = "epsilon"
    V_PREDICTION = "v_prediction"
    CLEAN_SAMPLE = "clean_sample"


MODEL_PARAMETERIZATIONS: dict[str, NativeParameterization] = {
    "cosmos3_super_text2image": NativeParameterization.FLOW,
    "flux1_dev": NativeParameterization.FLOW,
    "flux2_dev": NativeParameterization.FLOW,
    "ideogram4_nf4": NativeParameterization.PHYSICAL_VELOCITY,
    "qwen_image": NativeParameterization.FLOW,
    "qwen_image_2512": NativeParameterization.FLOW,
    "sd35_large": NativeParameterization.FLOW,
    "cogvideox_5b": NativeParameterization.V_PREDICTION,
    "hunyuan_video": NativeParameterization.FLOW,
    "joyai_echo": NativeParameterization.CLEAN_SAMPLE,
    "ltx_23": NativeParameterization.FLOW,
    "wan22_t2v_a14b": NativeParameterization.FLOW,
}


@dataclass(frozen=True)
class SchedulePoint:
    sigma: float
    alpha: float
    step_index: int
    timestep: float
    source: str


@dataclass
class CanonicalPrediction:
    native: torch.Tensor
    predicted_x0: torch.Tensor
    derivative: torch.Tensor
    parameterization: NativeParameterization
    schedule: SchedulePoint
    guidance_scale: float
    layout: dict[str, Any]
    branch: str

    def metadata(self) -> dict[str, Any]:
        return {
            "parameterization": self.parameterization.value,
            "schedule": asdict(self.schedule),
            "guidance_scale": self.guidance_scale,
            "layout": self.layout,
            "branch": self.branch,
        }


def parameterization_for_model(model_id: str) -> NativeParameterization:
    try:
        return MODEL_PARAMETERIZATIONS[model_id]
    except KeyError as exc:
        raise ValueError(f"No audited native parameterization for {model_id!r}") from exc


def _scheduler(adapter: Any) -> Any | None:
    for owner in (adapter, getattr(adapter, "pipeline", None), getattr(adapter, "pipe", None)):
        scheduler = getattr(owner, "scheduler", None) if owner is not None else None
        if scheduler is not None:
            return scheduler
    return None


def _float_timestep(timestep: Any) -> float:
    if torch.is_tensor(timestep):
        return float(timestep.detach().flatten()[0].cpu())
    return float(timestep)


def schedule_point(adapter: Any, timestep: Any, step_index: int) -> SchedulePoint:
    """Resolve sigma/alpha from the adapter's exact frozen scheduler state."""

    scheduler = _scheduler(adapter)
    sigma: float | None = None
    alpha: float | None = None
    source = "fallback"
    if scheduler is not None:
        sigmas = getattr(scheduler, "sigmas", None)
        if sigmas is not None and len(sigmas) > step_index:
            sigma = float(torch.as_tensor(sigmas[step_index]).detach().cpu())
            source = "scheduler.sigmas"
        alphas_cumprod = getattr(scheduler, "alphas_cumprod", None)
        if alphas_cumprod is not None:
            t = int(round(_float_timestep(timestep)))
            t = max(0, min(t, len(alphas_cumprod) - 1))
            alpha_bar = float(torch.as_tensor(alphas_cumprod[t]).detach().cpu())
            alpha = max(alpha_bar, 0.0) ** 0.5
            if sigma is None:
                sigma = max(1.0 - alpha_bar, 0.0) ** 0.5
                source = "scheduler.alphas_cumprod"
    if sigma is None:
        raw = abs(_float_timestep(timestep))
        sigma = raw / 1000.0 if raw > 1.0 else raw
    if alpha is None:
        alpha = max(1.0 - min(sigma, 1.0) ** 2, 0.0) ** 0.5
    return SchedulePoint(
        sigma=max(float(sigma), 1.0e-6),
        alpha=max(float(alpha), 1.0e-6),
        step_index=int(step_index),
        timestep=_float_timestep(timestep),
        source=source,
    )


def _layout_dict(adapter: Any, prediction: torch.Tensor) -> dict[str, Any]:
    try:
        layout = adapter.latent_layout(prediction)
    except Exception:
        layout = None
    if layout is None:
        return {"shape": list(prediction.shape)}
    if is_dataclass(layout):
        value = asdict(layout)
    elif hasattr(layout, "__dict__"):
        value = dict(vars(layout))
    else:
        value = {"value": str(layout)}
    value["shape"] = list(prediction.shape)
    return value


def predicted_x0(
    latents: torch.Tensor,
    native: torch.Tensor,
    kind: NativeParameterization,
    point: SchedulePoint,
) -> torch.Tensor:
    sigma = point.sigma
    alpha = point.alpha
    if kind is NativeParameterization.FLOW:
        return latents - sigma * native
    if kind is NativeParameterization.PHYSICAL_VELOCITY:
        return latents + sigma * native
    if kind is NativeParameterization.EPSILON:
        return (latents - sigma * native) / alpha
    if kind is NativeParameterization.V_PREDICTION:
        return alpha * latents - sigma * native
    if kind is NativeParameterization.CLEAN_SAMPLE:
        return native
    raise AssertionError(kind)


def prediction_from_x0(
    latents: torch.Tensor,
    x0: torch.Tensor,
    kind: NativeParameterization,
    point: SchedulePoint,
) -> torch.Tensor:
    sigma = point.sigma
    alpha = point.alpha
    if kind is NativeParameterization.FLOW:
        return (latents - x0) / sigma
    if kind is NativeParameterization.PHYSICAL_VELOCITY:
        return (x0 - latents) / sigma
    if kind is NativeParameterization.EPSILON:
        return (latents - alpha * x0) / sigma
    if kind is NativeParameterization.V_PREDICTION:
        return (alpha * latents - x0) / sigma
    if kind is NativeParameterization.CLEAN_SAMPLE:
        return x0
    raise AssertionError(kind)


def canonical_derivative(
    latents: torch.Tensor,
    x0: torch.Tensor,
    native: torch.Tensor,
    kind: NativeParameterization,
    point: SchedulePoint,
) -> torch.Tensor:
    if kind is NativeParameterization.FLOW:
        return native
    if kind is NativeParameterization.PHYSICAL_VELOCITY:
        return -native
    return (latents - x0) / point.sigma


def canonicalize_prediction(
    *,
    adapter: Any,
    model_id: str,
    latents: torch.Tensor,
    native: torch.Tensor,
    timestep: Any,
    step_index: int,
    guidance_scale: float,
    branch: str,
) -> CanonicalPrediction:
    kind = parameterization_for_model(model_id)
    point = schedule_point(adapter, timestep, step_index)
    x0 = predicted_x0(latents, native, kind, point)
    derivative = canonical_derivative(latents, x0, native, kind, point)
    return CanonicalPrediction(
        native=native,
        predicted_x0=x0,
        derivative=derivative,
        parameterization=kind,
        schedule=point,
        guidance_scale=float(guidance_scale),
        layout=_layout_dict(adapter, native),
        branch=branch,
    )


def infer_channel_dim(tensor: torch.Tensor, model_id: str, layout: dict[str, Any] | None = None) -> int:
    layout = layout or {}
    for key in ("feature_dim", "channel_dim", "channel_axis", "channels_dim", "channels_axis"):
        if key in layout and layout[key] is not None:
            value = int(layout[key])
            return value if value >= 0 else tensor.ndim + value
    if tensor.ndim == 5 and model_id == "cogvideox_5b":
        return 2
    return 1


def channel_tokens(
    tensor: torch.Tensor,
    model_id: str,
    layout: dict[str, Any] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Move channels last and flatten all sample/token axes without losing shape."""

    channel_dim = infer_channel_dim(tensor, model_id, layout)
    moved = tensor.movedim(channel_dim, -1)
    tokens = moved.reshape(-1, moved.shape[-1])
    return tokens, {
        "original_shape": list(tensor.shape),
        "moved_shape": list(moved.shape),
        "channel_dim": channel_dim,
    }


def restore_channel_tokens(tokens: torch.Tensor, metadata: dict[str, Any]) -> torch.Tensor:
    moved = tokens.reshape(metadata["moved_shape"])
    return moved.movedim(-1, int(metadata["channel_dim"]))
