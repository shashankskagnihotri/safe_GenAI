"""MidSteer on complete attention outputs after their output projection.

This module follows Atmyre/MidSteer commit
0f3b31e15cdda6ad0d46167e10319e896d6f1541.  The default upstream
``attn_output`` mode registers forward hooks on complete attention modules,
adds one singleton feature-group dimension, calibrates the full covariance at
the first diffusion step, and reuses that transform at later diffusion steps.
It is intentionally schema-incompatible with the rejected pre-projection,
per-head implementation in ``midsteer_full.py``.
"""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import torch


MIDSTEER_REVISION = "0f3b31e15cdda6ad0d46167e10319e896d6f1541"
MIDSTEER_SCHEMA_VERSION = 4
MIDSTEER_CONTROL_MODE = "attn_output_post_projection"
MIDSTEER_STEP_POLICY = "first_diffusion_step_reused"
MIDSTEER_TOKEN_SCOPE = "admitted_model_topology_image_token_suffix_v2"


def fractional_matrix_power_covariance(
    matrix: torch.Tensor, power: float
) -> torch.Tensor:
    """Apply the pinned upstream covariance eigenthreshold and pseudo-power."""

    matrix = matrix.to(dtype=torch.float64)
    eigenvalues, eigenvectors = torch.linalg.eigh(matrix)
    threshold = (
        eigenvalues[..., -1:]
        * matrix.shape[-1]
        * torch.finfo(eigenvalues.dtype).eps
    )
    keep = eigenvalues > threshold
    powered = torch.where(
        keep,
        eigenvalues.clamp_min(0).pow(power),
        torch.zeros_like(eigenvalues),
    )
    return eigenvectors @ torch.diag_embed(powered) @ eigenvectors.mT


def resolve_transformer_root(adapter: Any) -> torch.nn.Module:
    pipeline = getattr(adapter, "pipeline", None)
    if pipeline is None:
        raise RuntimeError("MidSteer requires a loaded adapter pipeline")
    for name in ("transformer", "unet"):
        value = getattr(pipeline, name, None)
        if isinstance(value, torch.nn.Module):
            return value
    raise RuntimeError("MidSteer requires a transformer or UNet root")


@dataclass(frozen=True)
class AttentionOutputSite:
    name: str
    module: torch.nn.Module


def _is_attention_module(module: torch.nn.Module) -> bool:
    query = any(
        isinstance(getattr(module, name, None), torch.nn.Module)
        for name in ("to_q", "q_proj", "query")
    )
    output = any(
        isinstance(getattr(module, name, None), torch.nn.Module)
        or isinstance(getattr(module, name, None), (torch.nn.ModuleList, torch.nn.Sequential))
        for name in ("to_out", "to_add_out", "out_proj", "o_proj")
    )
    class_name = module.__class__.__name__.lower()
    return query and (output or "attention" in class_name)


def attention_output_sites(root: torch.nn.Module) -> list[AttentionOutputSite]:
    """Resolve complete attention modules, never their internal projections."""

    sites: list[AttentionOutputSite] = []
    seen: set[int] = set()
    for name, module in root.named_modules():
        if not name or id(module) in seen or not _is_attention_module(module):
            continue
        seen.add(id(module))
        sites.append(AttentionOutputSite(name=name, module=module))
    if not sites:
        raise RuntimeError("MidSteer found no complete attention modules")
    return sites


def _primary_output(
    output: Any,
) -> tuple[torch.Tensor, Callable[[torch.Tensor], Any]]:
    if isinstance(output, torch.Tensor):
        return output, lambda replacement: replacement
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        rest = output[1:]
        return output[0], lambda replacement: (replacement, *rest)
    if isinstance(output, list) and output and isinstance(output[0], torch.Tensor):
        rest = output[1:]
        return output[0], lambda replacement: [replacement, *rest]
    raise TypeError(
        "MidSteer attention output must be a tensor or a tuple/list whose first "
        f"item is a tensor, observed {type(output).__name__}"
    )


def _output_tokens(value: torch.Tensor) -> torch.Tensor:
    if value.ndim < 2:
        raise ValueError(
            f"MidSteer attention output requires a feature axis: {tuple(value.shape)}"
        )
    return value.reshape(-1, 1, value.shape[-1])


def _scoped_attention_output(
    value: torch.Tensor,
    *,
    site_name: str,
    image_token_count: int | None,
) -> tuple[torch.Tensor, int]:
    """Preserve an admitted text prefix and expose only image-token outputs."""

    if image_token_count is None:
        return value, 0
    if image_token_count <= 0:
        raise ValueError("MidSteer image_token_count must be positive")
    if value.ndim < 3:
        raise ValueError(
            f"MidSteer packed output requires a sequence axis: {tuple(value.shape)}"
        )
    total_tokens = int(value.shape[-2])
    if image_token_count > total_tokens:
        raise ValueError(
            "MidSteer image-token count exceeds the attention-output sequence: "
            f"{image_token_count} > {total_tokens} at {site_name}"
        )
    preserved_prefix_tokens = total_tokens - image_token_count
    return value[..., preserved_prefix_tokens:, :], preserved_prefix_tokens


def merge_midsteer_moments(
    left: dict[str, Any] | None,
    right: dict[str, Any],
) -> dict[str, Any]:
    required = {"count", "sum"}
    if not required.issubset(right):
        raise ValueError(f"Incomplete MidSteer moments: {sorted(right)}")
    if left is None:
        result = {
            "count": int(right["count"]),
            "sum": right["sum"].clone(),
        }
        if "sum_outer" in right:
            result["sum_outer"] = right["sum_outer"].clone()
        return result
    if ("sum_outer" in left) != ("sum_outer" in right):
        raise ValueError("Cannot merge MidSteer moments with different covariance scope")
    if left["sum"].shape != right["sum"].shape:
        raise ValueError("Cannot merge MidSteer moments with different output widths")
    result = {
        "count": int(left["count"]) + int(right["count"]),
        "sum": left["sum"] + right["sum"],
    }
    if "sum_outer" in left:
        result["sum_outer"] = left["sum_outer"] + right["sum_outer"]
    return result


class AttentionOutputMomentCapture(
    AbstractContextManager["AttentionOutputMomentCapture"]
):
    """Capture exact CPU float64 moments of post-projection attention outputs."""

    def __init__(
        self,
        root: torch.nn.Module,
        *,
        include_covariance: bool,
        conditional_only: bool = True,
        image_token_count: int | None = None,
        token_aggregation: str = "all",
    ) -> None:
        if token_aggregation not in {"all", "average"}:
            raise ValueError(
                "MidSteer token_aggregation must be 'all' or 'average'"
            )
        self.sites = attention_output_sites(root)
        self.include_covariance = include_covariance
        self.conditional_only = conditional_only
        self.image_token_count = image_token_count
        self.token_aggregation = token_aggregation
        self.handles: list[Any] = []
        self.values: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> "AttentionOutputMomentCapture":
        for site in self.sites:

            def hook(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                _site: AttentionOutputSite = site,
            ) -> Any:
                value, _ = _primary_output(output)
                if (
                    self.conditional_only
                    and value.ndim >= 2
                    and int(value.shape[0]) == 2
                ):
                    value = value[1:]
                scoped, _ = _scoped_attention_output(
                    value,
                    site_name=_site.name,
                    image_token_count=self.image_token_count,
                )
                tokens = _output_tokens(scoped)
                if self.token_aggregation == "average":
                    tokens = tokens.mean(dim=0, keepdim=True)
                work = tokens.detach().to(device="cpu", dtype=torch.float64)
                moments: dict[str, Any] = {
                    "count": int(work.shape[0]),
                    "sum": work.sum(dim=0),
                }
                if self.include_covariance:
                    moments["sum_outer"] = torch.einsum(
                        "ngd,nge->gde", work, work
                    )
                self.values[_site.name] = merge_midsteer_moments(
                    self.values.get(_site.name), moments
                )
                return output

            self.handles.append(site.module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def moments(self) -> dict[str, dict[str, Any]]:
        if not self.values:
            raise RuntimeError("MidSteer captured no post-projection attention outputs")
        return self.values


def _neutral_geometry(
    neutral: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    if "sum_outer" not in neutral:
        raise ValueError("Neutral MidSteer moments require a full second moment")
    count = int(neutral["count"])
    if count < 2:
        raise ValueError("MidSteer neutral calibration has fewer than two observations")
    total = neutral["sum"].to(dtype=torch.float64)
    mean = total / count
    second = neutral["sum_outer"].to(dtype=torch.float64)
    covariance = (
        second - count * torch.einsum("gd,ge->gde", mean, mean)
    ) / (count - 1)
    covariance = 0.5 * (covariance + covariance.mT)
    sigma_minus_half = fractional_matrix_power_covariance(covariance, -0.5)
    sigma_plus_half = fractional_matrix_power_covariance(covariance, 0.5)
    return mean, sigma_minus_half, sigma_plus_half, count


def fit_midsteer_site_transforms(
    neutral: dict[str, Any],
    pairs: Mapping[str, Mapping[str, dict[str, Any]]],
) -> dict[str, dict[str, torch.Tensor]]:
    """Factor one site's neutral covariance once, then fit every concept pair."""

    neutral_mean, sigma_minus_half, sigma_plus_half, neutral_count = (
        _neutral_geometry(neutral)
    )
    fitted: dict[str, dict[str, torch.Tensor]] = {}
    for pair_id, sides in pairs.items():
        if set(sides) != {"source", "target"}:
            raise ValueError(f"MidSteer pair {pair_id!r} lacks source/target moments")
        source = sides["source"]
        target = sides["target"]
        source_count = int(source["count"])
        target_count = int(target["count"])
        if source_count < 1 or target_count < 1:
            raise ValueError(f"MidSteer pair {pair_id!r} has no observations")
        source_mean = source["sum"].to(dtype=torch.float64) / source_count
        target_mean = target["sum"].to(dtype=torch.float64) / target_count
        if source_mean.shape != neutral_mean.shape or target_mean.shape != neutral_mean.shape:
            raise ValueError(f"MidSteer output width changed for pair {pair_id!r}")
        source_white = sigma_minus_half @ (source_mean - neutral_mean).unsqueeze(-1)
        target_white = sigma_minus_half @ (target_mean - neutral_mean).unsqueeze(-1)
        steering_vector = source_white - target_white
        if float(steering_vector.norm().item()) == 0.0:
            raise RuntimeError(
                f"MidSteer pair {pair_id!r} has identical whitened source/target means"
            )
        projection_left = sigma_plus_half @ steering_vector
        projection_right = torch.linalg.pinv(source_white) @ sigma_minus_half
        if not all(
            torch.isfinite(value).all()
            for value in (neutral_mean, projection_left, projection_right)
        ):
            raise RuntimeError(f"MidSteer pair {pair_id!r} produced non-finite transforms")
        fitted[str(pair_id)] = {
            "neutral_mean": neutral_mean.to(dtype=torch.float32),
            "projection_left_t": projection_left.mT.to(dtype=torch.float32),
            "projection_right_t": projection_right.mT.to(dtype=torch.float32),
            "source_target_white_norm": steering_vector.norm().to(dtype=torch.float32),
            "neutral_observations": torch.tensor(neutral_count, dtype=torch.int64),
            "source_observations": torch.tensor(source_count, dtype=torch.int64),
            "target_observations": torch.tensor(target_count, dtype=torch.int64),
        }
    return fitted


def fit_midsteer_transform_from_moments(
    neutral: dict[str, Any],
    source: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, torch.Tensor]:
    return fit_midsteer_site_transforms(
        neutral, {"single_pair": {"source": source, "target": target}}
    )["single_pair"]


class MidSteerArtifact:
    def __init__(self, payload: dict[str, Any]) -> None:
        metadata = payload.get("metadata", {})
        required = {
            "schema_version": MIDSTEER_SCHEMA_VERSION,
            "upstream_revision": MIDSTEER_REVISION,
            "covariance": "full",
            "control_mode": MIDSTEER_CONTROL_MODE,
            "step_policy": MIDSTEER_STEP_POLICY,
            "token_scope": MIDSTEER_TOKEN_SCOPE,
            "intermediate_clipping": False,
        }
        mismatches = {
            key: (metadata.get(key), expected)
            for key, expected in required.items()
            if metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"MidSteer artifact contract mismatch: {mismatches}")
        transforms = payload.get("transforms")
        if not isinstance(transforms, list) or not transforms:
            raise ValueError("MidSteer artifact contains no transforms")
        self.metadata = metadata
        self.transforms = transforms
        self._index: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for item in transforms:
            key = (str(item["model_role"]), str(item["site"]))
            self._index.setdefault(key, []).append(item)

    @classmethod
    def load(cls, path: str | Path) -> "MidSteerArtifact":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        return cls(payload)

    @staticmethod
    def save(
        path: str | Path,
        metadata: dict[str, Any],
        transforms: list[dict[str, Any]],
    ) -> None:
        target = Path(path)
        if target.exists():
            raise FileExistsError(f"Refusing to overwrite MidSteer artifact: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": {
                    **metadata,
                    "schema_version": MIDSTEER_SCHEMA_VERSION,
                    "upstream_revision": MIDSTEER_REVISION,
                    "covariance": "full",
                    "control_mode": MIDSTEER_CONTROL_MODE,
                    "step_policy": MIDSTEER_STEP_POLICY,
                    "token_scope": MIDSTEER_TOKEN_SCOPE,
                    "intermediate_clipping": False,
                },
                "transforms": transforms,
            },
            target,
        )

    def for_role(self, model_role: str) -> dict[str, list[dict[str, Any]]]:
        available_roles = {role for role, _ in self._index}
        role = model_role if model_role in available_roles else "default"
        result = {
            site: values
            for (candidate_role, site), values in self._index.items()
            if candidate_role == role
        }
        if not result:
            raise KeyError(
                f"MidSteer has no transforms for role={model_role!r}; "
                f"available={sorted(available_roles)!r}"
            )
        return result


class MidSteerIntervention(AbstractContextManager["MidSteerIntervention"]):
    def __init__(
        self,
        root: torch.nn.Module,
        artifact: MidSteerArtifact,
        *,
        model_role: str,
        step_index: int,
        strength: float,
        intermediate_clipping: bool = False,
        image_token_count: int | None = None,
    ) -> None:
        if strength < 0:
            raise ValueError("MidSteer strength must be non-negative")
        self.root = root
        self.transforms = artifact.for_role(model_role)
        self.model_role = model_role
        self.step_index = int(step_index)
        self.strength = float(strength)
        self.intermediate_clipping = bool(intermediate_clipping)
        self.image_token_count = image_token_count
        self.handles: list[Any] = []
        self.records: list[dict[str, Any]] = []

    def __enter__(self) -> "MidSteerIntervention":
        sites = {site.name: site for site in attention_output_sites(self.root)}
        missing = sorted(set(self.transforms) - set(sites))
        if missing:
            raise RuntimeError(f"MidSteer calibrated sites are absent: {missing}")
        for name, transforms in self.transforms.items():
            site = sites[name]

            def hook(
                _module: torch.nn.Module,
                _inputs: tuple[Any, ...],
                output: Any,
                *,
                _site: AttentionOutputSite = site,
                _transforms: list[dict[str, Any]] = transforms,
            ) -> Any:
                value, rebuild = _primary_output(output)
                if value.ndim < 2:
                    raise ValueError(
                        f"MidSteer site {_site.name} returned shape {tuple(value.shape)}"
                    )
                batch_slice = (
                    slice(int(value.shape[0]) // 2, None)
                    if int(value.shape[0]) > 1 and int(value.shape[0]) % 2 == 0
                    else slice(None)
                )
                selected = value[batch_slice]
                scoped, preserved_prefix_tokens = _scoped_attention_output(
                    selected,
                    site_name=_site.name,
                    image_token_count=self.image_token_count,
                )
                tokens = _output_tokens(scoped)
                work = tokens.to(dtype=torch.float32)
                original_work = work.clone()
                pair_records: list[dict[str, Any]] = []
                for transform in _transforms:
                    mean = transform["neutral_mean"].to(work.device)
                    left_t = transform["projection_left_t"].to(work.device)
                    right_t = transform["projection_right_t"].to(work.device)
                    centered = work.permute(1, 0, 2) - mean.unsqueeze(1)
                    scores = centered @ right_t
                    positive_fraction = float((scores > 0).float().mean().item())
                    if self.intermediate_clipping:
                        scores = torch.where(scores > 0, scores, torch.zeros_like(scores))
                    delta = -self.strength * (scores @ left_t)
                    candidate = (work.permute(1, 0, 2) + delta).permute(1, 0, 2)
                    effective = (
                        candidate.to(dtype=value.dtype).to(dtype=torch.float32)
                        - work.to(dtype=value.dtype).to(dtype=torch.float32)
                    )
                    pair_records.append(
                        {
                            "pair_id": str(transform["pair_id"]),
                            "delta_norm": float(effective.norm().item()),
                            "mathematical_delta_norm": float(delta.norm().item()),
                            "positive_score_fraction": positive_fraction,
                            "intermediate_clipping": self.intermediate_clipping,
                        }
                    )
                    work = candidate
                replacement_scoped = work.reshape(scoped.shape).to(dtype=value.dtype)
                if preserved_prefix_tokens:
                    replacement_selected = selected.clone()
                    replacement_selected[..., preserved_prefix_tokens:, :] = (
                        replacement_scoped
                    )
                else:
                    replacement_selected = replacement_scoped
                effective_total = (
                    replacement_selected.to(dtype=torch.float32)
                    - selected.to(dtype=torch.float32)
                )
                mathematical_total = work - original_work
                replacement = value.clone()
                replacement[batch_slice] = replacement_selected
                self.records.append(
                    {
                        "site": _site.name,
                        "delta_norm": float(effective_total.norm().item()),
                        "mathematical_delta_norm": float(mathematical_total.norm().item()),
                        "steered_token_count": int(tokens.shape[0]),
                        "preserved_prefix_tokens": preserved_prefix_tokens,
                        "pairs": pair_records,
                    }
                )
                return rebuild(replacement)

            self.handles.append(site.module.register_forward_hook(hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def evidence(self) -> dict[str, Any]:
        called = {str(record["site"]) for record in self.records}
        expected = set(self.transforms)
        if called != expected:
            raise RuntimeError(
                f"MidSteer hook coverage mismatch: missing={sorted(expected-called)}, "
                f"unexpected={sorted(called-expected)}"
            )
        total = sum(float(record["delta_norm"]) for record in self.records)
        mathematical_total = sum(
            float(record["mathematical_delta_norm"]) for record in self.records
        )
        if not torch.isfinite(torch.tensor(total)) or total <= 0.0:
            raise RuntimeError("MidSteer produced no effective post-cast intervention")
        by_pair: dict[str, float] = {}
        mathematical_by_pair: dict[str, float] = {}
        for record in self.records:
            for pair in record["pairs"]:
                pair_id = str(pair["pair_id"])
                by_pair[pair_id] = by_pair.get(pair_id, 0.0) + float(
                    pair["delta_norm"]
                )
                mathematical_by_pair[pair_id] = mathematical_by_pair.get(
                    pair_id, 0.0
                ) + float(pair["mathematical_delta_norm"])
        if any(value <= 0.0 for value in by_pair.values()):
            raise RuntimeError("MidSteer produced a zero effective pair intervention")
        return {
            "site_count": len(expected),
            "total_delta_norm": total,
            "mathematical_total_delta_norm": mathematical_total,
            "pair_delta_norms": by_pair,
            "mathematical_pair_delta_norms": mathematical_by_pair,
            "control_mode": MIDSTEER_CONTROL_MODE,
            "step_policy": MIDSTEER_STEP_POLICY,
            "application_step_index": self.step_index,
            "model_role": self.model_role,
            "intermediate_clipping": self.intermediate_clipping,
            "token_scope": MIDSTEER_TOKEN_SCOPE,
            "image_token_count": self.image_token_count,
            "sites": self.records,
        }
