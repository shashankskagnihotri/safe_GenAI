"""Full-covariance, per-step, per-attention-site MidSteer adaptation.

The linear algebra follows Atmyre/MidSteer commit
0f3b31e15cdda6ad0d46167e10319e896d6f1541.  The adapter hooks the
concatenated attention-head tensor immediately before the output projection,
which preserves the upstream [batch, sequence, heads, head_dim] contract
without requiring model-specific attention-processor forks.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


MIDSTEER_REVISION = "0f3b31e15cdda6ad0d46167e10319e896d6f1541"


def fractional_matrix_power_covariance(matrix: torch.Tensor, power: float) -> torch.Tensor:
    """Match the pinned upstream eigenvalue threshold and pseudo-power."""

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


def _head_tokens(value: torch.Tensor, heads: int) -> tuple[torch.Tensor, tuple[int, ...]]:
    if value.ndim == 4:
        if int(value.shape[-2]) != heads:
            raise ValueError(
                f"MidSteer expected {heads} heads, observed shape {tuple(value.shape)}"
            )
        original = tuple(value.shape)
        return value.reshape(-1, heads, value.shape[-1]), original
    if value.ndim < 2 or int(value.shape[-1]) % heads:
        raise ValueError(
            f"MidSteer cannot split shape {tuple(value.shape)} into {heads} heads"
        )
    original = tuple(value.shape)
    head_dim = int(value.shape[-1]) // heads
    return value.reshape(-1, heads, head_dim), original


def _restore_head_tokens(tokens: torch.Tensor, original: tuple[int, ...]) -> torch.Tensor:
    return tokens.reshape(original)


@dataclass(frozen=True)
class AttentionHeadSite:
    name: str
    projection: torch.nn.Module
    heads: int


def resolve_transformer_root(adapter: Any) -> torch.nn.Module:
    pipeline = getattr(adapter, "pipeline", None)
    if pipeline is None:
        raise RuntimeError("MidSteer requires a loaded adapter pipeline")
    for name in ("transformer", "unet"):
        value = getattr(pipeline, name, None)
        if isinstance(value, torch.nn.Module):
            return value
    raise RuntimeError("MidSteer requires a transformer or UNet root")


def attention_head_sites(root: torch.nn.Module) -> list[AttentionHeadSite]:
    sites: list[AttentionHeadSite] = []
    seen: set[int] = set()
    for name, module in root.named_modules():
        heads = getattr(module, "heads", None)
        to_out = getattr(module, "to_out", None)
        if not name or not isinstance(heads, int) or heads <= 0 or to_out is None:
            continue
        projection: Any = None
        if isinstance(to_out, (torch.nn.ModuleList, torch.nn.Sequential, list, tuple)):
            if len(to_out):
                projection = to_out[0]
        elif isinstance(to_out, torch.nn.Module):
            projection = to_out
        if not isinstance(projection, torch.nn.Module) or id(projection) in seen:
            continue
        seen.add(id(projection))
        sites.append(
            AttentionHeadSite(
                name=f"{name}.to_out.0",
                projection=projection,
                heads=heads,
            )
        )
    if not sites:
        raise RuntimeError(
            "MidSteer found no attention output projections with an explicit head count"
        )
    return sites


class AttentionHeadCapture(AbstractContextManager["AttentionHeadCapture"]):
    def __init__(self, root: torch.nn.Module, *, conditional_only: bool = True) -> None:
        self.sites = attention_head_sites(root)
        self.conditional_only = conditional_only
        self.handles: list[Any] = []
        self.values: dict[str, list[torch.Tensor]] = {}

    def __enter__(self) -> "AttentionHeadCapture":
        for site in self.sites:
            def hook(
                _module: torch.nn.Module,
                inputs: tuple[Any, ...],
                *,
                _site: AttentionHeadSite = site,
            ) -> None:
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    raise RuntimeError(f"MidSteer site {_site.name} received no tensor input")
                value = inputs[0]
                if self.conditional_only and value.ndim >= 3 and int(value.shape[0]) == 2:
                    value = value[1:]
                tokens, _ = _head_tokens(value, _site.heads)
                self.values.setdefault(_site.name, []).append(
                    tokens.detach().to(device="cpu", dtype=torch.float32)
                )

            self.handles.append(site.projection.register_forward_pre_hook(hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def tensors(self) -> dict[str, torch.Tensor]:
        result = {
            name: torch.cat(chunks, dim=0)
            for name, chunks in self.values.items()
            if chunks
        }
        if set(result) != {site.name for site in self.sites}:
            missing = sorted({site.name for site in self.sites} - set(result))
            raise RuntimeError(f"MidSteer capture missed calibrated sites: {missing}")
        return result


def merge_midsteer_moments(
    left: dict[str, Any] | None,
    right: dict[str, Any],
) -> dict[str, Any]:
    """Merge exact sufficient statistics without retaining token activations."""

    required = {"count", "sum"}
    if not required.issubset(right):
        raise ValueError(f"Incomplete MidSteer moments: {sorted(right)}")
    if left is None:
        return {
            "count": int(right["count"]),
            "sum": right["sum"].clone(),
            **(
                {"sum_outer": right["sum_outer"].clone()}
                if "sum_outer" in right
                else {}
            ),
        }
    if ("sum_outer" in left) != ("sum_outer" in right):
        raise ValueError("Cannot merge MidSteer moments with different covariance scope")
    if left["sum"].shape != right["sum"].shape:
        raise ValueError("Cannot merge MidSteer moments with different head shapes")
    result = {
        "count": int(left["count"]) + int(right["count"]),
        "sum": left["sum"] + right["sum"],
    }
    if "sum_outer" in left:
        result["sum_outer"] = left["sum_outer"] + right["sum_outer"]
    return result


class AttentionHeadMomentCapture(AbstractContextManager["AttentionHeadMomentCapture"]):
    """Stream per-head first/full-second moments to CPU during one forward."""

    def __init__(
        self,
        root: torch.nn.Module,
        *,
        include_covariance: bool,
        conditional_only: bool = True,
    ) -> None:
        self.sites = attention_head_sites(root)
        self.include_covariance = include_covariance
        self.conditional_only = conditional_only
        self.handles: list[Any] = []
        self.values: dict[str, dict[str, Any]] = {}

    def __enter__(self) -> "AttentionHeadMomentCapture":
        for site in self.sites:
            def hook(
                _module: torch.nn.Module,
                inputs: tuple[Any, ...],
                *,
                _site: AttentionHeadSite = site,
            ) -> None:
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    raise RuntimeError(f"MidSteer site {_site.name} received no tensor input")
                value = inputs[0]
                if self.conditional_only and value.ndim >= 3 and int(value.shape[0]) == 2:
                    value = value[1:]
                tokens, _ = _head_tokens(value, _site.heads)
                work = tokens.detach().to(dtype=torch.float32)
                moments: dict[str, Any] = {
                    "count": int(work.shape[0]),
                    "sum": work.sum(dim=0).to(device="cpu", dtype=torch.float64),
                }
                if self.include_covariance:
                    moments["sum_outer"] = torch.einsum(
                        "nhd,nhe->hde", work, work
                    ).to(device="cpu", dtype=torch.float64)
                self.values[_site.name] = merge_midsteer_moments(
                    self.values.get(_site.name), moments
                )

            self.handles.append(site.projection.register_forward_pre_hook(hook))
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def moments(self) -> dict[str, dict[str, Any]]:
        expected = {site.name for site in self.sites}
        if set(self.values) != expected:
            missing = sorted(expected - set(self.values))
            raise RuntimeError(f"MidSteer moment capture missed sites: {missing}")
        return self.values


def fit_midsteer_transform_from_moments(
    neutral: dict[str, Any],
    source: dict[str, Any],
    target: dict[str, Any],
) -> dict[str, torch.Tensor]:
    """Fit the upstream affine transform from exact merged sufficient statistics."""

    if "sum_outer" not in neutral:
        raise ValueError("Neutral MidSteer moments require a full second moment")
    neutral_count = int(neutral["count"])
    source_count = int(source["count"])
    target_count = int(target["count"])
    if neutral_count < 2 or source_count < 1 or target_count < 1:
        raise ValueError("MidSteer calibration has insufficient observations")
    neutral_sum = neutral["sum"].to(dtype=torch.float64)
    source_sum = source["sum"].to(dtype=torch.float64)
    target_sum = target["sum"].to(dtype=torch.float64)
    if neutral_sum.shape != source_sum.shape or source_sum.shape != target_sum.shape:
        raise ValueError("MidSteer neutral/source/target head shapes differ")
    neutral_mean = neutral_sum / neutral_count
    second = neutral["sum_outer"].to(dtype=torch.float64)
    covariance = (
        second
        - neutral_count
        * torch.einsum("hd,he->hde", neutral_mean, neutral_mean)
    ) / (neutral_count - 1)
    covariance = 0.5 * (covariance + covariance.mT)
    sigma_minus_half = fractional_matrix_power_covariance(covariance, -0.5)
    sigma_plus_half = fractional_matrix_power_covariance(covariance, 0.5)
    source_mean = source_sum / source_count
    target_mean = target_sum / target_count
    source_white = sigma_minus_half @ (source_mean - neutral_mean).unsqueeze(-1)
    target_white = sigma_minus_half @ (target_mean - neutral_mean).unsqueeze(-1)
    steering_vector = source_white - target_white
    projection_left = sigma_plus_half @ steering_vector
    projection_right = torch.linalg.pinv(source_white) @ sigma_minus_half
    if not all(
        torch.isfinite(value).all()
        for value in (neutral_mean, projection_left, projection_right)
    ):
        raise RuntimeError("MidSteer calibration produced non-finite transforms")
    if float(steering_vector.norm().item()) == 0.0:
        raise RuntimeError("MidSteer source and target means are identical after whitening")
    return {
        "neutral_mean": neutral_mean.to(dtype=torch.float32),
        "projection_left_t": projection_left.mT.to(dtype=torch.float32),
        "projection_right_t": projection_right.mT.to(dtype=torch.float32),
        "source_target_white_norm": steering_vector.norm().to(dtype=torch.float32),
        "neutral_observations": torch.tensor(neutral_count, dtype=torch.int64),
        "source_observations": torch.tensor(source_count, dtype=torch.int64),
        "target_observations": torch.tensor(target_count, dtype=torch.int64),
    }


def fit_midsteer_transform(
    neutral: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Fit the pinned upstream full-covariance MidSteer projection."""

    for name, value in (("neutral", neutral), ("source", source), ("target", target)):
        if value.ndim != 3:
            raise ValueError(
                f"MidSteer {name} observations must be [samples,heads,dim], "
                f"got {tuple(value.shape)}"
            )
    if neutral.shape[1:] != source.shape[1:] or source.shape[1:] != target.shape[1:]:
        raise ValueError("MidSteer neutral/source/target head shapes differ")
    neutral64 = neutral.to(dtype=torch.float64)
    source64 = source.to(dtype=torch.float64)
    target64 = target.to(dtype=torch.float64)
    return fit_midsteer_transform_from_moments(
        {
            "count": int(neutral64.shape[0]),
            "sum": neutral64.sum(dim=0),
            "sum_outer": torch.einsum("nhd,nhe->hde", neutral64, neutral64),
        },
        {"count": int(source64.shape[0]), "sum": source64.sum(dim=0)},
        {"count": int(target64.shape[0]), "sum": target64.sum(dim=0)},
    )


class MidSteerArtifact:
    def __init__(self, payload: dict[str, Any]) -> None:
        metadata = payload.get("metadata", {})
        if metadata.get("upstream_revision") != MIDSTEER_REVISION:
            raise ValueError("MidSteer artifact has an unknown upstream revision")
        if metadata.get("covariance") != "full":
            raise ValueError("MidSteer artifact is not full-covariance")
        transforms = payload.get("transforms")
        if not isinstance(transforms, list) or not transforms:
            raise ValueError("MidSteer artifact contains no transforms")
        self.metadata = metadata
        self.transforms = transforms
        self._index: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
        for item in transforms:
            key = (str(item["model_role"]), int(item["step_index"]), str(item["site"]))
            self._index.setdefault(key, []).append(item)

    @classmethod
    def load(cls, path: str | Path) -> "MidSteerArtifact":
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu")
        return cls(payload)

    @staticmethod
    def save(path: str | Path, metadata: dict[str, Any], transforms: list[dict[str, Any]]) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "metadata": {
                    **metadata,
                    "upstream_revision": MIDSTEER_REVISION,
                    "covariance": "full",
                    "control_mode": "attention_heads_pre_output_projection",
                },
                "transforms": transforms,
            },
            target,
        )

    def for_step(self, model_role: str, step_index: int) -> dict[str, list[dict[str, Any]]]:
        result: dict[str, list[dict[str, Any]]] = {}
        for (role, step, site), values in self._index.items():
            if role == model_role and step == step_index:
                result[site] = values
        if not result:
            raise KeyError(
                f"MidSteer has no transforms for role={model_role!r}, step={step_index}"
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
    ) -> None:
        self.root = root
        self.transforms = artifact.for_step(model_role, step_index)
        self.strength = float(strength)
        self.handles: list[Any] = []
        self.records: list[dict[str, Any]] = []

    def __enter__(self) -> "MidSteerIntervention":
        sites = {site.name: site for site in attention_head_sites(self.root)}
        missing = sorted(set(self.transforms) - set(sites))
        if missing:
            raise RuntimeError(f"MidSteer calibrated sites are absent: {missing}")
        for name, transforms in self.transforms.items():
            site = sites[name]

            def hook(
                _module: torch.nn.Module,
                inputs: tuple[Any, ...],
                *,
                _site: AttentionHeadSite = site,
                _transforms: list[dict[str, Any]] = transforms,
            ) -> tuple[Any, ...]:
                if not inputs or not isinstance(inputs[0], torch.Tensor):
                    raise RuntimeError(f"MidSteer site {_site.name} received no tensor")
                value = inputs[0]
                tokens, original = _head_tokens(value, _site.heads)
                work = tokens.to(dtype=torch.float32)
                pair_records: list[dict[str, Any]] = []
                for transform in _transforms:
                    mean = transform["neutral_mean"].to(work.device)
                    left_t = transform["projection_left_t"].to(work.device)
                    right_t = transform["projection_right_t"].to(work.device)
                    centered = work.permute(1, 0, 2) - mean.unsqueeze(1)
                    scores = centered @ right_t
                    scores = torch.where(scores > 0, scores, torch.zeros_like(scores))
                    delta = -self.strength * (scores @ left_t)
                    work = (work.permute(1, 0, 2) + delta).permute(1, 0, 2)
                    pair_records.append(
                        {
                            "pair_id": str(transform["pair_id"]),
                            "delta_norm": float(delta.norm().item()),
                            "positive_gate_fraction": float((scores > 0).float().mean().item()),
                        }
                    )
                total_delta = work - tokens.to(dtype=torch.float32)
                self.records.append(
                    {
                        "site": _site.name,
                        "delta_norm": float(total_delta.norm().item()),
                        "pairs": pair_records,
                    }
                )
                replacement = _restore_head_tokens(work.to(dtype=value.dtype), original)
                return (replacement, *inputs[1:])

            self.handles.append(site.projection.register_forward_pre_hook(hook))
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
        if not torch.isfinite(torch.tensor(total)) or total <= 0.0:
            raise RuntimeError("MidSteer active step produced a zero or non-finite intervention")
        by_pair: dict[str, float] = {}
        for record in self.records:
            for pair in record["pairs"]:
                by_pair[pair["pair_id"]] = by_pair.get(pair["pair_id"], 0.0) + float(
                    pair["delta_norm"]
                )
        if any(value <= 0.0 for value in by_pair.values()):
            raise RuntimeError("MidSteer produced a zero pair intervention")
        return {
            "site_count": len(expected),
            "total_delta_norm": total,
            "pair_delta_norms": by_pair,
            "sites": self.records,
        }
