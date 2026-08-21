"""Exact CogVideoX-1.0 RIFLEx rotary-position implementation.

This is an isolated transcription of thu-ml/DiT-Extrapolation at revision
d54e1e2a0ec88626d3f5543f3c54c0c6cbbbc87c.  The only additions are explicit
contract checks; the frequency and 3-D composition equations are unchanged.
"""

from __future__ import annotations

from typing import Any

import torch
from diffusers.models.embeddings import get_1d_rotary_pos_embed
from diffusers.pipelines.cogvideo.pipeline_cogvideox import (
    get_resize_crop_region_for_grid,
)


UPSTREAM_REPOSITORY = "https://github.com/thu-ml/DiT-Extrapolation"
UPSTREAM_REVISION = "d54e1e2a0ec88626d3f5543f3c54c0c6cbbbc87c"


def get_1d_rotary_pos_embed_riflex(
    dim: int,
    pos: torch.Tensor | int,
    *,
    theta: float = 10_000.0,
    use_real: bool = False,
    k: int | None = None,
    L_test: int | None = None,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Return the upstream RIFLEx one-dimensional rotary frequencies."""

    if dim % 2 != 0:
        raise ValueError(f"RIFLEx rotary dimension must be even, got {dim}")
    if isinstance(pos, int):
        pos = torch.arange(pos)
    if not isinstance(pos, torch.Tensor):
        pos = torch.as_tensor(pos)

    freqs = 1.0 / (
        theta
        ** (
            torch.arange(0, dim, 2, device=pos.device)[: dim // 2].float()
            / dim
        )
    )
    if k is not None:
        if L_test is None or L_test <= 0:
            raise ValueError("L_test must be positive when RIFLEx k is set")
        if not 1 <= k <= freqs.numel():
            raise ValueError(f"RIFLEx k={k} is outside 1..{freqs.numel()}")
        freqs[k - 1] = 0.9 * 2 * torch.pi / L_test

    freqs = torch.outer(pos, freqs)
    if use_real:
        return (
            freqs.cos().repeat_interleave(2, dim=1).float(),
            freqs.sin().repeat_interleave(2, dim=1).float(),
        )
    return torch.polar(torch.ones_like(freqs), freqs)


def get_3d_rotary_pos_embed_riflex(
    *,
    embed_dim: int,
    crops_coords: tuple[tuple[int, int], tuple[int, int]],
    grid_size: tuple[int, int],
    temporal_size: int,
    theta: int = 10_000,
    device: torch.device,
    k: int,
    L_test: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compose temporal RIFLEx and unchanged spatial rotary frequencies."""

    start, stop = crops_coords
    grid_size_h, grid_size_w = grid_size
    grid_h = torch.linspace(
        start[0],
        stop[0] * (grid_size_h - 1) / grid_size_h,
        grid_size_h,
        device=device,
        dtype=torch.float32,
    )
    grid_w = torch.linspace(
        start[1],
        stop[1] * (grid_size_w - 1) / grid_size_w,
        grid_size_w,
        device=device,
        dtype=torch.float32,
    )
    grid_t = torch.linspace(
        0,
        temporal_size * (temporal_size - 1) / temporal_size,
        temporal_size,
        device=device,
        dtype=torch.float32,
    )

    dim_t = embed_dim // 4
    dim_h = embed_dim // 8 * 3
    dim_w = embed_dim // 8 * 3
    if dim_t + dim_h + dim_w != embed_dim:
        raise ValueError(
            f"CogVideoX rotary dimensions do not sum to {embed_dim}: "
            f"{dim_t}+{dim_h}+{dim_w}"
        )

    t_cos, t_sin = get_1d_rotary_pos_embed_riflex(
        dim_t,
        grid_t,
        theta=theta,
        use_real=True,
        k=k,
        L_test=L_test,
    )
    h_cos, h_sin = get_1d_rotary_pos_embed(
        dim_h, grid_h, theta=theta, use_real=True
    )
    w_cos, w_sin = get_1d_rotary_pos_embed(
        dim_w, grid_w, theta=theta, use_real=True
    )

    def combine(
        temporal: torch.Tensor,
        vertical: torch.Tensor,
        horizontal: torch.Tensor,
    ) -> torch.Tensor:
        temporal = temporal[:, None, None, :].expand(
            -1, grid_size_h, grid_size_w, -1
        )
        vertical = vertical[None, :, None, :].expand(
            temporal_size, -1, grid_size_w, -1
        )
        horizontal = horizontal[None, None, :, :].expand(
            temporal_size, grid_size_h, -1, -1
        )
        return torch.cat([temporal, vertical, horizontal], dim=-1).view(
            temporal_size * grid_size_h * grid_size_w, -1
        )

    return combine(t_cos, h_cos, w_cos), combine(t_sin, h_sin, w_sin)


def prepare_cogvideox_1_0_riflex_rotary(
    pipeline: Any,
    *,
    height: int,
    width: int,
    latent_frames: int,
    device: torch.device,
    k: int,
    L_test: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the exact upstream CogVideoX-1.0 RIFLEx rotary pair."""

    transformer_config = pipeline.transformer.config
    if transformer_config.patch_size_t is not None:
        raise ValueError("This benchmark contract supports CogVideoX-1.0 only")
    if latent_frames != L_test:
        raise ValueError(
            f"latent_frames={latent_frames} must equal RIFLEx L_test={L_test}"
        )

    patch_size = transformer_config.patch_size
    grid_height = height // (pipeline.vae_scale_factor_spatial * patch_size)
    grid_width = width // (pipeline.vae_scale_factor_spatial * patch_size)
    base_size_width = transformer_config.sample_width // patch_size
    base_size_height = transformer_config.sample_height // patch_size
    crop = get_resize_crop_region_for_grid(
        (grid_height, grid_width), base_size_width, base_size_height
    )
    return get_3d_rotary_pos_embed_riflex(
        embed_dim=transformer_config.attention_head_dim,
        crops_coords=crop,
        grid_size=(grid_height, grid_width),
        temporal_size=latent_frames,
        device=device,
        k=k,
        L_test=L_test,
    )
