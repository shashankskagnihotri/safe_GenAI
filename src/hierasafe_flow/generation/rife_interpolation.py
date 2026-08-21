"""Pinned RIFE midpoint interpolation and exact 8-to-16 fps frame arithmetic.

The small inference network below is an architecture-compatible transcription of
the RIFE HDv3 inference model distributed by the RIFE project.  Its parameter
names intentionally match the pinned ``flownet.pkl`` checkpoint.  RIFE is MIT
licensed; the upstream implementation and checkpoint identities are recorded in
the constants below so experiment reports can distinguish generated source
frames from neural interpolation.

Upstream code: https://github.com/hzwer/ECCV2022-RIFE
Pinned code revision: 5d8adbdd40e12c2c8f91930eff838aebe561c086
Pinned weights: AlexWortega/RIFE@440cdec905de98e1d7e81f65d2c88a08da7cb4e2

RIFE copyright (c) 2020 hzwer. Permission is hereby granted, free of
charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without
restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software,
and to permit persons to whom the Software is furnished to do so, subject to
including this copyright and permission notice in all copies or substantial
portions. THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO
EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES
OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE,
ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
DEALINGS IN THE SOFTWARE.
"""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

RIFE_CODE_REPOSITORY = "https://github.com/hzwer/ECCV2022-RIFE"
RIFE_CODE_REVISION = "5d8adbdd40e12c2c8f91930eff838aebe561c086"
RIFE_WEIGHTS_REPOSITORY = "AlexWortega/RIFE"
RIFE_WEIGHTS_REVISION = "440cdec905de98e1d7e81f65d2c88a08da7cb4e2"
RIFE_WEIGHTS_FILENAME = "flownet.pkl"
RIFE_WEIGHTS_SHA256 = "fe854fc8996547c953f732aaa3b78cae76cc0a12833ae856ea0749c4c570d7d8"


@dataclass(frozen=True)
class ExactInterpolationProvenance:
    schema_version: int
    source_frame_count: int
    source_fps: int
    inclusive_interpolated_frame_count: int
    inserted_midpoint_count: int
    endpoint_crop: str
    output_frame_count: int
    output_fps: int
    duration_seconds: float
    source_frame_pixel_sha256: list[str]
    inclusive_frame_pixel_sha256: list[str]
    output_frame_pixel_sha256: list[str]
    preserved_source_output_indices: list[int]
    dropped_source_frame_index: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RifeMidpointInterpolator:
    """Run the pinned RIFE HDv3 checkpoint at exactly ``t=0.5``.

    The backend is deliberately narrow: it inserts one midpoint and never
    duplicates a source frame, blends linearly, or silently changes resolution.
    Checkpoint download is revision-pinned and the bytes are verified before any
    pickle deserialization or model execution.
    """

    def __init__(
        self,
        *,
        device: torch.device | str,
        weights_path: str | Path | None = None,
        scale: float = 1.0,
    ) -> None:
        if scale not in {0.25, 0.5, 1.0, 2.0, 4.0}:
            raise ValueError(f"RIFE scale must be one of 0.25, 0.5, 1, 2, 4; got {scale}.")
        self.device = torch.device(device)
        self.scale = float(scale)
        resolved_weights = (
            Path(weights_path).expanduser().resolve()
            if weights_path is not None
            else _download_pinned_weights()
        )
        actual_sha256 = _sha256_file(resolved_weights)
        if actual_sha256 != RIFE_WEIGHTS_SHA256:
            raise RuntimeError(
                "Pinned RIFE checkpoint digest mismatch: "
                f"expected {RIFE_WEIGHTS_SHA256}, got {actual_sha256} at {resolved_weights}."
            )

        # Verify bytes before torch.load because this historical checkpoint is a
        # pickle-backed state dict. weights_only=True restricts deserialization
        # to tensor/state-dict types supported by PyTorch's safe loader.
        raw_state = torch.load(resolved_weights, map_location="cpu", weights_only=True)
        if not isinstance(raw_state, dict) or not raw_state:
            raise RuntimeError("Pinned RIFE checkpoint is not a non-empty state dictionary.")
        if any(not isinstance(key, str) or not key.startswith("module.") for key in raw_state):
            raise RuntimeError("Pinned RIFE checkpoint has unexpected unscoped parameter keys.")
        state = {key.removeprefix("module."): value for key, value in raw_state.items()}
        self.model = _RifeIfNet().to(device=self.device, dtype=torch.float32).eval()
        self.model.load_state_dict(state, strict=True)
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.weights_path = resolved_weights
        self.weights_sha256 = actual_sha256

    def __call__(self, frame0: Any, frame1: Any) -> Image.Image:
        array0 = _frame_to_rgb_uint8(frame0)
        array1 = _frame_to_rgb_uint8(frame1)
        if array0.shape != array1.shape:
            raise ValueError(
                f"RIFE frame shape mismatch: first={array0.shape}, second={array1.shape}."
            )
        height, width, _ = array0.shape
        multiple = max(32, int(32 / self.scale))
        padded_height = ((height - 1) // multiple + 1) * multiple
        padded_width = ((width - 1) // multiple + 1) * multiple
        padding = (0, padded_width - width, 0, padded_height - height)

        tensor0 = _array_to_tensor(array0, self.device)
        tensor1 = _array_to_tensor(array1, self.device)
        tensor0 = F.pad(tensor0, padding)
        tensor1 = F.pad(tensor1, padding)
        scale_list = [4.0 / self.scale, 2.0 / self.scale, 1.0 / self.scale]
        with torch.inference_mode(), torch.backends.cudnn.flags(
            enabled=True,
            benchmark=False,
            deterministic=True,
        ):
            midpoint = self.model(torch.cat((tensor0, tensor1), dim=1), scale_list)[2]
        midpoint = midpoint[0, :, :height, :width].clamp(0.0, 1.0)
        # Match the pinned RIFE inference script's uint8 conversion (truncate,
        # not a hidden rounding or image-encoding operation).
        array = (midpoint * 255.0).to(torch.uint8).permute(1, 2, 0).cpu().numpy()
        if array.shape != array0.shape:
            raise RuntimeError(
                f"RIFE changed frame shape from {array0.shape} to {array.shape}."
            )
        return Image.fromarray(array, mode="RGB")

    def provenance(self) -> dict[str, Any]:
        return {
            "backend": "RIFE_HDv3_midpoint",
            "code_repository": RIFE_CODE_REPOSITORY,
            "code_revision": RIFE_CODE_REVISION,
            "weights_repository": RIFE_WEIGHTS_REPOSITORY,
            "weights_revision": RIFE_WEIGHTS_REVISION,
            "weights_filename": RIFE_WEIGHTS_FILENAME,
            "weights_sha256": self.weights_sha256,
            "resolved_weights_path": str(self.weights_path),
            "scale": self.scale,
            "device": str(self.device),
            "dtype": "torch.float32",
            "timestep": 0.5,
        }


def rife_2x_and_crop_exact(
    source_frames: Sequence[Any],
    *,
    midpoint_interpolator: Callable[[Any, Any], Any],
    source_fps: int,
    output_fps: int,
    duration_seconds: float,
    output_frame_count: int,
) -> tuple[list[Any], ExactInterpolationProvenance]:
    """Insert one midpoint per source interval, then take ``[0:output_count]``.

    For the frozen CogVideoX protocol this proves the exact arithmetic
    ``121 -> 241 -> [0:240]``.  The inclusive endpoint is necessary to cover 15
    seconds at 8 fps; it is then omitted under the standard half-open 16-fps
    sample interval, yielding frames at times ``0/16`` through ``239/16``.
    """

    if source_fps <= 0 or output_fps != 2 * source_fps:
        raise ValueError(
            f"Exact midpoint protocol requires output_fps == 2 * source_fps; "
            f"got {source_fps} -> {output_fps}."
        )
    if duration_seconds <= 0 or int(duration_seconds * source_fps) != duration_seconds * source_fps:
        raise ValueError("Duration must contain an integral number of native frame intervals.")
    expected_source_count = int(duration_seconds * source_fps) + 1
    expected_output_count = int(duration_seconds * output_fps)
    if len(source_frames) != expected_source_count:
        raise ValueError(
            f"Expected {expected_source_count} source frames for {duration_seconds}s at "
            f"{source_fps} fps inclusive, got {len(source_frames)}."
        )
    if output_frame_count != expected_output_count:
        raise ValueError(
            f"Expected output_frame_count={expected_output_count}, got {output_frame_count}."
        )

    source_arrays = [_frame_to_rgb_uint8(frame) for frame in source_frames]
    reference_shape = source_arrays[0].shape
    if any(array.shape != reference_shape for array in source_arrays):
        raise ValueError("All RIFE source frames must have exactly the same RGB dimensions.")

    inclusive: list[Any] = []
    for index in range(len(source_frames) - 1):
        inclusive.append(source_frames[index])
        midpoint = midpoint_interpolator(source_frames[index], source_frames[index + 1])
        midpoint_array = _frame_to_rgb_uint8(midpoint)
        if midpoint_array.shape != reference_shape:
            raise RuntimeError(
                f"RIFE midpoint {index} has shape {midpoint_array.shape}; expected {reference_shape}."
            )
        inclusive.append(midpoint)
    inclusive.append(source_frames[-1])

    expected_inclusive_count = 2 * len(source_frames) - 1
    if len(inclusive) != expected_inclusive_count:
        raise RuntimeError(
            f"Interpolation produced {len(inclusive)} frames; expected {expected_inclusive_count}."
        )
    output = inclusive[:output_frame_count]
    if len(output) != output_frame_count:
        raise RuntimeError(
            f"Endpoint crop produced {len(output)} frames; expected {output_frame_count}."
        )

    source_hashes = [_pixel_sha256(array) for array in source_arrays]
    inclusive_hashes = [_pixel_sha256(_frame_to_rgb_uint8(frame)) for frame in inclusive]
    output_hashes = inclusive_hashes[:output_frame_count]
    preserved_indices = list(range(0, output_frame_count, 2))
    for source_index, output_index in enumerate(preserved_indices):
        if output_hashes[output_index] != source_hashes[source_index]:
            raise RuntimeError(
                f"Source frame {source_index} was not preserved exactly at output index {output_index}."
            )

    provenance = ExactInterpolationProvenance(
        schema_version=1,
        source_frame_count=len(source_frames),
        source_fps=source_fps,
        inclusive_interpolated_frame_count=len(inclusive),
        inserted_midpoint_count=len(source_frames) - 1,
        endpoint_crop=f"[0:{output_frame_count}]",
        output_frame_count=len(output),
        output_fps=output_fps,
        duration_seconds=float(duration_seconds),
        source_frame_pixel_sha256=source_hashes,
        inclusive_frame_pixel_sha256=inclusive_hashes,
        output_frame_pixel_sha256=output_hashes,
        preserved_source_output_indices=preserved_indices,
        dropped_source_frame_index=len(source_frames) - 1,
    )
    return output, provenance


def _download_pinned_weights() -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            repo_id=RIFE_WEIGHTS_REPOSITORY,
            filename=RIFE_WEIGHTS_FILENAME,
            revision=RIFE_WEIGHTS_REVISION,
        )
    ).resolve()


def _frame_to_rgb_uint8(frame: Any) -> np.ndarray:
    if hasattr(frame, "convert"):
        array = np.asarray(frame.convert("RGB"))
    else:
        array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] != 3 or array.dtype != np.uint8:
        raise ValueError(
            "RIFE frames must be RGB uint8 images with shape [height, width, 3]; "
            f"got shape={array.shape}, dtype={array.dtype}."
        )
    return np.ascontiguousarray(array)


def _array_to_tensor(array: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(array.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        .div_(255.0)
    )


def _pixel_sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _conv(in_channels: int, out_channels: int, *, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=True),
        nn.PReLU(out_channels),
    )


class _RifeIfBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int = 90) -> None:
        super().__init__()
        self.conv0 = nn.Sequential(
            _conv(in_channels, channels // 2, stride=2),
            _conv(channels // 2, channels, stride=2),
        )
        self.convblock0 = nn.Sequential(_conv(channels, channels), _conv(channels, channels))
        self.convblock1 = nn.Sequential(_conv(channels, channels), _conv(channels, channels))
        self.convblock2 = nn.Sequential(_conv(channels, channels), _conv(channels, channels))
        self.convblock3 = nn.Sequential(_conv(channels, channels), _conv(channels, channels))
        self.conv1 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1),
            nn.PReLU(channels // 2),
            nn.ConvTranspose2d(channels // 2, 4, 4, 2, 1),
        )
        self.conv2 = nn.Sequential(
            nn.ConvTranspose2d(channels, channels // 2, 4, 2, 1),
            nn.PReLU(channels // 2),
            nn.ConvTranspose2d(channels // 2, 1, 4, 2, 1),
        )

    def forward(
        self,
        images_and_mask: torch.Tensor,
        flow: torch.Tensor,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = F.interpolate(
            images_and_mask,
            scale_factor=1.0 / scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        scaled_flow = F.interpolate(
            flow,
            scale_factor=1.0 / scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        ) * (1.0 / scale)
        features = self.conv0(torch.cat((features, scaled_flow), dim=1))
        for block in (self.convblock0, self.convblock1, self.convblock2, self.convblock3):
            features = block(features) + features
        flow_delta = self.conv1(features)
        mask_delta = self.conv2(features)
        flow_delta = F.interpolate(
            flow_delta,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        ) * scale
        mask_delta = F.interpolate(
            mask_delta,
            scale_factor=scale,
            mode="bilinear",
            align_corners=False,
            recompute_scale_factor=False,
        )
        return flow_delta, mask_delta


class _RifeIfNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.block0 = _RifeIfBlock(11)
        self.block1 = _RifeIfBlock(11)
        self.block2 = _RifeIfBlock(11)
        # The teacher block is present in the released state dict even though
        # inference never executes it. Keeping it is required for strict load.
        self.block_tea = _RifeIfBlock(14)

    def forward(self, images: torch.Tensor, scale_list: list[float]) -> list[torch.Tensor]:
        if images.ndim != 4 or images.shape[1] != 6:
            raise ValueError(f"RIFE IFNet expects [B,6,H,W], got {tuple(images.shape)}.")
        image0, image1 = images[:, :3], images[:, 3:]
        warped0, warped1 = image0, image1
        flow = images[:, :4].detach() * 0.0
        mask = images[:, :1].detach() * 0.0
        merged: list[torch.Tensor] = []
        for block, scale in zip((self.block0, self.block1, self.block2), scale_list, strict=True):
            delta0, mask0 = block(torch.cat((warped0, warped1, mask), dim=1), flow, scale)
            swapped_flow = torch.cat((flow[:, 2:4], flow[:, :2]), dim=1)
            delta1, mask1 = block(
                torch.cat((warped1, warped0, -mask), dim=1),
                swapped_flow,
                scale,
            )
            flow = flow + (delta0 + torch.cat((delta1[:, 2:4], delta1[:, :2]), dim=1)) / 2.0
            mask = mask + (mask0 - mask1) / 2.0
            warped0 = _warp(image0, flow[:, :2])
            warped1 = _warp(image1, flow[:, 2:4])
            sigmoid_mask = torch.sigmoid(mask)
            merged.append(warped0 * sigmoid_mask + warped1 * (1.0 - sigmoid_mask))
        return merged


_WARP_GRID_CACHE: dict[tuple[str, str, tuple[int, ...]], torch.Tensor] = {}


def _warp(image: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
    key = (str(flow.device), str(flow.dtype), tuple(flow.shape))
    grid = _WARP_GRID_CACHE.get(key)
    if grid is None:
        horizontal = torch.linspace(
            -1.0,
            1.0,
            flow.shape[3],
            device=flow.device,
            dtype=flow.dtype,
        ).view(1, 1, 1, flow.shape[3]).expand(flow.shape[0], -1, flow.shape[2], -1)
        vertical = torch.linspace(
            -1.0,
            1.0,
            flow.shape[2],
            device=flow.device,
            dtype=flow.dtype,
        ).view(1, 1, flow.shape[2], 1).expand(flow.shape[0], -1, -1, flow.shape[3])
        grid = torch.cat((horizontal, vertical), dim=1)
        _WARP_GRID_CACHE[key] = grid
    normalized_flow = torch.cat(
        (
            flow[:, 0:1] / ((image.shape[3] - 1.0) / 2.0),
            flow[:, 1:2] / ((image.shape[2] - 1.0) / 2.0),
        ),
        dim=1,
    )
    sampling_grid = (grid + normalized_flow).permute(0, 2, 3, 1)
    return F.grid_sample(
        image,
        sampling_grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
