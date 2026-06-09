from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from hierasafe_flow.generation.video_utils import save_video_frames
from hierasafe_flow.utils.io import ensure_dir, write_json


def save_generation_output(
    media: Any,
    latents: torch.Tensor,
    trace: list[dict[str, Any]],
    output_dir: str | Path,
    sample_id: str,
    task: str,
    save_latents: bool = True,
    save_traces: bool = True,
    image_format: str = "png",
    video_format: str = "mp4",
    fps: int = 16,
) -> dict[str, str]:
    directory = ensure_dir(Path(output_dir) / sample_id)
    paths: dict[str, str] = {}

    if save_latents:
        latent_path = directory / "final_latents.pt"
        torch.save(latents.detach().cpu(), latent_path)
        paths["latents"] = str(latent_path)

    if save_traces:
        trace_path = directory / "steering_trace.json"
        write_json(trace_path, trace)
        paths["trace"] = str(trace_path)

    if media is None:
        return paths

    if isinstance(media, torch.Tensor):
        tensor_path = directory / "decoded_tensor.pt"
        torch.save(media.detach().cpu(), tensor_path)
        paths["media"] = str(tensor_path)
        return paths

    if task == "text_to_image":
        images = media if isinstance(media, list) else [media]
        for idx, image in enumerate(images):
            if not hasattr(image, "save"):
                continue
            path = directory / f"image_{idx:03d}.{image_format}"
            image.save(path)
            paths[f"image_{idx}"] = str(path)
        return paths

    if task == "text_to_video":
        videos = media if isinstance(media, list) else [media]
        for idx, frames in enumerate(videos):
            if not isinstance(frames, (list, tuple)):
                continue
            path = directory / f"video_{idx:03d}.{video_format}"
            save_video_frames(frames, path, fps=fps)
            paths[f"video_{idx}"] = str(path)
        return paths

    return paths


def save_generation_report(
    report: dict[str, Any],
    output_dir: str | Path,
    sample_id: str,
) -> str:
    directory = ensure_dir(Path(output_dir) / sample_id)
    report_path = directory / "report.json"
    write_json(report_path, report)
    return str(report_path)
