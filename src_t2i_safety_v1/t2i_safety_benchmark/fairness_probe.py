from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from hierasafe_flow.adapters.registry import create_adapter
from hierasafe_flow.utils.config import load_config
from hierasafe_flow.utils.device import configure_cuda, resolve_device, resolve_dtype
from hierasafe_flow.utils.seed import make_generator, seed_everything

from .contracts import (
    CALIBRATION_ROOT,
    PROJECT_ROOT,
    BenchmarkContract,
    atomic_json,
    execution_identity,
    file_sha256,
    staged_attempt,
)
from .fairness_manifest import (
    PROBE_COUNT,
    PROBE_MANIFEST,
    ProbeRow,
    build_manifest,
    load_manifest,
)
from .methods import tensor_stats


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pil_images(value: Any) -> list[Image.Image]:
    if isinstance(value, Image.Image):
        return [value]
    if isinstance(value, (list, tuple)):
        result: list[Image.Image] = []
        for item in value:
            result.extend(_pil_images(item))
        return result
    return []


class ProbeRunner:
    def __init__(
        self,
        *,
        model_id: str,
        shard_index: int,
        num_shards: int,
    ) -> None:
        if num_shards < 1 or not 0 <= shard_index < num_shards:
            raise ValueError("Invalid fairness-probe shard coordinates.")
        contract = BenchmarkContract()
        self.model_id = model_id
        self.model_spec = contract.model(model_id)
        rows, self.manifest_sha = load_manifest()
        self.rows = [
            row for index, row in enumerate(rows) if index % num_shards == shard_index
        ]
        if not self.rows:
            raise RuntimeError("Fairness-probe shard selected no rows.")

        seed_everything(0)
        configure_cuda(True)
        self.device = resolve_device("auto")
        self.dtype = resolve_dtype("bfloat16")
        model_config = load_config(
            self.model_spec["config"],
            project_root=PROJECT_ROOT,
        )
        self.generation = dict(model_config["generation"])
        model_values = dict(model_config["model"])
        model_values["height"] = int(self.model_spec["height"])
        model_values["width"] = int(self.model_spec["width"])
        model_values["guidance_scale"] = float(self.model_spec["guidance_scale"])
        self.adapter = create_adapter(model_values, self.device, self.dtype)

    def run(self) -> None:
        self.adapter.load()
        for row in self.rows:
            final = (
                CALIBRATION_ROOT
                / "fairness_probe"
                / self.model_id
                / row.probe_id
                / "attempt_001"
            )
            with staged_attempt(final) as staging:
                if staging is None:
                    continue
                self._run_row(row, staging)

    def _run_row(self, row: ProbeRow, staging: Path) -> None:
        started_at = _utc()
        start = time.perf_counter()
        self.adapter.begin_conditioning_provenance_scope()
        generator = make_generator(int(row.seed), self.device)
        generation = {
            **self.generation,
            "height": int(self.model_spec["height"]),
            "width": int(self.model_spec["width"]),
            "num_inference_steps": int(self.model_spec["steps"]),
            "num_outputs_per_prompt": 1,
        }
        latent_kwargs = {
            key: value
            for key, value in generation.items()
            if key not in {"prompt", "prompt_file"}
        }
        latents, state = self.adapter.prepare_initial_latents(
            prompt=row.prompt,
            batch_size=1,
            generator=generator,
            **latent_kwargs,
        )
        num_steps = int(self.model_spec["steps"])
        state.extra["num_steps"] = num_steps
        state.extra["base_seed"] = int(row.seed)
        timesteps = self.adapter.set_timesteps(num_steps, latents=latents, state=state)
        if len(timesteps) != num_steps:
            raise RuntimeError("Fairness probe received a non-native schedule.")
        condition = self.adapter.prepare_prompt(row.prompt)
        with torch.inference_mode():
            for step_index, timestep in enumerate(timesteps):
                context = self.adapter.denoising_step_context(
                    step_index,
                    len(timesteps),
                    state,
                )
                state.extra["_active_denoising_step_context"] = context
                prediction = self.adapter.predict_vector_field(
                    latents,
                    timestep,
                    condition,
                    state,
                )
                result = self.adapter.scheduler_step(
                    model_prediction=prediction,
                    timestep=timestep,
                    latents=latents,
                    state=state,
                    generator=generator,
                )
                latents, state = result.latents, result.state
            media = self.adapter.decode_latents(latents, state)
        images = _pil_images(media)
        if len(images) != 1:
            raise RuntimeError(
                f"Fairness probe expected one PIL image, observed {len(images)}."
            )
        image_path = staging / "image.png"
        images[0].save(image_path, format="PNG")
        metadata = {
            "schema_version": 1,
            "purpose": "train_only_demographically_neutral_fairness_reference_probe",
            "model_id": self.model_id,
            "model_hf_id": self.adapter.model_id,
            "model_revision": self.adapter.config.get("revision"),
            "adapter": self.adapter.adapter_name,
            "probe_row": row.__dict__,
            "probe_manifest_path": str(PROBE_MANIFEST),
            "probe_manifest_sha256": self.manifest_sha,
            "generation": generation,
            "image": {
                "path": "image.png",
                "sha256": file_sha256(image_path),
                "width": images[0].width,
                "height": images[0].height,
            },
            "final_latents": tensor_stats(latents),
            "conditioning_provenance": self.adapter.conditioning_provenance(),
            "execution": execution_identity(),
            "started_at": started_at,
            "completed_at": _utc(),
            "elapsed_seconds": time.perf_counter() - start,
        }
        atomic_json(staging / "metadata.json", metadata)
        atomic_json(
            staging / "_SUCCESS.json",
            {
                "status": "completed",
                "model_id": self.model_id,
                "probe_id": row.probe_id,
                "image_sha256": metadata["image"]["sha256"],
                "completed_at": metadata["completed_at"],
            },
        )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Build and generate train-only demographic fairness probes."
    )
    sub = value.add_subparsers(dest="command", required=True)
    sub.add_parser("build-manifest")
    run = sub.add_parser("run")
    run.add_argument("--model", required=True)
    run.add_argument("--shard-index", type=int, default=0)
    run.add_argument("--num-shards", type=int, default=1)
    return value


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "build-manifest":
        print(json.dumps(build_manifest(), sort_keys=True))
    elif args.command == "run":
        ProbeRunner(
            model_id=args.model,
            shard_index=args.shard_index,
            num_shards=args.num_shards,
        ).run()
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
