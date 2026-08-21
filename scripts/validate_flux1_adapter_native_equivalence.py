#!/usr/bin/env python3
"""Real-model, checkpoint-by-checkpoint FluxAdapter/native equivalence gate.

This diagnostic is deliberately fixed to the P1 seed-zero baseline used by the
sealed Flux.1 native-negative calibration.  It loads one pinned pipeline, runs
the native no-negative call, then reruns the denoising computation through the
adapter from a freshly generated seed-identical latent.  Every comparison is
strict: hashes, shapes, dtypes, and maximum absolute differences are recorded,
and any non-bitwise-identical checkpoint fails the process.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from importlib.metadata import distribution
from pathlib import Path
from typing import Any

import numpy as np
import torch

from hierasafe_flow.adapters.flux_adapter import FluxAdapter
from hierasafe_flow.utils.seed import make_generator, seed_everything


SCHEMA_VERSION = 1
MODEL_ID = "black-forest-labs/FLUX.1-dev"
MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"
DIFFUSERS_COMMIT = "577b28f8f5d30eabdd357d74944cd76568292faf"
DIFFUSERS_VERSION = "0.39.0.dev0"
PIPELINE_SOURCE_SHA256 = "ac0613aca45759fc2a5f4b5310ac9f2d8ea7f4cc34f04adada03b80896eaec92"
SCHEDULER_SOURCE_SHA256 = "8b96f25a6170480fb02decb58904e026250d6d68be5de8a7b8785b4c721e9d80"
MODEL_INDEX_SHA256 = "24946df21ff25e210486b5f6b14208983a90c9c73f8d48cfa724c0e4e03f7201"
MODEL_CONFIG_SHA256 = "5b958846975485c70f86a97695af5fed5ab6c609ed9ecea0f00b281c506d1843"
FLUX_ADAPTER_SHA256 = "114f76c09b76198f254bf5a7d5cf4a5b2cef5e12256835fd7b71fc035e19e439"

PROMPT = (
    "A sad young girl wearing a green jacket, sitting on a wooden park bench "
    "in a sunny park, eating a sandwich, realistic setting, natural lighting, great "
    "detail, full body visible, bench clearly visible, sandwich clearly visible. Only "
    "this one young girl is present and there are no other people anywhere in the scene. "
    "Use a portrait-oriented wide contextual composition with her complete head, torso, "
    "both legs, and both feet inside the frame, with visible space around the complete "
    "bench; do not crop any part of her body or the bench. Her sad expression is unmistakable "
    "and the sandwich touches her mouth as she visibly takes a bite."
)
SEED = 0
HEIGHT = 1216
WIDTH = 832
NUM_INFERENCE_STEPS = 28
GUIDANCE_SCALE = 3.5
TRUE_CFG_SCALE = 1.0
EXPECTED_LATENT_SHAPE = [1, 3952, 64]


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _clone_cpu(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.detach().contiguous().cpu().clone()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    value = tensor.detach().contiguous().cpu()
    return value.view(torch.uint8).numpy().tobytes()


def _tensor_summary(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().contiguous().cpu()
    finite = torch.isfinite(value.float())
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": _sha256_bytes(_tensor_bytes(value)),
        "finite": bool(finite.all().item()),
        "min": float(value.float().min().item()) if value.numel() else None,
        "max": float(value.float().max().item()) if value.numel() else None,
        "mean": float(value.float().mean().item()) if value.numel() else None,
    }


def _compare_tensors(
    comparison_id: str,
    native: torch.Tensor,
    adapter: torch.Tensor,
) -> dict[str, Any]:
    native_cpu = native.detach().contiguous().cpu()
    adapter_cpu = adapter.detach().contiguous().cpu()
    same_shape = native_cpu.shape == adapter_cpu.shape
    same_dtype = native_cpu.dtype == adapter_cpu.dtype
    exact = bool(same_shape and same_dtype and torch.equal(native_cpu, adapter_cpu))
    max_abs = None
    mean_abs = None
    if same_shape and native_cpu.numel():
        delta = (native_cpu.float() - adapter_cpu.float()).abs()
        max_abs = float(delta.max().item())
        mean_abs = float(delta.mean().item())
    return {
        "comparison_id": comparison_id,
        "passed": exact,
        "required_relation": "bitwise_identical",
        "same_shape": same_shape,
        "same_dtype": same_dtype,
        "max_abs_difference": max_abs,
        "mean_abs_difference": mean_abs,
        "native": _tensor_summary(native_cpu),
        "adapter": _tensor_summary(adapter_cpu),
    }


def _extract_tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, tuple) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    sample = getattr(output, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample
    raise TypeError(f"Cannot extract tensor from {type(output).__name__}.")


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New, nonexistent directory for authenticated diagnostic evidence.",
    )
    return parser


def _environment_preflight(project_root: Path) -> dict[str, Any]:
    import diffusers
    import transformers
    from diffusers import FluxPipeline
    from diffusers.schedulers import scheduling_flow_match_euler_discrete
    from huggingface_hub import snapshot_download

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Equivalence gate requires exactly one visible CUDA GPU.")
    gpu_name = torch.cuda.get_device_name(0)
    if "H100" not in gpu_name.upper():
        raise RuntimeError(f"Equivalence gate is qualified only on H100; observed {gpu_name!r}.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("Visible H100 does not report bfloat16 support.")
    if diffusers.__version__ != DIFFUSERS_VERSION:
        raise RuntimeError(
            f"Diffusers version drift: {diffusers.__version__!r} != {DIFFUSERS_VERSION!r}."
        )
    direct_url_raw = distribution("diffusers").read_text("direct_url.json")
    direct_url = json.loads(direct_url_raw) if direct_url_raw else {}
    installed_commit = direct_url.get("vcs_info", {}).get("commit_id")
    if installed_commit != DIFFUSERS_COMMIT:
        raise RuntimeError(
            f"Diffusers commit drift: {installed_commit!r} != {DIFFUSERS_COMMIT!r}."
        )

    pipeline_source = Path(inspect.getsourcefile(FluxPipeline) or "").resolve()
    scheduler_source = Path(
        inspect.getsourcefile(scheduling_flow_match_euler_discrete) or ""
    ).resolve()
    adapter_source = Path(inspect.getsourcefile(FluxAdapter) or "").resolve()
    guarded_sources = {
        "diffusers_flux_pipeline": (pipeline_source, PIPELINE_SOURCE_SHA256),
        "diffusers_flow_match_scheduler": (scheduler_source, SCHEDULER_SOURCE_SHA256),
        "hierasafe_flux_adapter": (adapter_source, FLUX_ADAPTER_SHA256),
        "model_config": (
            project_root / "configs/models/t2i_flux1_dev.yaml",
            MODEL_CONFIG_SHA256,
        ),
    }
    source_receipts: dict[str, dict[str, str]] = {}
    for name, (path, expected_sha256) in guarded_sources.items():
        observed = _sha256_file(path)
        if observed != expected_sha256:
            raise RuntimeError(
                f"Guarded source drift for {name}: {observed} != {expected_sha256}."
            )
        source_receipts[name] = {
            "path": str(path),
            "sha256": observed,
        }

    snapshot = Path(
        snapshot_download(
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
            local_files_only=True,
        )
    ).resolve()
    model_index = snapshot / "model_index.json"
    if _sha256_file(model_index) != MODEL_INDEX_SHA256:
        raise RuntimeError("Pinned FLUX.1-dev model_index.json digest drifted.")

    return {
        "gpu": {
            "name": gpu_name,
            "count": torch.cuda.device_count(),
            "compute_capability": list(torch.cuda.get_device_capability(0)),
        },
        "packages": {
            "torch": torch.__version__,
            "diffusers": diffusers.__version__,
            "diffusers_commit": installed_commit,
            "transformers": transformers.__version__,
        },
        "sources": source_receipts,
        "model_snapshot": {
            "model_id": MODEL_ID,
            "revision": MODEL_REVISION,
            "path": str(snapshot),
            "model_index_sha256": MODEL_INDEX_SHA256,
        },
        "slurm": {
            "job_id": os.environ.get("SLURM_JOB_ID"),
            "job_name": os.environ.get("SLURM_JOB_NAME"),
            "node_list": os.environ.get("SLURM_JOB_NODELIST"),
        },
    }


def _run_equivalence(output_dir: Path, project_root: Path) -> dict[str, Any]:
    from diffusers import FluxPipeline

    run_started_at = _utc_now()
    started = time.perf_counter()
    preflight = _environment_preflight(project_root)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")
    dtype = torch.bfloat16

    load_started = time.perf_counter()
    pipe = FluxPipeline.from_pretrained(
        MODEL_ID,
        revision=MODEL_REVISION,
        torch_dtype=dtype,
        local_files_only=True,
        low_cpu_mem_usage=True,
    )
    pipe.to(device)
    load_seconds = time.perf_counter() - load_started
    scheduler_config = dict(pipe.scheduler.config)
    if scheduler_config.get("use_dynamic_shifting") is not True:
        raise RuntimeError("Pinned scheduler no longer enables dynamic shifting.")
    if bool(scheduler_config.get("use_flow_sigmas", False)):
        raise RuntimeError("Pinned equivalence protocol does not admit use_flow_sigmas=True.")

    native_transformer_records: list[dict[str, torch.Tensor]] = []
    native_post_latents: list[torch.Tensor] = []
    native_timesteps: list[torch.Tensor] = []
    native_vae_outputs: list[torch.Tensor] = []

    def native_transformer_hook(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> None:
        required = {
            "hidden_states",
            "timestep",
            "guidance",
            "pooled_projections",
            "encoder_hidden_states",
            "txt_ids",
            "img_ids",
        }
        missing = sorted(required - set(kwargs))
        if missing:
            raise RuntimeError(f"Native transformer hook lacks required kwargs: {missing}.")
        native_transformer_records.append(
            {
                key: _clone_cpu(kwargs[key])
                for key in sorted(required)
            }
            | {"prediction": _clone_cpu(_extract_tensor(output))}
        )

    def native_vae_hook(
        _module: torch.nn.Module,
        _args: tuple[Any, ...],
        output: Any,
    ) -> None:
        native_vae_outputs.append(_clone_cpu(_extract_tensor(output)))

    def native_callback(
        _pipeline: Any,
        _step_index: int,
        timestep: torch.Tensor,
        callback_kwargs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        native_timesteps.append(_clone_cpu(timestep))
        native_post_latents.append(_clone_cpu(callback_kwargs["latents"]))
        return callback_kwargs

    transformer_handle = pipe.transformer.register_forward_hook(
        native_transformer_hook,
        with_kwargs=True,
    )
    # FluxPipeline calls ``vae.decode`` directly, so a hook on the VAE module's
    # ``forward`` method is never invoked.  The pinned VAE decode path invokes
    # its decoder module exactly once; instrument that internal forward call
    # without replacing or wrapping the numerical implementation.
    vae_handle = pipe.vae.decoder.register_forward_hook(native_vae_hook)
    seed_everything(SEED)
    native_started = time.perf_counter()
    with torch.inference_mode():
        native_output = pipe(
            prompt=PROMPT,
            negative_prompt=None,
            true_cfg_scale=TRUE_CFG_SCALE,
            height=HEIGHT,
            width=WIDTH,
            num_inference_steps=NUM_INFERENCE_STEPS,
            guidance_scale=GUIDANCE_SCALE,
            num_images_per_prompt=1,
            generator=make_generator(SEED, device),
            output_type="pil",
            return_dict=True,
            callback_on_step_end=native_callback,
            callback_on_step_end_tensor_inputs=["latents"],
        )
    torch.cuda.synchronize()
    native_seconds = time.perf_counter() - native_started
    native_sigmas = _clone_cpu(pipe.scheduler.sigmas)
    transformer_handle.remove()
    vae_handle.remove()
    if len(native_transformer_records) != NUM_INFERENCE_STEPS:
        raise RuntimeError(
            f"Native call made {len(native_transformer_records)} transformer calls, expected 28."
        )
    if len(native_post_latents) != NUM_INFERENCE_STEPS or len(native_timesteps) != NUM_INFERENCE_STEPS:
        raise RuntimeError("Native callback did not capture every denoising step.")
    if len(native_vae_outputs) != 1:
        raise RuntimeError("Native call did not produce exactly one VAE decoder output.")
    native_image = native_output.images[0]
    native_image.save(output_dir / "native_pipeline.png")

    adapter = FluxAdapter(
        model_id=MODEL_ID,
        device=device,
        dtype=dtype,
        config={
            "revision": MODEL_REVISION,
            "guidance_scale": GUIDANCE_SCALE,
        },
    )
    adapter.pipeline = pipe
    adapter.loaded = True
    seed_everything(SEED)
    adapter_started = time.perf_counter()
    with torch.inference_mode():
        condition = adapter.prepare_prompt(PROMPT)
        adapter_generator = make_generator(SEED, device)
        latents, state = adapter.prepare_initial_latents(
            prompt=PROMPT,
            batch_size=1,
            generator=adapter_generator,
            height=HEIGHT,
            width=WIDTH,
        )
        if list(latents.shape) != EXPECTED_LATENT_SHAPE:
            raise RuntimeError(
                f"Packed latent shape drift: {list(latents.shape)} != {EXPECTED_LATENT_SHAPE}."
            )
        initial_adapter_latents = _clone_cpu(latents)
        state.extra["num_steps"] = NUM_INFERENCE_STEPS
        adapter_timesteps = adapter.set_timesteps(
            NUM_INFERENCE_STEPS,
            latents=latents,
            state=state,
        )
        adapter_sigmas = _clone_cpu(pipe.scheduler.sigmas)
        adapter_predictions: list[torch.Tensor] = []
        adapter_post_latents: list[torch.Tensor] = []
        for timestep in adapter_timesteps:
            prediction = adapter.predict_vector_field(latents, timestep, condition, state)
            adapter_predictions.append(_clone_cpu(prediction))
            step = adapter.scheduler_step(
                model_prediction=prediction,
                timestep=timestep,
                latents=latents,
                state=state,
                generator=adapter_generator,
            )
            latents, state = step.latents, step.state
            adapter_post_latents.append(_clone_cpu(latents))

        adapter_vae_outputs: list[torch.Tensor] = []

        def adapter_vae_hook(
            _module: torch.nn.Module,
            _args: tuple[Any, ...],
            output: Any,
        ) -> None:
            adapter_vae_outputs.append(_clone_cpu(_extract_tensor(output)))

        adapter_vae_handle = pipe.vae.decoder.register_forward_hook(adapter_vae_hook)
        adapter_media = adapter.decode_latents(latents, state)
        adapter_vae_handle.remove()
    torch.cuda.synchronize()
    adapter_seconds = time.perf_counter() - adapter_started
    if len(adapter_vae_outputs) != 1:
        raise RuntimeError("Adapter route did not produce exactly one VAE decoder output.")
    adapter_image = adapter_media[0]
    adapter_image.save(output_dir / "flux_adapter.png")

    comparisons: list[dict[str, Any]] = []
    first_native = native_transformer_records[0]
    comparisons.extend(
        [
            _compare_tensors(
                "conditioning.prompt_embeds",
                first_native["encoder_hidden_states"],
                condition.data["prompt_embeds"],
            ),
            _compare_tensors(
                "conditioning.pooled_prompt_embeds",
                first_native["pooled_projections"],
                condition.data["pooled_prompt_embeds"],
            ),
            _compare_tensors(
                "conditioning.text_ids",
                first_native["txt_ids"],
                condition.data["text_ids"],
            ),
            _compare_tensors(
                "latents.initial_packed",
                first_native["hidden_states"],
                initial_adapter_latents,
            ),
            _compare_tensors(
                "latents.image_ids",
                first_native["img_ids"],
                state.extra["latent_image_ids"],
            ),
            _compare_tensors("schedule.sigmas", native_sigmas, adapter_sigmas),
            _compare_tensors(
                "schedule.timesteps",
                torch.stack(native_timesteps),
                torch.stack([_clone_cpu(value) for value in adapter_timesteps]),
            ),
            _compare_tensors(
                "transformer.embedded_guidance",
                first_native["guidance"],
                torch.full((1,), GUIDANCE_SCALE, dtype=torch.float32),
            ),
        ]
    )
    for step_index in range(NUM_INFERENCE_STEPS):
        comparisons.extend(
            [
                _compare_tensors(
                    f"step.{step_index:02d}.normalized_timestep",
                    native_transformer_records[step_index]["timestep"],
                    torch.as_tensor(adapter_timesteps[step_index])
                    .reshape(1)
                    .to(dtype=torch.bfloat16)
                    / 1000,
                ),
                _compare_tensors(
                    f"step.{step_index:02d}.prediction",
                    native_transformer_records[step_index]["prediction"],
                    adapter_predictions[step_index],
                ),
                _compare_tensors(
                    f"step.{step_index:02d}.post_scheduler_latents",
                    native_post_latents[step_index],
                    adapter_post_latents[step_index],
                ),
            ]
        )
    comparisons.append(
        _compare_tensors(
            "decode.vae_output",
            native_vae_outputs[0],
            adapter_vae_outputs[0],
        )
    )
    native_rgb = torch.from_numpy(np.asarray(native_image).copy())
    adapter_rgb = torch.from_numpy(np.asarray(adapter_image).copy())
    comparisons.append(_compare_tensors("decode.rgb_uint8", native_rgb, adapter_rgb))

    failures = [row["comparison_id"] for row in comparisons if not row["passed"]]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if not failures else "failed",
        "gate": "bitwise_native_pipeline_flux_adapter_equivalence",
        "started_at": run_started_at,
        "environment_preflight": preflight,
        "frozen_protocol": {
            "prompt": PROMPT,
            "prompt_sha256": _sha256_bytes(PROMPT.encode("utf-8")),
            "seed": SEED,
            "height": HEIGHT,
            "width": WIDTH,
            "num_inference_steps": NUM_INFERENCE_STEPS,
            "guidance_scale": GUIDANCE_SCALE,
            "true_cfg_scale": TRUE_CFG_SCALE,
            "negative_prompt": None,
            "dtype": "torch.bfloat16",
        },
        "scheduler_config": scheduler_config,
        "timing_seconds": {
            "model_load": load_seconds,
            "native_pipeline": native_seconds,
            "flux_adapter": adapter_seconds,
            "total": time.perf_counter() - started,
        },
        "comparisons": comparisons,
        "failed_comparison_ids": failures,
        "media": {
            "native_pipeline": {
                "path": str((output_dir / "native_pipeline.png").resolve()),
                "sha256": _sha256_file(output_dir / "native_pipeline.png"),
            },
            "flux_adapter": {
                "path": str((output_dir / "flux_adapter.png").resolve()),
                "sha256": _sha256_file(output_dir / "flux_adapter.png"),
            },
        },
        "ended_at": _utc_now(),
    }
    return result


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    project_root = Path(__file__).resolve().parents[1]
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    result_path = output_dir / "equivalence_result.json"
    try:
        result = _run_equivalence(output_dir, project_root)
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "error",
            "gate": "bitwise_native_pipeline_flux_adapter_equivalence",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "ended_at": _utc_now(),
        }
        _write_json_atomic(result_path, failure)
        raise
    _write_json_atomic(result_path, result)
    print(json.dumps({"result": str(result_path), "status": result["status"]}, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    sys.exit(main())
