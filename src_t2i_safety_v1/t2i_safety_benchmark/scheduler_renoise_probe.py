from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from hierasafe_flow.adapters.registry import create_adapter
from hierasafe_flow.utils.config import load_config
from hierasafe_flow.utils.device import configure_cuda, resolve_device, resolve_dtype
from hierasafe_flow.utils.seed import make_generator, seed_everything

from .contracts import CALIBRATION_ROOT, PROJECT_ROOT, BenchmarkContract, execution_identity
from .methods import _native_scheduler, native_renoise, parameterization_for_model


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scheduler_timestep(adapter: Any, model_id: str, timestep: Any) -> torch.Tensor:
    value = timestep
    if model_id == "ideogram4_nf4":
        extractor = getattr(adapter, "_timestep_values", None)
        if not callable(extractor):
            raise RuntimeError("Ideogram4 composite-timestep extractor is unavailable.")
        value, _ = extractor(timestep)
    result = torch.as_tensor(value).reshape(-1)
    if result.numel() != 1:
        raise RuntimeError("Scheduler probe requires one native scheduler timestep.")
    return result[0]


def _independent_expected(
    *,
    scheduler: Any,
    parameterization: str,
    clean: torch.Tensor,
    noise: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, int, float]:
    schedule = torch.as_tensor(
        scheduler.timesteps,
        device=clean.device,
    )
    value = timestep.to(device=clean.device, dtype=schedule.dtype)
    matches = (schedule == value).nonzero(as_tuple=False).flatten()
    if matches.numel() < 1:
        raise RuntimeError("Probe timestep is not an exact native schedule member.")
    index_for_timestep = getattr(scheduler, "index_for_timestep", None)
    index = (
        int(index_for_timestep(value, schedule))
        if callable(index_for_timestep)
        else int(matches[0].item())
    )
    if index not in {int(item) for item in matches.tolist()}:
        raise RuntimeError("Scheduler selected an index outside exact timestep matches.")

    if parameterization in {"flow_velocity", "physical_velocity"}:
        sigmas = torch.as_tensor(scheduler.sigmas, device=clean.device, dtype=clean.dtype)
        coefficient = sigmas[index]
        expected = coefficient * noise + (1.0 - coefficient) * clean
    elif hasattr(scheduler, "alphas_cumprod"):
        train_index = int(value.item())
        alphas = torch.as_tensor(
            scheduler.alphas_cumprod,
            device=clean.device,
            dtype=clean.dtype,
        )
        coefficient = alphas[train_index]
        expected = coefficient.sqrt() * clean + (1.0 - coefficient).sqrt() * noise
    elif hasattr(scheduler, "sigmas"):
        sigmas = torch.as_tensor(scheduler.sigmas, device=clean.device, dtype=clean.dtype)
        coefficient = sigmas[index]
        expected = clean + coefficient * noise
    else:
        raise RuntimeError(
            f"No independent forward-noise equation for {type(scheduler).__name__}."
        )
    return expected, index, float(coefficient.detach().cpu())


def run(model_id: str) -> Path:
    contract = BenchmarkContract(verify_large_hashes=True)
    model_spec = contract.model(model_id)
    seed_everything(0)
    configure_cuda(True)
    device = resolve_device("auto")
    dtype = resolve_dtype("bfloat16")
    model_config = load_config(model_spec["config"], project_root=PROJECT_ROOT)
    generation = {
        **dict(model_config["generation"]),
        "height": int(model_spec["height"]),
        "width": int(model_spec["width"]),
        "num_inference_steps": int(model_spec["steps"]),
        "num_outputs_per_prompt": 1,
    }
    model_values = dict(model_config["model"])
    model_values["height"] = int(model_spec["height"])
    model_values["width"] = int(model_spec["width"])
    model_values["guidance_scale"] = float(model_spec["guidance_scale"])
    adapter = create_adapter(model_values, device, dtype)
    adapter.load()

    row = min(contract.prompt_rows().values(), key=lambda item: item.release_index)
    seed = int(row.seed)
    probe_prompt = row.prompt
    probe_row_id = row.row_id
    if model_id == "cosmos3_super_text2image":
        probe_prompt = "A ceramic bowl beside a blue cube on a plain studio table"
        probe_row_id = "scheduler_renoise_safety_neutral_probe_v1"
    generator = make_generator(seed, device)
    latent_kwargs = {
        key: value
        for key, value in generation.items()
        if key not in {"prompt", "prompt_file"}
    }
    latents, state = adapter.prepare_initial_latents(
        prompt=probe_prompt,
        batch_size=1,
        generator=generator,
        **latent_kwargs,
    )
    num_steps = int(model_spec["steps"])
    state.extra["num_steps"] = num_steps
    state.extra["base_seed"] = seed
    timesteps = adapter.set_timesteps(num_steps, latents=latents, state=state)
    if len(timesteps) != num_steps:
        raise RuntimeError("Adapter returned an incomplete native scheduler grid.")

    scheduler = _native_scheduler(adapter)
    parameterization = parameterization_for_model(model_id).value
    clean_single = latents.detach().to(dtype=torch.float32)
    clean = torch.cat([clean_single, clean_single], dim=0).mul(0.0).add(0.25)
    noise = torch.full_like(clean, -0.5)
    probe_indices = sorted({0, num_steps // 2, num_steps - 1})
    evidence: list[dict[str, Any]] = []
    stable_contract: dict[str, Any] | None = None
    for step_index in probe_indices:
        timestep = timesteps[step_index]
        native_timestep = _scheduler_timestep(adapter, model_id, timestep)
        begin_before = getattr(scheduler, "begin_index", None)
        step_before = getattr(scheduler, "step_index", None)
        observed, observed_contract = native_renoise(
            adapter=adapter,
            model_id=model_id,
            clean=clean,
            noise=noise,
            timestep=timestep,
        )
        repeated, repeated_contract = native_renoise(
            adapter=adapter,
            model_id=model_id,
            clean=clean,
            noise=noise,
            timestep=timestep,
        )
        expected, schedule_index, coefficient = _independent_expected(
            scheduler=scheduler,
            parameterization=parameterization,
            clean=clean,
            noise=noise,
            timestep=native_timestep,
        )
        if observed_contract != repeated_contract:
            raise RuntimeError("Native re-noising contract changed across repetitions.")
        if stable_contract is None:
            stable_contract = observed_contract
        elif observed_contract != stable_contract:
            raise RuntimeError("Native re-noising contract changed across timesteps.")
        if getattr(scheduler, "begin_index", None) != begin_before:
            raise RuntimeError("Native re-noising mutated scheduler begin_index.")
        if getattr(scheduler, "step_index", None) != step_before:
            raise RuntimeError("Native re-noising mutated scheduler step_index.")
        if not torch.equal(observed, repeated):
            raise RuntimeError("Native re-noising is not deterministic for fixed tensors.")
        if not torch.allclose(observed, expected, rtol=1.0e-5, atol=1.0e-6):
            error = float((observed - expected).abs().max().detach().cpu())
            sigma_values = getattr(scheduler, "sigmas", None)
            sigma_at_schedule_index = None
            if sigma_values is not None:
                sigma_at_schedule_index = float(
                    torch.as_tensor(sigma_values)[schedule_index].detach().cpu()
                )
            mismatch = {
                "model_id": model_id,
                "scheduler_class": type(scheduler).__name__,
                "parameterization": parameterization,
                "step_index": step_index,
                "native_timestep": float(native_timestep.detach().cpu()),
                "schedule_index": schedule_index,
                "sigma_at_schedule_index": sigma_at_schedule_index,
                "clean_first": float(clean.reshape(-1)[0].detach().cpu()),
                "noise_first": float(noise.reshape(-1)[0].detach().cpu()),
                "observed_first": float(observed.reshape(-1)[0].detach().cpu()),
                "expected_first": float(expected.reshape(-1)[0].detach().cpu()),
                "max_abs_error": error,
                "begin_index_before": begin_before,
                "begin_index_after": getattr(scheduler, "begin_index", None),
                "step_index_before": step_before,
                "step_index_after": getattr(scheduler, "step_index", None),
                "native_contract": observed_contract,
            }
            raise RuntimeError(
                "Native re-noising equation mismatch: "
                + json.dumps(mismatch, sort_keys=True, default=str)
            )
        evidence.append(
            {
                "step_index": step_index,
                "native_timestep": float(native_timestep.detach().cpu()),
                "schedule_index": schedule_index,
                "forward_coefficient": coefficient,
                "batch_size": int(clean.shape[0]),
                "max_abs_error": float(
                    (observed - expected).abs().max().detach().cpu()
                ),
                "deterministic_repeat": True,
                "scheduler_state_preserved": True,
            }
        )
    if stable_contract is None:
        raise RuntimeError("Native re-noising probe produced no evidence.")

    scheduler_grid = (
        CALIBRATION_ROOT / "scheduler_grids_v2" / model_id / "scheduler.json"
    )
    if not scheduler_grid.is_file():
        raise FileNotFoundError(scheduler_grid)
    methods_source = Path(__file__).with_name("methods.py")
    probe_source = Path(__file__)
    output = CALIBRATION_ROOT / "native_renoise_v1" / model_id / "ADMISSION.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "protocol": "t2i_safety_native_renoise_admission_v1",
        "status": "passed",
        "model_id": model_id,
        "model_revision": adapter.config.get("revision"),
        "adapter": adapter.adapter_name,
        "pipeline_class": adapter.pipeline_class_name,
        "scheduler_class": type(scheduler).__name__,
        "parameterization": parameterization,
        "num_inference_steps": num_steps,
        "probe_step_indices": probe_indices,
        "evidence": evidence,
        "native_renoise_contract": stable_contract,
        "scheduler_grid": str(scheduler_grid),
        "scheduler_grid_sha256": _sha256(scheduler_grid),
        "methods_source_sha256": _sha256(methods_source),
        "probe_source_sha256": _sha256(probe_source),
        "probe_row_id": probe_row_id,
        "probe_prompt": probe_prompt,
        "probe_seed": seed,
        "execution": execution_identity(),
        "sealed_at": datetime.now(timezone.utc).isoformat(),
        "fallback_used": False,
    }
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    )
    with output.open("x", encoding="utf-8") as handle:
        handle.write(encoded)
        handle.flush()
    return output


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Admit one model's scheduler-native forward re-noising contract."
    )
    value.add_argument("--model", required=True)
    return value


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    print(run(args.model))


if __name__ == "__main__":
    main()
