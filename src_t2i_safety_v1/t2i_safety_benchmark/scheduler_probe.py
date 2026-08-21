from __future__ import annotations

import argparse
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from hierasafe_flow.adapters.registry import create_adapter
from hierasafe_flow.utils.config import load_config
from hierasafe_flow.utils.device import configure_cuda, resolve_device, resolve_dtype
from hierasafe_flow.utils.seed import make_generator, seed_everything

from .contracts import CALIBRATION_ROOT, PROJECT_ROOT, BenchmarkContract, execution_identity
from .pilot_context import runtime_timestep_values


def _write_once(path: Path, payload: dict[str, object]) -> None:
    encoded = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
        + "\n"
    ).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        comparable = {
            key: existing.get(key)
            for key in (
                "protocol",
                "status",
                "model_id",
                "num_inference_steps",
                "step_indices",
                "timesteps",
            )
        }
        expected = {key: payload[key] for key in comparable}
        if comparable != expected:
            raise FileExistsError(f"Existing scheduler grid conflicts with runtime: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.staging-", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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
        probe_row_id = "scheduler_safety_neutral_probe_v1"
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
    values = runtime_timestep_values(timesteps)
    if len(values) != num_steps:
        raise RuntimeError("Adapter returned an incomplete native scheduler grid")
    output = CALIBRATION_ROOT / "scheduler_grids_v2" / model_id / "scheduler.json"
    _write_once(
        output,
        {
            "schema_version": 2,
            "protocol": "t2i_safety_scheduler_grid_v2",
            "status": "sealed",
            "model_id": model_id,
            "num_inference_steps": num_steps,
            "step_indices": list(range(num_steps)),
            "timesteps": list(values),
            "adapter": adapter.adapter_name,
            "pipeline_class": adapter.pipeline_class_name,
            "model_revision": adapter.config.get("revision"),
            "probe_row_id": probe_row_id,
            "probe_prompt": probe_prompt,
            "probe_seed": seed,
            "execution": execution_identity(),
            "sealed_at": datetime.now(timezone.utc).isoformat(),
        },
    )
    return output


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Seal one model's native scheduler grid.")
    value.add_argument("--model", required=True)
    return value


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    print(run(args.model))


if __name__ == "__main__":
    main()
