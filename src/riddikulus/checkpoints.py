"""Full native solver and Qwen-prefix checkpoints with committed trace journals."""
from __future__ import annotations

import copy
import hashlib
import os
from pathlib import Path
import random

import numpy as np
import torch

from .adapters.base import AdapterState
from .numerics import tensor_sha256
from .artifacts import atomic, digest, read, utc
from .model_state import (
    export_model_state, restore_model_state, scheduler_hash, config_hash, freeze_tree, thaw_tree,
)


SCHEMA = "pool25-warm10.complete-native-checkpoint.v1"


def cpu(value):
    return freeze_tree(value)[0]


def device(value, target):
    if isinstance(value, torch.Tensor):
        return value.to(target)
    if isinstance(value, dict):
        return {k: device(v, target) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(device(v, target) for v in value)
    return copy.deepcopy(value)


def checkpoint(attempt, index, latents, state, scheduler, exposure, generator,
               spec_sha, condition_sha, selection_state, timing_state, *, adapter):
    if adapter.pipeline.scheduler is not scheduler:
        raise RuntimeError("Checkpoint scheduler is not owned by the native adapter")
    target = attempt / "checkpoints" / f"step_{index:02d}.pt"
    offsets = {p.name: p.stat().st_size for p in (attempt / "traces").glob("*.jsonl")}
    runtime, placements = freeze_tree({k: v for k, v in vars(scheduler).items() if k != "_internal_dict"})
    extra, extra_devices = freeze_tree(state.extra)
    saved = {
        "schema": SCHEMA, "spec_sha256": spec_sha, "next_step": index + 1, "completed_step": index,
        "latents": cpu(latents), "latent_sha256": tensor_sha256(latents),
        "state_extra": extra, "state_extra_devices": extra_devices,
        "scheduler_class": f"{type(scheduler).__module__}.{type(scheduler).__name__}",
        "scheduler_config": dict(scheduler.config), "scheduler_runtime": runtime,
        "scheduler_runtime_devices": placements, "scheduler_state_sha256": scheduler_hash(scheduler),
        "model_state": export_model_state(adapter),
        "exposure": cpu(vars(exposure)), "generator": generator.get_state().cpu(),
        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
        "python_rng": random.getstate(), "numpy_rng": np.random.get_state(),
        "base_condition_sha256": condition_sha, "selection_state": cpu(selection_state),
        "timing_state": cpu(timing_state), "attempt": str(attempt), "trace_offsets": offsets,
        "trace_sha256": {name: digest(attempt / "traces" / name) for name in offsets},
    }
    temporary = target.with_suffix(".partial")
    with temporary.open("xb") as stream:
        torch.save(saved, stream)
        stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, target)
    receipt = {"path": str(target), "sha256": digest(target), "completed_step": index,
               "next_step": index + 1, "spec_sha256": spec_sha, "utc": utc(),
               "scheduler_state_sha256": saved["scheduler_state_sha256"],
               "native_model_state_schema": saved["model_state"]["schema"]}
    atomic(target.with_suffix(".json"), receipt)
    atomic(target.parent / "latest.json", receipt)
    return receipt


def restore(receipt_path, attempt, spec_sha, adapter, generator, exposure,
            condition_sha, selection_state, timing_state):
    receipt = read(receipt_path)
    source = Path(receipt["path"])
    if receipt["spec_sha256"] != spec_sha or digest(source) != receipt["sha256"]:
        raise RuntimeError("Explicit checkpoint hash/configuration mismatch")
    saved = torch.load(source, map_location="cpu", weights_only=False)
    scheduler = adapter.pipeline.scheduler
    if (saved["schema"] != SCHEMA or saved["spec_sha256"] != spec_sha
            or saved["base_condition_sha256"] != condition_sha
            or saved["scheduler_class"] != f"{type(scheduler).__module__}.{type(scheduler).__name__}"
            or config_hash(saved["scheduler_config"]) != config_hash(scheduler.config)
            or tensor_sha256(saved["latents"]) != saved["latent_sha256"]
            or saved["next_step"] != saved["completed_step"] + 1
            or saved["next_step"] != receipt["next_step"]):
        raise RuntimeError("Checkpoint native solver, latent, boundary or conditioning identity mismatch")
    for name in ("timesteps", "sigmas"):
        if not torch.equal(getattr(scheduler, name).cpu(), saved["scheduler_runtime"][name]):
            raise RuntimeError(f"Checkpoint native schedule differs: {name}")
    if len(torch.cuda.get_rng_state_all()) != len(saved["cuda_rng"]):
        raise RuntimeError("Exact resume requires the same allocated GPU count")
    original_config = scheduler._internal_dict
    restored = thaw_tree(saved["scheduler_runtime"], saved["scheduler_runtime_devices"])
    vars(scheduler).clear()
    scheduler._internal_dict = original_config
    vars(scheduler).update(restored)
    if scheduler_hash(scheduler) != saved["scheduler_state_sha256"]:
        raise RuntimeError("Restored complete native scheduler state differs from checkpoint")
    restore_model_state(adapter, saved["model_state"])
    selection_state.update(saved["selection_state"])
    timing_state.update(saved["timing_state"])
    vars(exposure).update(saved["exposure"])
    generator.set_state(saved["generator"])
    torch.set_rng_state(saved["torch_rng"]); torch.cuda.set_rng_state_all(saved["cuda_rng"])
    random.setstate(saved["python_rng"]); np.random.set_state(saved["numpy_rng"])
    old = Path(saved["attempt"]) / "traces"
    for name, size in saved["trace_offsets"].items():
        with (old / name).open("rb") as stream:
            payload = stream.read(size)
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != saved["trace_sha256"][name]:
            raise RuntimeError("Checkpoint refers to corrupted committed trace bytes")
        with (attempt / "traces" / name).open("xb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    if "timing_state_after_commit" in receipt:
        journal = receipt["checkpoint_timing_commit"]
        with Path(journal["path"]).open("rb") as stream:
            payload = stream.read(journal["bytes"])
        if len(payload) != journal["bytes"] or hashlib.sha256(payload).hexdigest() != journal["sha256"]:
            raise RuntimeError("Checkpoint timing journal integrity mismatch")
        timing_state.update(receipt["timing_state_after_commit"])
        with (attempt / "traces/checkpoint_timing.jsonl").open("wb") as stream:
            stream.write(payload); stream.flush(); os.fsync(stream.fileno())
    return (saved["latents"].to(adapter.device),
            AdapterState(extra=thaw_tree(saved["state_extra"], saved["state_extra_devices"])),
            saved["next_step"])
