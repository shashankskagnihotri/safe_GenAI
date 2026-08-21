#!/usr/bin/env python
"""Measure real H100 model-load CUDA peaks for every benchmark model.

The parent process launches one short-lived child in the exact registered
Conda environment for each model.  A fresh process is important here: it
prevents allocator state from one very large pipeline from contaminating the
next measurement and guarantees that CUDA memory is released when the child
exits.  Children load the real pinned adapter but never prepare latents or run
a denoising step.

Only the final six-field aggregate is written to stdout so it can be embedded
verbatim in the production-smoke no-generation receipt.  Per-model commands,
timings, environment identities, pinned revisions, live implementation hashes,
and CUDA measurements are written to a separate immutable detail document.
The receipt proves twelve adapter ``load`` calls and zero latent preparation,
denoising, or generation calls; it is not a synthetic capacity estimate.
"""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
from copy import deepcopy
from datetime import datetime, timezone
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping


PEAK_DETAIL_SCHEMA_VERSION = 2
PEAK_DETAIL_CONTRACT = "finer_detailing_real_model_load_peak_v2"
PEAK_MEASUREMENT_SOURCE_FILES = (
    "scripts/measure_finer_detailing_model_load_peak.py",
    "scripts/finer_detailing_environment_dispatch.py",
    "src/hierasafe_flow/adapters/registry.py",
    "src/hierasafe_flow/benchmarks/finer_detailing_correction.py",
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(payload: Any) -> bytes:
    return (
        json.dumps(payload, allow_nan=False, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _atomic_write_new(path: Path, payload: bytes) -> None:
    """Publish one new file without replacement and fsync its directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary, path)
        temporary.unlink()
        temporary = None
        directory_fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _model_load_config(model_name: str, root: Path) -> dict[str, Any]:
    """Reconstruct the exact model subsection used by GenerationRunner."""

    from hierasafe_flow.benchmarks import finer_detailing_correction as finer

    config_path, resolved_name = finer._resolve_model_config(root, model_name)
    if resolved_name != model_name:
        raise RuntimeError(
            f"Model config resolution drifted: requested {model_name}, got {resolved_name}."
        )
    config = finer.load_yaml(config_path)
    model = deepcopy(dict(config["model"]))
    generation = dict(config["generation"])
    for key in ("height", "width", "num_frames", "fps"):
        if key in generation and key not in model:
            model[key] = generation[key]
    model["guidance_scale"] = generation.get("guidance_scale")
    if model.get("revision") != finer.EXPECTED_MODEL_REVISIONS[model_name]:
        raise RuntimeError(f"Pinned model revision drifted for {model_name}.")
    return model


def _child_measurement(model_name: str, root: Path, *, require_h100: bool) -> dict[str, Any]:
    import torch

    from hierasafe_flow.adapters.registry import create_adapter
    from hierasafe_flow.benchmarks import finer_detailing_correction as finer
    from hierasafe_flow.utils.device import configure_cuda
    from scripts.finer_detailing_environment_dispatch import (
        _validate_active_conda_environment,
        contract_for_model,
        validate_diffusers_install,
    )

    if model_name not in finer.MODEL_NAMES:
        raise ValueError(f"Unknown canonical finer-detailing model {model_name!r}.")
    contract = contract_for_model(model_name)
    environment = _validate_active_conda_environment(contract)
    diffusers = validate_diffusers_install(contract)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Peak-load measurement requires exactly one visible CUDA device.")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    device_name = str(properties.name)
    if require_h100 and "H100" not in device_name.upper():
        raise RuntimeError(f"Peak-load qualification requires an H100, found {device_name!r}.")

    configure_cuda(True)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    before_allocated = int(torch.cuda.memory_allocated(device))
    before_reserved = int(torch.cuda.memory_reserved(device))
    started_at = _utc_now()
    start = time.perf_counter()
    adapter = None
    try:
        model_config = _model_load_config(model_name, root)
        adapter = create_adapter(model_config, device=device, dtype=torch.bfloat16)
        # Third-party loaders occasionally print progress to stdout.  Keep the
        # machine-readable child result isolated on its final stdout line.
        with redirect_stdout(sys.stderr):
            adapter.load()
        torch.cuda.synchronize(device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        if not adapter.loaded:
            raise RuntimeError(f"Adapter {model_name} returned without entering loaded state.")
        # Sequential CPU offload is an exact production route for Flux.2 and
        # can intentionally leave zero CUDA bytes resident immediately after
        # load.  The authenticated `adapter.loaded` state proves that route;
        # the aggregate still requires a positive worst-case CUDA peak.
        if not (0 <= peak_allocated <= peak_reserved < int(properties.total_memory)):
            raise RuntimeError(
                f"Measured CUDA peak is infeasible for {model_name}: "
                f"allocated={peak_allocated}, reserved={peak_reserved}, "
                f"total={int(properties.total_memory)}."
            )
        completed_at = _utc_now()
        duration = time.perf_counter() - start
        result = {
            "schema_version": 2,
            "measurement": "real_adapter_model_load_cuda_peak",
            "model_name": model_name,
            "model_revision": model_config["revision"],
            "environment": environment,
            "diffusers": diffusers,
            "device_name": device_name,
            "total_memory_bytes": int(properties.total_memory),
            "baseline_allocated_bytes": before_allocated,
            "baseline_reserved_bytes": before_reserved,
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "headroom_bytes": int(properties.total_memory) - peak_reserved,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "duration_seconds": duration,
            "adapter_load_calls": 1,
            "latent_preparation_calls": 0,
            "denoising_step_calls": 0,
            "generation_calls": 0,
        }
    finally:
        del adapter
        gc.collect()
        torch.cuda.empty_cache()
    return result


def _parse_child_stdout(stdout: str, *, model_name: str) -> dict[str, Any]:
    lines = [line for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Model-load child {model_name} returned no JSON result.")
    try:
        payload = json.loads(lines[-1])
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Model-load child {model_name} final stdout line is not JSON.") from exc
    if not isinstance(payload, dict) or payload.get("model_name") != model_name:
        raise RuntimeError(f"Model-load child identity drifted for {model_name}.")
    return payload


def aggregate_measurements(
    records: list[Mapping[str, Any]], expected_models: list[str]
) -> dict[str, Any]:
    """Validate per-model measurements and derive the six-field gate payload."""

    if [record.get("model_name") for record in records] != expected_models:
        raise ValueError("Peak-load records are missing, duplicated, or reordered.")
    if len(set(expected_models)) != len(expected_models):
        raise ValueError("Peak-load model selection contains duplicates.")
    devices = {str(record.get("device_name")) for record in records}
    totals = {record.get("total_memory_bytes") for record in records}
    if len(devices) != 1 or len(totals) != 1:
        raise ValueError("All model-load measurements must use the same H100 device class.")
    device_name = next(iter(devices))
    total = next(iter(totals))
    if "H100" not in device_name.upper() or isinstance(total, bool) or not isinstance(total, int):
        raise ValueError("Peak-load aggregate is not bound to one H100 memory contract.")
    for record in records:
        allocated = record.get("peak_allocated_bytes")
        reserved = record.get("peak_reserved_bytes")
        if (
            record.get("schema_version") != 2
            or record.get("measurement") != "real_adapter_model_load_cuda_peak"
            or record.get("adapter_load_calls") != 1
            or record.get("latent_preparation_calls") != 0
            or record.get("denoising_step_calls") != 0
            or record.get("generation_calls") != 0
            or isinstance(allocated, bool)
            or not isinstance(allocated, int)
            or isinstance(reserved, bool)
            or not isinstance(reserved, int)
            or not 0 <= allocated <= reserved < total
        ):
            raise ValueError(f"Invalid real model-load measurement: {record.get('model_name')}.")
    peak_allocated = max(int(record["peak_allocated_bytes"]) for record in records)
    peak_reserved = max(int(record["peak_reserved_bytes"]) for record in records)
    if peak_allocated <= 0 or peak_reserved <= 0:
        raise ValueError("No real model load produced a positive CUDA allocation.")
    return {
        "device_name": device_name,
        "total_memory_bytes": total,
        "peak_allocated_bytes": peak_allocated,
        "peak_reserved_bytes": peak_reserved,
        "headroom_bytes": total - peak_reserved,
        "workload_condition_ids": [f"model_load:{model}" for model in expected_models],
    }


def _parent_measurement(
    *,
    models: list[str],
    root: Path,
    detail_output: Path,
    require_h100: bool,
) -> dict[str, Any]:
    from hierasafe_flow.benchmarks import finer_detailing_correction as finer
    from scripts.finer_detailing_environment_dispatch import contract_for_model

    if not models:
        models = list(finer.MODEL_NAMES)
    if len(set(models)) != len(models) or any(model not in finer.MODEL_NAMES for model in models):
        raise ValueError("--model must select distinct canonical benchmark models.")
    if (
        detail_output.exists()
        or detail_output.with_suffix(detail_output.suffix + ".sha256").exists()
    ):
        raise FileExistsError(f"Refusing to overwrite model-load evidence: {detail_output}")

    records: list[dict[str, Any]] = []
    invocations: list[dict[str, Any]] = []
    script = Path(__file__).resolve()
    raw_conda_executable = os.environ.get("CONDA_EXE") or shutil.which("conda")
    if not raw_conda_executable:
        raise RuntimeError("Cannot locate the registered Conda installation.")
    conda_executable = Path(raw_conda_executable).resolve()
    if conda_executable.parent.name != "bin":
        raise RuntimeError(f"Unexpected Conda executable layout: {conda_executable}")
    conda_root = conda_executable.parents[1]
    for model_name in models:
        contract = contract_for_model(model_name)
        prefix = conda_root / "envs" / contract.name
        python = (prefix / "bin" / "python").resolve()
        if not python.is_file():
            raise FileNotFoundError(
                f"Registered environment Python is missing for {model_name}: {python}"
            )
        command = [
            str(python),
            str(script),
            "--child",
            "--model",
            model_name,
            "--project-root",
            str(root),
        ]
        if require_h100:
            command.append("--require-h100")
        environment = dict(os.environ)
        environment.update(
            {
                "CONDA_DEFAULT_ENV": contract.name,
                "CONDA_PREFIX": str(prefix.resolve()),
                "PYTHONNOUSERSITE": "1",
            }
        )
        started_at = _utc_now()
        start = time.perf_counter()
        completed = subprocess.run(
            command,
            cwd=root,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
        )
        duration = time.perf_counter() - start
        completed_at = _utc_now()
        invocation = {
            "model_name": model_name,
            "environment_name": contract.name,
            "command": command,
            "started_at_utc": started_at,
            "completed_at_utc": completed_at,
            "duration_seconds": duration,
            "exit_code": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
        }
        invocations.append(invocation)
        if completed.returncode != 0:
            raise RuntimeError(
                f"Real model-load child failed for {model_name} (exit {completed.returncode}).\n"
                f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
            )
        records.append(_parse_child_stdout(completed.stdout, model_name=model_name))

    aggregate = aggregate_measurements(records, models)
    expected_revisions = {model: finer.EXPECTED_MODEL_REVISIONS[model] for model in models}
    expected_environments = {model: contract_for_model(model).name for model in models}
    source_files_sha256: dict[str, str] = {}
    for relative in PEAK_MEASUREMENT_SOURCE_FILES:
        source = (root / relative).resolve()
        if source != root and root not in source.parents or not source.is_file():
            raise RuntimeError(f"Peak-measurement source is missing or escaped root: {source}")
        source_files_sha256[relative] = hashlib.sha256(source.read_bytes()).hexdigest()
    detail = {
        "schema_version": PEAK_DETAIL_SCHEMA_VERSION,
        "contract": PEAK_DETAIL_CONTRACT,
        "project_root": str(root),
        "created_at_utc": _utc_now(),
        "require_h100": require_h100,
        "source_files_sha256": source_files_sha256,
        "models": models,
        "expected_model_revisions": expected_revisions,
        "expected_model_environments": expected_environments,
        "adapter_load_call_count": len(records),
        "latent_preparation_call_count": 0,
        "denoising_step_call_count": 0,
        "generation_call_count": 0,
        "records": records,
        "invocations": invocations,
        "aggregate": aggregate,
    }
    detail["document_sha256"] = _sha256_bytes(_canonical_bytes(detail))
    encoded = _canonical_bytes(detail)
    _atomic_write_new(detail_output, encoded)
    _atomic_write_new(
        detail_output.with_suffix(detail_output.suffix + ".sha256"),
        f"{detail['document_sha256']}  {detail_output.name}\n".encode("utf-8"),
    )
    return aggregate


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", action="append", default=[])
    parser.add_argument("--project-root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--detail-output")
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--require-h100", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.project_root).resolve()
    os.chdir(root)
    if args.child:
        if len(args.model) != 1 or args.detail_output is not None:
            raise ValueError("Internal --child mode requires exactly one model and no output path.")
        payload = _child_measurement(args.model[0], root, require_h100=args.require_h100)
        print(json.dumps(payload, allow_nan=False, sort_keys=True, separators=(",", ":")))
        return 0
    if args.detail_output is None:
        raise ValueError("Parent model-load measurement requires --detail-output.")
    detail_output = Path(args.detail_output)
    if not detail_output.is_absolute():
        detail_output = (root / detail_output).resolve()
    aggregate = _parent_measurement(
        models=list(args.model),
        root=root,
        detail_output=detail_output,
        require_h100=args.require_h100,
    )
    print(json.dumps(aggregate, allow_nan=False, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
