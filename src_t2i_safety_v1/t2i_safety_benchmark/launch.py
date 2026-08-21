from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import (
    CALIBRATION_ROOT,
    OUTPUT_ROOT,
    PROJECT_ROOT,
    WORK_ROOT,
    BenchmarkContract,
    atomic_json,
    file_sha256,
)


GENERATION_ENV = "safe_genai_conceptsteer"
COSMOS_ENV = "t2i_cosmos3"
IMAGEGUARD_ENV = "t2i_imageguard"
CONDA_SH = Path("/ceph/sagnihot/miniconda3/etc/profile.d/conda.sh")
ENVIRONMENT_CONTRACTS = {
    GENERATION_ENV: {
        "prefix": "/ceph/sagnihot/miniconda3/envs/safe_genai_conceptsteer",
        "cuda": "12.4",
    },
    COSMOS_ENV: {
        "prefix": "/ceph/sagnihot/miniconda3/envs/t2i_cosmos3",
        "cuda": "12.4",
        "python_major_minor": "3.11",
        "package_versions": {
            "torch": "2.6.0+cu124",
            "torchvision": "0.21.0+cu124",
            "diffusers": "0.39.0.dev0",
            "transformers": "5.12.1",
            "cosmos-guardrail": "0.3.1",
        },
        "required_imports": ("cosmos_guardrail",),
    },
    IMAGEGUARD_ENV: {
        "prefix": "/ceph/sagnihot/miniconda3/envs/t2i_imageguard",
        "cuda": "12.4",
    },
}
JOB_ROOT = WORK_ROOT / "jobs"
LOG_ROOT = WORK_ROOT / "logs"
RECEIPT_ROOT = WORK_ROOT / "submission"
PILOT_GATE = WORK_ROOT / "pilot_acceptance.json"
MODELS = (
    "cosmos3_super_text2image",
    "flux1_dev",
    "flux2_dev",
    "ideogram4_nf4",
    "qwen_image",
    "qwen_image_2512",
    "sd35_large",
)
VARIANTS = (
    "baseline",
    "native_negative_prompt",
    "conceptsteer",
    "midsteer",
    "sgf",
    "safe_denoiser",
)
RELATED_VARIANTS = ("midsteer", "sgf", "safe_denoiser")
NONFAIR_CATEGORIES = (
    "sexual",
    "hate",
    "humiliation",
    "violence",
    "illegal_activity",
    "disturbing",
    "public_figures",
    "personal_identification_documents",
    "intellectual_property_violation",
)
GPU_PARTITIONS = {"gpu-vram-48gb", "gpu-vram-94gb"}
MODEL_GPU_RESOURCES = {
    "cosmos3_super_text2image": {
        "partition": "gpu-vram-48gb",
        "gpus": 4,
        "cpus": 32,
        "memory": "256G",
    },
    "flux1_dev": {
        "partition": "gpu-vram-94gb",
        "gpus": 1,
        "cpus": 16,
        "memory": "192G",
    },
    "flux2_dev": {
        "partition": "gpu-vram-94gb",
        "gpus": 1,
        "cpus": 16,
        "memory": "192G",
    },
    "ideogram4_nf4": {
        "partition": "gpu-vram-48gb",
        "gpus": 1,
        "cpus": 16,
        "memory": "192G",
    },
    "qwen_image": {
        "partition": "gpu-vram-94gb",
        "gpus": 1,
        "cpus": 16,
        "memory": "192G",
    },
    "qwen_image_2512": {
        "partition": "gpu-vram-94gb",
        "gpus": 1,
        "cpus": 16,
        "memory": "192G",
    },
    "sd35_large": {
        "partition": "gpu-vram-48gb",
        "gpus": 1,
        "cpus": 16,
        "memory": "192G",
    },
}
IMAGEGUARD_GPU_RESOURCES = {
    "partition": "gpu-vram-94gb",
    "gpus": 1,
    "cpus": 16,
    "memory": "192G",
}
IMAGEGUARD_GPU_PROFILES = {
    "h100_94gb": IMAGEGUARD_GPU_RESOURCES,
    "48gb": {
        "partition": "gpu-vram-48gb",
        "gpus": 1,
        "cpus": 16,
        "memory": "128G",
    },
}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _slug(value: str, limit: int = 80) -> str:
    cleaned = "".join(
        character if character.isalnum() or character in "-_" else "-"
        for character in value
    )
    return cleaned[:limit].strip("-")


def _dependency(ids: list[str]) -> str | None:
    values = [str(value) for value in ids if str(value)]
    return "afterok:" + ":".join(values) if values else None


def _unmet_slurm_dependency(job_id: str) -> list[str]:
    if not job_id.isdigit():
        raise ValueError(f"SLURM dependency ID must be numeric: {job_id!r}")
    result = subprocess.run(
        [
            "sacct",
            "-n",
            "-P",
            "-j",
            job_id,
            "--format=JobIDRaw,State,ExitCode",
            "-X",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    records = [
        line.split("|")
        for line in result.stdout.splitlines()
        if line.strip()
    ]
    base = next((record for record in records if record[0] == job_id), None)
    if base is None or len(base) < 3:
        raise RuntimeError(f"No archival SLURM evidence exists for dependency {job_id}")
    state = base[1].split(maxsplit=1)[0]
    exit_code = base[2]
    if state == "COMPLETED" and exit_code == "0:0":
        return []
    if state in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}:
        return [job_id]
    raise RuntimeError(
        f"Dependency {job_id} is not successful or active: state={state}, "
        f"exit_code={exit_code}"
    )


def _runtime_prefix(
    *,
    environment: str,
    require_cuda: bool,
    expected_gpu_count: int,
) -> list[str]:
    if environment not in ENVIRONMENT_CONTRACTS:
        raise KeyError(f"Unknown runtime environment: {environment}")
    environment_contract = ENVIRONMENT_CONTRACTS[environment]
    commands = [
        f"source {shlex.quote(str(CONDA_SH))}",
        f"conda activate {shlex.quote(environment)}",
        f'test "${{CONDA_DEFAULT_ENV:-}}" = {shlex.quote(environment)}',
        f'test "${{CONDA_PREFIX:-}}" = {shlex.quote(environment_contract["prefix"])}',
        f"export PYTHONPATH={shlex.quote(str(PROJECT_ROOT / 'src_t2i_safety_v1'))}:{shlex.quote(str(PROJECT_ROOT / 'src'))}",
        "export PYTHONNOUSERSITE=1",
        "export TOKENIZERS_PARALLELISM=false",
        "export MALLOC_ARENA_MAX=2",
        "export MALLOC_TRIM_THRESHOLD_=131072",
        f"cd {shlex.quote(str(PROJECT_ROOT))}",
    ]
    python_major_minor = environment_contract.get("python_major_minor")
    package_versions = environment_contract.get("package_versions", {})
    required_imports = environment_contract.get("required_imports", ())
    if python_major_minor or package_versions or required_imports:
        validation_code = [
            "import importlib,importlib.metadata as md,sys",
            f"expected_python={python_major_minor!r}",
            f"expected_packages={package_versions!r}",
            f"required_imports={tuple(required_imports)!r}",
            "actual_python=f'{sys.version_info.major}.{sys.version_info.minor}'",
            "actual_packages={name:md.version(name) for name in expected_packages}",
            "assert expected_python is None or actual_python == expected_python, "
            "(actual_python,expected_python)",
            "assert actual_packages == expected_packages, "
            "(actual_packages,expected_packages)",
            "[importlib.import_module(name) for name in required_imports]",
            "print({'python_major_minor':actual_python,"
            "'package_versions':actual_packages,'required_imports':required_imports},"
            "flush=True)",
        ]
        commands.append(
            f"python -c {shlex.quote(';'.join(validation_code))}"
        )
    if require_cuda:
        commands.append(
            "python -c 'import os,sys,torch; "
            "assert sys.prefix == os.environ[\"CONDA_PREFIX\"]; "
            "assert torch.cuda.is_available(); "
            f"assert torch.version.cuda == \"{environment_contract['cuda']}\"; "
            f"assert torch.cuda.device_count() >= {expected_gpu_count}, "
            f"(torch.cuda.device_count(), {expected_gpu_count}); "
            "print({\"python\":sys.executable,\"torch\":torch.__version__,"
            "\"cuda\":torch.version.cuda,\"gpu_count\":torch.cuda.device_count(),"
            "\"gpus\":[torch.cuda.get_device_name(i) "
            "for i in range(torch.cuda.device_count())]},"
            "flush=True)'"
        )
    return commands


def _script_text(
    *,
    job_name: str,
    command: list[str],
    gpu: bool,
    environment: str,
    partition: str | None,
    gpus: int,
    time_limit: str,
    dependency: str | None,
    cpus: int = 16,
    memory: str = "192G",
) -> str:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    lines = [
        "#!/usr/bin/env bash",
        f"#SBATCH --job-name={_slug(job_name, 100)}",
        f"#SBATCH --account=ml-staff",
        f"#SBATCH --cpus-per-task={cpus}",
        f"#SBATCH --mem={memory}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={LOG_ROOT}/%j.out",
        f"#SBATCH --error={LOG_ROOT}/%j.err",
    ]
    if gpu:
        if partition not in GPU_PARTITIONS:
            raise ValueError(f"GPU job requires a valid partition, got {partition!r}.")
        if gpus < 1:
            raise ValueError("GPU job requires at least one GPU.")
        lines.extend(
            [
                f"#SBATCH --partition={partition}",
                f"#SBATCH --gres=gpu:{gpus}",
            ]
        )
    elif partition is not None or gpus != 0:
        raise ValueError("CPU job cannot request a GPU partition or GPU count.")
    if dependency:
        lines.append(f"#SBATCH --dependency={dependency}")
    lines.extend(
        [
            "",
            "set -euo pipefail",
            *_runtime_prefix(
                environment=environment,
                require_cuda=gpu,
                expected_gpu_count=gpus,
            ),
            f"exec {shlex.join(command)}",
            "",
        ]
    )
    text = "\n".join(lines)
    forbidden = ("#SBATCH --array", "%A", "%a", "SLURM_ARRAY_TASK_ID")
    found = [token for token in forbidden if token in text]
    if found:
        raise RuntimeError(f"Generated script contains forbidden array tokens: {found}")
    return text


class Submission:
    def __init__(self, stage: str) -> None:
        self.stage = stage
        self.stamp = _stamp()
        self.directory = JOB_ROOT / f"{self.stamp}_{_slug(stage)}"
        self.directory.mkdir(parents=True, exist_ok=False)
        self.jobs: list[dict[str, Any]] = []

    def submit(
        self,
        *,
        logical_id: str,
        command: list[str],
        gpu: bool,
        time_limit: str,
        dependencies: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        cpus: int = 16,
        memory: str = "192G",
        environment: str = GENERATION_ENV,
        partition: str | None = None,
        gpus: int = 0,
    ) -> str:
        dependency = _dependency(dependencies or [])
        script = self.directory / f"{_slug(logical_id)}.sbatch"
        script.write_text(
            _script_text(
                job_name=f"t2is-{logical_id}",
                command=command,
                gpu=gpu,
                environment=environment,
                partition=partition,
                gpus=gpus,
                time_limit=time_limit,
                dependency=dependency,
                cpus=cpus,
                memory=memory,
            ),
            encoding="utf-8",
        )
        script.chmod(0o750)
        result = subprocess.run(
            ["sbatch", "--parsable", str(script)],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        job_id = result.stdout.strip().split(";", 1)[0]
        if not job_id.isdigit():
            raise RuntimeError(f"Unexpected sbatch response: {result.stdout!r}")
        record = {
            "logical_id": logical_id,
            "job_id": job_id,
            "script": str(script),
            "script_sha256": file_sha256(script),
            "command": command,
            "gpu": gpu,
            "partition": partition,
            "gpus": gpus,
            "time_limit": time_limit,
            "dependency": dependency,
            "environment": environment,
            "slurm_array": False,
            "array_throttle": False,
            **(metadata or {}),
        }
        self.jobs.append(record)
        print(json.dumps(record, sort_keys=True), flush=True)
        return job_id

    def seal(self) -> Path:
        RECEIPT_ROOT.mkdir(parents=True, exist_ok=True)
        target = RECEIPT_ROOT / f"{self.stamp}_{_slug(self.stage)}.json"
        payload = {
            "schema_version": 1,
            "stage": self.stage,
            "submitted_at": _utc(),
            "project_root": str(PROJECT_ROOT),
            "environments": sorted({job["environment"] for job in self.jobs}),
            "slurm_arrays": False,
            "array_throttles": False,
            "jobs": self.jobs,
        }
        atomic_json(target, payload)
        print(json.dumps({"receipt": str(target), "jobs": len(self.jobs)}, sort_keys=True))
        return target


def _module(name: str, *arguments: str) -> list[str]:
    return ["python", "-m", f"t2i_safety_benchmark.{name}", *arguments]


def _model_gpu_kwargs(model_id: str) -> dict[str, Any]:
    try:
        resource = MODEL_GPU_RESOURCES[model_id]
    except KeyError as exc:
        raise KeyError(f"No GPU resource contract for model {model_id!r}.") from exc
    return {
        "gpu": True,
        "environment": (
            COSMOS_ENV
            if model_id == "cosmos3_super_text2image"
            else GENERATION_ENV
        ),
        **resource,
    }


def submit_scheduler_grids(selected_models: set[str] | None = None) -> Path:
    submission = Submission("scheduler_grids_v2")
    for model_id in MODELS:
        if selected_models is not None and model_id not in selected_models:
            continue
        submission.submit(
            logical_id=f"scheduler-{model_id}",
            command=_module("scheduler_probe", "--model", model_id),
            time_limit="04:00:00",
            metadata={
                "stage_contract": "t2i_safety_scheduler_grid_v2",
                "model_id": model_id,
                "output": str(
                    CALIBRATION_ROOT
                    / "scheduler_grids_v2"
                    / model_id
                    / "scheduler.json"
                ),
            },
            **_model_gpu_kwargs(model_id),
        )
    return submission.seal()


def _imageguard_gpu_kwargs(
    resource: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "gpu": True,
        "environment": IMAGEGUARD_ENV,
        **(resource or IMAGEGUARD_GPU_RESOURCES),
    }


def _parse_time_limit(value: str) -> int:
    pieces = value.split(":")
    if len(pieces) != 3 or any(not piece.isdigit() for piece in pieces):
        raise ValueError(f"Invalid SLURM time limit: {value!r}")
    hours, minutes, seconds = (int(piece) for piece in pieces)
    if minutes >= 60 or seconds >= 60:
        raise ValueError(f"Invalid SLURM time limit: {value!r}")
    total = hours * 3600 + minutes * 60 + seconds
    if total <= 0 or total > 24 * 3600:
        raise ValueError(f"SLURM time limit must be in (0, 24h]: {value!r}")
    return total


def _format_time_limit(seconds: float) -> str:
    rounded = max(3600, int(math.ceil(seconds / 1800.0) * 1800))
    if rounded > 24 * 3600:
        raise ValueError(
            f"Estimated job duration {seconds:.1f}s exceeds the 24-hour cap."
        )
    hours, remainder = divmod(rounded, 3600)
    minutes, final_seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{final_seconds:02d}"


def _validate_resource_fields(
    value: dict[str, Any],
    *,
    context: str,
) -> dict[str, Any]:
    required = {"partition", "gpus", "cpus", "memory"}
    missing = required - set(value)
    if missing:
        raise ValueError(f"{context} is missing resource fields: {sorted(missing)}")
    if value["partition"] not in GPU_PARTITIONS:
        raise ValueError(f"{context} has invalid partition {value['partition']!r}.")
    for field in ("gpus", "cpus"):
        if type(value[field]) is not int or value[field] < 1:
            raise ValueError(f"{context}.{field} must be a positive integer.")
    memory = value["memory"]
    if (
        not isinstance(memory, str)
        or not memory.endswith("G")
        or not memory[:-1].isdigit()
        or int(memory[:-1]) < 1
    ):
        raise ValueError(f"{context}.memory must be a positive integer GiB string.")
    return {
        "partition": value["partition"],
        "gpus": value["gpus"],
        "cpus": value["cpus"],
        "memory": value["memory"],
    }


def _plan_key(model_id: str, variant: str) -> str:
    return f"{model_id}::{variant}"


def _runnable_pairs(contract: BenchmarkContract) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for model_id in MODELS:
        for variant in VARIANTS:
            if (
                variant == "native_negative_prompt"
                and not contract.model(model_id)["native_negative_supported"]
            ):
                continue
            pairs.append((model_id, variant))
    return pairs


def _validate_production_plan(
    gate: dict[str, Any],
    contract: BenchmarkContract,
) -> dict[str, dict[str, Any]]:
    plan = gate.get("production_plan")
    if not isinstance(plan, dict):
        raise RuntimeError("Pilot gate lacks a production_plan object.")
    expected = {
        _plan_key(model_id, variant)
        for model_id, variant in _runnable_pairs(contract)
    }
    if set(plan) != expected:
        raise RuntimeError(
            "Production-plan pair mismatch: "
            f"missing={sorted(expected - set(plan))}, "
            f"extra={sorted(set(plan) - expected)}"
        )
    validated: dict[str, dict[str, Any]] = {}
    for model_id, variant in _runnable_pairs(contract):
        key = _plan_key(model_id, variant)
        value = plan[key]
        if not isinstance(value, dict):
            raise TypeError(f"Production plan {key} must be an object.")
        resource = _validate_resource_fields(value, context=f"production_plan.{key}")
        required = {
            "num_shards",
            "time_limit",
            "pilot_job_id",
            "pilot_row_count",
            "measured_load_seconds",
            "measured_p95_cell_seconds",
            "safety_factor",
            "pilot_partition",
            "pilot_gpus",
        }
        missing = required - set(value)
        if missing:
            raise ValueError(f"Production plan {key} is missing {sorted(missing)}.")
        if type(value["num_shards"]) is not int or value["num_shards"] < 1:
            raise ValueError(f"Production plan {key} has invalid num_shards.")
        total_cells = len(contract.cells(model_id=model_id, variant=variant))
        if value["num_shards"] > total_cells:
            raise ValueError(f"Production plan {key} has more shards than cells.")
        if not str(value["pilot_job_id"]).isdigit():
            raise ValueError(f"Production plan {key} lacks a numeric pilot job ID.")
        if type(value["pilot_row_count"]) is not int or value["pilot_row_count"] < 10:
            raise ValueError(f"Production plan {key} has insufficient pilot rows.")
        for field in (
            "measured_load_seconds",
            "measured_p95_cell_seconds",
            "safety_factor",
        ):
            if not isinstance(value[field], (int, float)) or value[field] <= 0:
                raise ValueError(f"Production plan {key}.{field} must be positive.")
        if value["safety_factor"] < 1.5:
            raise ValueError(f"Production plan {key} safety factor must be at least 1.5.")
        if (
            value["pilot_partition"] != resource["partition"]
            or value["pilot_gpus"] != resource["gpus"]
        ):
            raise ValueError(
                f"Production plan {key} requests hardware not proved by its pilot."
            )
        cells_per_shard = math.ceil(total_cells / value["num_shards"])
        projected_seconds = float(value["measured_load_seconds"]) + (
            cells_per_shard
            * float(value["measured_p95_cell_seconds"])
            * float(value["safety_factor"])
        )
        time_seconds = _parse_time_limit(str(value["time_limit"]))
        if time_seconds < projected_seconds:
            raise ValueError(
                f"Production plan {key} walltime {time_seconds}s is below "
                f"projected {projected_seconds:.1f}s."
            )
        validated[key] = {
            **resource,
            "num_shards": value["num_shards"],
            "time_limit": value["time_limit"],
            "pilot_job_id": str(value["pilot_job_id"]),
            "pilot_row_count": value["pilot_row_count"],
            "measured_load_seconds": float(value["measured_load_seconds"]),
            "measured_p95_cell_seconds": float(
                value["measured_p95_cell_seconds"]
            ),
            "safety_factor": float(value["safety_factor"]),
            "projected_seconds_per_shard": projected_seconds,
        }
    return validated


def _validate_imageguard_plan(gate: dict[str, Any]) -> dict[str, Any]:
    value = gate.get("imageguard_plan")
    if not isinstance(value, dict):
        raise RuntimeError("Pilot gate lacks an imageguard_plan object.")
    resource = _validate_resource_fields(value, context="imageguard_plan")
    required = {
        "preflight_job_id",
        "timing_job_id",
        "measured_load_seconds",
        "measured_p95_cell_seconds",
        "safety_factor",
    }
    missing = required - set(value)
    if missing:
        raise ValueError(f"ImageGuard plan is missing {sorted(missing)}.")
    for field in ("preflight_job_id", "timing_job_id"):
        if not str(value[field]).isdigit():
            raise ValueError(f"ImageGuard plan {field} must be a numeric job ID.")
    for field in (
        "measured_load_seconds",
        "measured_p95_cell_seconds",
        "safety_factor",
    ):
        if not isinstance(value[field], (int, float)) or value[field] <= 0:
            raise ValueError(f"ImageGuard plan {field} must be positive.")
    if value["safety_factor"] < 1.5:
        raise ValueError("ImageGuard safety factor must be at least 1.5.")
    return {
        **resource,
        "preflight_job_id": str(value["preflight_job_id"]),
        "timing_job_id": str(value["timing_job_id"]),
        "measured_load_seconds": float(value["measured_load_seconds"]),
        "measured_p95_cell_seconds": float(value["measured_p95_cell_seconds"]),
        "safety_factor": float(value["safety_factor"]),
    }


def _validate_model_contract() -> BenchmarkContract:
    contract = BenchmarkContract()
    if tuple(contract.models) != MODELS:
        raise RuntimeError(
            f"Model order changed: {tuple(contract.models)} != {MODELS}"
        )
    if contract.variants != VARIANTS:
        raise RuntimeError(f"Variant order changed: {contract.variants} != {VARIANTS}")
    return contract


def submit_extraction(download_job_id: str) -> Path:
    submission = Submission("corpus_extraction")
    submission.submit(
        logical_id="extract-corpus",
        command=["bash", str(WORK_ROOT / "extract_dataset.sh")],
        gpu=False,
        time_limit="06:00:00",
        dependencies=[download_job_id],
        cpus=8,
        memory="64G",
        metadata={"download_job_id": download_job_id},
    )
    return submission.seal()


def submit_midsteer_prompts() -> Path:
    _validate_model_contract()
    submission = Submission("midsteer_prompt_manifest")
    submission.submit(
        logical_id="midsteer-build-prompts",
        command=_module("midsteer_calibrate", "build-prompts"),
        gpu=False,
        time_limit="12:00:00",
        cpus=8,
        memory="96G",
        metadata={
            "kind": "midsteer_prompt_manifest",
            "repo_id": "laion/relaion2B-en-research",
            "revision": "cb2173cfd818b41c8370b287dabf93ae85231c42",
            "neutral_population": 50_000,
            "concept_population_per_side": 1_000,
        },
    )
    return submission.seal()


def submit_imageguard_preflight(gpu_profile: str = "h100_94gb") -> Path:
    _validate_model_contract()
    if gpu_profile not in IMAGEGUARD_GPU_PROFILES:
        raise ValueError(f"Unknown ImageGuard GPU profile: {gpu_profile!r}")
    resource = IMAGEGUARD_GPU_PROFILES[gpu_profile]
    submission = Submission("imageguard_preflight")
    submission.submit(
        logical_id="imageguard-preflight",
        command=_module("imageguard_eval", "preflight"),
        **_imageguard_gpu_kwargs(resource),
        time_limit="02:00:00",
        metadata={
            "kind": "imageguard_preflight",
            "imageguard_revision": "e40dad31ec43ea8b4c82b24527f1d39c441a2485",
            "base_revision": "c67bd06390dbe068a582c6561570725b1289a7c5",
            "image_size": 490,
            "do_sample": False,
            "gpu_profile": gpu_profile,
            "gpu_resource": resource,
        },
    )
    return submission.seal()


def submit_midsteer(shard_count: int) -> Path:
    _validate_model_contract()
    prompt_manifest = CALIBRATION_ROOT / "midsteer_prompt_manifest.json"
    if not prompt_manifest.is_file():
        raise FileNotFoundError(
            "Seal the exact MidSteer prompt manifest before submitting calibration."
        )
    submission = Submission("midsteer_calibration")
    for model_id in MODELS:
        shard_jobs: list[str] = []
        for shard_index in range(shard_count):
            job_id = submission.submit(
                logical_id=f"mid-{model_id}-s{shard_index:03d}",
                command=_module(
                    "midsteer_calibrate",
                    "shard",
                    "--model",
                    model_id,
                    "--shard-index",
                    str(shard_index),
                    "--shard-count",
                    str(shard_count),
                ),
                **_model_gpu_kwargs(model_id),
                time_limit="24:00:00",
                metadata={
                    "kind": "midsteer_moment_shard",
                    "model_id": model_id,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "prompt_manifest_sha256": file_sha256(prompt_manifest),
                },
            )
            shard_jobs.append(job_id)
        submission.submit(
            logical_id=f"mid-{model_id}-pack",
            command=_module(
                "midsteer_calibrate",
                "pack",
                "--model",
                model_id,
                "--shard-count",
                str(shard_count),
            ),
            gpu=False,
            time_limit="24:00:00",
            dependencies=shard_jobs,
            cpus=32,
            memory="384G",
            metadata={
                "kind": "midsteer_pack",
                "model_id": model_id,
                "shard_count": shard_count,
            },
        )
    return submission.seal()


def submit_fairness_probes(shard_count: int) -> Path:
    raise RuntimeError(
        "The one-shot fairness workflow is prohibited because a profile cannot "
        "be admitted before direct visual review. Use fairness-generation, then "
        "complete visual admission, then fairness-evaluations, and finally "
        "fairness-profiles."
    )


def submit_fairness_probe_generation(
    shard_count: int,
    model_filter: set[str] | None = None,
) -> Path:
    _validate_model_contract()
    requested_models = set(model_filter or MODELS)
    unknown_models = requested_models - set(MODELS)
    if unknown_models:
        raise ValueError(
            f"Unknown fairness-generation models: {sorted(unknown_models)}"
        )
    probe_manifest = WORK_ROOT / "manifests" / "fairness_probe.jsonl"
    if not probe_manifest.is_file():
        raise FileNotFoundError("Seal the fairness probe manifest before submission.")
    submission = Submission("fairness_probe_generation")
    for model_id in MODELS:
        if model_id not in requested_models:
            continue
        for shard_index in range(shard_count):
            submission.submit(
                logical_id=f"fairgen-{model_id}-s{shard_index:02d}",
                command=_module(
                    "fairness_probe",
                    "run",
                    "--model",
                    model_id,
                    "--shard-index",
                    str(shard_index),
                    "--num-shards",
                    str(shard_count),
                ),
                **_model_gpu_kwargs(model_id),
                time_limit="12:00:00",
                metadata={
                    "kind": "fairness_probe_generation",
                    "model_id": model_id,
                    "shard_index": shard_index,
                    "shard_count": shard_count,
                    "probe_manifest_sha256": file_sha256(probe_manifest),
                },
            )
    return submission.seal()


def submit_fairness_probe_evaluations(
    generation_receipt: Path,
    preflight_job_id: str,
    model_filter: set[str] | None = None,
) -> Path:
    receipt = json.loads(generation_receipt.read_text(encoding="utf-8"))
    if receipt.get("stage") != "fairness_probe_generation":
        raise RuntimeError(
            "Fairness evaluation requires a fairness-generation receipt."
        )
    requested_models = set(model_filter or MODELS)
    unknown_models = requested_models - set(MODELS)
    if unknown_models:
        raise ValueError(f"Unknown fairness-evaluation models: {sorted(unknown_models)}")
    selected_jobs = [
        job for job in receipt["jobs"] if job["model_id"] in requested_models
    ]
    if not selected_jobs:
        raise RuntimeError("Fairness evaluation selected no generation jobs.")
    selected_models = sorted({job["model_id"] for job in selected_jobs})
    for model_id in selected_models:
        model_jobs = [job for job in selected_jobs if job["model_id"] == model_id]
        shard_counts = {int(job["shard_count"]) for job in model_jobs}
        if len(shard_counts) != 1:
            raise RuntimeError(
                f"Fairness generation shard count changed within {model_id}."
            )
        shard_count = shard_counts.pop()
        shard_indices = {int(job["shard_index"]) for job in model_jobs}
        if shard_indices != set(range(shard_count)):
            raise RuntimeError(
                f"Fairness evaluation requires all {shard_count} generation shards "
                f"for {model_id}; observed {sorted(shard_indices)}."
            )
    upstream_job_ids = [preflight_job_id] + [
        str(job["job_id"]) for job in selected_jobs
    ]
    terminal_proofs: dict[str, dict[str, str]] = {}
    for job_id in upstream_job_ids:
        if not job_id.isdigit():
            raise ValueError(f"Invalid Slurm job ID in fairness receipt: {job_id!r}")
        result = subprocess.run(
            [
                "sacct",
                "-X",
                "-j",
                job_id,
                "-n",
                "-P",
                "-o",
                "JobIDRaw,JobName,State,ExitCode,End",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        rows = [
            line.split("|")
            for line in result.stdout.splitlines()
            if line.strip()
        ]
        exact = [row for row in rows if row[0] == job_id]
        if len(exact) != 1 or len(exact[0]) != 5:
            raise RuntimeError(
                f"Could not obtain one exact terminal proof for Slurm job {job_id}."
            )
        _, job_name, state, exit_code, ended_at = exact[0]
        normalized_state = state.split()[0].rstrip("+")
        if normalized_state != "COMPLETED" or exit_code != "0:0":
            raise RuntimeError(
                f"Fairness evaluation upstream {job_id} is not a successful terminal job: "
                f"state={state!r}, exit_code={exit_code!r}."
            )
        terminal_proofs[job_id] = {
            "job_id": job_id,
            "job_name": job_name,
            "state": normalized_state,
            "exit_code": exit_code,
            "ended_at": ended_at,
        }
    submission = Submission("fairness_probe_imageguard")
    eval_by_model: dict[str, list[str]] = {
        model_id: [] for model_id in selected_models
    }
    for job in selected_jobs:
        eval_id = submission.submit(
            logical_id=f"faireval-{job['model_id']}-s{job['shard_index']:02d}",
            command=_module(
                "imageguard_eval",
                "score-fairness-probe",
                "--model",
                job["model_id"],
                "--shard-index",
                str(job["shard_index"]),
                "--num-shards",
                str(job["shard_count"]),
            ),
            **_imageguard_gpu_kwargs(IMAGEGUARD_GPU_PROFILES["48gb"]),
            time_limit="12:00:00",
            metadata={
                "kind": "fairness_probe_imageguard",
                "model_id": job["model_id"],
                "shard_index": job["shard_index"],
                "shard_count": job["shard_count"],
                "generation_job_id": job["job_id"],
                "preflight_job_id": preflight_job_id,
                "generation_terminal_proof": terminal_proofs[str(job["job_id"])],
                "preflight_terminal_proof": terminal_proofs[preflight_job_id],
            },
        )
        eval_by_model[job["model_id"]].append(eval_id)
    return submission.seal()


def submit_fairness_profiles(
    evaluation_receipt: Path,
    model_filter: set[str] | None = None,
) -> Path:
    receipt = json.loads(evaluation_receipt.read_text(encoding="utf-8"))
    if receipt.get("stage") != "fairness_probe_imageguard":
        raise RuntimeError(
            "Fairness profiles require a fairness-probe ImageGuard receipt."
        )
    requested_models = set(model_filter or MODELS)
    unknown_models = requested_models - set(MODELS)
    if unknown_models:
        raise ValueError(f"Unknown fairness-profile models: {sorted(unknown_models)}")
    selected_jobs = [
        job
        for job in receipt["jobs"]
        if job.get("kind") == "fairness_probe_imageguard"
        and job.get("model_id") in requested_models
    ]
    if not selected_jobs:
        raise RuntimeError("Fairness profile selected no ImageGuard jobs.")
    selected_models = sorted({str(job["model_id"]) for job in selected_jobs})
    terminal_proofs: dict[str, dict[str, str]] = {}
    for model_id in selected_models:
        model_jobs = [job for job in selected_jobs if job["model_id"] == model_id]
        shard_counts = {int(job["shard_count"]) for job in model_jobs}
        if len(shard_counts) != 1:
            raise RuntimeError(
                f"Fairness evaluation shard count changed within {model_id}."
            )
        shard_count = shard_counts.pop()
        shard_indices = {int(job["shard_index"]) for job in model_jobs}
        if shard_indices != set(range(shard_count)):
            raise RuntimeError(
                f"Fairness profile requires all {shard_count} ImageGuard shards "
                f"for {model_id}; observed {sorted(shard_indices)}."
            )
        visual_admission_path = (
            WORK_ROOT
            / "visual_audits"
            / "fairness_full_v2"
            / model_id
            / "VISUAL_ADMISSION.json"
        )
        if not visual_admission_path.is_file():
            raise FileNotFoundError(
                f"Direct visual admission is required before profiling: "
                f"{visual_admission_path}"
            )
        visual_admission = json.loads(
            visual_admission_path.read_text(encoding="utf-8")
        )
        if (
            visual_admission.get("status") != "accepted"
            or visual_admission.get("model_id") != model_id
            or visual_admission.get("record_count") != 515
        ):
            raise RuntimeError(
                f"Direct fairness visual admission is malformed for {model_id}."
            )
        for job in model_jobs:
            job_id = str(job["job_id"])
            if not job_id.isdigit():
                raise ValueError(f"Invalid fairness ImageGuard job ID: {job_id!r}")
            result = subprocess.run(
                [
                    "sacct",
                    "-X",
                    "-j",
                    job_id,
                    "-n",
                    "-P",
                    "-o",
                    "JobIDRaw,JobName,State,ExitCode,End",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            rows = [
                line.split("|")
                for line in result.stdout.splitlines()
                if line.strip()
            ]
            exact = [row for row in rows if row[0] == job_id]
            if len(exact) != 1 or len(exact[0]) != 5:
                raise RuntimeError(
                    f"Could not obtain terminal proof for ImageGuard job {job_id}."
                )
            _, job_name, state, exit_code, ended_at = exact[0]
            normalized_state = state.split()[0].rstrip("+")
            if normalized_state != "COMPLETED" or exit_code != "0:0":
                raise RuntimeError(
                    f"Fairness ImageGuard job {job_id} is not successful: "
                    f"state={state!r}, exit_code={exit_code!r}."
                )
            terminal_proofs[job_id] = {
                "job_id": job_id,
                "job_name": job_name,
                "state": normalized_state,
                "exit_code": exit_code,
                "ended_at": ended_at,
            }

    submission = Submission("fairness_profiles")
    for model_id in selected_models:
        model_jobs = [job for job in selected_jobs if job["model_id"] == model_id]
        visual_admission_path = (
            WORK_ROOT
            / "visual_audits"
            / "fairness_full_v2"
            / model_id
            / "VISUAL_ADMISSION.json"
        )
        submission.submit(
            logical_id=f"fairprofile-{model_id}",
            command=_module(
                "imageguard_eval",
                "build-fairness-profile",
                "--model",
                model_id,
            ),
            gpu=False,
            environment=IMAGEGUARD_ENV,
            time_limit="01:00:00",
            cpus=4,
            memory="16G",
            metadata={
                "kind": "fairness_profile",
                "model_id": model_id,
                "evaluation_receipt": str(evaluation_receipt),
                "evaluation_receipt_sha256": file_sha256(evaluation_receipt),
                "evaluation_terminal_proofs": {
                    str(job["job_id"]): terminal_proofs[str(job["job_id"])]
                    for job in model_jobs
                },
                "visual_admission": str(visual_admission_path),
                "visual_admission_sha256": file_sha256(visual_admission_path),
            },
        )
    return submission.seal()


def submit_nonfair_references(
    extraction_job_id: str,
    selected_models: set[str] | None = None,
) -> Path:
    _validate_model_contract()
    unknown_models = sorted((selected_models or set()) - set(MODELS))
    if unknown_models:
        raise ValueError(f"Unknown non-fair reference models: {unknown_models}")
    extraction_dependencies = _unmet_slurm_dependency(extraction_job_id)
    submission = Submission("nonfair_reference_banks")
    reference_manifest_root = WORK_ROOT / "manifests" / "unsafe_references"
    manifest_summary_path = reference_manifest_root / "summary.json"
    manifest_summary_sha256: str | None = None
    if manifest_summary_path.is_file():
        summary = json.loads(manifest_summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict) or set(summary) != set(NONFAIR_CATEGORIES):
            raise RuntimeError(
                "Sealed non-fair reference summary does not contain exactly the "
                "required benchmark categories."
            )
        expected_counts = {
            category: 115 if category == "hate" else 515
            for category in NONFAIR_CATEGORIES
        }
        for category, expected_count in expected_counts.items():
            entry = summary.get(category)
            if not isinstance(entry, dict):
                raise RuntimeError(f"Invalid sealed manifest summary entry: {category}")
            expected_path = (reference_manifest_root / f"{category}.jsonl").resolve()
            observed_path = Path(str(entry.get("manifest", ""))).resolve()
            if observed_path != expected_path or not expected_path.is_file():
                raise RuntimeError(
                    f"Sealed reference manifest path mismatch for {category}: "
                    f"{observed_path} != {expected_path}"
                )
            if entry.get("manifest_sha256") != file_sha256(expected_path):
                raise RuntimeError(
                    f"Sealed reference manifest hash mismatch for {category}."
                )
            line_count = sum(
                1
                for line in expected_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
            if (
                line_count != expected_count
                or entry.get("selected_count") != expected_count
                or entry.get("effective_reference_population") != expected_count
                or float(entry.get("empirical_mass_multiplier", 0.0)) != 1.0
            ):
                raise RuntimeError(
                    f"Sealed reference population contract mismatch for {category}."
                )
        manifest_summary_sha256 = file_sha256(manifest_summary_path)
        manifest_dependencies: list[str] = []
        manifest_state = "verified_sealed_existing_manifests"
    else:
        existing_manifests = [
            reference_manifest_root / f"{category}.jsonl"
            for category in NONFAIR_CATEGORIES
            if (reference_manifest_root / f"{category}.jsonl").exists()
        ]
        if existing_manifests:
            raise RuntimeError(
                "Incomplete non-fair reference manifest state exists without a sealed "
                f"summary: {existing_manifests}"
            )
        manifest_job = submission.submit(
            logical_id="reference-manifests",
            command=_module("references", "build-category-manifests"),
            gpu=False,
            time_limit="06:00:00",
            dependencies=extraction_dependencies,
            cpus=8,
            memory="64G",
            metadata={"kind": "unsafe_reference_manifests"},
        )
        manifest_dependencies = [manifest_job]
        manifest_state = "new_manifest_job"
    for model_id in MODELS:
        if selected_models is not None and model_id not in selected_models:
            continue
        for category in NONFAIR_CATEGORIES:
            submission.submit(
                logical_id=f"ref-{model_id}-{category}",
                command=_module(
                    "references",
                    "encode",
                    "--model",
                    model_id,
                    "--category",
                    category,
                ),
                **_model_gpu_kwargs(model_id),
                time_limit="24:00:00",
                dependencies=manifest_dependencies,
                metadata={
                    "kind": "unsafe_reference_bank",
                    "model_id": model_id,
                    "category": category,
                    "manifest_state": manifest_state,
                    "manifest_summary": str(manifest_summary_path),
                    "manifest_summary_sha256": manifest_summary_sha256,
                    "effective_reference_population_policy": (
                        "actual unique human-annotated train images, capped at 515"
                    ),
                    "empirical_mass_multiplier": 1.0,
                    "selection": "all unique human-annotated train images up to 515",
                },
            )
    return submission.seal()


def submit_fairness_references() -> Path:
    _validate_model_contract()
    submission = Submission("fairness_reference_banks")
    for model_id in MODELS:
        profile = CALIBRATION_ROOT / "fairness_profiles" / f"{model_id}.json"
        if not profile.is_file():
            raise FileNotFoundError(f"Missing admitted fairness profile {profile}")
        manifest_id = submission.submit(
            logical_id=f"fairref-manifest-{model_id}",
            command=_module(
                "references",
                "build-fairness-manifest",
                "--model",
                model_id,
                "--profile",
                str(profile),
            ),
            gpu=False,
            time_limit="02:00:00",
            cpus=4,
            memory="32G",
            metadata={
                "kind": "fairness_reference_manifest",
                "model_id": model_id,
                "profile_sha256": file_sha256(profile),
            },
        )
        submission.submit(
            logical_id=f"fairref-encode-{model_id}",
            command=_module(
                "references",
                "encode",
                "--model",
                model_id,
                "--category",
                "fairness",
            ),
            **_model_gpu_kwargs(model_id),
            time_limit="24:00:00",
            dependencies=[manifest_id],
            metadata={
                "kind": "unsafe_reference_bank",
                "model_id": model_id,
                "category": "fairness",
                "effective_reference_population": 515,
                "selection": "unique_human_annotated_train_images_up_to_515",
            },
        )
    return submission.seal()


def _pilot_rows(contract: BenchmarkContract) -> list[str]:
    rows = contract.prompt_rows().values()
    first_by_category: dict[str, Any] = {}
    for row in sorted(rows, key=lambda value: value.release_index):
        first_by_category.setdefault(row.category, row)
    if set(first_by_category) != set(contract.categories):
        raise RuntimeError("Pilot selection does not cover every benchmark category.")
    return [
        first_by_category[category].row_id
        for category in contract.categories
    ]


def _admission(
    path: Path,
    *,
    required_identity: dict[str, Any],
) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Admission is not a JSON object: {path}")
    if value.get("status") != "accepted":
        raise RuntimeError(f"Artifact was not admitted: {path}")
    mismatches = {
        key: (value.get(key), expected)
        for key, expected in required_identity.items()
        if value.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Admission identity mismatch in {path}: {mismatches}")
    artifact_path_value = value.get("artifact_path")
    artifact_sha = value.get("artifact_sha256")
    if artifact_path_value is not None or artifact_sha is not None:
        artifact_path = Path(str(artifact_path_value)).resolve()
        if not artifact_path.is_file():
            raise FileNotFoundError(artifact_path)
        if file_sha256(artifact_path) != artifact_sha:
            raise RuntimeError(f"Admitted artifact hash changed: {artifact_path}")
    return {
        "path": str(path),
        "sha256": file_sha256(path),
        "value": value,
    }


def _paper_reproduction_ready(variant: str) -> dict[str, Any]:
    path = CALIBRATION_ROOT / "paper_reproductions_v2" / f"{variant}.json"
    evidence = _admission(
        path,
        required_identity={
            "protocol": "t2i_safety_paper_reproduction_gate_v2",
            "variant": variant,
        },
    )
    value = evidence["value"]
    if value.get("author_result_status") != "reproduced_within_predeclared_tolerance":
        raise RuntimeError(f"Author paper result was not reproduced for {variant}")
    if value.get("local_equivalence_status") != "passed":
        raise RuntimeError(f"Local implementation equivalence failed for {variant}")
    revision = str(value.get("author_repository_revision", ""))
    if len(revision) != 40 or any(c not in "0123456789abcdef" for c in revision):
        raise RuntimeError(f"Unpinned author repository revision for {variant}")
    return evidence


def _concept_hierarchy_ready(model_id: str) -> dict[str, Any]:
    from .hierarchy_store import validate_hierarchy_admission

    return validate_hierarchy_admission(model_id)


def _related_artifacts_ready(model_id: str, variant: str) -> dict[str, Any]:
    contract = _validate_model_contract()
    evidence: dict[str, Any] = {
        "paper_reproduction": _paper_reproduction_ready(variant),
        "categories": {},
    }
    if variant == "midsteer":
        for category in contract.categories:
            root = CALIBRATION_ROOT / "midsteer" / model_id / category
            admission = _admission(
                root / "ADMISSION.json",
                required_identity={
                    "protocol": "t2i_safety_midsteer_artifact_admission_v3",
                    "model_id": model_id,
                    "category": category,
                    "variant": variant,
                },
            )
            if admission["value"].get("intermediate_clipping") is not False:
                raise RuntimeError(
                    f"MidSteer clipping was not disabled for {model_id}/{category}"
                )
            calibration = _admission(
                CALIBRATION_ROOT
                / "method_parameters_v3"
                / model_id
                / category
                / "midsteer.json",
                required_identity={
                    "protocol": "t2i_safety_method_calibration_v3",
                    "model_id": model_id,
                    "category": category,
                    "method": variant,
                },
            )
            if (
                calibration["value"].get("midsteer_artifact_sha256")
                != admission["value"].get("artifact_sha256")
            ):
                raise RuntimeError(
                    f"MidSteer calibration/artifact binding mismatch for "
                    f"{model_id}/{category}"
                )
            evidence["categories"][category] = {
                "artifact": admission,
                "calibration": calibration,
            }
    elif variant in {"sgf", "safe_denoiser"}:
        for category in contract.categories:
            root = (
                CALIBRATION_ROOT
                / "unsafe_references_v3"
                / model_id
                / category
            )
            bank = _admission(
                root / "ADMISSION.json",
                required_identity={
                    "protocol": "t2i_safety_reference_bank_admission_v3",
                    "model_id": model_id,
                    "category": category,
                },
            )
            bank_value = bank["value"]
            if (
                int(bank_value.get("reference_count", -1))
                != int(bank_value.get("effective_reference_population", -2))
                or float(bank_value.get("empirical_mass_multiplier", -1.0)) != 1.0
            ):
                raise RuntimeError(
                    f"Reference-bank empirical mass is invalid for "
                    f"{model_id}/{category}"
                )
            calibration = _admission(
                CALIBRATION_ROOT
                / "method_parameters_v3"
                / model_id
                / category
                / f"{variant}.json",
                required_identity={
                    "protocol": "t2i_safety_method_calibration_v3",
                    "model_id": model_id,
                    "category": category,
                    "method": variant,
                },
            )
            if (
                calibration["value"].get("reference_bank_sha256")
                != bank_value.get("artifact_sha256")
            ):
                raise RuntimeError(
                    f"Calibration/reference binding mismatch for "
                    f"{model_id}/{category}/{variant}"
                )
            evidence["categories"][category] = {
                "reference_bank": bank,
                "calibration": calibration,
            }
    else:
        raise ValueError(f"Unknown related-work variant {variant}")
    return evidence


def _runner_command(
    model_id: str,
    variant: str,
    *,
    shard_index: int | None = None,
    num_shards: int | None = None,
    row_ids: list[str] | None = None,
) -> list[str]:
    command = _module(
        "runner",
        "--model",
        model_id,
        "--variant",
        variant,
    )
    if shard_index is not None and num_shards is not None:
        command.extend(
            [
                "--shard-index",
                str(shard_index),
                "--num-shards",
                str(num_shards),
            ]
        )
    for row_id in row_ids or []:
        command.extend(["--row-id", row_id])
    return command


def submit_pilots(selected_variants: list[str] | None = None) -> Path:
    contract = _validate_model_contract()
    row_ids = _pilot_rows(contract)
    variants = tuple(selected_variants or VARIANTS)
    unknown = sorted(set(variants) - set(VARIANTS))
    if unknown or len(set(variants)) != len(variants):
        raise ValueError(
            f"Pilot variant selection is invalid: unknown={unknown}, values={variants}"
        )
    submission = Submission(
        "generation_pilots_" + "_".join(_slug(value, 20) for value in variants)
    )
    for model_id in MODELS:
        for variant in variants:
            if (
                variant == "native_negative_prompt"
                and not contract.model(model_id)["native_negative_supported"]
            ):
                continue
            artifact_evidence: dict[str, Any] = {}
            if variant == "conceptsteer":
                artifact_evidence["concept_hierarchy"] = (
                    _concept_hierarchy_ready(model_id)
                )
            elif variant in RELATED_VARIANTS:
                artifact_evidence["related_work"] = _related_artifacts_ready(
                    model_id,
                    variant,
                )
            submission.submit(
                logical_id=f"pilot-{model_id}-{variant}",
                command=_runner_command(
                    model_id,
                    variant,
                    row_ids=row_ids,
                ),
                **_model_gpu_kwargs(model_id),
                time_limit="08:00:00",
                metadata={
                    "kind": "generation_pilot",
                    "model_id": model_id,
                    "variant": variant,
                    "row_ids": row_ids,
                    "row_count": len(row_ids),
                    "artifact_evidence": artifact_evidence,
                },
            )
    return submission.seal()


def submit_pilot_evaluations(generation_receipt: Path) -> Path:
    receipt = json.loads(generation_receipt.read_text(encoding="utf-8"))
    if not str(receipt.get("stage", "")).startswith("generation_pilots_"):
        raise RuntimeError("Pilot evaluation requires a generation-pilot receipt.")
    submission = Submission("pilot_imageguard")
    for job in receipt["jobs"]:
        command = _module(
            "imageguard_eval",
            "score-benchmark",
            "--model",
            job["model_id"],
            "--variant",
            job["variant"],
        )
        for row_id in job["row_ids"]:
            command.extend(["--row-id", row_id])
        submission.submit(
            logical_id=f"eval-{job['logical_id']}",
            command=command,
            **_imageguard_gpu_kwargs(),
            time_limit="08:00:00",
            dependencies=[job["job_id"]],
            metadata={
                "kind": "pilot_imageguard",
                "model_id": job["model_id"],
                "variant": job["variant"],
                "row_ids": job["row_ids"],
                "generation_job_id": job["job_id"],
            },
        )
    return submission.seal()


def _validate_pilot_gate() -> dict[str, Any]:
    contract = _validate_model_contract()
    gate = json.loads(PILOT_GATE.read_text(encoding="utf-8"))
    required = {
        "schema_version": 1,
        "status": "accepted",
        "models": list(MODELS),
        "variants": list(VARIANTS),
    }
    mismatches = {
        key: (gate.get(key), expected)
        for key, expected in required.items()
        if gate.get(key) != expected
    }
    if mismatches:
        raise RuntimeError(f"Pilot acceptance gate mismatch: {mismatches}")
    if not gate.get("direct_visual_review_sha256"):
        raise RuntimeError("Pilot gate lacks direct visual-review provenance.")
    hierarchy_admissions = {
        model_id: _concept_hierarchy_ready(model_id)
        for model_id in MODELS
    }
    if gate.get("hierarchy_admissions") != hierarchy_admissions:
        raise RuntimeError(
            "Pilot gate is not bound to the current hierarchy and fairness overlays."
        )
    _validate_production_plan(gate, contract)
    _validate_imageguard_plan(gate)
    return gate


def submit_production() -> Path:
    contract = _validate_model_contract()
    gate = _validate_pilot_gate()
    production_plan = _validate_production_plan(gate, contract)
    submission = Submission("production_generation")
    for model_id in MODELS:
        for variant in VARIANTS:
            if (
                variant == "native_negative_prompt"
                and not contract.model(model_id)["native_negative_supported"]
            ):
                continue
            artifact_evidence: dict[str, Any] = {}
            if variant == "conceptsteer":
                artifact_evidence["concept_hierarchy"] = (
                    _concept_hierarchy_ready(model_id)
                )
            elif variant in RELATED_VARIANTS:
                artifact_evidence["related_work"] = _related_artifacts_ready(
                    model_id,
                    variant,
                )
            plan = production_plan[_plan_key(model_id, variant)]
            num_shards = int(plan["num_shards"])
            for shard_index in range(num_shards):
                cells = contract.cells(
                    model_id=model_id,
                    variant=variant,
                    shard_index=shard_index,
                    num_shards=num_shards,
                )
                if not cells:
                    continue
                submission.submit(
                    logical_id=(
                        f"prod-{model_id}-{variant}-s{shard_index:03d}"
                    ),
                    command=_runner_command(
                        model_id,
                        variant,
                        shard_index=shard_index,
                        num_shards=num_shards,
                    ),
                    gpu=True,
                    environment=(
                        COSMOS_ENV
                        if model_id == "cosmos3_super_text2image"
                        else GENERATION_ENV
                    ),
                    partition=plan["partition"],
                    gpus=plan["gpus"],
                    cpus=plan["cpus"],
                    memory=plan["memory"],
                    time_limit=plan["time_limit"],
                    metadata={
                        "kind": "production_generation",
                        "model_id": model_id,
                        "variant": variant,
                        "shard_index": shard_index,
                        "num_shards": num_shards,
                        "cell_count": len(cells),
                        "pilot_gate_sha256": file_sha256(PILOT_GATE),
                        "pilot_gate": gate,
                        "resource_plan": plan,
                        "artifact_evidence": artifact_evidence,
                    },
                )
    return submission.seal()


def submit_production_evaluations(generation_receipt: Path) -> Path:
    gate = _validate_pilot_gate()
    imageguard_plan = _validate_imageguard_plan(gate)
    receipt = json.loads(generation_receipt.read_text(encoding="utf-8"))
    if receipt.get("stage") != "production_generation":
        raise RuntimeError("Production evaluation requires a production receipt.")
    submission = Submission("production_imageguard")
    for job in receipt["jobs"]:
        if job.get("pilot_gate_sha256") != file_sha256(PILOT_GATE):
            raise RuntimeError(
                f"Generation job {job.get('job_id')} used a different pilot gate."
            )
        estimated_seconds = imageguard_plan["measured_load_seconds"] + (
            int(job["cell_count"])
            * imageguard_plan["measured_p95_cell_seconds"]
            * imageguard_plan["safety_factor"]
        )
        time_limit = _format_time_limit(estimated_seconds)
        submission.submit(
            logical_id=f"eval-{job['logical_id']}",
            command=_module(
                "imageguard_eval",
                "score-benchmark",
                "--model",
                job["model_id"],
                "--variant",
                job["variant"],
                "--shard-index",
                str(job["shard_index"]),
                "--num-shards",
                str(job["num_shards"]),
            ),
            gpu=True,
            environment=IMAGEGUARD_ENV,
            partition=imageguard_plan["partition"],
            gpus=imageguard_plan["gpus"],
            cpus=imageguard_plan["cpus"],
            memory=imageguard_plan["memory"],
            time_limit=time_limit,
            dependencies=[
                job["job_id"],
                imageguard_plan["preflight_job_id"],
            ],
            metadata={
                "kind": "production_imageguard",
                "model_id": job["model_id"],
                "variant": job["variant"],
                "shard_index": job["shard_index"],
                "num_shards": job["num_shards"],
                "cell_count": job["cell_count"],
                "generation_job_id": job["job_id"],
                "imageguard_plan": imageguard_plan,
                "estimated_seconds": estimated_seconds,
            },
        )
    return submission.seal()


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Ordinary non-array T2ISafety SLURM launcher."
    )
    sub = value.add_subparsers(dest="command", required=True)
    extract = sub.add_parser("extraction")
    extract.add_argument("--download-job-id", required=True)
    sub.add_parser("midsteer-prompts")
    imageguard_preflight = sub.add_parser("imageguard-preflight")
    imageguard_preflight.add_argument(
        "--gpu-profile",
        choices=tuple(IMAGEGUARD_GPU_PROFILES),
        default="h100_94gb",
    )
    scheduler_grids = sub.add_parser("scheduler-grids")
    scheduler_grids.add_argument("--model", action="append", choices=MODELS)
    mid = sub.add_parser("midsteer")
    mid.add_argument("--shard-count", type=int, default=20)
    fair = sub.add_parser("fairness-probes")
    fair.add_argument("--shard-count", type=int, default=8)
    fair_generation = sub.add_parser("fairness-generation")
    fair_generation.add_argument("--shard-count", type=int, default=8)
    fair_generation.add_argument("--model", action="append", choices=MODELS)
    fair_evaluation = sub.add_parser("fairness-evaluations")
    fair_evaluation.add_argument("--generation-receipt", required=True, type=Path)
    fair_evaluation.add_argument("--preflight-job-id", required=True)
    fair_evaluation.add_argument("--model", action="append", choices=MODELS)
    fair_profiles = sub.add_parser("fairness-profiles")
    fair_profiles.add_argument("--evaluation-receipt", required=True, type=Path)
    fair_profiles.add_argument("--model", action="append", choices=MODELS)
    nonfair = sub.add_parser("nonfair-references")
    nonfair.add_argument("--extraction-job-id", required=True)
    nonfair.add_argument("--model", action="append", choices=MODELS)
    sub.add_parser("fairness-references")
    pilots = sub.add_parser("pilots")
    pilots.add_argument("--variant", action="append", choices=VARIANTS)
    pilot_eval = sub.add_parser("pilot-evaluations")
    pilot_eval.add_argument("--generation-receipt", required=True, type=Path)
    sub.add_parser("production")
    production_eval = sub.add_parser("production-evaluations")
    production_eval.add_argument("--generation-receipt", required=True, type=Path)
    return value


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "extraction":
        target = submit_extraction(args.download_job_id)
    elif args.command == "midsteer-prompts":
        target = submit_midsteer_prompts()
    elif args.command == "imageguard-preflight":
        target = submit_imageguard_preflight(args.gpu_profile)
    elif args.command == "scheduler-grids":
        target = submit_scheduler_grids(set(args.model) if args.model else None)
    elif args.command == "midsteer":
        target = submit_midsteer(args.shard_count)
    elif args.command == "fairness-probes":
        target = submit_fairness_probes(args.shard_count)
    elif args.command == "fairness-generation":
        target = submit_fairness_probe_generation(
            args.shard_count,
            set(args.model) if args.model else None,
        )
    elif args.command == "fairness-evaluations":
        target = submit_fairness_probe_evaluations(
            args.generation_receipt,
            args.preflight_job_id,
            set(args.model) if args.model else None,
        )
    elif args.command == "fairness-profiles":
        target = submit_fairness_profiles(
            args.evaluation_receipt,
            set(args.model) if args.model else None,
        )
    elif args.command == "nonfair-references":
        target = submit_nonfair_references(
            args.extraction_job_id,
            set(args.model) if args.model else None,
        )
    elif args.command == "fairness-references":
        target = submit_fairness_references()
    elif args.command == "pilots":
        target = submit_pilots(args.variant)
    elif args.command == "pilot-evaluations":
        target = submit_pilot_evaluations(args.generation_receipt)
    elif args.command == "production":
        target = submit_production()
    elif args.command == "production-evaluations":
        target = submit_production_evaluations(args.generation_receipt)
    else:
        raise AssertionError(args.command)
    print(json.dumps({"sealed_receipt": str(target)}, sort_keys=True))


if __name__ == "__main__":
    main()
