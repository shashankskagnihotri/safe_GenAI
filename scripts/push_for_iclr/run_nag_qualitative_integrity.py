#!/usr/bin/env python3
"""Generate one arm of the pinned NAG qualitative implementation audit."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import tempfile
import time
import types
from datetime import datetime, timezone
from pathlib import Path

import torch
from PIL import Image

from nag_paper_l1_attention_flux import build_paper_l1_processor_class


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(path: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
    ).strip()


def install_flux_only_nag_namespace(upstream: Path) -> None:
    """Expose pinned FLUX submodules without executing unrelated eager exports."""
    package_dir = (upstream / "nag").resolve()
    if not (package_dir / "transformer_flux.py").is_file():
        raise RuntimeError(f"Pinned NAG FLUX package is incomplete: {package_dir}")
    package = types.ModuleType("nag")
    package.__file__ = str(package_dir / "__init__.py")
    package.__package__ = "nag"
    package.__path__ = [str(package_dir)]
    sys.modules["nag"] = package


def make_pipeline(config: dict, arm_id: str):
    model_dir = Path(config["model"]["local_dir"])
    admission = load_json(model_dir / "ADMISSION.json")
    if admission.get("status") != "ADMITTED_IMMUTABLE_DIFFUSERS_SNAPSHOT":
        raise RuntimeError("Flux asset snapshot is not admitted")
    dtype = torch.bfloat16
    transform_audit = None

    if arm_id == "baseline_diffusers":
        from diffusers import FluxPipeline

        pipe = FluxPipeline.from_pretrained(
            model_dir,
            torch_dtype=dtype,
            local_files_only=True,
        )
    else:
        install_flux_only_nag_namespace(Path(config["upstream"]["path"]))
        from nag.transformer_flux import NAGFluxTransformer2DModel
        import nag.pipeline_flux_nag as pipeline_module

        if arm_id == "paper_equation_L1":
            transformed_class, transform_audit = build_paper_l1_processor_class(
                pipeline_module
            )
            pipeline_module.NAGFluxAttnProcessor2_0 = transformed_class
        elif arm_id != "author_release_L2":
            raise ValueError(f"Unknown NAG arm: {arm_id}")

        transformer = NAGFluxTransformer2DModel.from_pretrained(
            model_dir,
            subfolder="transformer",
            torch_dtype=dtype,
            local_files_only=True,
        )
        pipe = pipeline_module.NAGFluxPipeline.from_pretrained(
            model_dir,
            transformer=transformer,
            torch_dtype=dtype,
            local_files_only=True,
        )
    pipe.to("cuda")
    pipe.set_progress_bar_config(disable=False)
    return pipe, transform_audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--arm-index", type=int, required=True)
    args = parser.parse_args()
    config_path = args.config.resolve()
    config = load_json(config_path)
    environment_root = Path(config["environment"]).resolve()
    environment_admission_path = environment_root / "ADMISSION.json"
    environment_admission = load_json(environment_admission_path)
    if environment_admission.get("status") != "ADMITTED":
        raise RuntimeError("NAG repaired environment overlay is not admitted")
    if environment_admission.get("upstream_commit") != config["upstream"]["commit"]:
        raise RuntimeError("NAG environment admission is bound to another upstream commit")
    configured_overlay = Path(config["pythonpath_overlay"]).resolve()
    admitted_overlay = Path(environment_admission["overlay"]).resolve()
    if configured_overlay != admitted_overlay:
        raise RuntimeError("NAG configured overlay does not match its admission record")
    resolved_sys_path = {
        Path(entry).resolve() for entry in sys.path if entry
    }
    if configured_overlay not in resolved_sys_path:
        raise RuntimeError("NAG admitted package overlay is absent from sys.path")
    expected_packages = environment_admission.get(
        "effective_packages", environment_admission["packages"]
    )
    actual_packages = {
        name: importlib.metadata.version(name) for name in expected_packages
    }
    if actual_packages != expected_packages:
        raise RuntimeError(
            f"NAG package mismatch: expected {expected_packages}, got {actual_packages}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; no CPU fallback is permitted")

    upstream = Path(config["upstream"]["path"])
    actual_upstream = git_head(upstream)
    if actual_upstream != config["upstream"]["commit"]:
        raise RuntimeError(
            f"NAG upstream mismatch: expected {config['upstream']['commit']}, "
            f"got {actual_upstream}"
        )
    sys.path.insert(0, str(upstream))

    arms = config["arms"]
    if args.arm_index < 0 or args.arm_index >= len(arms):
        raise ValueError(f"arm-index must be in [0, {len(arms) - 1}]")
    arm = arms[args.arm_index]
    arm_id = arm["id"]
    output_root = Path(config["output_root"])
    arm_root = output_root / arm_id
    arm_root.mkdir(parents=True, exist_ok=True)
    manifest_path = arm_root / "MANIFEST.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite an existing immutable attempt: {manifest_path}"
        )

    started_at = utc_now()
    pipeline, transform_audit = make_pipeline(config, arm_id)
    records = []
    generation = config["generation"]
    nag = config["nag"]
    for prompt_index, prompt in enumerate(config["prompts"]):
        prompt_dir = arm_root / f"prompt_{prompt_index:02d}_{prompt['id']}"
        prompt_dir.mkdir(parents=True, exist_ok=True)
        for seed in generation["seeds"]:
            output_path = prompt_dir / f"seed_{seed:06d}.png"
            if output_path.exists():
                raise FileExistsError(f"Refusing to overwrite {output_path}")
            generator = torch.Generator(device="cuda").manual_seed(seed)
            kwargs = {
                "prompt": prompt["prompt"],
                "height": generation["height"],
                "width": generation["width"],
                "num_inference_steps": generation["num_inference_steps"],
                "guidance_scale": generation["guidance_scale"],
                "max_sequence_length": generation["max_sequence_length"],
                "generator": generator,
            }
            if arm_id != "baseline_diffusers":
                kwargs.update(
                    {
                        "nag_scale": nag["runtime_nag_scale"],
                        "nag_tau": nag["tau"],
                        "nag_alpha": nag["alpha"],
                        "nag_end": nag["nag_end"],
                        "nag_negative_prompt": prompt["negative_prompt"],
                    }
                )
            started = time.monotonic()
            result = pipeline(**kwargs)
            elapsed = time.monotonic() - started
            image = result.images[0]
            if not isinstance(image, Image.Image):
                raise TypeError(f"Pipeline returned {type(image)}, expected PIL.Image")
            if image.size != (generation["width"], generation["height"]):
                raise RuntimeError(f"Unexpected image dimensions {image.size}")
            image.save(output_path, format="PNG")
            records.append(
                {
                    "prompt_index": prompt_index,
                    "prompt_id": prompt["id"],
                    "prompt": prompt["prompt"],
                    "negative_prompt": prompt["negative_prompt"],
                    "seed": seed,
                    "path": str(output_path),
                    "sha256": sha256_file(output_path),
                    "width": image.width,
                    "height": image.height,
                    "generation_seconds": elapsed,
                }
            )
            print(json.dumps(records[-1], sort_keys=True), flush=True)

    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "experiment_id": config["experiment_id"],
            "claim_scope": config["claim_scope"],
            "arm": arm,
            "started_at_utc": started_at,
            "completed_at_utc": utc_now(),
            "config_path": str(config_path),
            "config_sha256": sha256_file(config_path),
            "campaign_code_commit": git_head(config_path.parents[2]),
            "upstream_commit": actual_upstream,
            "environment": {
                "root": str(environment_root),
                "admission_path": str(environment_admission_path),
                "admission_sha256": sha256_file(environment_admission_path),
                "base_python": config["base_python"],
                "pythonpath_overlay": str(configured_overlay),
                "nag_import_policy": config["nag_import_policy"],
                "packages": actual_packages,
            },
            "model_revision": config["model"]["revision"],
            "generation": generation,
            "nag": nag if arm_id != "baseline_diffusers" else None,
            "l1_transform_audit": transform_audit,
            "records": records,
            "runtime": {
                "host": platform.node(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "packages": actual_packages,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            },
        },
    )


if __name__ == "__main__":
    main()
