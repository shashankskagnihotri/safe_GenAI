#!/usr/bin/env python3
"""Run one immutable V8 cell through the archived legacy bottleneck controller."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import tempfile
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from PIL import Image

from hierasafe_flow.generation.runner import GenerationRunner


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_row(path: Path, expected_sha256: str, index: int) -> dict[str, Any]:
    require(sha256_file(path) == expected_sha256, "manifest file SHA-256 mismatch")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    require(all(row["job_index"] == position for position, row in enumerate(rows)), "non-contiguous manifest job_index")
    require(0 <= index < len(rows), "job index out of range")
    return rows[index]


def build_runner_config(row: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    model = row["model"]
    generation = row["generation"]
    controller = row["legacy_controller"]
    arm = row["arm"]
    return {
        "project": {"name": row["campaign_id"], "seed": row["seed"]},
        "runtime": {
            "device": "cuda",
            "dtype": model["dtype"],
            "allow_tf32": True,
            "compile": False,
        },
        "generation": {
            **generation,
            "prompt_file": None,
            "prompt": row["original_prompt"],
        },
        "model": {
            "adapter": model["adapter"],
            "model_id": model["model_id"],
            "revision": model["revision"],
            "variant": None,
            "torch_dtype": model["dtype"],
            "local_files_only": True,
            "load_kwargs": {},
            "diffusers_pipeline_class": model["pipeline_class"],
            "verified_model_index": True,
            "components": [
                "scheduler",
                "text_encoder",
                "text_encoder_2",
                "tokenizer",
                "tokenizer_2",
                "transformer",
                "vae",
            ],
            "guidance_scale": generation["guidance_scale"],
        },
        "concepts": {"hierarchy_path": row["hierarchy_path"]},
        "steering": {
            "enabled": bool(arm["enabled"]),
            "start_step": controller["start_step"],
            "end_step": controller["end_step"],
            "lambda_schedule": {
                "kind": "constant",
                "max_value": float(arm["strength"]),
                "min_value": float(arm["strength"]),
            },
            "margin": controller["margin"],
            "feature_dim": controller["feature_dim"],
            "mask": controller["mask"],
            "calibration": controller["calibration"],
            "mode": controller["mode"] if arm["enabled"] else "none",
            "start_fraction": controller["start_fraction"],
            "end_fraction": controller["end_fraction"],
            "active_pair_ids": arm["active_pair_ids"],
            "normalize_directions": controller["normalize_directions"],
            "prompt_composition": controller["prompt_composition"],
            "step_stride": controller["step_stride"],
            "pair_overrides": {},
        },
        "logging": {"level": "INFO", "tensorboard": True, "log_every_steps": 1, "output_dir": str(output_dir)},
        "output": {"save_latents": False, "save_traces": True, "image_format": "png", "video_format": "mp4", "decode": True},
        "native_negative_prompt": {},
        "benchmark": {
            "name": "push_for_iclr_legacy_bottleneck_v8",
            "stage": "development_exact_reproduction",
            "variant": arm["id"],
            "prompt_id": row["prompt_id"],
            "seed": row["seed"],
            "strength": arm["strength"],
            "margin": controller["margin"],
            "schedule": "full_window_constant",
            "schedule_window": [controller["start_fraction"], controller["end_fraction"]],
            "local_mask": controller["mask"]["enabled"],
            "normalize_directions": controller["normalize_directions"],
            "prompt_composition": controller["prompt_composition"],
            "step_stride": controller["step_stride"],
            "active_pair_ids": arm["active_pair_ids"],
        },
    }


def validate_trace(path: Path, row: dict[str, Any]) -> dict[str, Any]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    expected_steps = int(row["generation"]["num_inference_steps"])
    require(isinstance(trace, list) and len(trace) == expected_steps, "trace step count mismatch")
    active_ids = list(row["arm"]["active_pair_ids"])
    if not row["arm"]["enabled"]:
        require(all(step["enabled"] is False for step in trace), "baseline trace unexpectedly enabled")
        return {"status": "PASS_BASELINE", "step_count": len(trace), "active_pair_ids": []}
    nonzero = {pair_id: 0 for pair_id in active_ids}
    for step_index, step in enumerate(trace):
        require(step["step_index"] == step_index, "trace step index mismatch")
        require(step["enabled"] is True, "legacy full-window step is disabled")
        observed = [record["concept_id"] for record in step["concepts"]]
        require(observed == active_ids, "legacy pair order/coverage mismatch")
        for record in step["concepts"]:
            stats = record.get("steering_delta_stats") or {}
            values = [float(stats[name]) for name in ("mean", "std", "min", "max")]
            require(all(math.isfinite(value) for value in values), "non-finite steering delta")
            if max(abs(value) for value in values) > 0.0:
                nonzero[record["concept_id"]] += 1
    require(all(count > 0 for count in nonzero.values()), "one or more legacy pairs were inert")
    return {
        "status": "PASS_ACTIVE",
        "step_count": len(trace),
        "active_pair_ids": active_ids,
        "nonzero_delta_step_count": nonzero,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--index", type=int, required=True)
    parser.add_argument("--expected-code-commit", required=True)
    args = parser.parse_args()

    repository_root = Path(__file__).resolve().parents[2]
    row = load_row(args.manifest.resolve(), args.manifest_file_sha256, args.index)
    actual_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repository_root, text=True
    ).strip()
    require(actual_commit == args.expected_code_commit == row["code_commit"], "code commit mismatch")
    require(sha256_file(repository_root / row["hierarchy_path"]) == row["hierarchy_sha256"], "hierarchy hash mismatch")
    for relative, expected in row["runtime_dependency_sha256"].items():
        require(sha256_file(repository_root / relative) == expected, f"runtime dependency drift: {relative}")

    output_dir = Path(row["expected_output_dir"])
    allowed_root = Path("/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/BY_MODEL").resolve()
    require(allowed_root in output_dir.resolve().parents, "output path escaped campaign root")
    require(not output_dir.exists(), f"refusing to overwrite output: {output_dir}")
    config = build_runner_config(row, output_dir)
    started = datetime.now(timezone.utc).isoformat()
    try:
        result = GenerationRunner(config).run(prompt=row["original_prompt"])
        image_path = output_dir / "sample_0000" / "image_000.png"
        report_path = output_dir / "sample_0000" / "report.json"
        trace_path = output_dir / "sample_0000" / "steering_trace.json"
        require(image_path.is_file(), "missing final PNG")
        require(report_path.is_file(), "missing sample report")
        require(trace_path.is_file(), "missing steering trace")
        with Image.open(image_path) as image:
            image.verify()
        with Image.open(image_path) as image:
            require(image.size == (row["generation"]["width"], row["generation"]["height"]), "wrong image dimensions")
        trace_validation = validate_trace(trace_path, row)
        metadata = {
            "schema_version": "push-for-iclr.legacy-bottleneck-cell-metadata.v1",
            "status": "SUCCESS",
            "started_at_utc": started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "job": row,
            "manifest_path": str(args.manifest.resolve()),
            "manifest_file_sha256": args.manifest_file_sha256,
            "resolved_controller_contract": {
                "space": "model_native_vector_field",
                "schedule": "constant_full_window_every_step",
                "prompt_composition": "concept_only",
                "mask_enabled": False,
                "calibration_enabled": False,
                "normalize_directions": False,
                "sequential_pair_updates": True,
            },
            "trace_validation": trace_validation,
            "artifacts": {
                "image": str(image_path),
                "image_sha256": sha256_file(image_path),
                "report": str(report_path),
                "report_sha256": sha256_file(report_path),
                "trace": str(trace_path),
                "trace_sha256": sha256_file(trace_path),
            },
            "runner_output_dir": result.output_dir,
            "runner_records": [asdict(record) for record in result.records],
            "slurm": {
                "array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
                "array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
                "job_id": os.environ.get("SLURM_JOB_ID"),
                "node": os.environ.get("SLURMD_NODENAME"),
            },
        }
        atomic_json(output_dir / "metadata.json", metadata)
        (output_dir / "_SUCCESS").write_text("success\n", encoding="utf-8")
        print(json.dumps({"status": "SUCCESS", "output_dir": str(output_dir)}, sort_keys=True))
        return 0
    except Exception as exc:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_json(
            output_dir / "_FAILURE.json",
            {
                "schema_version": "push-for-iclr.legacy-bottleneck-failure.v1",
                "status": "FAILED",
                "job": row,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
                "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            },
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
