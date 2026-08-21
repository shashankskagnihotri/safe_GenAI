#!/usr/bin/env python3
import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from paper_i2p_common import (
    assert_git_commit,
    assert_sha256,
    atomic_json,
    atomic_jsonl,
    ensure_fresh_directory,
    ensure_runtime,
    load_json,
    load_jsonl,
    sha256_file,
)


def native_value(value):
    if isinstance(value, dict):
        return {str(key): native_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [native_value(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def evaluate_manifest(config, input_manifest, output_dir, mode, task_id):
    import onnxruntime
    import torch
    from PIL import Image

    if not torch.cuda.is_available():
        raise RuntimeError("Official Q16/NudeNet evaluation requires CUDA")
    providers = onnxruntime.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        raise RuntimeError("CUDAExecutionProvider is absent: %s" % providers)
    assets = Path(config["assets"]["root"])
    os.environ["HOME"] = str(assets / "home")
    os.environ["HF_HUB_CACHE"] = config["assets"]["clip_cache"]
    os.environ["TRANSFORMERS_CACHE"] = config["assets"]["clip_cache"]
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.chdir(assets)
    official_repo = config["official_evaluator"]["repo"]
    sys.path.insert(0, official_repo)
    from eval.nudenet_detector_gpu import Detector
    from eval.q16 import Q16

    output_dir = ensure_fresh_directory(output_dir)
    q16 = Q16()
    nudenet = Detector(model_name=config["evaluation"]["nudenet_model"])
    unsafe_labels = set(config["evaluation"]["unsafe_labels"])
    threshold = float(config["evaluation"]["nudenet_threshold"])
    source_records = load_jsonl(input_manifest)
    results = []
    for index, record in enumerate(source_records):
        image_path = Path(record["image_path"])
        if sha256_file(image_path) != record["image_sha256"]:
            raise RuntimeError("Image integrity mismatch: %s" % image_path)
        with Image.open(image_path) as loaded:
            image = loaded.convert("RGB")
            q16_unsafe = bool(q16.detect(image))
        detections = nudenet.detect(str(image_path), min_prob=threshold)
        if not isinstance(detections, list):
            raise RuntimeError("Official NudeNet detector returned a non-list result")
        labels = [detection["label"] for detection in detections]
        nudenet_unsafe = any(label in unsafe_labels for label in labels)
        result = dict(record)
        result.update({
            "q16_unsafe": q16_unsafe,
            "nudenet_unsafe": nudenet_unsafe,
            "unsafe": bool(q16_unsafe or nudenet_unsafe),
            "nudenet_labels": labels,
            "nudenet_detections": native_value(detections),
            "evaluator_mode": mode,
        })
        results.append(result)
        if (index + 1) % 25 == 0 or index + 1 == len(source_records):
            print("Evaluated %d/%d images" % (index + 1, len(source_records)), flush=True)
    atomic_jsonl(output_dir / "manifest.jsonl", results)
    atomic_json(output_dir / "STATUS.json", {
        "status": "evaluation_complete",
        "method": "sld_hypmax",
        "mode": mode,
        "task_id": task_id,
        "images": len(results),
        "onnxruntime_providers": providers,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("runtime-check")
    evaluation_parser = subparsers.add_parser("evaluate")
    evaluation_parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    ensure_runtime(config["environment"]["evaluation_python"])
    evaluator = config["official_evaluator"]
    assert_git_commit(evaluator["repo"], evaluator["commit"])
    assert_sha256(Path(evaluator["repo"]) / "eval/q16.py", evaluator["q16_sha256"])
    assert_sha256(Path(evaluator["repo"]) / "eval/nudenet_detector_gpu.py", evaluator["nudenet_sha256"])
    attempt_root = Path(config["attempt_root"])
    if args.command == "runtime-check":
        evaluate_manifest(
            config,
            attempt_root / "SMOKE/SLD_GENERATION/manifest.jsonl",
            attempt_root / "SMOKE/SLD_EVALUATION",
            "smoke",
            0,
        )
    else:
        task_id = args.task_id
        evaluate_manifest(
            config,
            attempt_root / ("GENERATION/shard_%03d/manifest.jsonl" % task_id),
            attempt_root / ("EVALUATION/shard_%03d" % task_id),
            "paper",
            task_id,
        )


if __name__ == "__main__":
    main()
