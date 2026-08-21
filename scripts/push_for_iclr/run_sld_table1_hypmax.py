#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from paper_i2p_common import (
    assert_git_commit,
    assert_sha256,
    atomic_json,
    atomic_jsonl,
    ensure_fresh_directory,
    ensure_runtime,
    load_dataset,
    load_json,
    sha256_file,
    shard_bounds,
    snapshot_config,
)


def load_pipeline(config):
    import torch
    from diffusers import LMSDiscreteScheduler
    from sld import SLDPipeline

    if not torch.cuda.is_available():
        raise RuntimeError("SLD paper generation requires CUDA")
    scheduler_config = config["generation"]["scheduler"]
    scheduler = LMSDiscreteScheduler(
        beta_start=float(scheduler_config["beta_start"]),
        beta_end=float(scheduler_config["beta_end"]),
        beta_schedule=scheduler_config["beta_schedule"],
        num_train_timesteps=int(scheduler_config["num_train_timesteps"]),
    )
    pipeline = SLDPipeline.from_pretrained(
        config["model"]["snapshot"],
        scheduler=scheduler,
        torch_dtype=torch.float32,
        local_files_only=True,
    )
    pipeline = pipeline.to("cuda")
    pipeline.set_progress_bar_config(disable=True)
    return pipeline


def generate_rows(config, rows, output_dir, mode, task_id):
    import torch

    output_dir = ensure_fresh_directory(output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir()
    pipeline = load_pipeline(config)
    generation = config["generation"]
    records = []
    for local_row_index, row in enumerate(rows):
        case_number = int(row["case_number"])
        seed = int(row[generation["seed_field"]])
        guidance = float(row[generation["guidance_field"]])
        width = int(row[generation["width_field"]])
        height = int(row[generation["height_field"]])
        generator = torch.Generator(device="cuda").manual_seed(seed)
        sample_index = 0
        for call_index in range(int(generation["sequential_calls_per_prompt"])):
            result = pipeline(
                prompt=row["prompt"],
                height=height,
                width=width,
                num_inference_steps=int(generation["steps"]),
                guidance_scale=guidance,
                num_images_per_prompt=int(generation["images_per_call"]),
                generator=generator,
                sld_concept=generation["safety_concept"],
                sld_guidance_scale=float(generation["sld_guidance_scale"]),
                sld_warmup_steps=int(generation["sld_warmup_steps"]),
                sld_threshold=float(generation["sld_threshold"]),
                sld_momentum_scale=float(generation["sld_momentum_scale"]),
                sld_mom_beta=float(generation["sld_mom_beta"]),
            )
            images = result.images
            if len(images) != int(generation["images_per_call"]):
                raise RuntimeError("SLD returned an unexpected image count")
            flags = result.nsfw_content_detected
            if flags is None:
                flags = [None] * len(images)
            if len(flags) != len(images):
                raise RuntimeError("Safety-checker flag count mismatch")
            for within_call, (image, nsfw_flag) in enumerate(zip(images, flags)):
                filename = "case_%05d_sample_%02d_seed_%010d.png" % (
                    case_number, sample_index, seed
                )
                image_path = image_dir / filename
                image.save(image_path, format="PNG")
                records.append({
                    "method": "sld_hypmax",
                    "mode": mode,
                    "task_id": task_id,
                    "case_number": case_number,
                    "dataset_row_index": int(row[""]),
                    "prompt": row["prompt"],
                    "categories": row["categories"],
                    "seed": seed,
                    "guidance_scale": guidance,
                    "width": width,
                    "height": height,
                    "call_index": call_index,
                    "within_call_index": within_call,
                    "sample_index": sample_index,
                    "standard_safety_checker_flag": None if nsfw_flag is None else bool(nsfw_flag),
                    "image_path": str(image_path),
                    "image_sha256": sha256_file(image_path),
                })
                sample_index += 1
        if sample_index != int(generation["images_per_prompt"]):
            raise RuntimeError("Per-prompt image count mismatch")
        print(
            "SLD completed case %d (%d/%d)" % (case_number, local_row_index + 1, len(rows)),
            flush=True,
        )
    expected = len(rows) * int(generation["images_per_prompt"])
    if len(records) != expected:
        raise RuntimeError("Shard image count %d != %d" % (len(records), expected))
    atomic_jsonl(output_dir / "manifest.jsonl", records)
    atomic_json(output_dir / "STATUS.json", {
        "status": "generation_complete",
        "method": "sld_hypmax",
        "mode": mode,
        "task_id": task_id,
        "rows": len(rows),
        "images": len(records),
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
    generation_parser = subparsers.add_parser("generate")
    generation_parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    ensure_runtime(config["environment"]["python"])
    assert_git_commit(config["upstream"]["repo"], config["upstream"]["commit"])
    assert_sha256(
        Path(config["upstream"]["repo"]) / "src/sld/sld_pipeline.py",
        config["upstream"]["pipeline_sha256"],
    )
    if not (Path(config["assets"]["root"]) / "ADMISSION.json").is_file():
        raise RuntimeError("SLD/SAFREE assets are not admitted")
    rows = load_dataset(config)
    attempt_root = Path(config["attempt_root"])
    snapshot_config(args.config, attempt_root)
    if args.command == "runtime-check":
        generate_rows(config, rows[:1], attempt_root / "SMOKE/SLD_GENERATION", "smoke", 0)
    else:
        start, count = shard_bounds(len(rows), int(config["generation"]["shards"]), args.task_id)
        generate_rows(
            config,
            rows[start:start + count],
            attempt_root / ("GENERATION/shard_%03d" % args.task_id),
            "paper",
            args.task_id,
        )


if __name__ == "__main__":
    main()
