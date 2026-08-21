#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from paper_i2p_common import (
    assert_git_commit,
    assert_sha256,
    atomic_json,
    atomic_jsonl,
    ensure_runtime,
    load_dataset,
    load_json,
    sha256_file,
    shard_bounds,
    snapshot_config,
)


def run_author_source(config, rows, start, count, output_dir, mode, task_id):
    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise RuntimeError("Refusing to overwrite SAFREE output: %s" % output_dir)
    generation = config["generation"]
    source = Path(config["upstream"]["repo"])
    command = [
        config["environment"]["python"],
        str(source / "generate_safree.py"),
        "--data", config["dataset"]["admitted_path"],
        "--save-dir", str(output_dir),
        "--model_id", config["model"]["snapshot"],
        "--num-samples", str(generation["images_per_prompt"]),
        "--nudenet-path", config["assets"]["nudenet_classifier_admitted"],
        "--category", generation["category"],
        "--config", config["assets"]["admitted_sd_config"],
        "--device", "cuda:0",
        "--nudity_thr", str(generation["nudenet_threshold"]),
        "--valid_case_numbers", "%d,%d" % (start, count),
        "--erase-id", "std",
        "--sf_alpha", str(generation["sf_alpha"]),
        "--re_attn_t=" + generation["re_attn_t"],
        "--freeu_hyp=" + generation["freeu_hyp"],
        "--up_t", str(generation["up_t"]),
    ]
    if generation["safree"]:
        command.append("--safree")
    if generation["self_validation_filter"]:
        command.append("--self_validation_filter")
    if generation["latent_re_attention"]:
        command.append("--latent_re_attention")
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": str(source),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "PYTHONHASHSEED": "0",
    })
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    print("+", " ".join(command), flush=True)
    subprocess.run(command, cwd=source, env=environment, check=True)
    detect_path = output_dir / "detect_dict.json"
    with detect_path.open("r", encoding="utf-8") as handle:
        detection = json.load(handle)
    unsafe = detection.get("unsafe")
    if not isinstance(unsafe, list) or len(unsafe) != count:
        raise RuntimeError("SAFREE detector result count does not match shard")
    selected_rows = rows[start:start + count]
    records = []
    for row, is_unsafe in zip(selected_rows, unsafe):
        case_number = int(row["case_number"])
        candidates = sorted((output_dir / "all").glob("%d_*.png" % case_number))
        if len(candidates) != 1:
            raise RuntimeError("Expected one SAFREE image for case %d, found %d" % (case_number, len(candidates)))
        image_path = candidates[0]
        records.append({
            "method": "safree",
            "mode": mode,
            "task_id": task_id,
            "case_number": case_number,
            "dataset_row_index": int(row[""]),
            "prompt": row["prompt"],
            "categories": row["categories"],
            "seed": int(row["evaluation_seed"]),
            "guidance_scale": float(row["evaluation_guidance"]),
            "width": int(row["sd_image_width"]),
            "height": int(row["sd_image_height"]),
            "sample_index": 0,
            "unsafe": bool(is_unsafe),
            "image_path": str(image_path),
            "image_sha256": sha256_file(image_path),
        })
    atomic_jsonl(output_dir / "manifest.jsonl", records)
    atomic_json(output_dir / "STATUS.json", {
        "status": "generation_and_author_evaluation_complete",
        "method": "safree",
        "mode": mode,
        "task_id": task_id,
        "dataset_start": start,
        "rows": count,
        "images": len(records),
        "author_command": command,
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
    assert_sha256(Path(config["upstream"]["repo"]) / "generate_safree.py", config["upstream"]["generator_sha256"])
    assert_sha256(config["assets"]["admitted_sd_config"], config["upstream"]["config_sha256"])
    assert_sha256(config["assets"]["nudenet_classifier_admitted"], config["assets"]["nudenet_classifier_sha256"])
    rows = load_dataset(config)
    attempt_root = Path(config["attempt_root"])
    snapshot_config(args.config, attempt_root)
    if args.command == "runtime-check":
        run_author_source(config, rows, 0, 1, attempt_root / "SMOKE/SAFREE", "smoke", 0)
    else:
        start, count = shard_bounds(len(rows), int(config["generation"]["shards"]), args.task_id)
        run_author_source(
            config,
            rows,
            start,
            count,
            attempt_root / ("GENERATION/shard_%03d" % args.task_id),
            "paper",
            args.task_id,
        )


if __name__ == "__main__":
    main()
