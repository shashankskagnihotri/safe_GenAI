#!/usr/bin/env python3
import argparse
import ast
import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ATTACK_KEYS = ("sneaky_prompt", "mma_diffusion", "ring_a_bell", "p4d")
NUDENET_KEYS = ATTACK_KEYS + ("i2p_sexual",)


def now():
    return datetime.now(timezone.utc).isoformat()


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%s" % os.getpid())
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def atomic_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%s" % os.getpid())
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(str(temporary), str(path))


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_sha256(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError("SHA-256 mismatch for %s: %s != %s" % (path, actual, expected))
    return actual


def load_csv_rows(asset):
    path = Path(asset["path"] if "path" in asset else asset["csv"])
    expected_hash = asset["sha256"]
    expected_rows = int(asset["rows"])
    assert_sha256(path, expected_hash)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise RuntimeError("CSV row mismatch for %s: %d != %d" % (path, len(rows), expected_rows))
    if not rows or "prompt" not in rows[0] or any(not row["prompt"].strip() for row in rows):
        raise RuntimeError("CSV has a missing prompt: %s" % path)
    return rows


def suite_map(config):
    return {suite["key"]: suite for suite in config["suites"]}


def assert_runtime(config):
    expected = Path(config["environment"]["python"]).resolve().parent.parent
    if Path(sys.prefix).resolve() != expected:
        raise RuntimeError("Wrong runtime: %s != %s" % (sys.prefix, expected))


def assert_protocol(config_path, config):
    assert_runtime(config)
    source = Path(config["upstream"]["repo"])
    actual_commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_commit != config["upstream"]["commit"]:
        raise RuntimeError("DES upstream commit mismatch: %s" % actual_commit)
    subprocess.run(["git", "-C", str(source), "diff", "--quiet", "HEAD", "--"], check=True)
    for relative, expected in config["upstream"]["files"].items():
        assert_sha256(source / relative, expected)
    model = Path(config["model"]["snapshot"])
    if model.name != config["model"]["revision"]:
        raise RuntimeError("Model revision/path mismatch")
    assert_sha256(model / "model_index.json", config["model"]["model_index_sha256"])
    assert_sha256(model / "scheduler/scheduler_config.json", config["model"]["scheduler_config_sha256"])
    assert_sha256(model / "text_encoder/config.json", config["model"]["text_encoder_config_sha256"])
    load_csv_rows(config["training"]["safe_csv"])
    load_csv_rows(config["training"]["unsafe_csv"])
    for suite in config["suites"]:
        load_csv_rows(suite)
    attempt = Path(config["attempt_root"])
    attempt.mkdir(parents=True, exist_ok=True)
    protocol = attempt / "PROTOCOL_CONFIG.json"
    source_bytes = Path(config_path).read_bytes()
    if protocol.exists() and protocol.read_bytes() != source_bytes:
        raise RuntimeError("Frozen protocol differs from submitted config")
    if not protocol.exists():
        temporary = protocol.with_name(protocol.name + ".tmp.%s" % os.getpid())
        temporary.write_bytes(source_bytes)
        os.replace(str(temporary), str(protocol))
    return attempt


def author_environment(config):
    environment = os.environ.copy()
    environment.update({
        "PYTHONPATH": config["upstream"]["repo"],
        "PYTHONHASHSEED": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
    })
    return environment


def run_author(command, config, cwd):
    print("+", " ".join(str(item) for item in command), flush=True)
    subprocess.run(
        [str(item) for item in command],
        cwd=str(cwd),
        env=author_environment(config),
        check=True,
    )


def preflight(config_path, config, attempt):
    blocked = {
        "metric": "FID",
        "status": config["evaluation"]["fid_status"],
        "reason": config["evaluation"]["fid_block_reason"],
        "missing_import": "T2IBenchmark",
        "missing_reference": "datasets/coco_10k",
        "fallback_used": False,
    }
    atomic_json(attempt / "QUALITY/FID_BLOCKED.json", blocked)
    result = {
        "status": "preflight_pass",
        "method": "DES",
        "upstream_commit": config["upstream"]["commit"],
        "model_revision": config["model"]["revision"],
        "protocol_sha256": sha256_file(config_path),
        "training_rows": 6911,
        "generation_rows": sum(int(item["rows"]) for item in config["suites"]),
        "denominator_repair": config["generation"]["paper_denominator_repair"],
        "fid": blocked,
        "clip_asset_status": config["evaluation"]["clip_asset_status"],
        "completed_at": now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_json(attempt / "PROVENANCE/PREFLIGHT.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


def build_codebook(config, attempt):
    output = attempt / "CODEBOOK"
    if output.exists():
        raise RuntimeError("Refusing to overwrite codebook: %s" % output)
    source = Path(config["upstream"]["repo"])
    command = [
        config["environment"]["python"], source / "save_codebook.py",
        "--model_path", config["model"]["snapshot"],
        "--device", "cuda:0",
        "--save_dir", output,
        "--csv_path", config["training"]["safe_csv"]["path"],
    ]
    run_author(command, config, source)
    files = {}
    for name in ("clip_embeddings.pt", "clip_embeddings.faiss"):
        path = output / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Missing codebook artifact: %s" % path)
        files[name] = {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
    atomic_json(output / "STATUS.json", {
        "status": "author_codebook_complete", "rows": 6911, "files": files,
        "author_command": [str(item) for item in command], "completed_at": now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    })


def train(config, attempt):
    if not (attempt / "CODEBOOK/STATUS.json").is_file():
        raise RuntimeError("Codebook admission is missing")
    output = attempt / "TRAINING"
    if output.exists():
        raise RuntimeError("Refusing to overwrite training output: %s" % output)
    source = Path(config["upstream"]["repo"])
    paired = output / "paired_data.pt"
    training = config["training"]
    command = [
        config["environment"]["python"], source / "train_des.py",
        "--model_path", config["model"]["snapshot"],
        "--device", "cuda:0",
        "--codebook_dir", attempt / "CODEBOOK",
        "--unsafe_csv_path", training["unsafe_csv"]["path"],
        "--safe_csv_path", training["safe_csv"]["path"],
        "--output_dir", output / "checkpoints",
        "--num_epochs", training["epochs"],
        "--learning_rate", training["learning_rate"],
        "--batch_size", training["batch_size"],
        "--save_every", training["save_every"],
        "--sampling_ratio", training["sampling_ratio"],
        "--lambda_safe", training["lambda_safe"],
        "--concept_prompt", training["concept_prompt"],
        "--concept_guidance_scale", training["concept_guidance_scale"],
        "--safe_embedding_path", paired,
        "--ablation",
    ] + training["losses"]
    run_author(command, config, source)
    checkpoints = sorted((output / "checkpoints").glob("*/checkpoint-2.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError("Expected one epoch-2 checkpoint, found %d" % len(checkpoints))
    checkpoint = checkpoints[0]
    atomic_json(output / "STATUS.json", {
        "status": "author_training_complete",
        "checkpoint": str(checkpoint),
        "checkpoint_bytes": checkpoint.stat().st_size,
        "checkpoint_sha256": sha256_file(checkpoint),
        "paired_data": str(paired),
        "paired_data_sha256": sha256_file(paired),
        "author_command": [str(item) for item in command],
        "hyperparameters": training,
        "completed_at": now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    })


def checkpoint_path(attempt):
    status = load_json(attempt / "TRAINING/STATUS.json")
    path = Path(status["checkpoint"])
    assert_sha256(path, status["checkpoint_sha256"])
    return path


def generate(config, attempt, task_id):
    suites = config["suites"]
    if task_id < 0 or task_id >= len(suites):
        raise ValueError("Generation task outside suite range")
    suite = suites[task_id]
    rows = load_csv_rows(suite)
    final = attempt / "GENERATIONS" / suite["key"]
    workspace = attempt / "WORK" / suite["key"]
    if final.exists() or workspace.exists():
        raise RuntimeError("Refusing to overwrite DES suite %s" % suite["key"])
    workspace.mkdir(parents=True)
    source = Path(config["upstream"]["repo"])
    generation = config["generation"]
    checkpoint = checkpoint_path(attempt)
    command = [
        config["environment"]["python"], source / "generate.py",
        "--model_path", config["model"]["snapshot"],
        "--device", "cuda:0",
        "--seed", generation["seed"],
        "--prompts_csv", suite["csv"],
        "--start_idx", generation["start_idx"],
        "--end_idx", suite["rows"],
        "--height", generation["height"],
        "--width", generation["width"],
        "--output_path", "images",
        "--num_inference_steps", generation["num_inference_steps"],
        "--guidance_scale", generation["guidance_scale"],
        "--training_method", "des",
        "--text_encoder_path", checkpoint,
        "--model_type", "sd_v1",
    ]
    run_author(command, config, workspace)
    generated = workspace / "results/des" / Path(config["model"]["snapshot"]).name / "images"
    candidates = list(generated.glob("images_des_*p.png"))
    indexed = {}
    for path in candidates:
        match = re.fullmatch(r"images_des_(\d+)p\.png", path.name)
        if not match:
            raise RuntimeError("Unexpected DES filename: %s" % path.name)
        indexed[int(match.group(1))] = path
    expected = set(range(len(rows)))
    if set(indexed) != expected:
        raise RuntimeError("Generated indices differ from full paper denominator for %s" % suite["key"])
    final.mkdir(parents=True)
    generated.rename(final / "images")
    records = []
    for index, row in enumerate(rows):
        image_path = final / "images" / indexed[index].name
        with Image.open(image_path) as image:
            image.load()
            if image.size != (int(generation["width"]), int(generation["height"])):
                raise RuntimeError("Wrong image dimensions: %s" % image_path)
            mode = image.mode
        records.append({
            "method": "DES", "suite": suite["key"], "prompt_index": index,
            "prompt": row["prompt"], "image_path": str(image_path),
            "image_sha256": sha256_file(image_path), "image_mode": mode,
            "width": int(generation["width"]), "height": int(generation["height"]),
            "seed_protocol": "seed_everything(42) once, then author sequential global RNG",
            "scheduler": generation["scheduler"], "guidance_scale": generation["guidance_scale"],
            "num_inference_steps": generation["num_inference_steps"],
        })
    atomic_jsonl(final / "manifest.jsonl", records)
    atomic_json(final / "STATUS.json", {
        "status": "author_generation_complete", "suite": suite["key"],
        "images": len(records), "full_paper_denominator": True,
        "paper_denominator_repair": generation["paper_denominator_repair"],
        "author_command": [str(item) for item in command], "completed_at": now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    })
    shutil.rmtree(str(workspace))


def parse_author_nudenet(path):
    text = Path(path).read_text(encoding="utf-8")
    percent_match = re.search(r"Average Classification Acc \[nudity\]=\s*([0-9.]+)%", text)
    total_match = re.search(r"Total images examined:\s*(\d+)", text)
    detected_match = re.search(r"Nudity detected in\s*(\d+) images", text)
    names_match = re.search(r"Detected image names:\s*(\[.*\])", text, flags=re.DOTALL)
    if not all((percent_match, total_match, detected_match, names_match)):
        raise RuntimeError("Could not parse author NudeNet output")
    categories = {}
    for key, count in re.findall(r"^- ([A-Z_]+):\s*(\d+) images", text, flags=re.MULTILINE):
        categories[key] = int(count)
    names = ast.literal_eval(names_match.group(1))
    return {
        "attack_success_rate_percent": float(percent_match.group(1)),
        "total_images": int(total_match.group(1)),
        "unsafe_images": int(detected_match.group(1)),
        "category_counts": categories,
        "unsafe_filenames": names,
    }


def evaluate_nudenet(config, attempt, task_id):
    if task_id < 0 or task_id >= len(NUDENET_KEYS):
        raise ValueError("NudeNet task outside suite range")
    key = NUDENET_KEYS[task_id]
    suite = suite_map(config)[key]
    images = attempt / "GENERATIONS" / key / "images"
    status = load_json(attempt / "GENERATIONS" / key / "STATUS.json")
    if int(status["images"]) != int(suite["rows"]):
        raise RuntimeError("Generation denominator mismatch")
    output = attempt / "EVALUATIONS" / key
    if output.exists():
        raise RuntimeError("Refusing to overwrite NudeNet evaluation: %s" % output)
    source = Path(config["upstream"]["repo"])
    command = [
        config["environment"]["python"], source / "tasks/img_batch_classify.py",
        "--job", "nudity", "--cls_class", "nudity",
        "--folder_dir", images, "--devices", "0,0",
    ]
    run_author(command, config, source)
    author_output = images / "classify.txt"
    result = parse_author_nudenet(author_output)
    if result["total_images"] != int(suite["rows"]):
        raise RuntimeError("NudeNet denominator mismatch for %s" % key)
    result.update({
        "status": "author_nudenet_complete", "method": "DES", "suite": key,
        "threshold": config["evaluation"]["nudenet_threshold"],
        "attack_success_rate_fraction": result["unsafe_images"] / result["total_images"],
        "author_command": [str(item) for item in command], "completed_at": now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
    })
    output.mkdir(parents=True)
    shutil.copy2(str(author_output), str(output / "classify.txt"))
    atomic_json(output / "NudeNet.json", result)


def load_manifest(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def all_generation_records(config, attempt):
    records = []
    for suite in config["suites"]:
        manifest = attempt / "GENERATIONS" / suite["key"] / "manifest.jsonl"
        suite_records = load_manifest(manifest)
        if len(suite_records) != int(suite["rows"]):
            raise RuntimeError("Manifest denominator mismatch for %s" % suite["key"])
        records.extend(suite_records)
    return records


def render_contact(config, attempt, page_index):
    contact = config["contact_sheets"]
    if page_index < 0 or page_index >= int(contact["pages"]):
        raise ValueError("Contact page index outside configured range")
    records = all_generation_records(config, attempt)
    expected_pages = int(math.ceil(len(records) / float(contact["page_size"])))
    if expected_pages != int(contact["pages"]):
        raise RuntimeError("Configured contact page count is wrong")
    start = page_index * int(contact["page_size"])
    page = records[start:start + int(contact["page_size"])]
    flagged = {}
    for key in NUDENET_KEYS:
        result = load_json(attempt / "EVALUATIONS" / key / "NudeNet.json")
        flagged[key] = set(result["unsafe_filenames"])
    columns = int(contact["columns"])
    rows = int(math.ceil(len(page) / float(columns)))
    tile_w = int(contact["tile_width"])
    tile_h = int(contact["tile_height"])
    header_h = 36
    canvas = Image.new("RGB", (columns * tile_w, header_h + rows * tile_h), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    draw.text((8, 10), "DES SD1.5 paper reproduction | page %d/%d | global rows %d-%d" % (
        page_index + 1, expected_pages, start, start + len(page) - 1), fill="black", font=font)
    metadata = []
    for offset, record in enumerate(page):
        x = (offset % columns) * tile_w
        y = header_h + (offset // columns) * tile_h
        with Image.open(record["image_path"]) as image:
            image = image.convert("RGB")
            image.thumbnail((tile_w - 8, tile_w - 8))
            px = x + (tile_w - image.width) // 2
            canvas.paste(image, (px, y + 4))
        filename = Path(record["image_path"]).name
        unsafe = filename in flagged.get(record["suite"], set())
        color = "red" if unsafe else "black"
        draw.rectangle((x + 2, y + 2, x + tile_w - 3, y + tile_w - 3), outline=color, width=3 if unsafe else 1)
        label = "%s #%d%s" % (record["suite"], record["prompt_index"], " NUDE+" if unsafe else "")
        draw.text((x + 4, y + tile_w + 1), label[:34], fill=color, font=font)
        prompt_lines = textwrap.wrap(record["prompt"].replace("\n", " "), width=31)[:2]
        for line_index, line in enumerate(prompt_lines):
            draw.text((x + 4, y + tile_w + 13 + line_index * 11), line, fill="black", font=font)
        metadata.append({
            "global_index": start + offset, "suite": record["suite"],
            "prompt_index": record["prompt_index"], "prompt": record["prompt"],
            "image_path": record["image_path"], "nudenet_unsafe": unsafe,
        })
    output = attempt / "CONTACT_SHEETS"
    output.mkdir(parents=True, exist_ok=True)
    image_path = output / ("page_%04d.jpg" % page_index)
    json_path = output / ("page_%04d.json" % page_index)
    if image_path.exists() or json_path.exists():
        raise RuntimeError("Refusing to overwrite contact page %d" % page_index)
    canvas.save(image_path, quality=92, subsampling=0)
    atomic_json(json_path, {"page_index": page_index, "records": metadata})


def verify_contacts(config, attempt):
    missing = []
    for page in range(int(config["contact_sheets"]["pages"])):
        for suffix in ("jpg", "json"):
            path = attempt / "CONTACT_SHEETS" / ("page_%04d.%s" % (page, suffix))
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    result = {
        "status": "contact_sheets_complete" if not missing else "contact_sheets_incomplete",
        "expected_pages": int(config["contact_sheets"]["pages"]),
        "missing": missing, "visual_review_status": "pending_exhaustive_manual_review",
        "completed_at": now(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_json(attempt / "CONTACT_SHEETS/INTEGRITY.json", result)
    if missing:
        raise SystemExit(6)


def aggregate(config, attempt):
    comparisons = {}
    observed = []
    tolerance = float(config["evaluation"]["absolute_asr_tolerance_fraction"])
    suites = suite_map(config)
    for key in ATTACK_KEYS:
        result = load_json(attempt / "EVALUATIONS" / key / "NudeNet.json")
        value = float(result["attack_success_rate_fraction"])
        target = float(suites[key]["paper_asr_percent"]) / 100.0
        observed.append(value)
        comparisons[key] = {
            "observed_fraction": value, "paper_fraction": target,
            "absolute_error": abs(value - target), "tolerance": tolerance,
            "pass": abs(value - target) <= tolerance,
            "unsafe_images": int(result["unsafe_images"]),
            "denominator": int(result["total_images"]),
        }
    average = sum(observed) / len(observed)
    target_average = float(config["evaluation"]["table_2_average_asr_percent"]) / 100.0
    average_gate = {
        "observed_fraction": average, "paper_fraction": target_average,
        "absolute_error": abs(average - target_average), "tolerance": tolerance,
        "pass": abs(average - target_average) <= tolerance,
    }
    i2p = load_json(attempt / "EVALUATIONS/i2p_sexual/NudeNet.json")
    observed_detections = sum(int(value) for value in i2p["category_counts"].values())
    target_detections = int(config["evaluation"]["table_1_i2p_total_detections"])
    i2p_tolerance = int(config["evaluation"]["table_1_i2p_detection_tolerance"])
    i2p_gate = {
        "observed_total_category_detections": observed_detections,
        "paper_total_category_detections": target_detections,
        "absolute_error": abs(observed_detections - target_detections),
        "tolerance": i2p_tolerance,
        "pass": abs(observed_detections - target_detections) <= i2p_tolerance,
        "observed_category_counts": i2p["category_counts"],
        "paper_category_counts": config["evaluation"]["table_1_i2p_category_counts"],
        "unsafe_images": i2p["unsafe_images"], "denominator": i2p["total_images"],
    }
    primary_pass = average_gate["pass"] and all(item["pass"] for item in comparisons.values())
    result = {
        "status": "paper_reproduction_numeric_pass" if primary_pass else "paper_reproduction_numeric_fail",
        "method": "DES", "primary_contract": "Table 2 SD1.5 four-attack unweighted ASR",
        "table_2_suite_comparisons": comparisons, "table_2_average": average_gate,
        "table_1_i2p": i2p_gate, "numeric_pass": primary_pass,
        "fid": load_json(attempt / "QUALITY/FID_BLOCKED.json"),
        "clip": {"status": config["evaluation"]["clip_asset_status"], "paper": config["evaluation"]["paper_clip"]},
        "visual_integrity_status": "pending_exhaustive_manual_review",
        "completed_at": now(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_json(attempt / "AGGREGATE/NUMERIC_ADMISSION.json", result)
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    if not primary_pass:
        raise SystemExit(5)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("preflight")
    subparsers.add_parser("codebook")
    subparsers.add_parser("train")
    generate_parser = subparsers.add_parser("generate")
    generate_parser.add_argument("--task-id", type=int, required=True)
    evaluate_parser = subparsers.add_parser("evaluate-nudenet")
    evaluate_parser.add_argument("--task-id", type=int, required=True)
    contact_parser = subparsers.add_parser("render-contact")
    contact_parser.add_argument("--page-index", type=int, required=True)
    subparsers.add_parser("verify-contacts")
    subparsers.add_parser("aggregate")
    args = parser.parse_args()
    config = load_json(args.config)
    attempt = assert_protocol(args.config, config)
    if args.command == "preflight":
        preflight(args.config, config, attempt)
    elif args.command == "codebook":
        build_codebook(config, attempt)
    elif args.command == "train":
        train(config, attempt)
    elif args.command == "generate":
        generate(config, attempt, args.task_id)
    elif args.command == "evaluate-nudenet":
        evaluate_nudenet(config, attempt, args.task_id)
    elif args.command == "render-contact":
        render_contact(config, attempt, args.page_index)
    elif args.command == "verify-contacts":
        verify_contacts(config, attempt)
    else:
        aggregate(config, attempt)


if __name__ == "__main__":
    main()
