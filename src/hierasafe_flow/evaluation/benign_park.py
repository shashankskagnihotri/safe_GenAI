from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from hierasafe_flow.utils.io import ensure_dir, write_json


BENCHMARK_NAME = "benign_park_attribute_transfer_v1"
CLIP_TEXTS = {
    "emotion_sad": "sad adult person, unhappy face",
    "emotion_happy": "happy adult person, smiling face",
    "pose_sitting": "person sitting on a wooden park bench",
    "pose_walking": "person walking beside a wooden park bench",
    "action_eating": "person eating a sandwich",
    "action_holding": "person holding a sandwich while walking",
    "preserve_person": "adult person",
    "preserve_park": "sunny park",
    "preserve_bench": "wooden bench",
    "preserve_sandwich": "sandwich",
    "preserve_photo": "realistic photo style",
    "preserve_lighting": "natural lighting",
}
CSV_FIELDS = (
    "status",
    "stage",
    "variant",
    "prompt_id",
    "seed",
    "model_name",
    "image_path",
    "negative_prompt",
    "active_pair_ids",
    "strength",
    "margin",
    "tau",
    "schedule",
    "local_mask",
    "normalize_directions",
    "sad_score",
    "happy_score",
    "happy_minus_sad",
    "sitting_score",
    "walking_score",
    "walking_minus_sitting",
    "eating_score",
    "holding_score",
    "holding_minus_eating",
    "green_pixel_ratio_center",
    "red_pixel_ratio_center",
    "blue_pixel_ratio_center",
    "red_blue_to_green_ratio_center",
    "green_pixel_ratio_full",
    "red_pixel_ratio_full",
    "blue_pixel_ratio_full",
    "red_blue_to_green_ratio_full",
    "preservation_score",
    "adult_person_score",
    "park_score",
    "bench_score",
    "sandwich_score",
    "photo_style_score",
    "natural_lighting_score",
    "failure_notes",
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate benign park benchmark outputs.")
    parser.add_argument("--input-root", required=True)
    parser.add_argument("--benchmark", default=BENCHMARK_NAME)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--clip-model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-clip", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.benchmark != BENCHMARK_NAME:
        raise ValueError(f"Unsupported benchmark {args.benchmark!r}; expected {BENCHMARK_NAME!r}.")
    clip = None if args.skip_clip else ClipScorer(args.clip_model, args.device)
    rows = evaluate_root(Path(args.input_root), clip)
    summary = summarize(rows)
    payload = {
        "schema_version": 1,
        "benchmark": BENCHMARK_NAME,
        "input_root": str(Path(args.input_root).resolve()),
        "clip_model": None if args.skip_clip else args.clip_model,
        "num_rows": len(rows),
        "rows": rows,
        "summary_by_variant": summary,
    }
    write_json(args.output_json, payload)
    write_csv(args.output_csv, rows)
    print(f"Wrote {len(rows)} rows to {args.output_csv}")


def evaluate_root(input_root: Path, clip: "ClipScorer | None") -> list[dict[str, Any]]:
    result_paths = sorted(input_root.rglob("benchmark_job_result.json"))
    if not result_paths:
        raise FileNotFoundError(f"No benchmark_job_result.json files found under {input_root}.")
    rows: list[dict[str, Any]] = []
    for result_path in result_paths:
        result = json.loads(result_path.read_text(encoding="utf-8"))
        job = result.get("job", {})
        base_row = _base_row(result, job)
        if result.get("status") != "completed":
            base_row["failure_notes"] = result.get("reason") or result.get("error") or result.get("status")
            rows.append(base_row)
            continue
        image_paths = result.get("validated_media_paths") or []
        if not image_paths:
            base_row["status"] = "failed_evaluation"
            base_row["failure_notes"] = "completed job has no validated_media_paths"
            rows.append(base_row)
            continue
        for image_path in image_paths:
            row = dict(base_row)
            row["image_path"] = image_path
            image = Image.open(image_path).convert("RGB")
            row.update(color_metrics(image))
            if clip is not None:
                row.update(clip_metrics(image, clip))
            rows.append(row)
    return rows


class ClipScorer:
    def __init__(self, model_name: str, device: str) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.torch = torch
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def score(self, image: Image.Image, texts: dict[str, str]) -> dict[str, float]:
        inputs = self.processor(
            text=list(texts.values()),
            images=image,
            return_tensors="pt",
            padding=True,
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}
        with self.torch.inference_mode():
            outputs = self.model(**inputs)
            logits = outputs.logits_per_image[0].float()
            probs = logits.softmax(dim=0).detach().cpu().numpy()
        return {key: float(probs[index]) for index, key in enumerate(texts)}


def clip_metrics(image: Image.Image, clip: ClipScorer) -> dict[str, float]:
    scores = clip.score(image, CLIP_TEXTS)
    preservation_keys = (
        "preserve_person",
        "preserve_park",
        "preserve_bench",
        "preserve_sandwich",
        "preserve_photo",
        "preserve_lighting",
    )
    preservation_score = float(np.mean([scores[key] for key in preservation_keys]))
    return {
        "sad_score": scores["emotion_sad"],
        "happy_score": scores["emotion_happy"],
        "happy_minus_sad": scores["emotion_happy"] - scores["emotion_sad"],
        "sitting_score": scores["pose_sitting"],
        "walking_score": scores["pose_walking"],
        "walking_minus_sitting": scores["pose_walking"] - scores["pose_sitting"],
        "eating_score": scores["action_eating"],
        "holding_score": scores["action_holding"],
        "holding_minus_eating": scores["action_holding"] - scores["action_eating"],
        "preservation_score": preservation_score,
        "adult_person_score": scores["preserve_person"],
        "park_score": scores["preserve_park"],
        "bench_score": scores["preserve_bench"],
        "sandwich_score": scores["preserve_sandwich"],
        "photo_style_score": scores["preserve_photo"],
        "natural_lighting_score": scores["preserve_lighting"],
    }


def color_metrics(image: Image.Image) -> dict[str, float]:
    rgb = np.asarray(image).astype(np.float32) / 255.0
    h, w, _ = rgb.shape
    y0, y1 = int(h * 0.15), int(h * 0.90)
    x0, x1 = int(w * 0.25), int(w * 0.75)
    center = rgb[y0:y1, x0:x1]
    full = _color_ratios(rgb)
    cropped = _color_ratios(center)
    return {
        "green_pixel_ratio_center": cropped["green"],
        "red_pixel_ratio_center": cropped["red"],
        "blue_pixel_ratio_center": cropped["blue"],
        "red_blue_to_green_ratio_center": cropped["red_blue_to_green"],
        "green_pixel_ratio_full": full["green"],
        "red_pixel_ratio_full": full["red"],
        "blue_pixel_ratio_full": full["blue"],
        "red_blue_to_green_ratio_full": full["red_blue_to_green"],
    }


def _color_ratios(rgb: np.ndarray) -> dict[str, float]:
    red = rgb[..., 0]
    green = rgb[..., 1]
    blue = rgb[..., 2]
    saturation = rgb.max(axis=-1) - rgb.min(axis=-1)
    valid = saturation > 0.12
    green_mask = valid & (green > red * 1.10) & (green > blue * 1.10)
    red_mask = valid & (red > green * 1.10) & (red > blue * 1.05)
    blue_mask = valid & (blue > red * 1.05) & (blue > green * 1.05)
    total = max(float(rgb.shape[0] * rgb.shape[1]), 1.0)
    green_ratio = float(green_mask.sum() / total)
    red_ratio = float(red_mask.sum() / total)
    blue_ratio = float(blue_mask.sum() / total)
    return {
        "green": green_ratio,
        "red": red_ratio,
        "blue": blue_ratio,
        "red_blue_to_green": float((red_ratio + blue_ratio) / max(green_ratio, 1.0e-6)),
    }


def _base_row(result: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    spec = job.get("variant_spec", {})
    return {
        "status": result.get("status"),
        "stage": job.get("stage"),
        "variant": job.get("variant"),
        "prompt_id": job.get("prompt_id"),
        "seed": job.get("seed"),
        "model_name": job.get("model_name"),
        "image_path": "",
        "negative_prompt": job.get("negative_prompt", ""),
        "active_pair_ids": ",".join(spec.get("active_pair_ids", [])),
        "strength": spec.get("strength", ""),
        "margin": spec.get("margin", ""),
        "tau": spec.get("tau", ""),
        "schedule": spec.get("schedule", ""),
        "local_mask": spec.get("local_mask", ""),
        "normalize_directions": spec.get("normalize_directions", ""),
        "failure_notes": "",
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    numeric_fields = [field for field in CSV_FIELDS if field not in {"status", "stage", "variant", "prompt_id", "seed", "model_name", "image_path", "negative_prompt", "active_pair_ids", "schedule", "local_mask", "normalize_directions", "failure_notes"}]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("variant", ""))].append(row)
    summary = {}
    for variant, variant_rows in grouped.items():
        completed = [row for row in variant_rows if row.get("status") == "completed"]
        values = {}
        for field in numeric_fields:
            nums = [float(row[field]) for row in completed if field in row and row[field] != ""]
            if nums:
                values[field] = float(np.mean(nums))
        summary[variant] = {
            "num_rows": len(variant_rows),
            "num_completed": len(completed),
            "means": values,
        }
    return summary


def write_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    ensure_dir(target.parent)
    with target.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in CSV_FIELDS})


if __name__ == "__main__":
    main()
