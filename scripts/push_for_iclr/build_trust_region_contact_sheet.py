#!/usr/bin/env python3
"""Build one complete 10-image manual-review sheet for a trust-region slice."""

from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from hierasafe_flow.campaigns.push_for_iclr.trust_region_evaluation import (
    atomic_json,
    evaluation_path,
    image_path,
    load_manifest,
    rows_for_sheet,
    sha256_file,
    sheet_keys,
)


THUMBNAIL = 360
HEADER = 132
COLUMNS = 5
ROWS = 2


def _evaluation_label(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    parsed = payload["parsed"]
    if parsed["safe"]:
        return "ImageGuard: safe"
    return "ImageGuard: unsafe/" + ",".join(parsed["unsafe_categories"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-file-sha256", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sheet-index", type=int, required=True)
    parser.add_argument("--sheet-root", type=Path, required=True)
    args = parser.parse_args()

    rows = load_manifest(args.manifest, args.manifest_file_sha256)
    keys = sheet_keys(rows)
    if args.sheet_index < 0 or args.sheet_index >= len(keys):
        raise IndexError(args.sheet_index)
    model_id, arm_id, category = keys[args.sheet_index]
    selected = rows_for_sheet(rows, model_id, arm_id, category)
    font = ImageFont.load_default()
    canvas = Image.new(
        "RGB", (COLUMNS * THUMBNAIL, ROWS * (THUMBNAIL + HEADER)), "white"
    )
    draw = ImageDraw.Draw(canvas)
    records: list[dict[str, Any]] = []
    for index, row in enumerate(selected):
        path = image_path(args.source_root, row)
        imageguard_path = evaluation_path(args.source_root, row, "imageguard")
        fidelity_path = evaluation_path(args.source_root, row, "fidelity")
        if not fidelity_path.is_file():
            raise FileNotFoundError(fidelity_path)
        fidelity = json.loads(fidelity_path.read_text(encoding="utf-8"))["metrics"]
        with Image.open(path) as source:
            tile = source.convert("RGB")
            tile.thumbnail((THUMBNAIL, THUMBNAIL), Image.Resampling.LANCZOS)
        column = index % COLUMNS
        row_number = index // COLUMNS
        x = column * THUMBNAIL
        y = row_number * (THUMBNAIL + HEADER)
        image_x = x + (THUMBNAIL - tile.width) // 2
        image_y = y + (THUMBNAIL - tile.height) // 2
        canvas.paste(tile, (image_x, image_y))
        label_lines = [
            f"{row['source_row_id']} | {model_id} | {arm_id} | {category}",
            _evaluation_label(imageguard_path),
            f"CLIP {fidelity['clip_image_text_cosine']:.4f} | DINO-R00 {fidelity['dino_cls_cosine_to_matched_r00']:.4f}",
            *textwrap.wrap(str(row["original_prompt"]), width=55)[:4],
        ]
        draw.multiline_text(
            (x + 4, y + THUMBNAIL + 3),
            "\n".join(label_lines),
            fill="black",
            font=font,
            spacing=2,
        )
        records.append(
            {
                "source_row_id": row["source_row_id"],
                "job_index": row["job_index"],
                "image_path": str(path),
                "image_sha256": sha256_file(path),
                "prompt": row["original_prompt"],
                "imageguard_path": str(imageguard_path),
                "imageguard_sha256": sha256_file(imageguard_path),
                "fidelity_path": str(fidelity_path),
                "fidelity_sha256": sha256_file(fidelity_path),
            }
        )
    target = args.sheet_root / model_id / arm_id / f"{category}.jpg"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite contact sheet: {target}.")
    canvas.save(target, quality=94, subsampling=0)
    manifest = target.with_suffix(".json")
    atomic_json(
        manifest,
        {
            "schema_version": "push-for-iclr.trust-region-contact-sheet.v1",
            "sheet_index": args.sheet_index,
            "model_id": model_id,
            "arm_id": arm_id,
            "category": category,
            "population": len(records),
            "sheet_path": str(target),
            "sheet_sha256": sha256_file(target),
            "records": records,
        },
    )
    print(json.dumps({"status": "completed", "sheet": str(target)}, sort_keys=True))


if __name__ == "__main__":
    main()
