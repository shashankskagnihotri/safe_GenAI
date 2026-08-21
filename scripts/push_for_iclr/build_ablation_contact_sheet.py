#!/usr/bin/env python3
"""Build one immutable category contact sheet for a completed ablation group."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hierasafe_flow.campaigns.push_for_iclr.ablation_manifests import (  # noqa: E402
    validate_sealed_job_manifest,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--ablation", required=True)
    parser.add_argument("--category", choices=["nudity", "violence"], required=True)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    current_commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if current_commit != args.code_commit:
        raise RuntimeError("Evaluation worktree HEAD changed")
    rows, _ = validate_sealed_job_manifest(args.manifest.resolve())
    selected = [
        row
        for row in rows
        if row["model_id"] == args.model
        and row["ablation_id"] == args.ablation
        and row["category"] == args.category
    ]
    if len(selected) != 15:
        raise RuntimeError(f"Expected 15 rows, got {len(selected)}")
    cells = [
        REPOSITORY_ROOT / "outputs/PUSH_FOR_ICLR" / row["expected_output_relative_path"]
        for row in selected
    ]
    if not all((cell / "_SUCCESS").exists() for cell in cells):
        raise RuntimeError("Contact-sheet group is incomplete")
    output = (
        REPOSITORY_ROOT
        / "outputs/PUSH_FOR_ICLR/CONTACT_SHEETS/stage2"
        / args.model
        / args.ablation
        / f"{args.category}_page_1.jpg"
    )
    if output.exists():
        print(f"CONTACT_SHEET_ALREADY_COMPLETE {output}")
        return 0
    columns, rows_count = 5, 3
    tile_width, image_height, label_height = 320, 320, 72
    canvas = Image.new("RGB", (columns * tile_width, rows_count * (image_height + label_height)), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default(size=14)
    for index, (row, cell) in enumerate(zip(selected, cells)):
        image = Image.open(cell / "image.png").convert("RGB")
        image.thumbnail((tile_width, image_height), Image.Resampling.LANCZOS)
        x = (index % columns) * tile_width
        y = (index // columns) * (image_height + label_height)
        canvas.paste(image, (x + (tile_width - image.width) // 2, y + (image_height - image.height) // 2))
        label = f"{row['source_row_id']} | {row['difficulty_stratum']}\n{row['original_prompt'][:90]}"
        draw.multiline_text((x + 5, y + image_height + 4), label, fill="black", font=font, spacing=2)
    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.partial.{os.getpid()}")
    canvas.save(partial, format="JPEG", quality=92, optimize=True)
    os.replace(partial, output)
    print(f"CONTACT_SHEET_COMPLETE {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
