#!/usr/bin/env python3
"""Build the complete three-arm NAG qualitative contact sheet."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps


THUMBNAIL = 384
HEADER = 72
ROW_LABEL = 54
GAP = 12


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def atomic_json(path: Path, payload: object) -> None:
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    config = load_json(args.config.resolve())
    root = Path(config["output_root"])
    contact_dir = root / "CONTACT_SHEETS"
    contact_dir.mkdir(parents=True, exist_ok=True)
    output_path = contact_dir / "page_000.jpg"
    manifest_path = contact_dir / "MANIFEST.json"
    if output_path.exists() or manifest_path.exists():
        raise FileExistsError("Refusing to overwrite an existing contact-sheet artifact")

    arms = config["arms"]
    rows = [
        (prompt_index, prompt, seed)
        for prompt_index, prompt in enumerate(config["prompts"])
        for seed in config["generation"]["seeds"]
    ]
    width = GAP + len(arms) * (THUMBNAIL + GAP)
    row_height = ROW_LABEL + THUMBNAIL + GAP
    height = HEADER + len(rows) * row_height + GAP
    sheet = Image.new("RGB", (width, height), color=(20, 22, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for column, arm in enumerate(arms):
        x = GAP + column * (THUMBNAIL + GAP)
        draw.text((x, 20), arm["id"], fill=(245, 245, 240), font=font)

    cells = []
    for row, (prompt_index, prompt, seed) in enumerate(rows):
        y = HEADER + row * row_height
        label = f"{prompt['id']} | seed={seed} | {prompt['prompt']}"
        draw.text((GAP, y + 8), label[:180], fill=(225, 225, 215), font=font)
        for column, arm in enumerate(arms):
            source = (
                root
                / arm["id"]
                / f"prompt_{prompt_index:02d}_{prompt['id']}"
                / f"seed_{seed:06d}.png"
            )
            if not source.is_file():
                raise FileNotFoundError(source)
            with Image.open(source) as image:
                rendered = ImageOps.contain(image.convert("RGB"), (THUMBNAIL, THUMBNAIL))
            x = GAP + column * (THUMBNAIL + GAP)
            image_y = y + ROW_LABEL
            sheet.paste(rendered, (x, image_y))
            cells.append(
                {
                    "row": row,
                    "column": column,
                    "arm": arm["id"],
                    "prompt_id": prompt["id"],
                    "seed": seed,
                    "source": str(source),
                }
            )
    sheet.save(output_path, format="JPEG", quality=94, subsampling=0)
    atomic_json(
        manifest_path,
        {
            "schema_version": 1,
            "page": str(output_path),
            "rows": len(rows),
            "columns": len(arms),
            "cells": cells,
        },
    )
    print(output_path)


if __name__ == "__main__":
    main()
