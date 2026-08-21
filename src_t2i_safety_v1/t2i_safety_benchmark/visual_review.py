from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .contracts import (
    OUTPUT_ROOT,
    WORK_ROOT,
    BenchmarkContract,
    PromptRow,
    atomic_json,
    file_sha256,
)


REVIEW_ROOT = WORK_ROOT / "visual_review"
VARIANTS = (
    "baseline",
    "native_negative_prompt",
    "conceptsteer",
    "midsteer",
    "sgf",
    "safe_denoiser",
)
THUMBNAIL = 512
HEADER = 64
PROMPT_HEADER = 128
COLUMNS = 3
ROWS = 2


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _attempt(model_id: str, variant: str, row: PromptRow) -> Path:
    return (
        OUTPUT_ROOT
        / model_id
        / variant
        / row.domain
        / row.category
        / row.row_id
        / "attempt_001"
    )


def _variant_supported(
    contract: BenchmarkContract,
    model_id: str,
    variant: str,
) -> bool:
    return not (
        variant == "native_negative_prompt"
        and not contract.model(model_id)["native_negative_supported"]
    )


def _source_records(
    contract: BenchmarkContract,
    model_id: str,
    row: PromptRow,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for variant in VARIANTS:
        supported = _variant_supported(contract, model_id, variant)
        attempt = _attempt(model_id, variant, row)
        if not supported:
            records.append(
                {
                    "variant": variant,
                    "supported": False,
                    "status": "unsupported_by_model",
                }
            )
            continue
        image = attempt / "image.png"
        success = attempt / "_SUCCESS.json"
        if not image.is_file() or not success.is_file():
            raise FileNotFoundError(
                f"Visual-review source is incomplete for {model_id}/{variant}/"
                f"{row.row_id}: {attempt}"
            )
        records.append(
            {
                "variant": variant,
                "supported": True,
                "attempt": str(attempt),
                "image": str(image),
                "image_sha256": file_sha256(image),
                "success_sha256": file_sha256(success),
            }
        )
    return records


def _render(
    *,
    model_id: str,
    row: PromptRow,
    records: list[dict[str, Any]],
) -> Image.Image:
    width = COLUMNS * THUMBNAIL
    height = PROMPT_HEADER + ROWS * (HEADER + THUMBNAIL)
    sheet = Image.new("RGB", (width, height), color=(244, 240, 231))
    draw = ImageDraw.Draw(sheet)
    title_font = _font(22)
    label_font = _font(20)
    detail_font = _font(16)
    title = f"{model_id} | {row.category} | {row.row_id}"
    draw.text((18, 12), title, fill=(20, 24, 25), font=title_font)
    wrapped = textwrap.wrap(row.prompt, width=145)
    draw.multiline_text(
        (18, 44),
        "\n".join(wrapped[:4]),
        fill=(45, 48, 48),
        font=detail_font,
        spacing=3,
    )
    for index, record in enumerate(records):
        column = index % COLUMNS
        row_index = index // COLUMNS
        left = column * THUMBNAIL
        top = PROMPT_HEADER + row_index * (HEADER + THUMBNAIL)
        background = (221, 228, 221) if record["supported"] else (229, 218, 212)
        draw.rectangle(
            (left, top, left + THUMBNAIL, top + HEADER),
            fill=background,
        )
        draw.text(
            (left + 14, top + 17),
            record["variant"],
            fill=(18, 25, 23),
            font=label_font,
        )
        image_top = top + HEADER
        if record["supported"]:
            with Image.open(record["image"]) as source:
                source = source.convert("RGB")
                source.thumbnail((THUMBNAIL, THUMBNAIL), Image.Resampling.LANCZOS)
                canvas = Image.new(
                    "RGB",
                    (THUMBNAIL, THUMBNAIL),
                    color=(16, 17, 17),
                )
                x = (THUMBNAIL - source.width) // 2
                y = (THUMBNAIL - source.height) // 2
                canvas.paste(source, (x, y))
            sheet.paste(canvas, (left, image_top))
        else:
            draw.rectangle(
                (
                    left,
                    image_top,
                    left + THUMBNAIL,
                    image_top + THUMBNAIL,
                ),
                fill=(66, 61, 58),
            )
            draw.text(
                (left + 150, image_top + 240),
                "NATIVE UNSUPPORTED",
                fill=(245, 238, 225),
                font=label_font,
            )
    return sheet


def build_sheet(
    contract: BenchmarkContract,
    model_id: str,
    row: PromptRow,
    target: Path,
) -> dict[str, Any]:
    records = _source_records(contract, model_id, row)
    metadata_path = target.with_suffix(".json")
    identity = {
        "schema_version": 1,
        "model_id": model_id,
        "row_id": row.row_id,
        "row_sha256": row.row_sha256,
        "variants": records,
    }
    if target.exists() or metadata_path.exists():
        if not target.is_file() or not metadata_path.is_file():
            raise RuntimeError(f"Incomplete existing review sheet {target}")
        existing = json.loads(metadata_path.read_text(encoding="utf-8"))
        if existing.get("identity") != identity:
            raise RuntimeError(f"Review sheet provenance changed for {target}")
        if existing.get("sheet_sha256") != file_sha256(target):
            raise RuntimeError(f"Review sheet hash changed for {target}")
        return existing

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.name}.tmp")
    sheet = _render(model_id=model_id, row=row, records=records)
    sheet.save(temporary, format="PNG")
    temporary.replace(target)
    metadata = {
        "identity": identity,
        "sheet_path": str(target),
        "sheet_sha256": file_sha256(target),
        "width": sheet.width,
        "height": sheet.height,
        "review_contract": (
            "direct_current_model_visual_inspection_required_no_gemini"
        ),
    }
    atomic_json(metadata_path, metadata)
    return metadata


def _pilot_rows(contract: BenchmarkContract) -> list[PromptRow]:
    first_by_category: dict[str, PromptRow] = {}
    for row in sorted(
        contract.prompt_rows().values(),
        key=lambda value: value.release_index,
    ):
        first_by_category.setdefault(row.category, row)
    if set(first_by_category) != set(contract.categories):
        raise RuntimeError("Pilot visual review lacks a benchmark category.")
    return [first_by_category[category] for category in contract.categories]


def build_pilot_sheets() -> dict[str, Any]:
    contract = BenchmarkContract()
    records: list[dict[str, Any]] = []
    for model_id in contract.models:
        for row in _pilot_rows(contract):
            target = REVIEW_ROOT / "pilot" / model_id / f"{row.category}.png"
            records.append(build_sheet(contract, model_id, row, target))
    result = {
        "schema_version": 1,
        "kind": "pilot_cross_variant_sheets",
        "sheet_count": len(records),
        "models": list(contract.models),
        "categories": list(contract.categories),
        "sheets": [
            {
                "path": record["sheet_path"],
                "sha256": record["sheet_sha256"],
            }
            for record in records
        ],
    }
    atomic_json(REVIEW_ROOT / "pilot" / "manifest.json", result)
    return result


def build_full_shard(
    model_id: str,
    shard_index: int,
    num_shards: int,
) -> dict[str, Any]:
    if num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError("Invalid visual-review shard coordinates.")
    contract = BenchmarkContract()
    contract.model(model_id)
    rows = [
        row
        for row in contract.prompt_rows().values()
        if row.release_index % num_shards == shard_index
    ]
    records: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda value: value.release_index):
        target = (
            REVIEW_ROOT
            / "full"
            / model_id
            / row.domain
            / row.category
            / f"{row.row_id}.png"
        )
        value = build_sheet(contract, model_id, row, target)
        records.append(
            {
                "row_id": row.row_id,
                "path": value["sheet_path"],
                "sha256": value["sheet_sha256"],
            }
        )
    result = {
        "schema_version": 1,
        "kind": "full_cross_variant_sheet_shard",
        "model_id": model_id,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "sheet_count": len(records),
        "sheets": records,
    }
    atomic_json(
        REVIEW_ROOT
        / "full"
        / model_id
        / f"manifest_{shard_index:04d}_of_{num_shards:04d}.json",
        result,
    )
    return result


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Build direct cross-variant visual-review sheets."
    )
    sub = value.add_subparsers(dest="command", required=True)
    sub.add_parser("pilots")
    full = sub.add_parser("full-shard")
    full.add_argument("--model", required=True)
    full.add_argument("--shard-index", type=int, required=True)
    full.add_argument("--num-shards", type=int, required=True)
    return value


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "pilots":
        result = build_pilot_sheets()
    elif args.command == "full-shard":
        result = build_full_shard(
            args.model,
            args.shard_index,
            args.num_shards,
        )
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
