from __future__ import annotations

import argparse
import json
import textwrap
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from .contracts import CALIBRATION_ROOT, WORK_ROOT, atomic_json, file_sha256
from .fairness_manifest import PROBE_COUNT, load_manifest


AUDIT_ROOT = WORK_ROOT / "visual_audits" / "fairness_full_v2"
SHEET_SIZE = 20
COLUMNS = 5
ROWS = 4
TILE_WIDTH = 320
IMAGE_HEIGHT = 256
HEADER_HEIGHT = 72
SHEET_WIDTH = COLUMNS * TILE_WIDTH
SHEET_HEIGHT = ROWS * (HEADER_HEIGHT + IMAGE_HEIGHT)
FAILURE_TYPES = frozenset(
    {
        "non_human",
        "no_visible_person",
        "multiple_people",
        "blocked_or_placeholder",
        "severely_corrupted",
        "not_a_portrait",
        "occupation_unreadable",
        "other",
    }
)
REVIEWER_CONTRACT = "gpt-5.6-sol-ultra_current_session_direct_visual_review"


def _font(size: int) -> ImageFont.ImageFont:
    path = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _attempt(model_id: str, probe_id: str) -> Path:
    return (
        CALIBRATION_ROOT
        / "fairness_probe"
        / model_id
        / probe_id
        / "attempt_001"
    )


def _records(model_id: str) -> tuple[list[dict[str, Any]], str]:
    rows, probe_manifest_sha = load_manifest()
    records: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        attempt = _attempt(model_id, row.probe_id)
        image = attempt / "image.png"
        metadata = attempt / "metadata.json"
        success = attempt / "_SUCCESS.json"
        for path in (image, metadata, success):
            if not path.is_file():
                raise FileNotFoundError(path)
        metadata_value = json.loads(metadata.read_text(encoding="utf-8"))
        if metadata_value.get("model_id") != model_id:
            raise RuntimeError(f"Fairness metadata model mismatch in {metadata}")
        probe_row = metadata_value.get("probe_row")
        if (
            not isinstance(probe_row, dict)
            or probe_row.get("probe_id") != row.probe_id
            or probe_row.get("prompt_sha256") != row.prompt_sha256
        ):
            raise RuntimeError(f"Fairness probe binding mismatch in {metadata}")
        image_sha = file_sha256(image)
        if metadata_value.get("image", {}).get("sha256") != image_sha:
            raise RuntimeError(f"Fairness image hash mismatch in {metadata}")
        records.append(
            {
                "index": index,
                "probe_id": row.probe_id,
                "prompt_sha256": row.prompt_sha256,
                "occupation": row.occupation,
                "prompt": row.prompt,
                "image_path": str(image),
                "image_sha256": image_sha,
                "metadata_path": str(metadata),
                "metadata_sha256": file_sha256(metadata),
                "success_sha256": file_sha256(success),
            }
        )
    if len(records) != PROBE_COUNT:
        raise RuntimeError(
            f"Expected {PROBE_COUNT} fairness images, observed {len(records)}"
        )
    return records, probe_manifest_sha


def _render_sheet(records: list[dict[str, Any]], target: Path) -> None:
    sheet = Image.new("RGB", (SHEET_WIDTH, SHEET_HEIGHT), (238, 235, 226))
    draw = ImageDraw.Draw(sheet)
    title_font = _font(16)
    detail_font = _font(13)
    for position, record in enumerate(records):
        column = position % COLUMNS
        row = position // COLUMNS
        left = column * TILE_WIDTH
        top = row * (HEADER_HEIGHT + IMAGE_HEIGHT)
        draw.rectangle(
            (left, top, left + TILE_WIDTH, top + HEADER_HEIGHT),
            fill=(212, 222, 216),
        )
        draw.text(
            (left + 8, top + 5),
            f"{record['index']:04d} | {record['probe_id']}",
            fill=(17, 24, 22),
            font=title_font,
        )
        occupation = "\n".join(
            textwrap.wrap(str(record["occupation"]), width=38)[:2]
        )
        draw.multiline_text(
            (left + 8, top + 29),
            occupation,
            fill=(38, 45, 43),
            font=detail_font,
            spacing=2,
        )
        with Image.open(record["image_path"]) as source:
            source = source.convert("RGB")
            source.thumbnail(
                (TILE_WIDTH, IMAGE_HEIGHT),
                Image.Resampling.LANCZOS,
            )
            canvas = Image.new(
                "RGB",
                (TILE_WIDTH, IMAGE_HEIGHT),
                (18, 19, 19),
            )
            x = (TILE_WIDTH - source.width) // 2
            y = (IMAGE_HEIGHT - source.height) // 2
            canvas.paste(source, (x, y))
        sheet.paste(canvas, (left, top + HEADER_HEIGHT))
    temporary = target.with_name(f".{target.name}.tmp")
    sheet.save(temporary, format="PNG")
    temporary.replace(target)


def build(model_id: str) -> dict[str, Any]:
    model_root = AUDIT_ROOT / model_id
    manifest_path = model_root / "SHEET_MANIFEST.json"
    if manifest_path.exists():
        value = json.loads(manifest_path.read_text(encoding="utf-8"))
        for sheet in value.get("sheets", []):
            path = Path(sheet["path"])
            if not path.is_file() or file_sha256(path) != sheet["sha256"]:
                raise RuntimeError(f"Existing fairness review sheet changed: {path}")
        return value
    if model_root.exists():
        raise RuntimeError(f"Refusing incomplete visual-audit directory {model_root}")
    model_root.mkdir(parents=True)
    records, probe_manifest_sha = _records(model_id)
    sheets: list[dict[str, Any]] = []
    page_count = (len(records) + SHEET_SIZE - 1) // SHEET_SIZE
    for page_index, start in enumerate(range(0, len(records), SHEET_SIZE)):
        page = records[start : start + SHEET_SIZE]
        target = model_root / f"sheet_{page_index:03d}_of_{page_count:03d}.png"
        _render_sheet(page, target)
        sheets.append(
            {
                "page_index": page_index,
                "path": str(target),
                "sha256": file_sha256(target),
                "first_index": page[0]["index"],
                "last_index": page[-1]["index"],
                "record_count": len(page),
            }
        )
    value = {
        "schema_version": 1,
        "kind": "full_fairness_probe_direct_visual_review_sheets",
        "review_contract": REVIEWER_CONTRACT,
        "model_id": model_id,
        "probe_manifest_sha256": probe_manifest_sha,
        "record_count": len(records),
        "sheet_size": SHEET_SIZE,
        "sheet_count": len(sheets),
        "records": records,
        "sheets": sheets,
    }
    atomic_json(manifest_path, value)
    return value


def admit(model_id: str, review_path: Path) -> dict[str, Any]:
    model_root = AUDIT_ROOT / model_id
    manifest_path = model_root / "SHEET_MANIFEST.json"
    admission_path = model_root / "VISUAL_ADMISSION.json"
    if admission_path.exists():
        raise FileExistsError(admission_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        record["probe_id"]: record
        for record in manifest["records"]
    }
    reviews: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(
        review_path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise RuntimeError(f"Blank visual-review line {line_number}")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise RuntimeError(f"Visual-review line {line_number} is not an object")
        probe_id = str(value.get("probe_id", ""))
        if probe_id not in expected or probe_id in reviews:
            raise RuntimeError(f"Unknown or duplicate visual review {probe_id}")
        if value.get("image_sha256") != expected[probe_id]["image_sha256"]:
            raise RuntimeError(f"Visual-review image hash mismatch for {probe_id}")
        if value.get("reviewer_contract") != REVIEWER_CONTRACT:
            raise RuntimeError(f"Visual-review model mismatch for {probe_id}")
        valid = value.get("valid_human_portrait")
        failures = value.get("failure_types")
        if type(valid) is not bool or not isinstance(failures, list):
            raise RuntimeError(f"Malformed visual decision for {probe_id}")
        if (
            any(failure not in FAILURE_TYPES for failure in failures)
            or (valid and failures)
            or (not valid and not failures)
        ):
            raise RuntimeError(f"Inconsistent visual decision for {probe_id}")
        reviews[probe_id] = value
    if set(reviews) != set(expected):
        raise RuntimeError(
            f"Visual review is incomplete: {len(reviews)}/{len(expected)}"
        )
    invalid = [
        value for value in reviews.values() if not value["valid_human_portrait"]
    ]
    failure_type_counts = Counter(
        failure
        for decision in invalid
        for failure in decision["failure_types"]
    )
    blocked_or_placeholder_probe_ids = sorted(
        decision["probe_id"]
        for decision in invalid
        if "blocked_or_placeholder" in decision["failure_types"]
    )
    value = {
        "schema_version": 2,
        "status": "accepted",
        "kind": "complete_direct_fairness_visual_admission",
        "reviewer_contract": REVIEWER_CONTRACT,
        "model_id": model_id,
        "record_count": len(reviews),
        "valid_count": len(reviews) - len(invalid),
        "valid_rate": (len(reviews) - len(invalid)) / len(reviews),
        "invalid_count": len(invalid),
        "invalid_probe_ids": sorted(value["probe_id"] for value in invalid),
        "invalid_records": sorted(
            (
                {
                    "probe_id": decision["probe_id"],
                    "failure_types": sorted(decision["failure_types"]),
                }
                for decision in invalid
            ),
            key=lambda decision: decision["probe_id"],
        ),
        "failure_type_counts": dict(sorted(failure_type_counts.items())),
        "blocked_or_placeholder_count": len(blocked_or_placeholder_probe_ids),
        "blocked_or_placeholder_rate": (
            len(blocked_or_placeholder_probe_ids) / len(reviews)
        ),
        "blocked_or_placeholder_probe_ids": blocked_or_placeholder_probe_ids,
        "sheet_manifest_path": str(manifest_path),
        "sheet_manifest_sha256": file_sha256(manifest_path),
        "review_path": str(review_path),
        "review_sha256": file_sha256(review_path),
    }
    atomic_json(admission_path, value)
    return value


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Build and admit complete direct fairness visual reviews."
    )
    sub = value.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--model", required=True)
    admit_parser = sub.add_parser("admit")
    admit_parser.add_argument("--model", required=True)
    admit_parser.add_argument("--review-jsonl", type=Path, required=True)
    return value


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "build":
        result = build(args.model)
    elif args.command == "admit":
        result = admit(args.model, args.review_jsonl)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
