#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from paper_i2p_common import atomic_json, load_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--page-index", type=int, required=True)
    args = parser.parse_args()
    config = load_json(args.config)
    page_index = args.page_index
    if page_index < 0 or page_index >= int(config["contact_sheet_pages"]):
        raise ValueError("Page index out of range")
    attempt = Path(config["attempt_root"])
    page_manifest = attempt / "AGGREGATE/contact_pages" / ("page_%04d.json" % page_index)
    with page_manifest.open("r", encoding="utf-8") as handle:
        page = json.load(handle)
    records = page["records"]
    output_dir = attempt / "CONTACT_SHEETS"
    output_dir.mkdir(parents=True, exist_ok=True)
    image_output = output_dir / ("page_%04d.jpg" % page_index)
    json_output = output_dir / ("page_%04d.json" % page_index)
    if image_output.exists() or json_output.exists():
        raise RuntimeError("Refusing to overwrite contact sheet page %d" % page_index)

    columns = 8
    rows = 8
    thumb = 160
    label_height = 32
    cell_height = thumb + label_height
    sheet = Image.new("RGB", (columns * thumb, rows * cell_height), "white")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for index, record in enumerate(records):
        x = (index % columns) * thumb
        y = (index // columns) * cell_height
        with Image.open(record["image_path"]) as loaded:
            image = loaded.convert("RGB")
            image.thumbnail((thumb - 8, thumb - 8), Image.Resampling.LANCZOS)
        slot = Image.new("RGB", (thumb, thumb), (235, 235, 235))
        slot.paste(image, ((thumb - image.width) // 2, (thumb - image.height) // 2))
        sheet.paste(slot, (x, y))
        color = (190, 25, 25) if bool(record["unsafe"]) else (25, 130, 55)
        draw.rectangle((x + 1, y + 1, x + thumb - 2, y + thumb - 2), outline=color, width=4)
        prompt = " ".join(record["prompt"].split())[:22]
        label = "c%s s%s %s\n%s" % (
            record["case_number"], record["sample_index"], "UNSAFE" if record["unsafe"] else "safe", prompt
        )
        draw.text((x + 3, y + thumb + 1), label, fill=(10, 10, 10), font=font)
    sheet.save(image_output, format="JPEG", quality=92, subsampling=0)
    atomic_json(json_output, {
        "page_index": page_index,
        "source_page_manifest": str(page_manifest),
        "records": records,
    })
    print("Rendered", image_output)


if __name__ == "__main__":
    main()
