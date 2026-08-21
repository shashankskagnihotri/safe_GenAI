#!/usr/bin/env python3
"""Prepare local images/contact sheets for manual inspection; no external API."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

from hierasafe_flow.campaigns.chatgpt_steering_22_july import (
    load_campaign_spec,
    primary_media_files,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default="configs/experiments/chatgpt_steering_22_july.yaml",
    )
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()
    spec = load_campaign_spec(args.config)
    root = Path(spec["_root"])
    review_root = root / spec["campaign"]["state_root"] / "manual_review"
    review_root.mkdir(parents=True, exist_ok=True)
    rows = [
        json.loads(line)
        for line in Path(args.manifest).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assets = []
    for row in rows:
        output = root / row["output_dir"]
        media = primary_media_files(output, row["modality"])
        if row["execution_status"] == "unsupported_by_model":
            assets.append(
                {
                    "cell_id": row["cell_id"],
                    "kind": "unsupported_by_model",
                    "path": str(output / "UNSUPPORTED_BY_MODEL.json"),
                }
            )
            continue
        if len(media) != 1:
            assets.append(
                {
                    "cell_id": row["cell_id"],
                    "kind": "missing_or_ambiguous",
                    "media_count": len(media),
                }
            )
            continue
        source = media[0]
        if source.suffix.lower() in {".mp4", ".webm", ".gif"}:
            target = review_root / f"{row['cell_id']}__contact.png"
            subprocess.run(
                [
                    "ffmpeg",
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(source),
                    "-vf",
                    "fps=1/1.5,scale=320:-1,tile=5x2",
                    "-frames:v",
                    "1",
                    str(target),
                ],
                check=True,
            )
            kind = "video_contact_sheet"
        else:
            target = source
            kind = "image"
        assets.append(
            {
                "cell_id": row["cell_id"],
                "prompt_id": row["prompt_id"],
                "model_id": row["model_id"],
                "variant": row["variant"],
                "kind": kind,
                "path": str(target),
                "source_media": str(source),
                "manual_status": "pending",
            }
        )
    target = review_root / f"{Path(args.manifest).stem}_assets.json"
    target.write_text(
        json.dumps(assets, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({"assets": len(assets), "manifest": str(target)}, sort_keys=True))


if __name__ == "__main__":
    main()
