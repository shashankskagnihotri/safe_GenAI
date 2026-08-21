#!/usr/bin/env python3
"""Build a deterministic custom-prompt ensemble for a MidSteer diagnostic."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


SCHEMA_VERSION = 1
PROTOCOL = "midsteer_custom_paired_prompt_ensemble_v1"


def read_row(path: Path, index: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for current, line in enumerate(handle):
            if current == index:
                return json.loads(line)
    raise IndexError(f"Manifest {path} has no row {index}")


def context(index: int) -> str:
    settings = (
        "in a spacious city park",
        "in a quiet botanical garden",
        "beside a tree-lined pedestrian path",
        "near a sunlit public plaza",
        "in an open neighborhood green space",
        "along a landscaped riverside walkway",
        "in a broad community garden",
        "near a calm outdoor pavilion",
    )
    lighting = (
        "soft morning daylight",
        "clear midday natural light",
        "warm late-afternoon sunlight",
        "diffuse overcast daylight",
        "gentle side lighting",
        "bright spring daylight",
        "balanced natural illumination",
        "subtle golden-hour light",
    )
    cameras = (
        "documentary photography",
        "realistic editorial photography",
        "naturalistic cinematic photography",
        "high-detail environmental portraiture",
        "observational lifestyle photography",
        "realistic full-scene photography",
        "unretouched location photography",
        "careful contextual portrait photography",
    )
    framings = (
        "wide contextual framing with the full subject visible",
        "portrait-oriented framing with generous surrounding space",
        "eye-level framing with coherent full-body anatomy",
        "balanced environmental composition without cropping",
        "clear foreground and background separation",
        "complete scene coverage with realistic scale",
        "stable perspective with all important objects visible",
        "natural depth and an uncluttered composition",
    )
    values = (
        settings[index % len(settings)],
        lighting[(index // len(settings)) % len(lighting)],
        cameras[(index // (len(settings) * len(lighting))) % len(cameras)],
        framings[
            (index // (len(settings) * len(lighting) * len(cameras)))
            % len(framings)
        ],
    )
    return ", ".join(values)


def neutral_prompt(index: int) -> str:
    subjects = (
        "a ceramic teapot and two cups on a wooden table",
        "a stone lighthouse above a quiet coastline",
        "a row of bicycles beside a brick wall",
        "an old library reading room with tall windows",
        "a small sailboat crossing a calm lake",
        "a bowl of citrus fruit beside folded linen",
        "a mountain cabin surrounded by pine trees",
        "an empty tram stop after light rain",
        "a greenhouse filled with tropical plants",
        "a vintage camera resting on a desk",
        "a narrow footbridge over a shallow stream",
        "an outdoor cafe before opening time",
        "a red mailbox beside a country road",
        "a workshop shelf holding hand tools",
        "a quiet museum gallery with framed landscapes",
        "a market stall displaying fresh vegetables",
    )
    subject = subjects[index % len(subjects)]
    return f"{subject}, {context(index + 97)}, realistic detail"


def build(
    row: dict[str, Any],
    *,
    pair_ids: list[str],
    neutral_count: int,
    concept_count: int,
) -> dict[str, Any]:
    if neutral_count < 2 or concept_count < 2:
        raise ValueError("MidSteer ensemble populations must each be at least two")
    concept_path = Path(str(row["concept_manifest"]))
    encoded_concept = concept_path.read_bytes()
    observed_sha = hashlib.sha256(encoded_concept).hexdigest()
    if observed_sha != row["concept_manifest_sha256"]:
        raise RuntimeError(
            f"Concept manifest hash mismatch: {observed_sha} != "
            f"{row['concept_manifest_sha256']}"
        )
    tree = yaml.safe_load(encoded_concept)
    by_id = {str(pair["id"]): dict(pair) for pair in tree["pairs"]}
    unknown = sorted(set(pair_ids) - set(by_id))
    if unknown:
        raise ValueError(f"Unknown MidSteer pair ids: {unknown}")
    pairs: dict[str, dict[str, list[str]]] = {}
    for pair_id in pair_ids:
        pair = by_id[pair_id]
        source = str(pair["unsafe_concept"])
        target = str(pair.get("target_concept", pair["safe_sibling_concept"]))
        pairs[pair_id] = {
            "source": [
                f"{source}. Render this as {context(index)}."
                for index in range(concept_count)
            ],
            "target": [
                f"{target}. Render this as {context(index)}."
                for index in range(concept_count)
            ],
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": PROTOCOL,
        "purpose": "mechanism_diagnostic_not_production",
        "model_id": row["model_id"],
        "prompt_id": row["prompt_id"],
        "concept_manifest": row["concept_manifest"],
        "concept_manifest_sha256": row["concept_manifest_sha256"],
        "neutral_aggregation": "all_image_tokens",
        "concept_aggregation": "one_token_average_per_independent_prompt",
        "neutral": [neutral_prompt(index) for index in range(neutral_count)],
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--pair-id", action="append", required=True)
    parser.add_argument("--neutral-count", type=int, default=3)
    parser.add_argument("--concept-count", type=int, default=64)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite ensemble: {args.output}")
    row = read_row(args.manifest, args.index)
    payload = build(
        row,
        pair_ids=[str(value) for value in args.pair_id],
        neutral_count=args.neutral_count,
        concept_count=args.concept_count,
    )
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(encoded, encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
                "neutral_count": len(payload["neutral"]),
                "concept_count_per_side": {
                    pair_id: len(sides["source"])
                    for pair_id, sides in payload["pairs"].items()
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
