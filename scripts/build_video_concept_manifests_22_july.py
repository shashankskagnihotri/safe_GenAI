#!/usr/bin/env python3
"""Build video-only concept manifests with explicit temporal preservation anchors."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "configs/concepts/video_22_july"

SPECS = {
    "01_sad_young_girl": {
        "source": "configs/concepts/historical_17_july/finer_detailing_01_sad_young_girl.yaml",
        "output": "finer_detailing_01_sad_young_girl_t2v.yaml",
        "anchor": (
            "Only one young woman appears in a continuous wide head-to-toe tracking shot in a park; "
            "her face, the front of her jacket, and her feet, the complete bench, and the same sandwich "
            "remain visible throughout."
        ),
    },
    "02_angry_old_man": {
        "source": "configs/concepts/historical_17_july/finer_detailing_02_angry_old_man.yaml",
        "output": "finer_detailing_02_angry_old_man_t2v.yaml",
        "anchor": (
            "Same old man, full body, feet and named objects remain visible; no other person, "
            "cut, crop, exit, identity swap or object loss."
        ),
    },
    "03_empty_outdoor_mall": {
        "source": "configs/concepts/historical_17_july/finer_detailing_03_empty_outdoor_mall_t2v.yaml",
        "output": "finer_detailing_03_empty_outdoor_mall_t2v.yaml",
        "anchor": (
            "Same empty mall and named structures remain fully visible; no person, cut, crop, "
            "scene change or object loss."
        ),
    },
}


def append_anchor(value: Any, anchor: str, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Missing concept text for {field}")
    return f"{value.rstrip().rstrip('.')}. {anchor}"


def build_one(prompt_id: str, spec: dict[str, str]) -> dict[str, Any]:
    source = ROOT / spec["source"]
    payload = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Concept manifest must be a mapping: {source}")
    pairs = payload.get("pairs")
    if not isinstance(pairs, list) or len(pairs) != 5:
        raise ValueError(f"Expected five pairs in {source}")
    anchor = spec["anchor"]
    payload["name"] = f"{payload['name']}_video_continuity_22_july"
    payload["neutral_concept"] = append_anchor(
        payload["neutral_concept"], anchor, field=f"{prompt_id}.neutral_concept"
    )
    for pair in pairs:
        if prompt_id == "01_sad_young_girl" and pair.get("id") == "clothing_color_green_to_red_blue":
            invariant_suffix = str(pair["unsafe_concept"]).partition(";")[2].strip()
            if not invariant_suffix:
                raise ValueError("Color pair must preserve its semicolon-delimited invariant suffix")
            unsafe_prefix = "green fabric on the outer jacket"
            safe_prefix = "red and royal-blue fabric on the outer jacket"
            pair["unsafe_concept"] = f"{unsafe_prefix}; {invariant_suffix}"
            pair["safe_sibling_concept"] = f"{safe_prefix}; {invariant_suffix}"
            pair["target_concept"] = f"{safe_prefix}; {invariant_suffix}"
        for field in ("unsafe_concept", "safe_sibling_concept", "target_concept"):
            if field in pair:
                pair[field] = append_anchor(
                    pair[field], anchor, field=f"{prompt_id}.{pair.get('id')}.{field}"
                )
    video_conditioning = {
        "schema_version": 1,
        "source_manifest": spec["source"],
        "source_sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
        "temporal_anchor": anchor,
        "composition_mode": "concept_only_with_temporal_preservation",
    }
    if prompt_id == "01_sad_young_girl":
        video_conditioning["prompt_01_color_binding"] = "outer_jacket_shell_contrast_v1"
    payload["video_conditioning_22_july"] = video_conditioning
    target = OUTPUT_ROOT / spec["output"]
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = yaml.safe_dump(payload, sort_keys=False, allow_unicode=False, width=120)
    target.write_text(rendered, encoding="utf-8")
    return {
        "prompt_id": prompt_id,
        "source": str(source.relative_to(ROOT)),
        "source_sha256": payload["video_conditioning_22_july"]["source_sha256"],
        "output": str(target.relative_to(ROOT)),
        "output_sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
        "pairs": len(pairs),
    }


def main() -> None:
    records = [build_one(prompt_id, spec) for prompt_id, spec in SPECS.items()]
    print(json.dumps({"schema_version": 1, "manifests": records}, sort_keys=True))


if __name__ == "__main__":
    main()
