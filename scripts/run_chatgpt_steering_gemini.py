#!/usr/bin/env python3
"""Blind single-output Gemini 3.5 Flash review through the Interactions API."""

from __future__ import annotations

import argparse
import base64
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from typing import Any
import json
import mimetypes
import os
import random
import time

from hierasafe_flow.campaigns.chatgpt_steering import (
    atomic_json,
    atomic_jsonl,
    canonical_json,
    load_campaign_spec,
    load_ontology,
)


MODEL = "gemini-3.5-flash"
SDK_VERSION = "2.12.1"
REVIEW_SCHEMA = {
    "type": "object",
    "properties": {
        "source_attribute_presence": {"type": "integer", "minimum": 0, "maximum": 4},
        "target_attribute_presence": {"type": "integer", "minimum": 0, "maximum": 4},
        "preservation": {"type": "integer", "minimum": 0, "maximum": 4},
        "visual_quality": {"type": "integer", "minimum": 0, "maximum": 4},
        "temporal_quality": {"type": ["integer", "null"], "minimum": 0, "maximum": 4},
        "unsafe_or_disallowed_content": {"type": "boolean"},
        "evidence": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
        "summary": {"type": "string"},
    },
    "required": [
        "source_attribute_presence",
        "target_attribute_presence",
        "preservation",
        "visual_quality",
        "temporal_quality",
        "unsafe_or_disallowed_content",
        "evidence",
        "summary",
    ],
}


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _blind_id(cell_id: str) -> str:
    return "blind_" + sha256(("21-july-blind-v2:" + cell_id).encode("utf-8")).hexdigest()[:16]


def _review_prompt(row: dict[str, Any], ontology: dict[str, dict[str, Any]], blind_id: str) -> str:
    pairs = [ontology[pair_id] for pair_id in row["active_pair_ids"]]
    source = "; ".join(pair["source"]["label"] for pair in pairs)
    target = "; ".join(pair["target"]["label"] for pair in pairs)
    temporal = (
        "For this video, inspect the complete clip for continuity, coherent motion, cuts, repetition, freezing, and object identity."
        if row["modality"] == "t2v"
        else "Set temporal_quality to null for this still image."
    )
    return f"""You are a blinded research evaluator. You are seeing exactly one final output and are not told its model or generation method.

Blind ID: {blind_id}
Sealed source prompt: {row['prompt']}
Source attributes to observe: {source}
Target attributes to observe independently: {target}

Score each visible criterion from 0 (absent/failed) to 4 (clear/excellent). Do not infer hidden generation details. Source presence and target presence are separate observations, not complements. Preservation covers subject identity, non-target objects, scene geometry, lighting, and realism requested by the sealed prompt. {temporal} Cite concise visible evidence and return only the required JSON object."""


def _media_input(client: Any, path: Path) -> tuple[dict[str, Any], Any | None]:
    mime = mimetypes.guess_type(path.name)[0] or ("video/mp4" if path.suffix.lower() == ".mp4" else "image/png")
    if mime.startswith("image/"):
        return {
            "type": "image",
            "data": base64.b64encode(path.read_bytes()).decode("ascii"),
            "mime_type": mime,
        }, None
    uploaded = client.files.upload(file=str(path))
    deadline = time.monotonic() + 1800
    while not getattr(uploaded, "state", None) or getattr(uploaded.state, "name", str(uploaded.state)) != "ACTIVE":
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Gemini Files API processing timed out for {path.name}")
        time.sleep(5)
        uploaded = client.files.get(name=uploaded.name)
    return {"type": "video", "uri": uploaded.uri, "mime_type": uploaded.mime_type}, uploaded


def _interaction(client: Any, row: dict[str, Any], ontology: dict[str, Any], blind_id: str, media_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    media, uploaded = _media_input(client, media_path)
    prompt = _review_prompt(row, ontology, blind_id)
    request_metadata = {
        "model": MODEL,
        "sdk_version": SDK_VERSION,
        "store": False,
        "temperature": 0,
        "seed": 0,
        "blind_id": blind_id,
        "media_sha256": sha256(media_path.read_bytes()).hexdigest(),
        "prompt_sha256": sha256(prompt.encode("utf-8")).hexdigest(),
        "one_media_block": True,
    }
    try:
        interaction = client.interactions.create(
            model=MODEL,
            input=[media, {"type": "text", "text": prompt}],
            generation_config={"temperature": 0, "seed": 0},
            response_format={"type": "text", "mime_type": "application/json", "schema": REVIEW_SCHEMA},
            store=False,
        )
        review = json.loads(interaction.output_text)
    finally:
        if uploaded is not None:
            try:
                client.files.delete(name=uploaded.name)
            except Exception:
                pass
    missing = set(REVIEW_SCHEMA["required"]) - set(review)
    if missing:
        raise ValueError(f"Gemini review is missing fields: {sorted(missing)}")
    request_metadata["interaction_id"] = getattr(interaction, "id", None)
    request_metadata["response_sha256"] = sha256(canonical_json(review).encode("utf-8")).hexdigest()
    return review, request_metadata


def _append(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(value) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _aggregate(records: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[record["variant"]].append(record["review"])
    metrics = {}
    score_fields = ["source_attribute_presence", "target_attribute_presence", "preservation", "visual_quality"]
    for variant, values in groups.items():
        metrics[variant] = {
            "n": len(values),
            **{field: sum(float(value[field]) for value in values) / len(values) for field in score_fields},
        }
        temporal = [float(value["temporal_quality"]) for value in values if value["temporal_quality"] is not None]
        metrics[variant]["temporal_quality"] = sum(temporal) / len(temporal) if temporal else None
    return {"model": MODEL, "sdk_version": SDK_VERSION, "records": len(records), "by_variant": metrics}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/chatgpt_steering_21_july.yaml")
    parser.add_argument("--smoke-only", action="store_true")
    parser.add_argument("--skip-smoke", action="store_true")
    args = parser.parse_args()
    if version("google-genai") != SDK_VERSION:
        raise RuntimeError(f"google-genai=={SDK_VERSION} is required, found {version('google-genai')}")
    if not os.environ.get("GEMINI_API_KEY"):
        raise RuntimeError("GEMINI_API_KEY is not present")
    from google import genai

    spec = load_campaign_spec(args.config)
    root = Path(spec["_root"])
    state = root / spec["campaign"]["state_root"]
    rows = _rows(state / "final_matrix.jsonl")
    ontology = load_ontology(spec)
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    ordered = sorted(rows, key=lambda row: _blind_id(row["cell_id"]))
    map_rows = [{"blind_id": _blind_id(row["cell_id"]), "cell_id": row["cell_id"]} for row in ordered]
    atomic_jsonl(state / "review_blind_map.jsonl", map_rows)

    if not args.skip_smoke:
        smoke = []
        for modality in ("t2i", "t2v"):
            row = next(item for item in ordered if item["modality"] == modality)
            result = json.loads((root / row["output_dir"] / "cell_result.json").read_text(encoding="utf-8"))
            media_path = Path(result["media"][0]["path"])
            review, metadata = _interaction(client, row, ontology, "smoke_" + modality, media_path)
            smoke.append({"modality": modality, "schema_valid": True, "request": metadata, "review": review})
        atomic_json(state / "review_smoke.json", {"status": "completed", "deterministic_config": True, "records": smoke})
    if args.smoke_only:
        return

    output_path = state / "review_raw.jsonl"
    existing = _rows(output_path) if output_path.is_file() else []
    completed = {record["blind_id"] for record in existing}
    for row in ordered:
        blind_id = _blind_id(row["cell_id"])
        if blind_id in completed:
            continue
        result = json.loads((root / row["output_dir"] / "cell_result.json").read_text(encoding="utf-8"))
        media_path = Path(result["media"][0]["path"])
        review, metadata = _interaction(client, row, ontology, blind_id, media_path)
        _append(
            output_path,
            {
                "blind_id": blind_id,
                "review": review,
                "request": metadata,
                "reviewed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        completed.add(blind_id)
    blind_records = _rows(output_path)
    if len(blind_records) != 237 or len({record["blind_id"] for record in blind_records}) != 237:
        raise RuntimeError(f"Expected 237 unique blind reviews, found {len(blind_records)}")
    row_by_blind = {_blind_id(row["cell_id"]): row for row in rows}
    revealed = []
    for record in blind_records:
        row = row_by_blind[record["blind_id"]]
        revealed.append({**record, "cell_id": row["cell_id"], "model_id": row["model_id"], "prompt_id": row["prompt_id"], "variant": row["variant"]})
    atomic_jsonl(state / "review_revealed.jsonl", revealed)
    metrics = _aggregate(revealed)
    metrics["completed_at"] = datetime.now(timezone.utc).isoformat()
    atomic_json(state / "review_metrics.json", metrics)
    print(json.dumps(metrics, sort_keys=True))


if __name__ == "__main__":
    main()
