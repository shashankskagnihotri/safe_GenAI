#!/usr/bin/env python3
"""Compile prompt-agnostic global-v2 concept pairs for the July-22 runner."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
GLOBAL_ROOT = ROOT / "configs/concepts/global_v2"
REGISTRY_PATH = GLOBAL_ROOT / "registry.yaml"
COMPATIBILITY_PATH = GLOBAL_ROOT / "compatibility_hierarchy.yaml"

PAIR_GROUPS = {
    "human": (
        "facial_affect_negative_to_happy",
        "body_pose_sitting_to_walking",
        "clothing_color_green_to_red_blue",
        "sandwich_action_eating_to_holding",
        "composition_static_to_dynamic",
    ),
    "mall": (
        "merchandise_handbags_to_cars",
        "signage_sale_to_new_arrival",
        "vertical_circulation_escalators_to_marble_stairs",
        "horizontal_floor_marble_to_tile",
        "sky_color_blue_to_pink",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")
    return value


def resolve_registered_path(entry: dict[str, Any]) -> Path:
    raw = entry.get("path", entry.get("file"))
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"Registry entry {entry.get('id')!r} has no path/file")
    declared = Path(raw)
    if declared.is_absolute():
        return declared
    root_relative = ROOT / declared
    registry_relative = REGISTRY_PATH.parent / declared
    if root_relative.is_file():
        return root_relative
    if registry_relative.is_file():
        return registry_relative
    raise FileNotFoundError(f"Registered concept definition does not exist: {raw}")


def compile_manifest(group: str) -> dict[str, Any]:
    registry = load_yaml(REGISTRY_PATH)
    if int(registry.get("schema_version", -1)) != 2:
        raise ValueError("Global ontology registry must use schema_version 2")
    entries = registry.get("pairs")
    if not isinstance(entries, list):
        raise TypeError("Global ontology registry pairs must be a list")
    by_id = {str(entry["id"]): entry for entry in entries}

    compatibility = load_yaml(COMPATIBILITY_PATH)
    neutral = compatibility.get("neutral_concept")
    if not isinstance(neutral, str) or not neutral.strip():
        raise ValueError("Compatibility hierarchy has no global neutral_concept")

    compiled_pairs = []
    for pair_id in PAIR_GROUPS[group]:
        if pair_id not in by_id:
            raise KeyError(f"Global registry has no pair {pair_id!r}")
        entry = by_id[pair_id]
        definition_path = resolve_registered_path(entry)
        definition_sha = sha256(definition_path)
        expected_sha = entry.get("sha256")
        if expected_sha is not None and definition_sha != expected_sha:
            raise ValueError(
                f"Registry hash mismatch for {pair_id}: "
                f"expected {expected_sha!r}, observed {definition_sha}"
            )
        definition = load_yaml(definition_path)
        if definition.get("id") != pair_id:
            raise ValueError(f"Definition ID mismatch in {definition_path}")
        source = definition.get("source")
        target = definition.get("target")
        if not isinstance(source, dict) or not isinstance(target, dict):
            raise TypeError(f"Pair {pair_id} must define source and target mappings")
        source_prompt = source.get("prompt")
        target_prompt = target.get("prompt")
        if not isinstance(source_prompt, str) or not isinstance(target_prompt, str):
            raise TypeError(f"Pair {pair_id} source/target prompts must be strings")
        if source_prompt == target_prompt:
            raise ValueError(f"Pair {pair_id} has identical source and target prompts")

        definition_priority = definition.get("priority")
        registry_priority = entry.get("priority")
        if (
            registry_priority is not None
            and float(registry_priority) != float(definition_priority)
        ):
            raise ValueError(f"Registry priority mismatch for {pair_id}")
        registry_parent = entry.get("parent")
        definition_parent = definition.get("parent")
        if registry_parent is not None and registry_parent != definition_parent:
            raise ValueError(f"Registry parent mismatch for {pair_id}")

        compiled_pairs.append(
            {
                "id": pair_id,
                "parent": definition_parent,
                "unsafe_concept": source_prompt,
                "safe_sibling_concept": target_prompt,
                "target_concept": target_prompt,
                "priority": float(definition_priority),
                "description": (
                    f"Global ontology transition from {source.get('label')} "
                    f"to {target.get('label')}."
                ),
                "intended_effect": target_prompt,
                "global_definition_path": str(definition_path.relative_to(ROOT)),
                "global_definition_sha256": definition_sha,
            }
        )

    return {
        "schema_version": 2,
        "name": f"global_v2_{group}_attribute_set",
        "ontology_scope": "global_prompt_agnostic",
        "ontology_registry": str(REGISTRY_PATH.relative_to(ROOT)),
        "ontology_registry_sha256": sha256(REGISTRY_PATH),
        "compatibility_hierarchy": str(COMPATIBILITY_PATH.relative_to(ROOT)),
        "compatibility_hierarchy_sha256": sha256(COMPATIBILITY_PATH),
        "neutral_concept": neutral,
        "pairs": compiled_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=tuple(PAIR_GROUPS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest = compile_manifest(args.group)
    output.write_text(
        yaml.safe_dump(manifest, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(f"{output}\t{sha256(output)}\t{len(manifest['pairs'])} pairs")


if __name__ == "__main__":
    main()
