from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ConceptPair:
    id: str
    parent: str
    unsafe_concept: str
    safe_sibling_concept: str
    description: str = ""


@dataclass(frozen=True)
class ConceptHierarchy:
    name: str
    neutral_concept: str
    pairs: tuple[ConceptPair, ...]

    @classmethod
    def from_dict(cls, data: dict) -> "ConceptHierarchy":
        if "name" not in data:
            raise ValueError("Concept hierarchy is missing required field 'name'.")
        neutral = data.get("neutral_concept")
        if not neutral:
            raise ValueError("Concept hierarchy is missing required field 'neutral_concept'.")
        raw_pairs = data.get("pairs")
        if not isinstance(raw_pairs, list) or not raw_pairs:
            raise ValueError("Concept hierarchy must define a non-empty 'pairs' list.")

        pairs: list[ConceptPair] = []
        seen_ids: set[str] = set()
        required = {"id", "parent", "unsafe_concept", "safe_sibling_concept"}
        for index, item in enumerate(raw_pairs):
            if not isinstance(item, dict):
                raise ValueError(f"Concept pair at index {index} must be a mapping.")
            missing = sorted(required - set(item))
            if missing:
                raise ValueError(f"Concept pair at index {index} is missing fields: {missing}")
            pair_id = str(item["id"])
            if pair_id in seen_ids:
                raise ValueError(f"Duplicate concept pair id: {pair_id}")
            seen_ids.add(pair_id)
            pairs.append(
                ConceptPair(
                    id=pair_id,
                    parent=str(item["parent"]),
                    unsafe_concept=str(item["unsafe_concept"]),
                    safe_sibling_concept=str(item["safe_sibling_concept"]),
                    description=str(item.get("description", "")),
                )
            )
        return cls(name=str(data["name"]), neutral_concept=str(neutral), pairs=tuple(pairs))

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "ConceptHierarchy":
        yaml_path = Path(path)
        if not yaml_path.exists():
            raise FileNotFoundError(f"Concept hierarchy file does not exist: {yaml_path}")
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Concept hierarchy must be a mapping: {yaml_path}")
        return cls.from_dict(data)

    def safe_sibling_for(self, unsafe_concept: str) -> str:
        for pair in self.pairs:
            if pair.unsafe_concept == unsafe_concept:
                return pair.safe_sibling_concept
        raise KeyError(f"No safe sibling registered for unsafe concept: {unsafe_concept}")

    def by_parent(self) -> dict[str, list[ConceptPair]]:
        grouped: dict[str, list[ConceptPair]] = {}
        for pair in self.pairs:
            grouped.setdefault(pair.parent, []).append(pair)
        return grouped


def compose_concept_prompt(prompt: str, concept: str) -> str:
    prompt = prompt.strip()
    concept = concept.strip()
    if not concept:
        return prompt
    if not prompt:
        return concept
    return f"{prompt}. Safety concept: {concept}."

