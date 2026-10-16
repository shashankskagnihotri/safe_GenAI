from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

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


@dataclass(frozen=True)
class MirroredConceptNode:
    """One prompt-independent unsafe/safe node in a mirrored concept tree."""

    id: str
    level: int
    unsafe_concept: str
    safe_concept: str
    parent_id: str | None
    child_ids: tuple[str, ...]
    steerable: bool

    @property
    def is_leaf(self) -> bool:
        return not self.child_ids


@dataclass(frozen=True)
class MirroredConceptHierarchy:
    """Backward-compatible full-tree representation used by routed ConceptSteer."""

    name: str
    neutral_concept: str
    root_id: str
    nodes: dict[str, MirroredConceptNode]
    prompt_composition: str = "concept_only"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MirroredConceptHierarchy":
        required = {"name", "neutral_concept", "root_id", "nodes"}
        missing = sorted(required - set(data))
        if missing:
            raise ValueError(f"Mirrored hierarchy is missing fields: {missing}")
        raw_nodes = data["nodes"]
        if not isinstance(raw_nodes, list) or not raw_nodes:
            raise ValueError("Mirrored hierarchy nodes must be a non-empty list.")
        nodes: dict[str, MirroredConceptNode] = {}
        for index, raw in enumerate(raw_nodes):
            if not isinstance(raw, dict):
                raise ValueError(f"Mirrored node {index} must be a mapping.")
            node = MirroredConceptNode(
                id=str(raw["id"]),
                level=int(raw["level"]),
                unsafe_concept=str(raw["unsafe_concept"]),
                safe_concept=str(raw["safe_concept"]),
                parent_id=(None if raw.get("parent_id") is None else str(raw["parent_id"])),
                child_ids=tuple(str(value) for value in raw.get("child_ids", ())),
                steerable=bool(raw["steerable"]),
            )
            if node.id in nodes:
                raise ValueError(f"Duplicate mirrored node id: {node.id}")
            nodes[node.id] = node
        hierarchy = cls(
            name=str(data["name"]),
            neutral_concept=str(data["neutral_concept"]),
            root_id=str(data["root_id"]),
            nodes=nodes,
            prompt_composition=str(data.get("prompt_composition", "concept_only")),
        )
        hierarchy.validate()
        return hierarchy

    @classmethod
    def from_yaml_file(cls, path: str | Path) -> "MirroredConceptHierarchy":
        yaml_path = Path(path)
        if not yaml_path.is_file():
            raise FileNotFoundError(f"Mirrored hierarchy file does not exist: {yaml_path}")
        data = yaml.safe_load(yaml_path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError(f"Mirrored hierarchy must be a mapping: {yaml_path}")
        return cls.from_dict(data)

    def validate(self) -> None:
        if self.prompt_composition != "concept_only":
            raise ValueError("Mirrored hierarchy requires prompt_composition=concept_only")
        if self.neutral_concept != "neutral":
            raise ValueError("This campaign requires the exact neutral concept 'neutral'.")
        if self.root_id not in self.nodes:
            raise ValueError("Hierarchy root is missing")
        root = self.nodes[self.root_id]
        if root.level != 0 or root.parent_id is not None:
            raise ValueError("Hierarchy root must be the only level-0 node")
        if sum(node.level == 0 for node in self.nodes.values()) != 1:
            raise ValueError("Hierarchy must contain exactly one level-0 node")
        for node_id, node in self.nodes.items():
            if node.id != node_id:
                raise ValueError(f"Node key/id mismatch: {node_id}")
            if node.level < 0 or node.level > 4:
                raise ValueError(f"Invalid level for {node_id}: {node.level}")
            if not node.unsafe_concept.strip() or not node.safe_concept.strip():
                raise ValueError(f"Empty mirrored concept at {node_id}")
            if len(node.child_ids) != len(set(node.child_ids)):
                raise ValueError(f"Duplicate child at {node_id}")
            if node.parent_id is not None:
                parent = self.nodes.get(node.parent_id)
                if parent is None:
                    raise ValueError(f"Missing parent for {node_id}")
                if parent.level + 1 != node.level:
                    raise ValueError(f"Non-consecutive levels at {node_id}")
                if node_id not in parent.child_ids:
                    raise ValueError(f"Parent does not list child {node_id}")
            for child_id in node.child_ids:
                child = self.nodes.get(child_id)
                if child is None:
                    raise ValueError(f"Missing child {child_id} listed by {node_id}")
                if child.parent_id != node_id:
                    raise ValueError(f"Child {child_id} does not point back to {node_id}")
            if node.level == 4 and node.child_ids:
                raise ValueError(f"Level-4 node cannot have children: {node_id}")
            if node.steerable != (node.level == 4 and node.is_leaf):
                raise ValueError(f"Only Level-4 leaves may steer: {node_id}")

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> None:
            if node_id in visiting:
                raise ValueError(f"Hierarchy contains a cycle at {node_id}")
            if node_id in visited:
                return
            visiting.add(node_id)
            for child_id in self.nodes[node_id].child_ids:
                visit(child_id)
            visiting.remove(node_id)
            visited.add(node_id)

        visit(self.root_id)
        unreachable = sorted(set(self.nodes) - visited)
        if unreachable:
            raise ValueError(f"Hierarchy contains unreachable nodes: {unreachable}")

    def traversal_order(self) -> tuple[str, ...]:
        result: list[str] = []

        def visit(node_id: str) -> None:
            result.append(node_id)
            for child_id in self.nodes[node_id].child_ids:
                visit(child_id)

        visit(self.root_id)
        return tuple(result)


def compose_concept_prompt(prompt: str, concept: str) -> str:
    prompt = prompt.strip()
    concept = concept.strip()
    if not concept:
        return prompt
    if not prompt:
        return concept
    return f"{prompt}. Safety concept: {concept}."
