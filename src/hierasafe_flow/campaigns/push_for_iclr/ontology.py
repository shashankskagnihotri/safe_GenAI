from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RuntimeProbe:
    probe_id: str
    text: str
    weight: float


@dataclass(frozen=True)
class GlobalOntology:
    path: Path
    sha256: str
    schema_version: int
    ontology_id: str
    category: str
    canonical_negative_prompt: str
    canonical_positive_prompt: str
    canonical_neutral_prompt: str
    generic_positive_suffix: str
    config_positive_suffix: str
    negative_probes: tuple[RuntimeProbe, ...]
    positive_probes: tuple[RuntimeProbe, ...]
    neutral_probes: tuple[RuntimeProbe, ...]
    preservation_axes: tuple[str, ...]
    raw: dict[str, Any]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Ontology field {key!r} must be a non-empty string.")
    return value


def _parse_probes(data: dict[str, Any], group: str) -> tuple[RuntimeProbe, ...]:
    groups = data.get("runtime_probe_groups")
    if not isinstance(groups, dict):
        raise ValueError("runtime_probe_groups must be an object.")
    rows = groups.get(group)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Runtime probe group {group!r} must be a non-empty list.")
    probes: list[RuntimeProbe] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Probe {group}[{index}] must be an object.")
        probe_id = _require_string(row, "id")
        if probe_id in seen:
            raise ValueError(f"Duplicate runtime probe id: {probe_id}")
        seen.add(probe_id)
        text = _require_string(row, "text")
        weight = float(row.get("weight", 0.0))
        if not weight > 0.0:
            raise ValueError(f"Probe {probe_id} has non-positive weight {weight}.")
        probes.append(RuntimeProbe(probe_id=probe_id, text=text, weight=weight))
    return tuple(probes)


def _validate_structure(data: dict[str, Any]) -> None:
    required = {
        "schema_version",
        "ontology_id",
        "category",
        "prompt_independent",
        "context_independent",
        "canonical_negative_prompt",
        "canonical_positive_prompt",
        "canonical_neutral_prompt",
        "generic_positive_suffix",
        "config_positive_suffix",
        "parents",
        "leaves",
        "runtime_probe_groups",
        "boundary_cases",
        "preservation_axes",
    }
    missing = sorted(required - set(data))
    if missing:
        raise ValueError(f"Ontology is missing required fields: {missing}")
    if data["schema_version"] != 1:
        raise ValueError("Only global ontology schema_version=1 is admitted.")
    if data["prompt_independent"] is not True or data["context_independent"] is not True:
        raise ValueError("The proposed ontology must be prompt- and context-independent.")
    if data["category"] not in {"nudity", "violence", "combined_nudity_violence"}:
        raise ValueError(f"Unsupported global category: {data['category']!r}")
    parents = data["parents"]
    leaves = data["leaves"]
    if not isinstance(parents, list) or len(parents) < 2:
        raise ValueError("Ontology requires at least two parent concepts.")
    if not isinstance(leaves, list) or len(leaves) < 12:
        raise ValueError("Ontology requires at least twelve explicit leaves.")
    parent_ids = {_require_string(row, "id") for row in parents}
    leaf_ids: set[str] = set()
    polarities: set[str] = set()
    for row in leaves:
        leaf_id = _require_string(row, "id")
        if leaf_id in leaf_ids:
            raise ValueError(f"Duplicate leaf id: {leaf_id}")
        leaf_ids.add(leaf_id)
        parent_id = _require_string(row, "parent_id")
        if parent_id not in parent_ids:
            raise ValueError(f"Leaf {leaf_id} references unknown parent {parent_id}.")
        polarity = _require_string(row, "polarity")
        if polarity not in {"unsafe", "safe", "boundary"}:
            raise ValueError(f"Leaf {leaf_id} has invalid polarity {polarity}.")
        polarities.add(polarity)
        _require_string(row, "text")
        _require_string(row, "description")
    if polarities != {"unsafe", "safe", "boundary"}:
        raise ValueError(f"Ontology must contain unsafe, safe, and boundary leaves; got {polarities}.")
    for parent in parents:
        children = parent.get("children")
        if not isinstance(children, list) or not children:
            raise ValueError(f"Parent {parent['id']} has no children.")
        unknown = sorted(set(children) - leaf_ids)
        if unknown:
            raise ValueError(f"Parent {parent['id']} references unknown leaves: {unknown}")
    axes = data["preservation_axes"]
    if not isinstance(axes, list) or len(axes) < 10 or len(axes) != len(set(axes)):
        raise ValueError("preservation_axes must contain at least ten unique values.")
    boundaries = data["boundary_cases"]
    if not isinstance(boundaries, list) or len(boundaries) < 3:
        raise ValueError("At least three explicit boundary cases are required.")


def load_global_ontology(path: str | Path) -> GlobalOntology:
    resolved = Path(path).expanduser().resolve()
    raw_bytes = resolved.read_bytes()
    data = yaml.safe_load(raw_bytes)
    if not isinstance(data, dict):
        raise ValueError(f"Ontology must decode to an object: {resolved}")
    _validate_structure(data)
    negative = _parse_probes(data, "negative")
    positive = _parse_probes(data, "positive")
    neutral = _parse_probes(data, "neutral")
    all_ids = [p.probe_id for p in (*negative, *positive, *neutral)]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Runtime probe IDs must be globally unique within an ontology.")
    return GlobalOntology(
        path=resolved,
        sha256=hashlib.sha256(raw_bytes).hexdigest(),
        schema_version=int(data["schema_version"]),
        ontology_id=_require_string(data, "ontology_id"),
        category=_require_string(data, "category"),
        canonical_negative_prompt=_require_string(data, "canonical_negative_prompt"),
        canonical_positive_prompt=_require_string(data, "canonical_positive_prompt"),
        canonical_neutral_prompt=_require_string(data, "canonical_neutral_prompt"),
        generic_positive_suffix=_require_string(data, "generic_positive_suffix"),
        config_positive_suffix=_require_string(data, "config_positive_suffix"),
        negative_probes=negative,
        positive_probes=positive,
        neutral_probes=neutral,
        preservation_axes=tuple(str(axis) for axis in data["preservation_axes"]),
        raw=json.loads(json.dumps(data, sort_keys=True)),
    )
