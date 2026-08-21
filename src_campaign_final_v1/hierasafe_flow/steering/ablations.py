from __future__ import annotations

from copy import deepcopy
from typing import Any

from hierasafe_flow.utils.config import deep_merge


def expand_ablations(base_config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    ablations = base_config.get("ablations") or []
    if not ablations:
        return [("base", deepcopy(base_config))]

    expanded: list[tuple[str, dict[str, Any]]] = []
    for item in ablations:
        if not isinstance(item, dict) or "name" not in item:
            raise ValueError("Each ablation must be a mapping with a 'name' field.")
        override = {k: v for k, v in item.items() if k != "name"}
        expanded.append((str(item["name"]), deep_merge(base_config, override)))
    return expanded

