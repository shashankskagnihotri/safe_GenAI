from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


ConfigDict = dict[str, Any]


def project_root_from(path: str | Path | None = None) -> Path:
    if path is None:
        return Path.cwd()
    return Path(path).expanduser().resolve()


def load_yaml(path: str | Path) -> ConfigDict:
    yaml_path = Path(path)
    if not yaml_path.exists():
        raise FileNotFoundError(f"YAML config does not exist: {yaml_path}")
    with yaml_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"YAML config must contain a mapping at top level: {yaml_path}")
    return data


def deep_merge(base: ConfigDict, override: ConfigDict) -> ConfigDict:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _resolve_config_reference(reference: str | Path, project_root: Path, source_path: Path) -> Path:
    ref_path = Path(reference).expanduser()
    if ref_path.is_absolute():
        return ref_path
    project_candidate = project_root / ref_path
    if project_candidate.exists():
        return project_candidate
    return source_path.parent / ref_path


def load_config(path: str | Path, project_root: str | Path | None = None) -> ConfigDict:
    root = project_root_from(project_root)
    config_path = Path(path).expanduser()
    if not config_path.is_absolute():
        config_path = root / config_path
    config_path = config_path.resolve()
    config = load_yaml(config_path)

    merged: ConfigDict = {}
    base_config = config.pop("base_config", None)
    if base_config is not None:
        merged = deep_merge(
            merged,
            load_config(_resolve_config_reference(base_config, root, config_path), root),
        )

    model_config = config.pop("model_config", None)
    if model_config is not None:
        merged = deep_merge(
            merged,
            load_config(_resolve_config_reference(model_config, root, config_path), root),
        )

    concept_config = config.pop("concept_config", None)
    if concept_config is not None:
        concept_path = _resolve_config_reference(concept_config, root, config_path)
        config.setdefault("concepts", {})["hierarchy_path"] = str(concept_path)

    merged = deep_merge(merged, config)
    merged.setdefault("_meta", {})["config_path"] = str(config_path)
    merged["_meta"]["project_root"] = str(root)
    return merged


def parse_cli_overrides(overrides: list[str] | None) -> ConfigDict:
    result: ConfigDict = {}
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be key=value, got: {item}")
        dotted_key, raw_value = item.split("=", 1)
        value = yaml.safe_load(raw_value)
        cursor = result
        parts = dotted_key.split(".")
        for part in parts[:-1]:
            cursor = cursor.setdefault(part, {})
            if not isinstance(cursor, dict):
                raise ValueError(f"Override path crosses a non-mapping value: {dotted_key}")
        cursor[parts[-1]] = value
    return result


def apply_overrides(config: ConfigDict, overrides: list[str] | None) -> ConfigDict:
    return deep_merge(config, parse_cli_overrides(overrides))


def get_path(config: ConfigDict, dotted_path: str, default: Any = None) -> Any:
    cursor: Any = config
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return default
        cursor = cursor[part]
    return cursor


def read_prompt_file(path: str | Path) -> list[str]:
    prompt_path = Path(path)
    if not prompt_path.exists():
        raise FileNotFoundError(f"Prompt file does not exist: {prompt_path}")
    prompts = [
        line.strip()
        for line in prompt_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not prompts:
        raise ValueError(f"Prompt file contains no runnable prompts: {prompt_path}")
    return prompts


def collect_prompts(config: ConfigDict, explicit_prompt: str | None = None) -> list[str]:
    if explicit_prompt:
        return [explicit_prompt]
    prompt_file = get_path(config, "generation.prompt_file")
    if prompt_file:
        return read_prompt_file(prompt_file)
    prompt = get_path(config, "generation.prompt")
    if not prompt:
        raise ValueError("No prompt was supplied in CLI, generation.prompt, or generation.prompt_file.")
    return [str(prompt)]

