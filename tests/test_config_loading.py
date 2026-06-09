from __future__ import annotations

from pathlib import Path

from hierasafe_flow.utils.config import apply_overrides, get_path, load_config


def test_load_experiment_config_merges_base() -> None:
    config = load_config("configs/experiments/smoke_t2i.yaml", project_root=Path.cwd())
    assert get_path(config, "model.adapter") == "dummy"
    assert get_path(config, "steering.enabled") is True
    assert get_path(config, "generation.num_inference_steps") == 4
    assert Path(get_path(config, "concepts.hierarchy_path")).exists()


def test_cli_overrides_nested_value() -> None:
    config = load_config("configs/experiments/smoke_t2i.yaml", project_root=Path.cwd())
    updated = apply_overrides(config, ["steering.margin=0.2", "generation.num_inference_steps=2"])
    assert get_path(updated, "steering.margin") == 0.2
    assert get_path(updated, "generation.num_inference_steps") == 2

