from __future__ import annotations

from pathlib import Path

from hierasafe_flow.generation.runner import GenerationRunner
from hierasafe_flow.utils.config import deep_merge, load_config


def test_full_dummy_generation_runner(tmp_path: Path) -> None:
    config = load_config("configs/experiments/smoke_t2i.yaml", project_root=Path.cwd())
    config = deep_merge(
        config,
        {
            "project": {"seed": 7},
            "runtime": {"device": "cpu", "dtype": "float32"},
            "logging": {"output_dir": str(tmp_path), "tensorboard": False},
            "generation": {"prompt_file": None, "prompt": "a clean smoke prompt", "num_inference_steps": 2},
        },
    )
    result = GenerationRunner(config).run()
    assert len(result.records) == 1
    sample_dir = tmp_path / "sample_0000"
    assert (sample_dir / "final_latents.pt").exists()
    assert (sample_dir / "steering_trace.json").exists()
    assert (sample_dir / "report.json").exists()
    assert (sample_dir / "decoded_tensor.pt").exists()
