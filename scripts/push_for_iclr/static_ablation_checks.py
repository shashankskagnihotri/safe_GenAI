#!/usr/bin/env python3
"""CPU-only Stage-2 algebra, controller, and matrix checks."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hierasafe_flow.campaigns.push_for_iclr.ablation_controller import (  # noqa: E402
    ABLATION_IDS,
    AblationController,
    AblationMode,
    apply_relative_direction,
    build_unified_time_map,
    compute_no_neutral_direction,
    conceptsteer_direction,
    mask_config_from_mapping,
    normalize_direction_rms,
    tensor_rms,
)


def main() -> int:
    config = yaml.safe_load(
        (REPOSITORY_ROOT / "configs/experiments/push_for_iclr/ablation_stage2_v1.yaml").read_text()
    )
    assert tuple(config["ablations"]) == ABLATION_IDS
    assert len(config["models"]) * len(ABLATION_IDS) * config["source_manifest"]["row_count"] == 780
    assert config["slurm"]["array_throttle"] is None
    assert config["slurm"]["user_hold"] is False

    timesteps = torch.linspace(1000.0, 0.0, 28)
    time_map = build_unified_time_map(timesteps)
    active_early = sum(800.0 <= value <= 1000.0 for value in time_map.unified_times)
    expected_recomputations = {
        AblationMode.BASELINE: 0,
        AblationMode.GENERIC_SUFFIX: 0,
        AblationMode.CONFIG_SUFFIX: 0,
        AblationMode.NO_NEUTRAL_AR: 28,
        AblationMode.PULSE_K0: 1,
        AblationMode.PULSE_K2: 1,
        AblationMode.PULSE_K5: 1,
        AblationMode.FROZEN_K0: 1,
        AblationMode.FROZEN_K2: 1,
        AblationMode.FROZEN_K5: 1,
        AblationMode.EARLY_STOP: active_early,
        AblationMode.EARLY_FROZEN: active_early,
        AblationMode.ALL_RECOMPUTE: 28,
    }
    expected_applications = {
        AblationMode.BASELINE: 0,
        AblationMode.GENERIC_SUFFIX: 0,
        AblationMode.CONFIG_SUFFIX: 0,
        AblationMode.NO_NEUTRAL_AR: 28,
        AblationMode.PULSE_K0: 1,
        AblationMode.PULSE_K2: 1,
        AblationMode.PULSE_K5: 1,
        AblationMode.FROZEN_K0: 28,
        AblationMode.FROZEN_K2: 26,
        AblationMode.FROZEN_K5: 23,
        AblationMode.EARLY_STOP: active_early,
        AblationMode.EARLY_FROZEN: 28,
        AblationMode.ALL_RECOMPUTE: 28,
    }
    controller_counts = {}
    for mode in AblationMode:
        controller = AblationController(mode, time_map.unified_times)
        computes = sum(controller.should_compute_direction(i) for i in range(28))
        applies = sum(controller.should_apply_direction(i) for i in range(28))
        assert computes == expected_recomputations[mode], (mode, computes)
        assert applies == expected_applications[mode], (mode, applies)
        assert controller.requires_neutral is (controller.requires_concept_fields and mode is not AblationMode.NO_NEUTRAL_AR)
        controller_counts[mode.value] = {"recomputations": computes, "applications": applies}

    generator = torch.Generator(device="cpu").manual_seed(20260821)
    base = torch.randn((1, 8, 6, 6), generator=generator, dtype=torch.float32)
    unsafe = torch.randn((1, 8, 6, 6), generator=generator, dtype=torch.float32)
    safe = torch.randn((1, 8, 6, 6), generator=generator, dtype=torch.float32)
    neutral = torch.randn((1, 8, 6, 6), generator=generator, dtype=torch.float32)
    mask = mask_config_from_mapping(config["models"]["flux1_dev"]["mask"])
    direction, metadata = conceptsteer_direction(
        v_base=base,
        v_unsafe=unsafe,
        v_safe=safe,
        v_neutral=neutral,
        feature_dim=1,
        margin=0.08,
        mask_config=mask,
    )
    assert metadata["neutral_used"] is True
    assert torch.isfinite(direction).all()
    no_neutral, no_neutral_metadata = compute_no_neutral_direction(
        base, unsafe, safe, feature_dim=1, margin=0.08, mask_config=mask
    )
    assert no_neutral_metadata["neutral_used"] is False
    assert torch.isfinite(no_neutral).all()
    assert (safe - unsafe).shape == direction.shape == no_neutral.shape

    unit = normalize_direction_rms(direction)
    base_bf16 = base.to(torch.bfloat16)
    identity = apply_relative_direction(base_bf16, unit, relative_strength=0.0)
    assert identity.dtype == base_bf16.dtype
    assert torch.equal(identity, base_bf16)
    steered = apply_relative_direction(base_bf16, unit, relative_strength=0.18)
    delta = steered.float() - base_bf16.float()
    relative = float(tensor_rms(delta, eps=0.0) / tensor_rms(base_bf16))
    assert abs(relative - 0.18) < 0.01, relative
    assert torch.isfinite(steered).all()

    result = {
        "status": "PASS",
        "matrix_cells": 780,
        "ablation_count": len(ABLATION_IDS),
        "early_active_steps_on_28_step_reference": active_early,
        "controller_counts": controller_counts,
        "zero_strength_exact_identity": True,
        "relative_rms_observed": relative,
        "no_generator_loaded": True,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
