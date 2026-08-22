#!/usr/bin/env python3
"""CPU-only algebra and frozen-contract checks for the trust-region redesign."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
import yaml

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

from hierasafe_flow.campaigns.push_for_iclr.ablation_controller import mask_config_from_mapping  # noqa: E402
from hierasafe_flow.campaigns.push_for_iclr.trust_region_controller import (  # noqa: E402
    TrustRegionArm,
    bounded_trust_region_delta,
    contextualize_probe,
    matched_pair_direction,
    project_semantic_component,
    raised_cosine_window,
    route_pair_directions,
)


def main() -> int:
    config = yaml.safe_load(
        (
            REPOSITORY_ROOT
            / "configs/experiments/push_for_iclr/trust_region_development_v1.yaml"
        ).read_text()
    )
    assert config["source_manifest"]["required_split"] == "development"
    assert "locked_validation" not in config["source_manifest"]["path"]
    assert config["slurm"]["array_throttle"] is None
    assert config["slurm"]["user_hold"] is False
    arms = [TrustRegionArm.from_mapping(value) for value in config["arms"]]
    assert len(arms) == 12
    assert len(config["models"]) * len(arms) * config["source_manifest"]["row_count"] == 480

    active_counts = {}
    for arm in arms:
        weights = [
            raised_cosine_window(
                index,
                28,
                start_fraction=arm.start_fraction,
                end_fraction=arm.end_fraction,
            )
            if arm.enabled
            else 0.0
            for index in range(28)
        ]
        if arm.enabled:
            assert 0.0 < max(weights) <= 1.0
        active_counts[arm.arm_id] = sum(weight > 0.0 for weight in weights)

    prompt = "A samurai in a moonlit courtyard"
    composed = contextualize_probe(prompt, "no visible blood or injury")
    assert composed.startswith("Safety concept:")
    assert composed.endswith(prompt)

    generator = torch.Generator(device="cpu").manual_seed(20260822)
    base = torch.randn((1, 8, 6, 6), generator=generator)
    neutral = torch.randn((1, 8, 6, 6), generator=generator)
    source = torch.randn((1, 8, 6, 6), generator=generator)
    target = torch.randn((1, 8, 6, 6), generator=generator)
    mask = mask_config_from_mapping(config["models"]["flux1_dev"]["mask"])
    pair = matched_pair_direction(
        pair_id="test_pair",
        parent="test_parent",
        v_base=base,
        v_source=source,
        v_target=target,
        v_neutral=neutral,
        feature_dim=1,
        margin=0.08,
        mask_config=mask,
        activation_top_fraction=0.10,
    )
    routed, routing = route_pair_directions(
        [pair],
        base=base,
        top_k_pairs=1,
        minimum_pair_score=0.0,
        routing_temperature=0.05,
    )
    assert routing["selected_pairs"] == ["test_pair"]
    projected, projection = project_semantic_component(
        routed,
        base - neutral,
        feature_dim=1,
        coefficient=1.0,
    )
    assert projection["mean_abs_cosine_after"] <= projection["mean_abs_cosine_before"] + 1e-5
    delta, trust = bounded_trust_region_delta(
        base=base,
        direction=projected,
        feature_dim=1,
        schedule_weight=0.8,
        max_local_relative=0.04,
        remaining_energy=0.0005,
    )
    assert torch.isfinite(delta).all()
    assert trust["maximum_observed_local_relative"] <= 0.032 + 2e-5
    assert trust["energy_after_budget_scale"] <= 0.0005 + 2e-7
    identity, identity_meta = bounded_trust_region_delta(
        base=base,
        direction=projected,
        feature_dim=1,
        schedule_weight=0.0,
        max_local_relative=0.04,
        remaining_energy=0.1,
    )
    assert torch.equal(identity, torch.zeros_like(base))
    assert identity_meta["energy_after_budget_scale"] == 0.0

    ontologies = {}
    for category, relative in config["ontologies"].items():
        ontology = yaml.safe_load((REPOSITORY_ROOT / relative).read_text())
        pair_ids = [pair_value["id"] for pair_value in ontology["runtime_probe_pairs"]]
        assert ontology["schema_version"] == 2
        assert ontology["category"] == category
        assert ontology["prompt_independent"] is True
        assert len(pair_ids) == len(set(pair_ids))
        assert ontology["neutral_probes"]
        ontologies[category] = len(pair_ids)

    print(
        json.dumps(
            {
                "status": "PASS",
                "matrix_cells": 480,
                "arm_count": len(arms),
                "active_steps_on_28_step_reference": active_counts,
                "ontology_pair_counts": ontologies,
                "context_conditioning_preserves_original_prompt": True,
                "semantic_projection_reduces_parallel_component": True,
                "local_cap_respected": True,
                "cumulative_energy_budget_respected": True,
                "gpu_or_generator_loaded": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
