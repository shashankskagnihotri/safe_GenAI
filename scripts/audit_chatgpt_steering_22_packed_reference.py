#!/usr/bin/env python3
"""Independent, one-pass audit for a packed CHS22 switch-reference artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import torch


PAIR_IDS = {
    "facial_affect_negative_to_happy",
    "body_pose_sitting_to_walking",
    "clothing_color_green_to_red_blue",
    "sandwich_action_eating_to_holding",
    "composition_static_to_dynamic",
}
ROLE = "primary::segment_000"
SAMPLE_SHAPE = (3952, 128)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def walk_tensors(value: Any, path: tuple[str, ...] = ()):
    if isinstance(value, torch.Tensor):
        yield path, value
        return
    if isinstance(value, dict):
        for key, child in value.items():
            yield from walk_tensors(child, path + (str(key),))
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from walk_tensors(child, path + (str(index),))


def walk_key_values(value: Any, key: str, path: tuple[str, ...] = ()):
    if isinstance(value, torch.Tensor):
        return
    if isinstance(value, dict):
        for child_key, child in value.items():
            child_path = path + (str(child_key),)
            if child_key == key:
                yield child_path, child
            yield from walk_key_values(child, key, child_path)
        return
    if isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            yield from walk_key_values(child, key, path + (str(index),))


def require_equal(payload: dict[str, Any], key: str, expected: Any) -> list[str]:
    matches = [
        "/".join(path)
        for path, actual in walk_key_values(payload, key)
        if actual == expected
    ]
    if not matches:
        observed = [
            {"path": "/".join(path), "value": actual}
            for path, actual in walk_key_values(payload, key)
        ]
        raise AssertionError(
            f"{key}: expected nested value {expected!r}; observed {observed!r}"
        )
    return matches


def classify_side(path: tuple[str, ...]) -> str:
    lowered = [component.lower() for component in path]
    source = any(component == "source" or component.endswith("_source") for component in lowered)
    target = any(component == "target" or component.endswith("_target") for component in lowered)
    if source == target:
        raise AssertionError(f"cannot uniquely classify tensor side at {'/'.join(path)}")
    return "source" if source else "target"


def audit(args: argparse.Namespace) -> dict[str, Any]:
    artifact = args.artifact.resolve()
    if not artifact.is_file():
        raise AssertionError(f"artifact is missing: {artifact}")

    artifact_sha256 = sha256_file(artifact)
    if artifact_sha256 != args.expected_sha256:
        raise AssertionError(
            f"artifact SHA-256 mismatch: expected {args.expected_sha256}, got {artifact_sha256}"
        )

    payload = torch.load(artifact, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise AssertionError(f"top-level payload must be a dict, got {type(payload).__name__}")

    metadata_paths: dict[str, list[str]] = {}
    metadata_paths["schema_version"] = require_equal(payload, "schema_version", 2)
    metadata_paths["model_id"] = require_equal(payload, "model_id", args.model_id)
    metadata_paths["prompt_id"] = require_equal(payload, "prompt_id", args.prompt_id)
    metadata_paths["construction"] = require_equal(
        payload, "construction", "full_paired_source_target_conditioned_trajectory"
    )
    metadata_paths["reference_dtype"] = require_equal(payload, "reference_dtype", "float32")
    metadata_paths["reference_population_per_side"] = require_equal(
        payload, "reference_population_per_side", 515
    )
    metadata_paths["reference_population_total"] = require_equal(
        payload, "reference_population_total", 1030
    )
    metadata_paths["terminal_sample"] = require_equal(
        payload,
        "terminal_sample",
        "canonical_predicted_x0_at_each_model_role_segment_end",
    )
    metadata_paths["trajectory_conditioning"] = require_equal(
        payload,
        "trajectory_conditioning",
        "paired_source_or_target_prompt_at_every_denoising_step",
    )
    metadata_paths["concept_manifest_sha256"] = require_equal(
        payload, "concept_manifest_sha256", args.concept_manifest_sha256
    )
    metadata_paths["model_config_sha256"] = require_equal(
        payload, "model_config_sha256", args.model_config_sha256
    )
    metadata_paths["reference_seeds"] = require_equal(
        payload, "reference_seeds", list(range(2_400_000, 2_400_515))
    )
    metadata_paths["pair_reference_counts_per_side"] = require_equal(
        payload,
        "pair_reference_counts_per_side",
        {pair_id: 103 for pair_id in PAIR_IDS},
    )
    metadata_paths["reference_population_by_role"] = require_equal(
        payload,
        "reference_population_by_role",
        {ROLE: {"source": 515, "target": 515}},
    )

    tensor_records = list(walk_tensors(payload))
    if not tensor_records:
        raise AssertionError("packed artifact contains no tensors")

    side_counts: Counter[str] = Counter()
    dtype_counts: Counter[str] = Counter()
    storage_shape_counts: Counter[str] = Counter()
    endpoint_hashes: set[str] = set()
    duplicate_hashes: list[str] = []
    tensor_paths: list[str] = []
    global_min = math.inf
    global_max = -math.inf

    for path, tensor in tensor_records:
        tensor_paths.append("/".join(path))
        side = classify_side(path)
        dtype_counts[str(tensor.dtype)] += 1
        storage_shape_counts[str(list(tensor.shape))] += 1

        if tensor.device.type != "cpu":
            raise AssertionError(f"tensor is not on CPU after map_location at {'/'.join(path)}")
        if tensor.dtype != torch.float32:
            raise AssertionError(
                f"expected float32 tensor at {'/'.join(path)}, got {tensor.dtype}"
            )
        if tensor.ndim < 3 or tuple(tensor.shape[-2:]) != SAMPLE_SHAPE:
            raise AssertionError(
                f"unexpected endpoint shape at {'/'.join(path)}: {list(tensor.shape)}"
            )
        if not bool(torch.isfinite(tensor).all().item()):
            raise AssertionError(f"non-finite values at {'/'.join(path)}")

        global_min = min(global_min, float(tensor.min().item()))
        global_max = max(global_max, float(tensor.max().item()))
        samples = tensor.contiguous().view(-1, *SAMPLE_SHAPE)
        side_counts[side] += int(samples.shape[0])
        for sample in samples:
            sample_sha256 = hashlib.sha256(sample.numpy().tobytes()).hexdigest()
            if sample_sha256 in endpoint_hashes:
                duplicate_hashes.append(sample_sha256)
            endpoint_hashes.add(sample_sha256)

    if dict(side_counts) != {"source": 515, "target": 515}:
        raise AssertionError(
            f"endpoint populations mismatch: expected source=515,target=515, got {dict(side_counts)}"
        )
    if duplicate_hashes:
        raise AssertionError(
            f"duplicate endpoint tensors detected: {len(duplicate_hashes)} duplicate hashes"
        )
    if len(endpoint_hashes) != 1030:
        raise AssertionError(
            f"expected 1030 unique endpoint hashes, got {len(endpoint_hashes)}"
        )

    return {
        "artifact": str(artifact),
        "artifact_bytes": artifact.stat().st_size,
        "artifact_sha256": artifact_sha256,
        "dtype_counts": dict(dtype_counts),
        "endpoint_population_by_side": dict(side_counts),
        "global_max": global_max,
        "global_min": global_min,
        "metadata_paths": metadata_paths,
        "model_id": args.model_id,
        "prompt_id": args.prompt_id,
        "schema_version": 2,
        "status": "passed",
        "storage_shape_counts": dict(storage_shape_counts),
        "tensor_leaf_count": len(tensor_records),
        "tensor_paths": tensor_paths,
        "unique_endpoint_tensor_count": len(endpoint_hashes),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--concept-manifest-sha256", required=True)
    parser.add_argument("--model-config-sha256", required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-id", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = audit(args)
    except Exception as exc:
        print(json.dumps({"error": str(exc), "status": "failed"}, sort_keys=True))
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
