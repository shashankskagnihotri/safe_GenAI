#!/usr/bin/env python3
"""Fail-closed generation-runtime preflight for the 22 July campaign."""

from __future__ import annotations

import argparse
from importlib import metadata
import json
from pathlib import Path
from typing import Any

from finer_detailing_environment_dispatch import (
    LTX_DIFFUSERS_REVISION,
    LTX_PIPELINE_SOURCE_SHA256,
    LTX_VIDEO_VAE_SOURCE_SHA256,
    MODEL_ENVIRONMENTS,
    WAN_FTFY_VERSION,
    _validate_active_conda_environment,
    _validate_diffusers_source_contract,
    validate_diffusers_install,
    validate_runtime_distribution,
)


EXPECTED_TORCH = "2.5.1"
EXPECTED_TORCHVISION = "0.20.1"
EXPECTED_TORCH_CUDA = "12.4"


def distribution_record(name: str, prefix: Path) -> dict[str, str]:
    distribution = metadata.distribution(name)
    location = Path(distribution.locate_file("")).resolve()
    try:
        location.relative_to(prefix)
    except ValueError as exc:
        raise RuntimeError(
            f"{name} resolves outside the authenticated environment: {location}"
        ) from exc
    return {"version": distribution.version, "location": str(location)}


def verify(model_id: str, expected_environment: str, require_cuda: bool) -> dict[str, Any]:
    if model_id not in MODEL_ENVIRONMENTS:
        raise ValueError(f"No frozen environment contract for model {model_id!r}")
    contract = MODEL_ENVIRONMENTS[model_id]
    if contract.name != expected_environment:
        raise RuntimeError(
            f"Model {model_id!r} belongs to {contract.name!r}, not "
            f"{expected_environment!r}"
        )
    active = _validate_active_conda_environment(contract)
    prefix = Path(active["prefix"])
    diffusers = validate_diffusers_install(contract)
    torch_record = validate_runtime_distribution(
        "torch", EXPECTED_TORCH, prefix=prefix
    )
    torchvision_record = validate_runtime_distribution(
        "torchvision", EXPECTED_TORCHVISION, prefix=prefix
    )
    ftfy_record = validate_runtime_distribution(
        "ftfy", WAN_FTFY_VERSION, prefix=prefix
    )
    package_records = {
        "torch": torch_record,
        "torchvision": torchvision_record,
        "ftfy": ftfy_record,
        "transformers": distribution_record("transformers", prefix),
        "diffusers": distribution_record("diffusers", prefix),
    }

    import torch
    import torchvision
    from transformers import AutoProcessor

    del torchvision, AutoProcessor
    if torch.version.cuda != EXPECTED_TORCH_CUDA:
        raise RuntimeError(
            f"Wrong PyTorch CUDA build: expected {EXPECTED_TORCH_CUDA!r}, "
            f"got {torch.version.cuda!r}"
        )
    if require_cuda and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable inside the allocated GPU job")

    source_contract = None
    if contract.ltx_source_contract:
        source_contract = _validate_diffusers_source_contract(
            {
                "diffusers_revision": LTX_DIFFUSERS_REVISION,
                "pipeline_source_sha256": LTX_PIPELINE_SOURCE_SHA256,
                "video_vae_source_sha256": LTX_VIDEO_VAE_SOURCE_SHA256,
            }
        )
    return {
        "schema_version": 1,
        "status": "generation_environment_verified",
        "model_id": model_id,
        "environment": active,
        "diffusers": diffusers,
        "packages": package_records,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "cuda_required": require_cuda,
        "source_contract": source_contract,
        "scope": "generation_only; temporal evaluation runtime is validated separately",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--expected-environment", required=True)
    parser.add_argument("--require-cuda", action="store_true")
    args = parser.parse_args()
    print(
        json.dumps(
            verify(args.model_id, args.expected_environment, args.require_cuda),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
