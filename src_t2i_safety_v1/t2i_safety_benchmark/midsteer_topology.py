from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch

from hierasafe_flow.steering.related_work.midsteer_attn_output import (
    attention_output_sites,
)

from .contracts import PROJECT_ROOT, file_sha256


TOPOLOGY_PROTOCOL = "t2isafety_midsteer_model_topology_admission_v1"
TOPOLOGY_ROOT = (
    PROJECT_ROOT
    / "debugging"
    / "t2i_safety_27_july"
    / "validation"
    / "midsteer_topology_admission_v1"
)


def _project_file(value: Any, *, field: str) -> Path:
    raw = Path(str(value))
    path = raw.resolve() if raw.is_absolute() else (PROJECT_ROOT / raw).resolve()
    if not path.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"{field} must remain inside PROJECT_ROOT")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


@lru_cache(maxsize=None)
def load_midsteer_topology_binding(model_id: str) -> dict[str, Any]:
    aggregate_path = TOPOLOGY_ROOT / "ADMISSION.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    expected_aggregate = {
        "protocol": TOPOLOGY_PROTOCOL,
        "status": "accepted",
        "model_count": 6,
        "site_count": 257,
    }
    aggregate_mismatches = {
        key: (aggregate.get(key), expected)
        for key, expected in expected_aggregate.items()
        if aggregate.get(key) != expected
    }
    if aggregate_mismatches:
        raise RuntimeError(
            f"MidSteer topology aggregate mismatch: {aggregate_mismatches}"
        )
    records = [
        item
        for item in aggregate.get("admissions", [])
        if item.get("model_id") == model_id
    ]
    if len(records) != 1:
        raise RuntimeError(
            f"Expected one admitted MidSteer topology for {model_id}, got {len(records)}"
        )
    record = records[0]
    admission_path = _project_file(record.get("path"), field="topology_admission")
    admission_sha256 = file_sha256(admission_path)
    if admission_sha256 != record.get("sha256"):
        raise RuntimeError(f"MidSteer topology admission changed for {model_id}")
    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    expected_admission = {
        "protocol": TOPOLOGY_PROTOCOL,
        "status": "accepted",
        "model_id": model_id,
        "topology_sha256": record.get("topology_sha256"),
        "site_contract_sha256": record.get("site_contract_sha256"),
    }
    admission_mismatches = {
        key: (admission.get(key), expected)
        for key, expected in expected_admission.items()
        if admission.get(key) != expected
    }
    if admission_mismatches:
        raise RuntimeError(
            f"MidSteer topology admission mismatch for {model_id}: "
            f"{admission_mismatches}"
        )
    site_contract = admission.get("site_contract")
    if not isinstance(site_contract, dict):
        raise RuntimeError(f"MidSteer topology lacks a site contract for {model_id}")
    groups = site_contract.get("groups")
    ordered_sites = site_contract.get("ordered_sites")
    if not isinstance(groups, list) or not isinstance(ordered_sites, list):
        raise RuntimeError(f"Malformed MidSteer site contract for {model_id}")
    counts = {
        int(group["image_token_count"])
        for group in groups
        if isinstance(group, dict)
    }
    policies = {
        str(group["image_token_policy"])
        for group in groups
        if isinstance(group, dict)
    }
    if len(counts) != 1 or not policies.issubset(
        {"entire_primary_output", "suffix_after_text_prefix"}
    ):
        raise RuntimeError(f"Invalid MidSteer image-token contract for {model_id}")
    if int(site_contract.get("total_site_count", -1)) != len(ordered_sites):
        raise RuntimeError(f"MidSteer topology site count changed for {model_id}")
    return {
        "model_id": model_id,
        "model_revision": admission["model_revision"],
        "adapter": admission["adapter"],
        "root_class_name": admission["root_class_name"],
        "latent_shape": list(admission["latent_shape"]),
        "prediction_shape": list(admission["prediction_shape"]),
        "ordered_sites": list(ordered_sites),
        "image_token_count": counts.pop(),
        "topology_sha256": admission["topology_sha256"],
        "site_contract_sha256": admission["site_contract_sha256"],
        "topology_admission_path": str(admission_path),
        "topology_admission_sha256": admission_sha256,
        "topology_aggregate_path": str(aggregate_path),
        "topology_aggregate_sha256": file_sha256(aggregate_path),
    }


def validate_midsteer_topology(
    *,
    model_id: str,
    adapter: Any,
    root: torch.nn.Module,
) -> dict[str, Any]:
    binding = load_midsteer_topology_binding(model_id)
    observed_identity = {
        "model_revision": adapter.config.get("revision"),
        "adapter": adapter.adapter_name,
        "root_class_name": root.__class__.__name__,
        "ordered_sites": [site.name for site in attention_output_sites(root)],
    }
    mismatches = {
        key: (observed, binding[key])
        for key, observed in observed_identity.items()
        if observed != binding[key]
    }
    if mismatches:
        raise RuntimeError(
            f"Live MidSteer topology differs from admission for {model_id}: {mismatches}"
        )
    return binding


def midsteer_image_token_count(
    binding: dict[str, Any],
    latents: torch.Tensor,
) -> int:
    observed = list(latents.shape)
    if observed != binding["latent_shape"]:
        raise RuntimeError(
            f"MidSteer latent topology changed for {binding['model_id']}: "
            f"{observed} != {binding['latent_shape']}"
        )
    return int(binding["image_token_count"])


__all__ = [
    "load_midsteer_topology_binding",
    "midsteer_image_token_count",
    "validate_midsteer_topology",
]
