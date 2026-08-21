#!/usr/bin/env python3
"""Audit one completed schema-v2 paired-reference shard from its event log."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
from typing import Any

import torch


FATAL_LOG_PATTERN = re.compile(
    r"traceback|out of memory|cuda error|exception:", re.IGNORECASE
)


def tensor_sha256(value: torch.Tensor) -> str:
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def event_rows(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if value.get("event") == "switch_reference_shard_completed":
                events.append(value)
    return events


def parse_shape(value: str | None) -> tuple[int, ...] | None:
    if value is None:
        return None
    shape = tuple(int(part) for part in value.split(","))
    if not shape or any(dimension <= 0 for dimension in shape):
        raise ValueError(f"Invalid expected shape: {value!r}")
    return shape


def audit(args: argparse.Namespace) -> dict[str, Any]:
    stdout_log = Path(args.stdout_log).resolve()
    stderr_log = (
        Path(args.stderr_log).resolve()
        if args.stderr_log
        else stdout_log.with_suffix(".err")
    )
    events = event_rows(stdout_log)
    if len(events) != args.pair_count:
        raise RuntimeError(
            f"Expected {args.pair_count} completed-pair events, found {len(events)}"
        )

    global_indices = [int(event["index"]) for event in events]
    local_indices = {
        index % args.pair_population_per_side for index in global_indices
    }
    if len(local_indices) != 1:
        raise RuntimeError(f"Events disagree on local pair index: {global_indices}")
    local_index = next(iter(local_indices))
    strata = sorted(
        index // args.pair_population_per_side for index in global_indices
    )
    if strata != list(range(args.pair_count)):
        raise RuntimeError(f"Pair strata are incomplete or duplicated: {strata}")

    expected_shape = parse_shape(args.expected_shape)
    pair_ids: list[str] = []
    tensor_hashes: list[str] = []
    pair_results: list[dict[str, Any]] = []
    role_count = 0

    for event in events:
        artifact_path = Path(event["path"]).resolve()
        payload = torch.load(artifact_path, map_location="cpu", weights_only=False)
        if payload.get("schema_version") != 2:
            raise RuntimeError(f"Wrong schema in {artifact_path}")
        if payload.get("construction") != (
            "full_paired_source_target_conditioned_trajectory"
        ):
            raise RuntimeError(f"Wrong construction in {artifact_path}")
        expected_identity = {
            "model_id": args.model_id,
            "prompt_id": args.prompt_id,
            "seed": int(event["seed"]),
            "pair_id": str(event["pair_id"]),
            "pair_sample_index": local_index,
        }
        for key, expected in expected_identity.items():
            if payload.get(key) != expected:
                raise RuntimeError(
                    f"{artifact_path}: {key}={payload.get(key)!r}, "
                    f"expected {expected!r}"
                )
        if payload.get("source_prompt_sha256") == payload.get(
            "target_prompt_sha256"
        ):
            raise RuntimeError(f"Source and target prompt hashes match: {artifact_path}")

        references = payload.get("references")
        if not isinstance(references, dict) or set(references) != {
            "source",
            "target",
        }:
            raise RuntimeError(f"Invalid reference sides: {artifact_path}")
        source = references["source"]
        target = references["target"]
        if not isinstance(source, dict) or not source or set(source) != set(target):
            raise RuntimeError(f"Source/target roles do not match: {artifact_path}")

        role_results = []
        for role in sorted(source):
            source_tensor = source[role]
            target_tensor = target[role]
            if not torch.is_tensor(source_tensor) or not torch.is_tensor(target_tensor):
                raise RuntimeError(f"Non-tensor reference role {role}: {artifact_path}")
            if source_tensor.shape != target_tensor.shape:
                raise RuntimeError(f"Source/target shape mismatch: {artifact_path}")
            if expected_shape is not None and tuple(source_tensor.shape) != expected_shape:
                raise RuntimeError(
                    f"{artifact_path}: shape {tuple(source_tensor.shape)}, "
                    f"expected {expected_shape}"
                )
            if source_tensor.dtype != torch.float32 or target_tensor.dtype != torch.float32:
                raise RuntimeError(f"Non-float32 reference tensor: {artifact_path}")
            if not torch.isfinite(source_tensor).all() or not torch.isfinite(
                target_tensor
            ).all():
                raise RuntimeError(f"Non-finite reference tensor: {artifact_path}")
            if torch.equal(source_tensor, target_tensor):
                raise RuntimeError(f"Identical source/target tensor: {artifact_path}")
            source_hash = tensor_sha256(source_tensor)
            target_hash = tensor_sha256(target_tensor)
            tensor_hashes.extend((source_hash, target_hash))
            role_count += 1
            role_results.append(
                {
                    "role": role,
                    "shape": list(source_tensor.shape),
                    "source_sha256": source_hash,
                    "target_sha256": target_hash,
                    "source_target_rmse": float(
                        torch.sqrt(
                            torch.mean(
                                (source_tensor.float() - target_tensor.float()) ** 2
                            )
                        )
                    ),
                }
            )

        pair_id = str(payload["pair_id"])
        pair_ids.append(pair_id)
        pair_results.append(
            {
                "pair_id": pair_id,
                "seed": int(payload["seed"]),
                "global_index": int(event["index"]),
                "local_pair_index": local_index,
                "artifact": str(artifact_path),
                "artifact_sha256": file_sha256(artifact_path),
                "roles": role_results,
            }
        )

    if len(set(pair_ids)) != args.pair_count:
        raise RuntimeError(f"Pair IDs are incomplete or duplicated: {pair_ids}")
    if len(set(tensor_hashes)) != len(tensor_hashes):
        raise RuntimeError("Reference tensor hashes are not globally unique in the shard")

    fatal_matches: list[dict[str, Any]] = []
    for log_path in (stdout_log, stderr_log):
        if not log_path.is_file():
            raise FileNotFoundError(log_path)
        with log_path.open("r", encoding="utf-8", errors="replace") as handle:
            for line_number, line in enumerate(handle, start=1):
                if FATAL_LOG_PATTERN.search(line):
                    fatal_matches.append(
                        {
                            "path": str(log_path),
                            "line": line_number,
                            "text": line.rstrip(),
                        }
                    )
    if fatal_matches:
        raise RuntimeError(f"Fatal log signatures found: {fatal_matches}")

    return {
        "status": "passed",
        "schema_version": 2,
        "model_id": args.model_id,
        "prompt_id": args.prompt_id,
        "local_pair_index": local_index,
        "global_indices": sorted(global_indices),
        "pair_count": len(pair_ids),
        "role_count": role_count,
        "tensor_count": len(tensor_hashes),
        "unique_tensor_count": len(set(tensor_hashes)),
        "fatal_log_matches": 0,
        "pairs": sorted(pair_results, key=lambda value: value["global_index"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stdout-log", required=True)
    parser.add_argument("--stderr-log")
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--prompt-id", required=True)
    parser.add_argument("--pair-count", type=int, default=5)
    parser.add_argument("--pair-population-per-side", type=int, default=103)
    parser.add_argument("--expected-shape")
    args = parser.parse_args()
    print(json.dumps(audit(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
