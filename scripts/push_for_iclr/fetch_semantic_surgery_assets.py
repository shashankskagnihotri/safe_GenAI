#!/usr/bin/env python3
"""Fetch and admit the exact model assets used by Semantic Surgery Table 2."""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def model_cache_name(repo_id: str) -> str:
    return "models--" + repo_id.replace("/", "--")


def snapshot_manifest(snapshot: Path, output: Path) -> dict[str, Any]:
    rows: list[tuple[str, int, str]] = []
    for path in sorted(snapshot.rglob("*")):
        if path.is_file():
            rows.append((path.relative_to(snapshot).as_posix(), path.stat().st_size, sha256_file(path)))
    with output.open("w", encoding="utf-8") as handle:
        for relative, size, digest in rows:
            handle.write(f"{digest}  {size}  {relative}\n")
    return {
        "snapshot": str(snapshot),
        "file_count": len(rows),
        "byte_count": sum(row[1] for row in rows),
        "manifest": str(output),
        "manifest_sha256": sha256_file(output),
    }


def offline_load_gate(cache_root: Path, contract: dict[str, Any]) -> None:
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["DIFFUSERS_OFFLINE"] = "1"

    from diffusers import AutoencoderKL, UNet2DConditionModel
    from diffusers.pipelines.stable_diffusion import StableDiffusionSafetyChecker
    from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer

    sd = contract["models"]["stable_diffusion"]["repo_id"]
    clip = contract["models"]["clip_text_encoder"]["repo_id"]
    loaders = [
        lambda: AutoencoderKL.from_pretrained(sd, subfolder="vae", cache_dir=cache_root, local_files_only=True),
        lambda: CLIPTokenizer.from_pretrained(clip, cache_dir=cache_root, local_files_only=True),
        lambda: CLIPTextModel.from_pretrained(clip, cache_dir=cache_root, local_files_only=True),
        lambda: UNet2DConditionModel.from_pretrained(sd, subfolder="unet", cache_dir=cache_root, local_files_only=True),
        lambda: CLIPFeatureExtractor.from_pretrained(sd, subfolder="feature_extractor", cache_dir=cache_root, local_files_only=True),
        lambda: StableDiffusionSafetyChecker.from_pretrained(sd, subfolder="safety_checker", cache_dir=cache_root, local_files_only=True),
    ]
    for load in loaders:
        value = load()
        del value
        gc.collect()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text(encoding="utf-8"))
    final = Path(contract["execution"]["asset_cache"])
    expected = {
        "sd_commit": contract["models"]["stable_diffusion"]["resolved_commit"],
        "clip_commit": contract["models"]["clip_text_encoder"]["resolved_commit"],
    }

    if (final / "ASSET_ADMISSION.json").is_file():
        admission = json.loads((final / "ASSET_ADMISSION.json").read_text(encoding="utf-8"))
        if admission.get("status") != "admitted" or any(admission.get(key) != value for key, value in expected.items()):
            raise RuntimeError("Existing Semantic Surgery asset admission does not match the contract")
        print(f"Exact asset cache already admitted: {final}")
        return 0
    if final.exists():
        raise RuntimeError(f"Refusing incomplete existing asset cache: {final}")

    job_id = os.environ.get("SLURM_JOB_ID", f"pid{os.getpid()}")
    temporary = final.parent / f".{final.name}.building_{job_id}"
    failed = final.parent / f"{final.name}.FAILED_{job_id}"
    if temporary.exists() or failed.exists():
        raise RuntimeError("Refusing to overwrite a previous build artifact")
    temporary.mkdir(parents=True)

    try:
        sd_contract = contract["models"]["stable_diffusion"]
        clip_contract = contract["models"]["clip_text_encoder"]
        sd_source = Path(sd_contract["source_cache_model_dir"])
        sd_target = temporary / model_cache_name(sd_contract["repo_id"])
        if not sd_source.is_dir():
            raise FileNotFoundError(sd_source)
        source_main = (sd_source / "refs" / "main").read_text(encoding="utf-8").strip()
        if source_main != sd_contract["resolved_commit"]:
            raise RuntimeError(f"SD 1.4 main ref mismatch: {source_main}")
        os.symlink(sd_source, sd_target, target_is_directory=True)

        os.environ.pop("HF_HUB_OFFLINE", None)
        os.environ.pop("TRANSFORMERS_OFFLINE", None)
        from huggingface_hub import snapshot_download

        clip_snapshot_result = Path(snapshot_download(
            repo_id=clip_contract["repo_id"],
            revision=clip_contract["resolved_commit"],
            cache_dir=temporary,
            max_workers=8,
        ))
        if clip_snapshot_result.name != clip_contract["resolved_commit"]:
            raise RuntimeError(f"CLIP revision mismatch: {clip_snapshot_result}")
        clip_model_root = temporary / model_cache_name(clip_contract["repo_id"])
        (clip_model_root / "refs").mkdir(exist_ok=True)
        (clip_model_root / "refs" / "main").write_text(clip_contract["resolved_commit"] + "\n", encoding="utf-8")

        sd_snapshot = sd_target / "snapshots" / sd_contract["resolved_commit"]
        clip_snapshot = clip_model_root / "snapshots" / clip_contract["resolved_commit"]
        if not sd_snapshot.is_dir() or not clip_snapshot.is_dir():
            raise RuntimeError("A required exact snapshot is missing")

        offline_load_gate(temporary, contract)
        sd_manifest = snapshot_manifest(sd_snapshot, temporary / "SD14_MAIN_SHA256.txt")
        clip_manifest = snapshot_manifest(clip_snapshot, temporary / "CLIP_L14_SHA256.txt")
        admission = {
            "schema_version": 1,
            "status": "admitted",
            "protocol": "semantic_surgery_table2_exact_assets_v1",
            "sd_repo_id": sd_contract["repo_id"],
            "sd_requested_revision": sd_contract["requested_revision"],
            "sd_commit": sd_contract["resolved_commit"],
            "clip_repo_id": clip_contract["repo_id"],
            "clip_requested_revision": clip_contract["requested_revision"],
            "clip_commit": clip_contract["resolved_commit"],
            "sd_manifest": sd_manifest,
            "clip_manifest": clip_manifest,
            "offline_author_load_contract_passed": True,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        write_json(temporary / "ASSET_ADMISSION.json", admission)
        temporary.rename(final)
    except BaseException:
        if temporary.exists():
            temporary.rename(failed)
        raise

    print(f"Admitted exact Semantic Surgery assets: {final}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
