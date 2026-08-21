from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from PIL import Image

from hierasafe_flow.adapters.registry import create_adapter
from hierasafe_flow.steering.canonical.contract import infer_channel_dim
from hierasafe_flow.utils.config import load_config
from hierasafe_flow.utils.device import configure_cuda, resolve_device, resolve_dtype
from hierasafe_flow.utils.seed import make_generator, seed_everything

from .contracts import (
    CALIBRATION_ROOT,
    PROJECT_ROOT,
    WORK_ROOT,
    BenchmarkContract,
    atomic_json,
    canonical_sha256,
    file_sha256,
)
from .exact_related_work import tensor_sha256


CORPUS_REPO = WORK_ROOT / "upstream" / "repos" / "t2i_safety_dataset"
CORPUS_DATA = WORK_ROOT / "upstream" / "data" / "t2i_safety_dataset"
REFERENCE_MANIFEST_ROOT = WORK_ROOT / "manifests" / "unsafe_references"
CORPUS_REVISION = "35cc5d6dc9f3c71c51cf8f81fae88ab493862af5"

CATEGORY_LABELS = {
    "sexual": "sexual",
    "hate": "hate",
    "humiliation": "humiliation",
    "violence": "violence",
    "illegal_activity": "illegal activity",
    "disturbing": "disturbing",
    "public_figures": "public figures",
    "personal_identification_documents": "personal identification documents",
    "intellectual_property_violation": "intellectual property violation",
}
EFFECTIVE_REFERENCE_POPULATION = 515


def _assistant_text(record: dict[str, Any]) -> str:
    for turn in record.get("conversations", []):
        if str(turn.get("from", "")).lower() in {"assistant", "gpt"}:
            return str(turn.get("value", "")).strip().lower()
    return ""


def _assistant_labels(record: dict[str, Any]) -> set[str]:
    recognized = set(CATEGORY_LABELS.values())
    labels: set[str] = set()
    for line in _assistant_text(record).splitlines():
        normalized = line.strip().lower()
        if not normalized or normalized in {"safe", "unsafe"}:
            continue
        for value in normalized.split(" and "):
            candidate = value.strip()
            if candidate in recognized:
                labels.add(candidate)
    return labels


def _image_path(record: dict[str, Any]) -> str:
    images = record.get("image")
    if not isinstance(images, list) or len(images) != 1:
        raise RuntimeError("Reference record must contain exactly one image path.")
    return str(images[0])


def _records(filename: str) -> list[dict[str, Any]]:
    with (CORPUS_REPO / filename).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, list):
        raise RuntimeError(f"Corpus metadata {filename} is not a list.")
    return value


def build_category_manifests() -> None:
    source_files = (
        "hf_train_toxicity_privacy_generated.json",
        "hf_train_toxicity_privacy_real.json",
    )
    all_records: list[dict[str, Any]] = []
    for filename in source_files:
        for record in _records(filename):
            all_records.append({"source_metadata": filename, **record})
    summary: dict[str, Any] = {}
    for category, label in CATEGORY_LABELS.items():
        candidates: list[dict[str, Any]] = []
        seen_paths: set[str] = set()
        seen_image_sha256: set[str] = set()
        excluded_duplicate_content_count = 0
        for record in all_records:
            if label not in _assistant_labels(record):
                continue
            relative = _image_path(record)
            if relative in seen_paths:
                continue
            absolute = CORPUS_DATA / relative
            if not absolute.is_file():
                raise FileNotFoundError(f"Corpus image is missing: {absolute}")
            seen_paths.add(relative)
            image_sha256 = file_sha256(absolute)
            if image_sha256 in seen_image_sha256:
                excluded_duplicate_content_count += 1
                continue
            seen_image_sha256.add(image_sha256)
            identity = {
                "category": category,
                "source_metadata": record["source_metadata"],
                "source_id": record.get("id"),
                "image": relative,
                "image_sha256": image_sha256,
                "assistant_annotation": _assistant_text(record),
                "real": bool(record.get("real", False)),
                "corpus_revision": CORPUS_REVISION,
            }
            identity["selection_key"] = canonical_sha256(identity)
            candidates.append(identity)
        candidates.sort(key=lambda item: item["selection_key"])
        if len(candidates) < 2:
            raise RuntimeError(
                f"Category {category} has fewer than two unique train references."
            )
        selected = candidates[:EFFECTIVE_REFERENCE_POPULATION]
        path = REFERENCE_MANIFEST_ROOT / f"{category}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite sealed reference manifest {path}")
        with path.open("w", encoding="utf-8") as handle:
            for item in selected:
                handle.write(
                    json.dumps(
                        item,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                    + "\n"
                )
        summary[category] = {
            "candidate_count": len(candidates),
            "candidate_unique_image_sha256_count": len(seen_image_sha256),
            "excluded_duplicate_content_count": excluded_duplicate_content_count,
            "selected_count": len(selected),
            "effective_reference_population": len(selected),
            "empirical_mass_multiplier": 1.0,
            "manifest": str(path),
            "manifest_sha256": file_sha256(path),
        }
    atomic_json(REFERENCE_MANIFEST_ROOT / "summary.json", summary)


def build_fairness_manifest(model_id: str, profile_path: Path) -> None:
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if profile.get("model_id") != model_id:
        raise RuntimeError("Fairness profile model identity mismatch.")
    labels = {
        str(value).strip().lower()
        for value in profile.get("overrepresented_demographics", [])
    }
    if not labels:
        raise RuntimeError("Fairness profile has no overrepresented demographics.")
    source_files = ("hf_train_fairness_generated.json", "hf_train_fairness_real.json")
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    seen_image_sha256: set[str] = set()
    for filename in source_files:
        for record in _records(filename):
            answer = _assistant_text(record)
            values = {
                line.split(":", 1)[1].strip()
                for line in answer.splitlines()
                if ":" in line
            }
            if not values.intersection(labels):
                continue
            relative = _image_path(record)
            if relative in seen:
                continue
            absolute = CORPUS_DATA / relative
            if not absolute.is_file():
                raise FileNotFoundError(absolute)
            seen.add(relative)
            image_sha256 = file_sha256(absolute)
            if image_sha256 in seen_image_sha256:
                continue
            seen_image_sha256.add(image_sha256)
            item = {
                "category": "fairness",
                "model_id": model_id,
                "source_metadata": filename,
                "source_id": record.get("id"),
                "image": relative,
                "image_sha256": image_sha256,
                "assistant_annotation": answer,
                "matched_overrepresented_demographics": sorted(
                    values.intersection(labels)
                ),
                "fairness_profile_sha256": file_sha256(profile_path),
                "corpus_revision": CORPUS_REVISION,
            }
            item["selection_key"] = canonical_sha256(item)
            candidates.append(item)
    candidates.sort(key=lambda item: item["selection_key"])
    if len(candidates) < 2:
        raise RuntimeError(
            f"Fairness profile for {model_id} yields fewer than two references."
        )
    target = REFERENCE_MANIFEST_ROOT / model_id / "fairness.jsonl"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    with target.open("w", encoding="utf-8") as handle:
        for item in candidates[:EFFECTIVE_REFERENCE_POPULATION]:
            handle.write(
                json.dumps(
                    item,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                )
                + "\n"
            )


def _latent_mode(encoded: Any) -> torch.Tensor:
    distribution = getattr(encoded, "latent_dist", None)
    if distribution is not None:
        if hasattr(distribution, "mode"):
            return distribution.mode()
        if hasattr(distribution, "sample"):
            return distribution.sample()
    if hasattr(encoded, "latents"):
        return encoded.latents
    if isinstance(encoded, tuple) and encoded and isinstance(encoded[0], torch.Tensor):
        return encoded[0]
    raise RuntimeError(f"Unsupported VAE encode output {type(encoded).__name__}")


def _preprocess_image(pipe: Any, image: Image.Image, height: int, width: int) -> torch.Tensor:
    value = pipe.image_processor.preprocess(image, height=height, width=width)
    vae_device = next(pipe.vae.parameters()).device
    return value.to(device=vae_device, dtype=pipe.vae.dtype)


def encode_reference(
    adapter: Any,
    model_id: str,
    image: Image.Image,
    *,
    height: int,
    width: int,
    generator: torch.Generator,
) -> torch.Tensor:
    pipe = adapter.pipeline
    if pipe is None:
        raise RuntimeError("Reference encoding requires a loaded pipeline.")
    if model_id == "cosmos3_super_text2image":
        video = pipe.video_processor.preprocess_video(
            [image], height=height, width=width
        ).to(device=adapter.device, dtype=pipe.vae.dtype)
        latent = _latent_mode(pipe.vae.encode(video))
        mean = pipe._vae_latents_mean.to(latent.device, latent.dtype).view(
            1, -1, 1, 1, 1
        )
        inv_std = pipe._vae_latents_inv_std.to(latent.device, latent.dtype).view(
            1, -1, 1, 1, 1
        )
        return ((latent - mean) * inv_std).to(dtype=adapter.dtype)

    if model_id == "flux2_dev":
        # Mirror Flux2Pipeline.prepare_image_latents exactly.  Flux2 uses
        # sequential CPU offload, so VAE parameters are meta-resident between
        # calls and cannot identify a tensor destination.  The native pipeline
        # deliberately stages image tensors on its Accelerate execution device.
        execution_device = torch.device(pipe._execution_device)
        if execution_device.type == "meta":
            raise RuntimeError("Flux2 native execution device resolved to meta.")
        pixels = pipe.image_processor.preprocess(
            image.convert("RGB"), height=height, width=width
        ).to(device=execution_device, dtype=pipe.vae.dtype)
        latent = pipe._encode_vae_image(pixels, generator)
        return pipe._pack_latents(latent).to(dtype=adapter.dtype)
    pixels = _preprocess_image(pipe, image.convert("RGB"), height, width)
    latent = _latent_mode(pipe.vae.encode(
        pixels.unsqueeze(2) if model_id.startswith("qwen_image") else pixels
    ))
    if model_id == "flux1_dev":
        shift = float(pipe.vae.config.shift_factor)
        scale = float(pipe.vae.config.scaling_factor)
        latent = (latent - shift) * scale
        return pipe._pack_latents(
            latent,
            latent.shape[0],
            latent.shape[1],
            latent.shape[2],
            latent.shape[3],
        ).to(dtype=adapter.dtype)
    if model_id.startswith("qwen_image"):
        mean = torch.tensor(pipe.vae.config.latents_mean).view(
            1, pipe.vae.config.z_dim, 1, 1, 1
        ).to(latent)
        inv_std = (
            1.0
            / torch.tensor(pipe.vae.config.latents_std).view(
                1, pipe.vae.config.z_dim, 1, 1, 1
            )
        ).to(latent)
        latent = (latent - mean) * inv_std
        return pipe._pack_latents(
            latent,
            latent.shape[0],
            latent.shape[1],
            latent.shape[-2],
            latent.shape[-1],
        ).to(dtype=adapter.dtype)
    if model_id == "ideogram4_nf4":
        patch = int(pipe.patch_size)
        batch, channels, latent_h, latent_w = latent.shape
        latent = latent.view(
            batch,
            channels,
            latent_h // patch,
            patch,
            latent_w // patch,
            patch,
        )
        latent = latent.permute(0, 2, 4, 3, 5, 1).contiguous()
        latent = latent.view(
            batch,
            (latent_h // patch) * (latent_w // patch),
            channels * patch * patch,
        )
        mean = pipe.vae.bn.running_mean.view(1, 1, -1).to(latent)
        std = torch.sqrt(
            pipe.vae.bn.running_var + pipe.vae.config.batch_norm_eps
        ).view(1, 1, -1).to(latent)
        return ((latent - mean) / std).to(dtype=torch.float32)
    if model_id == "sd35_large":
        shift = float(getattr(pipe.vae.config, "shift_factor", 0.0))
        scale = float(pipe.vae.config.scaling_factor)
        return ((latent - shift) * scale).to(dtype=adapter.dtype)
    raise ValueError(f"No audited reference encoder for {model_id}")


def _reference_manifest(model_id: str, category: str) -> Path:
    if category == "fairness":
        return REFERENCE_MANIFEST_ROOT / model_id / "fairness.jsonl"
    return REFERENCE_MANIFEST_ROOT / f"{category}.jsonl"


def _kernel_pairwise_squared_distances(
    kernel_references: torch.Tensor,
    *,
    device: torch.device,
    chunk_size: int = 4,
) -> torch.Tensor:
    flat = kernel_references.to(device=device, dtype=torch.float32).reshape(
        kernel_references.shape[0], -1
    )
    norms = flat.square().sum(dim=1)
    result = torch.empty(
        (flat.shape[0], flat.shape[0]),
        dtype=torch.float32,
        device="cpu",
    )
    with torch.inference_mode():
        for start in range(0, flat.shape[0], chunk_size):
            stop = min(start + chunk_size, flat.shape[0])
            distances = (
                norms[start:stop].unsqueeze(1)
                + norms.unsqueeze(0)
                - 2.0 * (flat[start:stop] @ flat.T)
            ).clamp_min_(0.0)
            result[start:stop].copy_(distances.cpu())
    result = 0.5 * (result + result.T)
    result.fill_diagonal_(0.0)
    if not bool(torch.isfinite(result).all()):
        raise RuntimeError("Safe Denoiser kernel distances are non-finite.")
    return result


def encode_bank(model_id: str, category: str) -> None:
    contract = BenchmarkContract()
    model_spec = contract.model(model_id)
    manifest = _reference_manifest(model_id, category)
    if not manifest.is_file():
        raise FileNotFoundError(manifest)
    records = [
        json.loads(line)
        for line in manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 2 <= len(records) <= EFFECTIVE_REFERENCE_POPULATION:
        raise RuntimeError(
            "Reference manifest must contain between 2 and "
            f"{EFFECTIVE_REFERENCE_POPULATION} unique records; observed {len(records)}."
        )

    seed_everything(0)
    configure_cuda(True)
    device = resolve_device("auto")
    dtype = resolve_dtype("bfloat16")
    config = load_config(model_spec["config"], project_root=PROJECT_ROOT)
    model_values = dict(config["model"])
    model_values["height"] = int(model_spec["height"])
    model_values["width"] = int(model_spec["width"])
    model_values["guidance_scale"] = float(model_spec["guidance_scale"])
    adapter = create_adapter(model_values, device, dtype)
    adapter.load()
    generator = make_generator(0, device)
    encoded: list[torch.Tensor] = []
    with torch.inference_mode():
        for record in records:
            image_path = CORPUS_DATA / record["image"]
            with Image.open(image_path) as image:
                latent = encode_reference(
                    adapter,
                    model_id,
                    image,
                    height=int(model_spec["height"]),
                    width=int(model_spec["width"]),
                    generator=generator,
                )
            if latent.shape[0] != 1:
                raise RuntimeError("Reference encoder returned a non-unit batch.")
            if encoded and latent.shape[1:] != encoded[0].shape[1:]:
                raise RuntimeError("Reference latent shapes changed within one bank.")
            encoded.append(latent.detach().to(device="cpu", dtype=torch.float32))
    references = torch.cat(encoded, dim=0)

    probe_generator = make_generator(0, device)
    probe, _ = adapter.prepare_initial_latents(
        prompt="reference shape admission probe",
        batch_size=1,
        generator=probe_generator,
        task="text_to_image",
        height=int(model_spec["height"]),
        width=int(model_spec["width"]),
        num_inference_steps=int(model_spec["steps"]),
    )
    if tuple(references.shape[1:]) != tuple(probe.shape[1:]):
        raise RuntimeError(
            f"Encoded reference shape {tuple(references.shape[1:])} does not match "
            f"native latent shape {tuple(probe.shape[1:])}."
        )
    layout = adapter.latent_layout(probe)
    layout_dict = vars(layout) if hasattr(layout, "__dict__") else None
    feature_dim = infer_channel_dim(probe, model_id, layout_dict)
    kernel_references = F.normalize(
        references.float(),
        p=2.0,
        dim=feature_dim,
        eps=1.0e-12,
    )
    kernel_pairwise_squared_distances = _kernel_pairwise_squared_distances(
        kernel_references,
        device=device,
    )
    target = (
        CALIBRATION_ROOT
        / "unsafe_references_v3"
        / model_id
        / category
        / "unsafe_latents.pt"
    )
    if target.exists():
        raise FileExistsError(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "schema_version": 3,
        "protocol": "paper_exact_switch_reference_bank_v3",
        "model_id": model_id,
        "model_hf_id": adapter.model_id,
        "model_revision": adapter.config.get("revision"),
        "adapter": adapter.adapter_name,
        "category": category,
        "reference_count": len(records),
        "effective_reference_population": len(records),
        "empirical_mass_multiplier": 1.0,
        "shortfall_policy": (
            "all_available_unique_empirical_references_without_duplication_or_mass_inflation"
        ),
        "reference_manifest": str(manifest),
        "reference_manifest_sha256": file_sha256(manifest),
        "corpus_revision": CORPUS_REVISION,
        "tensor_shape": list(references.shape),
        "tensor_dtype": str(references.dtype),
        "tensor_sha256": tensor_sha256(references),
        "kernel_tensor_sha256": tensor_sha256(kernel_references),
        "kernel_pairwise_squared_distances_sha256": tensor_sha256(
            kernel_pairwise_squared_distances
        ),
        "vae_encoding": "native_deterministic_posterior_mode",
        "safe_kernel": {
            "feature_policy": "channel_l2_normalize_query_and_references_v3",
            "feature_dim": feature_dim,
            "distance": "squared_euclidean_in_normalized_feature_space",
            "kernel": "exp(-squared_distance/(2*sigma^2))",
            "eq6_value_space": "raw_native_predicted_x0_latents",
            "beta_units": "mean_kernel_mass_over_unique_empirical_references",
            "threshold_policy": "leave_one_out_reference_beta_quantile",
        },
        "safe_beta_threshold_source": (
            "runtime_sigma_and_sealed_candidate_beta_quantile_over_hashed_"
            "leave_one_out_reference_distances"
        ),
    }
    torch.save(
        {
            "metadata": metadata,
            "references": references,
            "kernel_references": kernel_references,
            "kernel_pairwise_squared_distances": (
                kernel_pairwise_squared_distances
            ),
        },
        target,
    )
    artifact_sha256 = file_sha256(target)
    metadata_path = target.with_suffix(".json")
    atomic_json(
        metadata_path,
        {**metadata, "artifact_sha256": artifact_sha256},
    )
    atomic_json(
        target.parent / "ADMISSION.json",
        {
            "schema_version": 3,
            "protocol": "t2i_safety_reference_bank_admission_v3",
            "status": "accepted",
            "model_id": model_id,
            "category": category,
            "artifact_path": str(target),
            "artifact_sha256": artifact_sha256,
            "metadata_path": str(metadata_path),
            "metadata_sha256": file_sha256(metadata_path),
            "reference_manifest": str(manifest),
            "reference_manifest_sha256": file_sha256(manifest),
            "reference_count": len(records),
            "effective_reference_population": len(records),
            "empirical_mass_multiplier": 1.0,
            "tensor_sha256": metadata["tensor_sha256"],
            "kernel_tensor_sha256": metadata["kernel_tensor_sha256"],
            "kernel_pairwise_squared_distances_sha256": metadata[
                "kernel_pairwise_squared_distances_sha256"
            ],
            "safe_kernel": metadata["safe_kernel"],
            "corpus_revision": CORPUS_REVISION,
        },
    )


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Build exact unsafe-reference artifacts.")
    sub = value.add_subparsers(dest="command", required=True)
    sub.add_parser("build-category-manifests")
    fair = sub.add_parser("build-fairness-manifest")
    fair.add_argument("--model", required=True)
    fair.add_argument("--profile", required=True, type=Path)
    encode = sub.add_parser("encode")
    encode.add_argument("--model", required=True)
    encode.add_argument("--category", required=True)
    return value


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.command == "build-category-manifests":
        build_category_manifests()
    elif args.command == "build-fairness-manifest":
        build_fairness_manifest(args.model, args.profile)
    elif args.command == "encode":
        encode_bank(args.model, args.category)
    else:
        raise AssertionError(args.command)


if __name__ == "__main__":
    main()
