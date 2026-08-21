#!/usr/bin/env python3
"""Exact released-code GrOCE Snoopy Table 1 reproduction orchestration."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import runpy
import shutil
import subprocess
import sys
import tarfile
import time
from typing import Any
import urllib.request


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def read_contract(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema_version") != 1 or value.get("method") != "groce":
        raise RuntimeError("Unsupported GrOCE contract")
    return value


def verify_upstream(contract: dict[str, Any]) -> None:
    observed = subprocess.check_output(
        ["git", "-C", contract["upstream"]["repo"], "rev-parse", "HEAD"], text=True
    ).strip()
    if observed != contract["upstream"]["commit"]:
        raise RuntimeError(f"Pinned GrOCE upstream moved: {observed}")


def safe_extract(archive: Path, destination: Path) -> None:
    root = destination.resolve()
    with tarfile.open(archive, "r") as handle:
        for member in handle.getmembers():
            target = (destination / member.name).resolve()
            if target != root and root not in target.parents:
                raise RuntimeError(f"Unsafe archive member: {member.name}")
        handle.extractall(destination)


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def verify_environment(contract: dict[str, Any]) -> None:
    root = Path(contract["environment"]["root"])
    admission = json.loads((root / "ADMISSION.json").read_text(encoding="utf-8"))
    if admission.get("status") != contract["environment"]["required_status"]:
        raise RuntimeError("GrOCE v2 environment is not admitted for execution")
    expected = {
        "torch": contract["environment"]["torch"],
        "torchvision": contract["environment"]["torchvision"],
        "torchaudio": contract["environment"]["torchaudio"],
    }
    observed = {name: package_version(name) for name in expected}
    if observed != expected:
        raise RuntimeError(f"GrOCE CUDA stack mismatch: expected={expected}, observed={observed}")


def prepare(contract_path: Path, contract: dict[str, Any]) -> None:
    verify_upstream(contract)
    final = Path(contract["execution"]["attempt_root"])
    if final.exists():
        raise RuntimeError(f"Refusing existing GrOCE attempt: {final}")
    temporary = final.parent / f".{final.name}.preparing_{os.getpid()}"
    temporary.mkdir(parents=True)
    try:
        source = temporary / "UPSTREAM_SOURCE"
        tree = source / "tree"
        tree.mkdir(parents=True)
        archive = source / "groce_970ea07eb7ed.tar"
        subprocess.run(
            ["git", "-C", contract["upstream"]["repo"], "archive", "--format=tar", "--output", str(archive), contract["upstream"]["commit"]],
            check=True,
        )
        safe_extract(archive, tree)
        template_module = runpy.run_path(str(tree / "src" / "template.py"))
        templates = template_module["template_dict"][contract["generation"]["erase_type"]]
        if len(templates) != contract["generation"]["template_count"]:
            raise RuntimeError(f"Expected 80 instance templates, found {len(templates)}")
        if len(set(templates)) != len(templates):
            raise RuntimeError("Released instance templates are not unique")
        inputs = temporary / "INPUTS"
        inputs.mkdir()
        shutil.copy2(contract_path, inputs / "reproduction_contract.json")
        write_json(inputs / "templates.json", templates)
        for relative in (
            "ASSETS/torch_home/hub/checkpoints",
            "GRAPH",
            "OUTPUTS/pretrain",
            "OUTPUTS/logs/GrOCE/instance",
            "STATUS",
        ):
            (temporary / relative).mkdir(parents=True, exist_ok=True)
        manifest = {
            "schema_version": 1,
            "status": "prepared_not_executed",
            "upstream_commit": contract["upstream"]["commit"],
            "upstream_archive_sha256": sha256_file(archive),
            "contract_sha256": sha256_file(contract_path),
            "contents": contract["generation"]["contents"],
            "templates_per_content": len(templates),
            "samples_per_template": contract["generation"]["num_samples"],
            "expected_original_images": len(templates) * len(contract["generation"]["contents"]) * contract["generation"]["num_samples"],
            "expected_erased_images": len(templates) * len(contract["generation"]["contents"]) * contract["generation"]["num_samples"],
            "python_hash_seed": contract["generation"]["python_hash_seed"],
            "hash_seed_reason": "Released expand_concepts returns list(set(...)); fixed only to make ordering repeatable.",
        }
        write_json(temporary / "ATTEMPT_MANIFEST.json", manifest)
        final.parent.mkdir(parents=True, exist_ok=True)
        temporary.rename(final)
    except BaseException:
        failed = final.parent / f"{final.name}.FAILED_PREPARE_{os.getpid()}"
        if temporary.exists():
            temporary.rename(failed)
        raise
    print(f"Prepared immutable GrOCE attempt: {final}")


def fetch_metric_asset(contract: dict[str, Any]) -> None:
    attempt = Path(contract["execution"]["attempt_root"])
    asset = contract["metric_assets"]
    final = attempt / "ASSETS" / "torch_home" / "hub" / "checkpoints" / asset["filename"]
    admission_path = final.parent / "INCEPTION_ADMISSION.json"
    if admission_path.exists():
        admission = json.loads(admission_path.read_text(encoding="utf-8"))
        if admission.get("sha256") != sha256_file(final) or not admission["sha256"].startswith(asset["required_sha256_prefix"]):
            raise RuntimeError("Existing Torch-Fidelity asset admission is invalid")
        print(f"Torch-Fidelity Inception asset already admitted: {final}")
        return
    if final.exists():
        raise RuntimeError(f"Refusing unadmitted metric asset: {final}")
    temporary = final.with_suffix(final.suffix + f".downloading_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
    try:
        with urllib.request.urlopen(asset["torch_fidelity_inception_url"], timeout=120) as response, temporary.open("wb") as output:
            while True:
                block = response.read(8 * 1024 * 1024)
                if not block:
                    break
                output.write(block)
        digest = sha256_file(temporary)
        if not digest.startswith(asset["required_sha256_prefix"]):
            raise RuntimeError(f"Torch-Fidelity Inception SHA-256 prefix mismatch: {digest}")
        temporary.replace(final)
        write_json(admission_path, {
            "schema_version": 1,
            "status": "admitted",
            "source_url": asset["torch_fidelity_inception_url"],
            "path": str(final),
            "sha256": digest,
            "bytes": final.stat().st_size,
            "required_sha256_prefix": asset["required_sha256_prefix"],
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        })
    except BaseException:
        if temporary.exists():
            failed = final.with_suffix(final.suffix + f".FAILED_{os.environ.get('SLURM_JOB_ID', os.getpid())}")
            temporary.replace(failed)
        raise
    print(f"Admitted Torch-Fidelity Inception asset: {final}")


def runtime_environment(contract: dict[str, Any]) -> dict[str, str]:
    env = os.environ.copy()
    hf_cache = contract["model"]["hf_cache"]
    env.update({
        "HF_HUB_CACHE": hf_cache,
        "HUGGINGFACE_HUB_CACHE": hf_cache,
        "TRANSFORMERS_CACHE": hf_cache,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "DIFFUSERS_OFFLINE": "1",
        "TOKENIZERS_PARALLELISM": "false",
        "PYTHONHASHSEED": str(contract["generation"]["python_hash_seed"]),
        "TORCH_HOME": str(Path(contract["execution"]["attempt_root"]) / "ASSETS" / "torch_home"),
    })
    return env


def verify_model_assets(contract: dict[str, Any]) -> None:
    admission = json.loads((Path(contract["model"]["hf_cache"]) / "ASSET_ADMISSION.json").read_text(encoding="utf-8"))
    if admission.get("status") != "admitted" or admission.get("sd_commit") != contract["model"]["resolved_commit"]:
        raise RuntimeError("Shared exact SD 1.4 cache is not admitted")


def build_graph(contract: dict[str, Any]) -> None:
    verify_environment(contract)
    verify_model_assets(contract)
    attempt = Path(contract["execution"]["attempt_root"])
    output = attempt / "GRAPH" / "concept_network.json"
    status = attempt / "STATUS" / "GRAPH_COMPLETE.json"
    if status.exists():
        record = json.loads(status.read_text(encoding="utf-8"))
        if record.get("sha256") != sha256_file(output):
            raise RuntimeError("Existing GrOCE graph completion record is invalid")
        print("GrOCE graph already complete")
        return
    graph = contract["graph"]
    tree = attempt / "UPSTREAM_SOURCE" / "tree"
    command = [
        sys.executable, "knowledge.py",
        "--sd_ckpt", contract["model"]["repo_id"],
        "--seed", str(graph["seed"]),
        "--output_dir", str(attempt / "GRAPH"),
        "--batch_size", str(graph["batch_size"]),
        "--similarity_threshold", str(graph["similarity_threshold"]),
        "--max_connections", str(graph["max_connections"]),
        "--sigma", str(graph["sigma"]),
        "--lambda_param", str(graph["lambda_param"]),
    ]
    started = time.time()
    subprocess.run(command, cwd=tree, env=runtime_environment(contract), check=True)
    data = json.loads(output.read_text(encoding="utf-8"))["concept_network"]
    edges = sum(len(value) for value in data.values())
    write_json(status, {
        "status": "complete_not_paper_admitted",
        "nodes": len(data),
        "directed_edges": edges,
        "sha256": sha256_file(output),
        "bytes": output.stat().st_size,
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    })
    print(f"Completed GrOCE graph: nodes={len(data)} edges={edges}")


def image_paths(contract: dict[str, Any], mode: str) -> dict[str, list[Path]]:
    attempt = Path(contract["execution"]["attempt_root"])
    paths: dict[str, list[Path]] = {}
    for concept in contract["generation"]["contents"]:
        if mode == "original":
            root = attempt / "OUTPUTS" / "pretrain" / "instance" / concept / "original"
        else:
            root = attempt / "OUTPUTS" / "logs" / "GrOCE" / "instance" / "Snoopy" / concept / "edit"
        paths[concept] = sorted(root.glob("*.png")) if root.is_dir() else []
    return paths


def verify_generated_images(contract: dict[str, Any], mode: str) -> list[dict[str, Any]]:
    from PIL import Image

    expected_per_concept = contract["generation"]["template_count"] * contract["generation"]["num_samples"]
    records = []
    for concept, paths in image_paths(contract, mode).items():
        if len(paths) != expected_per_concept:
            raise RuntimeError(f"Expected {expected_per_concept} {mode} {concept} images, found {len(paths)}")
        for path in paths:
            with Image.open(path) as image:
                image.verify()
            with Image.open(path) as image:
                if image.size != (512, 512):
                    raise RuntimeError(f"Unexpected GrOCE image size: {path} {image.size}")
            records.append({
                "concept": concept,
                "path": str(path),
                "name": path.name,
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            })
    return records


def generate(contract: dict[str, Any], mode: str) -> None:
    verify_environment(contract)
    verify_model_assets(contract)
    attempt = Path(contract["execution"]["attempt_root"])
    graph_status = json.loads((attempt / "STATUS" / "GRAPH_COMPLETE.json").read_text(encoding="utf-8"))
    if graph_status.get("status") != "complete_not_paper_admitted":
        raise RuntimeError("GrOCE graph is not complete")
    status = attempt / "STATUS" / f"GENERATION_{mode.upper()}_COMPLETE.json"
    if status.exists():
        existing = json.loads(status.read_text(encoding="utf-8"))
        if existing.get("image_count") != 3200:
            raise RuntimeError("Existing GrOCE generation status is invalid")
        print(f"GrOCE {mode} generation already complete")
        return
    generation = contract["generation"]
    tree = attempt / "UPSTREAM_SOURCE" / "tree"
    common = [
        "--sd_ckpt", contract["model"]["repo_id"],
        "--seed", str(generation["seed"]),
        "--guidance_scale", str(generation["guidance_scale"]),
        "--total_timesteps", str(generation["total_timesteps"]),
        "--num_samples", str(generation["num_samples"]),
        "--batch_size", str(generation["batch_size"]),
        "--erase_type", generation["erase_type"],
        "--contents", ", ".join(generation["contents"]),
    ]
    if mode == "original":
        command = [
            sys.executable, "sample_origin.py",
            "--save_root", str(attempt / "OUTPUTS" / "pretrain"),
            "--target_concept", generation["erase_type"],
        ] + common
    else:
        working_graph = attempt / "GRAPH" / "concept_network.SNOOPY_WORKING.json"
        if not working_graph.exists():
            shutil.copy2(attempt / "GRAPH" / "concept_network.json", working_graph)
        command = [
            sys.executable, "sample_erase.py",
            "--save_root", str(attempt / "OUTPUTS" / "logs" / "GrOCE" / "instance"),
            "--mode", "edit",
            "--target_concepts", generation["target_concepts"],
            "--network_path", str(working_graph),
            "--n_step", str(generation["n_step"]),
            "--top_k", str(generation["top_k"]),
            "--decay_factor", str(generation["decay_factor"]),
            "--insert_topk", str(generation["insert_topk"]),
            "--similarity_threshold", str(generation["similarity_threshold"]),
            "--sigma", str(generation["sigma"]),
            "--lambda_param", str(generation["lambda_param"]),
            "--diffusion_steps", str(generation["diffusion_steps"]),
            "--projection_threshold", str(generation["projection_threshold"]),
            "--embedding_batch_size", str(generation["embedding_batch_size"]),
        ] + common
    started = time.time()
    subprocess.run(command, cwd=tree, env=runtime_environment(contract), check=True)
    records = verify_generated_images(contract, mode)
    write_json(status, {
        "status": "complete_not_paper_admitted",
        "mode": mode,
        "image_count": len(records),
        "elapsed_seconds": time.time() - started,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "packages": {name: package_version(name) for name in ("torch", "torchvision", "torchaudio", "diffusers", "transformers", "torch-fidelity")},
        "images": records,
    })
    print(f"Completed exact GrOCE {mode} generation: {len(records)} images")


def expected_filename(prompt: str, sample: int, mode: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", prompt)
    if mode == "original":
        cleaned = cleaned.replace(", ", "_")
    else:
        cleaned = cleaned.replace(" ", "_")
    return f"{cleaned}_{sample}.png"


def create_contact_sheets(contract: dict[str, Any], output: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    from PIL import Image, ImageDraw

    attempt = Path(contract["execution"]["attempt_root"])
    templates = json.loads((attempt / "INPUTS" / "templates.json").read_text(encoding="utf-8"))
    output.mkdir(exist_ok=False)
    pairs = []
    identity = {}
    for concept in contract["generation"]["contents"]:
        same = 0
        for template_index, template in enumerate(templates):
            prompt = template.format(concept)
            for sample in range(contract["generation"]["num_samples"]):
                original = attempt / "OUTPUTS" / "pretrain" / "instance" / concept / "original" / expected_filename(prompt, sample, "original")
                erased = attempt / "OUTPUTS" / "logs" / "GrOCE" / "instance" / "Snoopy" / concept / "edit" / expected_filename(prompt, sample, "erased")
                if not original.is_file() or not erased.is_file():
                    raise RuntimeError(f"Cannot pair GrOCE images: {original} ; {erased}")
                original_sha = sha256_file(original)
                erased_sha = sha256_file(erased)
                same += original_sha == erased_sha
                pairs.append((concept, template_index, sample, prompt, original, erased, original_sha, erased_sha))
        identity[concept] = {"identical_sha256_pairs": same, "total_pairs": len(templates) * contract["generation"]["num_samples"]}
    pages = []
    page_size = 32
    for page_index in range((len(pairs) + page_size - 1) // page_size):
        page_pairs = pairs[page_index * page_size:(page_index + 1) * page_size]
        canvas = Image.new("RGB", (4 * 300, 8 * 182), "white")
        draw = ImageDraw.Draw(canvas)
        descriptors = []
        for slot, (concept, template_index, sample, prompt, original, erased, _, _) in enumerate(page_pairs):
            x = (slot % 4) * 300
            y = (slot // 4) * 182
            for offset, path in ((0, original), (146, erased)):
                with Image.open(path) as image:
                    thumb = image.convert("RGB")
                    thumb.thumbnail((142, 142))
                    canvas.paste(thumb, (x + offset, y))
            draw.text((x + 2, y + 144), f"{concept} t{template_index:02d} s{sample} | original / erased", fill="black")
            draw.text((x + 2, y + 160), prompt[:46], fill="black")
            descriptors.append({"concept": concept, "template_index": template_index, "sample": sample})
        page = output / f"page_{page_index:03d}.jpg"
        canvas.save(page, quality=86, optimize=True)
        pages.append({"page": page.name, "pairs": descriptors, "sha256": sha256_file(page)})
    return pages, identity


def evaluate(contract: dict[str, Any]) -> None:
    verify_environment(contract)
    verify_model_assets(contract)
    attempt = Path(contract["execution"]["attempt_root"])
    for mode in ("original", "erased"):
        status = json.loads((attempt / "STATUS" / f"GENERATION_{mode.upper()}_COMPLETE.json").read_text(encoding="utf-8"))
        if status.get("image_count") != 3200:
            raise RuntimeError(f"GrOCE {mode} generation is incomplete")
    metric_asset = contract["metric_assets"]
    checkpoint = attempt / "ASSETS" / "torch_home" / "hub" / "checkpoints" / metric_asset["filename"]
    digest = sha256_file(checkpoint)
    if not digest.startswith(metric_asset["required_sha256_prefix"]):
        raise RuntimeError("Torch-Fidelity Inception checkpoint failed its hash gate")
    tree = attempt / "UPSTREAM_SOURCE" / "tree"
    root = attempt / "OUTPUTS" / "logs" / "GrOCE" / "instance"
    command = [
        sys.executable, "src/clip_score_cal.py",
        "--contents", ", ".join(contract["generation"]["contents"]),
        "--root_path", str(root),
        "--pretrained_path", str(attempt / "OUTPUTS" / "pretrain" / "instance"),
    ]
    subprocess.run(command, cwd=tree, env=runtime_environment(contract), check=True)
    record_path = root / "Snoopy" / "record_metrics.txt"
    text = record_path.read_text(encoding="utf-8")
    pattern = re.compile(r"^(\w+): CS is ([0-9.]+), FID is ([0-9.]+), PSNR is ([0-9.]+|inf)", re.MULTILINE)
    metrics = {
        match.group(1): {"clip": float(match.group(2)), "fid": float(match.group(3)), "psnr": float(match.group(4))}
        for match in pattern.finditer(text)
    }
    expected_names = set(contract["generation"]["contents"])
    if set(metrics) != expected_names:
        raise RuntimeError(f"Released GrOCE evaluator metric set mismatch: {sorted(metrics)}")
    pages, identity = create_contact_sheets(contract, attempt / "CONTACT_SHEETS_ALL_PAIRS")
    write_json(attempt / "CONTACT_SHEETS_ALL_PAIRS" / "INDEX.json", pages)
    paper = contract["paper_metrics"]
    tolerance = contract["tolerances"]
    gates = {
        "snoopy_clip": abs(metrics["Snoopy"]["clip"] - paper["snoopy_clip"]) <= tolerance["clip_absolute"],
        "mickey_fid": abs(metrics["Mickey"]["fid"] - paper["mickey_fid"]) <= tolerance["fid_absolute"],
        "spongebob_fid": abs(metrics["Spongebob"]["fid"] - paper["spongebob_fid"]) <= tolerance["fid_absolute"],
        "pikachu_fid": abs(metrics["Pikachu"]["fid"] - paper["pikachu_fid"]) <= tolerance["fid_absolute"],
    }
    result = {
        "schema_version": 1,
        "status": "numerical_gate_passed_visual_review_required" if all(gates.values()) else "numerical_gate_failed",
        "released_evaluator_metrics": metrics,
        "paper_metrics": paper,
        "tolerances": tolerance,
        "metric_gates": gates,
        "pair_identity": identity,
        "contact_sheet_pages": len(pages),
        "contact_sheet_pairs": 3200,
        "visual_review_complete": False,
        "python_hash_seed": contract["generation"]["python_hash_seed"],
        "torch_fidelity_inception_sha256": digest,
    }
    write_json(attempt / "EVALUATION_RESULT.json", result)
    write_json(attempt / "VISUAL_REVIEW_REQUIRED.json", {
        "required": True,
        "pages": len(pages),
        "pairs": 3200,
        "paper_admission_blocked_until_review": True,
    })
    print(json.dumps(result, indent=2, sort_keys=True))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", required=True, type=Path)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    commands.add_parser("fetch-metric-asset")
    commands.add_parser("build-graph")
    generate_parser = commands.add_parser("generate")
    generate_parser.add_argument("--mode", required=True, choices=("original", "erased"))
    commands.add_parser("evaluate")
    args = parser.parse_args()
    contract = read_contract(args.contract)
    if args.command == "prepare":
        prepare(args.contract, contract)
    elif args.command == "fetch-metric-asset":
        verify_environment(contract)
        fetch_metric_asset(contract)
    elif args.command == "build-graph":
        build_graph(contract)
    elif args.command == "generate":
        generate(contract, args.mode)
    elif args.command == "evaluate":
        evaluate(contract)
    return 0


if __name__ == "__main__":
    sys.exit(main())
