#!/usr/bin/env python3
"""Run the exact released DES SD1.5 CLIP/FID quality evaluation."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", encoding="utf-8", dir=str(path.parent), delete=False
    ) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def load_config(path: Path) -> Tuple[Dict[str, Any], str]:
    raw = path.read_bytes()
    config = json.loads(raw)
    if config.get("schema_version") != 1:
        raise RuntimeError("Unsupported DES quality configuration schema")
    if config.get("method") != "DES":
        raise RuntimeError("Configuration is not a DES protocol")
    if config["metric_protocol"].get("fallbacks_allowed") is not False:
        raise RuntimeError("DES quality protocol must explicitly prohibit fallbacks")
    return config, hashlib.sha256(raw).hexdigest()


def git_head(repository: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return completed.stdout.strip()


def require_file_hash(path: Path, expected: str, label: str) -> Dict[str, Any]:
    if not path.is_file():
        raise RuntimeError(f"Missing exact {label}: {path}")
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(
            f"Exact {label} hash mismatch: expected {expected}, observed {observed}"
        )
    return {"path": str(path), "sha256": observed, "bytes": path.stat().st_size}


def require_git_commit(path: Path, expected: str, label: str) -> Dict[str, str]:
    if not path.is_dir():
        raise RuntimeError(f"Missing exact {label} checkout: {path}")
    observed = git_head(path)
    if observed != expected:
        raise RuntimeError(
            f"Exact {label} commit mismatch: expected {expected}, observed {observed}"
        )
    for command in (
        ["git", "-C", str(path), "diff", "--quiet"],
        ["git", "-C", str(path), "diff", "--cached", "--quiet"],
    ):
        completed = subprocess.run(command, check=False)
        if completed.returncode != 0:
            raise RuntimeError(f"Pinned {label} checkout has tracked modifications: {path}")
    return {"path": str(path), "commit": observed}


def prompt_count(path: Path) -> int:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["prompt"]:
            raise RuntimeError(
                f"Released DES COCO CSV must contain only the prompt column, got {reader.fieldnames}"
            )
        rows = list(reader)
    if any(not isinstance(row.get("prompt"), str) or not row["prompt"].strip() for row in rows):
        raise RuntimeError("Released DES COCO CSV contains an empty or non-string prompt")
    return len(rows)


def static_asset_admission(config: Dict[str, Any], config_sha: str) -> Dict[str, Any]:
    paths = {key: Path(value) for key, value in config["paths"].items()}
    expected = config["expected_sha256"]
    upstreams = config["upstreams"]
    files = {}
    for label in (
        "prompt_csv",
        "clipscore_script",
        "fid_script",
        "mapping_admission",
        "image_admission",
        "reference_manifest",
        "clip_model_safetensors",
        "inception_checkpoint",
    ):
        files[label] = require_file_hash(paths[label], expected[label], label)

    repositories = {
        label: require_git_commit(Path(spec["path"]), spec["commit"], label)
        for label, spec in upstreams.items()
    }
    observed_prompts = prompt_count(paths["prompt_csv"])
    expected_count = int(config["generated_images"]["count"])
    if observed_prompts != expected_count:
        raise RuntimeError(
            f"Released DES prompt denominator mismatch: {observed_prompts} != {expected_count}"
        )

    reference_files = sorted(
        path
        for path in paths["reference_images"].iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    expected_references = int(config["reference_images"]["count"])
    if len(reference_files) != expected_references:
        raise RuntimeError(
            f"Exact COCO reference denominator mismatch: {len(reference_files)} != {expected_references}"
        )
    if len({path.name for path in reference_files}) != expected_references:
        raise RuntimeError("Exact COCO reference directory contains duplicate names")
    if any(path.stat().st_size <= 0 for path in reference_files):
        raise RuntimeError("Exact COCO reference directory contains an empty image")

    payload = {
        "status": "STATIC_ASSETS_ADMITTED",
        "method": "DES",
        "protocol_name": config["protocol_name"],
        "config_sha256": config_sha,
        "completed_at": utc_now(),
        "fallback_used": False,
        "files": files,
        "repositories": repositories,
        "prompt_count": observed_prompts,
        "reference_image_count": len(reference_files),
        "reference_image_name_digest": hashlib.sha256(
            "\n".join(path.name for path in reference_files).encode("utf-8")
        ).hexdigest(),
    }
    atomic_json(paths["quality_root"] / "STATIC_ASSET_ADMISSION.json", payload)
    return payload


def image_files(directory: Path) -> List[Path]:
    return sorted(
        (
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ),
        key=lambda path: path.name,
    )


def locate_generated_coco(config: Dict[str, Any], config_sha: str) -> Dict[str, Any]:
    spec = config["generated_images"]
    root = Path(config["paths"]["generated_search_root"])
    expected_count = int(spec["count"])
    start = int(spec["start_index"])
    stop = int(spec["stop_index_exclusive"])
    suffix = str(spec["suffix"])
    filename_template = str(spec.get("filename_template", "{index}" + suffix))
    expected_names = {filename_template.format(index=index) for index in range(start, stop)}
    if len(expected_names) != expected_count:
        raise RuntimeError("Generated image denominator and index interval disagree")
    if not root.is_dir():
        raise RuntimeError(f"DES generation root does not exist: {root}")

    admission_path = config.get("paths", {}).get("author_generation_admission")
    if admission_path is not None:
        admission_file = Path(admission_path)
        if not admission_file.is_file():
            raise RuntimeError(f"Missing DES author-generation admission: {admission_file}")
        admission = json.loads(admission_file.read_text(encoding="utf-8"))
        if admission.get("status") != "AUTHOR_GENERATION_ADMITTED":
            raise RuntimeError("DES author-generation admission did not pass")
        if admission.get("fallback_used") is not False:
            raise RuntimeError("DES author-generation admission used a prohibited fallback")
        if admission.get("config_sha256") != config_sha:
            raise RuntimeError("DES author-generation admission configuration hash mismatch")
        if admission.get("image_count") != expected_count:
            raise RuntimeError("DES author-generation admission denominator mismatch")
        if admission.get("author_generation_manifest_sha256") != config["expected_sha256"]["author_generation_manifest"]:
            raise RuntimeError("DES author-generation admission manifest hash mismatch")
        if admission.get("author_image_directory") != config["paths"]["author_generation_images"]:
            raise RuntimeError("DES author-generation admission image directory mismatch")
        if admission.get("all_source_hashes_verified") is not True:
            raise RuntimeError("DES author-generation admission did not verify every image hash")
        if admission.get("all_prompts_verified_against_released_csv") is not True:
            raise RuntimeError("DES author-generation admission did not verify every prompt")
        if admission.get("author_filename_contract_preserved") is not True:
            raise RuntimeError("DES author-generation admission changed the author filename contract")

    directories: Iterable[Path] = [root, *sorted(path for path in root.rglob("*") if path.is_dir())]
    candidates: List[Tuple[Path, List[Path]]] = []
    for directory in directories:
        files = image_files(directory)
        if len(files) != expected_count:
            continue
        if {path.name for path in files} == expected_names:
            candidates.append((directory, files))

    if len(candidates) != 1:
        rendered = [str(path) for path, _ in candidates]
        raise RuntimeError(
            f"Expected exactly one complete {filename_template} DES COCO directory; "
            f"found {len(candidates)}: {rendered}"
        )
    directory, files = candidates[0]
    if any(path.stat().st_size <= 0 for path in files):
        raise RuntimeError("DES COCO output contains an empty image")
    metadata_lines = [f"{path.name}\t{path.stat().st_size}" for path in files]
    return {
        "path": str(directory.resolve()),
        "count": len(files),
        "start_index": start,
        "stop_index_exclusive": stop,
        "suffix": suffix,
        "filename_and_size_digest": hashlib.sha256(
            "\n".join(metadata_lines).encode("utf-8")
        ).hexdigest(),
        "locator_rule": spec["locator_rule"],
    }


def run_author_command(command: Sequence[str], output_dir: Path, environment: Dict[str, str]) -> str:
    output_dir.mkdir(parents=True, exist_ok=False)
    atomic_json(
        output_dir / "COMMAND.json",
        {
            "command": list(command),
            "started_at": utc_now(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        },
    )
    completed = subprocess.run(
        list(command),
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )
    (output_dir / "AUTHOR_STDOUT.txt").write_text(completed.stdout, encoding="utf-8")
    (output_dir / "AUTHOR_STDERR.txt").write_text(completed.stderr, encoding="utf-8")
    atomic_json(
        output_dir / "PROCESS_STATUS.json",
        {"returncode": completed.returncode, "completed_at": utc_now()},
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"Pinned DES author evaluator failed with return code {completed.returncode}"
        )
    return completed.stdout + "\n" + completed.stderr


NUMBER = r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"


def parse_named_metric(text: str, metric: str) -> float:
    if metric == "clip":
        patterns = (
            rf"(?im)^.*average\s+clip(?:\s+similarity)?(?:\s+score)?\s*[:=]\s*{NUMBER}",
            rf"(?im)^.*clip(?:\s+similarity)?\s+score\s*[:=]\s*{NUMBER}",
        )
    elif metric == "fid":
        patterns = (
            rf"(?im)^.*fid(?:\s+score)?\s*[:=]\s*{NUMBER}",
            rf"(?im)^.*frechet(?:\s+inception)?(?:\s+distance)?\s*[:=]\s*{NUMBER}",
        )
    else:
        raise RuntimeError(f"Unknown metric parser: {metric}")
    values: List[float] = []
    for pattern in patterns:
        values.extend(float(match) for match in re.findall(pattern, text))
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        raise RuntimeError(f"Pinned author {metric} output did not contain a named finite score")
    distinct = {round(value, 12) for value in finite}
    if len(distinct) != 1:
        raise RuntimeError(f"Pinned author {metric} output contained ambiguous scores: {finite}")
    return finite[-1]


def metric_environment(config: Dict[str, Any]) -> Dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "HF_HOME": config["paths"]["hf_home"],
            "TORCH_HOME": config["paths"]["torch_home"],
            "HF_HUB_OFFLINE": "1",
            "TRANSFORMERS_OFFLINE": "1",
            "PYTHONHASHSEED": "0",
        }
    )
    upstream_python = [
        config["upstreams"]["text2image_benchmark"]["path"],
        config["upstreams"]["openai_clip"]["path"],
    ]
    if environment.get("PYTHONPATH"):
        upstream_python.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(upstream_python)
    return environment


def run_metric(config: Dict[str, Any], config_sha: str, task_id: int) -> Dict[str, Any]:
    if task_id not in (0, 1):
        raise RuntimeError("DES quality task id must be 0 (CLIP) or 1 (FID)")
    static_asset_admission(config, config_sha)
    generated = locate_generated_coco(config, config_sha)
    paths = config["paths"]
    quality_root = Path(paths["quality_root"])
    metric = "clip" if task_id == 0 else "fid"
    output_dir = quality_root / metric.upper()
    if output_dir.exists():
        raise RuntimeError(f"Refusing to overwrite an existing exact metric attempt: {output_dir}")

    if metric == "clip":
        command = [
            sys.executable,
            paths["clipscore_script"],
            "--image_folder",
            generated["path"],
            "--csv_file",
            paths["prompt_csv"],
            "--device",
            "cuda:0",
        ]
    else:
        command = [
            sys.executable,
            paths["fid_script"],
            "--job",
            "fid",
            "--gen_imgs_path",
            generated["path"],
            "--coco_imgs_path",
            paths["reference_images"],
            "--device",
            "cuda:0",
            "--output_file",
            str(output_dir / "AUTHOR_METRIC_OUTPUT.txt"),
        ]

    combined_output = run_author_command(command, output_dir, metric_environment(config))
    author_output = output_dir / "AUTHOR_METRIC_OUTPUT.txt"
    if author_output.is_file():
        combined_output += "\n" + author_output.read_text(encoding="utf-8")
    raw_score = parse_named_metric(combined_output, metric)

    targets = config["paper_targets"]
    if metric == "clip":
        normalized_score = raw_score * 100.0 if abs(raw_score) <= 1.0 else raw_score
        target = float(targets["clip_score_percent"])
        tolerance = float(targets["clip_absolute_tolerance"])
        score_key = "clip_score_percent"
    else:
        normalized_score = raw_score
        target = float(targets["fid"])
        tolerance = float(targets["fid_absolute_tolerance"])
        score_key = "fid"
    absolute_error = abs(normalized_score - target)
    result = {
        "status": "METRIC_COMPLETE",
        "metric": metric,
        "score_key": score_key,
        "raw_author_score": raw_score,
        "normalized_score": normalized_score,
        "paper_target": target,
        "absolute_tolerance": tolerance,
        "absolute_error": absolute_error,
        "numerical_gate_passed": absolute_error <= tolerance,
        "config_sha256": config_sha,
        "protocol_name": config["protocol_name"],
        "generated": generated,
        "fallback_used": False,
        "implementation_commit": git_head(Path.cwd()),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "completed_at": utc_now(),
    }
    atomic_json(output_dir / "RESULT.json", result)
    return result


def aggregate(config: Dict[str, Any], config_sha: str) -> Dict[str, Any]:
    quality_root = Path(config["paths"]["quality_root"])
    clip = json.loads((quality_root / "CLIP" / "RESULT.json").read_text(encoding="utf-8"))
    fid = json.loads((quality_root / "FID" / "RESULT.json").read_text(encoding="utf-8"))
    for result in (clip, fid):
        if result.get("config_sha256") != config_sha:
            raise RuntimeError("Metric result was produced from a different quality configuration")
        if result.get("fallback_used") is not False:
            raise RuntimeError("Metric result used a prohibited fallback")
    if clip["generated"] != fid["generated"]:
        raise RuntimeError("CLIP and FID did not evaluate the identical generated image set")
    numerical_pass = bool(clip["numerical_gate_passed"] and fid["numerical_gate_passed"])
    payload = {
        "status": (
            "PAPER_REPRODUCTION_NUMERICAL_ADMITTED"
            if numerical_pass
            else "PAPER_REPRODUCTION_NUMERICAL_FAILED"
        ),
        "admitted": numerical_pass,
        "manual_visual_gate": "PENDING",
        "method": "DES",
        "protocol_name": config["protocol_name"],
        "config_sha256": config_sha,
        "clip": clip,
        "fid": fid,
        "generated": clip["generated"],
        "fallback_used": False,
        "completed_at": utc_now(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_json(quality_root / "QUALITY_NUMERIC_ADMISSION.json", payload)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate-static")
    metric_parser = subparsers.add_parser("metric")
    metric_parser.add_argument("--task-id", type=int, required=True)
    subparsers.add_parser("aggregate")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config, config_sha = load_config(args.config)
    if args.command == "validate-static":
        result = static_asset_admission(config, config_sha)
    elif args.command == "metric":
        result = run_metric(config, config_sha, args.task_id)
    else:
        result = aggregate(config, config_sha)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
