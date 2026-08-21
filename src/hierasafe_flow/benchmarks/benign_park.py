from __future__ import annotations

import argparse
import json
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

from hierasafe_flow.cli.run_redteam_tri_condition import run_native_negative_prompt_baseline
from hierasafe_flow.generation.runner import GenerationRunner
from hierasafe_flow.utils.config import deep_merge, load_config, load_yaml
from hierasafe_flow.utils.io import ensure_dir, write_json, write_text, write_yaml


BENCHMARK_NAME = "benign_park_attribute_transfer_v1"
PROMPT_PRIORITY = (
    "P0_all_sources_main",
    "P2_emotion_source_only",
    "P3_pose_source_only",
    "P5_pose_action_source",
    "P1_color_source_only",
    "P4_action_source_only",
    "P6_target_control_no_source",
    "P7_green_background_control",
)
ALL_PROMPT_IDS = (
    "P0_all_sources_main",
    "P1_color_source_only",
    "P2_emotion_source_only",
    "P3_pose_source_only",
    "P4_action_source_only",
    "P5_pose_action_source",
    "P6_target_control_no_source",
    "P7_green_background_control",
)
CORE_PROMPT_IDS = (
    "P0_all_sources_main",
    "P2_emotion_source_only",
    "P3_pose_source_only",
    "P5_pose_action_source",
)
CONTROL_PROMPT_IDS = (
    "P0_all_sources_main",
    "P6_target_control_no_source",
    "P7_green_background_control",
)
PAIR_IDS = (
    "facial_affect_sad_to_happy",
    "body_pose_sitting_to_walking",
    "clothing_color_green_to_red_blue",
    "sandwich_action_eating_to_holding",
    "composition_static_to_dynamic",
)
PAIR_GROUPS = {
    "emotion": ("facial_affect_sad_to_happy",),
    "pose": ("body_pose_sitting_to_walking",),
    "color": ("clothing_color_green_to_red_blue",),
    "action": ("sandwich_action_eating_to_holding",),
    "composition": ("composition_static_to_dynamic",),
    "pose_action": ("body_pose_sitting_to_walking", "sandwich_action_eating_to_holding"),
    "emotion_pose": ("facial_affect_sad_to_happy", "body_pose_sitting_to_walking"),
    "emotion_pose_action": (
        "facial_affect_sad_to_happy",
        "body_pose_sitting_to_walking",
        "sandwich_action_eating_to_holding",
    ),
    "emotion_pose_color_action": (
        "facial_affect_sad_to_happy",
        "body_pose_sitting_to_walking",
        "clothing_color_green_to_red_blue",
        "sandwich_action_eating_to_holding",
    ),
    "full": PAIR_IDS,
}
SCHEDULE_WINDOWS = {
    "early": (0.00, 0.35),
    "middle": (0.25, 0.75),
    "late": (0.65, 1.00),
    "full_window": (0.00, 1.00),
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run or prepare the benign park ConceptSteer benchmark.")
    parser.add_argument("--benchmark", default=BENCHMARK_NAME)
    parser.add_argument("--model", default="flux2_dev")
    parser.add_argument("--models", default=None, help="Comma-separated model list or 'all' for every configs/models/*.yaml.")
    parser.add_argument("--stage", default="0", choices=["0", "1", "2", "3", "4", "5", "6", "all"])
    parser.add_argument("--variants", default=None, help="Comma-separated explicit variant list.")
    parser.add_argument("--prompt-ids", default=None, help="Comma-separated explicit prompt IDs.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-root", default="outputs/benign_park_conceptsteer_debug")
    parser.add_argument("--write-manifest", default=None)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--index", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--num-inference-steps", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--num-frames", type=int, default=None)
    parser.add_argument("--fps", type=int, default=None)
    parser.add_argument("--duration-seconds", type=float, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument(
        "--output-layout",
        default="stage",
        choices=["stage", "model_first"],
        help=(
            "Use 'stage' for the historical output_root/stage/model/variant layout, or "
            "'model_first' for direct comparisons under output_root/model/variant."
        ),
    )
    parser.add_argument(
        "--cpu-offload",
        default=None,
        choices=[None, "model", "sequential", "true", "false"],
        help=(
            "Optional memory/offload override. By default the benchmark respects each model config. "
            "Use 'sequential' only for memory-constrained nodes; H100 runs should usually leave this unset."
        ),
    )
    parser.add_argument("--tensorboard", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save-latents", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--save-traces", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    root = project_root()
    if args.benchmark != BENCHMARK_NAME:
        raise ValueError(f"Unsupported benchmark {args.benchmark!r}; expected {BENCHMARK_NAME!r}.")
    if args.seed != 0:
        raise ValueError("The benign debugging benchmark must use manual seed 0 only.")

    if args.manifest:
        manifest = _read_manifest(Path(args.manifest))
        if args.index is None:
            results = [run_job(job, root) for job in manifest["jobs"]]
            print(json.dumps({"num_jobs": len(results), "results": results}, indent=2))
            return
        jobs = manifest["jobs"]
        if args.index < 0 or args.index >= len(jobs):
            raise IndexError(f"Job index {args.index} is out of range 0..{len(jobs) - 1}.")
        print(json.dumps(run_job(jobs[args.index], root), indent=2))
        return

    manifest = build_manifest(args, root)
    if args.write_manifest:
        write_manifest(manifest, Path(args.write_manifest), root)
        print(f"Wrote {len(manifest['jobs'])} jobs to {args.write_manifest}")
        return

    results = [run_job(job, root) for job in manifest["jobs"]]
    print(json.dumps({"num_jobs": len(results), "results": results}, indent=2))


def build_manifest(args: argparse.Namespace, root: Path) -> dict[str, Any]:
    model_names = _selected_model_names(root, args.model, args.models)
    manifests = [_build_single_model_manifest(args, root, model_name) for model_name in model_names]
    if len(manifests) == 1:
        return manifests[0]
    jobs = [job for manifest in manifests for job in manifest["jobs"]]
    return {
        "schema_version": 1,
        "benchmark": BENCHMARK_NAME,
        "model": "multi_model",
        "models": [manifest["model"] for manifest in manifests],
        "stage": args.stage,
        "seed": 0,
        "num_jobs": len(jobs),
        "jobs": jobs,
    }


def _build_single_model_manifest(args: argparse.Namespace, root: Path, requested_model: str) -> dict[str, Any]:
    prompt_suite_path = root / "configs/experiments/benign_park_attribute_transfer_v1_prompts.yaml"
    negative_prompt_path = root / "configs/experiments/benign_park_attribute_transfer_v1_negative_prompts.yaml"
    concept_tree_path = root / "configs/concepts/benign_park_concept_tree.yaml"
    prompts = _load_prompt_suite(prompt_suite_path)
    negative_prompts = _load_negative_prompts(negative_prompt_path)
    model_config_path, model_name = _resolve_model_config(root, requested_model)
    model_config = load_yaml(model_config_path)
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = root / output_root

    variants = _explicit_csv(args.variants) if args.variants else None
    prompt_ids = _explicit_csv(args.prompt_ids) if args.prompt_ids else None
    jobs: list[dict[str, Any]] = []
    for stage_name, stage_variants, stage_prompt_ids in _stage_plan(args.stage, variants, prompt_ids):
        for variant in stage_variants:
            variant_spec = _variant_spec(variant, model_name=model_name)
            selected_prompt_ids = _validate_prompt_ids(stage_prompt_ids, prompts)
            for prompt_id in selected_prompt_ids:
                prompt_entry = prompts[prompt_id]
                negative_prompt = _negative_prompt_for(variant, prompt_id, negative_prompts)
                if args.output_layout == "model_first":
                    run_output_dir = output_root / model_name / variant / prompt_id
                else:
                    run_output_dir = output_root / stage_name / model_name / variant / prompt_id
                generation = _generation_config(args, model_config)
                jobs.append(
                    {
                        "schema_version": 1,
                        "benchmark": BENCHMARK_NAME,
                        "stage": stage_name,
                        "variant": variant,
                        "variant_spec": variant_spec,
                        "prompt_id": prompt_id,
                        "prompt": prompt_entry["prompt"],
                        "prompt_metadata": prompt_entry,
                        "negative_prompt": negative_prompt,
                        "seed": 0,
                        "model_name": model_name,
                        "model_config": str(model_config_path),
                        "base_config": str(root / "configs/default.yaml"),
                        "concept_tree": str(concept_tree_path),
                        "prompt_suite": str(prompt_suite_path),
                        "negative_prompt_config": str(negative_prompt_path),
                        "output_dir": str(run_output_dir),
                        "runtime": _runtime_config(args, model_config),
                        "generation": {
                            **generation,
                            "num_outputs_per_prompt": 1,
                        },
                        "logging": {
                            "tensorboard": bool(args.tensorboard),
                        },
                        "output": {
                            "decode": True,
                            "save_latents": bool(args.save_latents),
                            "save_traces": bool(args.save_traces),
                            "image_format": "png",
                            "video_format": "mp4",
                        },
                    }
                )

    if not jobs:
        raise ValueError("Benchmark manifest contains no jobs.")
    return {
        "schema_version": 1,
        "benchmark": BENCHMARK_NAME,
        "model": model_name,
        "stage": args.stage,
        "seed": 0,
        "num_jobs": len(jobs),
        "jobs": jobs,
    }


def run_job(job: dict[str, Any], root: Path | None = None) -> dict[str, Any]:
    root = root or project_root()
    output_dir = ensure_dir(Path(str(job["output_dir"])))
    write_yaml(output_dir / "benchmark_job.yaml", job)
    _write_job_notes(output_dir, job, status="started")
    try:
        config = _runner_config(job, root)
        if job["variant_spec"]["kind"] == "native_negative_prompt":
            result = run_native_negative_prompt_baseline(config, str(job["prompt"]))
            media_paths = _validate_media_outputs(output_dir) if result["status"] == "completed" else []
            payload = {
                "schema_version": 1,
                "status": result["status"],
                "job": job,
                "result": result.get("result"),
                "reason": result.get("reason"),
                "validated_media_paths": media_paths,
            }
            write_json(output_dir / "benchmark_job_result.json", payload)
            _write_job_notes(output_dir, job, status=result["status"], payload=payload)
            return payload

        result = GenerationRunner(config).run(prompt=str(job["prompt"]))
        media_paths = _validate_media_outputs(output_dir)
        payload = {
            "schema_version": 1,
            "status": "completed",
            "job": job,
            "runner_output_dir": result.output_dir,
            "records": [asdict(record) for record in result.records],
            "validated_media_paths": media_paths,
        }
        write_json(output_dir / "benchmark_job_result.json", payload)
        _write_job_notes(output_dir, job, status="completed", payload=payload)
        return payload
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "status": "failed",
            "job": job,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(output_dir / "benchmark_job_result.json", payload)
        _write_job_notes(output_dir, job, status="failed", payload=payload)
        raise


def write_manifest(manifest: dict[str, Any], path: Path, root: Path) -> None:
    if not path.is_absolute():
        path = root / path
    ensure_dir(path.parent)
    write_json(path, manifest)
    write_json(
        path.with_name("manifest_summary.json"),
        {key: value for key, value in manifest.items() if key != "jobs"},
    )
    path.with_name("num_jobs.txt").write_text(f"{len(manifest['jobs'])}\n", encoding="utf-8")


def _read_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("jobs"), list):
        raise ValueError(f"Manifest must be an object with a jobs list: {path}")
    if data.get("benchmark") != BENCHMARK_NAME:
        raise ValueError(f"Manifest benchmark mismatch in {path}: {data.get('benchmark')!r}")
    return data


def _runner_config(job: dict[str, Any], root: Path) -> dict[str, Any]:
    benchmark_name = str(job.get("benchmark", BENCHMARK_NAME))
    seed = int(job.get("seed", 0))
    base = load_config(job["base_config"], project_root=root)
    model_config = load_yaml(job["model_config"])
    variant_spec = job["variant_spec"]
    generation = dict(job["generation"])
    generation.update({"prompt": job["prompt"], "prompt_file": None})

    model = dict(model_config.get("model", {}))
    model["guidance_scale"] = generation.get("guidance_scale")
    dtype = job.get("runtime", {}).get("dtype")
    if dtype:
        model["torch_dtype"] = dtype
    runtime = job.get("runtime", {})
    if "cpu_offload" in runtime:
        cpu_offload = runtime["cpu_offload"]
        if cpu_offload in (False, "false"):
            model.pop("cpu_offload", None)
        elif cpu_offload in (None, "model"):
            pass
        elif cpu_offload == "true":
            model["cpu_offload"] = True
        else:
            model["cpu_offload"] = cpu_offload

    if variant_spec["kind"] == "native_negative_prompt":
        native_negative_prompt = {"prompt": job["negative_prompt"]}
    else:
        native_negative_prompt = {}

    if variant_spec["kind"] == "baseline":
        steering = {"mode": "none", "enabled": False}
    elif variant_spec["kind"] == "conceptsteer":
        steering = _steering_config(variant_spec)
    else:
        steering = {"mode": "none", "enabled": False}

    override = {
        "project": {"name": benchmark_name, "seed": seed},
        "runtime": {
            "device": job.get("runtime", {}).get("device", "cuda"),
            "dtype": model.get("torch_dtype", "bfloat16"),
        },
        "model": model,
        "generation": generation,
        "concepts": {"hierarchy_path": job["concept_tree"]},
        "steering": steering,
        "native_negative_prompt": native_negative_prompt,
        "logging": {
            "output_dir": job["output_dir"],
            "tensorboard": job.get("logging", {}).get("tensorboard", True),
            "level": "INFO",
        },
        "output": job["output"],
        "benchmark": {
            "name": benchmark_name,
            "stage": job["stage"],
            "variant": job["variant"],
            "prompt_id": job["prompt_id"],
            "seed": seed,
            "negative_prompt": job["negative_prompt"],
            "active_pair_ids": variant_spec.get("active_pair_ids", []),
            "strength": variant_spec.get("strength"),
            "margin": variant_spec.get("margin"),
            "tau": variant_spec.get("tau"),
            "schedule": variant_spec.get("schedule"),
            "schedule_window": variant_spec.get("schedule_window"),
            "local_mask": variant_spec.get("local_mask"),
            "normalize_directions": variant_spec.get("normalize_directions"),
            "prompt_composition": variant_spec.get("prompt_composition"),
            "step_stride": variant_spec.get("step_stride", 1),
        },
    }
    config = deep_merge(base, override)
    config["generation"] = _sanitize_generation_for_task(dict(config.get("generation", {})))
    return config


def _steering_config(variant_spec: dict[str, Any]) -> dict[str, Any]:
    start_fraction, end_fraction = variant_spec["schedule_window"]
    return {
        "mode": "bottleneck",
        "enabled": True,
        "start_fraction": start_fraction,
        "end_fraction": end_fraction,
        "lambda_schedule": {
            "kind": "constant",
            "max_value": variant_spec["strength"],
            "min_value": variant_spec["strength"],
        },
        "margin": variant_spec["margin"],
        "active_pair_ids": list(variant_spec["active_pair_ids"]),
        "normalize_directions": bool(variant_spec["normalize_directions"]),
        "prompt_composition": variant_spec.get("prompt_composition", "append"),
        "step_stride": int(variant_spec.get("step_stride", 1)),
        "pair_overrides": variant_spec.get("pair_overrides", {}),
        "mask": {
            "enabled": bool(variant_spec["local_mask"]),
            "mode": "max_normalized",
            "threshold": variant_spec["tau"],
            "percentile": 0.85,
            "eps": 1.0e-6,
        },
    }


def _load_prompt_suite(path: Path) -> dict[str, dict[str, Any]]:
    data = load_yaml(path)
    if data.get("benchmark") != BENCHMARK_NAME:
        raise ValueError(f"Prompt suite benchmark mismatch in {path}.")
    rows = data.get("prompts")
    if not isinstance(rows, list):
        raise ValueError(f"Prompt suite must contain a prompts list: {path}")
    prompts: dict[str, dict[str, Any]] = {}
    for row in rows:
        prompt_id = str(row["prompt_id"])
        prompt = str(row["prompt"])
        prompts[prompt_id] = dict(row)
        prompts[prompt_id]["prompt"] = prompt
    missing = [prompt_id for prompt_id in ALL_PROMPT_IDS if prompt_id not in prompts]
    if missing:
        raise ValueError(f"Prompt suite is missing required prompt IDs: {missing}")
    main_prompt = prompts["P0_all_sources_main"]["prompt"]
    if "sad adult" not in main_prompt or "sitting" not in main_prompt:
        raise ValueError("P0_all_sources_main must explicitly contain 'sad adult' and 'sitting'.")
    return prompts


def _load_negative_prompts(path: Path) -> dict[str, Any]:
    data = load_yaml(path)
    if data.get("benchmark") != BENCHMARK_NAME:
        raise ValueError(f"Negative prompt config benchmark mismatch in {path}.")
    variants = data.get("variants")
    if not isinstance(variants, dict):
        raise ValueError(f"Negative prompt config must contain variants mapping: {path}")
    return variants


def _resolve_model_config(root: Path, model_name: str) -> tuple[Path, str]:
    candidates = []
    for path in sorted((root / "configs/models").glob("*.yaml")):
        data = load_yaml(path)
        model = data.get("model", {})
        stem_aliases = {path.stem, path.stem.removeprefix("t2i_"), path.stem.removeprefix("t2v_")}
        aliases = stem_aliases | {
            str(model.get("adapter", "")),
            str(model.get("model_id", "")),
        }
        if model_name in aliases:
            candidates.append((path, path.stem.removeprefix("t2i_").removeprefix("t2v_")))
    if not candidates:
        raise ValueError(f"Could not resolve model {model_name!r} under configs/models.")
    if len(candidates) > 1:
        exact = [item for item in candidates if item[1] == model_name or item[0].stem == model_name]
        if len(exact) == 1:
            return exact[0]
        raise ValueError(f"Model name {model_name!r} is ambiguous: {[str(item[0]) for item in candidates]}")
    return candidates[0]


def _stage_plan(
    stage: str,
    variants: list[str] | None,
    prompt_ids: list[str] | None,
) -> list[tuple[str, list[str], tuple[str, ...]]]:
    if variants is not None:
        return [(f"custom_stage_{stage}", variants, tuple(prompt_ids or ALL_PROMPT_IDS))]
    if prompt_ids is not None:
        selected = tuple(prompt_ids)
    else:
        selected = ()
    plans = {
        "0": [("stage0_smoke", ["baseline", "negativeprompt", "conceptsteer_full_default"], ("P0_all_sources_main",))],
        "1": [("stage1_core", ["baseline", "negativeprompt", "negative_prompt_global", "conceptsteer_full_default"], selected or ALL_PROMPT_IDS)],
        "2": [("stage2_single_pair_ablations", [
            "conceptsteer_emotion_only",
            "conceptsteer_pose_only",
            "conceptsteer_color_only",
            "conceptsteer_action_only",
            "conceptsteer_composition_only",
            "conceptsteer_pose_action_only",
            "conceptsteer_emotion_pose_only",
            "conceptsteer_emotion_pose_action_only",
            "conceptsteer_full",
        ], selected or ALL_PROMPT_IDS)],
        "3": [("stage3_schedule_ablation", [
            "conceptsteer_full_early",
            "conceptsteer_full_middle",
            "conceptsteer_full_late",
            "conceptsteer_full_full_window",
        ], selected or CORE_PROMPT_IDS)],
        "4": [("stage4_strength_sweep", [
            "conceptsteer_full_strength_0p25",
            "conceptsteer_full_strength_0p50",
            "conceptsteer_full_strength_1p00",
            "conceptsteer_full_strength_1p50",
            "conceptsteer_full_strength_2p00",
        ], selected or CORE_PROMPT_IDS)],
        "5": [("stage5_margin_threshold_sweep", [
            "conceptsteer_full_margin_0p00_tau_0p05",
            "conceptsteer_full_margin_0p00_tau_0p10",
            "conceptsteer_full_margin_0p00_tau_0p20",
            "conceptsteer_full_margin_0p05_tau_0p05",
            "conceptsteer_full_margin_0p05_tau_0p10",
            "conceptsteer_full_margin_0p05_tau_0p20",
            "conceptsteer_full_margin_0p10_tau_0p05",
            "conceptsteer_full_margin_0p10_tau_0p10",
            "conceptsteer_full_margin_0p10_tau_0p20",
        ], selected or ("P0_all_sources_main",))],
        "6": [("stage6_masking_normalization", [
            "conceptsteer_full_mask_on_norm_on",
            "conceptsteer_full_mask_off_norm_on",
            "conceptsteer_full_mask_on_norm_off",
            "conceptsteer_full_mask_off_norm_off",
        ], selected or CONTROL_PROMPT_IDS)],
    }
    if stage == "all":
        rows: list[tuple[str, list[str], tuple[str, ...]]] = []
        for key in ("0", "1", "2", "3", "4", "5", "6"):
            rows.extend(plans[key])
        return rows
    return plans[stage]


def _variant_spec(name: str, model_name: str | None = None) -> dict[str, Any]:
    if name == "baseline":
        return {"kind": "baseline"}
    if name in {"negativeprompt", "negative_prompt_matched", "negative_prompt_global"}:
        return {"kind": "native_negative_prompt"}

    spec = {
        "kind": "conceptsteer",
        "active_pair_ids": PAIR_GROUPS["full"],
        "strength": 1.0,
        "margin": 0.05,
        "tau": 0.10,
        "schedule": "full_window",
        "schedule_window": SCHEDULE_WINDOWS["full_window"],
        "local_mask": False,
        "normalize_directions": False,
        "prompt_composition": "concept_only",
        "step_stride": 1,
    }
    if name == "conceptsteer_full_default":
        spec.update(_model_steering_overrides(model_name))
        return spec

    for stride in (2, 3, 4, 5, 8, 10):
        if name == f"conceptsteer_full_default_sparse_stride_{stride}":
            spec.update(_model_steering_overrides(model_name))
            spec["step_stride"] = stride
            return spec
        if name == f"conceptsteer_no_composition_default_sparse_stride_{stride}":
            spec.update(_model_steering_overrides(model_name))
            spec["active_pair_ids"] = PAIR_GROUPS["emotion_pose_color_action"]
            spec["step_stride"] = stride
            return spec
        for strength in (1.50, 2.00, 3.00):
            if name == f"conceptsteer_full_default_sparse_stride_{stride}_strength_{_float_id(strength)}":
                spec.update(_model_steering_overrides(model_name))
                spec["strength"] = strength
                spec["step_stride"] = stride
                return spec
            if name == (
                f"conceptsteer_no_composition_default_sparse_stride_{stride}_strength_{_float_id(strength)}"
            ):
                spec.update(_model_steering_overrides(model_name))
                spec["active_pair_ids"] = PAIR_GROUPS["emotion_pose_color_action"]
                spec["strength"] = strength
                spec["step_stride"] = stride
                return spec
            if name == (
                f"conceptsteer_no_composition_default_sparse_stride_{stride}_strength_{_float_id(strength)}"
                "_pose_action_weight3"
            ):
                spec.update(_model_steering_overrides(model_name))
                spec["active_pair_ids"] = PAIR_GROUPS["emotion_pose_color_action"]
                spec["strength"] = strength
                spec["step_stride"] = stride
                spec["pair_overrides"] = {
                    "body_pose_sitting_to_walking": {
                        "start_fraction": 0.0,
                        "end_fraction": 1.0,
                        "weight": 3.0,
                    },
                    "sandwich_action_eating_to_holding": {
                        "start_fraction": 0.0,
                        "end_fraction": 1.0,
                        "weight": 3.0,
                    },
                }
                return spec

    aliases = {
        "conceptsteer_full": "full",
        "conceptsteer_emotion": "emotion",
        "conceptsteer_emotion_only": "emotion",
        "conceptsteer_pose": "pose",
        "conceptsteer_pose_only": "pose",
        "conceptsteer_color": "color",
        "conceptsteer_color_only": "color",
        "conceptsteer_action": "action",
        "conceptsteer_action_only": "action",
        "conceptsteer_composition": "composition",
        "conceptsteer_composition_only": "composition",
        "conceptsteer_pose_action": "pose_action",
        "conceptsteer_pose_action_only": "pose_action",
        "conceptsteer_emotion_pose_only": "emotion_pose",
        "conceptsteer_emotion_pose_action_only": "emotion_pose_action",
        "conceptsteer_no_composition": "emotion_pose_color_action",
        "conceptsteer_emotion_pose_color_action_only": "emotion_pose_color_action",
    }
    if name in aliases:
        spec["active_pair_ids"] = PAIR_GROUPS[aliases[name]]
        return spec

    for schedule in SCHEDULE_WINDOWS:
        if name == f"conceptsteer_full_{schedule}":
            spec["schedule"] = schedule
            spec["schedule_window"] = SCHEDULE_WINDOWS[schedule]
            return spec
    for strength in (0.25, 0.50, 0.75, 1.00, 1.50, 2.00):
        if name == f"conceptsteer_full_strength_{_float_id(strength)}":
            spec["strength"] = strength
            return spec
    for margin in (0.00, 0.05, 0.10):
        for tau in (0.05, 0.10, 0.20):
            if name == f"conceptsteer_full_margin_{_float_id(margin)}_tau_{_float_id(tau)}":
                spec["margin"] = margin
                spec["tau"] = tau
                return spec
    mask_norm = {
        "conceptsteer_full_mask_on_norm_on": (True, True),
        "conceptsteer_full_mask_off_norm_on": (False, True),
        "conceptsteer_full_mask_on_norm_off": (True, False),
        "conceptsteer_full_mask_off_norm_off": (False, False),
    }
    if name in mask_norm:
        spec["local_mask"], spec["normalize_directions"] = mask_norm[name]
        return spec
    if name == "conceptsteer_full_append_mask_on_norm_on_middle":
        spec["local_mask"] = True
        spec["normalize_directions"] = True
        spec["prompt_composition"] = "append"
        spec["schedule"] = "middle"
        spec["schedule_window"] = SCHEDULE_WINDOWS["middle"]
        return spec
    if name == "conceptsteer_full_concept_only_mask_off_norm_off":
        spec["local_mask"] = False
        spec["normalize_directions"] = False
        spec["prompt_composition"] = "concept_only"
        return spec
    if name == "conceptsteer_full_concept_only_mask_off_norm_off_full_window":
        spec["local_mask"] = False
        spec["normalize_directions"] = False
        spec["prompt_composition"] = "concept_only"
        spec["schedule"] = "full_window"
        spec["schedule_window"] = SCHEDULE_WINDOWS["full_window"]
        return spec
    if name == "conceptsteer_full_append_mask_off_norm_off_full_window":
        spec["local_mask"] = False
        spec["normalize_directions"] = False
        spec["prompt_composition"] = "append"
        spec["schedule"] = "full_window"
        spec["schedule_window"] = SCHEDULE_WINDOWS["full_window"]
        return spec
    if name == "conceptsteer_full_append_strength_2p00_mask_off_norm_off_full_window":
        spec["strength"] = 2.0
        spec["local_mask"] = False
        spec["normalize_directions"] = False
        spec["prompt_composition"] = "append"
        spec["schedule"] = "full_window"
        spec["schedule_window"] = SCHEDULE_WINDOWS["full_window"]
        return spec
    for strength in (0.05, 0.10, 0.25, 0.50, 0.75, 1.00):
        if name == f"conceptsteer_full_append_strength_{_float_id(strength)}_mask_off_norm_off_full_window":
            spec["strength"] = strength
            spec["local_mask"] = False
            spec["normalize_directions"] = False
            spec["prompt_composition"] = "append"
            spec["schedule"] = "full_window"
            spec["schedule_window"] = SCHEDULE_WINDOWS["full_window"]
            return spec
        if name == (
            f"conceptsteer_no_composition_append_strength_{_float_id(strength)}"
            "_mask_off_norm_off_full_window"
        ):
            spec["strength"] = strength
            spec["active_pair_ids"] = PAIR_GROUPS["emotion_pose_color_action"]
            spec["local_mask"] = False
            spec["normalize_directions"] = False
            spec["prompt_composition"] = "append"
            spec["schedule"] = "full_window"
            spec["schedule_window"] = SCHEDULE_WINDOWS["full_window"]
            return spec
    if name in {
        "conceptsteer_qwen_append_no_composition_pose_action_weight2",
        "conceptsteer_qwen_append_no_composition_pose_action_weight3",
        "conceptsteer_qwen_concept_only_no_composition_pose_action_weight2",
        "conceptsteer_qwen_concept_only_no_composition_pose_action_weight3",
    }:
        weight = 2.0 if name.endswith("weight2") else 3.0
        spec["strength"] = 2.0
        spec["active_pair_ids"] = PAIR_GROUPS["emotion_pose_color_action"]
        spec["local_mask"] = False
        spec["normalize_directions"] = False
        spec["prompt_composition"] = "concept_only" if "_concept_only_" in name else "append"
        spec["schedule"] = "full_window"
        spec["schedule_window"] = SCHEDULE_WINDOWS["full_window"]
        spec["pair_overrides"] = {
            "body_pose_sitting_to_walking": {
                "start_fraction": 0.0,
                "end_fraction": 1.0,
                "weight": weight,
            },
            "sandwich_action_eating_to_holding": {
                "start_fraction": 0.0,
                "end_fraction": 1.0,
                "weight": weight,
            },
        }
        return spec
    for strength in (0.25, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00):
        if name == (
            f"conceptsteer_no_composition_concept_only_strength_{_float_id(strength)}"
            "_mask_off_norm_off_full_window"
        ):
            spec["strength"] = strength
            spec["active_pair_ids"] = PAIR_GROUPS["emotion_pose_color_action"]
            spec["local_mask"] = False
            spec["normalize_directions"] = False
            spec["prompt_composition"] = "concept_only"
            spec["schedule"] = "full_window"
            spec["schedule_window"] = SCHEDULE_WINDOWS["full_window"]
            return spec
    if name == "conceptsteer_flux1_pose_action_full_emotion_late":
        spec["active_pair_ids"] = PAIR_GROUPS["emotion_pose_action"]
        spec["pair_overrides"] = {
            "facial_affect_sad_to_happy": {"start_fraction": 0.65, "end_fraction": 1.0},
            "body_pose_sitting_to_walking": {"start_fraction": 0.0, "end_fraction": 1.0},
            "sandwich_action_eating_to_holding": {"start_fraction": 0.0, "end_fraction": 1.0},
        }
        return spec
    if name == "conceptsteer_flux1_pose_action_full_emotion_middle":
        spec["active_pair_ids"] = PAIR_GROUPS["emotion_pose_action"]
        spec["pair_overrides"] = {
            "facial_affect_sad_to_happy": {"start_fraction": 0.25, "end_fraction": 0.75},
            "body_pose_sitting_to_walking": {"start_fraction": 0.0, "end_fraction": 1.0},
            "sandwich_action_eating_to_holding": {"start_fraction": 0.0, "end_fraction": 1.0},
        }
        return spec
    if name == "conceptsteer_flux1_no_composition_color_weight2":
        spec["active_pair_ids"] = PAIR_GROUPS["emotion_pose_color_action"]
        spec["pair_overrides"] = {
            "facial_affect_sad_to_happy": {"start_fraction": 0.25, "end_fraction": 0.75},
            "body_pose_sitting_to_walking": {"start_fraction": 0.0, "end_fraction": 1.0},
            "clothing_color_green_to_red_blue": {
                "start_fraction": 0.0,
                "end_fraction": 1.0,
                "weight": 2.0,
            },
            "sandwich_action_eating_to_holding": {"start_fraction": 0.0, "end_fraction": 1.0},
        }
        return spec
    if name == "conceptsteer_flux1_no_composition_color_weight3":
        spec["active_pair_ids"] = PAIR_GROUPS["emotion_pose_color_action"]
        spec["pair_overrides"] = {
            "facial_affect_sad_to_happy": {"start_fraction": 0.25, "end_fraction": 0.75},
            "body_pose_sitting_to_walking": {"start_fraction": 0.0, "end_fraction": 1.0},
            "clothing_color_green_to_red_blue": {
                "start_fraction": 0.0,
                "end_fraction": 1.0,
                "weight": 3.0,
            },
            "sandwich_action_eating_to_holding": {"start_fraction": 0.0, "end_fraction": 1.0},
        }
        return spec
    if name == "conceptsteer_flux1_no_composition_color_late_weight2":
        spec["active_pair_ids"] = PAIR_GROUPS["emotion_pose_color_action"]
        spec["pair_overrides"] = {
            "facial_affect_sad_to_happy": {"start_fraction": 0.25, "end_fraction": 0.75},
            "body_pose_sitting_to_walking": {"start_fraction": 0.0, "end_fraction": 1.0},
            "clothing_color_green_to_red_blue": {
                "start_fraction": 0.65,
                "end_fraction": 1.0,
                "weight": 2.0,
            },
            "sandwich_action_eating_to_holding": {"start_fraction": 0.0, "end_fraction": 1.0},
        }
        return spec
    if name == "conceptsteer_flux1_no_composition_color_late_weight3":
        spec["active_pair_ids"] = PAIR_GROUPS["emotion_pose_color_action"]
        spec["pair_overrides"] = {
            "facial_affect_sad_to_happy": {"start_fraction": 0.25, "end_fraction": 0.75},
            "body_pose_sitting_to_walking": {"start_fraction": 0.0, "end_fraction": 1.0},
            "clothing_color_green_to_red_blue": {
                "start_fraction": 0.65,
                "end_fraction": 1.0,
                "weight": 3.0,
            },
            "sandwich_action_eating_to_holding": {"start_fraction": 0.0, "end_fraction": 1.0},
        }
        return spec
    raise ValueError(f"Unknown benchmark variant: {name}")


def _negative_prompt_for(variant: str, prompt_id: str, negative_prompts: dict[str, Any]) -> str:
    if variant in {"negativeprompt", "negative_prompt_matched"}:
        matched = negative_prompts.get("negative_prompt_matched", {})
        if prompt_id not in matched:
            raise ValueError(f"Matched negative prompt missing for {prompt_id}.")
        return str(matched[prompt_id])
    if variant == "negative_prompt_global":
        global_cfg = negative_prompts.get("negative_prompt_global", {})
        return str(global_cfg["prompt"])
    return ""


def _validate_prompt_ids(prompt_ids: tuple[str, ...], prompts: dict[str, Any]) -> tuple[str, ...]:
    missing = [prompt_id for prompt_id in prompt_ids if prompt_id not in prompts]
    if missing:
        raise ValueError(f"Unknown prompt IDs: {missing}")
    return prompt_ids


def _validate_media_outputs(output_dir: Path) -> list[str]:
    media_suffixes = {".png", ".jpg", ".jpeg", ".webp", ".mp4", ".gif", ".pt"}
    paths = sorted(path for path in output_dir.rglob("*") if path.is_file() and path.suffix.lower() in media_suffixes)
    if not paths:
        raise RuntimeError(f"No decoded media file was produced under {output_dir}.")
    empty = [str(path) for path in paths if path.stat().st_size <= 0]
    if empty:
        raise RuntimeError(f"Generated media files are empty: {empty}")
    return [str(path) for path in paths]


def _write_job_notes(
    output_dir: Path,
    job: dict[str, Any],
    status: str,
    payload: dict[str, Any] | None = None,
) -> None:
    variant_spec = dict(job.get("variant_spec", {}) or {})
    generation = dict(job.get("generation", {}) or {})
    output_paths = _collect_output_paths_from_payload(payload or {})
    media_paths = payload.get("validated_media_paths", []) if payload else []
    reason = payload.get("reason") if payload else None
    error = payload.get("error") if payload else None
    error_type = payload.get("error_type") if payload else None
    lines = [
        f"# {job.get('model_name')} / {job.get('variant')} / {job.get('prompt_id')}",
        "",
        f"Status: `{status}`",
        "",
        "## Run Purpose",
        "",
        _variant_purpose(job),
        "",
        "## User-Changeable Parameters",
        "",
        f"- `model_name`: `{job.get('model_name')}`",
        f"- `model_config`: `{job.get('model_config')}`",
        f"- `variant`: `{job.get('variant')}`",
        f"- `prompt_id`: `{job.get('prompt_id')}`",
        f"- `seed`: `{job.get('seed')}`",
        f"- `task`: `{generation.get('task')}`",
        f"- `height`: `{generation.get('height')}`",
        f"- `width`: `{generation.get('width')}`",
        f"- `num_inference_steps`: `{generation.get('num_inference_steps')}`",
        f"- `guidance_scale`: `{generation.get('guidance_scale')}`",
        f"- `num_outputs_per_prompt`: `{generation.get('num_outputs_per_prompt')}`",
    ]
    if generation.get("task") == "text_to_video":
        lines.extend(
            [
                f"- `duration_seconds`: `{generation.get('duration_seconds')}`",
                f"- `num_frames`: `{generation.get('num_frames')}`",
                f"- `fps`: `{generation.get('fps')}`",
            ]
        )
    lines.extend(
        [
            f"- `negative_prompt`: `{job.get('negative_prompt')}`",
            f"- `concept_tree`: `{job.get('concept_tree')}`",
            "",
            "## Steering / Variant Hyperparameters",
            "",
        ]
    )
    if variant_spec.get("kind") == "conceptsteer":
        for key in (
            "active_pair_ids",
            "strength",
            "margin",
            "tau",
            "schedule",
            "schedule_window",
            "local_mask",
            "normalize_directions",
            "prompt_composition",
            "step_stride",
            "pair_overrides",
        ):
            if key in variant_spec:
                lines.append(f"- `{key}`: `{variant_spec.get(key)}`")
    else:
        lines.append(f"- `kind`: `{variant_spec.get('kind')}`")
    lines.extend(
        [
            "",
            "## Prompt",
            "",
            str(job.get("prompt", "")),
            "",
            "## Expected Change",
            "",
            _expected_change(job),
            "",
            "## Output Files",
            "",
        ]
    )
    if media_paths:
        lines.extend(f"- media: `{path}`" for path in media_paths)
    for key, value in output_paths.items():
        lines.append(f"- `{key}`: `{value}`")
    run_level_paths = _existing_run_level_paths(output_dir)
    for key, value in run_level_paths.items():
        lines.append(f"- `{key}`: `{value}`")
    if not media_paths and not output_paths and not run_level_paths:
        lines.append("- No output files have been produced yet.")
    lines.extend(
        [
            "",
            "## Observation",
            "",
            _observation_placeholder(status, reason=reason, error_type=error_type, error=error),
            "",
            "## Next Step",
            "",
            _next_step(status, job, reason=reason, error=error),
            "",
        ]
    )
    write_text(output_dir / "RUN.md", "\n".join(lines))


def _collect_output_paths_from_payload(payload: dict[str, Any]) -> dict[str, str]:
    records: list[dict[str, Any]] = []
    if isinstance(payload.get("records"), list):
        records = payload["records"]
    result = payload.get("result")
    if isinstance(result, dict) and isinstance(result.get("records"), list):
        records = result["records"]
    paths: dict[str, str] = {}
    for record in records:
        sample_id = str(record.get("sample_id", "sample"))
        for key, value in dict(record.get("output_paths", {}) or {}).items():
            paths[f"{sample_id}.{key}"] = str(value)
    return paths


def _existing_run_level_paths(output_dir: Path) -> dict[str, str]:
    paths: dict[str, str] = {}
    for key, filename in (
        ("benchmark_job", "benchmark_job.yaml"),
        ("benchmark_job_result", "benchmark_job_result.json"),
        ("run_timing", "run_timing.json"),
        ("resolved_config", "resolved_config.yaml"),
        ("run_log", "run.log"),
        ("system_info", "system_info.json"),
    ):
        path = output_dir / filename
        if path.exists():
            paths[key] = str(path)
    return paths


def _variant_purpose(job: dict[str, Any]) -> str:
    kind = job.get("variant_spec", {}).get("kind")
    if kind == "baseline":
        return "Baseline run: measure the unmodified model's response to the source prompt before any steering."
    if kind == "native_negative_prompt":
        return (
            "Native negative-prompt run: test whether the vendor pipeline can suppress the source "
            "attributes using its own `negative_prompt` interface."
        )
    return (
        "Concept steering run: test whether vector-field steering transfers the requested target "
        "attributes while preserving the person, park, bench, and sandwich."
    )


def _expected_change(job: dict[str, Any]) -> str:
    kind = job.get("variant_spec", {}).get("kind")
    if kind == "baseline":
        return "Expected to preserve the source prompt: sad adult, green jacket, seated on bench, eating sandwich."
    if kind == "native_negative_prompt":
        return (
            "Expected to reduce the source attributes named in the negative prompt if this model "
            "natively supports negative prompting."
        )
    active_pairs = set(job.get("variant_spec", {}).get("active_pair_ids", []) or [])
    targets = []
    if "facial_affect_sad_to_happy" in active_pairs:
        targets.append("happier expression instead of sad expression")
    if "clothing_color_green_to_red_blue" in active_pairs:
        targets.append("red-blue clothing instead of green clothing")
    if "body_pose_sitting_to_walking" in active_pairs:
        targets.append("walking/upright pose beside the bench instead of sitting")
    if "sandwich_action_eating_to_holding" in active_pairs:
        targets.append("holding/carrying the sandwich rather than biting/eating it")
    if "composition_static_to_dynamic" in active_pairs:
        targets.append("more dynamic park snapshot instead of a static seated portrait")
    if not targets:
        return "Expected target transfer for the active concept pairs while preserving person, park, bench, and sandwich."
    return "Expected target transfer for active pairs: " + "; ".join(targets) + "."


def _observation_placeholder(
    status: str,
    reason: str | None = None,
    error_type: str | None = None,
    error: str | None = None,
) -> str:
    if status == "completed":
        return "Pending visual inspection of decoded media."
    if status == "not_supported":
        return f"Not visually inspected because the run is not supported: `{reason}`."
    if status == "failed":
        return f"Run failed before usable visual output: `{error_type}` - `{error}`."
    return "Run is in progress; observation will be updated after completion and visual inspection."


def _next_step(
    status: str,
    job: dict[str, Any],
    reason: str | None = None,
    error: str | None = None,
) -> str:
    if status == "completed":
        return "Extract representative frames, visually inspect prompt alignment, then compare against baseline and concept-steering runs."
    if status == "not_supported":
        return "Skip this negative-prompt variant for this model and compare baseline against concept steering."
    if status == "failed":
        return "Fix the root cause in code or configuration, regenerate this same job, and keep the output folder layout unchanged."
    del job, reason, error
    return "Monitor the job until it completes or fails."


def _explicit_csv(value: str) -> list[str]:
    rows = [item.strip() for item in value.split(",") if item.strip()]
    if not rows:
        raise ValueError(f"Empty comma-separated list: {value!r}")
    return rows


def _float_id(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def _selected_model_names(root: Path, model: str, models: str | None) -> list[str]:
    if not models:
        return [model]
    if models.strip().lower() == "all":
        return [
            path.stem.removeprefix("t2i_").removeprefix("t2v_")
            for path in sorted((root / "configs/models").glob("*.yaml"))
        ]
    return _explicit_csv(models)


def _runtime_config(args: argparse.Namespace, model_config: dict[str, Any]) -> dict[str, Any]:
    runtime = {
        "device": args.device,
        "dtype": args.dtype,
    }
    model_cpu_offload = model_config.get("model", {}).get("cpu_offload")
    if args.cpu_offload in (None, "model"):
        if model_cpu_offload is not None:
            runtime["cpu_offload"] = model_cpu_offload
        return runtime
    if args.cpu_offload == "true":
        runtime["cpu_offload"] = True
    elif args.cpu_offload == "false":
        runtime["cpu_offload"] = False
    else:
        runtime["cpu_offload"] = args.cpu_offload
    return runtime


def _generation_config(args: argparse.Namespace, model_config: dict[str, Any]) -> dict[str, Any]:
    model_generation = dict(model_config.get("generation", {}))
    task = str(model_generation.get("task", "text_to_image"))
    duration_seconds = (
        float(args.duration_seconds)
        if args.duration_seconds is not None
        else (
            float(model_generation["duration_seconds"])
            if "duration_seconds" in model_generation and model_generation["duration_seconds"] is not None
            else None
        )
    )
    generation = {
        "task": task,
        "num_inference_steps": int(args.num_inference_steps or model_generation.get("num_inference_steps", 28)),
        "height": int(args.height) if args.height is not None else int(model_generation.get("height", 1024)),
        "width": int(args.width) if args.width is not None else int(model_generation.get("width", 1024)),
        "guidance_scale": float(args.guidance_scale)
        if args.guidance_scale is not None
        else float(model_generation.get("guidance_scale", 4.0)),
    }
    if task == "text_to_video":
        fps = int(args.fps) if args.fps is not None else int(model_generation.get("fps", 16))
        if args.num_frames is not None:
            num_frames = int(args.num_frames)
        elif "num_frames" in model_generation and model_generation["num_frames"] is not None:
            num_frames = int(model_generation["num_frames"])
        elif duration_seconds is not None:
            num_frames = int(round(duration_seconds * fps))
        else:
            num_frames = 81
        if duration_seconds is not None and int(round(duration_seconds * fps)) != num_frames:
            raise ValueError(
                "Text-to-video generation has inconsistent duration/fps/frame settings: "
                f"duration_seconds={duration_seconds}, fps={fps}, num_frames={num_frames}."
            )
        generation["num_frames"] = num_frames
        generation["fps"] = fps
        if duration_seconds is not None:
            generation["duration_seconds"] = duration_seconds
    return generation


def _sanitize_generation_for_task(generation: dict[str, Any]) -> dict[str, Any]:
    task = str(generation.get("task", "text_to_image"))
    if task == "text_to_image":
        for key in ("num_frames", "fps", "duration_seconds", "frame_rate"):
            generation.pop(key, None)
    return generation


def _model_steering_overrides(model_name: str | None) -> dict[str, Any]:
    if model_name == "flux1_dev":
        return {
            "active_pair_ids": PAIR_GROUPS["emotion_pose_action"],
            "pair_overrides": {
                "facial_affect_sad_to_happy": {"start_fraction": 0.25, "end_fraction": 0.75},
                "body_pose_sitting_to_walking": {"start_fraction": 0.0, "end_fraction": 1.0},
                "sandwich_action_eating_to_holding": {"start_fraction": 0.0, "end_fraction": 1.0},
            },
        }
    if model_name in {"qwen_image", "qwen_image_2512"}:
        return {
            "strength": 2.0,
            "prompt_composition": "append",
            "local_mask": False,
            "normalize_directions": False,
            "schedule": "full_window",
            "schedule_window": SCHEDULE_WINDOWS["full_window"],
        }
    if model_name == "sd35_large":
        return {
            "strength": 0.5,
        }
    if model_name in {"hunyuan_video", "wan22_t2v_a14b"}:
        return {
            "step_stride": 10,
        }
    return {}


if __name__ == "__main__":
    main()
