from __future__ import annotations

import argparse
import inspect
import os
import traceback
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch

from hierasafe_flow.generation.runner import GenerationRunner
from hierasafe_flow.generation.save_outputs import save_generation_output, save_generation_report
from hierasafe_flow.utils.config import deep_merge, get_path, load_config
from hierasafe_flow.utils.device import resolve_device, resolve_dtype
from hierasafe_flow.utils.io import ensure_dir, write_json, write_text, write_yaml
from hierasafe_flow.utils.memory import clear_cuda_cache
from hierasafe_flow.utils.seed import make_generator, seed_everything


CONDITIONS = (
    {
        "name": "baseline_trace_only",
        "description": "Real-model baseline latent/vector trace. Decoding disabled to avoid saving explicit nude sexual media.",
        "override": {
            "steering": {"mode": "none", "enabled": False},
            "output": {"decode": False, "save_latents": True, "save_traces": True},
        },
    },
    {
        "name": "native_negative_prompt",
        "description": "Native diffusers negative_prompt baseline on the unmodified model, used only when the pipeline explicitly supports negative_prompt.",
        "native_negative_prompt": True,
    },
    {
        "name": "concept_bottleneck",
        "description": "Hierarchical local concept vector-field bottleneck steering.",
        "override": {
            "steering": {"mode": "bottleneck", "enabled": True},
            "output": {"decode": True, "save_latents": True, "save_traces": True},
        },
    },
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run red-team tri-condition real-model jobs.")
    parser.add_argument("--grid", required=True, help="Grid YAML with model config list and attempt metadata.")
    parser.add_argument("--only", action="append", default=[], help="Optional model name filter. Repeatable.")
    parser.add_argument(
        "--condition",
        action="append",
        default=[],
        choices=[condition["name"] for condition in CONDITIONS],
        help="Optional condition filter. Repeatable.",
    )
    parser.add_argument("--prompt", default=None, help="Optional prompt override.")
    parser.add_argument("--attempt-name", default=None)
    parser.add_argument("--report-path", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    grid_path = Path(args.grid)
    grid = load_config(grid_path)
    attempt = grid.get("attempt", {})
    attempt_name = args.attempt_name or str(attempt.get("name", "try_First"))
    report_path = Path(args.report_path or str(attempt.get("report_path", f"debugging/{attempt_name}.md")))
    prompt_label = args.prompt or (
        f"See {attempt['prompt_file']}" if attempt.get("prompt_file") else None
    )
    filters = set(args.only)
    condition_filters = set(args.condition)

    records: list[dict[str, Any]] = []
    for item in grid.get("models", []):
        name = str(item["name"])
        if filters and name not in filters:
            continue
        base_config = load_config(item["config"])
        model_records = run_model_conditions(
            name,
            base_config,
            attempt_name,
            args.prompt,
            condition_filters=condition_filters,
        )
        records.extend(model_records)
        write_attempt_report(report_path, attempt_name, records, prompt=prompt_label, final=False)

    write_json(Path("outputs/redteam") / attempt_name / "tri_condition_result.json", records)
    write_attempt_report(report_path, attempt_name, records, prompt=prompt_label, final=True)
    print(str(report_path))


def run_model_conditions(
    model_name: str,
    base_config: dict[str, Any],
    attempt_name: str,
    prompt: str | None,
    condition_filters: set[str] | None = None,
) -> list[dict[str, Any]]:
    records = []
    base_output = Path("outputs/redteam") / attempt_name / model_name
    prompts = _prompts_for_snapshot(base_config, prompt)
    active_conditions = [
        condition
        for condition in CONDITIONS
        if not condition_filters or condition["name"] in condition_filters
    ]
    write_model_config_snapshot(
        output_dir=base_output,
        model_name=model_name,
        attempt_name=attempt_name,
        base_config=base_config,
        prompts=prompts,
        conditions=active_conditions,
    )
    for condition in active_conditions:
        config = deepcopy(base_config)
        config = deep_merge(config, condition.get("override", {}))
        config = deep_merge(
            config,
            {
                "logging": {
                    "output_dir": str(base_output / condition["name"]),
                    "tensorboard": bool(get_path(config, "logging.tensorboard", True)),
                }
            },
        )
        record: dict[str, Any] = {
            "model": model_name,
            "condition": condition["name"],
            "description": condition["description"],
            "output_dir": get_path(config, "logging.output_dir"),
            "status": "started",
        }
        write_condition_config_snapshot(
            output_dir=Path(str(record["output_dir"])),
            model_name=model_name,
            condition_name=condition["name"],
            attempt_name=attempt_name,
            config=config,
            prompts=prompts,
            condition=condition,
        )
        runner: GenerationRunner | None = None
        try:
            if condition.get("policy_blocked"):
                record["status"] = "not_run_policy_boundary"
                record["reason"] = condition["reason"]
            elif condition.get("native_negative_prompt"):
                record.update(run_native_negative_prompt_baseline(config, prompt))
            else:
                runner = GenerationRunner(config)
                result = runner.run(prompt=prompt)
                record["status"] = "completed"
                record["result"] = asdict(result)
        except Exception as exc:
            record["status"] = "failed"
            record["error_type"] = type(exc).__name__
            record["error"] = str(exc)
            record["traceback"] = traceback.format_exc()
        finally:
            runner = None
            clear_cuda_cache()
        records.append(record)
    return records


def write_model_config_snapshot(
    output_dir: Path,
    model_name: str,
    attempt_name: str,
    base_config: dict[str, Any],
    prompts: list[str],
    conditions: list[dict[str, Any]],
) -> None:
    snapshot = {
        "schema_version": 1,
        "attempt_name": attempt_name,
        "model_name": model_name,
        "model": base_config.get("model", {}),
        "modality": get_path(base_config, "generation.task"),
        "generation": base_config.get("generation", {}),
        "steering": base_config.get("steering", {}),
        "native_negative_prompt": base_config.get("native_negative_prompt", {}),
        "concepts": base_config.get("concepts", {}),
        "prompts": prompts,
        "conditions": [
            {
                "name": condition["name"],
                "description": condition["description"],
                "policy_blocked": bool(condition.get("policy_blocked", False)),
                "native_negative_prompt": bool(condition.get("native_negative_prompt", False)),
                "override": condition.get("override", {}),
            }
            for condition in conditions
        ],
    }
    write_yaml(output_dir / "config.yaml", snapshot)


def write_condition_config_snapshot(
    output_dir: Path,
    model_name: str,
    condition_name: str,
    attempt_name: str,
    config: dict[str, Any],
    prompts: list[str],
    condition: dict[str, Any],
) -> None:
    snapshot = {
        "schema_version": 1,
        "attempt_name": attempt_name,
        "model_name": model_name,
        "condition_name": condition_name,
        "condition_description": condition["description"],
        "policy_blocked": bool(condition.get("policy_blocked", False)),
        "policy_block_reason": condition.get("reason"),
        "model": config.get("model", {}),
        "modality": get_path(config, "generation.task"),
        "generation": config.get("generation", {}),
        "output": config.get("output", {}),
        "steering": config.get("steering", {}),
        "concepts": config.get("concepts", {}),
        "prompts": prompts,
    }
    write_yaml(output_dir / "config.yaml", snapshot)


def _prompts_for_snapshot(config: dict[str, Any], prompt: str | None) -> list[str]:
    if prompt:
        return [prompt]
    prompt_file = get_path(config, "generation.prompt_file")
    if prompt_file and not Path(str(prompt_file)).is_absolute():
        prompt_file = str(Path(get_path(config, "_meta.project_root", Path.cwd())) / str(prompt_file))
    if prompt_file:
        return _read_prompts(str(prompt_file))
    return [str(get_path(config, "generation.prompt", ""))]


def run_native_negative_prompt_baseline(config: dict[str, Any], prompt: str | None) -> dict[str, Any]:
    import diffusers

    pipeline_class_name = str(get_path(config, "model.diffusers_pipeline_class"))
    if not hasattr(diffusers, pipeline_class_name):
        return {
            "status": "not_supported",
            "reason": f"diffusers does not expose {pipeline_class_name}",
        }
    pipeline_cls = getattr(diffusers, pipeline_class_name)
    call_signature = inspect.signature(pipeline_cls.__call__)
    if "negative_prompt" not in call_signature.parameters:
        return {
            "status": "not_supported",
            "reason": f"{pipeline_class_name}.__call__ does not expose negative_prompt",
        }

    seed = int(get_path(config, "project.seed", 1234))
    seed_everything(seed)
    device = resolve_device(str(get_path(config, "runtime.device", "cuda")))
    dtype = resolve_dtype(str(get_path(config, "runtime.dtype", get_path(config, "model.torch_dtype", "bfloat16"))))
    model_id = str(get_path(config, "model.model_id"))
    output_dir = Path(str(get_path(config, "logging.output_dir")))
    ensure_dir(output_dir)

    load_kwargs = dict(get_path(config, "model.load_kwargs", {}) or {})
    load_kwargs.setdefault("torch_dtype", dtype)
    load_kwargs.setdefault("low_cpu_mem_usage", True)
    if os.environ.get("HF_TOKEN") and "token" not in load_kwargs:
        load_kwargs["token"] = os.environ["HF_TOKEN"]
    pipe = pipeline_cls.from_pretrained(model_id, **load_kwargs)
    cpu_offload = get_path(config, "model.cpu_offload", False)
    if cpu_offload == "sequential" and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload(device=device)
    elif bool(cpu_offload) and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload(device=device)
    elif hasattr(pipe, "to"):
        pipe.to(device)

    generation = config.get("generation", {})
    prompt_file = generation.get("prompt_file")
    if prompt_file and not Path(str(prompt_file)).is_absolute():
        prompt_file = str(Path(get_path(config, "_meta.project_root", Path.cwd())) / str(prompt_file))
    selected_prompts = [prompt] if prompt else _read_prompts(str(prompt_file) if prompt_file else None)
    negative_prompt = str(
        get_path(
            config,
            "native_negative_prompt.prompt",
            (
                "explicit nudity, exposed intimate anatomy, pornographic content, sexualized bed scene, "
                "erotic pose, bare front body, fetish content, genital exposure, nipple exposure, "
                "see-through clothing, unsafe sexual content"
            ),
        )
    )
    sample_records = []
    for index, selected_prompt in enumerate(selected_prompts):
        sample_id = f"sample_{index:04d}"
        generator = make_generator(seed, device)
        call_kwargs = {
            "prompt": selected_prompt,
            "negative_prompt": negative_prompt,
            "height": generation.get("height"),
            "width": generation.get("width"),
            "num_frames": generation.get("num_frames"),
            "frame_rate": generation.get("fps"),
            "num_inference_steps": generation.get("num_inference_steps"),
            "guidance_scale": generation.get("guidance_scale"),
            "true_cfg_scale": get_path(config, "native_negative_prompt.true_cfg_scale"),
            "guidance_scale_2": get_path(config, "native_negative_prompt.guidance_scale_2"),
            "use_dynamic_cfg": get_path(config, "native_negative_prompt.use_dynamic_cfg"),
            "num_images_per_prompt": generation.get("num_outputs_per_prompt"),
            "num_videos_per_prompt": generation.get("num_outputs_per_prompt"),
            "generator": generator,
            "output_type": "pil",
            "return_dict": True,
        }
        call_kwargs = {
            key: value for key, value in call_kwargs.items() if key in call_signature.parameters and value is not None
        }
        with torch.inference_mode():
            output = pipe(**call_kwargs)

        media = _extract_pipeline_media(output)
        trace = [
            {
                "condition": "native_negative_prompt",
                "pipeline_class": pipeline_class_name,
                "negative_prompt": negative_prompt,
            }
        ]
        paths = save_generation_output(
            media=media,
            latents=torch.empty(0),
            trace=trace,
            output_dir=output_dir,
            sample_id=sample_id,
            task=str(generation.get("task", "text_to_image")),
            save_latents=False,
            save_traces=True,
            image_format=str(get_path(config, "output.image_format", "png")),
            video_format=str(get_path(config, "output.video_format", "mp4")),
            fps=int(generation.get("fps", 16)),
        )
        generation_config = dict(generation)
        generation_config["prompt"] = selected_prompt
        report = {
            "schema_version": 1,
            "prompt": selected_prompt,
            "sample_id": sample_id,
            "task": str(generation.get("task", "text_to_image")),
            "model": {
                "adapter": get_path(config, "model.adapter"),
                "model_id": model_id,
                "pipeline_class": pipeline_class_name,
            },
            "condition": {
                "steering_mode": "native_negative_prompt",
                "decode_outputs": True,
                "is_native_negative_prompt": True,
                "negative_prompt": negative_prompt,
            },
            "generation": generation_config,
            "output_paths": dict(paths),
            "interpretability": {
                "note": (
                    "Native negative-prompt baselines do not expose concept bottleneck activations "
                    "or steering deltas. The prompt, negative_prompt, and saved media/trace are "
                    "recorded for comparison."
                ),
                "timesteps": trace,
                "concepts_per_step": [],
            },
        }
        paths["report"] = save_generation_report(report, output_dir, sample_id)
        sample_records.append(
            {
                "prompt": selected_prompt,
                "sample_id": sample_id,
                "output_paths": paths,
            }
        )
    return {
        "status": "completed",
        "result": {
            "output_dir": str(output_dir),
            "records": sample_records,
        },
    }


def _read_prompts(path: str | None) -> list[str]:
    if not path:
        raise ValueError("No prompt was supplied and generation.prompt_file is empty.")
    prompts = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    if not prompts:
        raise ValueError(f"Prompt file contains no prompts: {path}")
    return prompts


def _extract_pipeline_media(output: Any) -> Any:
    if hasattr(output, "images"):
        return output.images
    if hasattr(output, "frames"):
        return output.frames
    if hasattr(output, "videos"):
        return output.videos
    if isinstance(output, tuple) and output:
        return output[0]
    return output


def write_attempt_report(
    path: str | Path,
    attempt_name: str,
    records: list[dict[str, Any]],
    prompt: str | None,
    final: bool,
) -> None:
    ensure_dir(Path(path).parent)
    prompt_line = prompt or "See configs/prompts/redteam_nudity_user_prompt.txt"
    lines = [
        f"# {attempt_name}",
        "",
        "## Prompt",
        "",
        f"`{prompt_line}`",
        "",
        "## Safety Boundary",
        "",
        (
            "The baseline condition is run as a real-model latent/vector trace with decoding disabled. "
            "The third condition uses the native diffusers `negative_prompt` API only when that API is "
            "explicitly present on the pipeline. Concept bottlenecking is the proposed method."
        ),
        "",
        "## Conditions",
        "",
    ]
    for record in records:
        lines.append(f"### {record['model']} / {record['condition']}")
        lines.append("")
        lines.append(f"Status: `{record['status']}`")
        lines.append("")
        lines.append(record["description"])
        lines.append("")
        lines.append(f"Output: `{record['output_dir']}`")
        lines.append("")
        if record["status"] == "failed":
            lines.append(f"Failure: `{record.get('error_type')}` - {record.get('error')}")
            lines.append("")
            if record.get("traceback"):
                lines.append("```text")
                lines.append(record["traceback"])
                lines.append("```")
                lines.append("")
        elif record["status"] == "not_supported":
            lines.append(f"Not supported: {record.get('reason')}")
            lines.append("")
        elif record["status"] == "not_run_policy_boundary":
            lines.append(f"Not run: {record.get('reason')}")
            lines.append("")
        elif record["status"] == "completed":
            result = record.get("result", {})
            for sample in result.get("records", []):
                lines.append(f"Sample `{sample['sample_id']}` paths:")
                for key, value in sample.get("output_paths", {}).items():
                    lines.append(f"- `{key}`: `{value}`")
                lines.append("")
    lines.extend(
        [
            "## Analysis",
            "",
            "Automated status analysis is available above. Visual quality and safety analysis should be filled after inspecting decoded safe-output media.",
            "",
            "## Next Improvements",
            "",
            "- Increase bottleneck lambda or margin if decoded safe-output conditions still contain target unsafe visual content.",
            "- For native negative-prompt baselines, tune only pipeline-supported negative-prompt arguments.",
            "- Reduce lambda or delay steering start if outputs are safe but prompt fidelity collapses.",
            "",
            f"Finalized: `{final}`",
            "",
        ]
    )
    write_text(path, "\n".join(lines))


if __name__ == "__main__":
    main()
