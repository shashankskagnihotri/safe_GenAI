from __future__ import annotations

import argparse
import inspect
import math
import os
import time
import traceback
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from hierasafe_flow.adapters.base import (
    configure_pipeline_vae_tiling,
    load_single_file_companion_components,
    resolve_single_file_checkpoint,
)
from hierasafe_flow.adapters.cogvideox_adapter import (
    COGVIDEOX_TEMPORAL_PROTOCOL_KEY,
    CogVideoXAdapter,
)
from hierasafe_flow.adapters.flux_dual_view_adapter import (
    FLUX_DUAL_VIEW_CONFIG_KEY,
    FluxDualViewAdapter,
)
from hierasafe_flow.adapters.hunyuan_video_adapter import (
    HUNYUAN_DUAL_VIEW_CONFIG_KEY,
    HUNYUAN_TEMPORAL_PROTOCOL_KEY,
    HunyuanVideoAdapter,
)
from hierasafe_flow.adapters.wan_adapter import (
    WAN_T2V_REVISION,
    WAN_TEMPORAL_PROTOCOL_KEY,
    WanAdapter,
    validate_wan_native_negative_prompt_cleaner,
)
from hierasafe_flow.generation.runner import (
    GenerationRunner,
    _bind_flux_dual_view_conditioning,
)
from hierasafe_flow.evaluation.flux1_dual_view_jobs_v3 import (
    CONTRACT_ID as FLUX1_DUAL_VIEW_ROUTE_CONTRACT_ID,
    NEGATIVE_MODE_EXPLICIT_NONE_CONTROL,
    NEGATIVE_MODE_NOT_APPLIED,
    NEGATIVE_MODE_PAIRED_REGISTERED,
    canonical_sha256 as flux1_dual_view_plan_sha256,
)
from hierasafe_flow.generation.save_outputs import save_generation_output, save_generation_report
from hierasafe_flow.logging_utils.experiment_tracker import ExperimentTracker
from hierasafe_flow.logging_utils.logger import setup_logger
from hierasafe_flow.logging_utils.system_info import collect_system_info
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


# This escape hatch is deliberately narrower than the generic native-negative
# interface.  It exists only for the sealed Flux.1 true-CFG calibration, whose
# same-pipeline control must pass ``negative_prompt=None`` explicitly while all
# other loader, call, extraction, and saving code remains identical to the
# registered-negative scale-1.0 sentinel.
_FLUX1_NEGATIVE_CALIBRATION_ID = "flux1_native_negative_true_cfg_scale_v1"
_FLUX1_NEGATIVE_CALIBRATION_STAGE = "flux1_native_negative_scale_calibration_v1"
_FLUX1_NEGATIVE_CALIBRATION_CONFIG_SHA256 = (
    "6f972bc66fe02732f2e175e82b92b011fc4cf0244e1adb99799bc54c222e41cd"
)
_FLUX1_NEGATIVE_CALIBRATION_SCALES = (
    1.0,
    1.25,
    1.5,
    1.75,
    2.0,
    2.25,
    2.5,
    2.75,
    3.0,
    3.25,
    3.5,
    3.75,
    4.0,
)
_FLUX1_NEGATIVE_CALIBRATION_V2_ID = "flux1_native_negative_true_cfg_scale_v2"
_FLUX1_NEGATIVE_CALIBRATION_V2_STAGE = "flux1_native_negative_scale_calibration_v2"
# This remains intentionally unset while the v2 protocol is a checked-in
# draft.  Sealing calibration v2 must pin the same raw YAML digest here and in
# its dedicated manifest builder before the explicit-null route can execute.
_FLUX1_NEGATIVE_CALIBRATION_V2_CONFIG_SHA256: str | None = None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run red-team tri-condition real-model jobs.")
    parser.add_argument(
        "--grid", required=True, help="Grid YAML with model config list and attempt metadata."
    )
    parser.add_argument(
        "--only", action="append", default=[], help="Optional model name filter. Repeatable."
    )
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
    report_path = Path(
        args.report_path or str(attempt.get("report_path", f"debugging/{attempt_name}.md"))
    )
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
        prompt_file = str(
            Path(get_path(config, "_meta.project_root", Path.cwd())) / str(prompt_file)
        )
    if prompt_file:
        return _read_prompts(str(prompt_file))
    return [str(get_path(config, "generation.prompt", ""))]


def run_native_negative_prompt_baseline(
    config: dict[str, Any], prompt: str | None
) -> dict[str, Any]:
    import diffusers

    output_dir = Path(str(get_path(config, "logging.output_dir")))
    ensure_dir(output_dir)
    run_started_at = _utc_now_iso()
    run_start = time.perf_counter()
    logger = setup_logger(
        output_dir=output_dir,
        level=str(get_path(config, "logging.level", "INFO")),
    )
    tracker = ExperimentTracker.create(output_dir, config)
    write_json(output_dir / "system_info.json", collect_system_info())

    pipeline_class_value = get_path(config, "model.diffusers_pipeline_class")
    pipeline_class_name = (
        pipeline_class_value.strip()
        if isinstance(pipeline_class_value, str) and pipeline_class_value.strip()
        else None
    )
    if pipeline_class_name is None:
        reason = (
            "model config does not define a native diffusers pipeline class; "
            "native negative prompting is unavailable for this adapter"
        )
        logger.info("Native negative prompt not supported: %s", reason)
        tracker.log_event(
            "native_negative_prompt_not_supported",
            {"reason": reason},
        )
        _write_native_run_timing(
            output_dir=output_dir,
            status="not_supported",
            started_at=run_started_at,
            run_start=run_start,
            config=config,
            model_id=str(get_path(config, "model.model_id")),
            pipeline_class_name=pipeline_class_name,
            records=[],
            reason=reason,
        )
        return {
            "status": "not_supported",
            "reason": reason,
        }
    if not hasattr(diffusers, pipeline_class_name):
        logger.info("Native negative prompt not supported: diffusers lacks %s", pipeline_class_name)
        tracker.log_event(
            "native_negative_prompt_not_supported",
            {"reason": f"diffusers does not expose {pipeline_class_name}"},
        )
        _write_native_run_timing(
            output_dir=output_dir,
            status="not_supported",
            started_at=run_started_at,
            run_start=run_start,
            config=config,
            model_id=str(get_path(config, "model.model_id")),
            pipeline_class_name=pipeline_class_name,
            records=[],
            reason=f"diffusers does not expose {pipeline_class_name}",
        )
        return {
            "status": "not_supported",
            "reason": f"diffusers does not expose {pipeline_class_name}",
        }
    pipeline_cls = getattr(diffusers, pipeline_class_name)
    call_signature = inspect.signature(pipeline_cls.__call__)
    if "negative_prompt" not in call_signature.parameters:
        logger.info(
            "Native negative prompt not supported: %s lacks negative_prompt", pipeline_class_name
        )
        tracker.log_event(
            "native_negative_prompt_not_supported",
            {"reason": f"{pipeline_class_name}.__call__ does not expose negative_prompt"},
        )
        _write_native_run_timing(
            output_dir=output_dir,
            status="not_supported",
            started_at=run_started_at,
            run_start=run_start,
            config=config,
            model_id=str(get_path(config, "model.model_id")),
            pipeline_class_name=pipeline_class_name,
            records=[],
            reason=f"{pipeline_class_name}.__call__ does not expose negative_prompt",
        )
        return {
            "status": "not_supported",
            "reason": f"{pipeline_class_name}.__call__ does not expose negative_prompt",
        }
    if str(get_path(config, "model.adapter") or "").lower() == "flux_dual_view":
        required_flux_call_keys = {
            "prompt",
            "prompt_2",
            "negative_prompt",
            "negative_prompt_2",
            "guidance_scale",
            "true_cfg_scale",
        }
        missing_flux_call_keys = sorted(
            required_flux_call_keys - set(call_signature.parameters)
        )
        if missing_flux_call_keys:
            raise RuntimeError(
                "Flux dual-view native-negative execution requires all four paired "
                "pipeline prompt arguments and both effective guidance arguments; "
                f"missing {missing_flux_call_keys}."
            )

    raw_wan_temporal_protocol = get_path(config, f"model.{WAN_TEMPORAL_PROTOCOL_KEY}")
    wan_dependency_preflight: dict[str, Any] | None = None
    if raw_wan_temporal_protocol is not None:
        if str(get_path(config, "model.adapter")) != "wan":
            raise ValueError("A Wan temporal protocol was attached to a non-Wan model.")
        wan_dependency_preflight = validate_wan_native_negative_prompt_cleaner()
        logger.info(
            "Wan native-negative dependency preflight passed with ftfy %s",
            wan_dependency_preflight["package_version"],
        )
        tracker.log_event(
            "wan_native_negative_dependency_preflight",
            wan_dependency_preflight,
        )

    seed = int(get_path(config, "project.seed", 1234))
    seed_everything(seed)
    device = resolve_device(str(get_path(config, "runtime.device", "cuda")))
    dtype = resolve_dtype(
        str(get_path(config, "runtime.dtype", get_path(config, "model.torch_dtype", "bfloat16")))
    )
    model_id = str(get_path(config, "model.model_id"))
    adapter_name = str(get_path(config, "model.adapter"))
    generation = _generation_for_task(dict(config.get("generation", {})))
    flux_dual_view_adapter = _build_flux_dual_view_native_negative_adapter(
        config,
        device=device,
        dtype=dtype,
    )
    raw_cogvideox_temporal_protocol = get_path(
        config, f"model.{COGVIDEOX_TEMPORAL_PROTOCOL_KEY}"
    )
    raw_hunyuan_plan = get_path(config, f"model.{HUNYUAN_DUAL_VIEW_CONFIG_KEY}")
    raw_hunyuan_temporal_protocol = get_path(
        config, f"model.{HUNYUAN_TEMPORAL_PROTOCOL_KEY}"
    )
    cogvideox_adapter: CogVideoXAdapter | None = None
    hunyuan_adapter: HunyuanVideoAdapter | None = None
    if raw_cogvideox_temporal_protocol is not None:
        if adapter_name != "cogvideox":
            raise ValueError("A CogVideoX temporal protocol was attached to another adapter.")
        cogvideox_adapter = CogVideoXAdapter(
            model_id=model_id,
            device=device,
            dtype=dtype,
            config=dict(config.get("model", {})),
        )
        _preauthenticate_schema2_native_adapter(cogvideox_adapter, family="cogvideox")
    if raw_hunyuan_plan is not None or raw_hunyuan_temporal_protocol is not None:
        if adapter_name != "hunyuan_video":
            raise ValueError(
                "Hunyuan dual-view or temporal configuration was attached to another adapter."
            )
        if raw_hunyuan_plan is None or raw_hunyuan_temporal_protocol is None:
            raise RuntimeError(
                "Native Hunyuan schema-2 execution requires both temporal protocol and "
                "frozen dual-view conditioning plan."
            )
        hunyuan_adapter = HunyuanVideoAdapter(
            model_id=model_id,
            device=device,
            dtype=dtype,
            config=dict(config.get("model", {})),
        )
        _preauthenticate_schema2_native_adapter(hunyuan_adapter, family="hunyuan_video")

    prompt_file = generation.get("prompt_file")
    if prompt_file and not Path(str(prompt_file)).is_absolute():
        prompt_file = str(
            Path(get_path(config, "_meta.project_root", Path.cwd())) / str(prompt_file)
        )
    selected_prompts = (
        [prompt] if prompt else _read_prompts(str(prompt_file) if prompt_file else None)
    )
    if any(
        adapter is not None
        for adapter in (
            cogvideox_adapter,
            hunyuan_adapter,
            wan_dependency_preflight,
        )
    ) and len(selected_prompts) != 1:
        raise ValueError(
            "A one-way segmented native temporal protocol requires exactly one prompt per process."
        )
    negative_prompt, flux1_calibration_provenance = _resolve_native_negative_prompt(
        config,
        pipeline_class_name=pipeline_class_name,
    )
    if flux_dual_view_adapter is not None:
        flux_dual_view_adapter.validate_primary_prompts(selected_prompts)
        plan = flux_dual_view_adapter.config[FLUX_DUAL_VIEW_CONFIG_KEY]
        negative_plan = plan["negative"]
        route_binding = _flux_dual_view_route_binding(config)
        if negative_plan is None:
            if not (
                negative_prompt is None
                and route_binding["negative_mode"]
                == NEGATIVE_MODE_EXPLICIT_NONE_CONTROL
                and flux1_calibration_provenance is not None
                and flux1_calibration_provenance["negative_prompt_mode"]
                == NEGATIVE_MODE_EXPLICIT_NONE_CONTROL
            ):
                raise ValueError(
                    "A null Flux dual-view negative plan is restricted to the exact "
                    "explicit-none calibration control."
                )
        elif (
            route_binding["negative_mode"] != NEGATIVE_MODE_PAIRED_REGISTERED
            or negative_prompt != negative_plan["clip_negative_prompt"]
        ):
            raise ValueError(
                "native_negative_prompt.prompt must equal the shared paired registered "
                "negative CLIP view."
            )
        for key, value in {
            "guidance_scale": generation.get("guidance_scale"),
            "true_cfg_scale": get_path(config, "native_negative_prompt.true_cfg_scale"),
        }.items():
            if type(value) is not float or not math.isfinite(value):
                raise ValueError(
                    "Flux dual-view native execution requires each effective guidance "
                    f"argument to be one exact finite float; {key}={value!r}."
                )

    logger.info("Loading native negative prompt pipeline %s for %s", pipeline_class_name, model_id)
    load_start = time.perf_counter()
    load_kwargs = dict(get_path(config, "model.load_kwargs", {}) or {})
    if get_path(config, "model.revision") is not None:
        load_kwargs.setdefault("revision", get_path(config, "model.revision"))
    if get_path(config, "model.variant") is not None:
        load_kwargs.setdefault("variant", get_path(config, "model.variant"))
    single_file = load_kwargs.pop("single_file", None)
    single_file_components = load_kwargs.pop("single_file_components", None)
    load_kwargs.setdefault("torch_dtype", dtype)
    load_kwargs.setdefault("low_cpu_mem_usage", True)
    if os.environ.get("HF_TOKEN") and "token" not in load_kwargs:
        load_kwargs["token"] = os.environ["HF_TOKEN"]
    if (
        str(get_path(config, "model.adapter")) == "wan"
        and get_path(config, f"model.{WAN_TEMPORAL_PROTOCOL_KEY}") is not None
    ):
        _inject_wan_float32_vae_for_native_temporal_load(
            diffusers_module=diffusers,
            model_id=model_id,
            config=config,
            load_kwargs=load_kwargs,
        )
    if single_file is not None:
        load_kwargs.update(
            load_single_file_companion_components(
                single_file_components,
                torch_dtype=load_kwargs.get("torch_dtype"),
                local_files_only=bool(load_kwargs.get("local_files_only", False)),
                token=load_kwargs.get("token"),
                cache_dir=load_kwargs.get("cache_dir"),
            )
        )
        checkpoint_path = resolve_single_file_checkpoint(
            single_file,
            token=load_kwargs.get("token"),
            revision=load_kwargs.get("revision"),
            local_files_only=bool(load_kwargs.get("local_files_only", False)),
            cache_dir=load_kwargs.get("cache_dir"),
        )
        pipe = pipeline_cls.from_single_file(checkpoint_path, **load_kwargs)
    else:
        pipe = pipeline_cls.from_pretrained(model_id, **load_kwargs)
    configure_pipeline_vae_tiling(pipe, get_path(config, "model.vae_tiling"))
    cpu_offload = get_path(config, "model.cpu_offload", False)
    if cpu_offload == "sequential" and hasattr(pipe, "enable_sequential_cpu_offload"):
        pipe.enable_sequential_cpu_offload(device=device)
    elif bool(cpu_offload) and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload(device=device)
    elif hasattr(pipe, "to"):
        pipe.to(device)
    pipeline_load_seconds = time.perf_counter() - load_start
    tracker.log_event(
        "native_negative_prompt_pipeline_loaded",
        {"pipeline_class": pipeline_class_name, "model_id": model_id},
    )

    flux_dual_view_preflight: dict[str, Any] | None = None
    if flux_dual_view_adapter is not None:
        flux_dual_view_adapter.pipeline = pipe
        flux_dual_view_adapter.loaded = True
        flux_dual_view_preflight = flux_dual_view_adapter.preflight_conditioning_plan()
        tracker.log_event(
            "flux_dual_view_conditioning_plan_preflight_completed",
            flux_dual_view_preflight,
        )

    cogvideox_native_contract: dict[str, Any] | None = None
    if cogvideox_adapter is not None:
        cogvideox_native_contract = _configure_native_temporal_adapter(
            cogvideox_adapter,
            pipe,
            family="cogvideox",
        )
        tracker.log_event(
            "cogvideox_native_temporal_protocol_configured",
            cogvideox_native_contract,
        )

    hunyuan_preflight: dict[str, Any] | None = None
    hunyuan_native_contract: dict[str, Any] | None = None
    if hunyuan_adapter is not None:
        hunyuan_adapter.pipeline = pipe
        hunyuan_adapter.loaded = True
        hunyuan_preflight = hunyuan_adapter.preflight_conditioning_plan()
        tracker.log_event(
            "conditioning_plan_preflight_completed",
            hunyuan_preflight,
        )
        hunyuan_native_contract = _configure_native_temporal_adapter(
            hunyuan_adapter,
            pipe,
            family="hunyuan_video",
        )
        tracker.log_event(
            "hunyuan_native_temporal_protocol_configured",
            hunyuan_native_contract,
        )

    wan_adapter: WanAdapter | None = None
    wan_native_contract: dict[str, Any] | None = None
    if raw_wan_temporal_protocol is not None:
        assert wan_dependency_preflight is not None
        wan_adapter = WanAdapter(
            model_id=model_id,
            device=device,
            dtype=dtype,
            config=dict(config.get("model", {})),
        )
        wan_adapter.record_native_negative_dependency_preflight(wan_dependency_preflight)
        wan_native_contract = _configure_native_temporal_adapter(
            wan_adapter,
            pipe,
            family="wan",
        )
        tracker.log_event(
            "wan_native_temporal_protocol_configured",
            wan_native_contract,
        )

    active_temporal_contracts = [
        name
        for name, contract in (
            ("cogvideox", cogvideox_native_contract),
            ("hunyuan_video", hunyuan_native_contract),
            ("wan", wan_native_contract),
        )
        if contract is not None
    ]
    if len(active_temporal_contracts) > 1:
        raise RuntimeError(
            "A native-negative run cannot activate multiple temporal protocols: "
            f"{active_temporal_contracts}."
        )

    sample_records = []
    for index, selected_prompt in enumerate(selected_prompts):
        sample_id = f"sample_{index:04d}"
        sample_started_at = _utc_now_iso()
        sample_start = time.perf_counter()
        logger.info("Generating %s with native negative prompt", sample_id)
        generator = make_generator(seed, device)
        task = str(generation.get("task", "text_to_image"))
        requested_num_frames = generation.get("num_frames")
        if cogvideox_native_contract is not None:
            native_call_num_frames = int(cogvideox_native_contract["num_frames"])
            native_call_height = int(cogvideox_native_contract["height"])
            native_call_width = int(cogvideox_native_contract["width"])
            native_contract_fps = int(cogvideox_native_contract["native_fps"])
        elif hunyuan_native_contract is not None:
            native_call_num_frames = int(hunyuan_native_contract["num_frames"])
            native_call_height = int(hunyuan_native_contract["height"])
            native_call_width = int(hunyuan_native_contract["width"])
            native_contract_fps = int(hunyuan_native_contract["native_fps"])
        elif wan_native_contract is not None:
            native_call_num_frames = int(wan_native_contract["num_frames"])
            native_call_height = int(generation["height"])
            native_call_width = int(generation["width"])
            native_contract_fps = int(wan_native_contract["native_fps"])
        else:
            native_call_num_frames = (
                _aligned_native_num_frames(config, int(requested_num_frames))
                if task == "text_to_video" and requested_num_frames is not None
                else requested_num_frames
            )
            native_call_height = generation.get("height")
            native_call_width = generation.get("width")
            native_contract_fps = generation.get("fps")
        call_kwargs = {
            "height": native_call_height,
            "width": native_call_width,
            "num_frames": native_call_num_frames,
            # Some pipelines expose an explicit frame-rate argument and others
            # do not. When present it must describe the model-native call, not
            # the eventual 16-fps file produced by deterministic postprocessing.
            "frame_rate": native_contract_fps,
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
        hunyuan_encoding = None
        flux_dual_view_call: dict[str, str | None] | None = None
        flux_dual_view_trace_call: dict[str, str | float | None] | None = None
        if flux_dual_view_adapter is not None:
            if hunyuan_adapter is not None:
                raise RuntimeError("Flux and Hunyuan dual-view routes cannot be combined.")
            if negative_prompt is None:
                flux_dual_view_call = (
                    flux_dual_view_adapter.native_explicit_none_prompt_kwargs(
                        positive_clip_prompt=selected_prompt,
                    )
                )
            else:
                flux_dual_view_call = flux_dual_view_adapter.native_negative_prompt_kwargs(
                    positive_clip_prompt=selected_prompt,
                    negative_clip_prompt=negative_prompt,
                )
            call_kwargs.update(flux_dual_view_call)
        elif hunyuan_adapter is None:
            call_kwargs.update(
                {
                    "prompt": selected_prompt,
                    "negative_prompt": negative_prompt,
                }
            )
        else:
            conditioning_kwargs, hunyuan_encoding = _hunyuan_native_conditioning_kwargs(
                hunyuan_adapter,
                selected_prompt,
                negative_prompt,
            )
            call_kwargs.update(conditioning_kwargs)
        call_kwargs = {
            key: value
            for key, value in call_kwargs.items()
            if key in call_signature.parameters and value is not None
        }
        if (
            flux1_calibration_provenance is not None
            and flux1_calibration_provenance["negative_prompt_mode"]
            == NEGATIVE_MODE_EXPLICIT_NONE_CONTROL
        ):
            # Preserve the explicit ``None`` call argument.  The generic
            # sanitizer drops None-valued optional kwargs, but the calibration
            # integrity gate needs the exact same call surface as its scale-1
            # registered-string row. The dual-view route controls both paired
            # negative arguments together; the legacy route controls one.
            call_kwargs["negative_prompt"] = None
            if flux_dual_view_adapter is not None:
                call_kwargs["negative_prompt_2"] = None
        if flux_dual_view_adapter is not None:
            required_flux_prompt_keys = {
                "prompt",
                "prompt_2",
                "negative_prompt",
                "negative_prompt_2",
            }
            if (
                required_flux_prompt_keys - set(call_kwargs)
                or flux_dual_view_call is None
                or any(
                    call_kwargs[key] != flux_dual_view_call[key]
                    for key in required_flux_prompt_keys
                )
            ):
                raise RuntimeError(
                    "Flux dual-view native-negative call lost a registered prompt view."
                )
            expected_guidance_call = {
                "guidance_scale": generation["guidance_scale"],
                "true_cfg_scale": get_path(
                    config, "native_negative_prompt.true_cfg_scale"
                ),
            }
            if any(
                key not in call_kwargs
                or type(call_kwargs[key]) is not float
                or not math.isfinite(call_kwargs[key])
                or call_kwargs[key] != expected
                for key, expected in expected_guidance_call.items()
            ):
                raise RuntimeError(
                    "Flux dual-view native call lost or changed its exact effective "
                    "guidance_scale/true_cfg_scale arguments."
                )
            flux_dual_view_trace_call = {
                **deepcopy(flux_dual_view_call),
                **{key: call_kwargs[key] for key in expected_guidance_call},
            }
        if hunyuan_adapter is not None:
            forbidden_raw_keys = {"prompt", "prompt_2", "negative_prompt", "negative_prompt_2"}
            if forbidden_raw_keys & set(call_kwargs):
                raise RuntimeError(
                    "Hunyuan native-negative dual-view routing must omit every raw prompt argument."
                )
            required_embedding_keys = {
                "prompt_embeds",
                "pooled_prompt_embeds",
                "prompt_attention_mask",
                "negative_prompt_embeds",
                "negative_pooled_prompt_embeds",
                "negative_prompt_attention_mask",
            }
            if not required_embedding_keys <= set(call_kwargs):
                raise RuntimeError(
                    "Hunyuan native-negative pipeline signature does not accept all six frozen "
                    "positive/negative conditioning tensors."
                )
        generation_start = time.perf_counter()
        with torch.inference_mode():
            if cogvideox_native_contract is not None:
                output = _invoke_native_temporal_first_call(
                    pipe=pipe,
                    family="cogvideox",
                    call_kwargs=call_kwargs,
                    contract=cogvideox_native_contract,
                )
            elif hunyuan_native_contract is not None:
                output = _invoke_native_temporal_first_call(
                    pipe=pipe,
                    family="hunyuan_video",
                    call_kwargs=call_kwargs,
                    contract=hunyuan_native_contract,
                )
            elif wan_native_contract is not None:
                output = _invoke_native_temporal_first_call(
                    pipe=pipe,
                    family="wan",
                    call_kwargs=call_kwargs,
                    contract=wan_native_contract,
                )
            else:
                output = pipe(**call_kwargs)
        media = _extract_pipeline_media(output)
        if cogvideox_adapter is not None:
            media = _complete_native_temporal_protocol(
                adapter=cogvideox_adapter,
                family="cogvideox",
                first_segment_media=media,
                call_kwargs=call_kwargs,
                contract=cogvideox_native_contract,
            )
        elif hunyuan_adapter is not None:
            media = _complete_native_temporal_protocol(
                adapter=hunyuan_adapter,
                family="hunyuan_video",
                first_segment_media=media,
                call_kwargs=call_kwargs,
                contract=hunyuan_native_contract,
            )
        elif wan_adapter is not None:
            media = _complete_native_temporal_protocol(
                adapter=wan_adapter,
                family="wan",
                first_segment_media=media,
                call_kwargs=call_kwargs,
                contract=wan_native_contract,
            )
        generation_seconds = time.perf_counter() - generation_start
        tracker.log_event(
            "native_negative_prompt_generation_finished",
            {
                "sample_id": sample_id,
                "num_inference_steps": generation.get("num_inference_steps"),
            },
        )
        if not active_temporal_contracts and task == "text_to_video" and requested_num_frames is not None:
            media = _crop_video_media(media, int(requested_num_frames))
        trace_entry = {
            "condition": "native_negative_prompt",
            "pipeline_class": pipeline_class_name,
            "negative_prompt": negative_prompt,
            "conditioning_mode": (
                "hunyuan_dual_view_precomputed_embeddings"
                if hunyuan_adapter is not None
                else (
                    (
                        "flux_dual_view_explicit_none_native_prompt_arguments"
                        if negative_prompt is None
                        else "flux_dual_view_paired_native_prompt_arguments"
                    )
                    if flux_dual_view_adapter is not None
                    else "native_raw_prompt_arguments"
                )
            ),
        }
        if flux_dual_view_adapter is not None:
            if flux_dual_view_trace_call is None:
                raise RuntimeError("Flux dual-view effective native-call evidence is absent.")
            trace_entry["flux_dual_view_native_call"] = deepcopy(
                flux_dual_view_trace_call
            )
            trace_entry["flux_dual_view_plan_preflight"] = deepcopy(
                flux_dual_view_preflight
            )
        if flux1_calibration_provenance is not None:
            trace_entry["flux1_native_negative_calibration"] = dict(
                flux1_calibration_provenance
            )
        if cogvideox_adapter is not None:
            trace_entry["cogvideox_temporal_generation"] = _completed_temporal_provenance(
                adapter=cogvideox_adapter,
                family="CogVideoX",
            )
            trace_entry["cogvideox_native_call_contract"] = cogvideox_native_contract
        if hunyuan_native_contract is not None:
            assert hunyuan_adapter is not None
            trace_entry["hunyuan_temporal_generation"] = _completed_temporal_provenance(
                adapter=hunyuan_adapter,
                family="HunyuanVideo",
            )
            trace_entry["hunyuan_native_call_contract"] = hunyuan_native_contract
        if wan_native_contract is not None:
            assert wan_adapter is not None
            trace_entry["wan_temporal_generation"] = _completed_temporal_provenance(
                adapter=wan_adapter,
                family="Wan",
            )
            trace_entry["wan_native_call_contract"] = wan_native_contract
            trace_entry["wan_native_negative_dependency_preflight"] = wan_dependency_preflight
        if hunyuan_encoding is not None:
            trace_entry["hunyuan_entry_ids"] = [
                record["entry_id"] for record in hunyuan_encoding.records
            ]
            trace_entry["hunyuan_plan_preflight"] = hunyuan_preflight
        if task == "text_to_video":
            trace_entry.update(
                {
                    "requested_num_frames": requested_num_frames,
                    "native_call_num_frames": native_call_num_frames,
                }
            )
        trace = [trace_entry]
        save_start = time.perf_counter()
        timing_path = output_dir / sample_id / "timing.json"
        generation_config = dict(generation)
        generation_config["prompt"] = selected_prompt
        if task == "text_to_video":
            generation_config["native_call_num_frames"] = native_call_num_frames
            generation_config["native_call_fps"] = native_contract_fps
        if cogvideox_native_contract is not None:
            generation_config["cogvideox_native_call_contract"] = dict(
                cogvideox_native_contract
            )
        if hunyuan_native_contract is not None:
            generation_config["hunyuan_native_call_contract"] = dict(hunyuan_native_contract)
        if wan_native_contract is not None:
            generation_config["wan_native_call_contract"] = dict(wan_native_contract)
            generation_config["wan_native_negative_dependency_preflight"] = (
                wan_dependency_preflight
            )
        if flux_dual_view_adapter is not None:
            conditioning_provenance = flux_dual_view_adapter.conditioning_provenance()
        elif hunyuan_adapter is not None:
            conditioning_provenance = hunyuan_adapter.conditioning_provenance()
        elif wan_adapter is not None:
            conditioning_provenance = wan_adapter.conditioning_provenance()
        elif cogvideox_adapter is not None:
            conditioning_provenance = cogvideox_adapter.conditioning_provenance()
        else:
            conditioning_provenance = {
                "conditioning_mode": "native_raw_prompt_arguments",
                "prompt": selected_prompt,
                "negative_prompt": negative_prompt,
            }
            if flux1_calibration_provenance is not None:
                conditioning_provenance["flux1_native_negative_calibration"] = dict(
                    flux1_calibration_provenance
                )
        temporal_adapter = cogvideox_adapter or hunyuan_adapter
        temporal_evidence = None
        segment_trace_validation = None
        if temporal_adapter is not None:
            temporal_evidence = temporal_adapter.take_temporal_evidence(None)
            if temporal_evidence is None:
                raise RuntimeError(
                    "Schema-2 native temporal completion did not publish temporal evidence."
                )
            temporal_evidence = _bind_native_temporal_execution_identity(
                temporal_evidence,
                config=config,
            )
            segment_trace_validation = _native_segment_trace_validation(
                temporal_evidence
            )
        report = {
            "schema_version": 1,
            "prompt": selected_prompt,
            "sample_id": sample_id,
            "task": task,
            "benchmark": config.get("benchmark", {}),
            "model": {
                "adapter": get_path(config, "model.adapter"),
                "model_id": model_id,
                "revision": get_path(config, "model.revision"),
                "pipeline_class": pipeline_class_name,
            },
            "condition": {
                "steering_mode": "native_negative_prompt",
                "decode_outputs": True,
                "is_native_negative_prompt": True,
                "negative_prompt": negative_prompt,
                "negative_prompt_2": (
                    None
                    if flux_dual_view_call is None
                    else flux_dual_view_call["negative_prompt_2"]
                ),
                "flux1_native_negative_calibration": flux1_calibration_provenance,
            },
            "generation": generation_config,
            "conditioning_provenance": conditioning_provenance,
            "steering": config.get("steering", {}),
            "output_paths": {},
            "interpretability": {
                "note": (
                    "Native negative-prompt baselines do not expose concept bottleneck activations "
                    "or steering deltas. The prompt, negative_prompt, and saved media/trace are "
                    "recorded for comparison."
                ),
                "segment_trace_validation": segment_trace_validation,
                "timesteps": trace,
                "concepts_per_step": [],
            },
        }
        paths = save_generation_output(
            media=media,
            latents=torch.empty(0),
            trace=trace,
            output_dir=output_dir,
            sample_id=sample_id,
            task=task,
            save_latents=False,
            save_traces=True,
            image_format=str(get_path(config, "output.image_format", "png")),
            video_format=str(get_path(config, "output.video_format", "mp4")),
            fps=int(generation.get("fps", 16)),
            temporal_evidence=temporal_evidence,
            report=report if temporal_evidence is not None else None,
        )
        if temporal_evidence is None:
            report["output_paths"] = dict(paths)
            paths["report"] = save_generation_report(report, output_dir, sample_id)
        paths["timing"] = str(timing_path)
        save_seconds = time.perf_counter() - save_start
        timing = {
            "schema_version": 1,
            "status": "completed",
            "started_at": sample_started_at,
            "ended_at": _utc_now_iso(),
            "total_seconds": time.perf_counter() - sample_start,
            "phases_seconds": {
                "pipeline_load": pipeline_load_seconds if index == 0 else 0.0,
                "native_pipeline_call": generation_seconds,
                "save_media_trace_and_report": save_seconds,
            },
            "sample_id": sample_id,
            "prompt": selected_prompt,
            "task": task,
            "benchmark": config.get("benchmark", {}),
            "model": {
                "adapter": get_path(config, "model.adapter"),
                "model_id": model_id,
                "revision": get_path(config, "model.revision"),
                "pipeline_class": pipeline_class_name,
            },
            "generation": generation_config,
            "media": _media_summary(media, task=task, fps=int(generation.get("fps", 16))),
            "output_paths": dict(paths),
        }
        write_json(timing_path, timing)
        logger.info("Finished %s", sample_id)
        tracker.log_event("sample_finished", {"sample_id": sample_id, "paths": paths})
        sample_records.append(
            {
                "prompt": selected_prompt,
                "sample_id": sample_id,
                "output_paths": paths,
            }
        )
    _write_native_run_timing(
        output_dir=output_dir,
        status="completed",
        started_at=run_started_at,
        run_start=run_start,
        config=config,
        model_id=model_id,
        pipeline_class_name=pipeline_class_name,
        records=sample_records,
        pipeline_load_seconds=pipeline_load_seconds,
    )
    return {
        "status": "completed",
        "result": {
            "output_dir": str(output_dir),
            "records": sample_records,
        },
    }


def _build_flux_dual_view_native_negative_adapter(
    config: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> FluxDualViewAdapter | None:
    """Resolve the runner-owned plan bridge before any native pipeline load."""

    model_config = dict(config.get("model", {}))
    generation_config = _generation_for_task(dict(config.get("generation", {})))
    has_plan = (
        model_config.get(FLUX_DUAL_VIEW_CONFIG_KEY) is not None
        or generation_config.get(FLUX_DUAL_VIEW_CONFIG_KEY) is not None
    )
    is_dual_view = str(model_config.get("adapter") or "").lower() == "flux_dual_view"
    if not has_plan and not is_dual_view:
        return None
    if not has_plan:
        raise ValueError(
            "A flux_dual_view native-negative run requires its registered paired plan."
        )
    bound_model = _bind_flux_dual_view_conditioning(
        model_config=model_config,
        generation_config=generation_config,
    )
    return FluxDualViewAdapter(
        model_id=str(bound_model.get("model_id") or ""),
        device=device,
        dtype=dtype,
        config=bound_model,
    )


def _hunyuan_native_conditioning_kwargs(
    adapter: HunyuanVideoAdapter,
    positive_prompt: str,
    negative_prompt: str,
) -> tuple[dict[str, torch.Tensor], Any]:
    """Resolve exact plan entries and split two encoded rows into six tensors."""

    views = adapter.resolve_prompt_views([positive_prompt, negative_prompt])
    if len(views) != 2 or views[0].role.lower() not in {"base", "baseline"}:
        raise RuntimeError("Hunyuan native-negative positive prompt did not resolve to base.")
    if views[1].role.lower() != "native_negative":
        raise RuntimeError(
            "Hunyuan native-negative negative prompt did not resolve to native_negative."
        )
    encoding = adapter.encode_prompt_views(views)
    tensor_rows = {
        "prompt_embeds": encoding.prompt_embeds,
        "pooled_prompt_embeds": encoding.pooled_prompt_embeds,
        "prompt_attention_mask": encoding.prompt_attention_mask,
    }
    for name, tensor in tensor_rows.items():
        if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != 2:
            raise RuntimeError(
                f"Hunyuan native-negative {name} must contain exactly two encoded rows."
            )
    return (
        {
            "prompt_embeds": encoding.prompt_embeds[0:1],
            "pooled_prompt_embeds": encoding.pooled_prompt_embeds[0:1],
            "prompt_attention_mask": encoding.prompt_attention_mask[0:1],
            "negative_prompt_embeds": encoding.prompt_embeds[1:2],
            "negative_pooled_prompt_embeds": encoding.pooled_prompt_embeds[1:2],
            "negative_prompt_attention_mask": encoding.prompt_attention_mask[1:2],
        },
        encoding,
    )


def _flux_dual_view_route_binding(config: dict[str, Any]) -> dict[str, Any]:
    """Authenticate the shared job-route role before any native Flux call."""

    raw = get_path(config, "benchmark.flux1_dual_view_route_v3")
    expected_keys = {
        "contract_id",
        "negative_mode",
        "native_pipeline_execution",
        "plan_sha256",
    }
    if not isinstance(raw, dict) or set(raw) != expected_keys:
        raise ValueError("Flux dual-view native execution lacks its exact shared route binding.")
    if (
        raw.get("contract_id") != FLUX1_DUAL_VIEW_ROUTE_CONTRACT_ID
        or raw.get("negative_mode")
        not in {
            NEGATIVE_MODE_NOT_APPLIED,
            NEGATIVE_MODE_EXPLICIT_NONE_CONTROL,
            NEGATIVE_MODE_PAIRED_REGISTERED,
        }
        or not isinstance(raw.get("native_pipeline_execution"), bool)
        or raw.get("native_pipeline_execution") is not True
    ):
        raise ValueError("Flux dual-view native route identity/role drifted.")
    generation_plan = get_path(config, f"generation.{FLUX_DUAL_VIEW_CONFIG_KEY}")
    model_plan = get_path(config, f"model.{FLUX_DUAL_VIEW_CONFIG_KEY}")
    if generation_plan is None and model_plan is None:
        raise ValueError("Flux dual-view native route omitted its conditioning plan.")
    if generation_plan is not None and model_plan is not None and generation_plan != model_plan:
        raise ValueError("Flux dual-view native route has conflicting model/generation plans.")
    plan = generation_plan if generation_plan is not None else model_plan
    if raw.get("plan_sha256") != flux1_dual_view_plan_sha256(plan):
        raise ValueError("Flux dual-view native route plan digest drifted.")
    return deepcopy(raw)


def _resolve_native_negative_prompt(
    config: dict[str, Any],
    *,
    pipeline_class_name: str,
) -> tuple[str | None, dict[str, Any] | None]:
    """Resolve the generic string or the single sealed Flux calibration control.

    Ordinary manifests cannot request a null negative prompt: doing so requires
    the exact calibration benchmark identity, stage, checkpoint, pipeline,
    role, config digest, and scale.  Registered calibration rows also carry the
    same metadata so traces prove that both routes used this shared function.
    """

    fallback = (
        "explicit nudity, exposed intimate anatomy, pornographic content, sexualized bed scene, "
        "erotic pose, bare front body, fetish content, genital exposure, nipple exposure, "
        "see-through clothing, unsafe sexual content"
    )
    raw_prompt = get_path(config, "native_negative_prompt.prompt", fallback)
    mode = get_path(config, "native_negative_prompt.calibration_negative_prompt_mode")
    model_adapter = str(get_path(config, "model.adapter") or "")
    route_binding = (
        _flux_dual_view_route_binding(config)
        if model_adapter == "flux_dual_view"
        else None
    )
    if mode is None:
        if raw_prompt is None:
            raise ValueError(
                "negative_prompt=None is forbidden outside the sealed Flux.1 calibration."
            )
        if route_binding is not None and route_binding["negative_mode"] != (
            NEGATIVE_MODE_PAIRED_REGISTERED
        ):
            raise ValueError(
                "A non-calibration Flux dual-view native call must use the shared paired mode."
            )
        return str(raw_prompt), None

    benchmark_name = get_path(config, "benchmark.name")
    benchmark_stage = get_path(config, "benchmark.stage")
    calibration_id = get_path(config, "native_negative_prompt.calibration_id")
    config_sha256 = get_path(
        config, "native_negative_prompt.calibration_config_sha256"
    )
    role = get_path(config, "native_negative_prompt.calibration_role")
    true_cfg_scale = get_path(config, "native_negative_prompt.true_cfg_scale")
    frozen_identity = {
        "benchmark_name": benchmark_name,
        "benchmark_stage": benchmark_stage,
        "calibration_id": calibration_id,
        "calibration_config_sha256": config_sha256,
        "model_adapter": get_path(config, "model.adapter"),
        "model_id": get_path(config, "model.model_id"),
        "model_revision": get_path(config, "model.revision"),
        "pipeline_class": pipeline_class_name,
    }
    protocol_identity = (benchmark_name, benchmark_stage, calibration_id)
    if model_adapter not in {"flux", "flux_dual_view"}:
        raise ValueError(
            "Calibration negative-prompt mode requires an authenticated Flux.1 adapter route."
        )
    if protocol_identity == (
        _FLUX1_NEGATIVE_CALIBRATION_ID,
        _FLUX1_NEGATIVE_CALIBRATION_STAGE,
        _FLUX1_NEGATIVE_CALIBRATION_ID,
    ):
        if model_adapter != "flux":
            raise ValueError("Historical calibration-v1 is restricted to the legacy Flux adapter.")
        expected_config_sha256 = _FLUX1_NEGATIVE_CALIBRATION_CONFIG_SHA256
        protocol_version = 1
    elif protocol_identity == (
        _FLUX1_NEGATIVE_CALIBRATION_V2_ID,
        _FLUX1_NEGATIVE_CALIBRATION_V2_STAGE,
        _FLUX1_NEGATIVE_CALIBRATION_V2_ID,
    ):
        if model_adapter != "flux_dual_view":
            raise ValueError("Calibration-v2 requires the versioned flux_dual_view adapter.")
        expected_config_sha256 = _FLUX1_NEGATIVE_CALIBRATION_V2_CONFIG_SHA256
        protocol_version = 2
    else:
        raise ValueError("Unknown Flux calibration identity at the native-call boundary.")
    if expected_config_sha256 is None:
        raise ValueError(
            "Calibration negative-prompt execution is blocked until its sealed "
            "adapter-specific configuration digest is pinned in the native-call guard."
        )
    expected_identity = {
        "benchmark_name": benchmark_name,
        "benchmark_stage": benchmark_stage,
        "calibration_id": calibration_id,
        "calibration_config_sha256": expected_config_sha256,
        "model_adapter": model_adapter,
        "model_id": "black-forest-labs/FLUX.1-dev",
        "model_revision": "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21",
        "pipeline_class": "FluxPipeline",
    }
    if (
        protocol_identity
        not in (
            (
                _FLUX1_NEGATIVE_CALIBRATION_ID,
                _FLUX1_NEGATIVE_CALIBRATION_STAGE,
                _FLUX1_NEGATIVE_CALIBRATION_ID,
            ),
            (
                _FLUX1_NEGATIVE_CALIBRATION_V2_ID,
                _FLUX1_NEGATIVE_CALIBRATION_V2_STAGE,
                _FLUX1_NEGATIVE_CALIBRATION_V2_ID,
            ),
        )
        or frozen_identity != expected_identity
    ):
        raise ValueError(
            "Calibration negative-prompt mode is forbidden outside the exact sealed "
            f"Flux.1 protocol: observed={frozen_identity!r}."
        )
    if isinstance(true_cfg_scale, bool) or float(true_cfg_scale) not in (
        _FLUX1_NEGATIVE_CALIBRATION_SCALES
    ):
        raise ValueError("Flux.1 calibration true_cfg_scale is not preregistered.")

    if protocol_version == 2:
        if route_binding is None or mode != route_binding["negative_mode"]:
            raise ValueError(
                "Calibration-v2 negative mode differs from the shared Flux-v3 job route."
            )
        if mode == NEGATIVE_MODE_NOT_APPLIED:
            raise ValueError("Calibration official baseline cannot enter the native pipeline.")
        legal_paired_mode = NEGATIVE_MODE_PAIRED_REGISTERED
    else:
        legal_paired_mode = "registered_negative_string"

    if mode == NEGATIVE_MODE_EXPLICIT_NONE_CONTROL:
        if role != "native_no_negative_control" or float(true_cfg_scale) != 1.0:
            raise ValueError(
                "The explicit-none calibration route is restricted to the scale-1.0 "
                "native_no_negative_control row."
            )
        negative_prompt: str | None = None
    elif mode == legal_paired_mode:
        if role != "native_negative_scale_ladder":
            raise ValueError(
                "The registered-string calibration route requires a ladder row."
            )
        if not isinstance(raw_prompt, str) or not raw_prompt.strip():
            raise ValueError("A calibration ladder row requires its registered negative string.")
        negative_prompt = raw_prompt
    else:
        raise ValueError(f"Unknown calibration negative-prompt mode {mode!r}.")

    provenance = {
        "schema_version": 1,
        "calibration_id": calibration_id,
        "calibration_config_sha256": config_sha256,
        "calibration_role": role,
        "negative_prompt_mode": mode,
        "negative_prompt_is_none": negative_prompt is None,
        "true_cfg_scale": float(true_cfg_scale),
        "only_controlled_call_difference": (
            "paired_negative_prompt_values"
            if model_adapter == "flux_dual_view"
            else "negative_prompt_value"
        ),
    }
    return negative_prompt, provenance


def _inject_wan_float32_vae_for_native_temporal_load(
    *,
    diffusers_module: Any,
    model_id: str,
    config: dict[str, Any],
    load_kwargs: dict[str, Any],
) -> None:
    """Load Wan's VAE in its pinned official float32 precision contract."""

    revision = get_path(config, "model.revision")
    if revision != WAN_T2V_REVISION:
        raise RuntimeError(
            "Wan native temporal generation requires the pinned T2V revision "
            f"{WAN_T2V_REVISION}; got {revision!r}."
        )
    if load_kwargs.get("single_file") is not None:
        raise RuntimeError("Wan native temporal generation does not support single-file loading.")
    vae_cls = getattr(diffusers_module, "AutoencoderKLWan", None)
    if vae_cls is None or not callable(getattr(vae_cls, "from_pretrained", None)):
        raise RuntimeError("Installed diffusers does not expose AutoencoderKLWan.from_pretrained.")
    vae_kwargs: dict[str, Any] = {
        "subfolder": "vae",
        "revision": WAN_T2V_REVISION,
        "torch_dtype": torch.float32,
        "local_files_only": bool(load_kwargs.get("local_files_only", False)),
    }
    for key in ("token", "cache_dir"):
        if load_kwargs.get(key) is not None:
            vae_kwargs[key] = load_kwargs[key]
    vae = vae_cls.from_pretrained(model_id, **vae_kwargs)
    if getattr(vae, "dtype", None) != torch.float32:
        raise RuntimeError(
            "AutoencoderKLWan.from_pretrained did not preserve the required torch.float32 dtype."
        )
    load_kwargs["vae"] = vae


def _preauthenticate_schema2_native_adapter(adapter: Any, *, family: str) -> None:
    """Authenticate schema-2 artifacts before direct native pipeline loading."""

    protocol_hook = getattr(adapter, "_temporal_protocol", None)
    gate_hook = getattr(adapter, "_validate_temporal_execution_gate", None)
    preload_hook = getattr(adapter, "_validate_preload_contract", None)
    artifact_hook = getattr(adapter, "_authenticate_artifact_manifest", None)
    if not all(
        callable(hook)
        for hook in (protocol_hook, gate_hook, preload_hook, artifact_hook)
    ):
        raise RuntimeError(f"{family} lacks schema-2 native artifact preflight hooks.")
    protocol = protocol_hook()
    if not isinstance(protocol, dict) or protocol.get("schema_version") != 2:
        raise RuntimeError(f"{family} native preflight requires temporal protocol schema 2.")
    gate_hook(protocol)
    preload_hook(protocol)
    authentication = artifact_hook(protocol)
    if not isinstance(authentication, dict) or not str(
        authentication.get("status", "")
    ):
        raise RuntimeError(f"{family} native artifact authentication was not completed.")
    adapter._artifact_authentication = authentication


def _bind_native_temporal_execution_identity(
    evidence: Any,
    *,
    config: dict[str, Any],
) -> Any:
    benchmark = config.get("benchmark")
    if not isinstance(benchmark, dict):
        raise RuntimeError("Schema-2 native temporal run lacks benchmark identity.")
    required = {
        "condition_id": benchmark.get("condition_id"),
        "attempt": benchmark.get("attempt"),
        "manifest_sha256": benchmark.get("manifest_sha256"),
    }
    if (
        not isinstance(required["condition_id"], str)
        or not required["condition_id"]
        or isinstance(required["attempt"], bool)
        or not isinstance(required["attempt"], int)
        or not isinstance(required["manifest_sha256"], str)
    ):
        raise RuntimeError("Schema-2 native temporal execution identity is incomplete.")
    slurm_job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
    slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
    job_id = (
        f"slurm:{slurm_job}:{slurm_task}"
        if slurm_job and slurm_task is not None
        else (
            f"slurm:{slurm_job}"
            if slurm_job
            else f"local:{required['condition_id']}"
        )
    )
    return evidence.with_execution_identity(
        condition_id=required["condition_id"],
        attempt=required["attempt"],
        manifest_sha256=required["manifest_sha256"],
        job_id=job_id,
        metadata={
            "checkpoint_set": benchmark.get("checkpoint_set"),
            "checkpoint_set_sha256": benchmark.get("checkpoint_set_sha256"),
            "artifact_manifest": benchmark.get("artifact_manifest"),
            "artifact_manifest_sha256": benchmark.get("artifact_manifest_sha256"),
            "segmented_temporal_contract": benchmark.get(
                "segmented_temporal_contract"
            ),
            "native_temporal_call": True,
        },
    )


def _native_segment_trace_validation(evidence: Any) -> dict[str, Any]:
    segments = []
    global_start = 0
    for segment in evidence.segments:
        scheduler = dict(segment.scheduler)
        local_steps = scheduler.get(
            "num_inference_steps", scheduler.get("denoising_steps")
        )
        if isinstance(local_steps, bool) or not isinstance(local_steps, int):
            raise RuntimeError("Native temporal segment lacks scheduler step evidence.")
        segments.append(
            {
                "segment_index": segment.segment_index,
                "model_role": segment.model_role,
                "model_id": segment.model_id,
                "model_revision": segment.model_revision,
                "condition_epoch": segment.segment_index,
                "anchor_sha256": segment.anchor_sha256,
                "segment_seed": segment.segment_seed,
                "local_num_steps": local_steps,
                "global_step_start": global_start,
                "global_step_end": global_start + local_steps - 1,
            }
        )
        global_start += local_steps
    return {
        "schema_version": 1,
        "status": "passed",
        "global_num_steps": global_start,
        "segment_count": len(segments),
        "segments": segments,
        "evidence_kind": "native_pipeline_complete_call_records",
    }


def _configure_native_temporal_adapter(
    adapter: Any,
    pipe: Any,
    *,
    family: str,
) -> dict[str, Any]:
    """Attach a native pipeline and validate the adapter's frozen call contract."""

    configure_hook = getattr(adapter, "configure_native_pipeline_for_temporal_protocol", None)
    if not callable(configure_hook):
        raise RuntimeError(
            f"{family} native temporal execution requires a callable "
            "configure_native_pipeline_for_temporal_protocol hook."
        )
    required_completion_hook = {
        "cogvideox": "complete_native_pipeline_temporal_protocol",
        "hunyuan_video": "complete_native_pipeline_temporal_protocol",
        "wan": "complete_native_pipeline_temporal_protocol",
    }.get(family)
    if required_completion_hook is None:
        raise ValueError(f"Unsupported native temporal family: {family!r}.")
    if not callable(getattr(adapter, required_completion_hook, None)):
        raise RuntimeError(
            f"{family} native temporal execution requires a callable "
            f"{required_completion_hook} hook."
        )

    adapter.pipeline = pipe
    adapter.loaded = True
    contract = configure_hook()
    if not isinstance(contract, dict):
        raise RuntimeError(f"{family} native temporal configure hook must return a mapping.")

    if family == "cogvideox":
        expected = {
            "schema_version": 2,
            "segment_count": 3,
            "num_frames": 49,
            "native_fps": 8,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "output_frames": 240,
            "output_fps": 16,
            "completion_hook": "complete_native_pipeline_temporal_protocol",
        }
        required_positive = ("height", "width")
    elif family == "hunyuan_video":
        expected = {
            "schema_version": 2,
            "native_temporal_call_schema_version": 2,
            "segment_trace_schema_version": 1,
            "temporal_evidence_schema_version": 1,
            "segment_count": 3,
            "num_frames": 121,
            "native_fps": 24,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "true_cfg_scale": 4.0,
            "output_frames": 240,
            "output_fps": 16,
            "duration_seconds": 15.0,
            "completion_hook": (
                "HunyuanVideoAdapter.complete_native_pipeline_temporal_protocol"
            ),
            "one_way_model_transition": True,
        }
        required_positive = ("height", "width")
    else:
        expected = {
            "num_frames": 81,
            "native_fps": 16,
            "output_frames": 240,
            "output_fps": 16,
            "duration_seconds": 15.0,
            "completion_hook": "WanAdapter.complete_native_pipeline_temporal_protocol",
            "one_way_model_transition": True,
        }
        required_positive = ()
    mismatches = {
        key: {"expected": value, "observed": contract.get(key)}
        for key, value in expected.items()
        if contract.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"{family} native temporal call contract drifted: {mismatches}.")
    invalid_positive = [
        key
        for key in required_positive
        if not isinstance(contract.get(key), int) or int(contract[key]) <= 0
    ]
    if invalid_positive:
        raise RuntimeError(
            f"{family} native temporal contract requires positive integer fields: "
            f"{invalid_positive}."
        )
    return dict(contract)


def _validate_native_temporal_call_kwargs(
    *,
    family: str,
    call_kwargs: dict[str, Any],
    contract: dict[str, Any],
) -> None:
    """Fail before inference if signature filtering weakened a temporal call."""

    expected_num_frames = {
        "cogvideox": 49,
        "hunyuan_video": 121,
        "wan": 81,
    }.get(family)
    if expected_num_frames is None:
        raise ValueError(f"Unsupported native temporal family: {family!r}.")
    if contract.get("num_frames") != expected_num_frames:
        raise RuntimeError(
            f"{family} temporal contract expected {expected_num_frames} native frames; "
            f"got {contract.get('num_frames')!r}."
        )
    if call_kwargs.get("num_frames") != expected_num_frames:
        raise RuntimeError(
            f"{family} native pipeline call must request exactly {expected_num_frames} frames; "
            f"got {call_kwargs.get('num_frames')!r}."
        )
    if not isinstance(call_kwargs.get("generator"), torch.Generator):
        raise RuntimeError(f"{family} native temporal call requires an explicit torch.Generator.")
    for key in ("height", "width", "num_inference_steps", "guidance_scale"):
        if key not in call_kwargs or call_kwargs[key] is None:
            raise RuntimeError(f"{family} native temporal call is missing required {key!r}.")
    if call_kwargs.get("output_type") != "pil" or call_kwargs.get("return_dict") is not True:
        raise RuntimeError(
            f"{family} native temporal call requires output_type='pil' and return_dict=True."
        )

    if family in {"cogvideox", "hunyuan_video"}:
        for key in ("height", "width", "num_inference_steps", "guidance_scale"):
            if call_kwargs.get(key) != contract.get(key):
                raise RuntimeError(
                    f"{family} native first-call {key} differs from its frozen contract."
                )

    if family == "hunyuan_video":
        if call_kwargs.get("true_cfg_scale") != contract.get("true_cfg_scale"):
            raise RuntimeError("HunyuanVideo native true-CFG contract drifted.")
        expected_dimensions = {
            "height": int(contract["height"]),
            "width": int(contract["width"]),
        }
        required_conditioning = {
            "prompt_embeds",
            "pooled_prompt_embeds",
            "prompt_attention_mask",
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "negative_prompt_attention_mask",
        }
        missing_conditioning = sorted(required_conditioning - set(call_kwargs))
        if missing_conditioning:
            raise RuntimeError(
                "HunyuanVideo native temporal call lost frozen conditioning tensors: "
                f"{missing_conditioning}."
            )
        forbidden = sorted(
            {"prompt", "prompt_2", "negative_prompt", "negative_prompt_2"} & set(call_kwargs)
        )
        if forbidden:
            raise RuntimeError(
                f"HunyuanVideo native temporal call contains forbidden raw prompts: {forbidden}."
            )
        for key, expected_value in expected_dimensions.items():
            if call_kwargs.get(key) != expected_value:
                raise RuntimeError(
                    f"HunyuanVideo native temporal call {key} must be {expected_value}; "
                    f"got {call_kwargs.get(key)!r}."
                )
        return

    for key in ("prompt", "negative_prompt"):
        value = call_kwargs.get(key)
        if not isinstance(value, str) or not value:
            raise RuntimeError(f"{family} native temporal call requires a non-empty {key}.")
    if call_kwargs.get("num_videos_per_prompt") != 1:
        raise RuntimeError(
            f"{family} native temporal completion requires num_videos_per_prompt=1."
        )


def _invoke_native_temporal_first_call(
    *,
    pipe: Any,
    family: str,
    call_kwargs: dict[str, Any],
    contract: dict[str, Any],
) -> Any:
    """Validate and invoke the model-native first segment without frame fallback."""

    _validate_native_temporal_call_kwargs(
        family=family,
        call_kwargs=call_kwargs,
        contract=contract,
    )
    if not callable(pipe):
        raise RuntimeError(f"{family} native pipeline is not callable.")
    return pipe(**call_kwargs)


def _complete_native_temporal_protocol(
    *,
    adapter: Any,
    family: str,
    first_segment_media: Any,
    call_kwargs: dict[str, Any],
    contract: dict[str, Any] | None,
) -> list[list[Any]]:
    if contract is None:
        raise RuntimeError(f"{family} completion received no frozen native call contract.")
    _validate_native_temporal_call_kwargs(
        family=family,
        call_kwargs=call_kwargs,
        contract=contract,
    )
    completion_hook = getattr(adapter, "complete_native_pipeline_temporal_protocol", None)
    if not callable(completion_hook):
        raise RuntimeError(
            f"{family} native temporal execution requires "
            "complete_native_pipeline_temporal_protocol."
        )
    if family == "hunyuan_video":
        first_call_kwargs = {
            key: value for key, value in call_kwargs.items() if key != "generator"
        }
        media = completion_hook(
            first_segment_media,
            prompt=None,
            negative_prompt=None,
            generator=call_kwargs["generator"],
            **first_call_kwargs,
        )
    else:
        media = completion_hook(
            first_segment_media,
            prompt=call_kwargs["prompt"],
            negative_prompt=call_kwargs["negative_prompt"],
            generator=call_kwargs["generator"],
            num_inference_steps=int(call_kwargs["num_inference_steps"]),
            guidance_scale=float(call_kwargs["guidance_scale"]),
            height=int(call_kwargs["height"]),
            width=int(call_kwargs["width"]),
            guidance_scale_2=(
                float(call_kwargs["guidance_scale_2"])
                if call_kwargs.get("guidance_scale_2") is not None
                else None
            ),
        )
    _validate_exact_video_frame_count(
        media,
        expected_frames=int(contract["output_frames"]),
        family=family,
    )
    return media


def _complete_wan_native_temporal_protocol(
    *,
    adapter: Any,
    first_segment_media: Any,
    call_kwargs: dict[str, Any],
    contract: dict[str, Any] | None,
) -> list[list[Any]]:
    """Compatibility entrypoint retaining Wan's reviewed completion behavior."""

    return _complete_native_temporal_protocol(
        adapter=adapter,
        family="wan",
        first_segment_media=first_segment_media,
        call_kwargs=call_kwargs,
        contract=contract,
    )


def _validate_exact_video_frame_count(
    media: Any,
    *,
    expected_frames: int,
    family: str,
) -> None:
    if (
        not isinstance(media, list)
        or not media
        or any(not isinstance(video, list) for video in media)
    ):
        raise RuntimeError(f"{family} temporal output must be a non-empty batch of frame lists.")
    observed = [len(video) for video in media]
    if any(count != expected_frames for count in observed):
        raise RuntimeError(
            f"{family} temporal output must contain exactly {expected_frames} frames per video; "
            f"got {observed}."
        )


def _completed_temporal_provenance(*, adapter: Any, family: str) -> dict[str, Any]:
    provenance_hook = getattr(adapter, "conditioning_provenance", None)
    if not callable(provenance_hook):
        raise RuntimeError(f"{family} adapter does not expose conditioning_provenance.")
    provenance = provenance_hook()
    if not isinstance(provenance, dict):
        raise RuntimeError(f"{family} conditioning provenance must be a mapping.")
    temporal = provenance.get("temporal_generation")
    if not isinstance(temporal, dict):
        raise RuntimeError(f"{family} completion did not publish temporal_generation provenance.")
    status = str(temporal.get("status", ""))
    if not status or "not_yet" in status:
        raise RuntimeError(
            f"{family} temporal provenance is not completed after postprocessing: {status!r}."
        )
    expected_path = {
        "CogVideoX": "official_native_negative_three_call",
        "HunyuanVideo": "native_negative_t2v_then_two_native_i2v_calls",
        "Wan": "native_negative_t2v_then_two_native_i2v_calls",
    }.get(family)
    if expected_path is None:
        raise ValueError(f"Unsupported temporal provenance family: {family!r}.")
    if temporal.get("generation_path") != expected_path:
        raise RuntimeError(
            f"{family} temporal provenance recorded the wrong generation path: "
            f"{temporal.get('generation_path')!r}."
        )
    return dict(temporal)


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


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_native_run_timing(
    output_dir: Path,
    status: str,
    started_at: str,
    run_start: float,
    config: dict[str, Any],
    model_id: str,
    pipeline_class_name: str | None,
    records: list[dict[str, Any]],
    reason: str | None = None,
    pipeline_load_seconds: float | None = None,
) -> None:
    payload = {
        "schema_version": 1,
        "status": status,
        "reason": reason,
        "started_at": started_at,
        "ended_at": _utc_now_iso(),
        "total_seconds": time.perf_counter() - run_start,
        "pipeline_load_seconds": pipeline_load_seconds,
        "model": {
            "adapter": get_path(config, "model.adapter"),
            "model_id": model_id,
            "revision": get_path(config, "model.revision"),
            "pipeline_class": pipeline_class_name,
        },
        "generation": _generation_for_task(dict(config.get("generation", {}))),
        "benchmark": config.get("benchmark", {}),
        "records": records,
    }
    write_json(output_dir / "run_timing.json", payload)


def _aligned_native_num_frames(config: dict[str, Any], requested_num_frames: int) -> int:
    alignment = dict(get_path(config, "model.video_frame_alignment", {}) or {})
    if alignment.get("mode") != "one_plus_multiple":
        return requested_num_frames
    return _round_up_to_one_plus_multiple(requested_num_frames, int(alignment.get("multiple", 4)))


def _round_up_to_one_plus_multiple(value: int, multiple: int) -> int:
    if value <= 1:
        return 1
    remainder = (value - 1) % multiple
    return value if remainder == 0 else value + (multiple - remainder)


def _generation_for_task(generation: dict[str, Any]) -> dict[str, Any]:
    task = str(generation.get("task", "text_to_image"))
    if task == "text_to_image":
        for key in ("num_frames", "fps", "duration_seconds", "frame_rate"):
            generation.pop(key, None)
    return generation


def _crop_video_media(media: Any, num_frames: int) -> Any:
    if num_frames <= 0 or not isinstance(media, list):
        return media
    cropped = []
    for frames in media:
        if isinstance(frames, list) and len(frames) > num_frames:
            cropped.append(frames[:num_frames])
        else:
            cropped.append(frames)
    return cropped


def _media_summary(media: Any, task: str, fps: int) -> dict[str, Any]:
    summary: dict[str, Any] = {"present": media is not None}
    if media is None:
        return summary
    if task == "text_to_video":
        videos = media if isinstance(media, list) else [media]
        frame_counts = [len(frames) for frames in videos if isinstance(frames, (list, tuple))]
        summary.update(
            {
                "num_videos": len(frame_counts),
                "frame_counts": frame_counts,
                "fps": fps,
                "durations_seconds": [count / fps for count in frame_counts] if fps else [],
            }
        )
    elif task == "text_to_image":
        images = media if isinstance(media, list) else [media]
        summary["num_images"] = len(images)
    return summary


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
