from __future__ import annotations

import hashlib
import json
import os
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from hierasafe_flow.adapters.base import DenoisingStepContext
from hierasafe_flow.adapters.flux_dual_view_adapter import (
    FLUX_DUAL_VIEW_CONFIG_KEY,
    FLUX_DUAL_VIEW_FORBIDDEN_RAW_CONFIG_KEYS,
    FLUX_DUAL_VIEW_MODEL_ID,
    FLUX_DUAL_VIEW_MODEL_REVISION,
    validate_flux_dual_view_conditioning,
)
from hierasafe_flow.adapters.registry import create_adapter
from hierasafe_flow.generation.conditioning_cache import (
    StateAwareConditionCache,
    encoding_fingerprint,
)
from hierasafe_flow.generation.save_outputs import save_generation_output, save_generation_report
from hierasafe_flow.logging_utils.experiment_tracker import ExperimentTracker
from hierasafe_flow.logging_utils.logger import setup_logger
from hierasafe_flow.logging_utils.system_info import collect_system_info
from hierasafe_flow.logging_utils.tensorboard import TensorBoardLogger
from hierasafe_flow.steering.bottleneck import BottleneckConfig, HierarchicalVectorFieldBottleneck
from hierasafe_flow.steering.concept_graph import ConceptHierarchy
from hierasafe_flow.steering.negative_guidance import (
    NegativeConceptVectorGuidance,
    NegativeGuidanceConfig,
)
from hierasafe_flow.steering.shapley import (
    SHAPLEY_INTERVENTION_IDENTITY,
    SHAPLEY_TRACE_SCHEMA_VERSION,
    ShapleyConceptSteerer,
    ShapleyConfig,
)
from hierasafe_flow.utils.config import collect_prompts, get_path
from hierasafe_flow.utils.device import configure_cuda, resolve_device, resolve_dtype
from hierasafe_flow.utils.io import ensure_dir, write_json
from hierasafe_flow.utils.seed import make_generator, seed_everything


@dataclass
class GenerationRecord:
    prompt: str
    sample_id: str
    output_paths: dict[str, str]


@dataclass
class RunResult:
    output_dir: str
    records: list[GenerationRecord]


_STEERING_MODE_CANONICAL: dict[str, str] = {}


def _canonicalize_steering_mode(mode: str) -> str:
    return _STEERING_MODE_CANONICAL.get(mode, mode)


class GenerationRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = dict(config)
        self.config["generation"] = _generation_for_task(dict(self.config.get("generation", {})))
        self.config["model"] = _bind_flux_dual_view_conditioning(
            model_config=dict(self.config.get("model", {})),
            generation_config=dict(self.config.get("generation", {})),
        )
        project_root = Path(get_path(config, "_meta.project_root", Path.cwd()))
        output_dir = Path(get_path(config, "logging.output_dir", "outputs"))
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        self.output_dir = ensure_dir(output_dir)

        self.logger = setup_logger(
            output_dir=self.output_dir,
            level=str(get_path(config, "logging.level", "INFO")),
        )
        tracker_config = (
            self.config if FLUX_DUAL_VIEW_CONFIG_KEY in self.config.get("model", {}) else config
        )
        self.tracker = ExperimentTracker.create(self.output_dir, tracker_config)
        self.tensorboard = TensorBoardLogger(
            self.output_dir / "tensorboard",
            enabled=bool(get_path(config, "logging.tensorboard", True)),
        )

        seed = int(get_path(config, "project.seed", 1234))
        seed_everything(seed)
        self.device = resolve_device(str(get_path(config, "runtime.device", "auto")))
        self.dtype = resolve_dtype(
            str(get_path(config, "runtime.dtype", get_path(config, "model.torch_dtype", "auto")))
        )
        configure_cuda(bool(get_path(config, "runtime.allow_tf32", True)))
        write_json(self.output_dir / "system_info.json", collect_system_info())

        model_config = dict(self.config.get("model", {}))
        generation_config = dict(self.config.get("generation", {}))
        task = str(generation_config.get("task", "text_to_image"))
        for key in ("height", "width"):
            if key in generation_config and key not in model_config:
                model_config[key] = generation_config[key]
        if task == "text_to_video":
            for key in ("num_frames", "fps"):
                if key in generation_config and key not in model_config:
                    model_config[key] = generation_config[key]
        model_config["guidance_scale"] = get_path(config, "generation.guidance_scale", None)
        self.adapter = create_adapter(model_config, device=self.device, dtype=self.dtype)

        hierarchy_path = Path(str(get_path(config, "concepts.hierarchy_path")))
        if not hierarchy_path.is_absolute():
            hierarchy_path = project_root / hierarchy_path
        self.hierarchy = ConceptHierarchy.from_yaml_file(hierarchy_path)
        requested_mode = str(get_path(config, "steering.mode", "bottleneck"))
        self.steering_mode = _canonicalize_steering_mode(requested_mode)
        self.condition_cache = StateAwareConditionCache(namespace="generation_runner")
        self.steerer = HierarchicalVectorFieldBottleneck(
            hierarchy=self.hierarchy,
            config=BottleneckConfig.from_dict(config.get("steering", {})),
            condition_cache=self.condition_cache,
        )
        self.shapley_steerer: ShapleyConceptSteerer | None = None
        if self.steering_mode == "shapley_concept_steering":
            self.shapley_steerer = ShapleyConceptSteerer(
                hierarchy=self.hierarchy,
                bottleneck_config=BottleneckConfig.from_dict(config.get("steering", {})),
                shapley_config=ShapleyConfig.from_dict(
                    config.get("steering", {}),
                    default_seed=seed,
                ),
            )
            self.shapley_steerer.set_condition_cache(self.condition_cache)
        self.negative_guider = NegativeConceptVectorGuidance(
            NegativeGuidanceConfig.from_dict(config.get("negative_guidance", {})),
            condition_cache=self.condition_cache,
        )
        self._conditioning_preflight: dict[str, Any] | None = None

    def run(self, prompt: str | None = None) -> RunResult:
        run_started_at = _utc_now_iso()
        run_start = time.perf_counter()
        prompts = self._collect_prompts(prompt)
        validate_primary_prompts = getattr(self.adapter, "validate_primary_prompts", None)
        if callable(validate_primary_prompts):
            validate_primary_prompts(prompts)
        self.logger.info(
            "Loading adapter %s for %s", self.adapter.adapter_name, self.adapter.model_id
        )
        adapter_load_start = time.perf_counter()
        self.adapter.load()
        adapter_load_seconds = time.perf_counter() - adapter_load_start
        self.tracker.log_event("adapter_loaded", self.adapter.inspect())

        conditioning_preflight_seconds = 0.0
        configured_conditioning_plan = (
            get_path(self.config, "model.hunyuan_dual_view_conditioning") is not None
            or get_path(self.config, f"model.{FLUX_DUAL_VIEW_CONFIG_KEY}") is not None
        )
        if configured_conditioning_plan:
            preflight = getattr(self.adapter, "preflight_conditioning_plan", None)
            if not callable(preflight):
                raise RuntimeError(
                    "A dual-view conditioning plan is configured, but the adapter "
                    "does not expose its required whole-plan preflight."
                )
            preflight_start = time.perf_counter()
            self._conditioning_preflight = preflight()
            conditioning_preflight_seconds = time.perf_counter() - preflight_start
            self.tracker.log_event(
                "conditioning_plan_preflight_completed",
                self._conditioning_preflight,
            )

        records: list[GenerationRecord] = []
        for index, item in enumerate(prompts):
            sample_id = f"sample_{index:04d}"
            with torch.inference_mode():
                records.append(self._run_prompt(item, sample_id))

        self.tensorboard.close()
        self.tracker.log_event("run_finished", {"num_records": len(records)})
        run_timing = {
            "schema_version": 1,
            "status": "completed",
            "started_at": run_started_at,
            "ended_at": _utc_now_iso(),
            "total_seconds": time.perf_counter() - run_start,
            "adapter_load_seconds": adapter_load_seconds,
            "conditioning_preflight_seconds": conditioning_preflight_seconds,
            "conditioning_preflight": self._conditioning_preflight,
            "model": {
                "adapter": self.adapter.adapter_name,
                "model_id": self.adapter.model_id,
                "revision": self.adapter.config.get("revision"),
                "pipeline_class": self.adapter.pipeline_class_name,
            },
            "generation": dict(self.config.get("generation", {})),
            "benchmark": self.config.get("benchmark", {}),
            "records": [asdict(record) for record in records],
        }
        write_json(self.output_dir / "run_timing.json", run_timing)
        return RunResult(output_dir=str(self.output_dir), records=records)

    def _collect_prompts(self, prompt: str | None) -> list[str]:
        cfg = self.config
        prompt_file = get_path(cfg, "generation.prompt_file")
        if prompt_file and not Path(str(prompt_file)).is_absolute():
            project_root = Path(get_path(cfg, "_meta.project_root", Path.cwd()))
            cfg = dict(cfg)
            cfg["generation"] = dict(cfg.get("generation", {}))
            cfg["generation"]["prompt_file"] = str(project_root / str(prompt_file))
        return collect_prompts(cfg, prompt)

    def _run_prompt(self, prompt: str, sample_id: str) -> GenerationRecord:
        sample_started_at = _utc_now_iso()
        sample_start = time.perf_counter()
        self.logger.info("Generating %s", sample_id)
        generation = self.config.get("generation", {})
        task = str(generation.get("task", "text_to_image"))
        num_steps = int(generation.get("num_inference_steps", 28))
        batch_size = int(generation.get("num_outputs_per_prompt", 1))
        seed = int(get_path(self.config, "project.seed", 1234))
        generator = make_generator(seed, self.device)
        sample_cache_mark = self.condition_cache.mark()

        latent_kwargs = {
            key: value
            for key, value in generation.items()
            if key not in {"prompt", "prompt_file", FLUX_DUAL_VIEW_CONFIG_KEY}
        }
        prepare_start = time.perf_counter()
        latents, state = self.adapter.prepare_initial_latents(
            prompt=prompt,
            batch_size=batch_size,
            generator=generator,
            **latent_kwargs,
        )
        state.extra["num_steps"] = num_steps
        timesteps = self.adapter.set_timesteps(num_steps, latents=latents, state=state)
        prepare_seconds = time.perf_counter() - prepare_start
        trace: list[dict[str, Any]] = []
        step_contexts: list[DenoisingStepContext] = []
        step_timings: list[dict[str, Any]] = []
        log_every_steps = max(1, int(get_path(self.config, "logging.log_every_steps", 1)))

        denoising_start = time.perf_counter()
        for step_index, timestep in enumerate(tqdm(timesteps, desc=sample_id, leave=False)):
            step_start = time.perf_counter()
            context = self.adapter.denoising_step_context(
                step_index,
                len(timesteps),
                state,
            )
            _validate_next_step_context(context, step_contexts)
            step_contexts.append(context)
            state.extra["_active_denoising_step_context"] = context
            cache_mark = self.condition_cache.mark()
            prediction, step_trace = self._predict_step(
                latents=latents,
                timestep=timestep,
                state=state,
                prompt=prompt,
                step_index=step_index,
                num_steps=len(timesteps),
            )
            step_trace.segment = context.to_dict()
            step_trace.condition_calls = list(self.condition_cache.evidence_since(cache_mark))
            step_trace.dynamic_latent_fingerprint = encoding_fingerprint(latents)
            step_trace.protected_state = {
                **step_trace.protected_state,
                **_protected_state_trace(state),
            }
            trace.append(asdict(step_trace))
            self._log_trace(step_trace, step_index)
            result = self.adapter.scheduler_step(
                model_prediction=prediction,
                timestep=timestep,
                latents=latents,
                state=state,
                generator=generator,
            )
            latents, state = result.latents, result.state
            if step_index % log_every_steps == 0 or step_index == len(timesteps) - 1:
                progress = {
                    "sample_id": sample_id,
                    "step": step_index + 1,
                    "num_steps": len(timesteps),
                    "timestep": _timestep_for_log(timestep),
                }
                self.logger.info(
                    "Completed %s step %d/%d",
                    sample_id,
                    step_index + 1,
                    len(timesteps),
                )
                self.tracker.log_event("generation_step", progress)
            step_timings.append(
                {
                    "step_index": step_index,
                    "timestep": _timestep_for_log(timestep),
                    "seconds": time.perf_counter() - step_start,
                }
            )
        denoising_seconds = time.perf_counter() - denoising_start
        segment_trace_validation = _validate_complete_step_contexts(
            step_contexts,
            expected_global_steps=len(timesteps),
        )
        self.tracker.log_event("segment_trace_validated", segment_trace_validation)

        conceptsteer_trace_validation: dict[str, Any] | None = None
        shapley_trace_validation: dict[str, Any] | None = None
        if self.steering_mode == "bottleneck":
            conceptsteer_trace_validation = self.steerer.validate_run_trace(
                trace,
                num_steps=len(timesteps),
            )
            self.tracker.log_event("conceptsteer_trace_validated", conceptsteer_trace_validation)
        if self.steering_mode == "shapley_concept_steering":
            if self.shapley_steerer is None:
                raise RuntimeError("Shapley steering mode was not initialized.")
            shapley_trace_validation = self.shapley_steerer.validate_run_trace(
                trace,
                num_steps=len(timesteps),
            )
            protocol_binding = _shapley_protocol_binding(self.config)
            shapley_trace_validation["protocol_binding"] = deepcopy(protocol_binding)
            _bind_shapley_protocol_to_trace(trace, protocol_binding)
            self.tracker.log_event("shapley_trace_validated", shapley_trace_validation)

        decode_outputs = bool(get_path(self.config, "output.decode", True))
        media = None
        decode_seconds = 0.0
        if decode_outputs:
            decode_start = time.perf_counter()
            media = self.adapter.decode_latents(latents, state)
            decode_seconds = time.perf_counter() - decode_start
        temporal_evidence = self.adapter.take_temporal_evidence(state)
        if _adapter_requires_temporal_evidence(self.adapter) and temporal_evidence is None:
            raise RuntimeError(
                "Temporal protocol schema 2 completed without a TemporalEvidenceBundle."
            )
        if temporal_evidence is not None:
            temporal_evidence = temporal_evidence.with_runner_trace_evidence(trace)
            benchmark = self.config.get("benchmark")
            if not isinstance(benchmark, dict):
                raise RuntimeError("Segmented temporal run lacks benchmark identity metadata.")
            condition_id = benchmark.get("condition_id")
            manifest_sha256 = benchmark.get("manifest_sha256")
            attempt = benchmark.get("attempt")
            if not isinstance(condition_id, str) or not condition_id:
                raise RuntimeError("Segmented temporal run lacks condition_id.")
            if not isinstance(manifest_sha256, str):
                raise RuntimeError("Segmented temporal run lacks manifest SHA-256.")
            if isinstance(attempt, bool) or not isinstance(attempt, int):
                raise RuntimeError("Segmented temporal run lacks an integer attempt.")
            slurm_job = os.environ.get("SLURM_ARRAY_JOB_ID") or os.environ.get("SLURM_JOB_ID")
            slurm_task = os.environ.get("SLURM_ARRAY_TASK_ID")
            job_id = (
                f"slurm:{slurm_job}:{slurm_task}"
                if slurm_job and slurm_task is not None
                else (f"slurm:{slurm_job}" if slurm_job else f"local:{condition_id}")
            )
            temporal_evidence = temporal_evidence.with_execution_identity(
                condition_id=condition_id,
                attempt=attempt,
                manifest_sha256=manifest_sha256,
                job_id=job_id,
                metadata={
                    "checkpoint_set": benchmark.get("checkpoint_set"),
                    "checkpoint_set_sha256": benchmark.get("checkpoint_set_sha256"),
                    "artifact_manifest": benchmark.get("artifact_manifest"),
                    "artifact_manifest_sha256": benchmark.get("artifact_manifest_sha256"),
                    "segmented_temporal_contract": benchmark.get("segmented_temporal_contract"),
                },
            )
        save_start = time.perf_counter()
        report = self._build_sample_report(
            prompt=prompt,
            sample_id=sample_id,
            task=task,
            trace=trace,
            output_paths={},
            decode_outputs=decode_outputs,
            conceptsteer_trace_validation=conceptsteer_trace_validation,
            shapley_trace_validation=shapley_trace_validation,
            segment_trace_validation=segment_trace_validation,
            condition_cache_records=list(self.condition_cache.evidence_since(sample_cache_mark)),
        )
        output_paths = save_generation_output(
            media=media,
            latents=latents,
            trace=trace,
            output_dir=self.output_dir,
            sample_id=sample_id,
            task=task,
            save_latents=bool(get_path(self.config, "output.save_latents", True)),
            save_traces=bool(get_path(self.config, "output.save_traces", True)),
            image_format=str(get_path(self.config, "output.image_format", "png")),
            video_format=str(get_path(self.config, "output.video_format", "mp4")),
            fps=int(generation.get("fps", 16)),
            temporal_evidence=temporal_evidence,
            report=report if temporal_evidence is not None else None,
        )
        if temporal_evidence is None:
            report["output_paths"] = dict(output_paths)
            output_paths["report"] = save_generation_report(report, self.output_dir, sample_id)
        timing_path = ensure_dir(self.output_dir / sample_id) / "timing.json"
        output_paths["timing"] = str(timing_path)
        save_seconds = time.perf_counter() - save_start
        timing = {
            "schema_version": 1,
            "status": "completed",
            "started_at": sample_started_at,
            "ended_at": _utc_now_iso(),
            "total_seconds": time.perf_counter() - sample_start,
            "phases_seconds": {
                "prepare_latents_and_timesteps": prepare_seconds,
                "denoising": denoising_seconds,
                "decode": decode_seconds,
                "save_media_trace_and_report": save_seconds,
            },
            "step_timings": step_timings,
            "sample_id": sample_id,
            "prompt": prompt,
            "task": task,
            "benchmark": self.config.get("benchmark", {}),
            "model": {
                "adapter": self.adapter.adapter_name,
                "model_id": self.adapter.model_id,
                "revision": self.adapter.config.get("revision"),
                "pipeline_class": self.adapter.pipeline_class_name,
            },
            "generation": dict(generation),
            "latent_shape": list(latents.shape) if hasattr(latents, "shape") else None,
            "media": _media_summary(media, task=task, fps=int(generation.get("fps", 16))),
            "output_paths": dict(output_paths),
        }
        write_json(timing_path, timing)
        self.tracker.log_event("sample_finished", {"sample_id": sample_id, "paths": output_paths})
        return GenerationRecord(prompt=prompt, sample_id=sample_id, output_paths=output_paths)

    def _predict_step(
        self,
        latents: torch.Tensor,
        timestep: Any,
        state: Any,
        prompt: str,
        step_index: int,
        num_steps: int,
    ) -> tuple[torch.Tensor, Any]:
        if self.steering_mode == "bottleneck":
            return self.steerer.steer_step(
                adapter=self.adapter,
                latents=latents,
                timestep=timestep,
                state=state,
                prompt=prompt,
                step_index=step_index,
                num_steps=num_steps,
            )
        if self.steering_mode == "negative_guidance":
            return self.negative_guider.steer_step(
                adapter=self.adapter,
                latents=latents,
                timestep=timestep,
                state=state,
                prompt=prompt,
                step_index=step_index,
                num_steps=num_steps,
            )
        if self.steering_mode == "shapley_concept_steering":
            if self.shapley_steerer is None:
                raise RuntimeError("Shapley steering mode was not initialized.")
            return self.shapley_steerer.steer_step(
                adapter=self.adapter,
                latents=latents,
                timestep=timestep,
                state=state,
                prompt=prompt,
                step_index=step_index,
                num_steps=num_steps,
            )
        if self.steering_mode == "none":
            condition = self.condition_cache.get_or_prepare_one(
                adapter=self.adapter,
                prompt=prompt,
                state=state,
                prompt_view="registered",
                call_role="base_current",
            )
            prediction = self.adapter.predict_vector_field(latents, timestep, condition, state)
            from hierasafe_flow.steering.bottleneck import BottleneckTrace
            from hierasafe_flow.utils.tensors import tensor_stats

            stats = tensor_stats(prediction)
            return prediction, BottleneckTrace(
                step_index=step_index,
                timestep=timestep.detach().flatten()[0].item()
                if hasattr(timestep, "detach")
                else timestep,
                enabled=False,
                concepts=[],
                base_stats=stats,
                steered_stats=stats,
            )
        raise ValueError(
            f"Unknown steering.mode '{self.steering_mode}'. Valid modes: "
            "bottleneck, shapley_concept_steering, negative_guidance, none."
        )

    def _log_trace(self, trace: Any, step_index: int) -> None:
        self.tensorboard.add_scalars("vector_field/base", trace.base_stats, step_index)
        self.tensorboard.add_scalars("vector_field/steered", trace.steered_stats, step_index)
        for concept in trace.concepts:
            prefix = f"concepts/{concept.concept_id}"
            self.tensorboard.add_scalars(f"{prefix}/activation", concept.activation, step_index)
            self.tensorboard.add_scalars(f"{prefix}/mask", concept.mask, step_index)

    def _build_sample_report(
        self,
        prompt: str,
        sample_id: str,
        task: str,
        trace: list[dict[str, Any]],
        output_paths: dict[str, str],
        decode_outputs: bool,
        conceptsteer_trace_validation: dict[str, Any] | None,
        shapley_trace_validation: dict[str, Any] | None,
        segment_trace_validation: dict[str, Any] | None,
        condition_cache_records: list[dict[str, Any]],
    ) -> dict[str, Any]:
        generation_config = dict(self.config.get("generation", {}))
        generation_config["prompt"] = prompt
        native_negative_prompt = dict(self.config.get("native_negative_prompt", {}))
        return {
            "schema_version": 1,
            "prompt": prompt,
            "sample_id": sample_id,
            "task": task,
            "benchmark": self.config.get("benchmark", {}),
            "model": {
                "adapter": self.adapter.adapter_name,
                "model_id": self.adapter.model_id,
                "revision": self.adapter.config.get("revision"),
                "pipeline_class": self.adapter.pipeline_class_name,
            },
            "condition": {
                "steering_mode": self.steering_mode,
                "decode_outputs": decode_outputs,
                "is_native_negative_prompt": bool(native_negative_prompt),
                "negative_prompt": native_negative_prompt.get("prompt"),
            },
            "generation": generation_config,
            "conditioning_provenance": self.adapter.conditioning_provenance(),
            "steering": self.config.get("steering", {}),
            "concept_hierarchy": {
                "name": self.hierarchy.name,
                "neutral_concept": self.hierarchy.neutral_concept,
                "pairs": [asdict(pair) for pair in self.hierarchy.pairs],
            },
            "output_paths": dict(output_paths),
            "interpretability": {
                "note": (
                    "steering_delta_stats summarize the vector-field update. Shapley-mode "
                    "values are functional attributions for feature-channel coordinates of the "
                    "denoiser vector field and use safe-background rollback with strict "
                    "model-dtype-quantized score descent; they are not text embeddings or "
                    "latent-state causal effects. This code does not compute backpropagation "
                    "gradients."
                ),
                "segment_trace_validation": segment_trace_validation,
                "conceptsteer_trace_validation": conceptsteer_trace_validation,
                "shapley_trace_validation": shapley_trace_validation,
                "timesteps": trace,
                "concepts_per_step": _concepts_per_step(trace),
            },
            "conditioning_cache": {
                "schema_version": 1,
                "namespace": self.condition_cache.namespace,
                "records": condition_cache_records,
            },
        }


def _timestep_for_log(timestep: Any) -> float | int | str:
    if isinstance(timestep, torch.Tensor):
        return float(timestep.detach().flatten()[0].item())
    if isinstance(timestep, (float, int, str)):
        return timestep
    return str(timestep)


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _generation_for_task(generation: dict[str, Any]) -> dict[str, Any]:
    task = str(generation.get("task", "text_to_image"))
    if task == "text_to_image":
        for key in ("num_frames", "fps", "duration_seconds", "frame_rate"):
            generation.pop(key, None)
    return generation


def _bind_flux_dual_view_conditioning(
    *,
    model_config: dict[str, Any],
    generation_config: dict[str, Any],
) -> dict[str, Any]:
    """Bind one structured FLUX.1 prompt-view plan to the adapter config.

    Benchmark manifests naturally carry generation inputs, while adapters are
    constructed from the model section.  This is the only bridge between those
    surfaces.  It admits either location for backwards-compatible orchestration,
    requires equality when both are present, and rejects every competing raw
    Diffusers prompt/embedding call key before an output directory can be used
    for generation.
    """

    generation_raw = generation_config.get(FLUX_DUAL_VIEW_CONFIG_KEY)
    model_raw = model_config.get(FLUX_DUAL_VIEW_CONFIG_KEY)
    if generation_raw is None and model_raw is None:
        return model_config

    generation_plan = (
        validate_flux_dual_view_conditioning(generation_raw) if generation_raw is not None else None
    )
    model_plan = validate_flux_dual_view_conditioning(model_raw) if model_raw is not None else None
    if generation_plan is not None and model_plan is not None and generation_plan != model_plan:
        raise ValueError("Generation and model Flux dual-view conditioning plans conflict.")
    plan = generation_plan if generation_plan is not None else model_plan
    assert plan is not None

    adapter_name = str(model_config.get("adapter") or "").lower()
    model_id = str(model_config.get("model_id") or "")
    revision = str(model_config.get("revision") or "")
    if (
        adapter_name != "flux_dual_view"
        or model_id != FLUX_DUAL_VIEW_MODEL_ID
        or revision != FLUX_DUAL_VIEW_MODEL_REVISION
    ):
        raise ValueError(
            f"{FLUX_DUAL_VIEW_CONFIG_KEY} requires the explicit flux_dual_view adapter "
            "and exact registered FLUX.1-dev model/revision pair."
        )

    forbidden_generation_keys = FLUX_DUAL_VIEW_FORBIDDEN_RAW_CONFIG_KEYS - {"prompt"}
    generation_conflicts = sorted(forbidden_generation_keys.intersection(generation_config))
    model_conflicts = sorted(FLUX_DUAL_VIEW_FORBIDDEN_RAW_CONFIG_KEYS.intersection(model_config))
    nested_conflicts: list[str] = []
    for container_key in ("call_kwargs", "pipeline_call_kwargs", "generation_kwargs"):
        nested = generation_config.get(container_key)
        if isinstance(nested, dict):
            nested_conflicts.extend(
                f"{container_key}.{key}"
                for key in sorted(FLUX_DUAL_VIEW_FORBIDDEN_RAW_CONFIG_KEYS.intersection(nested))
            )
    conflicts = [
        *(f"generation.{key}" for key in generation_conflicts),
        *(f"model.{key}" for key in model_conflicts),
        *(f"generation.{key}" for key in nested_conflicts),
    ]
    if conflicts:
        raise ValueError(
            f"{FLUX_DUAL_VIEW_CONFIG_KEY} conflicts with raw conditioning call keys: {conflicts}."
        )

    bound = dict(model_config)
    bound[FLUX_DUAL_VIEW_CONFIG_KEY] = deepcopy(plan)
    return bound


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


def _concepts_per_step(trace: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for step in trace:
        concepts = []
        for concept in step.get("concepts", []):
            activation = concept.get("activation") or {}
            concepts.append(
                {
                    "concept_id": concept.get("concept_id"),
                    "parent": concept.get("parent"),
                    "unsafe_concept": concept.get("unsafe_concept"),
                    "target_concept": concept.get("target_concept"),
                    "lambda_t": concept.get("lambda_t"),
                    "activated": float(activation.get("max", 0.0) or 0.0) > 0.0,
                    "activation": activation,
                    "mask": concept.get("mask"),
                    "steering_delta_stats": concept.get("steering_delta_stats"),
                }
            )
        rows.append(
            {
                "step_index": step.get("step_index"),
                "timestep": step.get("timestep"),
                "enabled": step.get("enabled"),
                "segment": step.get("segment"),
                "concepts": concepts,
            }
        )
    return rows


def _validate_next_step_context(
    context: DenoisingStepContext,
    prior: list[DenoisingStepContext],
) -> None:
    expected_global = len(prior)
    if context.global_step_index != expected_global:
        raise RuntimeError(
            "Adapter denoising context did not cover the next global step exactly once."
        )
    if context.global_num_steps <= expected_global:
        raise RuntimeError("Adapter denoising context declares too few global steps.")
    if not prior:
        if context.segment_index != 0 or context.local_step_index != 0:
            raise RuntimeError("Segmented denoising must start at segment 0, local step 0.")
        return
    previous = prior[-1]
    if context.global_num_steps != previous.global_num_steps:
        raise RuntimeError("Global denoising step count changed during generation.")
    if context.segment_count != previous.segment_count:
        raise RuntimeError("Declared temporal segment count changed during generation.")
    if context.segment_index == previous.segment_index:
        if context.local_step_index != previous.local_step_index + 1:
            raise RuntimeError("Segment-local denoising coverage has a gap or duplicate.")
        immutable = (
            "local_num_steps",
            "model_role",
            "model_id",
            "model_revision",
            "condition_epoch",
            "anchor_sha256",
            "segment_seed",
        )
        if any(getattr(context, key) != getattr(previous, key) for key in immutable):
            raise RuntimeError("Authenticated segment identity changed before its boundary.")
        return
    if context.segment_index != previous.segment_index + 1:
        raise RuntimeError("Temporal segment ordering has a gap or reversal.")
    if previous.local_step_index != previous.local_num_steps - 1:
        raise RuntimeError("Adapter transitioned before completing the prior native schedule.")
    if context.local_step_index != 0:
        raise RuntimeError("New temporal segment did not reset to local step zero.")


def _validate_complete_step_contexts(
    contexts: list[DenoisingStepContext],
    *,
    expected_global_steps: int,
) -> dict[str, Any]:
    if len(contexts) != expected_global_steps or not contexts:
        raise RuntimeError("Denoising step-context coverage is incomplete.")
    last = contexts[-1]
    if last.global_step_index != expected_global_steps - 1:
        raise RuntimeError("Denoising step contexts do not end at the final global step.")
    if last.segment_index != last.segment_count - 1:
        raise RuntimeError("Denoising ended before the declared final temporal segment.")
    if last.local_step_index != last.local_num_steps - 1:
        raise RuntimeError("Denoising ended before the final segment schedule completed.")
    per_segment: list[dict[str, Any]] = []
    for segment_index in range(last.segment_count):
        rows = [row for row in contexts if row.segment_index == segment_index]
        if not rows:
            raise RuntimeError(f"Temporal segment {segment_index} has no denoising contexts.")
        if [row.local_step_index for row in rows] != list(range(rows[0].local_num_steps)):
            raise RuntimeError(f"Temporal segment {segment_index} local coverage is incomplete.")
        per_segment.append(
            {
                "segment_index": segment_index,
                "model_role": rows[0].model_role,
                "model_id": rows[0].model_id,
                "model_revision": rows[0].model_revision,
                "condition_epoch": rows[0].condition_epoch,
                "anchor_sha256": rows[0].anchor_sha256,
                "segment_seed": rows[0].segment_seed,
                "local_num_steps": rows[0].local_num_steps,
                "global_step_start": rows[0].global_step_index,
                "global_step_end": rows[-1].global_step_index,
            }
        )
    return {
        "schema_version": 1,
        "status": "passed",
        "global_num_steps": expected_global_steps,
        "segment_count": last.segment_count,
        "segments": per_segment,
    }


def _protected_state_trace(state: Any) -> dict[str, Any]:
    raw = getattr(state, "extra", {}).get("protected_state_trace", {})
    if not isinstance(raw, dict):
        raise RuntimeError("Adapter protected_state_trace must be a mapping when present.")
    return _trace_safe_value(raw)


def _trace_safe_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "fingerprint": encoding_fingerprint(value),
        }
    if isinstance(value, dict):
        return {str(key): _trace_safe_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_trace_safe_value(item) for item in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return {"type": type(value).__name__, "fingerprint": encoding_fingerprint(value)}


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _shapley_protocol_binding(config: dict[str, Any]) -> dict[str, Any]:
    steering = config.get("steering")
    benchmark = config.get("benchmark") or {}
    if not isinstance(steering, dict) or not isinstance(benchmark, dict):
        raise RuntimeError("Shapley generation has malformed steering/benchmark metadata.")
    shapley_config = steering.get("shapley")
    if not isinstance(shapley_config, dict):
        raise RuntimeError("Shapley generation lacks its exact estimator config.")
    benchmark_config = benchmark.get("shapley", shapley_config)
    provenance = benchmark.get("shapley_provenance") or {
        "protocol_version": 2,
        "trace_schema_version": SHAPLEY_TRACE_SCHEMA_VERSION,
        "intervention_identity": SHAPLEY_INTERVENTION_IDENTITY,
        "qualification_eligible": False,
        "source": "unregistered_non_benchmark_runner_config",
    }
    if shapley_config != benchmark_config:
        raise RuntimeError("Shapley runner and frozen benchmark configs are not identical.")
    if not isinstance(provenance, dict):
        raise RuntimeError("Shapley generation lacks frozen protocol provenance.")
    if provenance.get("protocol_version") != 2 or provenance.get("trace_schema_version") != 2:
        raise RuntimeError("Shapley generation requires protocol and trace schema version 2.")
    return {
        "schema_version": 1,
        "shapley_config": deepcopy(shapley_config),
        "shapley_config_sha256": _canonical_sha256(shapley_config),
        "shapley_provenance": deepcopy(provenance),
        "shapley_provenance_sha256": _canonical_sha256(provenance),
    }


def _bind_shapley_protocol_to_trace(
    trace: list[dict[str, Any]], protocol_binding: dict[str, Any]
) -> None:
    bound_count = 0
    for step in trace:
        concepts = step.get("concepts")
        if not isinstance(concepts, list):
            raise RuntimeError("Shapley trace concepts are malformed.")
        for concept in concepts:
            if not isinstance(concept, dict):
                raise RuntimeError("Shapley trace concept record is malformed.")
            shapley = concept.get("shapley")
            if shapley is None:
                continue
            if not isinstance(shapley, dict) or shapley.get("schema_version") != 2:
                raise RuntimeError("Shapley trace contains malformed protocol-v2 evidence.")
            shapley["protocol_binding"] = deepcopy(protocol_binding)
            bound_count += 1
    if bound_count == 0:
        raise RuntimeError("Shapley trace contains no protocol-bound intervention records.")


def _adapter_requires_temporal_evidence(adapter: Any) -> bool:
    config = getattr(adapter, "config", {})
    if not isinstance(config, dict):
        return False
    return any(
        isinstance(value, dict)
        and value.get("schema_version") == 2
        and "temporal_protocol" in str(key)
        for key, value in config.items()
    )
