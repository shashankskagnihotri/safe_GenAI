from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from tqdm.auto import tqdm

from hierasafe_flow.adapters.registry import create_adapter
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


class GenerationRunner:
    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        project_root = Path(get_path(config, "_meta.project_root", Path.cwd()))
        output_dir = Path(get_path(config, "logging.output_dir", "outputs"))
        if not output_dir.is_absolute():
            output_dir = project_root / output_dir
        self.output_dir = ensure_dir(output_dir)

        self.logger = setup_logger(
            output_dir=self.output_dir,
            level=str(get_path(config, "logging.level", "INFO")),
        )
        self.tracker = ExperimentTracker.create(self.output_dir, config)
        self.tensorboard = TensorBoardLogger(
            self.output_dir / "tensorboard",
            enabled=bool(get_path(config, "logging.tensorboard", True)),
        )

        seed = int(get_path(config, "project.seed", 1234))
        seed_everything(seed)
        self.device = resolve_device(str(get_path(config, "runtime.device", "auto")))
        self.dtype = resolve_dtype(str(get_path(config, "runtime.dtype", get_path(config, "model.torch_dtype", "auto"))))
        configure_cuda(bool(get_path(config, "runtime.allow_tf32", True)))
        write_json(self.output_dir / "system_info.json", collect_system_info())

        model_config = dict(config.get("model", {}))
        model_config["guidance_scale"] = get_path(config, "generation.guidance_scale", None)
        self.adapter = create_adapter(model_config, device=self.device, dtype=self.dtype)

        hierarchy_path = Path(str(get_path(config, "concepts.hierarchy_path")))
        if not hierarchy_path.is_absolute():
            hierarchy_path = project_root / hierarchy_path
        self.hierarchy = ConceptHierarchy.from_yaml_file(hierarchy_path)
        self.steering_mode = str(get_path(config, "steering.mode", "bottleneck"))
        self.steerer = HierarchicalVectorFieldBottleneck(
            hierarchy=self.hierarchy,
            config=BottleneckConfig.from_dict(config.get("steering", {})),
        )
        self.negative_guider = NegativeConceptVectorGuidance(
            NegativeGuidanceConfig.from_dict(config.get("negative_guidance", {}))
        )
        self._base_condition_cache: dict[str, Any] = {}

    def run(self, prompt: str | None = None) -> RunResult:
        self.logger.info("Loading adapter %s for %s", self.adapter.adapter_name, self.adapter.model_id)
        self.adapter.load()
        self.tracker.log_event("adapter_loaded", self.adapter.inspect())

        prompts = self._collect_prompts(prompt)
        records: list[GenerationRecord] = []
        for index, item in enumerate(prompts):
            sample_id = f"sample_{index:04d}"
            with torch.inference_mode():
                records.append(self._run_prompt(item, sample_id))

        self.tensorboard.close()
        self.tracker.log_event("run_finished", {"num_records": len(records)})
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
        self.logger.info("Generating %s", sample_id)
        generation = self.config.get("generation", {})
        task = str(generation.get("task", "text_to_image"))
        num_steps = int(generation.get("num_inference_steps", 28))
        batch_size = int(generation.get("num_outputs_per_prompt", 1))
        seed = int(get_path(self.config, "project.seed", 1234))
        generator = make_generator(seed, self.device)

        latent_kwargs = {
            key: value
            for key, value in generation.items()
            if key not in {"prompt", "prompt_file"}
        }
        latents, state = self.adapter.prepare_initial_latents(
            prompt=prompt,
            batch_size=batch_size,
            generator=generator,
            **latent_kwargs,
        )
        state.extra["num_steps"] = num_steps
        timesteps = self.adapter.set_timesteps(num_steps, latents=latents, state=state)
        trace: list[dict[str, Any]] = []

        for step_index, timestep in enumerate(tqdm(timesteps, desc=sample_id, leave=False)):
            prediction, step_trace = self._predict_step(
                latents=latents,
                timestep=timestep,
                state=state,
                prompt=prompt,
                step_index=step_index,
                num_steps=len(timesteps),
            )
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

        decode_outputs = bool(get_path(self.config, "output.decode", True))
        media = None
        if decode_outputs:
            media = self.adapter.decode_latents(latents, state)
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
        )
        report = self._build_sample_report(
            prompt=prompt,
            sample_id=sample_id,
            task=task,
            trace=trace,
            output_paths=output_paths,
            decode_outputs=decode_outputs,
        )
        output_paths["report"] = save_generation_report(report, self.output_dir, sample_id)
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
        if self.steering_mode == "none":
            condition = self._base_condition_cache.get(prompt)
            if condition is None:
                condition = self.adapter.prepare_prompt(prompt)
                self._base_condition_cache[prompt] = condition
            prediction = self.adapter.predict_vector_field(latents, timestep, condition, state)
            from hierasafe_flow.steering.bottleneck import BottleneckTrace
            from hierasafe_flow.utils.tensors import tensor_stats

            stats = tensor_stats(prediction)
            return prediction, BottleneckTrace(
                step_index=step_index,
                timestep=timestep.detach().flatten()[0].item() if hasattr(timestep, "detach") else timestep,
                enabled=False,
                concepts=[],
                base_stats=stats,
                steered_stats=stats,
            )
        raise ValueError(
            f"Unknown steering.mode '{self.steering_mode}'. Valid modes: bottleneck, negative_guidance, none."
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
    ) -> dict[str, Any]:
        generation_config = dict(self.config.get("generation", {}))
        generation_config["prompt"] = prompt
        return {
            "schema_version": 1,
            "prompt": prompt,
            "sample_id": sample_id,
            "task": task,
            "model": {
                "adapter": self.adapter.adapter_name,
                "model_id": self.adapter.model_id,
                "pipeline_class": self.adapter.pipeline_class_name,
            },
            "condition": {
                "steering_mode": self.steering_mode,
                "decode_outputs": decode_outputs,
                "is_native_negative_prompt": False,
            },
            "generation": generation_config,
            "steering": self.config.get("steering", {}),
            "concept_hierarchy": {
                "name": self.hierarchy.name,
                "neutral_concept": self.hierarchy.neutral_concept,
                "pairs": [asdict(pair) for pair in self.hierarchy.pairs],
            },
            "output_paths": dict(output_paths),
            "interpretability": {
                "note": (
                    "steering_delta_stats summarize the vector-field update applied by the "
                    "concept bottleneck. This method does not compute backpropagation gradients."
                ),
                "timesteps": trace,
                "concepts_per_step": _concepts_per_step(trace),
            },
        }


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
                "concepts": concepts,
            }
        )
    return rows
