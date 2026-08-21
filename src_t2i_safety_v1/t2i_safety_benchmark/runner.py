from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from hierasafe_flow.adapters.registry import create_adapter
from hierasafe_flow.utils.config import load_config
from hierasafe_flow.utils.device import configure_cuda, resolve_device, resolve_dtype
from hierasafe_flow.utils.seed import make_generator, seed_everything

from .contracts import (
    BENCHMARK_ID,
    MATRIX_SHA256,
    PROJECT_ROOT,
    PROMPTS_SHA256,
    BenchmarkContract,
    PromptRow,
    atomic_json,
    execution_identity,
    file_sha256,
    staged_attempt,
)
from .methods import (
    MidSteerRuntime,
    NativeNegativeMethod,
    UnsafeReferenceBank,
    historical_conceptsteer,
    related_work_transition,
    tensor_stats,
    validate_conceptsteer_trace,
    validate_related_work_runtime_grid,
)
from .pilot_context import MethodPilotCandidate


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _pil_images(value: Any) -> list[Image.Image]:
    if isinstance(value, Image.Image):
        return [value]
    if isinstance(value, (list, tuple)):
        result: list[Image.Image] = []
        for item in value:
            result.extend(_pil_images(item))
        return result
    return []


class ShardRunner:
    def __init__(
        self,
        *,
        model_id: str,
        variant: str,
        shard_index: int,
        num_shards: int,
        row_ids: set[str] | None,
        midsteer_strength: float | None,
        sgf_strength: float,
        safe_sigma: float | None,
        safe_scale: float | None,
        pilot_candidate: MethodPilotCandidate | None = None,
    ) -> None:
        self.contract = BenchmarkContract()
        self.model_id = model_id
        self.variant = variant
        self.model_spec = self.contract.model(model_id)
        self.cells = self.contract.cells(
            model_id=model_id,
            variant=variant,
            shard_index=shard_index,
            num_shards=num_shards,
            row_ids=row_ids,
        )
        self.prompts = self.contract.prompt_rows()
        self.pilot_candidate = pilot_candidate
        if pilot_candidate is not None:
            pilot_candidate.require_identity(
                model_id=model_id,
                category=pilot_candidate.category,
                method=variant,
            )
            selected_rows = {cell.row_id for cell in self.cells}
            if selected_rows != set(pilot_candidate.row_ids):
                raise RuntimeError("Runner selection differs from sealed pilot population")
        if midsteer_strength is not None:
            raise RuntimeError(
                "MidSteer strength overrides are forbidden; runtime strength "
                "must come from the sealed per-model/category calibration."
            )
        self.sgf_strength = float(sgf_strength)
        self.safe_sigma = safe_sigma
        self.safe_scale = safe_scale
        self.reference_cache: dict[str, UnsafeReferenceBank] = {}
        self.midsteer_cache: dict[str, MidSteerRuntime] = {}

        seed_everything(0)
        configure_cuda(True)
        self.device = resolve_device("auto")
        self.dtype = resolve_dtype("bfloat16")
        model_config = load_config(
            self.model_spec["config"],
            project_root=PROJECT_ROOT,
        )
        self.generation = dict(model_config["generation"])
        model_values = dict(model_config["model"])
        model_values["height"] = int(self.model_spec["height"])
        model_values["width"] = int(self.model_spec["width"])
        model_values["guidance_scale"] = float(self.model_spec["guidance_scale"])
        self.adapter = create_adapter(model_values, self.device, self.dtype)

    def run(self) -> None:
        if not self.cells:
            raise RuntimeError(
                f"No runnable cells selected for {self.model_id}/{self.variant}."
            )
        self.adapter.load()
        for cell in self.cells:
            row = self.prompts[cell.row_id]
            with staged_attempt(cell.output_dir) as staging:
                if staging is None:
                    continue
                self._run_cell(cell=cell, row=row, staging=staging)

    def _run_cell(
        self,
        *,
        cell: Any,
        row: PromptRow,
        staging: Path,
        seed_override: int | None = None,
    ) -> dict[str, Any]:
        started_at = _utc()
        start = time.perf_counter()
        self.adapter.begin_conditioning_provenance_scope()
        seed = int(row.seed if seed_override is None else seed_override)
        generator = make_generator(seed, self.device)
        generation = {
            **self.generation,
            "height": int(self.model_spec["height"]),
            "width": int(self.model_spec["width"]),
            "num_inference_steps": int(self.model_spec["steps"]),
            "num_outputs_per_prompt": 1,
        }
        latent_kwargs = {
            key: value
            for key, value in generation.items()
            if key not in {"prompt", "prompt_file"}
        }
        latents, state = self.adapter.prepare_initial_latents(
            prompt=row.prompt,
            batch_size=1,
            generator=generator,
            **latent_kwargs,
        )
        num_steps = int(self.model_spec["steps"])
        state.extra["num_steps"] = num_steps
        state.extra["base_seed"] = seed
        timesteps = self.adapter.set_timesteps(num_steps, latents=latents, state=state)
        if len(timesteps) != num_steps:
            raise RuntimeError("Adapter returned the wrong number of native timesteps.")

        base_condition = None
        native_method = None
        conceptsteerer = None
        conceptsteer_skip = False
        method_metadata: dict[str, Any] = {}
        if self.variant == "native_negative_prompt":
            category = self.contract.category(row.category)
            native_method = NativeNegativeMethod(
                adapter=self.adapter,
                model_id=self.model_id,
                prompt=row.prompt,
                negative_prompt=str(category["native_negative_prompt"]),
                true_cfg_scale=float(self.model_spec["guidance_scale"]),
            )
            method_metadata = dict(native_method.evidence)
        elif self.variant == "conceptsteer":
            from .hierarchy_store import load_sealed_concept_pair

            category = load_sealed_concept_pair(
                model_id=self.model_id,
                row=row,
            )
            if category["steering_action"] == "skip":
                conceptsteer_skip = True
                base_condition = self.adapter.prepare_prompt(row.prompt)
                method_metadata = {
                    "implementation": "sealed_hierarchy_no_action_passthrough",
                    "equation": "identity_baseline_vector_field",
                    "strength": 0.0,
                    "steering_action": "skip",
                    "skip_reason": "sealed_hierarchy_uncertainty_no_action",
                    "sealed_hierarchy": dict(category["provenance"]),
                    "neutral_concept": str(category["neutral_concept"]),
                }
            else:
                conceptsteerer = historical_conceptsteer(
                    category=row.category,
                    unsafe_concept=str(category["unsafe_concept"]),
                    safe_concept=str(category["safe_concept"]),
                    strength=float(self.model_spec["conceptsteer_lambda"]),
                )
                method_metadata = {
                    "implementation": "historical_finer_detailing_bottleneck",
                    "equation": "v_plus_lambda_relu_cos_margin_times_safe_minus_unsafe",
                    "strength": float(self.model_spec["conceptsteer_lambda"]),
                    "margin": 0.05,
                    "feature_dim": 1,
                    "prompt_composition": "concept_only",
                    "endpoint_contract": "historical_concept_only_category_contrast_v2",
                    "source_prompt_in_endpoints": False,
                    "shared_non_target_context": True,
                    "mask_enabled": False,
                    "normalize_directions": False,
                    "schedule": "constant_all_native_steps",
                    "hierarchy": {
                        "name": conceptsteerer.hierarchy.name,
                        "neutral_concept": conceptsteerer.hierarchy.neutral_concept,
                        "pairs": [
                            asdict(item) for item in conceptsteerer.hierarchy.pairs
                        ],
                    },
                }
                method_metadata["endpoint_contract"] = (
                    "sealed_row_specific_invariant_preserving_v2"
                )
                method_metadata["source_prompt_in_endpoints"] = True
                method_metadata["sealed_hierarchy"] = dict(
                    category["provenance"]
                )
                method_metadata["neutral_concept"] = str(
                    category["neutral_concept"]
                )
        else:
            base_condition = self.adapter.prepare_prompt(row.prompt)
            if self.variant == "midsteer":
                runtime = self.midsteer_cache.get(row.category)
                if runtime is None:
                    runtime = MidSteerRuntime(
                        adapter=self.adapter,
                        model_id=self.model_id,
                        category=row.category,
                        pilot_candidate=self.pilot_candidate,
                    )
                    self.midsteer_cache[row.category] = runtime
                runtime.validate_runtime_timesteps(timesteps)
                method_metadata = dict(runtime.metadata)
            elif self.variant in {"sgf", "safe_denoiser"}:
                bank = self.reference_cache.get(row.category)
                if bank is None:
                    self.reference_cache.clear()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    bank = UnsafeReferenceBank(
                        model_id=self.model_id,
                        category=row.category,
                        method=self.variant,
                        device=self.device,
                    )
                    self.reference_cache[row.category] = bank
                scheduler_evidence = validate_related_work_runtime_grid(
                    model_id=self.model_id,
                    method=self.variant,
                    bank=bank,
                    timesteps=timesteps,
                    pilot_candidate=self.pilot_candidate,
                )
                method_metadata = {
                    "unsafe_reference": dict(bank.metadata),
                    "scheduler_grid": scheduler_evidence,
                    "native_steps": num_steps,
                    "window": [0.0, 0.2]
                    if self.variant == "sgf"
                    else [0.0, 0.22],
                }

        trace: list[dict[str, Any]] = []
        with torch.inference_mode():
            for step_index, timestep in enumerate(timesteps):
                context = self.adapter.denoising_step_context(
                    step_index,
                    len(timesteps),
                    state,
                )
                state.extra["_active_denoising_step_context"] = context
                if self.variant == "baseline":
                    prediction = self.adapter.predict_vector_field(
                        latents, timestep, base_condition, state
                    )
                    detail = {
                        "method": "baseline",
                        "prediction": tensor_stats(prediction),
                    }
                elif self.variant == "native_negative_prompt":
                    if self.model_id == "sd35_large":
                        state.extra["_capture_sd35_native_cfg_trace"] = True
                    try:
                        prediction, detail = native_method.predict(
                            latents,
                            timestep,
                            state,
                        )
                    finally:
                        state.extra.pop("_capture_sd35_native_cfg_trace", None)
                    if self.model_id == "sd35_large":
                        sd35_cfg_trace = state.extra.pop(
                            "_sd35_native_cfg_trace",
                            None,
                        )
                        if not isinstance(sd35_cfg_trace, dict):
                            raise RuntimeError(
                                "SD3.5 native negative CFG omitted direct tensor evidence."
                            )
                        detail.update(sd35_cfg_trace)
                elif self.variant == "conceptsteer":
                    if conceptsteer_skip:
                        prediction = self.adapter.predict_vector_field(
                            latents, timestep, base_condition, state
                        )
                        detail = {
                            "method": "conceptsteer_no_action_passthrough",
                            "steering_applied": False,
                            "prediction": tensor_stats(prediction),
                        }
                    else:
                        prediction, concept_trace = conceptsteerer.steer_step(
                            adapter=self.adapter,
                            latents=latents,
                            timestep=timestep,
                            state=state,
                            prompt=row.prompt,
                            step_index=step_index,
                            num_steps=len(timesteps),
                        )
                        detail = asdict(concept_trace)
                elif self.variant == "midsteer":
                    runtime = self.midsteer_cache[row.category]
                    prediction, detail = runtime.predict(
                        latents=latents,
                        timestep=timestep,
                        condition=base_condition,
                        state=state,
                        step_index=step_index,
                    )
                elif self.variant in {"sgf", "safe_denoiser"}:
                    base_prediction = self.adapter.predict_vector_field(
                        latents, timestep, base_condition, state
                    )
                    latents_override, detail = related_work_transition(
                        method=self.variant,
                        adapter=self.adapter,
                        model_id=self.model_id,
                        latents=latents,
                        native_prediction=base_prediction,
                        timestep=timestep,
                        step_index=step_index,
                        num_steps=len(timesteps),
                        guidance_scale=float(self.model_spec["guidance_scale"]),
                        bank=self.reference_cache[row.category],
                        generator=generator,
                        sgf_strength=self.sgf_strength,
                        safe_sigma=self.safe_sigma,
                        safe_scale=self.safe_scale,
                        pilot_candidate=self.pilot_candidate,
                    )
                    prediction = base_prediction
                    detail["base_prediction"] = tensor_stats(base_prediction)
                    if latents_override is not None:
                        latents = latents_override
                else:
                    raise AssertionError(self.variant)
                detail["step_index"] = step_index
                detail["segment"] = context.to_dict()
                trace.append(detail)
                result = self.adapter.scheduler_step(
                    model_prediction=prediction,
                    timestep=timestep,
                    latents=latents,
                    state=state,
                    generator=generator,
                )
                latents, state = result.latents, result.state
            media = self.adapter.decode_latents(latents, state)

        validation: dict[str, Any]
        if self.variant == "conceptsteer":
            validation = validate_conceptsteer_trace(trace, row.category)
        elif self.variant == "native_negative_prompt":
            nonzero = sum(
                float(step["cfg_delta"]["norm"]) > 0.0
                for step in trace
                if "cfg_delta" in step
            )
            if nonzero != num_steps:
                raise RuntimeError("Native negative CFG lacked a nonzero delta at every step.")
            validation = {
                "status": "passed",
                "conditioning_difference_proved": True,
                "nonzero_cfg_delta_steps": nonzero,
            }
        elif self.variant == "midsteer":
            runtime = self.midsteer_cache[row.category]
            if len(trace) != runtime.num_inference_steps:
                raise RuntimeError(
                    "MidSteer runtime step count does not match its sealed "
                    "calibration."
                )
            active = [
                step_index
                for step_index, step in enumerate(trace)
                if step.get("active") is True
            ]
            if active != list(runtime.active_step_indices):
                raise RuntimeError(
                    "MidSteer observed active steps do not match its sealed "
                    "calibration."
                )
            validation = {
                "status": "passed",
                "active_steps": len(active),
                "active_step_indices": active,
                "calibration_sha256": runtime.metadata["calibration_sha256"],
            }
        elif self.variant == "sgf":
            eligible = sum(bool(step.get("eligible")) for step in trace)
            active = sum(bool(step.get("active")) for step in trace)
            if eligible != active or active <= 0:
                raise RuntimeError("SGF did not apply on every author-window step.")
            validation = {
                "status": "passed",
                "eligible_steps": eligible,
                "active_steps": active,
            }
        elif self.variant == "safe_denoiser":
            eligible = sum(bool(step.get("eligible")) for step in trace)
            active = sum(bool(step.get("active")) for step in trace)
            expected = int(
                __import__("math").ceil(0.22 * num_steps)
            )
            if eligible != expected:
                raise RuntimeError(
                    "Safe Denoiser did not evaluate every author-window step."
                )
            validation = {
                "status": "passed",
                "eligible_steps": eligible,
                "density_admitted_steps": active,
            }
        else:
            validation = {"status": "passed", "intervention": "none"}

        images = _pil_images(media)
        if len(images) != 1:
            raise RuntimeError(
                f"Expected exactly one decoded PIL image, observed {len(images)}."
            )
        image_path = staging / "image.png"
        images[0].save(image_path, format="PNG")
        image_hash = file_sha256(image_path)
        atomic_json(staging / "trace.json", trace)
        metadata = {
            "schema_version": 1,
            "benchmark_id": BENCHMARK_ID,
            "prompt_manifest_sha256": PROMPTS_SHA256,
            "matrix_manifest_sha256": MATRIX_SHA256,
            "cell": asdict(cell),
            "prompt_row": asdict(row),
            "model": {
                "id": self.model_id,
                "adapter": self.adapter.adapter_name,
                "model_id": self.adapter.model_id,
                "revision": self.adapter.config.get("revision"),
                "pipeline_class": self.adapter.pipeline_class_name,
            },
            "variant": self.variant,
            "method": method_metadata,
            "method_validation": validation,
            "generation": generation,
            "seed": seed,
            "final_latents": tensor_stats(latents),
            "image": {
                "path": "image.png",
                "sha256": image_hash,
                "width": images[0].width,
                "height": images[0].height,
            },
            "trace_path": "trace.json",
            "trace_steps": len(trace),
            "conditioning_provenance": self.adapter.conditioning_provenance(),
            "execution": execution_identity(),
            "started_at": started_at,
            "completed_at": _utc(),
            "wall_seconds": time.perf_counter() - start,
        }
        if self.pilot_candidate is not None:
            metadata["calibration_pilot"] = self.pilot_candidate.provenance()
        atomic_json(staging / "metadata.json", metadata)
        metadata_hash = file_sha256(staging / "metadata.json")
        atomic_json(
            staging / "_SUCCESS.json",
            {
                "schema_version": 1,
                "cell_id": cell.cell_id,
                "cell_sha256": cell.cell_sha256,
                "image_sha256": image_hash,
                "metadata_sha256": metadata_hash,
                "completed_at": _utc(),
            },
        )
        return {"image_sha256": image_hash, "metadata_sha256": metadata_hash}


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description="Run one immutable T2ISafety shard.")
    value.add_argument("--model", required=True)
    value.add_argument("--variant", required=True)
    value.add_argument("--shard-index", type=int, default=0)
    value.add_argument("--num-shards", type=int, default=1)
    value.add_argument("--row-id", action="append", default=[])
    value.add_argument("--midsteer-strength", type=float)
    value.add_argument("--sgf-strength", type=float, default=0.03)
    value.add_argument("--safe-sigma", type=float)
    value.add_argument("--safe-scale", type=float)
    return value


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    runner = ShardRunner(
        model_id=args.model,
        variant=args.variant,
        shard_index=args.shard_index,
        num_shards=args.num_shards,
        row_ids=set(args.row_id) if args.row_id else None,
        midsteer_strength=args.midsteer_strength,
        sgf_strength=args.sgf_strength,
        safe_sigma=args.safe_sigma,
        safe_scale=args.safe_scale,
        pilot_candidate=None,
    )
    runner.run()


if __name__ == "__main__":
    main()
