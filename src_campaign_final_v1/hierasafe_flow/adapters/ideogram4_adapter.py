from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

import torch

from hierasafe_flow.adapters.base import (
    AdapterCapabilities,
    AdapterState,
    FrozenGeneratorAdapter,
    PromptCondition,
    SchedulerStepResult,
)


IDEOGRAM4_MAX_SEQUENCE_LENGTH = 2048
# Pinned Diffusers Ideogram4 uses this role value in both `_prepare_ids` and
# the transformer mask. `load` verifies it against the installed implementation.
IDEOGRAM4_LLM_TOKEN_INDICATOR = 3


class Ideogram4Adapter(FrozenGeneratorAdapter):
    """Frozen-vector-field adapter for the Diffusers Ideogram 4 pipeline.

    Ideogram 4 uses asymmetric classifier-free guidance: a text-conditioned
    transformer and a separate unconditional image-only transformer. The
    physical velocity is ``g * v_pos + (1 - g) * v_uncond``; Diffusers then
    passes its negative to ``FlowMatchEulerDiscreteScheduler.step``. Keeping
    that sign convention exact is essential—using the legacy standalone
    Ideogram API's update rule with the Diffusers checkpoint produces no valid
    denoising trajectory.

    The unconditional branch is prompt-independent. ConceptSteer requests many
    vector fields at the same latent/timestep, so this adapter caches that
    branch once per denoising step. This is algebraically exact and avoids
    repeating the large unconditional transformer for every concept prompt.

    Ideogram 4 was trained on a strict JSON caption schema.  Passing the study's
    plain natural-language prompts directly can trigger the checkpoint's
    caption verifier/safety path and materially reduce quality.  We therefore
    wrap *the identical input text* in a deterministic schema before encoding.
    This is model-native serialization, not prompt enhancement: it adds no new
    source or target concept and never calls an external rewriting service.
    """

    adapter_name = "ideogram4"
    task_type = "text_to_image"
    pipeline_class_name = "Ideogram4Pipeline"
    latent_feature_dim = -1  # packed [B, image_tokens, D]

    def __init__(
        self,
        model_id: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id=model_id, device=device, dtype=dtype, config=config)
        self.pipeline: Any | None = None
        self.timesteps: list[torch.Tensor] = []
        self._active_grid_h: int | None = None
        self._active_grid_w: int | None = None
        self._active_max_sequence_length: int | None = None
        self._active_height: int | None = None
        self._active_width: int | None = None
        self._native_caption_cache: dict[str, str] = {}
        self._conditioning_records: dict[str, dict[str, Any]] = {}

    def capabilities(self) -> AdapterCapabilities:
        return AdapterCapabilities(
            task_type=self.task_type,
            pipeline_class_name=self.pipeline_class_name,
            exposes_latents=True,
            exposes_timesteps=True,
            exposes_scheduler_step=True,
            exposes_vector_field=True,
        )

    def load(self) -> None:
        try:
            from diffusers import Ideogram4Pipeline, Ideogram4PromptEnhancerHead
            from diffusers.pipelines.ideogram4.pipeline_ideogram4 import LLM_TOKEN_INDICATOR
        except (ImportError, ModuleNotFoundError) as exc:
            raise NotImplementedError(
                "Ideogram 4 requires a Diffusers build exposing Ideogram4Pipeline. "
                "Install current Diffusers main before running this adapter."
            ) from exc
        if LLM_TOKEN_INDICATOR != IDEOGRAM4_LLM_TOKEN_INDICATOR:
            raise RuntimeError(
                "Ideogram4 LLM token indicator drifted from the verified Diffusers contract: "
                f"expected {IDEOGRAM4_LLM_TOKEN_INDICATOR}, got {LLM_TOKEN_INDICATOR}."
            )

        load_kwargs = dict(self.config.get("load_kwargs", {}))
        if self.config.get("revision") is not None:
            load_kwargs["revision"] = self.config["revision"]
        if self.config.get("variant") is not None:
            load_kwargs["variant"] = self.config["variant"]
        load_kwargs.setdefault("torch_dtype", self.dtype)
        load_kwargs.setdefault("low_cpu_mem_usage", True)
        load_kwargs.setdefault("local_files_only", bool(self.config.get("local_files_only", False)))
        if os.environ.get("HF_TOKEN") and "token" not in load_kwargs:
            load_kwargs["token"] = os.environ["HF_TOKEN"]
        upsampling = dict(self.config.get("prompt_upsampling", {}) or {})
        if bool(upsampling.get("enabled", True)):
            if bool(upsampling.get("require_schema_constraint", True)):
                try:
                    import outlines  # noqa: F401
                except ModuleNotFoundError as exc:
                    raise RuntimeError(
                        "Ideogram4 production prompt upsampling requires `outlines` so the local "
                        "enhancer is constrained to the official JSON schema."
                    ) from exc
            head_model_id = str(
                upsampling.get(
                    "head_model_id",
                    "diffusers/qwen3-vl-8b-instruct-lm-head",
                )
            )
            head_revision = upsampling.get("head_revision")
            if not isinstance(head_revision, str) or re.fullmatch(
                r"[0-9a-f]{40}", head_revision
            ) is None:
                raise ValueError(
                    "Ideogram4 production prompt upsampling requires a full 40-character "
                    "prompt_upsampling.head_revision commit."
                )
            load_kwargs["prompt_enhancer_head"] = Ideogram4PromptEnhancerHead.from_pretrained(
                head_model_id,
                revision=head_revision,
                torch_dtype=self.dtype,
                local_files_only=bool(self.config.get("local_files_only", False)),
                token=os.environ.get("HF_TOKEN"),
            )
        self.pipeline = Ideogram4Pipeline.from_pretrained(self.model_id, **load_kwargs)

        cpu_offload = self.config.get("cpu_offload", False)
        if cpu_offload == "sequential" and hasattr(self.pipeline, "enable_sequential_cpu_offload"):
            self.pipeline.enable_sequential_cpu_offload(device=self.device)
        elif bool(cpu_offload) and hasattr(self.pipeline, "enable_model_cpu_offload"):
            self.pipeline.enable_model_cpu_offload(device=self.device)
        elif hasattr(self.pipeline, "to"):
            self.pipeline.to(self.device)

        self._freeze_pipeline()
        self.loaded = True

    def prepare_initial_latents(
        self,
        prompt: str,
        batch_size: int,
        generator: torch.Generator | None,
        **generation_kwargs: Any,
    ) -> tuple[torch.Tensor, AdapterState]:
        self._require_loaded()
        del prompt
        assert self.pipeline is not None
        height = int(generation_kwargs.get("height", 1024))
        width = int(generation_kwargs.get("width", 1024))
        divisor = int(self.pipeline.vae_scale_factor) * int(self.pipeline.patch_size)
        if height % divisor != 0 or width % divisor != 0:
            raise ValueError(f"Ideogram4 height/width must be divisible by {divisor}.")

        grid_h = height // divisor
        grid_w = width // divisor
        num_image_tokens = grid_h * grid_w
        latent_dim = int(self.pipeline.transformer.config.in_channels)
        max_sequence_length = int(
            generation_kwargs.get(
                "max_sequence_length",
                self.config.get("max_sequence_length", IDEOGRAM4_MAX_SEQUENCE_LENGTH),
            )
        )
        if max_sequence_length != IDEOGRAM4_MAX_SEQUENCE_LENGTH:
            raise ValueError(
                "Ideogram4 conditioning must retain the official "
                f"max_sequence_length={IDEOGRAM4_MAX_SEQUENCE_LENGTH}; got {max_sequence_length}."
            )
        latents = self.pipeline.prepare_latents(
            batch_size=batch_size,
            num_image_tokens=num_image_tokens,
            latent_dim=latent_dim,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
            latents=generation_kwargs.get("latents"),
        )
        self._active_grid_h = grid_h
        self._active_grid_w = grid_w
        self._active_max_sequence_length = max_sequence_length
        self._active_height = height
        self._active_width = width
        return latents, AdapterState(
            extra={
                "height": height,
                "width": width,
                "grid_h": grid_h,
                "grid_w": grid_w,
                "num_image_tokens": num_image_tokens,
                "latent_dim": latent_dim,
                "max_sequence_length": max_sequence_length,
                "attention_kwargs": generation_kwargs.get("attention_kwargs"),
            }
        )

    def prepare_prompt(self, prompt: str) -> PromptCondition:
        return self.prepare_prompts([prompt])[0]

    def prepare_prompts(self, prompts: list[str]) -> list[PromptCondition]:
        self._require_loaded()
        if not prompts:
            return []
        if (
            self._active_grid_h is None
            or self._active_grid_w is None
            or self._active_max_sequence_length is None
        ):
            raise RuntimeError("Ideogram4 initial latents must be prepared before prompt encoding.")
        assert self.pipeline is not None

        # Chunking bounds peak text-encoder activations without changing any
        # prompt condition. Each returned condition remains a batch of one.
        chunk_size = max(1, int(self.config.get("prompt_batch_size", 2)))
        conditions: list[PromptCondition] = []
        for start in range(0, len(prompts), chunk_size):
            chunk = prompts[start : start + chunk_size]
            encoded_chunk = [self._model_native_caption_for_prompt(prompt) for prompt in chunk]
            llm_features, position_ids, segment_ids, indicator = self.pipeline.encode_prompt(
                prompt=encoded_chunk,
                grid_h=self._active_grid_h,
                grid_w=self._active_grid_w,
                max_sequence_length=self._active_max_sequence_length,
                device=self.device,
            )
            post_encode_counts = [
                self._validate_post_encode_indicator(
                    raw_prompt=prompt.strip(),
                    indicator=indicator[index : index + 1],
                )
                for index, prompt in enumerate(chunk)
            ]
            for index, prompt in enumerate(chunk):
                # Diffusers returns a very large zero suffix for all image
                # tokens. Retaining that identical suffix in every cached
                # concept condition would cost roughly a gigabyte per prompt
                # at 1024px. Store only the actual text slots and reconstruct
                # one shared zero suffix in predict_vector_field instead.
                prompt_features = (
                    llm_features[index : index + 1, : self._active_max_sequence_length]
                    .to(self.pipeline.transformer.dtype)
                    .clone()
                )
                prompt_position_ids = position_ids[index : index + 1].clone()
                prompt_segment_ids = segment_ids[index : index + 1].clone()
                prompt_indicator = indicator[index : index + 1].clone()
                conditions.append(
                    PromptCondition(
                        prompt=prompt,
                        data={
                            "model_native_prompt": encoded_chunk[index],
                            "model_native_prompt_sha256": hashlib.sha256(
                                encoded_chunk[index].encode("utf-8")
                            ).hexdigest(),
                            "chat_template_token_count": post_encode_counts[index],
                            "chat_template_token_ids_sha256": self._conditioning_records[
                                prompt.strip()
                            ]["token_ids_sha256"],
                            "llm_features": prompt_features,
                            "position_ids": prompt_position_ids,
                            "segment_ids": prompt_segment_ids,
                            "indicator": prompt_indicator,
                            "max_text_tokens": self._active_max_sequence_length,
                            "neg_position_ids": prompt_position_ids[
                                :, self._active_max_sequence_length :
                            ],
                            "neg_segment_ids": prompt_segment_ids[
                                :, self._active_max_sequence_length :
                            ],
                            "neg_indicator": prompt_indicator[
                                :, self._active_max_sequence_length :
                            ],
                        },
                    )
                )
        return conditions

    @staticmethod
    def _model_native_caption(prompt: str) -> str:
        """Return a minimal valid caption only for explicit fallback/testing.

        Production jobs enable the checkpoint's official local prompt enhancer
        and do not use this generic fallback. Already structured expert JSON is
        preserved byte-for-byte after validation.
        """

        stripped = prompt.strip()
        if not stripped:
            raise ValueError("Ideogram4 prompt must be non-empty.")
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            _validate_native_caption(stripped, original_prompt=None)
            return stripped

        caption = {
            "high_level_description": stripped,
            "compositional_deconstruction": {
                "background": "The setting, ground, sky, atmosphere, and lighting stated by the user.",
                "elements": [
                    {
                        "type": "obj",
                        "desc": stripped,
                    }
                ]
                + [
                    {
                        "type": "text",
                        "text": literal,
                        "desc": "Visible text exactly as quoted by the user.",
                    }
                    for literal in _quoted_literals(stripped)
                ],
            },
        }
        return json.dumps(caption, separators=(",", ":"), ensure_ascii=False)

    def _model_native_caption_for_prompt(self, prompt: str) -> str:
        """Create and cache the exact official model-native caption.

        The released Ideogram 4 checkpoint is trained on a structured JSON
        caption. Its official local Qwen3-VL enhancer is therefore part of the
        model interface, not an external prompt service. Sampling is seeded by
        the raw prompt, target aspect ratio, and retry index; schema-constrained
        candidates are rejected unless they retain quoted text and the study's
        controlled concept terms.
        """

        stripped = prompt.strip()
        if not stripped:
            raise ValueError("Ideogram4 prompt must be non-empty.")
        cached = self._native_caption_cache.get(stripped)
        if cached is not None:
            return cached
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            _validate_native_caption(stripped, original_prompt=None)
            token_preflight = self._preflight_model_native_caption(stripped)
            self._native_caption_cache[stripped] = stripped
            self._record_conditioning(
                stripped,
                stripped,
                method="expert_json",
                attempt=0,
                token_preflight=token_preflight,
            )
            return stripped

        self._require_loaded()
        assert self.pipeline is not None
        if self._active_height is None or self._active_width is None:
            raise RuntimeError(
                "Ideogram4 initial latents must be prepared before prompt upsampling."
            )
        upsampling = dict(self.config.get("prompt_upsampling", {}) or {})
        if not bool(upsampling.get("enabled", True)):
            fallback = self._model_native_caption(stripped)
            _validate_native_caption(fallback, original_prompt=stripped)
            token_preflight = self._preflight_model_native_caption(fallback)
            self._native_caption_cache[stripped] = fallback
            self._record_conditioning(
                stripped,
                fallback,
                method="generic_fallback",
                attempt=0,
                token_preflight=token_preflight,
            )
            return fallback
        if getattr(self.pipeline, "prompt_enhancer_head", None) is None:
            raise RuntimeError(
                "Ideogram4 prompt upsampling is enabled but no enhancer head was loaded."
            )

        base_seed = _stable_caption_seed(stripped, self._active_height, self._active_width)
        max_attempts = int(upsampling.get("max_attempts", 3))
        if max_attempts <= 0:
            raise ValueError("Ideogram4 prompt_upsampling.max_attempts must be positive.")
        errors: list[str] = []
        for attempt in range(max_attempts):
            generator = torch.Generator(device="cpu")
            generator.manual_seed((base_seed + attempt) % (2**63 - 1))
            candidates = self.pipeline.upsample_prompt(
                stripped,
                height=self._active_height,
                width=self._active_width,
                temperature=float(upsampling.get("temperature", 1.0)),
                max_new_tokens=int(upsampling.get("max_new_tokens", 1024)),
                generator=generator,
                device=self.device,
            )
            if len(candidates) != 1:
                errors.append(f"attempt {attempt + 1}: returned {len(candidates)} captions")
                continue
            candidate = candidates[0].strip()
            try:
                _validate_native_caption(candidate, original_prompt=None)
                candidate_token_preflight = self._preflight_model_native_caption(candidate)
                repaired, fidelity_repair = _repair_schema_valid_enhancer_caption(
                    candidate,
                    stripped,
                )
                token_preflight = (
                    candidate_token_preflight
                    if repaired == candidate
                    else self._preflight_model_native_caption(repaired)
                )
            except ValueError as exc:
                errors.append(f"attempt {attempt + 1}: {exc}")
                continue
            self._native_caption_cache[stripped] = repaired
            self._record_conditioning(
                stripped,
                repaired,
                method=(
                    "official_local_prompt_enhancer_with_verbatim_fidelity_repair"
                    if fidelity_repair["fidelity_repair_applied"]
                    else "official_local_prompt_enhancer"
                ),
                attempt=attempt + 1,
                token_preflight=token_preflight,
                enhancer_candidate=candidate,
                enhancer_candidate_token_preflight=candidate_token_preflight,
                fidelity_repair=fidelity_repair,
            )
            return repaired
        raise RuntimeError(
            "Ideogram4 official local prompt enhancer did not produce a faithful schema-valid caption "
            f"after {max_attempts} deterministic attempts: {'; '.join(errors)}"
        )

    def _record_conditioning(
        self,
        raw_prompt: str,
        model_native_prompt: str,
        *,
        method: str,
        attempt: int,
        token_preflight: dict[str, Any],
        enhancer_candidate: str | None = None,
        enhancer_candidate_token_preflight: dict[str, Any] | None = None,
        fidelity_repair: dict[str, Any] | None = None,
    ) -> None:
        upsampling = dict(self.config.get("prompt_upsampling", {}) or {})
        candidate = enhancer_candidate if enhancer_candidate is not None else model_native_prompt
        candidate_preflight = enhancer_candidate_token_preflight or token_preflight
        repair = fidelity_repair or {
            "fidelity_repair_applied": False,
            "inserted_elements": [],
            "missing_quoted_literals_before": [],
            "missing_controlled_terms_before": [],
            "missing_quoted_literals_after": [],
            "missing_controlled_terms_after": [],
            "all_inserted_semantic_payloads_are_verbatim_raw_spans": True,
            "target_injection_guard_status": "passed",
            "candidate_caption_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
            "repaired_caption_sha256": hashlib.sha256(
                model_native_prompt.encode("utf-8")
            ).hexdigest(),
        }
        self._conditioning_records[raw_prompt] = {
            "raw_prompt": raw_prompt,
            "raw_prompt_sha256": hashlib.sha256(raw_prompt.encode("utf-8")).hexdigest(),
            "model_native_prompt": model_native_prompt,
            "model_native_prompt_sha256": hashlib.sha256(
                model_native_prompt.encode("utf-8")
            ).hexdigest(),
            "method": method,
            "deterministic_attempt": attempt,
            "enhancer_candidate_sha256": hashlib.sha256(candidate.encode("utf-8")).hexdigest(),
            "enhancer_candidate_token_count": candidate_preflight["token_count"],
            **repair,
            "head_model_id": upsampling.get("head_model_id"),
            "head_revision": upsampling.get("head_revision"),
            "model_revision": self.config.get("revision"),
            "height": self._active_height,
            "width": self._active_width,
            **token_preflight,
            "post_encode_llm_indicator_value": None,
            "post_encode_llm_indicator_count": None,
            "post_encode_llm_indicator_matches_token_count": None,
        }

    def _preflight_model_native_caption(self, model_native_prompt: str) -> dict[str, Any]:
        """Mirror Diffusers' exact chat tokenization and reject any lossy path."""

        self._require_loaded()
        assert self.pipeline is not None
        if self._active_max_sequence_length is None:
            raise RuntimeError("Ideogram4 initial latents must be prepared before token preflight.")
        tokenizer = getattr(self.pipeline, "tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("Ideogram4 pipeline does not expose its official tokenizer.")

        messages = [
            {
                "role": "user",
                "content": [{"type": "text", "text": model_native_prompt}],
            }
        ]
        chat_text = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        if not isinstance(chat_text, str) or not chat_text:
            raise RuntimeError("Ideogram4 tokenizer returned a malformed chat-template string.")
        tokenized = tokenizer(
            chat_text,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=False,
        )
        if not isinstance(tokenized, dict) and not hasattr(tokenized, "__getitem__"):
            raise RuntimeError("Ideogram4 tokenizer returned an unsupported encoding object.")
        try:
            token_ids = _single_token_id_sequence(tokenized["input_ids"])
        except (KeyError, TypeError) as exc:
            raise RuntimeError("Ideogram4 tokenizer did not return input_ids.") from exc
        token_count = len(token_ids)
        if token_count > self._active_max_sequence_length:
            raise ValueError(
                f"caption has {token_count} chat-template tokens, exceeding the no-truncation "
                f"limit {self._active_max_sequence_length}"
            )

        decoded = tokenizer.decode(
            list(token_ids),
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if decoded != chat_text:
            raise ValueError(
                "caption failed exact chat-template tokenization roundtrip; refusing lossy conditioning"
            )
        chat_text_sha256 = hashlib.sha256(chat_text.encode("utf-8")).hexdigest()
        token_ids_sha256 = hashlib.sha256(
            json.dumps(list(token_ids), separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return {
            "chat_template_text_sha256": chat_text_sha256,
            "roundtrip_decoded_text_sha256": hashlib.sha256(decoded.encode("utf-8")).hexdigest(),
            "token_count": token_count,
            "token_ids_sha256": token_ids_sha256,
            "max_sequence_length": self._active_max_sequence_length,
            "chat_template_add_generation_prompt": True,
            "tokenizer_add_special_tokens": False,
            "tokenizer_truncation": False,
            "truncated": False,
            "exact_roundtrip": True,
        }

    def _validate_post_encode_indicator(
        self,
        *,
        raw_prompt: str,
        indicator: torch.Tensor,
    ) -> int:
        """Bind preflight token count to the packed sequence returned by Diffusers."""

        if raw_prompt not in self._conditioning_records:
            raise RuntimeError("Ideogram4 conditioning record is missing after prompt encoding.")
        if not isinstance(indicator, torch.Tensor):
            raise RuntimeError("Ideogram4 encode_prompt returned a non-tensor indicator.")
        count = int((indicator == IDEOGRAM4_LLM_TOKEN_INDICATOR).sum().detach().cpu().item())
        record = self._conditioning_records[raw_prompt]
        expected = int(record["token_count"])
        if count != expected:
            raise RuntimeError(
                "Ideogram4 post-encode LLM indicator count does not match token preflight: "
                f"expected {expected}, got {count}."
            )
        record["post_encode_llm_indicator_value"] = IDEOGRAM4_LLM_TOKEN_INDICATOR
        record["post_encode_llm_indicator_count"] = count
        record["post_encode_llm_indicator_matches_token_count"] = True
        return count

    def conditioning_provenance(self) -> dict[str, Any]:
        return {
            "schema_version": 3,
            "adapter": self.adapter_name,
            "model_id": self.model_id,
            "model_revision": self.config.get("revision"),
            "records": [
                self._conditioning_records[prompt] for prompt in sorted(self._conditioning_records)
            ],
        }

    def set_timesteps(
        self,
        num_inference_steps: int,
        latents: torch.Tensor | None = None,
        state: AdapterState | None = None,
    ) -> list[torch.Tensor]:
        self._require_loaded()
        del latents
        if state is None:
            raise RuntimeError("Ideogram4 set_timesteps requires adapter state.")
        assert self.pipeline is not None
        from diffusers.pipelines.ideogram4.pipeline_ideogram4 import (
            _logit_normal_sigmas,
            _resolution_aware_mu,
        )

        schedule_mu = _resolution_aware_mu(
            height=int(state.extra["height"]),
            width=int(state.extra["width"]),
            base_mu=float(self.config.get("mu", 0.0)),
        )
        sigmas = _logit_normal_sigmas(
            num_inference_steps,
            schedule_mu,
            std=float(self.config.get("std", 1.5)),
            device=torch.device(self.device),
        )
        self.pipeline.scheduler.set_timesteps(sigmas=sigmas.tolist(), device=self.device)
        scheduler_timesteps = list(self.pipeline.scheduler.timesteps)

        explicit_schedule = self.config.get("guidance_schedule")
        preset = str(self.config.get("guidance_schedule_preset", "ideogram4_quality_48"))
        if explicit_schedule is not None:
            guidance = [float(value) for value in explicit_schedule]
        elif preset == "ideogram4_quality_48" and num_inference_steps == 48:
            guidance = [7.0] * 45 + [3.0] * 3
        else:
            guidance = [float(self.config.get("guidance_scale", 7.0))] * num_inference_steps
        if len(guidance) != num_inference_steps:
            raise ValueError(
                f"Ideogram4 guidance schedule has {len(guidance)} values for {num_inference_steps} steps."
            )

        self.timesteps = [
            torch.stack(
                [
                    timestep.to(device=self.device, dtype=torch.float32),
                    torch.tensor(guidance[index], device=self.device, dtype=torch.float32),
                ]
            )
            for index, timestep in enumerate(scheduler_timesteps)
        ]
        return self.timesteps

    def predict_vector_field(
        self,
        latents: torch.Tensor,
        timestep: Any,
        condition: PromptCondition,
        state: AdapterState,
    ) -> torch.Tensor:
        self._require_loaded()
        assert self.pipeline is not None
        scheduler_timestep, guidance = self._timestep_values(timestep)
        batch_size = latents.shape[0]
        max_text_tokens = int(condition.data["max_text_tokens"])
        t_model = 1.0 - (
            scheduler_timestep.float() / self.pipeline.scheduler.config.num_train_timesteps
        )
        t_model = t_model.expand(batch_size).to(self.pipeline.transformer.dtype)
        text_padding = torch.zeros(
            batch_size,
            max_text_tokens,
            latents.shape[-1],
            dtype=torch.float32,
            device=latents.device,
        )
        pos_z = torch.cat([text_padding, latents], dim=1).to(self.pipeline.transformer.dtype)
        feature_dim = int(condition.data["llm_features"].shape[-1])
        image_feature_key = (
            "ideogram4_shared_image_feature_padding",
            batch_size,
            int(state.extra["num_image_tokens"]),
            feature_dim,
            str(self.pipeline.transformer.dtype),
        )
        if state.extra.get("ideogram4_shared_image_feature_padding_key") == image_feature_key:
            image_feature_padding = state.extra["ideogram4_shared_image_feature_padding"]
        else:
            image_feature_padding = torch.zeros(
                batch_size,
                int(state.extra["num_image_tokens"]),
                feature_dim,
                dtype=self.pipeline.transformer.dtype,
                device=latents.device,
            )
            state.extra["ideogram4_shared_image_feature_padding_key"] = image_feature_key
            state.extra["ideogram4_shared_image_feature_padding"] = image_feature_padding
        encoder_hidden_states = torch.cat(
            [
                condition.data["llm_features"].to(self.pipeline.transformer.dtype),
                image_feature_padding,
            ],
            dim=1,
        )
        pos_out = self.pipeline.transformer(
            hidden_states=pos_z,
            timestep=t_model,
            encoder_hidden_states=encoder_hidden_states,
            position_ids=condition.data["position_ids"],
            segment_ids=condition.data["segment_ids"],
            indicator=condition.data["indicator"],
            attention_kwargs=state.extra.get("attention_kwargs"),
            return_dict=False,
        )[0]
        pos_v = pos_out[:, max_text_tokens:].to(torch.float32)

        cache_key = (
            float(scheduler_timestep.detach().item()),
            int(latents.data_ptr()),
        )
        if state.extra.get("ideogram4_unconditional_cache_key") == cache_key:
            neg_v = state.extra["ideogram4_unconditional_prediction"]
        else:
            unconditional_features = image_feature_padding.to(
                self.pipeline.unconditional_transformer.dtype
            )
            neg_v = self.pipeline.unconditional_transformer(
                hidden_states=latents.to(self.pipeline.unconditional_transformer.dtype),
                timestep=t_model.to(self.pipeline.unconditional_transformer.dtype),
                encoder_hidden_states=unconditional_features,
                position_ids=condition.data["neg_position_ids"],
                segment_ids=condition.data["neg_segment_ids"],
                indicator=condition.data["neg_indicator"],
                attention_kwargs=state.extra.get("attention_kwargs"),
                return_dict=False,
            )[0].to(torch.float32)
            state.extra["ideogram4_unconditional_cache_key"] = cache_key
            state.extra["ideogram4_unconditional_prediction"] = neg_v
        return guidance * pos_v + (1.0 - guidance) * neg_v

    def scheduler_step(
        self,
        model_prediction: torch.Tensor,
        timestep: Any,
        latents: torch.Tensor,
        state: AdapterState,
        generator: torch.Generator | None = None,
    ) -> SchedulerStepResult:
        self._require_loaded()
        del generator
        assert self.pipeline is not None
        scheduler_timestep, _ = self._timestep_values(timestep)
        output = self.pipeline.scheduler.step(
            -model_prediction.float(),
            scheduler_timestep,
            latents,
            return_dict=False,
        )[0]
        state.extra.pop("ideogram4_unconditional_cache_key", None)
        state.extra.pop("ideogram4_unconditional_prediction", None)
        return SchedulerStepResult(latents=output, state=state)

    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> Any:
        self._require_loaded()
        assert self.pipeline is not None
        z = latents
        bn_mean = self.pipeline.vae.bn.running_mean.view(1, 1, -1).to(
            device=z.device, dtype=z.dtype
        )
        bn_std = torch.sqrt(
            self.pipeline.vae.bn.running_var + self.pipeline.vae.config.batch_norm_eps
        ).view(1, 1, -1)
        z = z * bn_std.to(device=z.device, dtype=z.dtype) + bn_mean

        patch = int(self.pipeline.patch_size)
        ae_channels = z.shape[-1] // (patch * patch)
        grid_h = int(state.extra["grid_h"])
        grid_w = int(state.extra["grid_w"])
        z = z.view(z.shape[0], grid_h, grid_w, patch, patch, ae_channels)
        z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
        z = z.view(z.shape[0], ae_channels, grid_h * patch, grid_w * patch)
        decoded = self.pipeline.vae.decode(z.to(self.pipeline.vae.dtype), return_dict=False)[0]
        return self.pipeline.image_processor.postprocess(decoded.float(), output_type="pil")

    def _freeze_pipeline(self) -> None:
        assert self.pipeline is not None
        for component_name in (
            "transformer",
            "unconditional_transformer",
            "text_encoder",
            "vae",
            "prompt_enhancer_head",
        ):
            component = getattr(self.pipeline, component_name, None)
            if component is not None and hasattr(component, "parameters"):
                component.eval()
                for parameter in component.parameters():
                    parameter.requires_grad_(False)

    def _timestep_values(self, timestep: Any) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(timestep, torch.Tensor):
            raise TypeError("Ideogram4 timesteps must be tensors returned by set_timesteps.")
        values = timestep.to(device=self.device, dtype=torch.float32).flatten()
        if values.numel() != 2:
            raise ValueError(
                "Ideogram4 timestep tensors must contain [scheduler_timestep, guidance], "
                f"got shape {tuple(timestep.shape)}."
            )
        return values[0], values[1]


_CONTROLLED_TERM_REQUIREMENTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("young girl", ("girl", "female child")),
    ("old man", ("old man", "elderly man", "older man")),
    ("sad", ("sad", "downcast", "frown")),
    ("angry", ("angry", "scowl", "furious")),
    ("happy", ("happy", "smil", "cheerful")),
    ("sitting", ("sitting", "seated")),
    ("seated", ("sitting", "seated")),
    ("walking", ("walk", "strid")),
    ("eating", ("eat", "biting", "takes a bite", "taking a bite")),
    ("biting", ("eat", "biting", "takes a bite", "taking a bite")),
    ("holding", ("hold", "held", "carry")),
    ("full body", ("full body", "full-body", "head to feet", "both feet")),
    ("green", ("green",)),
    ("red", ("red",)),
    ("blue", ("blue",)),
    ("pink", ("pink",)),
    ("wooden", ("wood", "timber")),
    ("metal", ("metal", "steel", "iron")),
    ("bench", ("bench",)),
    ("sandwich", ("sandwich",)),
    ("park", ("park",)),
    ("mall", ("mall", "shopping center", "shopping centre")),
    ("escalator", ("escalator",)),
    ("stair", ("stair",)),
    ("marble", ("marble",)),
    ("tile", ("tile",)),
    ("sale", ("sale",)),
    ("arrival", ("arrival",)),
    ("handbag", ("handbag",)),
    ("passenger car", ("car", "cars", "passenger vehicle", "automobile")),
    ("cars", ("car", "cars", "passenger vehicle", "automobile")),
)


def _stable_caption_seed(prompt: str, height: int, width: int) -> int:
    digest = hashlib.sha256(f"{height}x{width}\0{prompt}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False) % (2**63 - 1)


def _single_token_id_sequence(value: Any) -> tuple[int, ...]:
    if isinstance(value, torch.Tensor):
        if value.ndim == 2 and value.shape[0] == 1:
            value = value[0]
        if value.ndim != 1:
            raise RuntimeError(
                "Ideogram4 tokenizer input_ids must have shape [tokens] or [1, tokens]."
            )
        values = value.detach().cpu().tolist()
    elif isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) == 1 and isinstance(values[0], (list, tuple)):
            values = list(values[0])
    else:
        raise RuntimeError("Ideogram4 tokenizer input_ids must be a tensor, list, or tuple.")
    if not values:
        raise RuntimeError("Ideogram4 tokenizer returned an empty input_ids sequence.")
    if any(not isinstance(token, int) or isinstance(token, bool) or token < 0 for token in values):
        raise RuntimeError("Ideogram4 tokenizer returned malformed token IDs.")
    return tuple(values)


def _quoted_literals(text: str) -> list[str]:
    matches = re.findall(r'"([^"\n]+)"|“([^”\n]+)”', text)
    return [straight or curly for straight, curly in matches]


_NON_PERSON_WALKING_PHRASE = re.compile(
    r"\b(?:(?:horizontal|moving|mechanical|motorized)\s+)?walking\s+"
    r"(?:floor|walkway|belt|surface|conveyor)\b"
)


def _without_non_person_walking_phrases(text: str) -> str:
    return _NON_PERSON_WALKING_PHRASE.sub(" ", text)


def _controlled_trigger_present(trigger: str, text: str) -> bool:
    if trigger == "walking":
        text = _without_non_person_walking_phrases(text)
    prefix_triggers = {"escalator", "handbag", "stair", "tile"}
    suffix = "" if trigger in prefix_triggers else r"\b"
    return re.search(rf"\b{re.escape(trigger)}{suffix}", text) is not None


def _controlled_alternative_present(
    trigger: str,
    alternatives: tuple[str, ...],
    text: str,
) -> bool:
    if trigger == "walking":
        text = _without_non_person_walking_phrases(text)
        return re.search(r"\b(?:walk(?:s|ed|ing)?|strid(?:e|es|ing|ed)?)\b", text) is not None
    return any(
        (
            re.search(rf"\b{re.escape(term)}\b", text) is not None
            if term in {"car", "cars", "red"}
            else term in text
        )
        for term in alternatives
    )


def _quoted_literal_spans(text: str) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    for match in re.finditer(r'"([^"\n]+)"|“([^”\n]+)”', text):
        group = 1 if match.group(1) is not None else 2
        start, end = match.span(group)
        spans.append({"payload": text[start:end], "start": start, "end": end})
    return spans


def _verbatim_controlled_spans(raw_prompt: str) -> list[dict[str, Any]]:
    """Locate controlled triggers as exact, case-preserved raw-prompt spans."""

    non_person_walking_spans = [
        match.span() for match in _NON_PERSON_WALKING_PHRASE.finditer(raw_prompt.casefold())
    ]
    prefix_triggers = {"escalator", "handbag", "stair", "tile"}
    records: list[dict[str, Any]] = []
    for trigger, _alternatives in _CONTROLLED_TERM_REQUIREMENTS:
        if trigger in prefix_triggers:
            pattern = rf"\b{re.escape(trigger)}[^\W\d_]*\b"
        else:
            pattern = rf"\b{re.escape(trigger)}\b"
        for match in re.finditer(pattern, raw_prompt, flags=re.IGNORECASE):
            if trigger == "walking" and any(
                outer_start <= match.start() and match.end() <= outer_end
                for outer_start, outer_end in non_person_walking_spans
            ):
                continue
            start, end = match.span()
            records.append(
                {
                    "trigger": trigger,
                    "payload": raw_prompt[start:end],
                    "start": start,
                    "end": end,
                }
            )
            break
    return records


def _native_caption_fidelity_gaps(caption: str, raw_prompt: str) -> dict[str, list[str]]:
    """Return ordered literal/concept losses after strict schema validation."""

    _validate_native_caption(caption, original_prompt=None)
    payload = json.loads(caption)
    elements = payload["compositional_deconstruction"]["elements"]
    quoted_literals = _quoted_literals(raw_prompt)
    text_elements = [element["text"] for element in elements if element["type"] == "text"]
    missing_literals = [
        literal for literal in quoted_literals if not any(literal in text for text in text_elements)
    ]
    caption_lower = caption.casefold()
    original_lower = raw_prompt.casefold()
    missing_terms = [
        trigger
        for trigger, alternatives in _CONTROLLED_TERM_REQUIREMENTS
        if _controlled_trigger_present(trigger, original_lower)
        and not _controlled_alternative_present(trigger, alternatives, caption_lower)
    ]
    return {
        "missing_quoted_literals": missing_literals,
        "missing_controlled_terms": missing_terms,
    }


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _repair_schema_valid_enhancer_caption(
    schema_valid_candidate: str,
    raw_prompt: str,
) -> tuple[str, dict[str, Any]]:
    """Restore only missing verbatim semantics already present in ``raw_prompt``."""

    _validate_native_caption(schema_valid_candidate, original_prompt=None)
    before = _native_caption_fidelity_gaps(schema_valid_candidate, raw_prompt)
    candidate_sha256 = hashlib.sha256(schema_valid_candidate.encode("utf-8")).hexdigest()
    if not before["missing_quoted_literals"] and not before["missing_controlled_terms"]:
        return schema_valid_candidate, {
            "fidelity_repair_applied": False,
            "inserted_elements": [],
            "missing_quoted_literals_before": [],
            "missing_controlled_terms_before": [],
            "missing_quoted_literals_after": [],
            "missing_controlled_terms_after": [],
            "all_inserted_semantic_payloads_are_verbatim_raw_spans": True,
            "target_injection_guard_status": "passed",
            "candidate_caption_sha256": candidate_sha256,
            "repaired_caption_sha256": candidate_sha256,
        }

    parsed = json.loads(schema_valid_candidate)
    elements = parsed["compositional_deconstruction"]["elements"]
    inserted_elements: list[dict[str, Any]] = []
    inserted_payloads: set[str] = set()

    def append_element(element: dict[str, str], *, payload: str, start: int, end: int) -> None:
        if raw_prompt[start:end] != payload:
            raise RuntimeError("Ideogram4 fidelity repair lost its exact raw source-span binding.")
        if payload in inserted_payloads:
            return
        elements.append(element)
        inserted_payloads.add(payload)
        inserted_elements.append(
            {
                "element_type": element["type"],
                "semantic_payload": payload,
                "raw_span": [start, end],
                "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                "final_element_sha256": _canonical_json_sha256(element),
            }
        )

    missing_literal_set = set(before["missing_quoted_literals"])
    for literal_span in _quoted_literal_spans(raw_prompt):
        literal = str(literal_span["payload"])
        if literal not in missing_literal_set:
            continue
        append_element(
            {
                "type": "text",
                "text": literal,
                "desc": "Visible text exactly as quoted by the user.",
            },
            payload=literal,
            start=int(literal_span["start"]),
            end=int(literal_span["end"]),
        )

    with_literals = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    remaining = _native_caption_fidelity_gaps(with_literals, raw_prompt)
    spans_by_trigger = {
        str(record["trigger"]): record for record in _verbatim_controlled_spans(raw_prompt)
    }
    for trigger, _alternatives in _CONTROLLED_TERM_REQUIREMENTS:
        if trigger not in remaining["missing_controlled_terms"]:
            continue
        source = spans_by_trigger.get(trigger)
        if source is None:
            raise RuntimeError(
                f"Ideogram4 fidelity repair could not bind controlled term {trigger!r} "
                "to the raw prompt."
            )
        payload = str(source["payload"])
        append_element(
            {"type": "obj", "desc": payload},
            payload=payload,
            start=int(source["start"]),
            end=int(source["end"]),
        )

    repaired = json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))
    _validate_native_caption(repaired, original_prompt=raw_prompt)
    after = _native_caption_fidelity_gaps(repaired, raw_prompt)
    if after["missing_quoted_literals"] or after["missing_controlled_terms"]:
        raise RuntimeError(f"Ideogram4 fidelity repair left unresolved gaps: {after}.")
    source_span_proof = all(
        raw_prompt[item["raw_span"][0] : item["raw_span"][1]] == item["semantic_payload"]
        for item in inserted_elements
    )
    if not source_span_proof:
        raise RuntimeError("Ideogram4 fidelity repair failed its target-injection guard.")
    return repaired, {
        "fidelity_repair_applied": bool(inserted_elements),
        "inserted_elements": inserted_elements,
        "missing_quoted_literals_before": list(before["missing_quoted_literals"]),
        "missing_controlled_terms_before": list(before["missing_controlled_terms"]),
        "missing_quoted_literals_after": list(after["missing_quoted_literals"]),
        "missing_controlled_terms_after": list(after["missing_controlled_terms"]),
        "all_inserted_semantic_payloads_are_verbatim_raw_spans": source_span_proof,
        "target_injection_guard_status": "passed",
        "candidate_caption_sha256": candidate_sha256,
        "repaired_caption_sha256": hashlib.sha256(repaired.encode("utf-8")).hexdigest(),
    }


def _validate_native_caption(caption: str, original_prompt: str | None) -> None:
    """Fail closed on schema drift or loss of controlled experiment concepts."""

    try:
        payload = json.loads(caption)
    except json.JSONDecodeError as exc:
        raise ValueError(f"caption is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("caption root must be a JSON object")
    canonical_top_order = [
        "high_level_description",
        "style_description",
        "compositional_deconstruction",
    ]
    if (
        set(payload) - set(canonical_top_order)
        or "compositional_deconstruction" not in payload
        or list(payload) != [key for key in canonical_top_order if key in payload]
    ):
        raise ValueError("caption does not use the official ordered top-level schema")
    high_level = payload.get("high_level_description")
    composition = payload["compositional_deconstruction"]
    if high_level is not None and (not isinstance(high_level, str) or not high_level.strip()):
        raise ValueError("high_level_description must be a non-empty string")
    style = payload.get("style_description")
    if style is not None:
        if not isinstance(style, dict):
            raise ValueError("style_description must be an object")
        has_photo = "photo" in style
        has_art = "art_style" in style
        if has_photo == has_art:
            raise ValueError("style_description must contain exactly one of photo or art_style")
        canonical_style_order = (
            ["aesthetics", "lighting", "photo", "medium", "color_palette"]
            if has_photo
            else ["aesthetics", "lighting", "medium", "art_style", "color_palette"]
        )
        if (
            set(style) - set(canonical_style_order)
            or list(style) != [key for key in canonical_style_order if key in style]
            or not all(key in style for key in ("aesthetics", "lighting", "medium"))
        ):
            raise ValueError("style_description keys are missing, unknown, or out of order")
        _validate_color_palette(style.get("color_palette"), maximum=16, path="style_description")
    if not isinstance(composition, dict) or list(composition) != ["background", "elements"]:
        raise ValueError("compositional_deconstruction must contain ordered background/elements")
    if not isinstance(composition["background"], str) or not composition["background"].strip():
        raise ValueError("background must be a non-empty string")
    elements = composition["elements"]
    if not isinstance(elements, list) or not elements:
        raise ValueError("elements must contain at least one object or text element")
    for index, element in enumerate(elements):
        if not isinstance(element, dict) or element.get("type") not in {"obj", "text"}:
            raise ValueError(f"element {index} has an unsupported type")
        if element["type"] == "obj":
            canonical_element_order = ["type", "bbox", "desc", "color_palette"]
            required = {"type", "desc"}
        else:
            canonical_element_order = ["type", "bbox", "text", "desc", "color_palette"]
            required = {"type", "text", "desc"}
        if (
            set(element) - set(canonical_element_order)
            or not required <= set(element)
            or list(element) != [key for key in canonical_element_order if key in element]
        ):
            raise ValueError(f"element {index} has missing, unknown, or out-of-order keys")
        if not isinstance(element.get("desc"), str) or not element["desc"].strip():
            raise ValueError(f"element {index} description is malformed")
        if element["type"] == "text" and (
            not isinstance(element.get("text"), str) or not element["text"].strip()
        ):
            raise ValueError(f"text element {index} is malformed")
        if "bbox" in element:
            bbox = element["bbox"]
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(isinstance(value, int) and 0 <= value <= 1000 for value in bbox)
                or bbox[0] > bbox[2]
                or bbox[1] > bbox[3]
            ):
                raise ValueError(f"element {index} bbox is malformed")
        _validate_color_palette(
            element.get("color_palette"),
            maximum=5,
            path=f"element {index}",
        )

    if original_prompt is None:
        return
    gaps = _native_caption_fidelity_gaps(caption, original_prompt)
    if gaps["missing_quoted_literals"]:
        raise ValueError(
            f"caption dropped quoted literal text: {gaps['missing_quoted_literals']}"
        )
    if gaps["missing_controlled_terms"]:
        raise ValueError(f"caption dropped controlled concepts: {gaps['missing_controlled_terms']}")


def _validate_color_palette(value: Any, *, maximum: int, path: str) -> None:
    if value is None:
        return
    if not isinstance(value, list) or len(value) > maximum:
        raise ValueError(f"{path} color_palette is malformed")
    if any(re.fullmatch(r"#[0-9A-F]{6}", color) is None for color in value):
        raise ValueError(f"{path} color_palette must contain uppercase #RRGGBB strings")
