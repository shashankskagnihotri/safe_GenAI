from __future__ import annotations

import gc
from contextlib import nullcontext
from dataclasses import dataclass
import hashlib
from importlib import metadata
import inspect
import json
from pathlib import Path
import re
from typing import Any, Mapping, Sequence

import numpy as np
from PIL import Image
import torch

from hierasafe_flow.adapters.base import (
    AdapterState,
    DenoisingStepContext,
    DiffusersFrozenAdapter,
    PromptCondition,
    SchedulerStepResult,
    configure_pipeline_vae_tiling,
)
from hierasafe_flow.adapters.hunyuan_temporal import (
    resample_hunyuan_native_24_to_16_exact,
)
from hierasafe_flow.evaluation.temporal_qualification import (
    TEMPORAL_CRITERIA_BY_MODEL,
    validate_temporal_production_gate,
)
from hierasafe_flow.generation.conditioning_cache import encoding_fingerprint
from hierasafe_flow.generation.temporal_artifacts import (
    TemporalEvidenceBundle,
    TemporalSegmentEvidence,
    frame_rgb_sha256,
)


HUNYUAN_DUAL_VIEW_CONFIG_KEY = "hunyuan_dual_view_conditioning"
HUNYUAN_DUAL_VIEW_SCHEMA_VERSION = 1
HUNYUAN_LLAMA_MAX_SEQUENCE_LENGTH = 256
HUNYUAN_CLIP_MAX_SEQUENCE_LENGTH = 77
HUNYUAN_LLAMA_HIDDEN_SIZE = 4096
HUNYUAN_CLIP_POOLED_SIZE = 768
HUNYUAN_TEMPORAL_PROTOCOL_KEY = "hunyuan_temporal_protocol"
HUNYUAN_T2V_MODEL_ID = "hunyuanvideo-community/HunyuanVideo"
HUNYUAN_I2V_MODEL_ID = "hunyuanvideo-community/HunyuanVideo-I2V"
HUNYUAN_CHECKPOINT_REVISION = "e8c2aaa66fe3742a32c11a6766aecbf07c56e773"
HUNYUAN_I2V_CHECKPOINT_REVISION = "fb9d287ef02fe6d39f2e23df6dcec1294e6c28d2"
HUNYUAN_DIFFUSERS_REVISION = "577b28f8f5d30eabdd357d74944cd76568292faf"
HUNYUAN_SEGMENT_SEED_DOMAIN = "hunyuan_i2v_continuation_v1"
HUNYUAN_ARTIFACT_MANIFEST_PATH = "configs/artifacts/hunyuan_video_segmented_temporal_v2.json"
HUNYUAN_PRIMARY_RECEIPT_PATH = "configs/artifacts/hunyuan_video_t2v_streaming_receipt_v1.json"
# Updated together with the immutable manifest below.  Keeping the digest in
# executable code makes a changed artifact inventory a source change rather
# than an implicit runtime choice.
HUNYUAN_ARTIFACT_MANIFEST_SHA256 = (
    "65d49d0de533c374eaeb238dc989269c0d61c957d2fff92f6840cb1337837523"
)
HUNYUAN_PRIMARY_RECEIPT_SHA256 = "c2eb43ed26816db7d78553b9576231bc392a8c0a880d556a8a8be6558356a391"

_EXPECTED_HUNYUAN_TEMPORAL_PROTOCOL = {
    "schema_version": 2,
    "native_temporal_call_schema_version": 2,
    "segment_trace_schema_version": 1,
    "temporal_evidence_schema_version": 1,
    "segmented_temporal_audit_schema_version": 1,
    "strategy": "community_diffusers_t2v_i2v_three_segment_composite",
    "scientific_label": "HunyuanVideo community-Diffusers T2V+I2V continuation composite",
    "diffusers_revision": HUNYUAN_DIFFUSERS_REVISION,
    "t2v_model_id": HUNYUAN_T2V_MODEL_ID,
    "t2v_revision": HUNYUAN_CHECKPOINT_REVISION,
    "i2v_model_id": HUNYUAN_I2V_MODEL_ID,
    "i2v_revision": HUNYUAN_I2V_CHECKPOINT_REVISION,
    "artifact_manifest": HUNYUAN_ARTIFACT_MANIFEST_PATH,
    "artifact_manifest_sha256": HUNYUAN_ARTIFACT_MANIFEST_SHA256,
    "primary_streaming_receipt_path": HUNYUAN_PRIMARY_RECEIPT_PATH,
    "primary_streaming_receipt_sha256": HUNYUAN_PRIMARY_RECEIPT_SHA256,
    "segment_count": 3,
    "segment_roles": ["t2v_primary", "i2v_continuation", "i2v_continuation"],
    "native_segment_frames": [121, 121, 121],
    "native_fps": 24,
    "native_latent_frames": 31,
    "i2v_dynamic_latent_frames": 30,
    "image_condition_type": "token_replace",
    "image_embed_interleave": 4,
    "scheduler_class": "FlowMatchEulerDiscreteScheduler",
    "scheduler_shifts": [7.0, 17.0, 17.0],
    "num_inference_steps_per_segment": 50,
    "embedded_guidance_scale": 6.0,
    "native_negative_true_cfg_scale": 4.0,
    "segment_seed_domain": HUNYUAN_SEGMENT_SEED_DOMAIN,
    "stitch_strategy": "segment_0[0:121]+segment_1[1:121]+segment_2[1:121]",
    "stitched_native_frames": 361,
    "resampling_method": "nearest_timestamp_decimation_round_half_up",
    "terminal_endpoint_crop": "stitched[0:360]",
    "output_frames": 240,
    "output_fps": 16,
    "duration_seconds": 15.0,
    "height": 544,
    "width": 960,
}
_HUNYUAN_TEMPORAL_DYNAMIC_FIELDS = {"execution_phase", "production_gate"}
_HUNYUAN_PRODUCTION_CRITERIA = TEMPORAL_CRITERIA_BY_MODEL["hunyuan_video"]
HUNYUAN_CONDITIONING_ROLES = frozenset(
    {"base", "baseline", "native_negative", "neutral", "unsafe", "safe"}
)


@dataclass(frozen=True)
class HunyuanPromptView:
    """Frozen, independently authored text views for HunyuanVideo's two encoders.

    ``raw_prompt`` is the byte-exact experiment condition used as the lookup
    key. ``llama_prompt`` retains the complete condition tree, while
    ``clip_prompt`` is the compact, concept-first preservation view that must
    fit CLIP's hard 77-token context. Optional token fingerprints allow a
    preflight-only run to freeze the exact tokenizer result for production.
    """

    entry_id: str
    role: str
    pair_id: str | None
    raw_prompt: str
    raw_prompt_sha256: str
    llama_prompt: str
    clip_prompt: str
    llama_token_count: int | None = None
    llama_token_ids_sha256: str | None = None
    clip_token_count: int | None = None
    clip_token_ids_sha256: str | None = None

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HunyuanPromptView":
        required = (
            "entry_id",
            "role",
            "raw_prompt",
            "raw_prompt_sha256",
            "llama_prompt",
            "clip_prompt",
        )
        missing = [key for key in required if key not in value]
        if missing:
            raise ValueError(f"Hunyuan conditioning-plan entry is missing fields: {missing}.")

        entry_id = _require_nonempty_text(value["entry_id"], "entry_id")
        role = _require_nonempty_text(value["role"], f"{entry_id}.role")
        if role.lower() not in HUNYUAN_CONDITIONING_ROLES:
            raise ValueError(
                f"Hunyuan conditioning-plan entry '{entry_id}' has unsupported role {role!r}; "
                f"allowed roles are {sorted(HUNYUAN_CONDITIONING_ROLES)}."
            )
        raw_prompt = _require_nonempty_text(value["raw_prompt"], f"{entry_id}.raw_prompt")
        llama_prompt = _require_nonempty_text(value["llama_prompt"], f"{entry_id}.llama_prompt")
        clip_prompt = _require_nonempty_text(value["clip_prompt"], f"{entry_id}.clip_prompt")
        raw_prompt_sha256 = _require_sha256(
            value["raw_prompt_sha256"], f"{entry_id}.raw_prompt_sha256"
        )
        actual_raw_sha256 = _sha256_text(raw_prompt)
        if raw_prompt_sha256 != actual_raw_sha256:
            raise ValueError(
                f"Hunyuan conditioning-plan entry '{entry_id}' raw prompt SHA-256 mismatch: "
                f"declared {raw_prompt_sha256}, calculated {actual_raw_sha256}."
            )

        pair_value = value.get("pair_id")
        pair_id = (
            None
            if pair_value is None
            else _require_nonempty_text(pair_value, f"{entry_id}.pair_id")
        )
        llama_token_count, llama_token_ids_sha256 = _optional_token_fingerprint(
            value,
            prefix="llama",
            entry_id=entry_id,
        )
        clip_token_count, clip_token_ids_sha256 = _optional_token_fingerprint(
            value,
            prefix="clip",
            entry_id=entry_id,
        )

        for field_name, text_value in (
            ("llama_prompt_sha256", llama_prompt),
            ("clip_prompt_sha256", clip_prompt),
        ):
            declared = value.get(field_name)
            if declared is not None:
                declared_sha256 = _require_sha256(declared, f"{entry_id}.{field_name}")
                actual_sha256 = _sha256_text(text_value)
                if declared_sha256 != actual_sha256:
                    raise ValueError(
                        f"Hunyuan conditioning-plan entry '{entry_id}' {field_name} mismatch: "
                        f"declared {declared_sha256}, calculated {actual_sha256}."
                    )

        return cls(
            entry_id=entry_id,
            role=role,
            pair_id=pair_id,
            raw_prompt=raw_prompt,
            raw_prompt_sha256=raw_prompt_sha256,
            llama_prompt=llama_prompt,
            clip_prompt=clip_prompt,
            llama_token_count=llama_token_count,
            llama_token_ids_sha256=llama_token_ids_sha256,
            clip_token_count=clip_token_count,
            clip_token_ids_sha256=clip_token_ids_sha256,
        )


@dataclass(frozen=True)
class HunyuanConditioningPlan:
    """Validated immutable prompt-view plan keyed by exact raw prompt text."""

    schema_version: int
    plan_sha256: str
    entries: tuple[HunyuanPromptView, ...]

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "HunyuanConditioningPlan":
        schema_version = value.get("schema_version")
        if schema_version != HUNYUAN_DUAL_VIEW_SCHEMA_VERSION:
            raise ValueError(
                "Hunyuan dual-view conditioning plan requires schema_version "
                f"{HUNYUAN_DUAL_VIEW_SCHEMA_VERSION}, got {schema_version!r}."
            )
        declared_sha256 = _require_sha256(value.get("plan_sha256"), "plan_sha256")
        calculated_sha256 = hunyuan_conditioning_plan_sha256(value)
        if declared_sha256 != calculated_sha256:
            raise ValueError(
                "Hunyuan dual-view conditioning plan SHA-256 mismatch: "
                f"declared {declared_sha256}, calculated {calculated_sha256}."
            )
        raw_entries = value.get("entries")
        if not isinstance(raw_entries, list) or not raw_entries:
            raise ValueError(
                "Hunyuan dual-view conditioning plan requires a non-empty entries list."
            )
        if not all(isinstance(entry, Mapping) for entry in raw_entries):
            raise ValueError("Every Hunyuan conditioning-plan entry must be a mapping.")
        entries = tuple(HunyuanPromptView.from_mapping(entry) for entry in raw_entries)

        entry_ids = [entry.entry_id for entry in entries]
        duplicate_entry_ids = _duplicates(entry_ids)
        if duplicate_entry_ids:
            raise ValueError(
                f"Hunyuan conditioning-plan entry_id values must be unique: {duplicate_entry_ids}."
            )
        raw_prompts = [entry.raw_prompt for entry in entries]
        duplicate_raw_prompts = _duplicates(raw_prompts)
        if duplicate_raw_prompts:
            raise ValueError(
                "Hunyuan conditioning-plan raw prompts must be byte-exact and unique; "
                f"duplicates found for {duplicate_raw_prompts}."
            )
        _validate_plan_pairs(entries)
        return cls(
            schema_version=HUNYUAN_DUAL_VIEW_SCHEMA_VERSION,
            plan_sha256=declared_sha256,
            entries=entries,
        )

    def resolve(self, raw_prompt: str) -> HunyuanPromptView:
        matches = [entry for entry in self.entries if entry.raw_prompt == raw_prompt]
        if len(matches) != 1:
            digest = _sha256_text(raw_prompt)
            raise KeyError(
                "Hunyuan dual-view conditioning refused an unplanned or ambiguous raw prompt "
                f"(sha256={digest}, matches={len(matches)})."
            )
        return matches[0]


@dataclass(frozen=True)
class HunyuanPromptEncoding:
    """Model-native tensors plus JSON-serializable dual-view provenance."""

    prompt_embeds: torch.Tensor
    pooled_prompt_embeds: torch.Tensor
    prompt_attention_mask: torch.Tensor
    records: tuple[dict[str, Any], ...]


def hunyuan_conditioning_plan_sha256(value: Mapping[str, Any]) -> str:
    """Hash the complete plan payload, excluding only its self-referential digest."""

    payload = dict(value)
    payload.pop("plan_sha256", None)
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Hunyuan conditioning plan must be canonical JSON data.") from exc
    return _sha256_text(canonical)


@torch.no_grad()
def encode_hunyuan_prompt_views(
    pipeline: Any,
    views: Sequence[HunyuanPromptView],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    llama_max_sequence_length: int = HUNYUAN_LLAMA_MAX_SEQUENCE_LENGTH,
    clip_max_sequence_length: int = HUNYUAN_CLIP_MAX_SEQUENCE_LENGTH,
    prompt_template: Mapping[str, Any] | None = None,
    plan_sha256: str | None = None,
) -> HunyuanPromptEncoding:
    """Encode independent Llama and CLIP views without Hunyuan's ``prompt_2`` bug.

    Released Diffusers HunyuanVideo pipelines accept ``prompt_2`` but have
    historically called ``_get_clip_prompt_embeds(prompt, ...)`` instead. We
    therefore encode CLIP explicitly and inject the resulting pooled tensor
    into ``encode_prompt``. Supplying ``pooled_prompt_embeds`` bypasses that
    faulty branch while retaining the pipeline's exact Llama template/crop.
    """

    frozen_views = tuple(views)
    if not frozen_views:
        raise ValueError("Hunyuan prompt-view encoding requires at least one view.")
    if not all(isinstance(view, HunyuanPromptView) for view in frozen_views):
        raise TypeError("Hunyuan prompt-view encoding accepts only HunyuanPromptView entries.")
    if llama_max_sequence_length != HUNYUAN_LLAMA_MAX_SEQUENCE_LENGTH:
        raise ValueError(
            "HunyuanVideo Llama conditioning must retain the checkpoint's exact "
            f"{HUNYUAN_LLAMA_MAX_SEQUENCE_LENGTH}-token post-crop length."
        )
    if clip_max_sequence_length != HUNYUAN_CLIP_MAX_SEQUENCE_LENGTH:
        raise ValueError(
            "HunyuanVideo CLIP conditioning must retain the checkpoint's exact "
            f"{HUNYUAN_CLIP_MAX_SEQUENCE_LENGTH}-token context."
        )
    if not callable(getattr(pipeline, "_get_clip_prompt_embeds", None)):
        raise RuntimeError(
            "HunyuanVideoPipeline no longer exposes _get_clip_prompt_embeds; "
            "dual-view routing cannot be verified."
        )
    if not callable(getattr(pipeline, "encode_prompt", None)):
        raise RuntimeError("HunyuanVideoPipeline does not expose encode_prompt.")
    if not hasattr(pipeline, "tokenizer") or not hasattr(pipeline, "tokenizer_2"):
        raise RuntimeError("HunyuanVideoPipeline must expose both Llama and CLIP tokenizers.")

    resolved_template = _resolve_prompt_template(pipeline, prompt_template)
    crop_start = _resolve_crop_start(pipeline.tokenizer, resolved_template)
    template_sha256 = _canonical_sha256(resolved_template)
    records = tuple(
        _preflight_prompt_view(
            pipeline,
            view,
            prompt_template=resolved_template,
            prompt_template_sha256=template_sha256,
            crop_start=crop_start,
            llama_max_sequence_length=llama_max_sequence_length,
            clip_max_sequence_length=clip_max_sequence_length,
            plan_sha256=plan_sha256,
        )
        for view in frozen_views
    )

    llama_prompts = [view.llama_prompt for view in frozen_views]
    clip_prompts = [view.clip_prompt for view in frozen_views]
    clip_pooled = pipeline._get_clip_prompt_embeds(
        prompt=clip_prompts,
        num_videos_per_prompt=1,
        device=device,
        dtype=dtype,
        max_sequence_length=clip_max_sequence_length,
    )
    output = pipeline.encode_prompt(
        prompt=llama_prompts,
        prompt_template=dict(resolved_template),
        num_videos_per_prompt=1,
        pooled_prompt_embeds=clip_pooled,
        device=device,
        dtype=dtype,
        max_sequence_length=llama_max_sequence_length,
    )
    if not isinstance(output, tuple) or len(output) != 3:
        raise RuntimeError(
            "HunyuanVideo encode_prompt contract drifted; expected exactly "
            "(prompt_embeds, pooled_prompt_embeds, prompt_attention_mask)."
        )
    prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = output
    if not isinstance(clip_pooled, torch.Tensor) or not isinstance(
        pooled_prompt_embeds, torch.Tensor
    ):
        raise RuntimeError("HunyuanVideo CLIP encoder did not return a pooled tensor.")
    if clip_pooled.shape != pooled_prompt_embeds.shape or not torch.equal(
        clip_pooled, pooled_prompt_embeds
    ):
        raise RuntimeError(
            "HunyuanVideo encode_prompt did not preserve the explicitly injected CLIP embeddings."
        )
    _validate_hunyuan_encoding_shapes(
        pipeline,
        prompt_embeds,
        pooled_prompt_embeds,
        prompt_attention_mask,
        batch_size=len(frozen_views),
        sequence_length=llama_max_sequence_length,
    )
    _validate_pooled_pair_deltas(frozen_views, pooled_prompt_embeds)

    output_shapes = {
        "prompt_embeds": [1, *prompt_embeds.shape[1:]],
        "pooled_prompt_embeds": [1, *pooled_prompt_embeds.shape[1:]],
        "prompt_attention_mask": [1, *prompt_attention_mask.shape[1:]],
    }
    finalized_records = tuple(
        {
            **record,
            "encoder_route": "t2v_dual_view_private_clip_injection",
            "output_shapes": output_shapes,
            "output_dtypes": {
                "prompt_embeds": str(prompt_embeds.dtype),
                "pooled_prompt_embeds": str(pooled_prompt_embeds.dtype),
                "prompt_attention_mask": str(prompt_attention_mask.dtype),
            },
            "output_fingerprints": {
                "prompt_embeds": encoding_fingerprint(prompt_embeds[index : index + 1]),
                "pooled_prompt_embeds": encoding_fingerprint(
                    pooled_prompt_embeds[index : index + 1]
                ),
                "prompt_attention_mask": encoding_fingerprint(
                    prompt_attention_mask[index : index + 1]
                ),
            },
        }
        for index, record in enumerate(records)
    )
    return HunyuanPromptEncoding(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        records=finalized_records,
    )


@torch.no_grad()
def encode_hunyuan_i2v_prompt_views(
    pipeline: Any,
    image: Image.Image,
    views: Sequence[HunyuanPromptView],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
    llama_max_sequence_length: int = HUNYUAN_LLAMA_MAX_SEQUENCE_LENGTH,
    clip_max_sequence_length: int = HUNYUAN_CLIP_MAX_SEQUENCE_LENGTH,
    image_embed_interleave: int = 4,
    prompt_template: Mapping[str, Any] | None = None,
    plan_sha256: str | None = None,
) -> HunyuanPromptEncoding:
    """Encode I2V dual views through the two authenticated private helpers.

    The pinned public I2V ``encode_prompt`` ignores ``prompt_2`` when it builds
    CLIP pooled conditioning.  Calling the two helpers explicitly is therefore
    part of the experiment contract, not an optimization.  Each LLaVA branch
    receives the caller-supplied anchor while CLIP receives only its compact
    registered text view.
    """

    frozen_views = tuple(views)
    if not frozen_views or not all(isinstance(view, HunyuanPromptView) for view in frozen_views):
        raise TypeError("Hunyuan I2V prompt-view encoding requires HunyuanPromptView entries.")
    if not isinstance(image, Image.Image):
        raise TypeError("Hunyuan I2V conditioning requires one real PIL anchor image.")
    if image_embed_interleave != 4:
        raise ValueError("Hunyuan token-replace I2V requires image_embed_interleave=4.")
    if llama_max_sequence_length != HUNYUAN_LLAMA_MAX_SEQUENCE_LENGTH:
        raise ValueError("Hunyuan I2V LLaVA conditioning requires max_sequence_length=256.")
    if clip_max_sequence_length != HUNYUAN_CLIP_MAX_SEQUENCE_LENGTH:
        raise ValueError("Hunyuan I2V CLIP conditioning requires max_sequence_length=77.")
    for helper_name in ("_get_llama_prompt_embeds", "_get_clip_prompt_embeds"):
        if not callable(getattr(pipeline, helper_name, None)):
            raise RuntimeError(
                f"HunyuanVideoImageToVideoPipeline no longer exposes {helper_name}; "
                "the registered dual-view route cannot continue."
            )
    if not hasattr(pipeline, "tokenizer") or not hasattr(pipeline, "tokenizer_2"):
        raise RuntimeError("Hunyuan I2V pipeline must expose both authenticated tokenizers.")

    resolved_template = _resolve_prompt_template(pipeline, prompt_template)
    crop_start = _resolve_crop_start(pipeline.tokenizer, resolved_template)
    template_sha256 = _canonical_sha256(resolved_template)
    records = tuple(
        _preflight_prompt_view(
            pipeline,
            view,
            prompt_template=resolved_template,
            prompt_template_sha256=template_sha256,
            crop_start=crop_start,
            llama_max_sequence_length=llama_max_sequence_length,
            clip_max_sequence_length=clip_max_sequence_length,
            plan_sha256=plan_sha256,
        )
        for view in frozen_views
    )
    anchor_sha256 = frame_rgb_sha256(image)
    prompt_rows: list[torch.Tensor] = []
    mask_rows: list[torch.Tensor] = []
    pooled_rows: list[torch.Tensor] = []
    for view in frozen_views:
        prompt_row, mask_row = pipeline._get_llama_prompt_embeds(
            image=image,
            prompt=view.llama_prompt,
            prompt_template=dict(resolved_template),
            num_videos_per_prompt=1,
            device=device,
            dtype=dtype,
            max_sequence_length=llama_max_sequence_length,
            image_embed_interleave=image_embed_interleave,
        )
        pooled_row = pipeline._get_clip_prompt_embeds(
            prompt=view.clip_prompt,
            num_videos_per_prompt=1,
            device=device,
            dtype=dtype,
            max_sequence_length=clip_max_sequence_length,
        )
        prompt_rows.append(prompt_row)
        mask_rows.append(mask_row)
        pooled_rows.append(pooled_row)
    if frame_rgb_sha256(image) != anchor_sha256:
        raise RuntimeError("Hunyuan I2V prompt encoding mutated its anchor image.")

    prompt_embeds = torch.cat(prompt_rows, dim=0)
    pooled_prompt_embeds = torch.cat(pooled_rows, dim=0)
    prompt_attention_mask = torch.cat(mask_rows, dim=0)
    _validate_hunyuan_i2v_encoding_shapes(
        pipeline,
        prompt_embeds,
        pooled_prompt_embeds,
        prompt_attention_mask,
        batch_size=len(frozen_views),
    )
    _validate_pooled_pair_deltas(frozen_views, pooled_prompt_embeds)
    finalized_records = tuple(
        {
            **record,
            "encoder_route": "i2v_private_llava_real_anchor_and_private_clip_view",
            "anchor_image_sha256": anchor_sha256,
            "image_embed_interleave": image_embed_interleave,
            "output_shapes": {
                "prompt_embeds": list(prompt_embeds[index : index + 1].shape),
                "pooled_prompt_embeds": list(pooled_prompt_embeds[index : index + 1].shape),
                "prompt_attention_mask": list(prompt_attention_mask[index : index + 1].shape),
            },
            "output_dtypes": {
                "prompt_embeds": str(prompt_embeds.dtype),
                "pooled_prompt_embeds": str(pooled_prompt_embeds.dtype),
                "prompt_attention_mask": str(prompt_attention_mask.dtype),
            },
            "output_fingerprints": {
                "prompt_embeds": encoding_fingerprint(prompt_embeds[index : index + 1]),
                "pooled_prompt_embeds": encoding_fingerprint(
                    pooled_prompt_embeds[index : index + 1]
                ),
                "prompt_attention_mask": encoding_fingerprint(
                    prompt_attention_mask[index : index + 1]
                ),
            },
        }
        for index, record in enumerate(records)
    )
    return HunyuanPromptEncoding(
        prompt_embeds=prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        prompt_attention_mask=prompt_attention_mask,
        records=finalized_records,
    )


class HunyuanVideoAdapter(DiffusersFrozenAdapter):
    """Authenticated community-Diffusers T2V→I2V 15-second composite."""

    adapter_name = "hunyuan_video"
    task_type = "text_to_video"
    pipeline_class_name = "HunyuanVideoPipeline"
    latent_feature_dim = 1  # [B, C, F, H, W]
    required_components = ("transformer", "scheduler", "vae")
    encode_prompt_output_names = ("prompt_embeds", "pooled_prompt_embeds", "prompt_attention_mask")

    def __init__(
        self,
        model_id: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id=model_id, device=device, dtype=dtype, config=config)
        raw_plan = self.config.get(HUNYUAN_DUAL_VIEW_CONFIG_KEY)
        if raw_plan is not None and not isinstance(raw_plan, Mapping):
            raise ValueError(f"{HUNYUAN_DUAL_VIEW_CONFIG_KEY} must be a mapping.")
        self._conditioning_plan = (
            HunyuanConditioningPlan.from_mapping(raw_plan) if raw_plan is not None else None
        )
        self._conditioning_records: dict[str, dict[str, Any]] = {}
        self._plan_preflight_record: dict[str, Any] | None = None
        self._last_native_negative_views: tuple[HunyuanPromptView, HunyuanPromptView] | None = None
        self._pipeline_role = "t2v"
        self._native_pipeline_temporal_configured = False
        self._preloaded_i2v_scheduler: Any | None = None
        self._native_segment_timesteps: list[list[Any]] = []
        self._completed_segments: list[list[Any]] = []
        self._segment_seeds: list[int] = []
        self._segment_conditioning: list[dict[str, Any]] = []
        self._segment_protected_state: list[dict[str, Any]] = []
        self._temporal_evidence: TemporalEvidenceBundle | None = None
        self._last_temporal_provenance: dict[str, Any] | None = None
        self._artifact_authentication: dict[str, Any] | None = None
        self._transition_provenance: dict[str, Any] | None = None

    def load(self) -> None:
        protocol = self._temporal_protocol()
        if protocol is not None:
            self._validate_preload_contract(protocol)
            self._artifact_authentication = self._authenticate_artifact_manifest(protocol)
        super().load()
        self._pipeline_role = "t2v"
        if protocol is not None:
            self._validate_primary_checkpoint_contract()

    def configure_native_pipeline_for_temporal_protocol(self) -> dict[str, Any]:
        self._require_loaded()
        protocol = self._temporal_protocol()
        if protocol is None:
            raise RuntimeError(
                "Native HunyuanVideo completion requires temporal protocol schema 2."
            )
        if self._conditioning_plan is None:
            raise RuntimeError("Native HunyuanVideo completion requires the frozen dual-view plan.")
        self._validate_temporal_execution_gate(protocol)
        self._validate_primary_checkpoint_contract()
        self._native_pipeline_temporal_configured = True
        return {
            "schema_version": 2,
            "native_temporal_call_schema_version": 2,
            "segment_trace_schema_version": 1,
            "temporal_evidence_schema_version": 1,
            "strategy": protocol["strategy"],
            "segment_count": 3,
            "first_call_role": "t2v_primary",
            "checkpoint_id": HUNYUAN_T2V_MODEL_ID,
            "checkpoint_revision": HUNYUAN_CHECKPOINT_REVISION,
            "num_frames": 121,
            "native_fps": 24,
            "height": 544,
            "width": 960,
            "num_inference_steps": 50,
            "guidance_scale": 6.0,
            "true_cfg_scale": 4.0,
            "scheduler": "FlowMatchEulerDiscreteScheduler",
            "scheduler_shift": 7.0,
            "output_frames": 240,
            "output_fps": 16,
            "duration_seconds": 15.0,
            "completion_hook": "HunyuanVideoAdapter.complete_native_pipeline_temporal_protocol",
            "one_way_model_transition": True,
        }

    def complete_native_pipeline_temporal_protocol(
        self,
        first_segment_media: Any,
        *,
        prompt: str | None = None,
        negative_prompt: str | None = None,
        generator: torch.Generator,
        **first_call_kwargs: Any,
    ) -> list[list[Any]]:
        """Complete native-negative generation with two authenticated I2V calls."""

        if not self._native_pipeline_temporal_configured:
            raise RuntimeError("Configure the Hunyuan native temporal route before completion.")
        if (prompt is not None and prompt.strip()) or (
            negative_prompt is not None and negative_prompt.strip()
        ):
            raise RuntimeError("Hunyuan native completion forbids raw prompt routing.")
        if not isinstance(generator, torch.Generator):
            raise TypeError("Hunyuan native completion requires an explicit torch.Generator.")
        if self._last_native_negative_views is None:
            raise RuntimeError(
                "Hunyuan native completion lacks the adapter-owned base/native-negative views."
            )
        _validate_native_first_call_kwargs(first_call_kwargs)
        first = _validate_hunyuan_video_batch(first_segment_media, 121)
        if len(first) != 1:
            raise RuntimeError(
                "Hunyuan segmented temporal generation is qualified at batch size 1."
            )

        positive_view, negative_view = self._last_native_negative_views
        positive_tensor_keys = (
            "prompt_embeds",
            "pooled_prompt_embeds",
            "prompt_attention_mask",
        )
        negative_tensor_keys = (
            "negative_prompt_embeds",
            "negative_pooled_prompt_embeds",
            "negative_prompt_attention_mask",
        )
        positive_fingerprints = {
            key: encoding_fingerprint(first_call_kwargs[key]) for key in positive_tensor_keys
        }
        negative_fingerprints = {
            key: encoding_fingerprint(first_call_kwargs[key]) for key in negative_tensor_keys
        }
        tensor_shapes = {
            key: list(first_call_kwargs[key].shape)
            for key in (*positive_tensor_keys, *negative_tensor_keys)
        }
        base_seed = int(generator.initial_seed())
        self._completed_segments = [first[0]]
        self._segment_seeds = [base_seed]
        self._segment_conditioning = [
            {
                "mode": "six_explicit_t2v_tensors",
                "positive_entry_id": positive_view.entry_id,
                "negative_entry_id": negative_view.entry_id,
                "positive_prompt_sha256": positive_view.raw_prompt_sha256,
                "negative_prompt_sha256": negative_view.raw_prompt_sha256,
                "positive_output_fingerprints": positive_fingerprints,
                "negative_output_fingerprints": negative_fingerprints,
                "tensor_shapes": tensor_shapes,
                "batch_size": 1,
                "raw_prompt_arguments_present": False,
            }
        ]
        self._segment_protected_state = [
            {
                "model_role": "t2v_primary",
                "fixed_anchor_temporal_positions": 0,
                "runner_visible_dynamic_temporal_positions": 31,
                "steering_scope": "all_31_t2v_temporal_positions",
                "branch_observations": {
                    positive_view.entry_id: {
                        "role": positive_view.role,
                        "batch_size": 1,
                        "fixed_anchor_latent_fingerprint": None,
                        "conditioning_fingerprints": positive_fingerprints,
                    },
                    negative_view.entry_id: {
                        "role": negative_view.role,
                        "batch_size": 1,
                        "fixed_anchor_latent_fingerprint": None,
                        "conditioning_fingerprints": negative_fingerprints,
                    },
                },
            }
        ]
        self._temporal_evidence = None
        self._load_continuation_pipeline()
        assert self.pipeline is not None

        for segment_index in (1, 2):
            anchor = self._completed_segments[-1][-1]
            anchor_sha256 = frame_rgb_sha256(anchor)
            seed = _hunyuan_continuation_seed(base_seed, segment_index)
            segment_generator = _make_hunyuan_generator(seed, self.device)
            positive = self._encode_i2v_views(anchor, (positive_view,))
            black = Image.new("RGB", (960, 544), color=0)
            black_sha256 = frame_rgb_sha256(black)
            negative = self._encode_i2v_views(black, (negative_view,))
            if positive.records[0]["anchor_image_sha256"] != anchor_sha256:
                raise RuntimeError(
                    "Positive Hunyuan I2V conditioning did not bind the real anchor."
                )
            if negative.records[0]["anchor_image_sha256"] != black_sha256:
                raise RuntimeError(
                    "Negative Hunyuan I2V conditioning did not bind the exact black image."
                )
            self._reset_i2v_scheduler(segment_index)
            call_kwargs = {
                "image": anchor,
                "prompt_embeds": positive.prompt_embeds,
                "pooled_prompt_embeds": positive.pooled_prompt_embeds,
                "prompt_attention_mask": positive.prompt_attention_mask,
                "negative_prompt_embeds": negative.prompt_embeds,
                "negative_pooled_prompt_embeds": negative.pooled_prompt_embeds,
                "negative_prompt_attention_mask": negative.prompt_attention_mask,
                "height": 544,
                "width": 960,
                "num_frames": 121,
                "num_inference_steps": 50,
                "guidance_scale": 6.0,
                "true_cfg_scale": 4.0,
                "image_embed_interleave": 4,
                "generator": segment_generator,
                "output_type": "pil",
                "return_dict": True,
            }
            if {"prompt", "prompt_2", "negative_prompt", "negative_prompt_2"} & set(call_kwargs):
                raise RuntimeError("Hunyuan continuation call unexpectedly contains raw prompts.")
            output = self.pipeline(**call_kwargs)
            media = _extract_hunyuan_pipeline_media(output)
            segment = _validate_hunyuan_video_batch(media, 121)[0]
            self._completed_segments.append(segment)
            self._segment_seeds.append(seed)
            self._segment_conditioning.append(
                {
                    "mode": "six_explicit_i2v_tensors",
                    "positive_entry_id": positive_view.entry_id,
                    "negative_entry_id": negative_view.entry_id,
                    "positive_anchor_sha256": anchor_sha256,
                    "negative_anchor_sha256": black_sha256,
                    "positive_output_fingerprints": positive.records[0]["output_fingerprints"],
                    "negative_output_fingerprints": negative.records[0]["output_fingerprints"],
                    "raw_prompt_arguments_present": False,
                }
            )
            self._segment_protected_state.append(
                {"fixed_anchor_rgb_sha256": anchor_sha256, "steered": False}
            )
        return self._stitch_and_resample_segments(
            self._completed_segments,
            generation_path="native_negative_t2v_then_two_native_i2v_calls",
        )

    def resolve_prompt_views(
        self,
        prompts: str | Sequence[str],
    ) -> tuple[HunyuanPromptView, ...]:
        if self._conditioning_plan is None:
            raise RuntimeError(f"No {HUNYUAN_DUAL_VIEW_CONFIG_KEY} plan is configured.")
        raw_prompts = (prompts,) if isinstance(prompts, str) else tuple(prompts)
        if not raw_prompts:
            raise ValueError("At least one raw prompt is required.")
        if not all(isinstance(prompt, str) for prompt in raw_prompts):
            raise TypeError("Hunyuan raw prompts must be strings.")
        return tuple(self._conditioning_plan.resolve(prompt) for prompt in raw_prompts)

    def encode_prompt_views(
        self,
        views: Sequence[HunyuanPromptView],
    ) -> HunyuanPromptEncoding:
        self._require_loaded()
        if self._pipeline_role != "t2v":
            raise RuntimeError(
                "T2V prompt encoding is unreachable after the one-way I2V transition."
            )
        assert self.pipeline is not None
        encoding = encode_hunyuan_prompt_views(
            self.pipeline,
            views,
            device=self.device,
            dtype=self.dtype,
            llama_max_sequence_length=self._max_sequence_length(),
            clip_max_sequence_length=HUNYUAN_CLIP_MAX_SEQUENCE_LENGTH,
            prompt_template=self._prompt_template(),
            plan_sha256=self._conditioning_plan.plan_sha256 if self._conditioning_plan else None,
        )
        self._record_encoding(encoding, route="t2v")
        frozen_views = tuple(views)
        if len(frozen_views) == 2 and [view.role.lower() for view in frozen_views] == [
            "baseline",
            "native_negative",
        ]:
            self._last_native_negative_views = (frozen_views[0], frozen_views[1])
        elif len(frozen_views) == 2 and [view.role.lower() for view in frozen_views] == [
            "base",
            "native_negative",
        ]:
            self._last_native_negative_views = (frozen_views[0], frozen_views[1])
        return encoding

    def _encode_i2v_views(
        self,
        anchor: Image.Image,
        views: Sequence[HunyuanPromptView],
    ) -> HunyuanPromptEncoding:
        self._require_loaded()
        if self._pipeline_role != "i2v":
            raise RuntimeError(
                "I2V prompt encoding requires the authenticated continuation pipeline."
            )
        assert self.pipeline is not None
        encoding = encode_hunyuan_i2v_prompt_views(
            self.pipeline,
            anchor,
            views,
            device=self.device,
            dtype=self.dtype,
            llama_max_sequence_length=self._max_sequence_length(),
            clip_max_sequence_length=HUNYUAN_CLIP_MAX_SEQUENCE_LENGTH,
            image_embed_interleave=4,
            prompt_template=self._prompt_template(),
            plan_sha256=self._conditioning_plan.plan_sha256 if self._conditioning_plan else None,
        )
        self._record_encoding(encoding, route="i2v", anchor_sha256=frame_rgb_sha256(anchor))
        return encoding

    def preflight_conditioning_plan(self, *, batch_size: int | None = None) -> dict[str, Any]:
        self._require_loaded()
        if self._conditioning_plan is None:
            raise RuntimeError(
                f"No {HUNYUAN_DUAL_VIEW_CONFIG_KEY} plan is configured for preflight."
            )
        if batch_size is None:
            batch_size = int(self.config.get("prompt_batch_size", 2))
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("Hunyuan conditioning-plan preflight batch_size must be positive.")
        pooled_by_entry: dict[str, torch.Tensor] = {}
        fingerprints: dict[str, str] = {}
        entries = self._conditioning_plan.entries
        for start in range(0, len(entries), batch_size):
            chunk = entries[start : start + batch_size]
            encoding = self.encode_prompt_views(chunk)
            for index, view in enumerate(chunk):
                pooled_by_entry[view.entry_id] = (
                    encoding.pooled_prompt_embeds[index].detach().float().cpu()
                )
                fingerprints[view.entry_id] = encoding.records[index][
                    "dual_view_fingerprint_sha256"
                ]
        ordered_pooled = torch.stack([pooled_by_entry[entry.entry_id] for entry in entries])
        record = {
            "completed": True,
            "plan_sha256": self._conditioning_plan.plan_sha256,
            "entry_count": len(entries),
            "batch_size": batch_size,
            "entry_fingerprints": fingerprints,
            "unsafe_safe_clip_delta_l2": _validate_pooled_pair_deltas(entries, ordered_pooled),
        }
        self._plan_preflight_record = record
        return dict(record)

    def _call_encode_prompt(self, prompt: str | list[str]) -> Any:
        if self._conditioning_plan is None:
            return super()._call_encode_prompt(prompt)
        encoding = self.encode_prompt_views(self.resolve_prompt_views(prompt))
        return encoding.prompt_embeds, encoding.pooled_prompt_embeds, encoding.prompt_attention_mask

    def prepare_prompt(self, prompt: str) -> PromptCondition:
        return self.prepare_prompts([prompt])[0]

    def prepare_prompts(self, prompts: list[str]) -> list[PromptCondition]:
        if self._conditioning_plan is None:
            return super().prepare_prompts(prompts)
        if not prompts:
            return []
        encoding = self.encode_prompt_views(self.resolve_prompt_views(prompts))
        return self._conditions_from_encoding(prompts, encoding, state=None)

    def conditioning_cache_identity(
        self,
        prompt: str,
        state: AdapterState,
        *,
        prompt_view: str,
        call_role: str,
    ) -> dict[str, Any]:
        identity = super().conditioning_cache_identity(
            prompt, state, prompt_view=prompt_view, call_role=call_role
        )
        if self._conditioning_plan is not None:
            view = self._conditioning_plan.resolve(prompt)
            identity.update(
                {
                    "dual_view_entry_id": view.entry_id,
                    "dual_view_role": view.role,
                    "llama_prompt_sha256": _sha256_text(view.llama_prompt),
                    "clip_prompt_sha256": _sha256_text(view.clip_prompt),
                    "conditioning_plan_sha256": self._conditioning_plan.plan_sha256,
                    "llama_max_sequence_length": 256,
                    "clip_max_sequence_length": 77,
                    "image_embed_interleave": 4 if self._pipeline_role == "i2v" else None,
                }
            )
        return identity

    def prepare_prompt_for_state(
        self,
        prompt: str,
        state: AdapterState,
        *,
        prompt_view: str,
        call_role: str,
    ) -> PromptCondition:
        return self.prepare_prompts_for_state(
            [prompt], state, prompt_view=prompt_view, call_roles=[call_role]
        )[0]

    def prepare_prompts_for_state(
        self,
        prompts: list[str],
        state: AdapterState,
        *,
        prompt_view: str,
        call_roles: list[str],
    ) -> list[PromptCondition]:
        if len(prompts) != len(call_roles):
            raise ValueError("prompts and call_roles must have identical lengths.")
        if self._conditioning_plan is None:
            return super().prepare_prompts_for_state(
                prompts, state, prompt_view=prompt_view, call_roles=call_roles
            )
        views = self.resolve_prompt_views(prompts)
        if self._pipeline_role == "t2v":
            encoding = self.encode_prompt_views(views)
        else:
            anchor = state.extra.get("anchor_image")
            expected_sha = state.extra.get("anchor_sha256")
            if not isinstance(anchor, Image.Image) or frame_rgb_sha256(anchor) != expected_sha:
                raise RuntimeError("Hunyuan I2V state lacks its authenticated real anchor image.")
            encoding = self._encode_i2v_views(anchor, views)
        return self._conditions_from_encoding(
            prompts,
            encoding,
            state=state,
            prompt_view=prompt_view,
            call_roles=call_roles,
        )

    def _conditions_from_encoding(
        self,
        prompts: Sequence[str],
        encoding: HunyuanPromptEncoding,
        *,
        state: AdapterState | None,
        prompt_view: str = "registered",
        call_roles: Sequence[str] | None = None,
    ) -> list[PromptCondition]:
        roles = tuple(call_roles or ("unscoped",) * len(prompts))
        conditions: list[PromptCondition] = []
        for index, (prompt, record, call_role) in enumerate(zip(prompts, encoding.records, roles)):
            condition_record = dict(record)
            if state is not None:
                identity = self.conditioning_cache_identity(
                    prompt, state, prompt_view=prompt_view, call_role=call_role
                )
                condition_record["state_identity"] = identity
                condition_record["state_identity_sha256"] = _canonical_sha256(identity)
                self._store_conditioning_record(condition_record)
            conditions.append(
                PromptCondition(
                    prompt=prompt,
                    data={
                        "prompt_embeds": encoding.prompt_embeds[index : index + 1],
                        "pooled_prompt_embeds": encoding.pooled_prompt_embeds[index : index + 1],
                        "prompt_attention_mask": encoding.prompt_attention_mask[index : index + 1],
                        "hunyuan_conditioning": condition_record,
                        "hunyuan_dual_view_fingerprint": condition_record[
                            "dual_view_fingerprint_sha256"
                        ],
                    },
                )
            )
        return conditions

    def conditioning_provenance(self) -> dict[str, Any]:
        plan = self._conditioning_plan
        provenance: dict[str, Any] = {
            "schema_version": 2,
            "adapter": self.adapter_name,
            "conditioning_mode": "dual_view" if plan is not None else "legacy_single_view",
            "routing": {
                "t2v": "private CLIP view injected into T2V encode_prompt",
                "i2v": "private LLaVA(real anchor, llama view) plus private CLIP view",
            },
            "model_id": self.model_id,
            "model_revision": self.config.get("revision"),
            "plan_sha256": plan.plan_sha256 if plan is not None else None,
            "plan_entry_count": len(plan.entries) if plan is not None else 0,
            "plan_entries": [_prompt_view_provenance(entry) for entry in plan.entries]
            if plan
            else [],
            "plan_preflight": self._plan_preflight_record,
            "records": [
                self._conditioning_records[key] for key in sorted(self._conditioning_records)
            ],
            "artifact_authentication": self._artifact_authentication,
            "one_way_transition": self._transition_provenance,
        }
        if self.pipeline is not None:
            provenance["runtime"] = _hunyuan_runtime_provenance(self.pipeline)
        protocol = self._temporal_protocol()
        if protocol is not None:
            provenance["temporal_generation"] = (
                self._last_temporal_provenance
                if self._last_temporal_provenance is not None
                else self._configured_temporal_provenance(protocol)
            )
        return provenance

    def prepare_initial_latents(
        self,
        prompt: str,
        batch_size: int,
        generator: torch.Generator | None,
        **generation_kwargs: Any,
    ) -> tuple[torch.Tensor, AdapterState]:
        self._require_loaded()
        assert self.pipeline is not None
        protocol = self._temporal_protocol()
        requested = int(generation_kwargs.get("num_frames", 129))
        if protocol is None:
            if requested > 129:
                raise ValueError(
                    f"{self.model_id} supports at most 129 direct frames; refusing unsupported "
                    f"direct denoising of {requested}."
                )
            if requested <= 0 or (requested - 1) % 4:
                raise ValueError("HunyuanVideo native frame counts must satisfy 4*N+1.")
            return super().prepare_initial_latents(
                prompt, batch_size, generator, **generation_kwargs
            )
        self._validate_temporal_execution_gate(protocol)
        self._validate_requested_temporal_output(generation_kwargs, protocol)
        self._validate_primary_checkpoint_contract()
        if self._conditioning_plan is None:
            raise RuntimeError("Hunyuan segmented generation requires the frozen dual-view plan.")
        if batch_size != 1:
            raise RuntimeError("Hunyuan segmented generation is qualified only at batch size 1.")
        latents = self.pipeline.prepare_latents(
            batch_size=1,
            num_channels_latents=16,
            height=544,
            width=960,
            num_frames=121,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
            latents=generation_kwargs.get("latents"),
        )
        if tuple(latents.shape[:3]) != (1, 16, 31):
            raise RuntimeError(
                f"Hunyuan T2V segment expected [1,16,31,...], got {tuple(latents.shape)}."
            )
        base_seed = int(generator.initial_seed()) if generator is not None else 0
        state = AdapterState(
            extra={
                "prompt": prompt,
                "base_seed": base_seed,
                "segment_seed": base_seed,
                "segment_generator": generator,
                "segment_index": 0,
                "segment_count": 3,
                "segment_step_index": 0,
                "condition_epoch": 0,
                "model_role": "t2v_primary",
                "model_id": HUNYUAN_T2V_MODEL_ID,
                "model_revision": HUNYUAN_CHECKPOINT_REVISION,
                "anchor_image": None,
                "anchor_sha256": None,
                "height": 544,
                "width": 960,
                "native_num_frames": 121,
                "attention_kwargs": generation_kwargs.get("attention_kwargs"),
            }
        )
        self._pipeline_role = "t2v"
        self._completed_segments = []
        self._segment_seeds = [base_seed]
        self._segment_conditioning = [{"mode": "state_aware_dual_view", "records": {}}]
        self._segment_protected_state = [
            {
                "model_role": "t2v_primary",
                "fixed_anchor_temporal_positions": 0,
                "runner_visible_dynamic_temporal_positions": 31,
                "steering_scope": "all_31_t2v_temporal_positions",
                "branch_observations": {},
            }
        ]
        self._temporal_evidence = None
        self._last_temporal_provenance = None
        return latents, state

    def set_timesteps(
        self,
        num_inference_steps: int,
        latents: torch.Tensor | None = None,
        state: AdapterState | None = None,
    ) -> list[Any]:
        protocol = self._temporal_protocol()
        if protocol is None:
            return self._retrieve_timesteps(self.pipeline.scheduler, num_inference_steps)
        if num_inference_steps != 50:
            raise ValueError("Hunyuan schema-2 protocol requires exactly 50 steps per segment.")
        if state is None:
            raise RuntimeError("Hunyuan segmented scheduling requires AdapterState.")
        assert self.pipeline is not None
        self._require_scheduler_shift(self.pipeline.scheduler, 7.0, "T2V")
        t2v = self._retrieve_timesteps(self.pipeline.scheduler, 50)
        self._preloaded_i2v_scheduler = self._new_i2v_scheduler()
        self._require_scheduler_shift(self._preloaded_i2v_scheduler, 17.0, "I2V")
        i2v_first = self._retrieve_timesteps(self._preloaded_i2v_scheduler, 50)
        i2v_second = [
            value.clone() if isinstance(value, torch.Tensor) else value for value in i2v_first
        ]
        self._native_segment_timesteps = [list(t2v), list(i2v_first), list(i2v_second)]
        self.timesteps = [*t2v, *i2v_first, *i2v_second]
        state.extra.update(local_num_steps=50, global_num_steps=150)
        return self.timesteps

    def denoising_step_context(
        self,
        global_step_index: int,
        global_num_steps: int,
        state: AdapterState,
    ) -> DenoisingStepContext:
        if self._temporal_protocol() is None:
            return super().denoising_step_context(global_step_index, global_num_steps, state)
        if global_num_steps != 150:
            raise RuntimeError("Hunyuan segmented runner must execute exactly 150 global steps.")
        return DenoisingStepContext(
            global_step_index=global_step_index,
            global_num_steps=global_num_steps,
            segment_index=int(state.extra["segment_index"]),
            segment_count=3,
            local_step_index=int(state.extra["segment_step_index"]),
            local_num_steps=50,
            model_role=str(state.extra["model_role"]),
            model_id=str(state.extra["model_id"]),
            model_revision=str(state.extra["model_revision"]),
            condition_epoch=int(state.extra["condition_epoch"]),
            anchor_sha256=state.extra.get("anchor_sha256"),
            segment_seed=int(state.extra["segment_seed"]),
        )

    def predict_vector_field(
        self,
        latents: torch.Tensor,
        timestep: Any,
        condition: PromptCondition,
        state: AdapterState,
    ) -> torch.Tensor:
        self._require_loaded()
        assert self.pipeline is not None
        prompt_embeds = condition.data.get("prompt_embeds")
        pooled = condition.data.get("pooled_prompt_embeds")
        mask = condition.data.get("prompt_attention_mask")
        if not all(isinstance(value, torch.Tensor) for value in (prompt_embeds, pooled, mask)):
            raise RuntimeError("Hunyuan conditioning lacks its three explicit encoder tensors.")
        transformer_dtype = getattr(self.pipeline.transformer, "dtype", prompt_embeds.dtype)
        fixed_fingerprint: str | None = None
        if self._pipeline_role == "i2v":
            fixed = state.extra.get("fixed_anchor_latent")
            if not isinstance(fixed, torch.Tensor) or int(fixed.shape[2]) != 1:
                raise RuntimeError("Hunyuan I2V segment lacks its fixed one-token anchor latent.")
            fixed_fingerprint = encoding_fingerprint(fixed)
            if fixed_fingerprint != state.extra.get("fixed_anchor_latent_fingerprint"):
                raise RuntimeError("Hunyuan fixed anchor latent changed between steering branches.")
            if int(latents.shape[2]) != 30:
                raise RuntimeError("Only 30 dynamic I2V latent positions may be runner-visible.")
            model_input = torch.cat([fixed, latents], dim=2).to(transformer_dtype)
        else:
            model_input = latents.to(transformer_dtype)
            if int(model_input.shape[2]) != 31:
                raise RuntimeError("Hunyuan T2V segment must expose exactly 31 latent positions.")
        protected_trace = {
            "fixed_anchor_latent_fingerprint": fixed_fingerprint,
            "runner_visible_dynamic_fingerprint": encoding_fingerprint(latents),
            "transformer_input_fingerprint": encoding_fingerprint(model_input),
            "fixed_temporal_positions": 1 if self._pipeline_role == "i2v" else 0,
            "dynamic_temporal_positions": int(latents.shape[2]),
        }
        state.extra["protected_state_trace"] = protected_trace
        segment_index = int(state.extra.get("segment_index", 0))
        record = condition.data.get("hunyuan_conditioning", {})
        if isinstance(record, Mapping) and self._temporal_protocol() is not None:
            if segment_index >= len(self._segment_conditioning):
                raise RuntimeError(
                    "Hunyuan segmented conditioning state is missing for the active segment."
                )
            records = self._segment_conditioning[segment_index].setdefault("records", {})
            branch_key = str(record.get("dual_view_fingerprint_sha256"))
            records[branch_key] = {
                "entry_id": record.get("entry_id"),
                "role": record.get("role"),
                "pair_id": record.get("pair_id"),
                "anchor_image_sha256": record.get("anchor_image_sha256"),
                "output_fingerprints": record.get("output_fingerprints"),
            }
            protected = self._segment_protected_state[segment_index]
            branches = protected.setdefault("branch_observations", {})
            branches[branch_key] = {
                "entry_id": record.get("entry_id"),
                "role": record.get("role"),
                **protected_trace,
            }
            if self._pipeline_role == "i2v":
                observed_fixed = {
                    branch["fixed_anchor_latent_fingerprint"] for branch in branches.values()
                }
                if observed_fixed != {state.extra["fixed_anchor_latent_fingerprint"]}:
                    raise RuntimeError(
                        "Hunyuan steering branches did not preserve one unbatched fixed anchor."
                    )
        guidance = torch.full(
            (latents.shape[0],), 6000.0, dtype=transformer_dtype, device=self.device
        )
        timestep_batch = self._timestep_batch(timestep, latents.shape[0]).to(latents.dtype)
        cache_context = (
            self.pipeline.transformer.cache_context("cond")
            if hasattr(self.pipeline.transformer, "cache_context")
            else nullcontext()
        )
        with cache_context:
            output = self.pipeline.transformer(
                hidden_states=model_input,
                timestep=timestep_batch,
                encoder_hidden_states=prompt_embeds.to(transformer_dtype),
                encoder_attention_mask=mask,
                pooled_projections=pooled.to(transformer_dtype),
                guidance=guidance,
                attention_kwargs=state.extra.get("attention_kwargs"),
                return_dict=False,
            )
        prediction = _extract_hunyuan_prediction(output).float()
        if self._pipeline_role == "i2v":
            if tuple(prediction.shape) != tuple(model_input.shape):
                raise RuntimeError("Hunyuan I2V transformer prediction shape drifted.")
            prediction = prediction[:, :, 1:]
        if tuple(prediction.shape) != tuple(latents.shape):
            raise RuntimeError(
                "Hunyuan dynamic prediction shape differs from runner-visible state."
            )
        return prediction

    def scheduler_step(
        self,
        model_prediction: torch.Tensor,
        timestep: Any,
        latents: torch.Tensor,
        state: AdapterState,
        generator: torch.Generator | None = None,
    ) -> SchedulerStepResult:
        active_generator = state.extra.get("segment_generator", generator)
        result = super().scheduler_step(
            model_prediction, timestep, latents, state, generator=active_generator
        )
        if self._temporal_protocol() is None:
            return result
        local_step = int(state.extra["segment_step_index"]) + 1
        state.extra["segment_step_index"] = local_step
        if local_step < 50:
            return result
        if local_step > 50:
            raise RuntimeError("Hunyuan segment scheduler advanced beyond 50 steps.")
        segment_index = int(state.extra["segment_index"])
        if segment_index == 2:
            return result
        segment = self._decode_native_segment(result.latents, state)
        self._completed_segments.append(segment)
        anchor = segment[-1]
        if segment_index == 0:
            self._load_continuation_pipeline()
        next_latents = self._start_i2v_segment(
            anchor=anchor, segment_index=segment_index + 1, state=state
        )
        return SchedulerStepResult(latents=next_latents, state=state)

    def decode_latents(self, latents: torch.Tensor, state: AdapterState) -> Any:
        if self._temporal_protocol() is None:
            return [self._decode_native_segment(latents, state)]
        if (
            int(state.extra.get("segment_index", -1)) != 2
            or int(state.extra.get("segment_step_index", -1)) != 50
        ):
            raise RuntimeError("Hunyuan decode requested before all three segments completed.")
        self._completed_segments.append(self._decode_native_segment(latents, state))
        return self._stitch_and_resample_segments(
            self._completed_segments, generation_path="adapter_vector_field_runner"
        )

    def take_temporal_evidence(self, state: AdapterState) -> TemporalEvidenceBundle | None:
        del state
        evidence = self._temporal_evidence
        self._temporal_evidence = None
        return evidence

    def _start_i2v_segment(
        self,
        *,
        anchor: Image.Image,
        segment_index: int,
        state: AdapterState,
    ) -> torch.Tensor:
        if self._pipeline_role != "i2v" or segment_index not in {1, 2}:
            raise RuntimeError("Invalid Hunyuan I2V segment transition.")
        assert self.pipeline is not None
        anchor_sha256 = frame_rgb_sha256(anchor)
        seed = _hunyuan_continuation_seed(int(state.extra["base_seed"]), segment_index)
        generator = _make_hunyuan_generator(seed, self.device)
        image_tensor = self.pipeline.video_processor.preprocess(anchor, 544, 960).to(
            self.device, self.pipeline.vae.dtype
        )
        full_latents, fixed = self.pipeline.prepare_latents(
            image=image_tensor,
            batch_size=1,
            num_channels_latents=16,
            height=544,
            width=960,
            num_frames=121,
            dtype=torch.float32,
            device=self.device,
            generator=generator,
            latents=None,
            image_condition_type="token_replace",
        )
        if tuple(full_latents.shape[:3]) != (1, 16, 31) or tuple(fixed.shape[:3]) != (
            1,
            16,
            1,
        ):
            raise RuntimeError("Hunyuan token-replace latent layout drifted from 31=1+30.")
        dynamic = full_latents[:, :, 1:].contiguous()
        fixed_fingerprint = encoding_fingerprint(fixed)
        self._reset_i2v_scheduler(segment_index)
        state.extra.update(
            {
                "segment_index": segment_index,
                "segment_step_index": 0,
                "condition_epoch": int(state.extra["condition_epoch"]) + 1,
                "segment_seed": seed,
                "segment_generator": generator,
                "model_role": "i2v_continuation",
                "model_id": HUNYUAN_I2V_MODEL_ID,
                "model_revision": HUNYUAN_I2V_CHECKPOINT_REVISION,
                "anchor_image": anchor,
                "anchor_sha256": anchor_sha256,
                "fixed_anchor_latent": fixed,
                "fixed_anchor_latent_fingerprint": fixed_fingerprint,
            }
        )
        self._segment_seeds.append(seed)
        self._segment_conditioning.append({"mode": "state_aware_dual_view", "records": {}})
        self._segment_protected_state.append(
            {
                "fixed_anchor_latent_fingerprint": fixed_fingerprint,
                "anchor_rgb_sha256": anchor_sha256,
                "fixed_temporal_positions": 1,
                "runner_visible_dynamic_positions": 30,
                "steering_scope": "dynamic_positions_1_through_30_only",
                "branch_observations": {},
            }
        )
        return dynamic

    def _decode_native_segment(self, latents: torch.Tensor, state: AdapterState) -> list[Any]:
        assert self.pipeline is not None
        decode_latents = latents
        if self._pipeline_role == "i2v":
            fixed = state.extra.get("fixed_anchor_latent")
            if not isinstance(fixed, torch.Tensor):
                raise RuntimeError("Hunyuan I2V decode is missing its fixed anchor latent.")
            if encoding_fingerprint(fixed) != state.extra.get("fixed_anchor_latent_fingerprint"):
                raise RuntimeError("Hunyuan fixed anchor latent changed before decode.")
            decode_latents = torch.cat([fixed, latents], dim=2)
        if int(decode_latents.shape[2]) != 31:
            raise RuntimeError("Hunyuan native decode requires exactly 31 latent positions.")
        scaled = (
            decode_latents.to(self.pipeline.vae.dtype) / self.pipeline.vae.config.scaling_factor
        )
        video = self.pipeline.vae.decode(scaled, return_dict=False)[0]
        videos = self.pipeline.video_processor.postprocess_video(video, output_type="pil")
        return _validate_hunyuan_video_batch(videos, 121)[0]

    def _stitch_and_resample_segments(
        self,
        segments: Sequence[Sequence[Any]],
        *,
        generation_path: str,
    ) -> list[list[Any]]:
        if len(segments) != 3 or len(self._segment_seeds) != 3:
            raise RuntimeError(
                "Hunyuan composite requires exactly three completed segments and seeds."
            )
        validated = [_validate_hunyuan_video_batch([list(segment)], 121)[0] for segment in segments]
        stitched = [*validated[0], *validated[1][1:], *validated[2][1:]]
        if len(stitched) != 361:
            raise RuntimeError("Hunyuan stitch arithmetic must be exactly 121+120+120=361.")
        output, resampling = resample_hunyuan_native_24_to_16_exact(stitched)
        retained = (tuple(range(121)), tuple(range(1, 121)), tuple(range(1, 121)))
        discarded = ((), (0,), (0,))
        stitch_map = tuple(
            [(0, index) for index in retained[0]]
            + [(1, index) for index in retained[1]]
            + [(2, index) for index in retained[2]]
        )
        segment_evidence = tuple(
            TemporalSegmentEvidence(
                segment_index=index,
                model_role="t2v_primary" if index == 0 else "i2v_continuation",
                model_id=HUNYUAN_T2V_MODEL_ID if index == 0 else HUNYUAN_I2V_MODEL_ID,
                model_revision=(
                    HUNYUAN_CHECKPOINT_REVISION if index == 0 else HUNYUAN_I2V_CHECKPOINT_REVISION
                ),
                segment_seed=self._segment_seeds[index],
                native_fps=24,
                frames=tuple(validated[index]),
                retained_indices=retained[index],
                discarded_indices=discarded[index],
                anchor_sha256=frame_rgb_sha256(validated[index - 1][-1]) if index else None,
                reconstruction_index=0 if index else None,
                first_motion_index=1 if index else None,
                scheduler={
                    "class": "FlowMatchEulerDiscreteScheduler",
                    "shift": 7.0 if index == 0 else 17.0,
                    "num_inference_steps": 50,
                    "embedded_guidance_scale": 6.0,
                    "precision": "bfloat16",
                },
                conditioning=self._normalized_segment_record(self._segment_conditioning, index),
                protected_state=self._normalized_segment_record(
                    self._segment_protected_state, index
                ),
            )
            for index in range(3)
        )
        protocol = self._temporal_protocol()
        assert protocol is not None
        self._temporal_evidence = TemporalEvidenceBundle(
            temporal_protocol=protocol,
            segments=segment_evidence,
            stitch_map=stitch_map,
            output_fps=16,
            output_frame_count=240,
            stitch_strategy=protocol["stitch_strategy"],
            postprocess={
                "method": protocol["resampling_method"],
                "input_frame_count": 361,
                "half_open_input_frame_count": 360,
                "output_frame_count": 240,
                "selected_source_indices": resampling.selected_source_indices,
                "dropped_source_indices": resampling.dropped_source_indices,
                "record": resampling.to_dict(),
            },
            scientific_label=protocol["scientific_label"],
            trace_schema_version=1,
            metadata={
                "generation_path": generation_path,
                "segment_seeds": list(self._segment_seeds),
                "artifact_authentication": self._artifact_authentication,
                "one_way_transition": self._transition_provenance,
            },
        )
        self._last_temporal_provenance = {
            **self._configured_temporal_provenance(protocol),
            "status": "completed",
            "generation_path": generation_path,
            "segment_seeds": list(self._segment_seeds),
            "native_segment_frame_counts": [121, 121, 121],
            "scheduler_shifts": [7.0, 17.0, 17.0],
            "stitched_native_frame_count": 361,
            "output_frame_count": 240,
            "resampling": resampling.to_dict(),
        }
        return [output]

    def _load_continuation_pipeline(self) -> None:
        if self._pipeline_role == "i2v":
            return
        assert self.pipeline is not None
        before = _cuda_memory_record(self.device)
        shared = {
            "vae": self.pipeline.vae,
            "text_encoder_2": self.pipeline.text_encoder_2,
            "tokenizer": self.pipeline.tokenizer,
            "tokenizer_2": self.pipeline.tokenizer_2,
        }
        old_transformer = self.pipeline.transformer
        old_text_encoder = self.pipeline.text_encoder
        if callable(getattr(self.pipeline, "remove_all_hooks", None)):
            self.pipeline.remove_all_hooks()
        self.pipeline.transformer = None
        self.pipeline.text_encoder = None
        del old_transformer, old_text_encoder
        gc.collect()
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
        after_release = _cuda_memory_record(self.device)
        scheduler = self._preloaded_i2v_scheduler or self._new_i2v_scheduler()
        continuation = self._build_continuation_pipeline(shared, scheduler)
        for name, component in shared.items():
            if getattr(continuation, name, None) is not component:
                raise RuntimeError(
                    f"Hunyuan continuation did not preserve shared {name} object identity."
                )
        self.pipeline = continuation
        self._pipeline_role = "i2v"
        self._freeze_pipeline()
        self._validate_components()
        self._validate_i2v_checkpoint_contract()
        self._transition_provenance = {
            "one_way": True,
            "from": f"{HUNYUAN_T2V_MODEL_ID}@{HUNYUAN_CHECKPOINT_REVISION}",
            "to": f"{HUNYUAN_I2V_MODEL_ID}@{HUNYUAN_I2V_CHECKPOINT_REVISION}",
            "released_t2v_transformer_and_llama": True,
            "shared_object_identity": {name: True for name in shared},
            "shared_component_sha256": self._shared_component_hashes(),
            "cuda_memory_before": before,
            "cuda_memory_after_release": after_release,
            "cuda_memory_after_i2v_load": _cuda_memory_record(self.device),
        }

    def _build_continuation_pipeline(self, shared: Mapping[str, Any], scheduler: Any) -> Any:
        import diffusers

        pipeline_cls = getattr(diffusers, "HunyuanVideoImageToVideoPipeline", None)
        if pipeline_cls is None:
            raise RuntimeError("Pinned Diffusers does not expose HunyuanVideoImageToVideoPipeline.")
        kwargs: dict[str, Any] = {
            **dict(shared),
            "scheduler": scheduler,
            "revision": HUNYUAN_I2V_CHECKPOINT_REVISION,
            "torch_dtype": self.dtype,
            "local_files_only": bool(self.config.get("local_files_only", False)),
            "low_cpu_mem_usage": True,
        }
        if self.config.get("cache_dir") is not None:
            kwargs["cache_dir"] = self.config["cache_dir"]
        continuation = pipeline_cls.from_pretrained(HUNYUAN_I2V_MODEL_ID, **kwargs)
        configure_pipeline_vae_tiling(continuation, self.config.get("vae_tiling"))
        cpu_offload = self.config.get("cpu_offload", False)
        strategy = self._cpu_offload_strategy(cpu_offload)
        if strategy == "sequential" and hasattr(continuation, "enable_sequential_cpu_offload"):
            continuation.enable_sequential_cpu_offload(device=self.device)
        elif strategy == "model" and hasattr(continuation, "enable_model_cpu_offload"):
            continuation.enable_model_cpu_offload(device=self.device)
        elif hasattr(continuation, "to"):
            continuation.to(self.device)
        return continuation

    def _new_i2v_scheduler(self) -> Any:
        from diffusers import FlowMatchEulerDiscreteScheduler

        return FlowMatchEulerDiscreteScheduler.from_pretrained(
            HUNYUAN_I2V_MODEL_ID,
            subfolder="scheduler",
            revision=HUNYUAN_I2V_CHECKPOINT_REVISION,
            local_files_only=bool(self.config.get("local_files_only", False)),
        )

    def _reset_i2v_scheduler(self, segment_index: int) -> None:
        if segment_index not in {1, 2}:
            raise ValueError("Hunyuan I2V scheduler reset is valid only for segments 1 and 2.")
        assert self.pipeline is not None
        observed = self._retrieve_timesteps(self.pipeline.scheduler, 50)
        if self._native_segment_timesteps:
            expected = self._native_segment_timesteps[segment_index]
            if not _hunyuan_timesteps_equal(observed, expected):
                raise RuntimeError(
                    "Hunyuan I2V scheduler reset differs from its frozen shift-17 schedule."
                )

    def _retrieve_timesteps(self, scheduler: Any, num_inference_steps: int) -> list[Any]:
        from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import retrieve_timesteps

        sigmas = np.linspace(1.0, 0.0, num_inference_steps + 1)[:-1]
        timesteps, _ = retrieve_timesteps(
            scheduler, num_inference_steps, self.device, sigmas=sigmas
        )
        if hasattr(scheduler, "set_begin_index"):
            scheduler.set_begin_index(0)
        result = list(timesteps)
        if len(result) != num_inference_steps:
            raise RuntimeError("Hunyuan scheduler did not expose the exact requested step count.")
        return result

    def _configured_temporal_provenance(self, protocol: Mapping[str, Any]) -> dict[str, Any]:
        checkpoint_set = [
            {
                "role": "primary",
                "model_id": HUNYUAN_T2V_MODEL_ID,
                "revision": HUNYUAN_CHECKPOINT_REVISION,
                "conversion": "community_diffusers_conversion",
            },
            {
                "role": "continuation",
                "model_id": HUNYUAN_I2V_MODEL_ID,
                "revision": HUNYUAN_I2V_CHECKPOINT_REVISION,
                "conversion": "community_diffusers_conversion",
            },
        ]
        return {
            "schema_version": 2,
            "status": "configured_not_yet_completed",
            "execution_phase": protocol["execution_phase"],
            "strategy": protocol["strategy"],
            "scientific_label": protocol["scientific_label"],
            "protocol_sha256": _canonical_sha256(protocol),
            "checkpoint_set": checkpoint_set,
            "checkpoint_set_sha256": _canonical_sha256(checkpoint_set),
            "artifact_manifest_sha256": protocol["artifact_manifest_sha256"],
            "primary_streaming_receipt_sha256": protocol["primary_streaming_receipt_sha256"],
            "native_generation": {
                "segment_frames": [121, 121, 121],
                "native_fps": 24,
                "scheduler_shifts": [7.0, 17.0, 17.0],
                "steps_per_segment": 50,
                "i2v_state": "one fixed anchor token plus 30 steerable dynamic tokens",
                "stitch": protocol["stitch_strategy"],
            },
            "postprocessing": {
                "method": protocol["resampling_method"],
                "stitched_native_frames": 361,
                "output_frames": 240,
                "output_fps": 16,
                "duration_seconds": 15.0,
                "synthesized_frames": 0,
                "duplicated_frames": 0,
                "slow_motion_used": False,
            },
            "artifact_authentication": self._artifact_authentication,
            "production_gate": protocol.get("production_gate"),
            "obsolete_single_trajectory_route_used": False,
        }

    def _temporal_protocol(self) -> dict[str, Any] | None:
        raw = self.config.get(HUNYUAN_TEMPORAL_PROTOCOL_KEY)
        if raw is None:
            return None
        if not isinstance(raw, Mapping):
            raise ValueError(f"{HUNYUAN_TEMPORAL_PROTOCOL_KEY} must be a mapping.")
        protocol = dict(raw)
        required = set(_EXPECTED_HUNYUAN_TEMPORAL_PROTOCOL) | {"execution_phase"}
        missing = sorted(required - set(protocol))
        if missing:
            raise ValueError(f"{HUNYUAN_TEMPORAL_PROTOCOL_KEY} is missing fields: {missing}.")
        unknown = sorted(
            set(protocol)
            - set(_EXPECTED_HUNYUAN_TEMPORAL_PROTOCOL)
            - _HUNYUAN_TEMPORAL_DYNAMIC_FIELDS
        )
        if unknown:
            raise ValueError(f"{HUNYUAN_TEMPORAL_PROTOCOL_KEY} has unknown fields: {unknown}.")
        mismatches = {
            key: {"expected": expected, "actual": protocol[key]}
            for key, expected in _EXPECTED_HUNYUAN_TEMPORAL_PROTOCOL.items()
            if protocol[key] != expected
        }
        if mismatches:
            raise ValueError(f"Hunyuan temporal schema-2 contract drifted: {mismatches}.")
        if protocol["execution_phase"] not in {"pilot", "production"}:
            raise ValueError("Hunyuan temporal execution_phase must be pilot or production.")
        return protocol

    def _validate_temporal_execution_gate(self, protocol: Mapping[str, Any]) -> None:
        validate_temporal_production_gate(
            protocol,
            model_name="hunyuan_video",
            model_revision=HUNYUAN_CHECKPOINT_REVISION,
            criteria_names=_HUNYUAN_PRODUCTION_CRITERIA,
        )

    def _validate_requested_temporal_output(
        self,
        generation_kwargs: Mapping[str, Any],
        protocol: Mapping[str, Any],
    ) -> None:
        actual = {
            "num_frames": int(generation_kwargs.get("num_frames", -1)),
            "fps": int(generation_kwargs.get("fps", -1)),
            "duration_seconds": float(generation_kwargs.get("duration_seconds", -1.0)),
            "height": int(generation_kwargs.get("height", -1)),
            "width": int(generation_kwargs.get("width", -1)),
        }
        expected = {
            "num_frames": int(protocol["output_frames"]),
            "fps": int(protocol["output_fps"]),
            "duration_seconds": float(protocol["duration_seconds"]),
            "height": int(protocol["height"]),
            "width": int(protocol["width"]),
        }
        if actual != expected:
            raise ValueError(
                f"Hunyuan segmented route requires exact output contract {expected}; got {actual}."
            )

    def _validate_preload_contract(self, protocol: Mapping[str, Any]) -> None:
        if self.model_id != HUNYUAN_T2V_MODEL_ID:
            raise RuntimeError(
                f"Hunyuan primary repository must be {HUNYUAN_T2V_MODEL_ID!r}; got {self.model_id!r}."
            )
        if self.config.get("revision") != HUNYUAN_CHECKPOINT_REVISION:
            raise RuntimeError(
                "Hunyuan primary checkpoint revision is not the complete frozen commit."
            )
        if protocol["i2v_model_id"] != HUNYUAN_I2V_MODEL_ID:
            raise RuntimeError("Hunyuan continuation repository alias is forbidden.")
        if _installed_diffusers_revision() != HUNYUAN_DIFFUSERS_REVISION:
            raise RuntimeError(
                "Hunyuan segmented route requires the exact pinned Diffusers commit."
            )

    def _validate_primary_checkpoint_contract(self) -> None:
        self._require_loaded()
        assert self.pipeline is not None
        protocol = self._temporal_protocol()
        if protocol is None:
            raise RuntimeError("Hunyuan primary composite validation requires protocol schema 2.")
        self._validate_preload_contract(protocol)
        config = self.pipeline.transformer.config
        actual = {
            "in_channels": int(getattr(config, "in_channels", -1)),
            "out_channels": int(getattr(config, "out_channels", -1)),
            "vae_temporal": int(getattr(self.pipeline, "vae_scale_factor_temporal", -1)),
            "vae_spatial": int(getattr(self.pipeline, "vae_scale_factor_spatial", -1)),
        }
        if actual != {"in_channels": 16, "out_channels": 16, "vae_temporal": 4, "vae_spatial": 8}:
            raise RuntimeError(f"Hunyuan T2V architecture drifted: {actual}.")
        self._require_scheduler_shift(self.pipeline.scheduler, 7.0, "T2V")

    def _validate_i2v_checkpoint_contract(self) -> None:
        assert self.pipeline is not None
        config = self.pipeline.transformer.config
        actual = {
            "in_channels": int(getattr(config, "in_channels", -1)),
            "out_channels": int(getattr(config, "out_channels", -1)),
            "image_condition_type": getattr(config, "image_condition_type", None),
            "vae_temporal": int(getattr(self.pipeline, "vae_scale_factor_temporal", -1)),
            "vae_spatial": int(getattr(self.pipeline, "vae_scale_factor_spatial", -1)),
        }
        expected = {
            "in_channels": 16,
            "out_channels": 16,
            "image_condition_type": "token_replace",
            "vae_temporal": 4,
            "vae_spatial": 8,
        }
        if actual != expected:
            raise RuntimeError(f"Hunyuan I2V token-replace architecture drifted: {actual}.")
        self._require_scheduler_shift(self.pipeline.scheduler, 17.0, "I2V")

    @staticmethod
    def _require_scheduler_shift(scheduler: Any, expected: float, role: str) -> None:
        config = getattr(scheduler, "config", {})
        actual = config.get("shift") if hasattr(config, "get") else getattr(config, "shift", None)
        if float(actual) != expected:
            raise RuntimeError(
                f"Hunyuan {role} scheduler shift must be {expected}; got {actual!r}."
            )

    def _authenticate_artifact_manifest(self, protocol: Mapping[str, Any]) -> dict[str, Any]:
        root = Path(__file__).resolve().parents[3]
        path = (root / str(protocol["artifact_manifest"])).resolve()
        if not path.is_file() or _sha256_file(path) != protocol["artifact_manifest_sha256"]:
            raise RuntimeError("Hunyuan checkpoint-set artifact manifest is missing or changed.")
        payload = json.loads(path.read_text(encoding="utf-8"))
        top_level_identity = {
            "schema_version": payload.get("schema_version"),
            "scientific_identity": payload.get("scientific_identity"),
            "checkpoint_count": payload.get("checkpoint_count"),
            "file_count": payload.get("file_count"),
            "referenced_bytes": payload.get("referenced_bytes"),
        }
        expected_top_level_identity = {
            "schema_version": 1,
            "scientific_identity": (
                "hunyuan_video_community_diffusers_t2v_to_i2v_segmented_temporal_v2"
            ),
            "checkpoint_count": 2,
            "file_count": 58,
            "referenced_bytes": 85_549_596_031,
        }
        if top_level_identity != expected_top_level_identity:
            raise RuntimeError("Hunyuan checkpoint-set artifact manifest identity drifted.")
        receipt_binding = payload.get("primary_streaming_receipt")
        expected_receipt_binding = {
            "path": protocol["primary_streaming_receipt_path"],
            "sha256": protocol["primary_streaming_receipt_sha256"],
        }
        if receipt_binding != expected_receipt_binding:
            raise RuntimeError("Hunyuan primary streaming receipt binding drifted.")
        receipt_path = (root / str(receipt_binding["path"])).resolve()
        if not receipt_path.is_file() or _sha256_file(receipt_path) != receipt_binding["sha256"]:
            raise RuntimeError("Hunyuan primary streaming receipt is missing or changed.")
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt_identity = {
            key: receipt.get(key)
            for key in (
                "schema_version",
                "status",
                "repository",
                "revision",
                "pipeline_class",
                "file_count",
                "referenced_bytes",
                "inventory_sha256",
                "checkpoint_identity_sha256",
                "checkpoint_set_identity_sha256",
            )
        }
        expected_receipt_identity = {
            "schema_version": 1,
            "status": "PASS",
            "repository": HUNYUAN_T2V_MODEL_ID,
            "revision": HUNYUAN_CHECKPOINT_REVISION,
            "pipeline_class": "HunyuanVideoPipeline",
            "file_count": 28,
            "referenced_bytes": 41_903_676_146,
            "inventory_sha256": (
                "312639cf88488e2ceccb8e505c291b920b1c13ca585125e17951e8928c291408"
            ),
            "checkpoint_identity_sha256": (
                "b16b5bc4c468eda13ff5804832638c009187b18ca38658643588048cbf9689b8"
            ),
            "checkpoint_set_identity_sha256": (
                "103e35989b942e3dc4c2e7f1c2f741bfce1f9abac81b590b4ab8b35b1eb0a95a"
            ),
        }
        if receipt_identity != expected_receipt_identity:
            raise RuntimeError("Hunyuan primary streaming receipt identity drifted.")
        source = payload.get("authentication_source")
        if not isinstance(source, Mapping) or set(source) != {"path", "sha256"}:
            raise RuntimeError("Hunyuan checkpoint manifest lacks its sealed audit binding.")
        source_path = (root / str(source["path"])).resolve()
        if (
            not source_path.is_file()
            or source["sha256"]
            != "adb6036e8f962c79c975375de8e3c81ce2ae0e2e801f77cd1b6cd2dc320a221d"
            or _sha256_file(source_path) != source["sha256"]
        ):
            raise RuntimeError("Hunyuan sealed checkpoint-authentication audit changed.")

        expected_checkpoints = (
            {
                "role": "primary",
                "model_id": HUNYUAN_T2V_MODEL_ID,
                "revision": HUNYUAN_CHECKPOINT_REVISION,
                "pipeline_class": "HunyuanVideoPipeline",
                "file_count": 28,
                "referenced_bytes": 41_903_676_146,
            },
            {
                "role": "continuation",
                "model_id": HUNYUAN_I2V_MODEL_ID,
                "revision": HUNYUAN_I2V_CHECKPOINT_REVISION,
                "pipeline_class": "HunyuanVideoImageToVideoPipeline",
                "file_count": 30,
                "referenced_bytes": 43_645_919_885,
            },
        )
        checkpoints = payload.get("checkpoints")
        if not isinstance(checkpoints, list) or len(checkpoints) != 2:
            raise RuntimeError("Hunyuan artifact manifest checkpoint set is incomplete.")
        if receipt.get("files") != checkpoints[0].get("files"):
            raise RuntimeError(
                "Hunyuan primary receipt and checkpoint manifest inventories differ."
            )
        shared_hashes = self._shared_component_hashes()
        checkpoint_file_maps = [
            {
                str(entry["path"]): str(entry["sha256"])
                for entry in checkpoint.get("files", [])
                if isinstance(entry, Mapping) and "path" in entry and "sha256" in entry
            }
            for checkpoint in checkpoints
        ]
        for shared_path, expected_sha256 in shared_hashes.items():
            observed = [file_map.get(shared_path) for file_map in checkpoint_file_maps]
            if observed != [expected_sha256, expected_sha256]:
                raise RuntimeError(
                    "Hunyuan shared component bytes differ between T2V and I2V: "
                    f"{shared_path!r} -> {observed}."
                )
        from huggingface_hub import snapshot_download

        verified_checkpoints: list[dict[str, Any]] = []
        total_files = 0
        total_bytes = 0
        for checkpoint, expected in zip(checkpoints, expected_checkpoints):
            if not isinstance(checkpoint, Mapping):
                raise RuntimeError("Hunyuan checkpoint manifest entry is not a mapping.")
            observed_identity = {
                key: checkpoint.get(key)
                for key in (
                    "role",
                    "model_id",
                    "revision",
                    "pipeline_class",
                    "file_count",
                    "referenced_bytes",
                )
            }
            if observed_identity != expected:
                raise RuntimeError(
                    f"Hunyuan {expected['role']} checkpoint manifest identity drifted."
                )
            snapshot = Path(
                snapshot_download(
                    str(expected["model_id"]),
                    revision=str(expected["revision"]),
                    local_files_only=True,
                )
            ).resolve()
            files = checkpoint.get("files")
            if not isinstance(files, list) or len(files) != expected["file_count"]:
                raise RuntimeError(
                    f"Hunyuan {expected['role']} manifest has no complete file inventory."
                )
            declared_paths: set[str] = set()
            referenced_bytes = 0
            verified = 0
            for entry in files:
                if not isinstance(entry, Mapping) or set(entry) != {
                    "path",
                    "size_bytes",
                    "sha256",
                }:
                    raise RuntimeError("Hunyuan artifact file entry is malformed.")
                relative_text = str(entry["path"])
                relative = Path(relative_text)
                if (
                    not relative_text
                    or relative.is_absolute()
                    or ".." in relative.parts
                    or relative.as_posix() != relative_text
                    or relative_text in declared_paths
                ):
                    raise RuntimeError(
                        f"Hunyuan manifest path is unsafe or duplicated: {relative_text!r}."
                    )
                declared_paths.add(relative_text)
                declared_bytes = entry["size_bytes"]
                declared_sha256 = entry["sha256"]
                if (
                    isinstance(declared_bytes, bool)
                    or not isinstance(declared_bytes, int)
                    or declared_bytes < 0
                    or not isinstance(declared_sha256, str)
                    or re.fullmatch(r"[0-9a-f]{64}", declared_sha256) is None
                ):
                    raise RuntimeError(
                        f"Hunyuan manifest file entry is malformed: {relative_text!r}."
                    )
                candidate = snapshot / relative
                try:
                    resolved_target = candidate.resolve(strict=True)
                except OSError as exc:
                    raise RuntimeError(f"Hunyuan artifact is missing: {candidate}.") from exc
                if not candidate.is_file() or not resolved_target.is_file():
                    raise RuntimeError(f"Hunyuan artifact is not a regular file: {candidate}.")
                if (
                    candidate.stat().st_size != declared_bytes
                    or _sha256_file(candidate) != declared_sha256
                ):
                    raise RuntimeError(f"Hunyuan artifact bytes changed: {relative_text!r}.")
                referenced_bytes += declared_bytes
                verified += 1
            observed_paths = {
                candidate.relative_to(snapshot).as_posix()
                for candidate in snapshot.rglob("*")
                if candidate.is_file()
            }
            if observed_paths != declared_paths:
                raise RuntimeError(
                    "Hunyuan manifest is not the complete resolved snapshot inventory for "
                    f"{expected['role']}: missing={sorted(observed_paths - declared_paths)}, "
                    f"extra={sorted(declared_paths - observed_paths)}."
                )
            if (
                verified != expected["file_count"]
                or referenced_bytes != expected["referenced_bytes"]
            ):
                raise RuntimeError(f"Hunyuan {expected['role']} artifact totals are inconsistent.")
            total_files += verified
            total_bytes += referenced_bytes
            verified_checkpoints.append(
                {
                    **expected,
                    "snapshot_path": str(snapshot),
                    "verified_file_count": verified,
                    "verified_referenced_bytes": referenced_bytes,
                    "complete_snapshot_inventory_match": True,
                }
            )
        if total_files != payload["file_count"] or total_bytes != payload["referenced_bytes"]:
            raise RuntimeError("Hunyuan checkpoint-set artifact totals are inconsistent.")
        checkpoint_set = [
            {
                "role": checkpoint["role"],
                "model_id": checkpoint["model_id"],
                "revision": checkpoint["revision"],
                "conversion": "community_diffusers_conversion",
            }
            for checkpoint in verified_checkpoints
        ]
        return {
            "schema_version": 1,
            "manifest_path": str(path),
            "manifest_sha256": protocol["artifact_manifest_sha256"],
            "checkpoint_set": checkpoint_set,
            "checkpoint_set_sha256": _canonical_sha256(checkpoint_set),
            "verified_checkpoints": verified_checkpoints,
            "verified_file_count": total_files,
            "verified_referenced_bytes": total_bytes,
            "sealed_authentication_source_sha256": source["sha256"],
            "primary_streaming_receipt_path": str(receipt_path),
            "primary_streaming_receipt_sha256": receipt_binding["sha256"],
            "primary_inventory_sha256": receipt["inventory_sha256"],
            "shared_component_sha256": shared_hashes,
            "shared_component_bytes_equal": True,
            "streaming_cache_policy": "none_full_stream_every_job",
            "all_files_rehashed": True,
        }

    def _record_encoding(
        self,
        encoding: HunyuanPromptEncoding,
        *,
        route: str,
        anchor_sha256: str | None = None,
    ) -> None:
        for record in encoding.records:
            enriched = {
                **record,
                "model_role": route,
                "anchor_image_sha256": anchor_sha256 or record.get("anchor_image_sha256"),
            }
            self._store_conditioning_record(enriched)

    def _store_conditioning_record(self, record: Mapping[str, Any]) -> None:
        key = _canonical_sha256(
            {
                "dual_view": record.get("dual_view_fingerprint_sha256"),
                "encoder_route": record.get("encoder_route"),
                "anchor": record.get("anchor_image_sha256"),
                "state": record.get("state_identity_sha256"),
            }
        )
        value = dict(record)
        prior = self._conditioning_records.get(key)
        if prior is not None and prior != value:
            raise RuntimeError("Hunyuan state-aware conditioning provenance collided.")
        self._conditioning_records[key] = value

    def _max_sequence_length(self) -> int:
        value = self.config.get("max_sequence_length", 256)
        return 256 if value is None else int(value)

    def _prompt_template(self) -> Mapping[str, Any] | None:
        value = self.config.get("prompt_template")
        if value is not None and not isinstance(value, Mapping):
            raise ValueError("Hunyuan prompt_template must be a mapping.")
        return value

    def _shared_component_hashes(self) -> dict[str, str]:
        protocol = self._temporal_protocol()
        assert protocol is not None
        return {
            "vae/diffusion_pytorch_model.safetensors": (
                "7c68a6295f9034a88225fbafb1f3258291a08d57a1fdb938233fa57b1b8f4883"
            ),
            "text_encoder_2/model.safetensors": (
                "660c6f5b1abae9dc498ac2d21e1347d2abdb0cf6c0c0c8576cd796491d9a6cdd"
            ),
            "tokenizer/special_tokens_map.json": (
                "e87284810dc7a4d9dbab5ea6b713b3e72a6a909006483760773fb129534b9fe0"
            ),
            "tokenizer/tokenizer.json": (
                "d2c593db4aa75b17a42c1f74d7cc38e257eaeed222e6a52674c65544165dcbaa"
            ),
            "tokenizer/tokenizer_config.json": (
                "88d8723716a3368430f3a02fd4614c8368b9f87b28c0053fcbdd80b88888b0f7"
            ),
            "tokenizer_2/merges.txt": (
                "9fd691f7c8039210e0fced15865466c65820d09b63988b0174bfe25de299051a"
            ),
            "tokenizer_2/special_tokens_map.json": (
                "2cdb3b8331a60c92fc1e55a13e9fd61fd2293c5a51275fdcccd62b780052530e"
            ),
            "tokenizer_2/tokenizer_config.json": (
                "39bae5313c2da6467866dee758e754d69f0e302a875027b142531cb9c51c7c28"
            ),
            "tokenizer_2/vocab.json": (
                "e089ad92ba36837a0d31433e555c8f45fe601ab5c221d4f607ded32d9f7a4349"
            ),
        }

    @staticmethod
    def _normalized_segment_record(
        records: Sequence[Mapping[str, Any]], index: int
    ) -> dict[str, Any]:
        if index >= len(records):
            return {}
        value = dict(records[index])
        nested = value.get("records")
        if isinstance(nested, Mapping):
            value["records"] = [nested[key] for key in sorted(nested)]
        return value


def _validate_hunyuan_i2v_encoding_shapes(
    pipeline: Any,
    prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    prompt_attention_mask: Any,
    *,
    batch_size: int,
) -> None:
    """Validate the pinned token-replace I2V LLaVA/CLIP tensor contract.

    The community I2V pipeline retains 252 text positions after its exact
    assistant-marker crop and prepends 576/4=144 real-image positions.  Thus
    the registered ``image_embed_interleave=4`` route has exactly 396 LLaVA
    positions.  Accepting an arbitrary sequence length here would make a
    changed private-helper implementation scientifically indistinguishable
    from the sealed route.
    """

    tensors = {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "prompt_attention_mask": prompt_attention_mask,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                f"HunyuanVideo I2V {name} must be a tensor, got {type(tensor).__name__}."
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"HunyuanVideo I2V {name} contains non-finite values.")

    transformer = getattr(pipeline, "transformer", None)
    transformer_config = getattr(transformer, "config", None)
    text_embed_dim = _config_value(transformer_config, "text_embed_dim")
    pooled_projection_dim = _config_value(transformer_config, "pooled_projection_dim")
    image_condition_type = _config_value(transformer_config, "image_condition_type")
    if (text_embed_dim, pooled_projection_dim, image_condition_type) != (
        HUNYUAN_LLAMA_HIDDEN_SIZE,
        HUNYUAN_CLIP_POOLED_SIZE,
        "token_replace",
    ):
        raise RuntimeError(
            "HunyuanVideo I2V transformer conditioning architecture drifted: "
            f"text_embed_dim={text_embed_dim!r}, "
            f"pooled_projection_dim={pooled_projection_dim!r}, "
            f"image_condition_type={image_condition_type!r}."
        )
    expected_shapes = {
        "prompt_embeds": (batch_size, 396, HUNYUAN_LLAMA_HIDDEN_SIZE),
        "pooled_prompt_embeds": (batch_size, HUNYUAN_CLIP_POOLED_SIZE),
        "prompt_attention_mask": (batch_size, 396),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(tensors[name].shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"HunyuanVideo I2V {name} shape drift: "
                f"expected {expected_shape}, got {actual_shape}."
            )
    mask_values = torch.unique(prompt_attention_mask.detach())
    if any(int(value) not in {0, 1} for value in mask_values.cpu().tolist()):
        raise RuntimeError("HunyuanVideo I2V prompt_attention_mask must contain only zero and one.")


def _validate_native_first_call_kwargs(kwargs: Mapping[str, Any]) -> None:
    """Fail closed if the native T2V first call was weaker than schema 2."""

    forbidden = sorted({"prompt", "prompt_2", "negative_prompt", "negative_prompt_2"} & set(kwargs))
    if forbidden:
        raise RuntimeError(f"Hunyuan native T2V first call contains raw prompts: {forbidden}.")
    embedding_keys = {
        "prompt_embeds",
        "pooled_prompt_embeds",
        "prompt_attention_mask",
        "negative_prompt_embeds",
        "negative_pooled_prompt_embeds",
        "negative_prompt_attention_mask",
    }
    missing = sorted(embedding_keys - set(kwargs))
    if missing:
        raise RuntimeError(
            f"Hunyuan native T2V first call is missing explicit conditioning: {missing}."
        )
    for key in sorted(embedding_keys):
        tensor = kwargs[key]
        if not isinstance(tensor, torch.Tensor) or tensor.shape[0] != 1:
            raise RuntimeError(f"Hunyuan native T2V {key} must be a one-row tensor.")
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"Hunyuan native T2V {key} contains non-finite values.")
    expected = {
        "height": 544,
        "width": 960,
        "num_frames": 121,
        "num_inference_steps": 50,
        "guidance_scale": 6.0,
        "true_cfg_scale": 4.0,
        "output_type": "pil",
        "return_dict": True,
    }
    mismatches = {
        key: {"expected": value, "actual": kwargs.get(key)}
        for key, value in expected.items()
        if kwargs.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Hunyuan native T2V first-call contract drifted: {mismatches}.")


def _validate_hunyuan_video_batch(videos: Any, expected_frames: int) -> list[list[Image.Image]]:
    """Normalize and validate a complete RGB PIL video batch without synthesis."""

    if not isinstance(videos, list) or not videos:
        raise RuntimeError("Hunyuan decoded media must be a non-empty list of videos.")
    normalized: list[list[Image.Image]] = []
    expected_size: tuple[int, int] | None = None
    for video_index, frames in enumerate(videos):
        if not isinstance(frames, list) or len(frames) != expected_frames:
            observed = len(frames) if isinstance(frames, list) else None
            raise RuntimeError(
                f"Hunyuan video {video_index} must contain exactly {expected_frames} frames; "
                f"got {observed}."
            )
        normalized_frames: list[Image.Image] = []
        for frame_index, frame in enumerate(frames):
            if not isinstance(frame, Image.Image):
                raise RuntimeError(
                    f"Hunyuan frame {video_index}:{frame_index} must be a PIL image."
                )
            rgb = frame if frame.mode == "RGB" else frame.convert("RGB")
            array = np.asarray(rgb)
            if array.dtype != np.uint8 or array.ndim != 3 or array.shape[2] != 3:
                raise RuntimeError(
                    f"Hunyuan frame {video_index}:{frame_index} is not canonical RGB uint8."
                )
            if expected_size is None:
                expected_size = rgb.size
            elif rgb.size != expected_size:
                raise RuntimeError("Hunyuan decoded frames do not share one spatial resolution.")
            normalized_frames.append(rgb)
        normalized.append(normalized_frames)
    return normalized


def _hunyuan_continuation_seed(base_seed: int, segment_index: int) -> int:
    if isinstance(base_seed, bool) or not isinstance(base_seed, int) or base_seed < 0:
        raise ValueError("Hunyuan base seed must be a non-negative integer.")
    if segment_index not in {1, 2}:
        raise ValueError("Hunyuan continuation seed is defined only for segments 1 and 2.")
    payload = f"{HUNYUAN_SEGMENT_SEED_DOMAIN}|{base_seed}|{segment_index}".encode()
    # torch.Generator.manual_seed accepts signed 64-bit seeds portably.  Keep
    # zero available but make the domain-separated continuation deterministic.
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**63 - 1)


def _make_hunyuan_generator(seed: int, device: torch.device) -> torch.Generator:
    generator_device = device if device.type == "cuda" else torch.device("cpu")
    return torch.Generator(device=generator_device).manual_seed(seed)


def _extract_hunyuan_pipeline_media(output: Any) -> Any:
    media = getattr(output, "frames", None)
    if media is None and isinstance(output, tuple) and output:
        media = output[0]
    if media is None:
        raise RuntimeError("Hunyuan pipeline output does not expose decoded frames.")
    return media


def _extract_hunyuan_prediction(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    sample = getattr(output, "sample", None)
    if isinstance(sample, torch.Tensor):
        return sample
    raise RuntimeError("Hunyuan transformer did not return a tensor prediction.")


def _cuda_memory_record(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda" or not torch.cuda.is_available():
        return {
            "available": False,
            "device": str(device),
            "allocated_bytes": 0,
            "reserved_bytes": 0,
        }
    torch.cuda.synchronize(device)
    return {
        "available": True,
        "device": str(device),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
    }


def _hunyuan_timesteps_equal(first: Sequence[Any], second: Sequence[Any]) -> bool:
    if len(first) != len(second):
        return False
    for left, right in zip(first, second):
        if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
            try:
                left_tensor = torch.as_tensor(left).detach().cpu()
                right_tensor = torch.as_tensor(right).detach().cpu()
            except (TypeError, ValueError):
                return False
            if left_tensor.dtype != right_tensor.dtype or not torch.equal(
                left_tensor, right_tensor
            ):
                return False
        elif left != right:
            return False
    return True


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _installed_diffusers_revision() -> str | None:
    try:
        distribution = metadata.distribution("diffusers")
    except metadata.PackageNotFoundError:
        return None
    direct_url = distribution.read_text("direct_url.json")
    if not direct_url:
        return None
    try:
        payload = json.loads(direct_url)
    except json.JSONDecodeError:
        return None
    revision = payload.get("vcs_info", {}).get("commit_id")
    return str(revision) if revision else None


def _require_nonempty_text(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Hunyuan conditioning-plan field '{field_name}' must be non-empty text.")
    return value


def _require_sha256(value: Any, field_name: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(
            f"Hunyuan conditioning-plan field '{field_name}' must be a lowercase SHA-256 digest."
        )
    return value


def _optional_token_fingerprint(
    value: Mapping[str, Any],
    *,
    prefix: str,
    entry_id: str,
) -> tuple[int | None, str | None]:
    count_value = value.get(f"{prefix}_token_count")
    sha_value = value.get(f"{prefix}_token_ids_sha256")
    if (count_value is None) != (sha_value is None):
        raise ValueError(
            f"Hunyuan conditioning-plan entry '{entry_id}' must provide both "
            f"{prefix}_token_count and {prefix}_token_ids_sha256, or neither."
        )
    if count_value is None:
        return None, None
    if isinstance(count_value, bool) or not isinstance(count_value, int) or count_value <= 0:
        raise ValueError(
            f"Hunyuan conditioning-plan entry '{entry_id}' {prefix}_token_count "
            "must be a positive integer."
        )
    return count_value, _require_sha256(sha_value, f"{entry_id}.{prefix}_token_ids_sha256")


def _duplicates(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    return sorted(duplicates)


def _validate_plan_pairs(entries: Sequence[HunyuanPromptView]) -> None:
    paired: dict[str, dict[str, HunyuanPromptView]] = {}
    for entry in entries:
        normalized_role = entry.role.lower()
        if normalized_role not in {"unsafe", "safe"}:
            if entry.pair_id is not None:
                raise ValueError(
                    f"Hunyuan conditioning-plan role '{entry.role}' entry '{entry.entry_id}' "
                    "must not declare pair_id."
                )
            continue
        if entry.pair_id is None:
            raise ValueError(
                f"Hunyuan conditioning-plan {entry.role} entry '{entry.entry_id}' requires pair_id."
            )
        role_map = paired.setdefault(entry.pair_id, {})
        if normalized_role in role_map:
            raise ValueError(
                f"Hunyuan conditioning-plan pair '{entry.pair_id}' has multiple "
                f"{normalized_role} entries."
            )
        role_map[normalized_role] = entry
    for pair_id, role_map in paired.items():
        unsafe = role_map.get("unsafe")
        safe = role_map.get("safe")
        missing_roles = sorted({"unsafe", "safe"} - role_map.keys())
        if missing_roles:
            raise ValueError(
                f"Hunyuan conditioning-plan pair '{pair_id}' is incomplete; "
                f"missing roles {missing_roles}."
            )
        assert unsafe is not None and safe is not None
        if unsafe.clip_prompt == safe.clip_prompt:
            raise ValueError(
                f"Hunyuan conditioning-plan pair '{pair_id}' gives unsafe and safe CLIP "
                "the same prompt; concept steering would be unidentifiable."
            )


def _prompt_view_provenance(view: HunyuanPromptView) -> dict[str, Any]:
    return {
        "entry_id": view.entry_id,
        "role": view.role,
        "pair_id": view.pair_id,
        "raw_prompt": view.raw_prompt,
        "raw_prompt_sha256": view.raw_prompt_sha256,
        "llama_prompt": view.llama_prompt,
        "llama_prompt_sha256": _sha256_text(view.llama_prompt),
        "clip_prompt": view.clip_prompt,
        "clip_prompt_sha256": _sha256_text(view.clip_prompt),
        "frozen_llama_token_count": view.llama_token_count,
        "frozen_llama_token_ids_sha256": view.llama_token_ids_sha256,
        "frozen_clip_token_count": view.clip_token_count,
        "frozen_clip_token_ids_sha256": view.clip_token_ids_sha256,
    }


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return _sha256_text(canonical)


def _resolve_prompt_template(
    pipeline: Any,
    prompt_template: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if prompt_template is None:
        try:
            parameter = inspect.signature(pipeline.encode_prompt).parameters["prompt_template"]
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(
                "Cannot recover HunyuanVideo encode_prompt's frozen prompt template."
            ) from exc
        if parameter.default is inspect.Parameter.empty or not isinstance(
            parameter.default, Mapping
        ):
            raise RuntimeError(
                "HunyuanVideo encode_prompt no longer exposes a mapping prompt_template default."
            )
        resolved = dict(parameter.default)
    else:
        resolved = dict(prompt_template)
    template_text = resolved.get("template")
    if not isinstance(template_text, str) or not template_text:
        raise ValueError("Hunyuan prompt_template requires a non-empty 'template' string.")
    sentinel = "HUNYUAN_PROMPT_VIEW_SENTINEL"
    try:
        rendered = template_text.format(sentinel)
    except (IndexError, KeyError, ValueError) as exc:
        raise ValueError("Hunyuan prompt_template must accept one positional prompt.") from exc
    if sentinel not in rendered:
        raise ValueError("Hunyuan prompt_template must preserve its positional prompt.")
    return resolved


def _resolve_crop_start(tokenizer: Any, prompt_template: Mapping[str, Any]) -> int:
    crop_start = prompt_template.get("crop_start")
    if crop_start is not None:
        if isinstance(crop_start, bool) or not isinstance(crop_start, int) or crop_start < 0:
            raise ValueError("Hunyuan prompt_template crop_start must be a non-negative integer.")
        return crop_start
    template_inputs = tokenizer(
        prompt_template["template"],
        padding="max_length",
        return_tensors="pt",
        return_length=False,
        return_overflowing_tokens=False,
        return_attention_mask=False,
    )
    template_ids = _extract_single_token_ids(template_inputs, encoder_name="Llama template")
    derived = len(template_ids) - 2
    if derived < 0:
        raise RuntimeError(
            "Hunyuan Llama prompt-template tokenization returned fewer than two tokens."
        )
    return derived


def _tokenize_without_truncation(
    tokenizer: Any, text: str, *, encoder_name: str
) -> tuple[int, ...]:
    encoded = tokenizer(
        text,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        add_special_tokens=True,
    )
    return _extract_single_token_ids(encoded, encoder_name=encoder_name)


def _extract_single_token_ids(encoded: Any, *, encoder_name: str) -> tuple[int, ...]:
    if isinstance(encoded, Mapping):
        token_ids = encoded.get("input_ids")
    else:
        token_ids = getattr(encoded, "input_ids", None)
    if token_ids is None:
        raise RuntimeError(f"Hunyuan {encoder_name} tokenizer did not return input_ids.")
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim == 2 and token_ids.shape[0] == 1:
            token_ids = token_ids[0]
        if token_ids.ndim != 1:
            raise RuntimeError(
                f"Hunyuan {encoder_name} preflight expected one token sequence, "
                f"got tensor shape {tuple(token_ids.shape)}."
            )
        return tuple(int(token) for token in token_ids.detach().cpu().tolist())
    if isinstance(token_ids, (list, tuple)):
        if len(token_ids) == 1 and isinstance(token_ids[0], (list, tuple)):
            token_ids = token_ids[0]
        if not all(isinstance(token, int) and not isinstance(token, bool) for token in token_ids):
            raise RuntimeError(f"Hunyuan {encoder_name} tokenizer returned invalid token IDs.")
        return tuple(token_ids)
    raise RuntimeError(
        f"Hunyuan {encoder_name} tokenizer returned unsupported input_ids type "
        f"{type(token_ids).__name__}."
    )


def _token_ids_sha256(token_ids: Sequence[int]) -> str:
    return _sha256_text(json.dumps(list(token_ids), separators=(",", ":")))


def _preflight_prompt_view(
    pipeline: Any,
    view: HunyuanPromptView,
    *,
    prompt_template: Mapping[str, Any],
    prompt_template_sha256: str,
    crop_start: int,
    llama_max_sequence_length: int,
    clip_max_sequence_length: int,
    plan_sha256: str | None,
) -> dict[str, Any]:
    formatted_llama_prompt = str(prompt_template["template"]).format(view.llama_prompt)
    llama_token_ids = _tokenize_without_truncation(
        pipeline.tokenizer,
        formatted_llama_prompt,
        encoder_name="Llama",
    )
    clip_token_ids = _tokenize_without_truncation(
        pipeline.tokenizer_2,
        view.clip_prompt,
        encoder_name="CLIP",
    )
    llama_limit = llama_max_sequence_length + crop_start
    if len(llama_token_ids) > llama_limit:
        raise ValueError(
            f"Hunyuan Llama view '{view.entry_id}' has {len(llama_token_ids)} formatted tokens, "
            f"exceeding the no-truncation limit {llama_limit} "
            f"({llama_max_sequence_length}+crop_start {crop_start})."
        )
    if len(clip_token_ids) > clip_max_sequence_length:
        raise ValueError(
            f"Hunyuan CLIP view '{view.entry_id}' has {len(clip_token_ids)} tokens, "
            f"exceeding the no-truncation limit {clip_max_sequence_length}."
        )
    llama_ids_sha256 = _token_ids_sha256(llama_token_ids)
    clip_ids_sha256 = _token_ids_sha256(clip_token_ids)
    _verify_frozen_token_fingerprint(
        view,
        encoder_name="llama",
        actual_count=len(llama_token_ids),
        actual_sha256=llama_ids_sha256,
    )
    _verify_frozen_token_fingerprint(
        view,
        encoder_name="clip",
        actual_count=len(clip_token_ids),
        actual_sha256=clip_ids_sha256,
    )
    fingerprint_payload = {
        "plan_sha256": plan_sha256,
        "entry_id": view.entry_id,
        "role": view.role,
        "pair_id": view.pair_id,
        "raw_prompt_sha256": view.raw_prompt_sha256,
        "llama_prompt_sha256": _sha256_text(view.llama_prompt),
        "clip_prompt_sha256": _sha256_text(view.clip_prompt),
        "llama_token_ids_sha256": llama_ids_sha256,
        "clip_token_ids_sha256": clip_ids_sha256,
        "prompt_template_sha256": prompt_template_sha256,
        "llama_max_sequence_length": llama_max_sequence_length,
        "clip_max_sequence_length": clip_max_sequence_length,
    }
    return {
        "entry_id": view.entry_id,
        "role": view.role,
        "pair_id": view.pair_id,
        "raw_prompt": view.raw_prompt,
        "raw_prompt_sha256": view.raw_prompt_sha256,
        "llama_prompt": view.llama_prompt,
        "llama_prompt_sha256": _sha256_text(view.llama_prompt),
        "clip_prompt": view.clip_prompt,
        "clip_prompt_sha256": _sha256_text(view.clip_prompt),
        "llama_token_count": len(llama_token_ids),
        "llama_token_ids": list(llama_token_ids),
        "llama_token_ids_sha256": llama_ids_sha256,
        "clip_token_count": len(clip_token_ids),
        "clip_token_ids": list(clip_token_ids),
        "clip_token_ids_sha256": clip_ids_sha256,
        "prompt_template": dict(prompt_template),
        "prompt_template_sha256": prompt_template_sha256,
        "prompt_template_crop_start": crop_start,
        "llama_max_sequence_length": llama_max_sequence_length,
        "clip_max_sequence_length": clip_max_sequence_length,
        "plan_sha256": plan_sha256,
        "routing": (
            "llama=encode_prompt(prompt);"
            "clip=_get_clip_prompt_embeds(clip_prompt)->pooled_prompt_embeds"
        ),
        "truncated": False,
        "dual_view_fingerprint_sha256": _canonical_sha256(fingerprint_payload),
    }


def _verify_frozen_token_fingerprint(
    view: HunyuanPromptView,
    *,
    encoder_name: str,
    actual_count: int,
    actual_sha256: str,
) -> None:
    expected_count = getattr(view, f"{encoder_name}_token_count")
    expected_sha256 = getattr(view, f"{encoder_name}_token_ids_sha256")
    if expected_count is None:
        return
    if expected_count != actual_count or expected_sha256 != actual_sha256:
        raise ValueError(
            f"Hunyuan {encoder_name} token fingerprint drift for entry '{view.entry_id}': "
            f"expected count/hash {expected_count}/{expected_sha256}, "
            f"got {actual_count}/{actual_sha256}."
        )


def _config_value(config: Any, name: str) -> Any:
    if isinstance(config, Mapping):
        return config.get(name)
    return getattr(config, name, None)


def _validate_hunyuan_encoding_shapes(
    pipeline: Any,
    prompt_embeds: Any,
    pooled_prompt_embeds: Any,
    prompt_attention_mask: Any,
    *,
    batch_size: int,
    sequence_length: int,
) -> None:
    tensors = {
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "prompt_attention_mask": prompt_attention_mask,
    }
    for name, tensor in tensors.items():
        if not isinstance(tensor, torch.Tensor):
            raise RuntimeError(
                f"HunyuanVideo {name} must be a tensor, got {type(tensor).__name__}."
            )
        if not bool(torch.isfinite(tensor).all()):
            raise RuntimeError(f"HunyuanVideo {name} contains non-finite values.")
    transformer = getattr(pipeline, "transformer", None)
    transformer_config = getattr(transformer, "config", None)
    text_embed_dim = _config_value(transformer_config, "text_embed_dim")
    pooled_projection_dim = _config_value(transformer_config, "pooled_projection_dim")
    if not isinstance(text_embed_dim, int) or not isinstance(pooled_projection_dim, int):
        raise RuntimeError(
            "HunyuanVideo transformer config must expose text_embed_dim and pooled_projection_dim."
        )
    if text_embed_dim != HUNYUAN_LLAMA_HIDDEN_SIZE:
        raise RuntimeError(
            "HunyuanVideo transformer text_embed_dim drift: expected frozen checkpoint value "
            f"{HUNYUAN_LLAMA_HIDDEN_SIZE}, got {text_embed_dim}."
        )
    if pooled_projection_dim != HUNYUAN_CLIP_POOLED_SIZE:
        raise RuntimeError(
            "HunyuanVideo transformer pooled_projection_dim drift: expected frozen checkpoint value "
            f"{HUNYUAN_CLIP_POOLED_SIZE}, got {pooled_projection_dim}."
        )
    expected_shapes = {
        "prompt_embeds": (batch_size, sequence_length, text_embed_dim),
        "pooled_prompt_embeds": (batch_size, pooled_projection_dim),
        "prompt_attention_mask": (batch_size, sequence_length),
    }
    for name, expected_shape in expected_shapes.items():
        actual_shape = tuple(tensors[name].shape)
        if actual_shape != expected_shape:
            raise RuntimeError(
                f"HunyuanVideo {name} shape drift: expected {expected_shape}, got {actual_shape}."
            )
    mask_values = torch.unique(prompt_attention_mask.detach())
    if any(int(value) not in {0, 1} for value in mask_values.cpu().tolist()):
        raise RuntimeError("HunyuanVideo prompt_attention_mask must contain only zero and one.")


def _validate_pooled_pair_deltas(
    views: Sequence[HunyuanPromptView],
    pooled_prompt_embeds: torch.Tensor,
) -> dict[str, float]:
    pairs: dict[str, dict[str, int]] = {}
    for index, view in enumerate(views):
        role = view.role.lower()
        if role in {"unsafe", "safe"} and view.pair_id is not None:
            pairs.setdefault(view.pair_id, {})[role] = index
    delta_l2: dict[str, float] = {}
    for pair_id, role_indices in pairs.items():
        if "unsafe" not in role_indices or "safe" not in role_indices:
            continue
        delta = (
            pooled_prompt_embeds[role_indices["safe"]].float()
            - pooled_prompt_embeds[role_indices["unsafe"]].float()
        )
        norm = float(torch.linalg.vector_norm(delta))
        if not bool(torch.isfinite(delta).all()) or norm <= 0.0:
            raise RuntimeError(
                f"Hunyuan CLIP unsafe/safe pair '{pair_id}' produced no finite pooled delta."
            )
        delta_l2[pair_id] = norm
    return delta_l2


def _hunyuan_runtime_provenance(pipeline: Any) -> dict[str, Any]:
    try:
        diffusers_version = metadata.version("diffusers")
    except metadata.PackageNotFoundError:
        diffusers_version = None
    try:
        transformers_version = metadata.version("transformers")
    except metadata.PackageNotFoundError:
        transformers_version = None
    try:
        pipeline_source = inspect.getsourcefile(type(pipeline))
    except TypeError:
        pipeline_source = None
    source_sha256 = None
    if pipeline_source is not None:
        source_path = Path(pipeline_source)
        try:
            if source_path.is_file():
                source_sha256 = hashlib.sha256(source_path.read_bytes()).hexdigest()
        except OSError:
            source_sha256 = None
    return {
        "diffusers_version": diffusers_version,
        "diffusers_direct_url": _distribution_direct_url("diffusers"),
        "transformers_version": transformers_version,
        "pipeline_class": f"{type(pipeline).__module__}.{type(pipeline).__qualname__}",
        "pipeline_source": pipeline_source,
        "pipeline_source_sha256": source_sha256,
        "llama_tokenizer": _tokenizer_runtime_provenance(getattr(pipeline, "tokenizer", None)),
        "clip_tokenizer": _tokenizer_runtime_provenance(getattr(pipeline, "tokenizer_2", None)),
        "llama_text_encoder": _text_encoder_runtime_provenance(
            getattr(pipeline, "text_encoder", None)
        ),
        "clip_text_encoder": _text_encoder_runtime_provenance(
            getattr(pipeline, "text_encoder_2", None)
        ),
    }


def _distribution_direct_url(distribution_name: str) -> dict[str, Any] | None:
    try:
        distribution = metadata.distribution(distribution_name)
    except metadata.PackageNotFoundError:
        return None
    for relative_path in distribution.files or ():
        if relative_path.name != "direct_url.json":
            continue
        path = Path(distribution.locate_file(relative_path))
        try:
            parsed = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _tokenizer_runtime_provenance(tokenizer: Any) -> dict[str, Any] | None:
    if tokenizer is None:
        return None
    init_kwargs = getattr(tokenizer, "init_kwargs", {})
    return {
        "class": f"{type(tokenizer).__module__}.{type(tokenizer).__qualname__}",
        "name_or_path": getattr(tokenizer, "name_or_path", None),
        "model_max_length": getattr(tokenizer, "model_max_length", None),
        "padding_side": getattr(tokenizer, "padding_side", None),
        "truncation_side": getattr(tokenizer, "truncation_side", None),
        "commit_hash": init_kwargs.get("_commit_hash")
        if isinstance(init_kwargs, Mapping)
        else None,
    }


def _text_encoder_runtime_provenance(text_encoder: Any) -> dict[str, Any] | None:
    if text_encoder is None:
        return None
    config = getattr(text_encoder, "config", None)
    architectures = _config_value(config, "architectures")
    if isinstance(architectures, tuple):
        architectures = list(architectures)
    return {
        "class": f"{type(text_encoder).__module__}.{type(text_encoder).__qualname__}",
        "dtype": str(getattr(text_encoder, "dtype", None)),
        "name_or_path": _config_value(config, "_name_or_path"),
        "commit_hash": _config_value(config, "_commit_hash"),
        "architectures": architectures,
        "hidden_size": _config_value(config, "hidden_size"),
        "max_position_embeddings": _config_value(config, "max_position_embeddings"),
    }


def _round_up_to_one_plus_multiple(value: int, multiple: int) -> int:
    if value <= 1:
        return 1
    remainder = (value - 1) % multiple
    return value if remainder == 0 else value + (multiple - remainder)


def _crop_videos(videos: Any, num_frames: int) -> Any:
    if num_frames <= 0 or not isinstance(videos, list):
        return videos
    cropped = []
    for frames in videos:
        if isinstance(frames, list) and len(frames) > num_frames:
            cropped.append(frames[:num_frames])
        else:
            cropped.append(frames)
    return cropped
