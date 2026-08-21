from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any, Mapping

import torch

from hierasafe_flow.adapters.base import (
    AdapterState,
    _call_with_supported_kwargs,
)
from hierasafe_flow.adapters.flux_adapter import FluxAdapter


FLUX_DUAL_VIEW_CONFIG_KEY = "flux_dual_view_conditioning"
FLUX_DUAL_VIEW_SCHEMA_VERSION = 1
FLUX_CLIP_MAX_TOKENS = 77
FLUX_T5_MAX_TOKENS = 512
FLUX_DUAL_VIEW_MODEL_ID = "black-forest-labs/FLUX.1-dev"
FLUX_DUAL_VIEW_MODEL_REVISION = "3de623fc3c33e44ffbe2bad470d0f45bccf2eb21"

# A dual-view run must have exactly one authority for text conditioning. These
# raw Diffusers call keys or embedding bypasses could silently override one half
# of the registered CLIP/T5 pair if admitted beside the structured plan.
FLUX_DUAL_VIEW_FORBIDDEN_RAW_CONFIG_KEYS = frozenset(
    {
        "prompt",
        "prompt_2",
        "prompt_embeds",
        "pooled_prompt_embeds",
        "negative_prompt",
        "negative_prompt_2",
        "negative_prompt_embeds",
        "negative_pooled_prompt_embeds",
    }
)


def validate_flux_dual_view_conditioning(raw: Any) -> dict[str, Any]:
    """Validate and normalize the exact FLUX.1 CLIP/T5 text-view contract."""

    if not isinstance(raw, Mapping):
        raise ValueError(f"{FLUX_DUAL_VIEW_CONFIG_KEY} must be a mapping.")
    expected_top = {"schema_version", "positive", "negative"}
    if set(raw) != expected_top:
        raise ValueError(
            f"{FLUX_DUAL_VIEW_CONFIG_KEY} must contain exactly {sorted(expected_top)}."
        )
    if raw.get("schema_version") != FLUX_DUAL_VIEW_SCHEMA_VERSION:
        raise ValueError(
            f"{FLUX_DUAL_VIEW_CONFIG_KEY}.schema_version must be exactly "
            f"{FLUX_DUAL_VIEW_SCHEMA_VERSION}."
        )

    positive_keys = {
        "clip_prompt",
        "t5_prompt_2",
        "clip_string_sha256",
        "clip_token_count",
        "clip_token_ids_sha256",
        "t5_string_sha256",
        "t5_token_count",
        "t5_token_ids_sha256",
    }
    positive = raw.get("positive")
    if not isinstance(positive, Mapping) or set(positive) != positive_keys:
        raise ValueError(
            f"{FLUX_DUAL_VIEW_CONFIG_KEY}.positive must contain exactly {sorted(positive_keys)}."
        )
    normalized_positive = _validated_prompt_pair(
        positive,
        role="positive",
        clip_text_key="clip_prompt",
        t5_text_key="t5_prompt_2",
        clip_sha_key="clip_string_sha256",
        t5_sha_key="t5_string_sha256",
        clip_count_key="clip_token_count",
        t5_count_key="t5_token_count",
        clip_token_sha_key="clip_token_ids_sha256",
        t5_token_sha_key="t5_token_ids_sha256",
    )
    if normalized_positive["clip_prompt"] == normalized_positive["t5_prompt_2"]:
        raise ValueError(
            f"{FLUX_DUAL_VIEW_CONFIG_KEY} positive CLIP and T5 views must be independent."
        )

    negative = raw.get("negative")
    normalized_negative: dict[str, Any] | None
    if negative is None:
        normalized_negative = None
    else:
        negative_keys = {
            "clip_negative_prompt",
            "t5_negative_prompt_2",
            "clip_negative_string_sha256",
            "clip_negative_token_count",
            "clip_negative_token_ids_sha256",
            "t5_negative_string_sha256",
            "t5_negative_token_count",
            "t5_negative_token_ids_sha256",
        }
        if not isinstance(negative, Mapping) or set(negative) != negative_keys:
            raise ValueError(
                f"{FLUX_DUAL_VIEW_CONFIG_KEY}.negative must be null or contain exactly "
                f"{sorted(negative_keys)}."
            )
        normalized_negative = _validated_prompt_pair(
            negative,
            role="negative",
            clip_text_key="clip_negative_prompt",
            t5_text_key="t5_negative_prompt_2",
            clip_sha_key="clip_negative_string_sha256",
            t5_sha_key="t5_negative_string_sha256",
            clip_count_key="clip_negative_token_count",
            t5_count_key="t5_negative_token_count",
            clip_token_sha_key="clip_negative_token_ids_sha256",
            t5_token_sha_key="t5_negative_token_ids_sha256",
        )
        if (
            normalized_negative["clip_negative_prompt"]
            == normalized_negative["t5_negative_prompt_2"]
        ):
            raise ValueError(
                f"{FLUX_DUAL_VIEW_CONFIG_KEY} negative CLIP and T5 views must be independent."
            )
        if normalized_negative["clip_negative_prompt"] == normalized_positive["clip_prompt"]:
            raise ValueError(
                f"{FLUX_DUAL_VIEW_CONFIG_KEY} positive and negative CLIP views must differ."
            )
        if normalized_negative["t5_negative_prompt_2"] == normalized_positive["t5_prompt_2"]:
            raise ValueError(
                f"{FLUX_DUAL_VIEW_CONFIG_KEY} positive and negative T5 views must differ."
            )
    return {
        "schema_version": FLUX_DUAL_VIEW_SCHEMA_VERSION,
        "positive": normalized_positive,
        "negative": normalized_negative,
    }


def _validated_prompt_pair(
    value: Mapping[str, Any],
    *,
    role: str,
    clip_text_key: str,
    t5_text_key: str,
    clip_sha_key: str,
    t5_sha_key: str,
    clip_count_key: str,
    t5_count_key: str,
    clip_token_sha_key: str,
    t5_token_sha_key: str,
) -> dict[str, Any]:
    normalized = dict(value)
    for text_key, sha_key in (
        (clip_text_key, clip_sha_key),
        (t5_text_key, t5_sha_key),
    ):
        text = _nonempty_text(value.get(text_key), f"{role}.{text_key}")
        declared = _require_sha256(value.get(sha_key), f"{role}.{sha_key}")
        actual = _text_sha256(text)
        if declared != actual:
            raise ValueError(
                f"{FLUX_DUAL_VIEW_CONFIG_KEY}.{role}.{sha_key} mismatch: "
                f"declared {declared}, calculated {actual}."
            )
        normalized[text_key] = text
        normalized[sha_key] = declared
    for count_key, token_sha_key, maximum in (
        (clip_count_key, clip_token_sha_key, FLUX_CLIP_MAX_TOKENS),
        (t5_count_key, t5_token_sha_key, FLUX_T5_MAX_TOKENS),
    ):
        count = value.get(count_key)
        if isinstance(count, bool) or not isinstance(count, int) or not 0 < count <= maximum:
            raise ValueError(
                f"{FLUX_DUAL_VIEW_CONFIG_KEY}.{role}.{count_key} must be an integer "
                f"between 1 and {maximum}; zero-truncation is mandatory."
            )
        normalized[count_key] = count
        normalized[token_sha_key] = _require_sha256(
            value.get(token_sha_key), f"{role}.{token_sha_key}"
        )
    return normalized


def _nonempty_text(value: Any, role: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{FLUX_DUAL_VIEW_CONFIG_KEY}.{role} must be a non-empty string.")
    return value


def _require_sha256(value: Any, role: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{FLUX_DUAL_VIEW_CONFIG_KEY}.{role} must be a lowercase SHA-256 digest.")
    return value


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _prompt_view_record(*, call_key: str, encoder: str, text: str) -> dict[str, Any]:
    return {
        "call_key": call_key,
        "encoder": encoder,
        "text": text,
        "utf8_sha256": _text_sha256(text),
    }


def _token_ids_sha256(token_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(token_ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def _tokenize_without_truncation(tokenizer: Any, text: str, *, role: str) -> list[int]:
    encoded = tokenizer(
        text,
        padding=False,
        truncation=False,
        return_attention_mask=False,
        add_special_tokens=True,
    )
    token_ids = (
        encoded.get("input_ids")
        if isinstance(encoded, Mapping)
        else getattr(encoded, "input_ids", None)
    )
    if isinstance(token_ids, torch.Tensor):
        if token_ids.ndim == 2 and token_ids.shape[0] == 1:
            token_ids = token_ids[0]
        if token_ids.ndim != 1:
            raise RuntimeError(
                f"Flux {role} tokenizer preflight expected one sequence, got "
                f"shape {tuple(token_ids.shape)}."
            )
        token_ids = token_ids.detach().cpu().tolist()
    if (
        isinstance(token_ids, (list, tuple))
        and len(token_ids) == 1
        and isinstance(token_ids[0], (list, tuple))
    ):
        token_ids = token_ids[0]
    if not isinstance(token_ids, (list, tuple)) or not all(
        isinstance(token, int) and not isinstance(token, bool) for token in token_ids
    ):
        raise RuntimeError(f"Flux {role} tokenizer did not return one integer input_ids sequence.")
    return [int(token) for token in token_ids]


def _preflight_token_view(
    *,
    tokenizer: Any,
    text: str,
    role: str,
    expected_count: int,
    expected_sha256: str,
    maximum: int,
    call_key: str,
    encoder: str,
) -> dict[str, Any]:
    token_ids = _tokenize_without_truncation(tokenizer, text, role=role)
    actual_count = len(token_ids)
    actual_sha256 = _token_ids_sha256(token_ids)
    if actual_count > maximum:
        raise ValueError(
            f"Flux {role} has {actual_count} tokens, exceeding its no-truncation limit {maximum}."
        )
    if actual_count != expected_count or actual_sha256 != expected_sha256:
        raise ValueError(
            f"Flux {role} token fingerprint drift: expected count/hash "
            f"{expected_count}/{expected_sha256}, got {actual_count}/{actual_sha256}."
        )
    return {
        "role": role,
        "call_key": call_key,
        "encoder": encoder,
        "text_utf8_sha256": _text_sha256(text),
        "token_count": actual_count,
        "token_ids_sha256": actual_sha256,
        "maximum_tokens_including_special_tokens": maximum,
        "truncated": False,
    }


class FluxDualViewAdapter(FluxAdapter):
    """FLUX.1 adapter used only by a fingerprinted dual-view v3 job.

    It intentionally lives outside the sealed legacy ``flux_adapter.py``. The
    explicit adapter identity prevents a legacy FLUX.1 manifest from silently
    opting into this route while the model/checkpoint fields continue to record
    the unchanged denoiser.
    """

    adapter_name = "flux_dual_view"

    def __init__(
        self,
        model_id: str,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(model_id=model_id, device=device, dtype=dtype, config=config)
        if (
            self.model_id != FLUX_DUAL_VIEW_MODEL_ID
            or self.config.get("revision") != FLUX_DUAL_VIEW_MODEL_REVISION
        ):
            raise ValueError(
                "FluxDualViewAdapter is versioned only for the exact registered "
                f"model/revision pair {FLUX_DUAL_VIEW_MODEL_ID}@"
                f"{FLUX_DUAL_VIEW_MODEL_REVISION}."
            )
        raw_plan = self.config.get(FLUX_DUAL_VIEW_CONFIG_KEY)
        if raw_plan is None:
            raise ValueError(
                "FluxDualViewAdapter requires an explicit flux_dual_view_conditioning plan."
            )
        conflicts = sorted(FLUX_DUAL_VIEW_FORBIDDEN_RAW_CONFIG_KEYS.intersection(self.config))
        if conflicts:
            raise ValueError(
                f"{FLUX_DUAL_VIEW_CONFIG_KEY} conflicts with raw model conditioning keys: "
                f"{conflicts}."
            )
        self._flux_dual_view_plan = validate_flux_dual_view_conditioning(raw_plan)
        self._flux_dual_view_calls: list[dict[str, Any]] = []
        self._flux_dual_view_native_calls: list[dict[str, Any]] = []
        self._flux_dual_view_preflight: dict[str, Any] | None = None

    def validate_primary_prompts(self, prompts: list[str]) -> None:
        registered = self._flux_dual_view_plan["positive"]["clip_prompt"]
        if not prompts or any(prompt != registered for prompt in prompts):
            raise ValueError(
                "A Flux dual-view generation may run only its exact registered "
                "positive.clip_prompt; prompt-file mixtures and prompt overrides are forbidden."
            )

    def preflight_conditioning_plan(self) -> dict[str, Any]:
        """Re-tokenize all configured views and prove zero truncation."""

        self._require_loaded()
        assert self.pipeline is not None
        clip_tokenizer = getattr(self.pipeline, "tokenizer", None)
        t5_tokenizer = getattr(self.pipeline, "tokenizer_2", None)
        if not callable(clip_tokenizer) or not callable(t5_tokenizer):
            raise RuntimeError(
                "Flux dual-view preflight requires callable pipeline.tokenizer (CLIP) "
                "and pipeline.tokenizer_2 (T5)."
            )
        positive = self._flux_dual_view_plan["positive"]
        records = [
            _preflight_token_view(
                tokenizer=clip_tokenizer,
                text=positive["clip_prompt"],
                role="positive.clip",
                expected_count=positive["clip_token_count"],
                expected_sha256=positive["clip_token_ids_sha256"],
                maximum=FLUX_CLIP_MAX_TOKENS,
                call_key="prompt",
                encoder="CLIPTextModel",
            ),
            _preflight_token_view(
                tokenizer=t5_tokenizer,
                text=positive["t5_prompt_2"],
                role="positive.t5",
                expected_count=positive["t5_token_count"],
                expected_sha256=positive["t5_token_ids_sha256"],
                maximum=FLUX_T5_MAX_TOKENS,
                call_key="prompt_2",
                encoder="T5EncoderModel",
            ),
        ]
        negative = self._flux_dual_view_plan["negative"]
        if negative is not None:
            records.extend(
                [
                    _preflight_token_view(
                        tokenizer=clip_tokenizer,
                        text=negative["clip_negative_prompt"],
                        role="negative.clip",
                        expected_count=negative["clip_negative_token_count"],
                        expected_sha256=negative["clip_negative_token_ids_sha256"],
                        maximum=FLUX_CLIP_MAX_TOKENS,
                        call_key="negative_prompt",
                        encoder="CLIPTextModel",
                    ),
                    _preflight_token_view(
                        tokenizer=t5_tokenizer,
                        text=negative["t5_negative_prompt_2"],
                        role="negative.t5",
                        expected_count=negative["t5_negative_token_count"],
                        expected_sha256=negative["t5_negative_token_ids_sha256"],
                        maximum=FLUX_T5_MAX_TOKENS,
                        call_key="negative_prompt_2",
                        encoder="T5EncoderModel",
                    ),
                ]
            )
        self._flux_dual_view_preflight = {
            "schema_version": 1,
            "status": "passed",
            "plan_sha256": _canonical_sha256(self._flux_dual_view_plan),
            "require_no_primary_or_secondary_truncation": True,
            "token_ids_hash_encoding": "compact_json_integer_array_utf8",
            "views": records,
        }
        return deepcopy(self._flux_dual_view_preflight)

    def native_negative_prompt_kwargs(
        self,
        *,
        positive_clip_prompt: str,
        negative_clip_prompt: str,
    ) -> dict[str, str]:
        """Return the exact paired raw views for a native Flux true-CFG call."""

        self._require_loaded()
        if self._flux_dual_view_preflight is None:
            raise RuntimeError(
                "Flux dual-view native-negative routing requires successful whole-plan "
                "tokenizer preflight before call construction."
            )
        self.validate_primary_prompts([positive_clip_prompt])
        negative = self._flux_dual_view_plan["negative"]
        if negative is None:
            raise ValueError(
                "Flux dual-view native-negative routing requires a registered paired negative plan."
            )
        if negative_clip_prompt != negative["clip_negative_prompt"]:
            raise ValueError(
                "Flux dual-view native-negative routing accepts only the exact registered "
                "negative CLIP view."
            )
        positive = self._flux_dual_view_plan["positive"]
        kwargs = {
            "prompt": positive["clip_prompt"],
            "prompt_2": positive["t5_prompt_2"],
            "negative_prompt": negative["clip_negative_prompt"],
            "negative_prompt_2": negative["t5_negative_prompt_2"],
        }
        self._flux_dual_view_native_calls.append(
            {
                "sequence_index": len(self._flux_dual_view_native_calls),
                "method": "native_flux_pipeline_paired_prompt_views",
                "positive": {
                    "clip": _prompt_view_record(
                        call_key="prompt",
                        encoder="CLIPTextModel",
                        text=kwargs["prompt"],
                    ),
                    "t5": _prompt_view_record(
                        call_key="prompt_2",
                        encoder="T5EncoderModel",
                        text=kwargs["prompt_2"],
                    ),
                },
                "negative": {
                    "clip": _prompt_view_record(
                        call_key="negative_prompt",
                        encoder="CLIPTextModel",
                        text=kwargs["negative_prompt"],
                    ),
                    "t5": _prompt_view_record(
                        call_key="negative_prompt_2",
                        encoder="T5EncoderModel",
                        text=kwargs["negative_prompt_2"],
                    ),
                },
            }
        )
        return deepcopy(kwargs)

    def native_explicit_none_prompt_kwargs(
        self,
        *,
        positive_clip_prompt: str,
    ) -> dict[str, str | None]:
        """Return dual positive views plus two explicit-null native CFG controls."""

        self._require_loaded()
        if self._flux_dual_view_preflight is None:
            raise RuntimeError(
                "Flux dual-view explicit-none routing requires successful whole-plan "
                "tokenizer preflight before call construction."
            )
        self.validate_primary_prompts([positive_clip_prompt])
        if self._flux_dual_view_plan["negative"] is not None:
            raise ValueError(
                "Flux dual-view explicit-none routing requires a registered null negative plan."
            )
        positive = self._flux_dual_view_plan["positive"]
        kwargs: dict[str, str | None] = {
            "prompt": positive["clip_prompt"],
            "prompt_2": positive["t5_prompt_2"],
            "negative_prompt": None,
            "negative_prompt_2": None,
        }
        self._flux_dual_view_native_calls.append(
            {
                "sequence_index": len(self._flux_dual_view_native_calls),
                "method": "native_flux_pipeline_dual_positive_explicit_none_negative_views",
                "positive": {
                    "clip": _prompt_view_record(
                        call_key="prompt",
                        encoder="CLIPTextModel",
                        text=positive["clip_prompt"],
                    ),
                    "t5": _prompt_view_record(
                        call_key="prompt_2",
                        encoder="T5EncoderModel",
                        text=positive["t5_prompt_2"],
                    ),
                },
                "negative": None,
                "explicit_none_arguments": ["negative_prompt", "negative_prompt_2"],
            }
        )
        return deepcopy(kwargs)

    def _call_encode_prompt(self, prompt: str | list[str]) -> Any:
        assert self.pipeline is not None
        if not hasattr(self.pipeline, "encode_prompt"):
            raise NotImplementedError(
                "FluxPipeline does not expose encode_prompt; dual-view conditioning "
                "cannot be authenticated."
            )
        prompts = [prompt] if isinstance(prompt, str) else list(prompt)
        if not prompts or any(not isinstance(value, str) for value in prompts):
            raise TypeError(
                "Flux encode_prompt input must be a string or a non-empty list of strings."
            )
        prompt_2_values = [self._registered_t5_view(value) for value in prompts]
        encoded_prompt: str | list[str] = prompts[0] if isinstance(prompt, str) else prompts
        encoded_prompt_2: str | list[str] = (
            prompt_2_values[0] if isinstance(prompt, str) else prompt_2_values
        )
        output = _call_with_supported_kwargs(
            self.pipeline.encode_prompt,
            {
                "prompt": encoded_prompt,
                "prompt_2": encoded_prompt_2,
                "device": self.device,
                "dtype": self.dtype,
                "num_images_per_prompt": 1,
                "num_videos_per_prompt": 1,
                "do_classifier_free_guidance": self._manual_guidance_scale() > 1.0,
                "max_sequence_length": self.config.get("max_sequence_length"),
            },
        )
        for clip_text, t5_text in zip(prompts, prompt_2_values, strict=True):
            role = self._registered_view_role(clip_text)
            self._flux_dual_view_calls.append(
                {
                    "sequence_index": len(self._flux_dual_view_calls),
                    "role": role,
                    "registered_dual_view_applied": role
                    in {"registered_positive", "registered_negative"},
                    "clip": _prompt_view_record(
                        call_key="prompt", encoder="CLIPTextModel", text=clip_text
                    ),
                    "t5": _prompt_view_record(
                        call_key="prompt_2", encoder="T5EncoderModel", text=t5_text
                    ),
                }
            )
        return output

    def _registered_view_role(self, prompt: str) -> str:
        if prompt == self._flux_dual_view_plan["positive"]["clip_prompt"]:
            return "registered_positive"
        negative = self._flux_dual_view_plan["negative"]
        if negative is not None and prompt == negative["clip_negative_prompt"]:
            return "registered_negative"
        return "legacy_mirrored_internal_prompt"

    def _registered_t5_view(self, prompt: str) -> str:
        role = self._registered_view_role(prompt)
        if role == "registered_positive":
            return str(self._flux_dual_view_plan["positive"]["t5_prompt_2"])
        if role == "registered_negative":
            return str(self._flux_dual_view_plan["negative"]["t5_negative_prompt_2"])
        return prompt

    def conditioning_cache_identity(
        self,
        prompt: str,
        state: AdapterState,
        *,
        prompt_view: str,
        call_role: str,
    ) -> dict[str, Any]:
        identity = super().conditioning_cache_identity(
            prompt,
            state,
            prompt_view=prompt_view,
            call_role=call_role,
        )
        t5_view = self._registered_t5_view(prompt)
        identity["flux_dual_view"] = {
            "schema_version": FLUX_DUAL_VIEW_SCHEMA_VERSION,
            "plan_sha256": _canonical_sha256(self._flux_dual_view_plan),
            "role": self._registered_view_role(prompt),
            "clip_prompt_sha256": _text_sha256(prompt),
            "t5_prompt_2_sha256": _text_sha256(t5_view),
        }
        return identity

    def conditioning_provenance(self) -> dict[str, Any]:
        positive = self._flux_dual_view_plan["positive"]
        negative = self._flux_dual_view_plan["negative"]
        return {
            "schema_version": 1,
            "method": "flux1_registered_dual_prompt_views",
            "plan_sha256": _canonical_sha256(self._flux_dual_view_plan),
            "positive": {
                "clip": _prompt_view_record(
                    call_key="prompt",
                    encoder="CLIPTextModel",
                    text=positive["clip_prompt"],
                )
                | {
                    "frozen_token_count": positive["clip_token_count"],
                    "frozen_token_ids_sha256": positive["clip_token_ids_sha256"],
                },
                "t5": _prompt_view_record(
                    call_key="prompt_2",
                    encoder="T5EncoderModel",
                    text=positive["t5_prompt_2"],
                )
                | {
                    "frozen_token_count": positive["t5_token_count"],
                    "frozen_token_ids_sha256": positive["t5_token_ids_sha256"],
                },
            },
            "negative": (
                None
                if negative is None
                else {
                    "clip": _prompt_view_record(
                        call_key="negative_prompt",
                        encoder="CLIPTextModel",
                        text=negative["clip_negative_prompt"],
                    )
                    | {
                        "frozen_token_count": negative["clip_negative_token_count"],
                        "frozen_token_ids_sha256": negative["clip_negative_token_ids_sha256"],
                    },
                    "t5": _prompt_view_record(
                        call_key="negative_prompt_2",
                        encoder="T5EncoderModel",
                        text=negative["t5_negative_prompt_2"],
                    )
                    | {
                        "frozen_token_count": negative["t5_negative_token_count"],
                        "frozen_token_ids_sha256": negative["t5_negative_token_ids_sha256"],
                    },
                }
            ),
            "runtime_preflight": deepcopy(self._flux_dual_view_preflight),
            "encode_calls": deepcopy(self._flux_dual_view_calls),
            "native_paired_calls": deepcopy(self._flux_dual_view_native_calls),
        }
