"""Native current-state concept predictions with CPU conditioning reuse."""
from __future__ import annotations
from typing import Any
import torch
from .adapters.base import AdapterState, DenoisingStepContext, PromptCondition
from .conditioning_cache import StateAwareConditionCache, encoding_fingerprint

class PredictionPoolError(RuntimeError):
    pass

def _move_tree(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device=device)
    if isinstance(value, dict):
        return {key: _move_tree(item, device) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_move_tree(item, device) for item in value)
    if isinstance(value, list):
        return [_move_tree(item, device) for item in value]
    return value


class _NativePredictor:
    def __init__(self, adapter: Any):
        self.adapter = adapter
        self.cache = StateAwareConditionCache(namespace="nailing_flux_prediction_replica")
        self.data_fingerprints: dict[str, str] = {}
        self.last_conditioning_evidence: dict[str, Any] = {}

    def predict(self, text: str, role: str, latents: torch.Tensor,
                timestep: Any, state: AdapterState) -> torch.Tensor:
        def prepare() -> PromptCondition:
            condition = self.adapter.prepare_prompt_for_state(
                text, state, prompt_view="registered_concept_only", call_role=role,
            )
            # The cache is CPU resident, as in the existing corrected steerer.
            return PromptCondition(condition.prompt, _move_tree(condition.data, torch.device("cpu")))
        mark = self.cache.mark()
        condition = self.cache.get_or_prepare_one(
            adapter=self.adapter, prompt=text, state=state,
            prompt_view="registered_concept_only", call_role=role, prepare=prepare,
        )
        event, = self.cache.evidence_since(mark)
        key = event["identity_sha256"]
        if key not in self.data_fingerprints:
            self.data_fingerprints[key] = encoding_fingerprint(condition.data)
        self.last_conditioning_evidence = event | {
            "condition_data_sha256": self.data_fingerprints[key],
            "unconditional_prediction_cache_key": condition.data.get("uncond_prediction_cache_key"),
        }
        current = PromptCondition(condition.prompt, _move_tree(condition.data, latents.device))
        prediction = self.adapter.predict_vector_field(latents, timestep, current, state)
        _validate_prediction(prediction, latents)
        return prediction


def _validate_prediction(prediction: torch.Tensor, latents: torch.Tensor) -> None:
    if not torch.is_tensor(prediction) or prediction.shape != latents.shape or prediction.dtype != torch.bfloat16:
        raise PredictionPoolError("Prediction shape/dtype differs from the native latent contract")
    if not bool(torch.isfinite(prediction).all()):
        raise PredictionPoolError("Nonfinite replica prediction")
