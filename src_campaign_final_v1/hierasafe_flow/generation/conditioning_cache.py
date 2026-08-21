from __future__ import annotations

import dataclasses
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import torch

from hierasafe_flow.adapters.base import AdapterState, PromptCondition


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a tensor-free cache identity without lossy coercion."""

    normalized = _canonical_identity_value(value, path="identity")
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_identity_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def encoding_fingerprint(value: Any) -> str:
    """Content fingerprint an encoded condition, including every tensor byte."""

    digest = hashlib.sha256()
    _update_value_fingerprint(digest, value, path="encoding")
    return digest.hexdigest()


@dataclass(frozen=True)
class ConditionCacheRecord:
    sequence_index: int
    namespace: str
    status: str
    call_role: str
    prompt_view: str
    prompt_sha256: str
    identity_sha256: str
    encoding_fingerprint: str
    identity: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass
class _CacheEntry:
    canonical_identity: bytes
    identity: dict[str, Any]
    value: PromptCondition
    fingerprint: str


class StateAwareConditionCache:
    """One authenticated condition cache shared by every prediction subsystem.

    Cache values are immutable by contract.  A hit re-fingerprints the value,
    catches in-place tensor mutation, and authenticates the canonical identity
    bytes associated with the digest before returning anything.
    """

    def __init__(self, namespace: str = "generation") -> None:
        if not isinstance(namespace, str) or not namespace.strip():
            raise ValueError("Condition-cache namespace must be a non-empty string.")
        self.namespace = namespace
        self._entries: dict[str, _CacheEntry] = {}
        self._records: list[ConditionCacheRecord] = []

    def mark(self) -> int:
        return len(self._records)

    @property
    def evidence(self) -> tuple[dict[str, Any], ...]:
        return tuple(deepcopy(record.to_dict()) for record in self._records)

    def evidence_since(self, mark: int) -> tuple[dict[str, Any], ...]:
        if isinstance(mark, bool) or not isinstance(mark, int) or not 0 <= mark <= len(
            self._records
        ):
            raise ValueError("Condition-cache evidence mark is outside the record range.")
        return tuple(deepcopy(record.to_dict()) for record in self._records[mark:])

    def get_or_prepare_one(
        self,
        *,
        adapter: Any,
        prompt: str,
        state: AdapterState,
        prompt_view: str,
        call_role: str,
        prepare: Callable[[], PromptCondition] | None = None,
    ) -> PromptCondition:
        identity = _adapter_cache_identity(
            adapter,
            prompt,
            state,
            prompt_view=prompt_view,
            call_role=call_role,
        )
        canonical = canonical_json_bytes(identity)
        identity_sha = hashlib.sha256(canonical).hexdigest()
        entry = self._entries.get(identity_sha)
        status = "hit"
        if entry is not None:
            if entry.canonical_identity != canonical:
                raise RuntimeError(
                    "Condition-cache SHA-256 collision: identical digest has different identity bytes."
                )
            current_fingerprint = encoding_fingerprint(entry.value)
            if current_fingerprint != entry.fingerprint:
                raise RuntimeError(
                    "Condition-cache value fingerprint changed after insertion; refusing a mutated hit."
                )
            value = entry.value
            fingerprint = current_fingerprint
        else:
            status = "miss"
            value = (
                prepare()
                if prepare is not None
                else _adapter_prepare_one(
                    adapter,
                    prompt,
                    state,
                    prompt_view=prompt_view,
                    call_role=call_role,
                )
            )
            if not isinstance(value, PromptCondition):
                raise TypeError(
                    "State-aware condition preparation must return PromptCondition; "
                    f"got {type(value).__name__}."
                )
            fingerprint = encoding_fingerprint(value)
            self._entries[identity_sha] = _CacheEntry(
                canonical_identity=canonical,
                identity=deepcopy(identity),
                value=value,
                fingerprint=fingerprint,
            )
        self._append_record(
            status=status,
            call_role=call_role,
            prompt_view=prompt_view,
            prompt=prompt,
            identity_sha256=identity_sha,
            fingerprint=fingerprint,
            identity=identity,
        )
        return value

    def get_or_prepare_many(
        self,
        *,
        adapter: Any,
        prompts: Sequence[str],
        state: AdapterState,
        prompt_views: Sequence[str] | str,
        call_roles: Sequence[str],
    ) -> list[PromptCondition]:
        prompts = list(prompts)
        roles = list(call_roles)
        views = (
            [prompt_views] * len(prompts)
            if isinstance(prompt_views, str)
            else list(prompt_views)
        )
        if not (len(prompts) == len(views) == len(roles)):
            raise ValueError("prompts, prompt_views, and call_roles must have identical lengths.")

        # Resolve identities before encoding.  Equal identities are prepared
        # once, but every caller still receives an ordered hit/miss record.
        identities: list[dict[str, Any]] = []
        canonical_rows: list[bytes] = []
        digests: list[str] = []
        for prompt, view, role in zip(prompts, views, roles):
            identity = _adapter_cache_identity(
                adapter,
                prompt,
                state,
                prompt_view=view,
                call_role=role,
            )
            canonical = canonical_json_bytes(identity)
            identities.append(identity)
            canonical_rows.append(canonical)
            digests.append(hashlib.sha256(canonical).hexdigest())

        missing_unique: list[int] = []
        first_for_digest: dict[str, int] = {}
        for index, (digest, canonical) in enumerate(zip(digests, canonical_rows)):
            prior = first_for_digest.get(digest)
            if prior is not None and canonical_rows[prior] != canonical:
                raise RuntimeError("Condition-cache SHA-256 collision within one batch request.")
            first_for_digest.setdefault(digest, index)
            entry = self._entries.get(digest)
            if entry is not None and entry.canonical_identity != canonical:
                raise RuntimeError("Condition-cache SHA-256 collision against an existing entry.")
            if entry is None and prior is None:
                missing_unique.append(index)

        if missing_unique:
            missing_prompts = [prompts[index] for index in missing_unique]
            missing_roles = [roles[index] for index in missing_unique]
            missing_views = [views[index] for index in missing_unique]
            if len(set(missing_views)) == 1:
                prepared = _adapter_prepare_many(
                    adapter,
                    missing_prompts,
                    state,
                    prompt_view=missing_views[0],
                    call_roles=missing_roles,
                )
            else:
                prepared = [
                    _adapter_prepare_one(
                        adapter,
                        prompt,
                        state,
                        prompt_view=view,
                        call_role=role,
                    )
                    for prompt, view, role in zip(
                        missing_prompts, missing_views, missing_roles
                    )
                ]
            if len(prepared) != len(missing_unique):
                raise RuntimeError(
                    "Adapter returned the wrong number of state-aware prompt conditions."
                )
            for index, value in zip(missing_unique, prepared):
                if not isinstance(value, PromptCondition):
                    raise TypeError("State-aware batch preparation returned a non-condition value.")
                fingerprint = encoding_fingerprint(value)
                self._entries[digests[index]] = _CacheEntry(
                    canonical_identity=canonical_rows[index],
                    identity=deepcopy(identities[index]),
                    value=value,
                    fingerprint=fingerprint,
                )

        output: list[PromptCondition] = []
        seen_this_call: set[str] = set()
        preexisting = set(self._entries) - {digests[index] for index in missing_unique}
        for prompt, view, role, identity, canonical, digest in zip(
            prompts, views, roles, identities, canonical_rows, digests
        ):
            entry = self._entries[digest]
            if entry.canonical_identity != canonical:
                raise RuntimeError("Condition-cache identity changed during batch preparation.")
            fingerprint = encoding_fingerprint(entry.value)
            if fingerprint != entry.fingerprint:
                raise RuntimeError("Condition-cache batch value fingerprint mismatch.")
            status = "hit" if digest in preexisting or digest in seen_this_call else "miss"
            seen_this_call.add(digest)
            self._append_record(
                status=status,
                call_role=role,
                prompt_view=view,
                prompt=prompt,
                identity_sha256=digest,
                fingerprint=fingerprint,
                identity=identity,
            )
            output.append(entry.value)
        return output

    def invalidate_epoch(self, condition_epoch: int) -> int:
        if isinstance(condition_epoch, bool) or not isinstance(condition_epoch, int):
            raise TypeError("condition_epoch must be an integer.")
        doomed = [
            digest
            for digest, entry in self._entries.items()
            if entry.identity.get("condition_epoch") == condition_epoch
        ]
        for digest in doomed:
            del self._entries[digest]
        return len(doomed)

    def _append_record(
        self,
        *,
        status: str,
        call_role: str,
        prompt_view: str,
        prompt: str,
        identity_sha256: str,
        fingerprint: str,
        identity: Mapping[str, Any],
    ) -> None:
        self._records.append(
            ConditionCacheRecord(
                sequence_index=len(self._records),
                namespace=self.namespace,
                status=status,
                call_role=str(call_role),
                prompt_view=str(prompt_view),
                prompt_sha256=hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                identity_sha256=identity_sha256,
                encoding_fingerprint=fingerprint,
                identity=deepcopy(dict(identity)),
            )
        )


def _canonical_identity_value(value: Any, *, path: str) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Non-finite float in cache identity at {path}.")
        return value
    if isinstance(value, torch.Tensor):
        raise TypeError(f"Dynamic tensor is forbidden in a condition-cache identity at {path}.")
    if dataclasses.is_dataclass(value):
        return _canonical_identity_value(dataclasses.asdict(value), path=path)
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key in sorted(value):
            if not isinstance(key, str):
                raise TypeError(f"Cache identity mapping key at {path} must be a string.")
            output[key] = _canonical_identity_value(value[key], path=f"{path}.{key}")
        return output
    if isinstance(value, (list, tuple)):
        return [
            _canonical_identity_value(item, path=f"{path}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"Unsupported condition-cache identity type at {path}: {type(value).__name__}."
    )


def _adapter_cache_identity(
    adapter: Any,
    prompt: str,
    state: Any,
    *,
    prompt_view: str,
    call_role: str,
) -> dict[str, Any]:
    hook = getattr(adapter, "conditioning_cache_identity", None)
    if callable(hook):
        return hook(
            prompt,
            state,
            prompt_view=prompt_view,
            call_role=call_role,
        )
    extra = getattr(state, "extra", {})
    config = getattr(adapter, "config", {})
    return {
        "schema_version": 1,
        "namespace": "legacy_adapter_compatibility",
        "adapter": str(getattr(adapter, "adapter_name", type(adapter).__name__)),
        "model_role": str(extra.get("model_role", "primary")),
        "model_id": str(extra.get("model_id", getattr(adapter, "model_id", "test-double"))),
        "model_revision": str(
            extra.get("model_revision", config.get("revision", "unversioned"))
        ),
        "segment_index": int(extra.get("segment_index", 0)),
        "condition_epoch": int(extra.get("condition_epoch", 0)),
        "segment_seed": int(extra.get("segment_seed", extra.get("base_seed", 0))),
        "anchor_sha256": extra.get("anchor_sha256"),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "prompt_view": prompt_view,
    }


def _adapter_prepare_one(
    adapter: Any,
    prompt: str,
    state: Any,
    *,
    prompt_view: str,
    call_role: str,
) -> PromptCondition:
    hook = getattr(adapter, "prepare_prompt_for_state", None)
    if callable(hook):
        return hook(
            prompt,
            state,
            prompt_view=prompt_view,
            call_role=call_role,
        )
    return adapter.prepare_prompt(prompt)


def _adapter_prepare_many(
    adapter: Any,
    prompts: list[str],
    state: Any,
    *,
    prompt_view: str,
    call_roles: list[str],
) -> list[PromptCondition]:
    hook = getattr(adapter, "prepare_prompts_for_state", None)
    if callable(hook):
        return hook(
            prompts,
            state,
            prompt_view=prompt_view,
            call_roles=call_roles,
        )
    batch_hook = getattr(adapter, "prepare_prompts", None)
    if callable(batch_hook):
        return batch_hook(prompts)
    return [adapter.prepare_prompt(prompt) for prompt in prompts]


def _update_value_fingerprint(digest: Any, value: Any, *, path: str) -> None:
    digest.update(type(value).__qualname__.encode("utf-8"))
    digest.update(b"\0")
    if value is None:
        return
    if isinstance(value, PromptCondition):
        _update_value_fingerprint(digest, value.prompt, path=f"{path}.prompt")
        _update_value_fingerprint(digest, value.data, path=f"{path}.data")
        return
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode("ascii"))
        try:
            raw = tensor.numpy().tobytes(order="C")
        except TypeError:
            raw = tensor.view(torch.uint8).numpy().tobytes(order="C")
        digest.update(raw)
        return
    if isinstance(value, Mapping):
        for key in sorted(value, key=str):
            _update_value_fingerprint(digest, str(key), path=f"{path}.key")
            _update_value_fingerprint(digest, value[key], path=f"{path}.{key}")
        return
    if isinstance(value, (list, tuple)):
        digest.update(str(len(value)).encode("ascii"))
        for index, item in enumerate(value):
            _update_value_fingerprint(digest, item, path=f"{path}[{index}]")
        return
    if dataclasses.is_dataclass(value):
        _update_value_fingerprint(digest, dataclasses.asdict(value), path=path)
        return
    if isinstance(value, (str, bytes, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"Non-finite value in encoded condition at {path}.")
        digest.update(value if isinstance(value, bytes) else repr(value).encode("utf-8"))
        return
    # Prompt encoders occasionally return immutable tokenizer helper objects.
    # Their fully qualified type and deterministic repr are authenticated;
    # objects with address-bearing reprs are rejected to prevent false hits.
    rendered = repr(value)
    if " at 0x" in rendered:
        raise TypeError(
            f"Encoded condition contains an object without stable content at {path}: "
            f"{type(value).__name__}."
        )
    digest.update(rendered.encode("utf-8"))
