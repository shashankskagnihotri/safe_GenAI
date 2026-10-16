"""Exact native scheduler identities and checkpoint tensor placement."""
from __future__ import annotations
from collections.abc import Mapping
import copy
import hashlib
import json
import math
import numpy as np
import torch
MODEL_STATE = "pool25-warm10.native-model-state.v1"

def _canonical(value):
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu().contiguous()
        payload = tensor.reshape(-1).view(torch.uint8).numpy().tobytes()
        return {"tensor_dtype": str(tensor.dtype), "shape": list(tensor.shape),
                "data_sha256": hashlib.sha256(payload).hexdigest()}
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise TypeError("Object arrays cannot identify native scheduler state")
        return {"numpy_dtype": str(array.dtype), "shape": list(array.shape),
                "data_sha256": hashlib.sha256(array.tobytes()).hexdigest()}
    if isinstance(value, np.generic):
        return _canonical(value.item())
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("Native scheduler mappings require string keys")
        result = {}
        for key in sorted(value):
            item = value[key]
            # Diffusers constructs this metadata list from a Python set. Its
            # iteration order is not part of the native solver computation.
            if key == "_use_default_values" and isinstance(item, (tuple, list)):
                if any(not isinstance(name, str) for name in item):
                    raise TypeError("Diffusers default-parameter metadata must contain names")
                item = sorted(item)
            result[key] = _canonical(item)
        return result
    if isinstance(value, (tuple, list)):
        return {"sequence_type": type(value).__name__, "items": [_canonical(v) for v in value]}
    if isinstance(value, (torch.dtype, torch.device)):
        return {"torch_type": type(value).__name__, "value": str(value)}
    if value is None or type(value) in (bool, str, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            return {"float": value.hex()}
        return value
    raise TypeError(f"Unsupported native state identity value: {type(value).__module__}.{type(value).__qualname__}")


def scheduler_hash(scheduler):
    """Hash complete values/config/history, excluding physical tensor placement."""
    identity = {"class": f"{type(scheduler).__module__}.{type(scheduler).__qualname__}",
                "runtime": _canonical(vars(scheduler))}
    return hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def config_hash(config):
    return hashlib.sha256(json.dumps(_canonical(config), sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def freeze_tree(value):
    """CPU snapshot plus tensor-device tree, including nested solver history."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone(), str(value.device)
    if isinstance(value, dict):
        pairs = {key: freeze_tree(v) for key, v in value.items()}
        return {k: v[0] for k, v in pairs.items()}, {k: v[1] for k, v in pairs.items()}
    if isinstance(value, (tuple, list)):
        pairs = [freeze_tree(v) for v in value]
        return type(value)(v[0] for v in pairs), type(value)(v[1] for v in pairs)
    return copy.deepcopy(value), None


def thaw_tree(value, devices):
    if isinstance(value, torch.Tensor):
        if not isinstance(devices, str):
            raise RuntimeError("Saved native tensor lacks its original device")
        return value.to(torch.device(devices))
    if isinstance(value, dict):
        if not isinstance(devices, dict) or value.keys() != devices.keys():
            raise RuntimeError("Saved native mapping/device tree differs")
        return {k: thaw_tree(v, devices[k]) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        if type(devices) is not type(value) or len(value) != len(devices):
            raise RuntimeError("Saved native sequence/device tree differs")
        return type(value)(thaw_tree(v, d) for v, d in zip(value, devices))
    if devices is not None:
        raise RuntimeError("Saved scalar has a tensor device")
    return copy.deepcopy(value)


def setup_native_state_policy(adapter, prompt, state):
    hook = getattr(adapter, "setup_checkpoint_state_policy", None)
    return hook(prompt, state) if callable(hook) else {"enabled": False, "adapter_name": adapter.adapter_name}

def reset_arm_cache(adapter):
    if getattr(adapter, "_shared_scope", None) is not None or getattr(adapter, "_shared_prediction_scope", None) is not None:
        raise RuntimeError("Cannot reset an arm inside a native prediction scope")
    hook = getattr(adapter, "reset_generation_state", None)
    if callable(hook):
        hook()

def prefix_mode_counters(adapter):
    hook = getattr(adapter, "checkpoint_state_counters", None)
    return hook() if callable(hook) else {"enabled": False, "adapter_name": adapter.adapter_name}

def export_model_state(adapter):
    hook = getattr(adapter, "export_checkpoint_state", None)
    return {"schema": MODEL_STATE, "adapter_name": adapter.adapter_name,
            "native_state": hook() if callable(hook) else None}

def restore_model_state(adapter, record):
    if record.get("schema") != MODEL_STATE or record.get("adapter_name") != adapter.adapter_name:
        raise RuntimeError("Native model checkpoint belongs to another adapter/schema")
    if record["native_state"] is not None:
        hook = getattr(adapter, "restore_checkpoint_state", None)
        if not callable(hook):
            raise RuntimeError("Adapter cannot restore its declared native checkpoint state")
        hook(record["native_state"])
