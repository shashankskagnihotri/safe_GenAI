"""Atomic per-attempt records and explicit stop ownership."""
from __future__ import annotations
from contextlib import contextmanager
from contextvars import ContextVar
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time

_STOP_PATHS = ContextVar("riddikulus_stop_paths", default=())

def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def canonical_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f'.{path.name}.{os.getpid()}.{time.time_ns()}.partial')
    with temporary.open('x') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


def append(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a') as stream:
        stream.write(json.dumps(value, sort_keys=True, allow_nan=False) + '\n')
        stream.flush(); os.fsync(stream.fileno())


@contextmanager
def execution_scope(attempt, stop_paths=()):
    paths = tuple(Path(p) for p in stop_paths) + (Path(attempt) / "STOP", Path(attempt) / "REPAIR_HOLD")
    token = _STOP_PATHS.set(paths)
    try:
        check_stop()
        yield
    finally:
        _STOP_PATHS.reset(token)

def check_stop():
    present = [str(p) for p in _STOP_PATHS.get() if p.exists()]
    if present:
        raise RuntimeError("Riddikulus execution stopped: " + ", ".join(present))
