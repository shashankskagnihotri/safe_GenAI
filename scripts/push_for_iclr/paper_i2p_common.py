#!/usr/bin/env python3
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path


def load_json(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def sha256_file(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_sha256(path, expected):
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError("SHA256 mismatch for %s: %s != %s" % (path, actual, expected))
    return actual


def assert_git_commit(repo, expected):
    actual = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != expected:
        raise RuntimeError("Commit mismatch for %s: %s != %s" % (repo, actual, expected))
    return actual


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%s" % os.getpid())
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(temporary, path)


def atomic_jsonl(path, records):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp.%s" % os.getpid())
    with temporary.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    os.replace(temporary, path)


def load_jsonl(path):
    records = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise RuntimeError("Invalid JSONL %s line %d" % (path, line_number)) from exc
    return records


def load_dataset(config):
    dataset = config["dataset"]
    path = Path(dataset["admitted_path"])
    assert_sha256(path, dataset["sha256"])
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != int(dataset["rows"]):
        raise RuntimeError("Dataset row count %d != %d" % (len(rows), dataset["rows"]))
    return rows


def shard_bounds(total, shards, task_id):
    if task_id < 0 or task_id >= shards:
        raise ValueError("Task %d outside [0, %d)" % (task_id, shards))
    base, remainder = divmod(total, shards)
    start = task_id * base + min(task_id, remainder)
    count = base + (1 if task_id < remainder else 0)
    return start, count


def ensure_runtime(expected_python):
    expected_prefix = Path(expected_python).parent.parent.resolve()
    actual_prefix = Path(sys.prefix).resolve()
    if actual_prefix != expected_prefix:
        raise RuntimeError("Wrong environment: %s != %s" % (actual_prefix, expected_prefix))


def ensure_fresh_directory(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.mkdir()
    return path


def snapshot_config(config_path, attempt_root):
    source = Path(config_path)
    destination = Path(attempt_root) / "PROTOCOL_CONFIG.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = source.read_bytes()
    if destination.exists():
        if destination.read_bytes() != content:
            raise RuntimeError("Attempt config snapshot differs from submitted config")
        return
    temporary = destination.with_name(destination.name + ".tmp.%s" % os.getpid())
    temporary.write_bytes(content)
    try:
        os.link(temporary, destination)
    except FileExistsError:
        if destination.read_bytes() != content:
            raise
    finally:
        temporary.unlink(missing_ok=True)
