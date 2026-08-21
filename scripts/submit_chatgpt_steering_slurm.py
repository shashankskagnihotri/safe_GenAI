#!/usr/bin/env python3
"""Submit and continuously monitor the full staged Slurm dependency graph."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import json
import os
import shlex
import subprocess
import sys
import time

from hierasafe_flow.campaigns.chatgpt_steering import atomic_json, freeze_campaign, load_campaign_spec


TERMINAL_STATES = {
    "BOOT_FAIL", "CANCELLED", "COMPLETED", "DEADLINE", "FAILED", "NODE_FAIL",
    "OUT_OF_MEMORY", "PREEMPTED", "REVOKED", "TIMEOUT",
}


def _submit(options: list[str], command: list[str]) -> str:
    wrap = " ".join(shlex.quote(value) for value in command)
    result = subprocess.run(["sbatch", "--parsable", *options, "--wrap", wrap], check=True, text=True, capture_output=True)
    return result.stdout.strip().split(";")[0]


def _common(spec: dict[str, Any], name: str, log_pattern: Path, *, gpu: bool, duration: str, dependency: str | None, extra: list[str]) -> list[str]:
    slurm = spec["slurm"]
    values = [
        f"--job-name={name}",
        f"--chdir={spec['_root']}",
        f"--cpus-per-task={slurm['cpus_per_task']}",
        f"--mem={slurm['memory']}",
        f"--time={duration}",
        f"--partition={slurm['gpu_partition'] if gpu else slurm['cpu_partition']}",
        f"--account={slurm['account']}",
        f"--qos={slurm['qos']}",
        f"--output={log_pattern}",
        f"--error={log_pattern}",
        "--kill-on-invalid-dep=yes",
        f"--export=ALL,PYTHONPATH={Path(spec['_root']) / 'src'},PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True",
    ]
    if gpu:
        values.append(f"--gres={slurm['gpu_gres']}")
    if dependency:
        values.append(f"--dependency=afterok:{dependency}")
    return values + extra


def submit_graph(spec: dict[str, Any]) -> dict[str, Any]:
    root = Path(spec["_root"])
    state = root / spec["campaign"]["state_root"]
    logs = state / "slurm"
    logs.mkdir(parents=True, exist_ok=True)
    python = str(spec["slurm"]["python"])
    config = spec["_spec_path"]
    jobs: dict[str, str] = {}
    jobs["calibration"] = _submit(
        _common(spec, "chs-cal", logs / "cal-%A_%a.log", gpu=True, duration=spec["slurm"]["calibration_time"], dependency=None, extra=[f"--array=0-11%{spec['slurm']['calibration_concurrency']}"]),
        [python, "scripts/run_chatgpt_steering_calibration.py", "--config", config],
    )
    jobs["calibration_validation"] = _submit(
        _common(spec, "chs-calval", logs / "calval-%j.log", gpu=False, duration=spec["slurm"]["validation_time"], dependency=jobs["calibration"], extra=[]),
        [python, "scripts/validate_chatgpt_steering.py", "--config", config, "--stage", "calibration"],
    )
    jobs["baselines"] = _submit(
        _common(spec, "chs-base", logs / "base-%A_%a.log", gpu=True, duration=spec["slurm"]["generation_time"], dependency=jobs["calibration_validation"], extra=[f"--array=0-56%{spec['slurm']['baseline_concurrency']}"]),
        [python, "scripts/run_chatgpt_steering_cell.py", "--config", config, "--manifest", str(state / "final_matrix_baselines.jsonl")],
    )
    jobs["baseline_validation"] = _submit(
        _common(spec, "chs-baseval", logs / "baseval-%j.log", gpu=False, duration=spec["slurm"]["validation_time"], dependency=jobs["baselines"], extra=[]),
        [python, "scripts/validate_chatgpt_steering.py", "--config", config, "--stage", "baseline"],
    )
    jobs["steering"] = _submit(
        _common(spec, "chs-steer", logs / "steer-%A_%a.log", gpu=True, duration=spec["slurm"]["generation_time"], dependency=jobs["baseline_validation"], extra=[f"--array=0-179%{spec['slurm']['steering_concurrency']}"]),
        [python, "scripts/run_chatgpt_steering_cell.py", "--config", config, "--manifest", str(state / "final_matrix_steering.jsonl")],
    )
    jobs["output_validation"] = _submit(
        _common(spec, "chs-outval", logs / "outval-%j.log", gpu=False, duration=spec["slurm"]["validation_time"], dependency=jobs["steering"], extra=[]),
        [python, "scripts/validate_chatgpt_steering.py", "--config", config, "--stage", "final"],
    )
    jobs["gemini_review"] = _submit(
        _common(spec, "chs-gemini", logs / "gemini-%j.log", gpu=False, duration=spec["slurm"]["review_time"], dependency=jobs["output_validation"], extra=[]),
        [python, "scripts/run_chatgpt_steering_gemini.py", "--config", config],
    )
    ledger = {
        "campaign_id": spec["campaign"]["id"],
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "jobs": jobs,
        "task_counts": {"calibration": 12, "baselines": 57, "steering": 180, "gemini_reviews": 237},
        "gemini_key_present_at_submission": bool(os.environ.get("GEMINI_API_KEY")),
    }
    atomic_json(state / "job_ledger.json", ledger)
    return ledger


def _states(job_ids: list[str]) -> dict[str, str]:
    result = subprocess.run(
        ["sacct", "-n", "-P", "-j", ",".join(job_ids), "--format=JobIDRaw,State"],
        check=False, text=True, capture_output=True,
    )
    states: dict[str, str] = {}
    wanted = set(job_ids)
    for line in result.stdout.splitlines():
        parts = line.split("|")
        if len(parts) >= 2 and parts[0] in wanted:
            states[parts[0]] = parts[1].split()[0].split("+")[0]
    return states


def monitor(ledger: dict[str, Any], spec: dict[str, Any], poll_seconds: int) -> dict[str, Any]:
    state_dir = Path(spec["_root"]) / spec["campaign"]["state_root"]
    jobs = ledger["jobs"]
    last: dict[str, str] = {}
    while True:
        states_by_id = _states(list(jobs.values()))
        current = {name: states_by_id.get(job_id, "PENDING_ACCOUNTING") for name, job_id in jobs.items()}
        if current != last:
            snapshot = {"observed_at": datetime.now(timezone.utc).isoformat(), "states": current}
            atomic_json(state_dir / "job_monitor_latest.json", snapshot)
            print(json.dumps(snapshot, sort_keys=True), flush=True)
            last = current
        if all(state in TERMINAL_STATES for state in current.values()):
            break
        time.sleep(poll_seconds)
    failures = {name: state for name, state in current.items() if state != "COMPLETED"}
    result = {"completed_at": datetime.now(timezone.utc).isoformat(), "states": current, "failures": failures}
    atomic_json(state_dir / "job_monitor_terminal.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/experiments/chatgpt_steering_21_july.yaml")
    parser.add_argument("--gpu-partition")
    parser.add_argument("--cpu-partition")
    parser.add_argument("--account")
    parser.add_argument("--qos")
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--no-monitor", action="store_true")
    args = parser.parse_args()
    spec = load_campaign_spec(args.config)
    freeze_campaign(spec)
    for key in ("gpu_partition", "cpu_partition", "account", "qos"):
        value = getattr(args, key)
        if value:
            spec["slurm"][key] = value
    ledger = submit_graph(spec)
    print(json.dumps(ledger, sort_keys=True), flush=True)
    if not args.no_monitor:
        result = monitor(ledger, spec, args.poll_seconds)
        if result["failures"]:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
