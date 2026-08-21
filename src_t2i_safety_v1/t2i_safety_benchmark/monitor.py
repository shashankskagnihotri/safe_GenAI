from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import CALIBRATION_ROOT, OUTPUT_ROOT, WORK_ROOT, atomic_json


TERMINAL_SUCCESS = {"COMPLETED"}
TERMINAL_FAILURE = {
    "BOOT_FAIL",
    "CANCELLED",
    "DEADLINE",
    "FAILED",
    "NODE_FAIL",
    "OUT_OF_MEMORY",
    "PREEMPTED",
    "REVOKED",
    "TIMEOUT",
}
FATAL_PATTERNS = (
    "Traceback (most recent call last)",
    "CUDA out of memory",
    "Out Of Memory",
    "slurmstepd: error",
    "RuntimeError:",
    "ValueError:",
    "FileNotFoundError:",
)


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_receipts(paths: list[Path]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for path in paths:
        receipt = json.loads(path.read_text(encoding="utf-8"))
        if receipt.get("slurm_arrays") is not False:
            raise RuntimeError(f"Receipt is not explicitly non-array: {path}")
        for job in receipt.get("jobs", []):
            job_id = str(job["job_id"])
            if job_id in seen:
                raise RuntimeError(f"Duplicate monitored job id {job_id}")
            seen.add(job_id)
            jobs.append({**job, "receipt": str(path)})
    if not jobs:
        raise RuntimeError("No jobs found in monitor receipts.")
    return jobs


def slurm_states(jobs: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    result: dict[str, dict[str, str]] = {}
    ids = [str(job["job_id"]) for job in jobs]
    for offset in range(0, len(ids), 200):
        batch = ids[offset : offset + 200]
        process = subprocess.run(
            [
                "sacct",
                "-n",
                "-X",
                "-P",
                "-j",
                ",".join(batch),
                "--format=JobIDRaw,JobName,State,Elapsed,ExitCode,NodeList",
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        for line in process.stdout.splitlines():
            if not line.strip():
                continue
            fields = line.split("|")
            if len(fields) < 6:
                continue
            job_id, name, state, elapsed, exit_code, node = fields[:6]
            if job_id in ids:
                result[job_id] = {
                    "job_name": name,
                    "state": state.split("+", 1)[0],
                    "elapsed": elapsed,
                    "exit_code": exit_code,
                    "node": node,
                }
    for job_id in ids:
        result.setdefault(
            job_id,
            {
                "job_name": "",
                "state": "UNKNOWN",
                "elapsed": "",
                "exit_code": "",
                "node": "",
            },
        )
    return result


def fatal_logs(jobs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    findings: list[dict[str, Any]] = []
    log_root = WORK_ROOT / "logs"
    for job in jobs:
        job_id = str(job["job_id"])
        for suffix in ("out", "err"):
            path = log_root / f"{job_id}.{suffix}"
            if not path.is_file():
                continue
            with path.open("r", encoding="utf-8", errors="replace") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if any(pattern in line for pattern in FATAL_PATTERNS):
                        findings.append(
                            {
                                "job_id": job_id,
                                "path": str(path),
                                "line": line_number,
                                "text": line.rstrip()[:1000],
                            }
                        )
    return findings


def artifact_counts() -> dict[str, int]:
    counts = Counter()
    if OUTPUT_ROOT.exists():
        for path in OUTPUT_ROOT.rglob("*"):
            if not path.is_file():
                continue
            if path.name == "_SUCCESS.json":
                counts["benchmark_successes"] += 1
            elif path.name == "evaluation.json":
                counts["benchmark_evaluations"] += 1
            elif path.name == "image.png":
                counts["benchmark_images"] += 1
    fairness_root = CALIBRATION_ROOT / "fairness_probe"
    if fairness_root.exists():
        counts["fairness_probe_successes"] = sum(
            1 for _ in fairness_root.rglob("_SUCCESS.json")
        )
        counts["fairness_probe_evaluations"] = sum(
            1 for _ in fairness_root.rglob("evaluation.json")
        )
    midsteer_root = CALIBRATION_ROOT / "midsteer"
    if midsteer_root.exists():
        counts["midsteer_artifacts"] = sum(
            1 for _ in midsteer_root.glob("*/*/artifact.pt")
        )
    reference_root = CALIBRATION_ROOT / "unsafe_references_v3"
    if reference_root.exists():
        counts["unsafe_reference_banks"] = sum(
            1 for _ in reference_root.glob("*/*/unsafe_latents.pt")
        )
    return dict(counts)


def snapshot(receipts: list[Path]) -> dict[str, Any]:
    jobs = load_receipts(receipts)
    states = slurm_states(jobs)
    state_counts = Counter(value["state"] for value in states.values())
    failures = [
        {
            "job_id": job_id,
            **value,
        }
        for job_id, value in states.items()
        if value["state"] in TERMINAL_FAILURE
        or (
            value["state"] in TERMINAL_SUCCESS
            and value["exit_code"] not in {"0:0", ""}
        )
    ]
    value = {
        "schema_version": 1,
        "observed_at": _utc(),
        "receipts": [str(path) for path in receipts],
        "job_count": len(jobs),
        "state_counts": dict(sorted(state_counts.items())),
        "jobs": [
            {
                **job,
                **states[str(job["job_id"])],
            }
            for job in jobs
        ],
        "terminal_failures": failures,
        "fatal_log_findings": fatal_logs(jobs),
        "artifacts": artifact_counts(),
    }
    return value


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(
        description="Monitor immutable non-array T2ISafety submission receipts."
    )
    value.add_argument("--receipt", action="append", required=True, type=Path)
    value.add_argument("--watch", action="store_true")
    value.add_argument("--interval", type=int, default=60)
    value.add_argument(
        "--snapshot",
        type=Path,
        default=WORK_ROOT / "monitor" / "latest.json",
    )
    return value


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    if args.interval < 10:
        raise ValueError("Monitor interval must be at least ten seconds.")
    while True:
        value = snapshot(args.receipt)
        atomic_json(args.snapshot, value)
        print(
            json.dumps(
                {
                    "observed_at": value["observed_at"],
                    "job_count": value["job_count"],
                    "state_counts": value["state_counts"],
                    "terminal_failures": len(value["terminal_failures"]),
                    "fatal_log_findings": len(value["fatal_log_findings"]),
                    "artifacts": value["artifacts"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if not args.watch:
            return
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
