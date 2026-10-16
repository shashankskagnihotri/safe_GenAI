#!/usr/bin/env python3
"""Minimal durable Gemini dispatcher; generation and HERE Sol agents stay separate."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from riddikulus_evaluation import protocol
from riddikulus_evaluation.gemini import configuration, run_task


def verify_source(path: Path) -> dict:
    manifest = protocol.read_json(path)
    if Path(manifest["source_root"]).resolve() != ROOT:
        raise ValueError("Watcher must execute the exact sealed source root")
    for relative, expected in manifest["source_sha256"].items():
        candidate = ROOT / relative
        if Path(relative).is_absolute() or ".." in Path(relative).parts or protocol.sha256(candidate) != expected:
            raise ValueError("Sealed evaluation source mismatch: " + relative)
    return manifest


def generation_terminal(expected_rows: list[dict], state: dict) -> bool:
    # BLOCKED_MODEL and ordinary FAILED can later recover. Only the generation owner's
    # explicit final state or closure flag establishes that no new images will arrive.
    return bool(expected_rows) and (state.get("status") == "GENERATION_COMPLETE_AWAITING_EVALUATION"
                                   or state.get("generation_closed") is True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--expected", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign-state", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, default=2, choices=(1, 2))
    parser.add_argument("--poll-seconds", type=int, default=60)
    parser.add_argument("--requeue-after-seconds", type=int, default=2400)
    parser.add_argument("--max-planned-requeues", type=int, default=120)
    args = parser.parse_args()
    if not 10 <= args.poll_seconds <= 60:
        parser.error("poll seconds must be between 10 and 60")
    if not 0 <= args.requeue_after_seconds <= 2400 or not 1 <= args.max_planned_requeues <= 120:
        parser.error("Planned continuation must retain the bounded drain and restart limits")
    verify_source(args.source_manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    campaign_root = args.campaign_state.resolve().parent
    stop_requested = False
    requeue_requested = False
    segment_started = time.monotonic()
    restart_count = int(os.environ.get("SLURM_RESTART_COUNT", "0"))

    def stop(signum, frame):
        nonlocal stop_requested
        stop_requested = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    # One authoritative dispatcher. Advisory lock is on the campaign's shared filesystem.
    with (args.output / ".gemini_dispatcher.lock").open("a") as owner_lock:
        fcntl.flock(owner_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity = {"schema": protocol.SCHEMA, "pid": os.getpid(), "host": os.uname().nodename,
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_at": protocol.now(),
                    "source_manifest": str(args.source_manifest.resolve()),
                    "source_manifest_sha256": protocol.sha256(args.source_manifest),
                    "configuration": configuration(), "concurrency": args.concurrency}
        status_path = args.output / "gemini_dispatcher.json"
        futures = {}
        fatal_failure = None
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            while True:
                if args.requeue_after_seconds and time.monotonic() - segment_started >= args.requeue_after_seconds:
                    # Drain bounded requests before the one-hour allocation ends.
                    # Persisted task state survives a scheduler-native requeue.
                    requeue_requested = True
                    stop_requested = True
                stopped = stop_requested or (campaign_root / "STOP").exists() or (args.output / "STOP").exists()
                if stopped:
                    stop_requested = True
                failure = None
                for future in list(futures):
                    if not future.done():
                        continue
                    task_path = futures.pop(future)
                    try:
                        future.result()
                    except Exception as exc:
                        # A code bug or interrupted ownership must not look like a successful judge.
                        failure = {"task": str(task_path), "type": type(exc).__name__,
                                   "message": str(exc), "traceback": traceback.format_exc(), "utc": protocol.now()}
                        protocol.atomic_json(task_path.with_name("dispatcher_exception.json"), failure)
                if failure:
                    fatal_failure = failure
                    stop_requested = True
                expected_rows = protocol.read_manifest(args.expected) if args.expected.exists() else []
                campaign = protocol.read_json(args.campaign_state)
                if not stop_requested and args.images.exists() and args.expected.exists():
                    capacity = args.concurrency - len(futures)
                    if capacity:
                        # run_task owns its technical retry and backoff. Its persisted
                        # state may temporarily be TECHNICAL_FAILURE while its Future
                        # is still alive, so do not submit that task a second time.
                        prepared = protocol.prepare_next(args.images, args.expected, args.output, "gemini", args.concurrency)
                        active_paths = {path.resolve() for path in futures.values()}
                        for assignment in prepared["assignments"]:
                            task = Path(assignment["task_path"]).resolve()
                            if task in active_paths or len(futures) >= args.concurrency:
                                continue
                            futures[pool.submit(run_task, task, stop_paths=(campaign_root / "STOP", args.output / "STOP"),
                                                stop_check=lambda: stop_requested)] = task
                            active_paths.add(task)
                tasks = []
                for path in args.output.glob("tasks/*/task.json"):
                    if protocol.read_json(path)["judge"] == "gemini":
                        tasks.append(protocol.read_json(path.with_name("state.json")))
                counts = {name: sum(t["status"] == name for t in tasks) for name in sorted({t["status"] for t in tasks})}
                inflight_paths = {str(path) for path in futures.values()}
                # A persisted RUNNING task absent from this process is ambiguous, never rerun.
                ambiguous = [t["task_id"] for t in tasks if t["status"] == "RUNNING"
                             and str(args.output.resolve() / "tasks" / t["task_id"] / "task.json") not in inflight_paths]
                if ambiguous:
                    stop_requested = True
                all_generated_terminal = generation_terminal(expected_rows, campaign) or (campaign_root / "GENERATION_CLOSED").exists()
                unfinished = any(t["status"] not in protocol.TERMINAL for t in tasks)
                ready_count = len(protocol.read_manifest(args.images)) if args.images.exists() else 0
                all_tasks_registered = len(tasks) == 2 * ready_count
                done = all_generated_terminal and all_tasks_registered and not unfinished and not futures
                status = "STOPPING" if stop_requested and futures else "STOPPED" if stop_requested else "CONCLUDED" if done else "RUNNING"
                record = {**identity, "status": status, "heartbeat": protocol.now(), "active_tasks": sorted(inflight_paths),
                          "stage_status_counts": counts, "ambiguous_task_ids": ambiguous,
                          "generation_terminal": all_generated_terminal, "verified_available_images": ready_count,
                          "expected_images": len(expected_rows), "sol_dispatch": "HERE agents only; not launched by watcher",
                          "planned_restart_count": restart_count,
                          "planned_requeue_requested": requeue_requested,
                          "failure": fatal_failure, "stop_semantics": "No new calls or retries after STOP; current bounded API calls may finish."}
                protocol.atomic_json(status_path, record)
                if (stop_requested and not futures) or done:
                    break
                time.sleep(args.poll_seconds)
        if fatal_failure or ambiguous:
            raise RuntimeError("Gemini dispatcher stopped on recorded execution failure or ambiguous prior invocation")
        if requeue_requested and not done and not ((campaign_root / "STOP").exists() or (args.output / "STOP").exists()):
            if restart_count >= args.max_planned_requeues:
                raise RuntimeError("Bounded Gemini planned-continuation limit reached")
            job_id = os.environ.get("SLURM_JOB_ID", "")
            if not job_id.isdigit():
                raise RuntimeError("Planned continuation requires an actual scheduler allocation")
            registered = protocol.read_json(args.output / "gemini_scheduler.json")
            if registered.get("job_id") != job_id:
                raise RuntimeError("Refusing to requeue a different registered Gemini allocation")
            expected_comment = next((a.split("=", 1)[1] for a in registered["command"]
                                     if a.startswith("--comment=")), None)
            observed = subprocess.run(["scontrol", "show", "job", job_id, "-o"],
                                      text=True, capture_output=True, timeout=30, check=True).stdout
            if ("UserId=sagnihot(" not in observed or not expected_comment
                    or not expected_comment.startswith("riddikulus-evaluation:gemini:")
                    or "Comment=" + expected_comment not in observed):
                raise RuntimeError("Gemini requeue scheduler ownership differs")
            protocol.atomic_json(status_path, {**record, "status": "PLANNED_REQUEUE_REQUESTED"})
            subprocess.run(["scontrol", "requeue", job_id], timeout=30, check=True)


if __name__ == "__main__":
    main()
