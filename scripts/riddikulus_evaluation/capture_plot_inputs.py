#!/usr/bin/env python3
"""Freeze saved plotting inputs and renderer source without changing live work."""
from pathlib import Path
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
import sys

PROJECT = Path(__file__).resolve().parents[2]
ROOT_INPUTS = ("image_manifest.jsonl", "expected_image_manifest.jsonl", "campaign_state.json")
RENDER_SOURCE = (
    "src/riddikulus_evaluation/__init__.py",
    "src/riddikulus_evaluation/plots.py",
    "src/riddikulus_evaluation/protocol.py",
    "src/riddikulus_evaluation/rubric.py",
    "scripts/riddikulus_evaluation/plot_results.py",
    "scripts/riddikulus_evaluation/capture_plot_inputs.py",
)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def capture(campaign):
    paths = [campaign / name for name in ROOT_INPUTS]
    for task in sorted((campaign / "EVALUATION/tasks").glob("*/task.json")):
        # Include all JSON receipts, including original failed/refused attempts.
        paths.extend(task.parent / name for name in ("task.json", "provenance.json", "state.json"))
        paths.extend(task.parent.glob("*.json"))
    result = {str(p.relative_to(campaign)): (str(p), p.read_bytes()) for p in sorted(set(paths))}
    for name in RENDER_SOURCE:
        path = PROJECT / name
        result["SOURCE/" + name] = (str(path), path.read_bytes())
    return result


def validate(files, expected_count):
    decode = lambda name: json.loads(files[name][1])
    actual = [json.loads(line) for line in files["image_manifest.jsonl"][1].splitlines() if line.strip()]
    expected = [json.loads(line) for line in files["expected_image_manifest.jsonl"][1].splitlines() if line.strip()]
    identities = lambda rows: {(r["cell_id"], r["arm"]) for r in rows}
    if len(expected) != expected_count or len(identities(expected)) != expected_count:
        raise ValueError("Expected inventory differs from the explicitly required full cohort")
    if len(identities(actual)) != len(actual) or not identities(actual) <= identities(expected):
        raise ValueError("Duplicate observed identity or image outside the full inventory")
    by_image = {(r["cell_id"], r["arm"]): r for r in actual}
    states = Counter()
    for name in sorted(n for n in files if n.startswith("EVALUATION/tasks/") and n.endswith("/task.json")):
        prefix = name.removesuffix("task.json")
        task, private, state = (decode(prefix + n) for n in ("task.json", "provenance.json", "state.json"))
        if task["task_id"] != Path(name).parent.name or state["task_id"] != task["task_id"]:
            raise ValueError("Task directory/state identity mismatch")
        source = private["source"]
        if by_image.get((source["cell_id"], source["arm"])) != source:
            raise ValueError("Task source is outside or differs from the captured image manifest")
        result_name = prefix + "result.json"
        complete = state["status"] in {"COMPLETE", "COMPLETE_WITH_ISSUES"}
        if complete != (result_name in files):
            raise ValueError("Partial result commit; reconcile and capture a stable boundary")
        if complete:
            result = decode(result_name)
            if (state["result_sha256"] != digest(files[result_name][1])
                    or result["task_sha256"] != digest(files[name][1])
                    or result["provenance_sha256"] != digest(files[prefix + "provenance.json"][1])):
                raise ValueError("Completed result binding mismatch")
        states[task["judge"] + "/" + state["status"]] += 1
    return {"expected_images": len(expected), "expected_primary_stages": 4 * len(expected),
            "available_images": len(actual), "captured_task_states": dict(states)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    args = parser.parse_args()
    campaign, output = args.campaign.resolve(), args.output.resolve()
    if args.expected_images < 1 or output.exists():
        raise ValueError("Require a positive full inventory and a new output directory")
    first, second = capture(campaign), capture(campaign)
    if first != second:
        raise RuntimeError("Live inputs changed during capture; reject and retry at a stable boundary")
    counts = validate(first, args.expected_images)
    output.mkdir(parents=True, exist_ok=False)
    records = {}
    for name, (original, data) in sorted(first.items()):
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(data)
        if target.read_bytes() != data:
            raise RuntimeError("Snapshot write differs: " + name)
        target.chmod(0o444)
        records[name] = {"original_path": original, "sha256": digest(data), "bytes": len(data)}
    receipt = {"schema": "riddikulus.plot-input-snapshot.v1",
               "captured_at": datetime.now(timezone.utc).isoformat(), **counts,
               "files": records, "python": sys.version, "python_executable": sys.executable,
               "packages": {p: importlib.metadata.version(p) for p in ("numpy", "matplotlib", "Pillow")},
               "limitations": ["Two byte-equal reads are an observed stable capture, not a global filesystem transaction.",
                               "Original absolute images, neutral images, specs, generation/timing and raw-response dependencies must remain accessible and unchanged.",
                               "Capture verifies copy and receipt bindings; the unchanged plotter still authenticates source artifacts and judges.",
                               "Use the copied evaluation tree only for plotting, never preparation or ingestion.",
                               "Numeric replay is distinct from byte-identical timestamped PDF/SVG rendering."]}
    temporary = output / ".snapshot_receipt.json.partial"
    with temporary.open("x") as stream:
        json.dump(receipt, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n"); stream.flush(); os.fsync(stream.fileno())
    receipt_path = output / "snapshot_receipt.json"
    temporary.replace(receipt_path); receipt_path.chmod(0o444)
    print(json.dumps({"snapshot": str(output), "receipt_sha256": digest(receipt_path.read_bytes()), **counts}))


if __name__ == "__main__":
    main()
