#!/usr/bin/env python3
import argparse
import math
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from paper_i2p_common import atomic_json, atomic_jsonl, load_json, load_jsonl, sha256_file


def validate_images(records):
    for index, record in enumerate(records):
        path = Path(record["image_path"])
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError("Missing image: %s" % path)
        if sha256_file(path) != record["image_sha256"]:
            raise RuntimeError("Image hash mismatch: %s" % path)
        if (index + 1) % 1000 == 0:
            print("Integrity checked %d/%d" % (index + 1, len(records)), flush=True)


def write_pages(temporary, records, page_size, expected_pages):
    pages_dir = temporary / "contact_pages"
    pages_dir.mkdir()
    pages = int(math.ceil(len(records) / float(page_size)))
    if pages != expected_pages:
        raise RuntimeError("Contact page count %d != %d" % (pages, expected_pages))
    for page in range(pages):
        atomic_json(pages_dir / ("page_%04d.json" % page), {
            "page_index": page,
            "records": records[page * page_size:(page + 1) * page_size],
        })
    return pages


def aggregate(method, config):
    attempt = Path(config["attempt_root"])
    final = attempt / "AGGREGATE"
    if final.exists():
        raise RuntimeError("Refusing to overwrite aggregate: %s" % final)
    temporary = attempt / (".AGGREGATE_BUILDING_%s" % os.environ.get("SLURM_JOB_ID", "local"))
    if temporary.exists():
        raise RuntimeError("Aggregate temporary path exists")
    temporary.mkdir(parents=True)
    shards = int(config["generation"]["shards"])
    records = []
    for task_id in range(shards):
        if method == "sld":
            shard = attempt / ("EVALUATION/shard_%03d" % task_id)
        else:
            shard = attempt / ("GENERATION/shard_%03d" % task_id)
        if not (shard / "STATUS.json").is_file():
            raise RuntimeError("Incomplete shard: %s" % shard)
        records.extend(load_jsonl(shard / "manifest.jsonl"))
    records.sort(key=lambda item: (int(item["case_number"]), int(item["sample_index"])))
    if len(records) != int(config["expected_images"]):
        raise RuntimeError("Image count %d != %d" % (len(records), config["expected_images"]))
    keys = [(int(item["case_number"]), int(item["sample_index"])) for item in records]
    if len(set(keys)) != len(keys):
        raise RuntimeError("Duplicate case/sample keys in aggregate")
    counts = Counter(case for case, _ in keys)
    expected_per_case = int(config["generation"]["images_per_prompt"])
    if len(counts) != int(config["dataset"]["rows"]) or set(counts.values()) != {expected_per_case}:
        raise RuntimeError("Per-case image cardinality is invalid")
    validate_images(records)

    if method == "sld":
        targets = config["paper_targets"]
        metrics = {"overall": sum(bool(item["unsafe"]) for item in records) / len(records)}
        denominators = {"overall": len(records)}
        for category in targets:
            if category == "overall":
                continue
            selected = [item for item in records if category in item["categories"].lower()]
            if not selected:
                raise RuntimeError("No records for category %s" % category)
            metrics[category] = sum(bool(item["unsafe"]) for item in selected) / len(selected)
            denominators[category] = len(selected)
        tolerance = float(config["absolute_tolerance"])
        comparisons = {
            name: {
                "observed": metrics[name],
                "paper": float(target),
                "absolute_error": abs(metrics[name] - float(target)),
                "tolerance": tolerance,
                "pass": abs(metrics[name] - float(target)) <= tolerance,
                "denominator": denominators[name],
            }
            for name, target in targets.items()
        }
    else:
        observed = sum(bool(item["unsafe"]) for item in records) / len(records)
        paper = float(config["paper_target_attack_success_rate"])
        tolerance = float(config["absolute_tolerance"])
        metrics = {"attack_success_rate": observed}
        comparisons = {
            "attack_success_rate": {
                "observed": observed,
                "paper": paper,
                "absolute_error": abs(observed - paper),
                "tolerance": tolerance,
                "pass": abs(observed - paper) <= tolerance,
                "denominator": len(records),
            }
        }
    numeric_pass = all(item["pass"] for item in comparisons.values())
    atomic_jsonl(temporary / "manifest.jsonl", records)
    pages = write_pages(
        temporary,
        records,
        int(config["contact_sheet_page_size"]),
        int(config["contact_sheet_pages"]),
    )
    admission = {
        "status": "numeric_reproduction_pass" if numeric_pass else "numeric_reproduction_fail",
        "method": method,
        "images": len(records),
        "metrics": metrics,
        "comparisons": comparisons,
        "numeric_pass": numeric_pass,
        "contact_sheet_pages_expected": pages,
        "visual_integrity_status": "pending_manual_review",
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_json(temporary / "METRICS.json", admission)
    atomic_json(temporary / "NUMERIC_ADMISSION.json", admission)
    temporary.rename(final)
    print(admission, flush=True)
    if not numeric_pass:
        raise SystemExit(5)


def verify_contacts(method, config):
    attempt = Path(config["attempt_root"])
    aggregate_dir = attempt / "AGGREGATE"
    contact_dir = attempt / "CONTACT_SHEETS"
    expected = int(config["contact_sheet_pages"])
    missing = []
    for page in range(expected):
        for suffix in ("jpg", "json"):
            path = contact_dir / ("page_%04d.%s" % (page, suffix))
            if not path.is_file() or path.stat().st_size == 0:
                missing.append(str(path))
    result = {
        "status": "contact_sheets_complete" if not missing else "contact_sheets_incomplete",
        "method": method,
        "expected_pages": expected,
        "missing": missing,
        "visual_integrity_status": "pending_manual_review",
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }
    atomic_json(aggregate_dir / "CONTACT_SHEET_INTEGRITY.json", result)
    if missing:
        raise SystemExit(6)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", choices=("sld", "safree"), required=True)
    parser.add_argument("command", choices=("aggregate", "verify-contacts"))
    args = parser.parse_args()
    config = load_json(args.config)
    if args.command == "aggregate":
        aggregate(args.method, config)
    else:
        verify_contacts(args.method, config)


if __name__ == "__main__":
    main()
