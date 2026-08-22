#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def submit(root, expected_commit, script, dependency=None):
    command = [
        "sbatch", "--parsable",
        "--export=ALL,PUSH_FOR_ICLR_EXECUTION_ROOT=%s,PUSH_FOR_ICLR_EXPECTED_COMMIT=%s" % (
            root, expected_commit),
    ]
    if dependency:
        command.append("--dependency=afterok:%s" % dependency)
    command.append(str(root / script))
    print("+", " ".join(command), flush=True)
    output = subprocess.check_output(command, cwd=str(root), text=True).strip()
    return output.split(";", 1)[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[2]))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(root), text=True)
    if status.strip():
        raise RuntimeError("DES execution worktree must be clean before submission")
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(root), text=True).strip()
    jobs = {}
    jobs["preflight"] = submit(root, commit, "slurm/push_for_iclr_des_preflight.sbatch")
    jobs["codebook"] = submit(root, commit, "slurm/push_for_iclr_des_codebook.sbatch", jobs["preflight"])
    jobs["training"] = submit(root, commit, "slurm/push_for_iclr_des_train.sbatch", jobs["codebook"])
    jobs["generation"] = submit(root, commit, "slurm/push_for_iclr_des_generate.sbatch", jobs["training"])
    jobs["nudenet"] = submit(root, commit, "slurm/push_for_iclr_des_nudenet.sbatch", jobs["generation"])
    jobs["contacts"] = submit(root, commit, "slurm/push_for_iclr_des_contacts.sbatch", jobs["nudenet"])
    jobs["contacts_verify"] = submit(root, commit, "slurm/push_for_iclr_des_contacts_verify.sbatch", jobs["contacts"])
    jobs["aggregate"] = submit(root, commit, "slurm/push_for_iclr_des_aggregate.sbatch", jobs["contacts_verify"])
    registry = {
        "method": "DES", "worktree": str(root), "commit": commit, "jobs": jobs,
        "nice": 10000, "array_throttles": None,
        "quality_jobs_submitted": False,
        "quality_block": "Exact FID assets and exact CLIP-L/14 checkpoint are not admitted.",
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    path = Path("/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_REGISTRY")
    path.mkdir(parents=True, exist_ok=True)
    output = path / ("des_sd15_%s.json" % jobs["preflight"])
    temporary = output.with_name(output.name + ".tmp.%s" % os.getpid())
    temporary.write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(str(temporary), str(output))
    print(json.dumps(registry, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
