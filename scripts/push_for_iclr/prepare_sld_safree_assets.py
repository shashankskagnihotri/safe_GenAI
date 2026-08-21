#!/usr/bin/env python3
import argparse
import os
import shutil
import ssl
import urllib.request
from pathlib import Path

from paper_i2p_common import assert_git_commit, assert_sha256, atomic_json, load_json, sha256_file


def download(url, destination):
    request = urllib.request.Request(url, headers={"User-Agent": "PUSH_FOR_ICLR-exact-reproduction/1"})
    context = ssl.create_default_context()
    with urllib.request.urlopen(request, context=context, timeout=120) as response:
        with Path(destination).open("wb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sld-config", required=True)
    parser.add_argument("--safree-config", required=True)
    args = parser.parse_args()
    sld = load_json(args.sld_config)
    safree = load_json(args.safree_config)
    final = Path(sld["assets"]["root"])
    if final != Path(safree["assets"]["root"]):
        raise RuntimeError("SLD and SAFREE must share one admitted asset root")
    if (final / "ADMISSION.json").is_file():
        print("Assets already admitted:", final)
        return
    if final.exists():
        raise RuntimeError("Refusing to overwrite non-admitted asset path: %s" % final)
    tag = os.environ.get("SLURM_JOB_ID", "local")
    temporary = final.parent / (".building_sld_safree_i2p_v1_" + tag)
    failed = final.parent / ("FAILED_sld_safree_i2p_v1_" + tag)
    final.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists() or failed.exists():
        raise RuntimeError("Asset evidence path already exists")
    temporary.mkdir()
    success = False
    try:
        assert_git_commit(sld["upstream"]["repo"], sld["upstream"]["commit"])
        assert_git_commit(sld["official_evaluator"]["repo"], sld["official_evaluator"]["commit"])
        assert_git_commit(safree["upstream"]["repo"], safree["upstream"]["commit"])
        assert_sha256(sld["dataset"]["source_path"], sld["dataset"]["sha256"])
        assert_sha256(sld["assets"]["q16_prompts_source"], sld["assets"]["q16_prompts_sha256"])
        assert_sha256(safree["assets"]["nudenet_classifier_source"], safree["assets"]["nudenet_classifier_sha256"])
        assert_sha256(safree["upstream"]["config_path"], safree["upstream"]["config_sha256"])
        assert_sha256(Path(sld["upstream"]["repo"]) / "src/sld/sld_pipeline.py", sld["upstream"]["pipeline_sha256"])
        assert_sha256(Path(sld["official_evaluator"]["repo"]) / "eval/q16.py", sld["official_evaluator"]["q16_sha256"])
        assert_sha256(Path(sld["official_evaluator"]["repo"]) / "eval/nudenet_detector_gpu.py", sld["official_evaluator"]["nudenet_sha256"])
        assert_sha256(Path(safree["upstream"]["repo"]) / "generate_safree.py", safree["upstream"]["generator_sha256"])

        for model in (sld["model"], safree["model"]):
            snapshot = Path(model["snapshot"])
            if snapshot.name != model["revision"] or not (snapshot / "model_index.json").is_file():
                raise RuntimeError("Invalid frozen model snapshot: %s" % snapshot)
        clip_snapshot = Path(sld["assets"]["clip_cache"]) / (
            "models--openai--clip-vit-large-patch14/snapshots/" + sld["assets"]["clip_revision"]
        )
        if not (clip_snapshot / "config.json").is_file():
            raise RuntimeError("Frozen Q16 CLIP snapshot is incomplete: %s" % clip_snapshot)

        (temporary / "dataset").mkdir()
        shutil.copy2(sld["dataset"]["source_path"], temporary / "dataset/i2p.csv")
        (temporary / "eval").mkdir()
        shutil.copy2(sld["assets"]["q16_prompts_source"], temporary / "eval/Q16_prompts.p")
        (temporary / "safree").mkdir()
        shutil.copy2(safree["assets"]["nudenet_classifier_source"], temporary / "safree/nudenet_classifier_model.onnx")
        shutil.copy2(safree["upstream"]["config_path"], temporary / "safree/sd_config.json")
        detector_root = temporary / "home/.NudeNet"
        (detector_root / "classes").mkdir(parents=True)
        checkpoint = detector_root / "detector_v2_base_checkpoint.onnx"
        classes = detector_root / "classes/detector_v2_base_classes"
        download(sld["assets"]["nudenet_base_checkpoint_url"], checkpoint)
        download(sld["assets"]["nudenet_base_classes_url"], classes)
        if checkpoint.stat().st_size < 1024 * 1024 or classes.stat().st_size == 0:
            raise RuntimeError("Downloaded NudeNet base assets are incomplete")

        manifest = {
            "status": "assets_admitted_for_runtime",
            "dataset": {"path": "dataset/i2p.csv", "sha256": sha256_file(temporary / "dataset/i2p.csv")},
            "q16_prompts": {"path": "eval/Q16_prompts.p", "sha256": sha256_file(temporary / "eval/Q16_prompts.p")},
            "safree_classifier": {"path": "safree/nudenet_classifier_model.onnx", "sha256": sha256_file(temporary / "safree/nudenet_classifier_model.onnx")},
            "safree_config": {"path": "safree/sd_config.json", "sha256": sha256_file(temporary / "safree/sd_config.json")},
            "nudenet_base_checkpoint": {"url": sld["assets"]["nudenet_base_checkpoint_url"], "sha256": sha256_file(checkpoint), "bytes": checkpoint.stat().st_size},
            "nudenet_base_classes": {"url": sld["assets"]["nudenet_base_classes_url"], "sha256": sha256_file(classes), "bytes": classes.stat().st_size},
            "clip_snapshot": str(clip_snapshot),
            "source_commits": {
                "sld": sld["upstream"]["commit"],
                "official_i2p": sld["official_evaluator"]["commit"],
                "safree": safree["upstream"]["commit"],
            },
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        }
        atomic_json(temporary / "ASSET_MANIFEST.json", manifest)
        atomic_json(temporary / "ADMISSION.json", manifest)
        temporary.rename(final)
        success = True
        print("Assets admitted:", final)
    finally:
        if not success and temporary.exists():
            (temporary / "FAILURE.txt").write_text("Asset admission failed.\n", encoding="utf-8")
            temporary.rename(failed)


if __name__ == "__main__":
    main()
