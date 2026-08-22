#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
import shutil
import subprocess
from importlib import metadata
from pathlib import Path

from paper_i2p_common import atomic_json, load_json, sha256_file

CONDA = "/ceph/sagnihot/miniconda3/bin/conda"
SLD_CONFIG = "configs/experiments/push_for_iclr/sld_table1_hypmax_v1.json"
SAFREE_CONFIG = "configs/experiments/push_for_iclr/safree_table1_i2p_v1.json"


def run(command, **kwargs):
    print("+", " ".join(str(part) for part in command), flush=True)
    subprocess.run([str(part) for part in command], check=True, **kwargs)


def package_versions(python):
    code = """
import importlib.metadata, json, platform
names = ['torch','torchvision','diffusers','transformers','accelerate','numpy','scipy','Pillow','huggingface-hub','albumentations','albucore','onnxruntime','onnxruntime-gpu','nudenet','sld','python-sld','clip']
out = {'python': platform.python_version(), 'packages': {}}
for name in names:
    try: out['packages'][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError: out['packages'][name] = None
print(json.dumps(out, sort_keys=True))
"""
    return json.loads(subprocess.check_output([str(python), "-c", code], text=True))


def build(task_id):
    if task_id == 0:
        method = "sld"
        config = load_json(SLD_CONFIG)
        repair_notes = (
            "The admitted historical SLD environment is cloned immutably. NumPy 2.x "
            "cannot provide Torch 2.0.1's NumPy bridge, and current huggingface_hub "
            "removed cached_download required by Diffusers 0.20.2. Only NumPy 1.26.4 "
            "and huggingface_hub 0.16.4 are replaced. The published LMS scheduler "
            "requires Diffusers' optional SciPy backend, which the upstream package "
            "metadata omitted, so release-compatible SciPy 1.10.1 is added. Method "
            "code and paper settings are unchanged."
        )
    elif task_id == 1:
        method = "safree"
        config = load_json(SAFREE_CONFIG)
        repair_notes = (
            "The admitted author environment is cloned immutably. Albumentations "
            "1.4.14 declared albucore>=0.0.13 and resolved to incompatible 0.2.13; "
            "albucore 0.0.16 restores the released API. The author's exact CPU "
            "onnxruntime 1.18.1 package and repository-vendored "
            "nudenet/classify_pil.py implementation are preserved. Installing the "
            "PyPI nudenet package would shadow that implicit namespace package and "
            "is explicitly forbidden. SAFREE also pins Diffusers 0.29.0 while its "
            "unconditionally imported sld 0.0.1 dependency uses the removed "
            "diffusers.pipeline_utils path. That dependency import is migrated to "
            "diffusers.pipelines.pipeline_utils with before/after hashes; the SLD "
            "class is not exercised by the paper's erase_id=std SAFREE route. No "
            "SAFREE method code changes."
        )
    else:
        raise ValueError("task-id must be 0 or 1")

    base = Path(config["environment"]["base_root"])
    final = Path(config["environment"]["compatible_root"])
    if not (base / "ADMISSION.json").is_file():
        raise RuntimeError("Base environment is not admitted: %s" % base)
    if (final / "ADMISSION.json").is_file():
        print("Compatibility environment already admitted:", final)
        return
    if final.exists():
        raise RuntimeError("Refusing to overwrite non-admitted path: %s" % final)

    tag = "%s_%s" % (os.environ.get("SLURM_ARRAY_JOB_ID", "local"), task_id)
    temporary = final.parent / (".building_compatible_" + tag)
    failed = final.parent / ("FAILED_COMPATIBLE_BUILD_" + tag)
    final.parent.mkdir(parents=True, exist_ok=True)
    if temporary.exists() or failed.exists():
        raise RuntimeError("Build evidence path already exists")
    temporary.mkdir()
    success = False
    try:
        (temporary / "REPAIR_NOTES.txt").write_text(repair_notes + "\n", encoding="utf-8")
        shutil.copy2(base / "ADMISSION.json", temporary / "BASE_ADMISSION.json")
        atomic_json(temporary / "BASE_IDENTITY.json", {
            "base_root": str(base),
            "base_admission_sha256": sha256_file(base / "ADMISSION.json"),
            "method": method,
        })
        run([CONDA, "create", "--prefix", temporary / "env", "--clone", base / "env", "-y"])
        python = temporary / "env/bin/python"
        if method == "sld":
            run([python, "-m", "pip", "install", "--no-deps", "--force-reinstall",
                 "numpy==1.26.4", "huggingface_hub==0.16.4", "scipy==1.10.1"])
            probe = (
                "import numpy, scipy, torch, huggingface_hub; "
                "from huggingface_hub import cached_download; from sld import SLDPipeline; "
                "assert torch.arange(3).numpy().tolist()==[0,1,2]; "
                "assert numpy.__version__=='1.26.4'; assert scipy.__version__=='1.10.1'; "
                "print('SLD_COMPATIBILITY_OK')"
            )
            run([python, "-c", probe])
        else:
            run([python, "-m", "pip", "install", "--no-deps", "--force-reinstall",
                 "albucore==0.0.16"])
            purelib = Path(subprocess.check_output(
                [str(python), "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                text=True,
            ).strip())
            sld_pipeline = purelib / "sld/sld_pipeline.py"
            old_import = "from diffusers.pipeline_utils import DiffusionPipeline"
            new_import = "from diffusers.pipelines.pipeline_utils import DiffusionPipeline"
            source_text = sld_pipeline.read_text(encoding="utf-8")
            if source_text.count(old_import) != 1:
                raise RuntimeError("Unexpected sld 0.0.1 Diffusers import surface")
            sld_before_sha256 = sha256_file(sld_pipeline)
            sld_pipeline.write_text(source_text.replace(old_import, new_import), encoding="utf-8")
            atomic_json(temporary / "DEPENDENCY_COMPATIBILITY.json", {
                "distribution": "sld==0.0.1",
                "file": str(sld_pipeline.relative_to(temporary)),
                "reason": "SAFREE pins Diffusers 0.29.0, which moved pipeline_utils under diffusers.pipelines",
                "old_import": old_import,
                "new_import": new_import,
                "before_sha256": sld_before_sha256,
                "after_sha256": sha256_file(sld_pipeline),
                "safree_route": "erase_id=std; ModifiedSLDPipeline imported but not instantiated",
            })
            source = str(Path(config["upstream"]["repo"]).resolve())
            probe = f"""
import importlib.metadata as metadata
from pathlib import Path
import albumentations, albucore, onnxruntime, torch
import nudenet.classify_pil as classify_pil
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from sld.sld_pipeline import DiffusionPipeline as SLDDiffusionPipeline
assert albucore.__version__ == '0.0.16'
assert metadata.version('onnxruntime') == '1.18.1'
assert metadata.version('sld') == '0.0.1'
assert metadata.version('diffusers') == '0.29.0'
assert SLDDiffusionPipeline is DiffusionPipeline
for forbidden in ('onnxruntime-gpu', 'nudenet'):
    try:
        metadata.version(forbidden)
    except metadata.PackageNotFoundError:
        pass
    else:
        raise AssertionError('Forbidden shadowing package installed: ' + forbidden)
expected = Path({source!r}) / 'nudenet/classify_pil.py'
assert Path(classify_pil.__file__).resolve() == expected.resolve()
assert 'CPUExecutionProvider' in onnxruntime.get_available_providers()
assert torch.arange(3).numpy().tolist() == [0, 1, 2]
import generate_safree
print('SAFREE_COMPATIBILITY_OK')
"""
            environment = os.environ.copy()
            environment["PYTHONPATH"] = config["upstream"]["repo"]
            environment["NO_ALBUMENTATIONS_UPDATE"] = "1"
            run([python, "-c", probe], cwd=config["upstream"]["repo"], env=environment)

        freeze = subprocess.check_output([str(python), "-m", "pip", "freeze", "--all"], text=True)
        (temporary / "pip-freeze.txt").write_text(
            "\n".join(sorted(line for line in freeze.splitlines() if line)) + "\n",
            encoding="utf-8",
        )
        versions = package_versions(python)
        versions.update({"method": method, "repair_notes": repair_notes})
        atomic_json(temporary / "VERSIONS.json", versions)
        atomic_json(temporary / "ADMISSION.json", {
            "status": "runtime_compatible_not_paper_admitted",
            "method": method,
            "base_root": str(base),
            "repair_profile": final.name,
            "slurm_array_job_id": os.environ.get("SLURM_ARRAY_JOB_ID"),
            "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
            "versions": versions,
        })
        temporary.rename(final)
        success = True
        print("Compatibility environment admitted for runtime:", final)
    finally:
        if not success and temporary.exists():
            (temporary / "FAILURE.txt").write_text("Compatibility build failed.\n", encoding="utf-8")
            temporary.rename(failed)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", type=int, required=True)
    args = parser.parse_args()
    build(args.task_id)


if __name__ == "__main__":
    main()
