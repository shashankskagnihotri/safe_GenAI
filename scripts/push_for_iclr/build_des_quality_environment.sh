#!/usr/bin/env bash
set -euo pipefail

ROOT="${PUSH_FOR_ICLR_EXECUTION_ROOT:?PUSH_FOR_ICLR_EXECUTION_ROOT is required}"
EXPECTED_COMMIT="${PUSH_FOR_ICLR_EXPECTED_COMMIT:?PUSH_FOR_ICLR_EXPECTED_COMMIT is required}"
CONFIG="$ROOT/configs/experiments/push_for_iclr/des_sd15_quality_v1.json"
CONDA=/ceph/sagnihot/miniconda3/bin/conda
PRIMARY=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/REPRODUCTIONS/ENVIRONMENTS/des/1d77e0a2a720/env
PARENT=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/REPRODUCTIONS/ENVIRONMENTS/des_quality/1d77e0a2a720_532229f679d7_dcba3cb2e282
FINAL="$PARENT/env"
BUILDS="$PARENT/builds"
BUILD="$BUILDS/build_${SLURM_JOB_ID:-manual}_$(date -u +%Y%m%dT%H%M%SZ)"
DES=/ceph/sagnihot/projects/safety_genAI_upstreams/PUSH_FOR_ICLR/DES
T2I=/ceph/sagnihot/projects/safety_genAI_upstreams/PUSH_FOR_ICLR/text2image-benchmark
CLIP=/ceph/sagnihot/projects/safety_genAI_upstreams/PUSH_FOR_ICLR/openai_CLIP

cd "$ROOT"
test "$(git rev-parse HEAD)" = "$EXPECTED_COMMIT"
test "$(git -C "$DES" rev-parse HEAD)" = 1d77e0a2a72023470f201d3149e80429f7b60e15
test "$(git -C "$T2I" rev-parse HEAD)" = 532229f679d7e97ecba61914db7276f95733e707
test "$(git -C "$CLIP" rev-parse HEAD)" = dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1
test -x "$PRIMARY/bin/python"
mkdir -p "$PARENT" "$BUILDS"

if [[ -L "$FINAL" ]]; then
  test -x "$FINAL/bin/python"
  test -f "$PARENT/ENVIRONMENT_ADMISSION.json"
  "$FINAL/bin/python" scripts/push_for_iclr/run_des_sd15_quality.py \
    --config "$CONFIG" validate-static
  exit 0
fi
test ! -e "$FINAL"

"$CONDA" create --prefix "$BUILD" --clone "$PRIMARY" -y
"$BUILD/bin/python" -m pip install --no-input datasets==2.21.0 glob2==0.7
"$BUILD/bin/python" -m pip check

PYTHONPATH="$T2I:$CLIP" "$BUILD/bin/python" - <<'PY'
import importlib
import pathlib

expected = {
    "clip": pathlib.Path("/ceph/sagnihot/projects/safety_genAI_upstreams/PUSH_FOR_ICLR/openai_CLIP"),
    "T2IBenchmark": pathlib.Path("/ceph/sagnihot/projects/safety_genAI_upstreams/PUSH_FOR_ICLR/text2image-benchmark"),
}
for module_name in ("clip", "datasets", "glob2", "T2IBenchmark"):
    module = importlib.import_module(module_name)
    origin = pathlib.Path(module.__file__).resolve()
    if module_name in expected and expected[module_name] not in origin.parents:
        raise RuntimeError(f"{module_name} imported from unpinned path {origin}")
PY

"$BUILD/bin/python" scripts/push_for_iclr/run_des_sd15_quality.py \
  --config "$CONFIG" validate-static
"$BUILD/bin/python" -m pip freeze --all > "$PARENT/ENVIRONMENT_FREEZE.txt"

BUILD_PATH="$BUILD" FINAL_PATH="$FINAL" EXPECTED_COMMIT_VALUE="$EXPECTED_COMMIT" \
  "$BUILD/bin/python" - <<'PY'
import hashlib
import json
import os
import pathlib
import platform
from datetime import datetime, timezone

freeze = pathlib.Path(os.environ["FINAL_PATH"]).parent / "ENVIRONMENT_FREEZE.txt"
payload = {
    "status": "ENVIRONMENT_ADMITTED",
    "build_path": os.environ["BUILD_PATH"],
    "environment_link": os.environ["FINAL_PATH"],
    "implementation_commit": os.environ["EXPECTED_COMMIT_VALUE"],
    "python": platform.python_version(),
    "freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
    "created_at": datetime.now(timezone.utc).isoformat(),
    "fallback_used": False,
}
target = pathlib.Path(os.environ["FINAL_PATH"]).parent / "ENVIRONMENT_ADMISSION.json"
temporary = target.with_suffix(".json.tmp")
temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
temporary.replace(target)
PY

ln -s "$BUILD" "$PARENT/.env_${SLURM_JOB_ID:-manual}.tmp"
mv "$PARENT/.env_${SLURM_JOB_ID:-manual}.tmp" "$FINAL"
test -x "$FINAL/bin/python"
