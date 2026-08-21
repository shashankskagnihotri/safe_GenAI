#!/usr/bin/env bash
set -euo pipefail

TASK_ID=$1
METHODS_ROOT=${PUSH_FOR_ICLR_EXECUTION_ROOT:?PUSH_FOR_ICLR_EXECUTION_ROOT must identify the frozen submitted worktree}
MAIN_ROOT=/ceph/sagnihot/projects/safety_genAI
MATRIX=$METHODS_ROOT/scripts/push_for_iclr/reproduction_env_matrix.tsv
ENV_ROOT=$MAIN_ROOT/outputs/PUSH_FOR_ICLR/REPRODUCTIONS/ENVIRONMENTS
CONDA=/ceph/sagnihot/miniconda3/bin/conda

test -d "$METHODS_ROOT"

row=$(awk -F '	' -v id="$TASK_ID" 'NR > 1 && $1 == id {print; found=1} END {if (!found) exit 2}' "$MATRIX")
IFS=$'	' read -r matrix_id method repo commit pyver repair_profile <<< "$row"
test "$matrix_id" = "$TASK_ID"
test -d "$repo"
actual_commit=$(git -C "$repo" rev-parse HEAD)
test "$actual_commit" = "$commit"

short_commit=$(printf '%s' "$commit" | cut -c1-12)
method_root=$ENV_ROOT/$method
final=$method_root/$short_commit
tmp=$method_root/.building_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
failed=$method_root/FAILED_BUILD_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "$method_root"

if [ -f "$final/ADMISSION.json" ]; then
  printf 'Environment already admitted: %s\n' "$final"
  exit 0
fi
if [ -e "$final" ] || [ -e "$tmp" ] || [ -e "$failed" ]; then
  printf 'Refusing to overwrite an existing non-admitted build path.\n' >&2
  exit 3
fi

mkdir -p "$tmp"
export CONDA_PKGS_DIRS=$tmp/conda-pkgs
mkdir -p "$CONDA_PKGS_DIRS"
success=0
on_exit() {
  rc=$?
  if [ "$success" -ne 1 ] && [ -d "$tmp" ]; then
    printf 'Environment build failed with exit code %s\n' "$rc" > "$tmp/FAILURE.txt"
    mv "$tmp" "$failed"
  fi
  exit "$rc"
}
trap on_exit EXIT

printf '%s\n' "$row" > "$tmp/MATRIX_ROW.tsv"
printf '%s\n' "$actual_commit" > "$tmp/UPSTREAM_COMMIT.txt"
sha256sum "$repo/requirements.txt" > "$tmp/UPSTREAM_REQUIREMENTS.sha256"
cp "$repo/requirements.txt" "$tmp/requirements.upstream.txt"

"$CONDA" create --prefix "$tmp/env" "python=$pyver" pip -y
PY=$tmp/env/bin/python
PIP=$tmp/env/bin/pip
export PIP_DISABLE_PIP_VERSION_CHECK=1
export PIP_NO_CACHE_DIR=1

case "$method" in
  sld)
    cat > "$tmp/REPAIR_NOTES.txt" <<'EOF'
The upstream commit leaves every runtime dependency unbounded. To reconstruct a
2023-compatible environment, this build pins the contemporary Diffusers 0.20.2,
Transformers 4.31.0, Accelerate 0.21.0, Torch 2.0.1, Torchvision 0.15.2, and
Pillow 9.5.0. Its legacy setup.py imports pkg_resources, so installation
requires the April 2023 packaging stack: pip 23.1.2, setuptools 67.7.2, and
wheel 0.40.0. A self-contained wheel is built without isolation instead of
leaving an editable link to the source checkout.
The numerical paper-row gate, not this inference alone, determines admission.
EOF
    "$PIP" install pip==23.1.2 setuptools==67.7.2 wheel==0.40.0
    "$PIP" install torch==2.0.1 torchvision==0.15.2
    "$PIP" install diffusers==0.20.2 transformers==4.31.0 accelerate==0.21.0 Pillow==9.5.0
    mkdir -p "$tmp/wheels"
    "$PIP" wheel --no-build-isolation --no-deps --wheel-dir "$tmp/wheels" "$repo"
    sld_wheel=$(find "$tmp/wheels" -maxdepth 1 -type f -name 'sld-0.0.1-*.whl' -print -quit)
    test -n "$sld_wheel"
    "$PIP" install --no-deps "$sld_wheel"
    ;;
  safree)
    OPENAI_CLIP_COMMIT=d05afc436d78f1c48dc0dbf8e5980a9d471f35f6
    SAFREE_SLD_REPO=/ceph/sagnihot/projects/safety_genAI/debugging/t2i_safety_27_july/upstream/repos/safe-latent-diffusion
    SAFREE_SLD_COMMIT=a42923c3de0e4346bee3f61891a510bdcc2aedd2
    test "$(git -C "$SAFREE_SLD_REPO" rev-parse HEAD)" = "$SAFREE_SLD_COMMIT"
    awk '
      $0 == "Pillow==9.5.0" {next}
      $0 == "clip==1.0" {next}
      $0 == "skimage==0.0" {print "scikit-image==0.25.0"; next}
      $0 == "sld==0.0.1" {next}
      $0 ~ /^torch==/ {next}
      $0 ~ /^torchvision==/ {next}
      {print}
    ' "$repo/requirements.txt" > "$tmp/requirements.resolved.txt"
    cat > "$tmp/REPAIR_NOTES.txt" <<'EOF'
The official requirements are unsatisfiable because Pillow 9.5.0 and 10.4.0 are
both exact pins. The later 10.4.0 pin is retained. Torch and Torchvision are
installed from the official CUDA 11.8 index because the +cu118 pins are not on
the default package index. The declared clip==1.0 is not published on PyPI, while
generate_safree.py calls the API implemented by OpenAI/CLIP; that official source
is therefore installed at commit d05afc436d78f1c48dc0dbf8e5980a9d471f35f6.
The skimage==0.0 placeholder explicitly directs users to the scikit-image
distribution; version 0.25.0 is the release contemporary with this January 2025
commit. The unavailable sld==0.0.1 distribution is built from official Safe
Latent Diffusion commit a42923c3de0e4346bee3f61891a510bdcc2aedd2, whose package
metadata declares that exact version.
No runtime method code is changed.
EOF
    printf 'openai_clip\thttps://github.com/openai/CLIP.git\t%s\n' "$OPENAI_CLIP_COMMIT" > "$tmp/UPSTREAM_AUXILIARY_LOCKS.tsv"
    printf 'safe_latent_diffusion\t%s\t%s\n' "$SAFREE_SLD_REPO" "$SAFREE_SLD_COMMIT" >> "$tmp/UPSTREAM_AUXILIARY_LOCKS.tsv"
    "$PIP" install --extra-index-url https://download.pytorch.org/whl/cu118 torch==2.4.0+cu118 torchvision==0.19.0+cu118
    "$PIP" install -r "$tmp/requirements.resolved.txt"
    "$PIP" install setuptools==67.7.2 wheel==0.40.0
    mkdir -p "$tmp/wheels"
    "$PIP" wheel --no-build-isolation --no-deps --wheel-dir "$tmp/wheels" "$SAFREE_SLD_REPO"
    sld_wheel=$(find "$tmp/wheels" -maxdepth 1 -type f -name 'sld-0.0.1-*.whl' -print -quit)
    test -n "$sld_wheel"
    "$PIP" install --no-deps "$sld_wheel"
    "$PIP" install --no-deps "git+https://github.com/openai/CLIP.git@$OPENAI_CLIP_COMMIT"
    ;;
  stg)
    printf '%s\n' 'No dependency repair; installing the exact released requirements.' > "$tmp/REPAIR_NOTES.txt"
    "$PIP" install -r "$repo/requirements.txt"
    ;;
  nag)
    cat > "$tmp/REPAIR_NOTES.txt" <<'EOF'
The official release leaves all dependencies unpinned. This build reconstructs
a release-date-compatible stack: Torch 2.6.0 CUDA 12.4, Diffusers 0.33.1,
Transformers 4.51.3, Accelerate 1.6.0, and SentencePiece 0.2.0. The full lock is
captured and the COCO-5K paper metrics remain the admission authority.
EOF
    "$PIP" install --extra-index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
    "$PIP" install diffusers==0.33.1 transformers==4.51.3 accelerate==1.6.0 sentencepiece==0.2.0
    "$PIP" install --no-deps -e "$repo"
    ;;
  semantic_surgery)
    printf '%s\n' 'No dependency repair; installing the exact released requirements.' > "$tmp/REPAIR_NOTES.txt"
    "$PIP" install -r "$repo/requirements.txt"
    ;;
  gloce)
    sed 's/^xformers$/xformers==0.0.20/' "$repo/requirements.txt" > "$tmp/requirements.resolved.txt"
    cat > "$tmp/REPAIR_NOTES.txt" <<'EOF'
The release pins Torch 2.0.1 but leaves xformers unbounded. Current xformers
would replace Torch and invalidate the paper environment, so xformers 0.0.20,
the release compatible with Torch 2.0.1, is pinned. No method code is changed.
EOF
    "$PIP" install -r "$tmp/requirements.resolved.txt"
    ;;
  adavd)
    cat > "$tmp/REPAIR_NOTES.txt" <<'EOF'
The README requests Python 3.9, but the released requirements pin NumPy 2.2.3,
which requires Python 3.10 or newer. Python 3.10 is used so the exact package pin
is retained; no method code or package version is changed.
EOF
    "$PIP" install -r "$repo/requirements.txt"
    ;;
  groce)
    cat > "$tmp/REPAIR_NOTES.txt" <<'EOF'
The release omits Torch from requirements. Torch 2.6.0 CUDA 12.4 is supplied as
a compatibility reconstruction for Diffusers 0.34.0 and Python 3.11. The exact
resolved lock is retained and paper metrics determine admission.
EOF
    "$PIP" install --extra-index-url https://download.pytorch.org/whl/cu124 torch==2.6.0
    "$PIP" install -r "$repo/requirements.txt"
    ;;
  nlce)
    printf '%s\n' 'The exact Torch +cu124 pin is resolved from the official PyTorch CUDA 12.4 index.' > "$tmp/REPAIR_NOTES.txt"
    "$PIP" install --extra-index-url https://download.pytorch.org/whl/cu124 -r "$repo/requirements.txt"
    ;;
  des)
    cat > "$tmp/REPAIR_NOTES.txt" <<'EOF'
The release omits Torch and Torchvision. Torch 2.4.1 and Torchvision 0.19.1 from
the official CUDA 12.1 index are supplied to match the late-2024 dependency
stack used by Diffusers 0.32.2. Paper metrics determine final admission.
EOF
    "$PIP" install --extra-index-url https://download.pytorch.org/whl/cu121 torch==2.4.1 torchvision==0.19.1
    "$PIP" install -r "$repo/requirements.txt"
    ;;
  *)
    printf 'Unknown method: %s\n' "$method" >&2
    exit 4
    ;;
esac

"$PIP" freeze --all | LC_ALL=C sort > "$tmp/pip-freeze.txt"
"$CONDA" list --prefix "$tmp/env" --explicit > "$tmp/conda-explicit.txt"
"$PY" - "$method" "$commit" "$repair_profile" > "$tmp/VERSIONS.json" <<'PY'
import importlib.metadata
import json
import platform
import sys

method, commit, repair = sys.argv[1:]
names = [
    "torch", "torchvision", "diffusers", "transformers", "accelerate",
    "numpy", "pandas", "Pillow", "xformers", "ultralytics", "nudenet", "clip", "setuptools"
]
versions = {}
for name in names:
    try:
        versions[name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        versions[name] = None
print(json.dumps({
    "status": "built",
    "method": method,
    "upstream_commit": commit,
    "repair_profile": repair,
    "python": platform.python_version(),
    "packages": versions,
}, indent=2, sort_keys=True))
PY
cat > "$tmp/ADMISSION.json" <<EOF
{
  "status": "environment_built_not_paper_admitted",
  "method": "$method",
  "upstream_commit": "$commit",
  "repair_profile": "$repair_profile",
  "slurm_array_job_id": "$SLURM_ARRAY_JOB_ID",
  "slurm_array_task_id": "$SLURM_ARRAY_TASK_ID"
}
EOF
mv "$tmp" "$final"
success=1
printf 'Environment admitted for execution: %s\n' "$final"
