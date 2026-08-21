#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/ceph/sagnihot/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/ceph/sagnihot/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda.sh not found; cannot create the frozen environments." >&2
  exit 1
fi

# These are intentionally separate environments: their source-critical
# Diffusers commits are mutually exclusive. `conda env create` also refuses to
# mutate an existing environment, keeping this setup operation fail-closed.
conda env create --file environment.yml
conda env create --file environment-ltx23.yml

for environment in safe_genai_conceptsteer safe_genai_ltx23; do
  conda run --no-capture-output -n "${environment}" \
    python -m pip install --no-deps -e .
  conda run --no-capture-output -n "${environment}" \
    python scripts/finer_detailing_environment_dispatch.py verify-install \
      --environment "${environment}"
done

conda run --no-capture-output -n safe_genai_conceptsteer python -m compileall src scripts tests
conda run --no-capture-output -n safe_genai_conceptsteer python -m pytest
conda run --no-capture-output -n safe_genai_ltx23 \
  python -m pytest \
    tests/test_finer_detailing_environment_dispatch.py \
    tests/test_ltx_temporal_protocol.py
