#!/usr/bin/env bash
# Fail-closed environment dispatch for the selected-common-seed Flux v2 cohort.

set -euo pipefail

PROJECT_ROOT="/ceph/sagnihot/projects/safety_genAI"
cd "${PROJECT_ROOT}"

MANIFEST="${1:?Usage: run_flux1_native_negative_calibration_v2_dispatched.sh MANIFEST INDEX}"
INDEX="${2:?Usage: run_flux1_native_negative_calibration_v2_dispatched.sh MANIFEST INDEX}"
if [ "$#" -ne 2 ]; then
  echo "Expected exactly MANIFEST and INDEX; got $# arguments." >&2
  exit 2
fi

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/ceph/sagnihot/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/ceph/sagnihot/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda.sh not found; authenticated calibration-v2 dispatch is impossible." >&2
  exit 1
fi

export PYTHONPATH="${PROJECT_ROOT}/src:${PROJECT_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TOKENIZERS_PARALLELISM=false

_activate_conda_environment_nounset_safe() {
  local environment_name="${1:?environment name is required}"
  local restore_nounset=0
  local activation_status=0
  if [[ "$-" == *u* ]]; then
    restore_nounset=1
    set +u
  fi
  conda activate "${environment_name}" || activation_status=$?
  if (( restore_nounset )); then
    set -u
  fi
  return "${activation_status}"
}

if ! _activate_conda_environment_nounset_safe safe_genai_conceptsteer; then
  echo "Required conda env safe_genai_conceptsteer is absent or cannot be activated." >&2
  exit 1
fi

DISPATCH_RECORD="$(
  python scripts/finer_detailing_environment_dispatch.py resolve \
    --flux1-native-negative-calibration-v2 \
    --manifest "${MANIFEST}" \
    --index "${INDEX}" \
    --project-root "${PROJECT_ROOT}"
)"
if [[ "${DISPATCH_RECORD}" == *$'\n'* ]]; then
  echo "Authenticated calibration-v2 dispatcher returned more than one record." >&2
  exit 1
fi

TARGET_ENVIRONMENT=""
AUTHENTICATED_MODEL=""
EXPECTED_DIFFUSERS_REVISION=""
TRAILING_FIELD=""
IFS=$'\t' read -r \
  TARGET_ENVIRONMENT \
  AUTHENTICATED_MODEL \
  EXPECTED_DIFFUSERS_REVISION \
  TRAILING_FIELD <<<"${DISPATCH_RECORD}"
if [ -z "${TARGET_ENVIRONMENT}" ] || \
  [ -z "${AUTHENTICATED_MODEL}" ] || \
  [ -z "${EXPECTED_DIFFUSERS_REVISION}" ] || \
  [ -n "${TRAILING_FIELD}" ]; then
  echo "Malformed authenticated calibration-v2 dispatch record." >&2
  exit 1
fi

if ! _activate_conda_environment_nounset_safe "${TARGET_ENVIRONMENT}"; then
  echo "Required conda env ${TARGET_ENVIRONMENT} is absent or cannot be activated." >&2
  exit 1
fi

python scripts/finer_detailing_environment_dispatch.py verify-job \
  --flux1-native-negative-calibration-v2 \
  --manifest "${MANIFEST}" \
  --index "${INDEX}" \
  --project-root "${PROJECT_ROOT}" \
  --expected-model "${AUTHENTICATED_MODEL}" \
  --expected-environment "${TARGET_ENVIRONMENT}" \
  --expected-diffusers-revision "${EXPECTED_DIFFUSERS_REVISION}"

export HIERASAFE_REQUIRE_ENVIRONMENT_PREFLIGHT=1

exec python scripts/run_flux1_native_negative_calibration_v2.py \
  --manifest "${MANIFEST}" \
  --index "${INDEX}"
