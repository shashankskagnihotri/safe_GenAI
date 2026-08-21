#!/usr/bin/env bash
# Dedicated fail-closed environment dispatcher for atomic Q1/Q2 cohorts.

set -euo pipefail

PROJECT_ROOT="/ceph/sagnihot/projects/safety_genAI"
cd "${PROJECT_ROOT}"

PLAN="${1:?Usage: run_finer_detailing_qualification_dispatched.sh PLAN ROLE REGISTRY INDEX}"
ROLE="${2:?ROLE is required}"
REGISTRY="${3:?REGISTRY is required}"
INDEX="${4:?INDEX is required}"
if [ "$#" -ne 4 ]; then
  echo "Expected exactly PLAN ROLE REGISTRY INDEX; got $# arguments." >&2
  exit 2
fi

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/ceph/sagnihot/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/ceph/sagnihot/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda.sh not found; qualification dispatch is impossible." >&2
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
  echo "Bootstrap conda env safe_genai_conceptsteer cannot be activated." >&2
  exit 1
fi

DISPATCH_RECORD="$(
  python scripts/finer_detailing_qualification_dispatch.py resolve \
    --plan "${PLAN}" \
    --role "${ROLE}" \
    --registry "${REGISTRY}" \
    --index "${INDEX}" \
    --project-root "${PROJECT_ROOT}"
)"
if [[ "${DISPATCH_RECORD}" == *$'\n'* ]]; then
  echo "Qualification dispatcher returned more than one record." >&2
  exit 1
fi

TARGET_ENVIRONMENT=""
AUTHENTICATED_MODEL=""
EXPECTED_DIFFUSERS_REVISION=""
AUTHENTICATED_MANIFEST=""
TRAILING_FIELD=""
IFS=$'\t' read -r \
  TARGET_ENVIRONMENT \
  AUTHENTICATED_MODEL \
  EXPECTED_DIFFUSERS_REVISION \
  AUTHENTICATED_MANIFEST \
  TRAILING_FIELD <<<"${DISPATCH_RECORD}"
if [ -z "${TARGET_ENVIRONMENT}" ] || \
  [ -z "${AUTHENTICATED_MODEL}" ] || \
  [ -z "${EXPECTED_DIFFUSERS_REVISION}" ] || \
  [ -z "${AUTHENTICATED_MANIFEST}" ] || \
  [ -n "${TRAILING_FIELD}" ]; then
  echo "Malformed qualification environment dispatch record." >&2
  exit 1
fi

if ! _activate_conda_environment_nounset_safe "${TARGET_ENVIRONMENT}"; then
  echo "Required conda env ${TARGET_ENVIRONMENT} cannot be activated." >&2
  exit 1
fi

python scripts/finer_detailing_qualification_dispatch.py verify-job \
  --plan "${PLAN}" \
  --role "${ROLE}" \
  --registry "${REGISTRY}" \
  --index "${INDEX}" \
  --project-root "${PROJECT_ROOT}" \
  --expected-model "${AUTHENTICATED_MODEL}" \
  --expected-environment "${TARGET_ENVIRONMENT}" \
  --expected-diffusers-revision "${EXPECTED_DIFFUSERS_REVISION}" \
  --expected-manifest "${AUTHENTICATED_MANIFEST}"

export HIERASAFE_REQUIRE_ENVIRONMENT_PREFLIGHT=1
export HIERASAFE_REQUIRE_QUALIFICATION_LAUNCH_AUTHORIZATION=1

exec python scripts/run_finer_detailing_correction.py \
  --manifest "${AUTHENTICATED_MANIFEST}" \
  --index "${INDEX}"
