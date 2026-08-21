#!/usr/bin/env bash
# Fail-closed launcher for one authenticated production campaign array task.

set -euo pipefail

PROJECT_ROOT="/ceph/sagnihot/projects/safety_genAI"
cd "${PROJECT_ROOT}"

KIND="${1:?Usage: run_finer_detailing_campaign_dispatched.sh KIND ROLE INDEX REGISTRY}"
ROLE="${2:?Usage: run_finer_detailing_campaign_dispatched.sh KIND ROLE INDEX REGISTRY}"
INDEX="${3:?Usage: run_finer_detailing_campaign_dispatched.sh KIND ROLE INDEX REGISTRY}"
REGISTRY="${4:?Usage: run_finer_detailing_campaign_dispatched.sh KIND ROLE INDEX REGISTRY}"
if [ "$#" -ne 4 ]; then
  echo "Expected exactly KIND ROLE INDEX REGISTRY; got $# arguments." >&2
  exit 2
fi

if ! [[ "${SLURM_ARRAY_JOB_ID:-}" =~ ^[0-9]+$ ]] || \
  ! [[ "${SLURM_ARRAY_TASK_ID:-}" =~ ^[0-9]+$ ]] || \
  ! [[ "${SLURM_JOB_ID:-}" =~ ^[0-9]+$ ]] || \
  ! [[ "${INDEX}" =~ ^[0-9]+$ ]] || \
  ! [[ "${ROLE}" =~ ^[a-z0-9_]+$ ]]; then
  echo "Dedicated campaign launch requires numeric Slurm IDs/index and a safe role." >&2
  exit 1
fi
if [ "${INDEX}" != "${SLURM_ARRAY_TASK_ID}" ]; then
  echo "Campaign argument index differs from SLURM_ARRAY_TASK_ID." >&2
  exit 1
fi
case "${KIND}" in
  seed_ladder)
    [[ "${SLURM_JOB_NAME:-}" =~ ^fd-ladder-[0-9][0-9]$ ]] || {
      echo "Seed-ladder task has a noncanonical job name." >&2
      exit 1
    }
    ;;
  selected_seed_final)
    [[ "${SLURM_JOB_NAME:-}" =~ ^fd-final-[0-9][0-9]-(std|shp)$ ]] || {
      echo "Final campaign task has a noncanonical job name." >&2
      exit 1
    }
    ;;
  *)
    echo "Unknown campaign cohort kind ${KIND}." >&2
    exit 1
    ;;
esac

if [ -f "${HOME}/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "${HOME}/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/ceph/sagnihot/miniconda3/etc/profile.d/conda.sh" ]; then
  # shellcheck disable=SC1091
  source "/ceph/sagnihot/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda.sh not found; authenticated campaign dispatch is impossible." >&2
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
  python scripts/finer_detailing_campaign_dispatch.py resolve \
    --cohort-kind "${KIND}" --role "${ROLE}" --index "${INDEX}" \
    --submission-registry "${REGISTRY}" --project-root "${PROJECT_ROOT}"
)"
if [[ "${DISPATCH_RECORD}" == *$'\n'* ]]; then
  echo "Campaign dispatcher returned more than one resolve record." >&2
  exit 1
fi

TARGET_ENVIRONMENT=""
AUTHENTICATED_MODEL=""
EXPECTED_DIFFUSERS_REVISION=""
MANIFEST=""
TRAILING_FIELD=""
IFS=$'\t' read -r TARGET_ENVIRONMENT AUTHENTICATED_MODEL \
  EXPECTED_DIFFUSERS_REVISION MANIFEST TRAILING_FIELD <<<"${DISPATCH_RECORD}"
if [ -z "${TARGET_ENVIRONMENT}" ] || [ -z "${AUTHENTICATED_MODEL}" ] || \
  [ -z "${EXPECTED_DIFFUSERS_REVISION}" ] || [ -z "${MANIFEST}" ] || \
  [ -n "${TRAILING_FIELD}" ]; then
  echo "Malformed authenticated campaign dispatch record." >&2
  exit 1
fi

if ! _activate_conda_environment_nounset_safe "${TARGET_ENVIRONMENT}"; then
  echo "Required conda env ${TARGET_ENVIRONMENT} cannot be activated." >&2
  exit 1
fi

python scripts/finer_detailing_campaign_dispatch.py verify-job \
  --cohort-kind "${KIND}" --role "${ROLE}" --index "${INDEX}" \
  --submission-registry "${REGISTRY}" --project-root "${PROJECT_ROOT}" \
  --expected-model "${AUTHENTICATED_MODEL}" \
  --expected-environment "${TARGET_ENVIRONMENT}" \
  --expected-diffusers-revision "${EXPECTED_DIFFUSERS_REVISION}" \
  --expected-manifest "${MANIFEST}"

export HIERASAFE_REQUIRE_ENVIRONMENT_PREFLIGHT=1
export HIERASAFE_REQUIRE_CAMPAIGN_LAUNCH_AUTHORIZATION=1

exec python scripts/run_finer_detailing_correction.py \
  --manifest "${MANIFEST}" --index "${INDEX}"
