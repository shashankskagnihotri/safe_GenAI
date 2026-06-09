#!/usr/bin/env bash
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "HF_TOKEN is not set." >&2
  exit 2
fi

ATTEMPT_NAME="${1:-try_First}"
REPORT_PATH="${2:-debugging/${ATTEMPT_NAME}.md}"

sbatch --export=ALL,HF_TOKEN="${HF_TOKEN}",ATTEMPT_NAME="${ATTEMPT_NAME}",REPORT_PATH="${REPORT_PATH}" \
  slurm/redteam_four_models_h100.sbatch
