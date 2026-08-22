#!/usr/bin/env bash
#SBATCH --job-name=iclr-nag-assets
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=08:00:00
#SBATCH --nice=10000
#SBATCH --output=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/%x_%j.out
#SBATCH --error=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/JOB_LOGS/%x_%j.err

set -euo pipefail
: "${PUSH_FOR_ICLR_EXECUTION_ROOT:?missing PUSH_FOR_ICLR_EXECUTION_ROOT}"
: "${PUSH_FOR_ICLR_EXPECTED_COMMIT:?missing PUSH_FOR_ICLR_EXPECTED_COMMIT}"
test "$(git -C "$PUSH_FOR_ICLR_EXECUTION_ROOT" rev-parse HEAD)" = "$PUSH_FOR_ICLR_EXPECTED_COMMIT"
git -C "$PUSH_FOR_ICLR_EXECUTION_ROOT" diff --quiet
git -C "$PUSH_FOR_ICLR_EXECUTION_ROOT" diff --cached --quiet
export HF_HUB_DISABLE_TELEMETRY=1
export PYTHONUNBUFFERED=1
ENV_ROOT=/ceph/sagnihot/projects/safety_genAI/outputs/PUSH_FOR_ICLR/REPRODUCTIONS/ENVIRONMENTS/nag/8bb34a16e517/env
CONFIG="$PUSH_FOR_ICLR_EXECUTION_ROOT/configs/experiments/push_for_iclr/nag_qualitative_integrity_v1.json"
"$ENV_ROOT/bin/python" "$PUSH_FOR_ICLR_EXECUTION_ROOT/scripts/push_for_iclr/download_nag_flux_assets.py" --config "$CONFIG"
