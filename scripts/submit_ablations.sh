#!/usr/bin/env bash
set -euo pipefail

echo "Submitting ablations. Make sure smoke jobs passed before running this."
sbatch slurm/ablations_h100.sbatch

