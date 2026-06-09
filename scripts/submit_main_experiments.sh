#!/usr/bin/env bash
set -euo pipefail

echo "Submitting full experiments. Make sure smoke jobs passed before running this."
sbatch slurm/t2i_main_h100.sbatch
sbatch slurm/t2v_main_h100.sbatch

