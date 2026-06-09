#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
CONFIG="${1:-configs/experiments/main_all_safety_t2i.yaml}"
shift || true
python -m hierasafe_flow.cli.run_generation --config "${CONFIG}" "$@"
