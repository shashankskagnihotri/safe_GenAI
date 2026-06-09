#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="${PWD}/src:${PYTHONPATH:-}"
python -m hierasafe_flow.cli.run_smoke_tests --config configs/experiments/smoke_t2i.yaml "$@"
python -m hierasafe_flow.cli.run_smoke_tests --config configs/experiments/smoke_t2v.yaml "$@"
