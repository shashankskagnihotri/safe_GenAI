#!/usr/bin/env bash
set -euo pipefail

conda env create -f environment.yml
conda activate hierasafe-flow
pip install -e .
python -m pytest

