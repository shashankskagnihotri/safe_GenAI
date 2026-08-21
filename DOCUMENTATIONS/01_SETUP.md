# Setup

Only conda is supported for this project. Do not use `venv`, `virtualenv`, or `python -m venv`.

```bash
cd /ceph/sagnihot/projects/safety_genAI

conda env create -f environment.yml
conda activate safe_genai_conceptsteer
python -m pip install --no-deps -e .
python -m pytest
```

For gated Hugging Face models, authenticate before loading real adapters:

```bash
huggingface-cli login
```

The dummy adapter smoke tests do not download model weights:

```bash
cd /ceph/sagnihot/projects/safety_genAI
conda activate safe_genai_conceptsteer
scripts/run_smoke_test_local.sh
```
