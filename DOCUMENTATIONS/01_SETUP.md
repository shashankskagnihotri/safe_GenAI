# Setup

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate hierasafe-flow
pip install -e .
pytest
```

For gated Hugging Face models, authenticate before loading real adapters:

```bash
huggingface-cli login
```

The dummy adapter smoke tests do not download model weights:

```bash
scripts/run_smoke_test_local.sh
```

