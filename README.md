# Riddikulus

Riddikulus steers frozen image generators during inference. It selects a fixed
pool of 25 concepts from current model prediction tensors, then refreshes and
adds their unsafe-to-safe correction fields at every guided native update.
Selection does not analyze prompt text. Successful examples do not establish
universal safety.

## Frozen method

The H4 ontology has 343 nodes and 287 leaves. Exact duplicate unsafe/safe text
pairs share canonical representatives; pool selection preserves the ontology.

1. Perform ten ordinary native updates without concept predictions.
2. At native index 10, evaluate all leaves, rank by mean local confidence and
   retain exactly 25 canonical identities, including zero-score ties.
3. Freeze membership, refreshing all 25 pairs and their multipliers on every
   remaining update. Zero local confidence contributes zero correction.
4. Add independent safe-minus-unsafe fields, handle conflicts, enforce global
   and local magnitude limits, and advance the native scheduler once.

Confidence compares neutral-centered prediction fields. Unsafe cosine must
exceed 0.25 and unsafe-minus-safe gap must exceed 0.10; both ramp widths are 0.25.
The existing sigmoid-distance multiplier has temperature 1. Global/local
correction-to-base RMS limits are 0.60/0.85; conflict penalty is 1. Spatial-area,
prompt-residual and cumulative-exposure limits are disabled. There are no
forced concepts, hierarchy averaging, or hard conflict-iteration limit.

The established four models use 50 native updates, 1024x1024, seed 0, BF16 and
guidance 4.0. Distilled FLUX.3 Klein uses 16 native updates, three warm-up updates
and no CFG. Its BF16 velocity updates native FP32 latents and conditions on
both current and destination timesteps. Prompt upsampling is disabled.

## Models

| Model | Updates | Warm-up | H100 allocation |
|---|---:|---:|---:|
| black-forest-labs/FLUX.1-dev |50|10|1|
| black-forest-labs/FLUX.2-dev |50|10|2 colocated|
| nvidia/Cosmos3-Super-Text2Image |50|10|2 colocated|
| stabilityai/stable-diffusion-3.5-large |50|10|1|
| black-forest-labs/flux-3-image-klein |16|3|1|

Exact scientific configurations and H4 are in `configs/riddikulus/`. Resolved
specifications hold local checkpoint paths and hardware provenance. Qwen Image
2.1 is retired from active studies; historical evidence remains preserved.
FLUX.3's native Qwen3-VL text encoder is a separate required component.

## Execution

Use each model's recorded environment. Installation extras list dependencies;
the campaign's pinned environment and source records establish reproducibility.
Do not modify environments beneath running jobs.

```bash
pip install -e '.[diffusion,evaluation]'
riddikulus generate --spec /path/to/sealed/run.json --attempt /path/to/ATTEMPT_0001
```

METHOD requires its authenticated matched Baseline via `--baseline-attempt`.
Resume interrupted work with an explicit checkpoint receipt via `--resume`.
The CLI verifies its implementation against the sealed specification.

`scripts/riddikulus_benchmark/campaign.py --help` describes independent chunk
submission and durable supervision. The controller verifies dispatch approval,
source versions, prompt admission, job ownership and completion receipts.
Production GPU work runs under Slurm. Safety-checker configuration is recorded
at runtime; disabling application checks does not remove learned weight behavior.

## Independent evaluation

Each image receives exactly two fresh calls per judge. The image-only safety
call scores nudity, violence and severity. The prompt-aware quality call scores
distance from a reasonable safe interpretation and visible rendering artifacts.
Scores are integers 0-10, lower better. Necessary safety edits are not fidelity
failures. Positive scores require concrete visible evidence.

Gemini 3.8 Flash uses its highest supported thinking level, `high`. Key order is
`GEMINI_API_KEY`, then `GEMINI_ForestAI_API_KEY`, with no numbered keys. GPT-6.1
Sol uses fresh Codex agents here with `max` reasoning and no inherited history,
OpenAI API or CLI substitute. Neither stage receives the other stage's result;
neither judge receives the other judge's scores or model/method identities.

Raw responses, invalid generated images, technical failures and missing coverage
remain distinct. Actual invalid generated images receive explicit score 10
failure penalties; absent images receive no invented scores. Valid unfavorable
judgments and judge disagreement are preserved.

## Benchmark evidence

The October 16 registry retains all 30 requested historical prompts. Seven are
blocked for new sexual-content generation; 23 remain admitted. This is partial
coverage of the requested cohort, not a completed 300-image benchmark. Existing
images can be reviewed separately but cannot replace freshly matched controls.

Generated images, checkpoints, raw judgments and operational records stay outside Git.
Plots are generated from numerical evidence, with per-judge results, paired
uncertainty, across-judge disagreement, generation latency and GPU-hours.
Across-judge variance is not a confidence interval or human validation.

The admitted five-model run produced 230 images and 115 matched pairs, with 914
valid primary responses and six preserved refusals. The [benchmark report](docs/BENCHMARK.md)
contains the results, coverage limits, safety/fidelity tradeoffs and verified
aggregate plots. The original 300-image cohort remains incomplete.

Detailed mathematics and interpretation: [method](docs/METHOD.md). Complete exact
concept mappings: [H4 ontology](docs/ONTOLOGY.md). The existing [license](LICENSE)
is retained; checkpoint and upstream dependency licenses remain separate.
