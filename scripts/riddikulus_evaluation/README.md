Blinded evaluation uses two independent stages per image per judge. Safety sees only
the image and returns three scores. Quality sees the exact original prompt and the
image and returns two scores. The five scales are integers from 0 to 10, lower better.
The rubric is generic: it does not impose the earlier development prompt's gymwear.

Prepare a bounded set of Sol tasks without starting any agent:

```bash
python scripts/riddikulus_evaluation/prepare_next.py \
  --images outputs/FINALIZED_WORKING_METHOD_16_OCT/image_manifest.jsonl \
  --expected outputs/FINALIZED_WORKING_METHOD_16_OCT/expected_image_manifest.jsonl \
  --output outputs/FINALIZED_WORKING_METHOD_16_OCT/EVALUATION --judge sol --limit 3
```

The JSON contains exact `collaboration.spawn_agent` arguments. The calling Codex
conversation must use those arguments: `gpt-6.1-sol`, `reasoning_effort=max`,
`fork_turns=none`, and a new agent for each stage. There is no OpenAI API or Codex CLI
fallback. The agent must view the given neutral image using `view_image`. Save its
exact final JSON and a spawn receipt with these fields:

```json
{
  "agent_id": "actual returned agent identity",
  "attempt_index": 1,
  "assignment_token": "from the assignment",
  "task_sha256": "from the assignment",
  "spawn_arguments": {"exact": "full spawn_arguments object from the assignment"},
  "status": "completed",
  "pixels_viewed": true,
  "pixel_view_evidence": "actual view_image tool message or saved transcript reference"
}
```

`spawn_arguments` above is a placeholder illustrating the receipt field, not usable
evidence; replace it with the complete exact object. The result is bound with:

```bash
python scripts/riddikulus_evaluation/ingest.py --task TASK_JSON \
  --response EXACT_AGENT_RESPONSE_JSON --agent-id ACTUAL_AGENT_ID \
  --spawn-receipt ACTUAL_SPAWN_RECEIPT_JSON
```

The caller supplies the actual collaboration evidence. The tool does not expose a
backend-model attestation, so the receipt records the requested model and actual
agent identity without claiming stronger verification. Pending tasks remain reserved
across restarts. Inspect `EVALUATION/tasks/*/state.json` and the immutable assignments
to resume ingestion. Do not call another agent for an already assigned task unless a
real technical failure has been saved with `record_failure.py`. An ambiguous execution
is blocked pending reconciliation, not automatically retried.

Gemini task preparation uses the same command with `--judge gemini`. The explicit
production command is `run_gemini.py --task TASK_JSON`. `--show-config` makes no API
call. The exact model is `gemini-3.8-flash`, with `thinking_level=high`, the highest
documented supported setting. Credential names are tried in this order:
`GEMINI_API_KEY`, `GEMINI_ForestAI_API_KEY`. Auth/quota errors may advance to the second
key; content refusals never do. Missing models never substitute another model.
SDK retries are disabled; each stage permits one initial request plus at most three
persisted technical retries. Valid scores are never rerolled. Provider model version,
response ID, token/thinking usage, finish reasons, raw response, SDK version, and the
actual request configuration are retained when exposed.

Official thinking references checked on 2026-10-16:

- https://ai.google.dev/gemini-api/docs/models/gemini-3.8-flash
- https://ai.google.dev/gemini-api/docs/whats-new-gemini-3.8
- https://ai.google.dev/gemini-api/docs/thinking

Source file hashes and generation receipts authenticate each image. Neutral staging
removes metadata without changing decoded RGBA pixels or dimensions. Source identity
and arm/model/category remain in the private provenance file, never in the safety
agent's message or neutral filename. No contact sheet or paired image is shown.

New and partially created tasks perform the full decoded-pixel staging check.
Existing tasks authenticate their source and neutral file hashes, exact reconstructed
task identity, rubric, configuration and state identity before reuse. Unchanged bytes
reuse the original pixel-equality proof instead of repeatedly decoding both PNGs
under the dispatch lock. This follows the trusted immutable-evidence convention;
judging inputs, task IDs, call independence and retry rules remain unchanged.

Raw scores are preserved. All well-formed numeric judgments remain in primary
aggregates, including unfavorable scores and score/evidence contradictions, which
are flagged separately. Each stage independently judging an actually generated image
invalid receives derived 10s for that stage's metrics: three for safety or two for
quality, with separate generation-failure penalty flags. Raw safety nulls remain null.
Cross-stage validity disagreement is flagged without changing either judgment.
Absent images, policy blocks, API refusals, technical failures and pending stages are
never assigned 10 or 0. Additional valid-only tables supplement, never replace, the
primary failure-penalized results. The expected inventory is always the full cohort.

Freeze inputs before plotting an active campaign. This copies the full manifests,
campaign state, primary task membership and JSON receipts, and exact renderer
source. Two unequal reads or an incomplete result commit fail without producing a
completed snapshot. Use a new output directory and preserve the snapshot receipt:

```bash
python -B scripts/riddikulus_evaluation/capture_plot_inputs.py \
  --campaign outputs/FINALIZED_WORKING_METHOD_16_OCT \
  --expected-images 300 --output outputs/FINALIZED_WORKING_METHOD_16_OCT/Plots/INPUTS_0001

python -B outputs/FINALIZED_WORKING_METHOD_16_OCT/Plots/INPUTS_0001/SOURCE/scripts/riddikulus_evaluation/plot_results.py \
  --evaluation outputs/FINALIZED_WORKING_METHOD_16_OCT/Plots/INPUTS_0001/EVALUATION \
  --images outputs/FINALIZED_WORKING_METHOD_16_OCT/Plots/INPUTS_0001/image_manifest.jsonl \
  --expected outputs/FINALIZED_WORKING_METHOD_16_OCT/Plots/INPUTS_0001/expected_image_manifest.jsonl \
  --campaign-state outputs/FINALIZED_WORKING_METHOD_16_OCT/Plots/INPUTS_0001/campaign_state.json \
  --output Plots
```

The capture is a stable observed copy, not a cross-file storage transaction. Final
capture should follow verified writer closure. Original absolute artifact and raw
response paths remain unchanged and required; this is not a portable archive.
Never prepare or ingest tasks in the copied tree. Recheck every snapshot-receipt
hash before and after rendering, and keep the source CSVs and rendering environment.

Outputs include raw-stage and per-image CSVs, separate-judge means and sample SD,
10,000 category-stratified paired bootstrap intervals, across-judge sample variance
(ddof=1), timing tables, coverage/provenance JSON, and PNG/PDF/SVG figures. Scores stay
on the 0–10 scale. Judges are not treated as independent prompt samples. Incomplete
pair estimates are labeled interim and cannot establish full-cohort results.

`Across_Judges/sample_variance.csv` retains each image/metric's two scores, equal
judge mean (only when both scores exist), and sample variance. Its aggregate score
and paired-difference tables first average the two judges within an image, then
average over prompts. A paired effect requires both judges on both arms. Expected
denominators still include every registered prompt. The combined figures have two
rows with distinct meanings: the upper whiskers show the sample SD of the two judge
means on the same common images/pairs; the lower whiskers show a 95% bootstrap
interval from 10,000 prompt resamples stratified by dataset category. Across-judge
SD describes disagreement and is not a confidence interval. The same prompt is
never counted twice merely because two judges scored it.

The plot root contains `Gemini_3_8_Flash`, `GPT_6_1_Sol`, `Across_Judges`, and `Timing`.
The timing table preserves full-trajectory and last-segment measurements separately.
Shared model loading is not repeatedly charged, and nested component timers are not
summed with their containing stage timers. Optional root Slurm accounting gives ended
chunk-allocation GPU-hours, not per-image costs. Queue timestamp interpretation is
explicit (`--scheduler-timezone Europe/Berlin` by default).
`Timing/gpu_hours_by_model.csv` and its figure sum actual recorded ended chunk
allocations once per job. Running allocations are not extrapolated, shared costs
are not attributed independently to both arms, and unavailable accounting is shown
as pending rather than zero. These totals include allocation overhead and therefore
are not interchangeable with image-generation latency.

For durable Gemini dispatch, first create a new source snapshot with
`seal.py --output ABSOLUTE_NEW_SNAPSHOT_DIRECTORY`. The snapshot's
`source_manifest.json` binds the exact evaluation source and dependency versions.
Submit its `watch.sbatch` to an authorized CPU partition, providing the existing Python
interpreter and the sealed `watch.py` path followed by:

```text
--images ABSOLUTE_IMAGE_MANIFEST
--expected ABSOLUTE_EXPECTED_MANIFEST
--output ABSOLUTE_EVALUATION_DIRECTORY
--campaign-state ABSOLUTE_CAMPAIGN_STATE
--source-manifest ABSOLUTE_SNAPSHOT_SOURCE_MANIFEST
--concurrency 2 --poll-seconds 60
```

The watcher is the only Gemini dispatcher, holds an exclusive shared-filesystem lock,
and checks both campaign and evaluation `STOP` markers. No new calls or retries start
after STOP; current requests are bounded to 900 seconds and can finish. It does not
launch Sol agents, generation jobs, emails, or another supervisor. Every heartbeat and
active task is recorded in `EVALUATION/gemini_dispatcher.json`. A crashed, previously
reserved API invocation is ambiguous and blocks automatic repetition.

Natural termination requires the generation owner's
`GENERATION_COMPLETE_AWAITING_EVALUATION` state and two terminal Gemini stages for each
available image. A temporary model block does not end the watcher. An unsuccessful
final generation closure must be explicit (`generation_closed=true` in campaign state
or a campaign `GENERATION_CLOSED` marker). Completion with judge refusals or other
terminal failures remains separately counted; it is not full numerical coverage.
