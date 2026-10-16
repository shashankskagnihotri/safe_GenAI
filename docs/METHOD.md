# Riddikulus method reference

This document describes the frozen October 16 method.
See ONTOLOGY.md for every exact H4 endpoint and relationship. Scores and
geometric alignment are not calibrated probabilities of harmful content.

## Same-state predictions and evidence

At each guided update, obtain the ordinary prediction v_b, neutral prediction
v_n, and the unsafe/safe predictions v_uj,v_sj for each participating leaf j.
All predictions use the same current latent, native time and, for FLUX.3, the
same destination time. The original generation prompt is unchanged; selection
never reads or analyzes its text.

    r_b = v_b - v_n; r_uj = v_uj - v_n; r_sj = v_sj - v_n

Subtracting the shared neutral behavior lets us compare the parts of each
prediction associated with its conditioning. This is not subtraction of pixels
or an anatomical segmentation mask.

    c_uj(p) = cosine(r_b(p), r_uj(p))
    c_sj(p) = cosine(r_b(p), r_sj(p)); g_j(p) = c_uj(p) - c_sj(p)

At position p, ask whether the current image prediction points more toward the
unsafe endpoint than the safe endpoint. A positive gap is a geometric contrast,
not proof that the corresponding object is present in the decoded image.

    a_j(p) = reliable_j(p) * clip((c_uj(p)-0.25)/0.25,0,1)
                               * clip((g_j(p)-0.10)/0.25,0,1)

Both eligibility comparisons are strict. Residual reliability requires local
RMS above max(1e-6, 1e-3 times that residual field's global RMS). The two ramps
make evidence gradual: c_u=0.45 and gap=0.20 give confidence 0.8*0.4=0.32.
Increasing a threshold rejects evidence; increasing a ramp width reduces
confidence for the same scores.

## Fixed pool acquisition and fresh corrections

    A_j = mean_p a_j(p)
    S = first25(sort canonical pairs by descending A_j, then canonical ID)

Rank whole-field mean confidence once after native warm-up. Exact text-pair
aliases share canonical representatives. There is no additional leaf-admission
floor. Exactly 25 identities are retained, but some may have zero confidence.
Excluded identities cannot return; membership is fixed while scores change.

For the four established models, indices 0-9 are native-only. At 10, evaluate all
287 leaves once, select the pool and reuse the same predictions for correction.
At 11-49 refresh all 25 retained leaves and correct all 25. The approved
FLUX.3 profile uses indices 0-2 native-only, acquisition at 3 and refresh through 15.
Warm-up contains no neutral/concept predictions or steering.

    D_j = 1 - cosine(vec(r_uj), vec(r_sj))
    kappa_j = sigmoid(D_j / 1.0)
    d_j = kappa_j * (v_sj - v_uj)

The distance determines a current concept-specific strength, applied once to the
raw replacement direction and separately from local confidence. For reliable
geometry D is between 0 and 2: the sigmoid spans 0.5 to approximately 0.881.
At D=1 its weight is approximately 0.731. Identical endpoint
predictions still give zero raw direction. Unreliable geometry gets weight 0.
The distance is semantic separation, not a severity score.

    raw(p) = sum_{j in S} a_j(p) * d_j(p)

Every retained leaf contributes its own field. There is no division by 25, sum
of confidence, branch count or hierarchy depth. The hierarchy preserves concept
identities and taxonomy; it does not average independent requirements.

## Conflicts and bounds

At every position, minimize over z:

    0.5*||z-raw||^2 + 1.0*sum_j a_j*relu(-z dot unit(d_j))^2

Stay close to the summed edit while softly penalizing opposition to individual
contributing directions. This is not a hard guarantee that every requirement
will be satisfied. The additive method uses unnormalized individual confidence.
Fixed-step descent checks first-order residual against 1e-8+1e-5*||raw||. There
is no hard iteration allowance; nonfinite arithmetic or numerical stagnation
fails explicitly. The value 128 is spatial chunk size, not an iteration limit.

    local_cap(p) = 0.85 * RMS_features(v_b(p))
    global_cap = 0.60 * RMS_all(v_b)

Shrink corrections exceeding either cap. A cap limits prediction magnitude,
not the percentage of pixels edited. Raising an inactive cap cannot strengthen
a correction. Prompt-residual and accumulated-exposure caps are disabled;
exposure remains a reported diagnostic.

    v_applied = cast_native(v_b + bounded_correction)
    actual_delta = float(v_applied) - float(v_b)

Verify the correction after native rounding. Quantization backtracking is
separate from conflict convergence; it shrinks or reverts unrepresentable
updates rather than silently exceeding the caps.

## Native solver and actual intervention

    next_actual = native_step(v_applied, current_state)
    next_counterfactual = native_step(v_b, identical_current_state)
    effect = next_actual - next_counterfactual

Advance the real scheduler exactly once and compare it with isolated solver
clones sharing the complete prior trajectory. This costs zero extra model
predictions. It measures one-update causal intervention, not an entire-image
unsteered counterfactual. Preserve full scheduler history, RNG, latent, pool,
trace commitments and native cache state when resuming.

FLUX.3 uses the official distilled flow-map network, conditioned on the current
and destination timestep. Its native update is FP32_state + Python_dt *
BF16_velocity; preserve the BF16 increment rounding. It has no CFG and no prompt
upsampling. Do not replace it with FLUX.2 Klein or the undistilled base model.

## Native image decoding

Steering changes the denoising trajectory. Rendering that trajectory must still
use the model's native inverse VAE normalization. For SD3.5 Large:

    image = postprocess(VAE.decode(z_final / 1.5305 + 0.0609))

The scale and shift restore the representation expected by the decoder; they
are not steering strengths. Both Baseline and Riddikulus use the same transform
in native precision. Decoding and predicted-clean previews never feed back into
the latent, scheduler or controller.

An October 16 audit found that the inherited SD decoder omitted this shift.
The SD-specific override repairs rendering without changing the frozen steering
method. A restricted source and checkpoint handoff preserves completed native
updates and authenticates the two-file repair. Old images and their judgments
remain historical evidence, separate from corrected primary results. Old
predicted-clean previews cannot be recovered exactly from post-update checkpoints
alone and are explicitly labeled decoder-invalid when retained after recovery.
Runtime decode receipts check finite tensors, native dimensions, unchanged input
latents and zero denoiser calls. Source-audited VAE input equivalence is separate
from independently executing the complete native pipeline.

## Parameters and limitations

| Quantity | Frozen setting | Interpretation |
|---|---:|---|
| Candidate identities |25|Fixed after one scan; all refresh and can contribute.|
| Absolute cosine threshold |0.25|Reject weak unsafe alignment.|
| Unsafe-minus-safe gap |0.10|Require unsafe alignment to exceed safe alignment.|
| Ramp widths |0.25 each|Control gradual confidence growth.|
| Sigmoid temperature |1.0|Scale concept-distance weights.|
| Global/local RMS caps |0.60/0.85|Constrain overall and concentrated correction magnitude.|
| Conflict penalty |1.0|Discourage opposing an individual contributing direction.|
| Spatial area gate |Disabled|No minimum whole-image support fraction.|
| Prompt-residual/exposure caps |Disabled|No additional magnitude/exposure ceiling.|

A fixed pool can miss relevant leaves. Whole-image mean confidence favors
broad evidence. The top 25 may include false positives; 25 selected identities do
not establish 25 active hazards. Additive corrections can cancel, conflict handling
is soft, and magnitude limits can attenuate useful edits. The benchmark evaluates
these limitations without retuning on its outcomes. User acceptance, numerical
validity and independent visual safety judgments remain separate evidence.
