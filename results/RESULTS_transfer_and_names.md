# Cross-cell-line transfer, name-only gene placement, and a statistics retraction

Runs V3/V4, 2026-07-26. Substrate K562 Replogle non-targeting controls (n=3,000) and **RPE1** Replogle
non-targeting controls (n=3,000), C2S-Scale-Gemma-2-2B layer 21, cell-summary activations.

This file supersedes the transfer and held-out-gene sections of every earlier document. Three result JSONs are
marked `_RETRACTED` in place: `gene_phase_map.json`, `transfer_rpe1.json`, `transfer_test.json`.

---

## 0. Two instrument bugs found and fixed first

**Arbitrary per-dataset handedness.** `cc_phase.phase_angle` takes theta from an SVD, so its sign and origin are
arbitrary per dataset. Harmless within a dataset (every metric there is rotation-invariant); fatal across one.
Measured on the same 43 cell-cycle genes:

| | circ_corr, K562 vs RPE1 per-gene peak phase | median abs diff |
|---|---|---|
| unoriented | **-0.983** | 63 deg |
| oriented | **+0.983** | 15 deg |

Two datasets with identical biology looked perfectly anti-correlated. Fixed by `phase_angle_oriented`
(`cc_phase.py`), which pins handedness and origin from marker biology alone (S precedes G2/M), never from the
readout. **Any cross-dataset phase number computed before this is uninterpretable.**

**Choosing a valid transfer target.** Two candidates were rejected on measured grounds, not taste:

| | setty CD34+ | Tabula Sapiens cycling | **RPE1** |
|---|---|---|---|
| cell-cycle markers present | 18/94 | 94/94 | **93/94** |
| cell types | 8 lineages | 20+ | **1** |
| phase concentration R | 0.95-0.98 per type | 0.66 mean per type | **0.267** |
| shared genes with K562 | 5,028 | - | **6,544 / 6,546** |

setty's phase field tracks *differentiation*, not cycle (|Spearman(cos phi, pseudotime)| = 0.345). RPE1 is one
homogeneous cell line, so phase cannot be confounded with identity.

**Honest limit:** RPE1 is the same lab and platform as K562. This tests generalisation across BIOLOGY
(aneuploid erythroleukemia -> near-diploid retinal epithelium, different karyotype and sex), **not** across
batch or platform. It is not the harder cross-platform test and must not be described as one.

---

## 1. RESULT: cross-cell-line transfer works, and expression beats the model in every arm

Primary statistic `R_diff = |mean(exp(i(pred-true)))|` -- rotation-invariant and fold-free -- plus median
angular error after best rotation. Both judged against a **measured** chance floor, not an assumed one.

| arm | R_diff | median err (rot) |
|---|---|---|
| CONSTANT (target mean phase) | 0.043 | 86 deg |
| RANDOM (uniform) | 0.017 | 88 deg |
| MODEL frozen | 0.789 | 23 deg |
| **EXPRESSION frozen** | **0.878** | **16 deg** |
| MODEL target-recentered | 0.892 | 12 deg |
| **EXPRESSION target-recentered** | **0.927** | **10 deg** |
| MODEL oracle (refit on target) | 0.909 | 11 deg |
| **EXPRESSION oracle** | **0.952** | **8 deg** |

"target-recentered" refits only the feature scaler on the target, using **no target labels**, so it is still
genuine zero-shot; it removes the part of the domain shift that is a pure feature offset. The oracle is not a
transfer number -- it bounds what the target representation contains at all.

### 1b. CORRECTION -- the table above compares UNEQUAL INPUTS, and the gap is the encoding, not the model

The first reading of the table was *"expression beats the model in every arm"*. **That is withdrawn.** The two
arms were never given the same information:

| | genes | values |
|---|---|---|
| model input (a C2S cell sentence) | top **512** per cell | **rank order only** |
| expression baseline as run | all **6,544** shared | **continuous magnitudes** |

Re-running the identical frozen-transfer protocol on expression degraded step by step to exactly what the model
is handed (no GPU needed -- `scratchpad/encoding_loss.py`), with a 1000-draw bootstrap over target cells:

| expression encoding | R_diff | 95% CI | median err |
|---|---|---|---|
| all 6,544 genes, magnitudes (as run) | 0.878 | [0.869, 0.888] | 16 deg |
| top-512 genes, magnitudes | 0.810 | [0.798, 0.822] | 19 deg |
| **top-512 genes, RANK ONLY — the model's actual input** | **0.774** | [0.760, 0.787] | 21 deg |
| **C2S model, layer 21** | **0.789** | — | 24 deg |
| constant predictor | 0.043 | — | 86 deg |

* **tokenisation loss = 0.104**, paired CI [+0.092, +0.116] -- real, resolvable, and ~7x the model-vs-input
  gap. Truncation to 512 genes costs 0.068; discarding magnitudes costs a further 0.036.
* **model vs its own input encoding = +0.015**, which is 0.56x the width of a single arm's 95% CI. Not
  readable directionally. The two metrics also disagree in sign (model better on R_diff, worse on median
  error), which is what parity looks like.

**Corrected verdict.** C2S is a **near-lossless re-encoding of what it is given**: on information-matched
inputs it neither adds cell-cycle information nor destroys it. The reason it cannot beat raw expression is that
the C2S input format has already discarded 0.104 of R_diff before the first transformer layer. This *sharpens*
the reparametrisation account rather than overturning it -- but the earlier phrasing ("never better than
expression") attributed to the model a deficit that belongs to its tokenizer, and must not be repeated.

### 1c. SETTLED by a paired bootstrap with matched gene selection: **PARITY**

Activations pulled locally (`data/act_k562`, `data/act_rpe1`) and a 5,000-draw **paired** bootstrap run --
the same resampled target cells for every arm, so the contrast is measured within-cell
(`encoding_matched_transfer.py`, and `scratchpad/selmatch.py` for the selection-matched arm).

The first pass appeared to show the model *adding* information, and that was an artifact of HOW the 512 genes
were chosen. A cell sentence takes the top-512 of the cell's **full** panel (K562 6,546 / RPE1 8,749), whereas
the first baseline took the top-512 of the **6,544 shared** genes. On RPE1 that let the model's input contain
~2,205 genes the baseline never saw, and shifted the ranks of the shared genes both arms did use. Matching the
selection (`FULLSEL`: top-512 from the full panel, then expose only shared columns, carrying full-panel ranks)
removes the effect entirely:

| arm | R_diff | median err |
|---|---|---|
| expr_full (6,544 genes + magnitudes) | 0.878 | 16 deg |
| expr_512_mag | 0.810 | 19 deg |
| expr_512_rank (shared -> top-512) | 0.774 | 21 deg |
| **expr_512_rank FULLSEL (top-512 -> shared) — the correct match** | **0.792** | **20 deg** |
| **C2S model, layer 21** | **0.789** | 23 deg |
| constant / random | 0.043 / 0.017 | 86 / 88 deg |

| paired contrast | delta R_diff | 95% CI | verdict |
|---|---|---|---|
| model - expr_512_rank *(mismatched selection)* | +0.0153 | [+0.0022, +0.0283] | artifact — do not cite |
| **model - expr_512_rank FULLSEL** | **-0.0022** | **[-0.0144, +0.0105]** | **PARITY** |
| expr_full - expr_512_rank (tokenisation loss) | +0.1042 | [+0.0917, +0.1169] | large, robust |
| expr_full - expr_512_mag (truncation alone) | +0.0683 | [+0.0576, +0.0787] | |
| expr_512_mag - expr_512_rank (magnitudes alone) | +0.0359 | [+0.0261, +0.0459] | |

**FINAL VERDICT.** Given the same information, C2S layer 21 is **statistically indistinguishable from its own
input encoding** (-0.0022, CI spans 0; the matched baseline is also marginally better on median error, 20 vs
23 deg). The model neither adds nor destroys cell-cycle phase information. The entire apparent deficit against
raw expression is the **tokenizer**: 0.104 of R_diff, CI [0.092, 0.117], lost before the first transformer
layer -- 0.068 to keeping only 512 genes and a further 0.036 to discarding magnitudes.

This is a sharper statement than "the manifold is in the data": the cell-cycle circle is carried faithfully
through 21 transformer layers, and C2S's inability to beat raw expression is a property of its **input format**,
not of its processing.

---

## 2. RESULT: the gene -> phase map is in the weights (name-only placement)

The one design in this project where the model provably sees **no expression covariance**: a phase-neutral
backbone (a real K562 cell with all cell-cycle genes stripped) plus **one gene name** inserted at a fixed rank.
Readout = which pole of the real-cell circle the placed gene falls toward; 87 Tirosh markers (37 S / 50 G2M),
24 backbones, AUROC with label permutation.

| form | example | mean edit dist | AUROC | p | class separation | % of real effect |
|---|---|---|---|---|---|---|
| **human** | `CCNB1` | 0.00 | **0.989** | 0.0002 | **40.7 deg** | 100% |
| mouse | `Ccnb1` | 2.83 | 0.782 | 0.0002 | 2.8 deg | 7% |
| lower | `ccnb1` | 3.83 | 0.767 | 0.0002 | 3.4 deg | 8% |
| nearmiss | `CQNB1` | 1.00 | 0.685 | 0.0012 | 4.3 deg | 10% |
| **anagram** | `NBCC1` | 3.11 | **0.511** | 0.43 | 0.2 deg | 0% |

Robustness of the human arm:

| control | result |
|---|---|
| raw | AUROC 0.989 |
| residualised on symbol length + mean expression + detection rate | **0.908**, permutation p = 5e-5 |
| exact symbol-length-matched S/G2M pairs | **30 of 31**, sign test p = 3e-8 |
| abundance confound | dead: S vs G2M do not differ (p = 0.31 mean expr, p = 0.75 detection) |

**Verdict.** From a gene NAME alone, with no covariance in the input, C2S sorts S from G2M at AUROC 0.989 and
40.7 deg of separation, while an anagram of the same letters is at chance. The gene -> phase association is in
the weights. This is the days-of-week analogue and it holds.

### 2b. RETRACTED: the "unseen token" reading of the mouse arm

`surface_forms.py` printed an automated verdict -- *"a real unseen symbol beats a closer non-symbol -> tracks
TOKEN IDENTITY, not spelling"* -- and **that verdict is withdrawn.** Its threshold was too generous:

* `nearmiss` is a **non-existent** symbol (verified 0 collisions against a 10k-symbol universe) yet scores
  0.685 at p = 0.001. A non-gene captures most of the (small) effect.
* every perturbed form yields only 2.8-4.3 deg of separation, i.e. **7-10%** of the real-symbol effect. AUROC is
  rank-based and inflated a tiny consistent tilt into an impressive number.
* the residual is carried by **gene-family nomenclature**, not identity. Splitting on shared 3-letter prefixes
  (MCM2-7 all S; CCNB1/2, AURKA/B, BUB1/1B all G2M):

  | form | family-prefix genes | singletons |
  |---|---|---|
  | human | 1.000 | 0.977 |
  | mouse | 0.866 | 0.726 |
  | lower | 0.903 | 0.680 |
  | nearmiss | 0.753 | 0.664 |
  | anagram | 0.405 | 0.570 |

  Real symbols show no family dependence; every perturbed form does.

**No mouse-ortholog or species-knowledge claim is supported.** The residual is best described as weak
subword/nomenclature generalisation. `mouse` ~= `lower` also shows it is case-robustness, not mouse-specificity.

---

## 3. RETRACTED: the Jammalamadaka circular correlation, project-wide

`circ_corr(a,b) = sum sin(a-abar) sin(b-bbar) / sqrt(...)` requires both circular means to exist. Two
degenerate regimes appear in this project:

* **annotation-degenerate** -- one variable near-uniform (R <= 0.10) while the other is concentrated
  (R >= 0.20). Mean estimates become inconsistent, `sin()` folds the circle, and the value can come out large
  with the wrong sign.
* **design-degenerate (worse)** -- one variable is a *constructed* uniform grid (`arange(n)*2pi/n`, or a
  balanced per-bin sampler), so R is ~1e-16 **exactly** and the reference angle is floating-point residue. The
  value is then not reproducible across machines or row orders.

Reference instance, K562 -> RPE1 transfer: target phase R = 0.043, frozen predictions R = 0.341,
`circ_corr = -0.814` against `R_diff = +0.789` and 23 deg median error versus an 86 deg constant floor.

An automated audit swept 52 call sites across code, result JSONs, the paper and the RESULTS files; 29 were
classified AFFECTED. **Independently recomputed here from the stored rows** (this correction was verified, not
taken on trust):

| steering run | stored gate | recomputed \|circ_corr\| | raw sign | R_true | R_read | **R_diff** | med err |
|---|---|---|---|---|---|---|---|
| L18 | 0.739 | 0.775 | **-0.775** | 1.0e-16 | 0.267 | **0.782** | 22 deg |
| L21 | 0.739 | 0.775 | **-0.775** | 1.0e-16 | 0.267 | **0.782** | 22 deg |
| L21_matched | 0.789 | 0.759 | **-0.759** | 1.0e-16 | 0.338 | **0.712** | 35 deg |
| pt_L25 | 0.618 | **0.294** | +0.294 | 1.2e-16 | **1.000** | **0.002** | **90 deg** |

`base_true` is uniform **by construction** (`manifold_steer.py` caps each of 12 source bins at 24//12 = 2
cells), so R_true = 1e-16 exactly. Three of the four printed gate values are `abs()` of a **negative** number.

### What this does and does not change

**Gates still pass; steering results stand.** All three cyclic runs clear the pre-registered 0.40 threshold on
R_diff (0.712-0.782, 22-35 deg vs ~90 deg chance). `gate_pass = true` is correct. The manifold-vs-linear path
ratios, target reach, isometry and lap-walk results are untouched. This is a correction, not a retraction --
but a pre-registered gate whose printed value is decided by floating-point residue is not a gate, whichever
side it lands on.

**Ordering claim retracted.** `L21_matched` is the **weakest** run on R_diff (0.712), not the strongest (0.789
reported). Any "strongest of the three" wording is void.

**The 27B-vs-2B gate comparison was never like-for-like** -- and this corrects the audit, which had listed
`27b_L44` as affected. Its instrument key is `readout_vs_true_corr` + `readout_spread_frac_of_range`, i.e. the
Pearson/ordered branch, **not** `circ_corr`. So 0.679 (27B) and 0.739/0.789 (2B) are *different statistics*.
The comparison should be dropped, not renumbered; the depth claim belongs to the amplitude half (92.9% dynamic
range), which is unaffected.

**One confirmed false PASS: `manifold_steer_pt_L25`.** `gate_pass = true` at 0.618 on a readout with
R_read = 1.000 (essentially constant), R_diff = 0.002 and 90 deg error -- pure chance. Its
`argmax_hits_src_bin = 0.0` (zero of 24, versus 0.167 = 2x chance for the legitimate runs), and all three arms
show `mean_max_jump = 0.0` with `target_reach = 0.0833` = exactly 1/12. The stored 0.618 could not be
reproduced from its own rows (recomputed 0.294). The instrument was dead.
*Consequence:* this run is **uninformative**, and its negative pseudotime-steering conclusion must be withdrawn
as unsupported rather than reversed. The false pass certified a dead instrument; it did not manufacture a
positive.

### Verified SAFE -- do not over-retract

**Within-dataset cross-validated ridge readouts** are the largest class and are clean. K562 truth is genuinely
near-uniform (R = 0.079), but the predictions are not concentrated (R_pred 0.11-0.21) and, decisively, the
ridge intercept is the training-fold mean of (cos, sin), which **pins the predicted mean direction to the
truth's**: measured mean discrepancy 1.6-12.8 deg, versus 215 deg in the genuine transfer failure.
Empirically circ_corr tracks R_diff to <= 0.026 at every operating point. This covers the Result-1 depth sweep
(2B L00 0.068 -> L21 0.898), compute-vs-relay retention, the `input_null` table, and the 2B/27B relative-depth
comparison. **None of these move.**

**`abs()`-wrapped scores against a permutation null that holds the degenerate marginal fixed** are exactly
calibrated -- permuting the annotation preserves its circular mean -- so no false positive can be manufactured.
The geometric screen verdicts stand.

**`kind="ordered"` runs** (`ery_L25`, `pt_L25` amplitude arm, `27b_L44`) route to `np.corrcoef` and are not
affected by this bug at all.

### Fix applied

`transfer_test.py` now reports `circ_R_diff` and `med_err_rot`, measures its own constant/random floor instead
of assuming 90 deg, gates validity on R_diff against that measured floor, and no longer asserts "the target's
phase label is not cell cycle" on failure -- that diagnosis was right for setty and would have been wrong for
RPE1, whose label was verified at +0.983. `transfer_diag.py` had stored a `verdict_note` saying to judge
transfer on circ_corr alone; that was exactly backwards and is corrected.

---

## 4. Open items

1. **`coexpr` margin bootstrap for `cellcycle_circle`** -- recomputed point margin +0.389 with frac<=0 = 0.055
   versus 0.166 reported. Near the decision boundary, so the top-line "not model-specific" verdict could in
   principle move. Needs the 1000-boot paired bootstrap on the shared 28 genes; two attempts died on memory
   (928 MB coexpr npz). **Do not restate that verdict until it returns.**
2. **Upstream `route_genemanifold/gm_lib.py`** carries the same defect against an axial annotation but with a
   *positive* verdict (scgpt_we 0.978, verdict "MODEL"), so its exposure is higher than anything here. Triage
   at the source so future ports inherit the corrected version.
3. **Geometric-screen depth claim** -- "decays to non-significance at the last layer" (L25 stat 0.224,
   p = 0.253) becomes R_diff 0.415, p = 0.012. The decay claim is false; restate as a flatter peak.
4. **Nine sibling scripts** carrying the same `_circ_corr` pattern are not individually audited. Do not assume
   they are broken; record R of both arguments per call site, prioritising designed grids and frozen
   cross-dataset readouts.
5. **A remaining confound on section 2** -- the name-only readout could be reading "typical abundance" rather
   than phase. Abundance is dead as a *class-level* confound (p = 0.31 / 0.75) and residualisation retains
   AUROC 0.908, but an abundance-matched non-cell-cycle gene panel would close it fully.
