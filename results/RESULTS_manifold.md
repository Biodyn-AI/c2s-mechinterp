# A causally-used, steerable CELL-CYCLE MANIFOLD in C2S-Scale-Gemma-2-2B

**Date:** 2026-07-24 · **Model:** `vandijklab/C2S-Scale-Gemma-2-2B` · **Data:** 3,000 cycling K562
(`biotensor/data/cellcycle/k562_substrate_for_state.h5ad`) · **Inspired by:** Goodfire, *Manifold Steering
Reveals the Shared Geometry of Neural Network Representation and Behavior* (arXiv 2605.05115).

## Question
Goodfire showed that concepts with a natural ORDER or CYCLE occupy CURVED paths in LLM activation space, and
that steering ALONG the path (rather than in a straight line) produces smooth, on-distribution behavioural
trajectories. **Is there such a manifold inside a single-cell LLM — and is it causally used?**

The cell cycle is the exact biological analogue of Goodfire's days-of-the-week circle: a true closed loop
(G1 → S → G2M → G1) with known adjacency and unambiguous marker readouts.

## Ground truth (validated in expression space, before any model was touched)
Continuous phase angle from a tricycle-style construction (PCA on 87 cell-cycle genes, θ = atan2(PC2, PC1)):

| phase | circular mean | concentration |
|---|---|---|
| G2M | 20.3° | 0.89 |
| G1 | 150.4° | 0.94 |
| S | 249.5° | 0.79 |

Correct cyclic order (G2M → G1 → S → G2M), tight arcs, and S/G2M module scores in **quadrature (109°)** — a
genuine cycle. Per-bin marker readouts are textbook (G2M: TOP2A/CENPF/CCNB1/AURKA; G1: LGALS1/MT2A/TPM1;
S: PCNA/MCM3-4/CLSPN/TYMS/RRM2/histones) with **0% marker overlap between opposite bins**.

## ⚠️ CORRECTION (post-hoc controls, run 2026-07-24 after the first draft)
Two controls were run after the fact. One **supports** the discovery claim, one **refutes part of it**. Read
this section before Result 1 — some language below it is corrected here.

**(a) Input-geometry null — SUPPORTS the claim.** Phase was *defined* as the angle in a PCA plane of
cell-cycle-gene expression, so a ring exists in the input by construction and the model had to be compared
against it (`input_null.py`):

| representation | circ corr | median err | planar | monotone ring? |
|---|---|---|---|---|
| `expr_cc` — the definition (calibration only) | 0.978 | 4° | 0.95 | ✓ |
| `expr_all` — full expression (model never sees) | 0.577 | 34° | 0.79 | ✓ |
| **`expr_rank` — the encoding the model receives** | **0.512** | **40°** | 0.93 | **✗** |
| **model activations @L21** | **0.898** | **13°** | 0.85 | ✓ (uniform 30°) |

The model's own input supports phase only at 40° error with **no** monotone ring; the model's internal state
reaches 13° with a clean uniformly-spaced ring. It **sharpens** phase ~3× and creates ring structure not
linearly present in its input. *(Fair caveat: `expr_rank` is a linear readout and the model is nonlinear, so a
nonlinear input probe would close part of this gap; the defensible claim is the standard one — the model makes
linearly decodable what was not.)*

**(b) Compute-vs-relay — SUPPORTS the claim.** Re-extracted with all 87 cell-cycle genes **deleted from the
cell sentences** (`ring_analysis.py`), so the model cannot read the genes that define the label:

| layer | normal | cc-genes DELETED | retention |
|---|---|---|---|
| L09 | 0.737 (26°) | 0.624 (36°) | 0.85 |
| L15 | 0.847 (17°) | 0.785 (21°) | 0.93 |
| **L21** | **0.898 (12°)** | **0.855 (15°)** | **0.95** |

Phase survives almost intact. The model infers cycle state from the rest of the transcriptome, not from the
marker genes handed to it. *(Caveat: the cell cycle modulates transcription genome-wide, so this rules out the
trivial "copies the markers" account, not "reads a distributed expression signature".)*

**(c) Ring vs disk — REFUTES the "ring" language at the single-cell level.** Projecting every **cell** (not the
bin centroids) onto the fitted phase plane:

| | radius CV | fraction inside ½·median radius |
|---|---|---|
| observed @L21 | 0.38 | 0.108 |
| synthetic RING calibration | 0.15 | 0.000 |
| **synthetic DISK calibration** | **0.35** | **0.126** |

The single-cell cloud matches a **filled disk**, not an annulus. The earlier statement that the model "collapses
onto a low-dimensional ring" is **wrong at the single-cell level** — it was true only of the 12 bin centroids,
and averaging ~250 cells per bin makes almost any phase-structured cloud look ring-like. Likewise the quoted
`participation ratio 2.6` and `radius_cv 0.20` were computed on centroids and must not be read as single-cell
dimensionality.

**Corrected statement of the geometry:** phase is encoded as a genuine **angular coordinate** in a 2-plane
(12° median error per cell, and the phase-ordered centroid path is 85% planar with uniform 30° spacing), but
cells **fill the disk** rather than lying on a 1-D curve — radius carries additional variation (plausibly how
strongly a cell is cycling). The steering results are unaffected: they steer along the centroid path, which is
a valid 1-D path, and carry a norm-matched random control.

## Result 0 — CONSTRUCTIVE test: build an input at phase θ, does the representation go to θ? (`synth_sweep.py`)

Everything else here is observational on the discovery side (real cells in, phase label from the same cells).
This is the **sufficiency** counterpart to the gene-deletion **necessity** test: construct a cell that says
"I am at phase θ" and see where its representation lands.

**Design.** Backbone = a real cell's ranked gene list with all 99 cell-cycle genes stripped (phase-neutral,
on-distribution). Module = 8 genes whose **external** peak-phase annotation (Cyclebase / Whitfield 2002
synchronized time-course — a temporal measurement, *not* this dataset's covariance) is nearest θ, inserted at
**fixed rank positions** identical across all conditions. Readout = projection onto the phase plane fitted on
**real** cells, so a synthetic point landing at the right angle means it reached the *same* representation.
Controls: `random` (8 random non-cycle genes) and `shuffled` (**the same 43 cycle genes with phase labels
permuted** — this is the control that kills "any cycle genes make it look cycling").

**Result @L21 — angular separation between G1/S-modules and G2/M-modules, paired per backbone (n=20):**

| condition | separation |
|---|---|
| **annotated** | **114.3° ± 6.9** |
| shuffled (same genes, permuted labels) | 13.2° ± 3.7 |
| random (non-cycle genes) | 3.5° ± 0.8 |
| *real cells of those phases, for reference* | *175.7°* |

The construction **works**: phase-annotated modules rotate the representation by 114°, versus 3.5° for random
modules and 13.2° for the identical genes with scrambled phase labels (≈33σ and ≈10σ). The effect is driven by
**which phase the genes belong to**, not by cell-cycle gene content.

**Radius behaves exactly as the disk interpretation predicts** (angle = phase, radius = coherence/strength):
annotated 0.752 (0.87× real) · random 0.697 · **shuffled 0.508** — an incoherent mixture of cycle genes pulls
the point *toward the centre* (ambiguous phase), while a coherent module keeps it out at real-cell radius and
moves it **angularly**. So the response is a phase-manifold response, not a cell-cycle-intensity response.

**Honest limits.**
- Synthetic modules reach only **~65% of the real angular separation** (114° vs 176°) and land at a systematic
  **80–116° offset** from where real cells of that phase sit: 8 inserted genes are a far weaker signal than a
  real cell's full coordinated programme, so these are *partial displacements from the neutral backbone*, not
  full phase states.
- **Within-arc resolution is poor** — targets 0°/30°/60° all land at ≈105°. This discriminates two phase
  classes, it does not finely track phase.
- **Loop closure is NOT testable** with annotation alone, and the `loop_gap = 0°` printed by the script is an
  **artifact**: ~40% of the circle (roughly 270–330°) has no phase-specific transcripts, so those targets return
  essentially the same module as 0° and trivially land in the same place. Do not read it as the loop closing.
- Circular correlation across targets is a poor statistic here (effectively two clusters, and its sign depends
  on an arbitrary frame handedness); the paired two-class separation above is the meaningful test.
- The absolute landing angles are **not** interpretable: the PCA phase origin is arbitrary and sits ~180° from
  Whitfield's G1/S = 0 convention (visible directly — the co-expression module at θ=0 is mitotic, CENPE/PLK1,
  while the annotated θ=0 module is G1/S, E2F1/MCM). Only separations and control comparisons mean anything.

### Secondary arm (co-expression-derived modules) — ATTEMPTED AND FAILED; loop closure remains UNTESTED
To make loop closure testable, the same sweep was re-run with modules defined from **this dataset's own
covariance** (top genes enriched in cells at each phase, stopword-filtered), which covers all 12 phases
uniformly. This is circular by construction, so only the *geometry* would have been readable from it. It does
not get that far — **the arm is a broken instrument**:

| L21, paired two-class separation | value |
|---|---|
| co-expression "real" modules | 27.9° ± 5.5 |
| random modules | 3.0° ± 0.8 |
| **shuffled-label control** | **65.4° ± 8.6** ← *beats the real arm* |

A control outperforming the signal means the arm carries no valid information (circular correlation is ≈0 for
the real arm at every layer: −0.03 / +0.03 / +0.02). **Diagnosis:** bin-enrichment picks genes by *mean
difference*, which is dominated by overall expression level, whereas peak-phase annotation picks genes by
*phase specificity*. The resulting modules are near-duplicates across adjacent bins (0°/30°/60° all lead with
HIST1H1B, HIST2H2AC), so the two classes being contrasted receive nearly identical inputs.

**Consequence: loop closure is still untested**, and the honest scope of the constructive experiment stays as
stated above — a two-class (G1/S vs G2/M) discrimination, not a full circular sweep. Filling the circle would
need a genuinely phase-specific module source for the ~40% of the cycle that has no phase-specific
transcription; bin-enrichment is not it.

## Result 1 — the manifold exists, and the model BUILDS it with depth
Cell-state = residual at the last token (`--unit cell-summary`), 10 taps. Phase decodability = cross-validated
ridge on (cos θ, sin θ), scored by circular correlation vs a label-shuffle null.

| layer | circ corr | median err | chord_dev | participation ratio | radius_cv |
|---|---|---|---|---|---|
| L00 | 0.068 | 81° | 0.95 | 6.5 | 0.25 |
| L04 | 0.065 | 83° | 0.99 | 7.8 | 0.26 |
| L06 | 0.321 | 60° | 0.98 | 7.4 | 0.26 |
| L09 | 0.737 | 26° | 1.07 | 5.2 | 0.21 |
| L15 | 0.847 | 17° | 1.29 | 3.8 | 0.18 |
| L18 | 0.889 | 13° | 1.62 | 2.7 | 0.17 |
| **L21** | **0.898** (z=+44) | **12°** | **1.74** | **2.6** | 0.20 |

- Phase is **absent at the embedding and constructed progressively** — this is computation, not input echo.
- With depth the structure collapses onto a **low-dimensional ring**: participation ratio 7.8 → **2.6**,
  constant radius (radius_cv ≈ 0.2), while chord deviation *rises* 0.95 → **1.74**.
- Median angular error **12°** against a 90° chance baseline.

## Result 2 — the manifold is CAUSALLY USED (and random directions do nothing)
Steering = add a displacement to the cell-state position at layer L; readout = **the model's own next-gene
logits** over per-bin marker genes (no fitted probe → non-circular).

**Pre-registered instrument gate** (the failure that killed an earlier scGPT attempt, whose readout scored
circ r = 0.19): the unsteered readout must decode phase. **PASS: circ r = +0.739.**

Progress toward the target phase (radians, α = 0 → 1):

| arm | α=0.25 | α=0.5 | α=0.75 | α=1 |
|---|---|---|---|---|
| manifold (arc) | −0.04 | +0.29 | +0.65 | **+0.82** |
| linear (chord) | +0.13 | +0.31 | +0.52 | **+0.82** |
| **random (norm-matched)** | +0.007 | +0.004 | −0.002 | **+0.007** |

Steering along the manifold moves the model's own predictions ~0.8 rad toward the target phase; a
**norm-matched random direction of identical magnitude does nothing**. The manifold is a real causal handle.

## Result 3 — manifold vs linear: representation–behaviour ISOMETRY
Both arms end at the same knot, so "did it reach the target" cannot separate them. The discriminating statistic
is **how far the behaviour travels around the ring** (sum of |Δ readout phase| across the sweep):

| | L21 | L18 | L21 *matched* |
|---|---|---|---|
| manifold | **1.968 rad** | **0.930 rad** | **2.287 rad** |
| linear | 0.945 rad | 0.474 rad | 1.319 rad |
| random | 0.189 rad | 0.139 rad | 0.254 rad |
| paired ratio (mfld/lin) | **2.08×** [1.68, 2.67] | **1.96×** [1.63, 2.33] | **1.73×** [1.41, 2.17] |
| cells with manifold > linear | 88% (p = 1.4e-4) | 92% (p = 1.8e-5) | 75% (p = 0.011) |
| activation-space arc/chord | 2.37× | 2.38× | 2.37× |
| **behaviour/representation ratio** | **0.88** | **0.82** | **0.73** |

*matched* = steering sentences built with the same 512-gene cap as the knots (removes a length mismatch present
in the first two runs); instrument gate there is the strongest of the three (circ r = **0.789**). The effect
replicates in all three runs — ratio 1.7–2.1×, every CI excluding 1.

**The behaviour follows the path you steer along.** Steering along the arc drives the model's output *the long
way around the cycle*, through the intermediate phases; the chord cuts across. And the ratio of behavioural path
lengths (2.08×) closely tracks the ratio of activation-space path lengths (2.37×) — a near-isometry (0.82–0.88),
which is precisely Goodfire's M_h ↔ M_y correspondence, here in a single-cell LLM.

## What we did NOT reproduce
Goodfire's "linear steering produces incoherent, off-target output" claim. Readout concentration and peak decline
**identically** for manifold and linear (0.140 → 0.080 at L21) — we see no differential off-manifold collapse.
On this readout the difference between arms is **which path the behaviour traverses**, not output coherence.

## Caveats (kept explicit)
- The bin readout is only weakly peaked (peak ≈ 0.14 vs 0.083 uniform); the instrument works through the
  circular mean (gate 0.739), not through a sharp argmax — so argmax-based adjacency/jump metrics are noise-
  dominated here and were discarded in favour of path length.
- n = 24 cells/layer (32 in the matched re-run), one cell line (K562), one variable.
- The cell cycle is the most tightly co-expressed program in transcriptomics, so this manifold is substantially a
  co-expression clock. That does **not** invalidate the arm comparison: manifold, linear and random share the
  identical gene module and differ only in the direction rule, so co-expression cannot manufacture the arm gap.
- Prior in-house art (`route_cellcycle/RESULTS.md`) proves the loop advantage is **topological, not curvature-
  dependent**: ∮w·dx = 0 means a fixed direction must stall (~π/2 of arc) even on a perfectly flat circle
  (verified there: 0.34 laps fixed vs 4.53 local-tangent). Our single-shot dose sweep is the Goodfire-style
  comparison; an iterative lap-counting experiment is the natural follow-up.

## Result 4 — hypothesis #2: the DIFFERENTIATION manifold (open curve) — discovered, causal, but no manifold-over-linear gain

Same harness on hematopoiesis (Setty CD34+ bone marrow), ordering variable = palantir pseudotime.

**Discovery — a cleaner manifold than the cell cycle.**

| | decodability | participation ratio | chord_dev |
|---|---|---|---|
| all 10 lineages (4,000 cells) | ρ = 0.926 (z=+64) @L25 | 4.2 → 2.1 | 1.42 |
| **erythroid branch only** (3,109 cells: HSC_1/2 → Ery_1/2) | **ρ = 0.957** @L25 | **1.7–2.0 (essentially 1-D)** | **0.90–1.05** |

Restricting to one unbranched lineage yields a nearly perfect 1-D curve — the highest decodability in the whole
programme — but the **lowest curvature**. Geometrically expected: a closed loop must curve, an open
developmental curve need not.

**Two instrument failures, both diagnosed (this is where the work was):**
1. *All-lineage, 12-bin marker readout* — gate FAIL on amplitude (corr 0.62 but only **3.2%** dynamic range).
   Cause: hematopoiesis **branches**, so late-pseudotime bins mix Ery/Mono/DC and their markers cancel.
2. *Erythroid, 2-pole readout* — gate FAIL (corr −0.19). Cause: the raw DE contrast is swamped by
   **stopwords** (END pole = HBB, then MT-CO3/MT-CO2/MT-ND4/MALAT1), because late erythroid cells shift library
   composition. Applying the Thread-B stopword filter recovered the canonical programme
   (START: AVP, MDK, GNG11, EVI2B · END: **HBB, BLVRB, AHSP, CA1, KLF1, UROD, GLRX5**).

**Causal result (clean instrument, stopword-filtered poles, L25):**

| arm | dose curve (α = 0 → 1) | Δ at α=1 |
|---|---|---|
| manifold | 0.008, 0.014, 0.019, 0.029, 0.038, 0.055, 0.063 | **+0.074** |
| linear | 0.005, 0.022, 0.028, 0.038, 0.047, 0.055, 0.068 | **+0.074** |
| random | ≈ 0 throughout | −0.008 |

Steering along the differentiation manifold **causally drives the model's own erythroid programme** (monotone
dose-response, SNR **4.4×** vs a norm-matched random direction). But **manifold and linear are
indistinguishable**: readout path ratio **1.01× (95%CI 0.91–1.12, 50% of cells, p = 0.58)**.

**Why — and the general principle this establishes.** Two reasons, and the second is a caveat on the method itself:
1. The erythroid manifold is nearly straight (chord_dev ≈ 1.0), so arc ≈ chord.
2. **A monotone scalar readout cannot register a detour**: the total variation of any monotone path from A to B
   is fixed by its endpoints. So path length — the statistic that revealed the 2× cell-cycle effect — is
   *degenerate by construction* for an ordered variable read out as a scalar. (Activation arc/chord here is
   1.82×, yet behavioural ratio is 1.01× — the isometry that held for the cycle, 0.73–0.88, collapses to 0.55.)
   This is a limitation of the metric, so #2 is **"no advantage detected"**, not a strong null.

> **The emerging principle: manifold steering buys you something in proportion to how curved the manifold is,
> and only when the readout can represent the detour.** Closed loop (cell cycle, chord_dev 1.74, cyclic
> readout) → 1.7–2.1× advantage. Near-straight open curve (erythroid, chord_dev 1.0, monotone readout) → none.

## Files
`cc_phase.py` (cyclic ground truth) · `manifold_fit.py` (discovery: decodability, ring, curvature) ·
`manifold_steer.py` (manifold vs linear vs random + instrument gate) ·
`results/manifold_fit_cc.json`, `results/manifold_knots_cc.npz`, `results/manifold_steer_L{18,21}.json`.
