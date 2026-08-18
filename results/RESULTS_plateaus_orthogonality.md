# Activation plateaus (NO) and orthogonal properties (YES) on the cell-cycle manifold

**Date:** 2026-07-24 · **Model:** C2S-Scale-Gemma-2-2B (L09/L15/L21) · **Data:** 3,000 cycling K562 ·
**Code:** `plateau_test.py`, `orthogonal_props.py` · **Results:** `results/plateau_L21.json`,
`results/orthogonal_L21.json`

Prompted by two questions: (a) does this manifold show **activation plateaus** in the sense of the LessWrong
post *"Activation plateaus: where and how they emerge"*, and (b) is everything cell-cycle-related carried by the
circle, or are some properties **orthogonal** to it (as parts-of-day are to day-of-week)?

---

## PART 1 — Activation plateaus: **NOT PRESENT**. The manifold is smooth all the way to the output.

Method, faithful to the post: interpolate a cell state `h(λ) = (1−λ)h_A + λh_B`, patch it back, record output
logits, and measure the relative distance `d(λ) = ‖y−y_A‖ / (‖y−y_A‖+‖y−y_B‖)`. A plateau gives a flat-then-jump
(step) curve; a smooth manifold gives `d(λ) ≈ λ`. Pairs stratified by whether they **cross a real cell-cycle
transition**. 21 λ steps, 3 layers.

| stratum | plateau strength (dev from linear) L09 → L15 → L21 | transition width L09 → L21 |
|---|---|---|
| within-S | 0.179 → 0.115 → **0.056** | 0.54 → 0.52 |
| across G1/S | 0.192 → 0.142 → **0.068** | 0.54 → 0.49 |
| across G2/M | 0.185 → 0.112 → **0.058** | 0.58 → 0.52 |

*Reference: a pure plateau ⇒ dev ≈ 0.5 and width ≈ 0.05; a perfectly linear response ⇒ dev ≈ 0, width ≈ 0.5.*

**Three reasons this is a clear negative:**
1. **Transition width ≈ 0.5** — the change is spread over *half* the interpolation path. There is no sharp edge.
2. **Plateau strength DECREASES with depth** (0.18 → 0.06). The published behaviour is the opposite: plateaus
   *sharpen* with depth as each MLP flattens them further. Here the response becomes **more linear** deeper in.
3. **Crossing a real transition makes no difference** — across-G1/S (0.068) is indistinguishable from
   within-S (0.056). If phases were plateaus separated by boundaries, these strata would differ sharply.

Largest-step share is 0.078–0.102 against a uniform baseline of 0.050 — i.e. only ~1.6–2× concentration, mild
non-uniformity rather than plateau structure.

**Phase-resolved sensitivity** (walk the manifold in equal arc steps, record ‖Δlogits‖ per step) shows moderate
non-uniformity (CV 0.31–0.41) but the peaks are **not layer-consistent**:

| layer | CV | top-3 peak phases |
|---|---|---|
| L09 | 0.405 | 210°, 120°, 102° |
| L15 | 0.309 | 36°, 114°, 96° |
| L21 | 0.335 | 108°, 192°, 288° |

Real boundaries would sit at the same phases across layers and align with G2M (~20°) or G1→S (~150–240°). They
do not. **Notably, the representation-space stretch at 150–240° has no output-space counterpart** — where the
representation moves fastest is *not* where the output is most sensitive.

### Why plateaus probably don't form here — a testable explanation
In the days-of-week case the **output space is discrete**: the model must emit "Monday" or "Tuesday", with
nothing in between, so the map from a continuous representation to a discrete token necessarily has flats and
cliffs. Here the readout is a **graded distribution over hundreds of marker genes** — a continuous output space
with no token boundary to snap to. Plateaus appear to be a property of *discrete output alphabets*, not of
curved manifolds per se. This predicts plateaus **would** appear in C2S if the readout were forced to be
discrete (e.g. a single next-gene-name argmax over a small candidate set) — a clean follow-up.

**So: continuous representation AND continuous behaviour.** The hypothesised "continuous code, quasi-discrete
response" synthesis is *not* supported.

---

## PART 2 — Orthogonal properties: **YES**, and the structure mirrors day-of-week vs parts-of-day

Cross-validated R² at L21 (n = 3,000; phase subspace = 2 of 2,304 dims):

| covariate | R²_circle | R²_radius | R²_full | verdict |
|---|---|---|---|---|
| G2M_score | **0.683** | 0.018 | 0.910 | **on the circle** |
| S_score | **0.609** | 0.040 | 0.798 | **on the circle** |
| mitotic_spindle | **0.603** | 0.020 | 0.833 | **on the circle** |
| replication (MCM/PCNA) | **0.529** | 0.056 | 0.717 | **on the circle** |
| histone_S | 0.446 | 0.019 | 0.810 | partly on the circle |
| library_size | 0.350 | 0.002 | 0.850 | partly (cells grow through the cycle) |
| **cycling_strength** | 0.040 | **0.414** | 0.588 | **RADIAL — orthogonal to phase** |
| n_genes | 0.138 | 0.000 | 0.840 | off-circle |
| ribosome | 0.046 | 0.012 | 0.636 | off-circle |
| stress / IEG | 0.036 | 0.001 | 0.426 | off-circle |
| MYC-growth | 0.025 | 0.027 | 0.563 | off-circle |
| oxphos | 0.067 | 0.005 | 0.266 | off-circle |
| *true_phase_cos (positive control)* | *0.829* | *0.006* | *0.795* | *is the circle* ✓ |

**Three compartments:**
1. **The circle carries phase and only phase** — phase-specific programmes load at R²_circle 0.45–0.68, every
   non-phase programme at 0.03–0.14. A ~15× separation.
2. **Cycling strength lives on the RADIUS** — R²_radius **0.414** vs R²_circle 0.040 (10× preference), and it is
   the *only* covariate with meaningful radial loading (all others ≤ 0.056). So **angle = where in the cycle,
   radius = how strongly cycling**. This is the exact analogue of parts-of-day being orthogonal to day-of-week,
   it independently confirms the filled-disk geometry, and it explains why incoherent gene mixtures pulled the
   state toward the centre in the constructive sweep.
3. **Other programmes are encoded but entirely off the manifold** — stress/IEG, MYC-growth, ribosome biogenesis
   and n_genes are all strongly predictable from the full activation (R²_full 0.43–0.84) yet nearly invisible to
   the circle. The model represents them in directions the cycle does not touch.

### Methodological caveat (important)
The `orth_fraction` column in the JSON is ≈1.00 for **every** covariate, including the phase scores, because
projecting out 2 dimensions from 2,304 removes almost no capacity. **That statistic is not discriminative and no
conclusion here rests on it.** The load-bearing numbers are `R²_circle` and `R²_radius`.

## Caveats
- One model (2B), one layer for the orthogonality decomposition (L21), one cell line.
- Module scores are correlated with each other; R² attributions are not a clean variance partition.
- The plateau negative is for *this* readout. See the discrete-output prediction above for the test that could
  overturn it.
