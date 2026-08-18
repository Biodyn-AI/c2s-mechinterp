# Cell-cycle manifold: 2B vs 27B — the full battery replicated at scale

**Date:** 2026-07-24 · **Models:** `vandijklab/C2S-Scale-Gemma-2-2B` (26 layers, d=2304) and
`-27B` (46 layers, d=4608) · **Data:** the same 3,000 cycling K562 cells · **Hardware:** 27B on an H100 80GB,
fp32 activation storage (mandatory — Gemma-2 27B late layers overflow fp16 → inf → NaN).

Taps were **depth-matched**: the 2B run used layers 0,2,4,6,9,12,15,18,21,25 of 26; the 27B run used
0,4,7,11,16,21,27,32,37,44 of 46, i.e. the same relative depths, so every comparison below is like-for-like.

## Headline: the manifold is essentially SCALE-INVARIANT

| property | 2B | 27B | verdict |
|---|---|---|---|
| peak phase decodability | **0.898** (12° median err) @L21/26 | **0.902** (13°) @L44/46 | **identical** |
| single-cell geometry (radius CV / frac near centre) | 0.38 / 0.108 | **0.38 / 0.110** | **identical — still a filled DISK** |
| compute-vs-relay (cc genes deleted, retention) | 0.95 | **0.95** | **identical** |
| activation-space arc/chord | 2.37× | 2.45× | same |
| manifold-vs-linear behavioural path ratio | **1.73×** [1.41, 2.17], 75% of cells, p=0.011 | **1.54×** [1.31, 1.81], 62%, **p=0.15** | **weaker / less consistent at 27B** |
| representation–behaviour isometry | 0.73 | 0.63 | slightly lower |
| constructive test: separation (annotated) | **114.3° ± 6.9** | **96.8° ± 8.8** (L37) | comparable, slightly smaller |
| constructive controls (shuffled / random) | 13.2° / 3.5° | 14.7° / 4.9° | identical |

**13× more parameters buys essentially nothing on this manifold.** Phase decodability, the disk geometry, and
the marker-independence are the same to within noise. Interpretation: the cell-cycle signal in a 512-gene ranked
sentence is close to **saturated** — a 2B model already extracts nearly all of the recoverable phase
information. That is consistent with the cell cycle being the most tightly co-expressed programme in
transcriptomics (an "easy" target both models max out on), and it echoes Thread B's finding that superposition
was scale-invariant, and the source route's own note that MaxToki-1B's extra contextualisation was "largely
co-expression-like".

## Depth profile — the one real difference

Phase decodability against **relative** depth:

| relative depth | 2B | 27B |
|---|---|---|
| 0.0 | 0.068 | −0.000 |
| ~0.25 | 0.321 (L06) | 0.225 (L11) |
| ~0.35 | **0.737** (L09) | 0.581 (L16) |
| ~0.46 | — | 0.710 (L21) |
| ~0.58 | 0.847 (L15) | **0.880** (L27) |
| ~0.80 | 0.898 (L21) | 0.891 (L37) |
| ~0.96 | — | 0.902 (L44) |

**2B builds the representation faster in relative depth** (0.737 vs 0.581 at ~35%), 27B catches up by ~58%, and
both finish at the same place. So the larger model does not build a *better* cell-cycle representation — it
takes proportionally *longer* to reach the same one.

## Per-experiment detail (27B)

**1. Discovery (`manifold_fit_cc_27b.json`).** Best layer **L44**, circ corr **0.902**. Same qualitative shape as
2B: absent at the embedding (−0.000 at L07), rising monotonically, saturating in the last third.

**2. Ring vs disk (`ring_analysis_27b.json`).** At every deep layer the per-cell radial statistics match the
synthetic **disk** calibration (cv 0.35 / frac-centre 0.126), not the ring calibration (0.15 / 0.000):
L27 0.41/0.121 · L32 0.38/0.107 · L37 0.39/0.120 · L44 0.38/0.110. **Scale does not turn the disk into a ring.**
The corrected 2B description therefore holds unchanged at 27B: phase is a genuine *angular coordinate*, but the
cell population fills the disk rather than concentrating on a circle.

**3. Compute-vs-relay (`ring_analysis_27b.json`).** Deleting all 87 cell-cycle genes from the sentences:
L16 retention 0.79 · **L27 0.95** · **L37 0.94**. Same as 2B (0.85 / 0.93 / 0.95). Both models infer cycle state
from the rest of the transcriptome rather than reading the marker genes handed to them.

**4. Steering (`manifold_steer_27b_L44.json`).** Instrument gate **PASS** and notably stronger than 2B:
corr 0.679 with **92.9% dynamic range** (2B: 0.789 / not directly comparable, but the amplitude here is the
highest seen). Readout path: manifold **2.575** vs linear 1.673 vs random 0.302 rad. Steering works and random
does nothing (reach +0.64 for both steered arms vs +0.24 random). But the manifold-over-linear advantage is
**weaker and not per-cell significant**: ratio 1.54× with a bootstrap CI excluding 1, yet only 62% of cells
favour manifold (sign p = 0.15). The aggregate ratio and the per-cell test disagree because the ratio is a
sum-weighted statistic — honest reading: **the effect is present in aggregate but less consistent than at 2B.**

**5. Constructive sweep (`synth_sweep_27b.json`).** The sufficiency test replicates cleanly and **grows with
depth**: paired two-class separation (annotated / shuffled / random) = L16 **43.5±6.2** / 3.5 / 7.2 →
L27 **69.9±11.8** / 11.3 / 3.5 → L37 **96.8±8.8** / 14.7 / 4.9. Controls stay flat at every depth, so at 27B —
as at 2B — the representation moves according to **which phase the inserted genes belong to**, not their
cell-cycle membership. Radius stays near real-cell radius for annotated modules (0.78–0.94×) and drops for
shuffled (0.59–0.90×), reproducing the 2B angle-vs-radius signature.

## Caveats
- n = 24 cells for steering, 20 backbones for the sweep — the same as 2B (paired designs), but small.
- The 27B steering was run at L44 (its best-decodability layer); a sweep over more layers might find a stronger
  steering layer, since best-decodability ≠ best-steerability.
- One cell line (K562), one variable (cell cycle).
- `nan` retention entries in the 27B log are layers where the cc-deleted extraction was not run (only L16/27/37
  were extracted for that arm) — not failures.

## Files
`results/manifold_fit_cc_27b.json` · `results/manifold_knots_cc_27b.npz` · `results/ring_analysis_27b.json` ·
`results/manifold_steer_27b_L44.json` · `results/synth_sweep_27b.json`
