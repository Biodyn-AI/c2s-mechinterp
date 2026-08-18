# Is the cell-cycle manifold CONTINUOUS or a few DISCRETE states?

**Date:** 2026-07-24 · **Model:** C2S-Scale-Gemma-2-2B, layer 21 · **Code:** `continuity_test.py` ·
**Result:** `results/continuity_L21.json`

## Answer: continuous — with a non-uniform (stretched) metric

Three independent tests, each scored on a 0 (fully continuous) → 1 (fully discrete attractors) scale:

| test | statistic | discreteness score |
|---|---|---|
| **behavioural interpolation** (decisive) | largest-step share **0.257** (gradual 0.10 · snap 1.00) | **0.17** |
| **metric profile** | largest step **0.172** of the total rise (smooth 0.08 · step 1.00) | **0.10** |
| **occupancy clumping** | decoded entropy **0.976** vs true **0.952**, **0 empty bins** | **0.00** |

**Overall: ~10–17% of the way toward discreteness — i.e. predominantly continuous.**

### 1. Occupancy — the model does not clump cells
Decoded-phase density has **higher entropy than the true phase density** (0.976 vs 0.952) and **zero empty
bins**. A discrete-attractor code would do the opposite: *lower* entropy than the data and gaps between the
attractors. The model is, if anything, **more uniform around the loop than the biology is**.

### 2. Metric — distance grows smoothly with phase difference
Mean representation distance vs |Δphase| rises monotonically in 82% of bins, with the largest single step only
**0.172** of the total rise (a genuine step function would be ~1.0, perfectly smooth ~0.08):

```
0.0803 0.0809 0.0826 0.0855 0.0876 0.0881 0.0901 0.0932 0.0940 0.0924 0.0911 0.0929
```

No plateau-then-jump structure. (The mild non-monotonicity at large Δphase is expected on a *closed* loop, where
distance saturates as points approach the antipode.)

### 3. Behavioural interpolation — the decisive test
Interpolating two cells' states, `h(λ) = (1−λ)h_A + λ h_B`, injecting `h(λ)` in place of the last-token residual,
and reading the **model's own marker-gene logits** (a *linear* decoder would report intermediate phases by
construction and is useless here; the model's output is nonlinear).

- **largest-step share = 0.257 ± 0.129** — much nearer perfectly-gradual (0.10) than pure-snap (1.00).
- **mid-interpolation entropy excess = +0.0056 (positive)** — the intermediate state is a *genuine mixture*, not
  a commitment to one attractor. A discrete-attractor model would be *more* committed mid-way, not less.

The model's own predictions therefore pass **gradually through intermediate cell-cycle phases** as the
representation is interpolated. There is no snapping.

## The one genuinely non-uniform feature — and it is biologically sensible
Consecutive knot spacing around the loop is **not** uniform, and the pattern **reproduces across model scale**:

| | CV of gaps | max/min | where the largest gaps sit |
|---|---|---|---|
| 2B L21 | 0.318 | 2.52 | bins 5→8 |
| 27B L44 | 0.295 | 2.45 | bins 5→8 |
| *uniform circle* | 0 | 1.0 | — |
| *3 discrete states* | ~1.0 | ≫3 | — |

Bins 5→8 span the **G1 → S transition** — the restriction point, the cell cycle's major commitment step. The
representation moves ~2.5× further per unit phase there than elsewhere. So the manifold is a continuous ring
with a **stretched metric at the biologically decisive transition**, not a set of discrete states.

This also reconciles the one earlier observation that looked discrete: the constructive sweep resolved only ~2
classes (G1/S vs G2/M). That is a limit of the **input probe** (only two arcs are addressable from external
peak-phase annotation, and 8 inserted genes are a weak signal), **not** a property of the representation — the
representation itself resolves phase to 12° and interpolates smoothly.

## Supporting evidence from earlier experiments
- Phase decodes to **12° median error** — ~30 distinguishable positions, far beyond 3 states.
- The lap walk advances smoothly: only **0.2%** of steps are near-zero increments (a 3-attractor traversal would
  show ~60%+ dwell steps), with a tight unimodal increment distribution (CV 0.36).
- The 12 phase-bin centroids lie in one plane at **uniform 30° angular spacing**, monotone in phase.

## Caveats
- One layer (L21), one model for the interpolation test (2B), n = 24 pairs.
- K562 is a fast-cycling cancer line; a primary-tissue population with a real G0 compartment could well show a
  genuine discrete state (quiescence) that this cell line cannot exhibit.
- "Continuous" here means *no attractor snapping and no density gaps at the resolution tested* (12 bins / 36
  density bins); a finer discreteness than ~10° would not be detected.
