# Iterative steering: linear steering PROVABLY STALLS on the cell-cycle loop, manifold steering laps

**Date:** 2026-07-24 · **Model:** C2S-Scale-Gemma-2-2B, layer 21 · **Data:** 3,000 cycling K562 ·
**Code:** `lap_walk.py` · **Result:** `results/lap_walk_L21.json`

## Why this experiment exists
Every earlier manifold-vs-linear comparison here used a **single-shot dose sweep** (source → target, α ∈ [0,1]).
That design lands on the target *by construction*, so it can never expose the deepest difference between the two
steering rules. **Iterating** exposes it.

**The theorem** (`route_cellcycle/RESULTS.md`): for any **closed** curve and any **fixed** direction w,
∮ w·dx = w·∮ dx = 0. An iterative steerer using a constant w advances while w·t̂ > 0, retreats once w·t̂ < 0, and
**converges to a stall where w ⊥ t̂ — about a quarter-lap from the start — even on a perfectly flat circle.**
Making the direction *local* (the tangent at the current position) removes the obstruction entirely.

Since "linear steering" *is* a fixed direction, this predicts a categorical failure of linear steering on any
cyclic variable — not a quantitative disadvantage.

## Implementation validated before spending GPU time
Run on a synthetic zero-curvature circle (12 knots, 2 of 50 dims):

| arm | laps | peak advance |
|---|---|---|
| fixed | 0.246 | **1.545 rad** (theory: π/2 = 1.571) |
| fixed_proj | 0.208 | 1.310 rad |
| local_tangent | 3.487 | 21.9 rad |

The harness reproduces the predicted stall to within 2%.

## Result on the real learned manifold (n = 12 cells, 200 steps, step = ¼ knot spacing)

### Geometric laps (unwrapped decoded phase)
| arm | laps | off-manifold |
|---|---|---|
| `fixed` | **+0.186 ± 0.124** | 99.9 |
| `fixed_proj` (projected onto local tangent — stays near the manifold) | **+0.216 ± 0.170** | 43.1 |
| **`local_tangent`** | **+2.096 ± 0.687** | 18.9 |
| `random` (norm-matched) | +0.009 ± 0.052 | 20.0 |
| `fixed_proj_retract` (snapped onto the path every step) | **+0.159 ± 0.165** | **0.00** |
| **`local_tangent_retract`** | **+3.452 ± 0.129** | **0.00** |

- **The stall reproduces on a learned representation**: fixed arms reach ≈ 0.19–0.22 laps, against the theoretical
  0.25 and the synthetic 0.21–0.25. (scGPT, for reference, stalled at 0.34.)
- **Local-tangent traverses**: 2.10 laps, or **3.45 laps** when kept exactly on the manifold.
- **Retraction does NOT rescue the fixed direction** — 0.159 laps at *zero* off-manifold error. The stall is
  **topological, not a drift artifact**. This is the control that rules out "it just fell off the manifold".
- Paired per cell: local_tangent > fixed_proj in **12/12 cells** (sign p = 2.4e-4), ratio **9.7×**; on-manifold
  (retracted) ratio **21.8×**.

### Behavioural laps — does the MODEL's own output go around?
The walked state **replaces** the last-token residual in a real forward pass, and the model's own marker-gene
logits are read every 20 steps:

| arm | behavioural laps |
|---|---|
| **`local_tangent`** | **+1.579** |
| `fixed_proj` | +0.212 |

**7.5×.** The model's next-gene predictions genuinely cycle through the phases more than one and a half times
under manifold steering, while a fixed direction gets the output stuck after roughly a quarter lap.

**Behaviour follows geometry at 0.75** (1.579 behavioural / 2.096 geometric) — independently reproducing the
representation–behaviour isometry measured by the dose sweep (0.73–0.88). Two different experiments, same number.

## What this establishes
This is the sharpest manifold-vs-linear result in the programme, and it is **categorical rather than
quantitative**: on a cyclic variable, linear steering cannot go around the loop *at all* — not because the
manifold is curved, but because it is **closed**. The dose sweep's 1.7–2.1× advantage was the same phenomenon
seen through a design that hid its severity.

It also completes the causal chain for the cell-cycle manifold:
- input → representation (constructive insertion, 114° vs 13°/3.5° controls)
- representation is not relayed (95% retention after deleting the cell-cycle genes)
- representation → behaviour (steering moves the model's own logits; random flat)
- **and the geometry of the representation dictates what steering can achieve** (this experiment)

## Caveats
- n = 12 cells, one layer (L21), one model (2B). The 27B lap test was not run.
- The walk is in activation space; the behavioural readout samples it every 20 steps rather than continuously.
- `local_tangent` uses the fitted knot path, so it is "manifold steering" with a known manifold — it is not a
  method for discovering the path online.
- Off-manifold drift for non-retracted arms is large (19–100× step length); the retracted arms are the clean
  comparison and they show the effect *more* strongly, not less.
