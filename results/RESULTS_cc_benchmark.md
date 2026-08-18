# ⚠️ C2S on the EXISTING cell-cycle benchmark: it does **NOT** beat expression — 0 of 36

**Date:** 2026-07-24 · **Model:** C2S-Scale-Gemma-2-2B · **Code:** `cc_benchmark_c2s.py` ·
**Result:** `results/cc_benchmark_c2s.json`

## Why this document exists
It was suggested that earlier single-cell models (scGPT, Geneformer, …) failed to outperform co-expression
baselines on the cell cycle, whereas **C2S strongly outperforms them**. **That is not correct, and this file is
the correction.** The C2S manifold work in this folder never used the established benchmark, so C2S had never
been placed on the same ruler. It has now been, and it loses.

## The prior art (`biotensor/codebase/route_cellcycle`) — confirmed
The cell-cycle manifold was already run on **five** models (scGPT, Geneformer, MaxToki, STATE-SE, UCE) on this
exact substrate (3,000 Replogle K562 non-targeting controls). Their findings:

- **Models beat the expression baseline in 1 of 45** (model × k × task) cells.
- **Raw expression is the cleanest substrate of all**: linear circ-R² **0.929**, higher than any model (0.84–0.92).
- The lap-stall / local-tangent phenomenon **reproduces on naked expression with no model** (0.36 vs **5.08** laps).
- scGPT's own readout head could not read phase (r = +0.19) where a linear marker score reads r ≈ 0.95.
- Verdict: *"the data carries everything"* — for the cell cycle the model adds nothing unique.

Reproduced locally on the identical cells (k=50): EXPRESSION 0.929 / 0.907 / 0.992 vs scGPT 0.911 / 0.884 /
0.985 — expression wins all three tasks.

## C2S in the same benchmark — 0 of 36
Identical protocol (PCA to k ∈ {10,20,50} → small probe → held out), identical substrate, canonical φ and
G1/S/G2M labels from `k562_cc_substrate.npz` (verified cell-for-cell). Each cell is `order circ-R² / classify
bal-acc / S-vs-G2M AUROC`.

| representation | k=10 | k=20 | k=50 |
|---|---|---|---|
| **EXPRESSION** | **0.921 / 0.911 / 0.994** | **0.929 / 0.916 / 0.995** | **0.929 / 0.907 / 0.992** |
| C2S-2B L09 | 0.087 / 0.368 / 0.535 | 0.089 / 0.375 / 0.541 | 0.115 / 0.411 / 0.557 |
| C2S-2B L15 | 0.245 / 0.463 / 0.665 | 0.378 / 0.536 / 0.679 | 0.443 / 0.584 / 0.751 |
| C2S-2B L21 | 0.850 / 0.862 / 0.980 | 0.877 / 0.869 / 0.985 | 0.875 / 0.866 / 0.982 |
| C2S-2B L25 | 0.856 / 0.843 / 0.980 | 0.874 / 0.856 / 0.981 | 0.876 / 0.861 / 0.985 |

**C2S beats the expression baseline in 0 of 36 (layer × k × task) cells.**

It is not merely "the same as the other models" — it is **slightly worse**: best C2S ordering **0.877** vs
scGPT **0.911** vs expression **0.929**. A text LLM over ranked gene names is a *poorer* cell-cycle
representation than an encoder SCFM, which is itself poorer than the raw expression it was trained on.

## Where my earlier phrasing went wrong
The `input_null` experiment compared the model against:
- `expr_rank` (0.512) — the model's own rank-based input encoding, and
- `expr_all` (0.577) — the full transcriptome under a *ridge/circular-correlation* readout,

and the model won those (0.898). That is a real result — **the model does sharpen phase relative to what it
literally receives** — but it is **not** the "beats co-expression" claim. Against the marker-expression readout
in the same file, the model *lost* (0.898 vs `expr_cc` 0.978), and against the benchmark's expression baseline
on the proper ruler it loses everywhere. My summaries should have drawn that line explicitly.

Two further cautions on my own numbers:
- The lap-walk result is a **sixth replication** of an already-established cross-model phenomenon which also
  holds with **no model at all** — it is a property of the data geometry, not evidence of model value.
- My manifold numbers used circular *correlation* with a ridge probe; the benchmark uses circ-*R²* after PCA(k).
  Different rulers — exactly the project's documented "metric drift" trap. Only this file's numbers are
  comparable to the prior models.

## What in the C2S manifold work still stands as novel
Independent of the co-expression question, these have no counterpart in the prior work:
- **Behavioural lap-counting.** The prior work's readout gate *failed* for scGPT (r = 0.19), so it could only
  measure geometric laps. The C2S readout passed (0.74–0.79), so the model's **own next-gene logits** were shown
  to lap **1.579** vs **0.212** for a fixed direction — behaviour, not just geometry.
- **The orthogonal radial axis**: radius = cycling strength (R²_radius 0.414 vs R²_circle 0.040), the
  days-of-week / parts-of-day analogue.
- **Activation-plateau negative**, with the testable "plateaus need a discrete output alphabet" explanation.
- **Filled-disk (not ring) single-cell geometry**, and the **2B vs 27B scale-invariance** result.

## ⚠️ RETRACTION: "the model sharpens phase 3× over its own input" — that was an ENCODING ARTIFACT

The benchmark above compares C2S against a baseline with **strictly more information**: the expression baseline
sees all 6,546 genes *with values*, while the model sees only the **set of the top-512 gene names and their
order** (values discarded). That comparison is not information-matched. The fair baseline is one given exactly
what the model is given. Benchmark ruler, PCA-50:

| representation | order (circ-R²) | classify |
|---|---|---|
| FULL expression (6,546 genes + values) | 0.929 | 0.907 |
| top-512 genes **with** values | 0.885 | 0.853 |
| **top-512 SET, binary presence — the model's input** | **0.882** | 0.846 |
| **top-512 SET + rank, linear — the model's input** | **0.868** | 0.844 |
| **C2S-2B L21 (the model)** | **0.875** | **0.866** |
| top-512 SET + rank → MLP-256 (nonlinear probe) | 0.802 | 0.830 |

**The model sits exactly on top of a linear readout of its own input** (0.875 vs 0.868–0.882). A nonlinear probe
does no better (0.802). So:

- **`input_null`'s claim that the model "sharpens phase ~3×" (0.898 vs `expr_rank` 0.512) is RETRACTED.**
  `expr_rank` used **1/rank** weighting, which collapses to the top few genes and cripples PCA. Encode the same
  information faithfully (binary set membership, or rank linear in position) and the input already supports
  0.868–0.882 — level with the model. The 3× gap measured my encoding choice, not the model.
- **"Compute, not relay" (95% retention after deleting the 87 marker genes) is also NOT model-specific.** The
  matched baseline retains **0.94** under the identical ablation (0.882 → 0.827). Phase information is
  distributed genome-wide, and a linear readout recovers it just as well.

### Where the loss actually happens
| step | order circ-R² | cost |
|---|---|---|
| full expression | 0.929 | — |
| → tokenisation (values dropped, top-512 names only) | 0.882 | **−0.047** |
| → the model's representation | 0.875 | −0.007 |

**All of the loss is in the input format; the model itself is faithful.** C2S is, to within noise, an
**information-preserving reparametrisation** of its own input — it neither adds nor destroys decodable phase.

### So is it "just a reparametrisation"? — Yes, informationally. But the *form* is not trivial.
What the model changes is not *how much* phase information is present but *how it is laid out*: the G1→S metric
stretch (below) exists in the model and not in the data, and the layout is **causally manipulable** — a PCA has
no forward pass, no logits, and cannot be steered. The surviving results are about the model as a *dynamical
system*, not about it knowing more than its input.

## Audit: which geometric claims were model-specific, and which were data properties?
Run on the **expression substrate with no model anywhere** (same 3,000 cells, same φ):

| claim | expression (no model) | C2S-2B | verdict |
|---|---|---|---|
| filled disk, not ring | radius_cv **0.42**, frac-centre **0.122** | 0.38 / 0.108 | **DATA property** — expression is the same disk |
| radius = cycling strength (orthogonal axis) | R²_circle 0.036, **R²_radius 0.529** | 0.040 / 0.414 | **DATA property** — expression shows it *more* strongly |
| non-uniform metric, stretched at G1→S | CV **0.193**, max/min 1.77, largest gap at bin **10→11** | CV **0.318**, max/min 2.52, largest at bins **5→8 (G1→S)** | **MODEL-SPECIFIC** — C2S is ~1.6× less uniform *and* stretches at a different, biologically meaningful place; reproduces in 27B at the same bins |

So of the three geometric descriptions, **two were inherited from the data and one is genuinely added by the
model.** The disk geometry and the angle/radius decomposition should be attributed to the expression manifold,
not to C2S. The G1→S metric stretch survives as a model property.

## What survives, in one table

| result | survives? | why |
|---|---|---|
| phase decodable / manifold exists | ✗ deflated | expression does it better (0.929 vs 0.877) |
| filled-disk geometry | ✗ data property | expression identical |
| radius = cycling strength | ✗ data property | expression stronger |
| lap-stall / local-tangent navigation | ✗ data property | reproduces on naked expression (0.36 vs 5.08) |
| **G1→S metric stretch** | ✓ **model-specific** | expression is more uniform and peaks elsewhere; replicates at 27B |
| **behavioural lap-counting** (own logits lap 1.579 vs 0.212) | ✓ **novel** | expression has no output; prior work's readout gate failed |
| **representation→behaviour isometry (0.73–0.88)** | ✓ **novel** | no expression analogue |
| **causal steerability** (random control flat) | ✓ | you cannot steer a PCA |
| **constructive input→representation map** (114° vs 13°) | ✓ | about the model's encoder |
| **no activation plateaus** + discrete-alphabet account | ✓ | about the model's input→output map |
| **sharpening over its own input encoding** (0.898 vs 0.512) | ✓ | about C2S's internal computation |
| marker-independence (95% retention) | ~ | untested confound: expression-minus-markers might also work |
| 2B ↔ 27B scale-invariance | ✓ | a fact about the model family |

**The dividing line is clean:** claims about the model's **output** or its **internal map** survive; claims about
the **geometry of the cell population** were inherited from the data.

## Bottom line
**Deny.** C2S is in the same regime as every other single-cell model tested here — the cell-cycle manifold is
real, low-dimensional and causally steerable, but it is **in the data**, and the model does not beat expression
on any cell-cycle task. The cell cycle remains, as `route_cellcycle` concluded, a degenerate testbed for
demonstrating model value.
