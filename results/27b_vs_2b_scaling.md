# C2S-Scale SAE atlas — 2B → 27B scaling comparison

**Question:** does scaling a Cell2Sentence LLM from 2B to 27B change what its SAE atlas looks like —
sparsity, ontology-interpretability, superposition, co-activation structure?

**Setup.** Matched pipeline both models: TopK SAE, k=32, 4× overcomplete, mean-centred residual-stream
activations at each layer, K562 control cells, ~614K gene-token positions. 2B: d_model 2304, d_SAE 9216,
26 layers. 27B: d_model 4608, d_SAE 18432, layers L0–25 (late layers 26–45 not run — see status).
Compared on layers present for both; rates (not counts) since d_SAE differs. Raw numbers:
`scaling_comparison.md` / `.json` (via `code/compare_scaling.py`).

## Matched-metric summary (mean over overlapping layers)

| metric | 2B | 27B | direction |
|---|---|---|---|
| SAE variance explained | 0.933 | **0.975** | 27B reconstructs better |
| **dead-feature fraction** | **1.2%** | **47.5%** | 27B far sparser dictionary usage |
| mean \|decoder cos\| | 0.020 | 0.013 | 27B features more orthogonal (higher-dim) |
| **ontology annotation rate** | **72.6%** | **29.0%** | 27B features far less ontology-mappable |
| SVD-aligned fraction | **0.0%** | **0.0%** | 100% novel in BOTH (superposition) |
| variance in top-50 SVD axes | 0.572 | **0.785** | 27B more low-dim-dominant (massive activations) |
| modules / layer | 23.4 | 21.0 | similar |
| module coverage | 66.1% | 31.8% | 27B features less organised into modules |

## The one clean, robust finding
**Massive superposition is scale-invariant.** At both 2B and 27B, **100% of SAE features are "novel"**
(zero align with the top-50 SVD axes at |cos|>0.7). The headline claim of the atlas program — that these
models pack far more concepts than dimensions into quasi-orthogonal directions invisible to linear tools —
**holds identically at 13× the parameter count.** This is a genuine, confound-free robustness result.

## The suggestive-but-confounded findings
On matched metrics the 27B looks **harder to interpret via SAEs**: ~40× more dead features (47% vs 1%),
less than half the ontology annotation rate (29% vs 73%), and half the module coverage (32% vs 66%). Taken
at face value this says *scaling made the biology less SAE-recoverable, not more.* **But three confounds
must be cleared before believing that:**

1. **UNDERTRAINING (the big one).** The 27B SAE has 2× the features (18432 vs 9216) trained on the *same*
   ~614K positions → **33 vs 67 positions per feature**. Halving positions-per-feature directly inflates
   dead features and depresses annotation/coverage (a feature needs enough activating examples to survive).
   Much of the "27B is sparser/less-interpretable" gap is plausibly this, not a model property. **Fix:**
   re-train the 27B SAEs at matched positions-per-feature (≥1.2M positions, or 2× overcomplete instead of
   4×) before any scaling claim.
2. **MASSIVE ACTIVATIONS.** Gemma-2's residual stream develops a few enormous-magnitude dimensions in deep
   layers (this is exactly what overflowed fp16 and broke L26+). They dominate variance → inflate both
   "variance explained" and "variance in top-50 SVD" (0.79 vs 0.57), and the SAE spends capacity tracking
   them, starving biological features. The 2B doesn't have this.
3. **PARTIAL 2B DATA.** Annotation/superposition compared on 16 layers, modules on 11 (the 2B's earlier
   flaky pull lost some files; the 2B pod is gone). SAE-quality (var/dead) is the only all-26-layer metric.

## Depth trend — the sharpest signal
Dead-feature fraction **by layer** is the clearest difference and it is depth-driven:
- **2B:** stays low throughout — 0.2% (L0) rising only to ~3.6% (L16) and back to ~1% (L25).
- **27B:** climbs steadily with depth — 27% (L0) → 38% (L13) → **60% (L15) → 72% (L23–24).**

The 27B's late layers progressively lose usable dictionary capacity — the same massive-activation growth
that overflowed fp16 at L26. So even setting the confounds aside, **deep 27B layers are qualitatively
different: their residual stream concentrates into fewer, larger directions with depth.**

## Verdict
- **Robust:** superposition is universal and scale-invariant (100% novel features at both scales).
- **Suggestive, needs a matched-training control:** the 27B's SAE features are sparser, less
  ontology-mappable, and less modular — but this is entangled with an undertraining confound
  (½ the positions/feature) and Gemma-2's massive activations. **Do not report a scaling effect on
  interpretability until the 27B SAEs are re-trained at matched positions-per-feature.**
- **Real qualitative difference:** deep 27B layers concentrate variance (massive activations) far more than
  the 2B, visible as dead features climbing to ~72% with depth.

## Next steps to make this publishable
1. **Matched-training control** — re-train the 27B L0–25 SAEs at ≥1.2M positions (or 2× overcomplete) and
   re-compare dead-frac / annotation / coverage. Decides whether the interpretability drop is real.
2. **Massive-activation handling** — normalise or down-weight the massive dims before SAE training (or
   exclude them), to see the biology-carrying subspace; and it's the prerequisite for finishing L26–45.
3. **Re-pull the missing 2B layers** are gone with the pod — would need to re-run the 2B atlas (cheap) to
   fill the annotation/module gaps for an all-26-layer comparison.
