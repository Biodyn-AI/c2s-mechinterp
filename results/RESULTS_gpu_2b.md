# Geometry GPU replications on C2S-Scale-2B (Setty CD34+ bone marrow) — 2026-07-24

Chained on the Stage-2 pod after circuit tracing. Dataset: `setty19_cd34_bm.h5ad` (5780 cells, 8
hematopoietic lineage clusters, palantir pseudotime, gene-symbol var_names, `spliced` raw counts).
All three drafts adversarially reviewed before running (workflow wf_a1cf1533-3ae); patches applied.
Results JSON: `geometry/results/gene_manifold_gpu.json`, `geometry/results/cell_state_gpu.json`.

## 1. gene_manifold — WEAK POSITIVE (beyond co-expression), descriptive only

Extracted gene-token activations (layers 9/13/17, 4000 cells, 2.03M gene-rows). Trustworthy number is
the **C2 within-co-expression-bin Δcos** (the raw/nostop enrichment z's are co-expression-BLIND and, per
review, stopword-contaminated — CD74/RACK1/S100A4 survive the HK filter, `--df-frac` is dead code — so
they are NOT reported as the result).

| layer | participation ratio | TwoNN id | within-coexpr-bin Δcos (C2) | n_bins |
|---|---|---|---|---|
| L09 | 132.1 | 5.28 | **+0.062** | 6 |
| L13 | 95.6 | 6.16 | **+0.061** | 3 |
| L17 | 93.9 | 5.00 | **+0.032** | 3 |

- **High intrinsic dimensionality** (PR ~90–130; TwoNN ~5–6) — consistent with the earlier local finding
  that C2S gene geometry is high-dim, opposite of scGPT's low-dim gene space.
- Functionally-related gene pairs are ~0.06 more aligned (cosine) than co-expression-matched unrelated
  pairs, within the same co-expression bin — a small **beyond-co-expression** functional signal, strongest
  mid-network (L9/L13), halving by L17.
- **CAVEATS (do not over-read):** the C2 Δcos is a bare point estimate — NO permutation null / CI / FDR in
  the draft (review's open item), so significance is not established; single proliferative tissue (co-expr +
  cell-cycle not fully neutralized); bins drop to 3 at L13/L17 (low resolution). Read as *descriptive weak
  positive*, not a gated claim. Firming it up needs the null + a multi-tissue (Tabula Sapiens) dataset.

## 2. cell_state geometry — NEGATIVE (model does not beat raw expression PCA)

Layer-18 cell-summary (3000 cells). Curvature = kNN_R² − linear_R² of pseudotime in matched 20-D PCA.

| metric | C2S-2B L18 | expression-PCA baseline |
|---|---|---|
| linear R² (pseudotime) | 0.696 | 0.862 |
| kNN R² | 0.720 | 0.964 |
| curvature (kNN−lin) | +0.023 | +0.102 |
| curved vs synthetic-linear? | **yes** (synth −0.117) | yes (synth −0.043) |
| tangent rotation θ_far | 58° | 84° |
| pseudotime monotonicity ρ | 0.782 | 0.845 |
| fate/cluster balanced-acc (null 0.107) | 0.605 | 0.862 |
| **earns_its_place (beats expr-PCA)** | — | **FALSE** |

- The model **does** embed the hematopoiesis trajectory as a genuinely curved manifold (curvature +0.023 >
  its decodability-matched synthetic-linear control −0.117), decodes pseudotime (R² 0.70), and recovers
  lineage (bal-acc 0.60 ≫ 0.107 null). So the trajectory IS represented.
- **But raw expression PCA does all of it better** — higher pseudotime R² (0.86/0.96), more curvature
  (0.102), more tangent rotation (84°), better lineage recovery (0.86). The model's L18 cell representation
  does not add trajectory geometry beyond expression. Clean negative.
- **CAVEATS:** naive shuffled KFold on autocorrelated pseudotime inflates absolute curvature (affects both
  arms equally → the *relative* verdict is protected); Ridge ill-conditioned (rcond ~3e-9) → curvature
  numbers slightly unstable; `--fate-key clusters` conflates progenitor/terminal so "fate" ≈ cluster
  separability; single tissue; one layer (18) only.

## 3. gene_polysemy — NOT RUN this cycle (see review)

Dropped: sole active test has an uncorrectable token-position confound (C2S extraction stores no token
rank, unlike MaxToki) + no stopword filter + no co-expression null. Needs a v2 (extraction emits rank →
rank-residualization + HK/df filter + co-expression-matched null). See [[gene-polysemy-program]].

## Bottom line

Both results sit at the project's recurring **co-expression/expression ceiling**: the model represents the
biology (functional gene organization; a curved differentiation trajectory) but only the gene_manifold shows
a *small* signal beyond co-expression (Δcos ~0.06, ungated), while cell-state geometry does not beat raw
expression. The one thread worth pulling is the gene_manifold beyond-co-expression Δcos — with a proper
permutation null + a multi-tissue atlas.
