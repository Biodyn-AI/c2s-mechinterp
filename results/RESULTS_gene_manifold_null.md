# gene_manifold — beyond-co-expression signal, gated by a co-expression-matched null (2026-07-25)

The C2 control (`within_bin_mean_cos_delta`) measures, within (co-expression × mean-expr) quantile bins, how
much MORE cosine-similar same-function gene pairs are than different-function pairs — i.e. functional geometry
BEYOND co-expression. It was previously a bare point estimate (~0.03–0.06); now gated with a permutation null:
the binning already matches co-expression, so shuffling the gene→functional-term assignment and recomputing the
within-bin Δ is the co-expression-matched null (200 permutations). C2S-Scale-2B, Setty CD34+, layers 9/13/17.

| layer | Δcos (observed) | null mean | null std | **z** | p (one-sided) | n_perm | n_bins |
|---|---|---|---|---|---|---|---|
| 9  | 0.0619 | −0.0009 | 0.0095 | **6.57** | 0.005 | 199 | 6 |
| 13 | 0.0439 | +0.0013 | 0.0110 | **3.86** | 0.005 | 200 | 3 |
| 17 | 0.0547 | +0.0017 | 0.0117 | **4.52** | 0.010 | 200 | 4 |

**Verdict: the signal SURVIVES.** The null sits at ~0 (shuffling function labels removes the same-vs-diff gap,
as it should), and the observed Δcos clears it at z ≈ 3.9–6.6 (p ≤ 0.01) at every layer. So the model's gene
representation geometry encodes functional relatedness beyond what co-expression + mean-expression explain — the
one positive geometry result is now a gated, significant finding rather than a descriptive point estimate.

**Caveats.** (1) Single proliferative tissue (Setty CD34+); the null controls for THAT dataset's co-expression —
a broad multi-tissue panel would strengthen generality. (2) Functional labels are GO/KEGG/Reactome; annotation-
database circularity is neutralized by the label-shuffle (both observed and null use the same label pool) but a
broader ontology set is worth trying. (3) Only 3–6 bins had enough same+diff pairs (≥20 each) — modest bin
count, strong z. (4) Magnitude (z 3.9–6.6) is comparable to but below MaxToki's +8σ on the analogous test —
consistent cross-model, not identical. Artifact: `gene_manifold_null.json`.
