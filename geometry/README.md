# Stage 3 — representation geometry

| File | What it does | Status |
|---|---|---|
| `gene_manifold.py` | Does the model's gene geometry encode function **beyond co-expression**? Bins gene pairs on (co-expression × mean expression), compares same- vs different-function pairs within bins, and gates the result with a 200-draw label-permutation null. | **Run.** z = 3.9–6.6, p ≤ 0.01 (Fig. 5a) |
| `cell_state_geometry.py` | Does the model embed a differentiation trajectory better than a dimension-matched PCA of the same raw counts? Scores pseudotime recovery, curvature and lineage recovery. | **Run.** Negative — expression PCA wins |
| `local_feature_geometry.py`, `local_functional_geometry.py` | Intrinsic dimension, anisotropy and functional-organisation probes on the SAE feature space. | Exploratory |
| `gene_polysemy_NOT_RUN.py` | Does a gene's representation shift with cell context? | **NOT RUN — do not use as is.** |

## Why `gene_polysemy_NOT_RUN.py` was not run
Its only active test has an **uncorrectable token-position confound**: a gene ranks 5th in one cell
type and 500th in another, and positional drift alone reproduces the effect. On an earlier
encoder-model study this was removable because the extractor stored each occurrence's token rank; the
C2S extractor does not. The script also lacks a stopword filter (MALAT1/MT-*/RACK1/CD74 dominate every
ranked list) and a co-expression null. A valid v2 must (1) emit token rank at extraction time,
(2) residualise on rank, (3) filter housekeeping genes by document frequency, and (4) add a
co-expression-coherence-matched null. It is kept here for transparency, not for reuse.
