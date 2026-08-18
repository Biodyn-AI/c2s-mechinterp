# The three controls

These are the methodological contribution of the paper. Each one changed a conclusion in this study;
one reversed it. They transfer to any biomedical foundation model.

## 1. Information-matched baselines
**Encode the baseline with exactly the information the model receives.**

A C2S cell sentence is the top-512 genes *in rank order, without magnitudes*. Comparing the model
against the full expression vector charges the model for its own tokenizer; comparing it against a
carelessly encoded subset flatters it.

What this cost us: an early analysis appeared to show the model sharpening a biological signal
~3× over its own input. That gap was an artefact of the baseline encoding (1/rank weighting collapsed
the baseline's PCA onto a few genes). Encoding the baseline properly reduced the model's advantage to
**0.007**. Report the whole ladder — full expression, top-512 with values, top-512 as a binary set,
top-512 set plus rank — so the cost of tokenisation is separated from the contribution of the model.

Also match the **selection procedure**, not just the encoding: if the sentence picks its top-512 from
the cell's full panel, the baseline must too.

Implementation: `manifold/encoding_matched_transfer.py`, `manifold/input_null.py`,
`manifold/selection_matched_arm.py`.

## 2. Co-expression-matched nulls
**The null must preserve the dominant correlation structure of the data.**

Co-expression can imitate almost any biological signal, because pathways and cell types are largely
*defined* by co-expression. A null that shuffles features or labels without preserving co-expression
is vacuous.

In `geometry/gene_manifold.py` the binning on (co-expression × mean expression) matches co-expression
by construction, so the honest null permutes the gene→function assignment and recomputes the
within-bin statistic. This converted an ungated point estimate (Δcos ≈ 0.06, significance unknown)
into a gated finding (z = 3.9–6.6, p ≤ 0.01). Elsewhere the same discipline dissolved apparent
structure — see `manifold/ctx_coexpr_null_c2s.py`.

## 3. Capacity-matched scaling
**Equalise features per unit of training data before comparing model sizes.**

At a fixed expansion factor a wider model gets a proportionally larger SAE dictionary trained on the
same number of positions, so its features are systematically under-trained. Comparing a 27B model's
4× dictionary (18,432 features, ~33 positions/feature) against a 2B model's 4× dictionary (9,216
features, ~67 positions/feature) confounds scale with under-training.

Retraining the 27B autoencoders at 2× expansion — same absolute dictionary, same positions per
feature — recovered ~10 points on each interpretability measure. That part was artefact. The
remaining gap (31× more dead features, 19 points lower annotation) was real.

Implementation: `c2s_mechinterp/train_sae.py --expansion 2 --sae-subdir sae_matched_x2`, compared
with `c2s_mechinterp/compare_scaling.py`.
