# The three controls

Each result is evaluated against controls that preserve the relevant input, dependence structure or dictionary capacity. These checks delimit the inference that the measurement supports.

## 1. Information-matched baselines
**Encode the baseline with exactly the information the model receives.**

A C2S cell sentence is the top-512 genes *in rank order, without magnitudes*. Comparing the model
against the full expression vector charges the model for its own tokenizer; comparing it against a
carelessly encoded subset flatters it.

Report the whole ladder: full expression, top-512 with values, top-512 as a binary set,
and top-512 with rank. On the saved within-K562 cohort, model minus rank-input circular R² is
0.0060 with paired 95% interval −0.0048 to 0.0170. On frozen K562→RPE1 transfer, the difference
is −0.0011 with interval −0.0132 to 0.0111. Neither contrast resolves a gain; these intervals
also do not establish equivalence. The within-K562 PCA/scaling is transductive, whereas the
transfer preprocessing and readout are fitted only on source cells.

Also match the **selection procedure**, not just the encoding: if the sentence picks its top-512 from
the cell's full panel, the baseline must too.

Implementation: `revision/rev_within_cellcycle.py` and
`revision/rev_selection_matched_transfer.py`; results are `within_k562_cellcycle.json` and
`selection_matched_transfer.json` in the numerical release.

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

The controlled comparison fixes dictionary size at 9,216 features, TopK at 32, training at six
epochs, the input sentences, and relative depth across the full models. Held-out inactivity and
inactivity across all catalogue positions are reported separately. Annotation is compared with
a dictionary-specific frequency-matched random-gene floor. A raw annotation rate alone is not
a measure of biological specificity.

Implementation: `revision/run_revision.py --only 'scale-*' --include-optional` after the saved
atlas inputs and extraction are available. See the reproduction guide for the complete order.


## Annotation, perturbation and scaling validation

The active annotation entry point delegates to `revision/annotate_v2.py`. All eligible terms stay
in each feature's BH family. Database snapshot sensitivity changes only the library files;
it never restores overlap-selected correction. Report both annotation rate and its matched
random-gene floor. Pair-count and TF-membership discoveries are evidence types, not shared terms.

The controlled perturbation workflow tests measured expression, ranked input and model features
on identical cells. Specificity is unsigned target-set association; known-target coverage and
positive-control performance limit interpretation. The complete screened TF family remains in
the correction, including factors with poor target coverage. RPE1 repeats the assay with the
same dictionaries, and its reconstruction quality is recorded before interpretation.

The active scaling workflow matches relative depth across the full models and fixes both
dictionary size and training epochs: 9,216 features, k = 32 and six epochs. The depth mapping is
2B {0,4,8,13,17,21,25} to 27B {0,7,14,24,31,38,45}. Float32 activation storage prevents overflow.
Residual spectra, large-coordinate magnitudes and feature firing are measured rather than
inferred from reconstruction quality. The seven matched depth pairs are the primary scale comparison; the older 26-layer 27B atlas covers only part of that model and is a separate analysis.
