# The four-stage protocol

Each stage is independent: you can run Stage 1 alone and stop, or point Stage 2 at an existing atlas.

## Stage 1 — sparse-autoencoder feature atlas
`c2s_mechinterp/{extract_activations,train_sae,feature_catalog,annotate_features,coactivation,cross_layer}.py`

1. **Cell sentences.** A cell becomes the top-512 gene symbols ranked by expression; magnitudes are
   discarded. A gene symbol may span several sub-word tokens, so tokens are mapped back to genes via
   the tokenizer's character offsets and the residual stream is mean-pooled per gene. One activation
   vector per (cell, gene) pair — a *gene-token position*.
2. **Extraction** streams activations to per-layer memmaps. Two-pass design (count rows, then fill)
   keeps memory bounded.
3. **TopK SAE per layer.** Dictionary 4× the model width, k = 32, decoder columns unit-norm.
4. **Catalogue** each feature: activation frequency, mean activation, top driving genes, SVD alignment.
5. **Annotate** against GO BP, KEGG, Reactome and TRRUST regulons using one-sided
   hypergeometric tests. Each feature is corrected across **all** eligible terms and three
   network/membership tests, including p = 1 for term overlaps below two genes. The universe is
   the full observed extraction gene list, not the selected top-gene union. STRING and TRRUST
   within-set pair counts use one-sided Poisson approximations based on universe pair density;
   TF membership uses a hypergeometric test. Uniform and gene-frequency-matched random sets
   quantify chance. Named terms and network evidence are stored separately.
6. **Co-activation modules** by Leiden clustering; **cross-layer tracking** by decoder cosine matching.

**Two implementation details that are load-bearing.** The unit-norm constraint must be applied along
the *feature* axis — normalising the other axis silently kills ~35% of features. And activations must
be stored in `float32` for the 27B model: its late-layer values exceed the fp16 range and become
non-finite.

## Stage 2 — causal circuit tracing
`c2s_mechinterp/circuit_tracing.py`, `exhaustive_mapping.py`

For each selected source feature: one clean forward pass, one ablated forward pass with the feature's
contribution removed from the residual stream via a forward hook. Downstream feature activations are
compared with online mean/variance accumulation. The edge rule requires absolute standardised
mean change greater than 0.5 and a majority sign in more than 70% of active cells. This is an
effect/consistency rule, not a multiple-testing-corrected p-value test. `revision/rev_circuit.py`
saves complete statistic vectors; circuit baselines hold the source fixed and sample annotated
targets in the same layer. Confidence intervals resample source features.

Cost scales as cells × features × layers with one unbatched pass per pair. The published run used a
reduced but complete design (50 cells, 15 features, source layers 0/7/16 → 800 forward passes per
source layer). Batching the ablation loop is the obvious speed-up and is not yet implemented.

## Regulatory-logic test
`c2s_mechinterp/perturbation_response.py`

The test measures **unsigned target-set association** after a factor knockdown. It does not
identify activating versus repressing regulation. `revision/rev_phase8_extract.py` saves per-cell
feature matrices. `revision/rev_phase8_stats.py` compares knockdowns with non-targeting controls
using two-sided Mann–Whitney tests, BH within each response family, and absolute effect greater
than 0.5. Specificity uses a one-sided Fisher test of known targets among responses and BH across
all screened factors. The perturbed gene itself is excluded from downstream-target retrieval.

Measured expression and the ranked input are tested on the **same cells** to establish whether the
assay can recover downstream target association. Factors without measurable target coverage are
reported as unevaluable, alongside coverage sensitivities. Average precision/AUPRC and AUROC are
reported with target prevalence and finite-sample random-label average-precision expectations.
RPE1 provides a second cell line; K562-trained SAE reconstruction is measured on those inputs.
A negative model result is informative only within the assay's demonstrated sensitivity.

## Stage 3 — geometry, nulls and steering
`geometry/`, `manifold/`

* **Gene geometry** (`geometry/gene_manifold.py`): gene pairs are binned jointly on co-expression and
  mean expression; within bins, same-function pairs are compared with different-function pairs. The
  null permutes gene→function labels, so co-expression is matched by construction.
* **Cell-state geometry** (`geometry/cell_state_geometry.py`): the model representation is compared
  against a dimension-matched PCA of the raw counts of the same cells on pseudotime recovery,
  curvature and lineage recovery.
* **Steering** (`manifold/manifold_steer.py`, `steer_c2s.py`): internal states are displaced along a
  target direction and the model's own next-gene predictions are read out, against a norm-matched
  random direction.

## Stage 4 — public atlas
Built and deployed from a separate repository: https://github.com/Biodyn-AI/c2s-atlas
The generator converts Stage-1/2 artefacts into the atlas JSON schema and computes UMAP layouts from
the SAE decoder directions. `revision/export_revision_atlas.py` writes a new staging directory,
requires complete significant term IDs and their numeric details, and never overwrites the active
browser data. Source checksums and annotation-family definitions accompany the export.
