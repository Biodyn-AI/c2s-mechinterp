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
5. **Annotate** against GO BP, KEGG, Reactome, STRING and TRRUST (Fisher's exact + BH-FDR).
6. **Co-activation modules** by Leiden clustering; **cross-layer tracking** by decoder cosine matching.

**Two implementation details that are load-bearing.** The unit-norm constraint must be applied along
the *feature* axis — normalising the other axis silently kills ~35% of features. And activations must
be stored in `float32` for the 27B model: its late-layer values exceed the fp16 range and become
non-finite.

## Stage 2 — causal circuit tracing
`c2s_mechinterp/circuit_tracing.py`, `exhaustive_mapping.py`

For each selected source feature: one clean forward pass, one ablated forward pass with the feature's
contribution removed from the residual stream via a forward hook. Downstream feature activations are
compared with online mean/variance accumulation; an edge is recorded when the standardised effect is
significant and consistent across cells.

Cost scales as cells × features × layers with one unbatched pass per pair. The published run used a
reduced but complete design (50 cells, 15 features, source layers 0/7/16 → 800 forward passes per
source layer). Batching the ablation loop is the obvious speed-up and is not yet implemented.

## Regulatory-logic test
`c2s_mechinterp/perturbation_response.py`

Does a feature encode the *direction* of regulation, or only association? For each transcription
factor, knockdown cells are compared against non-targeting controls. A feature is **responding** if
its activation shifts (Mann–Whitney U, BH-corrected). A factor is **specific** if the genes driving
its responding features are enriched for that factor's known targets (Fisher's exact, BH across
factors). High detection with null specificity means the model tracks co-expression, not regulation.

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
the SAE decoder directions.
