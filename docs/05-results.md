# Results and their provenance

Every number in the paper traces to a file. This table says which.

| Claim | Value | Artefact |
|---|---|---|
| SAE reconstruction, 2B | variance explained 0.87–0.99 (mean 0.93) | `results/2b_atlas_RESULTS.md`, `sae/train_L*.json` |
| Dead features, 2B | mean 1.2% | `sae/train_L*.json` |
| Ontology annotation, 2B | mean 68% of features per layer | `feature_annotations_L*.json` (`summary.annotation_rate`) |
| Superposition | 0–1 SVD-aligned features/layer (99.99% novel) | `feature_catalog_L*.json` (`summary.n_svd_aligned`) |
| Cross-layer persistence | 50.4% at layer 25, mean cosine 0.877 | `results/cross_layer_tracking.json` |
| Causal edges | 27,633 edges, 45 sources → 7,959 targets | `results/circuit_analysis.json`, `circuit_graph.json` |
| Shared-ontology edges | 10,919 / 12,816 = 85.2% | `results/circuit_analysis.json` (`biological_motifs`) |
| Perturbation detection | 96.7% (58/60 targets) | `results/phase8v2_L16.json` (`summary.detection_rate`) |
| TF-target specificity | 2.1% (1/48, BH q<0.05) | `results/phase8v2_L16.json` (`summary.tf_specificity_rate`) |
| Scaling, matched capacity | dead 1.2% / 37.7% / 47.5%; annotated-of-live 73.4% / 54.5% / 44.9% | `results/RESULTS_matched_control_27b.md` |
| Gene geometry beyond co-expression | Δcos 0.062/0.044/0.055; z 6.57/3.86/4.52; p ≤ 0.01 | `results/RESULTS_gene_manifold_null.md` |
| Cell-state geometry | model curvature 0.023 vs expression-PCA 0.102 | `results/RESULTS_gpu_2b.md` |
| Steering path length | manifold 1.97/0.93/2.29 rad vs linear 0.95/0.47/1.32 | `results/RESULTS_manifold.md` |
| Information ladder | 0.929 / 0.885 / 0.882 / 0.875 / 0.868 | `results/RESULTS_cc_benchmark.md` |
| Public atlas | 13 layers, 119,808 features, 88,137 annotated, 188 modules | atlas `global_summary.json` |

## Reading the negatives correctly

**"No regulatory logic" is not "the model is useless".** The model detects nearly every knockdown; it
simply does not encode which transcription factor acts on which target. Treat its outputs as
statements about co-expression, not regulation.

**"Bigger is less readable" is about interpretability, not capability.** We did not measure downstream
task performance of the 27B model. The claim is that its residual stream is more anisotropic —
higher reconstruction fidelity from fewer live features, which align less well with curated biology.

**"No gain over the input" is cell-level only.** Given a gene *name* on a neutral background, the
model sorts genes by cell-cycle phase at AUROC 0.989 while an anagram of the same letters is at
chance. The gene→role mapping is genuinely in the weights.

## Known limitations
One cell line (K562) for the atlas and circuits; one progenitor tissue for geometry. Circuit tracing
at reduced scope (50 cells, 15 features, 3 source layers). The 27B atlas covers 26 of 46 layers. Two
layers excluded from the public atlas for corrupted catalogues. Ontology annotation carries database
circularity — neutralised by the permutation nulls for comparative claims, but not for absolute
annotation rates. `geometry/gene_polysemy_NOT_RUN.py` is included but **was not run**: its only active
test has an uncorrectable token-position confound, because the extractor does not store token rank. A
v2 must emit rank at extraction time.
