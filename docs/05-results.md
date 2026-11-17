# Results and their provenance

This page describes the controlled analysis and its sampling units. The older write-ups retained directly in `results/` are historical records; their annotation percentages, broad negative regulatory interpretation and partial-depth scaling numbers are superseded. See [the results notice](../results/README.md).

Numerical source paths below refer to the canonical experiment archive: `experiments/results/` is relative to the paper project, while `runs/`, `geometry/` and `route_genemanifold_c2s/` identify its associated analysis directories. Compact portable reports are tracked in `results/revision/`; complete catalogues, per-factor outputs and manifests are in the [versioned numerical release](https://github.com/Biodyn-AI/c2s-mechinterp/releases/tag/controlled-analysis-2026-11). Source paths identify provenance even when an archive member is not a Git file. Archive `results/` maps to the cited paper path `experiments/results/`; other analysis-directory paths are retained beneath the workspace root. The release inventory records every member hash.

## The complete 2B feature atlas

The atlas uses 1,200 K562 cell sentences, 614,400 cell–gene positions per layer and 9,216 TopK features at each of all 26 layers. Counts across layers are **feature instances**, not distinct biological programmes.

| Measurement | Controlled value | Source |
|---|---|---|
| Fresh held-out reconstruction variance explained | 87.3–99.3%; unweighted layer mean 93.3% | `experiments/results/atlas_reporting.json`; `2b_eval_L00.json` through `2b_eval_L25.json` |
| Inactive on held-out positions | 2,845 / 239,616 = 1.19% | Same report and fresh evaluations |
| Inactive across every catalogue position | 0 / 239,616 | Same report; complete layer catalogues |
| Named ontology/regulon annotations | 152,782 / 239,616 = 63.8% | Same report; full annotation files |
| Frequency-matched ontology floor | 34.4%; observed excess 29.4 percentage points | Same report; three saved seeded null draws per layer |
| Uniform ontology floor | 0.916% | Same report |
| Any ontology, network-pair or TF-membership evidence | 75.4%, against its matching frequency floor of 51.6% | Same report |
| Network/membership evidence without a named term | 27,902 features | Same report |
| Co-activation modules | 577; 124,302 features assigned (51.9%) | Same report; saved module memberships |
| Decoder directions aligned with a sampled top-50 singular axis | 2 / 239,616 at absolute cosine >0.7 | Same report; complete catalogues |

Held-out inactivity and inactivity over the whole catalogue answer different questions. A feature that does not fire in a finite validation sample can still fire elsewhere. Reconstruction variance explained measures recovery of the tapped activations; it is not a biological validation score.

Annotation uses the full observed gene universe and a complete per-feature BH family: all eligible ontology/regulon terms plus three network/membership tests. Terms with fewer than two overlapping genes remain in that family with p=1. Network-pair tests use a Poisson density approximation and are separated from named ontology evidence. Neither the floor comparison nor a significant term establishes that a feature is monosemantic or causal in a cell.

High annotation can coexist with a high chance floor. Layer 22 has 77.6% named annotation against an 80.7% frequency-matched floor; layer 23 has 74.9% against 77.0%. Both are below their own floors. A uniform random-gene reference alone would hide this dependence on gene occurrence frequency. Floor replicates are random gene sets, not independent trained models.

The complete atlas has stable layer/feature IDs, driving genes and weights, significant term IDs and statistics, separate network evidence and module membership. Display-limited search indices are entry points: a missing search hit is not an exhaustive absence test. The export manifest records those limits. The staged V4 transport audit decoded every resource, checked hashes and bounds, and compared every one of the 239,616 full feature-detail objects with the preserved prior export. This establishes transport integrity, not new biological validation. Source: `experiments/results/audits/atlas_v4_full_roundtrip_audit.json`. The [live atlas](https://biodyn-ai.github.io/c2s-atlas/) is deployed from `d74f5c8cf28459339f7869bb4d64e12de369ce43`. Its published metadata match the audited bundle byte-for-byte; the production HTTPS loader also matched 156 complete feature records across all 26 layers and chunk boundaries. This is a delivery/integrity check, not browser visual inspection.

## Dictionary choices change the result

Source for all tables in this section: `experiments/results/sensitivity_reporting.json`, which records the source files and hashes. Rates use **all dictionary features**; each matched floor uses the same denominator.

### Sparsity and expansion

Each of the 12 configurations uses six epochs and one training seed. Baseline metrics use fresh extraction of the saved checkpoint; the other configurations use their saved training-held-out metrics. These are configuration sensitivities, not estimates of variability across training seeds.

| Setting | Held-out variance explained (%) | Held-out inactive features (%) | Named annotation / matched floor (%) |
|---|---|---|---|
| L0 k=8, ×4 | 97.7 | 43.63 | 19.2 / 3.0 |
| L0 k=16, ×4 | 98.9 | 10.85 | 43.9 / 11.4 |
| L0 k=32, ×4 | 99.3 | 0.26 | 53.4 / 7.9 |
| L0 k=64, ×4 | 99.4 | 0.00 | 44.9 / 1.4 |
| L0 k=128, ×4 | 99.3 | 0.00 | 56.4 / 1.4 |
| L13 k=8, ×4 | 86.0 | 72.06 | 14.8 / 3.6 |
| L13 k=16, ×4 | 90.0 | 33.32 | 39.1 / 12.4 |
| L13 k=32, ×4 | 92.5 | 1.71 | 63.7 / 25.6 |
| L13 k=64, ×4 | 93.7 | 0.00 | 50.3 / 5.8 |
| L13 k=128, ×4 | 94.2 | 0.00 | 53.3 / 2.0 |
| L13 k=32, ×2 | 91.9 | 0.48 | 64.6 / 14.2 |
| L13 k=32, ×8 | 92.7 | 4.24 | 63.4 / 36.5 |

At L13, higher k improves reconstruction and reduces held-out inactivity in these fits, while annotation is nonmonotone. At layer 13, expansion changes the matched floor from 14.2% to 25.6% to 36.5% even though the raw annotation rates remain near 63–65%. The k=32, fourfold setting is a working comparison configuration, not a demonstrated biological optimum. For another denominator example, L13 k=8 has 72.1% held-out inactivity but only 11.9% inactivity across all catalogue positions.

### Token-to-gene pooling

At layer 13, the fixed mean-pooling SAE and centring vector were applied to mean, first-token and last-token gene vectors; alternative SAEs were not retrained.

| Pooling | Variance explained on the pooling comparison (%) | Named annotation / matched floor (%) | Gene-similarity matrix correlation with mean pooling |
|---|---|---|---|
| Mean | 92.6 | 63.7 / 25.6 | 1.000 |
| First token | 71.9 | 59.9 / 22.8 | 0.745 |
| Last token | 54.9 | 44.8 / 30.9 | 0.496 |

Pooling is not interchangeable under this fixed dictionary. These values do not establish that mean pooling would outperform separately retrained first/last-token dictionaries. The comparison's reconstruction score also has its own evaluation corpus; it is distinct from the training-held-out score above.

### Random dictionary directions

The control replaces learned directions while retaining learned row norms, biases and centring. It is a random-direction SAE control inside the trained language model, **not an untrained language model**. Reconstruction performance is not inferred from its annotation rate.

| Layer | Learned directions: annotation / matched floor (%) | Random directions: annotation / matched floor (%) |
|---|---|---|
| 0 | 53.4 / 7.9 | 36.0 / 1.3 |
| 13 | 63.7 / 25.6 | 75.4 / 29.8 |
| 22 | 77.6 / 80.7 | 75.7 / 82.9 |

Many random directions acquire labels. At L13 their raw annotation rate exceeds that of learned directions; at L22 both are below their respective frequency-matched floors. Annotation rate alone therefore cannot identify what SAE training adds.

### Archived versus full annotation libraries

Both library conditions use complete BH families. They differ in available terms, genes and network coverage as well as the correction-family size: 2,373 tests per feature in the archived library and 4,590 in the full library. This is a library-coverage comparison, not an isolated causal effect of publication year.

| Layer | Archived library: annotation / matched floor (%) | Full library: annotation / matched floor (%) |
|---|---|---|
| 0 | 12.2 / 1.4 | 53.4 / 7.9 |
| 13 | 12.8 / 0.9 | 63.7 / 25.6 |
| 22 | 7.9 / 0.4 | 77.6 / 80.7 |

The results are not library invariant. L22 is above the archived-library floor and below the full-library floor. Source versions, checksums and the pinned producer commit are recorded in `experiments/results/ontology_provenance.json` and the ontology `VERSIONS.json`; environment versions are in `experiments/results/environment.json`.

## Biological descriptions and fixed-dictionary transfer

At L13, 3,138 / 9,216 features carry translation/ribosome labels, 3,156 RNA-processing labels, 837 cell-cycle/mitosis labels and 55 haem/erythroid labels. Term-name categories overlap, including “other”; they are not an independent cell-state classifier. Co-activation module 1 has 382 features, with 248 RNA-processing and 207 translation/ribosome labels.

| L13 feature | Example term | q | Fold | Firing in K562 / RPE1 positions (%) |
|---|---|---|---|---|
| F9106 | Response of EIF2AK1 (HRI) to Heme Deficiency | 5.94e−10 | 154.4 | .322 / .124 |
| F1089 | SARS-CoV-1 Modulates Host Translation Machinery | 1.64e−44 | 161.8 | 2.44 / 4.79 |
| F254 | FBXL7 Down-Regulates AURKA During Mitotic Entry and in Early Mitosis | 1.10e−33 | 120.3 | 2.31 / 1.66 |

The HRI feature's leading genes are ATF-family/stress-associated names; the ribosomal feature lists RPS genes; F254 lists proteasome genes. A term name alone does not show haemoglobin expression, infection or a cell-cycle-exclusive pathway. Frequencies use the fixed K562 dictionary/mean on 300 verified non-targeting cells per line and count gene positions, not cells. Different gene composition and reconstruction quality also affect those measurements. Sources: `experiments/results/biology.json`, `lines2.json`, `lines2_freq.npz` and `audits/biology_lines_v1.json`.

For 1,744 L0→L25 decoder-matched pairs annotated at both ends, broad category overlap is 94.84% versus 93.92% for a fixed-source random annotated target. The excess is .923 percentage points, 95% source-bootstrap interval −.047 to 1.862. This does not resolve preservation above a high broad-category baseline; targets can be reused. The corresponding numerical/match arrays are released.

## Persistence and model-intervention circuits

At layer 25, 4,641 of the 9,216 layer-0 directions (50.4%) have an absolute-cosine match of at least 0.7, with mean best-match cosine about 0.88. The minimum matching fraction is 40.3% near layer 8. Across the trajectory, 5,782 layer-0 directions (62.7%) match at more than ten layers. Target reuse is allowed. These values measure directional similarity rather than one-to-one feature lineage or equality of activation functions. Sources: `runs/2b_atlas/cross_layer_tracking.json` and its retained-value checks in `experiments/results/retained_numeric_ledger.json`.

The circuit panel selects 15 features at each of source layers 0, 7 and 16 and uses 50 cells. There are 33,070 downstream effects passing the magnitude/sign-consistency rule, of which **27,633 stored edges** remain after the cap of 50 targets per source and downstream layer. The stored graph connects 45 sources to 7,959 distinct target-layer/feature pairs. These are selected model-intervention effects, not FDR-controlled discoveries of biological regulatory edges.

Of 20,924 stored edges with named terms at both endpoints, 17,399 (83.2%) share a term. The same-source, same-target-layer random-annotated-target baseline is 61.3%, an excess of 21.9 percentage points with source-bootstrap 95% interval 18.7–24.9. Randomising both annotated endpoints instead gives 28.9%; that easier reference does not preserve source selection. Source-feature resampling describes heterogeneity within the selected panel and does not remove selection bias. Sources: `runs/2b_atlas/circuit_graph.json`, complete `circuit_effects_*.npz`, and `experiments/results/circuit_baseline.json`.

Under independent normal cell-level changes, the joint edge rule reaches 80% detection at a population mean/standard-deviation ratio of 0.725 for 50 active-source cells and 0.675 for 100 on the simulated grid. This is a conditional sensitivity calculation, not a per-edge p-value or FDR guarantee. The same 15 L7 source features were checked on nested 50/100-cell panels, all firing in every sampled cell. Uncapped rule-passing sets contain 11,293 / 10,719 effects and share 10,121: Jaccard .85115, recovering 89.62% of the 50-cell set. Across all 2,488,320 tested effects, including zeros and nonpassing effects, Pearson correlation is .98242 and Spearman .77298. The shared-passing-only Pearson .99058 is separately selection-conditioned. These nested panels and dependent effects provide descriptive selected-panel stability, not independent network replication. Sources: `experiments/results/circuit_stability.json` and `circuit_stability_full.json`.

A named example is L7 F254→L9 F254, effect −12.3448 and sign consistency 1.0. Both leading gene lists include PSMA3/PSMA4/PSMA5, sharing GO:0010498 proteasomal protein catabolism with fold 23.99 and full-family q 7.20×10⁻¹⁸/7.14×10⁻¹⁸. This is a model intervention with an annotation, not a cell protein-degradation mechanism. Source: `experiments/results/audits/circuit_proteasome_worked_v1.json`.

## Matched scale, residual structure and coordinate removal

The primary comparison uses the same 614,400 sentence positions, 9,216 features, k=32 and six epochs at seven approximately matched relative depths, covering both final taps. The 27B activations are stored in float32. Held-out inactivity and all-position inactivity are different quantities; all 2B catalogue features fire somewhere.

Each slash-separated cell below gives **2B / 27B** values. Annotation and its floor in this table use catalogue-active features. The last column compares annotation excess on the **all-feature** denominator.

| 2B / 27B tap | Held-out inactive (%) | Held-out VE | Named annotation (%) | Weighted floor (%) | 27B minus 2B excess (points) |
|---|---|---|---|---|---|
| 0 / 0 | 0.2604 / 13.98 | 0.9931 / 0.9899 | 53.43 / 38.97 | 7.896 / 3.528 | -10.29 |
| 4 / 7 | 0.4449 / 19.98 | 0.9812 / 0.9828 | 50.95 / 39.66 | 3.469 / 4.043 | -12.18 |
| 8 / 14 | 0.803 / 33.17 | 0.9588 / 0.9745 | 56.08 / 36.02 | 8.218 / 7.517 | -20.03 |
| 13 / 24 | 1.714 / 67.21 | 0.9247 / 0.9452 | 63.66 / 25.2 | 25.61 / 5.603 | -21.88 |
| 17 / 31 | 3.244 / 35.16 | 0.907 / 0.9313 | 72.25 / 48.32 | 57.66 / 16.59 | +16.88 |
| 21 / 38 | 0.8898 / 10.51 | 0.8933 / 0.9139 | 76.45 / 64.34 | 74.68 / 31.76 | +30.81 |
| 25 / 45 | 1.161 / 30.84 | 0.884 / 0.9362 | 68.9 / 42.99 | 57.86 / 23.14 | +8.666 |

Raw annotation is lower and held-out inactivity higher in 27B at every pair. Annotation excess above its own frequency floor is larger in 27B at the last three pairs. Thus raw coverage does not support a uniformly lower above-floor biological association. All-position 27B inactivity ranges from zero to 17.50%, while held-out inactivity reaches 67.21%. Sources: `experiments/results/scale_reporting.json`, `residual_summary.json`, and their complete catalogue/annotation inputs in the release.

Centred covariance spectra cover all 26/46 taps on the same 200 sentences. At 2B L13 / 27B L24, PR divided by width is 2.60% / .317%. The maximum/median coordinate RMS ratio spans about 20–45 versus 80–1,299 at the seven pairs. Within the seven 27B fitted taps, descriptive Spearman correlations of held-out inactivity with PR and RMS ratio are −.643 and−.286. Layers share cells/weights; no independent-layer hypothesis test is claimed. A shared mean and centred covariance concentration are separate measurements.

| 27B layer | Removed high-RMS coordinates | Held-out inactive before → after (%) | All-position inactive before → after (%) | Original / projected-input VE |
|---|---|---|---|---|
| 24 | 20 | 67.21 → 47.46 | 17.5 → 3.44 | 0.94523 / 0.91764 |
| 38 | 3 | 10.51 → 7.389 | 0 → 0.01085 | 0.91388 / 0.90931 |

Projection zeroes columns with RMS more than ten times the median, retaining original dictionary width/capacity and training recipe. L38 all-position inactivity increases from zero to one feature despite its held-out reduction. VE uses a different variance reference after projection and is not a like-for-like improvement score. Source: `experiments/results/dropmassive_comparison.json` and `audits/e4_saved_v1.json`.

## Perturbation responses and unsigned TF-target enrichment

The assay asks whether curated downstream targets are enriched among responses. It excludes the perturbed factor from downstream labels and gene-retrieval candidates. Response direction is not compared with signed regulatory predictions. Thus neither a hit nor a non-hit establishes biological regulatory direction, and an insensitive positive control cannot support absence of regulatory information.

K562 and RPE1 are Replogle CRISPRi screens. The THP1 comparison is a distinct Papalexi **inducible-Cas9 loss-of-function** cohort, sampled by guide assignment in one HTO partition, `rep1-tx`; successful editing is not verified per sampled cell. It uses 12 prespecified factors, four non-TF groups, 400 controls and four 50-cell pseudo-control groups. The fixed K562 dictionaries are transferred rather than fitted to THP1.

All screened factors stay in the factor-level BH family, including factors with zero eligible targets (p=1). Zero eligibility is explicitly uninterpretable. Evaluable counts below require at least one distinct downstream target; sensitivities requiring at least three and five are saved separately.

| Cohort | Expression hits / screened (evaluable) | Rank-input hits / screened (evaluable) | Feature hits at L8 / L16 / L22 (evaluable per layer) |
|---|---|---|---|
| K562 | 0/48 (34) | 0/48 (32) | 2 / 2 / 2 of 48 (28) |
| RPE1 | 2/121 (102) | 0/121 (99) | 1 / 0 / 1 of 121 (85) |
| THP1 | 3/12 (12) | 1/12 (12) | 1 / 1 / 0 of 12 (12) |

K562 feature hits are PHB2 and MAX. PHB2's L16 target label comes from one distinct target gene represented in 12 features; those are not 12 independent target genes. Requiring three target genes leaves MAX alone and only nine factors evaluable; requiring five leaves no discoveries and only five evaluable factors, always retaining the full 48-factor BH family. The K562 expression positive control is weak: its 0/48 result withdraws the earlier broad negative inference.

RPE1 expression hits are HMGA1 and NPM1; its feature hit is SF1 at L8 and L22. THP1 expression hits are IRF1, STAT1 and STAT2; ranks identify STAT1 and features identify STAT2 at L8/L16. These are context- and readout-dependent target-set recoveries. Gene and feature assays use different testing units and do not have directly comparable power.

### Retrieval and finite random-label AP

The common-gene comparison retains the same candidate universe across representations within each factor and cohort. Macro-averages include 34 K562, 102 RPE1 and 12 THP1 factors with eligible gene targets. Their common-gene retrieval eligibility differs from representation-specific specificity eligibility above. Intervals in the source files resample factors, conditional on the saved cell responses. They do not describe new-cell, guide, culture or training-seed variability.

| Cohort and representation | Factors | Mean AUROC | Mean AP | Mean finite random-label AP | Mean target prevalence |
|---|---|---|---|---|---|
| K562 expression | 34 | 0.519 | 0.00288 | 0.00142 | 0.000409 |
| K562 ranks | 34 | 0.594 | 0.00405 | 0.00139 | 0.000409 |
| K562 model L16 | 34 | 0.496 | 0.00079 | 0.00140 | 0.000409 |
| RPE1 expression | 102 | 0.590 | 0.00624 | 0.00184 | 0.000848 |
| RPE1 ranks | 102 | 0.628 | 0.00333 | 0.00182 | 0.000848 |
| RPE1 model L16 | 102 | 0.461 | 0.00113 | 0.00182 | 0.000848 |
| THP1 expression | 12 | 0.548 | 0.00911 | 0.00238 | 0.001873 |
| THP1 ranks | 12 | 0.679 | 0.01097 | 0.00235 | 0.001873 |
| THP1 model L16 | 12 | 0.513 | 0.00340 | 0.00234 | 0.001873 |

Average precision (AP) is the stepwise precision–recall integral; AUROC is also retained. For small, sparse target sets, prevalence is not generally the finite random-label expectation of AP. The exact expectation preserves observed score ties. If a sorted tie group has size a and cumulative end b, with m positives among n>1 candidates, its contribution is `a/(n*b) * (1 + (b-1)*(m-1)/(n-1))`; sum over groups. All-tied scores give prevalence. `prevalence_lift` and `lift_over_random_expected_AP` are distinct descriptive ratios, not significance tests. A mean of per-factor lifts is not a ratio of the macro-mean AP to mean prevalence. Separate model/input intervals do not constitute paired superiority or equivalence tests.

Sources: `experiments/results/phase8_k562/statistics/regulatory_statistics.json`, corresponding `phase8_rpe1` and `phase8_thp1_pooled` files, and their `audits/phase8_*_saved_v1.json` saved-data checks. Implementation: [rev_phase8_stats.py](../revision/rev_phase8_stats.py).

### Guide units and dictionary transfer

In the original K562 selection, 43 of 48 TF groups contain only one guide. THP1 retains all selected guides; 46 of 48 have at least three cells. At L16 the median feature-effect correlation across 66 evaluable guide pairs is 0.089 and median responding-set Jaccard overlap is zero. Pairs reuse controls and include nonresponding groups. This is a descriptive guide diagnostic, not 66 independent replications or a guide-level confidence interval. Source: `phase8_thp1_pooled/statistics/guide_concordance.json` under `experiments/results/`; implementation: [rev_guide_concordance.py](../revision/rev_guide_concordance.py).

| Control cohort for the target assay | Cells | Fixed K562 dictionary variance explained at L8 / L16 / L22 (%) |
|---|---|---|
| K562 | 400 | 92.0 / 85.4 / 81.3 |
| RPE1 | 1,000 | 87.7 / 79.2 / 70.5 |
| THP1 | 400 | 74.6 / 60.9 / 53.2 |

These are control-corpus reconstruction measurements, not training-held-out scores. Lower THP1 reconstruction limits interpretation of its feature responses. A separate transfer comparison uses 300 verified controls per line at L0/L13/L22. For that separate cohort, K562 reconstruction is 97.7/87.9/81.3% and RPE1 is 95.5/82.6/70.5%. Do not mix its rows or evaluation sample with the target-assay controls. Sources: each `phase8_*/meta.json`, plus `experiments/results/lines2.json` and `lines2_freq.npz`.

The audited PHB2 example illustrates a programme-level use. L16 F7801 decreases after knockdown (standardised effect −1.71, response q=3.74×10⁻⁷); its mitochondrial driving genes also decrease. Its strongest Reactome association concerns FASTK-family mitochondrial-RNA processing/stability (fold enrichment 188.7, ontology q=1.36×10⁻¹⁹). This links a model response with a measured programme; it does not identify direct PHB2 targets. Sources: `phase8_k562/statistics/response_biology.json` and `audits/phb2_worked_audit.json` under `experiments/results/`.

Deleting a name from the model sentence is a separate in-silico test, not the biological knockdown. The paired score subtracts the effect of deleting a nearby name and compares the same original downstream genes in both branches. The control does not match subword token length. Its per-factor downstream universes and coverage differ from the measured-perturbation assay, so AP values are not interchangeable. Source: `experiments/results/text_deletion_reporting.json`; its explicit units and limitations govern that comparison.

## Geometry, steering and input-matched readouts

| Analysis | Retained measurement | Source |
|---|---|---|
| Gene functional geometry, L9/L13/L17 | Co-expression/expression-bin cosine gaps 0.062/0.044/0.055; null z=6.57/3.86/4.52; 199/200/200 valid permutations | `geometry/gene_manifold_null.json`; `experiments/results/retained_numeric_ledger.json` |
| Standardised gene-vector dimensions | Participation ratio 94–132; TwoNN 5.0–6.2 | Same sources |
| CD34+ cell pseudotime, matched 20 dimensions | Linear R²: model 0.696, expression PCA 0.862; neighbour-minus-linear gain 0.023/0.102 | `geometry/results/cell_state_gpu.json` |
| CD34+ lineage readout | Balanced accuracy: model 0.605, expression 0.862; shuffled-label reference about 0.107 | Same source |
| L21 steering target progress | Curved/straight endpoints 0.820 rad; norm-matched random direction 0.007 rad | `route_genemanifold_c2s/results/manifold_steer_L21.json`; `experiments/results/retained_steering_audit.json` |
| L21 steering readout path length | Curved 1.968, straight 0.945, random 0.189 rad | Same sources |
| Gene-symbol probes, 87 genes | Human AUROC/AP 0.989/0.993; anagram 0.511/0.607; G2M prevalence 0.575; finite random-label AP 0.595 | `experiments/results/surface_form_retrieval.json` |

Gene participation ratio summarises global covariance concentration; TwoNN uses local neighbour-distance ratios. These are not competing estimates of one quantity. Genes as points also differ from cells as points in single-cell manifold studies. A co-expression-binned permutation result establishes alignment beyond that particular conditional null, not independence from every expression-based explanation. CD34+ random-fold readouts measure interpolation among sampled cells.

Steering changes a model readout rather than producing a biological transition. Curved and straight paths share endpoints, so longer traversal does not mean greater endpoint progress. The three L21, L18 and length-matched L21 settings reuse the same 24 cells: curved paths are longer in 21, 22 and 18 cells, with one-sided binomial sign-test probabilities from 1.8×10⁻⁵ to 0.011. They are not 72 independent cells. Gene-symbol probes use fixed-rank insertions in shared phase-neutralised real-cell backbones, not literally context-free symbols or a mouse-orthologue experiment.

### Within-line and frozen-transfer phase readouts

On the same 3,000 K562 cells, circular R² is 0.929 for full expression, 0.885 for selected magnitudes, 0.882 for binary membership, 0.868 for rank input and 0.874 for model L21. The paired model-minus-rank difference is 0.006 (95% interval −0.005 to 0.017); model-minus-membership is −0.009 (−0.019 to 0.001). The saved protocol uses whole-cohort PCA/scaling before the five-fold ridge readout, and phase labels are themselves derived from marker expression. It is a controlled readout comparison, not an estimate from fully fold-fitted preprocessing or a new clinical cohort. Source: `experiments/results/within_k562_cellcycle.json`.

Frozen K562→RPE1 transfer on 3,000 target cells gives model phase agreement **R_diff=0.789443** versus full-panel rank input **0.790535**; paired difference −0.001092 (95% interval −0.013245 to 0.011112). Shared-gene expression gives 0.878 and selected magnitudes 0.826; a constant predictor gives 0.043. Selection occurs on each full panel before projection to shared columns. Source fits are frozen. R_diff is the absolute mean complex angular residual, invariant to global rotation; it is **not circular R²**. The expression arms retain shared columns while the model can use nonshared gene names, so this is not a claim of perfect input-identity matching. Source: `experiments/results/selection_matched_transfer.json`.

Both paired intervals leave the model–rank difference unresolved. No equivalence margin or equivalence test was specified. “No resolved gain” therefore does not mean equal representations, a universal information ceiling or absence of useful gene-name associations. Cell-bootstrap intervals condition on saved fits and do not include new-source-sample or training-seed uncertainty.

## Raw-coordinate dimensionality controls

These controls treat **genes as rows**, with exact shared cells/genes within each cohort. They are separate from the standardized-coordinate geometry analysis (11,507 genes, PR 94–132 / TwoNN 5.0–6.2), which uses different sampling and coordinates. They are not estimates of a cell-state manifold or of a number of biological processes.

| Cohort / representation | Genes × coordinates | Covariance PR | TwoNN | Valid / excluded neighbor rows |
|---|---|---|---|---|
| CD34+: expression_log1p_CP10k | 6,633 × 4,000 | 11.092 | 236.12 | 6,633 / 0 |
| CD34+: expression_cells_shuffled_within_gene | 6,633 × 4,000 | 11.844 | 346.22 | 6,633 / 0 |
| CD34+: expression_shape_iid_normal | 6,633 × 4,000 | 2494.7 | 406.65 | 6,633 / 0 |
| CD34+: model_L09 | 6,633 × 2,304 | 111.45 | 4.6445 | 6,633 / 0 |
| CD34+: model_L09_shape_iid_normal | 6,633 × 2,304 | 1709.9 | 309.64 | 6,633 / 0 |
| CD34+: model_L13 | 6,633 × 2,304 | 71.888 | 4.7368 | 6,633 / 0 |
| CD34+: model_L13_shape_iid_normal | 6,633 × 2,304 | 1709.7 | 298.66 | 6,633 / 0 |
| CD34+: model_L17 | 6,633 × 2,304 | 67.262 | 4.0713 | 6,633 / 0 |
| CD34+: model_L17_shape_iid_normal | 6,633 × 2,304 | 1709.8 | 301.96 | 6,633 / 0 |
| K562 replication: expression_log1p_CP10k | 1,905 × 1,200 | 1.9284 | 94.916 | 1,905 / 0 |
| K562 replication: expression_cells_shuffled_within_gene | 1,905 × 1,200 | 1.947 | 124.18 | 1,905 / 0 |
| K562 replication: expression_shape_iid_normal | 1,905 × 1,200 | 735.78 | 210.94 | 1,905 / 0 |
| K562 replication: model_L09 | 1,905 × 2,304 | 94.44 | 4.9311 | 1,905 / 0 |
| K562 replication: model_L09_shape_iid_normal | 1,905 × 2,304 | 1042.3 | 289.4 | 1,905 / 0 |
| K562 replication: model_L13 | 1,905 × 2,304 | 62.724 | 4.7301 | 1,905 / 0 |
| K562 replication: model_L13_shape_iid_normal | 1,905 × 2,304 | 1042.8 | 276.58 | 1,905 / 0 |
| K562 replication: model_L17 | 1,905 × 2,304 | 61.881 | 3.649 | 1,905 / 0 |
| K562 replication: model_L17_shape_iid_normal | 1,905 × 2,304 | 1042.6 | 283.07 | 1,905 / 0 |

CD34+ uses 4,000 cells and 6,633 shared genes; the K562 replication uses 1,200 cells and 1,905 shared genes. Expression is log1p CP10k; shuffle controls permute cells within each gene; Gaussian controls match each matrix shape. Low model TwoNN coexists with **higher model PR than expression**. The two estimands respond to different geometry and marginals, so a universal “lower-dimensional model” conclusion is unsupported. All nine arms per cohort have valid distances for every gene. Full independent source/count/sentence/gene-mean/PR/direct-distance TwoNN/null-RNG audits passed 94 checks each. Sources: `experiments/results/cd34_dimensions.json`, `k562_replication_dimensions.json`, `audits/dimensions_cd34_v1.json`, `audits/dimensions_k562_replication_v1.json`, `audits/dimensions_standardized_reference_v1.json`.

## Artifact provenance and checks

The [controlled analysis release](https://github.com/Biodyn-AI/c2s-mechinterp/releases/tag/controlled-analysis-2026-11)
contains seven numerical archives, `NUMERICAL_ARCHIVES.json`, `NUMERICAL_ARTIFACTS.md` and
`SHA256SUMS`. The included-file inventory is authoritative; per-archive collection manifests
also record excluded local-only checkpoint classes. Duplicate numerical paths across the seven
archives were verified byte-identical. Primary count matrices, foundation-model/checkpoint binaries
and large reproducible activation intermediates are excluded.

The current paper and exact numerical figure sidecars are under `paper/`. Compact portable
reports under `results/revision/` record both original-source and destination hashes where
personal filesystem prefixes were normalized. Exact original artifact bytes remain in the
numerical archives. Historical results elsewhere in this repository are identified by
`results/README.md` and are not substituted for the controlled comparisons here.
