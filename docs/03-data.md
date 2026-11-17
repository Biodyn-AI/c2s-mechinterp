# Data

No primary data is redistributed in this repository. Everything below is public.

## Models
| Model | Source |
|---|---|
| C2S-Scale-Gemma-2-2B | `vandijklab/C2S-Scale-Gemma-2-2B` on HuggingFace |
| C2S-Scale-Gemma-2-27B | `vandijklab/C2S-Scale-Gemma-2-27B` on HuggingFace |

Use the **HF safetensors** variants: they expose `output_hidden_states=True` through standard
`transformers` hooks, which the whole pipeline depends on.

## Single-cell datasets
| Dataset | Used for | Source |
|---|---|---|
| Replogle et al. genome-scale Perturb-seq, K562 | feature atlas, circuit tracing, unsigned target-set assay | Cell 185 (2022) 2559–2575.e28, doi:10.1016/j.cell.2022.05.013 |
| Setty et al. CD34+ human bone marrow | gene and cell-state geometry | Nat. Biotechnol. 37 (2019) 451–460, doi:10.1038/s41587-019-0068-4 |

Expected format: an `AnnData` `.h5ad` whose `var_names` are **gene symbols** (not Ensembl IDs and not
integer indices — several public files ship with numeric `var_names` and will not work), with raw
counts available either in `.X` or in a named layer (`--counts-layer`).

## Ontologies
The scorer accepts full versioned libraries or older snapshots. Use
`python revision/build_ontologies.py "$ONT" "$DOWNLOADS"` to build the versioned files and a
`VERSIONS.json` manifest recording source URLs, release/library identifiers and checksums.
The released build script selects GO Biological Process 2026, KEGG 2026, Reactome Pathways 2024,
STRING v12.5 score ≥ 700 and TRRUST v2. Libraries are stored in full, then intersected with the
observed gene universe at scoring time.

```
ontologies_v2/
├── go_bp.json                 {term: [gene, ...]}
├── kegg.json
├── reactome.json
├── string_v12.5_score700_pairs.tsv.gz
├── trrust_human.tsv
└── VERSIONS.json
```

Snapshot layout uses `go_bp_gene_sets.json`, `kegg_gene_sets.json`,
`reactome_gene_sets.json`, `string_ppi_edges.json[pairs_700]` and `trrust_human.tsv`.
The public annotation CLI detects a complete unambiguous layout; use
`--library-layout versioned` or `--library-layout snapshot` to select explicitly. Both layouts
use full-family v3 correction and chance floors.

`row_gene_names.json` contains all input gene occurrences from extraction; its distinct symbols
form the scoring universe. The CLI requires this file or an explicit full-universe JSON supplied
with `--universe`. It rejects feature genes missing from that universe.

V3 annotation records preserve complete `significant_term_ids` and numeric term details. Named
ontology/regulon records carry raw p, BH q, overlap and fold enrichment. STRING pair, TRRUST pair
and TF-membership evidence has separate source identities and observed/expected counts. The
compatibility `feature_annotations` mapping contains only true ontology/regulon terms;
`network_evidence` is a separate mapping. Fold enrichment is not labelled as an odds ratio.

RPE1 replication uses the Replogle RPE1 essential-gene screen, distributed as
`ReplogleWeissman2022_rpe1.h5ad` in scPerturb, alongside
`ReplogleWeissman2022_K562_essential.h5ad`. Cell indices and sentence construction must be kept
with each design to align expression, input and model controls.

## Disk
Gene-token activations dominate. At 614,400 positions per layer: ~2.8 GB/layer for the 2B model in
fp16 (~5.7 GB in fp32), ~11.3 GB/layer for the 27B in fp32. A full 26-layer 27B float32 extraction is ~294 GB — use a network volume.
