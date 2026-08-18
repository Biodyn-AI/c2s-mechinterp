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
| Replogle et al. genome-scale Perturb-seq, K562 | feature atlas, circuit tracing, regulatory-logic test | Cell 185 (2022) 2559–2575.e28, doi:10.1016/j.cell.2022.05.013 |
| Setty et al. CD34+ human bone marrow | gene and cell-state geometry | Nat. Biotechnol. 37 (2019) 451–460, doi:10.1038/s41587-019-0068-4 |

Expected format: an `AnnData` `.h5ad` whose `var_names` are **gene symbols** (not Ensembl IDs and not
integer indices — several public files ship with numeric `var_names` and will not work), with raw
counts available either in `.X` or in a named layer (`--counts-layer`).

## Ontologies
GO biological process, KEGG, Reactome, STRING and TRRUST v2, as JSON gene-set files in a single
directory passed via `--ontology-dir`:

```
ontologies/
├── go_bp_gene_sets.json      {term: [gene, ...]}
├── kegg_gene_sets.json
├── reactome_gene_sets.json
├── string_ppi_edges.json
└── trrust_human.tsv
```

Annotation entries come in two shapes: enrichment terms carry `p_adjusted`/`odds_ratio`/`n_overlap`,
while STRING and TRRUST *edge* entries carry `density`/`n_edges` instead. Downstream code must
tolerate both.

## Disk
Gene-token activations dominate. At 614,400 positions per layer: ~2.8 GB/layer for the 2B model in
fp32, ~5.7 GB/layer for the 27B. A full 26-layer 27B extraction is ~150 GB — use a network volume.
