# Papalexi THP1 cohort preparation and completed C6 analysis

C6 was activated after the Replogle K562 measured-expression control recovered no specific TF target sets among the 48 screened factors. The fixed pooled THP1 `rep1-tx` cohort has completed model extraction, unsigned target-enrichment/retrieval analysis and descriptive guide diagnostics. It uses inducible-Cas9 loss-of-function guide assignments; assignment does not establish successful editing in every cell. Selection used availability, not observed response outcomes. The prepared arrayed cohort was not analysed and is not an independent replication result.

## Verified sources

Download the full RNA datasets into `$WORK/data/papalexi/` on the compute machine or an internal disk. Set `WORK=/path/to/workspace`, `REPO=/path/to/c2s-mechinterp` and `ONT="$WORK/ontologies_v2"` as in [the reproduction guide](../../docs/04-reproduce.md). Primary datasets are external inputs, not bundled with this code.

| Dataset | Source dimensions | Deposited MD5 | Local SHA256 |
|---|---|---|---|
| `PapalexiSatija2021_eccite_RNA.h5ad` | 20,729 cells × 18,649 genes | `d9e8bfe20b0e7b0919be042e8fcf6d03` | `03e9602d124c261281936717d5795445805e6e704a52b0310bcde94225aa0929` |
| `PapalexiSatija2021_eccite_arrayed_RNA.h5ad` | 8,984 cells × 16,826 genes | `843820d48b024348d6132cd53be0da91` | `7f8b06d92955940c453b7bec5dec879d64c1b2e2667aa5b6f7bb4bbc6665d976` |

The [scPerturb deposition](https://zenodo.org/records/13350497) directly supplies these filenames, sizes and MD5 values. The primary paper is [Papalexi et al., *Nature Genetics* 53, 322–331 (2021)](https://doi.org/10.1038/s41588-021-00778-2), with source accession [GSE153056](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE153056). scPerturb is described by [Peidli et al.](https://doi.org/10.1038/s41592-023-02144-y).

The observed `X` matrices are float32 and sampled stored values are finite, nonnegative exact integers. There is no `raw` matrix or count layer. Every count used in the staged cohorts, not only the 128-row inspection samples, is checked for those properties. The original files have no `uns` source metadata, so deposition provenance is supplied explicitly rather than inferred.

## Identity and grouping policy

Both files have one uppercase collision: `C2orf15` and `C2ORF15`. Both columns have exactly the same `var.ensembl_id`, `ENSG00000273045`. The staged data sum their raw counts and use the explicit merged spelling `C2ORF15`. The manifest keeps both original names, column positions and identifiers. Unique symbols retain their original spelling. In any future input, a case collision with distinct or missing identifiers excludes **every** conflicting column; the helper never chooses one arbitrarily.

The pooled file's `perturbation` field mostly contains guide identifiers, while the arrayed file contains gene labels. The helper validates both against `guide_id`, removes only a terminal `g[0-9]+` suffix, and maps `NTg#` guides to `control`. It preserves `STAT5A` and other gene-name digits, rejects multi-guide/unrecognised identifiers and rejects disagreements. TF status means membership in TRRUST's factor list; no new alias mapping is inferred.

The pooled file's `hto` field contains `rep1-tx` (9,351 cells), `rep3-tx` (5,992), `rep4-tx` (5,309), `rep2-tx` (73) and `rep2-ctrl` (4). The staged primary cohort uses only `rep1-tx`: targeted cells and controls therefore share the same recorded HTO category. The helper refuses an `all` pooled design. Separate `--batch rep3-tx` or `rep4-tx` designs can be prepared as prespecified replication, without pooling their cells or changing selection based on results. These are literal recorded categories; their biological treatment interpretation is not guessed from the names.

Arrayed `hto` values encode targeted wells/interventions and cannot be used as a common batch covariate. The arrayed cohort remains a separate assay with that batch limitation stated. It is not merged with the pooled experiment.

Cells are sampled with seed 20261117, balancing guides in a shuffled round-robin procedure. Selection uses availability only: all TRRUST factors with at least 30 cells in the cohort, up to 50 cells per factor; four randomly selected eligible non-TF targets; 400 controls and four disjoint pseudo-control groups of 50 cells. Guide identities and selected guide counts are retained. The control group stores 520 ranked names so text-deletion sensitivity can replace a removed name; standard extraction reads its first 512.

## Prepared cohorts

| Cohort directory beneath `$WORK/data/papalexi/` | Staged dimensions | Design |
|---|---|---|
| `c6_pooled_rep1tx/` | 1,384 × 18,648 | 12 TFs, 4 non-TFs, 400 controls, 4 × 50 pseudo-controls |
| `c6_arrayed/` | 1,050 × 16,825 | 5 TFs, 4 non-TFs, 400 controls, 4 × 50 pseudo-controls |

Each directory contains `counts_cleaned.h5ad`, `design.json`, and `provenance.json`. Design cell indices refer to the staged h5ad. `obs.c6_source_cell_index` and each group's `source_cells` preserve the original h5ad row indices. `obs.c6_source_obs_name` preserves the original cell barcode. Every filename, source/cleaned checksum, collision decision, batch category, guide-to-target mapping, selected row and script hash is recorded.

Only selected cells are materialized, using sparse 256-row reads. Staging is capped at 2,000 cells. Measured peak RAM for preparation was 818 MB for the pooled cohort and 387 MB for the arrayed cohort. Preparation used no GPU; subsequent model extraction ran separately.

Recreate the staged primary cohort from these local inputs:

```bash
python3 "$REPO/revision/contingency/prepare_papalexi.py" \
  --h5ad "$WORK/data/papalexi/PapalexiSatija2021_eccite_RNA.h5ad" \
  --trrust "$ONT/trrust_human.tsv" \
  --out "$WORK/data/papalexi/c6_pooled_rep1tx"
cd "$REPO"
python3 -m unittest discover -s revision -p 'test_*.py'
```

## Reproduce the completed pooled analysis

Use the fixed primary cohort and the K562-trained atlas dictionaries. The following commands assume the `WORK`, `REPO` and `ONT` variables above, with `RUN="$WORK/runs/2b_atlas"`. They reproduce the completed cohort rather than selecting a new one. `run_pooled.py` is the original sequential-stage wrapper for a compute workspace with the scripts staged under `$WORK/scripts/`; it does not create cloud resources.

```bash
python -u "$REPO/revision/rev_phase8_extract.py" \
  --code-dir "$REPO/c2s_mechinterp" --run "$RUN" \
  --model vandijklab/C2S-Scale-Gemma-2-2B \
  --trrust "$ONT/trrust_human.tsv" \
  --design "$WORK/data/papalexi/c6_pooled_rep1tx/design.json" \
  --layers 8 16 22 --out "$WORK/results/phase8_thp1_pooled"
python -u "$REPO/revision/rev_phase8_stats.py" \
  --features-dir "$WORK/results/phase8_thp1_pooled" \
  --design "$WORK/data/papalexi/c6_pooled_rep1tx/design.json" \
  --catalog-dir "$RUN" \
  --h5ad "$WORK/data/papalexi/c6_pooled_rep1tx/counts_cleaned.h5ad" \
  --trrust "$ONT/trrust_human.tsv" \
  --layers 8 16 22 --out "$WORK/results/phase8_thp1_pooled/statistics"
```

The measured-expression arm recovered IRF1, STAT1 and STAT2 (3/12 screened TFs), rank input recovered STAT1 (1/12), and model features recovered STAT2 at layers 8 and 16 (1/12 each), with no recovery at layer 22. All 12 factors had at least one eligible target in each arm. These are the saved full-screened-family BH-threshold results, not estimated rates of biological regulation. Control-cell dictionary reconstruction declined across layers 8, 16 and 22 (74.6%, 60.9%, 53.2% variance explained), so cross-line dictionary transfer remains a sensitivity limitation. The arrayed cohort and text deletion were not run for C6.

Saved evidence is under `$WORK/results/phase8_thp1_pooled/`, particularly `statistics/regulatory_statistics.json`, `statistics/response_biology.json` and `statistics/guide_concordance.json`. Independent saved-output audits include `phase8_thp1_pooled_saved_v1.json` and `c6_guide_concordance_v1.json` in `results/audits/`. Numerical files are external release/archive inputs, not generated by importing this package.

The endpoint is **unsigned known-target enrichment/retrieval**, not causal direction. A negative result with a weak expression control remains inconclusive. A stronger expression control establishes sensitivity only for the curated, context-dependent target sets and this selected cohort.

## Descriptive guide concordance after C6 extraction

`revision/rev_guide_concordance.py` uses the existing fixed design and saved feature matrices; it performs no inference or cell reselection. Each selected guide is compared with the **same fixed 400 controls**. The helper verifies every extracted cell/group against staged metadata, the single recorded batch and the saved guide counts. Four synthetic tests cover shuffled row alignment, fixed controls, low-count guides, self-target exclusion and a complete tiny command fixture.

The pooled cohort has four selected guide labels per factor. Forty-six of its 48 guide labels have at least three selected cells. MYCg4 has one cell and STAT1g4 has two; they remain in the report as unevaluable. The existing response rule requires at least three cells, so those guides are not silently dropped or replaced. No guide-level confidence interval is estimated from these few labels.

The completed guide analysis can be reproduced after the fixed cohort extraction:

```bash
python -u "$REPO/revision/rev_guide_concordance.py" \
  --h5ad "$WORK/data/papalexi/c6_pooled_rep1tx/counts_cleaned.h5ad" \
  --design "$WORK/data/papalexi/c6_pooled_rep1tx/design.json" \
  --features-dir "$WORK/results/phase8_thp1_pooled" \
  --catalog-dir "$RUN" --trrust "$ONT/trrust_human.tsv" \
  --layers 8 16 22 \
  --out "$WORK/results/phase8_thp1_pooled/statistics/guide_concordance.json"
```

The outputs describe feature-effect correlations, response-mask Jaccard overlap, response-sign agreement and per-guide curated-target gene AP/AUROC using the fixed catalogue and gene universe. Empty response unions have undefined Jaccard rather than perfect agreement. Pairwise summaries share the controls and are not independent replicates; many feature columns also overlap in biological membership. Small per-guide cell counts limit significant-response overlap. Feature-response signs are not regulatory activation/repression signs. These are diagnostics, not a new selection criterion or independent culture-level replication. Guide effect/mask arrays and input hashes are retained for audit.

At layer 16, the 66 eligible within-factor guide pairs had median full-feature effect correlation 0.089 and median significant-response Jaccard overlap 0. These low concordance summaries limit the interpretation of pooled cell-level recovery; they do not establish independent guide or culture replication.
