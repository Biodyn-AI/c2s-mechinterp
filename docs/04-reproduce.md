# Reproducing controlled analyses

Run model extraction, SAE training, circuit tracing, UMAP and large dimensionality analyses on a
GPU/CPU machine with enough memory and disk. The synthetic tests below need no model, primary
dataset, GPU or cloud credentials. Commands assume the repository root as the working directory.

## Recorded compute environment

The controlled analyses ran on Linux with Python 3.12.3 and PyTorch 2.8.0+cu128. Use a separate
compute environment rather than the older core-pipeline dependency bounds in the root requirements:

```bash
python3.12 -m venv .venv-controlled
source .venv-controlled/bin/activate
pip install torch==2.8.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r revision/requirements.txt
```

The CUDA wheel command follows the [official PyTorch version archive](https://pytorch.org/get-started/previous-versions/).
`environment.json` and `environment_dependencies.json` in the numerical archive record observed
versions; the latter also records that `tiktoken` was absent in the successful run. These package
pins document the measured environment rather than testing every other platform or driver.

## Inputs and workspace

Use a separate workspace containing datasets, exact sentence rows, trained dictionaries and
result artefacts. The repository supplies code; primary data and model weights remain external.

```bash
WORK=/path/to/workspace
REPO=/path/to/c2s-mechinterp
RUN="$WORK/runs/2b_atlas"
ONT="$WORK/ontologies_v2"
K562="$WORK/data/ReplogleWeissman2022_K562_essential.h5ad"
RPE1="$WORK/data/ReplogleWeissman2022_rpe1.h5ad"
```

The resumable queue expects `code/` pointing to the repository's `c2s_mechinterp/` directory,
`runs/2b_atlas/` with all original SAE checkpoints and exact `row_gene_names.json` /
`row_cell_ids.npy`, `runs/27b_atlas/` with the existing comparison catalogues, `ontologies_v2/`,
`data/ontologies/` for snapshot sensitivity, `data/` for h5ad files, and `designs/` for cell/sample
JSONs. The queue replays these fixed inputs; it does not silently select replacement atlas cells.

```bash
ln -s "$REPO/c2s_mechinterp" "$WORK/code"
python revision/build_ontologies.py "$ONT" "$WORK/downloads"
python revision/run_revision.py --root "$WORK" --list
```

The queue writes `results/` and `stage_logs/`. It executes stages sequentially, stops on errors,
and writes a completion sentinel only after a successful exit and required output validation.
Inspect `--list` before starting. `--only 'pattern-*'` selects stages whose inputs already exist;
`--until STAGE_NAME` stops after a selected stage. Circuit replication and CD34 dimensionality
are explicit optional stages. The script does not create, stop or delete cloud resources.

```bash
python -u revision/run_revision.py --root "$WORK"
python -u revision/run_revision.py --root "$WORK" --include-optional --only 'circuit-rerun-*'
python -u revision/run_revision.py --root "$WORK" --include-optional --only 'geometry-*'
```

## Stage 1: dictionary, full-family annotation and floors

For a fresh demonstration, first prepare a **non-targeting-control-only** h5ad with gene-symbol
`var_names` and raw counts in `.X`. Selecting a different cell sample is a new run, not an exact
replay of the saved atlas.

```bash
python -m c2s_mechinterp.extract_activations \
  --model vandijklab/C2S-Scale-Gemma-2-2B --h5ad "$CONTROL_H5AD" \
  --layers 0 13 25 --max-cells 1200 --max-genes 512 \
  --unit gene-token --store-dtype float16 --out "$RUN"

for L in 0 13 25; do
  python -m c2s_mechinterp.train_sae --run "$RUN" --layer "$L" \
    --expansion 4 --k 32 --epochs 6
  python -m c2s_mechinterp.feature_catalog --run "$RUN" --layer "$L"
  python -m c2s_mechinterp.annotate_features --run "$RUN" --layer "$L" \
    --ontology-dir "$ONT" --universe "$RUN/row_gene_names.json" \
    --library-layout versioned --n-floor-seeds 3
  python -m c2s_mechinterp.coactivation --run "$RUN" --layer "$L"
done
```

The public annotation entry point writes `feature_annotations_Lxx.json`, retaining v3 records,
full test-family metadata, random floors and separate ontology/network compatibility mappings.
The direct scorer writes the canonical `ann_v2_Lxx.json` layout used by the revision-analysis
and atlas-export commands:

```bash
mkdir -p "$WORK/results/annotations/2b"
python revision/annotate_v2.py --catalog "$RUN/feature_catalog_L13.json" \
  --universe "$RUN/row_gene_names.json" --ont-dir "$ONT" \
  --out "$WORK/results/annotations/2b/ann_v2_L13.json" --n-floor-seeds 3
```

Run all required layers before invoking cross-layer analyses. Full-family correction is always
active: changing `--library-layout` on the public entry point, or `--legacy` on the direct scorer,
changes only the database file layout. Low-overlap terms still count as tests. Every annotation
rate must be interpreted with its matching random-gene floor.

For exact replay, use `revision/rev_extract.py --rows-dir "$RUN"` with the saved row files and
original dictionaries. Its mean/first/last pooling choices share the same gene-to-token mapping.
It rejects dropped genes, non-finite values and float16 overflow rather than silently clipping.
`revision/rev_sae_tools.py` supplies held-out/all-position firing evaluation, six-epoch sweeps and
norm-matched untrained dictionaries. Their complete commands are generated by the queue.

## Perturbations: exact cells, positive controls and retrieval

```bash
python revision/make_original48_design.py --h5ad "$K562" \
  --trrust "$ONT/trrust_human.tsv" --out "$WORK/designs/phase8_k562_original48.json"

python revision/rev_phase8_extract.py --code-dir c2s_mechinterp \
  --design "$WORK/designs/phase8_k562_original48.json" --run "$RUN" \
  --model vandijklab/C2S-Scale-Gemma-2-2B --trrust "$ONT/trrust_human.tsv" \
  --layers 8 16 22 --insilico --out "$WORK/results/phase8_k562"

python revision/rev_phase8_stats.py --features-dir "$WORK/results/phase8_k562" \
  --design "$WORK/designs/phase8_k562_original48.json" --catalog-dir "$RUN" \
  --h5ad "$K562" --trrust "$ONT/trrust_human.tsv" --layers 8 16 22 \
  --out "$WORK/results/phase8_k562/statistics"
```

Extraction saves per-cell features, reconstruction metrics and group firing frequencies. Statistics
save every response effect/p/q, measured-expression and input-rank controls, unsigned target-set
specificity, coverage sensitivities, average precision/AUPRC, AUROC and random-label baselines.
The primary TF family retains all screened factors; no-coverage cases are identified explicitly.
The perturbed gene itself is excluded from downstream target retrieval. Interpret specificity
only in light of the measured-expression positive control and target coverage.

Repeat extraction/statistics with a separately saved RPE1 design and `--h5ad "$RPE1"`, using the
same K562-trained dictionaries. Do not substitute new cells into an existing design silently.
`revision/rev_response_biology.py` maps responding features to saved annotation categories and
keeps generic programme membership separate from known-target association.

## Circuits: full effects, matching chance and stability

```bash
python revision/rev_circuit.py --code-dir c2s_mechinterp --run "$RUN" \
  --model vandijklab/C2S-Scale-Gemma-2-2B --cells "$WORK/designs/circuit_k562.json" \
  --features "$WORK/designs/circuit_original_features.json" --source-layers 7 \
  --n-cells 100 --out "$WORK/results/circuit_100"

python revision/rev_cpu_analysis.py circuits --circuits "$WORK/results/circuit_50" \
  --compare "$WORK/results/circuit_100" --annotations "$WORK/results/annotations/2b" \
  --out "$WORK/results/circuit_stability.json"
```

The effect/consistency rule uses absolute standardised effect greater than 0.5 and majority sign
fraction greater than 0.7. These are not adjusted p-value thresholds. Save full statistic vectors:
the older capped JSON graph omits measured effects beyond its top-effect limit. Shared functional
terms use complete significant ontology/regulon IDs, excluding pair/membership pseudo-terms.
The baseline retains each source and target layer; source features are the bootstrap unit.
The Normal-delta sensitivity calculation is conditional on source activity and reports its
assumptions. It is not a general biological power guarantee.

## Scaling, biology and dimensionality

The queue extracts 27B activations in float32 at layers {0,7,14,24,31,38,45}, matched to 2B
{0,4,8,13,17,21,25}. Every matched SAE has 9,216 features, k = 32 and six epochs. It records residual
spectra over all layers, magnitudes, reconstruction and held-out/all-position firing.
`revision/rev_residual_summary.py` joins the spectra and training/evaluation artefacts.
The optional `revision/rev_massive_dimensions.py` test prespecifies coordinates above ten times
median RMS, then trains selected-coordinate-drop controls at matched settings. Coordinate removal
happens in memory; stored activations remain unchanged. Empty selections are recorded as skips.
Reconstruction values are labelled by input space and are not subtracted across different spaces.

```bash
python revision/rev_cpu_analysis.py biology --run "$RUN" \
  --annotations "$WORK/results/annotations/2b" \
  --sae-from "$RUN/sae/sae_L00.pt" --sae-to "$RUN/sae/sae_L25.pt" \
  --out "$WORK/results/biology.json"

python revision/rev_cpu_analysis.py dimensions --h5ad "$CD34_H5AD" --counts-layer spliced \
  --model-vectors "$WORK/results/cd34_gene_means.npz" --layers 9 13 17 \
  --out "$WORK/results/cd34_dimensions.json"
```

The category rules and input files are saved with the results. Dimensions report covariance
participation ratio and Euclidean TwoNN, with within-gene cell shuffles and random matrices of the
same shape. Preserve cell indices when comparing model and expression vectors; `--cell-indices`
accepts an explicit saved JSON list. Existing geometry and steering analyses remain in `geometry/`
and `manifold/`; their numerical outputs must pass the same artefact audit before publication.

## Selection-matched frozen transfer

```bash
python revision/rev_selection_matched_transfer.py \
  --source-h5ad "$SOURCE_SUBSTRATE_H5AD" --target-h5ad "$TARGET_SUBSTRATE_H5AD" \
  --source-states "$SOURCE_CELL_STATES" --target-states "$TARGET_CELL_STATES" \
  --route-dir manifold --out "$WORK/results/selection_matched_transfer.json"
```

Before using cached states without source-file identity metadata, run the sampled validation
on the GPU machine for each line:

```bash
python revision/rev_validate_cell_states.py --code-dir c2s_mechinterp \
  --model vandijklab/C2S-Scale-Gemma-2-2B --h5ad "$SOURCE_SUBSTRATE_H5AD" \
  --states "$SOURCE_CELL_STATES" --layer 21 --n-cells 20 \
  --out "$WORK/results/source_state_validation.json"
```

The helper maps sampled cache rows through their stored h5ad indices, constructs the exact prompted
human cell sentence and compares freshly extracted BF16 states with cached vectors. It searches
all cached rows for the nearest match and saves vector errors, source/state/code hashes, model
commit and GPU/runtime measurements. Sampling does not prove identity for unsampled rows. A failed
sample check saves its JSON and exits with status 2; investigate or re-extract the cohort before
interpreting its probes.

Each state directory must contain `manifest.json`, `row_cell_ids.npy` and
`layer_21_activations.npy` from the same cell-summary model. Exact row alignment is required.
Positive genes are selected from the **full panel first**, with expression ties broken by the
original panel order, before projection onto shared gene columns. Restricting genes before
selection is retained as a diagnostic and never as the primary comparison. Source-only scaling
and ridge fits are frozen on the target cells. The saved output includes predictions, phase
labels, paired target-cell bootstrap draws, full-precision contrasts and input checksums.
An interval spanning zero is an unresolved difference, not an equivalence result.

Expression predictors still use only shared gene identities, whereas the model may use identities
absent from the source panel. That coverage limitation remains explicit even when selection and
rank gaps are matched. `R_diff` is angular-residual concentration, not the within-dataset
benchmark's circular R-squared; values from those two assays must not be substituted.

## Atlas staging and checks

```bash
python revision/export_revision_atlas.py --run "$RUN" \
  --annotations "$WORK/results/annotations/2b" --evaluation-dir "$WORK/results" \
  --positions-dir "$ATLAS_POSITIONS" \
  --compute-missing-positions --phase8-statistics "$WORK/results/phase8_k562/statistics" \
  --out "$WORK/results/atlas_staging"

python -m unittest discover -s revision -p 'test_*.py'
```

The exporter requires a new empty staging directory, complete numeric term details and fresh
`2b_eval_LXX.json` files. It uses the re-extracted variance explained and held-out inactive counts,
while all-position inactivity remains the catalogue definition. It separates
ontology/regulon terms from network evidence, labels q-values and fold enrichment, records full
ontology counts despite display-index caps, and writes source/output checksums, including the fresh
evaluation files. Observed rates and matching ontology/combined random-gene floors use all dictionary
features as denominator; explicit alive-feature rates remain available. Graph metadata distinguishes
stored edges from the further capped browser payload. It does not
modify the active browser data or publish anything. Final manuscript, figures, response letter,
atlas and repository numbers are synchronized only after the saved-output audit.

## Saved numerical artifacts and paper build

Download the [controlled analysis release](https://github.com/Biodyn-AI/c2s-mechinterp/releases/tag/controlled-analysis-2026-11)
and follow `NUMERICAL_ARTIFACTS.md` to verify and unpack the seven scientific archives beneath
`$WORK`. `NUMERICAL_ARCHIVES.json` lists exact member hashes; the archive path `results/`
corresponds to the paper's cited `experiments/results/`. Compact portable JSON views and their
source/destination hashes are tracked in `results/revision/`.

With the numerical archives installed, regenerate all measured figures using explicit workspace
paths. These commands write only the chosen paper-output directory:

```bash
python paper/make_design_schematic.py
python paper/make_revision_figures.py --figures all \
  --results-dir "$WORK/results" --runs-dir "$WORK/runs" \
  --route-dir "$WORK/route_genemanifold_c2s/results" --geometry-dir "$WORK/geometry" \
  --stage-logs-dir "$WORK/stage_logs" --out-dir "$REPO/paper/figures"
python paper/assemble_supplement.py --results-dir "$WORK/results" \
  --figures-dir "$REPO/paper/figures" --stage-logs-dir "$WORK/stage_logs" \
  --ontology-versions "$WORK/ontologies_v2/VERSIONS.json" \
  --compute-accounting "$REPO/paper/compute_accounting.json" \
  --out-dir "$REPO/paper/supplement_generated"
cd "$REPO/paper"
latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex
latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
```

The final generated supplement fragments are checked in, so recompiling the delivered paper
requires TeX but does not rerun statistical analyses. Regenerating fragments also requires the
saved compute-accounting record; it documents both reported billing and the separate conservative
lifetime estimate when the billing endpoint is delayed. The figure builder records exact plotted
values and input hashes in one provenance sidecar per numerical figure. The schematic has no
statistical sample. `paper/make_figures.py` and `paper/make_graphical_abstract.py` are historical
plotting sources; the active schematic and quantitative builders above produce the current figures.

Source text checks (the light parser is `pylatexenc==2.10`) are available separately:

```bash
pip install pylatexenc==2.10
python paper/tools/make_title_abstract_keywords.py --check
python paper/tools/validate_submission.py --supplement paper/supplement.tex
python -m unittest discover -s paper/tools -p 'test_*.py'
```

The front-matter sidecar pins the scientific source and generated-text hashes. The source checker
also scans included supplementary table fragments, but it cannot replace PDF rendering and
individual page inspection.
