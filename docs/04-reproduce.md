# Reproducing the paper

Every command below writes into a run directory; later stages read the artefacts of earlier ones.
`$RUN` is a run directory, `$ONT` the ontology directory, `$H5AD` the dataset (see `03-data.md`).

## Stage 1 — the 2B feature atlas (Fig. 2)
```bash
# extract all 26 layers (fp32 is mandatory for 27B; safe everywhere)
python -m c2s_mechinterp.extract_activations \
  --model vandijklab/C2S-Scale-Gemma-2-2B --h5ad $H5AD \
  --layers $(seq 0 25) --max-cells 1200 --max-genes 512 \
  --unit gene-token --store-dtype float32 --out $RUN

for L in $(seq 0 25); do
  python -m c2s_mechinterp.train_sae        --run $RUN --layer $L --expansion 4 --k 32 --epochs 6
  python -m c2s_mechinterp.feature_catalog  --run $RUN --layer $L
  python -m c2s_mechinterp.annotate_features --run $RUN --layer $L --ontology-dir $ONT
  python -m c2s_mechinterp.coactivation     --run $RUN --layer $L
done
python -m c2s_mechinterp.cross_layer --run $RUN
```
Produces `sae/sae_L*.pt`, `feature_catalog_L*.json`, `feature_annotations_L*.json`,
`coactivation_modules_L*.json`, `cross_layer_tracking.json`.
Approx. 3 h on one A40 for the 2B model.

## Stage 2 — causal circuit tracing (Fig. 4a,b)
```bash
python -m c2s_mechinterp.circuit_tracing \
  --run $RUN --model vandijklab/C2S-Scale-Gemma-2-2B --h5ad $H5AD \
  --source-layers 0,7,16 --n-features 15 --n-cells 50 \
  --gene-col gene --control-label non-targeting --out $RUN
```
~80 min per source layer on an RTX A5000. Add `--analysis-only` to rebuild the graph and motif
analysis from existing per-layer files without re-running any forward passes.

> Layers are selected for having a **valid** catalogue *and* annotation file. Two layers in our run
> had corrupted catalogues and were skipped; the code now skips malformed JSON with a warning rather
> than aborting.

## Regulatory-logic test (Fig. 4c)
```bash
python -m c2s_mechinterp.perturbation_response \
  --run $RUN --layer 16 --h5ad $PERTURB_H5AD \
  --n-tfs 48 --n-nontf 12 --cells-per-target 25 --n-controls 400
```

## Capacity-matched scaling control (Fig. 3)
```bash
# retrain the 27B SAEs at 2x expansion -> same absolute dictionary as the 2B
for L in $(seq 0 25); do
  python -m c2s_mechinterp.train_sae --run $RUN_27B --layer $L \
    --expansion 2 --k 32 --epochs 8 --sae-subdir sae_matched_x2
done
python -m c2s_mechinterp.compare_scaling --run-2b $RUN_2B --run-27b $RUN_27B \
  --sae-subdir sae_matched_x2
```
`--sae-subdir` keeps the matched dictionaries beside the originals instead of overwriting them.
To catalogue/annotate the matched SAEs without duplicating 150 GB of activations, build a run
directory of symlinks whose `sae/` points at `sae_matched_x2/`.

## Stage 3 — geometry (Fig. 5a) and steering (Fig. 5b)
```bash
python geometry/gene_manifold.py --run $RUN_GENETOK --h5ad $CD34_H5AD \
  --onto $ONT --layers 9 13 17 --counts-layer spliced --out results/gene_manifold.json

python geometry/cell_state_geometry.py --run $RUN_CELLSUM --h5ad $CD34_H5AD --layer 18 \
  --pseudotime-key palantir_pseudotime --fate-key clusters --counts-layer spliced \
  --out results/cell_state.json
```
`gene_manifold.py` runs the co-expression-matched permutation null (200 permutations by default) and
reports z and a one-sided p per layer.

## Figures
```bash
python paper/make_figures.py            # Figures 1-5 -> paper/figures/*.pdf
python paper/make_graphical_abstract.py # graphical abstract at the journal's required size
```
Both read the result artefacts in `results/` and the atlas JSON; no number is hard-coded that is not
traceable to a file.

## Hardware actually used
| Task | GPU | Wall time |
|---|---|---|
| 2B atlas, 26 layers | A40 48 GB | ~3 h |
| 27B extraction, 26 layers | A100/H100 80 GB | ~6 h |
| 27B matched SAE retraining | RTX 4090 24 GB | ~1 h |
| Circuit tracing, 3 source layers | RTX A5000 24 GB | ~4 h |
| Geometry + null | RTX A5000 24 GB | ~30 min |
