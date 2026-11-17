# c2s-mechinterp

**Mechanistic interpretability of Cell2Sentence (C2S-Scale) language models.**

Code and analysis pipelines for the paper *"Same ceiling, different machinery: mechanistic
interpretability of a text language model trained to read cells as sentences"*
(Ihor Kendiukhov, Institute of Medical Genetics and Applied Genomics, University of Tübingen).

C2S-Scale represents a cell as a **sentence of gene names** — the genes are ranked by expression and
written out as plain text, so a general-purpose language model (Gemma-2) can be trained on cells the
way it is trained on text. This repository opens that model up and asks what it actually computes.

> **Interactive feature atlas:** https://biodyn-ai.github.io/c2s-atlas/
> **Atlas source code:** https://github.com/Biodyn-AI/c2s-atlas

---

## What the study found

| Analysis | Result |
|---|---|
| Biological labels | 152,782 of 239,616 features (63.8%) carry named ontology/regulon terms, against a 34.4% frequency-matched floor; two late layers fall below their floors |
| Decoder persistence | 50.4% of layer-0 directions match layer 25 at absolute cosine ≥0.7; target reuse is allowed |
| Model-intervention circuits | 27,633 stored, capped edges; 83.2% of annotated edges share a term versus 61.3% for same-source random targets, an excess of 21.9 percentage points |
| Gene geometry | Functional alignment exceeds a co-expression-matched permutation null at three sampled layers (z = 3.86–6.57) |
| Steering | Internal interventions change the model's gene readout; a longer curved path does not improve endpoint progress over the linear path |
| Curated-target recovery | K562 expression recovers 0/48 factors, RPE1 2/121, and THP1 3/12; transferred feature dictionaries recover targets unevenly. This unsigned assay does not establish regulatory direction or absence of regulatory information |
| Dictionary design | Sparsity, token pooling, untrained directions and database coverage materially affect reconstruction and annotation; see the sensitivity results |
| Matched model sizes | 27B has higher held-out inactivity and lower raw annotation at all seven matched depths; its excess above matched annotation floors is higher at the three late pairs |
| Frozen phase readout | K562→RPE1 model minus rank-input phase agreement R_diff = −0.0011, paired 95% interval −0.0132 to 0.0111; no resolved gain |
| Public atlas | All 26 2B layers, 239,616 feature records and 577 co-activation modules, with complete per-feature details and separate ontology/network evidence |

Full provenance for every number: [`docs/05-results.md`](docs/05-results.md).

## The protocol in one picture

```
a cell  ─►  top-512 gene names, ranked      (expression values discarded)
            │
            ▼
        C2S-Scale (Gemma-2, 2B / 27B)  ── residual stream tapped
            │
   ┌────────┼─────────┬──────────────┬─────────────┐
   ▼        ▼         ▼              ▼             ▼
Stage 1   Stage 2   Target-set     Stage 3      Stage 4
feature   circuit   assay          geometry     public
atlas     tracing   (perturbation) + steering   atlas
```

The protocol uses **three controls**, wherever the design allows — see [`docs/02-controls.md`](docs/02-controls.md).
The controls available in each experiment define the scope of its inference.

## Install

```bash
git clone https://github.com/Biodyn-AI/c2s-mechinterp.git
cd c2s-mechinterp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

For the controlled paper analyses, use the recorded Python 3.12 / PyTorch 2.8 CUDA environment in
[`docs/04-reproduce.md`](docs/04-reproduce.md) and `revision/requirements.txt`. The root requirements
retain the earlier core-pipeline bounds.

Tokenizer and model-loading dependencies:

* `transformers==4.46.3` — 5.x breaks against torch 2.4 with a `DTensor` import error.
* The tested tokenizer uses `sentencepiece` and `protobuf`; `tiktoken` was absent in the recorded successful run.

A GPU is needed for extraction, circuit tracing and steering. The 2B model runs on a 24 GB card
(RTX 4090 / A5000); the 27B model needs 80 GB for forward passes, though its **sparse autoencoders
can be retrained on a 24 GB card** from stored activations.

## Quickstart — a minimal Stage-1 atlas

```bash
# 1. extract gene-token activations for a few layers
python -m c2s_mechinterp.extract_activations \
    --model vandijklab/C2S-Scale-Gemma-2-2B --h5ad data/k562.h5ad \
    --layers 0 8 16 --max-cells 1200 --max-genes 512 \
    --unit gene-token --store-dtype float32 --out runs/demo

# 2. train a TopK sparse autoencoder on one layer
python -m c2s_mechinterp.train_sae --run runs/demo --layer 16 --expansion 4 --k 32 --epochs 6

# 3. catalogue and annotate the features
python -m c2s_mechinterp.feature_catalog  --run runs/demo --layer 16
python -m c2s_mechinterp.annotate_features --run runs/demo --layer 16 \
    --ontology-dir data/ontologies_v2 --universe runs/demo/row_gene_names.json
```

Full commands for controlled analyses: [`docs/04-reproduce.md`](docs/04-reproduce.md).

The annotation command uses the **full observed gene universe**. It applies BH correction within
one feature across every eligible ontology/regulon term and the three network/membership tests.
Terms with fewer than two overlapping genes remain in the correction family with p = 1.
Uniform and gene-frequency-matched random sets provide chance floors by default. A snapshot
library can be selected with `--library-layout snapshot`; that changes file names, not the scorer.
Network evidence is saved separately from named ontology terms.

## Controlled analysis modules

[`revision/`](revision/) contains extraction replay, pooling and sparsity controls, complete circuit
effects, measured-expression/input positive controls, target retrieval metrics, residual-spectrum
summaries, biological categories, dimensionality nulls and atlas staging export. The modules use
public model/data inputs. They contain no cloud access credentials or resource-management commands.

```bash
# Synthetic statistical, extraction-guard, adapter and queue checks; no model download or GPU.
python -m unittest discover -s revision -p 'test_*.py'
```

See [`revision/README.md`](revision/README.md) for the workspace contract and command map.

## Repository map

| Path | Contents |
|---|---|
| `c2s_mechinterp/` | Core pipeline: cell-sentence construction, activation extraction, TopK SAEs, feature catalogues, ontology annotation, co-activation modules, cross-layer tracking, causal circuit tracing, perturbation response, scaling comparison |
| `geometry/` | Stage 3: gene-representation geometry with the co-expression-matched null, cell-state trajectory geometry, intrinsic-dimension probes |
| `manifold/` | Cell-cycle manifold, activation steering, lap-walk navigation, continuity and plateau tests, information-matched transfer experiments (supporting analyses) |
| `paper/` | Manuscript source and the scripts that regenerate every figure from the result artefacts |
| `results/` | Result write-ups and the small JSON artefacts the paper's numbers are read from |
| `scripts/` | Smoke tests and unit checks (e.g. sub-word→gene attribution) |
| `docs/` | Protocol, controls, data, reproduction and results documentation |
| `revision/` | Controlled analysis modules, annotation scorer, resumable queue and synthetic tests |

`manifold/_ported_from_encoder_routes/` holds reference implementations ported from earlier
encoder-model studies, kept verbatim for comparability; they are not part of the C2S pipeline.

## Data

The [versioned numerical release](https://github.com/Biodyn-AI/c2s-mechinterp/releases/tag/controlled-analysis-2026-11) supplies complete saved outputs and member checksums; see [artifact paths and extraction](docs/06-artifacts.md). The current paper and supplementary sources are in [paper/](paper/).

No primary count matrices are redistributed here. Saved numerical results and provenance are released separately from large reproducible activation intermediates. All datasets are public — see
[`docs/03-data.md`](docs/03-data.md) for accessions and preparation steps.

## Related repositories

| Repository | Purpose |
|---|---|
| [`Biodyn-AI/c2s-mechinterp`](https://github.com/Biodyn-AI/c2s-mechinterp) | This repository — analysis pipelines |
| [`Biodyn-AI/c2s-atlas`](https://github.com/Biodyn-AI/c2s-atlas) | The interactive SAE feature atlas (React application + released feature data) |

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite the C2S-Scale model and the datasets used.

## License

MIT — see [`LICENSE`](LICENSE).
