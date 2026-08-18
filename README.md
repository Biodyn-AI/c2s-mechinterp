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

| | Result |
|---|---|
| ✅ **Interpretable features** | 68% of sparse-autoencoder features per layer carry a significant ontology annotation; 119,808 features released in the public atlas |
| ✅ **Features persist across depth** | 50.4% of layer-0 features are still recoverable at layer 25 (cosine 0.88) — against 2–3% for an encoder-style model under the identical pipeline |
| ✅ **Coherent causal circuits** | 27,633 significant cross-layer edges; 85.2% join features that share an ontology term |
| ✅ **Geometry beyond co-expression** | functionally related genes are closer than a co-expression-matched permutation null predicts (z = 3.9–6.6) |
| ✅ **Causally used** | steering along internal directions moves the model's own predictions; norm-matched random directions do not |
| ❌ **No regulatory logic** | 96.7% of gene knockdowns are detected, but only 2.1% of transcription factors give a target-specific response |
| ❌ **Scale hurts legibility** | at matched dictionary capacity the 27B model has 31× more dead features and 19 points lower annotation than the 2B |
| ❌ **No gain over the input** | on cell-level tasks the model sits within 0.007 of a linear readout of its own input encoding |

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
Stage 1   Stage 2   Regulatory     Stage 3      Stage 4
feature   circuit   logic test     geometry     public
atlas     tracing   (perturbation) + steering   atlas
```

Every model claim is judged against **three controls** — see [`docs/02-controls.md`](docs/02-controls.md).
They are the methodological core of the paper, and each one changed a conclusion.

## Install

```bash
git clone https://github.com/Biodyn-AI/c2s-mechinterp.git
cd c2s-mechinterp
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Version pins that matter (both cost us a day):

* `transformers==4.46.3` — 5.x breaks against torch 2.4 with a `DTensor` import error.
* `tiktoken` + `protobuf` are required for the Gemma-2 tokenizer.

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
python -m c2s_mechinterp.annotate_features --run runs/demo --layer 16 --ontology-dir data/ontologies
```

Full commands for every result in the paper: [`docs/04-reproduce.md`](docs/04-reproduce.md).

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

`manifold/_ported_from_encoder_routes/` holds reference implementations ported from earlier
encoder-model studies, kept verbatim for comparability; they are not part of the C2S pipeline.

## Data

No primary data is redistributed here. All datasets are public — see
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
