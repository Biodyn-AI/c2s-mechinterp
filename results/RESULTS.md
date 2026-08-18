# route_genemanifold — Thread A replicated on C2S-Scale-Gemma-2-2B

**Date:** 2026-07-24 · **Model:** `vandijklab/C2S-Scale-Gemma-2-2B` (Gemma-2, 26 layers, d=2304, text LLM) ·
**Source route:** `biotensor/codebase/route_genemanifold` (Kendiukhov, 2026-07)

## Question

Does C2S-Scale's **gene representation** carry geometric structure aligned to an external coordinate
(pathway step-order, a cell-cycle circle, antipodal lineage axes, chromosome), **beyond** a co-expression
data baseline and a protein-sequence control — the `route_genemanifold` Thread-A test, ported to a text LLM?

## The key adaptation

C2S-Scale has **no learned gene-embedding table** (a cell is a rank-ordered "cell sentence" of gene names,
standard Gemma tokenizer). So the route's context-free weight bases (`scgpt_we`/`maxtoki_we`/`geneformer_we`)
have no analog. The C2S model basis is the **context-aware activation basis** `c2s_ctx_L{k}` — the per-gene
mean residual stream at that gene's token position, layer k, over a diverse Tabula Sapiens panel (the direct
analog of `gm_lib.build_ctx`). Everything else is preserved verbatim:

- **References:** `coexpr` (per-gene log1p-CP10k profile across the SAME 7,500 TS cells; DATA baseline) and
  `esm2` (UCE/STATE frozen ESM2 protein embeddings; SEQUENCE control).
- **Statistics** (`order_score`/`circle_score`/`grid_score`/`antipodal_score`/`margin_boot`) copied verbatim.
- **Null:** permute the annotation over the same genes (NEVER feature-shuffle).
- **Verdict:** a model basis counts only if it beats BOTH references AND its paired margin bootstrap over the
  shared genes clears 0.

**Panel & basis:** 7,500 TS cells (kidney + lung + immune, 2,500 each), 7-layer depth scan {0,4,8,13,17,21,25};
11,514 genes retained (≥12 tokens), occ-cap 200. Extraction on an A6000 (~35 min).

## Result 1 — Geometric screen (`run_geometric.py` → `results/geometric.json`)

**No hypothesis shows model-specific geometric structure surviving a fair margin bootstrap.**

| hypothesis | best model | model | coexpr | esm2 | verdict |
|---|---|---|---|---|---|
| cellcycle_circle *(pos. control)* | L13 | 0.838 | 0.495 | 0.112 | model>both (pt), **margin CI [−0.36,+0.83] touches 0** |
| complement_cascade | L04 | 0.759 | 0.593 | 0.666 | pt-win is a coverage artifact; on shared genes **esm2 beats model** (−0.16) |
| heme_biosynth / tca | L17 / L13 | 0.80 / 0.72 | 0.22 / 0.28 | 0.46 / 0.57 | model>both (pt) but margin CI touches 0 (n=9/15) |
| glycolysis, secretory_pathway | — | — | — | — | **sequence (esm2)** — isozyme paralogs |
| coagulation_cascade | — | — | 0.811 | — | **co-expression** |
| antipodal ×3 (GATA1/SPI1, TBX21/GATA3, PAX5/PRDM1) | — | — | — | — | **co-expression** (no model-specific antipodality) |
| ribosomal_blob *(neg. control)* | L08 | 0.500 | 0.431 | 0.407 | **null / n.s.** ✓ clean |
| urea_cycle, hox_grid, rorc_foxp3, sox2_cdx2, globin_switch, histone_blob | — | — | — | — | INCOMPLETE (genes too lowly expressed in TS panel to enter ctx basis) |

- **Positive control works, method is sound:** the cell-cycle circle is recovered strongly at L13 (0.838) —
  but is co-expression-confounded (margin CI crosses 0), exactly as `gene_sets.py` warns ("peak phase IS the
  co-expression structure").
- **Negative control is clean:** ribosomal_blob does not spuriously "recover" its naming index.
- **Depth:** structure peaks mid-stack (L13), consistent with the route's "context integrated early, decays".

## Result 2 — Chromosome, genome-wide (`run_genome_wide.py` → `results/genome_wide.json`)

**The C2S activation basis decodes chromosome only weakly, and NOT model-specifically.**

| basis | T1 bal-acc (all autosomal) | ×chance | vs refs |
|---|---|---|---|
| c2s_ctx_L13 (best) | 0.105 | 2.3× | — |
| **esm2 (sequence)** | **0.190** | **4.2×** | **beats every ctx layer** |
| coexpr (data) | 0.080 | 1.8× | — |

- Every C2S ctx layer sits **between** coexpr and esm2 → verdict **"no"** (does not beat both). The chromosome
  signal in C2S activations is protein-sequence/paralogy (ESM2 captures it better), not a model-specific
  genomic representation.
- Robust to removing HOX+PCDH tandem arrays (T2 unchanged). **T3 genomic-distance decay ≈ 0** (no local
  genomic structure).
- **Transfer test:** the route's strong chromosome-in-the-*table* finding (0.51 ≈ 11× chance on local SCFMs)
  does **not** carry to C2S's activation basis — as predicted for a model with no gene-embedding table.

## Result 3 — Gate-3 causal steering (`run_steering.py` → `results/steering_L13.json`)

Nothing passed Gate 2, so Gate 3 has no *promoted* target. As a harness check + causal test of the
best-covered decodable directions, we steer at L13 with **norm-relative doses** (fraction × residual norm):
push a random half of a cell's axis-gene positions toward +u / −u, read the pole-A−pole-B next-token logit
swing at the disjoint other half (readout = the model's own logits; never a fitted probe).

| axis | cells | dose 0.25 / 0.5 / 1.0 (signed swing) | random @1.0 | causal? |
|---|---|---|---|---|
| cellcycle G1/S ↔ G2/M | 25 | +0.13 / +0.32 / **+0.54** | +0.00 | **yes** — monotone, random flat, 80% of cells |
| antipodal GATA1/SPI1 (erythroid↔myeloid) | 30 | +0.06 / +0.25 / **+0.73** | +0.07 | **yes** — monotone, random flat, 80% of cells |
| antipodal PAX5/PRDM1 | 30 | +0.00 / +0.00 / −0.06 | −0.04 | no — indistinguishable from random |
| TBX21/GATA3, RORC/FOXP3, SOX2/CDX2 | — | — | — | SKIP — lineage markers too rare in TS panel |

**Reading it honestly:** C2S **causally uses** the cell-cycle and erythroid/myeloid directions (clean monotone
dose-response, norm-matched random ≈ 0). But these directions **failed Gate 2** — they are co-expression-level,
not model-specific. So this is causal use of **co-expression structure**, which is exactly the route's thesis
(single-cell FMs re-express co-expression, and co-expression directions are causally used — not novel model
structure). The steering validates the harness and shows C2S behaves like the other SCFMs.

## Overall verdict

**A rigorous negative.** On C2S-Scale-2B, the context-aware gene representation carries **no geometric or
genomic structure that is model-specific beyond co-expression + protein sequence.** This **replicates
`route_genemanifold`'s central deflationary finding** on a completely different architecture (Gemma-2 text LLM
vs. the local encoder/decoder SCFMs), and specifically shows the chromosome result does not transfer to a model
without a learned gene table.

## Caveats

- **Coverage:** HOX, globins, urea-cycle, developmental TFs are too lowly expressed in the adult
  kidney/lung/immune panel to reach ≥12 tokens, so several sequence-confounded hypotheses are INCOMPLETE, not
  tested. A developmental panel (fetal gut) would be needed for HOX/globin — the route added `coexpr_devel`
  for exactly this.
- **Power:** the pathway hypotheses are small-n (6–28 genes); the margin bootstrap correctly reports wide CIs.
- **Basis:** the model basis is context-aware activations (the only C2S option), not a static table — so this
  tests C2S's *representation*, and a genuine null here is the honest analog of the route's table test.
