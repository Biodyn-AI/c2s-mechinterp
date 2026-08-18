# route_genemanifold Thread B (context representation) replicated on C2S-Scale-Gemma-2-2B

**Date:** 2026-07-24 · **Model:** `vandijklab/C2S-Scale-Gemma-2-2B` · **Source:** `NOTE_gene_context_representation.md`
(the route's one result that survived every deflationary null on MaxToki-217M/1B).

## Question
A gene's representation inside the model is contextual (depends on the other genes in the cell). Is that
contextual movement **gene-specific**, **reproducible**, **not token position**, **organized along functional
axes**, **beyond co-expression**, and **causally used** — on C2S-Scale?

## Method (faithful port + the controls the C2S pilot lacked)
Cube `M[partition, cell-type, gene, dim]` = per-(gene, cell-type) mean residual over a **12-cell-type Tabula
Sapiens panel** (B, CD4T, CD8T, neutrophil, macrophage, cMono, iMono, kidneyEpi, capEndo, basal, AT1, AT2; 1,000
cells each). 8 taps {0,1,2,4,6,9,13,17}, **4,876 panel genes**, **10,912 balanced (gene,ctx) at cap in both
halves**. Controls baked in: **occurrence cap 50** (heteroscedasticity), **cell split-half** (disjoint-cell
reproducibility), **token-rank stored** (the position control the pilot couldn't run), **stopword/df filter**
(MT-*, MALAT1, B2M, ribosomal, … dropped), **pairwise panel** (FLOOR 25 in ≥2 contexts). Per-dim Timkey z-score;
`top_dim_share` ≈ 1–3% (no rogue Gemma-2 activation dim).

## Gate chain
| # | test | metric | result | vs MaxToki |
|---|---|---|---|---|
| 1 | **gene-specific & reproducible** (`ctx_polysemy`) | EXCESS, peak L09 | **+0.916**, CI [+0.914,+0.919] · mainrep 0.997 | ✓ matches (MaxToki +0.76–0.92) |
| — | position/rank control (`ctx_position`) | EXCESS residualized on [1,rank₁,rank₂,\|Δrank\|] | **SURVIVES +0.911** | ✓ matches (MaxToki 96% survives) |
| 2 | **organized along functional axes** (`ctx_functional_axes`) | z vs random-partition null, rank-controlled | **POSITIVE** — nuclear↔surface **z=+21.3**, mito↔cytoskeleton +4.8 | ✓ matches |
| 3 | **BEYOND co-expression** (`ctx_coexpr_null`) — *the load-bearing deflation* | σ above the power~coherence curve | **FAILS — best +2.4σ (<3σ), ≈0 at peak layers** | ✗ **DIVERGES** (MaxToki +8σ) |
| 4 | **causally used** (`ctx_causal`) | signed steering swing, dose-response | **PASSES** — nuclear↔surface @L4: signed +0.04/+0.13/**+0.20** (monotone), random ≈0, 100% of cells, sign_p<1e-12 | ✓ matches |

## Verdict — a **partial** replication with one decisive divergence
On C2S-Scale-2B the gene-specific context response is **real, reproducible, rank-independent,
functionally-organized, AND causally used** (gates 1, position, 2, 4 all pass, closely tracking MaxToki). **But
the route's distinctive, load-bearing claim — that the functional modulation is BEYOND co-expression — does NOT
hold for C2S-2B**: against a co-expression-coherence-matched null the functional axes sit essentially on the
curve (+2.4σ best, ~0 at peak layers), versus +8σ in MaxToki-217M.

So the direction C2S causally uses (gate 4) is a **co-expression-level** functional direction (it failed gate 3).
This is the same shape as the Thread-A steering result: co-expression directions are decodable and causally
used, but that is not novel model structure. **4 of 5 gates pass, but the one that would make it "beyond
co-expression" is the one that fails.**

**Interpretation.** C2S re-references genes per context in a functionally-organized way, but that organization is
**co-expression** — the model re-expresses which gene names co-occur, it does not add structure beyond it. This
is consistent with (a) the route's own hedge that MaxToki-**1B**'s extra contextualization was "largely
co-expression-like," and (b) C2S being a **text LLM over gene-name co-occurrence**, whose contextualization is
naturally co-expression-shaped. It also rhymes with the Thread-A negative: on C2S, gene structure beyond
co-expression + sequence is hard to find.

## Caveats
- **No raw-embedding zero-check.** Tap "L00" is post-block-0 (`hidden_states[1]`), not the raw embedding, so the
  route's EXCESS≈0 sanity (context-free ⇒ 0) wasn't reproduced; the same-vs-diff control + cross-partition
  design are the artifact guards, and position/co-expression nulls further constrain interpretation.
- **Co-expression matrix** built from the same 12-cell-type panel ⇒ very coherent modules ⇒ a demanding curve
  (which is the correct, conservative choice — it is exactly what "beyond co-expression" must beat).
- `transcription_vs_transport` has few transport genes after the panel filter; `nuclear_vs_surface` and
  `mito_vs_cytoskeleton` are the well-powered axes.
