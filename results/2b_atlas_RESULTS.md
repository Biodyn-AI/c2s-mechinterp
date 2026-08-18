# C2S-Scale-Gemma-2-2B — full SAE atlas (Phases 1–5), 26 layers

Completed 2026-07-21 on a RunPod A40 (~3 h, ~$1.5). Data: 1,200 K562 cells × 512 genes ≈ **614K
gene-token positions/layer**, `vandijklab/C2S-Scale-Gemma-2-2B` (d=2304, 26 layers). TopK SAE per
layer: d_SAE=9,216 (4×), k=32, 6 epochs, **decoder_norm_dim=0**. Gene-token unit (subword→gene
mean-pool). Annotation vs GO BP / KEGG / Reactome / STRING(≥700/900) / TRRUST.

## Per-layer summary

| L | var expl | dead % | annot % | modules | coverage % | SVD-aligned |
|--|--|--|--|--|--|--|
| 0 | 0.993 | 0.22 | 80.3 | 22 | 79.9 | 1 |
| 4 | 0.981 | 0.53 | 80.3 | 26 | 74.8 | 0 |
| 8 | 0.959 | 0.82 | 79.8 | 24 | 60.3 | 0 |
| 12 | 0.936 | 0.80 | 75.1 | 18 | 50.1 | 0 |
| 16 | 0.905 | 3.61 | 59.2 | 19 | 36.8 | 0 |
| 20 | 0.893 | 1.37 | 54.0 | 23 | 38.9 | 0 |
| 25 | 0.884 | 1.05 | 57.8 | 26 | 31.2 | 1 |
| **26-layer** | **0.87–0.99 (μ0.93)** | **0.15–3.61 (μ1.2)** | **50–81 (μ68)** | **16–26 (μ22)** | **31–80 (μ52)** | **~0 (99.99% novel)** |

## Findings

1. **High-quality SAEs, smooth depth decline.** Variance explained 0.99→0.87 with depth; dead
   features ~1% with a bump to 3.6% at L15–17 — **reproduces the reference's late-layer dead-feature
   signature** (Geneformer L15–17 also peaked). Decoder cosine ~0.02 (well-separated).

2. **Massive superposition confirmed.** 0–1 SVD-aligned features/layer → **99.99% of features novel**
   vs top-50 SVD, matching the reference's headline "SVD can't see SAE features."

3. **Biology concentrates in early layers (declining profile).** Annotation rate 80% (L0–5) → ~50%
   (late); module coverage 80% → 31%. This is a **monotonic decline**, unlike Geneformer's U-shape —
   early layers carry textbook gene programs (highly annotatable), late layers become
   prediction-oriented and abstract. (Caveat: absolute annotation rates run higher than Geneformer's
   45–59% — partly genuine LLM richness, partly the K562-focused background gene set; compare
   carefully, don't over-read the absolute level.)

4. **More, smaller co-activation modules.** ~22 modules/layer vs the reference's 6–12 — finer-grained
   gene-level modular structure, coverage declining with depth.

5. **HEADLINE — cross-layer feature persistence is high.** **5,782/9,216 (63%) of L0 features persist
   past 10 layers; 50% survive all the way to L25 at mean decoder cosine 0.88.** Geneformer: 2–3%,
   nothing past L11. The C2S-Scale (Gemma-2 decoder LLM) **residual stream preserves gene-level
   feature directions across its full depth** — features written early stay readable late — whereas
   encoder-style single-cell FMs rebuild features layer-by-layer. A genuine, architecture-level
   contrast (verified at scale; the tiny-validation's 100% was an artifact, this 50% is real).

## Feature biology (representative, from the tightening + catalogs)
Distinct monosemantic programs: ribosome (RPL/RPS separated), proteasome, spliceosome (Sm/snRNP),
chaperonin (CCT), hnRNP/RNA-processing, eIF3 translation-initiation, mitosis/G2-M (CCNB1, TPX2,
CENPF), glycolysis (ENO1, LDHA) — sensible for proliferative K562.

## Artifacts (`runs/2b_atlas/`)
Per layer: `feature_catalog_L{L}.json`, `feature_annotations_L{L}.json`,
`coactivation_modules_L{L}.json`, `sae/sae_L{L}.pt` + `sae/mu_L{L}.npy` + `sae/train_L{L}.json`;
plus `cross_layer_tracking.json`. 26 trained SAEs preserved.

## Phase 8 (regulatory-logic test) — preliminary, layer 16

CRISPRi perturbation-response on the Replogle **subset** (control 500 non-targeting cells; 14 TRRUST
TFs + 36 non-TF targets, ~12 cells each). Per-feature effect = (pert_mean − ctrl_mean)/ctrl_std over
gene-token SAE activations; "specific" = responding feature's top-20 genes overlap the TF's TRRUST targets.

- **Detection rate 84%** (42/50 targets respond) — model registers perturbations (ref Geneformer 92%).
- **TF-specificity 14.3% (2/14: GABPA, YBX1)** — ref Geneformer 6.2% (3/48).

**Read: directionally consistent with the "co-expression, not regulation" null** — the model broadly
detects perturbations but rarely produces TF-target-specific feature responses. **But NOT definitive:**
(1) n=14 TFs → 2/14 has a huge CI (~2–43%), statistically indistinguishable from the reference's 6.2%
or from chance; (2) ~12 cells/target; (3) the quick effect-size test blew up (max|eff| 80–142) where
control std → 0 for rarely-active features, inflating responding counts — a rigorous version needs a
per-feature Wilcoxon + reliable-feature filter + Fisher's-exact specificity (not loose top-20 overlap).

## Phase 8 v2 (ROBUST) — layer 16, definitive

Full **Replogle K562 essential** dataset (310K cells; `code/phase8_perturbation_v2.py` →
`phase8v2_L16.json`): **48 TRRUST TFs + 12 non-TF × 25 cells vs 400 non-targeting controls**, per-cell
**Mann-Whitney U + BH-FDR** for responding features, **Fisher's-exact + BH-across-TFs** for specificity.

- **Detection rate 96.7%** (58/60 targets respond) — near-perfect; the model reliably registers a knockdown.
- **TF-specificity 2.1% (1/48 TFs, BH q<0.05)** — at the false-discovery floor, **statistically
  indistinguishable from zero**. Reference Geneformer: 6.2% (3/48).

**Verdict: the "co-expression, not regulation" null ROBUSTLY replicates on C2S-Scale-2B.** TFs like MAX
(825 responding features) and ATF5 (354) shift hundreds of features — the model loudly *detects* the
perturbation — but those features' top genes do **not** overlap the TF's known targets. Broad
co-expression response, no directed TF→target wiring. **Scaling from encoder single-cell FMs to a 2B
generative LLM trained on 57M cells did NOT buy regulatory logic:** near-perfect detection, null
specificity. Same central negative the whole program keeps finding, now confirmed on a large LLM with a
properly-powered, multiple-testing-corrected test. (The earlier 14.3% preliminary was small-n noise.)

## Next
- **27B: does scale change it?** The real scaling question — atlas + this same robust Phase 8 on the 27B.
  If the 27B also lands at the null, that's a strong statement; if regulation emerges, that's a finding.
- Optional: Phase 8 v2 at more layers (does specificity peak anywhere in depth?).
