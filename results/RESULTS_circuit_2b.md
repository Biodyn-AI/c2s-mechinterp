# Stage-2 causal circuit tracing — C2S-Scale-2B (2026-07-25)

Reduced-scope fast pass (user-approved): n_cells=50, n_features=15 per source layer. Source layers
**0, 7, 16** (L07 substitutes the CORRUPT `feature_catalog_L08.json`). K562 Replogle cells, non-targeting
control. Ablate each selected SAE feature in the source layer's residual stream, measure downstream SAE
feature deltas at disjoint positions. Outputs in `runs/2b_atlas/`: `circuit_L{00,07,16}_features.json`,
`circuit_graph.json` (8.5 MB), `circuit_analysis.json`.

## Graph
- **27,633 significant cross-layer causal edges**; 45 source features (15×3) → 7,959 distinct downstream
  target features.
- **Attenuation with layer distance** (clean, expected): edges/target-layer decay monotonically —
  L00→L01 746 … L00→L25 196; L07→L08 729 … L07→L25 307; L16→L17 615 … L16→L25 391. Avg sig edges per
  downstream layer: L0 45.7, L7 39.7, L16 38.5. Influence spreads forward and fades — no long-range spikes.

## Biological motif — the headline
- Of **12,816 annotated source→target edge pairs, 10,919 (85.2%) share ≥1 ontology term.**
- Top circuits are same-function propagation across layers, e.g. **L7 DNA-Damage-Response →
  L14/L22/L12/L8 DNA-Damage-Response** (Δ −1.2 to −1.45; shared Recombinational Repair, APC/C:Cdh1
  mitotic-exit, etc.).

**Interpretation (honest).** The causal circuit overwhelmingly propagates *functional identity* — a feature
tagged "DNA damage response" at layer 7 causally drives DNA-damage-response features downstream. That is a
coherent, interpretable circuit, but 85.2% shared-ontology is **not by itself evidence of regulatory logic**:
features sharing an ontology term are annotated from the same (co-expressed) gene sets, so the overlap can be
annotation-circularity / co-expression rather than the model implementing gene-regulatory computation. This
matches this project's recurring negative verdict on causal *regulatory* logic (attention-GRN ↔ SAE mega-
pipeline). To upgrade "interpretable functional propagation" to "regulatory logic" needs a co-expression-
matched null on the shared-ontology fraction (not run here).

## Caveats
- **Reduced scope**: 50 cells, 15 features/layer, 3 source layers — lower statistical power; a magnitude, not
  a gated claim.
- **L07 substituted for corrupt L08** (`feature_catalog_L08.json` malformed); a separate task regenerates the
  corrupt L08 catalog + L09 annotations.
- **L09 annotations skipped** (also corrupt) — L09 target features contribute edges but no ontology overlap,
  slightly deflating the 85.2% (conservative).
- `pmi_comparison` is **skipped by design** (comparing cross-layer causal edges to within-layer coactivation
  PMI is meaningless — recorded as skipped in the JSON).
- Original full-scope run (200 cells/30 features) was killed for being ~15–20h (unbatched per-cell×per-feature
  forwards); its partial L00 is preserved as `circuit_tracing_partial_L00.SALVAGED.json`. A full-power rerun
  would need the ablation loop batched across cells.
