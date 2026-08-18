# Matched-training control — is the 27B's lower SAE interpretability real scaling or undertraining? (2026-07-25)

**Setup.** The scaling comparison was confounded: both models' atlas SAEs use 4× expansion, but the 27B's
d_model is 2× the 2B's, so the 27B SAE has 2× the features (18,432 vs 9,216) on the same ~614k-row data →
~33 positions/feature vs the 2B's ~67. Fewer positions/feature undertrains features (more dead, noisier →
worse annotation). Control: retrain the 27B SAEs at **expansion 2 → 9,216 features = the 2B's absolute count**,
restoring ~67 positions/feature. Now only d_model differs (the actual scaling variable). RTX 4090 in EU-RO-1,
26 layers, k=32, 8 epochs, decoder_norm_dim=0, written to `27b_atlas/sae_matched_x2/`; catalog+annotate via a
symlink-farm run dir `27b_matched/` (no clobbering of the 4× atlas).

## Three-way result

| Metric | 2B (4×, 9216) | 27B MATCHED (2×, 9216) | 27B original (4×, 18432) |
|---|---|---|---|
| variance explained | 0.933 | 0.977 | 0.975 |
| **dead fraction** | **0.012** | **0.377** | 0.475 |
| decoder mean\|cos\| | 0.020 | 0.015 | 0.013 |
| annotated / total feats | 0.726 | 0.349 | 0.239 |
| **annotated / ALIVE feats** | **0.734** | **0.545** | 0.449 |

(2B annotation over the 16 layers annotated locally; 27B over all 26. dead/var from per-layer train JSON.)

## Verdict: the undertraining confound is REAL but only PARTIAL — a large genuine scaling gap survives

- **Undertraining explained ~10 points on each metric.** Halving the 27B's features (4×→2×, matched capacity)
  moved dead fraction 0.475→0.377 and annotated/alive 0.449→0.545. So part of the 27B's apparent
  low interpretability WAS undertraining — the control was worth running.
- **But at matched capacity the 27B is still dramatically less interpretable than the 2B:**
  dead fraction **0.377 vs 0.012 (~31×)** and annotated/alive **0.545 vs 0.734 (−19 pts)**. Undertraining
  cannot explain a gap this size — positions/feature are now equal (~67).
- **Mechanism — the 27B's residual stream is more anisotropic / lower effective-rank relative to d_model.**
  It reaches *higher* variance-explained (0.977 vs 0.933) with far fewer active features (only ~62% of the
  dictionary alive vs ~99%). Its computation concentrates in fewer, denser directions, and those active
  features align less with known GO/KEGG/Reactome biology.

**Bottom line: bigger is NOT more interpretable here.** At matched SAE capacity the 27B has ~31× more dead
features and ~19-pt-lower biological annotation of its active features than the 2B — a real representational
scaling property, not an undertraining artifact. Undertraining was a genuine but partial (~10-pt) confound
that the original scaling write-up correctly flagged and this control now quantifies.

## Caveats
- 2B annotation rate is over 16 layers (only those annotated locally) vs 27B's 26; the 2B rate is high and
  flat across those layers, so the gap is very unlikely to be a layer-coverage artifact, but note the mismatch.
- Annotation uses GO/KEGG/Reactome enrichment (annotation-database circularity applies UNIFORMLY to all three,
  so the comparison is fair even though the absolute rate is inflated by circularity).
- Matched on feature COUNT / positions-per-feature, NOT on d_model (2304 vs 4608) — d_model IS the scaling
  variable, so that is intended.
- Activations stored fp16 for both (same input to all SAEs → comparison fair).
- Superposition/reconstruction scaling was already found clean; this control was specifically about the
  interpretability drop.
