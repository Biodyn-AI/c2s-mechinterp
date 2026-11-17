# Controlled analysis numerical artifacts

Release: `controlled-analysis-2026-11` in [Biodyn-AI/c2s-mechinterp](https://github.com/Biodyn-AI/c2s-mechinterp/releases/tag/controlled-analysis-2026-11).

These archives contain the saved numerical inputs and outputs used for the paper, with exact sample designs, reference-library versions, source hashes and stage measurements. `NUMERICAL_ARCHIVES.json` lists every included member, byte length and SHA256; `SHA256SUMS` checks the downloaded assets. The final code and paper are in the tagged repository.

| Archive | Contents | Bytes |
|---|---|---:|
| `stage1_results.tar.gz` | Complete 26-layer 2B and original 27B annotations, 2B catalogues, evaluations, modules and extraction/spectrum provenance. | 720,820,247 |
| `stage2_scientific_01.tar.gz` | Reference libraries and version metadata, exact designs, K562/RPE1 saved per-cell/statistical results and audit inputs. | 1,039,492,654 |
| `stage2_scientific_02.tar.gz` | Sparsity/width, pooling and random-direction controls; activated THP1 results; biological categories and cross-line firing. | 154,254,616 |
| `stage3_annotations_scientific_01.tar.gz` | Complete seven matched-capacity 27B annotation files and annotation-stage records. | 57,880,258 |
| `stage3_scientific_01.tar.gz` | Matched 27B catalogues/evaluations, 72-tap residual summary (raw spectra are in Stage 1), coordinate-drop controls, full 50/100-cell circuit statistics, CD34+/K562 dimension controls and final stage records. | 267,627,109 |
| `retained_results.tar.gz` | Retained geometry, steering, gene-name probe, within-K562 and selection-matched frozen-transfer statistics plus sampled-cache validation. | 2,097,387 |
| `stage3_local_science_audits.tar.gz` | Additional independent saved-array audit results and dense p/q verification arrays. Despite its collection filename, this archive contains only scientific audit material. | 1,271,290 |

## Paths and extraction

Run on the compute machine. The common archive root is `$WORK`. Members under `results/` map to the paper’s cited `experiments/results/`; members under `runs/`, `geometry/`, `route_genemanifold_c2s/`, `designs/`, `ontologies_v2/` and `stage_logs/` retain those paths beneath `$WORK`. Numerical duplicate members across archives were verified byte-identical. Preserve each archive’s own collection manifest separately (Stage 1 uses `results/collection_stage1_manifest.json`; the other six use `COLLECTION_MANIFEST.json`); it records the collection source inventory and can also list a local-only class that was deliberately excluded from this public release. The authoritative included-file inventory is `NUMERICAL_ARCHIVES.json`.

```bash
WORK=/path/to/workspace
mkdir -p "$WORK/downloads" "$WORK/provenance"
gh release download controlled-analysis-2026-11 --repo Biodyn-AI/c2s-mechinterp \
  --dir "$WORK/downloads" --pattern "*.tar.gz" --pattern NUMERICAL_ARCHIVES.json \
  --pattern SHA256SUMS --pattern NUMERICAL_ARTIFACTS.md
cd "$WORK/downloads"
sha256sum -c SHA256SUMS
for archive in stage1_results.tar.gz stage2_scientific_01.tar.gz \
  stage2_scientific_02.tar.gz stage3_annotations_scientific_01.tar.gz \
  stage3_scientific_01.tar.gz retained_results.tar.gz stage3_local_science_audits.tar.gz; do
  manifest_member=COLLECTION_MANIFEST.json
  if [ "$archive" = stage1_results.tar.gz ]; then
    manifest_member=results/collection_stage1_manifest.json
  fi
  tar -xOf "$archive" "$manifest_member" > "$WORK/provenance/$archive.manifest.json"
  tar -xzf "$archive" -C "$WORK" --exclude=COLLECTION_MANIFEST.json
done
```

Primary h5ad/count matrices, cell-level foundation-model caches, large activation memmaps, foundation-model weights and SAE checkpoint binaries are excluded. Acquire primary data and model weights from the sources documented in the repository; checkpoints and activation caches can be regenerated with the recorded designs and analysis commands. All 22 new dictionary checkpoints were retained privately for continuity, outside this public artifact set. The old selection-confounded frozen-transfer result is excluded; `results/selection_matched_transfer.json` is the controlled comparison.

These archives report computed associations and controls; annotation labels and feature interventions do not themselves demonstrate a cellular regulatory mechanism. No editorial correspondence or reviewer text is included.
