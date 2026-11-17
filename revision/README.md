# Controlled analysis modules

This directory contains executable analysis code and synthetic tests. Model checkpoints, cell
sentences, h5ad files, ontology downloads, cloud access files, run logs and saved result
arrays are external inputs. No cloud account/resource changes are implemented here.

| Module | Purpose |
|---|---|
| `annotate_v2.py` | Complete per-feature test family, measured random-gene floors and full significant term details |
| `annotation_entrypoint.py` | Public `--run/--layer/--ontology-dir` adapter and ontology-only compatibility mapping |
| `build_ontologies.py` | Public versioned reference downloads, full libraries and provenance manifest |
| `rev_extract.py` | Replay exact stored sentences, three pooling choices and residual-spectrum measurements |
| `rev_sae_tools.py` | SAE sweeps, held-out/all-position evaluation, untrained controls and decoder matching |
| `make_original48_design.py` | Reconstruct the fixed K562 TF/control selection and cell sentences |
| `rev_phase8_extract.py` | Per-cell SAE features, reconstruction, firing frequencies and in-silico deletion |
| `rev_phase8_stats.py` | Expression/input positive controls, response statistics, coverage, specificity and retrieval |
| `rev_response_biology.py` | Generic programme response versus curated downstream-target association |
| `rev_selection_matched_transfer.py` | Full-panel top-gene selection before shared-gene projection, frozen phase readouts and paired transfer intervals |
| `rev_within_cellcycle.py` | Saved within-dataset input ladder, fixed out-of-fold predictions and paired intervals |
| `rev_validate_cell_states.py` | Fresh GPU extraction of sampled cached states with row-ID mapping and nearest-row checks |
| `rev_circuit.py` | Complete circuit effects and split-half statistic vectors |
| `rev_cpu_analysis.py` | Circuit chance/stability, biological categories, persistence and dimensionality controls |
| `rev_residual_summary.py` | Residual geometry, magnitude and firing summaries for matched scaling |
| `rev_massive_dimensions.py` | Prespecified high-RMS coordinate removal and matched dictionary firing sensitivity |
| `rev_misc.py` | Feature selection, cell-line dictionary transfer, gene means and pooling comparisons |
| `export_revision_atlas.py` | Validated new-directory atlas export with separate evidence types |
| `run_revision.py` | Sequential resumable queue, stage logs and success-only completion sentinels |

See [the reproduction guide](../docs/04-reproduce.md) for workspace paths and commands. Each module
except the download builder supports `--help`. Model work and large CPU analyses belong on a
compute machine. The tests below use small fixtures and do not download weights or datasets:

```bash
python -m unittest discover -s revision -p 'test_*.py'
```

The primary annotation method has no overlap-selected BH mode. `--legacy` on the raw scorer and
`--library-layout snapshot` on the public adapter only select old library file names. The
historical scorer remains recoverable through Git history; it is not an active default.

Atlas export requires `--evaluation-dir` containing fresh `2b_eval_LXX.json` metrics. It never
substitutes original training metrics. Annotation/floor curves share an all-dictionary-feature
denominator, with explicit alive-feature rates retained; ontology/regulon and combined-evidence
floors are separate. Stored and browser-retained circuit edge counts are also reported separately.

The atlas export uses v4 delivery: complete annotation details in fixed 256-feature-ID gzip chunks,
with exact encoded/decoded SHA256 values in `data_transport.json`. Other resources at least 1 MiB
are compressed too; small metadata remains plain. The default delivery guards require the complete
published data (including the final manifest) below 950,000,000 bytes and every file below 100 MiB.
The corresponding atlas UI is required for v4. All scientific records/counts are preserved; the
feature page fetches a chunk on demand instead of a whole layer of detailed annotations.
