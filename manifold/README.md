# Cell-cycle manifold, steering and information-matched transfer

Supporting analyses. These establish that internal directions are **causally used** (the steering
result in Fig. 5b) and that, on cell-level tasks, the model does **not exceed its own input encoding**
(the information ladder in Fig. 5c).

## Main entry points
| File | Purpose |
|---|---|
| `cc_phase.py` | Ground-truth cell-cycle phase in expression space (PCA on cycle genes, θ = atan2(PC2, PC1)) |
| `manifold_fit.py` | Fit the phase manifold in the model's residual stream; decodability by layer |
| `manifold_steer.py`, `steer_c2s.py` | Activation steering with norm-matched random controls |
| `lap_walk.py` | Iterative navigation: a fixed direction stalls on a closed loop, local-tangent steering does not |
| `synth_sweep.py` | Constructive test — insert phase-annotated gene modules and read where the representation lands |
| `continuity_test.py`, `plateau_test.py`, `orthogonal_props.py` | Is the representation continuous? Are there activation plateaus? What loads on the circle vs the radius? |
| `encoding_matched_transfer.py`, `input_null.py`, `selection_matched_arm.py` | **The information-matched baseline machinery** — see `docs/02-controls.md` |
| `cc_benchmark_c2s.py` | Model vs expression baselines on one common ruler |
| `gene_phase_map.py`, `heldout_gene_placement.py`, `surface_forms.py` | Is the gene→role mapping in the weights? (name-only placement, anagram controls) |
| `ctx_*_c2s.py` | Context-representation line, including the co-expression null that is the decisive deflation |

`_ported_from_encoder_routes/` contains reference implementations carried over verbatim from earlier
encoder-model studies so that statistics are computed identically across model families. They are not
part of the C2S pipeline.

## A caution worth repeating
Several results here were **retracted during the study** once a control was applied properly — most
importantly, an apparent 3× "sharpening" over the model's own input, which was an artefact of how the
baseline was encoded. `docs/02-controls.md` records what happened and why. Read the write-ups in
`results/` before reusing any number from these scripts.
