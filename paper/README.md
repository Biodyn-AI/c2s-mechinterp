# Paper sources and figures

`manuscript.tex` and `manuscript.pdf` contain the current paper. `supplement.tex`,
`supplement_generated/` and `supplement.pdf` contain the supplementary methods, all numerical
tables, complete figure provenance and measured resource records. Figure 1 is a vector schematic;
Figures 2–6 and S1–S6 are computed from saved artifacts. Every numerical figure has a
`.provenance.json` sidecar with exact plotted values and source SHA256 hashes.

The delivered `compute_accounting.json` omits cloud resource IDs and shortens billing-source
paths; `BUILD_MANIFEST.json` retains the canonical source checksum and pins this delivered
view, with all table values and the charging basis preserved.

`make_revision_figures.py`, `make_design_schematic.py` and `assemble_supplement.py` are the
current build sources. See [reproduction commands](../docs/04-reproduce.md) and the
[versioned numerical artifacts](https://github.com/Biodyn-AI/c2s-mechinterp/releases/tag/controlled-analysis-2026-11).
The older `fig1_design.pdf` through `fig5_beyond_input.pdf` figures and
`make_figures.py` and `make_graphical_abstract.py` are preserved historical sources;
do not use them to regenerate the current figure set.

Reviewer correspondence, cover letters and submitted administrative documents are not part of
this public source directory. The clean source/upload package is maintained separately.
