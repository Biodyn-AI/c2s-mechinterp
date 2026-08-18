# Paper sources

`make_figures.py` regenerates Figures 1–5 and `make_graphical_abstract.py` the journal-format
graphical abstract, both directly from the artefacts in `results/` and the atlas summary JSON.
No number is hard-coded that is not traceable to a file.

```bash
python paper/make_figures.py
python paper/make_graphical_abstract.py
latexmk -pdf manuscript.tex     # requires the elsarticle class (TeX Live)
```

Both scripts auto-fit label text to its box, so edited labels cannot overflow.
