#!/usr/bin/env python
"""JBI graphical abstract: min 531 x 1328 px (h x w), readable at 5 x 13 cm.
Strict box layout + auto-fit so no label can escape its box."""
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
OUT = "/Volumes/Crucial X6/MacBook/Code/biomi_automation/projects/c2s-scale/paper_jbi/figures"
W, H, DPI = 13.28, 5.31, 300
C_M, C_G, C_R, C_P = "#2166ac", "#1a9850", "#b2182b", "#762a83"
fig, ax = plt.subplots(figsize=(W/2.54, H/2.54), dpi=DPI)
ax.axis("off"); ax.set_xlim(0, W); ax.set_ylim(0, H)
_t = []

def box(x, y, w, h, title, lines, ec, fc="#ffffff", ts=7.0, bs=5.6, tcol=None):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05", fc=fc, ec=ec, lw=1.0))
    cx, avail = x + w/2, w - 0.25
    t0 = ax.text(cx, y + h - 0.30, title, ha="center", va="center",
                 fontsize=ts, fontweight="bold", color=tcol or ec)
    _t.append((t0, avail))
    for i, ln in enumerate(lines):
        ti = ax.text(cx, y + h - 0.62 - i*0.25, ln, ha="center", va="center",
                     fontsize=bs, color="#222")
        _t.append((ti, avail))

def fit():
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = ax.transData.inverted()
    for tx, mw in _t:
        for _ in range(30):
            bb = tx.get_window_extent(renderer=r)
            if (inv.transform((bb.x1,0))[0] - inv.transform((bb.x0,0))[0]) <= mw \
               or tx.get_fontsize() <= 3.6: break
            tx.set_fontsize(tx.get_fontsize() - 0.15)

ax.text(W/2, 4.98, "Mechanistic interpretability of a language model that reads cells as sentences",
        ha="center", va="center", fontsize=8.6, fontweight="bold", color="#111")

# --- pipeline row ---
pw, py, ph = 4.05, 3.05, 1.42
for i, (t, l, c, f) in enumerate([
        ("A cell as a sentence", ["top-512 gene names, ranked", "expression values discarded"], C_M, "#eef3f8"),
        ("C2S-Scale (Gemma-2)", ["2B and 27B parameters", "26 and 46 layers"], "#444", "#f7f7f7"),
        ("Open the model", ["sparse autoencoders, causal", "circuit tracing, steering"], C_P, "#f7f2fa")]):
    box(0.30 + i*(pw + 0.35), py, pw, ph, t, l, c, f)
for i in range(2):
    x0 = 0.30 + i*(pw + 0.35) + pw
    ax.add_patch(FancyArrowPatch((x0 + 0.03, py + ph/2), (x0 + 0.32, py + ph/2),
                                 arrowstyle="-|>", mutation_scale=8, lw=1.1, color="#333"))

# --- findings row ---
fw, fy, fh = 4.05, 0.85, 1.85
for i, (t, l, c, f) in enumerate([
        ("✓  Interpretable, causally used",
         ["68% of 119,808 features annotated;", "half persist across all 26 layers;",
          "geometry beats a co-expression null"], C_G, "#f2faf4"),
        ("✗  No regulatory logic",
         ["detects 97% of gene knockdowns,", "but only 2% give a response",
          "specific to that factor's targets"], C_R, "#fdf5f4"),
        ("✗  Bigger is less readable",
         ["at matched capacity the 27B has", "31x more dead features and 19 pts",
          "lower annotation than the 2B"], C_R, "#fdf5f4")]):
    box(0.30 + i*(fw + 0.35), fy, fw, fh, t, l, c, f)

ax.text(W/2, 0.36, "Open feature atlas:  biodyn-ai.github.io/c2s-atlas",
        ha="center", va="center", fontsize=6.2, color="#01665e", style="italic")
fit()
fig.savefig(f"{OUT}/graphical_abstract.tiff", bbox_inches="tight", dpi=DPI,
            pil_kwargs={"compression": "tiff_lzw"})
fig.savefig(f"{OUT}/graphical_abstract.pdf", bbox_inches="tight", dpi=DPI)
fig.savefig(f"{OUT}/graphical_abstract.png", bbox_inches="tight", dpi=DPI)
plt.close(fig)
from PIL import Image
im = Image.open(f"{OUT}/graphical_abstract.png")
print(f"graphical abstract: {im.size[0]} x {im.size[1]} px (w x h)  [JBI min 1328 x 531]  "
      f"{'OK' if im.size[0]>=1328 and im.size[1]>=531 else 'TOO SMALL'}")
