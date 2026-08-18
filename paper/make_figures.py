#!/usr/bin/env python
"""JBI paper figures. Scope = the mechanistic-interpretability PIPELINES executed on C2S-Scale.
Every number traceable to a results file; sources noted per figure."""
import json, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from collections import Counter

BASE = "/Volumes/Crucial X6/MacBook/Code/biomi_automation/projects/c2s-scale"
OUT = f"{BASE}/paper_jbi/figures"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 8, "font.family": "DejaVu Sans", "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.8, "figure.dpi": 300})
C_MODEL, C_BASE, C_NULL, C_ACC = "#2166ac", "#b2182b", "#999999", "#1a9850"
def save(fig, n):
    for e in ("pdf", "png"): fig.savefig(f"{OUT}/{n}.{e}", bbox_inches="tight", dpi=300)
    plt.close(fig); print("wrote", n)

# ---------------- Fig 1: the protocol (strict box layout, no overlaps)
fig, ax = plt.subplots(figsize=(7.2, 3.9))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 5.7)

_texts = []   # (text_obj, max_width_in_data_units) -> shrunk to fit before saving

def _fit_all():
    """Shrink any label that is wider than its box. Guarantees no text escapes a box."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    inv = ax.transData.inverted()
    for tx, maxw in _texts:
        for _ in range(24):
            bb = tx.get_window_extent(renderer=r)
            x0 = inv.transform((bb.x0, 0))[0]; x1 = inv.transform((bb.x1, 0))[0]
            if (x1 - x0) <= maxw or tx.get_fontsize() <= 4.2:
                break
            tx.set_fontsize(tx.get_fontsize() - 0.2)

def box(x, y, w, h, title, lines, ec, fc="#ffffff", tsize=7.6, bsize=6.3):
    """Title sits below the top edge; body lines march down on a fixed pitch."""
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                fc=fc, ec=ec, lw=1.15))
    cx = x + w / 2
    avail = w - 0.22
    t0 = ax.text(cx, y + h - 0.30, title, ha="center", va="center",
                 fontsize=tsize, fontweight="bold", color=ec)
    _texts.append((t0, avail))
    for i, ln in enumerate(lines):
        ti = ax.text(cx, y + h - 0.66 - i * 0.27, ln, ha="center", va="center",
                     fontsize=bsize, color="#222")
        _texts.append((ti, avail))

# --- top row: input -> model ---
box(1.45, 4.15, 2.95, 1.40, "A cell as a sentence",
    ["top-512 gene names, ranked", "expression values discarded",
     '"MALAT1  MT-CO2  CD74 …"'], C_MODEL, "#eef3f8")
box(5.60, 4.15, 2.95, 1.40, "C2S-Scale (Gemma-2)",
    ["2B: 26 layers, d = 2304", "27B: 46 layers, d = 4608",
     "residual stream tapped"], "#444", "#f7f7f7")
ax.add_patch(FancyArrowPatch((4.45, 4.85), (5.55, 4.85), arrowstyle="-|>",
                             mutation_scale=12, lw=1.3, color="#333"))

# --- distribution bus from the model down to the four stages ---
xs = [0.25, 2.70, 5.15, 7.60]          # box left edges
W  = 2.15                               # box width
cxs = [x + W / 2 for x in xs]
ax.plot([7.075, 7.075], [4.13, 3.86], color="#999", lw=1.0)
ax.plot([cxs[0], cxs[-1]], [3.86, 3.86], color="#999", lw=1.0)
for cx in cxs:
    ax.add_patch(FancyArrowPatch((cx, 3.86), (cx, 3.60), arrowstyle="-|>",
                                 mutation_scale=9, lw=1.0, color="#999"))

# --- four protocol stages ---
stages = [
    ("Stage 1\nFeature atlas", ["sparse autoencoders", "26 layers x 9,216 features"], C_MODEL),
    ("Stage 2\nCircuit tracing", ["ablate one feature,", "measure downstream"], C_ACC),
    ("Stage 3\nGeometry + steering", ["beyond co-expression?", "causally used?"], "#e08214"),
    ("Stage 4\nPublic atlas", ["119,808 features,", "browsable online"], "#01665e"),
]
for (title, lines, col), x in zip(stages, xs):
    box(x, 2.10, W, 1.50, title, lines, col, "#fcfcfc", tsize=7.2, bsize=6.2)
for cx in cxs:
    ax.add_patch(FancyArrowPatch((cx, 2.08), (cx, 1.72), arrowstyle="-|>",
                                 mutation_scale=9, lw=1.0, color="#999"))

# --- controls banner ---
ax.add_patch(FancyBboxPatch((0.25, 0.35), 9.50, 1.30, boxstyle="round,pad=0.05",
                            fc="#fff8e7", ec="#e0b040", lw=1.15))
ax.text(5.0, 1.36, "Every model claim is judged against three controls",
        ha="center", va="center", fontsize=7.6, fontweight="bold", color="#8a6d1f")
ctrl = [("information-matched", "baseline gets exactly the", "data the model receives"),
        ("co-expression-matched", "null preserves the data's", "correlation structure"),
        ("capacity-matched", "equal features per token", "before comparing sizes")]
for i, (h1, l1, l2) in enumerate(ctrl):
    cx = 1.90 + i * 3.10
    ax.text(cx, 1.00, h1, ha="center", va="center", fontsize=6.8,
            fontweight="bold", color="#8a6d1f")
    ax.text(cx, 0.75, l1, ha="center", va="center", fontsize=6.1, color="#333")
    ax.text(cx, 0.54, l2, ha="center", va="center", fontsize=6.1, color="#333")
for sx in (3.45, 6.55):
    ax.plot([sx, sx], [0.50, 1.12], color="#e0b040", lw=0.8)
_fit_all()
save(fig, "fig1_design")

# ---------------- Fig 2: SAE atlas (runs/2b_atlas_RESULTS.md + cross_layer_tracking.json)
L=[0,4,8,12,16,20,25]; var=[.993,.981,.959,.936,.905,.893,.884]; dead=[.22,.53,.82,.80,3.61,1.37,1.05]
annot=[80.3,80.3,79.8,75.1,59.2,54.0,57.8]; cover=[79.9,74.8,60.3,50.1,36.8,38.9,31.2]
lr=json.load(open(f"{BASE}/runs/2b_atlas/cross_layer_tracking.json"))["long_range_persistence"]
pl=[x["target_layer"] for x in lr]; pv=[100*x["persistence_rate"] for x in lr]
fig, axes = plt.subplots(1,3, figsize=(7.2,2.35))
a=axes[0]; a.plot(L,var,"o-",color=C_MODEL,lw=1.4,ms=4); a.set_xlabel("layer")
a.set_ylabel("variance explained",color=C_MODEL); a.set_ylim(.85,1.0); a.tick_params(axis="y",labelcolor=C_MODEL)
a2=a.twinx(); a2.plot(L,dead,"s--",color=C_BASE,lw=1.2,ms=3.5); a2.set_ylabel("dead features (%)",color=C_BASE)
a2.tick_params(axis="y",labelcolor=C_BASE); a2.spines["top"].set_visible(False)
a.set_title("a  SAE reconstruction quality",fontsize=8,loc="left")
a=axes[1]; a.plot(L,annot,"o-",color=C_ACC,lw=1.4,ms=4,label="annotated (GO/KEGG/Reactome)")
a.plot(L,cover,"^--",color="#762a83",lw=1.2,ms=4,label="assigned to a module")
a.set_xlabel("layer"); a.set_ylabel("% of features"); a.set_ylim(20,92)
a.legend(frameon=False,fontsize=6.0,loc="lower left"); a.set_title("b  Interpretability declines with depth",fontsize=8,loc="left")
a=axes[2]; a.plot(pl,pv,"-",color=C_MODEL,lw=1.6,label="C2S-Scale-2B")
a.fill_between([0,25],2,3,color=C_BASE,alpha=.25)
a.text(13,7.5,"Geneformer atlas, same pipeline:\n2–3%, none past L11",fontsize=6.0,color=C_BASE,ha="center")
a.axhline(50.4,ls=":",lw=.8,color="#555"); a.text(25,54,"50.4% at L25\n(cos 0.88)",fontsize=6.0,ha="right")
a.set_xlabel("layer"); a.set_ylabel("% of L0 features still present"); a.set_ylim(0,105)
a.legend(frameon=False,fontsize=6.2,loc="upper right")
a.set_title("c  Features persist across full depth",fontsize=8,loc="left")
fig.tight_layout(); save(fig,"fig2_sae_atlas")

# ---------------- Fig 3: matched-capacity scaling (RESULTS_matched_control_27b.md)
labels=["2B\n9,216 feat.","27B matched\n9,216 feat.","27B original\n18,432 feat."]
deadf=[1.2,37.7,47.5]; annalive=[73.4,54.5,44.9]; ve=[93.3,97.7,97.5]
x=np.arange(3); fig,axes=plt.subplots(1,3,figsize=(7.2,2.3))
for a,v,t,c,yl,lo,hi in [(axes[0],deadf,"a  Dead features (lower better)",C_BASE,"% of dictionary",0,62),
                         (axes[1],annalive,"b  Annotated among live features",C_ACC,"% of live features",0,88),
                         (axes[2],ve,"c  Variance explained",C_MODEL,"%",90,101.5)]:
    a.bar(x,v,color=[c if i else "#7fb3d5" for i in range(3)],width=.6,edgecolor="black",lw=.5)
    off=(hi-lo)*.025
    for xi,vi in zip(x,v): a.text(xi,vi+off,f"{vi:.1f}",ha="center",fontsize=7,fontweight="bold")
    a.set_xticks(x); a.set_xticklabels(labels,fontsize=6.2); a.set_ylabel(yl,fontsize=7.5)
    a.set_title(t,fontsize=8,loc="left"); a.set_ylim(lo,hi)
axes[0].annotate("",xy=(1.15,52),xytext=(1.85,52),arrowprops=dict(arrowstyle="<->",color="#333",lw=.9))
axes[0].text(1.5,54,"capacity matched",fontsize=6,ha="center",color="#333")
fig.tight_layout(); save(fig,"fig3_scaling")

# ---------------- Fig 4: circuits + regulatory-logic negative
cg=json.load(open(f"{BASE}/runs/2b_atlas/circuit_graph.json"))
cnt=Counter((e["source_layer"],e["target_layer"]) for e in cg["edges"])
fig,axes=plt.subplots(1,3,figsize=(7.2,2.4))
a=axes[0]
for s,c,lab in [(0,C_MODEL,"source L0"),(7,C_ACC,"source L7"),(16,"#762a83","source L16")]:
    p=sorted((t,n) for (ss,t),n in cnt.items() if ss==s)
    if p: a.plot([t for t,_ in p],[n for _,n in p],"o-",ms=2.5,lw=1.2,color=c,label=lab)
a.set_xlabel("downstream layer"); a.set_ylabel("significant causal edges")
a.legend(frameon=False,fontsize=6.2); a.set_title(f"a  Causal influence attenuates\n     ({cg['n_edges']:,} edges)",fontsize=8,loc="left")
a=axes[1]
a.bar([0,1],[85.2,14.8],color=[C_ACC,C_NULL],width=.55,edgecolor="black",lw=.5)
a.set_xticks([0,1]); a.set_xticklabels(["share an\nontology term","do not"],fontsize=6.8)
a.set_ylabel("% of annotated edge pairs"); a.set_ylim(0,100)
a.text(0,87.5,"10,919 / 12,816",ha="center",fontsize=6.5,fontweight="bold")
a.set_title("b  Circuits propagate function",fontsize=8,loc="left")
a=axes[2]
a.bar([0,1],[96.7,2.1],color=[C_ACC,C_BASE],width=.55,edgecolor="black",lw=.5)
for i,v in enumerate([96.7,2.1]): a.text(i,v+3,f"{v}%",ha="center",fontsize=7.5,fontweight="bold")
a.set_xticks([0,1]); a.set_xticklabels(["detects the\nknockdown\n(58/60)","TF-target-specific\nresponse\n(1/48, q<0.05)"],fontsize=6.2)
a.set_ylabel("% of perturbation targets"); a.set_ylim(0,112)
a.axhline(6.2,ls="--",lw=.9,color="#555"); a.text(1.45,8,"Geneformer 6.2%",fontsize=6,ha="right",color="#555")
a.set_title("c  Detection without regulatory logic",fontsize=8,loc="left")
fig.tight_layout(); save(fig,"fig4_circuits")

# ---------------- Fig 5: what lies beyond the input
fig,axes=plt.subplots(1,3,figsize=(7.2,2.45))
a=axes[0]; lay=[9,13,17]; obs=[.0619,.0439,.0547]; nm=[-.0009,.0013,.0017]; ns=[.0095,.0110,.0117]; zs=[6.57,3.86,4.52]
xp=np.arange(3)
a.bar(xp,obs,.5,color=C_ACC,edgecolor="black",lw=.5,label="observed",zorder=3)
a.errorbar(xp,nm,yerr=[2*s for s in ns],fmt="o",color=C_NULL,capsize=4,ms=4,label="co-expression-matched\nnull (±2 SD)",zorder=4)
for i,(o,z) in enumerate(zip(obs,zs)): a.text(i,o+.004,f"z={z:.1f}",ha="center",fontsize=6.6,fontweight="bold",color=C_BASE)
a.set_xticks(xp); a.set_xticklabels([f"L{l}" for l in lay],fontsize=7); a.axhline(0,color="#333",lw=.7)
a.set_ylabel(r"$\Delta$cos, same- vs diff-function"); a.set_ylim(-.03,.088)
a.legend(frameon=False,fontsize=5.8,loc="upper right")
a.set_title("a  Functional geometry beyond\n     co-expression (positive)",fontsize=8,loc="left")
a=axes[1]
arms=["manifold\n(local tangent)","linear\n(fixed dir.)","random\n(norm-matched)"]
a.bar(range(3),[1.728,.913,.194],yerr=[.58,.35,.05],capsize=3,color=[C_ACC,C_MODEL,C_NULL],width=.6,edgecolor="black",lw=.5)
a.set_xticks(range(3)); a.set_xticklabels(arms,fontsize=6.2); a.set_ylabel("readout path length (rad)")
a.text(1,2.45,"75–92% of cells,\np 1e-5..0.01",ha="center",fontsize=6.2,color=C_BASE)
a.set_ylim(0,2.9); a.set_title("b  Representations are causally\n     used (steering)",fontsize=8,loc="left")
a=axes[2]
names=["full expression\n(6,544 genes+values)","top-512 +values","top-512 set (binary)","C2S-2B L21","top-512 set+rank"]
vals=[.929,.885,.882,.875,.868]; cols=[C_BASE,"#d6604d","#f4a582",C_MODEL,"#92c5de"]
a.barh(range(len(vals))[::-1],vals,color=cols,edgecolor="black",lw=.5,height=.62)
for i,v in enumerate(vals): a.text(v+.004,len(vals)-1-i,f"{v:.3f}",va="center",fontsize=6.4,fontweight="bold")
a.set_yticks(range(len(vals))[::-1]); a.set_yticklabels(names,fontsize=5.9)
a.set_xlabel("cell-cycle phase recovery (circ-$R^2$)"); a.set_xlim(.80,.96)
a.set_title("c  On cell-level tasks the model does\n     not exceed its own input",fontsize=8,loc="left")
fig.tight_layout(); save(fig,"fig5_beyond_input")
print("ALL FIGURES DONE")
