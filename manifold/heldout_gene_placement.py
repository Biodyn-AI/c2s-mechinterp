"""heldout_gene_placement — the DAYS-OF-THE-WEEK analogue: is the circle in the WEIGHTS, not the input?

THE PROBLEM WITH EVERY PREVIOUS EXPERIMENT HERE. We always fed the model a whole cell (512 gene names) whose
co-expression already determines the phase, so the manifold was reconstructible from the input alone — and
indeed C2S turned out to sit exactly on a linear readout of its own input (an information-preserving
reparametrisation). That cannot demonstrate learned structure.

THE DAYS-OF-THE-WEEK SETUP. Prompt an LLM with the single token "Monday" and it lands at the right place on a
circle. The input is a POINT with no internal structure; the circle can only come from the weights.

THE ANALOGUE. Prompt C2S with a MINIMAL input — a single gene name (or a few) — and ask where it lands on the
cell-state phase circle FITTED ON REAL CELLS. One gene has no covariance, so any correct placement must come
from the model.

THE DECISIVE VERSION — HELD-OUT GENES. Use phase-annotated genes that are **ABSENT from the substrate's
6,546-gene panel**:
    G1/S transition  GO:0000082   211 genes outside the panel
    G2/M transition  GO:0000086   190 genes outside the panel
For these genes the expression baseline has **no column, no weight, and no way to place them even in
principle**. If the model still puts G1/S-transition genes and G2/M-transition genes at the correct, opposite
places on a circle defined by CELLS, then the gene->phase mapping is in the model's weights, not in this
dataset. (It came from the model's pretraining — which is exactly the days-of-week claim: structure from
training, not from the prompt.)

CONTROLS
  shuffled   the same held-out genes with their phase labels permuted  -> must collapse
  random     genes outside the panel with NO cell-cycle annotation     -> must show no separation
  in-panel   the same test on genes that ARE in the panel              -> calibration
Also swept over input size k in {1, 4, 8}: at k=1 there is no within-input covariance at all.

Out: results/heldout_placement.json
"""

# --- make sibling modules importable when run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "c2s_mechinterp"))

from __future__ import annotations
import os, sys, json, argparse, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_phase import phase_angle
from cell_sentences import build_cell_sentence, anndata_to_ranked_genes

ENS_PATH = os.environ.get("C2S_ENSMAP",
    "/Volumes/Crucial X6/MacBook/Code/neuro-mechinterp/models/Geneformer/geneformer/gene_name_id_dict_gc104M.pkl")
G2G_PATH = os.environ.get("C2S_GENE2GO",
    "/Volumes/Crucial X6/MacBook/biomechinterp/biodyn-work/single_cell_mechinterp/data/perturb/gene2go_all.pkl")
GO_G1S, GO_G2M = "GO:0000082", "GO:0000086"
CC_ANY = {"GO:0000082", "GO:0000086", "GO:0051301", "GO:0006260", "GO:0007059", "GO:0000278", "GO:0007049"}


def circ_mean_deg(a):
    a = np.radians(np.asarray(a, float))
    return float(np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))) % 360)


def circ_dist_deg(a, b):
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--model", default="vandijklab/C2S-Scale-Gemma-2-2B")
    ap.add_argument("--ks", type=int, nargs="+", default=[1, 4, 8])
    ap.add_argument("--n-draws", type=int, default=40, help="gene groups per class per k")
    ap.add_argument("--out", default="results/heldout_placement.json")
    a = ap.parse_args()

    import anndata, torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    ad = anndata.read_h5ad(a.h5ad)
    theta, _, _, _ = phase_angle(ad)
    panel = set(np.char.upper(np.asarray(ad.var_names).astype(str)))
    g2g = pickle.load(open(G2G_PATH, "rb"))
    # CANONICAL SYMBOLS ONLY, and "held out" verified BY ENSEMBL ID -- gene2go is full of aliases
    # (AIK/ARK1 = AURKA etc.), so a naive symbol-difference would leak panel genes back in.
    sym2ens = {k.upper(): v for k, v in pickle.load(open(ENS_PATH, "rb")).items()}
    panel_ens = {sym2ens[s] for s in panel if s in sym2ens}
    def held(S):
        return sorted({g for g in S if sym2ens.get(g) is not None and sym2ens[g] not in panel_ens})
    def inpan(S):
        return sorted({g for g in S if sym2ens.get(g) is not None and sym2ens[g] in panel_ens})
    g1s = {g.upper() for g, ts in g2g.items() if GO_G1S in set(ts)}
    g2m = {g.upper() for g, ts in g2g.items() if GO_G2M in set(ts)}
    both = g1s & g2m
    g1s, g2m = g1s - both, g2m - both
    ccany = {g.upper() for g, ts in g2g.items() if CC_ANY & set(ts)}
    sets = {
        "heldout_G1S": held(g1s), "heldout_G2M": held(g2m),
        "inpanel_G1S": inpan(g1s), "inpanel_G2M": inpan(g2m),
        "heldout_random": held({g.upper() for g in g2g} - ccany),
    }
    for k, v in sets.items():
        print(f"  {k:<16} {len(v)} genes", flush=True)

    # ---- the circle, fitted on REAL CELLS ----
    ids = np.load(os.path.join(a.states, "row_cell_ids.npy"))
    H = np.load(os.path.join(a.states, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    t = theta[ids][:len(H)]; H = H[:len(t)]
    sc = StandardScaler().fit(H)
    dec = Ridge(alpha=1e3).fit(sc.transform(H), np.column_stack([np.cos(t), np.sin(t)]))
    P0 = dec.predict(sc.transform(H)); C0 = P0.mean(0)
    ang_cells = np.degrees(np.arctan2(*(P0 - C0)[:, ::-1].T)) % 360
    deg_true = np.degrees(np.mod(t, 2 * np.pi))
    anchor = {"real_G1S_cells": circ_mean_deg(ang_cells[(deg_true > 180) & (deg_true < 300)]),
              "real_G2M_cells": circ_mean_deg(ang_cells[(deg_true < 90) | (deg_true > 330)])}
    print(f"  anchor: real S-phase cells at {anchor['real_G1S_cells']:.0f}deg, "
          f"real G2M cells at {anchor['real_G2M_cells']:.0f}deg", flush=True)

    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    dev = next(model.parameters()).device
    maxlen = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)

    def place(genes):
        """Minimal input: a cell sentence containing ONLY these genes -> angle on the cell-fitted circle."""
        cs = build_cell_sentence(list(genes))
        enc = tok(cs.text, return_tensors="pt", truncation=True, max_length=maxlen)
        pos = int(enc["input_ids"].shape[1]) - 1
        with torch.no_grad():
            out = model(**{k: v.to(dev) for k, v in enc.items()}, output_hidden_states=True)
        h = out.hidden_states[a.layer + 1][0, pos].float().cpu().numpy()[None, :]
        p = dec.predict(sc.transform(h))[0] - C0
        return float(np.degrees(np.arctan2(p[1], p[0])) % 360), float(np.linalg.norm(p))

    rng = np.random.default_rng(0)
    res = {"anchor": anchor, "set_sizes": {k: len(v) for k, v in sets.items()}, "by_k": {}}
    for k in a.ks:
        rec = {}
        for cond, (A, B) in {
            "heldout":  (sets["heldout_G1S"], sets["heldout_G2M"]),
            "inpanel":  (sets["inpanel_G1S"], sets["inpanel_G2M"]),
            "random":   (sets["heldout_random"], sets["heldout_random"]),
        }.items():
            if len(A) < k or len(B) < k:
                continue
            angA, angB, radA, radB = [], [], [], []
            draws = (max(len(A), len(B)) if k == 1 else a.n_draws)
            for di in range(draws):
                if k == 1:
                    ga = [A[di % len(A)]]; gb = [B[di % len(B)]]
                else:
                    ga = list(rng.choice(A, k, replace=False)); gb = list(rng.choice(B, k, replace=False))
                x, r1 = place(ga); y, r2 = place(gb)
                angA.append(x); angB.append(y); radA.append(r1); radB.append(r2)
            mA, mB = circ_mean_deg(angA), circ_mean_deg(angB)
            sep = circ_dist_deg(mA, mB)
            # shuffled-label null: pool the two classes and re-split at random
            pool = angA + angB; rng.shuffle(pool)
            sep_shuf = circ_dist_deg(circ_mean_deg(pool[:len(angA)]), circ_mean_deg(pool[len(angA):]))
            rec[cond] = dict(mean_A_deg=mA, mean_B_deg=mB, separation_deg=sep,
                             separation_shuffled_deg=sep_shuf,
                             radius_A=float(np.mean(radA)), radius_B=float(np.mean(radB)),
                             n_draws=len(angA), angles_A=angA, angles_B=angB)
            print(f"  k={k} {cond:<9} A={mA:6.1f} B={mB:6.1f} | separation {sep:6.1f}deg "
                  f"(shuffled {sep_shuf:5.1f}) | radius {np.mean(radA):.3f}/{np.mean(radB):.3f}", flush=True)
        res["by_k"][str(k)] = rec

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
