"""heldout_v2 — the held-out gene test, with BOTH caveats of heldout_gene_placement.py fixed.

CAVEAT 1 THERE: "absent from the K562 panel" was confounded with "not expressed in K562", so the null could
  simply mean the gene is unfamiliar to the model.
  FIX: keep only held-out genes that are WELL EXPRESSED ELSEWHERE — Tabula Sapiens document frequency >= 5%
  (range 6-36%). The model has demonstrably seen these genes many times; they are just not in THIS panel.
  Plus a FAMILIARITY-MATCHED random control drawn from the same frequency band.

CAVEAT 2 THERE: a 1-8 gene "cell sentence" is far off-distribution (radius 1.15-1.76 vs a real-cell median of
  0.86) and every condition collapsed to ~180 deg.
  FIX: insert the genes into a FULL 512-gene phase-neutral backbone (a real cell with all cell-cycle genes
  stripped) at fixed rank positions — exactly the constructive-sweep design that gave 114.3 deg on in-panel
  genes, so the input stays on-distribution and the instrument is known to work.

READ: if held-out-but-familiar phase genes separate the two classes, the model carries gene->phase knowledge
that is NOT in this dataset's covariance. If they do not, while in-panel genes do, the association is tied to
the genes actually present in the data.
Out: results/heldout_v2.json
"""

# --- make sibling modules importable when run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "c2s_mechinterp"))

from __future__ import annotations
import os, sys, json, argparse, pickle, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_sentences import anndata_to_ranked_genes, build_cell_sentence
from cc_phase import phase_angle, S_GENES, G2M_GENES
from synth_sweep import PEAK, insert_at

ENS_PATH = os.environ.get("C2S_ENSMAP", "/root/hd/gene_name_id_dict.pkl")
G2G_PATH = os.environ.get("C2S_GENE2GO", "/root/hd/gene2go_all.pkl")
CC_ANY = {"GO:0000082", "GO:0000086", "GO:0051301", "GO:0006260", "GO:0007059", "GO:0000278", "GO:0007049"}


def cmean(a):
    a = np.radians(np.asarray(a, float))
    return float(np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))) % 360)


def cdist(a, b):
    d = abs(float(a) - float(b)) % 360.0
    return min(d, 360.0 - d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--ts-h5ad", required=True, help="Tabula Sapiens panel, for the familiarity measure")
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--model", default="vandijklab/C2S-Scale-Gemma-2-2B")
    ap.add_argument("--n-backbone", type=int, default=24)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-genes", type=int, default=512)
    ap.add_argument("--fam-min", type=float, default=0.05)
    ap.add_argument("--out", default="results/heldout_v2.json")
    a = ap.parse_args()
    import anndata, torch, scipy.sparse as sp
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    ad = anndata.read_h5ad(a.h5ad)
    panel = set(np.char.upper(np.asarray(ad.var_names).astype(str)))
    g2g = pickle.load(open(G2G_PATH, "rb"))
    sym2ens = {k.upper(): v for k, v in pickle.load(open(ENS_PATH, "rb")).items()}
    panel_ens = {sym2ens[s] for s in panel if s in sym2ens}
    # familiarity from Tabula Sapiens
    ts = anndata.read_h5ad(a.ts_h5ad)
    tv = np.char.upper(np.asarray(ts.var_names).astype(str))
    Xts = ts.X.toarray() if sp.issparse(ts.X) else np.asarray(ts.X)
    freq = {s: float(f) for s, f in zip(tv, (Xts > 0).mean(0))}
    g1s = {g.upper() for g, t in g2g.items() if "GO:0000082" in set(t)}
    g2m = {g.upper() for g, t in g2g.items() if "GO:0000086" in set(t)}
    both = g1s & g2m; g1s, g2m = g1s - both, g2m - both
    ccany = {g.upper() for g, t in g2g.items() if CC_ANY & set(t)}
    heldfam = lambda S: sorted({g for g in S if sym2ens.get(g) and sym2ens[g] not in panel_ens
                                and freq.get(g, 0.0) >= a.fam_min})
    hA, hB = heldfam(g1s), heldfam(g2m)
    inA = sorted({g for g in g1s if sym2ens.get(g) and sym2ens[g] in panel_ens})
    inB = sorted({g for g in g2m if sym2ens.get(g) and sym2ens[g] in panel_ens})
    # familiarity-MATCHED random control: held-out, non-cycle, same frequency band
    lo, hi = min(freq[g] for g in hA + hB), max(freq[g] for g in hA + hB)
    ctrl = sorted({g for g in freq if g not in ccany and sym2ens.get(g) and sym2ens[g] not in panel_ens
                   and lo <= freq[g] <= hi})
    print(f"held-out FAMILIAR: G1/S {len(hA)} {hA[:8]}", flush=True)
    print(f"                   G2/M {len(hB)} {hB[:8]}", flush=True)
    print(f"in-panel calibration: {len(inA)}/{len(inB)} | matched random control: {len(ctrl)} "
          f"(freq band {lo:.3f}-{hi:.3f})", flush=True)

    # circle fitted on REAL cells
    theta, _, _, _ = phase_angle(ad)
    ids = np.load(os.path.join(a.states, "row_cell_ids.npy"))
    H = np.load(os.path.join(a.states, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    t = theta[ids][:len(H)]; H = H[:len(t)]
    sc = StandardScaler().fit(H)
    dec = Ridge(alpha=1e3).fit(sc.transform(H), np.column_stack([np.cos(t), np.sin(t)]))
    P0 = dec.predict(sc.transform(H)); C0 = P0.mean(0)
    real_rad = float(np.median(np.linalg.norm(P0 - C0, axis=1)))

    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto").eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    dev = next(model.parameters()).device
    maxlen = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)

    cc_strip = (set(S_GENES) | set(G2M_GENES) | set(PEAK) | ccany) & panel
    rng = np.random.default_rng(0)
    order = rng.permutation(ad.n_obs)
    backbones = []
    for ci in order:
        g = [x for x in anndata_to_ranked_genes(ad, int(ci), max_genes=a.max_genes) if x.upper() not in cc_strip]
        if len(g) >= 200:
            backbones.append(g[: a.max_genes - a.k])
        if len(backbones) >= a.n_backbone:
            break
    positions = np.linspace(20, 120, a.k).astype(int)
    print(f"{len(backbones)} phase-neutral backbones (stripped {len(cc_strip)} cc genes), k={a.k}, "
          f"real-cell median radius {real_rad:.3f}", flush=True)

    def place(bb, mod):
        cs = build_cell_sentence(insert_at(bb, list(mod), positions))
        enc = tok(cs.text, return_tensors="pt", truncation=True, max_length=maxlen)
        pos = int(enc["input_ids"].shape[1]) - 1
        with torch.no_grad():
            out = model(**{k: v.to(dev) for k, v in enc.items()}, output_hidden_states=True)
        h = out.hidden_states[a.layer + 1][0, pos].float().cpu().numpy()[None, :]
        p = dec.predict(sc.transform(h))[0] - C0
        return float(np.degrees(np.arctan2(p[1], p[0])) % 360), float(np.linalg.norm(p))

    conds = {"heldout_familiar": (hA, hB), "inpanel": (inA, inB), "matched_random": (ctrl, ctrl)}
    res = {"real_radius": real_rad, "sets": {"heldout_G1S": hA, "heldout_G2M": hB,
                                             "n_inpanel": [len(inA), len(inB)], "n_ctrl": len(ctrl)}, "conds": {}}
    for cname, (A, B) in conds.items():
        if len(A) < a.k or len(B) < a.k:
            print(f"  {cname}: too few genes ({len(A)}/{len(B)}) — skipped", flush=True); continue
        angA, angB, radA, radB = [], [], [], []
        for bb in backbones:
            ga = list(rng.choice(A, a.k, replace=False)); gb = list(rng.choice(B, a.k, replace=False))
            x, r1 = place(bb, ga); y, r2 = place(bb, gb)
            angA.append(x); angB.append(y); radA.append(r1); radB.append(r2)
        # paired per-backbone separation
        seps = [cdist(x, y) for x, y in zip(angA, angB)]
        pool = angA + angB; rng.shuffle(pool)
        shuf = cdist(cmean(pool[:len(angA)]), cmean(pool[len(angA):]))
        res["conds"][cname] = dict(mean_A=cmean(angA), mean_B=cmean(angB),
                                   separation_of_means=cdist(cmean(angA), cmean(angB)),
                                   paired_sep_mean=float(np.mean(seps)), paired_sep_sem=float(np.std(seps)/np.sqrt(len(seps))),
                                   shuffled=shuf, radius=float(np.mean(radA + radB)),
                                   radius_vs_real=float(np.mean(radA + radB) / real_rad), n=len(seps))
        r = res["conds"][cname]
        print(f"  {cname:<18} A={r['mean_A']:6.1f} B={r['mean_B']:6.1f} | sep(means)={r['separation_of_means']:6.1f} "
              f"paired={r['paired_sep_mean']:6.1f}±{r['paired_sep_sem']:.1f} (shuf {r['shuffled']:5.1f}) | "
              f"radius {r['radius_vs_real']:.2f}x real", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
