"""gene_phase_map — is the gene -> cell-cycle-phase map in the WEIGHTS, or only read off this dataset's covariance?

WHY THIS REPLACES heldout_v2.py. That test tried to hold out gene IDENTITY -- find phase genes absent from the
K562 panel -- and failed mechanically: the only sizeable held-out sets are the GO transition annotations
(GO:0000082 / GO:0000086), far weaker markers than Whitfield's peak-phase modules. Its in-panel calibration arm
scored 35 deg against a 31 deg shuffled floor where the working constructive instrument gets 114 deg, so the
held-out arm's null was uninterpretable. The instrument failed, not the model.

THE FIX: hold out INFORMATION, not gene identity. Every probe gene is inserted into the same phase-neutral
backbone at the same rank, so the input says nothing about it except its NAME. Then split genes by how
phase-informative they are IN THIS DATASET and ask:

    does the model place a gene at its externally-measured phase even when that gene carries NO phase
    information in this dataset's expression covariance?

*** THIS SCRIPT'S PREMISE IS WITHDRAWN. IT IS KEPT ONLY AS THE RECORD OF A FAILED DESIGN. ***
*** The valid successor is surface_forms.py; do not cite results/gene_phase_map.json.        ***

An earlier version of this docstring claimed the co-expression baseline is "at chance by construction" in the
low-informativeness stratum, citing circular correlations of -0.004 (K562) and +0.005 (RPE1) as a verified
floor. That claim is retracted on three counts:

  1. WRONG SPLIT. Those two numbers came from a bottom-TERTILE split (n=14) run ad hoc during development,
     whereas the code below splits at the MEDIAN (n=21) and reports baseline_circ_corr = +0.524 for its own low
     stratum. The quoted floor was never a property of the stratification actually shipped.
  2. WRONG PREMISE. "Informativeness" is the resultant length of a gene's expression-weighted phase, which
     measures SPREAD, not accuracy. Low-|r| genes still carry a correct mean angle -- a median of 14 deg from
     their own class pole -- and the baseline scores AUROC 0.993 there. Nothing was ever held out.
  3. WRONG COMPARISON. In this design the model never sees expression covariance at all; it sees ONE GENE NAME
     in a phase-neutral backbone. It is already held out by construction, so a covariance baseline is not the
     relevant null. The relevant null is a token with matched surface statistics and no gene identity -- the
     anagram arm of surface_forms.py, which sits at AUROC 0.511, p = 0.43.

The A2 "graded" arm is separately void: it computes median error on FLIPPED model angles but circular
correlation on UNFLIPPED ones, while holding the baseline unflipped, so its headline contrast is between two
different reflection frames. Its truth (Whitfield peak phase) is also bimodal-antipodal, R = 0.105, making it
an axial not a graded statistic.

THREE ARMS. Statistics were chosen by an explicit power analysis BEFORE running, after a first design (circular
correlation on informativeness tertiles) was found underpowered -- oracle +0.602 against a shuffled 95th
percentile of +0.462 at n=14. Simulated power at p<0.05:

  A1 BINARY (primary)   external label = Tirosh's curated S vs G2M marker classes; 87 genes in panel, 43 in the
                        low-informativeness half. Score = which pole of the REAL-CELL circle the placed gene
                        falls toward. AUROC + label permutation. Power 1.00 down to +/-60 deg placement noise,
                        0.97 at +/-80 deg. This is the well-powered version of the question.
  A2 GRADED (paired)    external label = Whitfield 2002 synchronised-HeLa peak phase, in degrees. Per-gene
                        angular error, model vs the covariance baseline, PAIRED Wilcoxon within the same genes
                        -- far more powerful than comparing two circular correlations. n=21 in the low half;
                        power 1.00 at +/-40 deg, 0.87 at +/-60 deg.
  A3 SURFACE FORMS      the days-of-week analogue proper, scored on the well-powered A1 statistic:
                          human    CCNB1 ...  the in-panel symbols
                          mouse    Ccnb1 ...  mouse orthologs: ZERO occurrences in any K562 cell sentence, so
                                              they carry no covariance in this dataset at all
                          anagram  NBCC1 ...  letters permuted, checked against the real gene universe. Same
                                              surface statistics, identity destroyed -- the floor that
                                              distinguishes "reads gene identity" from "reacts to a rare token"
                        human >> anagram shows the instrument reads identity; mouse >> anagram shows a token
                        absent from the data still gets placed. Note that a mouse win is evidence against a
                        reparametrisation of THIS dataset either way -- whether the model is doing species
                        knowledge or surface-form generalisation, neither is in the input covariance.

Out: results/gene_phase_map.json
"""

# --- make sibling modules importable when run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "c2s_mechinterp"))

from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cell_sentences import anndata_to_ranked_genes, build_cell_sentence
from cc_phase import phase_angle_oriented, S_GENES, G2M_GENES
from synth_sweep import PEAK, insert_at


def cmean_deg(a):
    a = np.radians(np.asarray(a, float))
    return float(np.degrees(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a)))) % 360)


def circ_corr_deg(a, b):
    a, b = np.radians(np.asarray(a, float)), np.radians(np.asarray(b, float))
    sa = np.sin(a - np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))
    sb = np.sin(b - np.arctan2(np.mean(np.sin(b)), np.mean(np.cos(b))))
    return float(np.sum(sa * sb) / (np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2)) + 1e-12))


def adist(a, b):
    d = np.abs(np.asarray(a, float) - np.asarray(b, float)) % 360.0
    return np.minimum(d, 360.0 - d)


def best_rot(pred, true, flip):
    """Single best rigid rotation. Fitted ONCE on the full gene set and then frozen -- never re-fitted per
    stratum, which would let the null absorb the effect it is supposed to bound."""
    p = (-np.asarray(pred, float) if flip else np.asarray(pred, float))
    grid = np.arange(0, 360, 1.0)
    errs = [np.median(adist(p + g, true)) for g in grid]
    i = int(np.argmin(errs))
    return float(grid[i]), float(errs[i])


def auroc(score, y):
    score = np.asarray(score, float); y = np.asarray(y, int)
    r = score.argsort().argsort() + 1.0
    n1 = int(y.sum()); n0 = len(y) - n1
    if n0 == 0 or n1 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n0 * n1))


def perm_auroc(score, y, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = auroc(score, y)
    null = np.array([auroc(rng.permutation(np.asarray(score, float)), y) for _ in range(n // 20)])
    return obs, float((np.sum(null >= obs) + 1) / (len(null) + 1)), float(null.mean()), float(null.std())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--states", required=True)
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--model", default="vandijklab/C2S-Scale-Gemma-2-2B")
    ap.add_argument("--n-backbone", type=int, default=24)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--max-genes", type=int, default=512)
    ap.add_argument("--out", default="results/gene_phase_map.json")
    a = ap.parse_args()
    import anndata, torch, scipy.sparse as sp
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from scipy.stats import wilcoxon

    ad = anndata.read_h5ad(a.h5ad)
    vn = np.char.upper(np.asarray(ad.var_names).astype(str))
    panel = set(vn)

    # ---- circle decoder fitted on REAL cells, on the CANONICALLY ORIENTED phase ----
    theta, _, _, oinfo = phase_angle_oriented(ad)
    ids = np.load(os.path.join(a.states, "row_cell_ids.npy"))
    H = np.load(os.path.join(a.states, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    t = theta[ids][:len(H)]; H = H[:len(t)]
    sc = StandardScaler().fit(H)
    dec = Ridge(alpha=1e3).fit(sc.transform(H), np.column_stack([np.cos(t), np.sin(t)]))
    P0 = dec.predict(sc.transform(H)); C0 = P0.mean(0)
    real_rad = float(np.median(np.linalg.norm(P0 - C0, axis=1)))
    mu_S, mu_G = oinfo["s_peak_deg"], oinfo["g2m_peak_deg"]
    print(f"orientation: flipped={oinfo['flipped']} rot={oinfo['rotation_deg']}deg | real-cell S peak {mu_S}deg, "
          f"G2M peak {mu_G}deg | median radius {real_rad:.3f}", flush=True)

    # ---- external labels ----
    tirosh = {}
    for g in S_GENES:   tirosh[g.upper()] = 0
    for g in G2M_GENES: tirosh[g.upper()] = 1
    tirosh = {g: v for g, v in tirosh.items() if g in panel}
    peak = {g: v for g, v in PEAK.items() if g in panel}
    probes = sorted(set(tirosh) | set(peak))
    print(f"probe genes: {len(probes)} (Tirosh binary {len(tirosh)} = "
          f"{sum(1 for g in tirosh if tirosh[g] == 0)}S/{sum(1 for g in tirosh if tirosh[g] == 1)}G2M | "
          f"Whitfield graded {len(peak)})", flush=True)

    # ---- informativeness + covariance baseline angle, both from THIS dataset ----
    X = ad.X.toarray() if sp.issparse(ad.X) else np.asarray(ad.X)
    X = X[ids][:len(t)]
    gidx = {g: i for i, g in enumerate(vn)}
    z = np.exp(1j * t)
    base_ang, informative = {}, {}
    for g in probes:
        x = X[:, gidx[g]].astype(np.float64)
        x = np.clip(x - x.mean(), 0, None)
        if x.sum() <= 0:
            continue
        r = (x * z).sum() / x.sum()
        base_ang[g] = float(np.degrees(np.angle(r)) % 360)
        informative[g] = float(np.abs(r))
    probes = [g for g in probes if g in informative]
    print(f"baseline + informativeness defined for {len(probes)} | informativeness deciles "
          f"{np.percentile(list(informative.values()), [10, 50, 90]).round(3)}", flush=True)

    # ---- surface forms ----
    rng = np.random.default_rng(0)
    def anagram(g):
        for _ in range(400):
            s = "".join(rng.permutation(list(g)))
            if s != g and s not in panel:
                return s
        return g[::-1]
    forms = {"human": {g: g for g in probes},
             "mouse": {g: g.capitalize() for g in probes},
             "anagram": {g: anagram(g) for g in probes}}
    ex = probes[0]
    print("forms: " + " | ".join(f"{k}:{forms[k][ex]}" for k in forms), flush=True)

    # ---- model + phase-neutral backbones ----
    tok = AutoTokenizer.from_pretrained(a.model)
    try:
        model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, device_map="auto",
                                                     attn_implementation="eager").eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16, device_map="auto",
                                                     attn_implementation="eager").eval()
    device = next(model.parameters()).device
    maxlen = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)

    cc_strip = (set(S_GENES) | set(G2M_GENES) | set(PEAK)) & panel
    order = rng.permutation(ad.n_obs)
    backbones = []
    for ci in order:
        gs = [x for x in anndata_to_ranked_genes(ad, int(ci), max_genes=a.max_genes) if x.upper() not in cc_strip]
        if len(gs) >= 200:
            backbones.append(gs[: a.max_genes - 1])
        if len(backbones) >= a.n_backbone:
            break
    print(f"{len(backbones)} phase-neutral backbones (stripped {len(cc_strip)} cc genes), insert rank {a.rank}",
          flush=True)

    def place(bb, sym):
        cs = build_cell_sentence(insert_at(bb, [sym], [a.rank]))
        enc = tok(cs.text, return_tensors="pt", truncation=True, max_length=maxlen)
        pos = int(enc["input_ids"].shape[1]) - 1
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
        h = out.hidden_states[a.layer + 1][0, pos].float().cpu().numpy()[None, :]
        p = dec.predict(sc.transform(h))[0] - C0
        return float(np.degrees(np.arctan2(p[1], p[0])) % 360), float(np.linalg.norm(p))

    res = {"orientation": oinfo, "real_radius": real_rad, "n_backbone": len(backbones), "insert_rank": a.rank,
           "n_probes": len(probes), "informativeness": informative, "baseline_angles": base_ang,
           "gene_angles": {}, "A1_binary": {}, "A2_graded": {}, "A3_forms": {}}

    # ---- forward sweep: every form x every probe gene ----
    model_ang, model_rad = {}, {}
    for fname, mapping in forms.items():
        ang, rad = {}, {}
        for gi_, g in enumerate(probes):
            per = [place(bb, mapping[g]) for bb in backbones]
            ang[g] = cmean_deg([p[0] for p in per]); rad[g] = float(np.mean([p[1] for p in per]))
            if gi_ % 25 == 0:
                print(f"  [{fname}] {gi_ + 1}/{len(probes)}", flush=True)
        model_ang[fname] = ang; model_rad[fname] = rad
        res["gene_angles"][fname] = ang

    # ================= A1 BINARY: Tirosh S vs G2M, low-informativeness half =================
    # score = which pole of the REAL-CELL circle the placed gene falls toward. Both poles come from real cells,
    # never from the probe genes, so the axis is not fitted on the thing being tested.
    def pole_score(angles, gs):
        return [np.cos(np.radians(angles[g] - mu_G)) - np.cos(np.radians(angles[g] - mu_S)) for g in gs]

    tg = [g for g in probes if g in tirosh]
    tg_sorted = sorted(tg, key=lambda g: informative[g])
    half = len(tg_sorted) // 2
    subsets = {"all": tg_sorted, "low_informativeness": tg_sorted[:half], "high_informativeness": tg_sorted[half:]}
    print(f"\nA1 BINARY (Tirosh S vs G2M; AUROC of real-cell-pole score, label permutation)", flush=True)
    print(f"  {'subset':<22}{'n':>4}{'S/G2M':>9}{'inf':>7}{'MODEL auroc':>14}{'p':>9}{'BASE auroc':>13}{'p':>9}",
          flush=True)
    for sname, gs in subsets.items():
        y = [tirosh[g] for g in gs]
        m_a, m_p, _, _ = perm_auroc(pole_score(model_ang["human"], gs), y)
        b_a, b_p, _, _ = perm_auroc(pole_score(base_ang, gs), y)
        res["A1_binary"][sname] = dict(n=len(gs), n_S=int(len(y) - sum(y)), n_G2M=int(sum(y)),
                                       mean_informativeness=float(np.mean([informative[g] for g in gs])),
                                       model_auroc=m_a, model_p=m_p, baseline_auroc=b_a, baseline_p=b_p,
                                       genes=gs)
        print(f"  {sname:<22}{len(gs):>4}{f'{len(y)-sum(y)}/{sum(y)}':>9}"
              f"{np.mean([informative[g] for g in gs]):>7.3f}{m_a:>14.3f}{m_p:>9.4f}{b_a:>13.3f}{b_p:>9.4f}",
              flush=True)

    # ================= A2 GRADED: paired per-gene error vs Whitfield =================
    pg = [g for g in probes if g in peak]
    pred_all = [model_ang["human"][g] for g in pg]; true_all = [peak[g] for g in pg]
    r0, e0 = best_rot(pred_all, true_all, False); r1, e1 = best_rot(pred_all, true_all, True)
    flip = e1 < e0; rot = r1 if flip else r0
    br0, be0 = best_rot([base_ang[g] for g in pg], true_all, False)
    br1, be1 = best_rot([base_ang[g] for g in pg], true_all, True)
    bflip = be1 < be0; brot = br1 if bflip else br0
    res["A2_graded"]["alignment"] = dict(model_flip=bool(flip), model_rot_deg=rot,
                                         baseline_flip=bool(bflip), baseline_rot_deg=brot)

    def err_of(angles, gs, fl, ro):
        p = np.array([-angles[g] if fl else angles[g] for g in gs], float)
        return adist(p + ro, [peak[g] for g in gs])

    pg_sorted = sorted(pg, key=lambda g: informative[g]); ph = len(pg_sorted) // 2
    print(f"\nA2 GRADED (Whitfield peak phase; PAIRED per-gene error, Wilcoxon model<baseline)", flush=True)
    print(f"  {'subset':<22}{'n':>4}{'model err':>12}{'base err':>11}{'wilcoxon p':>13}{'model r':>10}{'base r':>9}",
          flush=True)
    for sname, gs in [("all", pg_sorted), ("low_informativeness", pg_sorted[:ph]),
                      ("high_informativeness", pg_sorted[ph:])]:
        me = err_of(model_ang["human"], gs, flip, rot); be = err_of(base_ang, gs, bflip, brot)
        try:    w = float(wilcoxon(me, be, alternative="less").pvalue)
        except Exception: w = float("nan")
        m_r = circ_corr_deg([model_ang["human"][g] for g in gs], [peak[g] for g in gs])
        b_r = circ_corr_deg([base_ang[g] for g in gs], [peak[g] for g in gs])
        res["A2_graded"][sname] = dict(n=len(gs), model_median_err_deg=float(np.median(me)),
                                       baseline_median_err_deg=float(np.median(be)), wilcoxon_p=w,
                                       model_circ_corr=m_r, baseline_circ_corr=b_r,
                                       mean_informativeness=float(np.mean([informative[g] for g in gs])), genes=gs)
        print(f"  {sname:<22}{len(gs):>4}{np.median(me):>11.0f}d{np.median(be):>10.0f}d{w:>13.4f}"
              f"{m_r:>+10.3f}{b_r:>+9.3f}", flush=True)

    # ================= A3 SURFACE FORMS, scored on the well-powered A1 statistic =================
    print(f"\nA3 SURFACE FORMS (A1 statistic; 'mouse' tokens never occur in this dataset)", flush=True)
    print(f"  {'form':<10}{'auroc(all)':>12}{'p':>9}{'auroc(low-inf)':>17}{'p':>9}{'radius/real':>13}", flush=True)
    for fname in forms:
        aa, ap_, _, _ = perm_auroc(pole_score(model_ang[fname], subsets["all"]),
                                   [tirosh[g] for g in subsets["all"]])
        la, lp, _, _ = perm_auroc(pole_score(model_ang[fname], subsets["low_informativeness"]),
                                  [tirosh[g] for g in subsets["low_informativeness"]])
        rr = float(np.mean([model_rad[fname][g] for g in probes]) / real_rad)
        res["A3_forms"][fname] = dict(auroc_all=aa, p_all=ap_, auroc_low=la, p_low=lp, radius_vs_real=rr,
                                      example={g: forms[fname][g] for g in probes[:5]})
        print(f"  {fname:<10}{aa:>12.3f}{ap_:>9.4f}{la:>17.3f}{lp:>9.4f}{rr:>13.2f}", flush=True)

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"\n[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
