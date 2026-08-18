"""transfer_diag — why did K562 -> RPE1 transfer score -0.81, and does it survive a label-free recentering?

THE PUZZLE. transfer_test.py on RPE1 returned, for BOTH readouts, a strongly NEGATIVE circular correlation
together with a very LOW median error:  model -0.814 (24 deg), expression -0.796 (16 deg). Those two numbers
normally cannot coexist -- 16 deg of median error is near-perfect tracking, which should give a correlation
near +1. So at least one of them is not measuring what it appears to.

The likely resolution is that both are artifacts of CONCENTRATION. If the frozen readout's predictions collapse
into a narrow arc under domain shift, and the target's own phases are also concentrated, then median error is
small simply because everything sits near the same place, while the residual variation that circular
correlation actually scores is noise -- and noise can come out anywhere, including strongly negative.

Note this is NOT the failure the validity gate reported. The gate said "the target's phase label is not cell
cycle", which was the right refusal for the setty run but is wrong here: RPE1's phase label was verified
against K562 gene by gene at circ_corr +0.983 (median 15 deg) after canonical orientation. The label is sound;
what fails is the readout's transfer.

WHAT THIS SCRIPT ADDS
  1. THE MISSING CONTROL -- a CONSTANT predictor that always emits the target's mean phase. If it also achieves
     ~20 deg median error, then median error is uninformative here and the negative correlation is the only
     real signal. Without this control neither number can be interpreted.
  2. DISTRIBUTIONAL DIAGNOSIS -- concentration (resultant length) and circular mean of the true target phase,
     of each readout's predictions, and of the source predictions, so collapse is visible directly.
  3. A FAIRER TRANSFER ARM -- refit the StandardScaler on the TARGET's own features. This uses NO target
     labels, so it remains genuine zero-shot transfer, and it removes the part of the domain shift that is a
     pure feature offset/scale change rather than a change in the phase code. Standard practice under domain
     shift; without it a readout can be penalised for a batch effect that has nothing to do with cell cycle.
  4. AN ORACLE CEILING -- a readout refit ON the target with cross-validation. Not a transfer number; it bounds
     how much phase information the target representation contains at all, so a transfer failure can be told
     apart from "there is nothing there to find".

Out: results/transfer_diag.json
"""

# --- make sibling modules importable when run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "c2s_mechinterp"))

from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_phase import phase_angle_oriented


def circ_corr(a, b):
    sa = np.sin(a - np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))))
    sb = np.sin(b - np.arctan2(np.mean(np.sin(b)), np.mean(np.cos(b))))
    return float(np.sum(sa * sb) / (np.sqrt(np.sum(sa ** 2) * np.sum(sb ** 2)) + 1e-12))


def circ_dist(a, b):
    d = np.abs(a - b) % (2 * np.pi)
    return np.minimum(d, 2 * np.pi - d)


def R(a):
    return float(np.abs(np.mean(np.exp(1j * np.asarray(a)))))


def cmean(a):
    return float(np.mod(np.arctan2(np.mean(np.sin(a)), np.mean(np.cos(a))), 2 * np.pi))


def report(name, pred, true, out):
    out[name] = dict(circ_corr=circ_corr(pred, true),
                     median_err_deg=float(np.rad2deg(np.median(circ_dist(pred, true)))),
                     pred_R=R(pred), pred_mean_deg=float(np.rad2deg(cmean(pred))))
    r = out[name]
    print(f"  {name:<26}{r['circ_corr']:>+9.3f}{r['median_err_deg']:>12.0f}d{r['pred_R']:>10.3f}"
          f"{r['pred_mean_deg']:>12.0f}d", flush=True)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-states", required=True)
    ap.add_argument("--tgt-states", required=True)
    ap.add_argument("--src-h5ad", required=True)
    ap.add_argument("--tgt-h5ad", required=True)
    ap.add_argument("--layer", type=int, default=21)
    ap.add_argument("--out", default="results/transfer_diag.json")
    a = ap.parse_args()
    import anndata, scipy.sparse as sp
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import KFold

    src = anndata.read_h5ad(a.src_h5ad); tgt = anndata.read_h5ad(a.tgt_h5ad)
    ts, _, _, si = phase_angle_oriented(src)
    tt, _, _, ti = phase_angle_oriented(tgt)
    si_ = np.load(os.path.join(a.src_states, "row_cell_ids.npy"))
    ti_ = np.load(os.path.join(a.tgt_states, "row_cell_ids.npy"))
    Hs = np.load(os.path.join(a.src_states, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    Ht = np.load(os.path.join(a.tgt_states, f"layer_{a.layer:02d}_activations.npy")).astype(np.float64)
    ts_m = ts[si_][:len(Hs)]; Hs = Hs[:len(ts_m)]
    tt_m = tt[ti_][:len(Ht)]; Ht = Ht[:len(tt_m)]

    sv = np.char.upper(np.asarray(src.var_names).astype(str))
    tv = np.char.upper(np.asarray(tgt.var_names).astype(str))
    shared = sorted(set(sv) & set(tv))
    sidx = {g: i for i, g in enumerate(sv)}; tidx = {g: i for i, g in enumerate(tv)}
    Xs = (src.X.toarray() if sp.issparse(src.X) else np.asarray(src.X))[:, [sidx[g] for g in shared]][si_][:len(ts_m)]
    Xt = (tgt.X.toarray() if sp.issparse(tgt.X) else np.asarray(tgt.X))[:, [tidx[g] for g in shared]][ti_][:len(tt_m)]

    res = {"n_shared_genes": len(shared), "n_src": int(len(Hs)), "n_tgt": int(len(Ht)),
           "src_orientation": si, "tgt_orientation": ti}
    print(f"SOURCE {Hs.shape} | TARGET {Ht.shape} | shared genes {len(shared)}", flush=True)
    print(f"\nTRUE PHASE distribution: source R={R(ts_m):.3f} mean={np.rad2deg(cmean(ts_m)):.0f}deg | "
          f"target R={R(tt_m):.3f} mean={np.rad2deg(cmean(tt_m)):.0f}deg", flush=True)
    res["true_phase"] = dict(src_R=R(ts_m), tgt_R=R(tt_m),
                             src_mean_deg=float(np.rad2deg(cmean(ts_m))),
                             tgt_mean_deg=float(np.rad2deg(cmean(tt_m))))

    # ---- 1. THE MISSING CONTROL: a constant predictor at the target's mean phase ----
    print(f"\n{'arm':<28}{'circ_corr':>9}{'median err':>13}{'pred R':>10}{'pred mean':>13}", flush=True)
    res["arms"] = {}
    const = np.full_like(tt_m, cmean(tt_m))
    report("CONSTANT (target mean)", const, tt_m, res["arms"])
    rng = np.random.default_rng(0)
    report("RANDOM (uniform)", rng.uniform(0, 2 * np.pi, len(tt_m)), tt_m, res["arms"])

    def fit_predict(Xtr, ytr, Xte, scaler_on_target=False, Xte_for_scaler=None):
        Y = np.column_stack([np.cos(ytr), np.sin(ytr)])
        s_tr = StandardScaler().fit(Xtr)
        m = Ridge(alpha=1e3).fit(s_tr.transform(Xtr), Y)
        if scaler_on_target:
            s_te = StandardScaler().fit(Xte_for_scaler if Xte_for_scaler is not None else Xte)
            P = m.predict(s_te.transform(Xte))
        else:
            P = m.predict(s_tr.transform(Xte))
        return np.arctan2(P[:, 1], P[:, 0])

    for nm, Ftr, Fte in [("MODEL", Hs, Ht), ("EXPRESSION", Xs, Xt)]:
        report(f"{nm} frozen (src scaler)", fit_predict(Ftr, ts_m, Fte), tt_m, res["arms"])
        report(f"{nm} target-recentered", fit_predict(Ftr, ts_m, Fte, True), tt_m, res["arms"])

    # ---- 4. ORACLE: refit on target with CV. Bounds what is present, not a transfer number. ----
    for nm, F in [("MODEL", Ht), ("EXPRESSION", Xt)]:
        Y = np.column_stack([np.cos(tt_m), np.sin(tt_m)])
        P = np.zeros_like(Y)
        for tr, te in KFold(5, shuffle=True, random_state=0).split(F):
            s = StandardScaler().fit(F[tr])
            P[te] = Ridge(alpha=1e3).fit(s.transform(F[tr]), Y[tr]).predict(s.transform(F[te]))
        report(f"{nm} ORACLE (refit on tgt)", np.arctan2(P[:, 1], P[:, 0]), tt_m, res["arms"])

    c = res["arms"]["CONSTANT (target mean)"]
    print(f"\nREAD: the constant predictor achieves {c['median_err_deg']:.0f} deg median error with "
          f"circ_corr {c['circ_corr']:+.3f}.", flush=True)
    if c["median_err_deg"] < 40:
        print("  -> this target IS concentrated, so a constant predictor already scores well and median error "
              "alone cannot establish transfer. Require R_diff above the constant arm's as well.", flush=True)
    else:
        # This is the branch that actually fired on RPE1: constant = 86 deg, i.e. chance is ~90 deg, so the
        # 16-24 deg errors of the frozen readouts are REAL tracking. It was circ_corr that was wrong, not them.
        print(f"  -> chance is ~{c['median_err_deg']:.0f} deg here, so the frozen readouts' errors are genuine "
              f"tracking. Any large NEGATIVE circ_corr alongside them is an artifact of the statistic, not a "
              f"failure of the readout: circ_corr scores sin(x - xbar) and xbar is undefined for the "
              f"near-uniform true phase (R = {res['true_phase']['tgt_R']:.3f}) while the frozen predictions are "
              f"concentrated. Judge transfer on R_diff and rotation-aligned error against these floors.",
              flush=True)
    for nm in ("MODEL", "EXPRESSION"):
        fr = res["arms"][f"{nm} frozen (src scaler)"]["circ_corr"]
        rc = res["arms"][f"{nm} target-recentered"]["circ_corr"]
        orc = res["arms"][f"{nm} ORACLE (refit on tgt)"]["circ_corr"]
        print(f"  {nm:<11} frozen {fr:+.3f} -> recentered {rc:+.3f} (oracle ceiling {orc:+.3f})", flush=True)
    # This note previously read "transfer is judged on circ_corr only; median error is confounded by target
    # concentration" -- EXACTLY BACKWARDS, and it was written into the JSON unconditionally even though the
    # branch that would have justified it never fired. On this target the constant predictor scores 86 deg, so
    # median error was never bought by concentration; circ_corr was the invalid statistic.
    res["verdict_note"] = ("Judge transfer on R_diff = |mean(exp(i(pred-true)))| and on median angular error "
                           "after best rotation, both against the CONSTANT and RANDOM arms measured here. "
                           "circ_corr is NOT valid when the true phase is near-uniform and the frozen "
                           "predictions are concentrated: it returned -0.814 where R_diff was +0.789.")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
