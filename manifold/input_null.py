"""input_null — is the model's cell-cycle ring BETTER than the ring already present in its own input?

The discovery claim in RESULTS_manifold.md is under-controlled: phase was DEFINED as the angle in the PC1-PC2
plane of cell-cycle-gene expression, so a ring exists in the input by construction. This script builds the same
statistics on input-side representations, so the model's activations can be compared against them:

  expr_cc    the cell-cycle-gene expression submatrix  -> the DEFINITION (circular by construction; an upper
             bound, NOT a fair null — reported only for calibration)
  expr_all   full log1p-CP10k expression over all genes -> the raw data baseline
  expr_rank  the encoding the model ACTUALLY receives: per cell, value 1/rank for genes present in the top-K
             ranked cell sentence, 0 otherwise -> the FAIR null (the model cannot use more than this)

For each: cross-validated circular decodability of phase, and the same ring geometry used on the model knots
(planarity of the 12 bin centroids, in-plane radius CV, monotone phase ordering).
Out: results/input_null.json
"""

# --- make sibling modules importable when run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "c2s_mechinterp"))

from __future__ import annotations
import os, sys, json, argparse, warnings; warnings.filterwarnings("ignore")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cc_phase import phase_angle, S_GENES, G2M_GENES
from manifold_fit import decodability, knots


def ring_geometry(K, centers):
    """Same test used on the model knots: planarity, in-plane radius CV, monotone phase ordering."""
    ok = np.isfinite(K).all(1)
    K = K[ok]; c = centers[ok]
    C = K.mean(0); X = K - C
    U, S, _ = np.linalg.svd(X, full_matrices=False)
    P = (U * S)[:, :2]
    ang = np.mod(np.arctan2(P[:, 1], P[:, 0]), 2 * np.pi)
    r = np.linalg.norm(P, axis=1)
    seq = np.unwrap(ang[np.argsort(c)])
    mono = bool(np.all(np.diff(seq) > 0) or np.all(np.diff(seq) < 0))
    return dict(planar_frac=float((S[:2] ** 2).sum() / (S ** 2).sum()),
                inplane_radius_cv=float(r.std() / (r.mean() + 1e-12)),
                phase_monotone=mono)


def build_inputs(adata, max_genes=512):
    import scipy.sparse as sp
    X = adata.X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    var = np.char.upper(np.asarray(adata.var_names).astype(str))
    pos = {s: i for i, s in enumerate(var)}
    cc = np.array([pos[g] for g in (S_GENES + G2M_GENES) if g in pos])
    out = {"expr_cc": X[:, cc].astype(np.float64), "expr_all": X.astype(np.float64)}
    # expr_rank: exactly what the cell sentence encodes — which genes, in which rank order
    R = np.zeros_like(X, dtype=np.float32)
    for i in range(X.shape[0]):
        nz = np.nonzero(X[i] > 0)[0]
        order = nz[np.argsort(-X[i][nz], kind="stable")][:max_genes]
        R[i, order] = 1.0 / (1.0 + np.arange(len(order)))
    out["expr_rank"] = R.astype(np.float64)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5ad", default="/Volumes/Crucial X6/MacBook/Code/biomi_automation/projects/"
                                      "biotensor/data/cellcycle/k562_substrate_for_state.h5ad")
    ap.add_argument("--n-bins", type=int, default=12)
    ap.add_argument("--max-genes", type=int, default=512)
    ap.add_argument("--out", default="results/input_null.json")
    a = ap.parse_args()
    import anndata
    ad = anndata.read_h5ad(a.h5ad)
    theta, s, g, info = phase_angle(ad)
    print(f"phase from {info['n_cc_genes']} cc genes", flush=True)
    reps = build_inputs(ad, a.max_genes)
    res = {}
    for name, H in reps.items():
        # PCA-compress very wide inputs so ridge is comparable to the 2304-d model space
        if H.shape[1] > 2304:
            Hc = H - H.mean(0)
            U, S_, _ = np.linalg.svd(Hc, full_matrices=False)
            H = (U * S_)[:, :2304]
        dec = decodability(H, theta, "cyclic")
        K, centers, b, n = knots(H, theta, "cyclic", a.n_bins)
        geo = ring_geometry(K, centers)
        res[name] = dict(decodability=dec, ring=geo, d=int(H.shape[1]))
        print(f"  {name:<10} circ_corr={dec['value']:+.3f} err={dec['median_abs_err_deg']:.0f}deg "
              f"| planar={geo['planar_frac']:.2f} radius_cv={geo['inplane_radius_cv']:.2f} "
              f"monotone={geo['phase_monotone']}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print(f"[done] -> {a.out}", flush=True)


if __name__ == "__main__":
    main()
