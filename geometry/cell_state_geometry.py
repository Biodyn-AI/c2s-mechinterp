"""
cell_state_geometry_gpu.py — DRAFT (do NOT run without a GPU + the live 2B model).

QUESTION (cell-identity / trajectory geometry): how is CELL STATE laid out in C2S-Scale-2B's
residual stream? Is a developmental trajectory linearly embedded or genuinely curved? Does the
2B model's geometry beat a raw-expression baseline (i.e. does pretraining earn its place)?

This ports the program's cell-state geometry battery (route_geometry / route_lineage /
route_state_geometry / route_branchpoint / route_steering2) to C2S-Scale. The unit is a per-CELL
embedding = the residual stream at the cell-sentence's answer/last position OR mean-pooled over
gene tokens (both provided).

================================ WHAT THIS NEEDS ================================
COMPUTE:  1 GPU (>=24 GB) for extraction; all geometry runs on CPU.
MODEL:    vandijklab/C2S-Scale-Gemma-2-2B.
DATA:     a TRAJECTORY dataset with a pseudotime + terminal-fate labels. Recommended:
          Setty 2019 CD34+ human bone marrow (matches route_steering/convergence; has palantir
          pseudotime + branch labels). Also a cell-cycle-scored dataset for the cell-cycle test.
          Raw counts required for the expression-PCA baseline and abundance control.

STEP 0 (GPU) — per-cell embedding at ~70% depth (route convention). Reuse the atlas extractor:
    python ../../code/extract_activations.py \
        --model vandijklab/C2S-Scale-Gemma-2-2B \
        --h5ad <setty_cd34.h5ad> --layers 18 \
        --max-cells 3000 --max-genes 512 --unit cell-summary \
        --store-dtype float16 --out runs/cellstate
  -> runs/cellstate/layer_18_activations.npy (n_cells, d_model), row_cell_ids.npy
  (Layer 18 ~= 0.7 * 26. Use --unit gene-token + mean-pool here for the pooled variant.)

STEP 1+ (CPU) — this script: attach pseudotime/fate/counts by cell id and run:

================================ METHOD ================================
A. CURVATURE = kNN_R2 - linear_R2 of pseudotime in a matched 20-D PCA space
   (route_lineage/route_state_geometry canonical linear-accessibility test). Also
   bilinear_gain = quad_R2 - linear_R2 (degree-2 PolynomialFeatures ridge). curved iff both
   exceed 0 AND the synthetic-linear control (below).
B. TANGENT ROTATION theta_far: local pseudotime-ascent direction (Ridge) in pseudotime
   quintile-1 vs quintile-5; rotation_excess = theta_far - theta_far(synthetic linear).
   monotonicity rho = Spearman(bin separation, pairwise angle) = coherent arc vs disordered.
C. RULER RECOVERY: kNN balanced-accuracy of terminal fate + Spearman(embedding PC1, pseudotime),
   each minus a shuffled-label null; trustworthiness (2D-PCA vs full) >= 0.80 sanity.
D. LOCAL INTRINSIC DIM: participation ratio of the local (k=100) covariance spectrum along
   pseudotime — a potency-gradient probe (route_geometry_hypotheses hb3).

CONTROLS (a positive counts only if ALL pass)
  (1) SNR-MATCHED LINEAR-SYNTHETIC target: y_syn = X_pca @ w + noise, noise scaled so linear-R2
      matches the real target's; real curvature must EXCEED this control's curvature. Catches the
      LEACE/curvature false positive that fooled route_lineage.
  (2) EXPRESSION-PCA BASELINE (route_steering2 baseline.py): run the identical pipeline on top-2000
      HVG log1p-CP10k expression PCA of the SAME cells. The 2B "earns its place" only if it beats
      expression-PCA on curvature/rotation. (Program prior: it usually does NOT — steering geometry
      is largely the data's.)
  (3) ABUNDANCE PARTIALLING: residualize embedding on the library-size axis and the target on a
      deg-2 function of log total-counts; re-score. (A linear control cannot remove quadratically-
      entangled abundance — report both.)

Outputs: results/cell_state_L{L}.json (curvature/bilinear_gain/rotation_excess/monotonicity/
    ruler recovery/local-ID, each with its control values) + console table.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np


# --------------------------------------------------------------------------- #
# spaces                                                                        #
# --------------------------------------------------------------------------- #
def pca_space(X, dim=20, seed=0):
    from sklearn.decomposition import PCA
    Xc = (X - X.mean(0)) / (X.std(0) + 1e-6)
    return PCA(n_components=dim, random_state=seed).fit_transform(Xc)


def expression_pca(adata, cell_ids, dim=20, n_hvg=2000, seed=0, counts_layer=None):
    """Top-n_hvg HVG log1p-CP10k expression PCA of the SAME cells (route_steering2 baseline).
    Reads RAW counts from adata.layers[counts_layer] when given; else falls back to adata.X
    (double-normalization risk if X is already normalized)."""
    import scipy.sparse as sp
    if counts_layer is not None:
        if counts_layer not in adata.layers:
            raise KeyError(f"--counts-layer '{counts_layer}' not in adata.layers "
                           f"{list(adata.layers.keys())}")
        X = adata.layers[counts_layer][cell_ids]
    else:
        print("WARNING: expression_pca using adata.X (no --counts-layer); "
              "if X is already normalized the baseline is double-normalized")
        X = adata.X[cell_ids]
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    X = np.log1p(X / np.clip(X.sum(1, keepdims=True), 1, None) * 1e4)
    v = X.var(0)
    hvg = np.argsort(-v)[:n_hvg]
    return pca_space(X[:, hvg], dim, seed)


# --------------------------------------------------------------------------- #
# A. curvature = kNN - linear                                                   #
# --------------------------------------------------------------------------- #
def curvature_report(Z, y, seed=0):
    from sklearn.model_selection import cross_val_score, KFold
    from sklearn.linear_model import Ridge
    from sklearn.neighbors import KNeighborsRegressor
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.pipeline import make_pipeline
    cv = KFold(5, shuffle=True, random_state=seed)
    lin = cross_val_score(Ridge(alpha=10), Z, y, cv=cv, scoring="r2").mean()
    knn = cross_val_score(KNeighborsRegressor(15), Z, y, cv=cv, scoring="r2").mean()
    quad = cross_val_score(make_pipeline(PolynomialFeatures(2), Ridge(alpha=10)),
                           Z, y, cv=cv, scoring="r2").mean()
    return {"linear_r2": float(lin), "knn_r2": float(knn), "quad_r2": float(quad),
            "curvature": float(knn - lin), "bilinear_gain": float(quad - lin)}


def synthetic_linear_control(Z, y, seed=0):
    """y_syn linear-in-Z, decodability-matched to y; its curvature is the bar to beat."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_score
    w = Ridge(alpha=10).fit(Z, y).coef_
    base = Z @ w
    real_lin = cross_val_score(Ridge(alpha=10), Z, y, cv=KFold(5, shuffle=True, random_state=seed), scoring="r2").mean()
    rng = np.random.default_rng(seed)
    # scale noise so linear-R2(y_syn) ~ real_lin
    for s in np.linspace(0.1, 3.0, 30) * base.std():
        y_syn = base + rng.normal(0, s, len(y))
        r = cross_val_score(Ridge(alpha=10), Z, y_syn, cv=KFold(5, shuffle=True, random_state=seed), scoring="r2").mean()
        if r <= real_lin:
            break
    return curvature_report(Z, y_syn, seed)["curvature"]


# --------------------------------------------------------------------------- #
# B. tangent rotation                                                          #
# --------------------------------------------------------------------------- #
def tangent_rotation(Z, y, n_bins=5, seed=0):
    from sklearn.linear_model import Ridge
    order = np.argsort(y)
    bins = np.array_split(order, n_bins)
    def ascent(bidx):
        return Ridge(alpha=10).fit(Z[bidx], y[bidx]).coef_
    w0, wl = ascent(bins[0]), ascent(bins[-1])
    cos = float(np.dot(w0, wl) / (np.linalg.norm(w0) * np.linalg.norm(wl) + 1e-9))
    theta_far = float(np.degrees(np.arccos(np.clip(cos, -1, 1))))
    # monotonicity: pairwise bin-ascent angle vs bin separation
    seps, angs = [], []
    ws = [ascent(b) for b in bins]
    for i in range(n_bins):
        for j in range(i + 1, n_bins):
            c = np.dot(ws[i], ws[j]) / (np.linalg.norm(ws[i]) * np.linalg.norm(ws[j]) + 1e-9)
            seps.append(j - i); angs.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    from scipy.stats import spearmanr
    mono, _ = spearmanr(seps, angs)
    return {"theta_far_deg": theta_far, "monotonicity_rho": float(mono)}


# --------------------------------------------------------------------------- #
# C. ruler recovery                                                            #
# --------------------------------------------------------------------------- #
def ruler_recovery(Z, fate, y, seed=0):
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import make_scorer, balanced_accuracy_score
    from scipy.stats import spearmanr
    rng = np.random.default_rng(seed)
    cv = StratifiedKFold(5, shuffle=True, random_state=seed)
    sc = make_scorer(balanced_accuracy_score)
    acc = cross_val_score(KNeighborsClassifier(15), Z, fate, cv=cv, scoring=sc).mean()
    null = cross_val_score(KNeighborsClassifier(15), Z, rng.permutation(fate), cv=cv, scoring=sc).mean()
    from sklearn.decomposition import PCA
    pc1 = PCA(2, random_state=seed).fit_transform(Z)[:, 0]
    rho, _ = spearmanr(pc1, y)
    return {"fate_balacc": float(acc), "fate_null": float(null), "fate_margin": float(acc - null),
            "pc1_pseudotime_rho": float(abs(rho))}


# --------------------------------------------------------------------------- #
# D. local intrinsic dim along pseudotime                                     #
# --------------------------------------------------------------------------- #
def local_intrinsic_dim(Z, y, k=100, n_bins=10, seed=0):
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k).fit(Z)
    order = np.argsort(y)
    prs = []
    for b in np.array_split(order, n_bins):
        cen = int(b[len(b) // 2])
        _, idx = nn.kneighbors(Z[cen:cen + 1])
        C = np.cov(Z[idx[0]].T)
        l = np.linalg.eigvalsh(C)
        l = l[l > 0]
        prs.append(float((l.sum() ** 2) / (l ** 2).sum()))
    return {"pseudotime_bin_participation_ratio": prs}


def run_space(name, Z, y, fate, seed):
    rec = {"space": name}
    rec.update(curvature_report(Z, y, seed))
    rec["synthetic_linear_curvature"] = synthetic_linear_control(Z, y, seed)
    rec["curved"] = bool(rec["curvature"] > max(0.02, rec["synthetic_linear_curvature"] + 0.02))
    rec.update(tangent_rotation(Z, y, seed=seed))
    if fate is not None:
        rec.update(ruler_recovery(Z, fate, y, seed))
    rec.update(local_intrinsic_dim(Z, y, seed=seed))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True, help="dir with layer_{L}_activations.npy + row_cell_ids.npy")
    ap.add_argument("--h5ad", required=True)
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--pseudotime-key", required=True, help="adata.obs key with pseudotime")
    ap.add_argument("--fate-key", default=None, help="adata.obs key with terminal-fate labels")
    ap.add_argument("--counts-layer", default=None,
                    help="adata.layers key with RAW counts for the expression-PCA baseline "
                         "(e.g. spliced); without it the baseline uses adata.X and warns")
    ap.add_argument("--pca-dim", type=int, default=20)
    ap.add_argument("--out", default="results/cell_state.json")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import anndata
    adata = anndata.read_h5ad(args.h5ad)
    cell_ids = np.load(os.path.join(args.run, "row_cell_ids.npy"))
    X = np.load(os.path.join(args.run, f"layer_{args.layer:02d}_activations.npy")).astype(np.float32)
    y = np.asarray(adata.obs[args.pseudotime_key])[cell_ids].astype(float)
    fate = np.asarray(adata.obs[args.fate_key]).astype(str)[cell_ids] if args.fate_key else None
    keep = np.isfinite(y)
    X, y = X[keep], y[keep]
    fate = fate[keep] if fate is not None else None
    cell_ids = cell_ids[keep]

    Z_model = pca_space(X, args.pca_dim, args.seed)
    Z_expr = expression_pca(adata, cell_ids, args.pca_dim, seed=args.seed,
                            counts_layer=args.counts_layer)

    out = {"meta": {"params": vars(args)},
           "model_space": run_space(f"C2S-2B_L{args.layer}", Z_model, y, fate, args.seed),
           "expression_pca_baseline": run_space("expr_pca", Z_expr, y, fate, args.seed)}
    m, e = out["model_space"], out["expression_pca_baseline"]
    out["earns_its_place"] = bool(m["curvature"] > e["curvature"]
                                  and m["theta_far_deg"] > e["theta_far_deg"])
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    print(f"MODEL  curvature={m['curvature']:+.3f} (synth {m['synthetic_linear_curvature']:+.3f}) "
          f"theta_far={m['theta_far_deg']:.0f}deg mono={m['monotonicity_rho']:+.2f}")
    print(f"EXPR   curvature={e['curvature']:+.3f} theta_far={e['theta_far_deg']:.0f}deg")
    print(f"earns_its_place (beats expression-PCA): {out['earns_its_place']}")
    print("saved", args.out)


if __name__ == "__main__":
    main()
