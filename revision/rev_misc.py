"""Small helpers for the revision runs.

  select-features  choose circuit-tracing source features at a layer: the N annotated features with
                   the smallest enrichment p-value among features that fire on >= 1% of positions and
                   have >= 10 top genes, plus M random features under the same firing/gene filters.
  lines-eval       apply stored SAEs to activations of cells from several cell lines: per-line
                   reconstruction and per-line firing frequency of every feature.
  gene-means       mean activation vector per gene (genes with >= min occurrences) for given layers.
  pool-eval        compare first-token / last-token pooling with mean pooling at one layer.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def cmd_select(a):
    out = {}
    rng = np.random.RandomState(a.seed)
    for L in a.layers:
        cat = json.load(open(os.path.join(a.run, f"feature_catalog_L{L:02d}.json")))["features"]
        ann = json.load(open(os.path.join(a.ann_dir, f"ann_v2_L{L:02d}.json")))["features"]
        ann = {f["feature_idx"]: f for f in ann}
        ok = [f for f in cat if not f.get("is_dead") and len(f.get("top_genes", [])) >= 10
              and f.get("activation_freq", 0) >= 0.01]
        cand = sorted([f for f in ok if ann.get(f["feature_idx"], {}).get("annotated")],
                      key=lambda f: (ann[f["feature_idx"]]["min_p"], f["feature_idx"]))
        chosen = [f["feature_idx"] for f in cand[: a.n_annotated]]
        rest = sorted(f["feature_idx"] for f in ok if f["feature_idx"] not in set(chosen))
        rnd = [int(x) for x in rng.choice(rest, min(a.n_random, len(rest)), replace=False)]
        out[str(L)] = {"annotated": chosen, "random": rnd, "n_eligible": len(ok), "n_eligible_annotated": len(cand),
                       "labels": {str(i): next((t["term"] for t in (ann[i].get("terms") or [])
                                                 if t.get("source") != "STRING"),
                                                (ann[i].get("terms") or [{}])[0].get("term", ""))
                                  for i in chosen}}
        print(f"L{L}: eligible {len(ok)} (annotated {len(cand)}) -> {len(chosen)} annotated + {len(rnd)} random")
    json.dump(out, open(a.out, "w"), indent=1)


def cmd_lines(a):
    import torch
    sys.path.insert(0, a.code_dir)
    from sae_topk import TopKSAE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    cells = json.load(open(os.path.join(a.acts, "cells.json")))
    row_cell = np.load(os.path.join(a.acts, "row_cell_ids.npy"))
    line_of = np.array([c["line"] for c in cells])[row_cell]
    lines = sorted(set(line_of))
    res = {"lines": lines, "layers": {}}
    freqs = {}
    for L in a.layers:
        X = np.load(os.path.join(a.acts, f"layer_{L:02d}_activations.npy")).astype(np.float32)
        sae = TopKSAE.load(os.path.join(a.sae_run, a.sae_subdir, f"sae_L{L:02d}.pt"), device=dev).eval()
        mu = np.load(os.path.join(a.sae_run, a.sae_subdir, f"mu_L{L:02d}.npy")).astype(np.float32).reshape(1, -1)
        rec = {}
        for ln in lines:
            m = line_of == ln
            x = torch.tensor(X[m] - mu, device=dev)
            with torch.no_grad():
                xh = torch.zeros_like(x)
                cnt = torch.zeros(sae.n_features, device=dev)
                for s in range(0, len(x), 20000):
                    o, h, _ = sae(x[s:s + 20000])
                    xh[s:s + 20000] = o
                    cnt += (h > 0).float().sum(0)
                tot = x.var(0).sum().item()
                ve = 1 - (x - xh).var(0).sum().item() / max(tot, 1e-10)
            f = (cnt / len(x)).cpu().numpy()
            freqs[f"L{L:02d}_{ln}"] = f.astype(np.float32)
            rec[ln] = {"n_rows": int(m.sum()), "variance_explained": ve,
                       "share_features_firing": float((f > 0).mean()),
                       "mu_shift_rel": float(np.linalg.norm(X[m].mean(0) - mu[0]) / np.linalg.norm(mu[0]))}
            print(f"L{L} {ln}: var_expl {ve:.4f} | features firing {100 * (f > 0).mean():.1f}%", flush=True)
        res["layers"][str(L)] = rec
    json.dump(res, open(a.out + ".json", "w"), indent=1)
    np.savez_compressed(a.out + "_freq.npz", **freqs)


def cmd_gene_means(a):
    names = np.array(json.load(open(os.path.join(a.acts, "row_gene_names.json"))))
    u, inv, cnt = np.unique(names, return_inverse=True, return_counts=True)
    keep = cnt >= a.min_occ
    out = {"genes": u[keep], "counts": cnt[keep]}
    for L in a.layers:
        X = np.load(os.path.join(a.acts, f"layer_{L:02d}_activations.npy"), mmap_mode="r")
        s = np.zeros((len(u), X.shape[1]), np.float64)
        for st in range(0, len(names), 50000):
            np.add.at(s, inv[st:st + 50000], np.asarray(X[st:st + 50000], dtype=np.float64))
        out[f"L{L:02d}"] = (s[keep] / cnt[keep][:, None]).astype(np.float32)
        print(f"L{L}: {int(keep.sum())} genes with >= {a.min_occ} occurrences", flush=True)
    np.savez_compressed(a.out, **out)


def cmd_pool(a):
    import torch
    sys.path.insert(0, a.code_dir)
    from sae_topk import TopKSAE
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    L = a.layer
    M = np.load(os.path.join(a.mean_run, f"layer_{L:02d}_activations.npy")).astype(np.float32)
    names = np.array(json.load(open(os.path.join(a.mean_run, "row_gene_names.json"))))
    sae = TopKSAE.load(os.path.join(a.sae_run, "sae", f"sae_L{L:02d}.pt"), device=dev).eval()
    mu = np.load(os.path.join(a.sae_run, "sae", f"mu_L{L:02d}.npy")).astype(np.float32).reshape(1, -1)
    u, inv, cnt = np.unique(names, return_inverse=True, return_counts=True)
    rng = np.random.RandomState(0)
    sub = rng.choice(np.where(cnt >= 20)[0], min(600, int((cnt >= 20).sum())), replace=False)

    def gmeans(X):
        s = np.zeros((len(u), X.shape[1]), np.float64)
        np.add.at(s, inv, X.astype(np.float64))
        g = (s / cnt[:, None])[sub]
        g = g - g.mean(0)
        g /= np.linalg.norm(g, axis=1, keepdims=True) + 1e-12
        return g @ g.T

    def var_expl(X):
        x = torch.tensor(X - mu, device=dev)
        with torch.no_grad():
            num = 0.0
            xh = torch.zeros_like(x)
            for s in range(0, len(x), 20000):
                xh[s:s + 20000] = sae(x[s:s + 20000])[0]
            tot = x.var(0).sum().item()
            num = (x - xh).var(0).sum().item()
        return 1 - num / max(tot, 1e-10)

    Sm = gmeans(M)
    iu = np.triu_indices(len(sub), 1)
    res = {"layer": L, "mean": {"sae_variance_explained": var_expl(M)}}
    for kind, run in (("first", a.first_run), ("last", a.last_run)):
        X = np.load(os.path.join(run, f"layer_{L:02d}_activations.npy")).astype(np.float32)
        cos = (X * M).sum(1) / (np.linalg.norm(X, axis=1) * np.linalg.norm(M, axis=1) + 1e-12)
        S = gmeans(X)
        res[kind] = {"row_cosine_with_mean_pool_mean": float(cos.mean()),
                     "row_cosine_with_mean_pool_p05": float(np.quantile(cos, 0.05)),
                     "gene_similarity_matrix_correlation": float(np.corrcoef(S[iu], Sm[iu])[0, 1]),
                     "sae_variance_explained": var_expl(X)}
        print(kind, json.dumps(res[kind]), flush=True)
    json.dump(res, open(a.out, "w"), indent=1)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("select-features")
    s.add_argument("--run", required=True)
    s.add_argument("--ann-dir", required=True)
    s.add_argument("--layers", type=int, nargs="+", required=True)
    s.add_argument("--n-annotated", type=int, default=30)
    s.add_argument("--n-random", type=int, default=15)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--out", required=True)
    l = sub.add_parser("lines-eval")
    l.add_argument("--code-dir", required=True)
    l.add_argument("--acts", required=True)
    l.add_argument("--sae-run", required=True)
    l.add_argument("--sae-subdir", default="sae")
    l.add_argument("--layers", type=int, nargs="+", required=True)
    l.add_argument("--out", required=True)
    g = sub.add_parser("gene-means")
    g.add_argument("--acts", required=True)
    g.add_argument("--layers", type=int, nargs="+", required=True)
    g.add_argument("--min-occ", type=int, default=20)
    g.add_argument("--out", required=True)
    q = sub.add_parser("pool-eval")
    q.add_argument("--code-dir", required=True)
    q.add_argument("--layer", type=int, required=True)
    q.add_argument("--mean-run", required=True)
    q.add_argument("--first-run", required=True)
    q.add_argument("--last-run", required=True)
    q.add_argument("--sae-run", required=True)
    q.add_argument("--out", required=True)
    a = p.parse_args()
    {"select-features": cmd_select, "lines-eval": cmd_lines, "gene-means": cmd_gene_means,
     "pool-eval": cmd_pool}[a.cmd](a)


if __name__ == "__main__":
    main()
