"""
Phase 7 adapter for C2S-Scale: SAE feature co-activation graph + Leiden modules.

Adapted from repos/bio-sae/src/07_feature_coactivation.py (pinned 9277f5d). Same logic
and parameters (PMI > 2.0, min co-activation 50, Leiden RBConfiguration resolution 1.0
seed 42, modules keep communities with >= 3 features); only I/O is changed to the
C2S-Scale atlas run-dir contract.

Two deliberate departures from the reference, both forced by the contract:
  * The reference built a DENSE (n_features, n_features) int matrix. Here activations can
    be millions of rows, so we accumulate co-activation counts in scipy.sparse via a
    per-chunk binary matrix product A.T @ A (A is chunk x n_features, one entry per active
    feature). No dense n_features^2 matrix is ever materialized.
  * The reference annotated modules from feature_annotations.json / feature_catalog.json,
    which are not part of the C2S-Scale contract. That section is dropped; we emit only the
    module assignment / size / edge / coverage summary the atlas needs.

A feature is "active" at a position iff it is nonzero in the TopK code (post-ReLU) — i.e.
in topk_indices with a positive value.

Usage:
  python coactivation.py --run runs/2b_atlas --layer 13
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from scipy import sparse

SAE_CODE_DIR = "/Volumes/Crucial X6/MacBook/Code/biomi_automation/projects/c2s-scale/code"

# Reference parameters (repos/bio-sae/src/07_feature_coactivation.py) — kept faithfully.
PMI_THRESHOLD = 2.0       # retain feature-feature edges with PMI > threshold
MIN_COACTIVATION = 50     # minimum raw co-activation count for a candidate edge
LEIDEN_RESOLUTION = 1.0   # RBConfigurationVertexPartition resolution parameter
LEIDEN_SEED = 42
MIN_MODULE_SIZE = 3       # communities smaller than this are not reported as modules
CHUNK = 10000             # stream activations in chunks; never load all onto GPU


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


def parse_args():
    p = argparse.ArgumentParser(description="C2S-Scale Phase-7 SAE feature co-activation graph")
    p.add_argument("--run", required=True, help="atlas run dir (holds layer_*_activations.npy, sae/)")
    p.add_argument("--layer", type=int, required=True)
    return p.parse_args()


def main():
    import torch

    sys.path.insert(0, SAE_CODE_DIR)
    from sae_topk import TopKSAE

    args = parse_args()
    L = args.layer
    run = args.run
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 64)
    print(f"C2S-Scale Phase 7: feature co-activation | layer {L:02d} | {run}")
    print("=" * 64)

    # ------------------------------------------------------------------ load SAE + mean
    sae = TopKSAE.load(os.path.join(run, "sae", f"sae_L{L:02d}.pt"), device=device)
    sae.eval()
    n_features, k = sae.n_features, sae.k
    mu = np.load(os.path.join(run, "sae", f"mu_L{L:02d}.npy")).astype(np.float32)  # (1, d_model)
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    print(f"  SAE: n_features={n_features}, k={k}, d_model={sae.d_model}, device={device}")

    # ------------------------------------------------------------------ stream activations
    act_path = os.path.join(run, f"layer_{L:02d}_activations.npy")
    activations = np.lib.format.open_memmap(act_path, mode="r")
    n_total = int(activations.shape[0])
    print(f"  activations: {n_total:,} positions x {activations.shape[1]}")

    feature_count = np.zeros(n_features, dtype=np.int64)          # P(i) numerator
    coact = sparse.csr_matrix((n_features, n_features), dtype=np.int64)  # symmetric, incl. diagonal

    t0 = time.time()
    n_done = 0
    for start in range(0, n_total, CHUNK):
        end = min(start + CHUNK, n_total)
        x = torch.tensor(np.asarray(activations[start:end]), dtype=torch.float32, device=device)
        x = x - mu_t
        with torch.no_grad():
            h_sparse, topk_idx = sae.encode(x)          # h_sparse: (b, n_feat); topk_idx: (b, k)
            vals = torch.gather(h_sparse, 1, topk_idx)  # (b, k) TopK values (post-ReLU)
            active = vals > 0                           # "active" == nonzero in TopK
        b = end - start
        row_ids = torch.arange(b, device=device).unsqueeze(1).expand(-1, k)
        rr = row_ids[active].cpu().numpy()
        cc = topk_idx[active].cpu().numpy()

        np.add.at(feature_count, cc, 1)
        # binary position x feature matrix for this chunk; A.T @ A -> co-activation counts
        A = sparse.csr_matrix(
            (np.ones(cc.shape[0], dtype=np.int64), (rr, cc)), shape=(b, n_features))
        coact = coact + (A.T @ A).tocsr()

        n_done += b
        if (start // CHUNK) % 50 == 0:
            print(f"    {n_done:>12,}/{n_total:,} | nnz(coact)={coact.nnz:,} "
                  f"| {time.time() - t0:.1f}s")

    print(f"  encoding done: {time.time() - t0:.1f}s | nnz(coact)={coact.nnz:,}")

    # ------------------------------------------------------------------ PMI edges (upper triangle)
    tri = sparse.triu(coact, k=1).tocoo()
    ii, jj, cnt = tri.row.astype(np.int64), tri.col.astype(np.int64), tri.data.astype(np.float64)
    keep = cnt >= MIN_COACTIVATION
    ii, jj, cnt = ii[keep], jj[keep], cnt[keep]
    p_ij = cnt / n_total
    p_i = feature_count[ii].astype(np.float64) / n_total
    p_j = feature_count[jj].astype(np.float64) / n_total
    valid = (p_i > 0) & (p_j > 0)
    ii, jj, cnt = ii[valid], jj[valid], cnt[valid]
    pmi = np.log2((cnt / n_total) / (p_i[valid] * p_j[valid]))
    edge_keep = pmi > PMI_THRESHOLD
    pmi_edges = [(int(ii[m]), int(jj[m]), float(pmi[m]), int(cnt[m]))
                 for m in np.nonzero(edge_keep)[0]]
    print(f"  PMI edges (> {PMI_THRESHOLD}): {len(pmi_edges):,}")

    del coact, tri

    # ------------------------------------------------------------------ Leiden clustering
    modules = []
    assignment = {}
    if len(pmi_edges) == 0:
        print("  WARNING: no PMI edges; no modules.")
    else:
        import igraph as ig
        import leidenalg

        node_list = sorted({n for e in pmi_edges for n in e[:2]})
        node_to_idx = {n: idx for idx, n in enumerate(node_list)}
        g = ig.Graph(n=len(node_list), directed=False)
        g.add_edges([(node_to_idx[i], node_to_idx[j]) for i, j, _, _ in pmi_edges])
        g.es["weight"] = [p for _, _, p, _ in pmi_edges]
        print(f"  graph: {g.vcount()} nodes, {g.ecount()} edges")

        partition = leidenalg.find_partition(
            g, leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=LEIDEN_RESOLUTION,
            n_iterations=-1,
            seed=LEIDEN_SEED,
        )
        labels = np.array(partition.membership)
        print(f"  Leiden communities: {len(partition)}")

        for comm_id in range(len(partition)):
            member_local = np.where(labels == comm_id)[0]
            if len(member_local) < MIN_MODULE_SIZE:
                continue
            members = sorted(node_list[i] for i in member_local)
            member_set = set(members)
            internal = [p for i, j, p, _ in pmi_edges if i in member_set and j in member_set]
            modules.append({
                "module_id": len(modules),
                "n_features": len(members),
                "features": members,
                "n_internal_edges": len(internal),
                "mean_pmi": float(np.mean(internal)) if internal else 0.0,
            })
        modules.sort(key=lambda m: -m["n_features"])
        # reindex + build per-feature assignment after sort
        for new_id, m in enumerate(modules):
            m["module_id"] = new_id
            for fi in m["features"]:
                assignment[str(fi)] = new_id
        print(f"  modules (>= {MIN_MODULE_SIZE} features): {len(modules)}")

    # ------------------------------------------------------------------ summary + save
    n_alive = int(np.sum(feature_count > 0))
    n_in_modules = sum(m["n_features"] for m in modules)
    coverage = n_in_modules / max(n_alive, 1)

    output = {
        "layer": L,
        "config": {
            "pmi_threshold": PMI_THRESHOLD,
            "min_coactivation": MIN_COACTIVATION,
            "leiden_resolution": LEIDEN_RESOLUTION,
            "leiden_seed": LEIDEN_SEED,
            "min_module_size": MIN_MODULE_SIZE,
            "chunk": CHUNK,
            "k": k,
        },
        "summary": {
            "n_positions": n_total,
            "n_features": n_features,
            "n_alive_features": n_alive,
            "n_pmi_edges": len(pmi_edges),
            "n_modules": len(modules),
            "n_features_in_modules": n_in_modules,
            "module_coverage": coverage,
        },
        "module_assignment": assignment,   # feature_idx (str) -> module_id, for features in a module
        "modules": modules,                # each: module_id, n_features, features, n_internal_edges, mean_pmi
        "top_pmi_edges": sorted(pmi_edges, key=lambda e: -e[2])[:100],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    out_path = os.path.join(run, f"coactivation_modules_L{L:02d}.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)

    print("-" * 64)
    print(f"  alive features:       {n_alive}/{n_features}")
    print(f"  PMI edges:            {len(pmi_edges)}")
    print(f"  modules:              {len(modules)}")
    print(f"  features in modules:  {n_in_modules} ({coverage * 100:.1f}% of alive)")
    if modules:
        sizes = [m["n_features"] for m in modules]
        print(f"  module sizes:         min={min(sizes)} median={int(np.median(sizes))} "
              f"max={max(sizes)}")
    print(f"  saved: {out_path}")


if __name__ == "__main__":
    main()
