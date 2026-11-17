"""Regulatory-logic test, GPU part.

Input is a design file made by make_designs.py: groups of cells (controls, knockdowns of transcription
factors and of non-TF genes, pseudo-targets made of control cells) with their cell sentences.

(1) Per-cell SAE feature activations at several probe layers, one forward pass per cell. A per-cell
    vector is the mean SAE feature activation over the cell's gene rows (as in
    code/phase8_perturbation_v2.py).
(2) In-silico knockdown: delete a transcription factor's name from control-cell sentences (the 513th
    gene is appended, so the sentence keeps 512 genes and the prompt is unchanged) and measure how far
    every later gene's representation moves (cosine distance per probe layer), against the deletion of
    a neighbouring gene that is neither a TF nor a known target of that TF.

Statistics are computed afterwards on CPU (rev_phase8_stats.py).

Usage:
  python rev_phase8_extract.py --code-dir CODE --design phase8_k562.json --run runs/2b_k562 \
      --model vandijklab/C2S-Scale-Gemma-2-2B --trrust ont/trrust_human.tsv --layers 8 16 22 \
      --insilico --out out/phase8_k562 [--sae-subdir sae] [--max-cells-per-group 0]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--code-dir", required=True)
    p.add_argument("--design", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--sae-subdir", default="sae")
    p.add_argument("--model", required=True)
    p.add_argument("--trrust", required=True)
    p.add_argument("--layers", type=int, nargs="+", default=[16])
    p.add_argument("--max-genes", type=int, default=512)
    p.add_argument("--max-cells-per-group", type=int, default=0, help="smoke-test limit")
    p.add_argument("--max-groups", type=int, default=0, help="smoke-test limit")
    p.add_argument("--insilico", action="store_true")
    p.add_argument("--insilico-cells", type=int, default=40)
    p.add_argument("--insilico-min-cells", type=int, default=15)
    p.add_argument("--out", required=True)
    return p.parse_args()


def load_trrust(path):
    tfs, targets = set(), {}
    with open(path) as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            tf, tgt = parts[0].upper(), parts[1].upper()
            tfs.add(tf)
            targets.setdefault(tf, set()).add(tgt)
    return tfs, targets


def reconstruction_summary(n, sum_x, sum_x2, sum_error, sum_error2, firing_counts):
    """Match the original variance metric while reporting MSE and whole-group firing separately."""
    if n < 2:
        raise ValueError('at least two gene positions are required for reconstruction metrics')
    arrays = [np.asarray(a, np.float64) for a in (sum_x, sum_x2, sum_error, sum_error2, firing_counts)]
    if not all(np.isfinite(a).all() for a in arrays):
        raise ValueError('nonfinite reconstruction/firing statistics')
    sx, sx2, se, se2, counts = arrays
    total = np.maximum(sx2 - sx*sx/n, 0).sum()
    residual = np.maximum(se2 - se*se/n, 0).sum()
    return {'n_gene_positions': n, 'variance_explained': float(1 - residual/max(total, 1e-12)),
            'mse': float(se2.sum() / (n * len(sx))),
            'dead_features': int((counts == 0).sum()),
            'dead_fraction': float((counts == 0).mean()),
            'mean_active_per_position': float(counts.sum()/n)}


def main():
    args = parse_args()
    sys.path.insert(0, args.code_dir)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cell_sentences import attribute_tokens_to_genes, build_cell_sentence
    from sae_topk import TopKSAE

    os.makedirs(args.out, exist_ok=True)
    device = "cuda"
    trrust_tfs, trrust_targets = load_trrust(args.trrust)
    with open(args.design) as f:
        design = json.load(f)
    groups = design["groups"]
    if args.max_groups:
        keep = [g for g in groups if g["kind"] == "control"] + \
               [g for g in groups if g["kind"] != "control"][: args.max_groups]
        groups = keep
    MG = args.max_genes

    tok = AutoTokenizer.from_pretrained(args.model)
    max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    saes, mus = {}, {}
    sdir = os.path.join(args.run, args.sae_subdir)
    for L in args.layers:
        saes[L] = TopKSAE.load(os.path.join(sdir, f"sae_L{L:02d}.pt"), device=device).eval()
        mus[L] = torch.tensor(np.load(os.path.join(sdir, f"mu_L{L:02d}.npy")).astype(np.float32)
                              .reshape(1, -1), device=device)
        if not bool(torch.isfinite(mus[L]).all()):
            raise ValueError(f'nonfinite centering mean at layer {L}')
    nf = saes[args.layers[0]].n_features

    def pooled_vectors(genes):
        """dict layer -> (n_present_genes, d) float32 on GPU; number of genes that got tokens."""
        cs = build_cell_sentence(genes, prompt=True)
        enc = tok(cs.text, return_offsets_mapping=True, return_tensors="pt", truncation=True,
                  max_length=max_len)
        t2g = attribute_tokens_to_genes(enc.pop("offset_mapping")[0].tolist(), cs.gene_spans)
        with torch.no_grad():
            out = model(**{k: v.to(device) for k, v in enc.items()}, output_hidden_states=True)
        valid = np.where(t2g >= 0)[0]
        present_g = sorted(set(int(g) for g in t2g[valid]))
        remap = {g: i for i, g in enumerate(present_g)}
        gidx = torch.tensor([remap[int(g)] for g in t2g[valid]], device=device)
        vpos = torch.tensor(valid, device=device)
        cnt = torch.zeros(len(present_g), device=device).index_add_(
            0, gidx, torch.ones(len(valid), device=device))
        res = {}
        for L in args.layers:
            H = out.hidden_states[L + 1][0].float()
            res[L] = torch.zeros(len(present_g), H.shape[1], device=device).index_add_(
                0, gidx, H[vpos]) / cnt[:, None]
            if not bool(torch.isfinite(res[L]).all()):
                raise ValueError(f'nonfinite pooled activations at layer {L}')
        if present_g != list(range(len(genes))):
            raise ValueError('tokenisation dropped genes; exact cell-sentence identity is required')
        return res, len(present_g)

    # ---------------- (1) per-cell feature matrices ----------------
    rows = []
    feats = {L: [] for L in args.layers}
    group_reconstruction, group_frequencies = [], {}
    t0 = time.time()
    for group_index, g in enumerate(groups):
        accumulators = {}
        sents = g["sentences"]
        cells = g["cells"]
        if args.max_cells_per_group:
            sents, cells = sents[: args.max_cells_per_group], cells[: args.max_cells_per_group]
        if len(sents) != len(cells):
            raise ValueError(f'group {g["group"]} has unequal cell/sentence lengths')
        if len(set(cells)) != len(cells):
            raise ValueError(f'group {g["group"]} contains repeated cells')
        for ci, s in zip(cells, sents):
            genes = s[:MG]
            if not genes:
                continue
            vec, _ = pooled_vectors(genes)
            for L in args.layers:
                with torch.no_grad():
                    x = vec[L] - mus[L]
                    reconstructed, h, _ = saes[L](x)
                    if not bool(torch.isfinite(h).all() and torch.isfinite(reconstructed).all()):
                        raise ValueError(f'nonfinite SAE output in cell {ci}, layer {L}')
                    if L not in accumulators:
                        accumulators[L] = {'n': 0, 'sum_x': torch.zeros(x.shape[1], dtype=torch.float64, device=device),
                                           'sum_x2': torch.zeros(x.shape[1], dtype=torch.float64, device=device),
                                           'sum_error': torch.zeros(x.shape[1], dtype=torch.float64, device=device),
                                           'sum_error2': torch.zeros(x.shape[1], dtype=torch.float64, device=device),
                                           'count': torch.zeros(h.shape[1], dtype=torch.float64, device=device)}
                    a = accumulators[L]
                    xd, error = x.double(), (x - reconstructed).double()
                    a['n'] += len(x)
                    a['sum_x'] += xd.sum(0)
                    a['sum_x2'] += (xd*xd).sum(0)
                    a['sum_error'] += error.sum(0)
                    a['sum_error2'] += (error*error).sum(0)
                    a['count'] += (h > 0).double().sum(0)
                feats[L].append(h.mean(0).cpu().numpy().astype(np.float32))
            rows.append({"group": g["group"], "kind": g["kind"], "cell_index": int(ci), "n_genes": len(genes)})
            if len(rows) % 500 == 0:
                print(f"  {len(rows)} cells | {time.time() - t0:.0f}s", flush=True)
        for L, a in accumulators.items():
            counts = a['count'].cpu().numpy()
            record = reconstruction_summary(a['n'], *[a[key].cpu().numpy() for key in
                                             ('sum_x', 'sum_x2', 'sum_error', 'sum_error2')], counts)
            key = f'G{group_index:03d}_L{L:02d}'
            group_reconstruction.append({'group': g['group'], 'kind': g['kind'], 'layer': L,
                                         'frequency_key': key, **record})
            group_frequencies[key] = (counts / a['n']).astype(np.float32)
    for L in args.layers:
        np.savez_compressed(os.path.join(args.out, f"feat_L{L:02d}.npz"), X=np.stack(feats[L]))
    with open(os.path.join(args.out, "rows.json"), "w") as f:
        json.dump(rows, f)
    meta = {"design": os.path.basename(args.design), "source": design.get("source"), "layers": args.layers,
            "n_features": nf, "n_cells": len(rows), "sae_dir": sdir,
            "seconds_features": round(time.time() - t0, 1), "gpu": torch.cuda.get_device_name(0),
            "reconstruction_by_group_layer": group_reconstruction,
            "feature_frequency_file": "feature_frequencies_by_group.npz"}
    np.savez_compressed(os.path.join(args.out, 'feature_frequencies_by_group.npz'), **group_frequencies)
    with open(os.path.join(args.out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"feature matrices done: {len(rows)} cells in {time.time() - t0:.0f}s", flush=True)

    # ---------------- (2) in-silico knockdown ----------------
    if not args.insilico:
        return
    t1 = time.time()
    ctrl = [g for g in groups if g["kind"] == "control"][0]
    long_sents = [s for s in ctrl["sentences"] if len(s) >= MG + 1]
    if args.max_cells_per_group:
        long_sents = long_sents[: args.max_cells_per_group]
    occ = {}
    for s in long_sents:
        for name in s[: MG - 6]:
            if name.upper() in trrust_tfs:
                occ[name.upper()] = occ.get(name.upper(), 0) + 1
    tf_list = sorted(t for t, v in occ.items() if v >= args.insilico_min_cells)
    quota = {t: args.insilico_cells for t in tf_list}
    gene_univ = sorted({x for s in long_sents for x in s[: MG + 1]})
    gix = {g: i for i, g in enumerate(gene_univ)}
    tix = {t: i for i, t in enumerate(tf_list)}
    nT, nG, nL = len(tf_list), len(gene_univ), len(args.layers)
    s_tf = np.zeros((nT, nG, nL))
    s_ct = np.zeros((nT, nG, nL))
    s_d2 = np.zeros((nT, nG, nL))
    cnt = np.zeros((nT, nG))
    s_dist = np.zeros((nT, nG))
    n_cells_tf = np.zeros(nT, int)
    r_sum = np.zeros(nT)
    pre_noise = []
    print(f"in-silico: {nT} TFs named in >= {args.insilico_min_cells} control sentences | "
          f"{len(long_sents)} control cells", flush=True)

    def cosdist(a, b):
        return 1.0 - torch.nn.functional.cosine_similarity(a, b, dim=1)

    n_pairs = 0
    for s in long_sents:
        todo = [(name.upper(), r) for r, name in enumerate(s[: MG - 6])
                if name.upper() in quota and quota[name.upper()] > 0]
        if not todo:
            continue
        base, n_pres = pooled_vectors(s[:MG])
        if n_pres != MG:
            continue
        for tf, r in todo:
            known = trrust_targets.get(tf, set())
            rc = None
            for cand in (r + 1, r + 2, r + 3, r + 4):
                nm = s[cand].upper()
                if nm not in known and nm not in trrust_tfs:
                    rc = cand
                    break
            if rc is None:
                continue
            v_del, n1 = pooled_vectors(s[:r] + s[r + 1: MG + 1])
            v_ctl, n2 = pooled_vectors(s[:rc] + s[rc + 1: MG + 1])
            if n1 != MG or n2 != MG:
                continue
            lo = rc + 1                                   # first compared gene (index in the original sentence)
            idx = torch.arange(lo, MG, device=device)
            gi = np.array([gix[x] for x in s[lo:MG]])
            ti = tix[tf]
            for li, L in enumerate(args.layers):
                d_tf = cosdist(base[L][idx], v_del[L][idx - 1]).double().cpu().numpy()
                d_ct = cosdist(base[L][idx], v_ctl[L][idx - 1]).double().cpu().numpy()
                s_tf[ti, gi, li] += d_tf
                s_ct[ti, gi, li] += d_ct
                s_d2[ti, gi, li] += (d_tf - d_ct) ** 2
                if li == 0 and r > 5:
                    pre = torch.arange(0, r, device=device)
                    pre_noise.append(float(cosdist(base[L][pre], v_del[L][pre]).max()))
            cnt[ti, gi] += 1
            s_dist[ti, gi] += np.arange(lo, MG) - r
            n_cells_tf[ti] += 1
            r_sum[ti] += r
            quota[tf] -= 1
            n_pairs += 1
            if n_pairs % 500 == 0:
                print(f"  {n_pairs} (TF, cell) pairs | {time.time() - t1:.0f}s", flush=True)
    np.savez_compressed(os.path.join(args.out, "insilico.npz"), sum_tf=s_tf, sum_ctrl=s_ct, sum_d2=s_d2,
                        count=cnt, sum_rank_distance=s_dist, n_cells=n_cells_tf,
                        mean_rank=r_sum / np.maximum(n_cells_tf, 1))
    with open(os.path.join(args.out, "insilico_meta.json"), "w") as f:
        json.dump({"tfs": tf_list, "genes": gene_univ, "layers": args.layers, "n_pairs": int(n_pairs),
                   "max_pre_deletion_cosdist_mean": float(np.mean(pre_noise)) if pre_noise else None,
                   "max_pre_deletion_cosdist_max": float(np.max(pre_noise)) if pre_noise else None,
                   "seconds": round(time.time() - t1, 1)}, f)
    print(f"in-silico done: {n_pairs} pairs in {time.time() - t1:.0f}s | max change before the deleted "
          f"gene: {np.mean(pre_noise) if pre_noise else float('nan'):.2e}", flush=True)


if __name__ == "__main__":
    main()
