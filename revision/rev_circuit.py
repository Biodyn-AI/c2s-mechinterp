"""Causal feature-to-feature circuit tracing (same algorithm as code/circuit_tracing.py, GPU-side pooling).

For every cell: one clean forward pass; for every source feature that is active in the cell, the
feature is removed from the source layer's residual stream (SAE encode -> zero the feature -> decode;
the per-gene difference is added to all sub-word tokens of the genes where the feature was active) and
one ablated forward pass is run. For every downstream layer the change in every SAE feature is
measured, averaged over the genes where the source feature was active: one observation per cell.
Observations are accumulated online (Welford) separately for even- and odd-numbered cells, so that
effect sizes can be compared between two independent halves.

Statistics per (source feature, downstream feature): Cohen's d = mean / sd over cells,
consistency = share of cells with the majority sign. An edge is called when |d| > 0.5 and
consistency > 0.7 (thresholds of the original pipeline).

Differences from code/circuit_tracing.py: mean-pooling of sub-word tokens is done on the GPU, cells are
read from a JSON file of sentences, source features come from an explicit list, and the full statistic
vectors are saved (the original kept only the 50 largest effects per feature and layer).

Usage:
  python rev_circuit.py --code-dir CODE --run runs/2b_k562 --model vandijklab/C2S-Scale-Gemma-2-2B \
      --cells circuit_k562.json --features features.json --source-layers 0 8 16 --n-cells 200 \
      --out out/circuit
  features.json: {"0": {"annotated": [idx,...], "random": [idx,...]}, "8": {...}, ...}
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

D_THR, C_THR = 0.5, 0.7


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--code-dir", required=True)
    p.add_argument("--run", required=True)
    p.add_argument("--sae-subdir", default="sae")
    p.add_argument("--model", required=True)
    p.add_argument("--cells", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--source-layers", type=int, nargs="+", required=True)
    p.add_argument("--n-cells", type=int, default=200)
    p.add_argument("--n-model-layers", type=int, default=26)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, args.code_dir)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cell_sentences import attribute_tokens_to_genes, build_cell_sentence
    from sae_topk import TopKSAE

    os.makedirs(args.out, exist_ok=True)
    dev = torch.device("cuda")
    with open(args.cells) as f:
        sents = json.load(f)["sentences"][: args.n_cells]
    with open(args.features) as f:
        feat_sel = json.load(f)

    tok = AutoTokenizer.from_pretrained(args.model)
    max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                 device_map="auto").eval()
    sdir = os.path.join(args.run, args.sae_subdir)
    saes, mus = {}, {}

    def get_sae(L):
        if L not in saes:
            saes[L] = TopKSAE.load(os.path.join(sdir, f"sae_L{L:02d}.pt"), device=dev).eval()
            mus[L] = torch.tensor(np.load(os.path.join(sdir, f"mu_L{L:02d}.npy")).astype(np.float32)
                                  .reshape(1, -1), device=dev)
        return saes[L], mus[L]

    # tokenise once
    cells = []
    for genes in sents:
        cs = build_cell_sentence(genes, prompt=True)
        enc = tok(cs.text, return_offsets_mapping=True, return_tensors="pt", truncation=True,
                  max_length=max_len)
        t2g = attribute_tokens_to_genes(enc.pop("offset_mapping")[0].tolist(), cs.gene_spans)
        valid = np.where(t2g >= 0)[0]
        present = sorted(set(int(g) for g in t2g[valid]))
        remap = {g: i for i, g in enumerate(present)}
        cells.append((enc["input_ids"], enc["attention_mask"],
                      torch.tensor([remap[int(g)] for g in t2g[valid]]), torch.tensor(valid), len(present)))
    print(f"{len(cells)} cells tokenised", flush=True)

    for sl in args.source_layers:
        sel = feat_sel[str(sl)]
        feats = list(sel["annotated"]) + list(sel.get("random", []))
        kinds = ["annotated"] * len(sel["annotated"]) + ["random"] * len(sel.get("random", []))
        dls = list(range(sl + 1, args.n_model_layers))
        src_sae, src_mu = get_sae(sl)
        for dl in dls:
            get_sae(dl)
        nf = src_sae.n_features
        F, ND = len(feats), len(dls)
        # accumulators per half: [half][F, ND, nf]
        acc_mean = torch.zeros(2, F, ND, nf, dtype=torch.float64, device=dev)
        acc_m2 = torch.zeros(2, F, ND, nf, dtype=torch.float64, device=dev)
        acc_pos = torch.zeros(2, F, ND, nf, dtype=torch.float32, device=dev)
        acc_n = torch.zeros(2, F, dtype=torch.int64, device=dev)
        t0 = time.time()
        n_clean = n_abl = 0
        for ci, (ids, am, gidx, vpos, ng) in enumerate(cells):
            half = ci % 2
            ids, am, gidx, vpos = ids.to(dev), am.to(dev), gidx.to(dev), vpos.to(dev)
            cnt = torch.zeros(ng, device=dev).index_add_(0, gidx, torch.ones(len(gidx), device=dev))

            def pool(H):
                return torch.zeros(ng, H.shape[1], device=dev).index_add_(0, gidx, H[vpos]) / cnt[:, None]

            with torch.no_grad():
                out = model(input_ids=ids, attention_mask=am, output_hidden_states=True)
                n_clean += 1
                clean_full = out.hidden_states[sl + 1][0].float()
                h_sparse, topk_idx = src_sae.encode(pool(clean_full) - src_mu)
                clean_ds = [saes[dl].encode(pool(out.hidden_states[dl + 1][0].float()) - mus[dl])[0]
                            for dl in dls]
                recon_normal = src_sae.decode(h_sparse)
            del out
            for fpos, fi in enumerate(feats):
                active = h_sparse[:, fi] > 0
                if not bool(active.any()):
                    continue
                with torch.no_grad():
                    h_abl = h_sparse.clone()
                    h_abl[:, fi] = 0.0
                    delta = src_sae.decode(h_abl) - recon_normal                # (ng, d)
                    modified = clean_full.clone()
                    tmask = active[gidx]
                    modified[vpos[tmask]] += delta[gidx[tmask]]
                    fired = [False]

                    def hook(module, inp, output, modified=modified, fired=fired):
                        if fired[0]:
                            return output
                        fired[0] = True
                        o = list(output)
                        o[0] = modified.unsqueeze(0).to(device=output[0].device, dtype=output[0].dtype)
                        return tuple(o)

                    hnd = model.model.layers[sl].register_forward_hook(hook)
                    out_a = model(input_ids=ids, attention_mask=am, output_hidden_states=True)
                    hnd.remove()
                    n_abl += 1
                    rows = torch.where(active)[0]
                    acc_n[half, fpos] += 1
                    n = acc_n[half, fpos].double()
                    for di, dl in enumerate(dls):
                        abl = saes[dl].encode(pool(out_a.hidden_states[dl + 1][0].float()) - mus[dl])[0]
                        x = (abl[rows] - clean_ds[di][rows]).double().mean(0)
                        d1 = x - acc_mean[half, fpos, di]
                        acc_mean[half, fpos, di] += d1 / n
                        acc_m2[half, fpos, di] += d1 * (x - acc_mean[half, fpos, di])
                        acc_pos[half, fpos, di] += (x > 0).float()
                    del out_a
            if (ci + 1) % 10 == 0:
                el = time.time() - t0
                print(f"  L{sl} cell {ci + 1}/{len(cells)} | {n_clean} clean + {n_abl} ablated passes | "
                      f"{el:.0f}s | ETA {el / (ci + 1) * (len(cells) - ci - 1) / 60:.1f} min", flush=True)

        # ---- finalise: per half and pooled ----
        def finalise(mean, m2, pos, n):
            n_ = n.double().clamp(min=2)[:, None, None]
            sd = torch.sqrt(m2 / (n_ - 1))
            d = mean / sd.clamp(min=1e-10)
            cons = torch.maximum(pos.double(), n_ - pos.double()) / n_
            ok = (n >= 2)[:, None, None]
            return torch.where(ok, d, torch.zeros_like(d)), torch.where(ok, cons, torch.zeros_like(cons))

        n0, n1 = acc_n[0].double(), acc_n[1].double()
        nt = (n0 + n1).clamp(min=1)
        mean_all = (n0[:, None, None] * acc_mean[0] + n1[:, None, None] * acc_mean[1]) / nt[:, None, None]
        m2_all = acc_m2[0] + acc_m2[1] + (n0 * n1 / nt)[:, None, None] * (acc_mean[0] - acc_mean[1]) ** 2
        pos_all = acc_pos[0] + acc_pos[1]
        d_all, c_all = finalise(mean_all, m2_all, pos_all, acc_n[0] + acc_n[1])
        d_h0, c_h0 = finalise(acc_mean[0], acc_m2[0], acc_pos[0], acc_n[0])
        d_h1, c_h1 = finalise(acc_mean[1], acc_m2[1], acc_pos[1], acc_n[1])
        sig = (d_all.abs() > D_THR) & (c_all > C_THR)
        f32 = lambda t: t.cpu().numpy().astype(np.float32)  # noqa: E731
        np.savez_compressed(
            os.path.join(args.out, f"circuit_L{sl:02d}.npz"),
            features=np.array(feats), kinds=np.array(kinds), downstream_layers=np.array(dls),
            n_cells_active=(acc_n[0] + acc_n[1]).cpu().numpy(), n_half=acc_n.cpu().numpy(),
            edge_mask=sig.cpu().numpy(),
            d_all=f32(d_all), cons_all=f32(c_all), mean_all=mean_all.cpu().numpy().astype(np.float32),
            d_half0=f32(d_h0), d_half1=f32(d_h1), cons_half0=f32(c_h0), cons_half1=f32(c_h1))
        summ = {"source_layer": sl, "n_cells": len(cells), "features": feats, "kinds": kinds,
                "downstream_layers": dls, "n_clean_passes": n_clean, "n_ablated_passes": n_abl,
                "seconds": round(time.time() - t0, 1), "d_threshold": D_THR, "consistency_threshold": C_THR,
                "n_significant_total": int(sig.sum()),
                "n_significant_by_downstream_layer": [int(x) for x in sig.sum(dim=(0, 2)).cpu().numpy()],
                "n_significant_by_feature": [int(x) for x in sig.sum(dim=(1, 2)).cpu().numpy()],
                "gpu": torch.cuda.get_device_name(0)}
        with open(os.path.join(args.out, f"circuit_L{sl:02d}_summary.json"), "w") as f:
            json.dump(summ, f, indent=2)
        print(f"source layer {sl}: {summ['n_significant_total']} significant effects | "
              f"{n_clean} clean + {n_abl} ablated passes | {summ['seconds'] / 60:.1f} min", flush=True)
        del acc_mean, acc_m2, acc_pos
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
