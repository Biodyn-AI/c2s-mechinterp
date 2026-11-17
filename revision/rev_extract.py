"""Gene-token activation extraction from stored cell sentences.

Same sentence construction, tokenisation and token -> gene attribution as code/extract_activations.py
(one cell per forward pass, bf16 model, hidden_states[L + 1] = output of block L). Differences:
  * sentences are read back from an atlas row file pair (row_gene_names.json + row_cell_ids.npy), so the
    model input is exactly the one the atlas autoencoders were trained on;
  * pooling over a gene's sub-word tokens is selectable: mean (atlas default), first token, last token;
  * optional per-layer moment accumulation (mean, second-moment matrix) for residual-stream statistics;
  * token statistics, wall time and peak GPU memory are written to the manifest.

Usage:
  python rev_extract.py --code-dir /workspace/rev1/code --model vandijklab/C2S-Scale-Gemma-2-2B \
      --rows-dir /workspace/rev1/atlas_rows --layers 0 1 2 --pool mean --store-dtype float16 \
      --out /workspace/rev1/runs/2b_atlas [--max-cells 1200] \
      [--stats-layers all --stats-cells 200 --stats-out resid_stats_2b.npz] [--no-store]
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
    p.add_argument("--model", required=True)
    p.add_argument("--rows-dir", required=True)
    p.add_argument("--layers", nargs="*", default=[])
    p.add_argument("--pool", choices=["mean", "first", "last"], default="mean")
    p.add_argument("--store-dtype", choices=["float16", "float32"], default="float16")
    p.add_argument("--out", required=True)
    p.add_argument("--max-cells", type=int, default=0)
    p.add_argument("--no-store", action="store_true", help="only accumulate statistics")
    p.add_argument("--stats-layers", nargs="*", default=[])
    p.add_argument("--stats-cells", type=int, default=0)
    p.add_argument("--stats-out", default="")
    return p.parse_args()


def load_sentences(rows_dir):
    with open(os.path.join(rows_dir, "row_gene_names.json")) as f:
        names = json.load(f)
    cells = np.load(os.path.join(rows_dir, "row_cell_ids.npy"))
    if len(names) != len(cells) or not len(cells):
        raise ValueError('stored gene rows and cell IDs must be nonempty and have equal lengths')
    sents, ids = [], []
    start = 0
    for i in range(1, len(cells) + 1):
        if i == len(cells) or cells[i] != cells[start]:
            sents.append(names[start:i])
            ids.append(int(cells[start]))
            start = i
    if len(set(ids)) != len(ids):
        raise ValueError('each cell ID must occupy one contiguous block of stored rows')
    return sents, ids


def storage_array(values, dtype):
    """Fail before writing corrupted/overflowing values; float32 is required for massive activations."""
    if not np.isfinite(values).all():
        raise ValueError('nonfinite pooled activations; extraction cannot continue')
    if np.max(np.abs(values), initial=0) > np.finfo(dtype).max:
        raise ValueError(f'pooled activation exceeds {np.dtype(dtype)} range; rerun with --store-dtype float32')
    return values.astype(dtype)


def main():
    args = parse_args()
    sys.path.insert(0, args.code_dir)
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cell_sentences import attribute_tokens_to_genes, build_cell_sentence

    os.makedirs(args.out, exist_ok=True)
    store_dtype = np.float16 if args.store_dtype == "float16" else np.float32
    sents, cell_ids = load_sentences(args.rows_dir)
    if args.max_cells:
        sents, cell_ids = sents[: args.max_cells], cell_ids[: args.max_cells]
    print(f"{len(sents)} sentences | genes per sentence: min {min(map(len, sents))} max {max(map(len, sents))}",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    assert tok.is_fast
    max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)
    t_load = time.time()
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model = model.eval()
    d_model = model.config.hidden_size
    n_layers = model.config.num_hidden_layers
    dev = model.device
    print(f"model loaded in {time.time() - t_load:.0f}s | d_model {d_model} | layers {n_layers}", flush=True)

    layers = [int(x) for x in args.layers]
    stats_layers = (list(range(n_layers)) if args.stats_layers == ["all"]
                    else [int(x) for x in args.stats_layers])
    if not set(layers + stats_layers) <= set(range(n_layers)):
        raise ValueError('requested layer is outside the model')
    if not args.no_store and not layers:
        raise ValueError('at least one storage layer is required unless --no-store is set')

    # ---- pass 1 (tokeniser only): row bookkeeping + token statistics ----
    enc_cache, row_gene_names, row_cell_ids = [], [], []
    tok_per_gene_occ, sent_len, n_trunc = [], [], 0
    sym_tok = {}
    for ci, genes in enumerate(sents):
        cs = build_cell_sentence(genes, organism="human", prompt=True)
        enc = tok(cs.text, return_offsets_mapping=True, return_tensors="pt", truncation=True,
                  max_length=max_len)
        t2g = attribute_tokens_to_genes(enc["offset_mapping"][0].tolist(), cs.gene_spans)
        present = [g for g in range(len(genes)) if (t2g == g).any()]
        if len(present) < len(genes):
            n_trunc += 1
            raise ValueError(f'sentence {ci} lost genes during tokenisation; exact stored-row identity is required')
        for g in present:
            c = int((t2g == g).sum())
            tok_per_gene_occ.append(c)
            sym_tok[genes[g]] = c
            row_gene_names.append(genes[g])
            row_cell_ids.append(cell_ids[ci])
        sent_len.append(int(enc["input_ids"].shape[1]))
        enc_cache.append((enc["input_ids"], enc["attention_mask"], t2g, present))
    n_rows = len(row_gene_names)
    tpg = np.array(tok_per_gene_occ)
    token_stats = {
        "n_gene_occurrences": int(n_rows), "n_unique_symbols": len(sym_tok),
        "tokens_per_gene_occurrence_mean": float(tpg.mean()),
        "tokens_per_gene_occurrence_median": float(np.median(tpg)),
        "tokens_per_gene_occurrence_max": int(tpg.max()),
        "share_single_token_occurrences": float((tpg == 1).mean()),
        "tokens_per_unique_symbol_mean": float(np.mean(list(sym_tok.values()))),
        "share_single_token_symbols": float(np.mean([v == 1 for v in sym_tok.values()])),
        "tokens_per_symbol_hist": {str(k): int(v) for k, v in
                                   zip(*np.unique(list(sym_tok.values()), return_counts=True))},
        "sentence_tokens_mean": float(np.mean(sent_len)), "sentence_tokens_min": int(np.min(sent_len)),
        "sentence_tokens_max": int(np.max(sent_len)), "n_truncated_sentences": int(n_trunc),
        "tokenizer_vocab_size": int(len(tok)),
    }
    print("token stats:", json.dumps({k: v for k, v in token_stats.items() if k != "tokens_per_symbol_hist"}),
          flush=True)

    mm = {}
    if not args.no_store:
        for L in layers:
            mm[L] = np.lib.format.open_memmap(os.path.join(args.out, f"layer_{L:02d}_activations.npy"),
                                              mode="w+", dtype=store_dtype, shape=(n_rows, d_model))
    acc = {}
    for L in stats_layers:
        acc[L] = {"s": torch.zeros(d_model, dtype=torch.float64, device=dev),
                  "xx": torch.zeros(d_model, d_model, dtype=torch.float64, device=dev),
                  "n": 0, "norm": 0.0, "maxabs": 0.0, "cell_median_abs": []}

    need = sorted(set(layers) | set(stats_layers))
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    off = 0
    for ci, (ids, am, t2g, present) in enumerate(enc_cache):
        with torch.no_grad():
            out = model(input_ids=ids.to(dev), attention_mask=am.to(dev), output_hidden_states=True)
        valid = np.where(t2g >= 0)[0]
        remap = {g: i for i, g in enumerate(present)}
        gidx = torch.tensor([remap[int(g)] for g in t2g[valid]], device=dev)
        vpos = torch.tensor(valid, device=dev)
        ng = len(present)
        if args.pool == "mean":
            cnt = torch.zeros(ng, device=dev).index_add_(0, gidx, torch.ones(len(valid), device=dev))
        else:
            pos = {}
            for p_, g in zip(valid.tolist(), t2g[valid].tolist()):
                if args.pool == "first":
                    pos.setdefault(g, p_)
                else:
                    pos[g] = p_
            sel = torch.tensor([pos[g] for g in present], device=dev)
        for L in need:
            H = out.hidden_states[L + 1][0].float()
            if args.pool == "mean":
                pooled = torch.zeros(ng, d_model, device=dev).index_add_(0, gidx, H[vpos]) / cnt[:, None]
            else:
                pooled = H[sel]
            if not bool(torch.isfinite(pooled).all()):
                raise ValueError(f'nonfinite pooled activations in cell {ci}, layer {L}')
            if L in mm:
                x = pooled.cpu().numpy()
                mm[L][off:off + ng] = storage_array(x, store_dtype)
            if L in acc and (args.stats_cells == 0 or ci < args.stats_cells):
                a = acc[L]
                xd = pooled.double()
                a["s"] += xd.sum(0)
                a["xx"] += xd.T @ xd
                a["n"] += ng
                a["norm"] += float(pooled.norm(dim=1).sum())
                a["maxabs"] = max(a["maxabs"], float(pooled.abs().max()))
                a["cell_median_abs"].append(float(pooled.abs().median()))
        off += ng
        del out
        if (ci + 1) % 100 == 0:
            el = time.time() - t0
            print(f"  {ci + 1}/{len(enc_cache)} cells | {el:.0f}s | {el / (ci + 1):.3f} s/cell", flush=True)
        if args.no_store and args.stats_cells and ci + 1 >= args.stats_cells:
            break
    wall = time.time() - t0
    for L in mm:
        mm[L].flush()
    if not args.no_store:
        assert off == n_rows, (off, n_rows)
        np.save(os.path.join(args.out, "row_cell_ids.npy"), np.asarray(row_cell_ids, dtype=np.int64))
        with open(os.path.join(args.out, "row_gene_names.json"), "w") as f:
            json.dump(row_gene_names, f)

    if acc and args.stats_out:
        res = {}
        for L, a in acc.items():
            n = a["n"]
            mean = a["s"] / n
            cov = a["xx"] / n - torch.outer(mean, mean)
            ev = torch.linalg.eigvalsh(cov).flip(0).clamp(min=0).cpu().numpy()
            res[f"eig_L{L:02d}"] = ev.astype(np.float64)
            res[f"rms_L{L:02d}"] = torch.sqrt(torch.diag(a["xx"]) / n).cpu().numpy()
            res[f"mean_L{L:02d}"] = mean.cpu().numpy()
            # Fourth entry is the median of per-cell medians, not the pooled scalar median.
            res[f"scalars_L{L:02d}"] = np.array([n, a["norm"] / n, a["maxabs"],
                                                 np.median(a["cell_median_abs"])])
        np.savez_compressed(args.stats_out, **res)
        print("wrote", args.stats_out, flush=True)

    manifest = {"model": args.model, "model_commit": getattr(model.config, "_commit_hash", None),
                "unit": "gene-token", "pool": args.pool, "layers": layers,
                "stats_layers": stats_layers, "max_genes": 512, "prompt": True,
                "store_dtype": args.store_dtype, "n_cells": len(enc_cache), "n_rows": int(n_rows),
                "d_model": int(d_model), "n_model_layers": int(n_layers),
                "wall_seconds_forward": round(wall, 1),
                "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2),
                "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                "token_stats": token_stats}
    name = "manifest.json" if not args.no_store else "manifest_stats.json"
    with open(os.path.join(args.out, name), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"done: {n_rows} rows x {len(layers)} layers in {wall:.0f}s | peak GPU "
          f"{manifest['peak_gpu_mem_gb']} GB", flush=True)


if __name__ == "__main__":
    main()
