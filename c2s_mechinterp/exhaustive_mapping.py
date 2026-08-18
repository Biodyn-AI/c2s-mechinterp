"""
Stage 3 (adapted for C2S-Scale-Gemma-2-2B): exhaustive circuit mapping, higher-order
combinatorial ablation, and trajectory-guided feature steering — at ONE source layer.

Reference (Geneformer encoder, 1 token == 1 gene):
  repos/sae-biological-map/src/exhaustive_feature_tracing.py   (Experiment 1)
  repos/sae-biological-map/src/higher_order_ablation.py        (Experiment 2)
  repos/sae-biological-map/src/trajectory_steering.py          (Experiment 3)
Spec: pipelines/sparse-autoencoders/03-exhaustive-mapping-and-steering.md

Three analyses, all sharing the SAME Gemma-2 residual-stream ablation/steering hook:
  (a) EXHAUSTIVE tracing — ablate EVERY active source feature (activation_freq >= 0.001,
      not is_dead) at --layer and measure Cohen's d on every downstream SAE feature at a
      few downstream layers. Same math as Stage-2 circuit tracing, but over the whole active
      dictionary instead of only the annotated features.
  (b) THREE-WAY COMBINATORIAL ablation — for a few feature triplets at --layer, run the 7
      ablation conditions {A, B, C, AB, AC, BC, ABC} measured at a downstream layer, then
      compute the redundancy ratio R_ABC = |d_ABC| / (|d_A|+|d_B|+|d_C|) and the higher-order
      interaction I_ABC = d_ABC - d_AB - d_AC - d_BC + d_A + d_B + d_C.
  (c) TRAJECTORY-GUIDED STEERING — amplify a feature by alpha via the activation-addition
      formula  h' = h + (alpha-1) * a_f * d_f  (a_f = per-position SAE activation coeff,
      d_f = unit decoder direction W_dec[:, f]) and propagate; measure a state shift.

============================ C2S-Scale format contract ============================
MODEL: vandijklab/C2S-Scale-Gemma-2-2B — Gemma2ForCausalLM, d_model=2304, 26 layers, bf16.
  hidden_states[L+1] is the POST-block residual stream of transformer layer L
  (hidden_states[0] = embedding). The SAE at layer L was trained (extract_activations.py)
  on hidden_states[L+1] mean-pooled over each gene's subword tokens ("gene-token" unit).

RESIDUAL-STREAM HOOK: register a forward hook on model.model.layers[L] (the decoder block).
  Its forward returns a tuple; output[0] is the (1, seq, d) hidden state == hidden_states[L+1].
  The hook replaces output[0] with the edited hidden (cast to the block-output dtype/device),
  so every downstream layer sees the edit. (Geneformer's reference hooked
  model.bert.encoder.layer[L]; ours is model.model.layers[L].)

GENE POSITIONS: built with cell_sentences.py. build_cell_sentence(genes, prompt=True) gives
  .text + .gene_spans; tok(text, return_offsets_mapping=True, ...) then
  attribute_tokens_to_genes(offsets, gene_spans) -> per-token gene index or SCAFFOLD(-1).
  GENE POSITIONS = np.where(tok2gene != -1)[0]. Per gene, its token positions are
  np.where(tok2gene == g)[0]. anndata_to_ranked_genes(adata, ci, max_genes) gives the
  ranked gene list. Precompute (input_ids, attention_mask, tok2gene) ONCE per cell (cheap,
  tokenizer only) and reuse for the clean + every ablated/steered pass.

SAE: sae_topk.py TopKSAE. sae = TopKSAE.load(f"{RUN}/sae/sae_L{L:02d}.pt", device); mu =
  np.load(f"{RUN}/sae/mu_L{L:02d}.npy").astype(float32).reshape(1, d). encode/decode operate
  in float32: hidden_states are bf16 -> .float() before the SAE; cast the EDITED hidden back
  to the block dtype (bf16) for the hook. d_SAE=9216 (4x), k=32.

ATLAS OUTPUTS (already produced, in RUN): feature_catalog_L{L:02d}.json
  ({"features":[{feature_idx, is_dead, activation_freq, top_genes:[{gene_name,...}]}]}) and
  feature_annotations_L{L:02d}.json ({"feature_annotations": {"<fid>": [{ontology, term,
  p_adjusted}]}}). Feature *selection* (triplets / steer) scores by annotation richness
  (n_ontologies*10 + n_annotations - log10(min p_adjusted)), requires >=10 top_genes and
  activation_freq>=0.01, skips is_dead. Tracing uses the looser activation_freq>=0.001 filter.

DATA: replogle_k562_subset.h5ad; control cells obs['gene']=='non-targeting'.
Thresholds unchanged from the reference: Cohen's d 0.5, consistency 0.7. N_FEATURES=9216,
N_LAYERS=26.

============================ Streaming / resume design =============================
The reference caches every cell's clean forward pass, then loops features-outer. To honour
"stream one cell at a time (never hold all cells' hidden states)", we instead precompute only
the (cheap) per-cell TOKENIZATION for all cells, then loop cells-OUTER: one clean forward per
cell, then for every intervention on that cell an ablated/steered forward, accumulating
per-feature effect sizes online (Welford). Only one cell's hidden states are alive at a time.
This yields exactly the same forward-pass count and the same Cohen's d as the reference.

Memory note: cells-outer means all traced source features' accumulators live at once. Each
Welford accumulator over 9216 dims is ~221 KB; per source feature x (#downstream layers) that
is ~0.66 MB. Tracing all ~half of 9216 active features x 3 downstream ~= 3 GB (float64). Bound
it with --max-source-features and/or --feature-start/--feature-end sharding (resume-safe like
the reference's feature ranges).

Usage:
  python exhaustive_mapping.py --run runs/2b_atlas --model vandijklab/C2S-Scale-Gemma-2-2B \
      --h5ad data/replogle_k562_subset.h5ad --layer 9 --out runs/2b_atlas/exhaustive_mapping_L09.json

NOT run here (no GPU). Verified to parse only.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from .cell_sentences import (  # noqa: E402
    SCAFFOLD,
    anndata_to_ranked_genes,
    attribute_tokens_to_genes,
    build_cell_sentence,
)
except ImportError:
    from cell_sentences import (  # noqa: E402
    SCAFFOLD,
    anndata_to_ranked_genes,
    attribute_tokens_to_genes,
    build_cell_sentence,
)
try:  # works as `python -m c2s_mechinterp.<mod>` and as a plain script
    from .sae_topk import TopKSAE  # noqa: E402
except ImportError:
    from sae_topk import TopKSAE  # noqa: E402

# ------------------------------------------------------------------ constants
D_MODEL = 2304
D_SAE = 9216          # N_FEATURES (4x expansion)
K_VAL = 32
N_LAYERS = 26

COHENS_D_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD = 0.7
TOP_EFFECTS = 50               # top-N downstream edges stored per downstream layer per feature

ACT_FREQ_TRACE = 0.001         # exhaustive-tracing active-feature filter (Phase 1)
ACT_FREQ_SELECT = 0.01         # triplet / steer feature-selection filter (annotation-rich)
MIN_TOP_GENES = 10             # selection requires >= this many top_genes


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ------------------------------------------------------------------ Welford online effect size
class WelfordAccumulator:
    """Online mean/variance + sign-consistency over per-cell effect vectors (reference-identical)."""

    def __init__(self, n_dims):
        self.n = 0
        self.mean = np.zeros(n_dims, dtype=np.float64)
        self.M2 = np.zeros(n_dims, dtype=np.float64)
        self.pos_count = np.zeros(n_dims, dtype=np.float64)

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
        self.pos_count += (x > 0).astype(np.float64)

    def finalize(self):
        if self.n < 2:
            return np.zeros_like(self.mean), np.zeros_like(self.mean)
        var = self.M2 / (self.n - 1)
        std = np.sqrt(var)
        cohens_d = self.mean / np.maximum(std, 1e-10)
        consistency = np.maximum(self.pos_count, self.n - self.pos_count) / self.n
        return cohens_d, consistency


# ------------------------------------------------------------------ SAE cache
class SAECache:
    """Lazy per-layer TopKSAE + mu loader (mu shape (1, d_model))."""

    def __init__(self, run_dir, device):
        self.run_dir = run_dir
        self.device = device
        self._saes = {}
        self._mus = {}

    def get(self, layer):
        if layer not in self._saes:
            sae = TopKSAE.load(os.path.join(self.run_dir, "sae", f"sae_L{layer:02d}.pt"),
                               device=self.device)
            sae.eval()
            mu = np.load(os.path.join(self.run_dir, "sae", f"mu_L{layer:02d}.npy"))
            mu = mu.astype(np.float32).reshape(1, -1)
            self._saes[layer] = sae
            self._mus[layer] = mu
            print(f"    loaded SAE + mu for L{layer:02d}", flush=True)
        return self._saes[layer], self._mus[layer]

    def decoder_direction(self, layer, fid):
        """Unit decoder direction d_f = W_dec.weight[:, fid] (columns are unit-norm, dim=0)."""
        sae, _ = self.get(layer)
        return sae.W_dec.weight.data[:, fid].detach().float().cpu().numpy()  # (d_model,)


# ------------------------------------------------------------------ Gemma-2 residual hook
def make_residual_hook(modified_hidden_np):
    """Replace decoder-block output[0] (the (1, seq, d) residual stream) with modified_hidden_np
    (seq, d) numpy, cast to the block's own dtype/device. Fires once per forward."""
    fired = [False]

    def hook_fn(module, inputs, output):
        import torch
        if fired[0]:
            return output
        fired[0] = True
        ref = output[0]  # (1, seq, d) — Gemma2DecoderLayer returns a tuple
        mod = torch.from_numpy(np.ascontiguousarray(modified_hidden_np)).to(
            device=ref.device, dtype=ref.dtype).unsqueeze(0)
        out = list(output)
        out[0] = mod
        return tuple(out)

    return hook_fn


def sae_encode_np(sae, mu, gene_rows_np, device):
    """gene_rows_np (n_genes, d) float32 -> (h_sparse (n_genes, d_sae), topk (n_genes, k)) numpy."""
    import torch
    x = gene_rows_np.astype(np.float32) - mu  # center by mu (1, d)
    with torch.no_grad():
        h, topk = sae.encode(torch.tensor(x, dtype=torch.float32, device=device))
    return h.detach().cpu().numpy(), topk.detach().cpu().numpy()


def sae_decode_np(sae, h_np, device):
    """h_np (n_genes, d_sae) float32 -> decode (n_genes, d) numpy (WITHOUT +mu; mu cancels in deltas)."""
    import torch
    with torch.no_grad():
        rec = sae.decode(torch.tensor(h_np, dtype=torch.float32, device=device))
    return rec.detach().cpu().numpy()


def meanpool_gene_rows(hidden_np, pos_list):
    """hidden_np (seq, d); pos_list = list of token-position arrays per present gene ->
    (n_present, d) mean-pooled rows (matches how the atlas SAE was trained)."""
    return np.stack([hidden_np[p].mean(axis=0) for p in pos_list], axis=0)


def apply_gene_delta(source_hidden_np, pos_list, delta_gene):
    """Broadcast each gene's residual-space delta to ALL of that gene's subword-token positions.

    TODO(assumption): the reference had 1 token == 1 gene, so the SAE delta landed on a single
    residual position. C2S-Scale genes span several subword tokens and the SAE is trained on the
    gene MEAN, so we add the gene-level correction uniformly to every subword token of the gene.
    This is the most faithful gene-token adaptation but is an approximation of a per-token edit;
    revisit if per-token attribution is later required.
    """
    modified = source_hidden_np.copy()
    for gi, p in enumerate(pos_list):
        modified[p] += delta_gene[gi]
    return modified


# ------------------------------------------------------------------ atlas catalog / annotations
def load_catalog(run_dir, layer):
    with open(os.path.join(run_dir, f"feature_catalog_L{layer:02d}.json")) as f:
        cat = json.load(f)
    info = {}
    for feat in cat["features"]:
        fi = int(feat["feature_idx"])
        info[fi] = {
            "is_dead": bool(feat.get("is_dead", False)),
            "activation_freq": float(feat.get("activation_freq", 0.0)),
            "n_top_genes": len(feat.get("top_genes", [])),
            "top_genes": [g["gene_name"] for g in feat.get("top_genes", [])],
        }
    return info


def load_annotations(run_dir, layer):
    """Return (labels: fid->best-term, annotated: set(fid), rich: fid->selection-score, term: fid->term)."""
    path = os.path.join(run_dir, f"feature_annotations_L{layer:02d}.json")
    labels, annotated, rich, top_term = {}, set(), {}, {}
    if not os.path.exists(path):
        return labels, annotated, rich, top_term
    with open(path) as f:
        data = json.load(f)
    for fid_str, anns in data.get("feature_annotations", {}).items():
        if not anns:
            continue
        fi = int(fid_str)
        annotated.add(fi)
        # best label: prefer pathway ontologies, else first
        chosen = None
        for a in anns:
            if a.get("ontology") in ("GO_BP", "KEGG", "Reactome"):
                chosen = a["term"]
                break
        labels[fi] = chosen if chosen is not None else anns[0]["term"]
        top_term[fi] = anns[0].get("term", labels[fi])
        # annotation-richness selection score (contract formula)
        onts = {a.get("ontology") for a in anns}
        pmin = min((a.get("p_adjusted", 1.0) for a in anns), default=1.0)
        pmin = max(pmin, 1e-300)
        rich[fi] = len(onts) * 10 + len(anns) - math.log10(pmin)
    return labels, annotated, rich, top_term


# ------------------------------------------------------------------ per-cell tokenization (cheap; held for all cells)
def build_cell_records(adata, cell_indices, tok, max_genes, max_len, use_prompt=True):
    """Precompute ONLY tokenization per cell (no model). Reused for clean + every edited pass.
    Returns list of dicts: input_ids/attention_mask (torch on CPU), present-gene token positions,
    present-gene symbols."""
    records = []
    for ci in cell_indices:
        genes = anndata_to_ranked_genes(adata, int(ci), max_genes=max_genes)
        if not genes:
            continue
        cs = build_cell_sentence(genes, prompt=use_prompt)
        enc = tok(cs.text, return_offsets_mapping=True, return_tensors="pt",
                  truncation=True, max_length=max_len)
        offsets = enc.pop("offset_mapping")[0].tolist()
        tok2gene = attribute_tokens_to_genes(offsets, cs.gene_spans)
        pos_list, gene_names = [], []
        for g in range(len(genes)):
            p = np.where(tok2gene == g)[0]
            if p.size:
                pos_list.append(p)
                gene_names.append(genes[g])
        if not pos_list:
            continue
        records.append({
            "cell_index": int(ci),
            "input_ids": enc["input_ids"],          # (1, seq) long, CPU
            "attention_mask": enc["attention_mask"],
            "pos_list": pos_list,                   # per present-gene token positions
            "gene_names": gene_names,
        })
    return records


def clean_forward(model, record, layers_needed):
    """Run one clean forward; return {layer: hidden_np (seq, d) float32} for layers_needed and
    the logits mean over gene positions is computed by the caller if needed."""
    import torch
    dev = next(model.parameters()).device
    enc = {"input_ids": record["input_ids"].to(dev),
           "attention_mask": record["attention_mask"].to(dev)}
    with torch.no_grad():
        out = model(**enc, output_hidden_states=True)
    hidden = {L: out.hidden_states[L + 1][0].float().cpu().numpy() for L in layers_needed}
    logits = out.logits[0].float().cpu().numpy()  # (seq, vocab)
    del out
    _empty_cache(torch, dev)
    return hidden, logits


def edited_forward(model, record, layer, modified_hidden_np, layers_needed, want_logits=False):
    """Run a forward with the residual hook injecting modified_hidden_np at `layer`.
    Return {layer: hidden_np} for layers_needed (+ logits if requested)."""
    import torch
    dev = next(model.parameters()).device
    enc = {"input_ids": record["input_ids"].to(dev),
           "attention_mask": record["attention_mask"].to(dev)}
    handle = model.model.layers[layer].register_forward_hook(make_residual_hook(modified_hidden_np))
    try:
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True)
    finally:
        handle.remove()
    hidden = {L: out.hidden_states[L + 1][0].float().cpu().numpy() for L in layers_needed}
    logits = out.logits[0].float().cpu().numpy() if want_logits else None
    del out
    _empty_cache(torch, dev)
    return hidden, logits


def _empty_cache(torch, dev):
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    elif dev.type == "mps":
        torch.mps.empty_cache()


# ==================================================================================
# (a) EXHAUSTIVE FEATURE TRACING
# ==================================================================================
def run_exhaustive_tracing(model, records, sae_cache, source_layer, downstream_layers,
                           catalog, labels, feature_ids, save_cb=None, save_every=5):
    """Cells-outer streaming. For every cell, clean-encode the source + downstream; then for each
    active source feature in that cell, run ONE ablated forward and accumulate per-downstream
    Cohen's d online. feature_ids = the source features to trace (already filtered/sharded)."""
    src_sae, src_mu = sae_cache.get(source_layer)
    dst = {}
    for dl in downstream_layers:
        dst[dl] = sae_cache.get(dl)  # (sae, mu)

    trace_set = set(int(f) for f in feature_ids)
    # accumulators[fid][dl] -> WelfordAccumulator(D_SAE); created lazily
    accumulators = {}
    n_active_cells = {}  # fid -> #cells where feature fired

    layers_needed = [source_layer] + list(downstream_layers)
    t0 = time.time()

    for ci, rec in enumerate(records):
        hidden, _ = clean_forward(model, rec, layers_needed)
        src_rows = meanpool_gene_rows(hidden[source_layer], rec["pos_list"])   # (n_genes, d)
        h_sparse, topk = sae_encode_np(src_sae, src_mu, src_rows, sae_cache.device)  # (n_g, d_sae),(n_g,k)
        recon_clean = sae_decode_np(src_sae, h_sparse, sae_cache.device)  # (n_g, d) — reused per feature

        # clean downstream SAE codes (per gene row)
        z_clean = {}
        for dl in downstream_layers:
            dsae, dmu = dst[dl]
            drows = meanpool_gene_rows(hidden[dl], rec["pos_list"])
            zc, _ = sae_encode_np(dsae, dmu, drows, sae_cache.device)
            z_clean[dl] = zc

        # features active somewhere in this cell, restricted to the trace set
        active_here = np.intersect1d(np.unique(topk), np.fromiter(trace_set, dtype=np.int64))
        for fid in active_here:
            fid = int(fid)
            gene_mask = (topk == fid).any(axis=1)          # (n_genes,) genes where feature fired
            active_gi = np.where(gene_mask)[0]
            if active_gi.size == 0:
                continue

            # ablate: zero this feature, decode, subtract cached clean recon (delta only where active)
            h_abl = h_sparse.copy()
            h_abl[:, fid] = 0.0
            delta_gene = sae_decode_np(src_sae, h_abl, sae_cache.device) - recon_clean  # (n_genes, d)

            modified = apply_gene_delta(hidden[source_layer], rec["pos_list"], delta_gene)
            hid_abl, _ = edited_forward(model, rec, source_layer, modified, downstream_layers)

            if fid not in accumulators:
                accumulators[fid] = {dl: WelfordAccumulator(D_SAE) for dl in downstream_layers}
                n_active_cells[fid] = 0
            n_active_cells[fid] += 1

            for dl in downstream_layers:
                dsae, dmu = dst[dl]
                drows = meanpool_gene_rows(hid_abl[dl], rec["pos_list"])
                z_abl, _ = sae_encode_np(dsae, dmu, drows, sae_cache.device)
                # mean delta over the gene rows where the source feature was active
                delta_z = (z_abl[active_gi] - z_clean[dl][active_gi]).astype(np.float64).mean(axis=0)
                accumulators[fid][dl].update(delta_z)

        if (ci + 1) % 5 == 0 or ci == 0:
            print(f"    [trace] cell {ci+1}/{len(records)} | features touched {len(accumulators)} "
                  f"| {time.time()-t0:.0f}s", flush=True)
        if save_cb is not None and (ci + 1) % save_every == 0:
            save_cb(_summarize_tracing(accumulators, n_active_cells, catalog, labels,
                                       source_layer, downstream_layers, len(records),
                                       partial_through=ci + 1))

    return _summarize_tracing(accumulators, n_active_cells, catalog, labels,
                              source_layer, downstream_layers, len(records))


def _summarize_tracing(accumulators, n_active_cells, catalog, labels, source_layer,
                       downstream_layers, n_cells, partial_through=None):
    """Finalize accumulators -> per-feature edge lists + hub / annotation-bias / attenuation summary."""
    features_out = []
    per_dl_totals = {int(dl): 0 for dl in downstream_layers}
    for fid, accs in accumulators.items():
        downstream_edges = {}
        total_sig = 0
        for dl in downstream_layers:
            d_vals, c_vals = accs[dl].finalize()
            sig = (np.abs(d_vals) > COHENS_D_THRESHOLD) & (c_vals > CONSISTENCY_THRESHOLD)
            n_sig = int(sig.sum())
            total_sig += n_sig
            per_dl_totals[int(dl)] += n_sig
            if n_sig > 0:
                idx = np.where(sig)[0]
                order = np.argsort(-np.abs(d_vals[idx]))[:TOP_EFFECTS]
                top = idx[order]
                downstream_edges[str(int(dl))] = {
                    "n_significant": n_sig,
                    "top_effects": [
                        {"target": int(t), "d": float(d_vals[t]), "consistency": float(c_vals[t])}
                        for t in top
                    ],
                }
        features_out.append({
            "feature_idx": int(fid),
            "label": labels.get(int(fid), "unannotated"),
            "activation_freq": catalog.get(int(fid), {}).get("activation_freq", 0.0),
            "n_cells_active": int(n_active_cells.get(int(fid), 0)),
            "total_significant_edges": total_sig,
            "downstream_edges": downstream_edges,
        })

    features_out.sort(key=lambda r: -r["total_significant_edges"])
    edge_counts = np.array([r["total_significant_edges"] for r in features_out], dtype=np.float64)

    annotated_ids = {r["feature_idx"] for r in features_out if r["label"] != "unannotated"}
    n_traced = len(features_out)

    def ann_rate(subset):
        if not subset:
            return None
        return sum(1 for r in subset if r["feature_idx"] in annotated_ids) / len(subset)

    summary = {
        "source_layer": int(source_layer),
        "downstream_layers": [int(d) for d in downstream_layers],
        "n_cells": int(n_cells),
        "partial_through_cells": partial_through,
        "n_features_traced": n_traced,
        "total_edges": int(edge_counts.sum()) if n_traced else 0,
        "mean_edges_per_feature": float(edge_counts.mean()) if n_traced else 0.0,
        "median_edges_per_feature": float(np.median(edge_counts)) if n_traced else 0.0,
        "max_edges": int(edge_counts.max()) if n_traced else 0,
        "n_zero_edge_features": int((edge_counts == 0).sum()),
        "n_features_gt_500": int((edge_counts > 500).sum()),
        "n_features_gt_1000": int((edge_counts > 1000).sum()),
        "per_downstream_edge_totals": per_dl_totals,
        "annotation_bias": {
            "overall_rate": ann_rate(features_out),
            "top20_rate": ann_rate(features_out[:20]),
            "top100_rate": ann_rate(features_out[:100]),
            # TODO: add Fisher's-exact enrichment top-k vs overall (Phase 4) for the full run.
        },
        "top20_hubs": [
            {"feature_idx": r["feature_idx"], "label": r["label"],
             "total_edges": r["total_significant_edges"], "n_cells_active": r["n_cells_active"]}
            for r in features_out[:20]
        ],
    }
    return {"summary": summary, "features": features_out}


# ==================================================================================
# (b) THREE-WAY COMBINATORIAL ABLATION
# ==================================================================================
def _ablate_condition_delta(src_sae, h_sparse, recon_clean, feats_to_zero, device):
    """decode(h with feats_to_zero set to 0) - recon_clean = sum of removed features' contributions.
    Returns (n_genes, d) residual-space delta. mu cancels."""
    h_cond = h_sparse.copy()
    for f in feats_to_zero:
        h_cond[:, f] = 0.0
    return sae_decode_np(src_sae, h_cond, device) - recon_clean


def run_combinatorial(model, records, sae_cache, source_layer, measure_layer, triplets):
    """For each triplet at source_layer, accumulate Cohen's d at measure_layer for the 7 ablation
    conditions over the gene rows where ANY of the triplet features fire. Cells-outer streaming."""
    src_sae, src_mu = sae_cache.get(source_layer)
    m_sae, m_mu = sae_cache.get(measure_layer)
    conditions = ["A", "B", "C", "AB", "AC", "BC", "ABC"]

    results = []
    for tri in triplets:
        fA, fB, fC = int(tri["A"]), int(tri["B"]), int(tri["C"])
        member = {"A": [fA], "B": [fB], "C": [fC], "AB": [fA, fB], "AC": [fA, fC],
                  "BC": [fB, fC], "ABC": [fA, fB, fC]}
        accs = {cond: WelfordAccumulator(D_SAE) for cond in conditions}
        n_active = 0

        for rec in records:
            hidden, _ = clean_forward(model, rec, [source_layer, measure_layer])
            src_rows = meanpool_gene_rows(hidden[source_layer], rec["pos_list"])
            h_sparse, topk = sae_encode_np(src_sae, src_mu, src_rows, sae_cache.device)
            recon_clean = sae_decode_np(src_sae, h_sparse, sae_cache.device)

            joint = ((topk == fA) | (topk == fB) | (topk == fC)).any(axis=1)
            active_gi = np.where(joint)[0]
            if active_gi.size == 0:
                continue
            n_active += 1

            m_rows_clean = meanpool_gene_rows(hidden[measure_layer], rec["pos_list"])
            z_clean, _ = sae_encode_np(m_sae, m_mu, m_rows_clean, sae_cache.device)

            for cond in conditions:
                delta_gene = _ablate_condition_delta(src_sae, h_sparse, recon_clean,
                                                     member[cond], sae_cache.device)
                modified = apply_gene_delta(hidden[source_layer], rec["pos_list"], delta_gene)
                hid_cond, _ = edited_forward(model, rec, source_layer, modified, [measure_layer])
                m_rows = meanpool_gene_rows(hid_cond[measure_layer], rec["pos_list"])
                z_cond, _ = sae_encode_np(m_sae, m_mu, m_rows, sae_cache.device)
                delta_z = (z_cond[active_gi] - z_clean[active_gi]).astype(np.float64).mean(axis=0)
                accs[cond].update(delta_z)

        # finalize + redundancy / interaction analysis (finalize once per condition)
        fin = {cond: accs[cond].finalize() for cond in conditions}
        d = {cond: fin[cond][0] for cond in conditions}
        c = {cond: fin[cond][1] for cond in conditions}
        ad = {cond: np.abs(d[cond]) for cond in conditions}
        sum_singles = ad["A"] + ad["B"] + ad["C"]

        # significant target-feature support (>=1 significant effect in any condition)
        sig_any = np.zeros(D_SAE, dtype=bool)
        for cond in conditions:
            sig_any |= (ad[cond] > COHENS_D_THRESHOLD) & (c[cond] > CONSISTENCY_THRESHOLD)
        support = np.where(sig_any)[0]

        with np.errstate(divide="ignore", invalid="ignore"):
            R_ABC_vec = np.where(sum_singles > 0.01, ad["ABC"] / sum_singles, 1.0)
        # Möbius higher-order interaction (signed, per spec)
        I_ABC_vec = d["ABC"] - d["AB"] - d["AC"] - d["BC"] + d["A"] + d["B"] + d["C"]
        marginal_C_vec = ad["ABC"] - ad["AB"]

        if support.size:
            R_sup = R_ABC_vec[support]
            n_sub = int((R_sup < 0.8).sum())
            n_add = int(((R_sup >= 0.8) & (R_sup <= 1.2)).sum())
            n_super = int((R_sup > 1.2).sum())
            # strict superadditivity per Phase 8: |d_ABC| > |d_A|+|d_B|+|d_C|
            n_strict_super = int((ad["ABC"][support] > sum_singles[support]).sum())
            R_ABC = float(np.median(R_sup))
            I_ABC = float(np.median(I_ABC_vec[support]))
            marg = float(np.median(marginal_C_vec[support]))
        else:
            n_sub = n_add = n_super = n_strict_super = 0
            R_ABC = 1.0
            I_ABC = 0.0
            marg = 0.0

        results.append({
            "triplet": {"A": fA, "B": fB, "C": fC},
            "labels": {"A": tri.get("label_A"), "B": tri.get("label_B"), "C": tri.get("label_C")},
            "type": tri.get("type", "same_pathway"),
            "pathway": tri.get("pathway"),
            "source_layer": int(source_layer),
            "measure_layer": int(measure_layer),
            "n_cells_active": n_active,
            "n_support_targets": int(support.size),
            "R_ABC_median": R_ABC,
            "I_ABC_median": I_ABC,
            "marginal_C_given_AB_median": marg,
            "n_subadditive": n_sub,
            "n_additive": n_add,
            "n_superadditive": n_super,
            "n_strict_superadditive": n_strict_super,
            "mean_abs_d": {cond: float(np.mean(ad[cond])) for cond in conditions},
        })
        print(f"    [triplet {fA}/{fB}/{fC}] R_ABC={R_ABC:.3f} I_ABC={I_ABC:.4f} "
              f"support={support.size} super={n_strict_super}", flush=True)

    return {"measure_layer": int(measure_layer), "triplets": results}


def select_triplets(catalog, rich, top_term, n_triplets):
    """Heuristic triplet picker (no Stage-2 pairwise summary available on C2S-Scale yet).

    TODO(assumption): the reference selects triplets from Stage-2 pairwise ablation results,
    grouping features at L0/L5/L11 by shared pathway. Here every feature is at ONE layer, so we
    group annotation-rich features (activation_freq>=0.01, >=10 top_genes, not dead, annotated)
    by their top ontology term to form SAME-pathway triplets, plus a couple of CROSS-pathway
    triplets from distinct terms. Replace with a proper Stage-2-driven selection once available.
    """
    cand = [fid for fid, sc in rich.items()
            if catalog.get(fid, {}).get("activation_freq", 0) >= ACT_FREQ_SELECT
            and catalog.get(fid, {}).get("n_top_genes", 0) >= MIN_TOP_GENES
            and not catalog.get(fid, {}).get("is_dead", False)]
    cand.sort(key=lambda f: -rich[f])

    by_term = {}
    for fid in cand:
        by_term.setdefault(top_term.get(fid, "?"), []).append(fid)

    triplets = []
    # same-pathway: any term with >= 3 candidate features
    for term, fids in sorted(by_term.items(), key=lambda kv: -len(kv[1])):
        if len(fids) >= 3:
            a, b, cc = fids[0], fids[1], fids[2]
            triplets.append({"A": a, "B": b, "C": cc, "type": "same_pathway", "pathway": term,
                             "label_A": term, "label_B": term, "label_C": term})
        if len(triplets) >= max(1, n_triplets - 1):
            break

    # one cross-pathway triplet from the three most distinct high-score terms
    distinct_terms = [t for t, fids in sorted(by_term.items(), key=lambda kv: -rich[kv[1][0]])
                      if fids]
    if len(distinct_terms) >= 3:
        a = by_term[distinct_terms[0]][0]
        b = by_term[distinct_terms[1]][0]
        cc = by_term[distinct_terms[2]][0]
        triplets.append({"A": a, "B": b, "C": cc, "type": "cross_pathway",
                         "pathway": f"{distinct_terms[0]} x {distinct_terms[1]} x {distinct_terms[2]}",
                         "label_A": distinct_terms[0], "label_B": distinct_terms[1],
                         "label_C": distinct_terms[2]})

    return triplets[:n_triplets]


# ==================================================================================
# (c) TRAJECTORY-GUIDED STEERING
# ==================================================================================
def _cosine(a, b):
    na = float(np.sqrt(np.sum(a * a)))
    nb = float(np.sqrt(np.sum(b * b)))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.sum(a * b) / (na * nb))


def run_steering(model, tok, records, sae_cache, source_layer, steer_features, alphas,
                 labels, signatures=None):
    """Amplify each feature by alpha via  h' = h + (alpha-1) * a_f * d_f  and propagate; measure
    the induced state shift. Cells-outer streaming, only in cells where the feature fires."""
    src_sae, src_mu = sae_cache.get(source_layer)
    dev = sae_cache.device
    vocab = None

    early_sig = late_sig = None
    if signatures is not None:
        early_sig = np.asarray(signatures["early_sig"], dtype=np.float64)
        late_sig = np.asarray(signatures["late_sig"], dtype=np.float64)

    results = []
    for fid in steer_features:
        fid = int(fid)
        d_f = sae_cache.decoder_direction(source_layer, fid)  # (d_model,) unit vector
        per_alpha = {}
        for alpha in alphas:
            shifts_ds, shift_mag, shift_cos = [], [], []
            logit_delta_acc = None
            n_active = 0
            for rec in records:
                hidden, clean_logits = clean_forward(model, rec, [source_layer])
                gene_positions = np.concatenate(rec["pos_list"])  # token positions with a gene
                src_rows = meanpool_gene_rows(hidden[source_layer], rec["pos_list"])
                h_sparse, topk = sae_encode_np(src_sae, src_mu, src_rows, dev)

                gene_mask = (topk == fid).any(axis=1)
                if not gene_mask.any():
                    continue
                n_active += 1
                a_f = h_sparse[:, fid]  # (n_genes,) per-gene activation coeff

                # activation-addition steering delta (spec formula; equals decode-difference):
                #   delta_gene = (alpha - 1) * a_f * d_f    (nonzero only where the feature fired)
                delta_gene = (alpha - 1.0) * a_f[:, None] * d_f[None, :]  # (n_genes, d_model)
                modified = apply_gene_delta(hidden[source_layer], rec["pos_list"], delta_gene)
                _, steer_logits = edited_forward(model, rec, source_layer, modified, [], want_logits=True)

                if vocab is None:
                    vocab = clean_logits.shape[1]
                # state = mean logit vector over gene-token positions (analog of the reference,
                # which meaned encoder logits over gene positions). TODO: for a causal LM the
                # logit at pos i predicts token i+1; the mean over gene positions is a coarse
                # state summary — a last-real-token readout is an alternative worth comparing.
                z_clean = clean_logits[gene_positions].mean(axis=0).astype(np.float64)
                z_steer = steer_logits[gene_positions].mean(axis=0).astype(np.float64)
                dlog = z_steer - z_clean

                shift_mag.append(float(np.mean(np.abs(dlog))))
                shift_cos.append(_cosine(z_clean, z_steer))
                logit_delta_acc = dlog if logit_delta_acc is None else logit_delta_acc + dlog

                if early_sig is not None and late_sig is not None:
                    ds = ((_cosine(z_steer, late_sig) - _cosine(z_steer, early_sig))
                          - (_cosine(z_clean, late_sig) - _cosine(z_clean, early_sig)))
                    shifts_ds.append(ds)

            if n_active == 0:
                per_alpha[str(alpha)] = {"n_cells_active": 0}
                continue

            mean_delta = logit_delta_acc / n_active
            up = np.argsort(-mean_delta)[:20]
            dn = np.argsort(mean_delta)[:20]
            entry = {
                "n_cells_active": n_active,
                "mean_state_shift_magnitude": float(np.mean(shift_mag)),
                "mean_state_cosine": float(np.mean(shift_cos)),
                "top_upregulated": [{"token": tok.convert_ids_to_tokens(int(t)),
                                     "token_id": int(t), "delta": float(mean_delta[t])} for t in up],
                "top_downregulated": [{"token": tok.convert_ids_to_tokens(int(t)),
                                       "token_id": int(t), "delta": float(mean_delta[t])} for t in dn],
            }
            if shifts_ds:
                arr = np.array(shifts_ds)
                entry.update({
                    "mean_delta_s": float(arr.mean()),
                    "median_delta_s": float(np.median(arr)),
                    "frac_positive_shift": float((arr > 0).mean()),
                })
            per_alpha[str(alpha)] = entry

        results.append({
            "feature_idx": fid,
            "label": labels.get(fid, "unannotated"),
            "source_layer": int(source_layer),
            "alpha_results": per_alpha,
        })
        print(f"    [steer F{fid}] " + " ".join(
            f"a{a}:mag={per_alpha[str(a)].get('mean_state_shift_magnitude', 0):.4g}" for a in alphas),
            flush=True)

    return {"alphas": [float(a) for a in alphas], "has_signatures": signatures is not None,
            "features": results}


# ==================================================================================
# main
# ==================================================================================
def parse_args():
    p = argparse.ArgumentParser(description="C2S-Scale Stage 3: exhaustive mapping + higher-order "
                                            "ablation + trajectory steering (one source layer)")
    p.add_argument("--run", required=True, help="atlas run dir (sae/, feature_catalog_L*.json, ...)")
    p.add_argument("--model", required=True)
    p.add_argument("--h5ad", required=True)
    p.add_argument("--layer", type=int, required=True, help="source layer L (0-based)")
    p.add_argument("--out", default=None, help="output JSON (default {run}/exhaustive_mapping_L{L:02d}.json)")
    # shared
    p.add_argument("--n-features", type=int, default=D_SAE, help="SAE dictionary width (default 9216)")
    p.add_argument("--n-cells", type=int, default=50, help="cells to stream per analysis")
    p.add_argument("--max-genes", type=int, default=512)
    p.add_argument("--gene-col", default="gene")
    p.add_argument("--control-label", default="non-targeting")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true")
    # (a) exhaustive tracing
    p.add_argument("--downstream-layers", default=None,
                   help="comma-separated downstream layers (default: adjacent, mid, last after L)")
    p.add_argument("--max-source-features", type=int, default=None,
                   help="cap #active source features traced (memory/compute); default all in range")
    p.add_argument("--feature-start", type=int, default=0, help="source-feature shard start (resume-safe)")
    p.add_argument("--feature-end", type=int, default=D_SAE, help="source-feature shard end (exclusive)")
    p.add_argument("--save-every", type=int, default=5, help="partial-save cadence (cells)")
    # (b) combinatorial
    p.add_argument("--n-triplets", type=int, default=4)
    p.add_argument("--triplets", default=None,
                   help="explicit triplets 'a,b,c;a2,b2,c2' (overrides auto-selection)")
    p.add_argument("--measure-layer", type=int, default=None,
                   help="downstream layer for triplet effects (default last layer > L)")
    # (c) steering
    p.add_argument("--steer-features", default=None, help="comma-separated feature ids to steer")
    p.add_argument("--n-steer-features", type=int, default=5, help="#auto-picked features if none given")
    p.add_argument("--alphas", default="2.0,5.0")
    p.add_argument("--signatures", default=None,
                   help="optional .npz with early_sig/late_sig vocab vectors for the trajectory Δs")
    # toggles
    p.add_argument("--skip-tracing", action="store_true")
    p.add_argument("--skip-combinatorial", action="store_true")
    p.add_argument("--skip-steering", action="store_true")
    return p.parse_args()


def _default_downstream(L):
    """adjacent, ~mid, last — all strictly after L and within [L+1, N_LAYERS-1]."""
    if L >= N_LAYERS - 1:
        return []
    last = N_LAYERS - 1
    mid = L + max(1, (last - L) // 2)
    return sorted(set(x for x in (L + 1, mid, last) if L < x <= last))


def main():
    args = parse_args()
    np.random.seed(args.seed)
    L = args.layer
    out_path = args.out or os.path.join(args.run, f"exhaustive_mapping_L{L:02d}.json")

    if os.path.exists(out_path) and not args.overwrite:
        print(f"[resume] {out_path} exists — skipping (use --overwrite to recompute).")
        return

    downstream = ([int(x) for x in args.downstream_layers.split(",")]
                  if args.downstream_layers else _default_downstream(L))
    measure_layer = args.measure_layer if args.measure_layer is not None else (N_LAYERS - 1)

    import torch  # noqa: F401
    import anndata
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"device: {device} | source L{L} | downstream {downstream} | measure L{measure_layer}", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if not tok.is_fast:
        raise RuntimeError("A fast tokenizer is required for offset_mapping (gene attribution).")
    max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)

    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16,
                                                     device_map="auto", output_hidden_states=True)
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16,
                                                     device_map="auto", output_hidden_states=True)
    model = model.eval()

    sae_cache = SAECache(args.run, device)
    catalog = load_catalog(args.run, L)
    labels, annotated, rich, top_term = load_annotations(args.run, L)

    adata = anndata.read_h5ad(args.h5ad)
    gene_col = adata.obs[args.gene_col].astype(str).values
    control_idx = np.where(gene_col == args.control_label)[0]
    if control_idx.size == 0:
        raise SystemExit(f"no control cells with obs[{args.gene_col!r}]=={args.control_label!r}")
    rng = np.random.RandomState(args.seed)
    if control_idx.size > args.n_cells:
        control_idx = np.sort(rng.choice(control_idx, args.n_cells, replace=False))

    print(f"building cell records for {control_idx.size} control cells...", flush=True)
    records = build_cell_records(adata, control_idx, tok, args.max_genes, max_len, use_prompt=True)
    print(f"  usable cell records: {len(records)}", flush=True)

    result = {
        "layer": int(L),
        "model": args.model,
        "config": {
            "n_features": int(args.n_features), "n_cells": len(records),
            "downstream_layers": downstream, "measure_layer": int(measure_layer),
            "cohens_d_threshold": COHENS_D_THRESHOLD, "consistency_threshold": CONSISTENCY_THRESHOLD,
            "act_freq_trace": ACT_FREQ_TRACE, "max_genes": args.max_genes,
        },
    }

    def save():
        tmp = out_path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(result, f, indent=2, default=_json_default)
        os.replace(tmp, out_path)

    # ---- (a) exhaustive tracing ----
    if not args.skip_tracing and downstream:
        active = [fid for fid, meta in catalog.items()
                  if meta["activation_freq"] >= ACT_FREQ_TRACE and not meta["is_dead"]
                  and args.feature_start <= fid < args.feature_end]
        active.sort(key=lambda f: -catalog[f]["activation_freq"])
        if args.max_source_features is not None:
            active = active[:args.max_source_features]
        print(f"\n(a) EXHAUSTIVE tracing: {len(active)} active source features "
              f"(freq>={ACT_FREQ_TRACE}) at L{L} -> downstream {downstream}", flush=True)
        result["exhaustive_tracing"] = {"summary": {}, "features": []}

        def trace_save(partial):
            result["exhaustive_tracing"] = partial
            save()

        result["exhaustive_tracing"] = run_exhaustive_tracing(
            model, records, sae_cache, L, downstream, catalog, labels, active,
            save_cb=trace_save, save_every=args.save_every)
        save()
    elif not args.skip_tracing:
        print("(a) EXHAUSTIVE tracing skipped: no downstream layers after L.", flush=True)

    # ---- (b) combinatorial ablation ----
    if not args.skip_combinatorial and measure_layer > L:
        if args.triplets:
            triplets = []
            for chunk in args.triplets.split(";"):
                a, b, c = (int(x) for x in chunk.split(","))
                triplets.append({"A": a, "B": b, "C": c, "type": "explicit", "pathway": "explicit",
                                 "label_A": labels.get(a), "label_B": labels.get(b),
                                 "label_C": labels.get(c)})
        else:
            triplets = select_triplets(catalog, rich, top_term, args.n_triplets)
        print(f"\n(b) COMBINATORIAL ablation: {len(triplets)} triplets at L{L}, measured L{measure_layer}",
              flush=True)
        result["combinatorial_ablation"] = run_combinatorial(model, records, sae_cache, L,
                                                             measure_layer, triplets)
        save()
    elif not args.skip_combinatorial:
        print("(b) COMBINATORIAL ablation skipped: measure_layer not after L.", flush=True)

    # ---- (c) steering ----
    if not args.skip_steering:
        if args.steer_features:
            steer_features = [int(x) for x in args.steer_features.split(",")]
        else:
            # TODO(assumption): real "switch features" require Stage-1 pseudotime identification
            # (spec Phase 10). Absent that on C2S-Scale, default to the most annotation-rich active
            # features at L as a placeholder steering set.
            cand = [fid for fid, sc in sorted(rich.items(), key=lambda kv: -kv[1])
                    if catalog.get(fid, {}).get("activation_freq", 0) >= ACT_FREQ_SELECT
                    and not catalog.get(fid, {}).get("is_dead", False)]
            steer_features = cand[:args.n_steer_features]
        alphas = [float(a) for a in args.alphas.split(",")]
        signatures = np.load(args.signatures) if args.signatures else None
        print(f"\n(c) STEERING: features {steer_features} alphas {alphas} at L{L}", flush=True)
        result["steering"] = run_steering(model, tok, records, sae_cache, L, steer_features,
                                          alphas, labels, signatures=signatures)
        save()

    save()
    print(f"\ndone -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
