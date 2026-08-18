"""
Stage 2 (C2S-Scale adapter): Causal feature-to-feature circuit tracing.

Adapted from repos/bio-sae-circuits/src/13_causal_circuit_tracing.py (Geneformer). The
methodology, significance test (Cohen's d + consistency), Welford accumulation, per-cell
resume/partial saves, and output schema (circuit_L{L:02d}_features.json + circuit_graph.json
+ circuit_analysis.json) are kept faithful. Only the model, tokenization/gene-position
attribution, SAE I/O, and feature-selection I/O are remapped to the C2S-Scale contract.

Ablate individual SAE features at source layers and measure how downstream SAE features
change — revealing directed information flow through the model.

For each source feature:
  1. Run clean forward pass -> capture hidden states at ALL downstream layers
  2. Mean-pool each gene's subword tokens -> gene rows; encode source gene rows through the
     source SAE -> clean activations (the atlas SAE was trained on mean-pooled gene rows)
  3. Ablate source feature: SAE encode -> zero feature -> decode -> per-gene reconstruction delta
  4. Inject the per-gene delta onto every subword-token position of that gene at the source
     layer's residual stream (forward hook on model.model.layers[L]); ablated forward pass
  5. Mean-pool + encode ablated downstream gene rows through downstream SAEs -> ablated activations
  6. Accumulate per-downstream-feature deltas across cells (Welford's algorithm), one
     observation per cell = mean over the genes in which the source feature is active
  7. Compute significance: |Cohen's d| > 0.5 AND consistency > 0.7

C2S-Scale format (see projects/c2s-scale/code contract):
  * MODEL: vandijklab/C2S-Scale-Gemma-2-2B (Gemma2ForCausalLM, d_model=2304, 26 layers, bf16).
    hidden_states[L+1] is the post-block residual stream of transformer layer L.
  * HOOK: register_forward_hook on model.model.layers[L]; the block returns a tuple, output[0]
    is the (1, seq, d) hidden state. The hook replaces output[0] with the edited hidden (cast
    back to bf16 on the block's device). (Geneformer used model.bert.encoder.layer[L].)
  * TOKENIZATION / GENE POSITIONS: cell_sentences.build_cell_sentence + a fast tokenizer's
    offset_mapping + attribute_tokens_to_genes -> per-token gene index or SCAFFOLD(-1). Genes
    span several subword tokens, so we mean-pool per gene (matches the atlas SAE distribution).
  * SAE: sae_topk.TopKSAE.load(sae/sae_L{L:02d}.pt) + mu_L{L:02d}.npy; hidden bf16 -> .float()
    before the SAE, SAE math in float32, edited hidden cast back to bf16 for the hook.
  * ATLAS OUTPUTS (run dir): feature_catalog_L{L:02d}.json + feature_annotations_L{L:02d}.json.

Note: the reference's compare_with_pmi is DROPPED. Our atlas produces
coactivation_modules_L{L}.json whose top_pmi_edges are WITHIN-layer feature coactivations
(feat_a, feat_b at the same layer), not the reference's CROSS-layer PMI dependency files
(deps_L{src}_to_L{dst}.json). Comparing cross-layer causal edges to within-layer coactivation
would be meaningless, so the comparison is skipped and recorded as such in circuit_analysis.json.

Usage:
    python circuit_tracing.py \
        --run runs/2b_atlas --model vandijklab/C2S-Scale-Gemma-2-2B \
        --h5ad data/replogle_k562_subset.h5ad \
        --source-layers 0,8,16 --n-features 30 --n-cells 200 \
        [--available-layers 0,1,...,25] [--out runs/2b_atlas] [--analysis-only]
"""

import os
import sys
import gc
import json
import time
import argparse
import warnings

warnings.filterwarnings('ignore')
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', 1)

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

# ---- C2S-Scale constants (Gemma-2-2B atlas) ----
HIDDEN_DIM = 2304
N_FEATURES = 9216   # d_SAE = 4x expansion
N_LAYERS = 26
K_VAL = 32

# Significance thresholds (unchanged from the reference)
COHENS_D_THRESHOLD = 0.5
CONSISTENCY_THRESHOLD = 0.7
TOP_EFFECTS_PER_LAYER = 50  # Keep top N effects per downstream layer in output

# Cell / control sampling
DEFAULT_GENE_COL = 'gene'
DEFAULT_CONTROL_LABEL = 'non-targeting'
MAX_GENES = 512
SEED = 42


def _json_default(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Not JSON serializable: {type(obj)}")


# ============================================================
# Feature selection (C2S-Scale atlas I/O)
# ============================================================

def select_features(run, layer, n_features=30):
    """Select well-annotated features for circuit tracing.

    Reads the C2S-Scale atlas outputs feature_catalog_L{L:02d}.json +
    feature_annotations_L{L:02d}.json from the run dir. Scoring is unchanged from the
    reference (08_causal_patching.py):
        score = n_ontologies * 10 + n_annotations - log10(min_p)
    Requires >=10 top_genes and activation_freq >= 0.01, skips dead features.
    """
    catalog_path = os.path.join(run, f"feature_catalog_L{layer:02d}.json")
    ann_path = os.path.join(run, f"feature_annotations_L{layer:02d}.json")

    with open(ann_path) as f:
        ann_data = json.load(f)
    with open(catalog_path) as f:
        catalog = json.load(f)

    feature_annotations = ann_data.get('feature_annotations', {})

    # Build gene sets and freq for each (alive) feature
    feature_genes = {}
    feature_freq = {}
    for feat in catalog['features']:
        if feat.get('is_dead'):
            continue
        fi = feat['feature_idx']
        if feat.get('top_genes'):
            feature_genes[fi] = [g['gene_name'] for g in feat['top_genes'][:20]]
        feature_freq[fi] = feat.get('activation_freq', 0)

    # Score features by annotation quality
    scored = []
    for fid_str, anns in feature_annotations.items():
        fi = int(fid_str)
        if fi not in feature_genes or len(feature_genes[fi]) < 10:
            continue
        if feature_freq.get(fi, 0) < 0.01:
            continue

        ontologies = set(a['ontology'] for a in anns)
        n_ont = len(ontologies)
        n_ann = len(anns)
        min_p = min(a.get('p_adjusted', 1.0) for a in anns) if anns else 1.0

        best_label = "unknown"
        for a in anns:
            if a['ontology'] in ('GO_BP', 'KEGG', 'Reactome'):
                best_label = a['term']
                break

        scored.append({
            'feature_idx': fi,
            'n_ontologies': n_ont,
            'n_annotations': n_ann,
            'min_p': min_p,
            'label': best_label,
            'top_genes': feature_genes[fi],
            'activation_freq': feature_freq.get(fi, 0),
            'score': n_ont * 10 + n_ann - np.log10(max(min_p, 1e-30)),
        })

    scored.sort(key=lambda x: -x['score'])
    selected = scored[:n_features]
    print(f"  Selected {len(selected)} features for circuit tracing at layer {layer}")
    for i, s in enumerate(selected[:10]):
        print(f"    [{i}] Feature {s['feature_idx']}: {s['n_ontologies']} ont, "
              f"{s['n_annotations']} ann, freq={s['activation_freq']:.3f} | {s['label'][:50]}")

    return selected


# ============================================================
# Online accumulator (verbatim from the reference)
# ============================================================

class WelfordAccumulator:
    """Online mean/variance accumulator using Welford's algorithm."""

    def __init__(self, n_features):
        self.n = 0
        self.mean = np.zeros(n_features, dtype=np.float64)
        self.M2 = np.zeros(n_features, dtype=np.float64)
        self.pos_count = np.zeros(n_features, dtype=np.float64)

    def update(self, delta_vector):
        """Update with a single observation vector."""
        self.n += 1
        d = delta_vector - self.mean
        self.mean += d / self.n
        d2 = delta_vector - self.mean
        self.M2 += d * d2
        self.pos_count += (delta_vector > 0).astype(np.float64)

    def finalize(self):
        """Compute Cohen's d and consistency for all features."""
        if self.n < 2:
            return (np.zeros_like(self.mean), np.zeros_like(self.mean))
        var = self.M2 / (self.n - 1)
        std = np.sqrt(var)
        cohens_d = self.mean / np.maximum(std, 1e-10)
        consistency = np.maximum(self.pos_count, self.n - self.pos_count) / self.n
        return cohens_d, consistency


class SAECache:
    """Lazy-loading cache for C2S-Scale SAE models (sae_L{L:02d}.pt + mu_L{L:02d}.npy)."""

    def __init__(self, run, device):
        self.run = run
        self.device = device
        self._saes = {}
        self._means = {}

    def get(self, layer):
        """Get (SAE, activation-mean (1, d) float32) for a layer."""
        if layer not in self._saes:
            sae = TopKSAE.load(os.path.join(self.run, "sae", f"sae_L{layer:02d}.pt"),
                               device=self.device)
            sae.eval()
            act_mean = np.load(os.path.join(self.run, "sae", f"mu_L{layer:02d}.npy"))
            act_mean = act_mean.astype(np.float32).reshape(1, -1)
            self._saes[layer] = sae
            self._means[layer] = act_mean
            print(f"    Loaded SAE for layer {layer}")

        return self._saes[layer], self._means[layer]

    def preload(self, layers):
        """Preload SAEs for given layers."""
        for l in layers:
            self.get(l)


# ============================================================
# Cell tokenization (C2S-Scale cell-sentence format)
# ============================================================

def load_and_tokenize_cells(h5ad, tok, n_cells, max_len,
                            gene_col=DEFAULT_GENE_COL,
                            control_label=DEFAULT_CONTROL_LABEL,
                            max_genes=MAX_GENES, seed=SEED):
    """Load control cells and precompute per-cell tokenization + gene attribution.

    Each returned cell dict carries everything the clean AND ablated passes reuse:
      input_ids, attention_mask (np int64, 1D), present_genes (gene ids with >=1 token),
      gene_masks (list of np int64 token-position arrays, one per present gene).
    Mirrors the reference's precomputed `all_tokens`, generalized to subword->gene spans.
    """
    import anndata

    print("  Loading cell data (control cells)...")
    adata = anndata.read_h5ad(h5ad)
    genes_col = adata.obs[gene_col].astype(str).values
    ctrl_idx = np.where(genes_col == control_label)[0]
    print(f"    {len(ctrl_idx)} control cells ('{control_label}') available")

    rng = np.random.RandomState(seed)
    if len(ctrl_idx) > n_cells:
        ctrl_idx = np.sort(rng.choice(ctrl_idx, n_cells, replace=False))

    cells = []
    for ci in ctrl_idx[:n_cells]:
        genes = anndata_to_ranked_genes(adata, int(ci), max_genes=max_genes)
        if not genes:
            continue
        cs = build_cell_sentence(genes, prompt=True)
        enc = tok(cs.text, return_offsets_mapping=True, return_tensors="pt",
                  truncation=True, max_length=max_len)
        offsets = enc.pop("offset_mapping")[0].tolist()
        tok2gene = attribute_tokens_to_genes(offsets, cs.gene_spans)

        present_genes = [g for g in range(len(genes)) if (tok2gene == g).any()]
        if not present_genes:
            continue
        gene_masks = [np.where(tok2gene == g)[0] for g in present_genes]

        cells.append({
            'input_ids': enc["input_ids"][0].numpy().astype(np.int64),
            'attention_mask': enc["attention_mask"][0].numpy().astype(np.int64),
            'present_genes': present_genes,
            'gene_masks': gene_masks,
        })
    print(f"    Tokenized {len(cells)} cells "
          f"(max {max_genes} genes/cell, gene positions = tok2gene != SCAFFOLD({SCAFFOLD}))")
    del adata
    gc.collect()
    return cells


def _pool_gene_rows(hidden_cpu, gene_idx_tensors):
    """Mean-pool a (seq_len, d) hidden tensor over each gene's subword-token positions.

    Returns (n_genes, d) float32 — one row per present gene, matching the gene-token unit
    the atlas SAE was trained on (extract_activations.py: hidden[mask].mean(axis=0)).
    """
    import torch
    return torch.stack([hidden_cpu[idx].mean(dim=0) for idx in gene_idx_tensors], dim=0)


def make_hook(modified_hidden):
    """Create a hook that replaces the source block's hidden state (bf16-safe).

    model.model.layers[L].forward returns a tuple; output[0] is (1, seq, d). We swap in
    modified_hidden (seq, d, float32 on CPU), unsqueezed to a batch dim and cast to the
    block's own device+dtype (bf16) so downstream layers see the edit.
    """
    fired = [False]

    def hook_fn(module, inp, output):
        if fired[0]:
            return output
        fired[0] = True
        modified = list(output)
        modified[0] = modified_hidden.unsqueeze(0).to(
            device=output[0].device, dtype=output[0].dtype)
        return tuple(modified)

    return hook_fn


# ============================================================
# Core tracing (one source layer)
# ============================================================

def trace_source_layer(source_layer, selected_features, cells, model, device,
                       sae_cache, out_dir, n_cells, available_layers=None):
    """Run circuit tracing for all source features at one source layer.

    Iterates over cells in the outer loop. For each cell, runs ONE clean forward pass, then
    processes all features active in that cell (30x fewer clean passes). Downstream tracing
    is restricted to `available_layers` if provided (layers with SAE models).
    """
    import torch

    partial_path = os.path.join(out_dir, f"circuit_tracing_partial_L{source_layer:02d}.json")
    final_path = os.path.join(out_dir, f"circuit_L{source_layer:02d}_features.json")

    if os.path.exists(final_path):
        print(f"  Already done: {final_path}")
        return final_path

    # Load source SAE + mean (on device)
    source_sae, source_act_mean = sae_cache.get(source_layer)
    source_mu_t = torch.tensor(source_act_mean, dtype=torch.float32, device=device)

    # Determine downstream layers
    if available_layers is not None:
        downstream_layers = sorted([l for l in available_layers if l > source_layer])
    else:
        downstream_layers = list(range(source_layer + 1, N_LAYERS))
    print(f"  Source layer {source_layer} -> tracing through {len(downstream_layers)} "
          f"downstream layers: {downstream_layers}")

    # Preload downstream SAEs and cache their device mean tensors
    sae_cache.preload(downstream_layers)
    dst_mu_tensors = {}
    for dl in downstream_layers:
        _, dst_mean = sae_cache.get(dl)
        dst_mu_tensors[dl] = torch.tensor(dst_mean, dtype=torch.float32, device=device)

    feature_indices = [sf['feature_idx'] for sf in selected_features]

    # Initialize Welford accumulators: one per (feature, downstream_layer)
    accumulators = {}
    n_cells_active = {}
    for sf in selected_features:
        fi = sf['feature_idx']
        accumulators[fi] = {dl: WelfordAccumulator(N_FEATURES) for dl in downstream_layers}
        n_cells_active[fi] = 0

    # Resume from partial results (cell level)
    start_cell = 0
    if os.path.exists(partial_path):
        with open(partial_path) as f:
            partial = json.load(f)
        start_cell = partial.get('cells_completed', 0)
        for feat_data in partial.get('accumulator_state', []):
            fi = feat_data['feature_idx']
            if fi not in accumulators:
                continue
            n_cells_active[fi] = feat_data.get('n_cells_active', 0)
            for dl_str, acc_state in feat_data.get('accumulators', {}).items():
                dl = int(dl_str)
                if dl in accumulators[fi]:
                    acc = accumulators[fi][dl]
                    acc.n = acc_state['n']
                    acc.mean = np.array(acc_state['mean'])
                    acc.M2 = np.array(acc_state['M2'])
                    acc.pos_count = np.array(acc_state['pos_count'])
        print(f"  Resuming from cell {start_cell}/{n_cells}")

    total_t0 = time.time()
    n_ablated_passes = 0
    n_clean_passes = 0

    cells_to_process = cells[:n_cells]
    print(f"  Processing {len(cells_to_process)} cells (starting from {start_cell})...")

    for ci in range(start_cell, len(cells_to_process)):
        cell = cells_to_process[ci]
        input_ids_np = cell['input_ids']
        seq_len = len(input_ids_np)
        gene_idx_tensors = [torch.from_numpy(m) for m in cell['gene_masks']]
        if len(gene_idx_tensors) == 0:
            continue

        input_ids = torch.tensor(input_ids_np, dtype=torch.long, device=device).unsqueeze(0)
        attention_mask = torch.tensor(cell['attention_mask'], dtype=torch.long,
                                      device=device).unsqueeze(0)

        # === ONE clean forward pass per cell ===
        with torch.no_grad():
            outputs_clean = model(input_ids=input_ids, attention_mask=attention_mask,
                                  output_hidden_states=True)
        n_clean_passes += 1

        # Source layer full hidden (post-block residual), float32 on CPU — reused for both
        # the SAE gene rows and building the injected modified hidden.
        clean_hidden_full = outputs_clean.hidden_states[source_layer + 1][0].float().cpu()
        gene_hidden = _pool_gene_rows(clean_hidden_full, gene_idx_tensors)  # (n_genes, d)

        # Encode source gene rows through the source SAE
        with torch.no_grad():
            gene_hidden_centered = gene_hidden.to(device) - source_mu_t
            h_sparse, topk_indices = source_sae.encode(gene_hidden_centered)

        # Pre-pool clean downstream gene rows once per cell (CPU float32)
        clean_downstream = {}
        for dl in downstream_layers:
            dl_full = outputs_clean.hidden_states[dl + 1][0].float().cpu()
            clean_downstream[dl] = _pool_gene_rows(dl_full, gene_idx_tensors)
            del dl_full

        # Process each source feature active in this cell
        for fi in feature_indices:
            active_mask = (topk_indices == fi).any(dim=1)  # (n_genes,)
            if not active_mask.any():
                continue

            n_cells_active[fi] += 1
            active_rows = torch.where(active_mask.cpu())[0]

            # --- Ablation: zero target feature, decode delta per gene row ---
            h_ablated = h_sparse.clone()
            h_ablated[:, fi] = 0.0

            with torch.no_grad():
                recon_normal = source_sae.decode(h_sparse) + source_mu_t
                recon_ablated = source_sae.decode(h_ablated) + source_mu_t

            delta = (recon_ablated - recon_normal).detach().cpu()  # (n_genes, d)

            # Inject each active gene's delta onto ALL its subword-token positions.
            # (Inactive genes have delta ~ 0 since the feature was already 0 in topk.)
            modified_hidden = clean_hidden_full.clone()
            for r in active_rows.tolist():
                modified_hidden[gene_idx_tensors[r]] += delta[r]

            # --- Ablated forward pass (hook on the source decoder block) ---
            hook_handle = model.model.layers[source_layer].register_forward_hook(
                make_hook(modified_hidden)
            )
            with torch.no_grad():
                outputs_ablated = model(input_ids=input_ids, attention_mask=attention_mask,
                                        output_hidden_states=True)
            hook_handle.remove()
            n_ablated_passes += 1

            # --- Encode downstream layers and accumulate one obs/cell ---
            active_positions = active_rows.numpy()
            for dl in downstream_layers:
                dst_sae, _ = sae_cache.get(dl)
                dst_mu_t = dst_mu_tensors[dl]

                clean_dl = clean_downstream[dl]
                ablated_full = outputs_ablated.hidden_states[dl + 1][0].float().cpu()
                ablated_dl = _pool_gene_rows(ablated_full, gene_idx_tensors)
                del ablated_full

                with torch.no_grad():
                    clean_sparse, _ = dst_sae.encode(clean_dl.to(device) - dst_mu_t)
                    ablated_sparse, _ = dst_sae.encode(ablated_dl.to(device) - dst_mu_t)

                diff = (ablated_sparse - clean_sparse).detach().cpu().numpy().astype(np.float64)
                # Average delta across genes where the source feature is active -> one obs/cell
                cell_delta = diff[active_positions].mean(axis=0)
                accumulators[fi][dl].update(cell_delta)

            del outputs_ablated, modified_hidden, h_ablated, delta

        # Cleanup after processing all features for this cell
        del outputs_clean, clean_hidden_full, gene_hidden, h_sparse, topk_indices
        del clean_downstream
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        elif device.type == 'mps':
            torch.mps.empty_cache()

        # Progress report every 20 cells
        if (ci + 1) % 20 == 0:
            elapsed = time.time() - total_t0
            rate = (ci + 1 - start_cell) / max(elapsed, 1e-9)
            remaining = (len(cells_to_process) - ci - 1) / max(rate, 0.001)
            print(f"    Cell {ci+1}/{len(cells_to_process)} | "
                  f"{n_clean_passes} clean + {n_ablated_passes} ablated passes | "
                  f"{rate:.2f} cells/s | ETA {remaining/60:.1f} min")

        # Incremental save every 50 cells
        if (ci + 1) % 50 == 0:
            _save_partial_v2(partial_path, source_layer, ci + 1, feature_indices,
                             accumulators, n_cells_active, n_cells)

    # === Finalize: compute significance and build results ===
    print(f"\n  Finalizing results for source layer {source_layer}...")
    feature_results = []

    for sf in selected_features:
        fi = sf['feature_idx']

        downstream_effects = {}
        total_sig_edges = 0
        attenuation_curve = []

        for dl in downstream_layers:
            acc = accumulators[fi][dl]
            if acc.n < 2:
                downstream_effects[str(dl)] = {
                    'n_significant': 0,
                    'n_cells_measured': acc.n,
                    'top_effects': [],
                }
                attenuation_curve.append(0.0)
                continue

            cohens_d, consistency = acc.finalize()

            sig_mask = (np.abs(cohens_d) > COHENS_D_THRESHOLD) & (consistency > CONSISTENCY_THRESHOLD)
            n_sig = int(sig_mask.sum())
            total_sig_edges += n_sig

            sig_indices = np.where(sig_mask)[0]
            if len(sig_indices) > 0:
                abs_d = np.abs(cohens_d[sig_indices])
                top_order = np.argsort(-abs_d)[:TOP_EFFECTS_PER_LAYER]
                top_indices = sig_indices[top_order]

                std = np.sqrt(acc.M2 / (acc.n - 1))
                top_effects = []
                for idx in top_indices:
                    top_effects.append({
                        'target_feature_idx': int(idx),
                        'mean_delta': float(acc.mean[idx]),
                        'std_delta': float(std[idx]),
                        'cohens_d': float(cohens_d[idx]),
                        'consistency': float(consistency[idx]),
                        'n_cells_measured': int(acc.n),
                    })
            else:
                top_effects = []

            downstream_effects[str(dl)] = {
                'n_significant': n_sig,
                'n_cells_measured': int(acc.n),
                'top_effects': top_effects,
            }
            attenuation_curve.append(n_sig)

        max_sig = max(attenuation_curve) if attenuation_curve else 1
        attenuation_curve_norm = [v / max_sig for v in attenuation_curve] if max_sig > 0 else attenuation_curve

        result = {
            'source_feature_idx': fi,
            'source_label': sf['label'],
            'source_n_ontologies': sf['n_ontologies'],
            'source_n_annotations': sf['n_annotations'],
            'source_activation_freq': sf['activation_freq'],
            'n_cells_with_activity': n_cells_active[fi],
            'downstream_effects': downstream_effects,
            'attenuation_curve_raw': attenuation_curve,
            'attenuation_curve_normalized': attenuation_curve_norm,
            'total_significant_edges': total_sig_edges,
        }
        feature_results.append(result)

        print(f"  Feature {fi} ({sf['label'][:40]}): "
              f"active in {n_cells_active[fi]} cells, "
              f"{total_sig_edges} sig edges")

    # Final save
    total_elapsed = time.time() - total_t0
    output = {
        'source_layer': source_layer,
        'config': {
            'n_source_features': len(selected_features),
            'n_cells': n_cells,
            'downstream_layers': downstream_layers,
            'cohens_d_threshold': COHENS_D_THRESHOLD,
            'consistency_threshold': CONSISTENCY_THRESHOLD,
        },
        'stats': {
            'n_clean_passes': n_clean_passes,
            'n_ablated_passes': n_ablated_passes,
            'total_forward_passes': n_clean_passes + n_ablated_passes,
        },
        'features': feature_results,
        'total_compute_time_sec': total_elapsed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    with open(final_path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)

    if os.path.exists(partial_path):
        os.remove(partial_path)

    print(f"  Source layer {source_layer} complete: {total_elapsed/60:.1f} min")
    print(f"  Forward passes: {n_clean_passes} clean + {n_ablated_passes} ablated = "
          f"{n_clean_passes + n_ablated_passes}")
    print(f"  Saved: {final_path}")

    return final_path


def _save_partial_v2(path, source_layer, cells_completed, feature_indices,
                     accumulators, n_cells_active, n_cells):
    """Save incremental accumulator state for resume capability."""
    acc_state = []
    for fi in feature_indices:
        fi_accs = {}
        for dl, acc in accumulators[fi].items():
            fi_accs[str(dl)] = {
                'n': int(acc.n),
                'mean': acc.mean.tolist(),
                'M2': acc.M2.tolist(),
                'pos_count': acc.pos_count.tolist(),
            }
        acc_state.append({
            'feature_idx': fi,
            'n_cells_active': n_cells_active[fi],
            'accumulators': fi_accs,
        })

    output = {
        'source_layer': source_layer,
        'cells_completed': cells_completed,
        'n_cells': n_cells,
        'accumulator_state': acc_state,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(path, 'w') as f:
        json.dump(output, f, indent=2, default=_json_default)
    print(f"    Partial save at cell {cells_completed}: {path}")


# ============================================================
# Post-compute analysis (model-agnostic; verbatim except annotation I/O paths)
# ============================================================

def build_circuit_graph(out_dir, source_layers):
    """Aggregate significant edges into a circuit graph."""
    print("\n  Building circuit graph...")

    all_edges = []
    feature_labels = {}  # (layer, idx) -> label

    for sl in source_layers:
        path = os.path.join(out_dir, f"circuit_L{sl:02d}_features.json")
        if not os.path.exists(path):
            print(f"    Skipping L{sl} — not found")
            continue

        with open(path) as f:
            data = json.load(f)

        for feat in data['features']:
            feature_labels[(sl, feat['source_feature_idx'])] = feat['source_label']

            for dl_str, effects in feat['downstream_effects'].items():
                dl = int(dl_str)
                for eff in effects.get('top_effects', []):
                    all_edges.append({
                        'source_layer': sl,
                        'source_feature': feat['source_feature_idx'],
                        'source_label': feat['source_label'],
                        'target_layer': dl,
                        'target_feature': eff['target_feature_idx'],
                        'cohens_d': eff['cohens_d'],
                        'consistency': eff['consistency'],
                        'mean_delta': eff['mean_delta'],
                    })

    print(f"    Total significant edges: {len(all_edges)}")

    in_degree = {}
    out_degree = {}
    for e in all_edges:
        src = f"L{e['source_layer']}_F{e['source_feature']}"
        tgt = f"L{e['target_layer']}_F{e['target_feature']}"
        out_degree[src] = out_degree.get(src, 0) + 1
        in_degree[tgt] = in_degree.get(tgt, 0) + 1

    top_out = sorted(out_degree.items(), key=lambda x: -x[1])[:20]
    top_in = sorted(in_degree.items(), key=lambda x: -x[1])[:20]

    graph = {
        'n_edges': len(all_edges),
        'n_source_features': len(set(f"L{e['source_layer']}_F{e['source_feature']}" for e in all_edges)),
        'n_target_features': len(set(f"L{e['target_layer']}_F{e['target_feature']}" for e in all_edges)),
        'top_hub_sources': [{'feature': k, 'out_degree': v} for k, v in top_out],
        'top_hub_targets': [{'feature': k, 'in_degree': v} for k, v in top_in],
        'edges': all_edges,
    }

    out_path = os.path.join(out_dir, "circuit_graph.json")
    with open(out_path, 'w') as f:
        json.dump(graph, f, indent=2, default=_json_default)
    print(f"    Saved: {out_path}")

    return graph


def analyze_biology(out_dir, run, source_layers):
    """Look for biological motifs in the circuit graph.

    Reads C2S-Scale feature_annotations_L{L:02d}.json from the run dir. Only the layers that
    actually appear in the circuit graph are loaded (the annotation files are ~50-80 MB each).
    """
    print("\n  Analyzing biological motifs...")

    graph_path = os.path.join(out_dir, "circuit_graph.json")
    with open(graph_path) as f:
        graph = json.load(f)

    # Only load annotations for layers referenced by the graph edges (memory-conscious).
    needed_layers = set()
    for edge in graph['edges']:
        needed_layers.add(edge['source_layer'])
        needed_layers.add(edge['target_layer'])

    layer_annotations = {}
    for l in sorted(needed_layers):
        ann_path = os.path.join(run, f"feature_annotations_L{l:02d}.json")
        if os.path.exists(ann_path):
            try:
                with open(ann_path) as f:
                    ann_data = json.load(f)
                layer_annotations[l] = ann_data.get('feature_annotations', {})
            except (json.JSONDecodeError, ValueError) as e:
                print(f"  WARNING: skipping corrupt annotations for L{l:02d} "
                      f"({ann_path}): {e}")

    annotated_edges = []
    shared_ontology_count = 0
    total_annotated_pairs = 0

    for edge in graph['edges']:
        sl, sf = edge['source_layer'], edge['source_feature']
        tl, tf = edge['target_layer'], edge['target_feature']

        src_anns = layer_annotations.get(sl, {}).get(str(sf), [])
        tgt_anns = layer_annotations.get(tl, {}).get(str(tf), [])

        if src_anns and tgt_anns:
            total_annotated_pairs += 1
            src_terms = set(a['term'] for a in src_anns)
            tgt_terms = set(a['term'] for a in tgt_anns)
            shared = src_terms & tgt_terms

            if shared:
                shared_ontology_count += 1

            src_label = edge['source_label']
            tgt_label = "unknown"
            for a in tgt_anns:
                if a['ontology'] in ('GO_BP', 'KEGG', 'Reactome'):
                    tgt_label = a['term']
                    break

            annotated_edges.append({
                'source': f"L{sl}_F{sf}",
                'target': f"L{tl}_F{tf}",
                'source_label': src_label,
                'target_label': tgt_label,
                'shared_terms': list(shared) if shared else [],
                'cohens_d': edge['cohens_d'],
                'n_source_annotations': len(src_anns),
                'n_target_annotations': len(tgt_anns),
            })

    annotated_edges.sort(key=lambda x: (-len(x['shared_terms']), -abs(x['cohens_d'])))

    bio_motifs = {
        'total_edges': len(graph['edges']),
        'total_annotated_pairs': total_annotated_pairs,
        'shared_ontology_edges': shared_ontology_count,
        'shared_ontology_frac': shared_ontology_count / max(total_annotated_pairs, 1),
        'top_interpretable_circuits': annotated_edges[:50],
    }

    print(f"    Annotated edge pairs: {total_annotated_pairs}")
    print(f"    Shared ontology terms: {shared_ontology_count} ({bio_motifs['shared_ontology_frac']:.1%})")

    if annotated_edges:
        print(f"    Top circuit examples:")
        for ae in annotated_edges[:5]:
            shared_str = ", ".join(ae['shared_terms'][:3]) if ae['shared_terms'] else "none"
            print(f"      {ae['source']} ({ae['source_label'][:30]}) -> "
                  f"{ae['target']} ({ae['target_label'][:30]}) | "
                  f"d={ae['cohens_d']:.2f} | shared: {shared_str}")

    return bio_motifs


def analyze_attenuation(out_dir, source_layers):
    """Analyze how effects attenuate across layers."""
    print("\n  Analyzing attenuation curves...")

    attenuation_data = {}

    for sl in source_layers:
        path = os.path.join(out_dir, f"circuit_L{sl:02d}_features.json")
        if not os.path.exists(path):
            continue

        with open(path) as f:
            data = json.load(f)

        layer_curves = []
        for feat in data['features']:
            if feat['total_significant_edges'] > 0:
                layer_curves.append({
                    'feature_idx': feat['source_feature_idx'],
                    'label': feat['source_label'],
                    'total_edges': feat['total_significant_edges'],
                    'curve_raw': feat['attenuation_curve_raw'],
                    'curve_normalized': feat['attenuation_curve_normalized'],
                })

        if layer_curves:
            max_len = max(len(c['curve_raw']) for c in layer_curves)
            avg_curve = np.zeros(max_len)
            count = np.zeros(max_len)
            for c in layer_curves:
                for i, v in enumerate(c['curve_raw']):
                    avg_curve[i] += v
                    count[i] += 1
            avg_curve = avg_curve / np.maximum(count, 1)

            attenuation_data[f"L{sl}"] = {
                'n_features_with_effects': len(layer_curves),
                'avg_attenuation_curve': avg_curve.tolist(),
                'features': layer_curves,
            }

            print(f"    L{sl}: {len(layer_curves)} features with effects, "
                  f"avg sig edges per downstream layer: {avg_curve.mean():.1f}")

    return attenuation_data


def run_analysis(out_dir, run, source_layers):
    """Run all post-compute analyses."""
    print("\n" + "=" * 70)
    print("POST-COMPUTE ANALYSIS")
    print("=" * 70)

    graph = build_circuit_graph(out_dir, source_layers)
    # compare_with_pmi is intentionally DROPPED for C2S-Scale — see module docstring.
    # Our coactivation_modules_L{L}.json store WITHIN-layer feature coactivations, not the
    # reference's CROSS-layer PMI dependency files (deps_L{src}_to_L{dst}.json).
    pmi_comparison = {
        'skipped': True,
        'reason': ("No cross-layer PMI dependency files in the C2S-Scale atlas. "
                   "coactivation_modules_L{L}.json holds within-layer feature coactivations "
                   "(top_pmi_edges = [feat_a, feat_b, pmi, coactivation_count] at the SAME "
                   "layer), which is not comparable to cross-layer causal edges."),
    }
    bio_motifs = analyze_biology(out_dir, run, source_layers)
    attenuation = analyze_attenuation(out_dir, source_layers)

    analysis = {
        'source_layers': source_layers,
        'circuit_graph_summary': {
            'n_edges': graph['n_edges'],
            'n_source_features': graph['n_source_features'],
            'n_target_features': graph['n_target_features'],
        },
        'pmi_comparison': pmi_comparison,
        'biological_motifs': bio_motifs,
        'attenuation': attenuation,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }

    out_path = os.path.join(out_dir, "circuit_analysis.json")
    with open(out_path, 'w') as f:
        json.dump(analysis, f, indent=2, default=_json_default)
    print(f"\n  Analysis saved: {out_path}")

    return analysis


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="C2S-Scale causal feature-to-feature circuit tracing")
    parser.add_argument('--run', required=True,
                        help='atlas run dir (holds sae/, feature_catalog_L*.json, '
                             'feature_annotations_L*.json)')
    parser.add_argument('--model', default="vandijklab/C2S-Scale-Gemma-2-2B",
                        help='HF model id (default: vandijklab/C2S-Scale-Gemma-2-2B)')
    parser.add_argument('--h5ad', required=True,
                        help='AnnData with control cells (obs[gene]==non-targeting)')
    parser.add_argument('--source-layers', type=str, default='0,8,16',
                        help='Comma-separated source layers (need catalog+annotations)')
    parser.add_argument('--n-features', type=int, default=30,
                        help='Number of source features per layer (default: 30)')
    parser.add_argument('--n-cells', type=int, default=200,
                        help='Number of cells to trace through (default: 200)')
    parser.add_argument('--out', type=str, default=None,
                        help='Output directory (default: the run dir)')
    parser.add_argument('--available-layers', type=str, default=None,
                        help='Comma-separated layers with SAE models (default: all 0-25). '
                             'Downstream tracing is restricted to these layers.')
    parser.add_argument('--max-genes', type=int, default=MAX_GENES,
                        help=f'Max genes per cell sentence (default: {MAX_GENES})')
    parser.add_argument('--gene-col', default=DEFAULT_GENE_COL)
    parser.add_argument('--control-label', default=DEFAULT_CONTROL_LABEL)
    parser.add_argument('--analysis-only', action='store_true',
                        help='Skip computation, only run analysis on existing results')
    args = parser.parse_args()

    source_layers = [int(x) for x in args.source_layers.split(',')]
    available_layers = None
    if args.available_layers:
        available_layers = [int(x) for x in args.available_layers.split(',')]

    out_dir = args.out or args.run
    os.makedirs(out_dir, exist_ok=True)

    total_t0 = time.time()

    print("=" * 70)
    print("STAGE 2: C2S-SCALE CAUSAL FEATURE-TO-FEATURE CIRCUIT TRACING")
    print(f"  Model: {args.model}")
    print(f"  Run: {args.run}")
    print(f"  Source layers: {source_layers}")
    print(f"  Features per layer: {args.n_features}")
    print(f"  Cells: {args.n_cells}")
    if available_layers:
        print(f"  Available SAE layers: {available_layers}")
    print(f"  Output: {out_dir}")
    print("=" * 70)

    if not args.analysis_only:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        print("\n  Loading C2S-Scale-Gemma-2-2B...")
        t0 = time.time()

        if torch.cuda.is_available():
            device = torch.device("cuda")
            print("    Using CUDA")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
            print("    Using MPS (Apple Silicon GPU)")
        else:
            device = torch.device("cpu")
            print("    Using CPU")

        tok = AutoTokenizer.from_pretrained(args.model)
        if not tok.is_fast:
            raise RuntimeError("A fast tokenizer is required for offset_mapping (gene attribution).")
        max_len = min(getattr(tok, "model_max_length", 8192) or 8192, 8192)

        try:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, dtype=torch.bfloat16, device_map="auto",
                output_hidden_states=True)
        except TypeError:
            model = AutoModelForCausalLM.from_pretrained(
                args.model, torch_dtype=torch.bfloat16, device_map="auto",
                output_hidden_states=True)
        model = model.eval()
        print(f"    Model loaded in {time.time()-t0:.1f}s")

        # Tokenize cells (shared across all source layers)
        cells = load_and_tokenize_cells(args.h5ad, tok, args.n_cells, max_len,
                                        gene_col=args.gene_col,
                                        control_label=args.control_label,
                                        max_genes=args.max_genes)

        sae_cache = SAECache(args.run, device)

        for sl in source_layers:
            print(f"\n{'=' * 70}")
            print(f"SOURCE LAYER {sl}")
            print(f"{'=' * 70}")

            selected = select_features(args.run, sl, args.n_features)
            trace_source_layer(sl, selected, cells, model, device,
                               sae_cache, out_dir, args.n_cells,
                               available_layers=available_layers)

            if device.type == 'cuda':
                torch.cuda.empty_cache()
            elif device.type == 'mps':
                torch.mps.empty_cache()
            gc.collect()

    run_analysis(out_dir, args.run, source_layers)

    total_time = time.time() - total_t0
    print(f"\n{'=' * 70}")
    print(f"DONE. Total time: {total_time/60:.1f} min ({total_time/3600:.1f} hrs)")
    print(f"{'=' * 70}")


if __name__ == '__main__':
    main()
