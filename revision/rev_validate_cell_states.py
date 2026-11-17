"""Validate sampled cached cell-summary rows by fresh model extraction on a GPU.

Rows are sampled by their position in the cached matrix, then mapped through row_cell_ids.npy
to the source h5ad. Nearest-neighbour checks search every cached state. This is a sampled
identity/representation check, not proof for unsampled rows. Mismatches are saved before exit 2.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np


def sample_rows(cell_ids, n_observations, n_cells, seed=0):
    cell_ids = np.asarray(cell_ids)
    if cell_ids.ndim != 1 or not len(cell_ids) or not np.issubdtype(cell_ids.dtype, np.integer):
        raise ValueError("Cached row identities must be a nonempty integer vector")
    if len(np.unique(cell_ids)) != len(cell_ids) or cell_ids.min() < 0 or cell_ids.max() >= n_observations:
        raise ValueError("Cached row identities are duplicated or outside the h5ad")
    if n_cells < 1:
        raise ValueError("n_cells must be positive")
    positions = np.sort(np.random.default_rng(seed).choice(len(cell_ids), min(n_cells, len(cell_ids)), replace=False))
    return [(int(position), int(cell_ids[position])) for position in positions]


def compare_state(fresh, cached, expected_row, min_cosine=0.9999, max_relative_l2=0.02):
    fresh, cached = np.asarray(fresh, dtype=np.float64), np.asarray(cached, dtype=np.float64)
    if cached.ndim != 2 or fresh.shape != (cached.shape[1],) or not 0 <= expected_row < len(cached):
        raise ValueError("Fresh/cached state shapes or expected row do not align")
    if not np.isfinite(fresh).all() or not np.isfinite(cached).all():
        raise ValueError("State vectors must be finite")
    norms, fresh_norm = np.linalg.norm(cached, axis=1), np.linalg.norm(fresh)
    if fresh_norm == 0 or np.any(norms == 0):
        raise ValueError("Zero-norm states cannot supply a cosine identity check")
    similarities = (cached @ fresh) / (norms * fresh_norm)
    nearest = int(similarities.argmax())
    reference = cached[expected_row]
    cosine = float(np.clip(similarities[expected_row], -1, 1))
    relative = float(np.linalg.norm(fresh - reference) / norms[expected_row])
    tied_for_nearest = bool(similarities[expected_row] >= similarities[nearest] - 1e-7)
    return {"cosine": cosine, "relative_l2": relative,
            "max_abs_difference": float(np.max(np.abs(fresh - reference))),
            "nearest_stored_row": nearest, "nearest_cosine": float(np.clip(similarities[nearest], -1, 1)),
            "nearest_is_expected": nearest == expected_row, "expected_is_tied_for_nearest": tied_for_nearest,
            "n_nearest_ties": int(np.sum(similarities >= similarities[nearest] - 1e-7)),
            "passes_sample_criteria": bool(cosine >= min_cosine and relative <= max_relative_l2 and tied_for_nearest)}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--h5ad", type=Path, required=True)
    parser.add_argument("--states", type=Path, required=True)
    parser.add_argument("--layer", type=int, default=21)
    parser.add_argument("--n-cells", type=int, default=20)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-cosine", type=float, default=0.9999)
    parser.add_argument("--max-relative-l2", type=float, default=0.02)
    args = parser.parse_args()
    started = time.monotonic()
    import anndata
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available():
        parser.error("Fresh extraction requires a GPU; run this helper on the pod")
    sys.path.insert(0, str(args.code_dir))
    from cell_sentences import anndata_to_ranked_genes, build_cell_sentence
    manifest_path = args.states / "manifest.json"
    ids_path = args.states / "row_cell_ids.npy"
    states_path = args.states / f"layer_{args.layer:02d}_activations.npy"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("unit") != "cell-summary" or manifest.get("max_genes") != 512 or not manifest.get("prompt"):
        raise ValueError("Cached manifest must describe prompted top-512 cell-summary extraction")
    cached = np.load(states_path, mmap_mode="r")
    cell_ids = np.load(ids_path)
    if cached.ndim != 2 or len(cached) != len(cell_ids) or not np.isfinite(cached).all():
        raise ValueError("Cached state rows must align exactly with finite cell identities")
    data = anndata.read_h5ad(args.h5ad, backed="r")
    selected = sample_rows(cell_ids, data.n_obs, args.n_cells, args.seed)
    torch.cuda.reset_peak_memory_stats()
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto").eval()
    if not 0 <= args.layer < model.config.num_hidden_layers or cached.shape[1] != model.config.hidden_size:
        raise ValueError("Cached vector width or requested layer disagrees with the model")
    max_length = min(getattr(tokenizer, "model_max_length", 8192) or 8192, 8192)
    rows = []
    try:
        for state_row, h5ad_row in selected:
            genes = anndata_to_ranked_genes(data, h5ad_row, max_genes=512)
            sentence = build_cell_sentence(genes, organism="human", prompt=True)
            encoded = tokenizer(sentence.text, return_tensors="pt", truncation=False)
            if not genes or encoded["input_ids"].shape[1] > max_length:
                raise ValueError("Empty or truncated cell sentences cannot validate exact cached inputs")
            with torch.no_grad():
                output = model(**{key: value.to(model.device) for key, value in encoded.items()}, output_hidden_states=True)
                fresh = output.hidden_states[args.layer + 1][0, -1].float().cpu().numpy()
            compared = compare_state(fresh, cached, state_row, args.min_cosine, args.max_relative_l2)
            rows.append({"stored_row": state_row, "h5ad_cell_index": h5ad_row,
                         "nearest_h5ad_cell_index": int(cell_ids[compared["nearest_stored_row"]]),
                         "n_genes": len(genes), "n_tokens": int(encoded["input_ids"].shape[1]),
                         "sentence_sha256": hashlib.sha256(sentence.text.encode()).hexdigest(), **compared})
            del output
            print(f"stored row {state_row} -> h5ad {h5ad_row}: cos={compared['cosine']:.8f}, relL2={compared['relative_l2']:.5g}, nearest={compared['nearest_stored_row']}", flush=True)
    finally:
        data.file.close()
    passed = all(row["passes_sample_criteria"] for row in rows)
    inputs = {"h5ad": args.h5ad, "states": states_path, "row_ids": ids_path, "manifest": manifest_path,
              "cell_sentence_code": args.code_dir / "cell_sentences.py", "analysis_code": Path(__file__)}
    record = {"schema": "sampled-cell-summary-cache-validation-v1", "model": args.model,
        "resolved_model_commit": getattr(model.config, "_commit_hash", None),
        "resolved_tokenizer_commit": tokenizer.init_kwargs.get("_commit_hash"),
        "model_config_sha256": hashlib.sha256(model.config.to_json_string().encode()).hexdigest(),
        "cached_manifest": manifest, "layer": args.layer, "compute_dtype": "bfloat16", "pooling": "hidden_states[layer+1][0,-1]",
        "n_cached_rows": len(cached), "n_h5ad_rows": int(data.n_obs), "n_sampled": len(rows),
        "sampled_fraction_of_cached_rows": len(rows) / len(cached), "n_passed": sum(row["passes_sample_criteria"] for row in rows),
        "all_sampled_rows_pass": passed, "rows": rows, "seed": args.seed,
        "criteria": {"minimum_cosine": args.min_cosine, "maximum_relative_l2": args.max_relative_l2,
                     "expected_row_must_be_nearest_or_tied": True, "nearest_cosine_tie_tolerance": 1e-7},
        "coverage_limit": "Only sampled cached rows are re-extracted; this does not establish identity or correctness for unsampled rows",
        "nearest_search": "all cached rows, not just the sampled subset",
        "seconds": time.monotonic() - started, "gpu": torch.cuda.get_device_name(0),
        "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "peak_gpu_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "inputs": {name: {"file": path.name, "sha256": file_sha256(path)} for name, path in inputs.items()},
        "software": {"torch": torch.__version__, "numpy": np.__version__},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.out.with_suffix(args.out.suffix + ".pending")
    temporary.write_text(json.dumps(record, indent=2, allow_nan=False))
    temporary.replace(args.out)
    print(json.dumps({"all_sampled_rows_pass": passed, "n_sampled": len(rows), "out": str(args.out)}), flush=True)
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
