"""Resumable, sequential revision queue. Run on the GPU pod, never on the Mac.

  python run_revision.py --root /workspace/rev1 --list
  nohup python -u run_revision.py --root /workspace/rev1 > queue.log 2>&1 &
  python run_revision.py --root /workspace/rev1 --only 'pool-*'
  python run_revision.py --root /workspace/rev1 --include-optional --only 'geometry-*'

No cloud-resource or credential operations occur here. The Mac-side owner enforces the
budget and pod lifetime. A stage is complete only after a zero exit and verified outputs;
the command and script digest are checked before reusing its completion sentinel.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fnmatch
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time


@dataclass
class Stage:
    name: str
    command: list = field(default_factory=list)
    outputs: list = field(default_factory=list)
    links: list = field(default_factory=list)
    optional: bool = False
    action: str = ""


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False))
    temporary.replace(path)


def fingerprint(stage):
    payload = {"command": [str(x) for x in stage.command], "links": [[str(a), str(b)] for a, b in stage.links], "action": stage.action}
    scripts = [Path(arg) for arg in stage.command if str(arg).endswith(".py") and Path(arg).is_file()]
    payload["scripts"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in scripts}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def valid_outputs(outputs):
    for path in outputs:
        path = Path(path)
        if not path.is_file() or path.stat().st_size == 0:
            return False
        if path.suffix == ".json":
            try:
                json.loads(path.read_text())
            except (ValueError, OSError):
                return False
    return True


def execute_stage(stage, root):
    """Execute one reviewed command; keep a separate attempt record even when it fails."""
    root = Path(root)
    logs = root / "stage_logs"
    logs.mkdir(parents=True, exist_ok=True)
    complete = logs / f"{stage.name}.complete.json"
    digest = fingerprint(stage)
    if complete.exists():
        previous = json.loads(complete.read_text())
        if previous.get("fingerprint") == digest and valid_outputs(stage.outputs):
            print(f"SKIP {stage.name}", flush=True)
            return "skipped"
    # A stale success sentinel must never survive an unsuccessful replacement attempt.
    complete.unlink(missing_ok=True)
    started, disk_before = time.monotonic(), shutil.disk_usage(root).used
    record = {"name": stage.name, "fingerprint": digest, "start_utc": utc_now(),
              "command": [str(arg) for arg in stage.command], "outputs": [str(path) for path in stage.outputs],
              "peak_process_tree_rss_bytes": 0, "peak_device_memory_used_mib": 0, "returncode": None}
    attempt = logs / f"{stage.name}.attempt.json"
    atomic_json(attempt, record)
    print(f"START {stage.name} at {record['start_utc']}", flush=True)
    process = None
    try:
        for destination, source in stage.links:
            destination, source = Path(destination), Path(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            if not source.exists():
                raise FileNotFoundError(f"Required link source is absent: {source}")
            if destination.is_symlink():
                if destination.resolve() != source.resolve():
                    raise ValueError(f"Link points to another input: {destination}")
            elif destination.exists():
                raise ValueError(f"Refusing to replace an existing file with a link: {destination}")
            else:
                destination.symlink_to(source)
        # Outputs may live inside a linked directory. Set up links first so
        # output-parent creation cannot occupy a planned link destination.
        for output in stage.outputs:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
        if stage.action:
            prepare_action(stage.action, root)
        if stage.command:
            environment = dict(os.environ)
            environment.setdefault("PYTHONUNBUFFERED", "1")
            environment.setdefault("OMP_NUM_THREADS", "8")
            environment.setdefault("OPENBLAS_NUM_THREADS", "8")
            with open(logs / f"{stage.name}.log", "a", buffering=1) as log:
                process = subprocess.Popen([str(arg) for arg in stage.command], cwd=root, env=environment,
                                           stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
                while process.poll() is None:
                    try:
                        import psutil
                        parent = psutil.Process(process.pid)
                        rss = sum(child.memory_info().rss for child in [parent] + parent.children(recursive=True) if child.is_running())
                        record["peak_process_tree_rss_bytes"] = max(record["peak_process_tree_rss_bytes"], rss)
                    except Exception:
                        # Child processes may exit while their RSS is sampled; monitoring is optional.
                        pass
                    if shutil.which("nvidia-smi"):
                        try:
                            measured = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=10)
                            if measured.returncode == 0:
                                used = sum(int(value.strip()) for value in measured.stdout.splitlines())
                                record["peak_device_memory_used_mib"] = max(record["peak_device_memory_used_mib"], used)
                        except (OSError, ValueError, subprocess.SubprocessError):
                            pass
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        pass
                record["returncode"] = process.returncode
                if process.returncode != 0:
                    raise RuntimeError(f"{stage.name} exited {process.returncode}; see {logs / (stage.name + '.log')}")
        else:
            record["returncode"] = 0
        if not valid_outputs(stage.outputs):
            raise RuntimeError(f"{stage.name} did not produce valid required outputs: {stage.outputs}")
    except BaseException as error:
        if process is not None and process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
        record["error"] = f"{type(error).__name__}: {error}"
        raise
    finally:
        record["end_utc"] = utc_now()
        record["seconds"] = round(time.monotonic() - started, 3)
        record["disk_used_delta_bytes"] = shutil.disk_usage(root).used - disk_before
        record["disk_free_bytes_end"] = shutil.disk_usage(root).free
        atomic_json(attempt, record)
    atomic_json(complete, record)
    print(f"DONE {stage.name} ({record['seconds']:.1f}s)", flush=True)
    return "completed"


def prepare_action(action, root):
    """Only small, deterministic input transformations; no model execution here."""
    if action == "circuit-inputs":
        original = json.loads((root / "runs/2b_atlas/circuit_L07_features.json").read_text())
        features = [int(row["source_feature_idx"]) for row in original["features"]]
        (root / "designs/circuit_original_features.json").write_text(json.dumps({"7": {"annotated": features, "random": []}}, indent=2))
    elif action == "geometry-inputs":
        import anndata
        import numpy as np
        import scipy.sparse as sparse
        adata = anndata.read_h5ad(root / "data/setty19_cd34_bm.h5ad", backed="r")
        indices = np.sort(np.random.default_rng(0).choice(adata.n_obs, min(4000, adata.n_obs), replace=False))
        subset = adata[indices].to_memory()
        adata.file.close()
        counts = subset.layers["spliced"]
        names = np.asarray(subset.var_names).astype(str)
        rows, cell_ids = [], []
        for ci in range(len(indices)):
            values = counts[ci].toarray().ravel() if sparse.issparse(counts) else np.asarray(counts[ci]).ravel()
            nonzero = np.flatnonzero(values > 0)
            selected = nonzero[np.argsort(-values[nonzero], kind="stable")[:512]]
            rows.extend(names[selected].tolist())
            cell_ids.extend([int(indices[ci])] * len(selected))
        out = root / "designs/geometry_cd34"
        out.mkdir(parents=True, exist_ok=True)
        (out / "row_gene_names.json").write_text(json.dumps(rows))
        np.save(out / "row_cell_ids.npy", np.asarray(cell_ids, dtype=np.int64))
    else:
        raise ValueError(f"Unknown preparation action: {action}")


def build_queue(root, scripts):
    root, scripts = Path(root), Path(scripts)
    code, runs, results, designs = (root / name for name in ("code", "runs", "results", "designs"))
    atlas = runs / "2b_atlas"
    base_rows = atlas / "row_gene_names.json"
    trrust = root / "ontologies_v2/trrust_human.tsv"
    model2, model27 = "vandijklab/C2S-Scale-Gemma-2-2B", "vandijklab/C2S-Scale-Gemma-2-27B"
    queue = []

    def add(name, script, arguments, outputs, optional=False):
        queue.append(Stage(name, [sys.executable, str(script), *map(str, arguments)], [Path(p) for p in outputs], optional=optional))

    def link_run(name, destination, source, layers, checkpoint_source=None, optional=False):
        links = [(destination / filename, source / filename) for filename in ("row_gene_names.json", "row_cell_ids.npy")]
        links += [(destination / f"layer_{layer:02d}_activations.npy", source / f"layer_{layer:02d}_activations.npy") for layer in layers]
        if checkpoint_source:
            links += [(destination / "sae", checkpoint_source / "sae")]
        queue.append(Stage(name, outputs=[link[0] for link in links if link[0].name != "sae"], links=links, optional=optional))

    def annotate(name, run, layers, destination, legacy=False, optional=False):
        for layer in layers:
            filename = destination / f"ann_v2_L{layer:02d}.json"
            add(f"{name}-L{layer:02d}", scripts / "annotate_v2.py",
                ["--catalog", run / f"feature_catalog_L{layer:02d}.json", "--universe", base_rows,
                 "--ont-dir", root / ("data/ontologies" if legacy else "ontologies_v2"), "--out", filename,
                 "--n-floor-seeds", 3, *(["--legacy"] if legacy else [])], [filename], optional)

    def catalogues(prefix, run, layers, optional=False):
        for layer in layers:
            add(f"{prefix}-catalog-L{layer:02d}", code / "feature_catalog.py", ["--run", run, "--layer", layer],
                [run / f"feature_catalog_L{layer:02d}.json"], optional)

    def train(prefix, run, layer, expansion=4, k=32, features=0):
        add(prefix, scripts / "rev_sae_tools.py", ["--code-dir", code, "train", "--run", run,
            "--layer", layer, "--expansion", expansion, "--k", k, "--n-features", features,
            "--epochs", 6, "--batch-size", 4096, "--lr", "0.0003", "--seed", 0],
            [run / f"sae/train_L{layer:02d}.json", run / f"sae/sae_L{layer:02d}.pt"])

    def extract(name, rows, destination, layers, model=model2, dtype="float16", pool="mean", stats=False, optional=False):
        destination.mkdir(parents=True, exist_ok=True)
        statistics = results / f"{name}_residual_stats.npz"
        arguments = ["--code-dir", code, "--model", model, "--rows-dir", rows, "--out", destination,
                     "--layers", *layers, "--store-dtype", dtype, "--pool", pool]
        outputs = [destination / "manifest.json", *[destination / f"layer_{layer:02d}_activations.npy" for layer in layers]]
        if stats:
            arguments += ["--stats-layers", "all", "--stats-cells", 200, "--stats-out", statistics]
            outputs.append(statistics)
        add(name, scripts / "rev_extract.py", arguments, outputs, optional)

    # D1/D2/H: exact saved sentences and original trained dictionaries, all layers.
    extract("atlas-extract", atlas, atlas, list(range(26)), stats=True)
    for layer in range(26):
        output = results / f"2b_eval_L{layer:02d}.json"
        add(f"atlas-eval-L{layer:02d}", scripts / "rev_sae_tools.py", ["--code-dir", code, "eval", "--run", atlas, "--layer", layer, "--out", output], [output])
        catalogues("atlas", atlas, [layer])
        annotate("atlas-annotate", atlas, [layer], results / "annotations/2b")
        add(f"atlas-modules-L{layer:02d}", code / "coactivation.py", ["--run", atlas, "--layer", layer], [atlas / f"coactivation_modules_L{layer:02d}.json"])
    annotate("legacy-annotate", atlas, [0, 13, 22], results / "annotations/2b_legacy", legacy=True)
    annotate("27b-original-annotate", runs / "27b_atlas", list(range(26)), results / "annotations/27b_x4")
    add("circuit-baseline", scripts / "rev_cpu_analysis.py", ["circuits", "--circuits", atlas / "circuit_graph.json",
        "--annotations", results / "annotations/2b", "--out", results / "circuit_baseline.json"], [results / "circuit_baseline.json"])

    # C1-C5: primary exact 48-factor replication plus RPE1, both with measured-input controls.
    original_design = designs / "phase8_k562_original48.json"
    add("phase8-original-design", scripts / "make_original48_design.py", ["--h5ad", root / "data/ReplogleWeissman2022_K562_essential.h5ad",
        "--trrust", trrust, "--old-result", atlas / "phase8v2_L16.json", "--max-genes", 520, "--out", original_design], [original_design])
    for tag, design in (("k562", original_design), ("rpe1", designs / "phase8_rpe1.json")):
        destination = results / f"phase8_{tag}"
        destination.mkdir(parents=True, exist_ok=True)
        add(f"phase8-{tag}-extract", scripts / "rev_phase8_extract.py", ["--code-dir", code, "--design", design,
            "--run", atlas, "--model", model2, "--trrust", trrust, "--layers", 8, 16, 22,
            "--insilico", "--out", destination], [destination / "rows.json", destination / "feat_L16.npz", destination / "insilico.npz"])
        h5ad = root / "data" / ("ReplogleWeissman2022_K562_essential.h5ad" if tag == "k562" else "ReplogleWeissman2022_rpe1.h5ad")
        output = destination / "statistics"
        add(f"phase8-{tag}-stats", scripts / "rev_phase8_stats.py", ["--features-dir", destination,
            "--design", design, "--catalog-dir", atlas, "--h5ad", h5ad, "--trrust", trrust,
            "--layers", 8, 16, 22, "--out", output], [output / "regulatory_statistics.json", output / "manifest.json"])
        add(f"phase8-{tag}-response-biology", scripts / "rev_response_biology.py", ["--stats-dir", output,
            "--annotations-dir", results / "annotations/2b", "--catalog-dir", atlas, "--trrust", trrust,
            "--layers", 8, 16, 22, "--out", output / "response_biology.json"], [output / "response_biology.json"])

    # D3: k=32/expansion4 uses the stored baseline; all other settings are trained anew.
    for layer in (0, 13):
        for k in (8, 16, 64, 128):
            variant = runs / f"sweep_L{layer:02d}_k{k}_x4"
            prefix = f"sweep-L{layer:02d}-k{k}-x4"
            link_run(prefix + "-inputs", variant, atlas, [layer])
            train(prefix + "-train", variant, layer, k=k)
            catalogues(prefix, variant, [layer])
            annotate(prefix + "-annotate", variant, [layer], results / "annotations" / variant.name)
    for expansion in (2, 8):
        variant = runs / f"sweep_L13_k32_x{expansion}"
        prefix = f"sweep-L13-k32-x{expansion}"
        link_run(prefix + "-inputs", variant, atlas, [13])
        train(prefix + "-train", variant, 13, expansion=expansion)
        catalogues(prefix, variant, [13])
        annotate(prefix + "-annotate", variant, [13], results / "annotations" / variant.name)

    # D4/A3: pooling and untrained dictionary floors.
    for pool in ("first", "last"):
        variant = runs / f"pool_{pool}"
        extract(f"pool-{pool}-extract", atlas, variant, [13], pool=pool)
        queue.append(Stage(f"pool-{pool}-checkpoint", links=[(variant / "sae", atlas / "sae")], outputs=[variant / "sae/sae_L13.pt"]))
        catalogues(f"pool-{pool}", variant, [13])
        annotate(f"pool-{pool}-annotate", variant, [13], results / "annotations" / variant.name)
    add("pool-comparison", scripts / "rev_misc.py", ["pool-eval", "--code-dir", code, "--layer", 13,
        "--mean-run", atlas, "--first-run", runs / "pool_first", "--last-run", runs / "pool_last", "--sae-run", atlas,
        "--out", results / "pool_comparison.json"], [results / "pool_comparison.json"])
    for layer in (0, 13, 22):
        variant = runs / f"untrained_L{layer:02d}"
        prefix = f"untrained-L{layer:02d}"
        link_run(prefix + "-inputs", variant, atlas, [layer])
        add(prefix + "-init", scripts / "rev_sae_tools.py", ["--code-dir", code, "untrained", "--sae-run", atlas,
            "--layer", layer, "--out-run", variant], [variant / f"sae/sae_L{layer:02d}.pt"])
        catalogues(prefix, variant, [layer])
        annotate(prefix + "-annotate", variant, [layer], results / "annotations" / variant.name)

    # F3: applying the same K562 dictionaries to independent cell-line samples.
    lines = runs / "lines2"
    extract("lines-extract", designs / "transfer_controls", lines, [0, 13, 22])
    queue.append(Stage("lines-labels", links=[(lines / "cells.json", designs / "transfer_controls/cells.json")], outputs=[lines / "cells.json"]))
    add("lines-eval", scripts / "rev_misc.py", ["lines-eval", "--code-dir", code, "--acts", lines,
        "--sae-run", atlas, "--layers", 0, 13, 22, "--out", results / "lines2"], [results / "lines2.json", results / "lines2_freq.npz"])
    add("biology-summary", scripts / "rev_cpu_analysis.py", ["biology", "--annotations", results / "annotations/2b",
        "--run", atlas, "--sae-from", atlas / "sae/sae_L00.pt", "--sae-to", atlas / "sae/sae_L25.pt",
        "--out", results / "biology.json"], [results / "biology.json", results / "biology.json.matches.npz"])
    add("atlas-gene-means", scripts / "rev_misc.py", ["gene-means", "--acts", atlas, "--layers", 9, 13, 17,
        "--out", results / "atlas_gene_means.npz"], [results / "atlas_gene_means.npz"])

    # E1-E3: same sentences, full relative depth, float32 storage, identical six training epochs.
    scaled = runs / "27b_matched"
    scaled_layers = [0, 7, 14, 24, 31, 38, 45]
    extract("scale-extract", atlas, scaled, scaled_layers, model=model27, dtype="float32", stats=True)
    for layer in scaled_layers:
        train(f"scale-train-L{layer:02d}", scaled, layer, expansion=2, features=9216)
        catalogues("scale", scaled, [layer])
        annotate("scale-annotate", scaled, [layer], results / "annotations/27b_matched")
    add("scale-residual-summary", scripts / "rev_residual_summary.py", ["--stats-2b", results / "atlas-extract_residual_stats.npz",
        "--stats-27b", results / "scale-extract_residual_stats.npz", "--train-2b", atlas / "sae",
        "--eval-2b", results, "--train-27b", scaled / "sae", "--out", results / "residual_summary.json"], [results / "residual_summary.json"])

    # E4 is opt-in. Project selected dimensions only in the existing RAM input; activation
    # files remain untouched. Empty presets write an explicit skip JSON and no checkpoint.
    massive_presets = results / "dropmassive_dimensions"
    add("scale-dropmassive-presets", scripts / "rev_massive_dimensions.py", ["preset", "--stats",
        results / "scale-extract_residual_stats.npz", "--layers", 24, 38, "--out-dir", massive_presets],
        [massive_presets / "presets.json", *[massive_presets / f"dimensions_L{layer:02d}.json" for layer in (24, 38)]], True)
    for layer in (24, 38):
        add(f"scale-dropmassive-train-L{layer:02d}", scripts / "rev_sae_tools.py", ["--code-dir", code,
            "train", "--run", scaled, "--layer", layer, "--n-features", 9216, "--k", 32,
            "--epochs", 6, "--batch-size", 4096, "--lr", "0.0003", "--seed", 0,
            "--sae-subdir", "sae_dropmassive", "--drop-dims", massive_presets / f"dimensions_L{layer:02d}.json"],
            [scaled / "sae_dropmassive" / f"train_L{layer:02d}.json"], True)
    add("scale-dropmassive-comparison", scripts / "rev_massive_dimensions.py", ["compare", "--run", scaled,
        "--layers", 24, 38, "--projected-subdir", "sae_dropmassive", "--out", results / "dropmassive_comparison.json"],
        [results / "dropmassive_comparison.json"], True)

    # These resource-intensive checks are individually selectable after the primary queue.
    queue.append(Stage("circuit-rerun-inputs", outputs=[designs / "circuit_original_features.json"], optional=True, action="circuit-inputs"))
    for n_cells in (50, 100):
        destination = results / f"circuit_{n_cells}"
        add(f"circuit-rerun-{n_cells}", scripts / "rev_circuit.py", ["--code-dir", code, "--run", atlas,
            "--model", model2, "--cells", designs / "circuit_k562.json", "--features", designs / "circuit_original_features.json",
            "--source-layers", 7, "--n-cells", n_cells, "--out", destination], [destination / "circuit_L07.npz"], True)
    add("circuit-rerun-comparison", scripts / "rev_cpu_analysis.py", ["circuits", "--circuits", results / "circuit_50",
        "--compare", results / "circuit_100", "--annotations", results / "annotations/2b", "--out", results / "circuit_stability.json"], [results / "circuit_stability.json"], True)
    queue.append(Stage("geometry-inputs", outputs=[designs / "geometry_cd34/row_gene_names.json"], optional=True, action="geometry-inputs"))
    geometry = runs / "geometry_cd34"
    extract("geometry-extract", designs / "geometry_cd34", geometry, [9, 13, 17], optional=True)
    add("geometry-gene-means", scripts / "rev_misc.py", ["gene-means", "--acts", geometry, "--layers", 9, 13, 17,
        "--out", results / "cd34_gene_means.npz"], [results / "cd34_gene_means.npz"], True)
    add("geometry-dimensions", scripts / "rev_cpu_analysis.py", ["dimensions", "--h5ad", root / "data/setty19_cd34_bm.h5ad",
        "--counts-layer", "spliced", "--model-vectors", results / "cd34_gene_means.npz", "--layers", 9, 13, 17,
        "--out", results / "cd34_dimensions.json"], [results / "cd34_dimensions.json"], True)
    # G2 is a separately labelled K562 replication. Its model and expression arms share the
    # same saved 1,200 cell indices; the original atlas has no recoverable h5ad row mapping.
    k562_geometry = runs / "geometry_k562_replication"
    extract("geometry-k562-extract", designs / "atlas_k562", k562_geometry, [9, 13, 17], optional=True)
    add("geometry-k562-gene-means", scripts / "rev_misc.py", ["gene-means", "--acts", k562_geometry,
        "--layers", 9, 13, 17, "--out", results / "k562_replication_gene_means.npz"],
        [results / "k562_replication_gene_means.npz"], True)
    add("geometry-k562-dimensions", scripts / "rev_cpu_analysis.py", ["dimensions", "--h5ad",
        root / "data/ReplogleWeissman2022_K562_essential.h5ad", "--cell-indices",
        designs / "atlas_k562/h5ad_cell_indices.json", "--model-vectors", results / "k562_replication_gene_means.npz",
        "--layers", 9, 13, 17, "--out", results / "k562_replication_dimensions.json"],
        [results / "k562_replication_dimensions.json"], True)
    return queue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/workspace/rev1"))
    parser.add_argument("--scripts", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--only", nargs="+", help="Stage glob patterns; required earlier stages must already be complete")
    parser.add_argument("--until", help="Stop after this exact selected stage name")
    parser.add_argument("--include-optional", action="store_true", help="Include massive-dimension ablations, circuit replication and matched-cell geometry stages")
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / "results").mkdir(exist_ok=True)
    queue = build_queue(args.root, args.scripts)
    selected = [stage for stage in queue if (args.include_optional or not stage.optional)
                and (not args.only or any(fnmatch.fnmatch(stage.name, pattern) for pattern in args.only))]
    if not selected:
        parser.error("No stages match the requested selection")
    if args.until and args.until not in [stage.name for stage in selected]:
        parser.error("--until is not among the selected stages")
    if args.list:
        for stage in selected:
            print(stage.name + (" [optional]" if stage.optional else ""))
            if args.until == stage.name:
                break
        return
    def terminate_current_stage(signum, frame):
        raise KeyboardInterrupt("termination requested")
    signal.signal(signal.SIGTERM, terminate_current_stage)
    for stage in selected:
        execute_stage(stage, args.root)
        if args.until == stage.name:
            break


if __name__ == "__main__":
    main()
