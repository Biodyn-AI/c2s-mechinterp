"""CONTEXT-FREE battery: does any gene basis place a hypothesis's genes in its annotated geometry?

"Context-free" = the gene token alone, with no cell around it: the model's WEIGHTS (input embedding W_E and,
for MaxToki, the untied lm_head OUTPUT embedding). Compared against the two reference bases that decide whether
any hit is real (gm_lib docstring):
    coexpr -- the gene's expression profile across 7,500 TS cells. THE DATA BASELINE.
    esm2   -- UCE/STATE's frozen ESM2 protein embeddings. THE SEQUENCE CONTROL.

A hypothesis counts as a MODEL result only if a model basis beats BOTH. Beating coexpr but not esm2 = protein
sequence similarity. Beating esm2 but not coexpr = co-expression, which is what route_utility already showed
the models merely re-encode.

Run: ../../.venv/bin/python -u run_contextfree.py
Out: results/contextfree.json
"""
import os, sys, json, warnings; warnings.filterwarnings("ignore")
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import gm_lib as G
import gene_sets as S

MODEL_BASES = ["scgpt_we", "maxtoki_we", "maxtoki_lmhead"]
REF_BASES = ["coexpr", "esm2"]
BASES = MODEL_BASES + REF_BASES
N_PERM = 2000


def run_one(h):
    out = {}
    for b in BASES:
        M, ok = G.subset(b, h["genes"])
        if ok.sum() < 6:
            out[b] = dict(error=f"only {int(ok.sum())} genes found"); continue
        if h["kind"] in ("order", "blob"):
            idx = np.asarray(h["coord"], float)[ok]
            out[b] = G.order_score(M, idx, n_perm=N_PERM)
        elif h["kind"] == "circle":
            ph = np.asarray(h["coord"], float)[ok]
            out[b] = G.circle_score(M, ph, n_perm=N_PERM)
        elif h["kind"] == "grid":
            c1 = np.asarray(h["coord"][0], float)[ok]; c2 = np.asarray(h["coord"][1], float)[ok]
            out[b] = G.grid_score(M, c1, c2, n_perm=N_PERM)
        elif h["kind"] == "antipodal":
            r = G.antipodal_score(b, h["a"], h["b"], h["axis_genes"], h["axis_sign"])
            out[b] = r if r is not None else dict(error=f"{h['a']}/{h['b']} not in basis")
        out[b]["n_found"] = int(ok.sum())
    return out


def main():
    res = {}
    for name, h in S.H.items():
        res[name] = dict(kind=h["kind"], role=h["role"], n=len(h["genes"]), by_basis=run_one(h))
        r = res[name]["by_basis"]
        if h["kind"] == "antipodal":
            cells = "  ".join(f"{b.split('_')[0][:5]}:{r[b]['cos']:+.2f}" for b in BASES if "cos" in r.get(b, {}))
            ax = "  ".join(f"{b.split('_')[0][:5]}ax:{r[b]['axis_rho']:+.2f}"
                           for b in BASES if "axis_rho" in r.get(b, {}))
            print(f"[{name:<24}] cos {cells} | {ax}", flush=True)
        else:
            cells = "  ".join(f"{b.split('_')[0][:5]}:{r[b]['stat']:.2f}(p{r[b]['p']:.3f})"
                              for b in BASES if "stat" in r.get(b, {}))
            print(f"[{name:<24}] {cells}", flush=True)

    # ---- verdicts ----
    print("\n" + "=" * 100)
    print("VERDICT: a model basis must beat BOTH reference bases (coexpr = data, esm2 = sequence)")
    print("=" * 100)
    print(f"  {'hypothesis':<24}{'role':<18}{'best model':>22}{'coexpr':>10}{'esm2':>10}   verdict")
    for name, v in res.items():
        r = v["by_basis"]
        if v["kind"] == "antipodal":
            key, better = "cos", (lambda a, b: a < b)          # antipodal: MORE NEGATIVE cosine is stronger
        else:
            key, better = "stat", (lambda a, b: a > b)
        vals = {b: r[b][key] for b in BASES if r.get(b) and key in r[b]}
        if len(vals) < len(BASES):
            print(f"  {name:<24}{v['role']:<18}{'INCOMPLETE':>22}"); continue
        mb = min(MODEL_BASES, key=lambda b: vals[b]) if v["kind"] == "antipodal" \
            else max(MODEL_BASES, key=lambda b: vals[b])
        m, ce, es = vals[mb], vals["coexpr"], vals["esm2"]
        sig = r[mb].get("p", 1.0) < 0.05 if v["kind"] != "antipodal" else r[mb].get("p_more_negative", 1) < 0.05
        wins = better(m, ce) and better(m, es) and sig
        verdict = "** MODEL STRUCTURE **" if wins else (
            "sequence (esm2)" if better(es, m) and better(es, ce) else
            "co-expression" if better(ce, m) else "null / not significant")
        v["verdict"] = verdict; v["best_model_basis"] = mb
        print(f"  {name:<24}{v['role']:<18}{mb + ' ' + format(m, '+.3f'):>22}{ce:>+10.3f}{es:>+10.3f}   {verdict}")

    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    json.dump(res, open(os.path.join(HERE, "results", "contextfree.json"), "w"), indent=1)
    print("\n[done] -> results/contextfree.json")


if __name__ == "__main__":
    main()
