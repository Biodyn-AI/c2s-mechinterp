"""Build full, versioned gene-set libraries for feature annotation.

Sources (all public, no login):
  GO biological process : Enrichr library GO_Biological_Process_2026
  KEGG pathways         : Enrichr library KEGG_2026
  Reactome pathways     : Enrichr library Reactome_Pathways_2024
  STRING                : v12.5 human protein links, combined score >= 700, mapped to gene symbols
  TRRUST                : v2 human TF -> target table

Gene sets are stored in full (no restriction to any gene list). The annotation scorer restricts them
to the gene universe of the data at scoring time.

Usage: python build_ontologies.py <out_dir> [<tmp_dir>]
"""
import gzip
import hashlib
import json
import os
import sys
import time
import urllib.request

OUT = sys.argv[1]
TMP = sys.argv[2] if len(sys.argv) > 2 else OUT
os.makedirs(OUT, exist_ok=True)
os.makedirs(TMP, exist_ok=True)

ENRICHR = "https://maayanlab.cloud/Enrichr/geneSetLibrary?mode=text&libraryName={}"
LIBS = {"go_bp": "GO_Biological_Process_2026", "kegg": "KEGG_2026", "reactome": "Reactome_Pathways_2024"}
STRING_VERSION = "12.5"
STRING_LINKS = "https://stringdb-downloads.org/download/protein.links.v{v}/9606.protein.links.v{v}.txt.gz"
STRING_INFO = "https://stringdb-downloads.org/download/protein.info.v{v}/9606.protein.info.v{v}.txt.gz"
TRRUST = "https://www.grnpedia.org/trrust/data/trrust_rawdata.human.tsv"
STRING_MIN_SCORE = 700


def fetch(url, path, tries=3):
    for i in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (research script)"})
            with urllib.request.urlopen(req, timeout=300) as r, open(path, "wb") as f:
                while True:
                    b = r.read(1 << 20)
                    if not b:
                        break
                    f.write(b)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  attempt {i + 1} failed for {url}: {e}")
            time.sleep(5 * (i + 1))
    raise SystemExit(f"could not download {url}")


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


versions = {"built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "sources": {}}

# ---- Enrichr gene-set libraries -------------------------------------------------------------
for key, lib in LIBS.items():
    raw = os.path.join(TMP, f"{lib}.txt")
    url = ENRICHR.format(lib)
    fetch(url, raw)
    sets = {}
    with open(raw, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            name = parts[0].strip()
            genes = sorted({g.split(",")[0].strip().upper() for g in parts[2:] if g.strip()})
            if name and genes:
                sets[name] = genes
    out = os.path.join(OUT, f"{key}.json")
    with open(out, "w") as f:
        json.dump(sets, f)
    n_genes = len({g for v in sets.values() for g in v})
    versions["sources"][key] = {"library": lib, "url": url, "n_terms": len(sets), "n_genes": n_genes,
                                "raw_md5": md5(raw), "file": os.path.basename(out), "file_md5": md5(out)}
    print(f"{key}: {lib}: {len(sets)} terms, {n_genes} genes")

# ---- STRING ---------------------------------------------------------------------------------
links = os.path.join(TMP, f"9606.protein.links.v{STRING_VERSION}.txt.gz")
info = os.path.join(TMP, f"9606.protein.info.v{STRING_VERSION}.txt.gz")
fetch(STRING_LINKS.format(v=STRING_VERSION), links)
info_version = STRING_VERSION
try:
    fetch(STRING_INFO.format(v=STRING_VERSION), info, tries=1)
except SystemExit:
    info_version = "12.0"
    info = os.path.join(TMP, "9606.protein.info.v12.0.txt.gz")
    fetch(STRING_INFO.format(v="12.0"), info)
id2sym = {}
with gzip.open(info, "rt") as f:
    next(f)
    for line in f:
        p = line.rstrip("\n").split("\t")
        id2sym[p[0]] = p[1].upper()
pairs = set()
n_lines = 0
with gzip.open(links, "rt") as f:
    next(f)
    for line in f:
        a, b, s = line.split()
        n_lines += 1
        if int(s) >= STRING_MIN_SCORE:
            ga, gb = id2sym.get(a), id2sym.get(b)
            if ga and gb and ga != gb:
                pairs.add((ga, gb) if ga < gb else (gb, ga))
out = os.path.join(OUT, f"string_v{STRING_VERSION}_score{STRING_MIN_SCORE}_pairs.tsv.gz")
with gzip.open(out, "wt") as f:
    for a, b in sorted(pairs):
        f.write(f"{a}\t{b}\n")
versions["sources"]["string"] = {
    "version": STRING_VERSION, "info_version": info_version, "min_combined_score": STRING_MIN_SCORE,
    "url_links": STRING_LINKS.format(v=STRING_VERSION), "links_md5": md5(links),
    "n_link_rows": n_lines, "n_pairs": len(pairs),
    "n_genes": len({g for p in pairs for g in p}), "file": os.path.basename(out), "file_md5": md5(out)}
print(f"string v{STRING_VERSION}: {len(pairs)} gene pairs at score >= {STRING_MIN_SCORE}")

# ---- TRRUST ---------------------------------------------------------------------------------
tr = os.path.join(OUT, "trrust_human.tsv")
fetch(TRRUST, tr)
rows = [l.rstrip("\n").split("\t") for l in open(tr) if l.strip()]
versions["sources"]["trrust"] = {"version": "v2", "url": TRRUST, "n_rows": len(rows),
                                 "n_tfs": len({r[0] for r in rows}), "n_targets": len({r[1] for r in rows}),
                                 "file": "trrust_human.tsv", "file_md5": md5(tr)}
print(f"trrust v2: {len(rows)} rows, {len({r[0] for r in rows})} TFs")

with open(os.path.join(OUT, "VERSIONS.json"), "w") as f:
    json.dump(versions, f, indent=2)
print("wrote", os.path.join(OUT, "VERSIONS.json"))
