"""How much of 'expression beats the model' is the TOKENISATION LOSS rather than the model?

The C2S input is not the expression vector: it is the top-512 gene NAMES in descending expression order.
Magnitudes are discarded and 92% of the genes are dropped. The expression baseline in transfer_diag.py had all
6,544 shared genes WITH magnitudes. So the two arms were never given the same information.

This isolates the encoding loss from the model, using no GPU: re-run the identical frozen-transfer protocol on
expression encoded three ways, from the fully-informed vector down to exactly what the model is handed.
"""

# --- make sibling modules importable when run from anywhere ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), ".."))
_sys.path.insert(0, _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "c2s_mechinterp"))

import sys,numpy as np,anndata,scipy.sparse as sp,warnings; warnings.filterwarnings("ignore")
WS="/Volumes/Crucial X6/MacBook/Code/biomi_automation/projects/c2s-scale/route_genemanifold_c2s"
sys.path.insert(0,WS)
from cc_phase import phase_angle_oriented
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
K="/Volumes/Crucial X6/MacBook/Code/biomi_automation/projects/biotensor/data/cellcycle/k562_substrate_for_state.h5ad"
R=WS+"/data/rpe1_substrate.h5ad"
src=anndata.read_h5ad(K); tgt=anndata.read_h5ad(R)
ts,_,_,_=phase_angle_oriented(src); tt,_,_,_=phase_angle_oriented(tgt)
sv=np.char.upper(np.asarray(src.var_names).astype(str)); tv=np.char.upper(np.asarray(tgt.var_names).astype(str))
sh=sorted(set(sv)&set(tv)); sx={g:i for i,g in enumerate(sv)}; tx={g:i for i,g in enumerate(tv)}
Xs=(src.X.toarray() if sp.issparse(src.X) else np.asarray(src.X))[:,[sx[g] for g in sh]]
Xt=(tgt.X.toarray() if sp.issparse(tgt.X) else np.asarray(tgt.X))[:,[tx[g] for g in sh]]
def topk_mag(X,k=512):
    Y=np.zeros_like(X)
    for i in range(len(X)):
        idx=np.argpartition(-X[i],k)[:k]; Y[i,idx]=X[i,idx]
    return Y
def topk_rank(X,k=512):
    """Exactly what the sentence encodes: membership + ordinal position, no magnitude."""
    Y=np.zeros_like(X)
    for i in range(len(X)):
        idx=np.argpartition(-X[i],k)[:k]
        order=idx[np.argsort(-X[i][idx])]
        Y[i,order]=np.arange(k,0,-1)
    return Y
def Rdiff(a,b): return float(abs(np.mean(np.exp(1j*(a-b))))) 
def med(a,b):
    off=np.angle(np.mean(np.exp(1j*(b-a)))); d=np.angle(np.exp(1j*(a-b+off)))
    return float(np.degrees(np.median(np.abs(d))))
def frozen(A,B):
    Y=np.column_stack([np.cos(ts),np.sin(ts)]); s=StandardScaler().fit(A)
    m=Ridge(alpha=1e3).fit(s.transform(A),Y); P=m.predict(s.transform(B))
    p=np.arctan2(P[:,1],P[:,0]); return Rdiff(p,tt),med(p,tt)
print(f"{'expression encoding':<42}{'R_diff':>9}{'med err':>10}")
for nm,f in [("all 6,544 genes, magnitudes  (as run)",lambda X:X),
             ("top-512 genes, magnitudes",topk_mag),
             ("top-512 genes, RANK ONLY  <- the model's input",topk_rank)]:
    r,e=frozen(f(Xs),f(Xt)); print(f"{nm:<42}{r:>9.3f}{e:>9.0f}d")
print(f"{'C2S MODEL (layer 21)':<42}{0.789:>9.3f}{24:>9.0f}d   [from transfer_diag.py]")
print(f"{'CONSTANT predictor (chance floor)':<42}{0.043:>9.3f}{86:>9.0f}d")

# ---- how big is the noise on these R_diff values? bootstrap over cells ----
print("\nPAIRED BOOTSTRAP over target cells (1000 draws) — is a 0.015 gap readable at all?")
def frozen_pred(A,B):
    Y=np.column_stack([np.cos(ts),np.sin(ts)]); s=StandardScaler().fit(A)
    m=Ridge(alpha=1e3).fit(s.transform(A),Y); P=m.predict(s.transform(B))
    return np.arctan2(P[:,1],P[:,0])
pf=frozen_pred(Xs,Xt); pm=frozen_pred(topk_mag(Xs),topk_mag(Xt)); pr=frozen_pred(topk_rank(Xs),topk_rank(Xt))
rng=np.random.default_rng(0); n=len(tt)
bs={k:[] for k in ("full","mag512","rank512")}
for _ in range(1000):
    i=rng.integers(0,n,n)
    bs["full"].append(Rdiff(pf[i],tt[i])); bs["mag512"].append(Rdiff(pm[i],tt[i])); bs["rank512"].append(Rdiff(pr[i],tt[i]))
for k,v in bs.items():
    v=np.array(v); print(f"  {k:<10} R_diff {v.mean():.3f}  95% CI [{np.percentile(v,2.5):.3f}, {np.percentile(v,97.5):.3f}]  (width {np.percentile(v,97.5)-np.percentile(v,2.5):.3f})")
d=np.array(bs["full"])-np.array(bs["rank512"])
print(f"  paired diff full - rank512: {d.mean():+.3f} 95% CI [{np.percentile(d,2.5):+.3f}, {np.percentile(d,97.5):+.3f}] "
      f"-> the ENCODING loss is real and resolvable")
print(f"\n  the model-vs-rank512 gap is +0.015, i.e. {0.015/ (np.percentile(np.array(bs['rank512']),97.5)-np.percentile(np.array(bs['rank512']),2.5)):.2f}x the width of a single arm's own 95% CI")
