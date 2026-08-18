"""Does the model's +0.0153 survive matching HOW the 512 genes are chosen?

The model's cell sentence takes the top-512 of the cell's FULL panel (K562 6,546 / RPE1 8,749). The
expr_512_rank baseline took the top-512 of the 6,544 SHARED genes. Those are different gene sets: on RPE1 the
model's sentence can contain ~2,205 genes the baseline never sees, and — more subtly — their presence shifts
the ranks of the shared genes that both arms do use. That asymmetry favours the model and is large enough to
manufacture a +0.015 gap on its own, so the 'MODEL ADDS' verdict cannot stand until it is matched.

FULLSEL arm: choose the top-512 from the FULL panel exactly as the sentence does, then expose to the ridge only
the shared-gene columns, carrying the ranks the full-panel ordering gave them. Same selection, same information.
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
src=anndata.read_h5ad(K); tgt=anndata.read_h5ad(WS+"/data/rpe1_substrate.h5ad")
ts,_,_,_=phase_angle_oriented(src); tt,_,_,_=phase_angle_oriented(tgt)
si=np.load(WS+"/data/act_k562/row_cell_ids.npy"); ti=np.load(WS+"/data/act_rpe1/row_cell_ids.npy")
Hs=np.load(WS+"/data/act_k562/layer_21_activations.npy").astype(np.float64)
Ht=np.load(WS+"/data/act_rpe1/layer_21_activations.npy").astype(np.float64)
ts=ts[si][:len(Hs)]; Hs=Hs[:len(ts)]; tt=tt[ti][:len(Ht)]; Ht=Ht[:len(tt)]
sv=np.char.upper(np.asarray(src.var_names).astype(str)); tv=np.char.upper(np.asarray(tgt.var_names).astype(str))
shared=sorted(set(sv)&set(tv)); sx={g:i for i,g in enumerate(sv)}; tx={g:i for i,g in enumerate(tv)}
FS=(src.X.toarray() if sp.issparse(src.X) else np.asarray(src.X))[si][:len(ts)]
FT=(tgt.X.toarray() if sp.issparse(tgt.X) else np.asarray(tgt.X))[ti][:len(tt)]
sc_i=np.array([sx[g] for g in shared]); tc_i=np.array([tx[g] for g in shared])
def rank_shared_only(F,cols,k=512):
    """top-k from the FULL panel, then keep only `cols`, carrying full-panel ranks."""
    Y=np.zeros((len(F),len(cols)))
    pos={c:j for j,c in enumerate(cols)}
    for i in range(len(F)):
        idx=np.argpartition(-F[i],k)[:k]
        order=idx[np.argsort(-F[i][idx])]
        for r,g in enumerate(order):
            j=pos.get(g)
            if j is not None: Y[i,j]=k-r
    return Y
def shared_then_rank(F,cols,k=512):
    X=F[:,cols]; Y=np.zeros_like(X)
    for i in range(len(X)):
        idx=np.argpartition(-X[i],k)[:k]
        Y[i,idx[np.argsort(-X[i][idx])]]=np.arange(k,0,-1)
    return Y
def Rd(a,b): return float(abs(np.mean(np.exp(1j*(a-b))))) 
def me(a,b):
    o=np.angle(np.mean(np.exp(1j*(b-a)))); return float(np.degrees(np.median(np.abs(np.angle(np.exp(1j*(a-b+o)))))))
def frozen(A,B):
    Y=np.column_stack([np.cos(ts),np.sin(ts)]); s=StandardScaler().fit(A)
    P=Ridge(alpha=1e3).fit(s.transform(A),Y).predict(s.transform(B)); return np.arctan2(P[:,1],P[:,0])
preds={"model":frozen(Hs,Ht),
       "expr_512_rank (shared-then-top512)":frozen(shared_then_rank(FS,sc_i),shared_then_rank(FT,tc_i)),
       "expr_512_rank_FULLSEL (top512-then-shared)":frozen(rank_shared_only(FS,sc_i),rank_shared_only(FT,tc_i))}
print(f"K562 panel {FS.shape[1]} | RPE1 panel {FT.shape[1]} | shared {len(shared)}")
print(f"\n{'arm':<44}{'R_diff':>9}{'med err':>10}")
for k,v in preds.items(): print(f"{k:<44}{Rd(v,tt):>9.3f}{me(v,tt):>9.0f}d")
rng=np.random.default_rng(0); n=len(tt); NB=5000
bs={k:np.empty(NB) for k in preds}
for b in range(NB):
    i=rng.integers(0,n,n)
    for k,v in preds.items(): bs[k][b]=Rd(v[i],tt[i])
print(f"\nPAIRED CONTRASTS ({NB} draws)")
for a,b in [("model","expr_512_rank (shared-then-top512)"),("model","expr_512_rank_FULLSEL (top512-then-shared)")]:
    d=bs[a]-bs[b]; lo,hi=np.percentile(d,[2.5,97.5])
    verdict="MODEL ADDS" if lo>0 else ("MODEL LOSES" if hi<0 else "PARITY (CI spans 0)")
    print(f"  model - {b.split(' ')[0]:<22} {d.mean():+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  -> {verdict}")
    print(f"      ({b})")
