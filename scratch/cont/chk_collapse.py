import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, benchmark_recognition as B
from lingestra import features as F
from lingestra.live import matcher as M
def old(X, Mk, eps, c_miss=1.0):
    n=len(X); starts=[]; a=0
    while a<n:
        starts.append(a); rest=n-a
        c=M.frame_cost(X[a:],Mk[a:],np.broadcast_to(X[a],(rest,X.shape[1])),np.broadcast_to(Mk[a],(rest,M.NB)),c_miss)
        over=np.nonzero(c>eps)[0]; a=a+(int(over[0]) if len(over) else rest)
    return starts
pool=B.load_pool()[:60]; bad=0
for l,t in pool:
    for raw in t[:2]:
        X,Mk=F.sign_features(F.trim_active(raw)); Xc,Mc,rg=M.collapse_holds(X,Mk,0.06,1.0,return_ranges=True)
        if rg[:,0].tolist()!=old(X,Mk,0.06): bad+=1
        X2,M2=F.sign_features(raw); Xc,Mc,rg=M.collapse_holds(X2,M2,0.06,1.0,return_ranges=True)
        if rg[:,0].tolist()!=old(X2,M2,0.06): bad+=1
print('mismatches',bad)
