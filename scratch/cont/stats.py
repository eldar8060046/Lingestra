import sys; sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, benchmark_recognition as B
from lingestra import features as F
from lingestra.live import matcher as M
cfg=M.MatcherConfig()
pool=[(l,t) for l,t in B.load_pool() if len(t)>=2][:40]
tot=[];nk=[];nf=[]
for l,t in pool:
    raw=F.trim_active(t[0]); X,Mk=F.sign_features(raw)
    Xc,Mc=M.collapse_holds(X,Mk,cfg.eps)
    seg=M.frame_cost(Xc[1:],Mc[1:],Xc[:-1],Mc[:-1])
    tot.append(seg.sum()); nk.append(len(Xc)); nf.append(len(raw))
print(np.percentile(tot,[10,25,50,75,90]).round(2)); print(np.percentile(nk,[10,50,90])); print(np.percentile(nf,[10,50,90]))
