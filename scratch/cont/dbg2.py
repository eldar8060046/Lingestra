import sys, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'scratch/cont'); sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, tune
from lingestra.live import matcher as M
D2=pickle.load(open(tune.CACHE2,'rb'))
lab=sys.argv[1]; take=int(sys.argv[2])
for sd,vocab,eps,un in D2:
    m=M.PersonalMatcher(); m.set_templates([dict(id=str(i),label=l,frames=t) for i,(l,t) in enumerate(vocab)])
    for tk,l,fr in eps:
        if l==lab and tk==take:
            ep=M.episode_features(fr)
            print(len(fr), 'partition',[(a,b,m.labels[li],round(d,2)) for a,b,li,d in m._partition(ep,1.0)])
            half=int(round(0.8*25)); X,Mk=M.local_xm(ep,half)
            Xc,Mc,rg=M.collapse_holds(X,Mk,m.cfg.eps,m.cfg.c_miss,return_ranges=True)
            print('groups',len(rg), rg[-6:].tolist())
            Xd,Md,grp=M._arc_step(Xc,Mc,m.cfg.spot_ds,m.cfg.c_miss,None,True); print('pts',len(Xd),grp[-8:])
            m.debug=[]; segs=m.decode_sequence(fr); print(m.debug)
