import sys, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'scratch/cont'); sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, tune
from lingestra.live import matcher as M
D1=pickle.load(open(tune.CACHE,'rb'))
want=eval(sys.argv[1]); take=int(sys.argv[2])
for sd,vocab,seqs in D1:
    m=M.PersonalMatcher(); m.set_templates([dict(id=str(i),label=l,frames=t) for i,(l,t) in enumerate(vocab)])
    for tk,ns,ref,fr in seqs:
        if tk==take and ref==want:
            ep=M.episode_features(fr); print(len(fr))
            print('part',[(a,b,m.labels[li],round(d,2)) for a,b,li,d in m._partition(ep,1.0)])
            half=20; X,Mk=M.local_xm(ep,half)
            Xc,Mc,rg=M.collapse_holds(X,Mk,m.cfg.eps,m.cfg.c_miss,return_ranges=True)
            print('groups',[(int(a),int(b)) for a,b in rg if b-a>5])
            print('hh', np.round(ep.hh.max(1)[::6],2))
            m.debug=[]; segs=m.decode_sequence(fr); print(m.debug)
