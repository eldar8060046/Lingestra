import sys, pickle, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'scratch/cont'); sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, tune
from lingestra.live import matcher as M
D1=pickle.load(open(tune.CACHE,'rb'))
want=[eval(a) for a in sys.argv[1:]]
for sd,vocab,seqs in D1:
    m=M.PersonalMatcher(); m.set_templates([dict(id=str(i),label=l,frames=t) for i,(l,t) in enumerate(vocab)])
    for take,ns,ref,fr in seqs:
        if take==0 and ref in want:
            ep=M.episode_features(fr)
            print(ref, len(fr))
            part=m._partition(ep,1.0)
            print(' partition',[(a,b,m.labels[li],round(d,2)) for a,b,li,d in part])
            # true spans: single-sign matching of each ref label distance for whole
            m.debug=[]
            for seg in m.decode_sequence(fr): print('  seg',seg.label,seg.kind,seg.start,seg.end,round(seg.ratio,2))
            print(m.debug)
