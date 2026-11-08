import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, benchmark_continuous as C, benchmark_recognition as B
from lingestra import features as F
pool=B.load_pool(); vocab=C.make_vocab(pool,30,101)
rec=C.build_rec(vocab,True)
rng=np.random.default_rng(1)
take=int(sys.argv[1])
for lab,t in vocab:
    span=F.trim_active(t[take])
    if len(span)<12: continue
    st,_=C.build_sequence([span],t[take][0],t[take][-1],rng,speeds=False,pauses=False)
    if len(F.segment_signing(st))!=1: continue
    ev,_=C.run_stream(rec,st)
    k=[e for e in ev if e.kind in('sign','clarify','unknown')]
    if len(k)>1 or [e.label for e in ev if e.kind=='sign']!=[lab] and take==0:
        print(lab,len(st),[(e.kind,e.label,round(e.t,2),None if e.window is None else len(e.window)) for e in ev])
