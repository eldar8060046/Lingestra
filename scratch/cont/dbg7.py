import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, benchmark_continuous as C, benchmark_recognition as B
from lingestra import features as F
from lingestra.live import matcher as M
pool=B.load_pool(); sd=2001; take=0; ns=2
vocab=C.make_vocab(pool,30,sd)
rec=C.build_rec(vocab,True); ms=rec.matcher
cand=np.array(list(range(len(vocab))))
rng=np.random.default_rng(sd*7+ns+10*take)
while True:
    pick=cand[rng.permutation(len(cand))[:ns]]
    spans=[F.trim_active(vocab[i][1][take]) for i in pick]; ref=[vocab[i][0] for i in pick]
    st,_=C.build_sequence(spans,vocab[pick[0]][1][take][0],vocab[pick[-1]][1][take][-1],rng)
    if len(F.segment_signing(st))!=1: continue
    if ref==['Mirror','Iron']: break
ep=F.segment_signing(st)[0]
fr=ep.frames
print(len(fr),[len(s) for s in spans])
e=M.episode_features(fr)
for n in (174,180,len(fr)):
    sub=M.episode_features(fr[:n])
    for fin in (False,True):
        print(n,fin,[(a,b,ms.labels[l],round(d,2)) for a,b,l,d in ms._partition(sub,1.0,fin)])
    ms.debug=[]; r=ms.decode_sequence(fr[:n],final=True); print([(s.label,s.kind,s.start,s.end,round(s.ratio,2)) for s in r]); print(ms.debug)
