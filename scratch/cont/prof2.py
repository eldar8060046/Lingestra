import sys, warnings, cProfile, pstats, time
warnings.filterwarnings("ignore")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, benchmark_continuous as C, benchmark_recognition as B
from lingestra import features as F
from lingestra.live import matcher as M
pool=[(l,t) for l,t in B.load_pool() if len(t)>=2]
elig=pool[:50]
m=M.PersonalMatcher(); m.set_templates([dict(id=str(i),label=l,frames=[t[0],t[1],t[0][::2]]) for i,(l,t) in enumerate(elig)])
print('dense lens', np.percentile(m._dlens,[10,50,90,100]), m._DX.shape)
rng=np.random.default_rng(7)
eps=[]
for r in range(6):
    pick=rng.permutation(len(elig))[:4]
    spans=[F.trim_active(elig[i][1][1]) for i in pick]
    st,_=C.build_sequence(spans,elig[pick[0]][1][1][0],elig[pick[-1]][1][1][-1],rng)
    ep=F.segment_signing(st)[0].frames
    while len(ep)<200:
        spans.append(spans[len(spans)%4]); st,_=C.build_sequence(spans,elig[pick[0]][1][1][0],elig[pick[-1]][1][1][-1],rng); ep=F.segment_signing(st)[0].frames
    eps.append(ep[:200])
for ep in eps:
    ts=[]
    for _ in range(3):
        t=time.perf_counter(); segs=m.decode_sequence(ep); ts.append((time.perf_counter()-t)*1000)
    print(round(min(ts)), len(segs), round(m.last_decode_ms))
pr=cProfile.Profile(); pr.enable(); m.decode_sequence(eps[0]); pr.disable()
pstats.Stats(pr).sort_stats('tottime').print_stats(8)
