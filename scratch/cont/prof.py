import sys, warnings, cProfile, pstats, time
warnings.filterwarnings("ignore")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, benchmark_continuous as C, benchmark_recognition as B
from lingestra import features as F
from lingestra.live import matcher as M
pool=[(l,t) for l,t in B.load_pool() if len(t)>=2]
elig=pool[:50]
m=M.PersonalMatcher(); m.set_templates([dict(id=str(i),label=l,frames=[t[0],t[1],t[0][::2]]) for i,(l,t) in enumerate(elig)])
rng=np.random.default_rng(3)
pick=rng.permutation(50)[:6]
spans=[F.trim_active(elig[i][1][1]) for i in pick]
st,_=C.build_sequence(spans,elig[pick[0]][1][1][0],elig[pick[-1]][1][1][-1],rng)
ep=F.segment_signing(st)[0].frames[:200]
print(len(ep), m.n_takes, m._DX.shape)
m.decode_sequence(ep)
ts=[]
for _ in range(5):
    t=time.perf_counter(); segs=m.decode_sequence(ep); ts.append((time.perf_counter()-t)*1000)
print(np.round(ts,1), [(s.label,s.kind) for s in segs])
pr=cProfile.Profile(); pr.enable(); m.decode_sequence(ep); pr.disable()
pstats.Stats(pr).sort_stats('cumtime').print_stats(14)
