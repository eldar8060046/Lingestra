import sys, numpy as np, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'.')
from lingestra import features as F, labels as L
from lingestra.live.matcher import PersonalMatcher
pool=[]
for lab,ts in sorted(L.takes_by_label().items()):
    tk=[e['id'] for e in ts if L.clip_quality(e['id'])[1]]
    if len(tk)>=2: pool.append((lab,tk))
print(len(pool))
ld=lambda i: np.load(f'data/landmarks/{i}.npy')
import itertools
rng=np.random.default_rng(1)
good=[]
for lab,tk in pool[:60]:
    m=PersonalMatcher(); m.set_templates([dict(id='a',label=lab,frames=[ld(tk[0])])])
    d,_=m.distances(ld(tk[1]))
    good.append((float(d[0]),lab,tk))
good.sort()
for g in good[:15]: print(g)
