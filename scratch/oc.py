import sys, numpy as np, collections
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import benchmark_recognition as B
from lingestra.live.matcher import PersonalMatcher, MatcherConfig
pool=B.load_pool(); elig=[(l,t) for l,t in pool if len(t)>=2]
rng=np.random.default_rng(1)
for takes in (1,2):
  fa=[];labs=collections.Counter();n=0;thrs=[]
  for r in range(3):
    idx=rng.permutation(len(elig))[:30]; voc=[elig[i] for i in idx]
    m=PersonalMatcher(); m.set_templates([dict(id=str(i),label=l,frames=t[:takes]) for i,(l,t) in enumerate(voc)])
    thrs+=list(m.thr)
    have={l for l,_ in voc}; oth=[(l,t[0]) for l,t in pool if l not in have]
    for i in rng.permutation(len(oth))[:150]:
        res=m.classify(oth[i][1],trim=True); n+=1
        if res.kind=='sign': fa.append(1); labs[res.label]+=1
  print(takes,'false accept',len(fa)/n,'top labels',labs.most_common(4),'thr mean',np.mean(thrs),np.percentile(thrs,[10,50,90]))
