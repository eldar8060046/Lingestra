import time, numpy as np, glob, sys
from lingestra.live.matcher import PersonalMatcher as Matcher
import inspect
fs = sorted(glob.glob('data/**/*.npy', recursive=True))[:400]
print(len(fs))
T=[]
for i,f in enumerate(fs):
    a=np.load(f)
    if a.ndim!=3 or a.shape[1]!=67: continue
    T.append(dict(id=str(i),label=f"s{i}",frames=[a,a[2:]],fps=[25.0,25.0]))
print(len(T))
m=Matcher()
for n in (50,200,len(T)):
    t=time.time(); m.set_templates(T[:n]); print(n,'cold',time.time()-t)
    t=time.time(); m.set_templates(T[:n]); print(n,'warm',time.time()-t)
