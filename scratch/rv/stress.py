import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"tests"); sys.path.insert(0,".")
import numpy as np
import synth
from test_continuous import chain, templates, run, signs_of, FPS
from lingestra.live.matcher import PersonalMatcher
from lingestra.live.recognizer import Recognizer
rng=np.random.default_rng(5)
tmpl,keys_of=[],{}
for k in range(50):
    keys=[(float(rng.uniform(.3,.7)),float(rng.uniform(.25,.55)),float(rng.uniform(0,1))) for _ in range(int(rng.integers(3,6)))]
    keys_of[k]=keys
    takes=[synth.sign_frames(keys,noise=.002 if t else 0.,rng=np.random.default_rng(100*k+t),move_s=.45+.05*t) for t in range(3)]
    tmpl.append(dict(id=f"s{k}",label=f"s{k}",frames=takes))
rec=Recognizer(None,templates=tmpl)
seq=[int(x) for x in rng.integers(0,50,size=14)]
ch,_=chain([(keys_of[i],{}) for i in seq],trans_s=.2)
print("len s",len(ch)/FPS)
for dur in (10,30):
    c=ch[:int(dur*FPS)]
    ts=[]
    rec.reset(); 
    # wrap
    orig=rec.matcher.decode_sequence
    def w(*a,**k):
        t=time.perf_counter(); r=orig(*a,**k); ts.append((time.perf_counter()-t)*1000); return r
    rec.matcher.decode_sequence=w
    lat=[]
    ev=[]
    for i,f in enumerate(c):
        t=time.perf_counter(); ev+=rec.push_frame(f,i/FPS); lat.append((time.perf_counter()-t)*1000)
    rec.matcher.decode_sequence=orig
    print(dur,"decodes",len(ts),"mean",np.mean(ts),"max",max(ts),"maxframe",max(lat),"words",len(signs_of(ev)),"ref",len(seq))
# direct 10s and 30s decode
for dur in (10,30):
    c=ch[:int(dur*FPS)]
    t=time.perf_counter(); s=rec.matcher.decode_sequence(c); print("direct",dur,(time.perf_counter()-t)*1000,len(s))
c=ch[:200]
ms=[]
for _ in range(8):
    t=time.perf_counter(); rec.matcher.decode_sequence(c); ms.append((time.perf_counter()-t)*1000)
print("8s",np.mean(ms),max(ms),min(ms)); import os; print(os.getloadavg())
