import sys, time
sys.path.insert(0,'tests'); sys.path.insert(0,'.')
import numpy as np
from test_continuous import *
names=list(SIGNS)
rec=Recognizer(templates=templates(names,takes=2))
for nw in (12,60,240):
    seq=[names[i%4] for i in range(nw)]
    raw,spans=named(seq)
    st=synth.stream(raw)[0]
    rec.reset(); ev=[]; mx=0; tt=[]
    for i,f in enumerate(st):
        t0=time.perf_counter()
        e=rec.push_frame(f,i/FPS); tt.append(time.perf_counter()-t0); ev+=e
    got=[e.label for e in ev if e.kind=='sign']
    print(nw,len(got),got==seq, 'max ms',max(tt)*1e3,'mean',np.mean(tt)*1e3,[ (e.kind) for e in ev if e.kind!='sign'], len(rec.zone._ep))
    if got!=seq: print(got[:30])
for i,(a,b) in enumerate(zip(got,seq)):
    if a!=b: print('diverge at',i,got[i-2:i+3],seq[i-2:i+3]); break
print(len(got))
# per frame growth
