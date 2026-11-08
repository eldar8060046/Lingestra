import sys, time
sys.path.insert(0,'tests'); sys.path.insert(0,'.')
import numpy as np
from test_continuous import *
names=list(SIGNS)
rec=Recognizer(templates=templates(names,takes=2))
seq=[names[i%4] for i in range(240)]
raw,spans=named(seq)
st=synth.stream(raw)[0]
off=LEAD
rec.reset(); ev=[]
for i,f in enumerate(st):
    e=rec.push_frame(f,i/FPS)
    for x in e: ev.append((i,x.kind,x.label))
signs=[x for x in ev if x[1]=='sign']
for k in range(100,108): print(k, spans[k][0]+off, signs[k] if k<len(signs) else None)
print([x for x in ev if x[1]!='sign'])
