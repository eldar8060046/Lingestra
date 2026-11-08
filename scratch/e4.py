import sys
sys.path.insert(0,'tests'); sys.path.insert(0,'.')
from test_continuous import *
names=list(SIGNS)
rec=Recognizer(templates=templates(names,takes=2)); import sys as _s; rec.MAX_TAIL_S=float(_s.argv[1])
for dx in (0.031,):
    items=[]
    seq=[names[i%4] for i in range(16)]
    for i,n in enumerate(seq):
        keys=SIGNS[n]
        if i==2: keys=[(x+dx,y+dx,c) for x,y,c in keys]
        items.append((keys,{}))
    raw,_=chain(items)
    st=synth.stream(raw)[0]
    ev,at=run(rec,st)
    got=[e.label for e in ev if e.kind=='sign']
    print(dx,len(got),got==seq,[ (e.kind,e.label) for e in ev if e.kind!='sign'], at[:6], at[-3:])
    if got!=seq: print(got)
