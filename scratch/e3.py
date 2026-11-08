import sys
sys.path.insert(0,'tests'); sys.path.insert(0,'.')
from test_continuous import *
names=list(SIGNS)
rec=Recognizer(templates=templates(names,takes=2))
seq=[names[i%4] for i in range(240)]
raw,spans=named(seq)
st=synth.stream(raw)[0]
orig=rec.matcher.decode_sequence
def wrap(tail,**kw):
    r=orig(tail,**kw)
    if 5150<=rec.n<=5430 and rec.n%25==0: print(rec.n,len(tail),rec._c_from,[(s.kind,s.label,s.start,s.end,round(s.ratio,2)) for s in r], rec.zone.hold_active)
    return r
rec.matcher.decode_sequence=wrap
for i,f in enumerate(st): rec.push_frame(f,i/FPS)
