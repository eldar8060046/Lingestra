import sys; sys.path.insert(0,'scratch/cont')
from syn import *
rec=Recognizer(None,templates=templates(["alpha","beta","gamma"]))
names=sys.argv[1].split(',')
ch,spans=chain([(SIGNS[n],{}) for n in names]); st,lead,rs,L,low=stream(ch)
orig=rec.matcher.decode_sequence
def dec(tail,fps=25.0,strictness=1.0):
    r=orig(tail,fps=fps,strictness=strictness)
    print('  decode at',rec.n,'n=%d'%len(tail),'cfrom',rec._c_from,[(s.label,s.kind,s.start,s.end,round(s.ratio,2)) for s in r])
    return r
rec.matcher.decode_sequence=dec
rec.reset()
print(spans,lead+rs)
for i,f in enumerate(st):
    for e in rec.push_frame(f,i/FPS):
        if e.kind!='sentence': print(i,e.kind,e.label)
