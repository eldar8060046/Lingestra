import sys; sys.path.insert(0,'scratch/cont')
from syn import *
rec=Recognizer(None,templates=templates(["alpha","beta","gamma","delta"]))
ms=rec.matcher
name=sys.argv[1]
st=stream(synth.sign(name,pause_s=1.2,pause_at=(1,)),tail_s=2.0)[0]
orig=ms.decode_sequence
def dec(tail,fps=25.0,strictness=1.0,final=True):
    r=orig(tail,fps=fps,strictness=strictness,final=final)
    print('  decode at',rec.n,'n=%d'%len(tail),'final',final,'cfrom',rec._c_from,'hold',rec.zone.hold_active,[(s.label,s.kind,s.start,s.end,round(s.ratio,2)) for s in r])
    return r
ms.decode_sequence=dec
rec.reset()
for i,f in enumerate(st):
    for e in rec.push_frame(f,i/FPS):
        if e.kind!='sentence': print(i,e.kind,e.label)
