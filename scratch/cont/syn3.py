import sys; sys.path.insert(0,'scratch/cont')
from syn import *
rec=Recognizer(None,templates=templates(["alpha","beta","gamma"]))
names=sys.argv[1].split(',')
ch,spans=chain([(SIGNS[n],{}) for n in names],trans_s=0.25)
if len(sys.argv)>2:
    g=np.repeat(ch[spans[0][1]][None],25,0); ch=np.concatenate([ch[:spans[0][1]+1],g,ch[spans[0][1]+1:]])
st,lead,rs,L,low=stream(ch,tail_s=2.0)
print(spans, lead+rs)
ms=rec.matcher
print('offline',[(s.label,s.kind,s.start,s.end,round(s.ratio,2)) for s in ms.decode_sequence(ch)])
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
