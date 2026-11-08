import sys; sys.path.insert(0,'scratch/cont')
from syn import *
rec=Recognizer(None,templates=templates(["alpha","beta"]))
ms=rec.matcher
cands={'delta':SIGNS['delta'],'gamma':SIGNS['gamma'],'eps1':[(0.30,0.30,0.0),(0.70,0.30,1.0),(0.70,0.55,0.0),(0.30,0.55,1.0)],
 'eps2':[(0.35,0.55,1.0),(0.35,0.28,0.0),(0.68,0.28,1.0),(0.68,0.55,0.0),(0.35,0.55,1.0)],
 'eps3':[(0.5,0.3,0.5),(0.3,0.5,0.0),(0.7,0.5,1.0)]}
for k,keys in cands.items():
    sg=synth.sign_frames(keys)
    r=ms.classify(sg,trim=False); print(k,r.kind,r.label,round(r.distance,2),round(r.threshold,2))
    ch,spans=chain([(SIGNS['alpha'],{}),(keys,{}),(SIGNS['beta'],{})],trans_s=0.25)
    print('  ',[(s.label,s.kind,s.start,s.end,round(s.ratio,2)) for s in ms.decode_sequence(ch)],spans)
