import sys; sys.path.insert(0,'scratch/cont')
from syn import *
rec=Recognizer(None,templates=templates(["alpha","beta","gamma"]))
ms=rec.matcher
ch,spans=chain([(SIGNS[n],{}) for n in ["alpha","delta","beta"]],trans_s=0.25)
tail=ch[72-32:]
print(len(tail))
for fin in (True,False):
    ep=M.episode_features(tail) if False else None
from lingestra.live import matcher as M
ep=M.episode_features(tail)
for fin in (True,False):
    print(fin,[(a,b,ms.labels[l],round(d,2)) for a,b,l,d in ms._partition(ep,1.0,fin)])
print([(s.label,s.kind,s.start,s.end,round(s.ratio,2)) for s in ms.decode_sequence(tail,final=False)])
import numpy as np
cfg=ms.cfg
orig=np.argmin
print(cfg.unexplained_tail, cfg.unexplained)
ms.debug=[]; ms.decode_sequence(tail,final=False); print(ms.debug)
