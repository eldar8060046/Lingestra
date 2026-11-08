import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"tests"); sys.path.insert(0,".")
import numpy as np, synth
from test_continuous import named, templates, FPS
from lingestra.live.recognizer import Recognizer
names=["alpha","beta","gamma","alpha","beta"]
c,_=named(names)
st=synth.stream(c,tail_s=2.0)[0]
for mode in ("noop_true","off","off_on"):
    r=Recognizer(None,templates=templates(["alpha","beta","gamma"])); r.reset()
    ev=[]; toggled=False
    for i,f in enumerate(st):
        ev+=r.push_frame(f,i/FPS)
        if not toggled and sum(e.kind=="sign" for e in ev)>=2:
            toggled=True; print(mode,"toggle at",i)
            if mode=="noop_true": r.set_continuous(True)
            elif mode=="off": r.set_continuous(False)
    print(mode,[(e.kind,e.label) for e in ev])
