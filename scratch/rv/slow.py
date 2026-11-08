import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"tests"); sys.path.insert(0,".")
import numpy as np, synth
from synth import SIGNS
from test_continuous import chain, templates, run, FPS
r=Recognizer=None
from lingestra.live.recognizer import Recognizer
r=Recognizer(None,templates=templates(["alpha","beta","gamma"]))
def ST(c): return synth.stream(c,tail_s=2.0)[0]
for sp in (0.1,0.08):
  for names in (["alpha"],["alpha","beta"]):
    c,_=chain([(SIGNS[n],dict(speed=sp)) for n in names])
    ev,_=run(r,ST(c)); print(sp,names,len(c)/FPS,[(e.kind,e.label) for e in ev])
# unknown sign for 14 s then known
c,_=chain([(SIGNS["alpha"],dict(speed=.1)),(SIGNS["beta"],{})])
