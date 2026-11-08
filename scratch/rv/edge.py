import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,"tests"); sys.path.insert(0,".")
import numpy as np
import synth
from synth import SIGNS
from test_continuous import chain, templates, run, signs_of, FPS, named
from lingestra.live.recognizer import Recognizer
from lingestra.live.matcher import PersonalMatcher
def ST(c): return synth.stream(c,tail_s=2.0)[0]
def summ(ev): return [(e.kind,e.label) for e in ev]
# zero templates
r=Recognizer(None,templates=[])
ch,_=named(["alpha","beta"]); ev,_=run(r,ST(ch)); print("zero",summ(ev))
# one sign
r=Recognizer(None,templates=templates(["alpha"]))
ev,_=run(r,ST(named(["alpha","beta"])[0])); print("one sign, a+b",summ(ev))
ev,_=run(r,ST(named(["alpha"])[0])); print("one sign, a",summ(ev))
ev,_=run(r,ST(named(["alpha","alpha"])[0])); print("one sign, a a",summ(ev))
# NaN streams
r=Recognizer(None,templates=templates(["alpha","beta","gamma"]))
c,_=named(["alpha","beta","gamma"]); s=ST(c)
s2=s.copy(); s2[40:60]=np.nan; ev,_=run(r,s2); print("nan gap",summ(ev))
s3=s.copy(); s3[:, 15:, :]=np.nan; 
try: ev,_=run(r,s3); print("allnan hands",summ(ev))
except Exception as e: print("ERR",repr(e))
s4=s.copy(); s4[::3,:,:]=np.nan
try: ev,_=run(r,s4); print("nan every 3rd",summ(ev))
except Exception as e: print("ERR",repr(e))
s5=np.full_like(s,np.nan)
try: ev,_=run(r,s5); print("allnan",summ(ev))
except Exception as e: print("ERR",repr(e))
# very slow / fast
for sp in (0.25,0.4,3.0):
    c,_=chain([(SIGNS[n],dict(speed=sp)) for n in ["alpha","beta","gamma"]])
    ev,_=run(r,ST(c)); print("speed",sp,len(c)/FPS,summ(ev))
# unknown-only for 25 s
rng=np.random.default_rng(1)
c,_=chain([(SIGNS[n],{}) for n in ["delta","epsilon"]*6]) if "delta" in SIGNS else (None,None)
print(list(SIGNS))
