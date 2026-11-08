import time
from lingestra.library import SignLibrary
lib=SignLibrary()
print(len(lib._order))
t=time.time()
for i in lib._order[:1244]: lib._active[i]=True
T=lib.templates(); print(len(T), time.time()-t)
from lingestra.live.matcher import PersonalMatcher
m=PersonalMatcher(); t=time.time(); m.set_templates(T); print('cold',time.time()-t)
t=time.time(); m.set_templates(T); print('warm',time.time()-t)
import numpy as np, inspect
print([n for n in dir(m) if not n.startswith('_')])
q=T[5]['frames'][0]
t=time.time()
for _ in range(5): m.classify(q)
print('classify ms',(time.time()-t)/5*1000)
