import time,numpy as np,tempfile
from lingestra.library import SignLibrary
t=time.time();lib=SignLibrary();print('ctor',time.time()-t,len(lib.entries()))
t=time.time();r=lib.search('hello');print('search1',time.time()-t,len(r))
ids=[e.id for e in lib.entries()][:100]
t=time.time()
for i in ids: lib.thumbnail_frame(i)
print('100 thumbs',time.time()-t)
t=time.time()
for i in [e.id for e in lib.entries()][:100]: lib._quality_score(i)
print('100 q',time.time()-t)
t=time.time();lib.resolve_phrase('hello how are you');print('phrase',time.time()-t)
import collections
c=collections.Counter(e.label for e in lib.entries());print(c.most_common(3), sum(1 for v in c.values() if v>1))
