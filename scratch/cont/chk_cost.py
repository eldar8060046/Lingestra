import sys, warnings; warnings.filterwarnings("ignore")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts'); sys.path.insert(0,'scratch/cont')
import numpy as np, benchmark_recognition as B
from lingestra.live import matcher as M
import old_cost
from lingestra import features as F
pool=B.load_pool()[:12]
m=M.PersonalMatcher(); m.set_templates([dict(id=str(i),label=l,frames=[t[0]]) for i,(l,t) in enumerate(pool)])
q=M.prepare(pool[3][1][0],m.cfg)
a=M.cost_matrices(m._TX,m._TM,q,1.0); b=old_cost.cost_matrices(m._TX,m._TM,q,1.0)
print(a.shape, np.abs(a-b).max())
