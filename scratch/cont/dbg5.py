import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, benchmark_continuous as C, benchmark_recognition as B
from lingestra import features as F
pool=B.load_pool(); vocab=C.make_vocab(pool,30,101)
rec=C.build_rec(vocab,True)
rng=np.random.default_rng(1)
want=sys.argv[1]; take=int(sys.argv[2])
for lab,t in vocab:
    span=F.trim_active(t[take])
    st,_=C.build_sequence([span],t[take][0],t[take][-1],rng,speeds=False,pauses=False)
    if lab!=want: continue
    eps=F.segment_signing(st); print('offline ep',eps[0].start,eps[0].end,len(eps[0].frames), 'span',len(span))
    orig=rec.matcher.decode_sequence
    def dec(tail,fps=25.0,strictness=1.0):
        r=orig(tail,fps=fps,strictness=strictness)
        print('  decode n=%d fps=%.1f'%(len(tail),fps),[(s.label,s.kind,s.start,s.end,round(s.ratio,2)) for s in r])
        return r
    rec.matcher.decode_sequence=dec
    rec.reset()
    for i,f in enumerate(st):
        for e in rec.push_frame(f,i/25.0):
            if e.kind!='sentence': print(i,e.kind,e.label)
