import sys, pickle, os, itertools, warnings, io, contextlib
warnings.filterwarnings("ignore")
sys.path.insert(0,'scratch/cont'); sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np
from multiprocessing import Pool
import tune
from lingestra.live import matcher as M
from lingestra import features as F
import benchmark_continuous as C
D1=pickle.load(open(tune.CACHE,'rb')); D2=pickle.load(open(tune.CACHE2,'rb'))
def run(kw):
    cfg=M.MatcherConfig(**kw)
    r={}
    for sd,vocab,seqs in D1:
        m=M.PersonalMatcher(cfg); m.set_templates([dict(id=str(i),label=l,frames=t) for i,(l,t) in enumerate(vocab)])
        for take,ns,ref,fr in seqs:
            segs=m.decode_sequence(fr); hyp=[s.label for s in segs if s.kind=='sign']
            x=r.setdefault((take,ns),[0,0,0,0]); x[0]+=1; x[1]+=hyp==ref
            s_,d_,i_=C.align(ref,hyp); x[2]+=s_+d_+i_; x[3]+=len(ref)
    n=split=nun=acc=0
    for sd,vocab,eps,un in D2:
        m=M.PersonalMatcher(cfg); m.set_templates([dict(id=str(i),label=l,frames=t) for i,(l,t) in enumerate(vocab)])
        for take,l,fr in eps:
            segs=m.decode_sequence(fr); n+=1; split+=len(segs)>1
        for fr in un:
            segs=m.decode_sequence(fr); nun+=1; acc+=any(s.kind=='sign' for s in segs)
    seq={k:v[1]/v[0] for k,v in r.items()}
    wer={k:1-v[2]/v[3] for k,v in r.items()}
    return kw, seq, wer, split/n, acc/nun
if __name__=='__main__':
    grids=[{}]
    for v in [(0.7,1.5),(0.65,1.7),(0.8,1.4),(0.7,1.9)]: grids.append({'spot_len':v})
    for v in [0.85,0.9]: grids.append({'seg_sure':v})
    grids.append({'spot_len':(0.7,1.5),'seg_sure':0.9})
    with Pool(18) as p:
        for kw,seq,wer,fs,un in p.imap(run,grids):
            print(kw, 'seq',[round(v,2) for k,v in sorted(seq.items())], 'wacc',[round(v,2) for k,v in sorted(wer.items())],'fsplit',round(fs,3),'unen',round(un,3), flush=True)
