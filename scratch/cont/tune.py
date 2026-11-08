import sys, pickle, time, os, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, benchmark_recognition as B, benchmark_continuous as C
from lingestra import features as F
from lingestra.live import matcher as M
CACHE='scratch/cont/tune_cache.pkl'
def build(seeds=(101,102,103), n=12, vocab_n=30):
    pool=B.load_pool(); data=[]
    for sd in seeds:
        vocab=C.make_vocab(pool,vocab_n,sd)
        rng=np.random.default_rng(sd*7)
        seqs=[]
        mm=M.PersonalMatcher(); mm.set_templates([dict(id=str(i),label=l,frames=t[:1]) for i,(l,t) in enumerate(vocab)])
        good=[i for i,(l,t) in enumerate(vocab) if mm.classify(F.trim_active(t[1]),trim=False).label==l]
        print(sd,'matchable',len(good),flush=True)
        for take in (0,1):
            for ns in (2,3):
                k=0
                while k<n:
                    pool_idx=good if take==1 else list(range(len(vocab)))
                    if len(pool_idx)<ns: break
                    pick=np.array(pool_idx)[rng.permutation(len(pool_idx))[:ns]]
                    spans=[F.trim_active(vocab[i][1][take]) for i in pick]
                    if any(len(s)<12 for s in spans): continue
                    st,_=C.build_sequence(spans,vocab[pick[0]][1][take][0],vocab[pick[-1]][1][take][-1],rng)
                    eps=F.segment_signing(st)
                    if len(eps)!=1: continue
                    seqs.append((take,ns,[vocab[i][0] for i in pick],eps[0].frames)); k+=1
        data.append((sd,[(l,t[:1]) for l,t in vocab],seqs))
    return data
def get():
    if os.path.exists(CACHE): return pickle.load(open(CACHE,'rb'))
    d=build(); pickle.dump(d,open(CACHE,'wb')); return d
def evaluate(data, cfg=None, verbose=False):
    res={}
    tms=[]
    for sd,vocab,seqs in data:
        m=M.PersonalMatcher(cfg); m.set_templates([dict(id=str(i),label=l,frames=t) for i,(l,t) in enumerate(vocab)])
        for take,ns,ref,fr in seqs:
            t0=time.perf_counter(); segs=m.decode_sequence(fr); tms.append((time.perf_counter()-t0)*1000)
            hyp=[s.label for s in segs if s.kind=='sign']
            s_,d_,i_=C.align(ref,hyp)
            r=res.setdefault((take,ns),[0,0,0,0,0,0,0]); r[0]+=1; r[1]+=hyp==ref; r[2]+=s_; r[3]+=d_; r[4]+=i_; r[5]+=len(ref); r[6]+=sum(s.kind!='sign' for s in segs)
            if verbose and (hyp!=ref or any(s.kind!='sign' for s in segs)): print(take,ns,ref,[(s.label,s.kind,s.start,s.end,round(s.ratio,2)) for s in segs])
    for k,r in sorted(res.items()):
        print(k,f"n={r[0]} seq={r[1]/r[0]:.2f} S={r[2]/r[5]:.2f} D={r[3]/r[5]:.2f} I={r[4]/r[5]:.2f} nonsign/seq={r[6]/r[0]:.2f}")
    print('ms mean/max',np.mean(tms).round(1),np.max(tms).round(1))
if __name__=='__main__':
    d=get(); evaluate(d, verbose='-v' in sys.argv)

CACHE2='scratch/cont/tune_cache2.pkl'
def build_single(seeds=(101,102,103), vocab_n=30):
    pool=B.load_pool(); data=[]
    for sd in seeds:
        vocab=C.make_vocab(pool,vocab_n,sd); rng=np.random.default_rng(sd)
        have={l for l,_ in vocab}
        eps=[]
        for take in (0,1):
            for l,t in vocab:
                span=F.trim_active(t[take])
                if len(span)<12: continue
                st,_=C.build_sequence([span],t[take][0],t[take][-1],rng,speeds=False,pauses=False)
                e=F.segment_signing(st)
                if len(e)==1: eps.append((take,l,e[0].frames))
        others=[(l,t) for l,t in pool if l not in have]
        sel=np.random.default_rng(sd+5).permutation(len(others))[:40]
        un=[]
        for i in sel:
            l,t=others[i]; span=F.trim_active(t[0])
            if len(span)<12: continue
            st,_=C.build_sequence([span],t[0][0],t[0][-1],rng,speeds=False,pauses=False)
            e=F.segment_signing(st)
            if len(e)==1: un.append(e[0].frames)
        data.append((sd,[(l,t[:1]) for l,t in vocab],eps,un))
    return data
def evaluate_single(cfg=None):
    if os.path.exists(CACHE2): data=pickle.load(open(CACHE2,'rb'))
    else: data=build_single(); pickle.dump(data,open(CACHE2,'wb'))
    n=split=ok=0; nun=accw=words=0; ms=[]
    for sd,vocab,eps,un in data:
        m=M.PersonalMatcher(cfg); m.set_templates([dict(id=str(i),label=l,frames=t) for i,(l,t) in enumerate(vocab)])
        for take,l,fr in eps:
            segs=m.decode_sequence(fr); n+=1; split+=len(segs)>1;
            if len(segs)>1 and '-v' in sys.argv: print('SPLIT',take,l,[(x.label,x.kind,x.start,x.end,round(x.ratio,2)) for x in segs],len(fr))
            ok+=[s.label for s in segs if s.kind=='sign']==[l]
        for fr in un:
            t0=time.perf_counter(); segs=m.decode_sequence(fr); ms.append((time.perf_counter()-t0)*1000)
            w=[s for s in segs if s.kind=='sign']; nun+=1
            if w and '-v' in sys.argv: print('UNEN',[(x.label,x.kind,x.start,x.end,round(x.ratio,2)) for x in segs],len(fr))
            accw+=len(w)>0; words+=len(w)
    print(f"single: n={n} false_split={split/n:.3f} correct={ok/n:.3f} | unenrolled n={nun} word={accw/nun:.3f} words/ep={words/nun:.3f} ms={np.mean(ms):.0f}")
if __name__=='__main__' and '-s' in sys.argv:
    evaluate_single()
