import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
import numpy as np, benchmark_continuous as C, benchmark_recognition as B
from lingestra import features as F
pool=B.load_pool(); sd=int(sys.argv[1]); take=int(sys.argv[2]); ns=int(sys.argv[3])
vocab=C.make_vocab(pool,30,sd)
rec=C.build_rec(vocab,True)
only=C.matchable_signs(vocab) if take==1 else None
cand=np.array(list(range(len(vocab))) if only is None else only)
rng=np.random.default_rng(sd*7+ns+10*take)
done=0
while done<30:
    pick=cand[rng.permutation(len(cand))[:ns]]
    spans=[F.trim_active(vocab[i][1][take]) for i in pick]; ref=[vocab[i][0] for i in pick]
    st,_=C.build_sequence(spans,vocab[pick[0]][1][take][0],vocab[pick[-1]][1][take][-1],rng)
    if len(F.segment_signing(st))!=1: continue
    done+=1
    ev,_=C.run_stream(rec,st)
    hyp=C.words_of(ev)
    if hyp!=ref and (len(sys.argv)<5 or ref==eval(sys.argv[4])):
        orig=rec.matcher.decode_sequence
        def dec(tail,fps=25.0,strictness=1.0,final=True,_o=orig):
            r=_o(tail,fps=fps,strictness=strictness,final=final)
            print('  decode at',rec.n,'n=%d'%len(tail),'final',final,'cfrom',rec._c_from,'hold',rec.zone.hold_active,[(s.label,s.kind,s.start,s.end,round(s.ratio,2)) for s in r])
            return r
        rec.matcher.decode_sequence=dec
        C.run_stream(rec,st)
        rec.matcher.decode_sequence=orig
        ep=F.segment_signing(st)[0]
        print(ref,'ep',ep.start,ep.end,'len',len(st),[(e.kind,e.label,round(e.t*25)) for e in ev if e.kind!='sentence'])
