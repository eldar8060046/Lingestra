import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0,'.'); sys.path.insert(0,'tests')
import numpy as np, synth
from synth import FPS, SIGNS, frame, stream, moves, _lerp
from lingestra.live.recognizer import Recognizer

def path_of(keys, speed=1.0, pause_s=0.0, pause_at=(), move_s=0.5, edge=0.25):
    mf=max(int(move_s*FPS/speed),2)
    p=moves(keys,mf,int(pause_s*FPS),pause_at)
    eh=max(int(edge*FPS),1)
    return np.concatenate([np.repeat(p[:1],eh,0),p,np.repeat(p[-1:],eh,0)])

def chain(items, trans_s=0.25, noise=0.0, seed=0):
    """items: list of (keys, kwargs). Returns landmark frames (n,67,3) of the chain (no hold between signs)."""
    rng=np.random.default_rng(seed)
    rows=[]; spans=[]
    for i,(keys,kw) in enumerate(items):
        p=path_of(keys,**kw)
        if i>0:
            rows.append(_lerp(rows[-1][-1],p[0],max(int(trans_s*FPS),2)))
        s=sum(len(r) for r in rows); rows.append(p); spans.append((s,s+len(p)-1))
    path=np.concatenate(rows)
    fr=[]
    for x,y,c in path:
        w=np.array([x,y])+(rng.normal(0,noise,2) if noise else 0)
        fr.append(frame(wr_r=w,curl_r=float(c)))
    return np.stack(fr), spans

def templates(names, takes=1):
    out=[]
    for k,n in enumerate(names):
        fr=[synth.sign(n, noise=0.002 if t else 0.0, rng=np.random.default_rng(t)) for t in range(takes)]
        out.append(dict(id=f"id{k}",label=n,frames=fr))
    return out

def run(rec, raw):
    rec.reset(); ev=[]; at=[]
    for i,f in enumerate(raw):
        e=rec.push_frame(f,i/FPS); ev+=e; at+=[i]*len(e)
    return ev, at
if __name__=='__main__':
    rec=Recognizer(None,templates=templates(["alpha","beta","gamma"]))
    for names in (["alpha","beta"],["beta","gamma"],["alpha","beta","gamma"],["gamma","alpha","beta"]):
        ch,spans=chain([(SIGNS[n],{}) for n in names])
        st,lead,rs,L,low=stream(ch)
        ev,at=run(rec,st)
        print(names, 'chain frames',lead+rs,'->',lead+rs+L, spans, [(e.kind,e.label,a) for e,a in zip(ev,at)])
