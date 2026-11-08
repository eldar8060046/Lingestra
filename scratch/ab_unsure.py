import sys
from pathlib import Path
ROOT = Path("/home/eldar/Desktop/Lingestra")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from lingestra import features as F
import benchmark_continuous as C, benchmark_recognition as B
FPS = C.FPS
pool = B.load_pool()
START = (int(1.0 * FPS) + int(0.45 * FPS)) / FPS

def trial(N, seed, words, take, old):
    vocab = C.make_vocab(pool, N, seed)
    rec = C.build_rec(vocab, True); rec.UNSURE_WAITS_FOR_SIGN = old
    cand = C.matchable_signs(vocab) if take == 1 else list(range(len(vocab)))
    if len(cand) < words:
        return None
    rng = np.random.default_rng(seed)
    pick = [cand[i] for i in rng.permutation(len(cand))[:words]]
    spans = [F.trim_active(vocab[i][1][take]) for i in pick]
    ref = [vocab[i][0] for i in pick]
    stream, chain = C.build_sequence(spans, vocab[pick[0]][1][take][0], vocab[pick[-1]][1][take][-1],
                                     np.random.default_rng(seed + 1), speeds=False, pauses=False)
    rec.reset(); ev = []
    for i, f in enumerate(stream):
        ev += rec.push_frame(f, i / FPS)
    evs = [e for e in ev if e.kind in ("sign", "clarify", "unknown")]
    dur = chain / FPS
    lag = [(e.t - START) - (k + 1) / max(len(evs), 1) * dur for k, e in enumerate(evs)]
    hyp = [e.label for e in ev if e.kind == "sign"]
    s, d, i = C.align(ref, hyp)
    stalled = sum(1 for a, b in zip(evs, evs[1:]) if abs(a.t - b.t) < 1e-6)
    return dict(lag=float(np.mean(lag)) if lag else 0.0, wer=(s + d + i) / words, n_ev=len(evs), stalled=stalled,
                exact=hyp == ref, asked=sum(e.kind == "clarify" for e in ev), unk=sum(e.kind == "unknown" for e in ev))

for words in (4, 6):
    for old in (True, False):
        rs = []
        for N in (30, 100):
            for seed in (101, 102, 103, 104, 105, 106):
                r = trial(N, seed, words, 1, old)
                if r: rs.append(r)
        print(f"{words} words, cross-take, {'OLD (unsure waits for a later word)' if old else 'NEW (released when stable)'}: "
              f"n={len(rs)} mean lag {np.mean([r['lag'] for r in rs]):+.1f} s, events emitted in a burst {sum(r['stalled'] for r in rs)}/{sum(r['n_ev'] for r in rs)}, "
              f"word errors {np.mean([r['wer'] for r in rs]):.0%}, sentences exact {np.mean([r['exact'] for r in rs]):.0%}, "
              f"questions {sum(r['asked'] for r in rs)} unknown {sum(r['unk'] for r in rs)}", flush=True)
