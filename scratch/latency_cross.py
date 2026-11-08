import sys, time
from pathlib import Path
ROOT = Path("/home/eldar/Desktop/Lingestra")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from lingestra import features as F
import benchmark_continuous as C, benchmark_recognition as B
FPS = C.FPS
pool = B.load_pool()
def go(N, seed, words=6, take=1):
    vocab = C.make_vocab(pool, N, seed)
    rec = C.build_rec(vocab, True)
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
    start = (int(1.0 * FPS) + int(0.45 * FPS)) / FPS; end = start + chain / FPS - start
    out = [(e.kind, e.label, round(e.t - start, 1)) for e in ev if e.kind in ("sign", "clarify", "unknown")]
    return dict(ref=ref, chain=round(chain / FPS, 1), events=out)
for N in (30, 100):
    for seed in (101, 102, 103):
        r = go(N, seed)
        if r is None: print(N, seed, "not enough matchable signs"); continue
        print(f"N={N} seed={seed} sentence {r['chain']} s, signed: {r['ref']}")
        print("     events (kind, word, seconds after the sentence started):", r["events"], flush=True)
