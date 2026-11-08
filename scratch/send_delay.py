import sys
from pathlib import Path
ROOT = Path("/home/eldar/Desktop/Lingestra")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from lingestra import features as F
import benchmark_continuous as C, benchmark_recognition as B
FPS = C.FPS; pool = B.load_pool()
START = (int(1.0 * FPS) + int(0.45 * FPS)) / FPS
rows = []
for N in (30, 100):
    for seed in range(101, 111):
        vocab = C.make_vocab(pool, N, seed); rec = C.build_rec(vocab, True)
        cand = C.matchable_signs(vocab)
        if len(cand) < 3: continue
        rng = np.random.default_rng(seed); pick = [cand[i] for i in rng.permutation(len(cand))[:3]]
        spans = [F.trim_active(vocab[i][1][1]) for i in pick]
        stream, chain = C.build_sequence(spans, vocab[pick[0]][1][1][0], vocab[pick[-1]][1][1][-1],
                                         np.random.default_rng(seed + 1), speeds=False, pauses=False)
        # a long idle tail so a late 'sentence' event is visible
        stream = np.concatenate([stream, np.repeat(stream[-1:], int(14 * FPS), 0)])
        rec.reset(); ev = []
        for i, f in enumerate(stream):
            ev += rec.push_frame(f, i / FPS)
        end = START + chain / FPS
        sent = [e for e in ev if e.kind == "sentence"]
        qs = [e for e in ev if e.kind == "clarify"]
        if sent:
            rows.append((sent[0].t - end, len(qs) > 0))
print("seconds from 'hands start to lower' until the sentence is sent:")
no_q = [d for d, q in rows if not q]; with_q = [d for d, q in rows if q]
print(f"  no question open: n={len(no_q)} median {np.median(no_q):.1f} s, max {max(no_q):.1f} s" if no_q else "  no question open: none")
print(f"  a question was asked: n={len(with_q)} median {np.median(with_q):.1f} s, max {max(with_q):.1f} s" if with_q else "  a question was asked: none")
