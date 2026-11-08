import sys
from pathlib import Path
ROOT = Path("/home/eldar/Desktop/Lingestra")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import measure_commit_timing as M
import benchmark_continuous as C
import benchmark_recognition as B

GRID = {
  "A baseline (old rule)": (0.40, 0.25, 0.60, 1e9),
  "B 0.85/0.5/0.6, settle 0.6 s": (0.85, 0.50, 0.60, 0.6),
  "C 0.75/0.4/0.5, settle 0.6 s": (0.75, 0.40, 0.50, 0.6),
  "D 1.0/0.6/0.7, settle 0.5 s": (1.00, 0.60, 0.70, 0.5),
  "E 0.85/0.5/0.6, settle 0.9 s": (0.85, 0.50, 0.60, 0.9),
}
seeds = [101, 102]
pool = B.load_pool()
for name, (cs, hs, ss, st) in GRID.items():
    def fac(vocab, cont):
        r = C.build_rec(vocab, cont)
        r.COMMIT_SURE, r.HOLD_SURE, r.SETTLE_SURE, r.SETTLE_S = cs, hs, ss, st
        return r
    out = []
    for take in (0, 1):
        r = M.run(take, 8, seeds, rec_factory=fac)
        out.append(r)
    # single-sign safety (cross-take, with un-enrolled signs): false split / right word / un-enrolled word
    sg = []
    for sd in seeds:
        vocab = C.make_vocab(pool, 30, sd); rec = fac(vocab, True)
        sg.append(C.run_single(pool, vocab, rec, 1, sd, unenrolled=40))
    fs = np.mean([x["false_split"] for x in sg]); ok = np.mean([x["correct"] for x in sg]); uw = np.mean([x.get("unenrolled_word", 0) for x in sg])
    print(f"{name:<32} same-take: exact {out[0]['exact']:.0%} early {out[0]['early_nonlast']:.0%} lastEarly {out[0]['last_word_before_lowering']:.0%}"
          f" | cross-take: exact {out[1]['exact']:.0%} early {out[1]['early_nonlast']:.0%} lastEarly {out[1]['last_word_before_lowering']:.0%}"
          f" | single(cross): split {fs:.0%} right {ok:.0%} unenrolled->word {uw:.0%}", flush=True)
