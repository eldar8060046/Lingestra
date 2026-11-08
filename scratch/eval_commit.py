import sys
from pathlib import Path
ROOT = Path("/home/eldar/Desktop/Lingestra")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
import measure_commit_timing as M, benchmark_continuous as C, benchmark_recognition as B
seeds = [7101, 7102, 7103, 7104, 7105]
pool = B.load_pool()
CFG = {"next-sign bar 0.85 only (NO settle rule)": (0.85, 0.25, 0.60, 1e9)}
for name, (cs, hs, ss, st) in CFG.items():
    def fac(vocab, cont):
        r = C.build_rec(vocab, cont); r.COMMIT_SURE, r.HOLD_SURE, r.SETTLE_SURE, r.SETTLE_S = cs, hs, ss, st; return r
    res = [M.run(t, 10, seeds, rec_factory=fac) for t in (0, 1)]
    sg = []
    for sd in seeds:
        vocab = C.make_vocab(pool, 30, sd); sg.append(C.run_single(pool, vocab, fac(vocab, True), 1, sd, unenrolled=40))
    print(f"{name}: same-take exact {res[0]['exact']:.0%}, words shown early {res[0]['early_nonlast']:.0%} | "
          f"cross-take exact {res[1]['exact']:.0%}, words shown early {res[1]['early_nonlast']:.0%} (n={res[1]['n_nonlast']}) | "
          f"single signs cross-take: split {np.mean([x['false_split'] for x in sg]):.0%}, right {np.mean([x['correct'] for x in sg]):.0%}, "
          f"un-enrolled->word {np.mean([x.get('unenrolled_word',0) for x in sg]):.0%}", flush=True)
