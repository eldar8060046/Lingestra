"""How soon does a recognised word appear while the hands are still up?

Streams 2-3 signs back to back WITHOUT lowering the hands (same generator as benchmark_continuous) and records,
for every emitted word, whether it appeared BEFORE the hands started to lower ("early") and when the LAST word
appeared relative to the moment the hands started to lower. Run with --take 0 (same clip as the example) or
--take 1 (second dictionary take, i.e. realistic distances) on signs the single-sign matcher gets right.

Usage: python scripts/measure_commit_timing.py [--take 1] [--n 20] [--seeds 101,102,103]
"""
import argparse, sys
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from lingestra import features as F                # noqa: E402
import benchmark_continuous as C                   # noqa: E402
import benchmark_recognition as B                  # noqa: E402

FPS = C.FPS
CHAIN_START = int(1.0 * FPS) + int(0.45 * FPS)      # frames before the chain in build_sequence


def run(take, n, seeds, ns_list=(2, 3), vocab_size=30, rec_factory=None):
    pool = B.load_pool()
    early_nonlast, nonlast, last_dt, exact, total = 0, 0, [], 0, 0
    first_dt = []
    for sd in seeds:
        vocab = C.make_vocab(pool, vocab_size, sd)
        rec = (rec_factory or C.build_rec)(vocab, True)
        only = C.matchable_signs(vocab) if take == 1 else None
        cand = np.array(list(range(len(vocab))) if only is None else only)
        for ns in ns_list:
            rng = np.random.default_rng(sd * 11 + ns)
            done = tries = 0
            while done < n and tries < 6 * n:
                tries += 1
                pick = cand[rng.permutation(len(cand))[:ns]]
                spans = [F.trim_active(vocab[i][1][take]) for i in pick]
                ref = [vocab[i][0] for i in pick]
                if len(set(ref)) < ns or any(len(s) < 12 for s in spans):
                    continue
                stream, chain = C.build_sequence(spans, vocab[pick[0]][1][take][0], vocab[pick[-1]][1][take][-1], rng)
                if len(F.segment_signing(stream)) != 1:
                    continue
                done += 1
                ev, _ = C.run_stream(rec, stream)
                signs = [e for e in ev if e.kind == "sign"]
                chain_end = (CHAIN_START + chain) / FPS
                total += 1
                exact += [e.label for e in signs] == ref
                for k, e in enumerate(signs):
                    if k < len(signs) - 1:
                        nonlast += 1
                        early_nonlast += e.t < chain_end - 0.05
                if signs:
                    last_dt.append(signs[-1].t - chain_end)
                    first_dt.append(signs[0].t - CHAIN_START / FPS)
    return dict(sequences=total, exact=exact / max(total, 1),
                early_nonlast=early_nonlast / max(nonlast, 1), n_nonlast=nonlast,
                last_word_vs_lowering_s=float(np.median(last_dt)) if last_dt else float("nan"),
                last_word_before_lowering=float(np.mean([d < -0.05 for d in last_dt])) if last_dt else 0.0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--take", type=int, default=1)
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--seeds", default="101,102,103")
    a = ap.parse_args()
    r = run(a.take, a.n, [int(x) for x in a.seeds.split(",")])
    print(f"take={a.take} sequences={r['sequences']} exact={r['exact']:.0%}  "
          f"words shown BEFORE the hands lowered (all but the last word): {r['early_nonlast']:.0%} of {r['n_nonlast']}  "
          f"| last word: {r['last_word_before_lowering']:.0%} before lowering, median {r['last_word_vs_lowering_s']:+.2f} s vs lowering")
