"""Why do words take long to appear? Streams a 6-sign sentence (hands never lowered) through the Recogniser with
N signs switched on and reports: real-time factor (processing time / signing time; > 1 means the app cannot keep up with a
live camera and frames pile up), slowest decode, and when each word appeared relative to when it was finished.

Usage: python scripts/measure_latency.py [--n 30,150,400,1000]
"""
import argparse, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
from lingestra import features as F                 # noqa: E402
import benchmark_continuous as C                    # noqa: E402
import benchmark_recognition as B                   # noqa: E402

FPS = C.FPS


def run(n_templates, words=6, seed=5, rec_setup=None):
    pool = [(lab, t) for lab, t in B.load_pool() if len(t) >= 1]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(pool))
    vocab = [pool[i] for i in order[:max(n_templates, words)]]
    rec = C.build_rec(vocab, True)
    if rec_setup:
        rec_setup(rec)
    spans = [F.trim_active(vocab[i][1][0]) for i in range(words)]
    ref = [vocab[i][0] for i in range(words)]
    stream, chain = C.build_sequence(spans, vocab[0][1][0][0], vocab[words - 1][1][0][-1],
                                     np.random.default_rng(seed + 1), speeds=False, pauses=False)
    rec.reset(); ev = []; lat = []
    t_wall = time.perf_counter()
    for i, f in enumerate(stream):
        t0 = time.perf_counter(); ev += rec.push_frame(f, i / FPS); lat.append((time.perf_counter() - t0) * 1000)
    wall = time.perf_counter() - t_wall
    signs = [e for e in ev if e.kind == "sign"]
    asks = [e for e in ev if e.kind == "clarify"]
    start = (int(1.0 * FPS) + int(0.45 * FPS)) / FPS
    end = start + chain / FPS
    return dict(n=n_templates, rtf=wall / (len(stream) / FPS), max_frame_ms=max(lat), mean_frame_ms=float(np.mean(lat)),
                n_over_40ms=int(sum(x > 40 for x in lat)), words=[e.label for e in signs], ref=ref,
                emit_t=[round(e.t - start, 1) for e in signs], chain_s=round(end - start, 1), clarify=len(asks),
                right=[e.label for e in signs] == ref)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", default="30,150,400,1000"); a = ap.parse_args()
    for n in [int(x) for x in a.n.split(",")]:
        r = run(n)
        print(f"{n:>5} signs switched on: real-time factor {r['rtf']:.2f}  slowest frame {r['max_frame_ms']:.0f} ms  "
              f"frames over 40 ms: {r['n_over_40ms']}  sentence {r['chain_s']} s  words appeared at {r['emit_t']} s  "
              f"right={r['right']}  questions={r['clarify']}", flush=True)
