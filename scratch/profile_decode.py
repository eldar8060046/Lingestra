import sys, time, cProfile, pstats, io
sys.path.insert(0, "/home/eldar/Desktop/Lingestra"); sys.path.insert(0, "/home/eldar/Desktop/Lingestra/scripts")
import numpy as np
from lingestra import features as F
from lingestra.live.matcher import PersonalMatcher
import benchmark_recognition as B
pool = [(l, t) for l, t in B.load_pool() if len(t) >= 1]
rng = np.random.default_rng(1)
order = rng.permutation(len(pool))
def build(n):
    return [dict(id=str(i), label=pool[order[i]][0], frames=[pool[order[i]][1][0]]) for i in range(n)]
spans = [F.trim_active(pool[order[i]][1][0]) for i in range(5)]
ep = np.concatenate(spans)                                 # a 5-sign episode
for n in (30, 150, 600):
    m = PersonalMatcher(); m.set_templates(build(n))
    m.decode_sequence(ep)                                  # warm up
    t0 = time.perf_counter()
    for _ in range(3): m.decode_sequence(ep)
    ms = (time.perf_counter() - t0) / 3 * 1000
    pr = cProfile.Profile(); pr.enable(); m.decode_sequence(ep); pr.disable()
    s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats("cumulative").print_stats(8)
    top = [l for l in s.getvalue().splitlines() if "matcher.py" in l or "numpy" in l][:6]
    print(f"{n:>4} templates: decode of a {len(ep)}-frame episode = {ms:.0f} ms")
    for l in top: print("      ", l.strip()[:150])
