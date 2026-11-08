import os, sys, time
sys.path.insert(0, "/home/eldar/Desktop/Lingestra"); sys.path.insert(0, "/home/eldar/Desktop/Lingestra/scripts")
import numpy as np
from lingestra import features as F
from lingestra.live import matcher as M, fastspot
import benchmark_recognition as B
t0 = time.time(); fastspot.warm_up(background=False)
print(f"compile / cache load: {time.time()-t0:.1f} s, ready={fastspot.ready()}, cores usable={fastspot.n_threads()}", flush=True)
pool = [(l, t) for l, t in B.load_pool() if len(t) >= 1]
rng = np.random.default_rng(1); order = rng.permutation(len(pool))
def build(n): return [dict(id=str(i), label=pool[order[i]][0], frames=[pool[order[i]][1][0]]) for i in range(n)]
ep = np.concatenate([F.trim_active(pool[order[i]][1][0]) for i in range(5)])
def timeit(m, n=4):
    m.decode_sequence(ep); t = time.perf_counter()
    for _ in range(n): segs = m.decode_sequence(ep)
    return (time.perf_counter() - t) / n * 1000, segs
for n in (30, 150, 600, 1100):
    m = M.PersonalMatcher(); m.set_templates(build(n))
    # raw spotting outputs: numpy vs compiled
    Q = M.episode_features(ep, 25.0) if hasattr(M, "episode_features") else None
    m.cfg.fast = False; t_np, seg_np = timeit(m)
    m.cfg.fast = True
    row = [f"{n:>4} examples: numpy {t_np:6.0f} ms"]
    for th in (1, 4, 8, 16):
        os.environ["LINGESTRA_THREADS"] = str(th)
        t_f, seg_f = timeit(m)
        same = [(s.kind, s.label, s.start, s.end) for s in seg_f] == [(s.kind, s.label, s.start, s.end) for s in seg_np]
        row.append(f"{th:>2} cores {t_f:6.1f} ms ({t_np/t_f:4.1f}x{'' if same else ' DIFFERENT'})")
    print("  |  ".join(row), flush=True)
