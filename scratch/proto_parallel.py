import sys, time
sys.path.insert(0, "/home/eldar/Desktop/Lingestra"); sys.path.insert(0, "/home/eldar/Desktop/Lingestra/scripts")
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from lingestra import features as F
from lingestra.live import matcher as M
import benchmark_recognition as B
pool = [(l, t) for l, t in B.load_pool() if len(t) >= 1]
rng = np.random.default_rng(1); order = rng.permutation(len(pool))
def build(n): return [dict(id=str(i), label=pool[order[i]][0], frames=[pool[order[i]][1][0]]) for i in range(n)]
ep = np.concatenate([F.trim_active(pool[order[i]][1][0]) for i in range(5)])
orig = M.spot_batch
POOLS = {}
def make_parallel(workers, chunks):
    ex = POOLS.setdefault(workers, ThreadPoolExecutor(max_workers=workers))
    def spot_batch(DX, DM, dlens, Q, cfg, buckets=4):
        N = len(DX); Lq = len(Q.X)
        out_d = np.full((N, Lq), 1e9, np.float32); out_s = np.zeros((N, Lq), int)
        order_ = np.argsort(dlens, kind="stable")
        parts = [idx for idx in np.array_split(order_, min(chunks, N)) if len(idx)]
        def work(idx):
            Lt = int(dlens[idx].max())
            return idx, M._spot_chunk(DX[idx, :Lt], DM[idx, :Lt], dlens[idx], Q, cfg)
        for idx, (d, st) in ex.map(work, parts):
            out_d[idx], out_s[idx] = d, st
        return out_d, out_s
    return spot_batch
def timeit(m, n=4):
    m.decode_sequence(ep); t0 = time.perf_counter()
    for _ in range(n): segs = m.decode_sequence(ep)
    return (time.perf_counter() - t0) / n * 1000, segs
for n in (30, 150, 600):
    m = M.PersonalMatcher(); m.set_templates(build(n))
    M.spot_batch = orig
    base, seg0 = timeit(m)
    row = [f"{n:>4} templates: serial {base:6.0f} ms"]
    for w, c in ((2, 4), (4, 8), (8, 16), (16, 32)):
        M.spot_batch = make_parallel(w, c)
        t, segs = timeit(m)
        same = [(s.label, s.start, s.end) for s in segs] == [(s.label, s.start, s.end) for s in seg0]
        row.append(f"{w} threads {t:5.0f} ms ({base/t:.1f}x{'' if same else ' DIFFERENT'})")
    print("  |  ".join(row), flush=True)
M.spot_batch = orig
