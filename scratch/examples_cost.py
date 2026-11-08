import sys, time, json
sys.path.insert(0, "/home/eldar/Desktop/Lingestra"); sys.path.insert(0, "/home/eldar/Desktop/Lingestra/tests")
import numpy as np
from lingestra import features as F
from lingestra.live.matcher import PersonalMatcher
from test_fingers import CLIPS
base = F.trim_active(CLIPS["Two"])
rng = np.random.default_rng(0)
def take(k):    # a slightly different performance of the same sign
    x = base.copy(); x[:, :, :2] += rng.normal(0, 0.004, x[:, :, :2].shape); return x
for n_signs in (4, 20):
    for n_ex in (1, 3, 6, 10):
        tpl = [dict(id=f"s{i}", label=f"sign{i}", frames=[take(k) for k in range(n_ex)]) for i in range(n_signs)]
        m = PersonalMatcher(); m.set_templates(tpl)
        q = take(99)
        t0 = time.perf_counter()
        for _ in range(20): m.classify(q, trim=False)
        one = (time.perf_counter() - t0) / 20 * 1000
        ep = np.concatenate([take(1), take(2), take(3)])
        t0 = time.perf_counter()
        for _ in range(5): m.decode_sequence(ep)
        dec = (time.perf_counter() - t0) / 5 * 1000
        print(f"{n_signs:>2} signs x {n_ex:>2} examples = {n_signs*n_ex:>3} stored examples: one sign decision {one:5.1f} ms, decoding a 3-sign stretch {dec:6.1f} ms", flush=True)
