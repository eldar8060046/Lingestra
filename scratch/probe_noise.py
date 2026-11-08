import sys
sys.path.insert(0, "/home/eldar/Desktop/Lingestra"); sys.path.insert(0, "/home/eldar/Desktop/Lingestra/tests")
import numpy as np
from test_continuous import *
rec = Recognizer(None, templates=templates(list(SIGNS)))
for name in list(SIGNS)[:3]:
    for noise, spd in ((0.0, 1.0), (0.02, 1.0), (0.04, 1.0), (0.06, 1.0), (0.04, 1.4), (0.06, 1.5)):
        ch, _ = chain([(SIGNS[name], {"speed": spd})], noise=noise, seed=3)
        st = stream(np.concatenate([ch, np.repeat(ch[-1:], int(5 * FPS), 0)]), tail_s=3.0)[0]
        rows = []
        orig = rec.matcher.decode_sequence
        def spy(*a, **k):
            segs = orig(*a, **k)
            if rec.zone.hold_active and segs:
                rows.append((segs[-1].kind, segs[-1].label, round(float(segs[-1].ratio), 2)))
            return segs
        rec.matcher.decode_sequence = spy
        ev, _ = run(rec, st); rec.matcher.decode_sequence = orig
        hold_start = 1.3 + len(ch) / FPS
        t = [round(e.t - hold_start, 2) for e in ev if e.kind == "sign"]
        print(f"{name:<6} noise {noise:.2f} speed {spd}: ratios during hold {[r[2] for r in rows][:4]}  sign emitted {t} s after hold start  words {signs_of(ev)}", flush=True)
