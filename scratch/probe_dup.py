import sys
sys.path.insert(0, "/home/eldar/Desktop/Lingestra"); sys.path.insert(0, "/home/eldar/Desktop/Lingestra/tests")
import numpy as np
from test_continuous import *
name = sys.argv[1] if len(sys.argv) > 1 else "alpha"
rec = Recognizer(None, templates=templates(list(SIGNS)))
ch, _ = chain([(SIGNS[name], {"speed": 1.0})], noise=0.04, seed=3)
hold_s = 5.0
st = stream(np.concatenate([ch, np.repeat(ch[-1:], int(hold_s * FPS), 0)]), tail_s=3.0)[0]
hold_start = 1.3 + len(ch) / FPS
orig = rec.matcher.decode_sequence
def spy(piece, *a, **k):
    segs = orig(piece, *a, **k)
    if k.get("final", True):
        print(f"  FINAL decode at stream t={rec.last_t - hold_start:+.2f}s after hold start: piece {len(piece)} frames "
              f"({len(piece)/FPS:.1f}s) -> {[(s.kind, s.label, s.start, s.end, round(float(s.ratio),2)) for s in segs]}")
    return segs
rec.matcher.decode_sequence = spy
ev, _ = run(rec, st)
print(name, "events:", [(e.kind, e.label, round(e.t - hold_start, 2)) for e in ev])
print("committed-from index at the end of episode:", rec._c_from)
