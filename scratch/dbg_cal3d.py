import sys
sys.path.insert(0, "/home/eldar/Desktop/Lingestra"); sys.path.insert(0, "/home/eldar/Desktop/Lingestra/tests")
import numpy as np
from test_fingers import FG, _pose_frame, _shrink, F
frames = {n: _shrink(_pose_frame(w)) for n, w in ((1, "One"), (2, "Two"), (3, "Three"), (4, "Four"), (5, "Five"))}
for n, f in frames.items():
    v = FG.finger_features(f[46:67]); print(n, np.round(v, 2))
cal = FG.FingerCalibrator()
for n in (1, 2, 3, 4, 5):
    for i in range(40): st = cal.feed(frames[n])
print(st["done"], st["ok"], st["message"])
if st["cal"] is None:
    ks = sorted(cal.cal)
    for i, a in enumerate(ks):
        for b in ks[i + 1:]:
            print(a, b, round(float(np.linalg.norm(np.array(cal.cal[a]) - np.array(cal.cal[b]))), 2))
