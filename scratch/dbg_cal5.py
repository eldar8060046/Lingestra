import sys
sys.path.insert(0, "/home/eldar/Desktop/Lingestra"); sys.path.insert(0, "/home/eldar/Desktop/Lingestra/tests")
import numpy as np
import test_fingers as T
from test_fingers import FG, _pose_frame, _shrink
frames = {n: _shrink(_pose_frame(w)) for n, w in ((1, "One"), (2, "Two"), (3, "Three"), (4, "Four"), (5, "Five"))}
for n, f in frames.items():
    v = FG.finger_features(f[46:67]); print(n, np.round(v, 2), "ext>1.15:", int(sum(v[:4] > 1.15)), "std count:", FG.hands_count(f))
cal = FG.FingerCalibrator()
for n in (1, 2, 3, 4, 5):
    for i in range(40):
        st = cal.feed(frames[n])
    print("after feeding pose", n, "-> step", st["step"], "target", st["target"], "|", st["message"])
