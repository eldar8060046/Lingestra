import sys
sys.path.insert(0, "/home/eldar/Desktop/Lingestra"); sys.path.insert(0, "/home/eldar/Desktop/Lingestra/tests")
import numpy as np
from test_fingers import _zone_with_question, _pose_frame, F, FG
r, t = _zone_with_question()
up = _pose_frame("Two"); rng = np.random.default_rng(0)
print("after question: in_zone", r.zone.in_zone, "hold_active", r.zone.hold_active, "armed", r.picker.armed, "answering", r._answering)
for i in range(120):
    f = up.copy(); off = rng.normal(0, 0.06, 2); f[[F.L_WR, F.R_WR], :2] += off; f[F.N_POSE:, :2] += off
    ev = r.push_frame(f, t + 0.04 + i * 0.04)
    if i % 12 == 0 or ev:
        print(i, "in_zone", r.zone.in_zone, "hold", r.zone.hold_active, "engaged", r._finger_engaged, "moving_since", r._moving_since, "pending", r.pending is not None, [e.kind for e in ev], "finger", r.finger, "seen", r.picker.seen)
