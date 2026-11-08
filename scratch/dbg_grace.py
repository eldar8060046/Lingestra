import sys
sys.path.insert(0, "/home/eldar/Desktop/Lingestra"); sys.path.insert(0, "/home/eldar/Desktop/Lingestra/tests")
import numpy as np
from test_fingers import _pose_frame, CLIPS, F, Recognizer
r = Recognizer(None, templates=[dict(id="a", label="Mother", frames=[CLIPS["One"]]), dict(id="b", label="Medicine", frames=[CLIPS["Two"]])])
up = _pose_frame("Two"); down = up.copy(); down[F.N_POSE:] = np.nan; down[[F.L_WR, F.R_WR], 1] = float(np.nanmean(up[[F.L_HIP, F.R_HIP], 1])) + 0.15
t = 0.0
for _ in range(10): r.push_frame(down, t); t += 0.04
for _ in range(40): r.push_frame(up, t); t += 0.04
print("in_zone after raising:", r.zone.in_zone)
r.words.append("ok"); r._clarify("Mother", 0.5, [("Mother", .6), ("Medicine", .5)], "personal", t)
hole = object(); r.words.append(hole); r.pending["hole"] = hole
tq = t
for i in range(int(16 / 0.04)):
    ev = r.push_frame(down, t); t += 0.04
    if ev:
        print(round(t - tq, 2), [(e.kind, e.label) for e in ev], "pending", r.pending is not None,
              "words", [w if isinstance(w, str) else "<hole>" for w in r.words], "in_zone", r.zone.in_zone)
print("end: pending", r.pending is not None, [w if isinstance(w, str) else "<hole>" for w in r.words], "rest_for", r.zone.rest_for(t), "in_zone", r.zone.in_zone)
