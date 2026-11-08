import sys
sys.path.insert(0, "/home/eldar/Desktop/Lingestra"); sys.path.insert(0, "/home/eldar/Desktop/Lingestra/tests")
import numpy as np
import test_fingers as T
from test_fingers import FG, F, _pose_frame, _bend_ring, _set_thumb
for lab in ("One", "Two", "Three"):
    f = _pose_frame(lab); b = _bend_ring(f)
    for name, fr in (("orig", f), ("ring->1.3", b)):
        v = FG.finger_features(fr[46:67]); print(lab, name, np.round(v, 2), "count", FG.hands_count(fr), "gap", FG._gap_count(v[:4]))
