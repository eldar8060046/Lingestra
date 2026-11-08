import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from pathlib import Path
ROOT = Path("/home/eldar/Desktop/Lingestra")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from test_app_friendly import make_win, pump, Ev, _fing_frames
w = make_win("test_store_shots_v10"); w.resize(1360, 860)
w.policy_btns["manual"].click()
w.worker.events.emit([Ev("sign", "Mother", 0.95, source="personal"), Ev("sign", "Medicine", 0.95, source="personal")])
pump(timeout=0.3); w.grab().save(str(ROOT / "docs" / "ui_send_toggle.png"))
from lingestra.app.finger_dialog import FingerDialog
import test_fingers as T
d = FingerDialog(w); d.show(); d.feed(T._bend_ring(T._pose_frame("Two"))); pump(timeout=0.2)
d.grab().save(str(ROOT / "docs" / "ui_finger_check_bent.png"))
w.close(); print("ok")
