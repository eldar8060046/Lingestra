import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from pathlib import Path
ROOT = Path("/home/eldar/Desktop/Lingestra")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import numpy as np
from test_app_friendly import make_win, pump, app, Ev, clip
import test_fingers as T
w = make_win("test_store_shots_v7"); w.resize(1360, 860)
for i in range(4):
    w.store.save_unknown(clip(3 + i), {"t": float(i), "confidence": 0.4 + 0.05 * i})
w.show_section("teach"); w.new_tab.refresh(); pump(timeout=0.5)
w.grab().save(str(ROOT / "docs" / "ui_teach_pending.png"))
from lingestra.app.finger_dialog import FingerDialog
d = FingerDialog(w); d.show()
f2 = T._pose_frame("Two"); d.feed(f2); pump(timeout=0.2)
d.grab().save(str(ROOT / "docs" / "ui_finger_check.png"))
d.start_teaching(); sh = T._shrink(f2)
for _ in range(10): d.feed(T._pose_frame("One"))
pump(timeout=0.2); d.grab().save(str(ROOT / "docs" / "ui_finger_teach.png"))
w.show_section("settings"); pump(timeout=0.3)
w.grab().save(str(ROOT / "docs" / "ui_settings_fingers.png"))
w.close(); print("ok")
