import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from pathlib import Path
ROOT = Path("/home/eldar/Desktop/Lingestra")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from test_app_friendly import make_win, pump, Ev
w = make_win("test_store_shots_v8"); w.resize(1360, 860)
wk = w.worker
words = ["Medicine", "Mother", "Brother", "Grandmother", "Doctor", "Aunt", "Uncle", "Daughter", "Sister", "Father"]
wk.events.emit([Ev("sign", "Mother", 0.95, source="personal")])
wk.events.emit([Ev("clarify", "Medicine", 0.5, [(x, 0.9 - i * 0.05) for i, x in enumerate(words)], source="personal")])
wk.stats.emit({"signing": True, "finger": (5, 0.6), "finger_seen": 5, "finger_hand": True, "finger_options": 10})
pump(timeout=0.4); w.grab().save(str(ROOT / "docs" / "ui_choices_10.png"))
wk.events.emit([Ev("clarify", "Medicine", 0.5, [("Medicine", .5), ("Mother", .4)], source="personal")])
wk.stats.emit({"signing": True, "finger": (0, 0.0), "finger_seen": None, "finger_hand": False, "finger_options": 4})
pump(timeout=0.3); w.grab().save(str(ROOT / "docs" / "ui_choices_2.png"))
w.close(); print("ok")
