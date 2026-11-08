import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from pathlib import Path
ROOT = Path("/home/eldar/Desktop/Lingestra")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from test_app_friendly import make_win, pump
w = make_win("test_store_shots_v9"); w.resize(1360, 960)
w.show_section("settings"); pump(timeout=0.4)
w.grab().save(str(ROOT / "docs" / "ui_settings_hold.png")); w.close(); print("ok")
