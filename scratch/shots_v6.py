import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from pathlib import Path
ROOT = Path("/home/eldar/Desktop/Lingestra")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from test_app_friendly import make_win, pump, app, Ev, _hand_lm
w = make_win("test_store_shots_v6")
w.resize(1360, 860)
wk = w.worker
wk.stats.emit({"signing": True})
wk.events.emit([Ev("sign", "Mother", 0.95, source="personal")])
wk.events.emit([Ev("clarify", "Medicine", 0.5, [("Medicine", .5), ("Mother", .4)], source="personal")])
wk.stats.emit({"signing": True, "finger": (2, 0.6)})
wk.events.emit([Ev("sentence", "Father loves Mother", 1.0)]); wk.sent.emit("Father loves Mother", True, "ok")
wk.events.emit([Ev("sign", "Mother", 0.95, source="personal")])
wk.events.emit([Ev("clarify", "Medicine", 0.5, [("Medicine", .5), ("Mother", .4)], source="personal")])
wk.stats.emit({"signing": True, "finger": (2, 0.6)})
pump(timeout=0.3)
w.grab().save(str(ROOT / "docs" / "ui_talk_choice.png"))
wk.events.emit([Ev("sign", "Medicine", 1.0, answered=True, source="personal")])
pump(timeout=0.2)
w.grab().save(str(ROOT / "docs" / "ui_talk_live_word.png"))
w.show_section("signs"); pump(timeout=0.4)
from lingestra.app.library_tab import GroupSwitchDialog
lt = w.lib_tab; lt.confirm_many = lambda n: True
d = GroupSwitchDialog(lt); d.show(); pump(timeout=0.3)
d.rows[("category", "Family")][1].click(); pump(timeout=0.2)
d.grab().save(str(ROOT / "docs" / "ui_groups.png"))
w.close(); print("ok")
