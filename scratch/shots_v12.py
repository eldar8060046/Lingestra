import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from pathlib import Path
ROOT = Path("/home/eldar/Desktop/Lingestra")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from test_app_recordings import *
store = scratch_store("test_store_shots_v12")
w = MainWindow(Config(source="landmarks:x.npy", model=MODEL, store_dir=str(store.dir)), worker_factory=FakeWorker, store=store)
w.resize(1360, 960); w.show(); w.start_worker(); w.show_section("signs"); pump(timeout=0.3)
lib = w.library
mother = next(e.id for e in lib.entries() if e.label == "Mother")
lib.add_take(mother, clip(4)); lib.add_take(mother, clip(5))
lib.set_active(mother, True); lib.add_auto_take(mother, good_window(), now=1_730_900_000.0)
panel = click(w, mother, open_box=False); pump(timeout=0.3)
w.grab().save(str(ROOT / "docs" / "ui_recordings_closed.png"))
w.lib_tab.rec_box.set_open(True); pump(timeout=0.3)
w.lib_tab.edit_scroll.verticalScrollBar().setValue(w.lib_tab.edit_scroll.verticalScrollBar().maximum()); pump(timeout=0.2)
panel.rows[1].b_del.click(); pump(timeout=0.2)
w.grab().save(str(ROOT / "docs" / "ui_recordings_open.png")); w.close(); print("ok")
