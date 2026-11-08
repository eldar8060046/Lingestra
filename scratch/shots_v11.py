import os, sys
os.environ["QT_QPA_PLATFORM"] = "offscreen"
from pathlib import Path
ROOT = Path("/home/eldar/Desktop/Lingestra")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from test_app_recordings import *
store = scratch_store("test_store_shots_v11")
w = MainWindow(Config(source="landmarks:x.npy", model=MODEL, store_dir=str(store.dir)), worker_factory=FakeWorker, store=store)
w.resize(1360, 860); w.show(); w.start_worker(); w.show_section("signs"); pump(timeout=0.3)
lib = w.library
mother = next(e.id for e in lib.entries() if e.label == "Mother")
lib.add_take(mother, clip(4)); lib.add_take(mother, clip(5))
lib.set_active(mother, True); lib.add_auto_take(mother, good_window(), now=1_730_900_000.0)
w.lib_tab.search.setText("Mother"); w.lib_tab.refresh(); pump(timeout=0.3)
it = next(w.lib_tab.list.item(i) for i in range(w.lib_tab.list.count()) if w.lib_tab.list.item(i).data(Qt.UserRole) == mother)
w.lib_tab.list.itemClicked.emit(it); pump(timeout=0.5)
dlg = w._rec_dlg; dlg.move(w.x() + 200, w.y() + 80); dlg.rows[1].b_del.click(); pump(timeout=0.2)
dlg.grab().save(str(ROOT / "docs" / "ui_recordings.png")); w.grab().save(str(ROOT / "docs" / "ui_recordings_window.png"))
dlg.close(); w.close(); print("ok")
