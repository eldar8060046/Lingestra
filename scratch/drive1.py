import os, sys
os.environ["QT_QPA_PLATFORM"]="offscreen"
sys.path.insert(0,"/home/eldar/Desktop/Lingestra"); sys.path.insert(0,"/home/eldar/Desktop/Lingestra/tests")
from test_app_library_ui import *
from PySide6.QtTest import QTest
store = scratch_store("review_drive1")
w = MainWindow(Config(source="landmarks:x.npy", model=MODEL, store_dir=str(store.dir)), worker_factory=FakeWorker, store=store)
w.show(); w.start_worker(); pump(timeout=0.3)
w.tabs.setCurrentWidget(w.lib_tab)
lt=w.lib_tab
calls=[]
orig=w.begin_capture
w.begin_capture=lambda *a,**k: calls.append(a)
lt.search.setFocus()
QTest.keyClicks(lt.search,"nr12")
print("search text", repr(lt.search.text()), "capture calls", calls)
# shortcut via window key event
QTest.keyClick(w.new_tab.name,"n"); print("new tab name", repr(w.new_tab.name.text()), calls)
# list focus
lt.list.setFocus(); QTest.keyClick(lt.list,"n"); print("list N calls", calls)
# edit + save with search active
lt.search.setText("mother"); lt.refresh()
print("count", lt.list.count())
it=lt.list.item(0); lt.list.setCurrentItem(it); cur=lt.cur
print("cur", cur, w.library.get(cur).label)
lt.ed_label.setText("zzmom"); lt.save(); pump(timeout=0.2)
print("after save list", lt.list.count(), "cur", lt.cur, "selected", lt.list.currentItem() and lt.list.currentItem().text(), "worker map", getattr(w.worker,'label_map',None) or [x for x in dir(w.worker) if 'map' in x])
# Category empty
lt.ed_cat.setCurrentText(""); lt.save(); print("msg", lt.ed_msg.text())
