import os,sys
os.environ["QT_QPA_PLATFORM"]="offscreen"
sys.path.insert(0,"tests"); sys.path.insert(0,".")
from test_app_library_ui import *
store=scratch_store("test_store_dbg1")
w=MainWindow(Config(source="landmarks:x.npy",model=MODEL,store_dir=str(store.dir)),worker_factory=FakeWorker,store=store)
w.show();w.start_worker();w.show_section("signs");w.resize(1360,900);pump(timeout=0.3)
lt=w.lib_tab
print(w.size(), lt.size(), lt.body.size(), lt.body.minimumSizeHint(), lt.split.size(), lt.split.sizes(), lt.split.ratio)
print(lt.split.widget(0).minimumHeight(), lt.split.widget(1).minimumHeight(), lt.page_scroll.verticalScrollBar().maximum())
w.grab().save("scratch/dbg1.png")
lt.list.setCurrentRow(0); pump(timeout=0.3)
print(lt.body.minimumSizeHint(), lt.body.size(), lt.split.size(), lt.split.sizes(), lt.split.minimumSizeHint(), lt.panel.minimumSizeHint(), lt.panel.minimumSize(), lt.page_scroll.verticalScrollBar().maximum())
print(lt.edit_scroll.minimumSizeHint(), lt.edit_box.minimumSizeHint())
