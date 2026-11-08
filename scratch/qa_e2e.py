import os, sys, shutil, time, warnings
os.environ["QT_QPA_PLATFORM"]="offscreen"
from pathlib import Path
ROOT=Path("/home/eldar/Desktop/Lingestra"); sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/"tests"))
warnings.filterwarnings("ignore")
import numpy as np
import test_end_to_end as E
from test_end_to_end import *
import synth
from synth import SIGNS, FPS
from lingestra.live import recognizer as R
import lingestra.output.chat_sender as CS
class MP:
    def __init__(s): s.u=[]
    def setattr(s,o,n,v): s.u.append((o,n,getattr(o,n))); setattr(o,n,v)
mp=MP()
d=ROOT/"scratch"/"qa_e2e_store"; shutil.rmtree(d,ignore_errors=True); d.mkdir(parents=True)
store=SecureStore(d,passphrase="test-pass"); store.save_settings({"seen_help":True})
src=QueueSource()
mp.setattr(W,"make_source",lambda spec,loop_speed=1.0: src)
FakeSender.sent=[]; mp.setattr(CS,"ChatSender",FakeSender)
win=MainWindow(Config(source="webcam",model=None,send_mode="clipboard"),store=store)
# empty library first
win.show(); win.start_worker(); pump(timeout=0.3)
print("empty: bulk")
lt=win.lib_tab
try:
    win._push_all(); lt.reload()
    for n in dir(lt):
        pass
except Exception as e: print("EMPTY FAIL",repr(e))
lib=win.library
ids={}
for n in ("alpha","beta","gamma"):
    ids[n]=lib.add_recorded(n, synth.sign_frames(SIGNS[n]))
win._on_library_changed()
pump(timeout=0.5)
pushes=[]
orig=win.worker.set_templates
def spy(t): pushes.append([x["label"] for x in t]); orig(t)
win.worker.set_templates=spy
ev=[]; win.worker.events.connect(lambda e: ev.extend(e))
def signs(): return [(e.kind,e.label) for e in ev if e.kind in("sign","clarify","unknown")]
def run(frames, tail=1.0, wait=6):
    ev.clear(); src.feed(frames); pump(timeout=wait)
rest0=synth.frame()
def rest(s): return np.repeat(rest0[None],int(s*FPS),0)
# back to back chain
def chain(names,gap=0.0, pause=0.0):
    parts=[]
    for i,n in enumerate(names):
        p=synth.sign_frames(SIGNS[n],pause_s=pause,pause_at=(1,) if pause else ())
        if i: w=np.linspace(0,1,6,endpoint=False)[:,None,None]; parts.append(parts[-1][-1][None]*(1-w)+p[0][None]*w)
        parts.append(p)
    p=np.concatenate(parts)
    return np.concatenate([rest(1),(lambda w: rest0[None]*(1-w)+p[0][None]*w)(np.linspace(0,1,12,endpoint=False)[:,None,None]),p,(lambda w: p[-1][None]*(1-w)+rest0[None]*w)(np.linspace(0,1,12,endpoint=False)[:,None,None]),rest(1.2)]).astype(np.float32)
run(chain(["alpha","beta"])); print("chain",signs(), win.words)
win.words[:]=[] if False else win.words
run(chain(["beta","gamma"],pause=0.2)); print("chain pause",signs(), win.words)
run(chain(["alpha"])); print("single",signs())
print("pushes so far",len(pushes))
# switch all off via UI
import inspect
print(inspect.signature(lt.update_bulk))
# ---- bulk off
lt.b_all_off.click(); pump(timeout=0.5)
print("after off: counts",lib.active_counts(),"pushes",pushes, "undo visible", lt.undo_bulk.isVisibleTo(lt) if hasattr(lt.undo_bulk,'isVisibleTo') else None)
run(chain(["alpha","beta"])); print("after all off, recognised:",signs(), win.words)
# empty templates: empty state?
print("empty hint:", getattr(win,"empty_box",None) and win.empty_box.isVisibleTo(win))
from lingestra.library import USER_CATEGORY
# undo
lt.b_undo.click(); pump(timeout=0.5)
print("after undo",lib.active_counts(),"pushes",len(pushes),pushes[-1])
# all off again, then category on
lt.b_all_off.click(); pump(timeout=0.3)
i=lt.cat_combo.findText(USER_CATEGORY); print("cat idx",i,[lt.cat_combo.itemText(k) for k in range(min(4,lt.cat_combo.count()))])
lt.cat_combo.setCurrentIndex(i); pump(timeout=0.3)
lt.b_cat_on.click(); pump(timeout=0.5)
print("cat on counts",lib.active_counts(),"user",lib.active_counts(USER_CATEGORY),"pushes",len(pushes),pushes[-1])
run(chain(["alpha","gamma"])); print("cat only:",signs())
# a dictionary sign still off
# dictionary category on, user signs off
lt.b_all_off.click(); pump(timeout=0.3)
lt.cat_combo.setCurrentIndex(1); pump(timeout=0.2)
lt.b_cat_on.click(); pump(timeout=0.8)
print("dict cat on",lib.active_counts(),lib.active_counts(lt.cat_combo.currentText()),"templates",len(lib.templates()), "alpha active",lib.is_active(ids["alpha"]))
run(chain(["alpha"])); print("user sign while dict cat on:",signs())
# restore: user signs on, plus twin
lt.b_all_off.click(); pump(timeout=0.3)
TW=[(x+0.004,y,c) for x,y,c in SIGNS["alpha"]]; QY=[(x+0.002,y,c) for x,y,c in SIGNS["alpha"]]
ids["twin"]=lib.add_recorded("alpha_twin", synth.sign_frames(TW)); 
for n in ("alpha","beta","gamma"): lib.set_active(ids[n],True)
win._on_library_changed(); pump(timeout=0.5)
print("active",lib.active_counts(), "templates",[t['label'] for t in lib.templates()])
def clar(): 
    ev.clear(); src.feed(np.concatenate([rest(1),synth.stream(synth.sign_frames(QY),tail_s=0.8)[0]])); pump(timeout=5)
clar(); print("clar:",signs(), "choices", [c for c in dir(win.clarify) if 'btn' in c.lower()][:5])
n0=len(lib.auto_takes(ids["alpha"])); pn=len(pushes)
win._on_confirm("alpha"); src.feed(rest(0.5)); pump(timeout=1.5)
print("learned",len(lib.auto_takes(ids["alpha"]))-n0,"pushes",len(pushes)-pn,"bar",win.learn_bar.isVisibleTo(win),win.words, win.learn_lbl.text())
win._undo_learn(); pump(timeout=0.5)
print("after undo",len(lib.auto_takes(ids["alpha"]))-n0,"pushes",len(pushes)-pn,"takes",lib.take_numbers(ids["alpha"]))
# switched-off sign must not learn: ask, switch alpha off meanwhile, confirm
time.sleep(0); 
clar(); print("clar2:",signs())
lib.set_active(ids["alpha"],False); win._on_library_changed(); pump(timeout=0.5)
n0=len(lib.auto_takes(ids["alpha"])); win._on_confirm("alpha"); src.feed(rest(0.5)); pump(timeout=1.5)
print("off sign learned?",len(lib.auto_takes(ids["alpha"]))-n0,"words",win.words)
# bulk off while question open, then confirm other
lib.set_active(ids["alpha"],True); win._on_library_changed(); pump(timeout=0.5)
clar(); print("clar3:",signs(), win.words)
lt.b_all_off.click(); pump(timeout=0.5)
n0=len(lib.auto_takes(ids["alpha_twin" if False else "alpha"]))
win._on_confirm("alpha"); src.feed(rest(0.5)); pump(timeout=1.5)
print("bulk-off then confirm: learned",len(lib.auto_takes(ids["alpha"]))-n0,"words",win.words,"empty?")
# splitter
sp=[a for a in dir(lt) if 'split' in a.lower()]; print(sp)
print("---splitter")
win.show_section("signs") if hasattr(win,"show_section") else None
pump(timeout=0.3)
win.resize(1360,960); pump(timeout=0.5)
sp=lt.split; print(type(sp), sp.sizes())
tot=sum(sp.sizes()); sp.setSizes([int(tot*0.45), tot-int(tot*0.45)]); sp.drag_finished(); pump(timeout=0.2)
print("saved",win.settings.get("my_signs_split"), store.load_settings().get("my_signs_split"))
win.worker.stop(); src.close(); win.close(); pump(timeout=0.5)
src2=QueueSource(); W.make_source=lambda spec,loop_speed=1.0: src2
win2=MainWindow(Config(source="webcam",model=None,send_mode="clipboard"),store=store); win2.show(); win2.resize(1360,960); pump(timeout=0.5)
win2.show_section("signs"); pump(timeout=0.5)
print("reopened sizes",win2.lib_tab.split.sizes(), win2.settings.get("my_signs_split"))
src2.close(); win2.close()
os._exit(0)
