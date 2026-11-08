"""Offscreen Qt smoke test: python tests/test_app_offscreen.py  (or pytest).
Uses a fake worker/store implementing the shared interfaces, so it runs without a webcam or the
other modules; if the real Recognizer exists, a second test replays a landmarks file through it."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys, time
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from PySide6.QtCore import QObject, Signal
from PySide6.QtWidgets import QApplication
from lingestra.app.main_window import MainWindow, Config

app = QApplication.instance() or QApplication([])


def scratch_store(name):
    """Throwaway encrypted store under scratch/ (never the user's real app-data store)."""
    import shutil
    from lingestra.security.store import SecureStore
    d = ROOT / "scratch" / name
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    return SecureStore(d, passphrase="test-pass")


def ensure_active_api():
    """The ENGINE builder adds SignLibrary.is_active / set_active / templates. Until they exist (or when
    running against an older library) this adds the same interface in memory so the UI tests can run."""
    from lingestra.library import SignLibrary
    if hasattr(SignLibrary, "set_active"):
        return

    def _act(self):
        if not hasattr(self, "_t_active"):
            self._t_active = {}
        return self._t_active

    def is_active(self, id):
        a = _act(self)
        if id in a:
            return a[id]
        e = self.get(id)
        return e.source == "recorded" or e.takes > 1

    def set_active(self, id, on, actor="signer-admin"):
        self.get(id); _act(self)[id] = bool(on)

    def templates(self):
        out = []
        for e in self.entries():
            if self.is_active(e.id):
                fr = []
                for t in range(e.takes):
                    try:
                        fr.append(np.asarray(self._raw(e.id, t)[0], np.float32))
                    except Exception:
                        pass
                out.append(dict(id=e.id, label=e.label, frames=fr))
        return out
    SignLibrary.is_active, SignLibrary.set_active, SignLibrary.templates = is_active, set_active, templates


ensure_active_api()


class Ev:
    def __init__(self, kind, label=None, confidence=1.0, candidates=(), latency_ms=12.0, t=0.0, source=None, answered=False):
        self.answered = answered
        self.kind, self.label, self.confidence = kind, label, confidence
        self.candidates, self.latency_ms, self.t, self.source = list(candidates), latency_ms, t, source


class FakeWorker(QObject):
    """Implements the shared worker interface (incl. set_templates / clear_sentence)."""
    preview = Signal(object, object); events = Signal(object); stats = Signal(dict)
    sent = Signal(str, bool, str); unknown_saved = Signal(str); failed = Signal(str)
    capture_state = Signal(object); capture_result = Signal(object, float)
    def __init__(self, *a, **k):
        super().__init__(); self.calls = []; self.paused = False; self.label_map = None
        self.templates = None; self.send_mode = k.get("send_mode", "clipboard")
    def start_capture(self): self.calls.append(("start_capture",))
    def cancel_capture(self): self.calls.append(("cancel_capture",))
    def set_label_map(self, m): self.label_map = dict(m); self.calls.append(("label_map", dict(m)))
    def set_templates(self, t): self.templates = list(t); self.calls.append(("templates", len(t)))
    def clear_sentence(self): self.calls.append(("clear_sentence",))
    def undo_last_word(self): self.calls.append(("undo_last_word",))
    def set_use_model(self, on): self.calls.append(("use_model", bool(on)))
    def start(self): pass
    def stop(self): pass
    def wait(self, ms=0): pass
    def confirm(self, l): self.calls.append(("confirm", l))
    def reject(self): self.calls.append(("reject",))
    def skip(self): self.calls.append(("skip",))
    def flush(self): self.calls.append(("flush",))
    def set_threshold(self, v): self.calls.append(("thr", v))


def _hand_lm():
    lm = np.full((67, 3), np.nan, np.float32)
    for i, (x, y) in {11: (.65, .4), 12: (.35, .4), 13: (.7, .55), 14: (.3, .55), 15: (.6, .35), 16: (.4, .35),
                      23: (.62, .85), 24: (.38, .85), 0: (.5, .25)}.items():
        lm[i] = (x, y, 0)
    rng = np.random.default_rng(0)
    lm[46:67, :2] = np.array([.4, .35]) + rng.normal(0, .03, (21, 2)); lm[46:67, 2] = 0
    return lm


def test_window():
    store = scratch_store("test_store_app_window")
    store.save_unknown(np.zeros((12, 67, 3), np.float32), {"confidence": 0.3})
    win = MainWindow(Config(source="landmarks:x.npy", model=str(ROOT / "models" / "demo_family.keras")),
                     worker_factory=FakeWorker, store=store)
    win.show(); win.start_worker(); w = win.worker
    assert len(win.new_tab.cards) == 1 and win.lib_tab.list.count() > 0
    assert w.label_map == {}                       # library label_map pushed at start
    assert w.templates is not None                 # ...and the templates of the signs switched on

    w.stats.emit({"fps": 24.8, "latency_ms_avg": 31.0, "scale_used": 1.0, "frames_skipped": 2})
    w.preview.emit(None, _hand_lm())
    w.events.emit([Ev("sign", "Mother", 0.93), Ev("sign", "Father", 0.88)])
    assert win.caption.text() == "Mother Father" and win.pill.text() == "Father"            # CHANGED: the word alone, big and blue
    assert win.fps_lbl.text() == "24.8" and win.lat_lbl.text() == "12 ms"      # kept up to date, shown only on request

    w.events.emit([Ev("clarify", "Brother", 0.6, [("Brother", .6), ("Sister", .3)])])
    assert win.clarify.isVisible() and "Brother or Sister" in win.pill.text()
    win.clarify._pick(1)                      # same path as pressing key "2"
    assert w.calls[-1] == ("confirm", "Sister") and not win.clarify.isVisible()
    w.events.emit([Ev("clarify", "Brother", 0.6, [("Brother", .6), ("Sister", .3)])])
    win.clarify._repeat(); assert w.calls[-1] == ("reject",)

    w.events.emit([Ev("unknown", None, 0.2)]); w.unknown_saved.emit("u2")
    assert "teach it" in win.pill.text() and win.pill_btn.isVisibleTo(win)
    w.events.emit([Ev("sentence", "Mother Father Sister", 1.0)])
    assert win.words == [] and win.last_msg is not None and win.last_msg[0] == "Mother Father Sister"   # CHANGED: only the last message is kept
    w.sent.emit("Mother Father Sister", True, "clipboard: pasted")
    assert "Sent" in win.last_status.text() and "\u2713" in win.last_status.text()

    win.thr.setValue(70); assert ("thr", 0.7) in w.calls
    win.pause.setChecked(True); assert w.paused
    win.close()


def test_real_pipeline():
    """Replay a landmarks file through the real Recognizer/worker, if those modules exist."""
    if not (ROOT / "lingestra/live/recognizer.py").exists() or not (ROOT / "models/demo_family.keras").exists():
        print("real recognizer not present - skipped"); return
    files = sorted((ROOT / "data/landmarks").glob("*.npy"))
    win = MainWindow(Config(source=f"landmarks:{files[0]}", model=str(ROOT / "models/demo_family.keras"),
                            send_mode="none"), store=scratch_store("test_store_app_real"))
    win.start_worker()
    t0 = time.time()
    while time.time() - t0 < 20 and win.worker.isRunning():
        app.processEvents(); time.sleep(0.02)
    win.stop_worker(); assert "failed" not in win.status.text() and "startup" not in win.status.text(), win.status.text()


if __name__ == "__main__":
    test_window(); test_real_pipeline(); print("ok")


def test_worker_coalesces_template_bursts_and_cancels_paste():
    from lingestra.app.worker import RecognitionWorker
    w = RecognitionWorker("webcam", None)
    for n in range(3):
        w.set_templates([{"id": str(n)}])
    w.cancel_send()
    assert w._cancel_send is True and w._cmds.qsize() == 3     # coalescing happens when drained, in run()
