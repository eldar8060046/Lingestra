"""Learn from corrections: a confirmed 'Not sure - did you mean A or B?' answer becomes another encrypted
example of the chosen sign. Synthetic streams, REAL Recogniser + SignLibrary + encrypted scratch store
(never the real app-data store); UI parts use a fake worker, the thread-safety test the real worker."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from PySide6.QtCore import Signal
from PySide6.QtWidgets import QApplication

import synth
from lingestra import features as F
from synth import FPS, SIGNS, stream
from lingestra import library as LIB
from lingestra.library import MAX_AUTO_TAKES, LibraryError, SignLibrary
from lingestra.live.recognizer import Recognizer
from test_app_offscreen import FakeWorker, scratch_store
from test_app_friendly import pump

app = QApplication.instance() or QApplication([])
TWIN = [(x + 0.004, y, c) for x, y, c in SIGNS["alpha"]]
QUERY = [(x + 0.002, y, c) for x, y, c in SIGNS["alpha"]]      # halfway between alpha and its twin: ambiguous


def fr(keys, **kw):
    return synth.sign_frames(keys, **kw)


class Env:
    """Library in a scratch store with alpha / alpha_twin (near identical) and beta, one recogniser."""

    def __init__(self, name):
        self.store = scratch_store(name)
        self.lib = SignLibrary(store=self.store, model_classes=[], data_root=ROOT / "scratch" / "_none")
        self.ids = {n: self.lib.add_recorded(n, fr(k)) for n, k in
                    (("alpha", SIGNS["alpha"]), ("alpha_twin", TWIN), ("beta", SIGNS["beta"]))}
        self.rec = Recognizer(None, templates=self.lib.templates())

    def ask(self, keys=None, t0=0.0, **kw):
        """Sign something ambiguous; returns (events, clarify event)."""
        self.rec.reset()
        raw = stream(fr(keys or QUERY, **kw), tail_s=0.3)[0]
        ev = []
        for i, f in enumerate(raw):
            ev += self.rec.push_frame(f, t0 + i / FPS)
        cl = [e for e in ev if e.kind == "clarify"]
        return ev, (cl[0] if cl else None)

    def refresh(self):
        self.rec.set_templates(self.lib.templates())


@pytest.fixture
def env():
    return Env("test_store_learning")


def learn(env, label="alpha", **kw):
    """ask + confirm + (as the UI does) store the example."""
    _, cl = env.ask(**kw)
    assert cl is not None
    out = env.rec.confirm(label)[0]
    if out.learn and out.learn["ok"]:
        env.lib.add_auto_take(out.learn["sign_id"], out.learn["window"], fps=out.learn["fps"])
        env.refresh()
    return out


# ------------------------------------------------------------------ engine
def test_clarify_carries_only_its_segment_window(env):
    _, cl = env.ask()
    assert cl is not None and cl.window is not None and cl.window.shape[1:] == (67, 3)
    n_stream = len(stream(fr(SIGNS["alpha"]), tail_s=0.3)[0])
    assert 8 <= len(cl.window) < n_stream * 0.8        # not the whole stream with rest / raise / lowering


def test_confirm_adds_example_refreshes_templates_and_recognises_confidently(env):
    ids = env.ids
    out = learn(env, "alpha")
    assert out.kind == "sign" and out.label == "alpha" and out.learn["ok"] and out.learn["sign_id"] == ids["alpha"]
    assert env.lib.get(ids["alpha"]).takes == 2 and env.lib.get(ids["alpha"]).auto == 1
    assert len(env.lib.auto_takes(ids["alpha"])) == 1
    tpl = {t["id"]: t for t in env.lib.templates()}
    assert len(tpl[ids["alpha"]]["frames"]) == 2 and len(tpl[ids["alpha_twin"]]["frames"]) == 1
    ev, cl = env.ask()                                  # the same movement: no question any more
    assert cl is None and [e.label for e in ev if e.kind == "sign"] == ["alpha"]
    assert [e.kind for e in ev if e.kind in ("clarify", "unknown")] == []


def test_reject_and_repeat_teach_nothing(env):
    env.ask()
    env.rec.reject()
    assert env.lib.get(env.ids["alpha"]).takes == 1
    assert env.rec.pending is None


def test_label_not_switched_on_teaches_nothing(env):
    env.ask()
    env.lib.set_active(env.ids["alpha"], False)
    env.refresh()                                        # templates no longer contain alpha
    out = env.rec.confirm("alpha")[0]
    assert out.learn is None
    with pytest.raises(LibraryError):
        env.lib.add_auto_take(env.ids["alpha"], fr(SIGNS["alpha"]))
    assert env.lib.get(env.ids["alpha"]).takes == 1


def test_learning_off_gives_no_hint(env):
    env.rec.learning = False
    _, cl = env.ask()
    assert cl is not None and env.rec.confirm("alpha")[0].learn is None


def test_gate_rejects_far_off_movement(env):
    _, cl = env.ask()
    # a wildly different movement (delta) is confirmed as 'beta' by mistake: far from beta's examples
    env.rec.pending["window"] = fr(SIGNS["delta"])
    out = env.rec.confirm("beta")[0]
    assert out.learn is not None and not out.learn["ok"] and out.learn["reason"] == "movement too different"
    assert out.learn["window"] is None


def test_gate_rejects_too_short_and_hidden_hands(env):
    _, cl = env.ask()
    short = cl.window[:8].copy()                         # 0.32 s
    env.rec.pending["window"] = short
    out = env.rec.confirm("alpha")[0]
    assert out.learn["reason"] == "too short"
    _, cl = env.ask()
    w = cl.window.copy(); w[: int(len(w) * 0.6), F.N_POSE:] = np.nan
    env.rec.pending["window"] = w
    assert env.rec.confirm("alpha")[0].learn["reason"] == "hands not visible"


def test_no_duplicate_within_two_seconds(env):
    _, cl = env.ask()
    first = env.rec.confirm("alpha")[0]
    assert first.learn["ok"]
    t = env.rec.last_t
    env.rec.pending = dict(window=cl.window, fps=25.0, source="personal", candidates=[], t=t)
    again = env.rec.confirm("alpha")[0]
    assert again.learn["ok"] is False and again.learn["reason"] == "duplicate"


# ------------------------------------------------------------------ library
def test_cap_evicts_oldest_auto_example_only(env):
    i = env.ids["alpha"]
    t0 = 1000.0
    nums = [env.lib.add_auto_take(i, fr(SIGNS["alpha"], noise=0.001, rng=np.random.default_rng(k)), now=t0 + k)
            for k in range(MAX_AUTO_TAKES + 2)]
    autos = env.lib.auto_takes(i)
    assert len(autos) == MAX_AUTO_TAKES
    assert [a["take"] for a in autos] == nums[2:]                  # the two oldest went
    assert 0 in env.lib.take_numbers(i)                            # the user's own recording is still there
    assert env.lib.get(i).takes == 1 + MAX_AUTO_TAKES and env.lib.get(i).auto == MAX_AUTO_TAKES
    ev = [a for a in env.store.read_audit() if a["action"] == "library_auto_learn"]
    assert ev[-1]["extra"] == {"auto": MAX_AUTO_TAKES, "evicted": 1}


def test_remove_take_rules_and_audit(env):
    i = env.ids["alpha"]
    n = env.lib.add_auto_take(i, fr(SIGNS["alpha"]))
    env.lib.remove_take(i, n)
    assert env.lib.take_numbers(i) == [0] and env.lib.auto_takes(i) == []
    with pytest.raises(LibraryError):
        env.lib.remove_take(i, 0)                                  # the only example
    with pytest.raises(KeyError):
        env.lib.remove_take(i, 7)
    acts = [a for a in env.store.read_audit() if a["action"] == "library_remove_take"]
    assert acts and acts[-1]["extra"] == {"remaining": 1}


def test_undo_restores_recognition(env):
    ids = env.ids
    out = learn(env, "alpha")
    assert env.ask()[1] is None
    n = env.lib.auto_takes(ids["alpha"])[0]["take"]
    env.lib.remove_take(ids["alpha"], n); env.refresh()
    assert env.lib.get(ids["alpha"]).takes == 1
    assert env.ask()[1] is not None                                # ambiguous again, exactly as before


def test_audit_has_ids_and_counts_but_no_content_and_nothing_in_plaintext(env):
    out = learn(env, "alpha")
    log = env.store.read_audit()
    blob = json.dumps(log)
    assert any(a["action"] == "library_auto_learn" for a in log)
    assert not any(isinstance(v, (list, dict)) and len(json.dumps(v)) > 100 for a in log for v in a.get("extra", {}).values())
    win = out.learn["window"]
    assert "landmark" not in blob and str(float(win[3, 20, 0])) not in blob
    # scan the scratch store for raw float32 landmark data and for decimal text of the values
    raw = np.ascontiguousarray(win[:, :, :2][5:8]).tobytes()
    probe = np.float32(win[4, 15, 0]).tobytes() + np.float32(win[4, 15, 1]).tobytes()
    root = env.store.dir if hasattr(env.store, "dir") else ROOT / "scratch" / "test_store_learning"
    files = [p for p in Path(root).rglob("*") if p.is_file()]
    assert files
    for p in files:
        b = p.read_bytes()
        assert probe not in b and raw not in b
        assert b"[[[" not in b and b"alpha_twin" not in b or p.name.endswith(".json") is False


# ------------------------------------------------------------------ UI
class LearnWorker(FakeWorker):
    learn_ready = Signal(dict); learn_skipped = Signal(str)
    def set_learning(self, on): self.calls.append(("learning", bool(on)))


def make_win(name):
    from lingestra.app.main_window import MainWindow, Config
    store = scratch_store(name); store.save_settings({"seen_help": True})
    w = MainWindow(Config(source="landmarks:x.npy", model=str(ROOT / "models" / "demo_family.keras")),
                   worker_factory=LearnWorker, store=store)
    w.show(); w.start_worker(); app.processEvents()
    return w


def hint_for(win, name="alpha", keys=None):
    return dict(ok=True, reason="", label=name, sign_id=win.library.search(name)[0].id, fps=25.0,
                window=fr(keys or SIGNS["alpha"]))


@pytest.fixture
def win():
    w = make_win("test_store_learning_ui")
    for n in ("alpha", "beta"):
        w.library.add_recorded(n, fr(SIGNS[n]))
    w._push_all()
    yield w
    w.close()


def test_ui_bar_undo_and_templates_pushed_once(win):
    wk = win.worker; wk.calls.clear()
    wk.learn_ready.emit(hint_for(win))
    assert win.learn_bar.isVisibleTo(win) and win.learn_lbl.text() == "Saved. I will recognise alpha better now."  # CHANGED: friendlier wording
    assert [c for c in wk.calls if c[0] == "templates"] == [("templates", 2)]
    i = win.library.search("alpha")[0].id
    assert win.library.get(i).takes == 2
    wk.calls.clear()
    win.learn_undo.click()
    assert win.library.get(i).takes == 1 and not win.learn_bar.isVisibleTo(win)
    assert [c for c in wk.calls if c[0] == "templates"] == [("templates", 2)]
    assert len(wk.templates[0]["frames"]) == 1


def test_ui_bar_hides_after_timeout_and_my_recordings_lists_the_learned_example_with_delete(win):
    """CHANGED: the old 'Learned from your corrections' list is folded into the 'My recordings' drop-down (under 'More options')."""
    win.learn_ready_ = None
    win.worker.learn_ready.emit(hint_for(win))
    assert win._learn_timer.isActive() and 14000 <= win._learn_timer.interval() <= 16000
    i = win.library.search("alpha")[0].id
    win.show_section("signs"); win.lib_tab.cur = i; win.lib_tab._fill_form(); app.processEvents()
    lt = win.lib_tab
    assert lt.rec_box.isVisibleTo(win) and lt.rec_box.toggle.text().strip() == "My recordings (2)"   # its own recording + the learned one
    assert "2 examples" in lt.ed_info.text()
    lt.rec_box.set_open(True); app.processEvents()
    learned = [r for r in lt.rec_panel.rows if r.kind == "learned"]
    assert len(learned) == 1 and "Learned from a correction" in learned[0].sub.text()
    learned[0].b_del.click(); app.processEvents()             # first tap only asks
    assert win.library.get(i).takes == 2 and learned[0].b_del.text() == "Tap again to delete"
    learned[0].b_del.click(); app.processEvents()
    assert win.library.get(i).takes == 1 and lt.rec_box.toggle.text().strip() == "My recordings (1)"   # the sign's own recording stays
    assert not win.learn_bar.isVisibleTo(win) and win._learned is None                # the 'Learned from you' bar followed


def test_ui_setting_default_on_persisted_audited_and_pushed(win):
    assert win.learn_sw.isChecked()
    assert ("learning", True) in win.worker.calls
    win.learn_sw.setChecked(False)
    assert win.settings["learn_corrections"] is False and ("learning", False) in win.worker.calls
    assert win.store.load_settings()["learn_corrections"] is False
    assert any(a["action"] == "settings_update" and a["extra"] == {"key": "learn_corrections"} for a in win.store.read_audit())
    win.worker.learn_ready.emit(hint_for(win))                      # even a late hint teaches nothing when OFF
    assert not win.learn_bar.isVisibleTo(win) and win.library.search("alpha")[0].takes == 1


def test_ui_inactive_sign_and_skip_message(win):
    i = win.library.search("alpha")[0].id
    win.library.set_active(i, False)
    win.worker.learn_ready.emit(hint_for(win))
    assert win.library.get(i).takes == 1 and not win.learn_bar.isVisibleTo(win)
    win.worker.learn_skipped.emit("movement too different")
    assert "too different" in win.status.text()


def test_ui_wording_has_no_jargon(win):
    from test_app_friendly import visible_texts
    win.worker.learn_ready.emit(hint_for(win))
    win.show_section("settings")
    txt = " ".join(visible_texts(win)).lower() + win.learn_lbl.text().lower()
    assert "learn from my corrections" in txt
    for bad in ("template", "dtw", "threshold", "landmark"):
        assert bad not in txt


# ------------------------------------------------------------------ real worker: thread safety
def test_real_worker_confirm_emits_learn_and_commands_are_thread_safe(monkeypatch):
    from lingestra.app import worker as W
    import lingestra.live.recognizer as R
    # this fake camera plays about 3x faster than real time, so the UI-thread tap arrives late in STREAM time: the
    # new 2 s grace for an unanswered question (hands lowered) must not expire before the test taps
    monkeypatch.setattr(R.Recognizer, "QUESTION_GRACE_S", 1000.0)
    e = Env("test_store_learning_worker")
    raw = stream(fr(QUERY), tail_s=6.0)[0]

    class Slow:                              # a camera-like source: frames arrive over a few seconds
        last_frame_bgr = None
        def __iter__(self):
            for i, f in enumerate(raw):
                time.sleep(0.015)
                yield f, i / FPS
        def close(self): pass
    monkeypatch.setattr(W, "make_source", lambda spec, loop_speed=1.0: Slow())
    wk = W.RecognitionWorker("webcam", None, send_mode="none", templates=e.lib.templates())
    got, skipped, evs = [], [], []
    wk.learn_ready.connect(got.append); wk.learn_skipped.connect(skipped.append)
    wk.events.connect(evs.extend)
    wk.set_learning(True)
    for _ in range(50):                      # a burst of commands from the UI thread before / during the run
        wk.set_learning(True)
    wk.start()
    t0 = time.time()
    while not any(x.kind == "clarify" for x in evs) and time.time() - t0 < 20:
        pump()
    assert any(x.kind == "clarify" for x in evs)
    wk.confirm("alpha")
    t0 = time.time()
    while not got and not skipped and time.time() - t0 < 20 and wk.isRunning():
        pump()
    wk.stop(); wk.wait(5000)
    assert len(got) == 1 and got[0]["label"] == "alpha" and got[0]["window"].shape[1:] == (67, 3)
    assert all(getattr(x, "learn", None) is None for x in evs)       # the window is not part of the UI events


def test_library_itself_refuses_junk_auto_examples_and_oversized_bulk(env):
    i = env.ids["alpha"]
    good = fr(SIGNS["alpha"])
    junk = good.copy(); junk[:, F.N_POSE:, :] = np.nan                  # hands never visible
    before = len(env.lib.auto_takes(i))
    with pytest.raises(LibraryError):
        env.lib.add_auto_take(i, junk)
    with pytest.raises(LibraryError):
        env.lib.add_auto_take(i, good, fps=0.5)
    assert len(env.lib.auto_takes(i)) == before
    ids = [e.id for e in env.lib.entries()][:10]
    env.lib.set_active_many(ids, False)
    with pytest.raises(LibraryError):
        env.lib.set_active_many(ids, True, max_new=0)
    hid = [e.id for e in env.lib.entries()][-1]
    env.lib.hide(hid)
    with pytest.raises(LibraryError):
        env.lib.set_active_many([hid], True)
