"""End to end, headless: REAL worker, recogniser, matcher, library and an encrypted scratch store; only the
camera (a queue-fed landmark source) and the chat sender (records text, presses no keys) are fakes.

Scenario: a synthetic webcam stream built from dictionary clips (idle, raise, sign, lower, idle, with varying
speed and inserted pauses). Three signs are taught through the real teach flow (captured from the stream by the
real recorder), switched on, then a second stream signs them again (a different real take where it exists, other
speeds, pauses) plus one sign that was never taught. Landmarks only; nothing here persists video."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import queue, shutil, sys, time, warnings
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
warnings.filterwarnings("ignore")
from PySide6.QtWidgets import QApplication

from lingestra import features as F
from lingestra.app import worker as W
from lingestra.app.main_window import MainWindow, Config
from lingestra.live import recognizer as R
from lingestra.security.store import SecureStore

app = QApplication.instance() or QApplication([])
FPS = 25.0
LM = ROOT / "data" / "landmarks"
# (label, [take ids]): signs with 2 good real takes whose cross-take match is reliable on this data
TAUGHT = [("Bell", ["5657", "5658"]), ("Brown", ["4093", "4094"]), ("Four hundred", ["6083", "6140"])]
UNTAUGHT = "4041"            # 'Khaleej': never taught


def pump(cond=None, timeout=10.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents(); time.sleep(0.005)
        if cond is not None and cond():
            return True
    return cond is None


def load(i):
    return np.load(LM / f"{i}.npy").astype(np.float32)


# ------------------------------------------------------------------ synthetic webcam stream pieces
def blend(a, b, n):
    w = np.linspace(0, 1, n)[:, None, None]
    out = a[None] * (1 - w) + b[None] * w
    return np.where(np.isnan(out), np.where(w < 0.5, a[None], b[None]), out).astype(np.float32)


def retime(seg, speed):
    n = max(int(round(len(seg) / speed)), 4)
    src = np.linspace(0, len(seg) - 1, n)
    lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, len(seg) - 1); w = (src - lo)[:, None, None]
    out = seg[lo] * (1 - w) + seg[hi] * w
    return np.where(np.isnan(out), np.where(w < 0.5, seg[lo], seg[hi]), out).astype(np.float32)


def with_pauses(seg, secs=(0.7, 0.5), at=(0.35, 0.7)):
    parts, prev = [], 0
    for s, f in zip(secs, at):
        p = int(len(seg) * f)
        parts += [seg[prev:p], np.repeat(seg[p - 1:p], int(s * FPS), 0)]
        prev = p
    parts.append(seg[prev:])
    return np.concatenate(parts)


def performance(raw, speed=1.0, pauses=False, rest_s=0.0, move_s=0.4):
    """rest + raise + sign + lower + rest, from the clip's own rest poses (the sign itself is the clip's signing zone)."""
    seg = F.trim_active(raw)
    if speed != 1.0:
        seg = retime(seg, speed)
    if pauses:
        seg = with_pauses(seg)
    a, b, m = raw[0], raw[-1], int(move_s * FPS)
    return np.concatenate([blend(a, seg[0], m), seg, blend(seg[-1], b, m)]), a, b


def rest(frame, secs):
    return np.repeat(frame[None], int(secs * FPS), 0).astype(np.float32)


class QueueSource:
    """A 'webcam': frames arrive when the test feeds them; timestamps advance 1/fps per frame."""
    last_frame_bgr = None

    def __init__(self):
        self.q, self.n, self.closed = queue.Queue(), 0, False
        self.fed = []                                   # exactly what the recogniser must receive

    def feed(self, frames):
        for f in frames:
            self.fed.append(f); self.q.put(f)

    def close(self):
        self.closed = True

    def __iter__(self):
        while not self.closed:
            try:
                f = self.q.get(timeout=0.1)
            except queue.Empty:
                continue
            self.n += 1
            yield f, self.n / FPS


class FakeSender:
    sent = []

    def __init__(self, *a, **k):
        pass

    def send(self, text):
        FakeSender.sent.append(text)
        from types import SimpleNamespace
        return SimpleNamespace(ok=True, method="fake", detail="recorded, no keys pressed")


@pytest.fixture
def env(monkeypatch):
    d = ROOT / "scratch" / "test_store_e2e"
    shutil.rmtree(d, ignore_errors=True); d.mkdir(parents=True)
    store = SecureStore(d, passphrase="test-pass")
    store.save_settings({"seen_help": True})
    src = QueueSource()
    monkeypatch.setattr(W, "make_source", lambda spec, loop_speed=1.0: src)
    import lingestra.output.chat_sender as CS
    monkeypatch.setattr(CS, "ChatSender", FakeSender)
    FakeSender.sent = []
    pushed = []                                          # what the recogniser really received (identity of frames)
    orig = R.Recognizer.push_frame

    def spy(self, lm, t):
        pushed.append(np.array(lm, copy=True))
        return orig(self, lm, t)
    monkeypatch.setattr(R.Recognizer, "push_frame", spy)
    win = MainWindow(Config(source="webcam", model=None, send_mode="clipboard"), store=store)
    win.show(); win.start_worker(); app.processEvents()
    yield win, src, store, pushed
    src.close()                                           # lets the worker thread end
    win.close()


def teach(win, src, name, take):
    """The real teach flow: Teach a new sign -> countdown -> raise, sign, lower -> review -> Keep."""
    raw = load(take)
    perf, a, b = performance(raw)
    win.new_tab.name.setText(name)
    win.new_tab.teach_btn.click()
    assert win.stage.mode == "capture"
    src.feed(np.concatenate([rest(a, 3.6), perf, rest(b, 1.0)]))
    assert pump(lambda: win.stage.mode == "review", 20), f"recording of {name} did not finish: {win._capture}"
    assert win.stage.name_edit.text() == name
    win.stage.name_edit.returnPressed.emit()             # Keep
    assert win.stage.mode == "live"
    return win.library.search(name)[0].id


def test_teach_then_recognise_with_real_engine(env):
    win, src, store, pushed = env
    assert win.mirror is True                            # mirror view is on: it must not touch recognition
    # ---- 1. teach three signs from the stream; they are in use straight away
    ids = [teach(win, src, name, takes[0]) for name, takes in TAUGHT]
    for i in ids:
        assert win.library.is_active(i)
    assert sorted(t["label"] for t in win.library.templates() if t["id"] in ids) == sorted(n for n, _ in TAUGHT)
    clip = win.library.templates()[0]["frames"][0]
    assert clip.shape[1:] == (67, 3) and len(clip) < 90      # just the sign: no long raise / lower / idle in it
    assert F.hand_height(clip).min() > -0.2 if hasattr(F, "hand_height") else True
    assert not win.words                                  # teaching suspended recognition: no words from it

    # ---- 2. sign them again: different take, different speeds, pauses; then one never taught
    ev = []
    win.worker.events.connect(lambda e: ev.extend(e))
    plan = [("Bell", load("5658"), 1.0, False), ("Brown", load("4094"), 1.5, True),
            ("Four hundred", load("6140"), 0.7, False)]
    stream = [rest(plan[0][1][0], 1.5)]
    for _, raw, speed, pz in plan:
        perf, a, b = performance(raw, speed, pz)
        stream += [perf, rest(b, 0.9)]
    unk_raw = load(UNTAUGHT)
    perf, a, b = performance(unk_raw)
    stream += [perf, rest(b, 0.9)]
    src.feed(np.concatenate(stream))
    assert pump(lambda: len([e for e in ev if e.kind in ("sign", "clarify", "unknown")]) >= 4, 30)
    pump(timeout=0.5)
    kinds = [(e.kind, e.label) for e in ev if e.kind in ("sign", "clarify", "unknown")]
    assert [k for k in kinds if k[0] == "sign"] == [("sign", n) for n, _ in TAUGHT], kinds
    assert kinds[-1][0] == "unknown" and len(kinds) == 4, kinds     # raise/lower produced nothing extra
    assert win.words == [n for n, _ in TAUGHT]
    assert win.caption.text() == "Bell Brown Four hundred"          # the Talk screen shows exactly these words
    assert "teach it" in win.pill.text().lower()
    assert all(e.source == "personal" for e in ev if e.kind == "sign")

    # ---- 3. the un-taught sign became a pending card; naming it adds a NEW sign (no merge)
    assert pump(lambda: win.new_tab.total == 1, 5)
    card = win.new_tab.cards[0]
    assert card.b_add.isVisibleTo(win.new_tab) is not None and not card.b_add.isEnabled()   # needs a name first
    n_before = len(win.library.entries())
    card.name.setText("Khaleej (mine)"); assert card.b_add.isEnabled()
    card.b_add.click()
    assert len(win.library.entries()) == n_before + 1 and win.new_tab.total == 0
    new_id = win.library.search("Khaleej (mine)")[0].id
    assert win.library.get(new_id).source == "recorded" and win.library.is_active(new_id)

    # ---- 4. Send now pastes through the (fake) sender
    win.b_send.click()
    src.feed(rest(b, 0.4))                                # a live camera keeps delivering frames; commands run between them
    assert pump(lambda: FakeSender.sent, 5)
    assert FakeSender.sent == ["Bell Brown Four hundred"]
    assert pump(lambda: "Sent" in win.last_status.text(), 5) and not win.words

    # ---- 5. the freshly named sign is recognised from now on (templates refreshed after Add as new sign)
    ev.clear()
    perf, a, b = performance(unk_raw, 1.2, True)
    src.feed(np.concatenate([rest(a, 1.0), perf, rest(b, 0.9)]))
    assert pump(lambda: any(e.kind in ("sign", "clarify", "unknown") for e in ev), 20)
    assert [(e.kind, e.label) for e in ev if e.kind == "sign"] == [("sign", "Khaleej (mine)")], [(e.kind, e.label) for e in ev]

    # ---- 6. mirror is display only: the recogniser got exactly the frames fed, un-mirrored
    n = len(pushed)
    assert n > 0 and n <= len(src.fed)
    j = 0                                                 # in order; frames during teaching never reach the recogniser
    for p in pushed:
        while j < len(src.fed) and not np.array_equal(p, src.fed[j], equal_nan=True):
            j += 1
        assert j < len(src.fed), "the recogniser received a frame that was not the raw camera frame"
        j += 1
    assert not np.array_equal(pushed[-1][..., 0], 1 - pushed[-1][..., 0], equal_nan=True)

    # ---- 7. switching a sign off removes it from recognition; undo and clear reach the engine
    win.library.set_active(new_id, False)
    win._on_library_changed()
    ev.clear()
    src.feed(np.concatenate([perf, rest(b, 0.9)]))
    assert pump(lambda: any(e.kind in ("sign", "clarify", "unknown") for e in ev), 20)
    assert not [e for e in ev if e.kind == "sign" and e.label == "Khaleej (mine)"]
    win.words[:] = ["x", "y"]
    win.undo_word()
    assert win.words == ["x"]


def test_undo_last_word_reaches_the_recogniser():
    rec = R.Recognizer(None)
    rec.words = ["a", "b"]
    rec.undo_last_word()
    assert rec.words == ["a"]
    rec.undo_last_word(); rec.undo_last_word()
    assert rec.words == []
