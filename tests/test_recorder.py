"""SignRecorder + Recognizer.label_map/suspend tests (landmarks only, timestamps only)."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lingestra import features as F                                   # noqa: E402
from lingestra.live.recognizer import Recognizer                       # noqa: E402
from lingestra.live.recorder import SignRecorder                       # noqa: E402

FPS = 25


def _clip():
    man = json.load(open(ROOT / "data" / "manifest.json"))
    for e in man:
        raw = np.load(ROOT / "data" / "landmarks" / f"{e['id']}.npy")
        on = np.where(F.hand_activity(raw) > 0.25)[0]
        if len(on) > 25 and on[0] > 5 and on[-1] < len(raw) - 5 and on[-1] - on[0] < 150:
            return raw, on[0], on[-1]
    pytest.skip("no suitable clip")


def seg_of(raw, a, b, speed=1.0):
    seg = raw[a:b + 1]
    n = max(int(len(seg) / speed), 4)
    src = np.linspace(0, len(seg) - 1, n)
    lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, len(seg) - 1); w = (src - lo)[:, None, None]
    out = seg[lo] * (1 - w) + seg[hi] * w
    nn = np.where(w < 0.5, seg[lo], seg[hi])      # nearest frame where interpolation hit a NaN
    return np.where(np.isnan(out), nn, out).astype(np.float32)


def build(raw, a, b, lead_s=1.0, tail_s=2.0, speed=1.0):
    return np.concatenate([np.repeat(raw[:1], int(lead_s * FPS), 0), seg_of(raw, a, b, speed),
                           np.repeat(raw[-1:], int(tail_s * FPS), 0)])


def feed(rec, data, t0=0.0, fps=FPS):
    st = None
    for i, f in enumerate(data):
        st = rec.push(f, t0 + i / fps)
        if st.state in ("done", "failed"):
            break
    return st


def test_clip_trimmed_to_active_part():
    raw, a, b = _clip()
    rec = SignRecorder(countdown_s=3.0)
    rec.start(0.0)
    st = feed(rec, np.concatenate([np.repeat(raw[:1], 3 * FPS, 0), build(raw, a, b)]))
    assert st.state == "done" and st.clip.shape[1:] == (F.N_PTS, 3)
    # CHANGED: the clip no longer carries pre/post padding. It is the signing part only: the raise
    # from rest and the lowering are excluded (lowering the hands is the stop gesture).
    active = b - a + 1
    assert 8 <= len(st.clip) <= active
    assert st.frames == len(st.clip)
    h = F.hand_height(st.clip)
    assert h[0] >= 0.38 and h[-1] >= 0.25                  # starts and ends inside signing space
    assert rec.quality(st.clip)[0]


@pytest.mark.parametrize("speed", [0.5, 1.0, 2.0])
def test_fast_and_slow(speed):
    raw, a, b = _clip()
    rec = SignRecorder(countdown_s=0.0)
    rec.start(0.0)
    st = feed(rec, build(raw, a, b, speed=speed))
    assert st.state == "done", st.message


def test_no_hands_fails():
    raw, a, b = _clip()
    data = np.repeat(raw[:1], 30 * FPS, 0).copy()
    rec = SignRecorder(countdown_s=0.5, wait_timeout_s=5.0)
    rec.start(0.0)
    st = feed(rec, data)
    assert st.state == "failed" and "Timed out" in st.message


def test_hands_not_visible_fails():
    raw, a, b = _clip()
    seg = build(raw, a, b).copy()
    seg[:, F.N_POSE:] = np.nan                     # pose says hands up, hand landmarks missing
    rec = SignRecorder(countdown_s=0.0)
    rec.start(0.0)
    st = feed(rec, seg)
    assert st.state == "failed" and "hands" in st.message.lower()


def test_person_not_in_frame_and_nan_stream():
    rec = SignRecorder(countdown_s=0.0)
    rec.start(0.0)
    st = feed(rec, np.full((10 * FPS, F.N_PTS, 3), np.nan, np.float32))
    assert st.state == "failed" and "not in frame" in st.message


def test_too_short():
    raw, a, b = _clip()
    rec = SignRecorder(countdown_s=0.0, min_frames=1000)
    rec.start(0.0)
    st = feed(rec, build(raw, a, b))
    assert st.state == "failed" and "short" in st.message.lower()
    assert SignRecorder().quality(np.zeros((3, F.N_PTS, 3)))[0] is False


def test_countdown_by_timestamps_and_cancel():
    raw, a, b = _clip()
    rest = raw[0]
    rec = SignRecorder(countdown_s=3.0)
    assert rec.push(rest, 0.0).state == "idle"
    rec.start(100.0)
    st = rec.push(rest, 100.0)
    assert st.state == "countdown" and abs(st.seconds_left - 3.0) < 1e-6
    st = rec.push(rest, 101.5)
    assert st.state == "countdown" and abs(st.seconds_left - 1.5) < 1e-6
    # a huge timestamp gap (non-real-time source) finishes the countdown immediately
    st = rec.push(rest, 103.2)
    assert st.state == "waiting" and "Raise your hands" in st.message
    rec.cancel()
    assert rec.push(rest, 104.0).state == "idle"
    # cancel while recording
    rec.start(0.0)
    data = build(raw, a, b, lead_s=0.2)
    for i in range(0, 3 * FPS + 25):
        st = rec.push(data[min(i - 3 * FPS, len(data) - 1)] if i > 3 * FPS else rest, i / FPS)
        if st.state == "recording":
            break
    assert st.state == "recording"
    rec.cancel()
    assert rec.push(rest, 99.0).state == "idle"


def test_noise_does_not_trigger():
    raw, a, b = _clip()
    rec = SignRecorder(countdown_s=0.0, wait_timeout_s=100)
    rec.start(0.0)
    up = raw[(a + b) // 2]
    for i in range(200):
        f = up if i % 25 == 0 else raw[0]          # single-frame blips
        assert rec.push(f, i / FPS).state == "waiting"


def test_max_s_stops_recording():
    raw, a, b = _clip()
    up = raw[(a + b) // 2]
    rec = SignRecorder(countdown_s=0.0, max_s=2.0)
    rec.start(0.0)
    st = feed(rec, np.stack([up] * 200))
    assert st.state in ("done", "failed")
    if st.state == "done":
        assert len(st.clip) <= 2.0 * FPS + 3


# ---------------------------------------------------------------- recognizer additions
class FakeModel:
    def __init__(self, probs):
        self.probs = np.asarray(probs, float)

    def __call__(self, x):
        return np.tile(self.probs, (len(x), 1))


def fake_rec(probs):
    classes = ["a", "b", "c", "none"]
    return Recognizer("unused", model=FakeModel(probs), classes=classes)


def run(rec, data):
    out = []
    for i, f in enumerate(data):
        out += rec.push_frame(f, i / FPS)
    return out


def test_label_map_sign_sentence():
    raw, a, b = _clip()
    rec = fake_rec([0.95, 0.02, 0.02, 0.01])
    rec.label_map = {"a": "Hello there"}
    ev = run(rec, build(raw, a, b))
    assert [(e.kind, e.label) for e in ev] == [("sign", "Hello there"), ("sentence", "Hello there")]


def test_label_map_clarify_and_confirm():
    raw, a, b = _clip()
    rec = fake_rec([0.6, 0.3, 0.05, 0.05])
    rec.label_map = {"a": "Alpha", "b": "Beta"}
    ev = run(rec, build(raw, a, b))
    cl = [e for e in ev if e.kind == "clarify"][0]
    assert cl.label == "Alpha" and [c[0] for c in cl.candidates] == ["Alpha", "Beta"]
    out = rec.confirm("Beta")                       # display label accepted
    assert out[0].label == "Beta"
    rec.confirm("a")                                # raw class accepted
    assert rec.flush_sentence()[0].label == "Beta Alpha"


def test_suspend_no_phantom_sign():
    raw, a, b = _clip()
    rec = fake_rec([0.95, 0.02, 0.02, 0.01])
    data = build(raw, a, b)
    mid = a + 1 + int(FPS)
    rec.suspend(True)
    assert run(rec, data) == [] and rec.n == 0 and not rec.buf
    rec.suspend(False)
    # resume with hands at rest: nothing happens
    assert run(rec, np.repeat(raw[:1], 3 * FPS, 0)) == []
    # normal signing afterwards still works
    ev = run(rec, data)
    assert [e.kind for e in ev] == ["sign", "sentence"]
    # suspend mid-sign discards the partial sign
    rec.reset()
    part = data[: FPS + 10 + (b - a) // 2]
    assert run(rec, part) == []
    rec.suspend(True); rec.suspend(False)
    assert not rec.in_episode and rec.votes == 0
    assert run(rec, np.repeat(raw[:1], 3 * FPS, 0)) == []
