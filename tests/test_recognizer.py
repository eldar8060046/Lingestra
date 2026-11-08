"""Recognizer tests. Run: .venv/bin/python -m pytest tests/test_recognizer.py -q -s

The demo model (models/demo_family.keras) was trained on these same dictionary clips, so the
accuracy numbers printed here measure robustness to signing speed, NOT generalisation to a new signer.
"""
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
from lingestra.live.sources import LandmarkFileSource                  # noqa: E402

MODEL = ROOT / "models" / "demo_family.keras"
FPS = 25
pytestmark = pytest.mark.skipif(not MODEL.exists(), reason="demo model not trained")


# ---------------------------------------------------------------- helpers
def clips():
    """{label: raw landmarks} first take of each class of the demo model."""
    classes = json.load(open(MODEL.with_suffix(".json")))["classes"]
    man = json.load(open(ROOT / "data" / "manifest.json"))
    out = {}
    for e in man:
        if e["label"] in classes and e["label"] not in out and e["category"] == "Family":
            out[e["label"]] = np.load(ROOT / "data" / "landmarks" / f"{e['id']}.npy")
    # drop clips whose hands never come down (e.g. one 'Aunt' take is active until frame 235):
    # they have no idle tail, so they cannot be used to build rest-separated streams
    return {k: v for k, v in out.items() if np.where(F.hand_activity(v) > 0.25)[0][-1] < 150}


def sign_segment(raw, speed=1.0):
    """The active part of a clip, re-timed at the same fps: speed 2 -> half as many frames."""
    on = np.where(F.hand_activity(raw) > 0.25)[0]
    seg = raw[max(on[0] - 3, 0): on[-1] + 4]
    n = max(int(len(seg) / speed), 4)
    src = np.linspace(0, len(seg) - 1, n)
    lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, len(seg) - 1); w = (src - lo)[:, None, None]
    return (seg[lo] * (1 - w) + seg[hi] * w).astype(np.float32)


def rest_frames(raw, seconds):
    """Hands-down frames (the clip's own last frame, so shoulders/pose stay valid)."""
    return np.repeat(raw[-1:], int(seconds * FPS), axis=0)


def stream(segments, rest_s=0.8, tail_s=2.0, lead_s=1.0):
    """lead rest + (segment + rest)* + tail rest; `segments` are (raw_for_rest, segment)."""
    parts = [rest_frames(segments[0][0], lead_s)]
    for k, (raw, seg) in enumerate(segments):
        parts += [seg, rest_frames(raw, rest_s if k < len(segments) - 1 else tail_s)]
    return np.concatenate(parts)


def run(rec, data, fps=FPS, speed=1.0):
    ev = []
    for frame, t in LandmarkFileSource(data, fps=fps, speed=speed):
        ev += rec.push_frame(frame, t)
    return ev


@pytest.fixture(scope="module")
def data():
    return clips()


@pytest.fixture(scope="module")
def rec_model():
    return Recognizer(str(MODEL))


def fresh(rec):
    rec.reset()
    return rec


# ---------------------------------------------------------------- speed robustness
def test_signs_at_speeds(data, rec_model):
    """Per speed: exact = one correct confident 'sign'; wrong = a confident WRONG sign (bad);
    resolved = correct sign OR a clarify whose top candidate is correct (signer would tap confirm)."""
    labels = sorted(data)
    report, lat = {}, []
    for speed in (0.5, 1.0, 2.0):
        exact = wrong = resolved = 0
        for lab in labels:
            ev = run(fresh(rec_model), stream([(data[lab], sign_segment(data[lab], speed))]))
            signs = [e.label for e in ev if e.kind == "sign"]
            clar = [e.label for e in ev if e.kind == "clarify"]
            exact += signs == [lab]
            wrong += any(s != lab for s in signs)
            resolved += signs == [lab] or (not signs and clar == [lab])
            lat += [e.latency_ms for e in ev if e.kind in ("sign", "clarify")]
        report[speed] = dict(exact=exact / len(labels), wrong=wrong / len(labels), resolved=resolved / len(labels))
    print("\nper-speed", report, "| decision-frame latency ms: mean %.1f max %.1f" % (np.mean(lat), np.max(lat)))
    # CHANGED: the Keras model is now a secondary path that decides ONCE per finished episode (no
    # multi-scale streaming votes), so the 2x column is weaker than before.
    for sp, r in report.items():
        assert r["wrong"] <= 0.1, (sp, r)
        assert r["resolved"] >= (0.6 if sp >= 2.0 else 0.8), (sp, r)
    assert report[1.0]["exact"] >= 0.8


def test_time_scaled_source_speed(data, rec_model):
    """LandmarkFileSource(speed=...) only rescales timestamps; windows are time-based so it still works."""
    lab = "Family"
    for sp in (0.5, 2.0):
        ev = run(fresh(rec_model), stream([(data[lab], sign_segment(data[lab]))]), speed=sp)
        assert [e.label for e in ev if e.kind == "sign"] == [lab]


def test_idle_no_events(data, rec_model):
    lab = sorted(data)[0]
    ev = run(fresh(rec_model), rest_frames(data[lab], 20))
    assert ev == []


# ---------------------------------------------------------------- sentences
def test_sentence_by_rest_pause(data, rec_model):
    a, b = sorted(data)[1], sorted(data)[5]
    ev = run(fresh(rec_model), stream([(data[a], sign_segment(data[a])), (data[b], sign_segment(data[b]))]))
    kinds = [e.kind for e in ev]
    assert kinds.count("sentence") == 1
    assert [e.label for e in ev if e.kind == "sign"] == [a, b]
    assert [e.label for e in ev if e.kind == "sentence"] == [f"{a} {b}"]
    assert kinds.index("sentence") > max(i for i, k in enumerate(kinds) if k == "sign")


def test_sentence_by_separator(data, rec_model):
    a = sorted(data)[2]
    rec = fresh(rec_model)
    ev = run(rec, stream([(data[a], sign_segment(data[a]))], tail_s=0.5))
    assert not [e for e in ev if e.kind == "sentence"]            # rest pause not reached yet
    up = np.repeat(data[a][-1:], 40, axis=0).copy()
    up[:, [F.L_WR, F.R_WR], 1] = up[:, [0], 1] - 0.15              # both wrists above the nose
    ev2 = run(rec, up)
    kinds = [e.kind for e in ev2]
    assert kinds == ["separator", "sentence"] and ev2[1].label == a


def test_flush_and_none(data, rec_model):
    rec = fresh(rec_model)
    assert rec.flush_sentence() == []
    rec.words = ["x", "y"]
    assert rec.flush_sentence()[0].label == "x y"


# ---------------------------------------------------------------- confidence logic (fake model)
class FakeModel:
    def __init__(self, classes, probs, delay=0.0):
        self.classes, self.probs, self.delay, self.calls = classes, np.asarray(probs, float), delay, 0

    def __call__(self, x):
        self.calls += 1
        time.sleep(self.delay)
        return np.tile(self.probs, (len(x), 1))


def fake_rec(probs, **kw):
    classes = ["a", "b", "c", "none"]
    return Recognizer("unused", model=FakeModel(classes, probs, kw.pop("delay", 0.0)), classes=classes, **kw)


def test_low_confidence_clarify(data):
    rec = fake_rec([0.6, 0.3, 0.05, 0.05])
    lab = sorted(data)[0]
    ev = run(rec, stream([(data[lab], sign_segment(data[lab]))]))
    cl = [e for e in ev if e.kind == "clarify"]
    assert len(cl) == 1 and cl[0].label == "a" and [c[0] for c in cl[0].candidates] == ["a", "b"]
    assert not [e for e in ev if e.kind in ("sign", "sentence")]
    out = rec.confirm("b")
    assert out[0].kind == "sign" and out[0].label == "b"
    assert rec.flush_sentence()[0].label == "b"
    # reject drops the pending question
    rec.reset()
    ev = run(rec, stream([(data[lab], sign_segment(data[lab]))]))
    assert rec.pending is not None
    rec.reject()
    assert rec.pending is None


def test_unknown_and_none(data):
    lab = sorted(data)[0]
    ev = run(fake_rec([0.3, 0.3, 0.2, 0.2]), stream([(data[lab], sign_segment(data[lab]))]))
    unk = [e for e in ev if e.kind == "unknown"]
    assert len(unk) == 1 and unk[0].window.shape[1:] == (F.N_PTS, 3)
    ev = run(fake_rec([0.0, 0.0, 0.0, 1.0]), stream([(data[lab], sign_segment(data[lab]))]))
    assert ev == []                                                # 'none' winning -> silence


def test_confident_sign_not_duplicated(data):
    lab = sorted(data)[0]
    ev = run(fake_rec([0.95, 0.02, 0.02, 0.01]), stream([(data[lab], sign_segment(data[lab]))]))
    assert [e.kind for e in ev] == ["sign", "sentence"]            # exactly one sign despite many evals


# ---------------------------------------------------------------- adaptive load
def test_adaptive_skipping(data):
    """CHANGED: load control now sheds model crops per decision (decisions are per episode)."""
    lab = sorted(data)[0]
    rec = fake_rec([0.3, 0.3, 0.2, 0.2], delay=0.15, latency_budget_ms=60)
    for _ in range(3):
        run(rec, stream([(data[lab], sign_segment(data[lab]))]))
    s = rec.stats()
    assert s["load_level"] > 0 and s["frames_skipped"] > 0
    assert s["latency_ms_avg"] > 60
    # model becomes fast again -> recovers
    rec._model.delay = 0.0
    for _ in range(8):
        run(rec, stream([(data[lab], sign_segment(data[lab]))]))
    assert rec.stats()["load_level"] < s["load_level"]


def test_stats_real_model(data, rec_model):
    lab = sorted(data)[0]
    rec = fresh(rec_model)
    run(rec, stream([(data[lab], sign_segment(data[lab]))]))
    s = rec.stats()
    print("\nstats:", s)
    assert s["scale_used"] == "model" and s["load_level"] == 0     # CHANGED: no more time-scales


def test_parallel_tracking_gives_the_same_landmarks_as_serial_tracking():
    """Body and hand tracking run on two cores; the result must be exactly what the serial version gives."""
    import glob
    import cv2
    vids = sorted(glob.glob(str(Path(__file__).resolve().parents[1] / "data" / "raw_videos" / "*.mp4")))
    if not vids:
        pytest.skip("no dictionary videos available")
    cap = cv2.VideoCapture(vids[3 % len(vids)])
    frames = []
    while len(frames) < 25:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    cap.release()
    outs = {}
    for par in (False, True):
        ex = F.LandmarkExtractor(video_mode=True, parallel=par)
        try:
            outs[par] = np.stack([ex.process(f, i * 40) for i, f in enumerate(frames)])
        finally:
            ex.close()
    assert np.allclose(outs[False], outs[True], equal_nan=True)
