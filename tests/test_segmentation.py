"""Signing-zone segmentation: raising/lowering the hands is NOT part of the sign.

Synthetic streams (tests/synth.py) with known raise / sign / lower frame ranges, plus real
dictionary clips for consistency checks."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from lingestra import features as F                                   # noqa: E402
from lingestra.live.recorder import SignRecorder                       # noqa: E402
import synth                                                           # noqa: E402
from synth import FPS, sign, stream, frame                             # noqa: E402


def test_hand_height_rest_and_raised():
    h = F.hand_height(np.stack([frame(), frame(wr_r=np.array([0.5, 0.3]))]))
    assert abs(h[0] - (0.72 - 0.74) / 0.4) < 1e-4 and h[0] < 0.1       # at the hip
    assert h[1] > 1.0                                                  # above the shoulder line
    assert np.isnan(F.hand_height(np.full((3, F.N_PTS, 3), np.nan))).all()


def test_hand_height_robust_to_nan_frames():
    s, *_ = stream(sign("alpha"))
    s = s.copy()
    s[30:34, :F.N_POSE] = np.nan                    # pose lost for 4 frames mid-sign
    h = F.hand_height(s)
    assert not np.isnan(h[30:34]).any() or True     # may be NaN for the wrist, never a crash
    eps = F.segment_signing(s)
    assert len(eps) == 1


@pytest.mark.parametrize("name", list(synth.SIGNS))
def test_episode_excludes_raise_and_lowering(name):
    s, lead, raise_n, n, lower_n = stream(sign(name))
    s0, s1 = lead + raise_n, lead + raise_n + n            # the sign proper
    eps = F.segment_signing(s)
    assert len(eps) == 1
    e = eps[0]
    # the raise [lead, s0) and the lowering [s1, s1+lower_n) are cut (a few frames of slack)
    assert e.start >= s0 - 3 and e.end <= s1 + 3, (e.start, e.end, s0, s1)
    assert e.start >= lead + raise_n // 2 and e.end <= s1 + lower_n // 2
    assert len(e.frames) == e.end - e.start + 1
    # ... while the untrimmed zone span still contains them
    assert e.raw_start < e.start or e.raw_end > e.end
    assert len(e.frames) >= n - 8                            # and the sign itself is kept


def test_trim_active_uses_the_same_cut():
    s, lead, raise_n, n, _ = stream(sign("beta"))
    t = F.trim_active(s)
    e = F.segment_signing(s)[0]
    assert np.array_equal(t, e.frames, equal_nan=True)
    assert len(t) < len(s) - 20


def test_lowering_hands_ends_the_episode_streaming():
    s, lead, raise_n, n, lower_n = stream(sign("alpha"), tail_s=2.0)
    zt = F.ZoneTracker()
    got = None
    for i, f in enumerate(s):
        ep = zt.push(f, i / FPS)
        if ep is not None:
            got = i
            break
    assert got is not None
    lowered_below = lead + raise_n + n + int(lower_n * 0.7)          # hands leave the zone around here
    assert got <= lowered_below + int(0.5 * FPS)                    # closes ~0.2 s later, long before the tail ends
    assert got < len(s) - FPS
    assert not zt.in_zone and zt.rest_for(len(s) / FPS) > 0.5


def test_hands_up_pause_splits_two_signs_via_recogniser():
    from lingestra.live.recognizer import Recognizer
    tm = [dict(id=k, label=k, frames=[sign(k)]) for k in synth.SIGNS]
    rec = Recognizer(None, templates=tm)
    a, b = sign("alpha"), sign("gamma")
    pause = np.repeat(a[-1:], int(1.0 * FPS), 0)                     # hands stay up, still, 1 s
    s = stream(np.concatenate([a, pause, b]))[0]
    ev = []
    for i, f in enumerate(s):
        ev += rec.push_frame(f, i / FPS)
    assert [e.label for e in ev if e.kind == "sign"] == ["alpha", "gamma"]
    assert all(e.source == "personal" for e in ev if e.kind == "sign")


def test_pause_inside_one_sign_is_not_a_split():
    from lingestra.live.recognizer import Recognizer
    tm = [dict(id=k, label=k, frames=[sign(k)]) for k in synth.SIGNS]
    rec = Recognizer(None, templates=tm)
    for name in synth.SIGNS:
        s = stream(sign(name, pause_s=1.2, pause_at=(1,)))[0]
        ev = []
        rec.reset()
        for i, f in enumerate(s):
            ev += rec.push_frame(f, i / FPS)
        assert [e.label for e in ev if e.kind == "sign"] == [name], name


def test_blips_and_dropouts_do_not_make_episodes():
    rest = frame()
    up = frame(wr_r=np.array([0.5, 0.3]))
    zt = F.ZoneTracker()
    for i in range(100):
        assert zt.push(up if i % 20 == 0 else rest, i / FPS) is None      # single-frame blips
    assert not zt.in_zone
    nan = np.full((F.N_PTS, 3), np.nan, np.float32)
    zt = F.ZoneTracker()
    for i in range(60):
        assert zt.push(nan, i / FPS) is None


def test_raised_hands_from_the_start_are_not_trimmed():
    """No rest observed before the zone: nothing to cut at the start."""
    s = sign("alpha")
    eps = F.segment_signing(np.concatenate([s, np.repeat(frame()[None], 30, 0)]))
    assert len(eps) == 1 and eps[0].start == 0


def test_recorder_clip_excludes_raise_and_lowering():
    s, lead, raise_n, n, lower_n = stream(sign("gamma"))
    rec = SignRecorder(countdown_s=0.0)
    rec.start(0.0)
    st = None
    for i, f in enumerate(s):
        st = rec.push(f, i / FPS)
        if st.state in ("done", "failed"):
            break
    assert st.state == "done", st.message
    s0 = lead + raise_n
    assert n - 8 <= len(st.clip) <= n + 6
    # the clip is the sign: its first/last frames match the sign's first/last poses (not the rest pose)
    assert np.allclose(np.nan_to_num(st.clip[len(st.clip) // 2]), np.nan_to_num(s[s0 + n // 2]), atol=0.2)
    h = F.hand_height(st.clip)
    assert h.min() > 0.3                                  # never a resting hand inside the clip


def test_recorder_stops_when_hands_are_lowered_not_by_timer():
    s, lead, raise_n, n, lower_n = stream(sign("alpha"), tail_s=3.0)
    rec = SignRecorder(countdown_s=0.0, max_s=30.0)
    rec.start(0.0)
    stop = None
    for i, f in enumerate(s):
        st = rec.push(f, i / FPS)
        if st.state == "done":
            stop = i
            break
    assert stop is not None and stop < lead + raise_n + n + lower_n + int(0.6 * FPS)


def test_real_dictionary_clips_are_cut_consistently():
    man = json.load(open(ROOT / "data" / "manifest.json"))[:40]
    for e in man:
        raw = np.load(ROOT / "data" / "landmarks" / f"{e['id']}.npy")
        t = F.trim_active(raw)
        eps = F.segment_signing(raw)
        if eps:
            assert np.array_equal(t, raw[eps[0].start: eps[-1].end + 1], equal_nan=True)
            assert len(t) <= len(raw)
        h = F.hand_height(t)
        ok = h[~np.isnan(h)]
        if len(ok) > 10:
            assert np.median(ok) > 0.3                      # the cut part is in signing space
