"""Answering 'which word did you mean?' by holding up fingers (1 = first word, 2 = second, 3 = sign again)."""
import collections
import json
from pathlib import Path

import numpy as np
import pytest

from lingestra import features as F
from lingestra.live import fingers as FG
from lingestra.live.recognizer import Recognizer

ROOT = Path(__file__).resolve().parents[1]
NUMS = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}


def _clips():
    try:
        man = json.load(open(ROOT / "data" / "manifest.json"))
    except OSError:
        return {}
    out = {}
    for e in man:
        if e["category"] == "Numbers" and e["label"] in NUMS and e["label"] not in out:
            f = ROOT / "data" / "landmarks" / f"{e['id']}.npy"
            if f.is_file():
                out[e["label"]] = np.load(f)
    return out


CLIPS = _clips()
needs_data = pytest.mark.skipif(len(CLIPS) < 5, reason="dictionary number-sign landmarks not available")


def _pose_frame(label):
    """A real frame (67, 3) of the dictionary's number sign where the hand shows the number cleanly."""
    n = NUMS[label]
    for fr in CLIPS[label]:
        if FG.hands_count(fr) == n:
            return fr.copy()
    raise AssertionError(label)


@needs_data
@pytest.mark.parametrize("label", sorted(NUMS))
def test_counts_real_number_signs(label):
    """On the dictionary's own number signs the held shape is counted correctly (transition frames may differ)."""
    c = collections.Counter(FG.hands_count(f) for f in CLIPS[label])
    c.pop(None, None)
    assert c.most_common(1)[0][0] == NUMS[label]
    assert c[NUMS[label]] >= 25


def test_unclean_shapes_are_not_counted():
    h = np.zeros((21, 3))
    assert FG.count_fingers(h) is None                       # degenerate
    assert FG.count_fingers(np.full((21, 3), np.nan)) is None


def _two_hands(left_label, right_label):
    """One frame with the LEFT hand showing one number and the RIGHT hand another (6..10 = 5 + the rest)."""
    fr = _pose_frame(right_label)
    other = _pose_frame(left_label)
    fr[F.N_POSE:F.N_POSE + 21] = other[F.N_POSE + 21:F.N_POSE + 42] + np.array([0.25, 0.0, 0.0])
    return fr


@needs_data
def test_two_hands_add_up_to_6_10_and_lowered_hands_are_ignored():
    # CHANGED: two clean counts used to cancel each other; now they are ADDED (five on one hand + the rest on the other)
    assert FG.hands_count(_two_hands("Three", "Two")) == 5
    assert FG.hands_count(_two_hands("Five", "Two")) == 7
    assert FG.hands_count(_two_hands("Five", "Five")) == 10
    assert FG.hands_count(_two_hands("Five", "One")) == 6
    f2 = _pose_frame("Two")
    low = f2.copy()
    hip = np.nanmean(low[[F.L_HIP, F.R_HIP], 1])
    low[F.N_POSE:, 1] += (hip - np.nanmin(low[F.N_POSE:, 1])) + 0.3    # move the hands below the hip line
    assert FG.hands_count(low) is None


@needs_data
def test_picker_needs_arming_delay_and_steady_hold():
    fr = _pose_frame("Two")
    p = FG.FingerPicker(); p.arm(10.0, 3)
    for i in range(20):                                      # shown during the arming delay: ignored
        assert p.push(fr, 10.0 + i * 0.03)[0] is None
    t = 10.0 + FG.FingerPicker.ARM_S + 0.05
    got = None
    for i in range(40):
        got = p.push(fr, t + i * 0.04)[0] or got
    assert got == 2
    # a pose shown only briefly picks nothing
    p.arm(0.0, 3); t = FG.FingerPicker.ARM_S + 0.01
    assert all(p.push(fr, t + i * 0.04)[0] is None for i in range(6))     # 0.24 s
    assert p.push(np.full((67, 3), np.nan), t + 0.3)[0] is None
    # a number above the number of options (here 3 options -> 1..3) is ignored
    p.arm(0.0, 3); fr4 = _pose_frame("Four"); t = FG.FingerPicker.ARM_S + 0.01
    assert all(p.push(fr4, t + i * 0.04)[0] is None for i in range(40))


def _rec_with_question(t0=0.0):
    r = Recognizer(None, templates=[dict(id="a", label="Mother", frames=[CLIPS["One"]]),
                                    dict(id="b", label="Medicine", frames=[CLIPS["Two"]])])
    rest = _pose_frame("Two").copy()
    rest[F.N_POSE:] = np.nan                                  # no hands
    rest[:F.N_POSE, 1] = _pose_frame("Two")[:F.N_POSE, 1]
    for i in range(10):
        r.push_frame(rest, t0 + i * 0.04)
    r.words.append("ok")
    r._clarify("Mother", 0.5, [("Mother", 0.6), ("Medicine", 0.5)], "personal", t0 + 0.4)
    hole = object(); r.words.append(hole); r.pending["hole"] = hole
    return r, t0 + 0.4


@needs_data
@pytest.mark.parametrize("label,expect", [("Three", "Mother"), ("Four", "Medicine")])
def test_recogniser_answers_with_fingers(label, expect):
    """CHANGED numbering: 1 = skip, 2 = retry, 3.. = the suggested words (here 3 = Mother, 4 = Medicine)."""
    r, t = _rec_with_question()
    fr = _pose_frame(label)
    evs, asked_progress = [], 0.0
    for i in range(80):
        evs += r.push_frame(fr, t + 0.04 + i * 0.04)
        asked_progress = max(asked_progress, r.finger[1])
    signs = [e for e in evs if e.kind == "sign"]
    assert [e.label for e in signs] == [expect]
    assert signs[0].answered is True and signs[0].source == "personal"
    assert r.pending is None and r.words == ["ok", expect]       # the answer took the question's place
    assert asked_progress == 1.0 or asked_progress > 0.9


@needs_data
def test_two_fingers_mean_retry_and_one_finger_means_skip():
    for label, kind in (("Two", "retry"), ("One", "skip")):
        r, t = _rec_with_question()
        fr = _pose_frame(label)
        evs = []
        for i in range(80):
            evs += r.push_frame(fr, t + 0.04 + i * 0.04)
        assert [e.kind for e in evs if e.kind in ("sign", "retry", "skip")] == [kind]
        assert r.pending is None and r.words == ["ok"]           # the placeholder is gone, 'ok' stays


@needs_data
def test_up_to_ten_options_and_two_hands_pick_6_to_10():
    words = [(f"W{i}", 0.9 - i * 0.05) for i in range(8)]       # 8 suggestions + skip + retry = 10 options
    r, t = _rec_with_question()
    r._drop_pending(); r.words = ["ok"]
    r._clarify("W0", 0.5, words, "personal", t); hole = object(); r.words.append(hole); r.pending["hole"] = hole
    assert r.picker.options == 10
    fr = _two_hands("Five", "Two")                                 # 7 fingers = option 7 = the 5th suggestion
    evs = []
    for i in range(80):
        evs += r.push_frame(fr, t + 0.04 + i * 0.04)
    assert [e.label for e in evs if e.kind == "sign"] == ["W4"] and r.words == ["ok", "W4"]
    # fewer similar signs: only as many options as there are suggestions (+ skip + retry)
    r2, t2 = _rec_with_question()
    assert r2.picker.options == 4
    # a number beyond the options picks nothing
    evs = []
    for i in range(80):
        evs += r2.push_frame(_pose_frame("Five"), t2 + 0.04 + i * 0.04)
    assert not [e for e in evs if e.kind in ("sign", "retry", "skip")] and r2.pending is not None


@needs_data
def test_number_pose_after_answer_does_not_become_a_sign_or_unknown():
    r, t = _rec_with_question()
    fr = _pose_frame("Three")                                     # option 3 = the first suggested word
    evs = []
    for i in range(250):                                          # keeps holding the number for 10 s
        evs += r.push_frame(fr, t + 0.04 + i * 0.04)
    kinds = [e.kind for e in evs]
    assert kinds.count("sign") == 1 and "unknown" not in kinds and "clarify" not in kinds


@needs_data
def test_stats_report_signing_and_finger_progress():
    r, t = _rec_with_question()
    fr = _pose_frame("Two")
    seen = []
    for i in range(30):
        r.push_frame(fr, t + 0.04 + i * 0.04)
        seen.append(r.stats())
    assert all("signing" in s and "finger" in s and "asking" in s for s in seen)
    assert any(s["finger"][0] == 2 and 0 < s["finger"][1] <= 1 for s in seen)


# ------------------------------------------------------------------ fist on the other hand, taught finger poses
def _shrink(frame, f=0.5):
    """The same frame with every finger foreshortened (joints pulled toward the knuckle): what a closer camera or a
    tilted hand does. Extended fingers then measure ~1.45 instead of ~1.9: not clean for the standard counting."""
    fr = frame.copy()
    for sl in (slice(F.N_POSE, F.N_POSE + 21), slice(F.N_POSE + 21, F.N_POSE + 42)):
        h = fr[sl]
        if np.isnan(h[:, 0]).any():
            continue
        for mcp, joints in ((5, (6, 7, 8)), (9, (10, 11, 12)), (13, (14, 15, 16)), (17, (18, 19, 20))):
            for j in joints:
                h[j] = h[mcp] + (h[j] - h[mcp]) * f
    return fr


def _fist_hand(frame):
    h = frame[F.N_POSE + 21:].copy()
    w = h[0]
    for tip, mcp in zip((8, 12, 16, 20), (5, 9, 13, 17)):
        h[tip] = h[mcp] + (w - h[mcp]) * 0.25
    return h


@needs_data
def test_a_resting_fist_on_the_other_hand_does_not_cancel_the_pick():
    """User report: holding up fingers did nothing. One cause: the other hand resting as a fist read as a clean '0',
    which conflicted with the shown number and cancelled it."""
    fr = _pose_frame("Two")
    both = fr.copy()
    both[F.N_POSE:F.N_POSE + 21] = _fist_hand(fr) + np.array([0.15, 0.0, 0.0])      # left hand: a fist, raised
    assert FG.count_fingers(both[F.N_POSE:F.N_POSE + 21]) == 0
    assert FG.hands_count(both) == 2                                                  # the fist is simply not a number
    assert FG.hands_count(fr) == 2
    only_fist = both.copy(); only_fist[F.N_POSE + 21:] = np.nan
    assert FG.hands_count(only_fist) is None


def _set_ratio(frame, finger_tip, finger_mcp, ratio, hand=slice(F.N_POSE + 21, F.N_POSE + 42)):
    """Frame whose finger (tip index, knuckle index) measures `ratio` (tip / knuckle distance from the wrist, in 3D)."""
    fr = frame.copy(); h = fr[hand]
    P = FG._xyz(h, FG.ASPECT); w = P[0]
    cur = np.linalg.norm(P[finger_tip] - w) / np.linalg.norm(P[finger_mcp] - w)
    q = w + (P[finger_tip] - w) * (ratio / cur)
    h[finger_tip] = (q[0] / FG.ASPECT, q[1], q[2] / FG.ASPECT)
    return fr


def _tilt(frame, deg, hand=slice(F.N_POSE + 21, F.N_POSE + 42)):
    """The hand rotated `deg` degrees about the horizontal axis through the wrist: it points toward / away from the camera."""
    fr = frame.copy(); h = fr[hand]
    P = FG._xyz(h, FG.ASPECT); w = P[0].copy()
    a = np.deg2rad(deg); c, s_ = np.cos(a), np.sin(a)
    Q = (P - w) @ np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]]).T + w
    h[:, 0], h[:, 1], h[:, 2] = Q[:, 0] / FG.ASPECT, Q[:, 1], Q[:, 2] / FG.ASPECT
    return fr


def _bend_ring(frame, ratio=1.3):
    """The user's own '2': the ring finger half-bent (neither clearly up nor clearly down)."""
    return _set_ratio(frame, 16, 13, ratio)


def _set_thumb(frame, ratio, hand=slice(F.N_POSE + 21, F.N_POSE + 42)):
    fr = frame.copy(); h = fr[hand]
    P = FG._xyz(h, FG.ASPECT)
    palm = np.linalg.norm(P[5] - P[17])
    d = P[4] - P[5]; d = d / max(np.linalg.norm(d), 1e-9)
    q = P[5] + d * palm * ratio
    h[4] = (q[0] / FG.ASPECT, q[1], q[2] / FG.ASPECT)
    return fr


@needs_data
def test_the_thumb_does_not_change_the_count_except_for_a_fully_open_hand():
    """A fist with fingers up very often has its thumb resting alongside: it must not become an extra finger."""
    for label, n in (("One", 1), ("Two", 2), ("Three", 3)):
        base = _pose_frame(label)
        for thumb in (0.3, 0.9, 1.3, 1.6):                  # folded, resting alongside, sticking out, far out
            assert FG.hands_count(_set_thumb(base, thumb)) == n, (label, thumb)
    four = _pose_frame("Four")
    assert [FG.hands_count(_set_thumb(four, t)) for t in (0.3, 0.9)] == [4, 4]      # thumb folded / resting alongside
    assert [FG.hands_count(_set_thumb(four, t)) for t in (1.3, 1.6)] == [5, 5]      # four fingers + thumb out = open hand
    five = _pose_frame("Five")
    assert FG.hands_count(_set_thumb(five, 1.3)) == 5       # four fingers up + thumb out = the whole hand
    assert FG.hands_count(_set_thumb(five, 0.4)) == 4       # four fingers up, thumb folded in
    fist = _pose_frame("One").copy()
    fist = _set_ratio(fist, 8, 5, 0.6)                      # index folded too: a fist
    assert FG.count_fingers(_set_thumb(fist, 1.4)[F.N_POSE + 21:F.N_POSE + 42]) == 0    # a fist with its thumb out is not a number


@needs_data
def test_a_slightly_bent_hand_is_read_without_any_teaching_when_the_fingers_split_into_two_clear_groups():
    two = _pose_frame("Two")
    foreshortened = _set_ratio(_set_ratio(two, 8, 5, 1.34), 12, 9, 1.33)      # both fingers only 1.33x (neither up nor down)
    v = FG.finger_features(foreshortened[F.N_POSE + 21:F.N_POSE + 42])
    assert 1.2 < v[0] < 1.5 and 1.2 < v[1] < 1.5                              # the absolute rule alone could not read this
    assert FG.hands_count(foreshortened) == 2                                  # the gap between the groups (up vs curled) can
    mixed = _set_ratio(_set_ratio(two, 8, 5, 1.3), 12, 9, 1.05)               # one finger up-ish, the other at the border
    assert FG.hands_count(mixed) is None                                       # no clear split: no guess


@needs_data
def test_taught_finger_poses_read_hands_the_standard_way_cannot():
    one, three = _pose_frame("One"), _pose_frame("Three")
    two = _bend_ring(_pose_frame("Two"))                    # the signer's own '2': the ring finger always half-bent
    assert FG.hands_count(two) is None                      # standard counting: that finger is neither up nor down
    cal = FG.FingerCalibrator(targets=(1, 2, 3))
    for f in (one, two, three):                              # teach: 40 camera frames per pose (grace + 22 steady samples)
        for _ in range(40):
            st = cal.feed(f)
    assert st["done"] and st["ok"] and set(st["cal"]) == {"1", "2", "3", "ver"}      # (ver = measured in 3D)
    assert [FG.hands_count(f, st["cal"]) for f in (one, two, three)] == [1, 2, 3]
    fist_frame = two.copy(); fist_frame[F.N_POSE + 21:] = _fist_hand(two)
    assert FG.hands_count(fist_frame, st["cal"]) is None    # a fist matches none of the taught poses


@needs_data
def test_teaching_refuses_a_pose_that_is_not_the_requested_number():
    cal = FG.FingerCalibrator(targets=(1, 2, 3))
    two = _pose_frame("Two")
    for _ in range(40):
        st = cal.feed(two)                                                   # asked for 1 finger, shows 2
    assert st["step"] == 0 and "does not look like 1" in st["message"] and not cal.cal
    cal.restart()
    for _ in range(40):
        st = cal.feed(np.full((67, 3), np.nan))                              # no hand at all
    assert st["step"] == 0 and "camera" in st["message"]


@needs_data
def test_teaching_reports_poses_that_look_the_same():
    one = _pose_frame("One")
    cal = FG.FingerCalibrator(targets=(1, 2))
    # the same 1-finger pose shown for both 1 and 2 cannot be accepted for '2'
    for _ in range(30):
        cal.feed(one)
    for _ in range(40):
        st = cal.feed(one)
    assert st["step"] == 1 and "does not look like 2" in st["message"] and not st["done"]


@needs_data
def test_recogniser_answers_with_taught_poses_and_shows_what_it_sees():
    # the signer's own poses: ring finger half-bent in the '3' (neither up nor down for the standard counting)
    frames = {1: _pose_frame("One"), 2: _pose_frame("Two"), 3: _bend_ring(_pose_frame("Three"))}
    assert FG.hands_count(frames[3]) is None
    cal = FG.FingerCalibrator(targets=(1, 2, 3))
    for n in (1, 2, 3):
        for _ in range(40):
            st = cal.feed(frames[n])
    r, t = _rec_with_question()
    evs = []
    for i in range(80):                                                       # no taught poses: nothing is read
        evs += r.push_frame(frames[3], t + 0.04 + i * 0.04)
    assert not [e for e in evs if e.kind == "sign"] and r.stats()["finger_hand"] is True and r.stats()["finger_seen"] is None
    r, t = _rec_with_question()
    r.set_finger_cal(st["cal"])
    evs = []
    seen = set()
    for i in range(80):
        evs += r.push_frame(frames[3], t + 0.04 + i * 0.04)
        seen.add(r.stats()["finger_seen"])
    assert [e.label for e in evs if e.kind == "sign"] == ["Mother"]      # 3 fingers = the first suggested word
    assert 3 in seen                                                          # the live hint can say 'I see 2 fingers'
    r.reset(); assert r.finger_cal == st["cal"]                               # taught poses survive a reset


@needs_data
def test_an_unanswered_question_does_not_hold_the_message_for_ten_seconds():
    """User report: words took ~10 s to reach the message. Cause: with a question open the sentence waited for the whole
    10 s question timeout (measured 10.6 s instead of 1.8 s). The hands are down (no finger answer possible): after a
    short grace the word is skipped and the rest is sent."""
    r = Recognizer(None, templates=[dict(id="a", label="Mother", frames=[CLIPS["One"]]),
                                    dict(id="b", label="Medicine", frames=[CLIPS["Two"]])])
    up = _pose_frame("Two")                                       # hands raised in front of the body
    down = up.copy(); down[F.N_POSE:] = np.nan                    # hands lowered / out of view:
    hip = float(np.nanmean(up[[F.L_HIP, F.R_HIP], 1]))
    down[[F.L_WR, F.R_WR], 1] = hip + 0.15                        # the wrists are below the hip line
    t = 0.0
    for _ in range(10):
        r.push_frame(down, t); t += 0.04
    for _ in range(40):                                           # the signer signs: hands in the signing area
        r.push_frame(up, t); t += 0.04
    r.words.append("ok")
    r._clarify("Mother", 0.5, [("Mother", 0.6), ("Medicine", 0.5)], "personal", t)
    hole = object(); r.words.append(hole); r.pending["hole"] = hole
    t_q, evs, first_sent = t, [], None
    for _ in range(int(16 / 0.04)):                               # the hands go down; nobody taps a button
        new = r.push_frame(down, t); t += 0.04
        evs += new
        if first_sent is None and any(e.kind == "sentence" for e in new):
            first_sent = t - t_q
    assert [e.label for e in evs if e.kind == "skip"] == ["timeout"]
    sent = [e for e in evs if e.kind == "sentence"]
    assert len(sent) == 1 and sent[0].label == "ok"                  # the unanswered word is left out, 'ok' is sent
    assert first_sent is not None and first_sent < 6.0, first_sent   # was about 10.6 s
    assert r.pending is None and r.words == []


@needs_data
def test_slow_decodes_are_throttled_so_frames_do_not_pile_up():
    """With hundreds of signs on, one decode takes longer than a camera frame. Re-decoding at a fixed 0.25 s then
    uses most of the time; the throttle keeps decoding to about a quarter of it."""
    def run(fake_ms):
        r = Recognizer(None, templates=[dict(id="a", label="Mother", frames=[CLIPS["One"]]),
                                        dict(id="b", label="Medicine", frames=[CLIPS["Two"]])])
        orig, calls = r.matcher.decode_sequence, [0]

        def counted(*a, **k):
            calls[0] += 1
            out = orig(*a, **k)
            r.matcher.last_decode_ms = fake_ms            # pretend this decode was slow / fast
            return out
        r.matcher.decode_sequence = counted
        up, t = _pose_frame("Two"), 0.0
        for _ in range(10):
            r.push_frame(up, t); t += 0.04
        # a moving hand for 8 s: the hands stay up, nothing is committed yet
        rng = np.random.default_rng(0)
        for _ in range(200):
            f = up.copy(); f[F.N_POSE:, :2] += rng.normal(0, 0.08, 2)
            r.push_frame(f, t); t += 0.04
        return calls[0]
    fast, slow = run(8.0), run(600.0)
    assert fast >= 12                                         # about every 0.25 s
    assert slow <= fast * 0.45, (fast, slow)                  # 600 ms decodes -> one every 2.4 s


@needs_data
def test_without_auto_finish_lowering_the_hands_does_not_end_the_message():
    """'Send with Space': the message keeps growing; only flush (Space / Send now) ends it."""
    def run(auto):
        r = Recognizer(None, templates=[dict(id="a", label="Mother", frames=[CLIPS["One"]])])
        r.auto_finish = auto
        up = _pose_frame("Two"); down = up.copy(); down[F.N_POSE:] = np.nan
        down[[F.L_WR, F.R_WR], 1] = float(np.nanmean(up[[F.L_HIP, F.R_HIP], 1])) + 0.15
        t = 0.0
        for _ in range(10):
            r.push_frame(down, t); t += 0.04
        for _ in range(40):
            r.push_frame(up, t); t += 0.04
        r.words.append("ok"); r.sentence_flushed = False
        evs = []
        for _ in range(int(8 / 0.04)):                          # the hands are down for 8 s
            evs += r.push_frame(down, t); t += 0.04
        return r, evs, t
    r, evs, t = run(True)
    assert [e.label for e in evs if e.kind == "sentence"] == ["ok"]            # automatic: the message ended
    r, evs, t = run(False)
    assert not [e for e in evs if e.kind == "sentence"] and r.words == ["ok"]   # manual: still waiting
    sent = r.flush_sentence()                                                   # Space / Send now
    assert [e.label for e in sent if e.kind == "sentence"] == ["ok"] and r.words == []


def test_worker_set_auto_finish_is_a_thread_safe_command():
    from lingestra.app.worker import RecognitionWorker
    w = RecognitionWorker("webcam", None, send_mode="none")
    assert w.auto_finish is True
    w.set_auto_finish(False)
    assert w.auto_finish is False and w._cmds.get_nowait() == ("auto_finish", False)


@needs_data
def test_a_hand_tilted_toward_the_camera_is_still_read_correctly():
    """User report: only '2' selected anything. Flat (2D) readings shrink a finger that points at the camera (the index
    finger of a '1' read 1.10 at 65 degrees: 'folded') and fluctuate frame to frame; with depth the reading is steady."""
    for label, n in (("One", 1), ("Two", 2), ("Three", 3), ("Four", 4)):
        base = _pose_frame(label)
        for deg in (-40, 0, 30, 50, 65):
            assert FG.hands_count(_tilt(base, deg)) == n, (label, deg)


@needs_data
def test_the_camera_aspect_ratio_is_part_of_the_measurement():
    base = _pose_frame("One")
    for aspect in (4 / 3, 16 / 9, 1.0):
        assert FG.hands_count(base, aspect=aspect) == 1                   # robust to the real camera's shape
    v43 = FG.finger_features(base[F.N_POSE + 21:F.N_POSE + 42], 4 / 3)
    v169 = FG.finger_features(base[F.N_POSE + 21:F.N_POSE + 42], 16 / 9)
    assert not np.allclose(v43, v169)                                      # (it is used, not ignored)


def test_taught_poses_from_an_older_version_are_ignored():
    from lingestra.app.main_window import MainWindow
    old = {"1": [1.9, .8, .8, .8, .3], "2": [1.9, 1.9, .8, .8, .3]}                  # measured the flat way: no version
    assert MainWindow._valid_cal(old) is None
    new = dict(old, ver=FG.CAL_VERSION)
    assert MainWindow._valid_cal(new) == new


@needs_data
def test_a_flickering_reading_does_not_cancel_a_pose_that_is_clearly_held():
    """User report: only '2' ever selected. A strict 'same number on every frame' rule restarted on every unclear or missing
    frame; a hold is now a majority vote over 0.6 s."""
    two, bent, none = _pose_frame("Two"), _bend_ring(_pose_frame("Two")), np.full((67, 3), np.nan)
    pattern = [two, two, bent, two, two, none, two, two, bent, two]           # 7 of 10 frames read '2', 3 are unclear / missing
    p = FG.FingerPicker(); p.arm(0.0, 4)
    t, got = FG.FingerPicker.ARM_S + 0.01, None
    for i in range(60):
        got = p.push(pattern[i % len(pattern)], t + i * 0.04)[0] or got
    assert got == 2


@needs_data
def test_an_indecisive_hand_and_a_brief_pose_pick_nothing():
    three, four = _pose_frame("Three"), _pose_frame("Four")
    p = FG.FingerPicker(); p.arm(0.0, 5)
    t = FG.FingerPicker.ARM_S + 0.01
    assert all(p.push((three, four)[i % 2], t + i * 0.04)[0] is None for i in range(80))        # 50 / 50: no majority
    p = FG.FingerPicker(); p.arm(0.0, 5)
    assert all(p.push(three, t + i * 0.04)[0] is None for i in range(8))                       # 0.3 s: too short
    # progress and the number being held are reported while the vote builds
    p = FG.FingerPicker(); p.arm(0.0, 5)
    seen = [p.push(three, t + i * 0.04) for i in range(10)]
    assert seen[-1][1] == 3 and 0 < seen[-1][2] <= 1.0


def _reverse_fingers(frame, hand=slice(F.N_POSE + 21, F.N_POSE + 42)):
    """The hand as a tracker reports it when it mistakes it for the mirror-image hand: index <-> pinky, middle <-> ring."""
    fr = frame.copy(); h = fr[hand].copy()
    for a, b in ((5, 17), (9, 13)):
        for k in range(4):
            fr[hand][a + k], fr[hand][b + k] = h[b + k].copy(), h[a + k].copy()
    return fr


@needs_data
def test_a_pose_is_picked_even_when_the_tracker_loses_the_hand_for_a_third_of_the_frames():
    """User report: 'hand visible in 39 of 60 frames'. Frames with no tracked hand abstain; they must not vote against."""
    three, none = _pose_frame("Three"), np.full((67, 3), np.nan)
    rng = np.random.default_rng(1)
    for trial in range(10):
        pattern = [none if rng.random() < 0.40 else three for _ in range(80)]
        p = FG.FingerPicker(); p.arm(0.0, 5)
        got = None
        for i, f in enumerate(pattern):
            got = p.push(f, FG.FingerPicker.ARM_S + 0.01 + i * 0.04)[0] or got
        assert got == 3, trial
    # ... but a hand that is tracked only in a few frames is not enough
    p = FG.FingerPicker(); p.arm(0.0, 5)
    sparse = [three if i in (0, 5, 10) else none for i in range(40)]
    assert all(p.push(f, FG.FingerPicker.ARM_S + 0.01 + i * 0.04)[0] is None for i, f in enumerate(sparse))


@needs_data
def test_the_count_and_taught_poses_do_not_depend_on_which_finger_is_which():
    """User report: the finger raised for '1' was reported as the PINKY while the index stayed at 0.93 (mirror-hand labels)."""
    for label, n in (("One", 1), ("Two", 2), ("Three", 3), ("Four", 4)):
        base = _pose_frame(label)
        assert FG.hands_count(_reverse_fingers(base)) == n, label
    one = _reverse_fingers(_pose_frame("One"))
    v = FG.finger_features(one[F.N_POSE + 21:F.N_POSE + 42])
    assert v[3] > 1.5 and v[0] < 1.2                       # the raised finger is reported as the pinky, like in the report
    # poses taught with normal labels are recognised on reversed hands and the other way round
    cal = FG.FingerCalibrator(targets=(1, 2, 3))
    for label in ("One", "Two", "Three"):
        for _ in range(40):
            st = cal.feed(_pose_frame(label))
    assert st["ok"]
    assert [FG.hands_count(_reverse_fingers(_pose_frame(l)), st["cal"]) for l in ("One", "Two", "Three")] == [1, 2, 3]
    rev = FG.FingerCalibrator(targets=(1, 2, 3))
    for label in ("One", "Two", "Three"):
        for _ in range(40):
            st2 = rev.feed(_reverse_fingers(_pose_frame(label)))
    assert st2["ok"]
    assert [FG.hands_count(_pose_frame(l), st2["cal"]) for l in ("One", "Two", "Three")] == [1, 2, 3]


# ------------------------------------------------------------------ answer mode: a held hand answers, moving on skips
def _zone_with_question():
    """Hands raised in the signing area, a question open (3 suggestions: options 1-5), the message 'ok' + the open word."""
    r = Recognizer(None, templates=[dict(id="a", label="Mother", frames=[CLIPS["One"]]),
                                    dict(id="b", label="Medicine", frames=[CLIPS["Two"]])])
    up = _pose_frame("Two"); down = up.copy(); down[F.N_POSE:] = np.nan
    t = 0.0
    for _ in range(10):
        r.push_frame(down, t); t += 0.04
    for _ in range(40):
        r.push_frame(up, t); t += 0.04
    r.words.append("ok")
    r._clarify("Mother", 0.5, [("Mother", .6), ("Medicine", .5), ("Brother", .4)], "personal", t)
    hole = object(); r.words.append(hole); r.pending["hole"] = hole
    return r, t


@needs_data
def test_a_held_hand_is_never_decoded_as_a_sign_while_a_question_is_open():
    """The app cannot know whether a held hand shape is an answer or the start of the next sign. While a question is open
    a HELD hand (even one the picker cannot read) must not raise a new question / 'unknown' card, which would replace
    the open question under the signer's hand."""
    r, t = _zone_with_question()
    unreadable = _bend_ring(_pose_frame("Two"))              # a hand shape that gives no clean number
    evs = []
    for i in range(int(4.0 / 0.04)):
        evs += r.push_frame(unreadable, t + 0.04 + i * 0.04)
    assert not [e for e in evs if e.kind in ("sign", "clarify", "unknown", "skip", "retry")], [(e.kind, e.label) for e in evs]
    assert r.pending is not None and r._answering                       # the question is still there


@needs_data
def test_carrying_on_signing_instead_of_answering_skips_the_question_after_a_moment():
    r, t = _zone_with_question()
    up = _pose_frame("Two")
    rng = np.random.default_rng(0)
    evs, first = [], None
    for i in range(int(4.0 / 0.04)):                                    # the hands keep MOVING (the next sign)
        f = up.copy(); off = rng.normal(0, 0.10, 2)                   # the arms and hands travel (the next sign)
        f[[F.L_WR, F.R_WR], :2] += off; f[F.N_POSE:, :2] += off
        new = r.push_frame(f, t + 0.04 + i * 0.04)
        evs += new
        if first is None and any(e.kind == "skip" for e in new):
            first = (i + 1) * 0.04
    skips = [e for e in evs if e.kind == "skip"]
    assert [e.label for e in skips] == ["carry_on"]
    # (the tracker needs ~0.6 s to notice that a hold has ended, then CARRY_ON_S of movement: about 2.6 s in all)
    assert first is not None and Recognizer.CARRY_ON_S <= first <= Recognizer.CARRY_ON_S + 1.2, first
    assert r.pending is None and r.words == ["ok"]                       # the unanswered word is left out, the rest stays


@needs_data
def test_a_held_finger_pose_still_answers_in_answer_mode_and_nothing_else_is_decoded():
    r, t = _zone_with_question()
    evs = []
    for i in range(int(3.0 / 0.04)):
        evs += r.push_frame(_pose_frame("Three"), t + 0.04 + i * 0.04)         # option 3 = the first suggestion
    assert [e.kind for e in evs if e.kind in ("sign", "clarify", "unknown", "skip", "retry")] == ["sign"]
    assert [e.label for e in evs if e.kind == "sign"] == ["Mother"] and r.words == ["ok", "Mother"]


@needs_data
def test_finger_answers_can_be_switched_off_for_vocabularies_with_number_signs():
    r, t = _zone_with_question()
    assert r.picker.armed
    r.set_finger_answers(False)
    assert not r.picker.armed and r.finger == (0, 0.0)
    evs = []
    for i in range(int(3.0 / 0.04)):
        evs += r.push_frame(_pose_frame("Three"), t + 0.04 + i * 0.04)
    assert not [e for e in evs if e.kind in ("sign", "skip", "retry")] and r.pending is not None   # fingers do nothing
    assert not r._answering                                                   # ... and no answer mode: signing is decoded as usual
    r2, t2 = _zone_with_question()
    r2.finger_answers = False
    r2._clarify("Mother", 0.5, [("Mother", .6)], "personal", t2 + 0.1)
    assert not r2.picker.armed                                                 # a new question is not armed either
    r2.set_finger_answers(True)
    assert r2.picker.armed and r2.picker.options == 3                          # switching it on arms the open question


def test_worker_set_finger_answers_is_a_thread_safe_command():
    from lingestra.app.worker import RecognitionWorker
    w = RecognitionWorker("webcam", None, send_mode="none")
    assert w.finger_answers is True
    w.set_finger_answers(False)
    assert w.finger_answers is False and w._cmds.get_nowait() == ("finger_answers", False)


@needs_data
def test_a_clean_finger_shape_that_is_moving_is_not_taken_for_an_answer():
    """User concern: the app cannot tell a hand shape that is part of a sign from a selection. A selection is a HELD pose:
    the same shape travelling through the signing space (the next sign) selects nothing."""
    two = _pose_frame("Two")
    p = FG.FingerPicker(); p.arm(0.0, 4)
    t0, got = FG.FingerPicker.ARM_S + 0.01, None
    rng = np.random.default_rng(0)
    for i in range(80):                                       # a clean '2' shape, travelling across the signing space
        f = two.copy(); off = np.array([0.012 * i, 0.0])      # about 0.5 shoulder widths per 0.8 s
        f[[F.L_WR, F.R_WR], :2] += off; f[F.N_POSE:, :2] += off
        got = p.push(f, t0 + i * 0.04)[0] or got
    assert got is None
    p = FG.FingerPicker(); p.arm(0.0, 4)
    for i in range(80):                                       # the same shape held (a little natural sway)
        f = two.copy(); off = rng.normal(0, 0.004, 2)
        f[[F.L_WR, F.R_WR], :2] += off; f[F.N_POSE:, :2] += off
        got = p.push(f, t0 + i * 0.04)[0] or got
    assert got == 2
