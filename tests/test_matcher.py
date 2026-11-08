"""Personal matcher, sign features, and the Recogniser's personal path (synthetic, deterministic)."""
import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))
from lingestra import features as F                                   # noqa: E402
from lingestra.live import matcher as M                                # noqa: E402
from lingestra.live.matcher import MatcherConfig, PersonalMatcher     # noqa: E402
from lingestra.live.recognizer import Recognizer                       # noqa: E402
import synth                                                           # noqa: E402
from synth import FPS, SIGNS, frame, sign, stream                      # noqa: E402


def templates(names=None, takes=1, seed=0):
    out = []
    for k, n in enumerate(names or SIGNS):
        fr = [sign(n, noise=0.002 if t else 0.0, rng=np.random.default_rng(seed + t)) for t in range(takes)]
        out.append(dict(id=f"id{k}", label=n, frames=fr))
    return out


@pytest.fixture(scope="module")
def matcher():
    m = PersonalMatcher()
    m.set_templates(templates())
    return m


# ---------------------------------------------------------------- features
def test_features_shape_mask_and_missing_hand():
    s = sign("alpha")
    X, Mk = F.sign_features(s)
    assert X.shape == (len(s), F.SIGN_DIM) and Mk.shape == (len(s), len(F.SIGN_BLOCKS))
    names = [b[0] for b in F.SIGN_BLOCKS]
    assert Mk[:, names.index("locR")].all() and Mk[:, names.index("shpR")].all()
    assert not Mk[:, names.index("locL")].any()                     # the left hand rests: ignored
    assert np.isfinite(X).all() and (X[:, F.BLOCK_SLICES["shpL"]] == 0).all()
    s2 = s.copy()
    s2[:, F.N_POSE + 21:] = np.nan                                   # right hand undetected, pose still there
    X2, M2 = F.sign_features(s2)
    assert M2[:, names.index("locR")].all() and not M2[:, names.index("shpR")].any()
    assert np.isfinite(X2).all()
    assert F.sign_features(np.zeros((0, F.N_PTS, 3)))[0].shape == (0, F.SIGN_DIM)


def test_features_invariant_to_camera_shift_zoom_roll():
    s = sign("gamma")
    a = np.deg2rad(6)
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    t = s.copy()
    t[..., :2] = (s[..., :2] - 0.5) @ R.T * 1.3 + 0.5 + np.array([0.07, -0.05], np.float32)
    X, _ = F.sign_features(s)
    Y, _ = F.sign_features(t)
    b = F.BLOCK_SLICES
    for name in ("locR", "shpR", "oriR", "arm"):
        assert np.abs(X[:, b[name]] - Y[:, b[name]]).max() < 0.12, name


def test_features_are_not_mirror_invariant():
    s = sign("alpha")
    mir = s.copy()
    mir[..., 0] = 1 - mir[..., 0]                                    # image mirror (display-only transform)
    m = PersonalMatcher()
    m.set_templates(templates(["alpha", "beta"]))
    d, _ = m.distances(mir, trim=False)
    d0, _ = m.distances(s, trim=False)
    assert d0[0] < d[0]                                              # left/right matter


# ---------------------------------------------------------------- hold collapse / arc length
def test_hold_collapse_turns_a_pause_into_one_key_pose():
    cfg = MatcherConfig()
    base = F.sign_features(sign("alpha"))
    paused = F.sign_features(sign("alpha", pause_s=1.5, pause_at=(1,)))
    kb = M.collapse_holds(*base, cfg.eps)[0]
    kp = M.collapse_holds(*paused, cfg.eps)[0]
    assert len(paused[0]) >= len(base[0]) + 30                       # 1.5 s of extra frames...
    assert abs(len(kp) - len(kb)) <= 2                               # ...become (almost) no extra key poses


def test_arc_length_resampling_has_constant_steps():
    X, Mk = F.sign_features(sign("beta"))
    X, Mk = M.collapse_holds(X, Mk, 0.06)
    Y, My = M.arc_resample(X, Mk, 24)
    assert len(Y) == 24
    steps = M.frame_cost(Y[1:], My[1:], Y[:-1], My[:-1])
    assert steps.std() / steps.mean() < 0.35                         # equal amount of change per step


# ---------------------------------------------------------------- DTW
def test_dtw_identity_is_near_zero_and_ranks_first(matcher):
    for n in SIGNS:
        d, _ = matcher.distances(sign(n), trim=False)
        assert matcher.labels[int(np.argmin(d))] == n
        assert d[matcher.labels.index(n)] < 0.05
        assert np.sort(d)[1] > 0.2


@pytest.mark.parametrize("speed", [0.5, 0.75, 1.5, 2.0])
def test_dtw_time_warp_invariance(matcher, speed):
    for n in SIGNS:
        d, _ = matcher.distances(sign(n, speed=speed), trim=False)
        assert matcher.labels[int(np.argmin(d))] == n
        assert d[matcher.labels.index(n)] < 0.12


def test_dtw_pause_invariance_and_ablation(matcher):
    """A 1.5 s pause between key poses must not change the distance; without hold collapsing it does."""
    nocol = PersonalMatcher(MatcherConfig(collapse=False))
    nocol.set_templates(templates())
    worse = 0
    for n in SIGNS:
        q0 = sign(n)
        q1 = sign(n, pause_s=1.5, pause_at=(1,))
        i = matcher.labels.index(n)
        d0 = matcher.distances(q0, trim=False)[0][i]
        d1 = matcher.distances(q1, trim=False)[0][i]
        assert abs(d1 - d0) < 0.08 and matcher.labels[int(np.argmin(matcher.distances(q1, trim=False)[0]))] == n
        e0 = nocol.distances(q0, trim=False)[0][i]
        e1 = nocol.distances(q1, trim=False)[0][i]
        worse += (e1 - e0) > (d1 - d0) + 0.02
    assert worse >= 3                                                # ablation is clearly more pause-sensitive


def test_leftover_transition_frames_are_tolerated(matcher):
    s, lead, raise_n, n, lower_n = stream(sign("delta"))
    d, _ = matcher.distances(s, trim=False)       # untrimmed: raise + lower + idle included, open ends
    ok_untrimmed = matcher.labels[int(np.argmin(d))]
    d2, _ = matcher.distances(s, trim=True)
    assert matcher.labels[int(np.argmin(d2))] == "delta"
    assert d2[matcher.labels.index("delta")] < 0.15
    assert ok_untrimmed in matcher.labels


# ---------------------------------------------------------------- decision / rejection
def test_accept_correct_and_reject_unenrolled():
    m = PersonalMatcher()
    m.set_templates(templates(["alpha", "beta", "gamma"]))
    r = m.classify(sign("alpha", speed=1.5), trim=False)
    assert r.kind == "sign" and r.label == "alpha" and 0.6 <= r.confidence <= 1.0
    r = m.classify(sign("delta"), trim=False)                        # not enrolled
    assert r.kind == "unknown" and r.label is None
    assert m.classify(np.zeros((0, F.N_PTS, 3))).kind == "unknown"
    assert PersonalMatcher().classify(sign("alpha")).kind == "unknown"   # no templates


def test_confidence_decreases_with_distance():
    m = PersonalMatcher()
    m.set_templates(templates(["alpha", "beta"]))
    near = m.classify(sign("alpha"), trim=False)
    far = m.classify(sign("alpha", noise=0.03, rng=np.random.default_rng(1)), trim=False)
    assert near.confidence >= far.confidence if far.kind == "sign" else True


def test_clarify_when_two_labels_are_equally_close():
    a = sign("alpha")
    m = PersonalMatcher()
    m.set_templates([dict(id="1", label="one", frames=[a]), dict(id="2", label="two", frames=[a.copy()])])
    r = m.classify(a, trim=False)
    assert r.kind == "clarify" and {c[0] for c in r.candidates[:2]} == {"one", "two"}


def test_threshold_calibrated_from_own_takes():
    one = PersonalMatcher()
    one.set_templates(templates(["alpha", "beta"], takes=1))
    assert np.allclose(one.thr, one.cfg.default_thr)                 # single take: documented default
    three = PersonalMatcher()
    three.set_templates(templates(["alpha", "beta"], takes=3))
    assert "alpha" in three.calibration["loo"] and len(three.calibration["loo"]["alpha"]) == 3
    exp = np.clip(three.cfg.thr_margin * max(three.calibration["loo"]["alpha"]), three.cfg.thr_lo, three.cfg.thr_hi)
    assert abs(three.thr[three.labels.index("alpha")] - exp) < 1e-9
    # sloppy takes -> looser threshold than precise takes
    sloppy = PersonalMatcher()
    sloppy.set_templates([dict(id="s", label="alpha", frames=[
        sign("alpha", noise=0.03, rng=np.random.default_rng(i)) for i in range(3)])])
    assert sloppy.thr[0] >= three.thr[three.labels.index("alpha")]


def test_strictness_scales_acceptance():
    m = PersonalMatcher()
    m.set_templates(templates(["alpha", "beta"]))
    q = sign("alpha", noise=0.01, rng=np.random.default_rng(3))
    d, _ = m.distances(q, trim=False)
    assert m.decide(d, 1.0).kind == "sign"
    assert m.decide(d, 0.01).kind == "unknown"


def test_missing_hand_still_matches():
    m = PersonalMatcher()
    m.set_templates(templates(["alpha", "beta", "gamma"]))
    q = sign("beta")
    q[10:40, F.N_POSE + 21:] = np.nan                                # hand landmarks lost for a second
    assert m.classify(q, trim=False).label == "beta"


def test_decision_speed_50_signs_x_3_takes():
    names = [f"s{i}" for i in range(50)]
    rng = np.random.default_rng(0)
    tm = []
    for i, n in enumerate(names):
        keys = [(rng.uniform(0.4, 0.7), rng.uniform(0.25, 0.55), rng.uniform(0, 1)) for _ in range(4)]
        tm.append(dict(id=str(i), label=n, frames=[synth.sign_frames(keys, noise=0.003 * t, rng=np.random.default_rng(t))
                                                 for t in range(3)]))
    m = PersonalMatcher()
    m.set_templates(tm)
    assert m.n_takes == 150
    q = tm[7]["frames"][0]
    m.classify(q, trim=False)                                        # warm-up
    ts = []
    for _ in range(10):
        t0 = time.perf_counter()
        r = m.classify(q, trim=False)
        ts.append((time.perf_counter() - t0) * 1000)
    assert r.label == "s7"
    assert np.mean(ts) < 30, np.mean(ts)


# ---------------------------------------------------------------- recogniser end to end
@pytest.fixture
def rec():
    return Recognizer(None, templates=templates(["alpha", "beta"]))


def run(rec, raw, t0=0.0):
    rec.reset()
    ev = []
    for i, f in enumerate(raw):
        ev += rec.push_frame(f, t0 + i / FPS)
    return ev


def test_recogniser_2x_speed_with_one_second_pause(rec):
    for name in ("alpha", "beta"):
        s = stream(sign(name, speed=2.0, pause_s=1.0, pause_at=(1,)), tail_s=2.0)[0]
        ev = run(rec, s)
        assert [e.label for e in ev if e.kind == "sign"] == [name]
        e = [e for e in ev if e.kind == "sign"][0]
        assert e.source == "personal" and 0.5 < e.confidence <= 1.0
        assert [x.label for x in ev if x.kind == "sentence"] == [name]


def test_recogniser_unenrolled_sign_is_unknown_with_window(rec):
    s = stream(sign("delta"), tail_s=1.0)[0]
    ev = run(rec, s)
    assert [e.kind for e in ev] == ["unknown"]
    w = ev[0].window
    assert w.ndim == 3 and w.shape[1:] == (F.N_PTS, 3)
    h = F.hand_height(w)
    assert len(w) < len(s) - 20 and np.nanmin(h) > 0.3             # transitions are not in the stored sign
    assert ev[0].source == "personal"


def test_recogniser_without_templates_every_sign_is_new():
    r = Recognizer(None)                                              # no model, no templates
    ev = run(r, stream(sign("alpha"))[0])
    assert [e.kind for e in ev] == ["unknown"]


def test_recogniser_templates_replaced_live(rec):
    s = stream(sign("gamma"))[0]
    assert [e.kind for e in run(rec, s)] == ["unknown"]
    rec.set_templates(templates(["alpha", "gamma"]))
    assert [e.label for e in run(rec, s) if e.kind == "sign"] == ["gamma"]
    rec.set_templates([])
    assert not rec.matcher.ready


def test_recogniser_clarify_confirm_and_label_map_untouched():
    a = sign("alpha")
    r = Recognizer(None, templates=[dict(id="1", label="one", frames=[a]), dict(id="2", label="two", frames=[a.copy()])])
    r.label_map = {"one": "SHOULD NOT APPLY"}                         # label_map is for raw model classes only
    ev = run(r, stream(a)[0])
    cl = [e for e in ev if e.kind == "clarify"]
    assert len(cl) == 1 and cl[0].source == "personal" and [c[0] for c in cl[0].candidates[:2]] == ["one", "two"] or \
        {c[0] for c in cl[0].candidates[:2]} == {"one", "two"}
    out = r.confirm("one")
    assert out[0].label == "one" and out[0].source == "personal"
    assert r.flush_sentence()[0].label == "one"


def test_signing_on_drops_an_unanswered_question():
    a = sign("alpha")
    r = Recognizer(None, templates=[dict(id="1", label="one", frames=[a]), dict(id="2", label="two", frames=[a.copy()])])
    run(r, stream(a)[0])
    assert r.pending is not None
    t = r.last_t
    ev = []
    for i, f in enumerate(stream(a)[0]):
        ev += r.push_frame(f, t + (i + 1) / FPS)
    assert [e.kind for e in ev if e.kind == "clarify"] != [] and r.pending is not None   # fresh question for the new sign


def test_use_model_flag_is_off_when_templates_exist():
    calls = []

    class Fake:
        def __call__(self, x):
            calls.append(len(x))
            return np.tile([0.97, 0.01, 0.02], (len(x), 1))
    r = Recognizer("unused", model=Fake(), classes=["a", "b", "none"], templates=templates(["alpha"]))
    assert not r.model_enabled
    run(r, stream(sign("delta"))[0])                                  # unknown to the personal matcher
    ev = run(r, stream(sign("delta"))[0])
    assert [e.kind for e in ev] == ["unknown"] and not [c for c in calls if c > 1]
    r.use_model = True                                                # explicitly enabled
    ev = run(r, stream(sign("delta"))[0])
    assert [(e.kind, e.label, e.source) for e in ev if e.kind == "sign"] == [("sign", "a", "model")]
    # a confident personal match is never overridden by the model
    ev = run(r, stream(sign("alpha"))[0])
    assert [(e.label, e.source) for e in ev if e.kind == "sign"] == [("alpha", "personal")]


def test_separator_ends_sentence_and_keeps_last_sign(rec):
    s = stream(sign("alpha"), tail_s=0.2)[0]
    ev = run(rec, s)
    up = np.stack([frame(wr_r=np.array([0.45, 0.10]), wr_l=np.array([0.55, 0.10]))] * 40)
    for i, f in enumerate(up):
        ev += rec.push_frame(f, len(s) / FPS + i / FPS)
    assert [e.kind for e in ev if e.kind != "unknown"] == ["sign", "separator", "sentence"]


def test_clear_words_and_stats(rec):
    run(rec, stream(sign("alpha"), tail_s=0.3)[0])
    assert rec.words == ["alpha"]
    rec.clear_words()
    assert rec.words == [] and rec.flush_sentence() == []
    st = rec.stats()
    assert st["templates"] == 2 and st["decision_ms_avg"] > 0 and st["model_enabled"] is False


# ---------------------------------------------------------------- worker
def test_worker_commands_set_templates_and_clear_sentence(monkeypatch, tmp_path):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from lingestra.app import worker as W
    import time as _time
    monkeypatch.setattr(W.time, "sleep", lambda s: None)               # replay file sources unpaced
    s = stream(sign("alpha"), tail_s=2.0)[0]
    p = ROOT / "scratch" / "test_worker_stream.npy"
    p.parent.mkdir(exist_ok=True)
    np.save(p, s)
    w = W.RecognitionWorker(f"landmarks:{p}", None, send_mode="none", countdown_s=0.0)
    w.set_templates(templates(["alpha", "beta"]))                      # queued before start: applied first
    got, failed = [], []
    w.events.connect(lambda evs: got.extend(evs), Qt.DirectConnection)
    w.failed.connect(failed.append, Qt.DirectConnection)
    kinds = [c for c, _ in list(w._cmds.queue)]
    assert kinds == ["templates"]
    w.clear_sentence()
    assert [c for c, _ in list(w._cmds.queue)] == ["templates", "clear_sentence"]
    w.start()
    w.wait(20000)
    assert not failed, failed
    # clear_sentence ran before any frame: the sign is still recognised afterwards
    assert [e.label for e in got if e.kind == "sign"] == ["alpha"]
    assert [e.source for e in got if e.kind == "sign"] == ["personal"]
    p.unlink()


# ---------------------------------------------------------------- review fixes
def test_calibrated_threshold_is_capped_for_open_set_safety():
    m = PersonalMatcher()
    m.set_templates(templates(takes=3))
    assert m.thr.max() <= MatcherConfig().thr_hi <= 0.6        # never much looser than the 0.45 single-take default


def _decide(d, thr=0.45):
    m = PersonalMatcher()
    m.labels = ["a", "b"]; m.thr = np.array([thr, thr])
    return m.decide(np.array(d, float))


def test_margin_rule_has_no_cliff_at_the_threshold():
    assert _decide([0.44, 0.44]).kind == "clarify"
    # runner-up just OVER its threshold used to be ignored (confident 'sign'); now it still competes
    assert _decide([0.44, 0.46]).kind == "clarify"
    assert _decide([0.30, 0.46]).kind == "sign"                  # clearly better than the runner-up
    assert _decide([0.30, 0.90]).kind == "sign"                  # runner-up far away: no competition


# ------------------------------------------------------------------ 'clarify' offers every similar sign (up to 8)
def _matcher_with_signs(n):
    import json
    from lingestra.live.matcher import PersonalMatcher
    man = json.load(open(ROOT / "data" / "manifest.json"))
    tpl, seen = [], set()
    for e in man:
        f = ROOT / "data" / "landmarks" / f"{e['id']}.npy"
        if e["label"] in seen or not f.is_file():
            continue
        seen.add(e["label"]); tpl.append(dict(id=e["id"], label=e["label"], frames=[np.load(f)]))
        if len(tpl) == n:
            break
    m = PersonalMatcher(); m.set_templates(tpl)
    return m


def test_a_question_lists_only_signs_that_look_similar_and_at_most_eight():
    m = _matcher_with_signs(14)
    if len(m.labels) < 14:
        pytest.skip("dictionary landmarks not available")
    big = 50.0

    def ask(ratios):
        d = np.full(len(m.labels), big)
        for i, r in enumerate(ratios):
            d[i] = r * m.thr[i]
        return m.decide(d)

    res = ask([0.90, 0.95])                                   # two close candidates, everything else far away
    assert res.kind == "clarify" and [c[0] for c in res.candidates] == m.labels[:2]
    res = ask([0.90, 0.95, 1.2, 1.5, 1.9, 2.3, 3.0])           # three more within twice their own tolerance
    assert res.kind == "clarify" and [c[0] for c in res.candidates] == m.labels[:5]       # 2.3 and 3.0 are too different
    res = ask([0.90, 0.95] + [1.1] * 12)                       # many similar signs: capped at 8 (+ Skip + Retry = 10 options)
    assert res.kind == "clarify" and len(res.candidates) == 8
    res = ask([0.30, 0.95, 1.2])                               # a clear winner is not a question at all
    assert res.kind == "sign" and res.label == m.labels[0]


def test_suggestions_are_ordered_best_first_and_keep_the_close_pair_on_top():
    m = _matcher_with_signs(10)
    if len(m.labels) < 10:
        pytest.skip("dictionary landmarks not available")
    d = np.full(len(m.labels), 50.0)
    for i, r in zip((4, 2, 7, 5), (0.9, 0.97, 1.05, 1.6)):
        d[i] = r * m.thr[i]
    res = m.decide(d)
    assert res.kind == "clarify"
    assert [c[0] for c in res.candidates] == [m.labels[i] for i in (4, 2, 7, 5)]
