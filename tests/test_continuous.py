"""Continuous signing: several signs in a row WITHOUT lowering the hands (synthetic, deterministic).

Streams are built from tests/synth.py signs joined by a smooth transition (no hold), wrapped in a raise
from rest and a lowering to rest, so the zone tracker sees exactly ONE episode. The recogniser always
receives un-mirrored landmarks. See docs/recognition.md ('Continuous signing') and
scripts/benchmark_continuous.py for the dictionary-data evaluation."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
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
from lingestra.live.matcher import MatcherConfig, PersonalMatcher, Segment   # noqa: E402
from lingestra.live.recognizer import Recognizer                       # noqa: E402
import synth                                                           # noqa: E402
from synth import FPS, SIGNS, frame, moves, stream, _lerp             # noqa: E402

LEAD = int(0.8 * FPS) + int(0.5 * FPS)       # synth.stream: idle + raise before the chain starts


# ---------------------------------------------------------------- builders
def path_of(keys, speed=1.0, pause_s=0.0, pause_at=(), move_s=0.5, edge=0.25):
    mf = max(int(move_s * FPS / speed), 2)
    p = moves(keys, mf, int(pause_s * FPS), pause_at)
    eh = max(int(edge * FPS), 1)
    return np.concatenate([np.repeat(p[:1], eh, 0), p, np.repeat(p[-1:], eh, 0)])


def chain(items, trans_s=0.25, noise=0.0, seed=0):
    """items: [(key poses, kwargs of path_of)] -> (landmark frames of the chain, [(start, end)] per sign).
    Signs are joined by a straight transition (movement epenthesis); the hands never come down."""
    rng = np.random.default_rng(seed)
    rows, spans = [], []
    for i, (keys, kw) in enumerate(items):
        p = path_of(keys, **kw)
        if i > 0:
            rows.append(_lerp(rows[-1][-1], p[0], max(int(trans_s * FPS), 2)))
        s0 = sum(len(r) for r in rows)
        rows.append(p)
        spans.append((s0, s0 + len(p) - 1))
    path = np.concatenate(rows)
    out = []
    for x, y, c in path:
        w = np.array([x, y]) + (rng.normal(0, noise, 2) if noise else 0)
        out.append(frame(wr_r=w, curl_r=float(c)))
    return np.stack(out), spans


def named(names, **kw):
    return chain([(SIGNS[n], kw.get("per", {})) for n in names], **{k: v for k, v in kw.items() if k != "per"})


def templates(names, extra=None, takes=1):
    out = []
    for k, n in enumerate(names):
        keys = (extra or {}).get(n) or SIGNS[n]
        fr = [synth.sign_frames(keys, noise=0.002 if t else 0.0, rng=np.random.default_rng(t)) for t in range(takes)]
        out.append(dict(id=f"id{k}", label=n, frames=fr))
    return out


def run(rec, raw, t0=0.0):
    """Push a stream; returns events and the frame index at which each was emitted."""
    rec.reset()
    ev, at = [], []
    for i, f in enumerate(raw):
        e = rec.push_frame(f, t0 + i / FPS)
        ev += e
        at += [i] * len(e)
    return ev, at


def signs_of(ev):
    return [e.label for e in ev if e.kind == "sign"]


@pytest.fixture(scope="module")
def rec3():
    return Recognizer(None, templates=templates(["alpha", "beta", "gamma"]))


# ---------------------------------------------------------------- 1. several signs, no hold
@pytest.mark.parametrize("names", [["alpha", "beta"], ["beta", "gamma"], ["gamma", "alpha"],
                                   ["alpha", "beta", "gamma"], ["gamma", "alpha", "beta"],
                                   ["beta", "alpha", "gamma", "alpha"]])
@pytest.mark.parametrize("trans", [0.1, 0.4])
def test_signs_in_a_row_without_a_hold_are_decoded_in_order(rec3, names, trans):
    ch, _ = named(names, trans_s=trans)
    st = stream(ch, tail_s=2.0)[0]
    assert len(F.segment_signing(st)) == 1                       # really ONE episode: the hands never came down
    ev, _ = run(rec3, st)
    assert signs_of(ev) == names, [(e.kind, e.label) for e in ev]
    assert [e.label for e in ev if e.kind == "sentence"] == [" ".join(names)]
    assert all(e.source == "personal" for e in ev if e.kind == "sign")
    assert not [e for e in ev if e.kind in ("clarify", "unknown")]


def test_decode_sequence_returns_ordered_segments(rec3):
    ch, spans = named(["alpha", "beta", "gamma"])
    segs = rec3.matcher.decode_sequence(ch)
    assert [s.label for s in segs] == ["alpha", "beta", "gamma"]
    assert all(isinstance(s, Segment) and s.kind == "sign" and not s.ambiguous for s in segs)
    for s, (a, b) in zip(segs, spans):                            # every segment sits on its own sign
        assert abs(s.start - a) <= 6 and abs(s.end - b) <= 12, (s, a, b)   # a segment owns its trailing pose / transition
    assert all(x.end < y.start for x, y in zip(segs, segs[1:]))
    assert all(0 < s.confidence <= 1 and s.distance < 0.45 and s.runner_up for s in segs)


def test_a_single_sign_is_one_segment(rec3):
    for n in SIGNS:
        if n == "delta":
            continue
        segs = rec3.matcher.decode_sequence(synth.sign(n))
        assert [s.label for s in segs] == [n], (n, segs)


# ---------------------------------------------------------------- 2. no false splits
def test_single_multi_pose_sign_is_not_split():
    """gamma has four key poses; its first two poses are ALSO an enrolled sign. gamma must stay one word."""
    gamma = SIGNS["gamma"]
    rec = Recognizer(None, templates=templates(["alpha", "gamma", "gamma_head"], extra={"gamma_head": gamma[:2]}))
    for _ in range(2):
        ev, _ = run(rec, stream(synth.sign("gamma"), tail_s=2.0)[0])
        assert signs_of(ev) == ["gamma"], [(e.kind, e.label) for e in ev]
    assert not [e for e in ev if e.kind in ("clarify", "unknown")]
    # the same, as a plain decode of the episode
    assert [s.label for s in rec.matcher.decode_sequence(synth.sign("gamma"))] == ["gamma"]


def test_speed_change_and_short_pause_inside_a_sign_do_not_split(rec3):
    for per in ({"speed": 1.6}, {"speed": 0.6}, {"pause_s": 0.4, "pause_at": (1,)},
                {"speed": 1.6, "pause_s": 0.45, "pause_at": (2,)}):
        ch, _ = chain([(SIGNS["gamma"], per)])
        ev, _ = run(rec3, stream(ch, tail_s=2.0)[0])
        assert signs_of(ev) == ["gamma"], (per, [(e.kind, e.label) for e in ev])
        ch, _ = chain([(SIGNS["alpha"], {}), (SIGNS["gamma"], per)])      # ... and inside a sequence
        ev, _ = run(rec3, stream(ch, tail_s=2.0)[0])
        assert signs_of(ev) == ["alpha", "gamma"], (per, [(e.kind, e.label) for e in ev])


def test_a_long_pause_inside_a_sign_is_not_a_split_even_inside_a_sequence():
    """A pause >= 0.6 s is a HOLD. The first half of a sign matches its own template at ~0.5 of the threshold,
    so a hold must not be taken as 'the word is over' unless the match is very good."""
    rec = Recognizer(None, templates=templates(list(SIGNS)))
    for name in SIGNS:
        for pause in (1.0, 1.5):
            ch, _ = chain([(SIGNS[name], {"pause_s": pause, "pause_at": (1,)})])
            ev, _ = run(rec, stream(ch, tail_s=2.0)[0])
            assert signs_of(ev) == [name], (name, pause, [(e.kind, e.label) for e in ev])
    ch, _ = chain([(SIGNS["beta"], {}), (SIGNS["alpha"], {"pause_s": 1.2, "pause_at": (1,)})])
    ev, _ = run(rec, stream(ch, tail_s=2.0)[0])
    assert signs_of(ev) == ["beta", "alpha"], [(e.kind, e.label) for e in ev]


def test_a_short_pause_between_signs_is_not_a_hold_but_still_two_words(rec3):
    items = [(SIGNS["alpha"], {}), (SIGNS["beta"], {})]
    ch, spans = chain(items, trans_s=0.2)
    gap = np.repeat(ch[spans[0][1]][None], int(0.35 * FPS), 0)       # 0.35 s of stillness: below the 0.6 s hold
    ch = np.concatenate([ch[:spans[0][1] + 1], gap, ch[spans[0][1] + 1:]])
    ev, _ = run(rec3, stream(ch, tail_s=2.0)[0])
    assert signs_of(ev) == ["alpha", "beta"]


def test_holds_between_signs_still_separate_them(rec3):
    """The old behaviour (hands up, still >= 0.6 s) keeps working, in both modes."""
    ch, spans = named(["alpha", "beta"], trans_s=0.2)
    gap = np.repeat(ch[spans[0][1]][None], int(1.0 * FPS), 0)
    ch = np.concatenate([ch[:spans[0][1] + 1], gap, ch[spans[0][1] + 1:]])
    st = stream(ch, tail_s=2.0)[0]
    for flag in (True, False):
        r = Recognizer(None, templates=templates(["alpha", "beta"]), continuous=flag)
        ev, _ = run(r, st)
        assert signs_of(ev) == ["alpha", "beta"], (flag, [(e.kind, e.label) for e in ev])


# ---------------------------------------------------------------- 3. un-enrolled sign in the middle
def test_unenrolled_sign_between_two_enrolled_signs_is_unknown_for_that_segment_only():
    rec = Recognizer(None, templates=templates(["alpha", "beta"]))
    ch, spans = named(["alpha", "delta", "beta"], trans_s=0.25)        # delta is not enrolled
    st = stream(ch, tail_s=2.0)[0]
    ev, at = run(rec, st)
    kinds = [(e.kind, e.label) for e in ev if e.kind != "sentence"]
    assert kinds == [("sign", "alpha"), ("unknown", None), ("sign", "beta")], kinds
    unk = next(e for e in ev if e.kind == "unknown")
    d0, d1 = spans[1]
    assert len(unk.window) <= (d1 - d0 + 1) + int(0.8 * FPS)           # the window is delta, not the sentence
    assert len(unk.window) >= 0.6 * (d1 - d0 + 1)
    assert [e.label for e in ev if e.kind == "sentence"] == ["alpha beta"]
    # the unknown window really is delta's pose track: it starts after alpha and ends before beta
    wr = unk.window[:, F.R_WR, :2]
    assert wr[:, 0].min() < 0.34 and np.median(wr[:, 0]) < 0.46 and wr[0, 0] < 0.45   # delta lives at x ~ 0.30-0.45, alpha starts at 0.62


# ---------------------------------------------------------------- 4. timing and stability
def test_words_are_committed_before_the_hands_are_lowered():
    rec = Recognizer(None, templates=templates(["alpha", "beta", "gamma"]))
    ch, spans = named(["alpha", "beta", "gamma"], trans_s=0.25)
    st, lead, rs, L, low = stream(ch, tail_s=2.0)
    chain_end = lead + rs + len(ch) - 1                                # last frame with the hands up
    ev, at = run(rec, st)
    when = {e.label: a for e, a in zip(ev, at) if e.kind == "sign"}
    assert list(when) == ["alpha", "beta", "gamma"]
    assert when["alpha"] < lead + rs + spans[1][1], when               # 'alpha' appears while beta is being signed
    assert when["alpha"] > lead + rs + spans[0][1]                     # ... and never before alpha is over
    assert when["beta"] < chain_end, (when, chain_end)                 # beta appears while gamma is being signed
    assert when["gamma"] >= chain_end - int(0.5 * FPS)                 # the last word can only wait for the end


def test_committed_words_never_change():
    rec = Recognizer(None, templates=templates(["alpha", "beta", "gamma"]))
    names = ["gamma", "alpha", "beta", "alpha"]
    ch, _ = named(names)
    st = stream(ch, tail_s=0.2)[0]                                     # sentence not flushed: words stay in rec.words
    rec.reset()
    snaps = []
    for i, f in enumerate(st):
        rec.push_frame(f, i / FPS)
        snaps.append(list(rec.words))
    for a, b in zip(snaps, snaps[1:]):
        assert b[:len(a)] == a, (a, b)                                 # only ever appended to
    assert snaps[-1] == names


def test_streaming_matches_offline_decode_on_a_noisy_long_stream():
    rec = Recognizer(None, templates=templates(["alpha", "beta", "gamma", "delta"]))
    names = ["delta", "alpha", "gamma", "beta", "delta", "alpha"]
    ch, _ = named(names, noise=0.003, seed=3)
    ev, _ = run(rec, stream(ch, tail_s=2.0)[0])
    assert signs_of(ev) == names
    assert [s.label for s in rec.matcher.decode_sequence(ch)] == names


# ---------------------------------------------------------------- 5. continuous = False is the old behaviour
def test_continuous_false_reproduces_the_old_behaviour():
    names = ["alpha", "beta", "gamma"]
    old = Recognizer(None, templates=templates(names), continuous=False)
    new = Recognizer(None, templates=templates(names))
    assert old.continuous is False and new.continuous is True
    # (a) single signs: identical events
    for n in names:
        st = stream(synth.sign(n, speed=1.3, pause_s=0.5, pause_at=(1,)), tail_s=2.0)[0]
        a, _ = run(old, st)
        b, _ = run(new, st)
        assert [(e.kind, e.label) for e in a] == [(e.kind, e.label) for e in b] == [("sign", n), ("sentence", n)]
    # (b) the user's bug: with no hold the old path asks instead of answering, the new path answers
    ch, _ = named(["alpha", "beta"], trans_s=0.2)
    st = stream(ch, tail_s=2.0)[0]
    a, _ = run(old, st)
    assert signs_of(a) != ["alpha", "beta"] and len(signs_of(a)) < 2
    b, _ = run(new, st)
    assert signs_of(b) == ["alpha", "beta"]
    # (c) the toggle is safe at any moment and the flag is honoured
    new.set_continuous(False)
    c, _ = run(new, st)
    assert [(e.kind, e.label) for e in c] == [(e.kind, e.label) for e in a]
    new.set_continuous(True)


def test_label_map_suspend_confirm_reject_undo_stats_still_work():
    rec = Recognizer(None, templates=templates(["alpha", "beta"]))
    ev, _ = run(rec, stream(synth.sign("alpha"), tail_s=0.2)[0])
    assert rec.words == ["alpha"] and signs_of(ev) == ["alpha"]
    st = rec.stats()
    assert st["templates"] == 2 and st["decision_ms_avg"] > 0 and st["model_enabled"] is False
    rec.undo_last_word()
    assert rec.words == []
    rec.suspend(True)
    assert rec.push_frame(synth.frame(), 99.0) == []
    rec.suspend(False)
    ev, _ = run(rec, stream(synth.sign("beta"), tail_s=2.0)[0])
    assert signs_of(ev) == ["beta"]


# ---------------------------------------------------------------- 6. a question for one segment only
def _twin_templates():
    """alpha and alpha_twin are nearly the same sign: a clarify between them."""
    twin = [(x + 0.004, y, c) for x, y, c in SIGNS["alpha"]]
    t = templates(["beta", "gamma"])
    t += templates(["alpha"], takes=1) + templates(["alpha_twin"], extra={"alpha_twin": twin})
    return t


def test_ambiguous_segment_is_a_question_for_that_segment_only_and_keeps_its_place():
    rec = Recognizer(None, templates=_twin_templates())
    ch, _ = named(["beta", "alpha", "gamma"])
    ev, at = run(rec, stream(ch, tail_s=0.2)[0])
    kinds = [(e.kind, e.label) for e in ev]
    assert [k for k, _ in kinds] == ["sign", "clarify", "sign"], kinds    # beta and gamma go through
    assert kinds[0][1] == "beta" and kinds[2][1] == "gamma"
    q = next(e for e in ev if e.kind == "clarify")
    assert {c[0] for c in q.candidates[:2]} == {"alpha", "alpha_twin"}
    assert rec.pending is not None
    assert rec.words[0] == "beta" and rec.words[2] == "gamma"           # the question holds its place in the sentence
    out = rec.confirm("alpha_twin")
    assert [(e.kind, e.label) for e in out] == [("sign", "alpha_twin")]
    assert rec.words == ["beta", "alpha_twin", "gamma"] and rec.pending is None
    assert [e.label for e in rec.flush_sentence()] == ["beta alpha_twin gamma"]


def test_rejected_question_leaves_the_other_words():
    rec = Recognizer(None, templates=_twin_templates())
    ch, _ = named(["beta", "alpha", "gamma"])
    run(rec, stream(ch, tail_s=0.2)[0])
    assert rec.pending is not None
    rec.reject()
    assert rec.pending is None and rec.words == ["beta", "gamma"]


# ---------------------------------------------------------------- 7. features / cost
def test_segment_features_equal_sign_features_of_the_same_frames():
    for n in ("alpha", "gamma"):
        s = synth.sign(n)
        ep = M.episode_features(s)
        X, Mk = M.segment_xm(ep, 0, len(s))
        X0, M0 = F.sign_features(s)
        assert (Mk == M0).all()
        dom = F.BLOCK_SLICES["dom"]
        keep = np.ones(F.SIGN_DIM, bool)
        keep[dom] = False
        assert np.abs(X[:, keep] - X0[:, keep]).max() < 1e-4
        assert np.abs(X[:, dom] - X0[:, dom]).max() < 0.01              # dom: path length of smoothed vs raw wrist
    # a hand that is only used in part of the episode is judged per segment
    ch, spans = chain([(SIGNS["alpha"], {})])
    both = ch.copy()
    ep = M.episode_features(both)
    names = [b[0] for b in F.SIGN_BLOCKS]
    X, Mk = M.segment_xm(ep, 0, len(both))
    assert Mk[:, names.index("locR")].all() and not Mk[:, names.index("locL")].any()


def test_decode_time_bound_50_signs_x_3_takes_8_seconds():
    rng = np.random.default_rng(5)
    tmpl, keys_of = [], {}
    for k in range(50):
        keys = [(float(rng.uniform(0.3, 0.7)), float(rng.uniform(0.25, 0.55)), float(rng.uniform(0, 1)))
                for _ in range(int(rng.integers(3, 6)))]
        keys_of[k] = keys
        takes = [synth.sign_frames(keys, noise=0.002 if t else 0.0, rng=np.random.default_rng(100 * k + t),
                                   move_s=0.45 + 0.05 * t) for t in range(3)]
        tmpl.append(dict(id=f"s{k}", label=f"s{k}", frames=takes))
    m = PersonalMatcher()
    m.set_templates(tmpl)
    assert m.n_takes == 150
    pick = (3, 11, 20, 31, 44)
    ch, _ = chain([(keys_of[i], {}) for i in pick], trans_s=0.2)
    ch = ch[:int(8 * FPS)]
    assert len(ch) == int(8 * FPS)
    ms = []
    for _ in range(5):
        t0 = time.perf_counter()
        segs = m.decode_sequence(ch)
        ms.append((time.perf_counter() - t0) * 1000)
    assert [s.label for s in segs][:3] == ["s3", "s11", "s20"], [s.label for s in segs]
    assert min(ms) <= 150, ms                                           # best of 5: the machine may be busy
    assert np.median(ms) <= 250, ms


# ---------------------------------------------------------------- 8. end to end through the real worker
def test_worker_end_to_end_continuous_and_toggle(monkeypatch):
    from PySide6.QtCore import Qt
    from PySide6.QtWidgets import QApplication
    QApplication.instance() or QApplication([])
    from lingestra.app import worker as W
    monkeypatch.setattr(W.time, "sleep", lambda s: None)               # replay the file unpaced
    names = ["alpha", "beta", "gamma"]
    ch, _ = named(names)
    st = stream(ch, tail_s=2.0)[0]
    p = ROOT / "scratch" / "test_continuous_stream.npy"
    p.parent.mkdir(exist_ok=True)
    np.save(p, st)
    try:
        w = W.RecognitionWorker(f"landmarks:{p}", None, send_mode="none", countdown_s=0.0)
        assert w.continuous is True
        w.set_templates(templates(["alpha", "beta", "gamma"]))
        got, failed, sent = [], [], []
        w.events.connect(lambda evs: got.extend(evs), Qt.DirectConnection)
        w.failed.connect(failed.append, Qt.DirectConnection)
        w.sent.connect(lambda t, ok, d: sent.append(t), Qt.DirectConnection)
        w.start()
        w.wait(60000)
        assert not failed, failed
        assert [e.label for e in got if e.kind == "sign"] == names
        assert [e.label for e in got if e.kind == "sentence"] == ["alpha beta gamma"] and sent == ["alpha beta gamma"]
        # the toggle travels as a worker command and reaches the recogniser
        w2 = W.RecognitionWorker(f"landmarks:{p}", None, send_mode="none", countdown_s=0.0)
        w2.set_continuous(False)
        assert w2.continuous is False and [c for c, _ in list(w2._cmds.queue)] == ["continuous"]
        w2.set_templates(templates(["alpha", "beta", "gamma"]))
        got2, failed2 = [], []
        w2.events.connect(lambda evs: got2.extend(evs), Qt.DirectConnection)
        w2.failed.connect(failed2.append, Qt.DirectConnection)
        w2.start()
        w2.wait(60000)
        assert not failed2, failed2
        assert len([e for e in got2 if e.kind == "sign"]) < 3         # old behaviour: the sequence is not split
    finally:
        p.unlink(missing_ok=True)


# ---------------------------------------------------------------- 9. UI: one plain-language setting
def test_settings_toggle_for_continuous_signing_is_persisted_and_reaches_the_worker():
    from PySide6.QtWidgets import QApplication, QLabel
    app = QApplication.instance() or QApplication([])
    from test_app_offscreen import FakeWorker, scratch_store
    from lingestra.app.main_window import MainWindow, Config
    from lingestra.security.store import SecureStore

    class W(FakeWorker):
        def set_continuous(self, on): self.calls.append(("continuous", bool(on)))

    store = scratch_store("test_store_continuous_ui")
    store.save_settings({"seen_help": True})
    win = MainWindow(Config(source="landmarks:x.npy", model=str(ROOT / "models" / "demo_family.keras")),
                     worker_factory=W, store=store)
    win.show(); win.start_worker(); app.processEvents()
    try:
        texts = [l.text() for l in win.findChildren(QLabel)]
        assert "I can sign several signs in a row without lowering my hands" in texts
        assert win.continuous_sw.isChecked()                              # default ON
        assert ("continuous", True) in win.worker.calls                   # pushed at start
        win.continuous_sw.click()
        assert not win.continuous_sw.isChecked() and ("continuous", False) in win.worker.calls
        assert store.load_settings().get("continuous") is False           # persisted like the other settings
        win.continuous_sw.click()
        assert ("continuous", True) in win.worker.calls[-2:] and store.load_settings()["continuous"] is True
        win.continuous_sw.click()                                         # off again, then reopen the window
    finally:
        win.close()
    store2 = SecureStore(ROOT / "scratch" / "test_store_continuous_ui", passphrase="test-pass")
    win2 = MainWindow(Config(source="landmarks:x.npy", model=str(ROOT / "models" / "demo_family.keras")),
                      worker_factory=W, store=store2)
    win2.show(); win2.start_worker(); app.processEvents()
    try:
        assert not win2.continuous_sw.isChecked()
        assert ("continuous", False) in win2.worker.calls                 # the saved choice is applied at start
    finally:
        win2.close()


# ---------------------------------------------------------------- 6. review fixes
def test_repeated_set_continuous_mid_episode_does_not_re_emit_words():
    names = ["alpha", "beta", "gamma", "alpha", "beta"]
    rec = Recognizer(None, templates=templates(["alpha", "beta", "gamma"]))
    ch, _ = named(names)
    st = stream(ch, tail_s=2.0)[0]
    rec.reset()
    ev, toggled = [], False
    for i, f in enumerate(st):
        ev += rec.push_frame(f, i / FPS)
        if not toggled and len(signs_of(ev)) >= 2:
            toggled = True
            rec.set_continuous(True)                                   # a no-op request
    assert toggled
    assert signs_of(ev) == names


def test_switching_off_mid_episode_does_not_re_emit_committed_words():
    rec = Recognizer(None, templates=templates(["alpha", "beta", "gamma"]))
    ch, _ = named(["alpha", "beta", "gamma", "alpha"])
    rec.reset()
    ev, off = [], False
    for i, f in enumerate(stream(ch, tail_s=2.0)[0]):
        ev += rec.push_frame(f, i / FPS)
        if not off and len(signs_of(ev)) >= 2:
            off = True
            rec.set_continuous(False)
    assert signs_of(ev)[:2] == ["alpha", "beta"]
    assert signs_of(ev).count("alpha") <= 2 and signs_of(ev).count("beta") == 1


def test_a_word_that_never_becomes_sure_does_not_swallow_the_words_after_it():
    """Words are committed in order; one unsure word must not make the cap drop the confident ones behind it."""
    names = ["alpha", "beta", "gamma", "delta"] * 4
    rec = Recognizer(None, templates=templates(["alpha", "beta", "gamma", "delta"]))
    ch, _ = named(names, noise=0.004, seed=5)
    ev, at = run(rec, stream(ch, tail_s=2.0)[0])
    got = signs_of(ev)
    assert len(got) >= len(names) - 2, got                             # none lost silently ...
    kinds = [e.kind for e in ev]
    assert kinds.count("sign") + kinds.count("unknown") + kinds.count("clarify") >= len(names) - 1
    first_end = at[0]
    last_hands_up = len(ch) + LEAD
    assert first_end < last_hands_up - 5 * FPS                         # ... and committed progressively


def test_cap_emits_what_it_drops(monkeypatch):
    monkeypatch.setattr(Recognizer, "MAX_TAIL_S", 3.0)
    monkeypatch.setattr(Recognizer, "STALE_S", 100.0)
    names = ["alpha", "beta", "gamma", "alpha", "beta", "gamma"]
    rec = Recognizer(None, templates=templates(["alpha", "beta", "gamma"]))
    ch, _ = named(names)
    ev, _ = run(rec, stream(ch, tail_s=2.0)[0])
    assert signs_of(ev) == names


def test_caption_keeps_a_question_in_place_while_signing_goes_on():
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication([])
    from test_app_offscreen import FakeWorker, scratch_store
    from lingestra.app.main_window import MainWindow, Config
    from lingestra.live.recognizer import Event
    store = scratch_store("test_store_continuous_hole")
    store.save_settings({"seen_help": True})
    win = MainWindow(Config(source="landmarks:x.npy", model=str(ROOT / "models" / "demo_family.keras")),
                     worker_factory=FakeWorker, store=store)
    win.show(); win.start_worker(); app.processEvents()
    try:
        ev = lambda k, lab=None, c=(): Event(k, lab, 0.9, 0.0, list(c), 0.0, source="personal")
        win.on_events([ev("sign", "alpha"), ev("clarify", "beta", [("beta", .5), ("gamma", .4)]), ev("sign", "alpha")])
        assert win.words == ["alpha", "...", "alpha"] and win.clarify.isVisibleTo(win)
        win._on_confirm("beta")
        win.on_events([ev("sign", "beta")])
        assert win.words == ["alpha", "beta", "alpha"] and not win.clarify.isVisibleTo(win)
        win.on_events([ev("clarify", "beta", [("beta", .5), ("gamma", .4)])])
        win._on_repeat()
        assert win.words == ["alpha", "beta", "alpha"]
    finally:
        win.close()


# ------------------------------------------------------------------ words appear while the hands are still up
def test_words_appear_before_the_hands_are_lowered_even_for_a_second_performance():
    """Regression (user report): while signing several words in a row the earlier words must show up as soon
    as they are recognised, not when the hands are lowered. With the old strict rule only ~half of them did
    for realistic (second-take) matches; the next-sign rule must keep this well above that."""
    import sys
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root / "scripts"))
    import measure_commit_timing as M
    r = M.run(1, 6, [101], ns_list=(2,))
    assert r["n_nonlast"] >= 4
    assert r["early_nonlast"] >= 0.7, r


# ------------------------------------------------------------------ holding the last pose ends the word
def _held_last_pose(name, noise, hold=5.0, secs=None, seed=3):
    """A sign matched imperfectly (like a real repeat), then its last pose held still for `hold` s, then the hands lowered."""
    rec = Recognizer(None, templates=templates(list(SIGNS)))
    if secs is not None:
        rec.hold_commit_s = secs
    ch, _ = chain([(SIGNS[name], {})], noise=noise, seed=seed)
    st = stream(np.concatenate([ch, np.repeat(ch[-1:], int(hold * FPS), 0)]), tail_s=3.0)[0]
    ev, _ = run(rec, st)
    return ev, 1.3 + len(ch) / FPS                              # synth.stream: 0.8 s lead + 0.5 s raise before the chain


def test_a_held_last_pose_writes_the_word_before_the_hands_are_lowered():
    """User report: holding the last pose of the last word still, the word took ages. A held last word needed a
    near-perfect match (distance / tolerance <= 0.25) and otherwise waited for the 4 s backstop. A hold that lasts
    HOLD_COMMIT_S now ends the word. (noise 0.06 = a match at about a third of the tolerance, like a real repeat)"""
    for name in SIGNS:
        ev, hold_start = _held_last_pose(name, noise=0.06)
        sg = [e for e in ev if e.kind == "sign"]
        assert [e.label for e in sg] == [name], (name, [(e.kind, e.label) for e in ev])
        assert sg[0].t - hold_start < Recognizer.HOLD_COMMIT_S + 1.0, (name, sg[0].t - hold_start)   # not the 4 s backstop
        assert sg[0].t < hold_start + 5.0 - 1.0                  # before the hands start to lower


def test_holding_the_last_pose_does_not_add_a_repeated_or_look_alike_word_when_the_hands_lower():
    """The frozen pose after a committed word used to be decoded again when the hands were lowered ('alpha alpha',
    or 'gamma beta'). The sentence must be exactly the word signed."""
    for name in SIGNS:
        for noise in (0.0, 0.04, 0.06):
            ev, _ = _held_last_pose(name, noise=noise, hold=5.0)
            assert signs_of(ev) == [name], (name, noise, [(e.kind, e.label) for e in ev])
            assert [e.label for e in ev if e.kind == "sentence"] == [name]


def test_the_hold_length_that_ends_a_word_can_be_changed():
    when = {}
    for secs in (0.8, 2.5):
        ev, hs = _held_last_pose("beta", noise=0.06, secs=secs)
        when[secs] = [e.t for e in ev if e.kind == "sign"][0] - hs
    assert when[0.8] + 0.8 < when[2.5], when                       # a shorter setting writes the word sooner


def test_a_sign_followed_by_a_held_pose_and_then_a_new_sign_still_gives_two_words():
    rec = Recognizer(None, templates=templates(list(SIGNS)))
    items = [(SIGNS["alpha"], {}), (SIGNS["beta"], {})]
    ch, spans = chain(items, trans_s=0.2)                            # (noisy joins sometimes add a stray look-alike word, old code too)
    hold = np.repeat(ch[spans[0][1]][None], int(2.0 * FPS), 0)       # 2 s of stillness between the two words
    ch = np.concatenate([ch[:spans[0][1] + 1], hold, ch[spans[0][1] + 1:]])
    ev, _ = run(rec, stream(ch, tail_s=2.0)[0])
    assert signs_of(ev) == ["alpha", "beta"], [(e.kind, e.label) for e in ev]
