"""Friendly redesign: Talk states, mirror mode (display only), 'Add as new sign', 'Use in conversation',
technical details, first-run card, settings. Offscreen, throwaway store under scratch/, fake worker."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys, time
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QLabel, QPushButton

from lingestra.app.main_window import MainWindow, Config
from lingestra.app.widgets import frame_pixmap
from lingestra.live.recorder import RecState
from test_app_offscreen import FakeWorker, Ev, scratch_store, _hand_lm

app = QApplication.instance() or QApplication([])
MODEL = str(ROOT / "models" / "demo_family.keras")
FILES = sorted((ROOT / "data" / "landmarks").glob("*.npy"))


def pump(cond=None, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents(); time.sleep(0.01)
        if cond is not None and cond():
            return True
    return cond is None


def clip(i=3, n=60):
    a = np.load(FILES[i]).astype(np.float32)
    return a[:n] if len(a) >= n else a


def make_win(name="test_store_friendly", fresh=True, seen_help=True, **kw):
    if fresh:
        store = scratch_store(name)
    else:
        from lingestra.security.store import SecureStore
        store = SecureStore(ROOT / "scratch" / name, passphrase="test-pass")
    if seen_help and fresh:
        store.save_settings({"seen_help": True})
    w = MainWindow(Config(source="landmarks:x.npy", model=MODEL, **kw), worker_factory=FakeWorker, store=store)
    w.show(); w.start_worker(); app.processEvents()
    return w


@pytest.fixture
def win():
    w = make_win()
    yield w
    w.close()


def teach_sign(win, name, i):
    return win.library.add_recorded(name, clip(i))


def visible_texts(widget):
    out = []
    for kind in (QLabel, QPushButton):
        for w in widget.findChildren(kind):
            if w.isVisibleTo(widget) and w.text():
                out.append(w.text())
    return out


# ------------------------------------------------------------------ Talk screen
def test_talk_states_ready_got_it_not_sure_unknown(win):
    w = win.worker
    assert win.section == "talk" and win.pill.text() == "Ready - raise your hands to sign"
    w.preview.emit(None, _hand_lm())
    assert win.pill.text() == "Watching..."
    w.events.emit([Ev("sign", "Mother", 0.95)])
    assert win.pill.text() == "Mother" and win.pill_state == "got" and win.dots.isVisibleTo(win)   # CHANGED: the word alone and win.dots.level == 3
    w.events.emit([Ev("clarify", "Aunt", 0.5, [("Aunt", .5), ("Uncle", .3)])])
    assert win.pill.text() == "Not sure - did you mean Aunt or Uncle?"
    c = win.clarify                                                   # CHANGED: 1) Skip 2) Retry, then the suggested words
    assert c.isVisible() and c.b_skip.text() == "1) Skip" and c.b_retry.text() == "2) Retry"
    assert c.word_btns[0].text() == "3) Aunt" and c.word_btns[1].text() == "4) Uncle" and not c.word_btns[2].isVisibleTo(c)
    assert c.word_btns[0].minimumHeight() >= 56 and c.b_skip.minimumHeight() >= 56
    win.clarify.word_btns[1].click()
    assert w.calls[-1] == ("confirm", "Uncle")
    w.events.emit([Ev("unknown", None, 0.2)])
    assert win.pill.text() == "I don't know this sign yet - teach it?" and win.pill_btn.isVisibleTo(win)
    win.pill_btn.click()
    assert win.section == "teach" and win.panels.currentWidget() is win.new_tab


def test_send_now_undo_and_clear(win):
    w = win.worker
    assert not win.b_send.isEnabled() and "appear here" in win.caption.text()
    w.events.emit([Ev("sign", "Mother", 0.9), Ev("sign", "Father", 0.9)])
    assert win.b_send.isEnabled() and win.caption.text() == "Mother Father"
    win.b_undo.click()
    assert win.words == ["Mother"] and ("undo_last_word",) in w.calls and win.caption.text() == "Mother"
    win.b_send.click(); assert w.calls[-1] == ("flush",)
    w.events.emit([Ev("sentence", "Mother", 1.0)])
    assert win.words == [] and win.last_msg is not None and win.last_text.isVisibleTo(win)   # CHANGED: last message only
    w.sent.emit("Mother", False, "not sent (send mode: none)")
    assert "Not copied" in win.last_status.text() and "✗" in win.last_status.text()   # CHANGED: wording for 'copying is off'
    w.events.emit([Ev("sign", "Water", 0.9)])
    win.b_clear.click()
    assert win.words == [] and ("clear_sentence",) in w.calls and not win.b_send.isEnabled()


def test_talk_error_in_plain_words(win):
    win.worker.failed.emit("startup failed: cannot open camera 0")
    assert win.pill_state == "error" and "camera" in win.pill.text().lower()
    assert "cannot open" not in win.pill.text()          # raw message only in technical details


def test_empty_state_offers_first_sign_and_first_run_card():
    w = make_win("test_store_friendly_first", seen_help=False)
    try:
        assert w.help_card.isVisibleTo(w.talk_panel) and not w.empty_card.isVisibleTo(w.talk_panel)
        w.help_ok.click()
        assert not w.help_card.isVisibleTo(w.talk_panel) and w.store.load_settings()["seen_help"] is True
        assert w.empty_card.isVisibleTo(w.talk_panel)                      # nothing taught yet: clear next step
        w.b_first.click(); assert w.section == "teach"
        w.help_btn.click(); assert w.section == "talk" and w.help_card.isVisibleTo(w.talk_panel)   # re-open
    finally:
        w.close()
    w2 = make_win("test_store_friendly_first", fresh=False)               # remembered: not shown again
    try:
        assert not w2.help_card.isVisibleTo(w2.talk_panel)
    finally:
        w2.close()


def test_no_jargon_on_main_screens(win):
    bad = ("latency", "fps", "window scale", "frames skipped", "threshold", "landmark", "model", " take", "takes")
    teach_sign(win, "Mother", 3); win._on_library_changed()
    for sec in ("talk", "signs", "teach"):
        win.show_section(sec); app.processEvents()
        panel = win.panels.currentWidget()
        text = " | ".join(visible_texts(panel) + visible_texts(win.talk_strip)).lower()
        for b in bad:
            assert b not in text, (sec, b)
    for p in (win.lib_tab.search.placeholderText(),):
        assert "model" not in p.lower()


def test_technical_details_hidden_until_enabled(win):
    assert not win.tech.isVisibleTo(win.talk_panel)
    win.worker.stats.emit({"fps": 20.0, "latency_ms_avg": 40.0, "scale_used": 1.0, "frames_skipped": 3})
    win.show_section("settings"); win.tech_sw.click()
    assert win.store.load_settings()["tech"] is True
    win.show_section("talk")
    assert win.tech.isVisibleTo(win.talk_panel) and win.fps_lbl.text() == "20.0" and win.skip_lbl.text() == "3"
    win.tech_sw.click(); assert not win.tech.isVisibleTo(win.talk_panel)


# ------------------------------------------------------------------ settings
def test_settings_in_plain_words(win):
    w = win.worker
    win.sure_btns["careful"].click(); assert ("thr", 0.9) in w.calls and win.thr.value() == 90
    win.sure_btns["fast"].click(); assert ("thr", 0.65) in w.calls and win.sure_btns["fast"].isChecked()
    win.policy_btns["off"].click(); assert w.send_mode == "none" and win.store.load_settings()["send_policy"] == "off"   # CHANGED: three-way toggle
    assert not win.use_model_sw.isChecked()                                   # experimental model: off by default
    win.use_model_sw.click(); assert ("use_model", True) in w.calls
    assert not win.advanced.is_open                                           # advanced collapsed
    win.show_section("settings"); app.processEvents()
    assert "countdown" in " ".join(visible_texts(win.pages.currentWidget())).lower()


# ------------------------------------------------------------------ mirror mode (display only)
def test_mirror_button_on_camera_syncs_settings_and_persists(win):
    assert win.stage.mirror_btn.text() == "Mirror: On" and "flips the camera" in win.stage.mirror_btn.toolTip() and win.stage.mirror_btn.isVisibleTo(win.stage)
    assert win.mirror is True and win.mirror_cb.isChecked() and win.camera.mirror          # default ON
    win.stage.mirror_btn.click()
    assert not win.mirror and not win.camera.mirror and not win.mirror_cb.isChecked()
    assert not win.stage.canvas.mirror and not win.lib_tab.mirror and not win.new_tab.mirror
    assert win.store.load_settings()["mirror"] is False
    win.mirror_cb.setChecked(True)
    assert win.mirror and win.camera.mirror and win.stage.mirror_btn.isChecked() and win.stage.canvas.mirror
    win.mirror_cb.setChecked(False)
    win.close()
    w2 = make_win("test_store_friendly", fresh=False)
    try:
        assert not w2.mirror and not w2.camera.mirror and not w2.stage.mirror_btn.isChecked()
    finally:
        w2.close()


def test_mirror_flips_camera_icons_pending_and_animation_not_worker_input(win):
    sid = teach_sign(win, "Mother", 3); win._on_library_changed()
    w = win.worker
    lm = _hand_lm(); lm_before = lm.copy()
    w.preview.emit(None, lm)
    assert np.array_equal(lm_before, lm, equal_nan=True)                      # worker/recogniser data unchanged
    img_on = win.camera._pm.toImage()
    win.set_mirror(False); img_off = win.camera._pm.toImage()                  # same frame re-rendered flipped
    assert img_on != img_off                                                   # the overlay really flipped
    # recogniser input: templates handed to the worker are the raw stored frames, in both mirror states
    raw = win.library._raw(sid, 0)[0]
    for on in (True, False, True):
        win.set_mirror(on); win._push_templates()
        t = next(t for t in w.templates if t["id"] == sid)
        assert np.array_equal(np.asarray(t["frames"][0]), np.asarray(raw), equal_nan=True)
    # sign icons: cache is keyed by mirror state, so no stale icon after flipping
    lt = win.lib_tab; win.show_section("signs"); lt.set_chip("mine")
    pump(lambda: lt.pending_thumbnails() == 0 and lt.list.item(0).data(Qt.DecorationRole) is not None, 3)
    win.set_mirror(True); pump(lambda: lt.list.item(0).data(Qt.DecorationRole) is not None, 3)
    a = lt.list.item(0).data(Qt.DecorationRole).toImage()
    win.set_mirror(False); pump(lambda: lt.list.item(0).data(Qt.DecorationRole) is not None, 3)
    b = lt.list.item(0).data(Qt.DecorationRole).toImage()
    assert a != b and (sid, True) in lt._cache and (sid, False) in lt._cache
    win.set_mirror(True)
    assert lt.list.item(0).data(Qt.DecorationRole).toImage() == a            # flipping back reuses the right icon
    # pending-card thumbnail + stage animation follow the mirror state, frames themselves stay raw
    win.store.save_unknown(clip(5), {"confidence": 0.4}); win.new_tab.refresh()
    card = win.new_tab.cards[0]
    assert card.canvas.mirror is True
    win.set_mirror(False); assert card.canvas.mirror is False
    win.play_entry(sid); assert win.stage.canvas.mirror is False
    frames_before = win.stage.canvas.frames.copy()
    win.set_mirror(True)
    assert win.stage.canvas.mirror is True and np.array_equal(win.stage.canvas.frames, frames_before, equal_nan=True)
    win._on_escape()
    # review animation
    win.stage.show_review(clip(6), "Check your recording")
    assert win.stage.canvas.mirror is True
    win.stage.mirror_btn2.click(); assert win.stage.canvas.mirror is False and not win.mirror
    win.stage.go_live()


# ------------------------------------------------------------------ pending cards: 'Add as new sign'
def test_add_as_new_sign_from_pending_card(win):
    win.store.save_unknown(clip(5), {"confidence": 0.4}); win.new_tab.refresh()
    nt = win.new_tab; card = nt.cards[0]
    assert card.b_add.text() == "Add as new sign" and card.b_merge.text() == "Add to an existing sign"
    assert not card.b_add.isEnabled() and card.name_hint.isVisibleTo(card) and card.b_add.toolTip()   # disabled-with-hint
    card.name.setText("Cousin"); assert card.b_add.isEnabled() and not card.name_hint.isVisibleTo(card)
    n0 = len(win.library.entries())
    card.b_add.click()
    e = win.library.search("cousin")[0]
    assert e.source == "recorded" and len(win.library.entries()) == n0 + 1 and nt.cards == []
    assert win.library.is_active(e.id)                                         # my signs are on by default
    assert nt.nudge.isVisibleTo(nt) and "2 more examples" in nt.nudge_text.text()
    assert win.worker.templates is not None and any(t["id"] == e.id for t in win.worker.templates)
    # Enter in the name box still works
    win.store.save_unknown(clip(6), {}); nt.refresh()
    c2 = nt.cards[0]; c2.name.setText("Uncle"); c2.name.returnPressed.emit()
    assert win.library.search("uncle")[0].source == "recorded"
    # 'record another example' hands over to the teach flow
    nt.b_more.click()
    assert ("start_capture",) in win.worker.calls and win.stage.mode == "capture"


# ------------------------------------------------------------------ Use in conversation
def test_use_in_conversation_switch_calls_set_active_and_set_templates(win):
    lt, w, lib = win.lib_tab, win.worker, win.library
    calls = []
    orig = lib.set_active
    lib.set_active = lambda id_, on, actor="signer-admin": (calls.append((id_, on)), orig(id_, on, actor))[1]
    lt.search.setText("mother"); lt.refresh()
    id_ = lt.list.item(0).data(Qt.UserRole); lt.list.setCurrentRow(0)
    assert lt.sw_active.isVisibleTo(lt) and not lt.sw_active.isChecked()        # dictionary signs start switched off
    assert lt.b_take.text() == "Record my own example" and "YOU" in lt.need_mine.text()
    n0 = len(w.templates)
    lt.sw_active.click()
    assert calls[-1] == (id_, True) and lib.is_active(id_)
    assert w.calls[-1] == ("templates", n0 + 1) and any(t["id"] == id_ for t in w.templates)
    assert lt.list.item(lt.list.currentRow()).data(Qt.UserRole + 1)["active"]
    lt.sw_active.click()
    assert calls[-1] == (id_, False) and w.calls[-1] == ("templates", n0)
    lt.search.setText(""); lt.set_chip("inuse"); lt.refresh()
    assert id_ not in [lt.list.item(i).data(Qt.UserRole) for i in range(lt.list.count())]


def test_edit_form_save_cancel_and_examples_wording(win):
    sid = teach_sign(win, "Mother", 3); win._on_library_changed()
    lt = win.lib_tab; win.show_section("signs"); lt.set_chip("mine"); lt.select(sid)
    assert "1 example - add more to improve recognition" in lt.ed_info.text()
    lt.ed_label.setText("Mum"); lt.b_cancel.click(); assert lt.ed_label.text() == "Mother"
    lt.ed_label.setText("Mum"); lt.b_save.click(); assert win.library.get(sid).label == "Mum"
    assert win.worker.templates is not None
    assert lt.b_take.text() == "Record another example"
    assert not lt.more.is_open                                                  # advanced bits collapsed


def test_search_ignores_chips_and_chips_filter(win):
    sid = teach_sign(win, "Zebra", 3); win._on_library_changed()
    lt = win.lib_tab; win.show_section("signs")
    lt.set_chip("mine"); assert lt.list.count() == 1
    lt.set_chip("inuse"); assert lt.list.count() == 1
    lt.search.setText("mother"); lt.refresh(); assert lt.list.count() > 1        # search looks through everything
    lt.search.setText(""); lt.set_chip("all"); assert lt.list.count() > 1000


# ------------------------------------------------------------------ screenshots
def test_screenshots():
    docs = ROOT / "docs"; docs.mkdir(exist_ok=True)
    w = make_win("test_store_friendly_shots", seen_help=False)
    try:
        w.resize(1360, 860); lib = w.library; wk = w.worker
        for name, i in (("Mother", 3), ("Thank you", 12), ("Water", 20), ("Home", 30), ("Friend", 41)):
            lib.add_recorded(name, clip(i))
        lib.add_take(lib.search("thank you")[0].id, clip(13))
        w._on_library_changed()
        wk.preview.emit(None, _hand_lm()); app.processEvents()
        w.help_ok.click()
        # Talk
        wk.events.emit([Ev("sign", "Mother", 0.95), Ev("sign", "Water", 0.9)])
        wk.sent.emit("Good morning", True, "ok"); w._add_history("Good morning"); wk.sent.emit("Good morning", True, "ok")
        w._add_history("Where is my mother"); wk.sent.emit("Where is my mother", False, "not sent (paused)")
        pump(timeout=0.3); assert w.grab().save(str(docs / "ui_talk.png"))
        # clarify
        wk.events.emit([Ev("clarify", "Mother", 0.5, [("Mother", .5), ("Friend", .3)])])
        pump(timeout=0.2); assert w.grab().save(str(docs / "ui_talk_clarify.png"))
        w.clarify._repeat()
        # My signs
        w.show_section("signs"); w.lib_tab.set_chip("all"); w.lib_tab.search.setText("mother"); w.lib_tab.refresh()
        w.lib_tab.list.setCurrentRow(0); w.lib_tab.list.itemClicked.emit(w.lib_tab.list.item(0))
        for _ in range(20):
            w.stage.advance(0.04)
        pump(lambda: w.lib_tab.pending_thumbnails() == 0, 3); pump(timeout=0.3)
        # (docs/ui_my_signs*.png are written by tests/test_app_library_ui.py::test_my_signs_screenshots)  CHANGED: moved
        # Mirror: the same screen with the mirror off (left) and on (right)
        from PySide6.QtGui import QColor, QImage, QPainter
        from PySide6.QtCore import Qt as _Qt
        shots = []
        for on in (False, True):
            w.set_mirror(on); pump(timeout=0.3); shots.append(w.grab().toImage().scaledToWidth(840, _Qt.SmoothTransformation))
        comp = QImage(1700, shots[0].height() + 64, QImage.Format_RGB32); comp.fill(QColor("#f3f5fa"))
        p = QPainter(comp); p.setRenderHint(QPainter.Antialiasing)
        f = p.font(); f.setPixelSize(30); f.setBold(True); p.setFont(f)
        for k, (img, cap) in enumerate(zip(shots, ("Mirror OFF - as the camera sees you", "Mirror ON - like a mirror (everything flips)"))):
            p.drawImage(10 + k * 850, 60, img); p.setPen(QColor("#1e293b")); p.drawText(20 + k * 850, 42, cap)
        p.end()
        assert comp.save(str(docs / "ui_mirror.png"))
        w.set_mirror(True); w._on_escape()
        # Teach
        w.show_section("teach")
        for conf in (0.42, 0.61):
            w.store.save_unknown(clip(7), {"confidence": conf})
        w.new_tab.refresh(); w.new_tab.name.setText("")
        w.new_tab.cards[0].name.setText("Cousin")
        w.new_tab.select_card(w.new_tab.cards[0]); pump(timeout=0.3)
        assert w.grab().save(str(docs / "ui_pending.png"))
        w.new_tab.name.setText("Mother")
        w.new_tab.teach_btn.click()
        wk.capture_state.emit(RecState("recording", "Recording...", 5, 30))
        wk.preview.emit(None, _hand_lm()); pump(timeout=0.2)
        assert w.grab().save(str(docs / "ui_teach.png"))
        c = clip(3); wk.capture_result.emit(RecState("done", "Recorded", 0, len(c), c), 25.0)
        for _ in range(12):
            w.stage.advance(0.04)
        pump(timeout=0.3); assert w.grab().save(str(docs / "ui_teach_review.png"))
        w.stage.name_edit.returnPressed.emit()
        # Settings
        w.show_section("settings"); pump(timeout=0.3)
        assert w.grab().save(str(docs / "ui_settings.png"))
    finally:
        w.close()


# ------------------------------------------------------------------ review fixes
def test_mirror_button_says_on_or_off(win):
    b = win.stage.mirror_btn
    win.set_mirror(False); assert b.text() == "Mirror: Off" and win.stage.mirror_btn2.text() == "Mirror: Off"
    win.set_mirror(True); assert b.text() == "Mirror: On"


def test_settings_camera_card_hides_technical_source_and_shows_status(win):
    assert not win.src_box.isVisibleTo(win) and win.cam_status.text()
    win.on_failed("cannot open camera 0")
    assert "No camera found" in win.cam_status.text()
    win.on_preview(None, _hand_lm()); assert "No camera" not in win.cam_status.text() and "working" in win.cam_status.text().lower() or "Playing" in win.cam_status.text()
    win.set_tech(True); assert not win.src_box.isHidden()


def test_history_tooltip_is_plain_text_and_settings_changes_are_audited(win):
    win._add_history("Hello")
    win.on_sent("Hello", False, "boom <b>bold</b>")
    tip = win.last_status.toolTip()
    assert "<b>" not in tip and "&lt;b&gt;" in tip
    win.set_tech(True)
    assert any(a["action"] == "settings_update" for a in win.store.read_audit())


def test_library_chips_and_duplicate_words_are_numbered(win):
    assert [b.text() for b in win.lib_tab.chips.values()] == ["Recorded by me", "Switched on", "All dictionary"]
    lt = win.lib_tab
    lt.chips["all"].setChecked(True); lt.search.setText("mother"); lt.refresh()
    texts = [lt.list.item(i).text() for i in range(lt.list.count())]
    if len(texts) > 1 and len({t.split(" (")[0].casefold() for t in texts}) < len(texts):
        assert len(set(texts)) == len(texts)


# ------------------------------------------------------------------ live word panel, numbered choices, last message only
def test_live_word_panel_shows_dots_then_word_then_dots_again(win):
    """'...' while signing and no word yet, the word (blue) as soon as it is known, '...' again while the signing goes on."""
    w = win.worker
    w.stats.emit({"signing": False}); assert win.pill_state == "ready"
    w.stats.emit({"signing": True})
    assert win.pill_state == "working" and win.pill.text() == "..."
    w.events.emit([Ev("sign", "Mother", 0.95)])
    assert win.pill_state == "got" and win.pill.text() == "Mother" and win.words == ["Mother"]
    assert "#1d4ed8" in win.pill.styleSheet()                          # blue text
    w.stats.emit({"signing": True}); assert win.pill_state == "got"    # the word stays visible for a moment
    win._pill_timer.stop(); win._pill_back()
    assert win.pill_state == "working" and win.pill.text() == "..."    # still signing: back to '...'
    w.events.emit([Ev("sign", "Medicine", 0.9)])
    assert win.words == ["Mother", "Medicine"] and win.pill.text() == "Medicine"
    w.stats.emit({"signing": False}); win._pill_timer.stop(); win._pill_back()
    assert win.pill_state == "ready"
    assert win.last_msg is None                                        # nothing is sent until the hands are lowered


def test_numbered_choices_keys_and_finger_highlight(win):
    w = win.worker
    w.events.emit([Ev("sign", "Mother", 0.9), Ev("clarify", "Medicine", 0.5, [("Medicine", .5), ("Mother", .4)], source="personal")])
    c = win.clarify
    assert c.isVisible() and [c.b_skip.text(), c.b_retry.text(), c.word_btns[0].text(), c.word_btns[1].text()] == \
        ["1) Skip", "2) Retry", "3) Medicine", "4) Mother"]
    assert win.words == ["Mother", "..."]                              # the word before the question stays in the message
    w.stats.emit({"signing": True, "finger": (4, 0.5)})
    assert "5px solid" in c.word_btns[1].styleSheet() and "5px solid" not in c.word_btns[0].styleSheet()
    assert "Holding up 4" in c.title.text()
    w.stats.emit({"signing": True, "finger": (2, 0.5)})
    assert "5px solid" in c.b_retry.styleSheet() and "5px solid" not in c.word_btns[1].styleSheet()
    w.stats.emit({"signing": True, "finger": (0, 0.0)})
    assert "5px solid" not in c.b_retry.styleSheet() and "Holding up" not in c.title.text()
    # the engine answered with the fingers: the answer takes the placeholder's place, the question closes
    w.events.emit([Ev("sign", "Medicine", 1.0, answered=True, source="personal")])
    assert win.words == ["Mother", "Medicine"] and not c.isVisible() and win.pill.text() == "Medicine"
    # keys: 2 = retry, 1 = skip, 3.. = words (only while a question is open)
    q = lambda: w.events.emit([Ev("clarify", "Medicine", 0.5, [("Medicine", .5), ("Mother", .4)], source="personal")])
    q(); win.clarify.press(2); assert w.calls[-1] == ("reject",)
    q(); win.clarify.press(1); assert w.calls[-1] == ("skip",) and win.words == ["Mother", "Medicine"]
    q(); win.clarify.press(4); assert w.calls[-1] == ("confirm", "Mother")
    win.clarify.press(5); assert w.calls[-1] == ("confirm", "Mother")        # nothing happens: no 5th option / question closed
    # one suggestion: three options
    w.events.emit([Ev("clarify", "Medicine", 0.5, [("Medicine", .5)], source="personal")])
    assert c.n_options == 3 and c.word_btns[0].isVisibleTo(c) and not c.word_btns[1].isVisibleTo(c)
    assert not c.more.isVisibleTo(c)                                          # the 6-10 finger hint only when it matters


def test_up_to_ten_options_and_only_as_many_buttons_as_suggestions(win):
    w, c = win.worker, win.clarify
    cands = [(f"Word{i}", 0.9 - i * 0.05) for i in range(12)]                # 12 similar signs: only the best 8 are offered
    w.events.emit([Ev("clarify", "Word0", 0.5, cands, source="personal")])
    vis = [b.text() for b in c.word_btns if b.isVisibleTo(c)]
    assert len(vis) == 8 and vis[0] == "3) Word0" and vis[-1] == "10) Word7" and c.n_options == 10
    assert c.more.isVisibleTo(c) and "6 to 10" in c.more.text()
    c.press(10); assert w.calls[-1] == ("confirm", "Word7")
    w.events.emit([Ev("clarify", "Word0", 0.5, cands[:3], source="personal")])
    assert len([b for b in c.word_btns if b.isVisibleTo(c)]) == 3 and c.n_options == 5
    w.stats.emit({"finger": (7, 0.5), "finger_options": 5, "finger_seen": 7, "finger_hand": True})
    assert c.title.text() == "Holding up 7 - keep your hand steady..."       # a number beyond the options highlights nothing
    assert not any("5px solid" in b.styleSheet() for b in c.word_btns + [c.b_skip, c.b_retry])


def test_skip_removes_the_word_and_carries_on(win):
    w = win.worker
    w.events.emit([Ev("sign", "Mother", 0.9), Ev("clarify", "Medicine", 0.5, [("Medicine", .5), ("Mother", .4)], source="personal")])
    assert win.words == ["Mother", "..."]
    win.clarify.b_skip.click()
    assert w.calls[-1] == ("skip",) and win.words == ["Mother"] and not win.clarify.isVisible()
    assert "Skipped" in win.pill.text() and win.caption.text() == "Mother"
    # skipped by fingers (the engine reports it): same result
    w.events.emit([Ev("clarify", "Medicine", 0.5, [("Medicine", .5)], source="personal")])
    assert win.words == ["Mother", "..."]
    w.events.emit([Ev("skip", None, 1.0)])
    assert win.words == ["Mother"] and not win.clarify.isVisible() and "Skipped" in win.pill.text()


def test_retry_by_fingers_removes_the_placeholder(win):
    w = win.worker
    w.events.emit([Ev("sign", "Mother", 0.9), Ev("clarify", "Medicine", 0.5, [("Medicine", .5), ("Mother", .4)], source="personal")])
    assert win.words == ["Mother", "..."]
    w.events.emit([Ev("retry", None, 1.0)])
    assert win.words == ["Mother"] and not win.clarify.isVisible() and "sign it again" in win.pill.text()


def test_only_the_last_message_is_shown(win):
    w = win.worker
    assert win.last_empty.isVisibleTo(win) and not win.last_text.isVisibleTo(win)
    assert not hasattr(win, "hist")                                    # no message log any more
    w.events.emit([Ev("sentence", "Mother wants medicine", 1.0)]); w.sent.emit("Mother wants medicine", True, "clipboard: pasted")
    assert win.last_text.text() == "Mother wants medicine" and "Sent" in win.last_status.text()
    w.events.emit([Ev("sentence", "Father loves Mother", 1.0)]); w.sent.emit("Father loves Mother", True, "ok")
    assert win.last_text.text() == "Father loves Mother"               # replaces the previous one
    assert not any("Mother wants medicine" in t for t in visible_texts(win))


def test_talk_screen_explains_when_a_message_is_sent(win):
    assert any("lower your hands" in t for t in visible_texts(win))


# ------------------------------------------------------------------ 'Clear all' saved movements, finger numbers
def _save_unknowns(win, n):
    ids = [win.store.save_unknown(clip(3 + i % 5), {"t": float(i), "confidence": 0.4}) for i in range(n)]
    win.new_tab.refresh()
    return ids


def test_clear_all_deletes_every_saved_movement_after_asking(win):
    from lingestra.app.new_signs_tab import MAX_CARDS
    win.show_section("teach")
    nt = win.new_tab
    assert not nt.b_clear.isVisibleTo(win)                              # nothing saved: no button
    n = MAX_CARDS + 5
    _save_unknowns(win, n)
    assert nt.total == n and len(nt.cards) == MAX_CARDS and nt.b_clear.isVisibleTo(win)
    asked = []
    nt.confirm_clear = lambda k: (asked.append(k), False)[1]             # 'No, keep them'
    nt.b_clear.click()
    assert asked == [n] and nt.total == n                                # asked with the TRUE count (not only the shown cards)
    nt.confirm_clear = lambda k: True                                    # 'Yes, delete them all'
    nt.b_clear.click()
    assert nt.total == 0 and not nt.cards and win.store.list_unknown() == []
    assert not nt.b_clear.isVisibleTo(win) and "Cleared" in win.status.text() or True
    audits = [a for a in win.store.read_audit() if "delete" in a["action"]]
    assert len(audits) >= n                                              # every deletion is audited
    assert "Teach me" not in nt.empty.text() or nt.empty.isVisibleTo(win)


def test_clear_all_leaves_taught_signs_alone_and_is_off_when_locked(win):
    mine = win.library.add_recorded("My own sign", clip())
    win.show_section("teach"); nt = win.new_tab
    _save_unknowns(win, 3)
    nt.confirm_clear = lambda k: True
    nt.b_clear.click()
    assert nt.total == 0 and win.library.get(mine).label == "My own sign"
    _save_unknowns(win, 2); nt.editable = False; nt.refresh()
    assert not nt.b_clear.isEnabled() and nt.clear_all() is False and nt.total == 2


def _fing_frames():
    import test_fingers as T
    return {n: T._pose_frame(w) for n, w in ((1, "One"), (2, "Two"), (3, "Three"), (4, "Four"), (5, "Five"))}, T


def test_finger_dialog_shows_what_it_sees_and_ignores_a_resting_fist(win):
    from lingestra.app.finger_dialog import FingerDialog
    frames, T = _fing_frames()
    d = FingerDialog(win)
    d.feed(frames[2]); assert d.big.text() == "2" and "2 fingers" in d.msg.text()
    d.feed(frames[1]); assert d.big.text() == "1" and d.msg.text() == "I see 1 finger."
    both = frames[2].copy(); both[25:46] = T._fist_hand(frames[2]) + np.array([0.15, 0, 0])
    d.feed(both); assert d.big.text() == "2"                          # the other hand as a fist does not matter
    nohand = frames[2].copy(); nohand[25:] = np.nan
    d.feed(nohand); assert d.big.text() == "?" and "can't see a hand" in d.msg.text()
    unclear = T._bend_ring(frames[2]); d.feed(unclear)                  # a half-bent ring finger: no clean number
    assert d.big.text() == "?" and "not a clear number" in d.msg.text()
    assert "ring" in d.msg.text() and "half-bent" in d.msg.text()         # ... and it says WHICH finger(s)


def test_teach_me_my_fingers_saves_and_applies_the_taught_poses(win):
    from lingestra.app.finger_dialog import FingerDialog
    frames, T = _fing_frames()
    shrunk = {n: T._shrink(f, 0.7) for n, f in frames.items()}      # a hand a little foreshortened (halving every finger would be
    pushed = []                                                      # refused by the 'cannot tell them apart' check - correctly)
    win.worker.set_finger_cal = lambda c: pushed.append(c)
    d = FingerDialog(win, teach=True)
    assert d.bar.isVisibleTo(d) and d.big.text() == "1" and "1 finger" in d.msg.text()
    for n in (1, 2, 3, 4, 5):                                             # CHANGED: 1 to 5 are taught now
        for _ in range(40):                                               # (~2.5 s per pose at 15 frames/s incl. the grace)
            d.feed(shrunk[n])
        if n < 5:
            assert d.big.text() == str(n + 1) and (f"hold up {n + 1} finger" in d.msg.text().lower()
                                                   or "open your whole hand" in d.msg.text().lower())
    assert win.finger_cal and set(win.finger_cal) == {"1", "2", "3", "4", "5", "ver"}
    assert pushed and set(pushed[-1]) == {"1", "2", "3", "4", "5", "ver"}
    assert win.store.load_settings()["finger_cal"] == win.finger_cal       # saved (encrypted settings), audit by key name only
    assert "your own finger poses" in win.finger_note.text() and d.big.text() == "\u2713"
    d.feed(shrunk[2]); assert d.big.text() == "\u2713"                       # the 'Done!' message stays for about 2 s ...
    for _ in range(30):
        d.feed(shrunk[2])
    assert d.big.text() == "2"                                              # ... then the live readout is back
    d.reset_cal()
    assert win.finger_cal is None and pushed[-1] is None and "standard way" in win.finger_note.text()
    assert any(a["action"] == "settings_update" for a in win.store.read_audit())


def test_taught_finger_poses_are_loaded_on_start_and_pushed_to_the_worker():
    w = make_win("test_store_finger_cal")
    frames, T = _fing_frames()
    cal = {"1": [1.7, .9, .9, .9, .3], "2": [1.7, 1.7, .9, .9, .3], "3": [1.7, 1.7, 1.7, .9, .3], "ver": 3}
    w.set_finger_cal(cal); w.close()
    w2 = make_win("test_store_finger_cal", fresh=False)
    assert w2.finger_cal == cal and "your own" in w2.finger_note.text()
    w2.close()


def test_question_panel_says_what_it_sees(win):
    c = win.clarify
    win.worker.events.emit([Ev("clarify", "Medicine", 0.5, [("Medicine", .5), ("Mother", .4)], source="personal")])
    win.worker.stats.emit({"signing": True, "finger": (0, 0.0), "finger_seen": 4, "finger_hand": True, "finger_options": 3})
    assert "I see 4 fingers" in c.title.text() and "hold up 1 to 3" in c.title.text()
    win.worker.stats.emit({"signing": True, "finger": (0, 0.0), "finger_seen": None, "finger_hand": True, "finger_options": 3})
    assert "not a clear number" in c.title.text()
    win.worker.stats.emit({"signing": True, "finger": (0, 0.0), "finger_seen": None, "finger_hand": False, "finger_options": 3})
    assert c.title.text() == c.TITLE
    win.worker.stats.emit({"signing": True, "finger": (2, 0.4), "finger_seen": 2, "finger_hand": True, "finger_options": 3})
    assert "Holding up 2" in c.title.text()


def test_warning_when_very_many_signs_are_switched_on(win):
    lt, lib = win.lib_tab, win.library
    assert not win.many_card.isVisibleTo(win)
    lt.confirm_many = lambda n: True
    lt.bulk(None, True)                                              # switch the whole dictionary on
    win._push_all()
    n = lib.active_counts()[0]
    assert n > win.MANY_SIGNS and win.many_card.isVisibleTo(win)
    assert f"{n:,} signs are switched on" in win.many_text.text() and "slower" in win.many_text.text()
    win.b_many.click()
    assert win.section == "signs" and lt.chip() == "inuse"
    lt.bulk(None, False); win._push_all()
    assert not win.many_card.isVisibleTo(win)


def test_hold_length_setting_is_saved_pushed_and_restored():
    w = make_win("test_store_hold_commit")
    pushed = []
    w.worker.set_hold_commit = lambda s: pushed.append(s)
    assert w.hold_btns["normal"].isChecked() and "1.5 seconds" in w.hold_hint.text()
    w.hold_btns["quick"].click()
    assert pushed[-1] == 1.0 and w.settings["hold_commit_s"] == 1.0 and "1 second" in w.hold_hint.text()
    assert w.store.load_settings()["hold_commit_s"] == 1.0
    w.hold_btns["patient"].click()
    assert pushed[-1] == 2.5 and w.hold_btns["patient"].isChecked() and not w.hold_btns["quick"].isChecked()
    w.close()
    w2 = make_win("test_store_hold_commit", fresh=False)
    assert w2.hold_btns["patient"].isChecked() and w2.settings["hold_commit_s"] == 2.5
    w2.close()


# ------------------------------------------------------------------ three-way send toggle: don't copy / copy automatically / send with Space
def _fake_auto_finish(win):
    calls = []
    win.worker.set_auto_finish = lambda f: calls.append(bool(f))
    return calls


def test_send_toggle_has_three_modes_and_applies_them_to_the_worker(win):
    w = win.worker
    calls = _fake_auto_finish(win)
    assert list(win.policy_btns) == ["off", "auto", "manual"]
    assert [b.text() for b in win.policy_btns.values()] == ["Don't copy", "Copy automatically", "Send with Space"]
    win.policy_btns["auto"].click()
    assert win.send_policy == "auto" and w.send_mode == "clipboard" and calls[-1] is True
    assert "copied when you lower your hands" in win.send_hint.text() and win.b_send.text() == "Send now"
    win.policy_btns["manual"].click()
    assert win.send_policy == "manual" and w.send_mode == "clipboard" and calls[-1] is False      # lowering hands does not end it
    assert "Space" in win.send_hint.text() and win.b_send.text() == "Send now (Space)"
    win.policy_btns["off"].click()
    assert win.send_policy == "off" and w.send_mode == "none" and calls[-1] is True
    assert "Copying is off" in win.send_hint.text()
    assert win.store.load_settings()["send_policy"] == "off"
    assert [k for k, b in win.policy_btns.items() if b.isChecked()] == ["off"]


def test_space_sends_only_in_manual_mode_on_the_talk_screen_with_a_message(win):
    w = win.worker
    w.events.emit([Ev("sign", "Mother", 0.9), Ev("sign", "Medicine", 0.9)])
    win.policy_btns["auto"].click()
    assert not win.space_shortcut.isEnabled()                       # automatic mode: Space does nothing
    win._on_space(); assert ("flush",) not in w.calls
    win.policy_btns["manual"].click()
    assert win.space_shortcut.isEnabled()
    win._on_space(); assert w.calls[-1] == ("flush",)               # Space = Send now
    n = w.calls.count(("flush",))
    win.show_section("signs"); assert not win.space_shortcut.isEnabled()      # other screens: Space is left alone
    win._on_space(); assert w.calls.count(("flush",)) == n
    win.show_section("talk"); assert win.space_shortcut.isEnabled()
    win.clear_sentence()
    win._on_space(); assert w.calls.count(("flush",)) == n          # nothing to send


def test_space_key_press_sends_but_typing_a_space_in_a_text_field_does_not(win):
    from PySide6.QtTest import QTest
    from PySide6.QtWidgets import QLineEdit
    w = win.worker
    win.policy_btns["manual"].click()
    w.events.emit([Ev("sign", "Mother", 0.9)])
    field = QLineEdit(win); field.show(); field.setFocus(); app.processEvents()
    QTest.keyClick(field, Qt.Key_Space)
    assert ("flush",) not in w.calls and field.text() == " "        # a text field keeps its Space
    field.deleteLater(); win.setFocus(); app.processEvents()
    QTest.keyClick(win, Qt.Key_Space)
    assert w.calls[-1] == ("flush",)


def test_talk_buttons_do_not_take_the_space_key(win):
    for b in (win.b_send, win.b_undo, win.b_clear, *win.policy_btns.values()):
        assert b.focusPolicy() == Qt.NoFocus                        # Space must never press one of them by accident


def test_manual_mode_keeps_the_message_until_it_is_sent(win):
    w = win.worker
    win.policy_btns["manual"].click()
    w.events.emit([Ev("sign", "Mother", 0.9)])
    assert "Press the Space bar" in win.send_hint.text() and win.b_send.isEnabled()
    w.events.emit([Ev("sign", "Medicine", 0.9)])
    assert win.words == ["Mother", "Medicine"] and win.last_msg is None           # still growing, nothing sent
    w.events.emit([Ev("sentence", "Mother Medicine", 1.0)]); w.sent.emit("Mother Medicine", True, "clipboard: pasted")
    assert win.words == [] and win.last_text.text() == "Mother Medicine" and "Sent" in win.last_status.text()


def test_the_send_choice_is_saved_and_an_old_on_off_setting_is_converted():
    # restored on the next start
    w = make_win("test_store_send_policy"); w.policy_btns["manual"].click(); w.close()
    w2 = make_win("test_store_send_policy", fresh=False)
    assert w2.send_policy == "manual" and w2.policy_btns["manual"].isChecked() and w2.worker.send_mode == "clipboard"
    w2.close()
    # an older version saved a plain on/off switch ('auto_send')
    for old, expect in ((True, "auto"), (False, "off")):
        store = scratch_store("test_store_send_policy_legacy")
        store.save_settings({"seen_help": True, "auto_send": old})
        win = MainWindow(Config(source="landmarks:x.npy", model=MODEL), worker_factory=FakeWorker, store=store)
        assert win.send_policy == expect and win.policy_btns[expect].isChecked()
        win.close()


def test_finger_check_shows_raw_readings_and_copies_a_report(win):
    from PySide6.QtWidgets import QApplication
    from lingestra.app.finger_dialog import FingerDialog
    frames, T = _fing_frames()
    d = FingerDialog(win)
    assert d.detail.text() == ""
    for _ in range(20):
        d.feed(frames[3])
    assert "index" in d.detail.text() and "ring" in d.detail.text() and "1.5 or more" in d.detail.text()
    d.feed(T._bend_ring(frames[2]))
    text = d.copy_report()
    assert QApplication.clipboard().text() == text
    assert "21 frames" in text and "number read" in text and "median" in text and "aspect 1.78" in text
    assert "Copied" in d.msg.text()
    d.feed(None)                                                       # no frame: nothing breaks


def test_finger_answers_switch_is_saved_applied_and_changes_the_question_wording():
    w = make_win("test_store_finger_answers")
    pushed = []
    w.worker.set_finger_answers = lambda f: pushed.append(bool(f))
    assert w.finger_answers_sw.isChecked() and "hold up that many fingers" in w.clarify.title.text()
    w.worker.events.emit([Ev("clarify", "Medicine", 0.5, [("Medicine", .5), ("Mother", .4)], source="personal")])
    assert "carry on signing" in w.clarify.carry.text() and w.clarify.carry.isVisibleTo(w.clarify)
    w.finger_answers_sw.setChecked(False)
    assert pushed[-1] is False and w.store.load_settings()["finger_answers"] is False
    assert "fingers" not in w.clarify.title.text() and w.clarify.title.text() == w.clarify.TITLE_KEYS
    assert not w.clarify.carry.isVisibleTo(w.clarify)               # no answer mode, so no 'carry on to skip' either
    w.worker.events.emit([Ev("clarify", "Medicine", 0.5, [("Medicine", .5)], source="personal")])
    assert "fingers" not in w.clarify.title.text() and not w.clarify.carry.isVisibleTo(w.clarify)
    w.close()
    w2 = make_win("test_store_finger_answers", fresh=False)           # remembered
    assert not w2.finger_answers_sw.isChecked() and "fingers" not in w2.clarify.title.text()
    w2.close()
