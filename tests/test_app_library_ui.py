"""Sign library / playback / teach-a-sign UI, offscreen, real SignLibrary + SecureStore in scratch/."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import shutil, sys, time
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from lingestra.app.main_window import MainWindow, Config
from lingestra.live.recorder import RecState
from test_app_offscreen import FakeWorker, Ev, scratch_store, _hand_lm

app = QApplication.instance() or QApplication([])
MODEL = str(ROOT / "models" / "demo_family.keras")


def pump(cond=None, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents(); time.sleep(0.01)
        if cond is not None and cond():
            return True
    return cond is None


def clip(n=60):
    f = sorted((ROOT / "data" / "landmarks").glob("*.npy"))[3]
    a = np.load(f).astype(np.float32)
    return a[:n] if len(a) >= n else a


@pytest.fixture
def win():
    store = scratch_store("test_store_library_ui")
    w = MainWindow(Config(source="landmarks:x.npy", model=MODEL, store_dir=str(store.dir)),
                   worker_factory=FakeWorker, store=store)
    w.show(); w.start_worker(); app.processEvents()
    w.show_section("signs")
    yield w
    w.close()


def first_with(win, text):
    lt = win.lib_tab
    lt.search.setText(text); lt.refresh()
    return lt.list.item(0).data(Qt.UserRole)


def test_library_lists_everything_and_searches(win):
    lt = win.lib_tab
    assert len(win.library.entries()) > 1000 and lt.list.count() > 1000
    lt.search.setText("mother"); lt._debounce.stop(); lt.refresh()
    assert 0 < lt.list.count() < 100
    assert any("mother" in lt.list.item(i).text().lower() for i in range(lt.list.count()))
    lt.search.setText("zzzzqqq"); lt.refresh()
    assert lt.list.count() == 0 and lt.empty.isVisibleTo(lt) and "couldn't find" in lt.empty.text()
    lt.search.setText("")
    lt.set_chip("mine"); assert lt.list.count() == 0 and "first sign" in lt.empty.text()    # no user signs yet
    assert lt.b_first.isVisibleTo(lt)                                                         # call to action
    lt.set_chip("inuse"); assert lt.list.count() == 0                                         # nothing switched on yet
    lt.set_chip("all"); assert lt.list.count() > 1000


def test_lazy_thumbnails_fill_visible_items(win):
    lt = win.lib_tab
    pump(lambda: lt.list.item(0).data(Qt.DecorationRole) is not None, 5)
    assert lt.list.item(0).data(Qt.DecorationRole) is not None
    assert lt.list.item(lt.list.count() - 1).data(Qt.DecorationRole) is None     # far away: not rendered yet


def test_click_plays_in_main_view_and_esc_returns(win):
    lt, st = win.lib_tab, win.stage
    first_with(win, "mother")
    lt.list.itemClicked.emit(lt.list.item(0))
    assert st.mode == "playback" and st.skeleton_page.isVisible() and not st.camera_page.isVisible()
    assert st.title.text() == lt.list.item(0).text() and "listening in the background" in st.note.text()
    i0 = st.canvas.index
    assert pump(lambda: st.canvas.index != i0 or st.pos > 3, 2.0), "frames did not advance"
    st.play_btn.click(); assert not st.playing
    st.set_speed(2.0); assert st.speed == 2.0
    win._on_escape()
    assert st.mode == "live" and st.camera_page.isVisible() and not st.playing


def test_phrase_playback_sequence(win):
    lt, st = win.lib_tab, win.stage
    lt.search.setText("mother father xyzzy"); lt.refresh()
    assert lt.phrase_btn.isVisibleTo(lt) and "xyzzy" in lt.missing.text() and lt.missing.text().startswith("No sign found for")
    seen = []
    st.segment_changed.connect(lambda i, t: seen.append(i))
    lt.phrase_btn.click()
    assert st.mode == "playback" and len(st.segs) == 2 and st.loop_box.isChecked() is False
    assert st.subtitle.text().startswith("Sign 1 of 2")
    for _ in range(400):
        st.advance(0.1)
        if not st.playing:
            break
    assert 1 in seen and st.progress.value() == 1000 and not st.playing      # reached the end, not looping
    # Enter plays the phrase too
    win._on_escape(); lt.on_enter()
    assert st.mode == "playback" and len(st.segs) == 2


def test_edit_roundtrip_reaches_worker(win):
    lt, w = win.lib_tab, win.worker
    lt.search.setText("mother"); lt.refresh()
    it = next(lt.list.item(i) for i in range(lt.list.count()) if lt.list.item(i).text().lower() == "mother")
    lt.list.setCurrentItem(it)
    id_ = it.data(Qt.UserRole)
    assert lt.ed_label.isEnabled() and not lt.b_delete.isEnabled()
    lt.ed_label.setText("Mum"); lt.ed_alias.setText("mama, أم"); lt.ed_notes.setPlainText("<b>x</b>\nline2")
    lt.save()
    e = win.library.get(id_)
    assert e.label == "Mum" and e.edited and "mama" in e.aliases and e.notes.startswith("<b>")
    assert w.label_map and w.label_map.get(e.model_class) == "Mum"
    assert win.library.label_map() == w.label_map
    assert "Mum" in [lt.list.item(i).text() for i in range(lt.list.count())] or lt.search.text()
    # Arabic alias search finds it
    lt.search.setText("أم"); lt.refresh()
    assert id_ in [lt.list.item(i).data(Qt.UserRole) for i in range(lt.list.count())]
    # invalid text is rejected and shown, nothing changes
    lt.select(id_); lt.ed_label.setText("bad/name"); lt.save()
    assert "path" in lt.ed_msg.text() and win.library.get(id_).label == "Mum"
    # plain text everywhere
    from PySide6.QtWidgets import QLabel
    assert lt.ed_head.textFormat() == Qt.PlainText and lt.ed_msg.textFormat() == Qt.PlainText
    # hide / reset
    lt.select(id_); lt.toggle_hidden(); assert win.library.get(id_).hidden and lt.b_hide.text() == "Show again"
    lt.toggle_hidden()
    lt.confirm = lambda *a: True
    lt.reset_sign(); e = win.library.get(id_)
    assert e.label == e.default_label and not e.edited and w.label_map == {}
    # store audit log has our edits
    acts = [a["action"] for a in win.store.read_audit()]
    assert "library_update" in acts and "library_hide" in acts and "library_reset" in acts


def teach(win, mode_btn=None, name=""):
    w, st = win.worker, win.stage
    w.calls.clear()
    if mode_btn is None:
        win.new_tab.name.setText(name); win.new_tab.teach_btn.click()
    else:
        mode_btn.click()
    assert ("start_capture",) in w.calls and st.mode == "capture" and st.camera_page.isVisible()
    for stt in (RecState("countdown", "Get ready: 3", 2.6, 0), RecState("waiting", "Raise your hands and sign", 8, 0),
                RecState("recording", "Recording...", 5, 12)):
        w.capture_state.emit(stt)
        if stt.state == "countdown":
            assert st.ov_big.text() == "3"
        if stt.state == "recording":
            assert st.ov_big.text() == "Recording"
    return w, st


def test_teach_new_sign_end_to_end(win):
    n0 = len(win.library.entries())
    w, st = teach(win, name="Zebra test")
    c = clip()
    w.capture_result.emit(RecState("done", "Recorded", 0, len(c), c), 25.0)
    assert st.mode == "review" and st.skeleton_page.isVisible() and st.name_edit.text() == "Zebra test"
    # retake restarts capture
    w.calls.clear(); st.b_retake.click()
    assert ("start_capture",) in w.calls and st.mode == "capture"
    w.capture_result.emit(RecState("failed", "No hands seen: ...", 0, 5), 25.0)
    assert st.ov_retry.isVisibleTo(st) and "No hands seen" in st.ov_sub.text()
    st.ov_retry.click(); assert st.mode == "capture"
    w.capture_result.emit(RecState("done", "Recorded", 0, len(c), c), 25.0)
    st.name_edit.setText(""); st.name_edit.returnPressed.emit()
    assert st.mode == "review" and "type" in st.review_msg.text().lower()       # empty name refused
    st.name_edit.setText("Zebra test"); st.name_edit.returnPressed.emit()       # Enter = Keep
    assert st.mode == "live" and st.camera_page.isVisible()
    assert len(win.library.entries()) == n0 + 1
    e = win.library.search("zebra test")[0]
    assert e.source == "recorded" and e.category == "My signs" and win.library.animation(e.id).shape[1] == 67
    win.lib_tab.set_chip("mine")
    assert win.lib_tab.list.count() == 1
    assert win.new_tab.nudge.isVisibleTo(win.new_tab) and "2 more examples" in win.new_tab.nudge_text.text()
    assert win.lib_tab.list.item(0).data(Qt.UserRole + 1)["mine"]


def test_discard_and_cancel_resume_recognition(win):
    w, st = teach(win)
    c = clip()
    w.capture_result.emit(RecState("done", "Recorded", 0, len(c), c), 25.0)
    n0 = len(win.library.entries())
    st.b_discard.click()
    assert st.mode == "live" and len(win.library.entries()) == n0
    w, st = teach(win)
    w.calls.clear(); win._on_escape()          # Esc during capture cancels it
    assert ("cancel_capture",) in w.calls and st.mode == "live"


def test_rerecord_and_extra_take(win):
    lt = win.lib_tab
    id_ = first_with(win, "mother")
    lt.select(id_)
    w, st = teach(win, lt.b_take)
    c = clip(); w.capture_result.emit(RecState("done", "Recorded", 0, len(c), c), 25.0)
    assert not st.name_edit.isVisibleTo(st)
    st.b_keep.click()
    assert win.library.get(id_).takes == 2 and st.mode == "live"
    # take selector in the stage
    lt.list.itemClicked.emit(lt.list.currentItem())
    assert st.take_box.isVisibleTo(st) and st.take_box.count() == 2
    st.take_box.setCurrentIndex(1); assert st.mode == "playback"
    win._on_escape()
    w, st = teach(win, lt.b_rerec)
    w.capture_result.emit(RecState("done", "Recorded", 0, len(c), c), 25.0)
    st.b_keep.click()
    assert win.library.get(id_).edited


def test_pending_cards_promote_merge_discard(win):
    store, nt = win.store, win.new_tab
    for conf in (0.3, 0.4, 0.5):
        store.save_unknown(clip(), {"confidence": conf})
    win.worker.unknown_saved.emit("x")
    assert len(nt.cards) == 3 and win.nav["teach"].badge == 3
    card = nt.cards[0]
    card.name.setText("Cousin"); card.promote()                     # Enter in the name box does this
    assert win.library.search("cousin")[0].source == "recorded" and len(nt.cards) == 2
    assert len(store.list_unknown()) == 2
    target = first_with(win, "father")
    c2 = nt.cards[0]; nt.merge(c2, target)
    assert win.library.get(target).takes == 2 and len(nt.cards) == 1
    c3 = nt.cards[0]; c3.promote(); assert "name" in c3.msg.text().lower()          # needs a name
    played = []; nt.play_array.connect(lambda t, f: played.append(len(f)))
    c3.b_play.click(); assert played and win.stage.mode == "playback"
    win._on_escape()
    c3.discard()
    assert nt.cards == [] and store.list_unknown() == [] and nt.total == 0
    assert win.nav["teach"].badge == 0
    # hover animates a card
    store.save_unknown(clip(), {}); nt.refresh()
    cd = nt.cards[0]; assert not cd.canvas.is_playing
    nt.select_card(cd); assert cd.canvas.is_playing
    pump(lambda: cd.canvas.pos > 1, 1.5)
    assert cd.canvas.pos > 0


def test_store_unavailable_is_read_only(monkeypatch):
    import lingestra.security.store as S
    def boom(*a, **k): raise RuntimeError("no keyring")
    monkeypatch.setattr(S, "SecureStore", boom)
    w = MainWindow(Config(source="landmarks:x.npy", model=MODEL), worker_factory=FakeWorker)
    w.show(); w.start_worker()
    lt = w.lib_tab
    assert lt.list.count() > 1000 and not w.editable
    lt.list.setCurrentRow(0)
    assert not lt.ed_label.isEnabled() and not lt.b_save.isEnabled() and "looked at" in lt.ed_msg.text()
    assert "safe storage could not be opened" in w.status.text() and not w.new_tab.teach_btn.isEnabled()
    assert w.store_card.isVisibleTo(w.talk_panel)
    w.begin_capture("new", None, "x"); assert w.stage.mode == "live"
    lt.list.itemClicked.emit(lt.list.item(0)); assert w.stage.mode == "playback"      # browsing still works
    w.close()


def test_no_model_json_and_long_labels(win, tmp_path):
    w = MainWindow(Config(source="landmarks:x.npy", model=str(tmp_path / "nomodel.keras")),
                   worker_factory=FakeWorker, store=scratch_store("test_store_nomodel"))
    w.show()
    assert w.lib_tab.list.count() > 1000
    w.close()
    lt = win.lib_tab
    id_ = first_with(win, "mother"); lt.select(id_)
    lt.ed_label.setText("L" * 80); lt.save()
    assert win.library.get(id_).label == "L" * 80
    win.play_entry(id_)
    assert win.stage.title.text() == "L" * 80 and win.stage.title.styleSheet().count("21px")


def test_real_worker_capture(monkeypatch):
    """Real RecognitionWorker + SignRecorder on a replayed landmark stream: capture_result carries a clip,
    recognition is suspended meanwhile (no sentences)."""
    from PySide6.QtCore import QCoreApplication
    from lingestra.app import worker as W
    from lingestra.live import sources
    c = clip(60)
    idle = np.repeat(c[:1], 90, axis=0).copy()
    idle[:, 25:] = np.nan; idle[:, [15, 16], 1] = idle[:, [23, 24], 1].mean() + 0.1
    stream = np.concatenate([idle, c, idle[:50]]).astype(np.float32)
    monkeypatch.setattr(W, "make_source", lambda spec, loop_speed=1.0: sources.LandmarkFileSource(stream, fps=50))
    wk = W.RecognitionWorker("landmarks:x", MODEL, send_mode="none", store=None)
    res, evs = [], []
    wk.capture_result.connect(lambda st, fps: res.append((st, fps)), Qt.DirectConnection)
    wk.events.connect(lambda e: evs.extend(e), Qt.DirectConnection)
    wk.set_label_map({"Mother": "Mum"}); wk.start_capture()
    wk.start()
    pump(lambda: res or not wk.isRunning(), 30)
    wk.stop(); wk.wait(5000)
    assert res and res[0][0].state in ("done", "failed")
    if res[0][0].state == "done":
        assert res[0][0].clip.shape[1:] == (67, 3) and 20 <= res[0][1] <= 60
    assert not [e for e in evs if e.kind in ("sign", "sentence")]


# ------------------------------------------------------------------ resizable panel + bulk switches
from PySide6.QtCore import QPoint, QPointF, QEvent
from PySide6.QtGui import QMouseEvent
from PySide6.QtTest import QTest
from lingestra.app.library_tab import SPLIT_KEY, SPLIT_DEFAULT, BULK_CONFIRM, CELL


def sizes(lt):
    return lt.split.sizes()


def drag_handle(lt, dy):
    h = lt.split.handle(1)
    c = h.rect().center()
    QTest.mousePress(h, Qt.LeftButton, Qt.NoModifier, c); app.processEvents()
    QTest.mouseMove(h, c + QPoint(0, dy)); app.processEvents()
    QTest.mouseRelease(h, Qt.LeftButton, Qt.NoModifier, c + QPoint(0, dy)); app.processEvents()


def test_splitter_has_grip_tooltip_and_cursor(win):
    lt = win.lib_tab; pump(timeout=0.1)
    h = lt.split.handle(1)
    assert lt.split.orientation() == Qt.Vertical and h.height() >= 14
    assert "taller or shorter" in h.toolTip() and h.cursor().shape() == Qt.SplitVCursor
    assert not lt.split.childrenCollapsible()
    assert lt.split.widget(0).isAncestorOf(lt.list) and lt.split.widget(1).isAncestorOf(lt.b_save)
    img = h.grab().toImage()                                           # the grip is really painted (dots in the middle)
    mid = img.pixelColor(img.width() // 2, img.height() // 2); edge = img.pixelColor(4, img.height() // 2)
    assert mid != edge


def test_splitter_drag_respects_minimum_heights(win):
    lt = win.lib_tab; win.resize(1360, 1100); pump(timeout=0.15)
    t0, b0 = sizes(lt)
    drag_handle(lt, -60); t1, b1 = sizes(lt)
    assert t1 < t0 and b1 > b0 and t0 + b0 == t1 + b1
    drag_handle(lt, -2000); t2, b2 = sizes(lt)                          # all the way up
    assert t2 == lt.split.widget(0).minimumHeight() >= 2 * CELL.height() and b2 >= lt.PANEL_MIN
    assert lt.list.height() >= 2 * CELL.height()
    drag_handle(lt, 3000); t3, b3 = sizes(lt)                           # all the way down
    assert b3 >= lt.PANEL_MIN and t3 >= 2 * CELL.height()
    lt.list.setCurrentRow(0); app.processEvents()
    assert lt.save_bar.isVisibleTo(lt) and lt.b_save.height() >= 40 and lt.b_cancel.height() >= 40
    # Save / Cancel stay inside the panel at the smallest panel size
    drag_handle(lt, 3000)
    pr = lt.panel.rect(); bb = lt.b_save.mapTo(lt.panel, QPoint(0, 0))
    assert bb.y() >= 0 and bb.y() + lt.b_save.height() <= pr.height()


@pytest.mark.parametrize("size", [(1000, 640), (1100, 760), (1360, 860), (1800, 1100)])
def test_no_clipping_at_several_window_sizes(win, size):
    lt = win.lib_tab; win.resize(*size); pump(timeout=0.15)
    lt.list.setCurrentRow(0); pump(timeout=0.1)
    t, b = sizes(lt)
    assert t >= 2 * CELL.height() and b >= lt.PANEL_MIN
    # nothing of the splitter is cut off: either it fits in the view or the whole screen scrolls
    vp = lt.page_scroll.viewport()
    sp = lt.split.mapTo(lt.body, QPoint(0, 0))
    fits = sp.y() + lt.split.height() <= lt.body.height() + 1
    assert fits and lt.body.height() >= lt.body.minimumSizeHint().height()
    assert lt.b_save.mapTo(lt.panel, QPoint(0, 0)).y() + lt.b_save.height() <= lt.panel.height()
    assert lt.page_scroll.horizontalScrollBar().maximum() == 0
    if size[1] >= 860:
        assert lt.page_scroll.verticalScrollBar().maximum() == 0           # roomy window: no scrolling needed


def test_double_click_resets_to_default_ratio(win):
    lt = win.lib_tab; win.resize(1360, 1100); pump(timeout=0.15)
    drag_handle(lt, -80)
    assert abs(lt.split.ratio - SPLIT_DEFAULT) > 0.03
    h = lt.split.handle(1)
    QTest.mouseDClick(h, Qt.LeftButton, Qt.NoModifier, h.rect().center()); pump(timeout=0.1)
    t, b = sizes(lt)
    assert lt.split.ratio == SPLIT_DEFAULT and abs(t / (t + b) - SPLIT_DEFAULT) < 0.03


def test_split_position_is_saved_with_audit_and_restored_in_a_new_window():
    store = scratch_store("test_store_split_persist")
    cfg = Config(source="landmarks:x.npy", model=MODEL, store_dir=str(store.dir))
    w = MainWindow(cfg, worker_factory=FakeWorker, store=store); w.show(); w.start_worker(); w.show_section("signs")
    w.resize(1360, 1100); pump(timeout=0.2)
    n0 = len([a for a in store.read_audit() if a["action"] == "settings_update"])
    drag_handle(w.lib_tab, 90); r = w.lib_tab.split.ratio
    assert store.load_settings()[SPLIT_KEY] == pytest.approx(r, abs=0.001)
    assert len([a for a in store.read_audit() if a["action"] == "settings_update"]) == n0 + 1
    assert w.settings[SPLIT_KEY] == pytest.approx(r, abs=0.001)                 # the window's own settings agree
    w.close()
    w2 = MainWindow(cfg, worker_factory=FakeWorker, store=store); w2.show(); w2.start_worker(); w2.show_section("signs")
    w2.resize(1360, 1100); pump(timeout=0.2)
    t, b = sizes(w2.lib_tab)
    assert w2.lib_tab.split.ratio == pytest.approx(r, abs=0.001) and abs(t / (t + b) - r) < 0.03
    w2.close()


def test_split_position_works_without_a_store():
    from lingestra.library import SignLibrary
    from lingestra.app.library_tab import LibraryTab
    lt = LibraryTab(SignLibrary(store=None, model_classes=[]), editable=False)
    lt.resize(900, 900); lt.show(); pump(timeout=0.15)
    drag_handle(lt, -60); r = lt.split.ratio
    assert lt._mem_settings[SPLIT_KEY] == pytest.approx(r, abs=0.001)
    lt.close()


# ---- bulk switches
def counts_text(lt):
    return lt.all_label.text()


def test_bulk_all_on_off_row_and_counts_text(win):
    lt, lib = win.lib_tab, win.library
    n = lib.active_counts()[1]
    assert counts_text(lt) == f"Signs switched on: 0 of {n:,}"
    assert lt.b_all_on.text() == "Switch all on" and lt.b_all_off.text() == "Switch all off"
    assert not lt.b_all_off.isEnabled()
    assert not lt.b_cat_on.isEnabled() and not lt.b_cat_off.isEnabled()      # 'All categories': disabled with a hint
    assert "Pick a category" in lt.cat_label.text() and lt.cat_combo.currentText() == "All categories"
    assert not lt.cat_row.isVisibleTo(lt)        # CHANGED: the per-category row is hidden until a category is chosen


def small_category(lib):
    cs = [(len(lib.entries(category=c)), c) for c in lib.categories() if c]
    return min(x for x in cs if x[0] >= 3)[1]


def test_category_switch_affects_whole_category_regardless_of_search(win):
    lt, lib, w = win.lib_tab, win.library, win.worker
    cat = small_category(lib); ids = {e.id for e in lib.entries(category=cat)}
    lt.set_category(cat); lt.set_chip("all")
    assert lt.cat_label.text() == f"{cat}: 0 of {len(ids)} switched on"  # CHANGED: one wording
    assert "all" in lt.b_cat_on.toolTip() and "not only" in lt.b_cat_on.toolTip()
    lt.search.setText("zzzz"); lt.refresh()                                  # search matches nothing at all
    lt.b_cat_on.click()
    assert {i for i in ids if lib.is_active(i)} == ids
    assert lib.active_counts()[0] == len(ids)                               # nothing outside the category
    assert lt.cat_label.text() == f"{cat}: {len(ids)} of {len(ids)} switched on"  # CHANGED: one wording
    assert not lt.b_cat_on.isEnabled() and lt.b_cat_off.isEnabled()


def test_confirmation_above_100_and_not_below(win):
    lt, lib = win.lib_tab, win.library
    asked = []
    lt.confirm_many = lambda n: (asked.append(n), False)[1]                  # user says Cancel
    lt.b_all_on.click()
    assert asked == [lib.active_counts()[1]] and lib.active_counts()[0] == 0       # cancelled: nothing changed
    asked.clear()
    cat = small_category(lib); assert len(lib.entries(category=cat)) <= BULK_CONFIRM
    lt.set_category(cat); lt.b_cat_on.click()
    assert asked == [] and lib.active_counts(cat)[0] == lib.active_counts(cat)[1]      # small: no question
    lt.set_category(None); asked.clear()
    lt.confirm_many = lambda n: (asked.append(n), True)[1]
    lt.b_all_on.click()
    assert len(asked) == 1 and lib.active_counts()[0] == lib.active_counts()[1]
    # exactly 100 new signs: no question; 101: question
    lt.b_all_off.click(); asked.clear()
    ids = [e.id for e in lib.entries()]
    for i in ids[:BULK_CONFIRM]:
        lib.set_active(i, False)
    lib.set_active_many(ids, True); lib.set_active_many(ids[:BULK_CONFIRM], False); lt.reload()
    lt.b_all_on.click(); assert asked == []
    lib.set_active_many(ids[:BULK_CONFIRM + 1], False); lt.reload()
    lt.b_all_on.click(); assert asked == [BULK_CONFIRM + 1]


def test_undo_bar_restores_the_exact_previous_mixed_state(win):
    lt, lib = win.lib_tab, win.library
    ids = [e.id for e in lib.entries()][:30]
    lib.set_active(ids[0], True); lib.set_active(ids[1], False); lib.set_active(ids[2], True)
    mine = teach_mine(win)
    lt.reload()
    before = {i: (lib._active.get(i), lib.is_active(i)) for i in [e.id for e in lib.entries()]}
    lt.b_all_off.click()
    assert lib.active_counts()[0] == 0
    assert lt.undo_bar.isVisibleTo(lt) and lt.b_undo.isVisibleTo(lt)
    assert lt.undo_label.text().startswith("Switched off") and "sign" in lt.undo_label.text()
    lt.b_undo.click()
    assert {i: (lib._active.get(i), lib.is_active(i)) for i in [e.id for e in lib.entries()]} == before
    assert lib.is_active(mine) and not lt.b_undo.isVisibleTo(lt)
    assert lt.all_label.text() == f"Signs switched on: {lib.active_counts()[0]:,} of {lib.active_counts()[1]:,}"


def teach_mine(win):
    return win.library.add_recorded("Zebra", clip())


def test_undo_bar_disappears_after_about_ten_seconds(win):
    lt = win.lib_tab; lt.undo_ms = 120
    cat = small_category(win.library); lt.set_category(cat); lt.b_cat_on.click(); lt.b_cat_off.click()
    assert lt.undo_bar.isVisibleTo(lt)
    pump(lambda: not lt.undo_bar.isVisibleTo(lt), 1.5)
    assert not lt.undo_bar.isVisibleTo(lt)
    assert LibraryTabDefault.UNDO_MS == 30000      # CHANGED: longer, so a switch-off is not lost while away


from lingestra.app import library_tab as LibraryTabDefault
LibraryTabDefault.UNDO_MS = LibraryTabDefault.UNDO_MS


def test_bulk_refreshes_badges_counts_and_pushes_templates_once(win):
    lt, lib, w = win.lib_tab, win.library, win.worker
    cat = small_category(lib); n_cat = len(lib.entries(category=cat))
    lt.set_chip("all"); lt.set_category(cat)
    pushes = []
    orig = w.set_templates; w.set_templates = lambda t: (pushes.append(len(t)), orig(t))[1]
    singles = []
    o2 = lib.set_active; lib.set_active = lambda *a, **k: (singles.append(a), o2(*a, **k))[1]
    lt.b_cat_on.click()
    assert pushes == [n_cat] and singles == []                          # ONE push, no per-sign calls
    assert all(lt.list.item(i).data(Qt.UserRole + 1)["active"] for i in range(lt.list.count()))
    lt.b_cat_off.click()
    assert pushes == [n_cat, 0]
    assert not any(lt.list.item(i).data(Qt.UserRole + 1)["active"] for i in range(lt.list.count()))
    lt.b_undo.click()
    assert pushes == [n_cat, 0, n_cat]


def test_bulk_only_changes_active_flags_on_disk(win):
    lt, lib, store = win.lib_tab, win.library, win.store
    keys = lambda: sorted((ns, k) for ns in ("library", "lib_takes")
                          for k in store.list_blobs(ns) if (ns, k) != ("library", "active")) if hasattr(store, "list_blobs") else []
    sid = teach_mine(win); lib.update(sid, label="Zebra2"); lib.hide(lib.entries()[5].id)
    before = keys(); assert before; ov = lib._overrides.copy()
    cat = small_category(lib); lt.set_category(cat); lt.b_cat_on.click(); lt.b_cat_off.click(); lt.b_undo.click()
    assert keys() == before and lib._overrides == ov
    assert [a["action"] for a in store.read_audit() if a["action"].startswith("library_")][-3:] == [
        "library_set_active_many", "library_set_active_many", "library_restore_active"]


def test_bulk_is_fast_in_the_screen(win):
    lt = win.lib_tab
    lt.confirm_many = lambda n: True
    t = time.perf_counter(); lt.b_all_on.click(); dt = time.perf_counter() - t
    print(f"switch all on incl. screen refresh: {dt*1000:.0f} ms")
    assert dt < 1.5


# ------------------------------------------------------------------ screenshots (docs/ui_my_signs*.png)
def test_my_signs_screenshots():
    docs = ROOT / "docs"; docs.mkdir(exist_ok=True)
    store = scratch_store("test_store_library_shots")
    w = MainWindow(Config(source="landmarks:x.npy", model=MODEL, store_dir=str(store.dir)),
                   worker_factory=FakeWorker, store=store)
    try:
        w.show(); w.start_worker(); w.resize(1360, 960)
        lib = w.library
        for name, i in (("Mother", 3), ("Water", 20), ("Home", 30)):
            lib.add_recorded(name, clip())
        w._on_library_changed(); w.show_section("signs"); lt = w.lib_tab
        lt.confirm_many = lambda n: True
        lt.set_chip("all"); pump(lambda: lt.pending_thumbnails() == 0, 3); pump(timeout=0.4)
        assert w.grab().save(str(docs / "ui_my_signs.png"))
        # panel dragged taller + a sign selected
        lt.search.setText("mother"); lt.refresh(); lt.list.setCurrentRow(0); lt.list.itemClicked.emit(lt.list.item(0))
        w.resize(1360, 1000); pump(timeout=0.3)
        drag_handle(lt, 150)
        lt.search.setText(""); lt.refresh(keep=lt.cur); pump(lambda: lt.pending_thumbnails() == 0, 3); pump(timeout=0.4)
        assert w.grab().save(str(docs / "ui_my_signs_resized.png"))
        # category chosen, switched on, then another category switched off -> undo bar
        lt.list.setCurrentItem(None); w.resize(1360, 960)
        big = [c for n, c in sorted(((len(lib.entries(category=c)), c) for c in lib.categories() if c)) if 15 <= n <= 40]
        cat, other = big[0], big[1]
        lib.set_active_many([e.id for e in lib.entries(category=cat)][:8], True)
        lib.set_active_many([e.id for e in lib.entries(category=other)][:12], True); lt.reload()
        lt.set_chip("all"); lt.undo_ms = 60000
        lt.set_category(other); lt.b_cat_off.click()                          # 'Switched off 12 signs  [Undo]'
        lt.set_category(cat); pump(lambda: lt.pending_thumbnails() == 0, 3); pump(timeout=0.4)
        assert w.grab().save(str(docs / "ui_my_signs_bulk.png"))
    finally:
        w.close()


def test_switch_on_has_undo_and_switch_off_mentions_own_signs(win):
    lt, lib = win.lib_tab, win.library
    cat = small_category(lib); lt.set_category(cat)
    lt.b_cat_on.click()
    assert lt.b_undo.isVisibleTo(lt) and "Switched on" in lt.undo_label.text()
    lt.b_undo.click()
    assert lib.active_counts(cat)[0] == 0
    assert lt.b_cat_on.text() == f"Switch on all {cat}"
    lt.confirm_many = lambda n: True
    lt.set_category(None); lt.b_all_on.click(); lt.b_all_off.click()
    assert "Switched off" in lt.undo_label.text()



# ------------------------------------------------------------------ switch by group (collections + every category)
def test_groups_list_collections_and_every_category(win):
    from lingestra.app.library_tab import GroupSwitchDialog
    lt, lib = win.lib_tab, win.library
    dlg = GroupSwitchDialog(lt)
    keys = list(dlg.rows)
    assert [k for k in keys if k[0] != "category"] == [("mine", None), ("inuse", None), ("all", None)]
    assert {k[1] for k in keys if k[0] == "category"} == set(c for c in lib.categories() if c)
    assert len(keys) == 3 + len([c for c in lib.categories() if c])
    n_all = lib.active_counts()[1]
    assert dlg.rows[("all", None)][0].text() == f"All dictionary: 0 of {n_all:,} switched on"
    assert dlg.rows[("inuse", None)][1].isEnabled() is False        # nothing is on yet: 'Switch on' has nothing to do
    assert dlg.rows[("inuse", None)][2].isEnabled() is False        # ... and 'Switch off' neither


def test_group_buttons_switch_category_and_collections_with_undo(win):
    from lingestra.app.library_tab import GroupSwitchDialog
    lt, lib, w = win.lib_tab, win.library, win.worker
    lt.confirm_many = lambda n: True
    cat = small_category(lib); ids = {e.id for e in lib.entries(category=cat)}
    dlg = GroupSwitchDialog(lt)
    pushes = len([c for c in w.calls if c[0] == "templates"])
    dlg.rows[("category", cat)][1].click()                              # Switch on <category>
    assert {i for i in ids if lib.is_active(i)} == ids and lib.active_counts()[0] == len(ids)
    assert dlg.rows[("category", cat)][0].text() == f"{cat}: {len(ids)} of {len(ids)} switched on"
    assert dlg.rows[("inuse", None)][0].text() == f"Switched on right now: {len(ids)}"
    assert dlg.note.text().startswith("Switched on")
    pump(timeout=0.5)
    assert len([c for c in w.calls if c[0] == "templates"]) - pushes <= 2       # one refresh, not one per sign
    # 'Switched on' collection: Switch off turns everything off, with the Undo bar on the My signs screen
    dlg.rows[("inuse", None)][2].click()
    assert lib.active_counts()[0] == 0 and lt.undo_bar.isVisibleTo(lt) and lt.b_undo.isVisibleTo(lt)
    lt.b_undo.click()
    assert {i for i in ids if lib.is_active(i)} == ids


def test_recorded_by_me_collection_includes_dictionary_signs_with_my_own_example(win):
    from lingestra.app.library_tab import GroupSwitchDialog, is_mine
    lt, lib = win.lib_tab, win.library
    mine = lib.add_recorded("My own sign", clip())
    d_id = next(e.id for e in lib.entries() if e.source == "dictionary")
    lib.add_take(d_id, clip())                                          # my own example of a dictionary sign
    lt.reload()
    ids = set(lt.group_ids("mine"))
    assert mine in ids and d_id in ids and len(ids) == 2
    assert all(is_mine(lib.get(i)) for i in ids)
    lt.set_chip("mine"); assert lt.list.count() == 2                    # the chip uses the same definition
    dlg = GroupSwitchDialog(lt)
    dlg.rows[("mine", None)][1].click()
    assert all(lib.is_active(i) for i in ids) and lib.active_counts()[0] == 2
    dlg.rows[("mine", None)][2].click()
    assert lib.active_counts()[0] == 0


def test_group_dialog_is_disabled_when_the_store_is_locked(win):
    from lingestra.app.library_tab import GroupSwitchDialog
    lt = win.lib_tab
    lt.editable = False
    dlg = GroupSwitchDialog(lt)
    assert all(not b_on.isEnabled() and not b_off.isEnabled() for _l, b_on, b_off in dlg.rows.values())
