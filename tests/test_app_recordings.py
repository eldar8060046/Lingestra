"""'My recordings' drop-down (in the sign detail, under 'More options'): view and delete the examples of a sign that you
recorded. Offscreen, real SignLibrary + SecureStore in scratch/, fake worker."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import sys, time
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from lingestra import features as F
from lingestra.app.main_window import MainWindow, Config
from lingestra.library import LibraryError
from test_app_offscreen import FakeWorker, scratch_store

app = QApplication.instance() or QApplication([])
MODEL = str(ROOT / "models" / "demo_family.keras")
FILES = sorted((ROOT / "data" / "landmarks").glob("*.npy"))


def clip(i=3, n=60):
    a = np.load(FILES[i]).astype(np.float32)
    return a[:n] if len(a) >= n else a


def good_window():
    """A real signing span (hands visible) for an automatically learned example."""
    for f in FILES[:40]:
        a = F.trim_active(np.load(f).astype(np.float32))
        if len(a) >= 20 and float((~np.isnan(a[:, F.N_POSE:, 0])).any(axis=1).mean()) > 0.8:
            return a
    raise AssertionError("no usable clip")


def pump(cond=None, timeout=3.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        app.processEvents(); time.sleep(0.01)
        if cond is not None and cond():
            return True
    return cond is None


@pytest.fixture
def win():
    store = scratch_store("test_store_recordings")
    w = MainWindow(Config(source="landmarks:x.npy", model=MODEL, store_dir=str(store.dir)),
                   worker_factory=FakeWorker, store=store)
    w.show(); w.start_worker(); app.processEvents()
    w.show_section("signs"); w.lib_tab.set_chip("all")
    yield w
    w.close()


def click(win, sign_id, open_box=True):
    """Select the sign's card (as a click does); the detail panel shows it. Optionally open 'My recordings'."""
    lt = win.lib_tab
    lt.search.setText(win.library.get(sign_id).label); lt.set_chip("all"); lt.refresh(keep=sign_id)
    for i in range(lt.list.count()):
        it = lt.list.item(i)
        if it.data(Qt.UserRole) == sign_id:
            lt.list.setCurrentItem(it); lt._on_current(it, None)       # (refresh() selected it with signals blocked)
            lt.list.itemClicked.emit(it); app.processEvents()
            break
    else:
        raise AssertionError("card not found")
    if open_box and lt.rec_box.isVisibleTo(win):
        lt.rec_box.set_open(True); app.processEvents()
    return lt.rec_panel


def same(a, b):
    return a.shape == b.shape and bool(np.allclose(a, b, equal_nan=True))


def dict_id(win, n=0):
    return [e.id for e in win.library.entries() if e.source == "dictionary"][n]


# ------------------------------------------------------------------ library: examples() / remove_take()
def test_examples_lists_kinds_and_a_replaced_dictionary_clip_can_be_removed(win):
    lib = win.library
    d = dict_id(win, 5)
    assert [x["kind"] for x in lib.examples(d)] == ["dictionary"]
    original = lib.animation(d).copy()
    n1 = lib.add_take(d, clip())
    lib.set_active(d, True)
    n2 = lib.add_auto_take(d, good_window(), now=1_700_000_000.0)
    kinds = {x["take"]: x for x in lib.examples(d)}
    assert kinds[0]["kind"] == "dictionary" and kinds[n1]["kind"] == "mine"
    assert kinds[n2]["kind"] == "learned" and kinds[n2]["time"] == 1_700_000_000.0
    with pytest.raises(LibraryError):
        lib.remove_take(d, 0)                                       # the built-in clip itself stays
    lib.replace_animation(d, clip(5))                               # the user re-records example 0
    assert [x["kind"] for x in lib.examples(d) if x["take"] == 0] == ["mine"]
    assert not same(lib.animation(d), original)
    lib.remove_take(d, 0)                                           # ... deleting it brings the built-in clip back
    assert [x["kind"] for x in lib.examples(d) if x["take"] == 0] == ["dictionary"]
    assert same(lib.animation(d), original)
    lib.remove_take(d, n1)
    assert [x["take"] for x in lib.examples(d)] == [0, n2]
    mine = lib.add_recorded("Only one", clip())
    assert [x["kind"] for x in lib.examples(mine)] == ["mine"]
    with pytest.raises(LibraryError):
        lib.remove_take(mine, 0)                                    # the only example of a sign you created


# ------------------------------------------------------------------ the drop-down in the sign detail
def test_my_recordings_is_a_dropdown_right_under_more_options_for_signs_with_recordings(win):
    lib, lt = win.library, win.lib_tab
    mine = lib.add_recorded("My own sign", clip()); lib.add_take(mine, clip(4))
    plain = dict_id(win, 3)
    click(win, plain, open_box=False)
    assert not lt.rec_box.isVisibleTo(win)                         # a dictionary-only sign: nothing of mine to manage
    panel = click(win, mine, open_box=False)
    assert lt.rec_box.isVisibleTo(win) and lt.rec_box.toggle.text().strip() == "My recordings (2)"
    g = lt.edit_box.layout()
    assert g.indexOf(lt.rec_box) == g.indexOf(lt.more) + 1         # directly below the 'More options' drop-down
    assert not lt.rec_box.is_open and lt.more.is_open is False     # both start closed
    lt.rec_box.set_open(True); app.processEvents()
    assert lt.rec_box.is_open
    assert [r.kind for r in panel.rows] == ["mine", "mine"]
    assert [r.title.text() for r in panel.rows] == ["Example 1", "Example 2"]
    assert all("Recorded by you" in r.sub.text() for r in panel.rows)
    click(win, plain, open_box=False)                              # another sign without recordings: the drop-down goes away
    assert not lt.rec_box.isVisibleTo(win)


def test_selecting_another_sign_with_recordings_updates_the_same_dropdown(win):
    lib, lt = win.library, win.lib_tab
    a = lib.add_recorded("Alpha sign", clip()); lib.add_take(a, clip(4))
    b = lib.add_recorded("Beta sign", clip(5))
    click(win, a)
    assert lt.rec_box.toggle.text().strip() == "My recordings (2)" and len(lt.rec_panel.rows) == 2
    click(win, b, open_box=False)
    assert lt.rec_box.toggle.text().strip() == "My recordings (1)" and lt.rec_panel.sign_id == b and len(lt.rec_panel.rows) == 1
    assert lt.rec_box.is_open                                      # it stays open while you browse


def test_a_dictionary_sign_with_my_example_shows_the_builtin_one_but_cannot_delete_it(win):
    lib = win.library
    d = dict_id(win, 7)
    lib.add_take(d, clip())
    lib.set_active(d, True); lib.add_auto_take(d, good_window(), now=1_700_000_000.0)
    panel = click(win, d)
    assert [r.kind for r in panel.rows] == ["dictionary", "mine", "learned"]
    builtin, mine, learned = panel.rows
    assert builtin.title.text() == "Built-in example" and not builtin.b_del.isEnabled()
    assert "cannot be deleted" in builtin.b_del.toolTip()
    assert mine.title.text() == "Example 1" and mine.b_del.isEnabled()
    assert learned.title.text() == "Example 2" and "Learned from a correction" in learned.sub.text() and learned.b_del.isEnabled()
    assert win.lib_tab.rec_box.toggle.text().strip() == "My recordings (2)"           # the built-in clip is not counted


def test_the_only_example_of_a_sign_i_created_cannot_be_deleted_here(win):
    mine = win.library.add_recorded("Solo", clip())
    panel = click(win, mine)
    assert len(panel.rows) == 1 and not panel.rows[0].b_del.isEnabled()
    assert "only example" in panel.rows[0].b_del.toolTip() and "More options" in panel.rows[0].b_del.toolTip()


# ------------------------------------------------------------------ view and delete
def test_watch_plays_that_example_in_the_main_view(win):
    lib = win.library
    mine = lib.add_recorded("Watcher", clip()); lib.add_take(mine, clip(4)); lib.add_take(mine, clip(5))
    panel = click(win, mine)
    panel.rows[2].b_watch.click()
    assert win._stage_entry == mine and win.stage.take_box.currentIndex() == 2


def test_delete_needs_two_taps_then_removes_the_example_and_refreshes_everything(win):
    lib, w, lt = win.library, win.worker, win.lib_tab
    mine = lib.add_recorded("Deleter", clip()); lib.add_take(mine, clip(4)); lib.add_take(mine, clip(5))
    lib.set_active(mine, True); win._push_all()
    panel = click(win, mine)
    n_templates = len([c for c in w.calls if c[0] == "templates"])
    row = panel.rows[1]
    row.b_del.click()
    assert row.b_del.text() == "Tap again to delete" and len(lib.take_numbers(mine)) == 3        # first tap: only asks
    row.disarm(); assert row.b_del.text() == "Delete"                                           # it forgets if you wait
    row.b_del.click(); assert len(lib.take_numbers(mine)) == 3
    row.b_del.click()                                                                          # second tap: deleted
    assert len(lib.take_numbers(mine)) == 2 and len(lt.rec_panel.rows) == 2
    assert lt.rec_panel.note.text() == "Deleted." and lt.rec_box.toggle.text().strip() == "My recordings (2)"
    assert lt.rec_box.is_open                                                                   # the drop-down stays open
    pump(timeout=0.5)
    assert len([c for c in w.calls if c[0] == "templates"]) > n_templates                     # the recogniser got the new set
    assert any(a["action"] == "library_remove_take" for a in win.store.read_audit())
    assert win.library.get(mine).takes == 2


def test_example_numbers_with_gaps_still_play_correctly_after_a_deletion(win):
    lib = win.library
    mine = lib.add_recorded("Gappy", clip()); lib.add_take(mine, clip(4)); lib.add_take(mine, clip(5))
    panel = click(win, mine)
    panel.rows[1].b_del.click(); panel.rows[1].b_del.click()                                    # delete example 2 of 3
    nums = lib.take_numbers(mine)
    assert nums == [0, 2]                                                                      # a gap in the NUMBERS
    win.play_entry(mine)
    assert win.stage.take_box.count() == 2
    win.stage.take_box.setCurrentIndex(1)                                                     # 'Example 2 of 2' = take number 2
    pump(timeout=0.3)
    assert "not available" not in win.status.text()
    win.lib_tab.rec_panel.rows[1].b_watch.click()
    assert win.stage.take_box.currentIndex() == 1


def test_a_learned_example_can_be_deleted_and_the_learn_bar_follows(win):
    lib = win.library
    d = dict_id(win, 9)
    lib.set_active(d, True); n = lib.add_auto_take(d, good_window(), now=1_700_000_000.0)
    win._learned = (d, n, "x"); win.learn_bar.show()
    panel = click(win, d)
    learned = [r for r in panel.rows if r.kind == "learned"][0]
    learned.b_del.click(); learned.b_del.click()
    assert lib.auto_takes(d) == [] and [x["kind"] for x in lib.examples(d)] == ["dictionary"]
    assert not win.lib_tab.rec_box.isVisibleTo(win)                                           # nothing of mine is left
    assert win._learned is None and not win.learn_bar.isVisibleTo(win)


def test_record_another_example_mirror_and_read_only_mode(win):
    lib, lt = win.library, win.lib_tab
    mine = lib.add_recorded("Extras", clip()); lib.add_take(mine, clip(4))
    panel = click(win, mine)
    asked = []
    lt.record_requested.connect(lambda mode, sid: asked.append((mode, sid)))
    panel.b_more.click()
    assert asked == [("take", mine)]
    win.set_mirror(False)
    assert not any(r.canvas.mirror for r in panel.rows)
    win.set_mirror(True)
    assert all(r.canvas.mirror for r in panel.rows)
    from lingestra.app.recordings_panel import RecordingsPanel
    ro = RecordingsPanel(lib, editable=False); ro.set_sign(mine)
    assert not ro.b_more.isEnabled() and ro.rows and not any(r.b_del.isEnabled() for r in ro.rows)


def test_cards_show_how_many_recordings_of_mine_a_sign_has(win):
    from lingestra.app.library_tab import ROLE_FLAGS, ROLE_ID, recordings_count
    lib, lt = win.library, win.lib_tab
    plain = dict_id(win, 11)
    d = dict_id(win, 12); lib.add_take(d, clip()); lib.add_take(d, clip(4))
    mine = lib.add_recorded("Counted", clip())
    assert recordings_count(lib.get(plain)) == 0 and recordings_count(lib.get(d)) == 2 and recordings_count(lib.get(mine)) == 1
    lt.reload(); lt.search.setText(""); lt.set_chip("all"); lt.refresh()
    flags, tips = {}, {}
    for i in range(lt.list.count()):
        it = lt.list.item(i)
        import html
        flags[it.data(ROLE_ID)] = it.data(ROLE_FLAGS)["mine"]; tips[it.data(ROLE_ID)] = html.unescape(it.toolTip())   # (escaped on purpose)
    assert flags[mine] == 1 and "1 recording of this sign" in tips[mine]
    assert flags.get(d) == 2 and "2 recordings of this sign: open 'My recordings' below" in tips[d]
    assert flags.get(plain, 0) == 0 and "My recordings" not in tips.get(plain, "")
