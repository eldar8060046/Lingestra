"""Bulk switch on/off in the library: one write, one audit record, exact undo, counts, speed."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import shutil, sys, time
from pathlib import Path
import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
from lingestra.library import SignLibrary
from test_app_offscreen import scratch_store

MODEL = ROOT / "models" / "demo_family.keras"


def make_lib(name):
    store = scratch_store(name)
    return SignLibrary(store=store, model_classes=[]), store


def audit_actions(store):
    return [a["action"] for a in store.read_audit()]


def test_set_active_many_correct_and_single_audit_record():
    lib, store = make_lib("test_store_bulk_a")
    ids = [e.id for e in lib.entries()][:50]
    n0 = len(store.read_audit())
    saves = []
    orig = lib._save; lib._save = lambda k, o, a: (saves.append(k), orig(k, o, a))[1]
    prev = lib.set_active_many(ids, True)
    assert all(lib.is_active(i) for i in ids) and len(prev) == 50
    assert saves == ["active"]                                           # ONE persisted write
    new = store.read_audit()[n0:]                                        # the store logs its own blob_save too
    assert [a["action"] for a in new] == ["blob_save", "library_set_active_many"]
    assert new[1]["extra"] == {"on": True, "count": 50}
    assert lib.set_active_many(ids, True) == {} and len(store.read_audit()) == n0 + 2    # nothing to do: nothing written
    lib.set_active_many(ids[:10], False)
    assert not any(lib.is_active(i) for i in ids[:10]) and all(lib.is_active(i) for i in ids[10:])
    with pytest.raises(KeyError):
        lib.set_active_many(ids[:3] + ["nope"], False)
    assert all(lib.is_active(i) for i in ids[10:13])                      # all-or-nothing


def test_persistence_round_trip():
    lib, store = make_lib("test_store_bulk_b")
    ids = [e.id for e in lib.entries()]
    lib.set_active_many(ids[:30], True)
    lib2 = SignLibrary(store=store, model_classes=[])
    assert [i for i in ids if lib2.is_active(i)] == ids[:30]


def test_performance_all_signs_under_half_a_second():
    lib, store = make_lib("test_store_bulk_c")
    ids = [e.id for e in lib.entries(include_hidden=True)]
    assert len(ids) > 1000
    t = time.perf_counter(); lib.set_active_many(ids, True); dt_on = time.perf_counter() - t
    t = time.perf_counter(); lib.set_active_many(ids, False); dt_off = time.perf_counter() - t
    t = time.perf_counter(); lib.active_counts(); dt_c = time.perf_counter() - t
    print(f"bulk on {dt_on*1000:.0f} ms, off {dt_off*1000:.0f} ms, counts {dt_c*1000:.1f} ms for {len(ids)} signs")
    assert dt_on < 0.5 and dt_off < 0.5 and dt_c < 0.1


def test_counts_exclude_hidden_and_filter_by_category():
    lib, _ = make_lib("test_store_bulk_d")
    cats = [c for c in lib.categories() if c]
    assert cats
    n_on, n = lib.active_counts()
    assert n == len(lib.entries()) and n_on == 0
    cat = cats[0]; ids = [e.id for e in lib.entries(category=cat)]
    assert lib.active_counts(cat) == (0, len(ids))
    lib.set_active_many(ids, True); assert lib.active_counts(cat) == (len(ids), len(ids))
    lib.hide(ids[0])
    assert lib.active_counts(cat) == (len(ids) - 1, len(ids) - 1)           # hidden sign is not counted
    assert lib.active_counts()[1] == n - 1


def test_restore_active_gives_back_the_exact_previous_mixed_state():
    lib, store = make_lib("test_store_bulk_e")
    ids = [e.id for e in lib.entries()][:40]
    lib.set_active(ids[0], True); lib.set_active(ids[1], False); lib.set_active(ids[2], True)
    before = {i: (lib._active.get(i), lib.is_active(i)) for i in ids}
    n0 = len(store.read_audit())
    prev = lib.set_active_many(ids, False)
    assert not any(lib.is_active(i) for i in ids)
    lib.restore_active(prev)
    assert {i: (lib._active.get(i), lib.is_active(i)) for i in ids} == before
    assert audit_actions(store)[n0:] == ["blob_save", "library_set_active_many", "blob_save", "library_restore_active"]
    assert [t["id"] for t in lib.templates()] == [t["id"] for t in SignLibrary(store=store, model_classes=[]).templates()]
