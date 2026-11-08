import json
import shutil
import time
from pathlib import Path

import numpy as np
import pytest

from lingestra import features as F
from lingestra.library import SignLibrary, LibraryError, norm_text, prepare_animation
from lingestra.security.store import SecureStore, StoreError

SCRATCH = Path(__file__).resolve().parents[1] / "scratch"
FAMILY = ["Mother", "Father", "Naphew", "Family", "none"]


@pytest.fixture
def sdir():
    d = SCRATCH / "test_store_library"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


def mkstore(d):
    return SecureStore(d, passphrase="test-pass")


def window(n=40, seed=0, nan_gap=None):
    rng = np.random.default_rng(seed)
    w = np.zeros((n, F.N_PTS, 3), np.float32)
    w[:] = rng.uniform(0.4, 0.6, (n, F.N_PTS, 3)) * 0.01
    w[:, F.L_SH] = [0.6, 0.4, 0]; w[:, F.R_SH] = [0.4, 0.4, 0]
    w[:, F.L_HIP] = [0.58, 0.8, 0]; w[:, F.R_HIP] = [0.42, 0.8, 0]
    w[:, F.L_WR] = [0.7, 0.3, 0]; w[:, F.R_WR] = [0.3, 0.3, 0]
    w[:, F.N_POSE:] = np.nan
    w[:, F.N_POSE + 21:] = 0.3 + rng.uniform(0, 0.01, (n, 21, 3))
    if nan_gap:
        w[nan_gap[0]:nan_gap[1], F.N_POSE + 21:] = np.nan
    return w


def test_construction_fast():
    t = time.time()
    lib = SignLibrary()
    assert time.time() - t < 1.0
    assert len(lib.entries()) == 1244


def test_search_ranking_and_typo_and_arabic():
    lib = SignLibrary(model_classes=FAMILY)
    r = lib.search("mother")
    assert r[0].label.lower() == "mother"
    assert lib.search("MOTHER")[0].label == r[0].label
    typo = lib.search("Nephew")
    assert any(e.label == "Naphew" for e in typo)
    lib.update(r[0].id, aliases=["أُمّ", "ماما"])
    assert lib.search("ام")[0].id == r[0].id or any(e.id == r[0].id for e in lib.search("ام"))
    assert lib.search("اُم")[0].id == r[0].id
    assert norm_text("آمَنة") == norm_text("امنه")
    assert len(lib.search("")) == 200
    assert all(e.in_model for e in lib.search("", only_model=True))


def test_phrase_resolution():
    lib = SignLibrary(model_classes=FAMILY)
    ents, missing = lib.resolve_phrase("mother and father zzzz")
    assert [e.label for e in ents] == ["Mother", "Father"]
    assert missing == ["and", "zzzz"]
    assert ents[0].in_model
    # multi-word alias wins over single words
    mid = lib.search("mother")[0].id
    lib.update(mid, aliases=["how are you"])
    ents, missing = lib.resolve_phrase("how are you")
    assert [e.id for e in ents] == [mid] and missing == []


def test_edit_reset_persist_and_audit(sdir):
    lib = SignLibrary(mkstore(sdir), FAMILY)
    e = lib.search("mother")[0]
    u = lib.update(e.id, label="Mum", notes="line1\nline2", aliases=["mom"], category="Family2")
    assert u.label == "Mum" and u.edited and u.default_label == e.default_label
    lib2 = SignLibrary(mkstore(sdir), FAMILY)
    g = lib2.get(e.id)
    assert (g.label, g.notes, g.aliases, g.category) == ("Mum", "line1\nline2", ["mom"], "Family2")
    assert lib2.label_map() == {"Mother": "Mum"}
    assert lib2.display_name("Mother") == "Mum" and lib2.display_name("Father") == "Father"
    lib2.reset(e.id)
    assert not lib2.get(e.id).edited and lib2.get(e.id).label == e.default_label
    assert lib2.label_map() == {}
    st = mkstore(sdir)
    log = st.read_audit()
    assert any(r["action"] == "library_update" and r["id"] == e.id and r["actor"] == "signer-admin" for r in log)
    assert any(r["action"] == "library_reset" for r in log)
    assert "Mum" not in (sdir / "audit.log").read_text()
    assert st.verify_audit()
    # data on disk is encrypted
    assert all(b"Mum" not in p.read_bytes() for p in sdir.rglob("*.enc"))
    # dictionary files untouched / manifest still original
    assert lib2.get(e.id).default_label == e.default_label


def test_validation():
    lib = SignLibrary()
    i = lib.search("mother")[0].id
    for bad in ("", "a\x00b", "x" * 81, "../etc", "a\nb", "a\\b"):
        with pytest.raises(LibraryError):
            lib.update(i, label=bad)
    with pytest.raises(LibraryError):
        lib.update(i, aliases=["ok", "bad\x07"])
    with pytest.raises(LibraryError):
        lib.update(i, notes="x" * 501)
    assert "<b>x</b>" == lib.update(i, notes="<b>x</b>").notes  # stored verbatim, UI must show as plain text


def test_hidden():
    lib = SignLibrary(model_classes=FAMILY)
    e = lib.search("mother")[0]
    n = len(lib.entries())
    lib.hide(e.id)
    assert len(lib.entries()) == n - 1 and len(lib.entries(include_hidden=True)) == n
    assert e.id not in [x.id for x in lib.search("mother")]
    assert e.id not in [x.id for x in lib.resolve_phrase("mother")[0]]
    lib.unhide(e.id)
    assert len(lib.entries()) == n


def test_recorded_lifecycle(sdir):
    lib = SignLibrary(mkstore(sdir), FAMILY)
    uid = lib.add_recorded("Hello friend", window(), notes="mine", aliases=["مرحبا"])
    assert uid.startswith("u_")
    e = lib.get(uid)
    assert e.source == "recorded" and e.category == "My signs" and e.takes == 1
    assert "My signs" in lib.categories()
    lib2 = SignLibrary(mkstore(sdir), FAMILY)
    assert lib2.get(uid).label == "Hello friend"
    assert lib2.search("hello")[0].id == uid and lib2.search("مرحبا")[0].id == uid
    assert lib2.animation(uid).shape[1:] == (F.N_PTS, 3)
    assert lib2.add_take(uid, window(seed=1)) == 1
    assert lib2.get(uid).takes == 2
    assert lib2.animation(uid, 1).shape[1:] == (F.N_PTS, 3)
    with pytest.raises(LibraryError):
        lib2.add_recorded("bad", np.zeros((40, 10, 3), np.float32))
    with pytest.raises(LibraryError):
        lib2.add_recorded("bad", np.zeros((40, F.N_PTS, 3), np.uint8))
    lib2.delete_recorded(uid)
    with pytest.raises(KeyError):
        lib2.get(uid)
    assert SignLibrary(mkstore(sdir), FAMILY).search("hello friend") == [] or \
        SignLibrary(mkstore(sdir), FAMILY).search("hello friend")[0].label != "Hello friend"
    assert mkstore(sdir).list_blobs("lib_takes") == []
    acts = [r["action"] for r in mkstore(sdir).read_audit()]
    assert "library_add_recorded" in acts and "library_delete_recorded" in acts


def test_replace_animation_and_takes(sdir):
    lib = SignLibrary(mkstore(sdir), FAMILY)
    e = lib.search("father")[0]
    a0 = lib.animation(e.id)
    lib.replace_animation(e.id, window())
    assert lib.get(e.id).edited and lib.get(e.id).takes == 1
    a1 = lib.animation(e.id)
    assert a1.shape != a0.shape or not np.allclose(a0, a1, equal_nan=True)
    assert lib.add_take(e.id, window(seed=2)) == 1
    assert lib.get(e.id).takes == 2
    lib.reset(e.id)
    assert np.allclose(lib.animation(e.id), a0, equal_nan=True)


def test_animation_shape_and_nan():
    lib = SignLibrary()
    e = lib.search("mother")[0]
    a = lib.animation(e.id)
    assert a.ndim == 3 and a.shape[1:] == (F.N_PTS, 3) and a.dtype == np.float32
    assert lib.animation(e.id) is a  # cached
    t = lib.thumbnail_frame(e.id)
    assert t.shape == (F.N_PTS, 3)
    # raw with a short gap (3 frames = 0.12 s) gets bridged, long gap (8 frames) stays NaN
    w = window(60)
    w[10:13, F.N_POSE + 21:] = np.nan
    w[30:38, F.N_POSE + 21:] = np.nan
    a = prepare_animation(w)
    hand = np.isnan(a[:, F.N_POSE + 21, 0])
    assert not hand[11] and hand[34]
    assert not np.isnan(a[:, F.L_SH]).any() and np.allclose(a[:, F.L_SH, 0] - a[:, F.R_SH, 0], 1.0, atol=1e-4)
    # fully empty hands: all NaN, no crash
    w2 = window(30); w2[:, F.N_POSE:] = np.nan
    assert np.isnan(prepare_animation(w2)[:, F.N_POSE:]).all()
    assert prepare_animation(np.zeros((0, F.N_PTS, 3))).shape == (0, F.N_PTS, 3)
    # fps resampling
    assert len(prepare_animation(window(50), fps=50)) <= 30


def test_label_map_only_edited_model():
    lib = SignLibrary(model_classes=FAMILY)
    m = lib.search("mother")[0]
    o = lib.search("zebra")[0]
    lib.update(o.id, label="Zeb")
    assert lib.label_map() == {}
    lib.update(m.id, label="Mum")
    assert lib.label_map() == {"Mother": "Mum"}


def test_export(sdir, tmp_path):
    lib = SignLibrary(mkstore(sdir), FAMILY)
    uid = lib.add_recorded("Mine", window())
    lib.add_take(uid, window(seed=3))
    e = lib.search("father")[0]
    lib.replace_animation(e.id, window(seed=4))
    out = SCRATCH / "test_export_library"
    shutil.rmtree(out, ignore_errors=True)
    try:
        assert lib.export_user_signs(out) == 3
        meta = json.loads((out / "labels.json").read_text(encoding="utf-8"))
        assert len(meta) == 3 and all((out / k).exists() for k in meta)
        assert {m["label"] for m in meta.values()} == {"Mine", "Father"}
        assert np.load(out / f"{uid}_t0.npy").shape == (40, F.N_PTS, 3)
    finally:
        shutil.rmtree(out, ignore_errors=True)


def test_store_blobs(sdir):
    st = mkstore(sdir)
    st.save_blob("ns", "k1", {"a": [1, "é"]}, actor="me")
    st.save_blob("ns", "k2", np.zeros((9, F.N_PTS, 3), np.float32))
    assert st.load_blob("ns", "k1") == {"a": [1, "é"]}
    assert st.load_blob("ns", "k2").shape == (9, F.N_PTS, 3)
    assert st.list_blobs("ns") == ["k1", "k2"] and st.list_blobs("none") == []
    for bad in ("../x", "a/b", "", "x" * 65):
        with pytest.raises(StoreError):
            st.save_blob("ns", bad, {})
        with pytest.raises(StoreError):
            st.save_blob(bad, "k", {})
    with pytest.raises(StoreError):
        st.save_blob("ns", "img", np.zeros((480, 640, 3), np.float32))
    st.delete_blob("ns", "k1", actor="me")
    with pytest.raises(StoreError):
        st.load_blob("ns", "k1")
    log = st.read_audit()
    assert [r["action"] for r in log] == ["blob_save", "blob_save", "blob_delete"]
    assert st.verify_audit()
    assert oct((sdir / "blobs" / "ns" / "k2.enc").stat().st_mode & 0o777) == "0o600"
    with pytest.raises(StoreError):
        SecureStore(sdir, passphrase="wrong")


def test_review_fixes_text_and_search(sdir):
    lib = SignLibrary(mkstore(sdir), FAMILY)
    e = lib.search("mother")[0]
    with pytest.raises(LibraryError):
        lib.update(e.id, label="a b")
    lib.update(e.id, label="zz‌mom")                 # ZWNJ allowed
    assert any(x.id == e.id for x in lib.search(e.default_label))
    found, missing = lib.resolve_phrase(e.default_label)
    assert found and not missing


# ---------------------------------------------------------------- personal recognition templates
def test_templates_active_default_and_toggle_persist(sdir):
    st = mkstore(sdir)
    lib = SignLibrary(store=st)
    mother = lib.search("mother")[0]
    assert not lib.is_active(mother.id)                              # dictionary signs start off
    assert lib.templates() == []
    rid = lib.add_recorded("My Hello", window(40))
    assert lib.is_active(rid)                                        # own signs start on
    t = lib.templates()
    assert [x["id"] for x in t] == [rid] and t[0]["label"] == "My Hello"
    assert len(t[0]["frames"]) == 1 and t[0]["frames"][0].shape == (40, F.N_PTS, 3)
    lib.add_take(rid, window(30, seed=1))
    assert len(lib.templates()[0]["frames"]) == 2                    # all takes
    # a user take on a dictionary sign switches it on; its dictionary clip stays take 0
    n = lib.add_take(mother.id, window(35, seed=2))
    assert lib.is_active(mother.id) and n >= 1
    mt = [x for x in lib.templates() if x["id"] == mother.id][0]
    assert len(mt["frames"]) == 2
    lib.set_active(mother.id, False)
    lib.set_active(rid, False)
    assert lib.templates() == []
    fath = lib.search("father")[0]
    lib.set_active(fath.id, True)
    lib.update(fath.id, label="Dad")                                 # current output text is the label
    assert [(x["id"], x["label"]) for x in lib.templates()] == [(fath.id, "Dad")]
    assert len(lib.templates()[0]["frames"]) == 1 and lib.templates()[0]["frames"][0].ndim == 3
    lib.hide(fath.id)
    assert lib.is_active(fath.id) and lib.templates() == []          # hidden signs are not recognised
    lib.unhide(fath.id)
    # persisted (encrypted) and audited
    lib2 = SignLibrary(store=mkstore(sdir))
    assert lib2.is_active(fath.id) and not lib2.is_active(rid) and not lib2.is_active(mother.id)
    acts = [r["action"] for r in mkstore(sdir).read_audit()]
    assert acts.count("library_set_active") == 3
    assert mkstore(sdir).verify_audit()
    for f in sdir.rglob("*"):
        if f.is_file() and f.name != "audit.log":
            assert b"library_set_active" not in f.read_bytes()
    with pytest.raises(KeyError):
        lib.set_active("nope", True)
    assert lib.is_active("nope") is False
    lib.delete_recorded(rid)
    assert not lib.is_active(rid)


def test_animation_excludes_raise_and_lowering():
    lib = SignLibrary()
    sid = lib.search("mother")[0].id
    raw, fps = lib._raw(sid)
    anim = lib.animation(sid)
    assert 8 <= len(anim) < len(raw) * 0.8                           # idle + raise/lower dropped
    # raw data stays intact
    assert lib._raw(sid)[0].shape == raw.shape
