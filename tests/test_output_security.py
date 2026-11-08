import json
import numpy as np
import pytest

from lingestra.output.chat_sender import ChatSender
from lingestra.security.store import SecureStore, StoreError


def lm(n=48):
    return np.random.rand(n, 67, 3).astype(np.float32)


@pytest.fixture
def store(tmp_path):
    return SecureStore(tmp_path / "s", passphrase="pw")


def test_roundtrip_and_encrypted_on_disk(tmp_path):
    s = SecureStore(tmp_path / "s", passphrase="pw")
    a = lm()
    i = s.save_unknown(a, {"note": "x"})
    arr, meta = s.load_unknown(i)
    assert np.array_equal(arr, a) and meta["note"] == "x"
    raw = (s.unknown_dir / f"{i}.enc").read_bytes()
    assert b"note" not in raw and b"PK" not in raw[:4]
    assert [m["id"] for m in s.list_unknown()] == [i]


def test_wrong_passphrase(tmp_path):
    SecureStore(tmp_path / "s", passphrase="right")
    with pytest.raises(StoreError):
        SecureStore(tmp_path / "s", passphrase="wrong")


def test_keyfile_fallback_persists(tmp_path):
    s = SecureStore(tmp_path / "s")
    i = s.save_unknown(lm(), {})
    s2 = SecureStore(tmp_path / "s")
    assert s2.load_unknown(i)[0].shape == (48, 67, 3)


@pytest.mark.parametrize("bad", [
    np.zeros((480, 640, 3), np.uint8),
    np.zeros((480, 640, 3), np.float32),
    np.zeros((48, 67, 3), np.uint8),
    np.zeros((10,), np.float32),
])
def test_rejects_image_like(store, bad):
    with pytest.raises(StoreError):
        store.save_unknown(bad, {})


@pytest.mark.parametrize("bad", ["../x", "a/b", "..", "", "a\\b", "x" * 100, None])
def test_traversal_ids_rejected(store, bad):
    with pytest.raises(StoreError):
        store.load_unknown(bad)
    with pytest.raises(StoreError):
        store.delete_unknown(bad)


def test_audit_chain_and_tamper(store):
    i = store.save_unknown(lm(), {})
    store.label_unknown(i, "hello", "alice")
    store.delete_unknown(i, "bob")
    assert store.verify_audit()
    lines = store.audit_path.read_text().splitlines()
    rec = json.loads(lines[1])
    assert rec["actor"] == "alice"
    rec["actor"] = "mallory"
    lines[1] = json.dumps(rec, sort_keys=True)
    store.audit_path.write_text("\n".join(lines) + "\n")
    assert not store.verify_audit()


def test_audit_deleted_middle_record_detected(store):
    ids = [store.save_unknown(lm(), {}) for _ in range(3)]
    for i in ids:
        store.label_unknown(i, "w", "a")
    lines = store.audit_path.read_text().splitlines()
    del lines[1]
    store.audit_path.write_text("\n".join(lines) + "\n")
    assert not store.verify_audit()


def test_label_persists(store):
    i = store.save_unknown(lm(), {})
    store.label_unknown(i, "thanks", "me")
    assert store.load_unknown(i)[1]["label"] == "thanks"


# ---------------- ChatSender with fakes
class FakeClip:
    def __init__(self, ok=True):
        self.ok, self.text = ok, None

    def copy(self, t):
        self.text = t
        return self.ok


class FakeKb:
    def __init__(self, fail=False):
        self.fail, self.calls = fail, []

    def paste(self, enter):
        if self.fail:
            raise RuntimeError("no injection")
        self.calls.append(enter)


def test_clipboard_paste():
    c, k, slept = FakeClip(), FakeKb(), []
    r = ChatSender(clipboard=c, keyboard=k, countdown_s=3, sleep=slept.append).send("hello world")
    assert r.ok and r.method == "clipboard+paste" and c.text == "hello world"
    assert k.calls == [True] and abs(sum(slept) - 3) < 1e-6


def test_clipboard_only_fallback():
    r = ChatSender(clipboard=FakeClip(), keyboard=FakeKb(fail=True)).send("hi")
    assert r.ok and r.method == "clipboard-only" and "Ctrl/Cmd+V" in r.detail


def test_no_clipboard_fails_cleanly():
    r = ChatSender(clipboard=FakeClip(ok=False), keyboard=FakeKb()).send("hi")
    assert not r.ok


def test_empty_text():
    assert not ChatSender(clipboard=FakeClip(), keyboard=FakeKb()).send("  ").ok


class FakePW:
    def __init__(self, results):
        self.results, self.n = results, 0

    def send(self, text, enter):
        r = self.results[self.n]
        self.n += 1
        if isinstance(r, Exception):
            raise r
        return r


def test_playwright_retry_then_ok():
    pw = FakePW([False, True])
    r = ChatSender(mode="playwright", clipboard=FakeClip(), keyboard=FakeKb(), playwright_driver=pw).send("x")
    assert r.method == "playwright" and pw.n == 2


def test_playwright_fails_falls_back_to_clipboard():
    pw = FakePW([RuntimeError("a"), False])
    c = FakeClip()
    r = ChatSender(mode="playwright", clipboard=c, keyboard=FakeKb(), playwright_driver=pw).send("x")
    assert r.method == "clipboard+paste" and c.text == "x"


def test_bad_mode():
    with pytest.raises(ValueError):
        ChatSender(mode="bot")


@pytest.mark.skipif(not __import__("sys").platform.startswith("linux"), reason="Wayland simulation is Linux-only")
def test_default_keyboard_degrades_headless(monkeypatch):
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.setenv("WAYLAND_DISPLAY", "w")
    r = ChatSender(clipboard=FakeClip()).send("hi")
    assert r.ok and r.method == "clipboard-only"


def test_countdown_is_cancellable_before_paste():
    c, k, slept = FakeClip(), FakeKb(), []
    s = ChatSender(clipboard=c, keyboard=k, countdown_s=3, sleep=slept.append)
    s.should_abort = lambda: len(slept) >= 2           # user pressed Pause / Cancel during the countdown
    r = s.send("hello")
    assert not r.ok and r.method == "cancelled" and k.calls == []
