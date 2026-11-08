"""Encrypted-at-rest local storage for unknown-sign landmark windows.

Stores ONLY landmark arrays (float, frames x points x coords) and small
metadata. Image-like arrays are rejected. Key sources, in order:
  1. passphrase (scrypt, salt kept beside the data)
  2. OS keyring (if a backend works)
  3. 0600 keyfile in the store directory
The audit log is a hash chain: each record holds SHA-256 of the previous one.
Limitation: it is tamper-EVIDENT for edits/reordering/mid-deletion; truncating
the tail is not detectable without an external anchor.

Generic blobs (used by lingestra.library for sign-library edits and recorded signs):
  save_blob(namespace, key, obj, actor=None)  obj = JSON-able value OR a floating ndarray of
  landmarks (same image-like rejection as unknown windows). Encrypted with the same Fernet key,
  written atomically with 0600 perms under <store>/blobs/<namespace>/<key>.enc.
  load_blob(namespace, key) / list_blobs(namespace) / delete_blob(namespace, key, actor=None).
  Namespaces and keys must match [A-Za-z0-9_-]{1,64}. Every save/delete appends an audit record
  (action blob_save / blob_delete, id "<namespace>/<key>", actor); content is never logged.
  audit(action, id, actor, extra) lets callers add their own records (extra must not hold content).
"""
from __future__ import annotations

import base64
import hashlib
import io
import threading
import json
import os
import re
import sys
import tempfile
import time
import uuid
from pathlib import Path

import numpy as np
from cryptography.fernet import Fernet, InvalidToken
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt

_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_GENESIS = "0" * 64
_CHECK_PLAINTEXT = b"lingestra-key-check-v1"
MAX_ARRAY_BYTES = 20 * 1024 * 1024
MAX_POINTS = 600  # MediaPipe holistic has 543; a wider 2nd axis smells like an image


_AUDIT_LOCK = threading.Lock()


class StoreError(Exception):
    pass


def _default_dir() -> Path:
    from lingestra.output.chat_sender import app_data_dir
    return app_data_dir() / "secure_store"


def _chmod(path: Path, mode: int):
    try:
        os.chmod(path, mode)
    except OSError:
        pass  # e.g. Windows: ACLs apply instead


def _atomic_write(path: Path, data: bytes, mode: int = 0o600):
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        _chmod(Path(tmp), mode)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _check_landmarks(a) -> np.ndarray:
    a = np.asarray(a)
    if not np.issubdtype(a.dtype, np.floating):
        raise StoreError(f"landmark arrays must be floating point, got {a.dtype}")
    if a.ndim not in (2, 3):
        raise StoreError(f"unexpected landmark array ndim {a.ndim}")
    if a.ndim == 2 and a.shape[1] > MAX_POINTS * 4:
        raise StoreError(f"image-like shape {a.shape} rejected")
    if a.ndim == 3 and a.shape[2] not in (2, 3, 4):
        raise StoreError(f"image-like shape {a.shape} rejected")
    if a.ndim == 3 and a.shape[1] > MAX_POINTS:
        raise StoreError(f"image-like shape {a.shape} rejected")
    if a.nbytes > MAX_ARRAY_BYTES:
        raise StoreError("array too large for a landmark window")
    return a


class SecureStore:
    def __init__(self, directory=None, passphrase: str | None = None):
        self.dir = Path(directory) if directory else _default_dir()
        self.dir.mkdir(parents=True, exist_ok=True)
        _chmod(self.dir, 0o700)
        self.unknown_dir = self.dir / "unknown"
        self.unknown_dir.mkdir(exist_ok=True)
        _chmod(self.unknown_dir, 0o700)
        self.audit_path = self.dir / "audit.log"
        self._fernet = Fernet(self._get_key(passphrase))
        self._verify_key()

    # ---- keys
    def _get_key(self, passphrase) -> bytes:
        if passphrase:
            salt_path = self.dir / "kdf_salt"
            if salt_path.exists():
                salt = salt_path.read_bytes()
            else:
                salt = os.urandom(16)
                _atomic_write(salt_path, salt)
            raw = Scrypt(salt=salt, length=32, n=2**15, r=8, p=1).derive(passphrase.encode("utf-8"))
            return base64.urlsafe_b64encode(raw)
        key = self._keyring_key()
        if key:
            return key
        keyfile = self.dir / "store.key"
        if keyfile.exists():
            return keyfile.read_bytes().strip()
        key = Fernet.generate_key()
        _atomic_write(keyfile, key)
        return key

    def _keyring_key(self):
        """Key kept in the OS keyring, if a real backend exists; else None."""
        try:
            import keyring
            from keyring.backends import fail
            if isinstance(keyring.get_keyring(), fail.Keyring):
                return None
            name = "store-" + hashlib.sha256(str(self.dir.resolve()).encode()).hexdigest()[:16]
            k = keyring.get_password("Lingestra", name)
            if not k:
                k = Fernet.generate_key().decode()
                keyring.set_password("Lingestra", name, k)
            return k.encode()
        except Exception:
            return None

    def _verify_key(self):
        """A check blob makes a wrong passphrase/key fail immediately."""
        check = self.dir / "key_check"
        if check.exists():
            try:
                ok = self._fernet.decrypt(check.read_bytes()) == _CHECK_PLAINTEXT
            except InvalidToken:
                ok = False
            if not ok:
                raise StoreError("wrong passphrase or key")
        else:
            _atomic_write(check, self._fernet.encrypt(_CHECK_PLAINTEXT))

    # ---- ids
    def _path(self, id_: str) -> Path:
        if not isinstance(id_, str) or not _ID_RE.match(id_):
            raise StoreError("invalid id")
        return self.unknown_dir / f"{id_}.enc"

    # ---- (de)serialisation
    def _write(self, id_: str, arr: np.ndarray, meta: dict):
        buf = io.BytesIO()
        np.savez(buf, arr=arr, meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8))
        _atomic_write(self._path(id_), self._fernet.encrypt(buf.getvalue()))

    def save_unknown(self, landmarks_window, meta: dict) -> str:
        arr = _check_landmarks(landmarks_window)
        meta = dict(meta or {})
        meta.setdefault("created", time.time())
        meta.setdefault("label", None)
        id_ = uuid.uuid4().hex
        self._write(id_, arr, meta)
        self._audit("save", id_, "system", {})
        return id_

    def load_unknown(self, id):
        p = self._path(id)
        if not p.exists():
            raise StoreError("not found")
        try:
            data = self._fernet.decrypt(p.read_bytes())
        except InvalidToken:
            raise StoreError("decryption failed (corrupt or wrong key)")
        with np.load(io.BytesIO(data), allow_pickle=False) as z:
            arr = z["arr"]
            meta = json.loads(z["meta"].tobytes().decode("utf-8"))
        return arr, meta

    def list_unknown(self) -> list[dict]:
        out = []
        for p in sorted(self.unknown_dir.glob("*.enc")):
            try:
                _, meta = self.load_unknown(p.stem)
                out.append({"id": p.stem, **meta})
            except StoreError:
                continue
        return out

    def label_unknown(self, id, label: str, actor: str):
        arr, meta = self.load_unknown(id)
        old = meta.get("label")
        meta["label"] = label
        self._write(id, arr, meta)
        self._audit("label", id, actor, {"old": old, "new": label})

    def delete_unknown(self, id, actor: str = "user"):
        p = self._path(id)
        if not p.exists():
            raise StoreError("not found")
        p.unlink()
        self._audit("delete", id, actor, {})

    # ---- generic encrypted blobs
    def _blob_dir(self, namespace: str, create=False) -> Path:
        if not isinstance(namespace, str) or not _ID_RE.match(namespace):
            raise StoreError("invalid namespace")
        d = self.dir / "blobs" / namespace
        if create:
            d.mkdir(parents=True, exist_ok=True)
            _chmod(d.parent, 0o700)
            _chmod(d, 0o700)
        return d

    def _blob_path(self, namespace: str, key: str, create=False) -> Path:
        if not isinstance(key, str) or not _ID_RE.match(key):
            raise StoreError("invalid id")
        return self._blob_dir(namespace, create) / f"{key}.enc"

    def save_blob(self, namespace: str, key: str, obj, actor: str | None = None):
        p = self._blob_path(namespace, key, create=True)
        if isinstance(obj, np.ndarray):
            arr = _check_landmarks(obj)
            buf = io.BytesIO()
            np.save(buf, arr, allow_pickle=False)
            payload = b"A" + buf.getvalue()
        else:
            try:
                payload = b"J" + json.dumps(obj, ensure_ascii=False).encode("utf-8")
            except (TypeError, ValueError):
                raise StoreError("blob is not JSON serialisable")
        _atomic_write(p, self._fernet.encrypt(payload))
        self._audit("blob_save", f"{namespace}/{key}", actor or "system", {})

    def load_blob(self, namespace: str, key: str):
        p = self._blob_path(namespace, key)
        if not p.exists():
            raise StoreError("not found")
        try:
            data = self._fernet.decrypt(p.read_bytes())
        except InvalidToken:
            raise StoreError("decryption failed (corrupt or wrong key)")
        kind, body = data[:1], data[1:]
        if kind == b"A":
            return np.load(io.BytesIO(body), allow_pickle=False)
        if kind == b"J":
            return json.loads(body.decode("utf-8"))
        raise StoreError("unknown blob format")

    def list_blobs(self, namespace: str) -> list[str]:
        d = self._blob_dir(namespace)
        return sorted(p.stem for p in d.glob("*.enc")) if d.exists() else []

    def delete_blob(self, namespace: str, key: str, actor: str | None = None):
        p = self._blob_path(namespace, key)
        if not p.exists():
            raise StoreError("not found")
        p.unlink()
        self._audit("blob_delete", f"{namespace}/{key}", actor or "system", {})

    def audit(self, action: str, id_: str, actor: str, extra: dict | None = None):
        self._audit(action, id_, actor, extra or {})

    def read_audit(self) -> list[dict]:
        if not self.audit_path.exists():
            return []
        return [json.loads(l) for l in self.audit_path.read_text().splitlines() if l.strip()]

    # ---- settings (small encrypted JSON)
    def save_settings(self, settings: dict):
        _atomic_write(self.dir / "settings.enc", self._fernet.encrypt(json.dumps(settings).encode()))

    def load_settings(self) -> dict:
        p = self.dir / "settings.enc"
        if not p.exists():
            return {}
        try:
            return json.loads(self._fernet.decrypt(p.read_bytes()))
        except InvalidToken:
            raise StoreError("settings decryption failed")

    # ---- audit log
    @staticmethod
    def _hash(body: dict) -> str:
        return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    def _last_hash(self) -> str:
        if not self.audit_path.exists():
            return _GENESIS
        lines = self.audit_path.read_text().splitlines()
        return json.loads(lines[-1])["hash"] if lines else _GENESIS

    def _audit(self, action: str, id_: str, actor: str, extra: dict):
        with _AUDIT_LOCK:
            self._audit_locked(action, id_, actor, extra)

    def _audit_locked(self, action: str, id_: str, actor: str, extra: dict):
        body = {"ts": time.time(), "action": action, "id": id_, "actor": actor,
                "extra": extra, "prev": self._last_hash()}
        rec = {**body, "hash": self._hash(body)}
        with open(self.audit_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        _chmod(self.audit_path, 0o600)

    def verify_audit(self) -> bool:
        if not self.audit_path.exists():
            return True
        prev = _GENESIS
        try:
            for line in self.audit_path.read_text().splitlines():
                rec = json.loads(line)
                h = rec.pop("hash")
                if rec.get("prev") != prev or self._hash(rec) != h:
                    return False
                prev = h
        except (ValueError, KeyError, TypeError):
            return False
        return True
