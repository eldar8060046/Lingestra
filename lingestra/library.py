"""Sign library: dictionary clips + user edits + user-recorded signs (Qt-free).

Dictionary files under data/ are NEVER modified. All user changes live in the encrypted
SecureStore (or in memory when no store is given):
  blobs 'library'/'overrides'  {id: {label, notes, aliases, category, hidden}}   (JSON)
  blobs 'library'/'recorded'   {id: {default_label, notes, aliases, category, created}}
  blobs 'library'/'fps'        {take key: source fps}
  blobs 'lib_takes'/'<id>_t<n>' raw landmark windows (frames,67,3); t0 replaces the dictionary
                               animation (or is the recording of a user sign), t1.. are extra takes.
Only landmarks are stored, never images. Ids are generated (dictionary ids or 'u_<hex>'), never
derived from labels. Text fields are validated (length, no control characters, no path separators)
and must be rendered as PLAIN text by the UI.
"""
from __future__ import annotations

import dataclasses
import difflib
import json
import os
import re
import secrets
import time
import unicodedata
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import features as F
from . import labels as L

OUT_FPS = 25.0
AUTO_MIN_S, AUTO_MIN_HANDS = 0.4, 0.6      # least movement / share of frames with hands for an automatic example
MAX_GAP_S = 0.15
MAX_LABEL, MAX_NOTES, MAX_ALIAS, MAX_ALIASES, MAX_CATEGORY = 80, 500, 80, 20, 40
MIN_FRAMES, MAX_FRAMES = 8, 3000
MAX_AUTO_TAKES = 6          # examples learned from corrections, per sign (oldest dropped first)
USER_CATEGORY = "My signs"
_NS, _TAKES = "library", "lib_takes"
_AR_STRIP = re.compile("[ً-ٰٟـۖ-ۭ]")
_AR_MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ی": "ي", "ئ": "ي",
                         "ة": "ه", "ؤ": "و", "ک": "ك"})


class LibraryError(ValueError):
    pass


@dataclass
class SignEntry:
    id: str
    label: str
    default_label: str
    notes: str = ""
    aliases: list = field(default_factory=list)
    category: str = ""
    source: str = "dictionary"
    in_model: bool = False
    takes: int = 1
    edited: bool = False
    hidden: bool = False
    model_class: str | None = None
    auto: int = 0                 # examples learned automatically from corrections (part of `takes`)


# ---------------------------------------------------------------- text helpers
def _fix_mojibake(s: str) -> str:
    if any(0x80 <= ord(c) <= 0xff for c in s):
        for enc in ("latin-1", "cp1252"):
            try:
                return s.encode(enc).decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
    return s


def norm_text(s: str) -> str:
    """Case-insensitive, Arabic-tolerant normal form used for matching."""
    s = unicodedata.normalize("NFKC", s or "").lower()
    s = _AR_STRIP.sub("", s).translate(_AR_MAP)
    return re.sub(r"\s+", " ", s).strip()


def _words(s: str) -> list[str]:
    return [w for w in re.split(r"[^\w]+", norm_text(s)) if w]


def _clean_text(v, name, maxlen, multiline=False, allow_empty=False) -> str:
    if not isinstance(v, str):
        raise LibraryError(f"{name} must be text")
    v = unicodedata.normalize("NFC", v)
    v = "".join(" " if unicodedata.category(c) == "Zs" else c for c in v).strip()
    for ch in v:
        cat = unicodedata.category(ch)
        if ch in "/\\" and not multiline:
            raise LibraryError(f"{name} may not contain path separators")
        if cat in ("Zl", "Zp"):
            raise LibraryError(f"{name} contains control characters")
        if cat == "Cf" and ch not in "\u200c\u200d\u200e\u200f\u061c":
            raise LibraryError(f"{name} contains control characters")
        if cat in ("Cc", "Cs", "Co", "Cn") and not (multiline and ch == "\n"):
            raise LibraryError(f"{name} contains control characters")
    if not multiline:
        v = re.sub(" {2,}", " ", v)
    if len(v) > maxlen:
        raise LibraryError(f"{name} too long (max {maxlen})")
    if not v and not allow_empty:
        raise LibraryError(f"{name} may not be empty")
    return v


def _clean_aliases(al) -> list[str]:
    if isinstance(al, str):
        al = [al]
    out, seen = [], set()
    for a in al:
        a = _clean_text(a, "alias", MAX_ALIAS, allow_empty=True)
        if a and norm_text(a) not in seen:
            seen.add(norm_text(a))
            out.append(a)
    if len(out) > MAX_ALIASES:
        raise LibraryError(f"too many aliases (max {MAX_ALIASES})")
    return out


def _check_window(w) -> np.ndarray:
    a = np.asarray(w)
    if not np.issubdtype(a.dtype, np.floating):
        raise LibraryError("landmark window must be floating point")
    if a.ndim != 3 or a.shape[1:] != (F.N_PTS, 3):
        raise LibraryError(f"expected (frames,{F.N_PTS},3) landmarks, got {a.shape}")
    if not (MIN_FRAMES <= a.shape[0] <= MAX_FRAMES):
        raise LibraryError(f"window must have {MIN_FRAMES}..{MAX_FRAMES} frames")
    return a.astype(np.float32)


# ---------------------------------------------------------------- in-memory store stand-in
class _MemStore:
    def __init__(self):
        self.d, self.log = {}, []

    def save_blob(self, ns, key, obj, actor=None):
        self.d[(ns, key)] = obj.copy() if isinstance(obj, np.ndarray) else json.loads(json.dumps(obj))
        self.log.append({"action": "blob_save", "id": f"{ns}/{key}", "actor": actor or "system", "extra": {}})

    def load_blob(self, ns, key):
        if (ns, key) not in self.d:
            raise KeyError(key)
        v = self.d[(ns, key)]
        return v.copy() if isinstance(v, np.ndarray) else json.loads(json.dumps(v))

    def list_blobs(self, ns):
        return sorted(k for (n, k) in self.d if n == ns)

    def delete_blob(self, ns, key, actor=None):
        self.d.pop((ns, key), None)
        self.log.append({"action": "blob_delete", "id": f"{ns}/{key}", "actor": actor or "system", "extra": {}})

    def audit(self, action, id_, actor, extra=None):
        self.log.append({"action": action, "id": id_, "actor": actor, "extra": extra or {}})

    def read_audit(self):
        return list(self.log)


# ---------------------------------------------------------------- animation preparation
def _fill_ref(arr):
    valid = ~np.isnan(arr.reshape(len(arr), -1)).any(axis=1)
    if not valid.any():
        return None
    idx = np.where(valid, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    idx[:np.argmax(valid)] = np.argmax(valid)
    return arr[idx]


def _bridge(a, max_gap):
    """Linear interpolation across NaN runs <= max_gap frames (per landmark, per frame)."""
    n = len(a)
    for j in range(a.shape[1]):
        miss = np.isnan(a[:, j, 0])
        if not miss.any() or miss.all():
            continue
        i = 0
        while i < n:
            if miss[i]:
                k = i
                while k < n and miss[k]:
                    k += 1
                if i > 0 and k < n and k - i <= max_gap:
                    t = (np.arange(i, k) - (i - 1)) / (k - (i - 1))
                    a[i:k, j] = a[i - 1, j] * (1 - t)[:, None] + a[k, j] * t[:, None]
                i = k
            else:
                i += 1
    return a


def _smooth(a):
    """NaN-aware [1 2 1]/4 filter; NaN frames stay NaN."""
    if len(a) < 3:
        return a
    w = np.array([0.25, 0.5, 0.25], np.float32).reshape(3, 1, 1, 1)
    pad = np.concatenate([a[:1], a, a[-1:]])
    stack = np.stack([pad[:-2], pad[1:-1], pad[2:]])
    ok = ~np.isnan(stack)
    num = (np.where(ok, stack, 0) * w).sum(0)
    den = (ok * w).sum(0)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    out[np.isnan(a)] = np.nan
    return out.astype(np.float32)


def prepare_animation(raw, fps=OUT_FPS) -> np.ndarray:
    """Raw landmarks -> drawing-ready (frames,67,3): trimmed, shoulder-normalised (NaN kept),
    short gaps bridged, smoothed, resampled to OUT_FPS."""
    raw = np.asarray(raw, np.float32)
    if raw.ndim != 3 or len(raw) == 0:
        return np.zeros((0, F.N_PTS, 3), np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        seq = F.trim_active(raw, fps=fps).copy()
        c = (seq[:, F.L_SH] + seq[:, F.R_SH]) / 2
        s = np.linalg.norm(seq[:, F.L_SH, :2] - seq[:, F.R_SH, :2], axis=1)[:, None]
        c, s = _fill_ref(c), _fill_ref(s)
        if c is None or s is None:                     # shoulders never seen: best-effort fallback
            c = np.broadcast_to(np.array([0.5, 0.5, 0.0], np.float32), (len(seq), 3))
            s = np.full((len(seq), 1), 0.25, np.float32)
        s = np.maximum(s, 1e-3)[:, None, :]
        out = ((seq - c[:, None, :]) / s).astype(np.float32)
        out = _bridge(out, max(int(round(MAX_GAP_S * fps)), 1))
        out = _smooth(out)
        n_out = max(int(round(len(out) * OUT_FPS / fps)), 1)
        if n_out != len(out) and len(out) > 1:
            src = np.linspace(0, len(out) - 1, n_out)
            lo = np.floor(src).astype(int)
            hi = np.minimum(lo + 1, len(out) - 1)
            w = (src - lo)[:, None, None].astype(np.float32)
            out = (out[lo] * (1 - w) + out[hi] * w).astype(np.float32)
    return out


# ---------------------------------------------------------------- library
class SignLibrary:
    def __init__(self, store=None, model_classes=None, data_root=None):
        self.store = store if store is not None else _MemStore()
        self.persistent = store is not None
        self.root = Path(data_root) if data_root else F.ROOT / "data"
        self._manifest_path = self.root / "manifest.json"
        self._lm_dir = self.root / "landmarks"
        self._anim = OrderedDict()
        self._quality = {}
        self.model_classes = list(model_classes or [])
        self._class_by_norm = {norm_text(c): c for c in self.model_classes if c != "none"}
        self._overrides = self._load("overrides")
        self._recorded = self._load("recorded")
        self._fps = self._load("fps")
        self._active = self._load("active")          # id -> bool, only explicit choices
        self._auto = self._load("auto")              # id -> {take number: unix time} of auto-learned examples
        self._take_keys = set(self.store.list_blobs(_TAKES))
        self._base = {}
        if self._manifest_path.exists():
            man = json.loads(self._manifest_path.read_text(encoding="utf-8"))
            ov = L._overrides() if self.root == F.ROOT / "data" else {}
            for e in man:
                eid = str(e["id"])
                e = dict(e, label=_fix_mojibake(e["label"]))
                lab = L.clean_label(ov[eid]) if eid in ov else L.canonical_label(e) if eid in L.MERGE_TO else L.clean_label(e["label"])
                self._base[eid] = dict(default_label=lab, category=e["category"], notes="", aliases=[],
                                       variant=e.get("variant", 1))
        self._entries = {}
        self._order = []
        self._rebuild_all()

    # ---- persistence helpers
    def _load(self, key):
        try:
            v = self.store.load_blob(_NS, key)
            return v if isinstance(v, dict) else {}
        except Exception:
            return {}

    def _save(self, key, obj, actor):
        self.store.save_blob(_NS, key, obj, actor=actor)

    def _audit(self, action, id_, actor, fields=()):
        self.store.audit(action, id_, actor, {"fields": sorted(fields)} if fields else {})

    # ---- entry construction
    def _n_takes(self, id_):
        extra = sum(1 for k in self._take_keys if k.startswith(id_ + "_t") and k[len(id_) + 2:].isdigit())
        if id_ in self._recorded:
            return max(extra, 1)
        return 1 + max(extra - (1 if f"{id_}_t0" in self._take_keys else 0), 0)

    def _make(self, id_) -> SignEntry:
        rec = self._recorded.get(id_)
        b = rec if rec is not None else self._base[id_]
        ov = self._overrides.get(id_, {})
        default = b["default_label"]
        label = ov.get("label", default)
        mc = self._class_by_norm.get(norm_text(default))
        return SignEntry(
            id=id_, label=label, default_label=default,
            notes=ov.get("notes", b.get("notes", "")),
            aliases=list(ov.get("aliases", b.get("aliases", []))),
            category=ov.get("category", b["category"]),
            source="recorded" if rec is not None else "dictionary",
            in_model=mc is not None, takes=self._n_takes(id_),
            edited=bool(ov) or (rec is None and f"{id_}_t0" in self._take_keys),
            hidden=bool(ov.get("hidden", False)), model_class=mc,
            auto=sum(1 for n in self._auto.get(id_, {}) if f"{id_}_t{n}" in self._take_keys))

    def _refresh(self, id_):
        if id_ in self._base or id_ in self._recorded:
            self._entries[id_] = self._make(id_)
        else:
            self._entries.pop(id_, None)
        self._order = [i for i in self._order if i in self._entries] if id_ not in self._entries else (
            self._order if id_ in self._order else [id_] + self._order)
        self._index = None
        self._anim = OrderedDict((k, v) for k, v in self._anim.items() if k[0] != id_)

    def _rebuild_all(self):
        self._entries = {i: self._make(i) for i in list(self._recorded) + list(self._base)}
        self._order = list(self._entries)
        self._index = None

    def _cp(self, e):
        return dataclasses.replace(e, aliases=list(e.aliases))

    # ---- queries
    def get(self, id) -> SignEntry:
        if id not in self._entries:
            raise KeyError(id)
        return self._cp(self._entries[id])

    def entries(self, include_hidden=False, category=None, only_model=False):
        out = []
        for i in self._order:
            e = self._entries[i]
            if (e.hidden and not include_hidden) or (category and e.category != category) \
                    or (only_model and not e.in_model):
                continue
            out.append(self._cp(e))
        return out

    def categories(self):
        return sorted({e.category for e in self._entries.values() if not e.hidden}, key=str.lower)

    def _search_index(self):
        if self._index is None:
            self._index = {i: (norm_text(e.label), [norm_text(a) for a in e.aliases + ([e.default_label] if e.default_label else [])],
                           norm_text(e.notes))
                           for i, e in self._entries.items()}
        return self._index

    def search(self, query, category=None, only_model=False, limit=200):
        q = norm_text(query or "")
        pool = self.entries(False, category, only_model)
        if not q:
            return pool[:limit]
        qw = q.split()
        idx = self._search_index()
        scored = []
        for e in pool:
            lab, als, notes = idx[e.id]
            lw = lab.split()
            s = 0.0
            if lab == q:
                s = 100
            elif lab.startswith(q):
                s = 90
            elif all(w in lw for w in qw):
                s = 80
            elif q in als:
                s = 75
            elif any(a.startswith(q) for a in als):
                s = 72
            elif any(all(w in a.split() for w in qw) for a in als):
                s = 70
            elif q in lab:
                s = 62
            elif any(q in a for a in als):
                s = 60
            elif q in notes:
                s = 50
            else:
                cands = [lab] + lw + als
                r = max(difflib.SequenceMatcher(None, q, c).ratio() for c in cands) if len(q) >= 3 else 0
                if r >= 0.75:
                    s = 20 + 20 * r
            if s:
                scored.append((-s, len(lab), lab, e))
        scored.sort(key=lambda t: t[:3])
        return [t[3] for t in scored[:limit]]

    def _quality_score(self, id_):
        if id_ not in self._quality:
            try:
                st, good = L.clip_quality(id_) if self._lm_dir == L.LANDMARK_DIR else (None, False)
                if st is None and (self._lm_dir / f"{id_}.npy").exists():
                    st = L.clip_stats(np.load(self._lm_dir / f"{id_}.npy"))
                    good = L.is_good(st)
                self._quality[id_] = 0.0 if st is None else float(good) + st["any_hand"] + st["pose"]
            except Exception:
                self._quality[id_] = 0.0
        return self._quality[id_]

    def resolve_phrase(self, query):
        by_text, maxn = {}, 1
        for e in self._entries.values():
            if e.hidden:
                continue
            for t in [e.label] + e.aliases + ([e.default_label] if e.default_label else []):
                k = " ".join(_words(t))
                if k:
                    by_text.setdefault(k, {})[e.id] = e
                    maxn = max(maxn, len(k.split()))
        words = _words(query)
        found, missing, i = [], [], 0
        while i < len(words):
            for n in range(min(maxn, len(words) - i), 0, -1):
                c = by_text.get(" ".join(words[i:i + n]))
                if c:
                    found.append(self._pick(list(c.values())))
                    i += n
                    break
            else:
                missing.append(words[i])
                i += 1
        return found, missing

    def _pick(self, cands):
        def pri(e):
            return (e.in_model, e.edited, e.source == "recorded")
        top = max(pri(e) for e in cands)
        cands = [e for e in cands if pri(e) == top]
        if len(cands) > 1:
            cands = [max(cands, key=lambda e: (self._quality_score(e.id) if e.source == "dictionary" else 0,
                                               -int(e.id) if e.id.isdigit() else 0))]
        return self._cp(cands[0])

    # ---- animation
    def _raw(self, id_, take=0):
        key = f"{id_}_t{take}"
        if key in self._take_keys:
            return self.store.load_blob(_TAKES, key), self._fps.get(key, OUT_FPS)
        if take == 0 and id_ in self._base:
            p = self._lm_dir / f"{id_}.npy"
            if p.exists():
                return np.load(p), OUT_FPS
        raise KeyError(f"no landmark data for {id_} take {take}")

    def animation(self, id, take=0):
        if id not in self._entries:
            raise KeyError(id)
        k = (id, take)
        if k in self._anim:
            self._anim.move_to_end(k)
            return self._anim[k]
        raw, fps = self._raw(id, take)
        a = prepare_animation(raw, fps)
        a.setflags(write=False)
        self._anim[k] = a
        while len(self._anim) > 64:
            self._anim.popitem(last=False)
        return a

    def thumbnail_frame(self, id):
        a = self.animation(id)
        if len(a) == 0:
            return np.full((F.N_PTS, 3), np.nan, np.float32)
        hands = (~np.isnan(a[:, F.N_POSE:, 0])).sum(axis=1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            wy = np.nanmin(a[:, [F.L_WR, F.R_WR], 1], axis=1) if not np.isnan(a[:, [F.L_WR, F.R_WR], 1]).all() \
                else np.zeros(len(a))
        up = np.where(np.isnan(wy), -9.0, -wy)       # smaller y = higher
        score = up + np.where(hands > 0, 10.0, 0.0)
        return a[int(np.argmax(score))].copy()

    # ---- edits
    def _need(self, id_):
        if id_ not in self._entries:
            raise KeyError(id_)

    def _set_override(self, id_, ov, actor):
        if ov:
            self._overrides[id_] = ov
        else:
            self._overrides.pop(id_, None)
        self._save("overrides", self._overrides, actor)

    def update(self, id, *, label=None, notes=None, aliases=None, category=None, actor="signer-admin"):
        self._need(id)
        b = self._recorded.get(id) or self._base[id]
        ov = dict(self._overrides.get(id, {}))
        changed = []
        if label is not None:
            v = _clean_text(label, "label", MAX_LABEL)
            ov["label"] = v
            changed.append("label")
        if notes is not None:
            ov["notes"] = _clean_text(notes, "notes", MAX_NOTES, multiline=True, allow_empty=True)
            changed.append("notes")
        if aliases is not None:
            ov["aliases"] = _clean_aliases(aliases)
            changed.append("aliases")
        if category is not None:
            ov["category"] = _clean_text(category, "category", MAX_CATEGORY)
            changed.append("category")
        for f, base in (("label", b["default_label"]), ("notes", b.get("notes", "")),
                        ("aliases", b.get("aliases", [])), ("category", b["category"])):
            if f in ov and ov[f] == base:
                del ov[f]
        self._set_override(id, ov, actor)
        self._audit("library_update", id, actor, changed)
        self._refresh(id)
        return self.get(id)

    def reset(self, id, actor="signer-admin"):
        self._need(id)
        self._set_override(id, {}, actor)
        if id in self._base and f"{id}_t0" in self._take_keys:
            self._del_take(id, 0, actor)
        self._audit("library_reset", id, actor)
        self._refresh(id)

    def hide(self, id, actor="signer-admin"):
        self._set_hidden(id, True, actor)

    def unhide(self, id, actor="signer-admin"):
        self._set_hidden(id, False, actor)

    def _set_hidden(self, id, flag, actor):
        self._need(id)
        ov = dict(self._overrides.get(id, {}))
        if flag:
            ov["hidden"] = True
        else:
            ov.pop("hidden", None)
        self._set_override(id, ov, actor)
        self._audit("library_hide" if flag else "library_unhide", id, actor)
        self._refresh(id)

    # ---- takes
    def _put_take(self, id_, n, raw, fps, actor):
        key = f"{id_}_t{n}"
        self.store.save_blob(_TAKES, key, _check_window(raw), actor=actor)
        self._take_keys.add(key)
        if fps != OUT_FPS:
            self._fps[key] = float(fps)
            self._save("fps", self._fps, actor)

    def _del_take(self, id_, n, actor):
        key = f"{id_}_t{n}"
        if key in self._take_keys:
            self.store.delete_blob(_TAKES, key, actor=actor)
            self._take_keys.discard(key)
            if self._fps.pop(key, None) is not None:
                self._save("fps", self._fps, actor)
            if str(n) in self._auto.get(id_, {}):
                del self._auto[id_][str(n)]
                if not self._auto[id_]:
                    del self._auto[id_]
                self._save("auto", self._auto, actor)

    def replace_animation(self, id, raw_window, actor="signer-admin", fps=OUT_FPS):
        self._need(id)
        self._put_take(id, 0, raw_window, fps, actor)
        self._audit("library_replace_animation", id, actor)
        self._refresh(id)

    def add_take(self, id, raw_window, actor="signer-admin", fps=OUT_FPS) -> int:
        self._need(id)
        nums = [int(k[len(id) + 2:]) for k in self._take_keys
                if k.startswith(id + "_t") and k[len(id) + 2:].isdigit()]
        n = max(nums + [0]) + 1
        self._put_take(id, n, raw_window, fps, actor)
        self._audit("library_add_take", id, actor)
        self._refresh(id)
        return n

    def auto_takes(self, id) -> list:
        """Examples learned automatically from corrections: [{'take': n, 'time': unix seconds}], oldest first."""
        if id not in self._entries:
            raise KeyError(id)
        out = [dict(take=int(n), time=float(t)) for n, t in self._auto.get(id, {}).items()
               if f"{id}_t{n}" in self._take_keys]
        return sorted(out, key=lambda d: (d["time"], d["take"]))

    def add_auto_take(self, id, window, actor="auto-learn", fps=OUT_FPS, now=None) -> int:
        """Save a corrected movement (encrypted, landmarks only) as another example of a sign that is
        switched on. At most MAX_AUTO_TAKES are kept per sign: the OLDEST auto-learned one goes first;
        the user's own recorded examples are never dropped. One audit record (ids and counts only)."""
        self._need(id)
        if not self.is_active(id) or self._entries[id].hidden:
            raise LibraryError("this sign is not switched on")
        a = _check_window(window)
        fps = float(fps)
        if not (5.0 <= fps <= 120.0):
            raise LibraryError("unusable frame rate")
        if len(a) < AUTO_MIN_S * fps:
            raise LibraryError("movement too short to learn from")
        hands = float((~np.isnan(a[:, F.N_POSE:, 0])).any(axis=1).mean())
        if hands < AUTO_MIN_HANDS:
            raise LibraryError("hands not visible enough to learn from")
        n = self.add_take(id, window, actor=actor, fps=fps)
        self._auto.setdefault(id, {})[str(n)] = float(time.time() if now is None else now)
        evicted = 0
        while len(self.auto_takes(id)) > MAX_AUTO_TAKES:
            self._del_take(id, self.auto_takes(id)[0]["take"], actor)
            evicted += 1
        self._save("auto", self._auto, actor)
        self.store.audit("library_auto_learn", id, actor, {"auto": len(self.auto_takes(id)), "evicted": evicted})
        self._refresh(id)
        return n

    def examples(self, id) -> list:
        """The examples of a sign, in order: [{'take': n, 'kind': 'dictionary' | 'mine' | 'learned', 'time': unix | None}].
        'dictionary' = the built-in clip (cannot be removed); 'mine' = recorded by the user (a new sign's first example, an
        extra example, or a re-recorded dictionary example); 'learned' = added automatically from a correction (time = when)."""
        self._need(id)
        auto = {int(n): float(t) for n, t in self._auto.get(id, {}).items()}
        out = []
        for n in self.take_numbers(id):
            if id in self._base and n == 0 and f"{id}_t0" not in self._take_keys:
                out.append(dict(take=n, kind="dictionary", time=None))
            elif n in auto:
                out.append(dict(take=n, kind="learned", time=auto[n]))
            else:
                out.append(dict(take=n, kind="mine", time=None))
        return out

    def remove_take(self, id, take_no, actor="signer-admin"):
        """Remove one example of a sign. A dictionary clip (example 0 of a built-in sign) and the last
        remaining example of a sign cannot be removed (delete the sign instead)."""
        self._need(id)
        take_no = int(take_no)
        if take_no not in self.take_numbers(id):
            raise KeyError(take_no)
        if id in self._base and take_no == 0 and f"{id}_t0" not in self._take_keys:
            raise LibraryError("the built-in example cannot be removed")
        # (a dictionary sign whose example 0 was RE-RECORDED by the user: removing that recording restores the built-in clip)
        if id in self._recorded and len(self.take_numbers(id)) <= 1:
            raise LibraryError("this is the only example of the sign")
        self._del_take(id, take_no, actor)
        self.store.audit("library_remove_take", id, actor, {"remaining": len(self.take_numbers(id))})
        self._refresh(id)

    def add_recorded(self, label, raw_window, notes="", aliases=(), category=USER_CATEGORY,
                     actor="signer-admin", fps=OUT_FPS) -> str:
        label = _clean_text(label, "label", MAX_LABEL)
        notes = _clean_text(notes, "notes", MAX_NOTES, multiline=True, allow_empty=True)
        aliases = _clean_aliases(aliases)
        category = _clean_text(category, "category", MAX_CATEGORY)
        _check_window(raw_window)
        id_ = "u_" + secrets.token_hex(6)
        while id_ in self._entries:
            id_ = "u_" + secrets.token_hex(6)
        self._put_take(id_, 0, raw_window, fps, actor)
        self._recorded[id_] = dict(default_label=label, notes=notes, aliases=aliases, category=category)
        self._save("recorded", self._recorded, actor)
        self._audit("library_add_recorded", id_, actor)
        self._entries[id_] = self._make(id_)
        self._order.insert(0, id_)
        self._index = None
        return id_

    def delete_recorded(self, id, actor="signer-admin"):
        if id not in self._recorded:
            raise KeyError(id)
        for k in [k for k in self._take_keys if k.startswith(id + "_t")]:
            self._del_take(id, int(k[len(id) + 2:]), actor)
        del self._recorded[id]
        self._save("recorded", self._recorded, actor)
        if self._active.pop(id, None) is not None:
            self._save("active", self._active, actor)
        if id in self._overrides:
            self._set_override(id, {}, actor)
        self._audit("library_delete_recorded", id, actor)
        self._refresh(id)

    # ---- personal recognition templates
    def _has_user_take(self, id_):
        return any(k.startswith(id_ + "_t") and k[len(id_) + 2:].isdigit() for k in self._take_keys)

    def is_active(self, id) -> bool:
        """Is this sign used for live recognition ('Use in conversation')? Explicit choice wins;
        default: signs the user recorded and any sign carrying a user take are on, plain
        dictionary signs are off until switched on."""
        if id not in self._entries:
            return False
        if id in self._active:
            return bool(self._active[id])
        return id in self._recorded or self._has_user_take(id)

    def set_active(self, id, on: bool, actor="signer-admin"):
        self._need(id)
        on = bool(on)
        if self.is_active(id) == on and id in self._active:
            return
        self._active[id] = on
        self._save("active", self._active, actor)
        self._audit("library_set_active", id, actor, fields=["on" if on else "off"])

    def set_active_many(self, ids, on: bool, actor="signer-admin", max_new=None):
        """Switch many signs on/off with ONE persisted write and ONE audit record (all-or-nothing).
        Returns {id: previous explicit flag (True/False) or None when it was on its default} for the
        signs that changed, which `restore_active` accepts to undo exactly."""
        ids = list(dict.fromkeys(ids))
        for i in ids:
            self._need(i)
        on = bool(on)
        if on and any(self._entries[i].hidden for i in ids):
            raise LibraryError("hidden signs cannot be switched on in bulk")
        if on and max_new is not None and sum(1 for i in ids if not self.is_active(i)) > int(max_new):
            raise LibraryError(f"too many signs at once (max {int(max_new)})")
        prev = {i: self._active.get(i) for i in ids if self._active.get(i) != on}
        if not prev:
            return {}
        for i in prev:
            self._active[i] = on
        self._save("active", self._active, actor)
        self.store.audit("library_set_active_many", "batch", actor, {"on": on, "count": len(prev)})
        return prev

    def restore_active(self, previous, actor="signer-admin"):
        """Undo of `set_active_many`: put every listed sign back to its previous explicit state
        (None = no explicit choice, i.e. its default). One write, one audit record."""
        for i in previous:
            self._need(i)
        if not previous:
            return
        for i, v in previous.items():
            if v is None:
                self._active.pop(i, None)
            else:
                self._active[i] = bool(v)
        self._save("active", self._active, actor)
        self.store.audit("library_restore_active", "batch", actor, {"count": len(previous)})

    def active_counts(self, category=None):
        """(switched on, total) over the signs that are not hidden, optionally for one category."""
        n_on = n = 0
        for e in self._entries.values():
            if e.hidden or (category and e.category != category):
                continue
            n += 1
            n_on += self.is_active(e.id)
        return n_on, n

    def take_numbers(self, id):
        """Existing take numbers of a sign (0 = the dictionary clip or the first recording)."""
        if id not in self._entries:
            raise KeyError(id)
        nums = {int(k[len(id) + 2:]) for k in self._take_keys
                if k.startswith(id + "_t") and k[len(id) + 2:].isdigit()}
        if id in self._base and (self._lm_dir / f"{id}.npy").exists():
            nums.add(0)
        return sorted(nums)

    def templates(self):
        """Matcher input: [{id, label, frames: [raw (n,67,3), ...], fps: [..]}] for every active,
        non-hidden sign, all takes (raw data, untouched; the matcher cuts the sign itself)."""
        out = []
        for i in self._order:
            e = self._entries.get(i)
            if e is None or e.hidden or not self.is_active(i):
                continue
            frames, fps = [], []
            for n in self.take_numbers(i):
                try:
                    raw, f = self._raw(i, n)
                except Exception:
                    continue
                frames.append(np.asarray(raw, np.float32))
                fps.append(float(f))
            if frames:
                out.append(dict(id=i, label=e.label, frames=frames, fps=fps))
        return out

    # ---- model integration
    def label_map(self):
        """One output text per model class (first edited clip wins). Texts that equal another class's
        name or an already-used text are ignored so the recogniser's reverse lookup stays unambiguous."""
        classes = {e.model_class for e in self._entries.values() if e.model_class}
        out, used = {}, set()
        for i in self._order:
            e = self._entries[i]
            if e.model_class and e.label != e.default_label and not e.hidden and e.model_class not in out:
                k = norm_text(e.label)
                if k in used or any(k == norm_text(c) for c in classes if c != e.model_class):
                    continue
                out[e.model_class] = e.label; used.add(k)
        return out

    def display_name(self, model_class):
        return self.label_map().get(model_class, model_class)

    # ---- export for training
    def export_user_signs(self, directory) -> int:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True, mode=0o700)
        try:
            os.chmod(d, 0o700)
        except OSError:
            pass
        for old in d.glob("*.npy"):                 # drop stale takes from earlier exports
            old.unlink()
        meta, n = {}, 0
        for key in sorted(self._take_keys):
            id_, _, t = key.rpartition("_t")
            if id_ not in self._entries:
                continue
            e = self._entries[id_]
            fn = f"{key}.npy"
            np.save(d / fn, self.store.load_blob(_TAKES, key).astype(np.float32), allow_pickle=False)
            os.chmod(d / fn, 0o600)
            meta[fn] = dict(id=id_, take=int(t), label=e.label, default_label=e.default_label,
                            category=e.category, source=e.source, fps=self._fps.get(key, OUT_FPS))
            n += 1
        (d / "labels.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
        os.chmod(d / "labels.json", 0o600)
        self.store.audit("library_export", "export", "signer-admin", {"count": n})
        return n
