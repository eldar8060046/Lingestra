"""Honest, reproducible evaluation of the personal matcher on the dictionary landmarks.

    python scripts/benchmark_recognition.py            # full run, writes docs/recognition.md
    python scripts/benchmark_recognition.py --quick    # small subset, prints only (no docs write)

What is measured (see docs/recognition.md for the interpretation and the limits):
  * CROSS-TAKE: for signs with >= 2 good real takes, take 1 is the enrolled template and take 2 is
    the query (a genuinely different performance of the same sign; no augmentation involved).
  * STRESS on the query: speed, inserted pauses, raise/lower transitions + idle lead-in/out, camera
    shift/zoom/rotation, landmark jitter and hand drop-outs, and a combination.
  * OPEN SET: queries of signs that are NOT enrolled must not be accepted.
  * Methods: (a) Keras dictionary model path, (b) DTW without hold collapsing, (c) full matcher.
  * Vocabulary sizes 10 / 30 / 100.
Everything uses fixed seeds. The data are ONE dictionary signer: cross-take is the most realistic
proxy available, real webcam validation is still required.
"""
import argparse
from collections import Counter
import sys
import time
import zlib
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
from lingestra import features as F, labels as L                       # noqa: E402
from lingestra.live.matcher import MatcherConfig, PersonalMatcher     # noqa: E402

FPS = 25.0
DOCS = ROOT / "docs" / "recognition.md"


# ----------------------------------------------------------------------------- data
def load_pool():
    """[(label, [raw takes (good quality only)])] for signs with a good take, sorted by label."""
    pool = []
    for lab, ts in sorted(L.takes_by_label().items()):
        takes = []
        for e in ts:
            if L.clip_quality(e["id"])[1]:
                takes.append(np.load(L.LANDMARK_DIR / f"{e['id']}.npy"))
        if takes:
            pool.append((lab, takes))
    return pool


# ----------------------------------------------------------------------------- query stress ops
def _fix_nan(out, near):
    return np.where(np.isnan(out), near, out)


def time_scale(seg, speed):
    n = max(int(round(len(seg) / speed)), 4)
    src = np.linspace(0, len(seg) - 1, n)
    lo = np.floor(src).astype(int)
    hi = np.minimum(lo + 1, len(seg) - 1)
    w = (src - lo)[:, None, None]
    out = seg[lo] * (1 - w) + seg[hi] * w
    return _fix_nan(out, np.where(w < 0.5, seg[lo], seg[hi])).astype(np.float32)


def insert_pauses(seg, rng, k=None, lo_s=0.3, hi_s=1.5):
    """Hold the pose at 1-3 random points: pauses between key poses. A real held pose is not frozen:
    landmark jitter ~0.004 plus a slow random-walk drift (hand sag / sway) are added."""
    k = k or int(rng.integers(1, 4))
    pts = sorted(rng.integers(2, max(len(seg) - 2, 3), k))
    parts, prev = [], 0
    for p in pts:
        parts.append(seg[prev:p])
        n = int(rng.uniform(lo_s, hi_s) * FPS)
        drift = np.cumsum(rng.normal(0, 0.0015, (n, 1, 3)), 0).astype(np.float32) * np.array([1, 1, 0], np.float32)
        parts.append(np.repeat(seg[p - 1:p], n, 0) + drift
                     + rng.normal(0, 0.004, (n,) + seg.shape[1:]).astype(np.float32) * np.array([1, 1, 0], np.float32))
        prev = p
    parts.append(seg[prev:])
    return np.concatenate(parts).astype(np.float32)


def add_transitions(clip, seg, lead_s=1.0, move_s=0.4, tail_s=1.0):
    """idle + raise-from-rest + sign + lower-to-rest + idle, built from the clip's own rest frames."""
    rest_a, rest_b = clip[0], clip[-1]

    def blend(a, b, n):
        w = np.linspace(0, 1, n)[:, None, None]
        out = a[None] * (1 - w) + b[None] * w
        return _fix_nan(out, np.where(w < 0.5, a[None], b[None])).astype(np.float32)
    m = int(move_s * FPS)
    return np.concatenate([np.repeat(rest_a[None], int(lead_s * FPS), 0), blend(rest_a, seg[0], m), seg,
                           blend(seg[-1], rest_b, m), np.repeat(rest_b[None], int(tail_s * FPS), 0)])


def fidget(raw, rng, dur_s=4.0):
    """Non-sign motion: hands raised from rest, wandering around, put down again (no dictionary sign)."""
    n = int(dur_s * FPS)
    base = raw[len(raw) // 2].copy()
    base[F.N_POSE:] = np.nan_to_num(base[F.N_POSE:], nan=0.5)
    path = np.cumsum(rng.normal(0, 0.012, (n, 1, 3)), 0).astype(np.float32) * np.array([1, 1, 0], np.float32)
    mid = np.repeat(base[None], n, 0)
    mid[:, F.N_POSE:] += path
    mid[:, :F.N_POSE] += path[:, :, :] * 0.2
    return add_transitions(raw, mid)


def camera(clip, rng, shift=0.08, zoom=0.25, rot=10.0):
    a = np.deg2rad(rng.uniform(-rot, rot))
    z = np.exp(rng.uniform(-np.log(1 + zoom), np.log(1 + zoom)))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    out = clip.copy()
    xy = (out[..., :2] - 0.5) @ R.T * z + 0.5 + rng.uniform(-shift, shift, 2)
    out[..., :2] = xy
    return out


def jitter_dropout(clip, rng, sigma=0.004, drop=0.15):
    out = clip + rng.normal(0, sigma, clip.shape).astype(np.float32) * np.array([1, 1, 0], np.float32)
    for sl in (slice(F.N_POSE, F.N_POSE + 21), slice(F.N_POSE + 21, F.N_PTS)):
        i = 0
        while i < len(out):
            if rng.random() < drop / 5:
                n = int(rng.integers(3, 9))
                out[i:i + n, sl] = np.nan
                i += n
            i += 1
    return out


def sign_of(raw):
    seg = F.trim_active(raw)
    return seg


CONDITIONS = [
    ("clean (cross-take as is)", lambda raw, rng: raw),
    ("speed 0.5x", lambda raw, rng: time_scale(sign_of(raw), 0.5)),
    ("speed 0.75x", lambda raw, rng: time_scale(sign_of(raw), 0.75)),
    ("speed 1.5x", lambda raw, rng: time_scale(sign_of(raw), 1.5)),
    ("speed 2x", lambda raw, rng: time_scale(sign_of(raw), 2.0)),
    ("1-3 pauses of 0.3-1.5 s", lambda raw, rng: insert_pauses(sign_of(raw), rng)),
    ("raise/lower + idle lead-in/out", lambda raw, rng: add_transitions(raw, sign_of(raw))),
    ("camera shift/zoom/rotation", lambda raw, rng: camera(raw, rng)),
    ("jitter + hand drop-outs", lambda raw, rng: jitter_dropout(raw, rng)),
    ("combined (1.5x + pauses + transitions + camera + jitter)",
     lambda raw, rng: jitter_dropout(camera(add_transitions(raw, insert_pauses(
         time_scale(sign_of(raw), 1.5), rng)), rng), rng)),
]


# ----------------------------------------------------------------------------- comparator (b)
def dtw_plain(C):
    """Textbook DTW (steps right / down / diagonal) on a local-cost matrix, normalised by path
    length. Anti-diagonal vectorised. No band, no slack: the baseline without hold collapsing."""
    n, m = C.shape
    INF = 1e9
    D = np.full((n + 1, m + 1), INF)
    P = np.zeros((n + 1, m + 1))
    D[0, 0] = 0.0
    for k in range(2, n + m + 1):
        i = np.arange(max(1, k - m), min(n, k - 1) + 1)
        j = k - i
        st = np.stack([D[i - 1, j - 1], D[i - 1, j], D[i, j - 1]])
        ps = np.stack([P[i - 1, j - 1], P[i - 1, j], P[i, j - 1]])
        a = st.argmin(0)
        D[i, j] = np.take_along_axis(st, a[None], 0)[0] + C[i - 1, j - 1]
        P[i, j] = np.take_along_axis(ps, a[None], 0)[0] + 1
    return float(D[n, m] / max(P[n, m], 1))


class NativeDTW(PersonalMatcher):
    """Same features, same frame cost, same decision rule and thresholds; but sequences are kept at
    their native frame rate (no hold collapse, no arc-length resampling) and aligned with plain DTW."""

    def set_templates(self, templates):
        from lingestra.live import matcher as Mt
        self.clear()
        self._tk = []
        labels = {}
        for t in templates:
            li = labels.setdefault(t["label"], len(labels))
            for fr in t["frames"]:
                X, Mk = F.sign_features(F.trim_active(fr))
                self._tk.append((li, Mt._Seq(X, Mk, len(X))))
        self.labels = list(labels)
        self.thr = np.full(len(self.labels), self.cfg.default_thr)
        self.n_takes = len(self._tk)

    def distances(self, raw, fps=25.0, trim=None):
        from lingestra.live import matcher as Mt
        raw = F.trim_active(raw) if (self.cfg.trim if trim is None else trim) else raw
        X, Mk = F.sign_features(raw)
        out = np.full(len(self.labels), np.inf)
        if len(X) < 2:
            return out, len(X)
        q = Mt._Seq(X, Mk, len(X))
        for li, sq in self._tk:
            C = Mt.cost_matrices(sq.X[None], sq.M[None], q, self.cfg.c_miss)[0]
            out[li] = min(out[li], dtw_plain(C))
        return out, len(X)


# ----------------------------------------------------------------------------- evaluation
def build_matcher(cfg, vocab):
    m = NativeDTW(cfg) if cfg.collapse is False else PersonalMatcher(cfg)
    m.set_templates([dict(id=str(i), label=lab, frames=[takes[0]]) for i, (lab, takes) in enumerate(vocab)])
    return m


def classify_all(m, queries, strictness=1.0):
    """queries: [(true_label, raw)] -> dict of rates."""
    n = len(queries)
    top1 = acc_ok = acc_bad = clar_ok = unk = 0
    ms = []
    absorb = Counter()
    for lab, raw in queries:
        t0 = time.perf_counter()
        d, _ = m.distances(raw, FPS, trim=True)
        res = m.decide(d, strictness)
        ms.append((time.perf_counter() - t0) * 1000)
        top1 += bool(np.isfinite(d).any() and m.labels[int(np.argmin(d))] == lab)
        if res.kind == "sign":
            acc_ok += res.label == lab
            acc_bad += res.label != lab
            if res.label != lab:
                absorb[res.label] += 1
        elif res.kind == "clarify":
            clar_ok += lab in [c[0] for c in res.candidates[:2]]
        else:
            unk += 1
    return dict(n=n, top1=top1 / n, accept_ok=acc_ok / n, accept_bad=acc_bad / n,
                clarify_ok=clar_ok / n, unknown=unk / n, ms=float(np.mean(ms)), absorb=absorb)


def avg(runs):
    """Mean of the numeric rates over draws; absorbing labels are summed."""
    out = {k: float(np.mean([r[k] for r in runs])) for k in runs[0] if k != "absorb"}
    out["absorb"] = sum((r["absorb"] for r in runs), Counter())
    return out


def precision(r):
    n = r["accept_ok"] + r["accept_bad"]
    return pct(r["accept_ok"] / n) if n > 0 else "-"


def top_absorbers(r, k=3):
    tot = sum(r["absorb"].values())
    return ", ".join(f"{lab} {c}" for lab, c in r["absorb"].most_common(k)) + (f" (of {tot})" if tot else "") if tot else "-"


def pick_vocab(eligible, size, rng):
    idx = rng.permutation(len(eligible))[:size]
    return [eligible[i] for i in idx]


def run_matcher_suite(pool, cfg, sizes, repeats, seed, conditions, query_take=1):
    """Returns {(size, condition): rates averaged over repeats} and open-set rates."""
    eligible = [(lab, t) for lab, t in pool if len(t) >= 2]
    res, openset = {}, {}
    for size in sizes:
        reps = 1 if size >= len(eligible) else repeats
        acc = {c[0]: [] for c in conditions}
        op = []
        for r in range(reps):
            rng = np.random.default_rng(seed + 1000 * size + r)
            vocab = pick_vocab(eligible, size, rng)
            m = build_matcher(cfg, vocab)
            for name, fn in conditions:
                qs = [(lab, fn(t[query_take], np.random.default_rng(seed + zlib.crc32(lab.encode()) % 9973 + r))) for lab, t in vocab]
                acc[name].append(classify_all(m, qs))
            # open set: queries of signs that are not enrolled (their first take)
            have = {lab for lab, _ in vocab}
            others = [(lab, t[0]) for lab, t in pool if lab not in have]
            sel = np.random.default_rng(seed + r).permutation(len(others))[:150]
            qs = [others[i] for i in sel]
            op.append(classify_all(m, qs))
        for name in acc:
            res[(size, name)] = avg(acc[name])
        openset[size] = avg(op)
    return res, openset


# ----------------------------------------------------------------------------- stressed open set
OPEN_CONDS = [
    ("clean", lambda raw, rng: raw),
    ("1-3 drifting pauses (0.3-1.5 s)", lambda raw, rng: insert_pauses(sign_of(raw), rng)),
    ("raise/lower + idle lead-in/out", lambda raw, rng: add_transitions(raw, sign_of(raw))),
    ("1.5x + pauses + transitions + camera + jitter",
     lambda raw, rng: jitter_dropout(camera(add_transitions(raw, insert_pauses(
         time_scale(sign_of(raw), 1.5), rng)), rng), rng)),
    ("non-sign motion (hands raised, wandering, lowered)", lambda raw, rng: fidget(raw, rng)),
]


def run_openset_extra(pool, cfg, seed, size=30, repeats=3, n_query=100):
    """False accepts for un-enrolled signs under stress, with 1 enrolled take per sign and with 2 takes
    per sign (which switches on the per-label calibrated threshold). Returns {(takes, cond): rates}."""
    eligible = [(lab, t) for lab, t in pool if len(t) >= 2]
    acc = {(k, c[0]): [] for k in (1, 2) for c in OPEN_CONDS}
    for r in range(repeats):
        rng = np.random.default_rng(seed + 77 + r)
        vocab = pick_vocab(eligible, size, rng)
        have = {lab for lab, _ in vocab}
        others = [(lab, t[0]) for lab, t in pool if lab not in have]
        sel = np.random.default_rng(seed + r).permutation(len(others))[:n_query]
        for k in (1, 2):
            m = PersonalMatcher(cfg)
            m.set_templates([dict(id=str(i), label=lab, frames=list(takes[:k])) for i, (lab, takes) in enumerate(vocab)])
            for name, fn in OPEN_CONDS:
                qs = [(others[i][0], fn(others[i][1], np.random.default_rng(seed + 31 * i + r))) for i in sel]
                acc[(k, name)].append(classify_all(m, qs))
    return {key: avg(v) for key, v in acc.items()}


def run_hold_split(pool, cfg, seed, size=30, n_query=40):
    """Full Recogniser path (hold splitting, raise/lower segmentation) on a continuous stream: an
    un-enrolled sign performed with a 1 s drifting mid-sign pause, then rest. Counts streams that
    produced at least one 'sign' event (any = false accept) and the mean number of 'sign' events."""
    from lingestra.live.recognizer import Recognizer
    eligible = [(lab, t) for lab, t in pool if len(t) >= 2]
    rng = np.random.default_rng(seed + 5)
    vocab = pick_vocab(eligible, size, rng)
    have = {lab for lab, _ in vocab}
    others = [(lab, t[0]) for lab, t in pool if lab not in have]
    sel = np.random.default_rng(seed).permutation(len(others))[:n_query]
    rec = Recognizer(None, templates=[dict(id=str(i), label=lab, frames=[takes[0]]) for i, (lab, takes) in enumerate(vocab)])
    out = {}
    for name, fn in (("clean", lambda raw, r: raw),
                     ("one 1 s drifting pause mid-sign", lambda raw, r: insert_pauses(sign_of(raw), r, k=1, lo_s=1.0, hi_s=1.0)),
                     ("non-sign motion", lambda raw, r: fidget(raw, r))):
        any_sign = n_sign = 0
        for i in sel:
            q = fn(others[i][1], np.random.default_rng(seed + 31 * int(i)))
            stream = np.concatenate([np.repeat(q[:1], 10, 0), q, np.repeat(q[-1:], 40, 0)])
            rec.reset()
            ev = []
            for j, f in enumerate(stream):
                ev += rec.push_frame(f, j / FPS)
            k = sum(e.kind == "sign" for e in ev)
            any_sign += k > 0; n_sign += k
        out[name] = (any_sign / len(sel), n_sign / len(sel))
    return out


# ----------------------------------------------------------------------------- Keras path
def run_keras(conditions, seed, query_take=1):
    """The old dictionary model path on its own 12 Family signs (it cannot do other vocabularies).
    Streams go through the Recognizer in model-only mode; counts correct / wrong / no decision.
    Uses every take of those signs regardless of the quality filter (only a handful exist)."""
    import json
    mp = ROOT / "models" / "demo_family.keras"
    if not mp.exists():
        return None
    from lingestra.live.recognizer import Recognizer
    classes = json.load(open(mp.with_suffix(".json")))["classes"]
    rec = Recognizer(str(mp), use_model=True)
    items = []
    for lab, ts in sorted(L.takes_by_label().items()):
        if lab in classes and len(ts) >= (2 if query_take else 1):
            items.append((lab, [np.load(L.LANDMARK_DIR / f"{e['id']}.npy") for e in ts]))
    rows = {}
    for name, fn in conditions:
        ok = bad = none = 0
        for lab, t in items:
            q = fn(t[query_take], np.random.default_rng(seed + zlib.crc32(lab.encode()) % 9973))
            h = F.hand_height(q)
            if np.nanmax(h[:3]) > 0.38 or np.nanmax(h[-3:]) > 0.38:       # starts / ends with raised hands
                q = add_transitions(t[query_take], q)                     # (the live stream always has rest)
            stream = np.concatenate([np.repeat(q[:1], 10, 0), q, np.repeat(q[-1:], 40, 0)])
            rec.reset()
            ev = []
            for i, f in enumerate(stream):
                ev += rec.push_frame(f, i / FPS)
            signs = [e.label for e in ev if e.kind == "sign"]
            ok += signs == [lab]
            bad += bool(signs) and signs != [lab]
            none += not signs
        rows[name] = dict(n=len(items), ok=ok / len(items), bad=bad / len(items), none=none / len(items))
    return rows


# ----------------------------------------------------------------------------- reporting
def pct(x):
    return f"{100 * x:.0f}%"


def table(rows, header):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def speed_check(pool, cfg):
    """50 signs x 3 takes: decision time."""
    elig = [(lab, t) for lab, t in pool if len(t) >= 2][:50]
    tmpl = [dict(id=str(i), label=lab, frames=[t[0], t[1], t[0][::2]]) for i, (lab, t) in enumerate(elig)]
    m = PersonalMatcher(cfg)
    t0 = time.perf_counter()
    m.set_templates(tmpl)
    setup = time.perf_counter() - t0
    qs = [F.trim_active(t[1]) for _, t in elig[:20]]
    ts = []
    for q in qs:
        t0 = time.perf_counter()
        m.classify(q, trim=False)
        ts.append((time.perf_counter() - t0) * 1000)
    return len(elig), m.n_takes, setup, float(np.mean(ts)), float(np.max(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-keras", action="store_true")
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args()
    pool = load_pool()
    n_elig = sum(len(t) >= 2 for _, t in pool)
    print(f"dictionary signs with a good take: {len(pool)}; with >= 2 good takes: {n_elig}")
    sizes = [10] if a.quick else [10, 30, 100]
    repeats = 1 if a.quick else 5
    conds = CONDITIONS[:3] if a.quick else CONDITIONS
    cfg_full = MatcherConfig()
    cfg_nocollapse = MatcherConfig(collapse=False)
    t0 = time.time()
    full, op_full = run_matcher_suite(pool, cfg_full, sizes, repeats, a.seed, conds)
    nat_sizes = [x for x in sizes if x <= 30]            # plain DTW at native frame rate is ~100x slower
    nocol, op_nocol = run_matcher_suite(pool, cfg_nocollapse, nat_sizes, min(repeats, 3), a.seed, conds)
    keras = None if (a.quick or a.no_keras) else run_keras(conds, a.seed)
    keras_same = None if (a.quick or a.no_keras) else run_keras(conds, a.seed, query_take=0)
    same = None if a.quick else run_matcher_suite(pool, cfg_full, [30], repeats, a.seed, conds, query_take=0)[0]
    same_nc = None if a.quick else run_matcher_suite(pool, cfg_nocollapse, [30], 2, a.seed, conds, query_take=0)[0]
    extra = None if a.quick else run_openset_extra(pool, cfg_full, a.seed)
    hold = None if a.quick else run_hold_split(pool, cfg_full, a.seed)
    sp = speed_check(pool, cfg_full)
    print(f"benchmark time {time.time() - t0:.0f}s")

    md = ["# Recognition benchmark", "",
          "Generated by `python scripts/benchmark_recognition.py` (seed %d). Do not edit by hand." % a.seed, ""]
    md += ["## Honest scope", "",
           "* All data are the UAE Sign Language Dictionary landmarks: **one dictionary signer** (plus a few "
           "re-takes). The cross-take test (template = take 1, query = take 2 of the same sign) is the most "
           "realistic proxy available; it is NOT a test on a new signer or on your webcam.",
           "* **Real webcam validation by the team is still required** (different camera, lighting, framing, "
           "the signer's own style). Numbers below will differ in daily use. Enrolling 2-3 of your own takes "
           "per sign is recommended, but this benchmark only verifies that extra takes do NOT loosen "
           "rejection (open-set 1-take vs 2-take table); it cannot measure the gain from the user's own takes.",
           f"* Pool: {len(pool)} signs with a good-quality clip, {n_elig} with >= 2 good takes. Vocabulary "
           "of size V = V random eligible signs, one enrolled template each (take 1), queries = take 2. "
           f"Sizes with repeats: {sizes} x {repeats} random draws (mean).",
           "* Columns: **top-1** = nearest enrolled sign is the right one (ignores rejection); **accepted ok** "
           "= answered 'sign' with the right label; **wrong** = answered 'sign' with a WRONG label; "
           "**clarify ok** = asked 'did you mean A or B' and the right sign is one of the two; "
           "**unknown** = rejected. accepted ok + wrong + clarify (ok or not) + unknown = 100%. "
           "**precision** = accepted ok / (accepted ok + wrong): of the words actually written, the share that were right. "
           "**absorbing labels** = the labels that received the most wrong answers (summed over draws): a label "
           "that dominates is an 'attractor'.",
           "* **How to read the cross-take stress rows:** they barely move (0-5 points) because the difference "
           "between two dictionary takes of the SAME sign dominates; they are a floor set by take mismatch, "
           "NOT evidence of speed / pause invariance. The evidence for invariance is the same-take table, "
           "where that difference is removed.", ""]
    for size in sizes:
        md += [f"## Vocabulary of {size} signs: full matcher vs plain DTW (no hold collapsing)", ""]
        rows = []
        for name, _ in conds:
            f, n_ = full[(size, name)], nocol.get((size, name))
            nat = [pct(n_["top1"]), pct(n_["accept_ok"]), pct(n_["accept_bad"]), pct(n_["unknown"])] if n_ else ["-"] * 4
            rows.append([name, pct(f["top1"]), pct(f["accept_ok"]), pct(f["accept_bad"]), precision(f),
                         pct(f["clarify_ok"]), pct(f["unknown"]), top_absorbers(f)] + nat)
        md += [table(rows, ["query condition", "top-1", "accepted ok", "wrong", "precision", "clarify ok", "unknown",
                            "labels absorbing the wrong answers",
                            "native top-1", "native accepted ok", "native wrong", "native unknown"]), "",
               "(native = comparator (b), run for vocabularies <= 30 only, 3 draws (2 for the same-take table): it is ~100x slower; same features, cost, decision rule and thresholds, but plain DTW on the "
               "native frames: no hold collapsing, no arc-length resampling, no band / open ends.)", ""]
    if same:
        md += ["## Invariance isolated: the SAME take as template and query, stress applied to the query only (30 signs)", "",
               "Template = take 1, query = take 1 under each stress. This removes the cross-take difference, so it "
               "measures only what the speed / pause / camera / noise handling does. Perfect = 100% accepted ok. "
               "(The clean row is trivially 100%: the query is the template. Pauses here are drifting holds: "
               "jitter 0.004 plus a random-walk drift, not frozen frames.)", ""]
        rows = []
        for name, _ in conds:
            f, n_ = same[(30, name)], same_nc[(30, name)]
            rows.append([name, pct(f["top1"]), pct(f["accept_ok"]), pct(f["accept_bad"]), pct(f["unknown"]),
                         pct(n_["top1"]), pct(n_["accept_ok"]), pct(n_["accept_bad"]), pct(n_["unknown"])])
        md += [table(rows, ["query condition", "top-1", "accepted ok", "wrong", "unknown",
                            "native top-1", "native accepted ok", "native wrong", "native unknown"]), ""]
    md += ["## Open set: signs that are not enrolled must be rejected", ""]
    rows = []
    for size in sizes:
        f, n_ = op_full[size], op_nocol.get(size)
        rows.append([size, pct(f["accept_bad"]), pct(f["unknown"]), top_absorbers(f)] +
                    ([pct(n_["accept_bad"]), pct(n_["unknown"])] if n_ else ["-", "-"]))
    md += [table(rows, ["vocabulary", "FALSE ACCEPT (full)", "rejected (full)", "labels absorbing the false accepts",
                        "false accept (plain DTW)", "rejected (plain DTW)"]), "",
           "False accept = an un-enrolled sign was answered with some enrolled sign (150 queries per draw, signs "
           "that are not in the vocabulary). The remainder up to 100% was asked as 'clarify' (two candidates).", ""]
    if extra:
        md += ["## Open set under stress, 1 take vs 2 takes per enrolled sign (30 signs)", "",
               "Un-enrolled signs (100 per draw, 3 draws) plus non-sign motion; any 'sign' answer is a false accept. "
               "With 2 takes per sign the per-label acceptance threshold is calibrated from the takes "
               "(capped at %.2f, the 1-take default is %.2f); this row checks that having more takes does not loosen "
               "rejection. Caveat: dictionary second takes differ more from each other than a user's own takes do."
               % (cfg_full.thr_hi, cfg_full.default_thr), ""]
        rows = []
        for name, _ in OPEN_CONDS:
            e1, e2 = extra[(1, name)], extra[(2, name)]
            rows.append([name, pct(e1["accept_bad"]), top_absorbers(e1), pct(e2["accept_bad"]), top_absorbers(e2)])
        md += [table(rows, ["query", "false accept, 1 take", "absorbing labels (1 take)",
                            "false accept, 2 takes", "absorbing labels (2 takes)"]), ""]
    if hold:
        md += ["## Full Recogniser path (hold splitting) on un-enrolled signs, 30 enrolled signs", "",
               "Continuous stream through `Recognizer.push_frame` (segmentation + hold-split + matcher). "
               "'streams with a sign' = share of streams (40 un-enrolled signs) in which at least one sign was "
               "written (every one is a false accept); 'signs per stream' can exceed the share when a sign is "
               "split into fragments that each match an enrolled sign.", "",
               table([[k, pct(v[0]), f"{v[1]:.2f}"] for k, v in hold.items()],
                     ["stream", "streams with a sign", "signs per stream"]), ""]
    if keras:
        md += ["## Old Keras dictionary-model path (12 Family signs, its only vocabulary)", "",
               "Recogniser in model-only mode; ok = exactly one sign event, correct; bad = a wrong confident sign; "
               "none = no sign (clarify / unknown / silence). n = %d cross-take queries." % next(iter(keras.values()))["n"], ""]
        md += [table([[k, pct(v["ok"]), pct(v["bad"]), pct(v["none"])] for k, v in keras.items()],
                     ["query condition", "ok", "bad", "none"]), "",
               "The model was trained on take 1 of each sign (augmented), so cross-take is its honest test. "
               "(n is tiny: only two of the 12 Family signs have a second take.) ", ""]
        if keras_same:
            md += ["Same model, template-take as query (the take it was trained on) under the same stresses, "
                   "n = %d signs; an upper bound for the model:" % next(iter(keras_same.values()))["n"], "",
                   table([[k, pct(v["ok"]), pct(v["bad"]), pct(v["none"])] for k, v in keras_same.items()],
                         ["query condition", "ok", "bad", "none"]), ""]
        md += ["In real webcam use it collapses to 'Aunt'/'Brother'/'Lactation Brother' (domain gap, one "
               "dictionary signer), which is why it is now only an optional secondary path.", ""]
    md += ["## Speed", "",
           f"{sp[0]} signs / {sp[1]} templates (takes): enrolment (feature extraction + calibration) "
           f"{sp[2]:.2f} s, then **{sp[3]:.1f} ms mean / {sp[4]:.1f} ms max per decision** (made once per signed "
           "sign, not per frame; budget was < 30 ms).", ""]
    text = "\n".join(md)
    print(text)
    if not a.quick and not a.no_write:
        DOCS.parent.mkdir(exist_ok=True)
        head = DOCS.read_text(encoding="utf-8").split("\n<!-- BENCHMARK -->")[0] + "\n" if DOCS.exists() else ""
        DOCS.write_text(head + "<!-- BENCHMARK -->\n" + text + "\n", encoding="utf-8")
        print(f"\nwrote {DOCS}")


if __name__ == "__main__":
    main()
