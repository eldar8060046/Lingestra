"""Personal sign matcher: recognise signs from the user's own examples, immediately, no training.

Pipeline for every sequence (template take or live query), see docs/recognition.md
  1. cut the sign out of the raw landmarks (raise / lowering removed: features.trim_active);
  2. per-frame signing features (features.sign_features): hand shape, hand location vs shoulders and
     face, orientation, elbows, dominant hand; body-anchored, so camera shift / zoom / roll cancel;
  3. HOLD COLLAPSE: consecutive frames that differ by less than `eps` (our own frame cost) are merged,
     so a 1 s pause between key poses becomes one frame;
  4. ARC-LENGTH RESAMPLING: points at a constant amount of feature change (`step`), not constant
     time, so slow / fast / uneven signing yields the same point sequence;
  5. DTW (vectorised over all templates at once): the whole template must be consumed, the query
     may have leftover frames at both ends (open begin / open end), Sakoe-Chiba style band on the
     warp, step penalties, normalised by path length;
  6. open-set decision: per-label acceptance threshold calibrated from the user's own takes
     (leave-one-out distances, documented default for single takes) AND a margin over the second
     best label; otherwise 'clarify' (two close candidates) or 'unknown'.

Pure numpy; ~2-5 ms per decision for 50 signs x 3 takes.

CONTINUOUS SIGNING: `PersonalMatcher.decode_sequence` splits an episode in which the hands never came
down into several words (segmental spotting with subsequence DTW over fixed-step arc-length points, a
segmentation DP, per-segment open-set re-judging and a preference for the single-sign explanation);
see the section 'Continuous signing' below and docs/recognition.md.
"""
import os
from dataclasses import dataclass, field

import numpy as np

from lingestra import features as F
from lingestra.live import fastspot

BLOCKS = F.SIGN_BLOCKS
NB = len(BLOCKS)
_W = np.array([b[2] for b in BLOCKS], np.float32)
_T = np.array([b[3] for b in BLOCKS], np.float32)
_D = np.array([b[1] for b in BLOCKS], np.float32)
_SL = [F.BLOCK_SLICES[b[0]] for b in BLOCKS]


@dataclass
class MatcherConfig:
    eps: float = 0.06            # hold-collapse radius (frame-cost units; jitter is ~0.02/frame)
    n_points: int = 24           # points per sequence after arc-length resampling
    slack: int = 3               # open begin / end: free query points at each end
    band: int = 7                # Sakoe-Chiba half width (points)
    c_miss: float = 1.0          # cost of a block present in one frame but not the other
    step_pen: float = 0.08       # extra cost for non-diagonal DTW steps
    default_thr: float = 0.45    # acceptance distance when a label has a single take (docs/recognition.md)
    thr_margin: float = 1.35     # calibrated thr = margin * max leave-one-out distance
    thr_lo: float = 0.30
    thr_hi: float = 0.55         # ceiling: never looser than ~1.2x the single-take default (open-set safety)
    near_thr: float = 1.25       # a runner-up within this multiple of its own threshold still competes
    second_ratio: float = 0.85   # best must be <= this * second-best distance, else clarify
    sug_ratio: float = 2.0       # a 'clarify' also offers every sign within this multiple of ITS acceptance distance ...
    max_suggest: int = 8         # ... best first, at most this many (the panel adds Skip and Retry: 10 options at most)
    min_points: int = 2          # collapsed query with fewer key poses than this -> unknown
    trim: bool = True            # cut transitions from queries / templates inside the matcher
    collapse: bool = True        # False = ablation: no hold collapse / arc length (uniform time resampling)
    # ---- continuous signing (decode_sequence), see docs/recognition.md 'Continuous signing'
    spot_ds: float = 0.10        # arc-length step of the dense point sequences used for spotting
    spot_max_pts: int = 100      # a template longer than this (in dense points) is compressed to it
    spot_gate: float = 1.35      # a spotted candidate must be within this multiple of its label threshold
    spot_len: tuple = (0.55, 1.9)    # accepted span length / template length
    unexplained: float = 0.75    # DP cost of one dense point that no template explains (>> accept distance)
    unexplained_tail: float = 0.45   # ... at the open end of a still-running episode (a sign in progress is not 'unexplained')
    tail_pts: int = 30           # ... for at most this many points (about one long sign)
    seg_penalty: float = 1.5     # DP cost of one more segment: prefers the single-sign explanation
    max_gap_s: float = 0.30      # an unlabelled stretch up to this long between signs is a transition
    min_unknown_s: float = 0.6   # an unexplained stretch must be at least this long to be an 'unknown' sign
    transition_lin: float = 0.7  # ... and not one straight move (chord / path length of its key poses above this)
    whole_sure: float = 0.0      # the whole episode matching this well (distance / threshold) is ONE sign
    merge_k: float = 1.0         # neighbours merge when the joint span matches (sign) within k x their mean ratio
    seg_sure: float = 1.0        # a sign of a MULTI-segment explanation needs distance / threshold below this
    seg_sure_unk: float = 0.7    # ... below this when part of the episode matches nothing (fragment safety)
    edge_unknown_s: float = 1.0  # an unexplained stretch at the start / end of an episode must be this long ...
    transit_drop: float = 0.2    # ... and must not be the hands travelling to / from rest (height change, span units)
    same_gap_s: float = 0.12     # the same label twice, closer than this, is ONE sign (cyclic motion longer than its template)
    min_word_s: float = 0.45     # a spotted stretch shorter than this is a fragment, not a word
    min_seg_frames: int = 6
    fast: bool = True            # use the compiled multi-core spotting (live/fastspot.py) when numba is installed and warmed up


@dataclass
class MatchResult:
    kind: str                    # 'sign' | 'clarify' | 'unknown'
    label: str | None
    confidence: float
    candidates: list = field(default_factory=list)     # [(label, confidence)] best first
    distance: float = np.inf
    threshold: float = np.inf
    ms: float = 0.0


@dataclass
class _Seq:
    X: np.ndarray                # (L, D)
    M: np.ndarray                # (L, NB) bool
    n_keys: int = 0              # key poses after hold collapse (before resampling)
    Xc: np.ndarray | None = None     # hold-collapsed (not resampled) copy, for the dense spotting sequences
    Mc: np.ndarray | None = None


@dataclass
class Segment:
    """One word found in an episode by decode_sequence. start / end are frame indices (inclusive) into
    the episode that was decoded. kind: 'sign' (accepted label) | 'clarify' (two close candidates,
    `ambiguous`) | 'unknown' (matches nothing: the caller keeps the window of THIS segment only)."""
    label: str | None
    start: int
    end: int
    distance: float = np.inf
    confidence: float = 0.0
    runner_up: str | None = None
    ambiguous: bool = False
    kind: str = "sign"
    candidates: list = field(default_factory=list)
    ratio: float = np.inf        # distance / the label's acceptance threshold


# ----------------------------------------------------------------------------- frame cost
def frame_cost(Xa, Ma, Xb, Mb, c_miss=1.0):
    """Cost between frames (broadcast over leading dims by pairing a[i] with b[i]); (n,)."""
    num = 0.0
    den = 0.0
    for b in range(NB):
        sl = _SL[b]
        d = np.sqrt(((Xa[..., sl] - Xb[..., sl]) ** 2).mean(-1)) / _T[b]
        both = Ma[..., b] & Mb[..., b]
        one = Ma[..., b] ^ Mb[..., b]
        num = num + _W[b] * (np.where(both, d, 0.0) + np.where(one, c_miss, 0.0))
        den = den + _W[b] * (Ma[..., b] | Mb[..., b])
    return num / np.maximum(den, 1e-6)


# ----------------------------------------------------------------------------- sequence prep
def collapse_holds(X, M, eps, c_miss=1.0, return_ranges=False):
    """Merge runs of frames within `eps` of the run's first frame; returns group means
    (and, with return_ranges, the (n_groups, 2) first / last+1 frame of every group)."""
    n = len(X)
    if n == 0:
        return (X, M, np.zeros((0, 2), int)) if return_ranges else (X, M)
    # cost of frame i to frame i+k for k = 1..W in one vectorised pass (groups are short in motion; a long
    # hold is continued with direct comparisons below)
    W = 12
    band = np.full((n, W + 1), np.inf, np.float32)
    for k in range(1, min(W, n - 1) + 1):
        band[:n - k, k] = frame_cost(X[k:], M[k:], X[:-k], M[:-k], c_miss)
    over = band > eps                               # inf (beyond the end) counts as 'left the radius'
    starts = []
    a = 0
    while a < n:
        starts.append(a)
        row = over[a, 1:]
        if row.any():
            a = min(a + int(row.argmax()) + 1, n)
            continue
        w = 4 * W                                   # still within the radius after W frames (a hold): look further
        while True:
            hi = min(a + w, n)
            rest = hi - a
            c = frame_cost(X[a:hi], M[a:hi], np.broadcast_to(X[a], (rest, X.shape[1])),
                           np.broadcast_to(M[a], (rest, NB)), c_miss)
            ov = np.nonzero(c > eps)[0]
            if len(ov):
                a += int(ov[0])
                break
            if hi == n:
                a = n
                break
            w *= 4
    st = np.array(starts, int)
    en = np.append(st[1:], n)
    ln = en - st
    Xo = np.zeros((len(st), X.shape[1]), np.float32)
    Mo = np.zeros((len(st), NB), bool)
    for b in range(NB):
        sl = _SL[b]
        mb = M[:, b]
        cnt = np.add.reduceat(mb.astype(np.int64), st)
        sm = np.add.reduceat(X[:, sl] * mb[:, None], st, axis=0)
        Xo[:, sl] = np.where(cnt[:, None] > 0, sm / np.maximum(cnt, 1)[:, None], 0)
        Mo[:, b] = cnt >= ln / 2.0                # block present if present in half the group
    if return_ranges:
        return Xo, Mo, np.stack([st, en], 1)
    return Xo, Mo


def arc_resample(X, M, n_points, c_miss=1.0):
    """Resample the polyline X (L,D) to `n_points` points equally spaced in ARC LENGTH (feature
    space, frame-cost units): every step covers the same amount of change, whatever the timing."""
    L = len(X)
    if L == 0:
        return X, M
    if L == 1:
        return np.repeat(X, n_points, 0), np.repeat(M, n_points, 0)
    seg = frame_cost(X[1:], M[1:], X[:-1], M[:-1], c_miss)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1]
    if total < 1e-6:
        return np.repeat(X[:1], n_points, 0), np.repeat(M[:1], n_points, 0)
    pos = np.linspace(0.0, total, n_points)
    hi = np.clip(np.searchsorted(s, pos, side="right"), 1, L - 1)
    lo = hi - 1
    w = np.clip((pos - s[lo]) / np.maximum(s[hi] - s[lo], 1e-9), 0, 1).astype(np.float32)[:, None]
    both = np.zeros((len(pos), X.shape[1]), bool)
    for b in range(NB):
        both[:, _SL[b]] = (M[lo, b] & M[hi, b])[:, None]
    near = np.where(w[:, 0] < 0.5, lo, hi)
    out = np.where(both, X[lo] * (1 - w) + X[hi] * w, X[near])
    out_m = np.where(w < 0.5, M[lo], M[hi])
    return out.astype(np.float32), out_m


def prepare(raw, cfg: MatcherConfig, fps=25.0, trim=None):
    """Raw landmarks (n,67,3) -> _Seq (hold-collapsed, arc-length resampled)."""
    raw = np.asarray(raw, np.float32)
    if raw.ndim != 3 or len(raw) == 0:
        return _Seq(np.zeros((0, F.SIGN_DIM), np.float32), np.zeros((0, NB), bool))
    if cfg.trim if trim is None else trim:
        raw = F.trim_active(raw, fps=fps)
    X, M = F.sign_features(raw, fps)
    return prepare_features(X, M, cfg)


def prepare_features(X, M, cfg: MatcherConfig):
    """Per-frame features -> _Seq (hold-collapsed, arc-length resampled); the second half of prepare()."""
    nk = len(X)
    Xc = Mc = None
    if cfg.collapse:
        X, M = collapse_holds(X, M, cfg.eps, cfg.c_miss)
        nk = len(X)
        Xc, Mc = X, M
        X, M = arc_resample(X, M, cfg.n_points, cfg.c_miss)
    else:                                   # ablation: plain DTW, time-normalised to the same length
        X, M = _time_resample(X, M, cfg.n_points)
    return _Seq(X, M, nk, Xc, Mc)


def _time_resample(X, M, n_points):
    """Ablation: uniform resampling in TIME to n_points (no hold collapse, no arc length)."""
    n = len(X)
    if n == 0:
        return X, M
    idx = np.minimum(np.round(np.linspace(0, n - 1, n_points)).astype(int), n - 1)
    return X[idx], M[idx]


# ----------------------------------------------------------------------------- batched DTW
def _stack(seqs):
    L = max(len(s.X) for s in seqs)
    N = len(seqs)
    X = np.zeros((N, L, F.SIGN_DIM), np.float32)
    M = np.zeros((N, L, NB), bool)
    lens = np.zeros(N, int)
    for i, s in enumerate(seqs):
        k = len(s.X)
        X[i, :k], M[i, :k], lens[i] = s.X, s.M, k
    return X, M, lens


def cost_matrices(TX, TM, Q, c_miss):
    """Local costs (N, Lt, Lq) between every template frame and every query frame.

    Same value as frame_cost() for every pair. The mask terms (a block present in one frame only costs
    `c_miss`, absent from both costs nothing and does not count) are folded into two small matrix
    products, and the per-block distances are built in place, so the (N, Lt, Lq) arrays are touched
    ~8 times per block instead of ~15."""
    N, Lt, _ = TX.shape
    Lq = len(Q.X)
    ta = TM.reshape(N * Lt, NB).astype(np.float32)           # (N*Lt, NB)
    qa = Q.M.astype(np.float32)                               # (Lq, NB)
    taw = ta * _W
    both_w = taw @ qa.T                                       # sum_b W_b * present_in_both
    ta_w = taw.sum(1)[:, None]
    qa_w = (qa * _W).sum(1)[None, :]
    den = ta_w + qa_w - both_w                                # sum_b W_b * present_in_either
    num = np.float32(c_miss) * (ta_w + qa_w - 2.0 * both_w)   # one-sided blocks
    for b in range(NB):
        sl = _SL[b]
        scale = np.float32(1.0 / (np.sqrt(sl.stop - sl.start) * _T[b]))
        A = TX[:, :, sl].reshape(N * Lt, -1) * scale
        B = Q.X[:, sl] * scale
        S = A @ B.T
        S *= -2.0
        S += (A * A).sum(1)[:, None]
        S += (B * B).sum(1)[None, :]
        np.maximum(S, 0.0, out=S)
        np.sqrt(S, out=S)
        S *= taw[:, b:b + 1]                                  # W_b and 'template has the block'
        S *= qa[None, :, b]                                   # 'query has the block'
        num += S
    num /= np.maximum(den, 1e-6)
    return num.reshape(N, Lt, Lq)


def dtw_batch(TX, TM, lens, Q, cfg: MatcherConfig):
    """DTW of every template against query Q, all templates at once.

    Open begin / open end: the template may start at any of the first `slack` query points and end
    at any of the last `slack` ones, so leftover transition frames at either end of the query are
    skipped for free (but not more than that: a template cannot match a fragment of a long query).
    Steps from template row i-1 to row i: query j -> j, j-1 -> j (diagonal), j-2 -> j (one query
    point skipped); the two non-diagonal steps cost `step_pen`. Sakoe-Chiba band |i - j| <= band
    (+ slack). Normalised by path length. Returns (N,) distances."""
    C = cost_matrices(TX, TM, Q, cfg.c_miss)
    N, Lt, Lq = C.shape
    INF = np.float32(1e9)
    slack = int(cfg.slack)
    j = np.arange(Lq)[None, :]
    start_ok = j <= slack
    D = np.where(start_ok, C[:, 0, :], INF)
    P = np.ones((N, Lq), np.float32)
    pen = np.float32(cfg.step_pen)
    w = cfg.band + slack
    inf1 = np.full((N, 1), INF, np.float32)
    inf2 = np.full((N, 2), INF, np.float32)
    one1 = np.ones((N, 1), np.float32)
    one2 = np.ones((N, 2), np.float32)
    for i in range(1, Lt):
        st = np.stack([D + pen, np.concatenate([inf1, D[:, :-1]], 1),
                       np.concatenate([inf2, D[:, :-2]], 1) + pen])
        ps = np.stack([P, np.concatenate([one1, P[:, :-1]], 1), np.concatenate([one2, P[:, :-2]], 1)])
        k = st.argmin(0)[None]
        Dn = np.take_along_axis(st, k, 0)[0] + C[:, i, :]
        P = np.take_along_axis(ps, k, 0)[0] + 1
        D = np.where(np.abs(j - i) > w, INF, Dn)
    lo = max(Lq - 1 - slack, 0)
    return (D[:, lo:] / P[:, lo:]).min(1)


# ----------------------------------------------------------------------------- continuous signing
# decode_sequence() finds SEVERAL signs in one episode (hands never lowered). Outline (details in
# docs/recognition.md, 'Continuous signing'):
#   1. per-frame features of the episode, once (hand 'used' / dominant hand are decided per window or
#      per segment, because they depend on the stretch they are computed over);
#   2. hold collapse + arc-length resampling at a FIXED step (spot_ds) of the episode and of every
#      template take: the dense point sequences are speed / pause invariant, and every point is a
#      potential segment boundary (so no separate boundary proposals are needed);
#   3. SPOTTING: subsequence DTW (template fully consumed, free start and end inside the episode) of
#      all templates at once -> for every end point the best start and the normalised distance;
#   4. segmentation DP over the points: cost of a partition = sum(span x distance) + seg_penalty per
#      segment + `unexplained` per point no template explains (transitions, unknown signs);
#   5. every chosen segment is re-judged with the ordinary single-sign matcher (24 points, calibrated
#      per-label threshold + margin over its own runner-up): sign / clarify / unknown;
#   6. single-sign preference: neighbouring segments are merged back when the joint span matches as
#      ONE sign about as well (merge_k), and the whole episode matching confidently short-circuits.
@dataclass
class _Episode:
    raw: np.ndarray
    X: np.ndarray                # (n, D) per-frame features, both hands filled wherever a wrist was seen
    M: np.ndarray                # (n, NB)
    hh: np.ndarray               # (n, 2) wrist heights [L, R]
    fps: float = 25.0

    @property
    def n(self):
        return len(self.X)


_BI = {b[0]: i for i, b in enumerate(BLOCKS)}
_HAND_BLOCKS = (("locL", "shpL", "oriL"), ("locR", "shpR", "oriR"))


def episode_features(raw, fps=25.0) -> _Episode:
    raw = np.asarray(raw, np.float32)
    X, M, ex = F.sign_features(raw, fps, extra=True)
    return _Episode(raw, X, M, ex["hh"], fps)


def segment_xm(ep: _Episode, a, b):
    """Features of frames [a, b) as if that stretch had been featurised on its own (used-hand rule and
    dominant hand recomputed for the stretch; identical to sign_features of the same frames up to
    smoothing at the cut)."""
    X = ep.X[a:b].copy()
    M = ep.M[a:b].copy()
    for side, names in enumerate(_HAND_BLOCKS):
        if not F.hand_used(ep.hh[a:b, side]):
            for nm in names:
                X[:, F.BLOCK_SLICES[nm]] = 0
                M[:, _BI[nm]] = False
    lp = F._path_len(X[:, F.BLOCK_SLICES["locL"]][:, :2], M[:, _BI["locL"]])
    rp = F._path_len(X[:, F.BLOCK_SLICES["locR"]][:, :2], M[:, _BI["locR"]])
    X[:, F.BLOCK_SLICES["dom"]] = (rp - lp) / (rp + lp + 0.3)
    return X, M


def _window_mean(v, half):
    """Centred moving average (edges shrink the window)."""
    c = np.concatenate([[0.0], np.cumsum(v, dtype=np.float64)])
    n = len(v)
    lo = np.clip(np.arange(n) - half, 0, n)
    hi = np.clip(np.arange(n) + half + 1, 0, n)
    return (c[hi] - c[lo]) / np.maximum(hi - lo, 1)


def local_xm(ep: _Episode, half):
    """Per-frame features for SPOTTING: the 'hand used' rule and the dominant hand are decided in a
    window of +-`half` frames (about one sign long) instead of per sequence."""
    X = ep.X.copy()
    M = ep.M.copy()
    inc = []
    for side, names in enumerate(_HAND_BLOCKS):
        used = _window_mean((ep.hh[:, side] >= 2 * F.REST_HEIGHT).astype(np.float64), half) >= F.USED_FRAC
        for nm in names:
            M[:, _BI[nm]] &= used
            X[:, F.BLOCK_SLICES[nm]] *= M[:, _BI[nm]][:, None]
        loc = X[:, F.BLOCK_SLICES[names[0]]][:, :2]
        m = M[:, _BI[names[0]]]
        d = np.zeros(len(X))
        if len(X) > 1:
            d[1:] = np.linalg.norm(np.diff(loc, axis=0), axis=1) * (m[1:] & m[:-1])
        inc.append(d)
    lp, rp = (_window_mean(d, half) * (2 * half + 1) for d in inc)
    X[:, F.BLOCK_SLICES["dom"]] = ((rp - lp) / (rp + lp + 0.3))[:, None]
    return X, M


def _arc_step(X, M, ds, c_miss, max_pts=None, groups=False):
    """Arc-length resampling at a FIXED step `ds` (not a fixed point count). Returns X, M and, with
    `groups`, the index of the input row nearest to every output point (to map points back to frames)."""
    L = len(X)
    if L == 0:
        return X, M, (np.zeros(0, int) if groups else None)
    if L == 1:
        return X, M, (np.zeros(1, int) if groups else None)
    seg = frame_cost(X[1:], M[1:], X[:-1], M[:-1], c_miss)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = float(s[-1])
    n_pts = int(np.clip(round(total / ds) + 1, 2, max_pts or 10 ** 9))
    pos = np.linspace(0.0, total, n_pts)
    hi = np.clip(np.searchsorted(s, pos, side="right"), 1, L - 1)
    lo = hi - 1
    w = np.clip((pos - s[lo]) / np.maximum(s[hi] - s[lo], 1e-9), 0, 1).astype(np.float32)[:, None]
    both = np.zeros((n_pts, X.shape[1]), bool)
    for b in range(NB):
        both[:, _SL[b]] = (M[lo, b] & M[hi, b])[:, None]
    near = np.where(w[:, 0] < 0.5, lo, hi)
    out = np.where(both, X[lo] * (1 - w) + X[hi] * w, X[near]).astype(np.float32)
    out_m = np.where(w < 0.5, M[lo], M[hi])
    return out, out_m, (near if groups else None)


def spot_batch(DX, DM, dlens, Q: _Seq, cfg: MatcherConfig, buckets=4):
    """Subsequence DTW of every dense template against the dense episode Q (one pass per length bucket).

    Same local cost / step set / step penalty as dtw_batch, but the template may start and end
    ANYWHERE in the episode. Returns (dist (N, Lq), start (N, Lq)): the normalised distance of the best
    alignment of template k that ENDS at point j, and the point where it starts. Templates are
    processed in buckets of similar length so short ones do not pay for the longest."""
    if cfg.fast and fastspot.ready():
        return fastspot.spot_batch(DX, DM, dlens, Q, cfg)          # same maths, compiled, spread over several cores
    N = len(DX)
    Lq = len(Q.X)
    out_d = np.full((N, Lq), 1e9, np.float32)
    out_s = np.zeros((N, Lq), int)
    order = np.argsort(dlens, kind="stable")
    for idx in np.array_split(order, min(buckets, N)):
        if not len(idx):
            continue
        Lt = int(dlens[idx].max())
        d, st = _spot_chunk(DX[idx, :Lt], DM[idx, :Lt], dlens[idx], Q, cfg)
        out_d[idx], out_s[idx] = d, st
    return out_d, out_s


def _spot_chunk(DX, DM, dlens, Q, cfg):
    C = cost_matrices(DX, DM, Q, cfg.c_miss)
    N, Lt, Lq = C.shape
    INF = np.float32(1e9)
    pen = np.float32(cfg.step_pen)
    D = C[:, 0, :].copy()
    S = np.broadcast_to(np.arange(Lq)[None, :], (N, Lq)).copy()
    P = np.ones((N, Lq), np.float32)
    out_d = np.full((N, Lq), INF, np.float32)
    out_s = np.zeros((N, Lq), int)
    inf1 = np.full((N, 1), INF, np.float32)
    inf2 = np.full((N, 2), INF, np.float32)
    s1 = np.zeros((N, 1), int)
    s2 = np.zeros((N, 2), int)
    one1 = np.ones((N, 1), np.float32)
    one2 = np.ones((N, 2), np.float32)
    done = dlens - 1
    for i in range(1, Lt + 1):
        fin = done == i - 1
        if fin.any():
            out_d[fin] = D[fin] / P[fin]
            out_s[fin] = S[fin]
        if i == Lt:
            break
        st = np.stack([D + pen, np.concatenate([inf1, D[:, :-1]], 1),
                       np.concatenate([inf2, D[:, :-2]], 1) + pen])
        ps = np.stack([P, np.concatenate([one1, P[:, :-1]], 1), np.concatenate([one2, P[:, :-2]], 1)])
        ss = np.stack([S, np.concatenate([s1, S[:, :-1]], 1), np.concatenate([s2, S[:, :-2]], 1)])
        k = st.argmin(0)[None]
        D = np.take_along_axis(st, k, 0)[0] + C[:, i, :]
        P = np.take_along_axis(ps, k, 0)[0] + 1
        S = np.take_along_axis(ss, k, 0)[0]
    return out_d, out_s


# ----------------------------------------------------------------------------- the matcher
class PersonalMatcher:
    """Holds the prepared templates; `classify(raw)` is one decision per finished sign."""

    def __init__(self, cfg: MatcherConfig | None = None):
        self.cfg = cfg or MatcherConfig()
        self._cache = {}            # (id, take, len, checksum) -> _Seq
        self.clear()

    def clear(self):
        self.labels = []            # label per label index
        self.take_label = np.zeros(0, int)
        self.thr = np.zeros(0)      # per-label acceptance distance
        self._TX = self._TM = None
        self._lens = np.zeros(0, int)
        self.n_takes = 0
        self.calibration = {}
        self._DX = self._DM = None            # dense templates for continuous spotting
        self._dlens = np.zeros(0, int)
        self.last_decode_ms = 0.0
        self._act = None                      # decode_sequence: template takes worth comparing with (pruning)
        self.debug = None                     # set to a list to collect decode_sequence traces (tests / tuning)

    @property
    def ready(self):
        return self.n_takes > 0

    def set_templates(self, templates):
        """templates: iterable of dict(id, label, frames=[raw (n,67,3), ...], fps optional)."""
        cfg = self.cfg
        seqs, tl, labels, seen = [], [], [], {}
        keep = {}
        for t in templates or []:
            lab = str(t.get("label"))
            li = seen.setdefault(lab, len(labels))
            if li == len(labels):
                labels.append(lab)
            fpl = t.get("fps", 25.0)
            for ti, fr in enumerate(t.get("frames") or []):
                fr = np.asarray(fr)
                if fr.ndim != 3 or len(fr) < 4:
                    continue
                fps = float(fpl[ti] if isinstance(fpl, (list, tuple, np.ndarray)) else fpl)
                key = (str(t.get("id")), ti, len(fr), float(np.nansum(fr[::7, 15:17, :2])), fps)
                sq = self._cache.get(key)
                if sq is None:
                    sq = prepare(fr, cfg, fps)
                keep[key] = sq
                if sq.n_keys >= cfg.min_points:
                    seqs.append(sq)
                    tl.append(li)
        self._cache = keep
        self.clear()
        if not seqs:
            return
        self.labels = labels
        self.take_label = np.array(tl, int)
        self._TX, self._TM, self._lens = _stack(seqs)
        self._seqs = seqs
        self.n_takes = len(seqs)
        self._calibrate()
        self._build_dense()
        fastspot.warm_up(background=not os.environ.get("LINGESTRA_SYNC_WARM"))   # compile the multi-core kernel (cached on disk)

    def _calibrate(self):
        """Per-label acceptance distance from leave-one-out distances among the label's own takes."""
        cfg = self.cfg
        thr = np.full(len(self.labels), cfg.default_thr)
        loo = {}
        for li in range(len(self.labels)):
            idx = np.nonzero(self.take_label == li)[0]
            if len(idx) < 2:
                continue
            ds = []
            for qi in idx:
                others = idx[idx != qi]
                d = dtw_batch(self._TX[others], self._TM[others], self._lens[others], self._seqs[qi], cfg)
                ds.append(float(d.min()))
            loo[self.labels[li]] = ds
            thr[li] = float(np.clip(cfg.thr_margin * max(ds), cfg.thr_lo, cfg.thr_hi))
        self.thr = thr
        self.calibration = {"loo": loo, "default_thr": cfg.default_thr}


    # ------------------------------------------------------------------ continuous signing
    def _build_dense(self):
        cfg = self.cfg
        self._DX = self._DM = None
        dx, dm, dl = [], [], []
        for sq in self._seqs:
            if sq.Xc is None:
                dx = []
                break
            x, m, _ = _arc_step(sq.Xc, sq.Mc, cfg.spot_ds, cfg.c_miss, cfg.spot_max_pts)
            dx.append(x)
            dm.append(m)
            dl.append(len(x))
        if not dx:
            return
        L = max(dl)
        N = len(dx)
        self._DX = np.zeros((N, L, F.SIGN_DIM), np.float32)
        self._DM = np.zeros((N, L, NB), bool)
        for i, (x, m) in enumerate(zip(dx, dm)):
            self._DX[i, :len(x)], self._DM[i, :len(x)] = x, m
        self._dlens = np.array(dl, int)

    def _span(self, ep: _Episode, a, b, strictness):
        """Single-sign judgement of frames [a, b) of an episode: (MatchResult, label distances)."""
        if b - a < self.cfg.min_seg_frames:
            return MatchResult("unknown", None, 0.0), np.full(len(self.labels), np.inf)
        X, M = segment_xm(ep, a, b)
        q = prepare_features(X, M, self.cfg)
        d = self._label_distances(q, self._act)
        return self.decide(d, strictness), d

    @staticmethod
    def _ratio(res):
        return res.distance / max(res.threshold, 1e-9) if np.isfinite(res.distance) else np.inf

    def _segment(self, res, a, b):
        """MatchResult of frames [a, b) -> Segment (end inclusive)."""
        cands = list(res.candidates)
        ru = cands[1][0] if len(cands) > 1 else None
        if res.kind == "sign":
            return Segment(res.label, a, b - 1, res.distance, res.confidence, ru, False, "sign", cands, self._ratio(res))
        if res.kind == "clarify":
            return Segment(res.label, a, b - 1, res.distance, res.confidence, ru, True, "clarify", cands, self._ratio(res))
        return Segment(None, a, b - 1, res.distance, 0.0, cands[0][0] if cands else None, False, "unknown", cands,
                       self._ratio(res))

    def _partition(self, ep: _Episode, strictness, final=True):
        """Spotting + segmentation DP. Returns [(a, b, label_index, distance)] with frame spans [a, b)."""
        cfg = self.cfg
        n = ep.n
        half = int(round(0.8 * ep.fps))
        X, M = local_xm(ep, half)
        Xc, Mc, rg = collapse_holds(X, M, cfg.eps, cfg.c_miss, return_ranges=True)
        Xd, Md, grp = _arc_step(Xc, Mc, cfg.spot_ds, cfg.c_miss, None, True)
        Lq = len(Xd)
        if Lq < 3:
            return []
        Q = _Seq(Xd, Md, len(Xc))
        dist, start = spot_batch(self._DX, self._DM, self._dlens, Q, cfg)
        j = np.arange(Lq)[None, :]
        span = j - start + 1
        lo, hi = cfg.spot_len
        okm = (span >= np.maximum(lo * self._dlens[:, None], 3)) & (span <= hi * self._dlens[:, None] + 2)
        dist = np.where(okm, dist, np.inf)
        nl = len(self.labels)
        Dl = np.full((nl, Lq), np.inf, np.float32)
        np.minimum.at(Dl, self.take_label, dist)
        Sl = np.zeros((nl, Lq), int)
        ki, ji = np.nonzero((dist == Dl[self.take_label]) & np.isfinite(dist))
        Sl[self.take_label[ki], ji] = start[ki, ji]
        gate = (cfg.spot_gate * self.thr * strictness)[:, None]
        valid = Dl <= gate
        # pruning for the per-segment re-judging: only labels that matched SOME stretch of the episode
        # (within spot_gate x their threshold) or are among the 12 nearest are compared in full
        keep = valid.any(1)
        keep[np.argsort(Dl.min(1))[:12]] = True
        self._act = np.nonzero(keep[self.take_label])[0]
        G = np.zeros(Lq + 1)
        back = [None] * (Lq + 1)
        beta, cu = cfg.seg_penalty, cfg.unexplained
        for p in range(1, Lq + 1):
            e = p - 1
            best, bk = G[p - 1] + cu, None
            v = np.nonzero(valid[:, e])[0]
            if len(v):
                s0 = Sl[v, e]
                cost = G[s0] + (e - s0 + 1) * Dl[v, e] + beta
                m = int(np.argmin(cost))
                if cost[m] < best:
                    best, bk = float(cost[m]), (int(v[m]), int(s0[m]))
            G[p], back[p] = best, bk
        p = Lq
        if not final:                        # episode still running: the last stretch may be a sign in progress
            rem = Lq - np.arange(1, Lq + 1)             # at most one sign's worth of points gets the cheaper rate
            tot = G[1:] + np.minimum(rem, cfg.tail_pts) * cfg.unexplained_tail + np.maximum(rem - cfg.tail_pts, 0) * cu
            p = 1 + int(np.argmin(tot))
        out = []
        while p > 0:
            bk = back[p]
            if bk is None:
                p -= 1
            else:
                li, s0 = bk
                out.append((s0, p - 1, li, float(Dl[li, p - 1])))
                p = s0
        out.reverse()
        res, prev_end = [], 0
        for s0, e0, li, d in out:
            fa = int(rg[grp[s0], 0])            # a segment owns the whole hold-group of its first / last point
            fb = int(rg[grp[e0], 1])
            fa = max(fa, prev_end)
            fb = min(max(fb, fa + 1), n)
            res.append((fa, fb, li, d))
            prev_end = fb
        return res

    def decode_sequence(self, episode, fps=25.0, strictness=1.0, final=True):
        """Words of ONE episode (hands up the whole time) -> list[Segment], in time order.

        `episode`: raw landmarks (n, 67, 3) or an _Episode from episode_features(). Every Segment
        has kind 'sign' (accepted), 'clarify' (this segment only is ambiguous) or 'unknown' (matches
        nothing). Frames between segments (<= max_gap_s) are transitions and carry no word.
        final=False: the episode is still running, so its last stretch may be a sign in progress and is
        not forced into fragments of other signs (the caller commits only words that are over)."""
        import time
        t0 = time.perf_counter()
        cfg = self.cfg
        self._act = None
        ep = episode if isinstance(episode, _Episode) else episode_features(episode, fps)
        n = ep.n
        out = []
        if self.ready and n >= cfg.min_seg_frames:
            out = self._decode(ep, n, strictness, final)
        self.last_decode_ms = (time.perf_counter() - t0) * 1000
        return out

    def _decode(self, ep, n, strictness, final=True):
        cfg = self.cfg
        memo = {}

        def span(a, b):
            if (a, b) not in memo:
                memo[(a, b)] = self._span(ep, a, b, strictness)[0]
            return memo[(a, b)]

        def whole_seg():
            return self._segment(span(0, n), 0, n)

        if cfg.whole_sure > 0:
            w = span(0, n)
            if w.kind == "sign" and self._ratio(w) <= cfg.whole_sure:
                return [self._segment(w, 0, n)]
        if self._DX is None:
            return [whole_seg()]
        part = self._partition(ep, strictness, final)
        if not part:
            return [whole_seg()] if final else []
        gap_f = cfg.max_gap_s * ep.fps
        min_unk = max(gap_f, cfg.min_unknown_s * ep.fps)

        def unknown_gap(a, b):
            return Segment(None, a, b - 1, np.inf, 0.0, None, False, "unknown", [], 1.5)

        hmax = ep.hh.max(1)

        def transit(a, b, leading):
            """Leftover at an end of the episode: hands still travelling to / from rest (the tracker cuts
            the fast part of the raise / lowering; a slow tail can remain). Not a sign."""
            if b - a < 3:
                return True
            lo, hi = float(np.median(hmax[a:a + 3])), float(np.median(hmax[b - 3:b]))
            return (hi - lo if leading else lo - hi) >= cfg.transit_drop

        def straight(a, b):
            """A single straight move through feature space (chord ~ path length): hands travelling
            from one sign to the next, not a sign of their own."""
            if b - a < 3:
                return True
            X, M = segment_xm(ep, a, b)
            Xc, Mc = collapse_holds(X, M, cfg.eps, cfg.c_miss)
            if len(Xc) < 3:
                return True
            path = float(frame_cost(Xc[1:], Mc[1:], Xc[:-1], Mc[:-1], cfg.c_miss).sum())
            chord = float(frame_cost(Xc[-1:], Mc[-1:], Xc[:1], Mc[:1], cfg.c_miss)[0])
            return chord >= cfg.transition_lin * max(path, 1e-6)

        edge_min = max(min_unk, cfg.edge_unknown_s * ep.fps)
        segs = []
        cur = 0
        for fa, fb, li, d in part:
            if (fa - cur >= (edge_min if not segs else min_unk) and not (not segs and transit(cur, fa, True))
                    and not straight(cur, fa)):
                segs.append(unknown_gap(cur, fa))       # an unexplained stretch: an unknown sign
            if fb - fa < cfg.min_word_s * ep.fps:           # nobody signs a word in under ~0.45 s: a fragment
                continue
            sg = self._segment(span(fa, fb), fa, fb)
            if sg.kind == "unknown" and fb - fa < min_unk:  # a short leftover is a transition, not a sign
                continue
            segs.append(sg)
            cur = fb
        if n - cur >= edge_min and not transit(cur, n, False) and not straight(cur, n):
            segs.append(unknown_gap(cur, n))
        if self.debug is not None:
            self.debug.append(("segments", [(g.label, g.kind, g.start, g.end, round(g.ratio, 2)) for g in segs]))
        # single-sign preference: merge neighbours that match better as ONE sign
        guard = 0
        merged = True
        while merged and len(segs) > 1 and guard < 8:
            merged = False
            guard += 1
            for i in range(len(segs) - 1):
                a, b = segs[i], segs[i + 1]
                if b.start - a.end - 1 > gap_f:
                    continue
                res = span(a.start, b.end + 1)
                if res.kind != "sign":
                    continue
                refs = [x.ratio for x in (a, b) if x.kind in ("sign", "clarify")]
                rp = float(np.mean(refs)) if refs else 2.0       # two stretches that match nothing: any sign beats them
                if self.debug is not None:
                    self.debug.append(("merge?", a.label, b.label, res.label, round(self._ratio(res), 2), round(rp, 2)))
                if self._ratio(res) <= cfg.merge_k * rp:
                    segs[i:i + 2] = [self._segment(res, a.start, b.end + 1)]
                    merged = True
                    break
        # the same label twice with no transition between is one (cyclic) sign performed longer than its template
        i = 0
        while i < len(segs) - 1:
            a, b = segs[i], segs[i + 1]
            if (a.kind == b.kind == "sign" and a.label == b.label
                    and b.start - a.end - 1 <= cfg.same_gap_s * ep.fps):
                best = a if a.ratio <= b.ratio else b
                segs[i:i + 2] = [Segment(a.label, a.start, b.end, best.distance, best.confidence, best.runner_up,
                                         False, "sign", best.candidates, best.ratio)]
            else:
                i += 1
        # ... and finally the whole episode against whatever is left of the explanation
        w = span(0, n)
        if w.kind == "sign":
            refs = [x.ratio for x in segs if x.kind in ("sign", "clarify")]
            if self._ratio(w) <= cfg.merge_k * (float(np.mean(refs)) if refs else 2.0):
                return [self._segment(w, 0, n)]
        if len(segs) > 1:
            lim = cfg.seg_sure_unk if any(sg.kind == "unknown" for sg in segs) else cfg.seg_sure
            for i, sg in enumerate(segs):                   # multi-segment explanation: stricter per sign
                if sg.kind == "sign" and sg.ratio > lim:
                    segs[i] = Segment(None, sg.start, sg.end, sg.distance, 0.0, sg.label, False, "unknown",
                                      sg.candidates, sg.ratio)
            # neighbouring stretches that match nothing are ONE unknown sign, not several
            i = 0
            while i < len(segs) - 1:
                a, b = segs[i], segs[i + 1]
                if a.kind == "unknown" and b.kind == "unknown" and b.start - a.end - 1 <= gap_f:
                    segs[i:i + 2] = [Segment(None, a.start, b.end, min(a.distance, b.distance), 0.0, None, False,
                                             "unknown", [], min(a.ratio, b.ratio))]
                else:
                    i += 1
        return segs or [whole_seg()]

    def n_keys(self, raw, fps=25.0):
        """Number of distinct key poses (after hold collapse) of a clip."""
        return prepare(raw, self.cfg, fps, False).n_keys

    # ------------------------------------------------------------------ decision
    def distances(self, raw, fps=25.0, trim=None):
        """(label_distances (n_labels,), n_points) for a query clip."""
        q = prepare(raw, self.cfg, fps, trim)
        return self._label_distances(q), q.n_keys

    def _label_distances(self, q, takes=None):
        """Distance of the query to every label (min over its takes); `takes` = index array to compare
        with only those template takes (the others get inf)."""
        if q.n_keys < self.cfg.min_points or not self.ready:
            return np.full(len(self.labels), np.inf)
        out = np.full(len(self.labels), np.inf)
        if takes is None:
            d = dtw_batch(self._TX, self._TM, self._lens, q, self.cfg)
            np.minimum.at(out, self.take_label, d)
        else:
            d = dtw_batch(self._TX[takes], self._TM[takes], self._lens[takes], q, self.cfg)
            np.minimum.at(out, self.take_label[takes], d)
        return out

    def classify(self, raw, fps=25.0, strictness=1.0, trim=None) -> MatchResult:
        import time
        t0 = time.perf_counter()
        if not self.ready:
            return MatchResult("unknown", None, 0.0)
        d, npts = self.distances(raw, fps, trim)
        res = self.decide(d, strictness)
        res.ms = (time.perf_counter() - t0) * 1000
        return res

    def decide(self, d, strictness=1.0) -> MatchResult:
        """Open-set decision from per-label distances."""
        cfg = self.cfg
        thr = self.thr * float(strictness)
        if not np.isfinite(d).any():
            return MatchResult("unknown", None, 0.0)
        r = d / thr
        order = np.argsort(d)
        b = int(order[0])
        conf = lambda i: float(np.clip(1.0 - 0.5 * r[i], 0.0, 1.0))     # noqa: E731
        within = [int(i) for i in order if r[i] <= 1.0]
        cands = [(self.labels[int(i)], conf(int(i))) for i in order[:3] if np.isfinite(d[i])]
        if not within:
            return MatchResult("unknown", None, 0.0, cands, float(d[b]), float(thr[b]))
        # the runner-up competes if it is within threshold OR just over it (no hard cliff at the threshold)
        rivals = [int(i) for i in order[1:] if r[i] <= cfg.near_thr and np.isfinite(d[i])]
        j = within[1] if len(within) > 1 else (rivals[0] if rivals else None)
        if j is None or d[within[0]] <= cfg.second_ratio * d[j]:
            i = within[0]
            margin = 1.0 if j is None else min((d[j] / max(d[i], 1e-6) - 1.0) / 0.5, 1.0)
            c = conf(i) * (0.8 + 0.2 * margin) if j is not None else conf(i)
            c = float(np.clip(c, 0.0, 1.0))
            return MatchResult("sign", self.labels[i], c, cands, float(d[i]), float(thr[i]))
        i = within[0]
        first = [i, j]
        similar = [int(k) for k in order if int(k) not in first and np.isfinite(d[k]) and r[k] <= cfg.sug_ratio]
        ids = (first + similar)[:max(cfg.max_suggest, 2)]
        cl = [(self.labels[k], conf(k) * (0.8 if k in first else 1.0)) for k in ids]
        return MatchResult("clarify", self.labels[i], cl[0][1], cl, float(d[i]), float(thr[i]))


def decode_sequence(episode, templates, cfg: MatcherConfig | None = None, fps=25.0, strictness=1.0):
    """Module-level convenience for PersonalMatcher.decode_sequence.

    `episode`: raw landmarks (n, 67, 3) or episode_features(...); `templates`: a ready PersonalMatcher
    (its own cfg is used) or an iterable of template dicts as for PersonalMatcher.set_templates."""
    m = templates if isinstance(templates, PersonalMatcher) else None
    if m is None:
        m = PersonalMatcher(cfg)
        m.set_templates(templates)
    return m.decode_sequence(episode, fps=fps, strictness=strictness)
