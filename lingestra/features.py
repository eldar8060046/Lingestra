"""Landmark extraction + speed-invariant preprocessing shared by training and the live app.

Speed handling (the same code path is used offline and live, so features always match):
  1. trim_active()     - cut idle frames (hands at rest) so only the sign itself remains;
  2. resample()        - linearly resample the trimmed clip to a fixed T frames, which removes
                         overall signing speed;
  3. time_warp()       - training augmentation: random non-uniform speed profiles, so the model
                         also tolerates "fast start / slow end" style variation;
  4. multi-scale windows at inference time live in lingestra/live/recognizer.py.
"""
import warnings
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
T = 48                      # frames per normalised clip
N_POSE = 25                 # pose landmarks 0..24 (head, arms, torso)
N_PTS = N_POSE + 21 + 21    # + left hand + right hand
FEAT = N_PTS * 3            # per-frame feature size (x, y, z)
L_SH, R_SH = 11, 12         # shoulder indices
L_WR, R_WR = 15, 16         # wrist indices
L_HIP, R_HIP = 23, 24


class LandmarkExtractor:
    """Runs MediaPipe pose + hand landmarkers (Tasks API) and returns per-frame arrays.

    Output per frame: (N_PTS, 3) float32 raw image-space coordinates, NaN where nothing was
    detected. Hands are assigned to left/right by proximity to the pose wrists rather than
    MediaPipe's handedness label, so mirrored webcams and un-mirrored videos both work.
    """

    def __init__(self, video_mode=True, parallel=True, hand_conf=0.5):
        import mediapipe as mp
        from mediapipe.tasks.python import vision, BaseOptions
        self.mp = mp
        mode = vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE
        self.video_mode = video_mode
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "pose_landmarker_full.task")),
            running_mode=mode))
        self.hands = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(MODELS / "hand_landmarker.task")),
            running_mode=mode, num_hands=2, min_hand_detection_confidence=hand_conf,
            min_hand_presence_confidence=hand_conf, min_tracking_confidence=hand_conf))
        # Body and hand tracking are two independent models: run the hand model on another core while the body model
        # runs here (both release the interpreter lock while they work). Same landmarks, ~40% less time per frame.
        from concurrent.futures import ThreadPoolExecutor
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lingestra-hands") if parallel else None

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        self.pose.close(); self.hands.close()

    def process(self, rgb, ts_ms):
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        hand_fn = self.hands.detect_for_video if self.video_mode else self.hands.detect
        pose_fn = self.pose.detect_for_video if self.video_mode else self.pose.detect
        args = (img, ts_ms) if self.video_mode else (img,)
        if self._pool is not None:
            fut = self._pool.submit(hand_fn, *args)         # hands on another core ...
            pr = pose_fn(*args)                              # ... while the body is tracked here
            hr = fut.result()
        else:
            pr, hr = pose_fn(*args), hand_fn(*args)
        out = np.full((N_PTS, 3), np.nan, np.float32)
        if not pr.pose_landmarks:
            return out
        pose = np.array([[p.x, p.y, p.z] for p in pr.pose_landmarks[0][:N_POSE]], np.float32)
        out[:N_POSE] = pose
        for h in hr.hand_landmarks:
            pts = np.array([[p.x, p.y, p.z] for p in h], np.float32)
            dl = np.linalg.norm(pts[0, :2] - pose[L_WR, :2])
            dr = np.linalg.norm(pts[0, :2] - pose[R_WR, :2])
            sl = slice(N_POSE, N_POSE + 21) if dl < dr else slice(N_POSE + 21, N_POSE + 42)
            if np.isnan(out[sl][0, 0]):     # keep the first (closest) assignment
                out[sl] = pts
        return out


def extract_video(path, extractor=None):
    """All frames of a video -> (frames, N_PTS, 3) raw landmarks and the video fps."""
    import cv2
    own = extractor is None
    ex = extractor or LandmarkExtractor(video_mode=True)
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frames, i = [], 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        frames.append(ex.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int(i * 1000 / fps)))
        i += 1
    cap.release()
    if own:
        ex.close()
    return np.stack(frames) if frames else np.zeros((0, N_PTS, 3), np.float32), fps


# ----------------------------------------------------------------------------- preprocessing
def normalise(seq):
    """Centre on shoulder midpoint, scale by shoulder width. Missing landmarks -> 0.

    Per-frame normalisation (not per-clip) so the live app can compute it frame by frame.
    Returns (frames, N_PTS*3).
    """
    seq = seq.copy()
    c = (seq[:, L_SH, :] + seq[:, R_SH, :]) / 2
    s = np.linalg.norm(seq[:, L_SH, :2] - seq[:, R_SH, :2], axis=1)
    # carry the last valid reference forward (and backward) through detection drop-outs
    for arr in (c, s):
        valid = ~np.isnan(arr.reshape(len(arr), -1)).any(axis=1)
        if valid.any():
            idx = np.where(valid, np.arange(len(arr)), 0)
            np.maximum.accumulate(idx, out=idx)
            first = np.argmax(valid)
            idx[:first] = first
            arr[:] = arr[idx]
    s = np.maximum(s, 1e-3)[:, None, None]
    out = (seq - c[:, None, :]) / s
    return np.nan_to_num(out, nan=0.0).reshape(len(seq), -1).astype(np.float32)


def hand_activity(seq):
    """Per-frame 'is signing' score from RAW landmarks: how high the wrists are and how fast
    the hands move. Independent of signing speed because we only threshold, never time-scale."""
    n = len(seq)
    if n == 0:
        return np.zeros(0)
    ref = np.nanmedian(seq[:, [L_HIP, R_HIP], 1], axis=1)
    sh = np.nanmedian(seq[:, [L_SH, R_SH], 1], axis=1)
    up = np.nan_to_num(ref - np.nanmin(seq[:, [L_WR, R_WR], 1], axis=1), nan=-1.0)  # wrist above hip
    scale = np.nanmedian(np.abs(ref - sh)) + 1e-6
    return up / scale


def trim_for_model(seq, thresh=0.25, pad=3, min_len=8):
    """The cut the Keras dictionary model was trained with (first to last frame with a wrist above
    the hips, +-pad). Kept ONLY for that model's input pipeline (clip_to_tensor / scripts/train.py)
    so the existing models/*.keras keep working; everything else uses trim_active()."""
    act = hand_activity(seq)
    on = np.where(act > thresh)[0]
    if len(on) < min_len:
        return seq
    return seq[max(on[0] - pad, 0): on[-1] + pad + 1]


def trim_active(seq, fps=25.0, pad=1, min_len=8, **_ignored):
    """Cut a clip down to the sign itself: from 'arrived in signing space' to 'started to leave'.

    Raising the hands from rest and lowering them again are NOT part of the sign. This is the one
    cut used by the recorder, the live recogniser, library animations, training and the matcher
    (see ZoneTracker / segment_signing). Returns `seq` unchanged when no signing is found.
    """
    seq = np.asarray(seq)
    eps = segment_signing(seq, fps=fps, pad=pad, min_len=min_len)
    if not eps:
        return seq
    return seq[eps[0].start: eps[-1].end + 1]


# ----------------------------------------------------------------------------- signing zone
# 'Height' of a hand = how far the wrist is above the hip line, in units of the shoulder-to-hip
# span (0 = hip, 1 = shoulder). It only needs pose, so it works when hand landmarks drop out, and it
# is invariant to camera distance. Signing space is the zone well above rest.
ZONE_ENTER_MIN, ZONE_EXIT_MIN = 0.38, 0.25    # absolute floors of the hysteresis band
ENTER_ABOVE_REST, EXIT_ABOVE_REST = 0.30, 0.15
TRIGGER_S = 0.08         # hands must stay above 'enter' this long to start an episode
EXIT_S = 0.20            # ... and below 'exit' this long to end it (debounces tracking glitches)
V_UP = V_DOWN = 0.8      # span-units/s: faster vertical travel below signing height = approach/retreat
SOFT_V = 0.3             # ... threshold for the tail of a piece cut at a hold
ARRIVE_FRAC = 0.9        # 'arrived' = this fraction of the height reached in the first/last 0.8 s
MAX_TRIM_S = 0.7         # never trim more than this at either end
MIN_EPISODE_S = 0.25
HOLD_S = 0.6             # hands (nearly) static this long while up = a hold / pause
HOLD_EXTENT = 0.10       # ... 'static' = all tracked points move < this many shoulder widths


def _carry(arr):
    """Forward-fill (then back-fill) NaN rows of a per-frame reference array."""
    valid = ~np.isnan(arr.reshape(len(arr), -1)).any(axis=1)
    if not valid.any():
        return None
    idx = np.where(valid, np.arange(len(arr)), 0)
    np.maximum.accumulate(idx, out=idx)
    idx[:np.argmax(valid)] = np.argmax(valid)
    return arr[idx]


def hand_height(seq):
    """Per-frame height of the highest wrist (0 = hip, 1 = shoulder), NaN when unknown.

    Reference hip/shoulder lines are carried across drop-outs, so a few missing frames neither
    raise nor lower the hands."""
    seq = np.asarray(seq)
    n = len(seq)
    if n == 0:
        return np.zeros(0, np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        hip = _carry(np.nanmean(seq[:, [L_HIP, R_HIP], 1], axis=1))
        sh = _carry(np.nanmean(seq[:, [L_SH, R_SH], 1], axis=1))
        if hip is None or sh is None:
            return np.full(n, np.nan, np.float32)
        wr = np.nanmin(seq[:, [L_WR, R_WR], 1], axis=1)
    span = np.maximum(hip - sh, 0.05)
    return ((hip - wr) / span).astype(np.float32)


def zone_thresholds(rest):
    rest = float(np.clip(rest, 0.0, 0.3))
    return (min(max(ZONE_ENTER_MIN, rest + ENTER_ABOVE_REST), 0.6),
            min(max(ZONE_EXIT_MIN, rest + EXIT_ABOVE_REST), 0.42))


def _vertical_speed(h, t):
    """Smoothed d(height)/dt (span units per second), central difference over +-2 frames."""
    n = len(h)
    if n < 3:
        return np.zeros(n)
    hh = np.nan_to_num(h, nan=0.0)
    lo = np.maximum(np.arange(n) - 2, 0)
    hi = np.minimum(np.arange(n) + 2, n - 1)
    return (hh[hi] - hh[lo]) / np.maximum(t[hi] - t[lo], 1e-3)


def trim_bounds(h, t, s, e, seen_rest_before, seen_rest_after, pad=1, v_up=None, v_down=None):
    """Approach/retreat trimming of one in-zone span [s, e] (inclusive indices into h/t).

    Start: skip the frames where the wrist is still travelling up fast from rest and has not yet
    reached signing height. End: drop the frames where it has started to travel down toward rest.
    'Signing height' is a fraction of the span's median height, so a sign that itself moves
    vertically keeps its strokes once the hand has arrived. Only done on a side where rest was
    actually observed (a clip that starts with raised hands has no approach to cut)."""
    v = _vertical_speed(h, t)
    v_up = V_UP if v_up is None else v_up
    v_down = V_DOWN if v_down is None else v_down
    a, b = s, e
    if e < s:
        return s, e
    hs = h[s:e + 1]
    # 'signing height' = what the hand reaches within the first / last 0.8 s of the span
    head = hs[t[s:e + 1] <= t[s] + 0.8]
    tail = hs[t[s:e + 1] >= t[e] - 0.8]
    arrive_a = ARRIVE_FRAC * float(np.nanmax(head)) if len(head) else 0.0
    arrive_b = ARRIVE_FRAC * float(np.nanmax(tail)) if len(tail) else 0.0
    if seen_rest_before:
        lim = t[s] + MAX_TRIM_S
        while a < b and v[a] > v_up and h[a] < arrive_a and t[a] < lim:
            a += 1
        a = max(a - pad, s)
    if seen_rest_after:
        lim = t[e] - MAX_TRIM_S
        while b > a and v[b] < -v_down and h[b] < arrive_b and t[b] > lim:
            b -= 1
        b = min(b + pad, e)
    return a, b


def trim_tail(frames, fps=25.0):
    """Drop the slow start of a lowering from the end of a piece (used after a hold split, where
    what is left of the episode would otherwise be mostly the hands leaving)."""
    frames = np.asarray(frames)
    if len(frames) < 6:
        return frames
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        h = hand_height(frames).astype(np.float64)
    if np.isnan(h).all():
        return frames
    t = np.arange(len(h)) / fps
    _, b = trim_bounds(h, t, 0, len(h) - 1, False, True, pad=0, v_down=SOFT_V)
    return frames[:b + 1]


@dataclass
class Episode:
    start: int               # first / last kept frame, indices into the pushed stream
    end: int
    frames: np.ndarray       # (n, 67, 3) raw landmarks of the kept (trimmed) part
    t0: float
    t1: float
    raw_start: int = 0       # untrimmed zone span (includes approach / retreat)
    raw_end: int = 0
    reason: str = "lowered"  # 'lowered' | 'flushed'
    raw_frames: Any = None   # untrimmed zone span (approach + retreat included), for the model path


class ZoneTracker:
    """Streaming signing-zone detector shared by recorder, recogniser and offline cutting.

    push(raw, t) returns an Episode when the hands have been lowered (or None). While inside the
    zone, `frames` / `hold_*` describe the episode so far. Hysteresis + debounce make it robust
    to detection noise; `rest_for(t)` is how long the hands have been at rest."""

    def __init__(self, rest=None, pad=1, min_len=8, trigger_s=None, exit_s=None):
        self.fixed_rest, self.pad, self.min_len = rest, pad, min_len
        self.trigger_s = TRIGGER_S if trigger_s is None else trigger_s
        self.exit_s = EXIT_S if exit_s is None else exit_s
        self.reset()

    def reset(self):
        self.n = 0
        self.rest_level = 0.10 if self.fixed_rest is None else float(self.fixed_rest)
        self.in_zone = False
        self._pre = []                 # (idx, t, raw, h) just before the zone (context for trimming)
        self._ep = []                  # same tuples inside the zone span (untrimmed)
        self._enter_since = None
        self._low_since = None
        self._last_in = None           # index (in _ep) of last frame at/above 'exit'
        self.rest_since = None
        self._ref = None               # carried (hip_y, sh_y, c, width)
        self._hold_buf = []            # (t, pts (K,2))
        self.hold_active = False
        self.hold_start = None         # stream index where the static window began
        self._last_h = np.nan
        self._last_h_t = -1e9

    # -- per-frame height with carried references
    def _height(self, raw, t):
        hip = np.nanmean(raw[[L_HIP, R_HIP], 1]) if not np.isnan(raw[[L_HIP, R_HIP], 1]).all() else np.nan
        sh = np.nanmean(raw[[L_SH, R_SH], 1]) if not np.isnan(raw[[L_SH, R_SH], 1]).all() else np.nan
        w = raw[[L_WR, R_WR], 1]
        c = (raw[L_SH, :2] + raw[R_SH, :2]) / 2
        wd = np.linalg.norm(raw[L_SH, :2] - raw[R_SH, :2])
        if not (np.isnan(hip) or np.isnan(sh)):
            self._ref = (hip, sh)
        if not (np.isnan(c).any() or np.isnan(wd)):
            self._ref_c = (c, max(float(wd), 1e-3))
        if self._ref is None or np.isnan(w).all():
            # no wrist (or no person yet): hold the last height briefly, then call it 'unknown'
            return self._last_h if t - self._last_h_t <= 0.15 else np.nan
        h = (self._ref[0] - float(np.nanmin(w))) / max(self._ref[0] - self._ref[1], 0.05)
        self._last_h, self._last_h_t = h, t
        return h

    def _points(self, raw):
        ref = getattr(self, "_ref_c", None)
        if ref is None:
            return None
        c, wd = ref
        idx = [L_WR, R_WR] + [N_POSE + k for k in (4, 8, 12, 20)] + [N_POSE + 21 + k for k in (4, 8, 12, 20)]
        return (raw[idx, :2] - c) / wd

    def _update_hold(self, raw, t, in_zone):
        if not in_zone:
            self._hold_buf, self.hold_active, self.hold_start = [], False, None
            return
        pts = self._points(raw)
        if pts is None:
            return
        self._hold_buf.append((t, pts, self.n - 1))
        while len(self._hold_buf) > 1 and t - self._hold_buf[1][0] >= HOLD_S:
            self._hold_buf.pop(0)
        if t - self._hold_buf[0][0] < HOLD_S * 0.9:
            self.hold_active, self.hold_start = False, None
            return
        stack = np.stack([b[1] for b in self._hold_buf])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            ext = np.nanmax(np.nanmax(stack, 0) - np.nanmin(stack, 0))
        still = bool(ext < HOLD_EXTENT)           # NaN extent (nothing tracked) -> not a hold
        if still and not self.hold_active:
            self.hold_start = self._hold_buf[0][2]
        self.hold_active = still
        if not still:
            self.hold_start = None

    # -- main entry
    def push(self, raw, t):
        raw = np.asarray(raw, np.float32)
        t = float(t)
        idx = self.n
        self.n += 1
        h = self._height(raw, t)
        enter, exit_ = zone_thresholds(self.rest_level)
        hv = -1.0 if np.isnan(h) else h
        out = None
        if not self.in_zone:
            self._pre.append((idx, t, raw, h))
            while self._pre and (t - self._pre[0][1] > 2.0 or len(self._pre) > 80):
                self._pre.pop(0)
            if hv >= enter:
                self._enter_since = t if self._enter_since is None else self._enter_since
                if t - self._enter_since >= self.trigger_s:
                    # episode starts at the first frame of the continuous run above 'enter'
                    k = len(self._pre) - 1
                    while k > 0 and self._pre[k - 1][3] >= enter and not np.isnan(self._pre[k - 1][3]):
                        k -= 1
                    self._ctx = self._pre[:k]
                    self._ep = self._pre[k:]
                    self._pre = []
                    self.in_zone, self._low_since, self.rest_since = True, None, None
                    self._last_in = len(self._ep) - 1
                    self._hold_buf = []
                    for q in self._ep:
                        self._update_hold(q[2], q[1], True)
            else:
                self._enter_since = None
                if hv < exit_ and self.fixed_rest is None and not np.isnan(h):
                    self.rest_level += 0.05 * (min(h, 0.3) - self.rest_level)
                if self.rest_since is None:
                    self.rest_since = t
        else:
            self._ep.append((idx, t, raw, h))
            self._update_hold(raw, t, True)
            if hv >= exit_:
                self._low_since, self._last_in = None, len(self._ep) - 1
            else:
                self._low_since = t if self._low_since is None else self._low_since
                if t - self._low_since >= self.exit_s:
                    out = self._close("lowered", t)
        return out

    def _close(self, reason, t):
        ep, ctx = self._ep, getattr(self, "_ctx", [])
        last = self._last_in if reason == "lowered" else len(ep) - 1
        ep_in = ep[:last + 1]
        rest_start = self._low_since if reason == "lowered" else None
        self.in_zone, self._ep, self._ctx = False, [], []
        self._hold_buf, self.hold_active, self.hold_start = [], False, None
        self._pre = [] if reason == "lowered" else self._pre
        self._enter_since = None
        self.rest_since = rest_start if rest_start is not None else t
        out = None
        if len(ep_in) >= 2:
            hs = np.array([q[3] for q in ep_in], np.float64)
            ts = np.array([q[1] for q in ep_in])
            # drop context for velocity only when it exists: approach is real if the zone was
            # preceded by observed frames below 'enter'
            before = len(ctx) > 0
            a, b = trim_bounds(hs, ts, 0, len(ep_in) - 1, before, reason == "lowered", self.pad)
            if b - a + 1 < self.min_len:           # trimming ate the sign: keep the whole span
                a, b = 0, len(ep_in) - 1
            if (ts[b] - ts[a]) >= MIN_EPISODE_S and b - a + 1 >= 3:
                out = Episode(ep_in[a][0], ep_in[b][0], np.stack([q[2] for q in ep_in[a:b + 1]]),
                              float(ts[a]), float(ts[b]), ep_in[0][0], ep_in[-1][0], reason,
                              np.stack([q[2] for q in (ctx[-12:] + ep)]))
        return out

    def flush(self, t=None):
        """Close an open episode without waiting for the hands to come down."""
        if self.in_zone:
            return self._close("flushed", self._ep[-1][1] if t is None else t)
        return None

    # -- views on the episode in progress
    def current_span(self):
        """(start_idx, frames (n,67,3), h) of the in-progress zone span with the approach already
        trimmed (once the hands arrived), or None when not in the zone."""
        if not self.in_zone or len(self._ep) < 2:
            return None
        ep = self._ep
        hs = np.array([q[3] for q in ep], np.float64)
        ts = np.array([q[1] for q in ep])
        a, _ = trim_bounds(hs, ts, 0, len(ep) - 1, len(getattr(self, "_ctx", [])) > 0, False, self.pad)
        return ep[a][0], np.stack([q[2] for q in ep[a:]]), hs[a:]

    def rest_for(self, t):
        return 0.0 if self.in_zone or self.rest_since is None else max(t - self.rest_since, 0.0)


def segment_signing(seq, fps=25.0, t=None, pad=1, min_len=8):
    """Offline cut of a whole clip into sign episodes (same state machine as the live tracker).
    Returns a list of Episode; `.frames` excludes the raise and the lowering."""
    seq = np.asarray(seq, np.float32)
    if seq.ndim != 3 or len(seq) < 3:
        return []
    ts = np.arange(len(seq)) / fps if t is None else np.asarray(t, float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        h = hand_height(seq)
    ok = h[~np.isnan(h)]
    if len(ok) == 0:
        return []
    rest = float(np.percentile(ok, 10))
    if rest > 0.3:                       # hands never came down in this clip: no rest level observed
        rest = 0.1
    rest = max(rest, 0.0)
    tr = ZoneTracker(rest=rest, pad=pad, min_len=min_len)
    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        for f, tt in zip(seq, ts):
            e = tr.push(f, tt)
            if e is not None:
                out.append(e)
        e = tr.flush()
    if e is not None:
        out.append(e)
    return out


def resample(x, t=T):
    """Linear time-resample (frames, F) -> (t, F). This is what removes overall speed."""
    n = len(x)
    if n == 0:
        return np.zeros((t, x.shape[1] if x.ndim == 2 else FEAT), np.float32)
    if n == 1:
        return np.repeat(x, t, axis=0)
    src = np.linspace(0, n - 1, t)
    lo = np.floor(src).astype(int); hi = np.minimum(lo + 1, n - 1); w = (src - lo)[:, None]
    return (x[lo] * (1 - w) + x[hi] * w).astype(np.float32)


def add_velocity(x):
    """(t, F) -> (t, 2F): positions plus first differences (already speed-normalised by resample)."""
    v = np.vstack([np.zeros((1, x.shape[1]), np.float32), np.diff(x, axis=0)]) * x.shape[0] / 10.0
    return np.concatenate([x, v], axis=1)


def clip_to_tensor(raw_seq, trim=True):
    """Raw landmarks of a clip -> model input (T, 2*FEAT). Used offline and for live windows."""
    if trim:
        raw_seq = trim_for_model(raw_seq)
    return add_velocity(resample(normalise(raw_seq)))


# ----------------------------------------------------------------------------- augmentation
def time_warp(seq, rng, max_speed=2.0, strength=0.5):
    """Random monotone time warp of a normalised (frames, F) clip.

    A global speed factor in [1/max_speed, max_speed] (applied by cropping/stretching the frame
    count) plus a smooth random speed profile, so the sign is performed 'faster at the start /
    slower at the end' etc. Output has the same number of frames as the input before resample().
    """
    n = len(seq)
    if n < 4:
        return seq
    speed = np.exp(rng.uniform(-np.log(max_speed), np.log(max_speed)))
    m = max(int(n / speed), 4)
    knots = np.cumsum(np.exp(rng.normal(0, strength, 6)))      # random speed profile
    knots = (knots - knots[0]) / (knots[-1] - knots[0])
    cdf = np.interp(np.linspace(0, 1, m), np.linspace(0, 1, 6), knots)
    return _sample_at(seq, cdf * (n - 1))


def _sample_at(x, positions):
    n = len(x)
    lo = np.floor(positions).astype(int); hi = np.minimum(lo + 1, n - 1); w = (positions - lo)[:, None]
    return (x[lo] * (1 - w) + x[hi] * w).astype(np.float32)


def spatial_augment(x, rng, rot_deg=12.0, scale=0.12, shift=0.08, noise=0.01):
    """Camera-angle/position robustness on a normalised (frames, FEAT) clip."""
    pts = x.reshape(len(x), N_PTS, 3).copy()
    a = np.deg2rad(rng.uniform(-rot_deg, rot_deg))
    R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]], np.float32)
    present = np.abs(pts).sum(axis=2) > 0
    pts[..., :2] = pts[..., :2] @ R.T
    pts[..., :2] *= rng.uniform(1 - scale, 1 + scale, 2)       # anisotropic scale ~ slight yaw/zoom
    pts[..., :2] += rng.uniform(-shift, shift, 2)
    pts += rng.normal(0, noise, pts.shape)
    pts[~present] = 0.0                                          # keep "not detected" as zeros
    return pts.reshape(len(x), -1).astype(np.float32)


# ----------------------------------------------------------------------------- signing features
# Per-frame features for the personal matcher (not for the Keras model). Everything is expressed
# in a body-anchored frame: origin = shoulder midpoint, x axis = the shoulder line (so small camera
# roll cancels), unit = shoulder width (so camera distance/zoom cancel). Hand shape is relative to
# the wrist and scaled by palm size. Left and right are NOT interchangeable (no mirror invariance).
# A block is (name, n_dims, weight, typical_scale); `typical_scale` makes blocks comparable.
SIGN_BLOCKS = [
    ("locL", 4, 1.0, 0.45), ("locR", 4, 1.0, 0.45),        # wrist vs shoulders / vs nose
    ("shpL", 30, 1.0, 0.55), ("shpR", 30, 1.0, 0.55),      # fingertip + mid-finger joints vs wrist
    ("oriL", 6, 0.5, 0.60), ("oriR", 6, 0.5, 0.60),        # palm normal + hand direction
    ("arm", 4, 0.3, 0.45),                                  # both elbows
    ("dom", 1, 0.5, 0.60),                                  # which hand does the work (-1 L .. +1 R)
]
BLOCK_SLICES = {}
_o = 0
for _n, _d, _w, _t in SIGN_BLOCKS:
    BLOCK_SLICES[_n] = slice(_o, _o + _d)
    _o += _d
SIGN_DIM = _o
NOSE_IDX = 0
USED_FRAC = 0.25
REST_HEIGHT = 0.30       # a hand lower than this (span units) is 'resting' and carries no information
_SHAPE_PTS = [3, 4, 6, 8, 10, 12, 14, 16, 18, 20]


def _bridge_cols(x, miss, max_gap):
    """Linear interpolation across short runs where `miss` is True (in place, per column group)."""
    n = len(x)
    i = 0
    while i < n:
        if miss[i]:
            k = i
            while k < n and miss[k]:
                k += 1
            if i > 0 and k < n and k - i <= max_gap:
                w = (np.arange(i, k) - (i - 1)) / (k - (i - 1))
                x[i:k] = x[i - 1] * (1 - w)[:, None] + x[k] * w[:, None]
                miss[i:k] = False
            i = k
        else:
            i += 1


def sign_features(seq, fps=25.0, extra=False):
    """Raw landmarks (n, 67, 3) -> (X (n, SIGN_DIM) float32, M (n, len(SIGN_BLOCKS)) bool).

    M[i, b] says block b carries information in frame i (hand visible AND raised, etc.); X is zero
    where M is False. Returns empty arrays for empty input.

    extra=True (continuous signing, live/matcher.py): a hand is treated as 'used' for the WHOLE
    sequence (every frame with a detected wrist fills its blocks) and a third value is returned,
    dict(hh=(n, 2) per-wrist height [left, right] (-1 where missing)). The caller then decides per
    SEGMENT which hand is used (`hand_used`), because the 'used' rule and the dominant-hand value
    depend on the sequence they are computed over."""
    seq = np.asarray(seq, np.float32)
    n = len(seq)
    nb = len(SIGN_BLOCKS)
    if n == 0:
        out = (np.zeros((0, SIGN_DIM), np.float32), np.zeros((0, nb), bool))
        return out + (dict(hh=np.zeros((0, 2), np.float32)),) if extra else out
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lsh, rsh = _carry(seq[:, L_SH, :2]), _carry(seq[:, R_SH, :2])
        if lsh is None or rsh is None:
            out = (np.zeros((n, SIGN_DIM), np.float32), np.zeros((n, nb), bool))
            return out + (dict(hh=np.full((n, 2), -1.0, np.float32)),) if extra else out
        c = (lsh + rsh) / 2
        ex = lsh - rsh
        w = np.maximum(np.linalg.norm(ex, axis=1), 1e-3)
        ex = ex / w[:, None]
        ey = np.stack([-ex[:, 1], ex[:, 0]], 1)
        ey = np.where(ey[:, 1:2] < 0, -ey, ey)         # y axis points 'down the body' (image y grows down)
        ex = np.where(ex[:, 0:1] < 0, -ex, ex)

        def to_body(p):                                 # (n, 2) image points -> body coords
            d = p - c
            return np.stack([(d * ex).sum(1), (d * ey).sum(1)], 1) / w[:, None]

        def rot_vec(v):                                 # (n, k, 2/3) -> rotate xy into body axes
            xy = v[..., :2]
            a = (xy * ex[:, None, :]).sum(-1)
            b = (xy * ey[:, None, :]).sum(-1)
            return np.concatenate([a[..., None], b[..., None], v[..., 2:3]], -1)

        X = np.zeros((n, SIGN_DIM), np.float32)
        M = np.zeros((n, nb), bool)
        h = hand_height(seq)
        nose = _carry(seq[:, NOSE_IDX, :2])
        nose_b = to_body(nose) if nose is not None else np.zeros((n, 2))
        elb = []
        hhs = []
        for side, (wr_i, el_i, base) in enumerate([(L_WR, 13, N_POSE), (R_WR, 14, N_POSE + 21)]):
            # --- wrist location (pose, fall back to hand landmark 0)
            wr = seq[:, wr_i, :2].copy()
            hand = seq[:, base:base + 21, :]
            vis = ~np.isnan(hand[:, 0, 0])
            wr = np.where(np.isnan(wr), hand[:, 0, :2], wr)
            ok = ~np.isnan(wr[:, 0])
            wb = to_body(np.nan_to_num(wr))
            # per-wrist height for 'resting' detection
            hip = _carry(np.nanmean(seq[:, [L_HIP, R_HIP], 1], axis=1))
            sh = _carry(np.nanmean(seq[:, [L_SH, R_SH], 1], axis=1))
            if hip is not None and sh is not None:
                hh = (hip - np.nan_to_num(wr[:, 1], nan=1e3)) / np.maximum(hip - sh, 0.05)
            else:
                hh = h
            # a hand that is raised for under a quarter of the sequence is not 'used': ignore it
            # completely, so a resting hand flickering around a threshold adds no noise; a used hand
            # counts in every frame (it may dip toward the hip mid-sign)
            used = (ok & (hh >= 2 * REST_HEIGHT)).mean() >= USED_FRAC
            hhs.append(np.where(ok, hh, -1.0).astype(np.float32))
            up = ok if (used or extra) else np.zeros_like(ok)
            lb = "locL" if side == 0 else "locR"
            loc = np.concatenate([wb, wb - nose_b], 1)
            X[:, BLOCK_SLICES[lb]] = np.where(up[:, None], loc, 0)
            M[:, SIGN_BLOCKS.index(next(b for b in SIGN_BLOCKS if b[0] == lb))] = up
            # --- hand shape + orientation
            shp = "shpL" if side == 0 else "shpR"
            ori = "oriL" if side == 0 else "oriR"
            hv = np.nan_to_num(hand)
            rel = rot_vec(hv - hv[:, :1, :])
            palm = np.maximum(np.linalg.norm(rel[:, 9, :], axis=1), 1e-3)
            sh_f = (rel[:, _SHAPE_PTS, :] / palm[:, None, None]).reshape(n, -1)
            nrm = np.cross(rel[:, 5, :], rel[:, 17, :])
            nrm = nrm / np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-6)
            dr = rel[:, 9, :] / palm[:, None]
            seen = vis & up
            X[:, BLOCK_SLICES[shp]] = np.where(seen[:, None], sh_f, 0)
            X[:, BLOCK_SLICES[ori]] = np.where(seen[:, None], np.concatenate([nrm, dr], 1), 0)
            M[:, [i for i, b in enumerate(SIGN_BLOCKS) if b[0] == shp][0]] = seen
            M[:, [i for i, b in enumerate(SIGN_BLOCKS) if b[0] == ori][0]] = seen
            # --- elbow
            e = seq[:, el_i, :2]
            elb.append((to_body(np.nan_to_num(e)), ~np.isnan(e[:, 0])))
        X[:, BLOCK_SLICES["arm"]] = np.concatenate([elb[0][0], elb[1][0]], 1) * (elb[0][1] & elb[1][1])[:, None]
        M[:, [i for i, b in enumerate(SIGN_BLOCKS) if b[0] == "arm"][0]] = elb[0][1] & elb[1][1]
        # --- dominant hand: wrist path length (left vs right) over the sequence
        lp = _path_len(X[:, BLOCK_SLICES["locL"]][:, :2], M[:, 0])
        rp = _path_len(X[:, BLOCK_SLICES["locR"]][:, :2], M[:, 1])
        dom = (rp - lp) / (rp + lp + 0.3)
        X[:, BLOCK_SLICES["dom"]] = dom
        M[:, -1] = True
        # --- bridge short detection gaps of hand shape, then smooth
        gap = max(int(round(0.2 * fps)), 1)
        for b in ("shpL", "shpR", "oriL", "oriR"):
            bi = [i for i, bb in enumerate(SIGN_BLOCKS) if bb[0] == b][0]
            loc_bi = 0 if b.endswith("L") else 1
            miss = (~M[:, bi]) & M[:, loc_bi]
            if miss.any() and M[:, bi].any():
                _bridge_cols(X[:, BLOCK_SLICES[b]], miss, gap)
                M[:, bi] |= ~miss & M[:, loc_bi]
    Xs = _smooth_feats(X, M)
    if extra:
        return Xs, M, dict(hh=np.stack(hhs, 1))
    return Xs, M


def hand_used(hh):
    """The 'used hand' rule of sign_features for a stretch: raised (>= 2 x REST_HEIGHT) in at least
    USED_FRAC of its frames. hh: (n,) wrist heights (-1 = missing)."""
    return len(hh) > 0 and bool((np.asarray(hh) >= 2 * REST_HEIGHT).mean() >= USED_FRAC)


def _path_len(xy, m):
    if len(xy) < 2:
        return 0.0
    ok = m[1:] & m[:-1]
    return float((np.linalg.norm(np.diff(xy, axis=0), axis=1) * ok).sum())


def _smooth_feats(X, M):
    """[1 2 1]/4 smoothing along time, block-wise, only mixing frames where the block is present."""
    n = len(X)
    if n < 3:
        return X
    out = X.copy()
    for bi, (name, d, _, _) in enumerate(SIGN_BLOCKS):
        if name == "dom":
            continue
        sl = BLOCK_SLICES[name]
        m = M[:, bi].astype(np.float32)
        x = X[:, sl]
        pm = np.concatenate([m[:1], m, m[-1:]])
        px = np.concatenate([x[:1], x, x[-1:]])
        num = 0.25 * px[:-2] * pm[:-2, None] + 0.5 * px[1:-1] * pm[1:-1, None] + 0.25 * px[2:] * pm[2:, None]
        den = 0.25 * pm[:-2] + 0.5 * pm[1:-1] + 0.25 * pm[2:]
        out[:, sl] = np.where(m[:, None] > 0, num / np.maximum(den, 1e-6)[:, None], 0)
    return out.astype(np.float32)
