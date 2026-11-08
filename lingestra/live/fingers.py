"""Answer a 'which word did you mean?' question by holding up fingers: 1 finger = first option,
2 = second option, 3 = 'sign it again' (the Recognizer builds the option list; this module only
turns hands into a number and decides when a number has been 'held' long enough).

No training data is needed. Two ways to read a number from the hand landmarks:

  * STANDARD (geometry): a finger counts as extended when its tip is much further from the wrist than
    the finger's knuckle (MCP); curled fingers are close to the wrist. Calibrated on the dictionary's own
    number signs (One..Five): extended fingers measure 1.8-2.0, curled ones 0.6-0.9. Anything in between is
    'not a clean count' and is ignored.
  * PERSONAL (calibration): the user shows 1, 2 and 3 fingers once ('Teach me my fingers' in Settings); the
    five finger measurements of each pose are stored and later readings are matched to the nearest of them.
    This copes with a closer camera, a tilted hand, shorter fingers or a thumb held differently.

Safeguards against picking by accident (the signing hand often ends in a hand shape that looks like a
number):
  * the picker is armed ARM_S after the question appears (the last sign has been finished by then),
  * the same count must be held steadily for HOLD_S,
  * a closed fist (0 fingers) is NOT a number: the other hand resting as a fist must not cancel the pick,
  * two raised hands: their numbers are ADDED (6..10 = five on one hand plus the rest on the other),
  * only hands raised above the hip line count.
"""
from __future__ import annotations

from collections import Counter, deque

import numpy as np

from lingestra import features as F

TIPS = (8, 12, 16, 20)          # index, middle, ring, pinky tips
MCPS = (5, 9, 13, 17)
EXT_R = 1.5                     # tip / knuckle distance (both measured from the wrist) above this = extended
CURL_R = 1.2                    # ... below this = curled; in between = unclear -> no count
THUMB_EXT = 0.95                # thumb tip to index knuckle, in palm widths: above this = extended
CAL_TOL = 0.55                  # personal poses: a reading must be this close (L2 over the 5 measurements) ...
CAL_MARGIN = 0.20               # ... and this much closer than the second-nearest pose
CAL_TARGETS = (1, 2, 3, 4, 5)   # the numbers taught in 'Teach me my fingers' (more are made by adding both hands)
TIGHT = 0.25                    # ... and the fingers of the extended group must measure within this of each other
ASPECT = 16 / 9                 # default frame width / height (the dictionary videos); real cameras report their own
CAL_VERSION = 3                 # taught poses are measured in 3D; poses saved by older versions (flat measurements) are ignored
GAP_MIN = 0.30                  # a hand with a half-bent finger is still read when its fingers split into two clear groups
THUMB_W = 0.35                  # weight of the thumb when matching taught poses (it varies most between people)
CAL_MIN_SEPARATION = 0.5        # the taught poses must be at least this far apart
CAL_EXT_R = 1.15                # while TEACHING a finger counts as extended above this (curled ones stay below ~1.0)


def _xyz(h, aspect):
    """Hand landmarks as one isotropic 3D space: x and depth are fractions of the image WIDTH, y of the HEIGHT, so x and
    z are scaled by width / height. (Depth is what stops a finger pointing at the camera from looking short.)"""
    P = np.asarray(h, float)[:, :3].copy()
    P[:, 0] *= aspect
    P[:, 2] *= aspect
    return P


def finger_features(h, aspect=ASPECT):
    """The five measurements of one hand (21, 3), in 3D: extension of index, middle, ring, pinky (tip-to-wrist distance /
    knuckle-to-wrist distance) and the thumb (tip to index knuckle in palm widths). In 3D these hardly change when the
    hand tilts or points at the camera and fluctuate far less from frame to frame than flat 2D measurements (measured
    on the dictionary's number signs: frame-to-frame spread 0.05-0.13 instead of up to 0.8). None if not fully detected."""
    h = np.asarray(h, float)
    if h.shape[0] < 21 or np.isnan(h[:21, :3]).any():
        return None
    P = _xyz(h, aspect)
    w = P[0]
    r = []
    for tip, mcp in zip(TIPS, MCPS):
        base = np.linalg.norm(P[mcp] - w)
        if base < 1e-6:
            return None
        r.append(np.linalg.norm(P[tip] - w) / base)
    palm = np.linalg.norm(P[5] - P[17])
    if palm < 1e-6:
        return None
    # the thumb is 'out' when its tip is far from BOTH outer knuckles (index side and pinky side): the same whether the
    # tracker labels the hand correctly or as its mirror image
    r.append(min(np.linalg.norm(P[4] - P[5]), np.linalg.norm(P[4] - P[17])) / palm)
    return np.array(r, float)


def _gap_count(r):
    """No finger is half-way for sure, but the four fingers split into an EXTENDED group and a CURLED group with a clear
    gap between them (a foreshortened or slightly bent hand): read it from that gap. None if there is no clear split."""
    s = sorted((float(x) for x in r), reverse=True)
    g, i = max((s[k] - s[k + 1], k) for k in range(3))
    if g < GAP_MIN:
        return None
    top, bottom = s[:i + 1], s[i + 1:]
    if max(top) - min(top) > TIGHT:        # foreshortening shortens ALL extended fingers alike; a lone half-bent finger is an
        return None                        # outlier, not a clean 'up' - better no number than a wrong one
    return len(top) if (min(top) >= CURL_R and max(bottom) <= CURL_R - 0.05) else None


def _standard_count(v):
    """Fingers held up (0..5). The thumb is IGNORED unless all four fingers are up: a fist very often has its thumb resting
    alongside, and a relaxed 'fist' on the other hand must not add a phantom finger. Four fingers up + thumb out = 5."""
    n, unclear = 0, False
    for r in v[:4]:
        if r >= EXT_R:
            n += 1
        elif r > CURL_R:
            unclear = True                  # half-bent finger
    if unclear:
        g = _gap_count(v[:4])
        if g is None:
            return None                     # not a clean count
        n = g
    if n == 4:
        return 5 if v[4] > THUMB_EXT else 4
    return n


def describe_hand(h, aspect=ASPECT):
    """What the app reads from one hand, finger by finger (for the 'Check what I see' window):
    [(name, measurement, 'up' | 'down' | 'unclear')]; None if the hand is not fully detected."""
    v = finger_features(h, aspect)
    if v is None:
        return None
    out = []
    for name, r in zip(("index", "middle", "ring", "pinky"), v[:4]):
        out.append((name, float(r), "up" if r >= EXT_R else "down" if r <= CURL_R else "unclear"))
    out.append(("thumb", float(v[4]), "up" if v[4] > THUMB_EXT else "down"))
    return out


def _by_length(v):
    """The four finger measurements largest first (+ the thumb). Which FINGER is up does not matter for a number, and a hand
    tracker that mistakes your hand for the mirror-image hand reports the raised index finger as the pinky: comparing
    the sorted measurements makes taught poses immune to that."""
    v = np.asarray(v, float)
    return np.concatenate([np.sort(v[:4])[::-1], v[4:5]])


def _personal_count(v, cal):
    """Nearest taught pose, or None when nothing is close enough / two poses are equally close."""
    w = np.array([1.0, 1.0, 1.0, 1.0, THUMB_W])
    q = _by_length(v)
    best = sorted(((float(np.linalg.norm((q - _by_length(c)) * w)), int(k)) for k, c in cal.items()
                   if k.isdigit() and len(c) == 5))
    if not best or best[0][0] > CAL_TOL:
        return None
    if len(best) > 1 and best[1][0] - best[0][0] < CAL_MARGIN:
        return None
    return best[0][1]


def count_fingers(h, cal=None, aspect=ASPECT) -> int | None:
    """Number of extended fingers of one hand (21, 3), or None when it is not a clean count or the hand
    is not fully detected. With `cal` (taught poses) the nearest taught pose wins; otherwise (or when no
    taught pose fits) the standard geometry decides."""
    v = finger_features(h, aspect)
    if v is None:
        return None
    if cal:
        n = _personal_count(v, cal)
        if n is not None:
            return n
    return _standard_count(v)


def _hip_y(raw) -> float:
    ys = [raw[i, 1] for i in (F.L_HIP, F.R_HIP) if not np.isnan(raw[i, 1])]
    return float(np.mean(ys)) if ys else float("nan")


def _hands(raw):
    """The two hands (21, 3) of a frame (67, 3) that are raised above the hip line (others are left out)."""
    raw = np.asarray(raw, float)
    hip = _hip_y(raw)
    out = []
    for sl in (slice(F.N_POSE, F.N_POSE + 21), slice(F.N_POSE + 21, F.N_POSE + 42)):
        h = raw[sl]
        if np.isnan(h[:, :2]).any():
            continue
        if not np.isnan(hip) and h[0, 1] > hip:        # image y grows downwards: wrist below the hips = lowered
            continue
        out.append(h)
    return out


def hand_visible(raw) -> bool:
    """A raised hand is fully detected (so fingers could be read)."""
    return bool(_hands(raw))


def hands_count(raw, cal=None, aspect=ASPECT) -> int | None:
    """The number (1..10) shown by the raised hands in one frame (67, 3), or None (nothing clean).
    One hand: its fingers (1..5). Both hands: the two counts added up, so 6..10 = five on one hand plus the rest on
    the other. A closed fist (0) and a hand that is not a clean count are 'no number here', so a resting fist on the
    other hand does not matter."""
    seen = []
    for h in _hands(raw):
        n = count_fingers(h, cal, aspect)
        if n:                                         # None (unclear) and 0 (fist) are both 'no number here'
            seen.append(n)
    if not seen:
        return None
    total = sum(seen)
    return total if total <= 10 else None


class FingerPicker:
    """Feed it every frame while a question is open; it says when a number has been held steadily.

    'Held steadily' is a MAJORITY VOTE over the last WINDOW_S seconds, not 'the same on every frame':
      * a frame with a raised hand that gives no clean number (half-bent finger, ...) votes AGAINST;
      * a frame in which no hand is tracked at all ABSTAINS: hand trackers drop out often on a fist with fingers up
        (35% of the frames in a real report), and a dropout says nothing about which number is shown.
    A number is chosen when at least MIN_VOTES frames voted, the votes span at least HOLD_S, and at least MIN_SHARE of
    the votes are for that number."""
    ARM_S = 0.7           # ignore the first moments after the question appears
    HOLD_S = 0.5          # the votes must span at least this long
    WINDOW_S = 0.8        # ... looking back this far
    MIN_SHARE = 0.7       # ... with this share of the votes for the same number
    MIN_VOTES = 4         # ... and at least this many frames in which a hand was tracked
    STILL = 0.30          # ... and the hand(s) must be HELD: the wrist may wander at most this many shoulder widths over the
                          # window. Holding up fingers for a button is a held pose; a hand shape in the middle of a sign is
                          # moving, and must not be taken for an answer.

    def __init__(self, cal=None, aspect=ASPECT):
        self.cal = cal
        self.aspect = aspect          # frame width / height of the camera (3D measurements need it)
        self.seen = None              # the number read in the latest frame (any 1..10), None = no clean number
        self.hand = False             # a raised hand is visible in the latest frame
        self._sh = None               # shoulder width (isotropic units) of the latest frame with a tracked hand
        self.reset()

    def reset(self):
        self.armed_at = None
        self.options = 0
        self.win = deque()            # (time, number | 0, wrist x, wrist y) of the latest frames in which a hand was tracked

    def arm(self, t, options):
        self.reset()
        self.armed_at, self.options = float(t), int(options)

    @property
    def armed(self):
        return self.armed_at is not None

    def push(self, raw, t):
        """-> (choice | None, number being held | 0, progress 0..1). choice is 1..options."""
        n = hands_count(raw, self.cal, self.aspect)
        self.seen, self.hand = n, hand_visible(raw)
        if self.armed_at is None or t < self.armed_at + self.ARM_S:
            self.win.clear()
            return None, 0, 0.0
        if self.hand:                                                # no tracked hand: abstain
            wr = np.mean([_xyz(h, self.aspect)[0, :2] for h in _hands(raw)], axis=0)
            self.win.append((t, n if (n is not None and 1 <= n <= self.options) else 0, float(wr[0]), float(wr[1])))
            self._sh = self._shoulder_width(raw)
        while self.win and t - self.win[0][0] > self.WINDOW_S:
            self.win.popleft()
        votes = Counter(v for _t, v, _x, _y in self.win if v)
        if not votes:
            return None, 0, 0.0
        xs = [w[2] for w in self.win]; ys = [w[3] for w in self.win]
        moved = max(max(xs) - min(xs), max(ys) - min(ys))
        if self._sh and moved > self.STILL * self._sh:
            return None, 0, 0.0                                      # the hand is travelling: that is signing, not answering
        cand, k = votes.most_common(1)[0]
        total = len(self.win)
        share = k / total
        span = self.win[-1][0] - self.win[0][0]
        if total >= self.MIN_VOTES and span >= self.HOLD_S and share >= self.MIN_SHARE:
            self.reset()
            return cand, cand, 1.0
        prog = min(span / self.HOLD_S, 1.0) * min(share / self.MIN_SHARE, 1.0) * min(total / self.MIN_VOTES, 1.0)
        return None, (cand if share >= 0.5 else 0), min(prog, 0.99)

    def _shoulder_width(self, raw):
        """Distance between the shoulders in the same isotropic units (None if the pose is not tracked)."""
        a, b = np.asarray(raw, float)[F.L_SH, :2], np.asarray(raw, float)[F.R_SH, :2]
        if np.isnan(a).any() or np.isnan(b).any():
            return None
        d = float(np.hypot((a[0] - b[0]) * self.aspect, a[1] - b[1]))
        return d if d > 1e-6 else None


class FingerCalibrator:
    """'Teach me my fingers': the user holds up 1, then 2, then 3 fingers. Feed it every preview frame.

    state() -> dict(step=index in CAL_TARGETS | None when finished, target=number to show, progress 0..1,
                    message=plain-words hint, done=bool, ok=bool, cal=dict | None)."""
    NEED = 22             # steady samples per pose (about 1.5 s of camera frames)
    STEADY = 0.12         # the measurements must not wander more than this while they are collected
    GRACE = 12            # frames after a finished pose in which the old pose is still shown: no complaints

    def __init__(self, targets=CAL_TARGETS, aspect=ASPECT):
        self.targets = tuple(targets)
        self.aspect = aspect
        self.restart()

    def restart(self):
        self.step = 0
        self.samples = []
        self.cal = {}
        self.done = False
        self.ok = False
        self.message = "Hold up 1 finger"
        self._grace = 0

    @staticmethod
    def _hold(n):
        """The instruction for pose n in plain words."""
        return "Open your whole hand, fingers and thumb apart" if n == 5 else f"Hold up {n} finger{'s' if n != 1 else ''}"

    def _one_hand(self, raw):
        hs = [finger_features(h, self.aspect) for h in _hands(raw)]
        hs = [v for v in hs if v is not None]
        return hs[0] if len(hs) == 1 else None          # exactly one raised, fully detected hand

    def feed(self, raw):
        if self.done:
            return self.state()
        if self._grace > 0:                              # the signer is still changing hands: keep the message, collect nothing
            self._grace -= 1
            return self.state()
        v = self._one_hand(raw)
        target = self.targets[self.step]
        if v is None:
            self.samples = []
            self.message = (f"{self._hold(target)} - keep one hand up in front of the "
                            "camera, palm toward you, and the other hand down")
            return self.state()
        # a pose that cannot be the requested number is not collected (stops a wrong pose being saved)
        ext = int(sum(r > CAL_EXT_R for r in v[:4]))
        # 1-4: that many FINGERS up (the thumb does not count): at least `target` are clearly up and at most `target`
        # could be (a half-bent finger is allowed); 5: the whole hand open, thumb out as well
        sure = int(sum(r >= EXT_R - 0.15 for r in v[:4]))
        if (not (sure <= target <= ext)) if target <= 4 else (ext != 4 or v[4] < 0.8):
            self.samples = []
            self.message = f"That does not look like {target}. {self._hold(target)}" + (" (the others folded in)" if target < 5 else "")
            return self.state()
        if self.samples and np.max(np.abs(v - np.median(np.array(self.samples), 0))) > 0.5:
            self.samples = []                           # the hand moved a lot: start this pose again
        self.samples.append(v)
        self.message = f"{self._hold(target)} and keep steady"
        if len(self.samples) >= self.NEED:
            a = np.array(self.samples[-self.NEED:])
            if np.max(np.ptp(a, axis=0)) > self.STEADY * 4:     # too shaky to trust
                self.samples = self.samples[-5:]
                self.message = "Keep your hand a little steadier"
                return self.state()
            self.cal[str(target)] = [float(x) for x in np.median(a, axis=0)]
            self.samples = []
            self.step += 1
            if self.step >= len(self.targets):
                self._finish()
            else:
                nxt = self.targets[self.step]
                self._grace = self.GRACE
                self.message = f"Good. Now: {self._hold(nxt).lower()}" if nxt == 5 else f"Good. Now hold up {nxt} finger{'s' if nxt != 1 else ''}"
        return self.state()

    def _finish(self):
        self.done = True
        keys = sorted(self.cal)
        worst = min((np.linalg.norm(_by_length(self.cal[a]) - _by_length(self.cal[b])), a, b)
                    for i, a in enumerate(keys) for b in keys[i + 1:])
        if worst[0] < CAL_MIN_SEPARATION:
            self.ok = False
            self.message = (f"I can't tell {worst[1]} and {worst[2]} fingers apart yet. Try again with the fingers "
                            "clearly apart and the other fingers folded in")
        else:
            self.ok = True
            self.message = "Done! I will now recognise these finger numbers."

    def state(self):
        n = len(self.targets)
        prog = (self.step + (0 if self.done else len(self.samples) / self.NEED)) / n
        return dict(step=None if self.done else self.step, target=None if self.done else self.targets[self.step],
                    progress=min(prog, 1.0), message=self.message, done=self.done, ok=self.ok,
                    cal=dict(self.cal, ver=CAL_VERSION) if self.done and self.ok else None)
