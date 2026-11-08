"""Hands-free sign recorder (Qt-free, landmarks only).

Fed with the same per-frame raw landmarks (67, 3) and timestamps the Recognizer gets. Only the
timestamps are used for timing, never the wall clock, so replayed / non-real-time sources work.

    countdown -> waiting (raise your hands) -> recording -> done | failed

Segmentation is features.ZoneTracker, the same 'signing zone' model the live recogniser uses:
the clip starts when the hands have ARRIVED in signing space and ends when they START to leave.
Raising the hands from rest and lowering them again are not part of the clip, and lowering the
hands is the stop gesture.
"""
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from lingestra import features as F

N_HAND = slice(F.N_POSE, F.N_PTS)


@dataclass
class RecState:
    state: str                 # idle | countdown | waiting | recording | done | failed
    message: str
    seconds_left: float
    frames: int
    clip: Any = None           # (frames, 67, 3) raw landmarks when state == 'done'


def _person_visible(raw):
    return not (np.isnan(raw[[F.L_SH, F.R_SH], 1]).all() or np.isnan(raw[[F.L_HIP, F.R_HIP], 1]).all())


class SignRecorder:
    ABSENT_S = 3.0           # no person in frame this long while waiting -> fail
    MIN_HANDS_FRAC = 0.5

    def __init__(self, countdown_s=3.0, max_s=8.0, rest_s=0.3, min_frames=10, wait_timeout_s=10.0):
        """`rest_s`: how long the hands must stay lowered before the recording stops."""
        self.countdown_s, self.max_s, self.rest_s = countdown_s, max_s, rest_s
        self.min_frames, self.wait_timeout_s = min_frames, wait_timeout_s
        self.cancel()

    # ------------------------------------------------------------------ control
    def cancel(self):
        self._set('idle', 'Idle')
        self._t0 = None
        self._zone = F.ZoneTracker(exit_s=self.rest_s)
        self._rec_t0 = None
        self._absent_since = None
        self._wait_t0 = None
        self._clip = None
        self._frames = 0

    def start(self, t):
        self.cancel()
        self._t0 = float(t)
        self._set('countdown', 'Get ready')
        self._left = self.countdown_s
        if self.countdown_s <= 0:
            self._enter_waiting(t)

    def _set(self, state, message):
        self._state, self._msg, self._left = state, message, 0.0

    def _enter_waiting(self, t):
        self._set('waiting', 'Raise your hands and sign')
        self._wait_t0 = float(t)

    def _fail(self, msg):
        self._set('failed', msg)

    def _snapshot(self):
        frames = self._frames if self._state == 'recording' else (
            len(self._clip) if self._clip is not None else 0)
        return RecState(self._state, self._msg, max(self._left, 0.0), frames,
                        self._clip if self._state == 'done' else None)

    # ------------------------------------------------------------------ per frame
    def push(self, lm, t) -> RecState:
        t = float(t)
        if self._state in ('idle', 'done', 'failed'):
            return self._snapshot()
        raw = np.asarray(lm, np.float32)

        if self._state == 'countdown':
            left = self._t0 + self.countdown_s - t
            if left > 0:
                self._left = left
                self._msg = f'Get ready: {math.ceil(left)}'
                return self._snapshot()
            self._enter_waiting(t)

        if self._state in ('waiting', 'recording'):
            self._track(raw, t)
        return self._snapshot()

    def _track(self, raw, t):
        if self._state == 'waiting':
            self._left = self._wait_t0 + self.wait_timeout_s - t
            if _person_visible(raw):
                self._absent_since = None
            else:
                self._absent_since = t if self._absent_since is None else self._absent_since
                if t - self._absent_since >= self.ABSENT_S:
                    return self._fail('Person not in frame: step back so your head, shoulders and hips are visible')
        was_in = self._zone.in_zone
        ep = self._zone.push(raw, t)
        if ep is not None:
            return self._finish(ep.frames)
        if self._zone.in_zone and self._state == 'waiting':
            self._set('recording', 'Recording... lower your hands to finish')
            self._rec_t0 = t
        elif was_in and not self._zone.in_zone:      # a blip too short to be a sign: keep waiting
            self._set('waiting', 'Raise your hands and sign')
            self._wait_t0 = t
            self._rec_t0 = None
        if self._state == 'recording':
            span = self._zone.current_span()
            self._frames = len(span[1]) if span else 0
            self._left = self._rec_t0 + self.max_s - t
            if self._left <= 0:
                ep = self._zone.flush(t)
                return self._finish(ep.frames) if ep is not None else self._fail(
                    'Too short: sign a little longer or more clearly')
        elif self._left <= 0:
            self._fail('Timed out waiting for hands to be raised' if self._absent_since is None
                       else 'Person not in frame')

    def _finish(self, clip):
        clip = np.asarray(clip, np.float32)
        ok, msg = self.quality(clip)
        if not ok:
            return self._fail(msg)
        self._clip = clip
        self._set('done', 'Recorded %d frames' % len(clip))

    # ------------------------------------------------------------------ quality
    def quality(self, clip):
        clip = np.asarray(clip)
        if clip.ndim != 3 or len(clip) < self.min_frames:
            return False, 'Too short: sign a little longer or more clearly'
        hands = (~np.isnan(clip[:, N_HAND, 0])).any(axis=1)
        if hands.mean() < self.MIN_HANDS_FRAC:
            return False, 'No hands seen: your hands were visible in under half of the frames'
        if not any(_person_visible(f) for f in clip):
            return False, 'Person not in frame'
        return True, 'OK'
