"""Streaming sign recogniser: raw landmarks frame by frame -> Events.

Pipeline
  1. every frame goes through features.ZoneTracker ('signing zone': how high the hands are relative
     to the shoulder-hip span, with hysteresis). An EPISODE starts when the hands have arrived in
     signing space and ends when they are lowered. The raise and the lowering are cut off (they are
     the 'stop' gesture, not part of the sign). The recogniser ALWAYS receives un-mirrored
     landmarks; mirroring is display-only.
  2. Hands that stay up between signs are separated by HOLDS: stretches where the hands are (nearly)
     still for >= 0.6 s are remembered. When the episode ends the whole episode is compared with
     the pieces between the holds: if the whole is a confident match it is ONE sign (the holds were
     pauses between its key poses), otherwise, if the pieces match, they are separate signs. A very
     long episode is resolved at a hold so words do not wait forever. The separator gesture (both
     hands above the head, held) always ends the sentence.
  3. each finished piece is classified ONCE by the personal matcher (live/matcher.py) against the
     active templates (the user's own recordings): speed- and pause-invariant, open-set. Events
     carry source='personal'. The optional Keras dictionary model is a secondary path, used only
     when enabled (`use_model=True`, or automatically when there are no templates) and it never
     overrides a confident personal match (source='model').
  3b. CONTINUOUS SIGNING (`Recognizer.continuous`, default True): the signer may sign several signs
     in a row WITHOUT lowering the hands. The episode is decoded into words by
     matcher.decode_sequence (segmental spotting, see docs/recognition.md 'Continuous signing'):
     while the hands stay up the episode so far is re-decoded every ~0.25 s (and at every hold) and a
     word is COMMITTED once its segment ended more than LOOKAHEAD_S ago, the same word was found by
     the previous decode too, and it matched confidently, so 'ok' appears while '7000' has begun.
     Committed words never change. When the episode ends (hands lowered / separator) the rest is
     decoded and everything is emitted. An ambiguous segment becomes a 'clarify' for THAT segment
     only (a placeholder keeps its place in the sentence), an unmatched one an 'unknown' carrying
     only its own landmark window. MAX_EPISODE_S no longer cuts a sign. continuous=False is the
     older behaviour: one decision per episode, holds as the only split cue.
  4. events: 'sign', 'clarify' (two close candidates), 'unknown' (carries the trimmed landmark
     window), 'separator', 'sentence' (hands at rest long enough or separator).
  5. adaptive load control applies to the (optional) model path; the personal matcher is a few ms.
"""
import json
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from lingestra import features as F
from lingestra.live.fingers import ASPECT, FingerPicker, hands_count
from lingestra.live.matcher import MatcherConfig, MatchResult, PersonalMatcher

NOSE = 0


@dataclass
class Event:
    kind: str                 # 'sign' | 'unknown' | 'clarify' | 'retry' | 'skip' | 'separator' | 'sentence'
    label: str | None
    confidence: float
    latency_ms: float
    candidates: list = field(default_factory=list)
    t: float = 0.0
    # Extra (optional): for 'unknown' the raw landmark window (frames, 67, 3) so the caller can
    # store it via SecureStore.save_unknown; `source` is 'personal' or 'model'.
    window: Any = None
    source: str | None = None
    # 'sign' events produced by confirm(): the 'learn from corrections' hint (see Recognizer._learn_hint)
    # {ok, reason, label, sign_id, window (raw, that segment only), fps}; None = nothing to learn.
    learn: Any = None
    # True on the 'sign' event that ANSWERS an open question (tap or fingers): it takes the question's place.
    answered: bool = False


class Recognizer:
    REST_PAUSE_S = 1.5        # hands down this long -> sentence is finished
    SEP_HOLD_S = 1.0          # both hands above the head this long -> separator gesture
    PENDING_TIMEOUT_S = 10.0  # unanswered clarify question is dropped (while the hands are up)
    QUESTION_GRACE_S = 2.0    # hands lowered with a question still open: wait this long for a tap, then skip that word and send
                              # (before: the message waited for the whole 10 s timeout - measured 10.6 s instead of 1.8 s)
    MAX_SUGGEST = 8           # a question offers at most this many words (+ Skip and Retry = 10 options)
    CARRY_ON_S = 2.0          # a question is open and the hands keep MOVING (not held) this long: the signer carries on
                              # signing instead of answering -> the question is skipped
    MIN_PIECE_FRAMES = 8
    NOT_YET = 1 << 60
    MAX_EPISODE_S = 8.0       # an episode this long is resolved at the next hold
    MIN_HAND_FRAC = 0.3       # an unknown sign needs hands visible in this share of its frames
    HOLD_FILL_S = 0.15        # short landmark drop-outs are bridged this long
    DECODE_EVERY_S = 0.25     # continuous: re-decode the episode tail after this much new signing
    DECODE_DUTY = 4.0         # ... but never more often than (this x the last decode's duration): with hundreds of signs
                              # on, one decode takes 0.3-0.6 s (> a camera frame); unthrottled, frames pile up and every
                              # later word arrives later. Decoding uses at most ~1/4 of the time.
    LOOKAHEAD_S = 0.30        # continuous: a word is committed once its segment ended this long ago
    COMMIT_SURE = 0.85        # continuous: ... and matched with distance / threshold below this (else wait for the end);
    HOLD_SURE = 0.25          # ... below this when the only evidence that the word is over is a hold. (The first half of
                              # a sign matches its own template at ~0.5, so a pause INSIDE a sign must not commit it.)
    UNSURE_WAITS_FOR_SIGN = True    # True = old rule: a question / unknown waits until a LATER word is final (stalls what follows)
    HOLD_COMMIT_S = 1.6       # continuous: hands held STILL this long (counted from when they stopped) = the word before it is
                              # over: it is then accepted at COMMIT_SURE like a word followed by another sign. Before this a
                              # held last word needed a near-perfect match (HOLD_SURE) and otherwise waited for STALE_S (4 s).
                              # 1.6 s is above the longest pause inside a sign that is tolerated (1.5 s, tests); adjustable.
    SETTLE_S = None           # (off) the LAST word, hands still up and moving on, shown once its match stayed the same this
    SETTLE_SURE = 0.60        # ... long, with distance / threshold below this. Measured: it rarely fires (the hands lower right
                              # after the last word) and costs accuracy (single signs split 7% -> 10%), so it is disabled.
    MAX_TAIL_S = 10.0         # continuous: never decode more than this much uncommitted signing at once (what falls
                              # off the front is emitted as decoded, never dropped; one sign slower than this is cut)
    STALE_S = 4.0             # continuous: a word that has waited this long is committed even if not 'sure'
    # model path load levels: number of crops evaluated per decision
    LOAD_LEVELS = [4, 4, 2, 1]
    # learning from corrections (quality gate)
    LEARN_MIN_S = 0.4         # the confirmed movement must last at least this long
    LEARN_MIN_HANDS = 0.6     # ... with hands visible in this share of its frames
    LEARN_MAX_RATIO = 2.0     # ... and be within this multiple of the chosen sign's acceptance distance
    LEARN_DUP_S = 2.0         # the same sign is not learned twice within this time

    def __init__(self, model_path=None, threshold: float = 0.80, clarify_floor: float = 0.45,
                 separator: str = 'both_hands_up_hold', latency_budget_ms: float = 60.0,
                 model=None, classes=None, templates=None, use_model=None,
                 matcher_config: MatcherConfig | None = None, continuous: bool = True, **_legacy):
        """`model`/`classes` let tests inject a stand-in with __call__(x)->probs. `use_model`:
        None = automatic (the Keras model is used only while there are no personal templates),
        True/False = force. `threshold` is the user-facing strictness (0.8 = calibrated defaults)."""
        if model is None and model_path and Path(str(model_path)).exists():
            from tensorflow import keras
            import tensorflow as tf
            km = keras.models.load_model(model_path, compile=False)
            # a compiled tf.function is ~30x faster per call than eager keras (2 ms vs 65 ms)
            fn = tf.function(lambda x: km(x, training=False),
                             input_signature=[tf.TensorSpec((None, F.T, 2 * F.FEAT), tf.float32)])
            model = lambda x: fn(tf.constant(x)).numpy()          # noqa: E731
            classes = json.load(open(Path(model_path).with_suffix(".json")))["classes"]
        self._model, self.classes = model, list(classes or [])
        self.none_idx = self.classes.index("none") if "none" in self.classes else -1
        self.threshold, self.clarify_floor = threshold, clarify_floor
        self.separator, self.budget = separator, latency_budget_ms
        self.use_model = use_model
        self.continuous = bool(continuous)   # several signs in a row without lowering the hands
        self.matcher = PersonalMatcher(matcher_config)
        self.label_map = {}                  # raw model class -> display label (model path only)
        self.suspended = False
        self.learning = True                 # confirm() may hand back the movement as a new example
        self._label_ids = {}                 # label -> id of the first template with that label
        self._learned_t = {}                 # label -> stream time of the last accepted learn hint
        self.finger_cal = None               # the signer's own taught finger poses (set_finger_cal)
        self.finger_answers = True           # False: questions are answered by tap / keys only, never by fingers
        self.aspect = ASPECT                 # camera frame width / height (finger readings are 3D); the worker sets it
        self.auto_finish = True              # False ('send with Space'): lowering the hands does NOT end the message
        self.hold_commit_s = self.HOLD_COMMIT_S   # how long a still hold must last to end the word before it
        self.reset()
        if templates:
            self.set_templates(templates)
        self._warm()

    # ------------------------------------------------------------------ templates / settings
    def set_templates(self, templates):
        """Replace the personal templates: [{id, label, frames: [raw (n,67,3)...], fps?}]."""
        self.matcher.set_templates(templates or [])
        self._label_ids = {}
        for t in templates or []:
            self._label_ids.setdefault(str(t.get("label")), str(t.get("id")))

    def set_finger_answers(self, flag: bool) -> None:
        """Questions may be answered by holding up fingers (True) or only by tap / keys (False)."""
        self.finger_answers = bool(flag)
        if not self.finger_answers:
            self.picker.reset(); self.finger = (0, 0.0)
        elif self.pending is not None:
            n = min(len(list(self.pending.get('candidates', []))), self.MAX_SUGGEST) + 2
            self.picker.arm(self.last_t, n)

    def set_finger_cal(self, cal) -> None:
        """The signer's own taught finger poses ({'1': [5 numbers], '2': ..., '3': ...}); None = standard geometry."""
        self.finger_cal = dict(cal) if cal else None
        self.picker.cal = self.finger_cal

    def set_continuous(self, flag: bool) -> None:
        """Switch continuous signing on / off (safe at any moment; an episode in progress is decoded
        from the first word not yet emitted in the new mode; a repeated request changes nothing)."""
        flag = bool(flag)
        if flag == self.continuous:
            return
        self.continuous = flag
        self._c_n, self._c_prev = 0, []
        if flag:                             # words the old mode already emitted are not decoded again
            self._c_from = self.piece_start if 0 < self.piece_start < self.NOT_YET else None
        else:
            if self._c_from is not None and self.piece_start < self.NOT_YET:
                self.piece_start = max(self.piece_start, self._c_from)
            self._c_from = None

    @property
    def model_enabled(self) -> bool:
        if self._model is None:
            return False
        return bool(self.use_model) if self.use_model is not None else not self.matcher.ready

    @property
    def strictness(self) -> float:
        """Threshold slider -> multiplier of the calibrated acceptance distances (0.8 -> 1.0)."""
        return float(np.clip(1.0 + (0.8 - self.threshold) * 2.0, 0.6, 1.4))

    # ------------------------------------------------------------------ state
    def reset(self):
        self.buf = deque(maxlen=150)         # (idx, t, raw)
        self.n = 0
        self.zone = F.ZoneTracker()
        self.words = []                      # display strings
        self.pending = None
        self.in_episode = False
        self.sep_since = None
        self.sep_fired = False
        self.sep_idx = None                  # first frame of the current 'both hands up' run
        self.sentence_flushed = True
        self.last_t = 0.0
        self.piece_start = 0                 # stream index where the unclassified part begins (0 = episode start)
        self.holds = []                      # [start_idx, end_idx|None] of holds inside the piece
        self._hold_seen = False
        self._ignore = False                 # rest of the episode belongs to the separator pose
        self.votes, self.vote_label = 0, None     # kept for API compatibility (unused)
        self.load = 0
        self._since_change = 0
        self.lat_eval = deque(maxlen=10)     # ms per model evaluation
        self.ms_per_window = None
        self.skipped = 0
        self.scale_used = None
        self.dec_ms = deque(maxlen=20)       # ms per personal decision
        self._hold = None
        self._hold_t = None
        self._wall = deque(maxlen=30)
        self._lat_all = deque(maxlen=200)
        self._was_in = False                 # continuous state: zone state of the previous frame
        self._ep_id = 0
        self._c_from = None                  # stream index where the uncommitted part of the episode begins
        self._c_n = 0                        # episode frames seen at the last decode
        self._c_prev = []                    # (kind, label, start, end) of the previous decode, absolute indices
        self._last_decode_ms = 0.0           # duration of the latest continuous decode (throttles the next one)
        self.picker = FingerPicker(getattr(self, 'finger_cal', None), getattr(self, 'aspect', ASPECT))   # answers a question by finger count
        self.finger = (0, 0.0)               # (count held right now, progress 0..1) for the UI
        self._finger_engaged = False         # a clean finger count is showing: do not decode these frames
        self._answering = False              # a question is open and fingers may answer it: ANSWER MODE (no sign decoding)
        self._moving_since = None            # answer mode: since when the hands have been moving (not held)
        self._finger_release_until = None    # after an answer: keep cutting while the number pose is still shown
        self._skip_still = False             # a word was committed during a hold: the frozen pose after it is not a new word

    def _warm(self):
        """First model call is slow (graph build); do it now so it does not look like latency."""
        if self._model is None:
            return
        try:
            self._model(np.zeros((1, F.T, 2 * F.FEAT), np.float32))
        except Exception:
            pass

    # ------------------------------------------------------------------ per-frame helpers
    def _fill_dropouts(self, raw, t):
        """Hold the last valid value of each landmark for short gaps (<= HOLD_FILL_S)."""
        raw = raw.copy()
        if self._hold is None or self._hold.shape != raw.shape:
            self._hold = np.full(raw.shape, np.nan, np.float32)
            self._hold_t = np.full(raw.shape[:1], -np.inf)
        valid = ~np.isnan(raw).any(axis=-1)
        self._hold[valid] = raw[valid]
        self._hold_t[valid] = t
        fill = ~valid & ((t - self._hold_t) <= self.HOLD_FILL_S)
        raw[fill] = self._hold[fill]
        return raw

    @staticmethod
    def _both_up(raw):
        wr, nose = raw[[F.L_WR, F.R_WR], 1], raw[NOSE, 1]
        return bool((~np.isnan(wr)).all() and not np.isnan(nose) and (wr < nose).all())

    # ------------------------------------------------------------------ main entry
    def push_frame(self, raw_landmarks, t: float) -> list:
        if self.suspended:
            return []
        w0 = time.perf_counter()
        raw = self._fill_dropouts(np.asarray(raw_landmarks, np.float32), t)
        idx = self.n
        self.n += 1
        self.last_t = t
        self.buf.append((idx, t, raw))
        events = []
        up = self._both_up(raw)

        if self.pending is not None and t - self.pending['t'] > self.PENDING_TIMEOUT_S:
            self._drop_pending()

        # ---- ANSWER MODE: a question is open and fingers may answer it (1 = skip, 2 = retry, 3.. = the suggested words).
        #      The app cannot know whether a held hand shape is an answer or the start of the next sign, so while the question
        #      is open the hands are answering: a HELD hand (even one the picker cannot read yet) is never decoded as a sign -
        #      it would raise a NEW question or an 'unknown' card that replaces this one. If the signer keeps MOVING instead,
        #      that is the next sign: after CARRY_ON_S the question is skipped and normal decoding resumes.
        self._finger_engaged = False
        self._answering = self.pending is not None and self.picker.armed
        if self._answering:
            self.picker.aspect = self.aspect
            choice, cnt, prog = self.picker.push(raw, t)
            self.finger = (cnt, prog)
            self._finger_engaged = cnt > 0
            if choice is not None:
                events += self._answer_by_fingers(choice, idx, t)
                self._finger_engaged = True
                self._moving_since = None
            elif self.zone.hold_active or self._finger_engaged:
                self._moving_since = None
                self._c_from = idx + 1                  # the frames of a held pose are not part of any sign
                self._finger_engaged = True
            elif self.zone.in_zone:                     # hands up and moving: answering, or signing the next word?
                if self._moving_since is None:
                    self._moving_since = t
                if t - self._moving_since >= self.CARRY_ON_S:
                    self.skip()
                    events.append(Event('skip', 'carry_on', 1.0, 0.0, [], t))
                    self._answering = False
            else:
                self._moving_since = None               # hands down: the rest-pause / grace rules decide
        else:
            self.finger = (0, 0.0)
            self.picker.seen, self.picker.hand = None, False
            if self._finger_release_until is not None:      # answered: the number pose is not part of the next sign
                if t < self._finger_release_until and hands_count(raw, self.finger_cal, self.aspect) is not None:
                    self._c_from = idx + 1
                    self._finger_engaged = True
                else:
                    self._finger_release_until = None

        ep = self.zone.push(raw, t)
        self.in_episode = self.zone.in_zone
        if self.zone.in_zone and not self._was_in:           # a new episode: forget the previous one's question
            self._ep_id += 1
            self._c_from, self._c_n, self._c_prev = None, 0, []
            if self.pending is not None and self.pending.get('ep', self._ep_id) != self._ep_id:
                self._drop_pending()
        self._was_in = self.zone.in_zone
        if not self.in_episode and not ep:
            self.piece_start, self.holds, self._hold_seen = 0, [], False

        # ---- separator gesture: both hands above the head, held
        if self.separator == 'both_hands_up_hold':
            if up:
                if self.sep_since is None:
                    self.sep_since, self.sep_idx = t, idx
                if not self.sep_fired and t - self.sep_since >= self.SEP_HOLD_S:
                    self.sep_fired = True
                    events += self._on_separator(t)
            else:
                self.sep_since, self.sep_fired, self.sep_idx = None, False, None

        # ---- a word was committed while the hands were held still: that frozen pose (it can last seconds) is NOT part of
        #      the next sign. Without this the final check, when the hands are lowered, read it as another word
        #      (the same word again, or a look-alike). Decoding resumes when the hands move.
        if self._skip_still:
            if self.zone.hold_active and self.zone.in_zone:
                self._c_from = idx + 1
                self._finger_engaged = True
            else:
                self._skip_still = False

        # ---- holds: hands up but still. Remembered; resolved when the episode ends
        hold_started = False
        if self.zone.hold_active and not self._hold_seen:
            self._hold_seen = True
            hold_started = True
            if (self.sep_since is None and not self._ignore and self.zone.hold_start is not None
                    and not self._finger_engaged):
                self.holds.append([self.zone.hold_start, None])
                events += self._maybe_commit(t)
        elif not self.zone.hold_active:
            if self._hold_seen and self.holds and self.holds[-1][1] is None:
                self.holds[-1][1] = idx
                if self.piece_start >= self.NOT_YET:
                    self.piece_start = idx           # a long episode was cut at this hold
            self._hold_seen = False

        # ---- continuous signing: decode the episode so far, commit the words that are finished
        if (self.continuous and self.matcher.ready and ep is None and self.zone.in_zone
                and not self._ignore and self.sep_since is None and not self._finger_engaged and not self._answering):
            new = self._stream_step(t, force=hold_started)
            events += new
            if new and self.zone.hold_active:
                self._skip_still = True

        # ---- episode finished: the hands were lowered
        if ep is not None:
            if self._ignore:
                self._ignore = False
            else:
                events += self._on_episode(ep, t)
            self.piece_start, self.holds, self._hold_seen = 0, [], False
            self._c_from, self._c_n, self._c_prev = None, 0, []

        # ---- sentence ends when the hands rest long enough
        if self.auto_finish and not self.zone.in_zone and self.words and self.zone.rest_for(t) >= self.REST_PAUSE_S:
            if self.pending is None:
                events += self._finish_sentence(t)
            elif self.zone.rest_for(t) >= self.REST_PAUSE_S + self.QUESTION_GRACE_S:
                # the hands are down (fingers cannot answer any more) and nobody tapped: skip that word, send the rest
                self.skip()
                events.append(Event('skip', 'timeout', 1.0, 0.0, [], t))
                events += self._finish_sentence(t)

        # ---- latency bookkeeping: every event gets the wall time of the frame that made it
        lat = (time.perf_counter() - w0) * 1000
        self._lat_all.append(lat)
        self._wall.append(time.perf_counter())
        for e in events:
            if e.latency_ms == 0.0:
                e.latency_ms = lat
        return events

    def _answer_by_fingers(self, choice, idx, t):
        """The signer held up `choice` fingers: 1 = skip this sign, 2 = sign it again, 3.. = the suggested words."""
        cands = [c[0] if isinstance(c, (tuple, list)) else c
                 for c in (self.pending or {}).get('candidates', [])][:self.MAX_SUGGEST]
        # the frames showing the number are not part of any sign: decoding resumes after them
        self._c_from, self._c_prev = idx + 1, []
        self.piece_start, self.holds = idx + 1, []
        self._finger_release_until = t + 3.0
        if choice == 1:
            self.skip()
            return [Event('skip', None, 1.0, 0.0, [], t)]
        if choice == 2:
            self.reject()
            return [Event('retry', None, 1.0, 0.0, [], t)]
        if 0 <= choice - 3 < len(cands):
            return self.confirm(cands[choice - 3])
        self.reject()                                    # (cannot happen: the picker only accepts existing options)
        return [Event('retry', None, 1.0, 0.0, [], t)]

    # ------------------------------------------------------------------ pieces
    def _piece(self, lo_idx, hi_idx=None):
        """Raw frames of the in-progress episode with stream index in [lo_idx, hi_idx]."""
        span = self.zone.current_span()
        if span is None:
            return None
        s0, frames, _ = span
        a = max(lo_idx - s0, 0)
        b = len(frames) if hi_idx is None else max(hi_idx - s0 + 1, 0)
        return frames[a:b]

    def _rel_holds(self, base, n):
        """Holds as (start, end) indices relative to a piece starting at stream index `base`."""
        out = []
        for hs, he in self.holds:
            he = base + n - 1 if he is None else he
            if hs >= base and he > hs and hs - base < n:
                out.append((hs - base, min(he - base, n - 1)))
        return out

    def _maybe_commit(self, t):
        """A very long episode must not hold its words back: resolve what came before this hold."""
        if self.continuous and self.matcher.ready:       # continuous decoding commits words by itself
            return []
        if self.pending is not None or self.zone.hold_start is None:
            return []
        span = self.zone.current_span()
        if span is None:
            return []
        s0 = max(self.piece_start, span[0])
        start = self.zone.hold_start
        if (start - s0) / max(self._fps(), 1.0) < self.MAX_EPISODE_S:
            return []
        piece = self._piece(s0, start + 1)
        holds = self._rel_holds(s0, len(piece))
        self.holds = [h for h in self.holds if h[0] >= start]
        self.piece_start = self.NOT_YET
        if len(piece) < self.MIN_PIECE_FRAMES:
            return []
        return self._classify(piece, t, holds=[h for h in holds if h[0] < len(piece) - 1])

    def _fps(self):
        if len(self.buf) < 2:
            return 25.0
        return (len(self.buf) - 1) / max(self.buf[-1][1] - self.buf[0][1], 1e-3)

    def _on_separator(self, t):
        events = []
        if self.continuous and self.matcher.ready:
            if self.zone.in_zone and not self._ignore:
                span = self.zone.current_span()
                if span is not None:
                    hi = (self.sep_idx - 1) if self.sep_idx is not None else span[0] + len(span[1]) - 1
                    events += self._decode_final(span[0], span[1][:max(hi - span[0] + 1, 0)], t, separator=True)
            events.append(Event('separator', None, 1.0, 0.0, [], t))
            events += self._finish_sentence(t)
            if self.zone.in_zone:
                self._ignore = True
            self._c_from = None
            return events
        if self.zone.in_zone and self.pending is None and not self._ignore:
            span = self.zone.current_span()
            hi = (self.sep_idx - 1) if self.sep_idx is not None else None
            if span is not None:
                lo = max(self.piece_start, span[0])
                piece = self._piece(lo, hi)
                if piece is not None and len(piece) >= self.MIN_PIECE_FRAMES:
                    events += self._classify(piece, t, holds=self._rel_holds(lo, len(piece)))
        events.append(Event('separator', None, 1.0, 0.0, [], t))
        events += self._finish_sentence(t)
        if self.zone.in_zone:
            self._ignore = True              # the hands-up pose is not a sign
        return events

    def _on_episode(self, ep, t):
        if self.continuous and self.matcher.ready:
            return self._decode_final(ep.start, ep.frames, t)
        if self.pending is not None:         # the signer carried on: drop the unanswered question
            self._drop_pending()
        a = max(self.piece_start - ep.start, 0)
        piece = ep.frames[a:]
        holds = self._rel_holds(ep.start + a, len(piece))
        if a > 0:                            # what is left after a cut: drop the slow start of the lowering
            piece = F.trim_tail(piece)
            holds = [h for h in holds if h[0] < len(piece) - 1]
        if len(piece) < self.MIN_PIECE_FRAMES:
            return []
        raw_piece = ep.raw_frames if a == 0 and ep.raw_frames is not None else piece
        return self._classify(piece, t, holds=holds, raw_piece=raw_piece)

    # ------------------------------------------------------------------ continuous signing
    def _drop_pending(self):
        """Forget the open question; its placeholder leaves the sentence."""
        hole = (self.pending or {}).get('hole')
        self.pending = None
        self.picker.reset(); self.finger = (0, 0.0)
        self._moving_since = None
        if hole is not None:
            self.words = [w for w in self.words if w is not hole]
            self.sentence_flushed = not self.words

    def _stable(self, kind, label, a, b, fps):
        """The previous decode found the same word at about the same place."""
        ta, tb = int(0.35 * fps) + 2, int(0.25 * fps) + 2
        return any(k == kind and lb == label and abs(s0 - a) <= ta and abs(e0 - b) <= tb
                   for k, lb, s0, e0 in self._c_prev)

    def _emit(self, sg, tail, base, t):
        """One decoded segment -> events (sign / clarify / unknown). `tail` = frames the segment indexes."""
        if sg.kind == 'sign':
            return [self._sign(sg.label, sg.confidence, 'personal', t)]
        if self.model_enabled:               # optional secondary path: only for stretches the personal matcher could not decide
            piece = tail[sg.start:sg.end + 1]
            if len(piece) >= self.MIN_PIECE_FRAMES:
                res = MatchResult(sg.kind, sg.label, sg.confidence, list(sg.candidates))
                ev = self._model_decision(piece, t, res, piece)
                if ev is not None:
                    return ev
        if sg.kind == 'clarify':
            if self.pending is not None:
                self._drop_pending()         # a newer question replaces an unanswered one
            ev = self._clarify(sg.label, sg.confidence, sg.candidates, 'personal', t,
                               window=tail[sg.start:sg.end + 1])
            hole = object()
            self.words.append(hole)
            self.sentence_flushed = False
            self.pending['hole'] = hole
            self.pending['ep'] = self._ep_id
            return ev
        return self._unknown(tail[sg.start:sg.end + 1], t, 0.0, True)

    def _stream_step(self, t, force=False):
        """While the hands stay up: re-decode the uncommitted tail of the episode and commit the words
        that are finished (ended > LOOKAHEAD_S ago, found by the previous decode too, confident)."""
        span = self.zone.current_span()
        if span is None:
            return []
        s0, frames, _ = span
        fps = self._fps()
        n = len(frames)
        gap = max(self.DECODE_EVERY_S, self.DECODE_DUTY * self._last_decode_ms / 1000.0)
        if not force and n - self._c_n < max(int(gap * fps), 1):
            return []
        self._c_n = n
        lo = max(self._c_from if self._c_from is not None else s0, s0)
        cap = int(self.MAX_TAIL_S * fps)
        forced_until = lo                    # segments ending before this are emitted as decoded (the cap drops them)
        if s0 + n - lo > cap:                # nothing matched for ages: do not let the decode grow without bound
            forced_until = s0 + n - cap
        stale = int(self.STALE_S * fps)
        tail = frames[lo - s0:]
        if len(tail) < self.MIN_PIECE_FRAMES:
            return []
        segs = self.matcher.decode_sequence(tail, fps=fps, strictness=self.strictness, final=False)
        self.dec_ms.append(self.matcher.last_decode_ms)
        self._last_decode_ms = float(self.matcher.last_decode_ms)
        self.scale_used = 'personal'
        last, look = s0 + n - 1, int(np.ceil(self.LOOKAHEAD_S * fps))
        hold = self.zone.hold_start if self.zone.hold_active else None      # hands still: the word before it is over
        ok = []
        for i, sg in enumerate(segs):
            a, b = lo + sg.start, lo + sg.end
            if hold is not None and b >= hold:
                b = hold
            nxt = i < len(segs) - 1                              # something came after it: a next sign ...
            held_long = hold is not None and (last - hold) / fps >= self.hold_commit_s
            lim = self.COMMIT_SURE if (nxt or held_long) else (self.HOLD_SURE if hold is not None else self.SETTLE_SURE)
            sure = sg.kind != 'sign' or sg.ratio <= lim
            settled = (self.SETTLE_S is not None and sg.kind == 'sign'
                       and last - b >= int(np.ceil(self.SETTLE_S * fps)))                    # ... or it has stayed put
            follows = nxt or hold is not None or settled        # ... or a hold
            old = follows and (last - b >= stale or lo + sg.end < forced_until)   # waited too long: no more waiting
            ok.append(old or (follows and last - b >= look and sure
                              and self._stable(sg.kind, sg.label, a, lo + sg.end, fps)))
        # a word is final only once it is stable; a question / unknown only when a LATER sign is final too
        # (a sign still in progress looks like 'unknown' until it is complete)
        last_ok_sign = max((i for i, sg in enumerate(segs) if ok[i] and sg.kind == 'sign'), default=-1)
        if forced_until > lo:                # everything the cap drops is emitted, whatever its kind
            last_ok_sign = max(last_ok_sign, max((i for i, sg in enumerate(segs) if lo + sg.end < forced_until), default=-1))
        events = []
        for i, sg in enumerate(segs):
            # words are committed strictly in order. A question / unknown that is NOT the last segment (something came
            # after it, so it is not a sign still in progress) goes out as soon as it is stable; only the segment at the
            # live edge waits for a later word (a half-finished sign looks like 'unknown' until it is complete)
            wait = sg.kind != 'sign' and i > last_ok_sign and (self.UNSURE_WAITS_FOR_SIGN or i == len(segs) - 1)
            if not ok[i] or wait:
                break
            events += self._emit(sg, tail, lo, t)
            self._c_from = lo + (segs[i + 1].start if i + 1 < len(segs) else sg.end + 1)   # transitions are skipped
        if forced_until > lo and (self._c_from is None or self._c_from < forced_until):
            self._c_from = forced_until      # whatever is left of the dropped part was only a transition
        self._c_prev = [(sg.kind, sg.label, lo + sg.start, lo + sg.end) for sg in segs]
        return events

    def _decode_final(self, start, frames, t, separator=False):
        """The episode is over (hands lowered / separator): decode and emit everything not yet committed."""
        lo = max(self._c_from if self._c_from is not None else start, start)
        a = lo - start
        piece = frames[a:]                   # (the tracker already cut the lowering off the episode's end)
        self._c_from, self._c_n, self._c_prev = None, 0, []
        if len(piece) < self.MIN_PIECE_FRAMES:
            return []
        segs = self.matcher.decode_sequence(piece, fps=self._fps(), strictness=self.strictness)
        self.dec_ms.append(self.matcher.last_decode_ms)
        self.scale_used = 'personal'
        if separator:                        # the arms rising to the separator pose are not a sign
            fps = self._fps()
            while segs and segs[-1].kind == 'unknown' and len(piece) - segs[-1].start <= int(1.0 * fps):
                segs.pop()
        events = []
        for sg in segs:
            events += self._emit(sg, piece, lo, t)
        return events

    # ------------------------------------------------------------------ classification
    def _personal(self, piece):
        res = self.matcher.classify(piece, strictness=self.strictness, trim=False)
        self.dec_ms.append(res.ms)
        self.scale_used = 'personal'
        return res

    def _sign(self, label, conf, source, t):
        self.words.append(label)
        self.sentence_flushed = False
        return Event('sign', label, float(conf), 0.0, [], t, source=source)

    def _classify(self, piece, t, holds=(), raw_piece=None):
        res = self._personal(piece) if self.matcher.ready else None
        if res is not None and holds:
            ev = self._split_decision(piece, t, res, holds)
            if ev is not None:
                return ev
        if res is not None and res.kind == 'sign':
            return [self._sign(res.label, res.confidence, 'personal', t)]
        # optional secondary path: the Keras dictionary model, never over a confident personal match
        if self.model_enabled:
            ev = self._model_decision(piece, t, res, raw_piece)
            if ev is not None:
                return ev
        if res is None:                      # nothing to compare with: every sign is new
            return self._unknown(piece, t, 0.0, None)
        if res.kind == 'clarify':
            return self._clarify(res.label, res.confidence, res.candidates, 'personal', t, window=piece)
        return self._unknown(piece, t, 0.0, res)

    def _split_decision(self, piece, t, whole, holds):
        """Whole episode vs the pieces between its holds (hands stayed up, hands were still >= 0.6 s).
        Returns events, or None to fall through to the whole-piece handling."""
        cuts, lo = [], 0
        for hs, he in sorted(holds):
            cuts.append((lo, hs + 1))
            lo = he
        cuts.append((lo, len(piece)))
        parts = []
        for lo, hi in cuts:
            sub = piece[lo:hi]
            if len(sub) >= self.MIN_PIECE_FRAMES and self.matcher.n_keys(sub) >= 2:
                parts.append((sub, self._personal(sub)))
        if len(parts) < 2:
            return None                      # the holds were only leading / trailing stillness
        signs = [(sub, r) for sub, r in parts if r.kind == 'sign']
        if whole.kind == 'sign':
            # parts must beat the whole clearly to overrule it: a pause inside one sign makes its
            # first half look like something (or like itself) too
            if len(signs) == len(parts):
                rw = whole.distance / max(whole.threshold, 1e-6)
                rp = np.mean([r.distance / max(r.threshold, 1e-6) for _, r in signs])
                if rp < 0.7 * rw:
                    return [self._sign(r.label, r.confidence, 'personal', t) for _, r in signs]
            return None
        if not signs:
            return None
        events = []
        for sub, r in parts:
            if r.kind == 'sign':
                events.append(self._sign(r.label, r.confidence, 'personal', t))
            else:
                events += self._unknown(sub, t, 0.0, r)
        return events

    def _unknown(self, piece, t, conf, res):
        if len(piece) < self.MIN_PIECE_FRAMES:
            return []
        hands = (~np.isnan(piece[:, F.N_POSE:, 0])).any(axis=1).mean()
        if hands < self.MIN_HAND_FRAC:
            return []
        if res is not None:                  # static / featureless piece: not a sign
            q = self.matcher.distances(piece, trim=False)[1]
            if q < 3:
                return []
        elif len(piece) < 2 * self.MIN_PIECE_FRAMES:
            return []
        return [Event('unknown', None, float(conf), 0.0, [], t, window=piece.copy(), source='personal')]

    def _clarify(self, label, conf, cands, source, t, window=None):
        """Open a question. `window` = the raw frames of THAT segment only (kept with the question so a
        confirmed answer can be learned; also carried by the event)."""
        win = None if window is None else np.array(window, copy=True)
        self.pending = {'t': t, 'candidates': list(cands), 'source': source, 'window': win, 'fps': self._fps()}
        if self.finger_answers:
            self.picker.arm(t, min(len(list(cands)), self.MAX_SUGGEST) + 2)    # 1 = skip, 2 = retry, 3.. = the suggested words
        else:
            self.picker.reset()                                                  # fingers are off: nothing is armed
        self._moving_since = None
        self.finger = (0, 0.0)
        return [Event('clarify', label, float(conf), 0.0, list(cands), t, window=win, source=source)]

    # ------------------------------------------------------------------ optional Keras path
    def _crops(self, raw_piece, piece=None):
        """Model inputs: the span WITH its approach/retreat (the Keras model applies its own
        training-time cut, features.trim_for_model), the clean transition-free piece, and a
        head-trimmed and a tail-trimmed variant (robust to a ragged cut)."""
        n = len(raw_piece)
        k = max(int(0.12 * n), 1)
        out = [raw_piece]
        if piece is not None and piece is not raw_piece:
            out.append(piece)
        if n >= 16:
            out += [raw_piece[k:], raw_piece[:-k]]
        return out

    def _model_decision(self, piece, t, personal, raw_piece=None):
        t0 = time.perf_counter()
        crops = self._crops(piece if raw_piece is None else raw_piece, piece)
        use = crops[:self.LOAD_LEVELS[self.load]]
        self.skipped += len(crops) - len(use)
        x = np.stack([F.clip_to_tensor(c) for c in use])
        probs = np.asarray(self._model(x))
        self._load_control((time.perf_counter() - t0) * 1000, len(use))
        k = int(np.argmax(probs.max(axis=1)))
        p = probs[k]
        self.scale_used = 'model'
        top = int(np.argmax(p))
        conf = float(p[top])
        if top == self.none_idx:
            return [] if personal is None else None      # 'none' = silence when it is the only path
        label = self.classes[top]
        disp = self.label_map.get(label, label)
        if conf >= self.threshold:
            return [self._sign(disp, conf, 'model', t)]
        if personal is not None and personal.kind == 'clarify':
            return None                      # the personal question stands
        if conf >= self.clarify_floor:
            order = np.argsort(p)[::-1]
            cands = [(self.label_map.get(self.classes[i], self.classes[i]), float(p[i]))
                     for i in order if i != self.none_idx][:2]
            return self._clarify(disp, conf, cands, 'model', t)
        return None

    def _finish_sentence(self, t):
        if not self.words:
            return []
        if not any(isinstance(w, str) for w in self.words):
            self.words = []
            self.sentence_flushed = True
            return []
        s = " ".join(w for w in self.words if isinstance(w, str))      # open-question placeholders are not words
        self.words = []
        self.sentence_flushed = True
        return [Event('sentence', s, 1.0, 0.0, [], t)]

    # ------------------------------------------------------------------ public control API
    def suspend(self, flag: bool = True) -> None:
        """Pause recognition (e.g. while recording a new sign). Resuming clears partial state
        (buffer, episode, pending question) but keeps the finished words of the sentence."""
        flag = bool(flag)
        if flag == self.suspended:
            return
        self.suspended = flag
        words = self.words
        self.reset()
        self.words = words
        self.sentence_flushed = not words

    def confirm(self, label: str) -> list:
        """The signer picked a candidate (display label or raw model class)."""
        src = (self.pending or {}).get('source')
        if src != 'personal':
            raw = label
            if label not in self.classes:
                inv = [k for k, v in self.label_map.items() if v == label]
                raw = inv[0] if inv else label
            label = self.label_map.get(raw, raw)
        hole = (self.pending or {}).get('hole')
        learn = self._learn_hint(label, self.pending) if src == 'personal' else None
        self.pending = None
        self.picker.reset(); self.finger = (0, 0.0)
        if hole is not None and any(w is hole for w in self.words):
            self.words = [label if w is hole else w for w in self.words]    # the answer takes the question's place
        else:
            self.words.append(label)
        self.sentence_flushed = False
        return [Event('sign', label, 1.0, 0.0, [], self.last_t, source=src or 'model', learn=learn, answered=True)]

    def _learn_hint(self, label, pending):
        """Learn from a correction: the confirmed segment may become another example of `label`.
        None = nothing to learn (learning off, no window, label not switched on). Otherwise a dict with
        ok=False + reason when the quality gate refuses (too short / hands missing / movement too
        different from the chosen sign: probably a wrong tap / just learned), or ok=True + the window."""
        win = (pending or {}).get('window')
        if not self.learning or win is None or not len(win):
            return None
        sign_id = self._label_ids.get(label)
        if label not in self.matcher.labels or sign_id is None:     # not switched on: teaches nothing
            return None
        fps = float((pending or {}).get('fps') or self._fps())
        out = dict(ok=False, reason="", label=label, sign_id=sign_id, window=None, fps=fps)
        hands = float((~np.isnan(win[:, F.N_POSE:, 0])).any(axis=1).mean())
        if len(win) < self.LEARN_MIN_S * fps:
            out['reason'] = "too short"
            return out
        if hands < self.LEARN_MIN_HANDS:
            out['reason'] = "hands not visible"
            return out
        li = self.matcher.labels.index(label)
        d = float(self.matcher.distances(win, fps, trim=False)[0][li])
        if not np.isfinite(d) or d > self.LEARN_MAX_RATIO * float(self.matcher.thr[li]):
            out['reason'] = "movement too different"
            return out
        last = self._learned_t.get(label)
        if last is not None and 0 <= self.last_t - last < self.LEARN_DUP_S:
            out['reason'] = "duplicate"
            return out
        self._learned_t[label] = self.last_t
        out.update(ok=True, window=np.array(win, copy=True))
        return out

    def reject(self) -> None:
        """'Retry': drop the open question; the signer will sign that word again."""
        self._drop_pending()

    def skip(self) -> None:
        """'Skip': drop the open question and its word; carry on listening for the next sign of the sentence."""
        self._drop_pending()

    def clear_words(self) -> None:
        """Drop the words collected so far (and any open question)."""
        self.words = []
        self.pending = None
        self.sentence_flushed = True

    def undo_last_word(self) -> None:
        """Drop the newest collected word (and any open question)."""
        real = [i for i, w in enumerate(self.words) if isinstance(w, str)]
        if real:
            self.words.pop(real[-1])
        self._drop_pending()
        self.sentence_flushed = not self.words

    def flush_sentence(self) -> list:
        self._drop_pending()
        return self._finish_sentence(self.last_t)

    # ------------------------------------------------------------------ load control + stats
    def _load_control(self, ms, n_windows):
        """Model path: escalate when an evaluation exceeds the budget; recover when the estimated
        cost of a full evaluation is comfortably below it."""
        self.lat_eval.append(ms)
        per = ms / max(n_windows, 1)
        self.ms_per_window = per if self.ms_per_window is None else 0.5 * self.ms_per_window + 0.5 * per
        self._since_change += 1
        if self._since_change < 1:
            return
        avg = float(np.mean(list(self.lat_eval)[-2:]))
        if avg > self.budget and self.load < len(self.LOAD_LEVELS) - 1:
            self.load += 1; self._since_change = 0
        elif self.load > 0 and self.ms_per_window * self.LOAD_LEVELS[0] < 0.6 * self.budget:
            self.load -= 1; self._since_change = 0

    def stats(self) -> dict:
        w = self._wall
        fps = (len(w) - 1) / (w[-1] - w[0]) if len(w) > 1 and w[-1] > w[0] else 0.0
        lat = float(np.mean(self.lat_eval)) if self.lat_eval else (
            float(np.mean(self.dec_ms)) if self.dec_ms else 0.0)
        return {'fps': fps, 'latency_ms_avg': lat, 'scale_used': self.scale_used,
                'frames_skipped': self.skipped, 'load_level': self.load,
                'frame_ms_avg': float(np.mean(self._lat_all)) if self._lat_all else 0.0,
                'decision_ms_avg': float(np.mean(self.dec_ms)) if self.dec_ms else 0.0,
                'templates': self.matcher.n_takes, 'model_enabled': self.model_enabled,
                'signing': bool(self.zone.in_zone and not self.suspended), 'asking': self.pending is not None,
                'finger': tuple(self.finger), 'finger_seen': self.picker.seen, 'finger_hand': self.picker.hand,
                'finger_options': (self.picker.options if self.pending is not None else 0)}
