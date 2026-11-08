"""Recognition worker thread. All heavy work (camera, MediaPipe, model, sending) happens here;
the UI only receives Qt signals. Other modules are imported lazily, strictly via the agreed
interfaces (Recognizer, sources, ChatSender, SecureStore)."""
import queue
import time

import numpy as np
from PySide6.QtCore import QThread, Signal


def make_source(spec: str, loop_speed=1.0):
    """'webcam' | 'video:<path>' | 'landmarks:<path>' -> frame source object."""
    from ..live import sources
    if spec.startswith("video:"):
        return sources.VideoFileSource(spec[6:], speed=loop_speed)
    if spec.startswith("landmarks:"):
        return sources.LandmarkFileSource(spec[10:], speed=loop_speed)
    return sources.WebcamSource(0)


class RecognitionWorker(QThread):
    preview = Signal(object, object)   # (frame_bgr ndarray | None, landmarks (67,3))
    events = Signal(object)            # list[Event]
    stats = Signal(dict)
    sent = Signal(str, bool, str)      # (sentence text, ok, detail)
    unknown_saved = Signal(str)        # SecureStore id
    failed = Signal(str)
    countdown = Signal(float)          # seconds left before a sentence is pasted (0 = finished / cancelled)
    capture_state = Signal(object)     # recorder.RecState on every visible change while teaching a sign
    learn_ready = Signal(dict)         # a confirmed correction passed the quality gate: {label, sign_id, window, fps}
    learn_skipped = Signal(str)        # a confirmed correction was NOT learned (reason in plain words)
    capture_result = Signal(object, float, int)   # (final RecState: done carries .clip | failed, source fps, session id)

    def __init__(self, source_spec, model_path, threshold=0.80, send_mode="clipboard",
                 loop=False, store=None, parent=None,
                 countdown_s=3.0, press_enter=False, use_model=None, templates=None):
        super().__init__(parent)
        self.source_spec, self.model_path = source_spec, model_path
        self.threshold, self.send_mode, self.loop = threshold, send_mode, loop
        self.paused = False             # 'Pause sending' toggle (plain bool; atomic enough)
        self.store = store
        self.countdown_s, self.press_enter = countdown_s, press_enter
        self.use_model = use_model      # None = automatic (model only while there are no templates)
        self.learning = True            # 'learn from my corrections'
        self.continuous = True          # continuous signing (several signs without lowering the hands)
        self.finger_cal = None          # the signer's own taught finger poses (None = standard finger counting)
        self.hold_commit_s = None       # how long holding still ends a word (None = the recogniser's default)
        self.auto_finish = True         # False: the message is only finished by Send now / Space / the separator gesture
        self.finger_answers = True      # False: questions are answered by tap / keys only
        self._templates0 = templates    # applied at start (set_templates() before start() also works)
        self._cmds = queue.Queue()      # UI -> worker commands, drained between frames
        self._stop = False
        self._cancel_send = False       # UI 'Cancel' for a pending paste countdown
        self.capture_gen = 0            # id of the newest start_capture (UI thread writes, results echo it)

    # ---- thread-safe controls called from the UI thread ----
    def confirm(self, label): self._cmds.put(("confirm", label))
    def reject(self): self._cmds.put(("reject", None))
    def skip(self): self._cmds.put(("skip", None))
    def flush(self): self._cmds.put(("flush", None))
    def set_threshold(self, v): self._cmds.put(("threshold", float(v)))
    def stop(self): self._stop = True
    def cancel_send(self): self._cancel_send = True      # stops a paste countdown in progress
    def start_capture(self):
        self.capture_gen += 1
        self._cmds.put(("start_capture", self.capture_gen))
    def cancel_capture(self): self._cmds.put(("cancel_capture", None))
    def set_templates(self, templates):
        """Personal templates (SignLibrary.templates()); applied between frames, thread-safe."""
        self._cmds.put(("templates", list(templates or [])))
    def clear_sentence(self): self._cmds.put(("clear_sentence", None))
    def undo_last_word(self): self._cmds.put(("undo_last_word", None))
    def set_continuous(self, flag):
        """Several signs in a row without lowering the hands (Recognizer.continuous); thread-safe."""
        self.continuous = bool(flag)
        self._cmds.put(("continuous", bool(flag)))
    def set_learning(self, flag):
        """Learn from corrections on / off (Recognizer.learning); thread-safe."""
        self.learning = bool(flag)
        self._cmds.put(("learning", bool(flag)))
    def set_use_model(self, flag): self._cmds.put(("use_model", None if flag is None else bool(flag)))
    def set_finger_answers(self, flag):
        """Answer questions by holding up fingers (True) or only by tap / keys (False); thread-safe."""
        self.finger_answers = bool(flag)
        self._cmds.put(("finger_answers", self.finger_answers))
    def set_auto_finish(self, flag):
        """True: a message ends when you lower your hands. False ('send with Space'): it keeps growing until you send it; thread-safe."""
        self.auto_finish = bool(flag)
        self._cmds.put(("auto_finish", self.auto_finish))
    def set_hold_commit(self, seconds):
        """How long holding your hands still ends the word before it (seconds; None = default); thread-safe."""
        self.hold_commit_s = None if seconds is None else float(seconds)
        self._cmds.put(("hold_commit", self.hold_commit_s))
    def set_finger_cal(self, cal):
        """The signer's taught finger poses for answering questions with 1 / 2 / 3 fingers (None = standard); thread-safe."""
        self.finger_cal = dict(cal) if cal else None
        self._cmds.put(("finger_cal", self.finger_cal))
    def set_label_map(self, mapping): self._cmds.put(("label_map", dict(mapping or {})))   # copy: thread-safe

    # ---- internals ----
    def _handle_events(self, evs, window):
        if not evs:
            return
        for e in evs:
            ew = getattr(e, "window", None)
            win = ew if ew is not None and len(ew) else window
            if e.kind == "unknown" and self.store is not None and win is not None and len(win):
                try:
                    self.unknown_saved.emit(self.store.save_unknown(
                        np.asarray(win), {"t": e.t, "confidence": e.confidence}))
                except Exception as ex:  # storage trouble must never kill recognition
                    self.failed.emit(f"could not store unknown sign: {ex}")
        hints = []
        for e in evs:                    # the learn hint (with its landmark window) is not part of what the UI sees
            if e.kind == "sign" and getattr(e, "learn", None):
                hints.append(e.learn)
                e.learn = None
        self.events.emit(evs)            # before sending: the UI adds the history row that `sent` then updates
        for hint in hints:
            if hint.get("ok"):
                self.learn_ready.emit(dict(hint))
            else:
                self.learn_skipped.emit(str(hint.get("reason", "")))
        for e in evs:
            if e.kind == "sentence" and e.label:
                self._send(e.label)

    def _send(self, text):
        if self.send_mode == "none" or self.paused:
            self.sent.emit(text, False, "not sent (paused)" if self.paused else "not sent (send mode: none)")
            return
        self._cancel_send = False
        try:
            res = self._sender.send(text)
            self.sent.emit(text, bool(res.ok), f"{res.method}: {res.detail}")
        except Exception as ex:
            self.sent.emit(text, False, f"send error: {ex}")
        finally:
            self.countdown.emit(0.0)

    def run(self):
        try:
            from ..live.recognizer import Recognizer
            from ..output.chat_sender import ChatSender
            rec = Recognizer(self.model_path, threshold=self.threshold, use_model=self.use_model,
                             continuous=self.continuous)
            rec.learning = self.learning
            rec.set_finger_cal(self.finger_cal)
            rec.auto_finish = self.auto_finish
            rec.set_finger_answers(self.finger_answers)
            if self.hold_commit_s is not None:
                rec.hold_commit_s = self.hold_commit_s
            if self._templates0:
                rec.set_templates(self._templates0)
            # countdown gives the signer time to focus the chat window; never inject keys instantly
            self._sender = ChatSender(mode="clipboard", press_enter=self.press_enter,
                                      countdown_s=self.countdown_s)
            # re-checked every 0.1 s during the countdown and just before pasting
            self._sender.should_abort = lambda: (self._cancel_send or self.paused or self._stop
                                                 or self.send_mode == "none")
            self._sender.on_tick = lambda left: self.countdown.emit(float(left))
        except Exception as ex:
            self.failed.emit(f"startup failed: {ex}")
            return
        from ..live.recorder import SignRecorder
        recorder = SignRecorder()
        cap = {"on": False, "pending": False, "times": [], "key": None, "gen": 0}   # capture session state

        def end_capture(st=None, fps=25.0):
            gen = cap["gen"]
            cap.update(on=False, pending=False, times=[], key=None)
            rec.suspend(False)
            if st is not None:
                self.capture_result.emit(st, float(fps), int(gen))

        recent = []                      # rolling landmark window for saving unknown signs
        last_prev = 0.0
        try:
            while not self._stop:
                src = make_source(self.source_spec)
                is_live = self.source_spec == "webcam"
                wall0 = t0 = None
                if cap["on"]:            # source restarted (loop): timestamps jumped back
                    from ..live.recorder import RecState
                    end_capture(RecState("failed", "Source restarted during recording", 0.0, 0), 25.0)
                for lm, t in src:
                    if self._stop:
                        break
                    pending = []
                    while not self._cmds.empty():
                        pending.append(self._cmds.get())
                    last_t = max((i for i, (c_, _) in enumerate(pending) if c_ == "templates"), default=-1)
                    # a burst of library changes queues many 'templates' commands: only the newest matters
                    pending = [cv for i, cv in enumerate(pending) if cv[0] != "templates" or i == last_t]
                    for c, v in pending:
                        if c == "start_capture":
                            cap.update(on=True, pending=True, times=[], key=None, gen=v); rec.suspend(True)
                            continue
                        if c == "cancel_capture":
                            recorder.cancel()
                            if cap["on"]:
                                end_capture()
                            continue
                        if c == "label_map":
                            rec.label_map = dict(v)
                            continue
                        if c == "finger_answers":
                            rec.set_finger_answers(v)
                            continue
                        if c == "auto_finish":
                            rec.auto_finish = bool(v)
                            continue
                        if c == "hold_commit":
                            rec.hold_commit_s = rec.HOLD_COMMIT_S if v is None else float(v)
                            continue
                        if c == "finger_cal":
                            rec.set_finger_cal(v)
                            continue
                        if c == "templates":
                            try:
                                rec.set_templates(v)
                            except Exception as ex:      # bad template data must not stop recognition
                                self.failed.emit(f"could not load signs for recognition: {ex}")
                            continue
                        if c == "clear_sentence":
                            rec.clear_words()
                            continue
                        if c == "undo_last_word":
                            rec.undo_last_word()
                            continue
                        if c == "continuous":
                            rec.set_continuous(v)
                            continue
                        if c == "learning":
                            rec.learning = bool(v)
                            continue
                        if c == "use_model":
                            rec.use_model = v
                            continue
                        evs = {"confirm": lambda: rec.confirm(v), "reject": lambda: rec.reject(), "skip": lambda: rec.skip(),
                               "flush": rec.flush_sentence}.get(c, lambda: None)()
                        if c == "threshold":
                            rec.threshold = v
                        self._handle_events(evs or [], np.asarray(recent))
                    if not is_live:      # pace file sources to real time
                        wall0 = time.time() if wall0 is None else wall0
                        t0 = t if t0 is None else t0
                        lag = (t - t0) - (time.time() - wall0)
                        if lag > 0:
                            time.sleep(min(lag, 0.2))
                    a_ = getattr(src, "aspect", None)
                    if a_:
                        rec.aspect = float(a_)                   # finger readings use depth: they need the frame shape
                    recent.append(lm); del recent[:-90]
                    if cap["on"]:        # teaching a sign: recognition is suspended, only the recorder sees frames
                        if cap["pending"]:
                            recorder.start(t); cap["pending"] = False
                        cap["times"].append(t)
                        st = recorder.push(lm, t)
                        key = (st.state, st.message, st.frames // 5)
                        if key != cap["key"]:
                            cap["key"] = key; self.capture_state.emit(st)
                        if st.state in ("done", "failed"):
                            ts = cap["times"]
                            fps = (len(ts) - 1) / (ts[-1] - ts[0]) if len(ts) > 2 and ts[-1] > ts[0] else 25.0
                            end_capture(st, min(max(fps, 5.0), 60.0))
                    else:
                        self._handle_events(rec.push_frame(lm, t), np.asarray(recent))
                    now = time.time()
                    if now - last_prev > 1 / 15:       # throttle UI updates to ~15 Hz
                        last_prev = now
                        self.preview.emit(getattr(src, "last_frame_bgr", None), lm)
                        self.stats.emit(rec.stats())
                if hasattr(src, "close"):
                    src.close()
                if cap["on"] and not (self.loop and not self._stop):
                    from ..live.recorder import RecState
                    end_capture(RecState("failed", "Source ended before the sign was captured", 0.0, 0), 25.0)
                if not (self.loop and not self._stop):
                    break
                self._handle_events(rec.flush_sentence(), np.asarray(recent))
        except Exception as ex:
            self.failed.emit(f"recognition stopped: {ex}")
