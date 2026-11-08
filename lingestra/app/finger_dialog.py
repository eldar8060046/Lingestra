"""'Finger numbers': check what the app sees when you hold up fingers, and teach it your own finger poses.

Answering 'which word did you mean?' by holding up 1, 2 or 3 fingers works without any training (the app
counts extended fingers from your hand landmarks). If it does not pick up YOUR hand reliably (closer camera,
tilted hand, different thumb), this window shows live what it reads and can learn your 1 / 2 / 3 in about
half a minute. Only numbers describing finger positions are stored (never pictures), in the encrypted settings.
"""
from collections import Counter, deque

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication, QDialog, QHBoxLayout, QLabel, QProgressBar, QVBoxLayout

from ..live.fingers import ASPECT, FingerCalibrator, _hands, describe_hand, finger_features, hand_visible, hands_count
from . import theme
from .widgets import btn, heading, hint, plain_label


def plural(n):
    return f"{n} finger" + ("" if n == 1 else "s")


class FingerDialog(QDialog):
    def __init__(self, win, teach=False):
        super().__init__(win)
        self.win = win
        self.setWindowTitle("Finger numbers")
        self.setStyleSheet(f"QDialog{{background:{theme.BG};}}")
        self.resize(560, 520)
        self.runner = None
        self._log = deque(maxlen=60)       # latest readings: (number read | None, five measurements | None, hand visible)
        self.aspect = ASPECT
        self._hold_done = 0                # frames the 'Done!' message stays before the live readout comes back
        v = QVBoxLayout(self); v.setContentsMargins(20, 18, 20, 18); v.setSpacing(10)
        v.addWidget(heading("Choose a word with your fingers", 22))
        v.addWidget(hint("When I am not sure, I show numbered buttons: 1 = skip, 2 = retry, 3 and up = the words I suggest. "
                         "Hold up that many fingers and keep your hand steady. For 6 to 10, show 5 on one hand and the rest "
                         "on the other. Hold one hand up in front of the camera, palm toward you, the other fingers folded in.", 14))
        self.big = QLabel("?"); self.big.setAlignment(Qt.AlignCenter); self.big.setMinimumHeight(130)
        self.big.setStyleSheet(f"font-size:96px;font-weight:bold;color:{theme.PRIMARY};background:#ffffff;border-radius:16px;")
        v.addWidget(self.big)
        self.msg = plain_label("Hold a hand up in front of the camera.", wrap=True, style="font-size:17px;font-weight:bold;")
        v.addWidget(self.msg)
        self.bar = QProgressBar(); self.bar.setRange(0, 100); self.bar.setTextVisible(False); self.bar.setFixedHeight(14)
        self.bar.hide(); v.addWidget(self.bar)
        self.detail = plain_label("", wrap=True, style=f"font-size:13px;color:{theme.MUTED};")
        v.addWidget(self.detail)
        self.mode_note = plain_label("", wrap=True, style=f"font-size:14px;color:{theme.MUTED};")
        v.addWidget(self.mode_note)
        row = QHBoxLayout(); row.setSpacing(8)
        self.b_teach = btn("Teach me my fingers", "primary", 50); self.b_teach.clicked.connect(self.start_teaching)
        self.b_reset = btn("Use the standard way", "", 50); self.b_reset.clicked.connect(self.reset_cal)
        row.addWidget(self.b_teach, 2); row.addWidget(self.b_reset, 1)
        v.addLayout(row)
        self.b_copy = btn("Copy my readings (to send to support)", "", 40); self.b_copy.clicked.connect(self.copy_report)
        self.b_copy.setToolTip("Copies a few numbers about what I read from your hand (never a picture) so you can paste them in a message")
        v.addWidget(self.b_copy)
        self.b_done = btn("Done", "", 46); self.b_done.clicked.connect(self.accept)
        v.addWidget(self.b_done)
        self._refresh_mode()
        if teach:
            self.start_teaching()

    # ------------------------------------------------------------------ state
    @property
    def cal(self):
        return self.win.finger_cal

    def _refresh_mode(self):
        self.b_reset.setEnabled(bool(self.cal))
        self.mode_note.setText("I am using your own finger poses." if self.cal
                               else "I am using the standard way of counting fingers.")
        self.b_teach.setText("Teach me my fingers" if self.runner is None else "Start again")

    def start_teaching(self):
        self.runner = FingerCalibrator()
        self.bar.setValue(0); self.bar.show()
        self.big.setText("1"); self.msg.setText(self.runner.message)
        self.b_teach.setText("Start again")

    def report(self):
        """A short text about the latest readings (numbers only) for the person helping with a problem."""
        log = list(self._log)
        counts = Counter("none" if n is None else str(n) for n, _v, _h in log)
        vs = np.array([v for _n, v, _h in log if v is not None])
        med = ("index/middle/ring/pinky/thumb median " + " ".join(f"{x:.2f}" for x in np.median(vs, 0)) +
               "; spread " + " ".join(f"{x:.2f}" for x in vs.std(0))) if len(vs) else "no hand measured"
        return (f"Lingestra finger check: {len(log)} frames, hand visible in {sum(1 for *_x, h in log if h)}, aspect {self.aspect:.2f}, "
                f"own poses {'yes' if self.cal else 'no'}; number read: {dict(counts)}; {med}")

    def copy_report(self):
        text = self.report()
        QApplication.clipboard().setText(text)
        self.msg.setText("Copied. Paste it into your message. (Only numbers about your fingers, never a picture.)")
        return text

    def reset_cal(self):
        self.win.set_finger_cal(None)
        self.runner = None; self.bar.hide()
        self._refresh_mode()

    # ------------------------------------------------------------------ live frames (called by the main window)
    @staticmethod
    def _why(lm, aspect=ASPECT):
        """Plain words for what is unclear: a half-bent finger, or the general advice."""
        for h in _hands(lm):
            d = describe_hand(h, aspect)
            if d:
                bent = [n for n, _r, st in d[:4] if st == "unclear"]
                if bent:
                    one = len(bent) == 1
                    return (f"The {' and '.join(bent)} finger{'' if one else 's'} look{'s' if one else ''} half-bent: "
                            f"straighten {'it' if one else 'them'} or fold {'it' if one else 'them'} in fully.")
        return "Hold it up with the palm toward the camera, fingers apart, and the other fingers folded in."

    def feed(self, lm, aspect=None):
        if lm is None:
            return
        aspect = aspect or ASPECT
        if self.runner is not None:
            self.runner.aspect = aspect
            st = self.runner.feed(lm)
            self.bar.setValue(int(round(st["progress"] * 100)))
            self.msg.setText(st["message"])
            if st["done"]:
                if st["ok"]:
                    self.win.set_finger_cal(st["cal"])
                    self.big.setText("✓")
                    self.runner = None; self.bar.hide()
                    self._refresh_mode()
                    self.msg.setText("Done! I will now recognise your finger numbers. Try holding up 1, 2 or 3.")
                    self._hold_done = 30                           # about 2 seconds of camera frames
                else:
                    self.big.setText("?")
                    self.runner = None
                    self.b_teach.setText("Try again")
            elif st["target"] is not None:
                self.big.setText(str(st["target"]))
            return
        if self._hold_done > 0:
            self._hold_done -= 1
            return
        n = hands_count(lm, self.cal, aspect)
        self.aspect = aspect
        hs = _hands(lm)
        v = finger_features(hs[0], aspect) if hs else None
        self._log.append((n, None if v is None else [float(x) for x in v], bool(hs)))
        self.detail.setText("" if v is None else
                            f"What I measure: index {v[0]:.2f}  middle {v[1]:.2f}  ring {v[2]:.2f}  pinky {v[3]:.2f}  thumb {v[4]:.2f}"
                            "   (up at 1.5 or more, folded at 1.2 or less). Only HOW MANY fingers are up counts - the names can be "
                            "mixed up when a fist hides your thumb.")
        if n:
            self.big.setText(str(n)); self.msg.setText(f"I see {plural(n)}.")
            return
        self.big.setText("?")
        if hand_visible(lm):
            self.msg.setText("I can see your hand but not a clear number. " + self._why(lm, aspect))
        else:
            self.msg.setText("I can't see a hand. Hold one hand up in front of the camera.")
