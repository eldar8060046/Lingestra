"""The big 'main view' on the left: live camera <-> skeleton stage (playback / review of a clip).

Modes: 'live' (camera), 'capture' (camera + big overlay while teaching a sign), 'playback'
(skeleton animation of a library sign or phrase), 'review' (captured clip looping with
Keep / Retake / Discard). All text shown here may come from the user and is plain text.
"""
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QElapsedTimer, QTimer, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (QButtonGroup, QComboBox, QFrame, QGridLayout, QHBoxLayout,
                               QLineEdit, QProgressBar, QPushButton, QStackedWidget, QVBoxLayout, QWidget)

from . import theme
from .widgets import (CameraView, SkeletonCanvas, auto_direction, btn, mirror_button, plain_label, TickBox)

FPS = 25.0
GAP_FRAMES = 6            # hold on the last pose between words / loops
MAX_CHIPS = 8
SPEEDS = ((0.5, "Slow"), (1.0, "Normal"), (2.0, "Fast"))


@dataclass
class Segment:
    title: str
    frames: np.ndarray
    subtitle: str = ""


class MainStage(QWidget):
    back_to_live = Signal()
    take_changed = Signal(int)
    keep = Signal(str)
    retake = Signal()
    discard = Signal()
    capture_retry = Signal()
    capture_cancel = Signal()
    segment_changed = Signal(int, str)
    finished = Signal()
    mirror_toggled = Signal(bool)        # the user pressed a mirror button

    def __init__(self, parent=None):
        super().__init__(parent)
        self.mode = "live"
        self.segs: list[Segment] = []
        self.si = 0; self.pos = 0.0; self.speed = 1.0; self.playing = False
        self.mirror = True
        self._clock = QElapsedTimer()
        self._timer = QTimer(self); self._timer.setInterval(33); self._timer.timeout.connect(self._on_timer)
        lay = QVBoxLayout(self); lay.setContentsMargins(0, 0, 0, 0)
        self.stack = QStackedWidget(); lay.addWidget(self.stack)
        self._build_camera_page()
        self._build_skeleton_page()
        self.set_mirror(True)
        for key, fn in (("K", self._do_keep), ("T", self._do_retake), ("D", self._do_discard),
                        ("Alt+K", self._do_keep), ("Alt+T", self._do_retake), ("Alt+D", self._do_discard)):
            QShortcut(QKeySequence(key), self, activated=fn)

    # ------------------------------------------------------------------ mirror (display only)
    def set_mirror(self, on):
        """Flip everything shown here (camera + overlay, skeleton playback / review). Never touches data."""
        on = bool(on); self.mirror = on
        self.camera.set_mirror(on); self.canvas.set_mirror(on)
        for b in (self.mirror_btn, self.mirror_btn2):
            b.blockSignals(True); b.setChecked(on); b.blockSignals(False); b.refresh()
        self._update_note()

    def _user_mirror(self, on):
        self.set_mirror(on); self.mirror_toggled.emit(bool(on))

    # ------------------------------------------------------------------ construction
    def _build_camera_page(self):
        page = QWidget(); g = QGridLayout(page); g.setContentsMargins(0, 0, 0, 0)
        self.camera = CameraView(); g.addWidget(self.camera, 0, 0)
        self.mirror_btn = mirror_button(); self.mirror_btn.toggled.connect(self._user_mirror)
        g.addWidget(self.mirror_btn, 0, 0, Qt.AlignTop | Qt.AlignRight)
        g.setRowMinimumHeight(0, 0)
        self.overlay = QFrame(); self.overlay.setObjectName("overlay")
        self.overlay.setStyleSheet("QFrame#overlay{background:rgba(10,14,20,205);border-radius:20px;}"
                                   "QLabel{background:transparent;}")
        ol = QVBoxLayout(self.overlay); ol.setContentsMargins(40, 24, 40, 24)
        self.ov_big = plain_label("", style="font-size:96px;font-weight:bold;color:#f7fafc;")
        self.ov_big.setAlignment(Qt.AlignCenter)
        self.ov_sub = plain_label("", wrap=True, style="font-size:26px;color:#e2e8f0;")
        self.ov_sub.setAlignment(Qt.AlignCenter)
        row = QHBoxLayout()
        self.ov_retry = btn("Try again", "primary", 52); self.ov_cancel = btn("Cancel", "", 52)
        for b in (self.ov_retry, self.ov_cancel):
            b.setStyleSheet("font-size:19px;"); row.addWidget(b)
        self.ov_retry.clicked.connect(self.capture_retry); self.ov_cancel.clicked.connect(self.capture_cancel)
        ol.addWidget(self.ov_big); ol.addWidget(self.ov_sub); ol.addLayout(row)
        g.addWidget(self.overlay, 0, 0, Qt.AlignCenter)
        self.overlay.setMaximumWidth(640)
        self.overlay.hide()
        self.stack.addWidget(page); self.camera_page = page

    def _build_skeleton_page(self):
        page = QWidget(); v = QVBoxLayout(page); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(8)
        self.title = plain_label("", wrap=True, style=f"font-size:32px;font-weight:bold;color:{theme.TEXT};")
        self.title.setMinimumHeight(44)
        self.subtitle = plain_label("", wrap=True, style=f"font-size:16px;color:{theme.MUTED};")
        self.chips_row = QHBoxLayout(); self.chips_row.setSpacing(6); self._chips = []
        self.canvas = SkeletonCanvas(); self.canvas.setMinimumSize(420, 300)
        self.progress = QProgressBar(); self.progress.setRange(0, 1000); self.progress.setTextVisible(False)
        self.progress.setFixedHeight(8)
        v.addWidget(self.title); v.addWidget(self.subtitle); v.addLayout(self.chips_row)
        v.addWidget(self.canvas, 1); v.addWidget(self.progress)

        # playback controls
        self.controls = QWidget(); cv = QVBoxLayout(self.controls); cv.setContentsMargins(0, 0, 0, 0); cv.setSpacing(8)
        c = QHBoxLayout(); c.setSpacing(8); cv.addLayout(c)
        c2 = QHBoxLayout(); c2.setSpacing(8); cv.addLayout(c2)
        self.play_btn = btn("Pause", "", 46); self.play_btn.setMinimumWidth(96)
        self.play_btn.clicked.connect(self.toggle_play)
        c.addWidget(self.play_btn)
        self.speed_group = QButtonGroup(self); self.speed_btns = {}
        for sp, name in SPEEDS:
            b = btn(name, "seg", 46); b.setCheckable(True); b.setChecked(sp == 1.0)
            b.setToolTip({0.5: "Half speed", 1.0: "Normal speed", 2.0: "Double speed"}[sp])
            self.speed_group.addButton(b); self.speed_btns[sp] = b; c.addWidget(b)
            b.clicked.connect(lambda _=False, s=sp: self.set_speed(s))
        self.loop_box = TickBox("Repeat"); self.loop_box.setChecked(True); c.addWidget(self.loop_box)
        self.take_box = QComboBox(); self.take_box.setMinimumHeight(40); self.take_box.setVisible(False)
        self.take_box.currentIndexChanged.connect(self._on_take); c.addWidget(self.take_box)
        c.addStretch(1)
        self.mirror_btn2 = mirror_button(); self.mirror_btn2.setStyleSheet(
            f"QPushButton{{background:#ffffff;color:{theme.TEXT};border:1px solid #cbd5e1;border-radius:21px;padding:4px 16px;"
            f"font-size:15px;font-weight:bold;}}QPushButton:checked{{background:{theme.GOOD};color:#ffffff;border-color:{theme.GOOD};}}")
        from PySide6.QtGui import QIcon
        self.mirror_btn2.setIcon(QIcon(theme.icon("mirror", 28, theme.TEXT)))
        self.mirror_btn2.toggled.connect(self._user_mirror); c2.addWidget(self.mirror_btn2); c2.addStretch(1)
        self.back_btn = btn("Back to camera", "good", 50, tip="Esc"); self.back_btn.setMinimumWidth(190)
        self.back_btn.clicked.connect(self.go_live); c2.addWidget(self.back_btn)
        v.addWidget(self.controls)

        # review bar (captured clip)
        self.review = QWidget(); r = QHBoxLayout(self.review); r.setContentsMargins(0, 0, 0, 0); r.setSpacing(8)
        self.name_edit = QLineEdit(); self.name_edit.setPlaceholderText("Type the word for this sign")
        self.name_edit.setMaxLength(80); self.name_edit.setMinimumHeight(56)
        self.name_edit.setStyleSheet("font-size:20px;")
        self.name_edit.textChanged.connect(lambda t: auto_direction(self.name_edit, t))
        self.name_edit.returnPressed.connect(self._do_keep)
        self.b_keep = btn("Keep", "good", 56, tip="K - keep this recording")
        self.b_retake = btn("Try again", "warn", 56, tip="T - record it again")
        self.b_discard = btn("Delete", "danger", 56, tip="D - throw this recording away")
        for b in (self.b_keep, self.b_retake, self.b_discard):
            b.setMinimumWidth(110); b.setStyleSheet("font-size:19px;")
        self.b_keep.clicked.connect(self._do_keep); self.b_retake.clicked.connect(self._do_retake)
        self.b_discard.clicked.connect(self._do_discard)
        r.addWidget(self.name_edit, 1)
        for b in (self.b_keep, self.b_retake, self.b_discard):
            r.addWidget(b)
        v.addWidget(self.review)
        self.review_msg = plain_label("", wrap=True, style=f"color:{theme.BAD};font-size:15px;")
        v.addWidget(self.review_msg)
        self.note = plain_label("", wrap=True, style=f"color:{theme.MUTED};font-size:14px;")
        v.addWidget(self.note)
        self.review.hide(); self.review_msg.hide()
        self.stack.addWidget(page); self.skeleton_page = page

    def _update_note(self):
        self.note.setText("Your camera is still listening in the background."
                          + ("  Shown like a mirror - copy it just as you see it." if self.mirror else ""))

    # ------------------------------------------------------------------ live / capture
    def go_live(self):
        was = self.mode
        self._stop_clock()
        self.mode = "live"; self.segs = []
        self.overlay.hide()
        self.stack.setCurrentWidget(self.camera_page)
        if was != "live":
            self.back_to_live.emit()

    def show_capture(self):
        """Camera + overlay while the signer is being recorded."""
        self._stop_clock()
        self.mode = "capture"
        self.stack.setCurrentWidget(self.camera_page)
        self.set_overlay("", "Starting...")

    def set_overlay(self, big, sub, color="#f7fafc", retry=False, compact=False):
        """compact = small banner at the top so the signer can still see themselves while recording."""
        self.ov_big.setText(big); self.ov_big.setVisible(bool(big))
        self.ov_big.setStyleSheet(f"font-size:{40 if compact else 110}px;font-weight:bold;color:{color};background:transparent;")
        self.ov_sub.setText(sub)
        self.ov_sub.setStyleSheet(f"font-size:{20 if compact else 26}px;color:#e2e8f0;background:transparent;")
        self.ov_retry.setVisible(retry)
        self.ov_cancel.setMinimumHeight(40 if compact else 52)
        self.overlay.setStyleSheet("QFrame#overlay{background:rgba(10,14,20,%d);border-radius:20px;}" % (170 if compact else 220))
        self.camera_page.layout().setAlignment(self.overlay, (Qt.AlignTop | Qt.AlignHCenter) if compact else Qt.AlignCenter)
        self.overlay.show()

    @property
    def camera_visible(self):
        return self.stack.currentWidget() is self.camera_page

    # ------------------------------------------------------------------ playback
    def play(self, segments, loop=True, takes=1, take=0, autoplay=True):
        """Show the skeleton stage and play the segments back-to-back."""
        self.segs = list(segments)
        if not self.segs:
            return
        self.mode = "playback"
        self.overlay.hide(); self.review.hide(); self.review_msg.hide()
        self.controls.show(); self.note.show(); self._update_note()
        self.loop_box.setChecked(loop)
        self.take_box.blockSignals(True); self.take_box.clear()
        for i in range(takes):
            self.take_box.addItem(f"Example {i + 1} of {takes}")
        self.take_box.setCurrentIndex(min(take, max(takes - 1, 0)))
        self.take_box.blockSignals(False); self.take_box.setVisible(takes > 1)
        self._build_chips()
        self.si, self.pos = 0, 0.0
        self._load_segment()
        self.stack.setCurrentWidget(self.skeleton_page)
        self.set_playing(autoplay)

    def replace_take(self, segment, take):
        """Swap the frames of a single-sign view (take selector) without leaving the stage."""
        self.segs = [segment]; self.si, self.pos = 0, 0.0
        self._load_segment(); self.set_playing(True)

    def _build_chips(self):
        for c in self._chips:
            c.setParent(None); c.deleteLater()
        self._chips = []
        if 1 < len(self.segs) <= MAX_CHIPS:
            for s in self.segs:
                lb = plain_label(s.title, style="padding:3px 10px;border-radius:12px;background:#e2e8f0;color:#475569;")
                self.chips_row.addWidget(lb); self._chips.append(lb)

    def _load_segment(self):
        seg = self.segs[self.si]
        self.canvas.set_frames(seg.frames); self.canvas.set_index(0)
        self.title.setText(seg.title); auto_direction(self.title, seg.title)
        self.title.setStyleSheet("font-size:%dpx;font-weight:bold;color:%s;" % (32 if len(seg.title) < 36 else 21, theme.TEXT))
        sub = seg.subtitle
        if len(self.segs) > 1:
            sub = f"Sign {self.si + 1} of {len(self.segs)}" + (f"  -  {seg.subtitle}" if seg.subtitle else "")
        self.subtitle.setText(sub); self.subtitle.setVisible(bool(sub))
        for i, c in enumerate(self._chips):
            on = i == self.si
            c.setStyleSheet("padding:3px 10px;border-radius:12px;" + (
                f"background:{theme.PRIMARY};color:#ffffff;font-weight:bold;" if on else
                "background:#e2e8f0;color:#475569;"))
        self._update_progress()

    def set_playing(self, on):
        self.playing = bool(on)
        self.play_btn.setText("Pause" if self.playing else "Play")
        if self.playing and self.segs:
            self._clock.start(); self._timer.start()
        else:
            self._stop_clock()

    def _stop_clock(self):
        self._timer.stop(); self.playing = False
        self.play_btn.setText("Play")

    def toggle_play(self):
        if self.playing:
            self.set_playing(False)
        else:
            if self.segs and self.pos >= len(self.segs[self.si].frames) - 1 and self._at_end:
                self.si, self.pos = 0, 0.0; self._load_segment()
            self._at_end = False
            self.set_playing(True)

    _at_end = False

    def set_speed(self, s):
        self.speed = float(s)
        if s in self.speed_btns:
            self.speed_btns[s].setChecked(True)

    def _on_take(self, i):
        if i >= 0 and self.mode == "playback":
            self.take_changed.emit(i)

    def _on_timer(self):
        dt = self._clock.restart() / 1000.0
        self.advance(min(dt, 0.25))

    def advance(self, dt):
        """Advance the animation clock by dt seconds (called by the timer; tests call it directly)."""
        if not self.segs or not self.playing:
            return
        self.pos += dt * FPS * self.speed
        loop = self.loop_box.isChecked()
        while True:
            n = len(self.segs[self.si].frames)
            last = self.si == len(self.segs) - 1
            total = n + (GAP_FRAMES if (not last or loop) else 0)
            if self.pos < total:
                break
            self.pos -= total
            if not last:
                self.si += 1; self._load_segment()
                self.segment_changed.emit(self.si, self.segs[self.si].title)
            elif loop:
                self.si = 0; self._load_segment()
                self.segment_changed.emit(self.si, self.segs[self.si].title)
            else:
                self.pos = n - 1; self._at_end = True
                self.canvas.set_index(n - 1); self._update_progress(1.0)
                self.set_playing(False); self.finished.emit()
                return
        self.canvas.set_index(int(self.pos)); self._update_progress()

    def _update_progress(self, forced=None):
        lens = [len(s.frames) for s in self.segs] or [1]
        done = sum(lens[:self.si]) + min(self.pos, lens[self.si])
        frac = forced if forced is not None else done / max(sum(lens), 1)
        self.progress.setValue(int(frac * 1000))

    # ------------------------------------------------------------------ review of a captured clip
    def show_review(self, frames, title, name="", need_name=True, subtitle=""):
        self.segs = [Segment(title, frames, subtitle)]; self.mode = "review"
        self.overlay.hide(); self.controls.hide(); self.note.hide()
        self.review.show(); self.name_edit.setVisible(need_name)
        self.take_box.setVisible(False)
        self.review_msg.clear(); self.review_msg.hide()
        self.loop_box.setChecked(True); self._build_chips()
        self.si, self.pos = 0, 0.0; self._load_segment()
        self.name_edit.setText(name)
        self.stack.setCurrentWidget(self.skeleton_page)
        self.set_playing(True)
        if need_name and not name:
            self.name_edit.setFocus()
        else:
            self.b_keep.setFocus()          # plain K / T / D shortcuts work when no text field has focus

    def set_review_message(self, text):
        self.review_msg.setText(text); self.review_msg.setVisible(bool(text))

    def _do_keep(self):
        if self.mode == "review":
            self.keep.emit(self.name_edit.text().strip())

    def _do_retake(self):
        if self.mode == "review":
            self.retake.emit()

    def _do_discard(self):
        if self.mode == "review":
            self.discard.emit()

    @property
    def current_title(self):
        return self.title.text()
