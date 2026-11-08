"""Small custom widgets: camera preview with landmark overlay, skeleton icons, friendly building blocks
(buttons, cards, switch, collapsible section, nav button, dots meter) and the 'did you mean' panel.

MIRROR MODE is display-only. Every drawing helper here takes a `mirror` flag that flips x when
painting; the landmark arrays themselves (which the recogniser sees) are never modified."""
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QImage, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractButton, QCheckBox, QFrame, QGridLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy,
                               QToolButton, QVBoxLayout, QWidget)

from . import theme

# Layout of the 67 points: pose 0..24, left hand 25..45, right hand 46..66.
_ARMS = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
_HAND = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11),
         (11, 12), (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20),
         (5, 9), (9, 13), (13, 17)]
BONES = _ARMS + [(a + 25, b + 25) for a, b in _HAND] + [(a + 46, b + 46) for a, b in _HAND]


def draw_skeleton(p: QPainter, lm, w, h, color=QColor("#4fd1c5"), width=2, radius=2.5):
    """Draw landmarks given in 0..1 coordinates onto a w x h area (NaN points skipped)."""
    lm = np.asarray(lm, float)
    ok = ~np.isnan(lm[:, :2]).any(axis=1)
    xy = np.nan_to_num(lm[:, :2]) * np.array([w, h])
    _paint(p, xy, ok, width, radius, head=True, body=color)


def plain_label(text="", wrap=False, style="") -> QLabel:
    """QLabel that never interprets its text as HTML (all user-provided strings go through this)."""
    lb = QLabel(); lb.setTextFormat(Qt.PlainText); lb.setText(text)
    lb.setWordWrap(wrap)
    lb.setStyleSheet("background:transparent;" + style)
    return lb


def has_rtl(text: str) -> bool:
    return any("\u0590" <= c <= "\u08ff" or "\ufb1d" <= c <= "\ufdff" or "\ufe70" <= c <= "\ufeff"
               for c in text)


def auto_direction(widget, text: str):
    """Right-to-left layout for Arabic/Hebrew text, left-to-right otherwise."""
    widget.setLayoutDirection(Qt.RightToLeft if has_rtl(text) else Qt.LeftToRight)


HEAD = QColor("#cbd5e0")


def _paint(p, xy, ok, width, radius, head=False, body=QColor("#4fd1c5"), hand=QColor("#90cdf4")):
    """Draw a stick figure from pixel coordinates xy (N,2) with validity mask ok."""
    pt = lambda i: QPointF(float(xy[i, 0]), float(xy[i, 1]))
    for k, (a, b) in enumerate(BONES):
        if ok[a] and ok[b]:
            p.setPen(QPen(body if k < len(_ARMS) else hand, width, Qt.SolidLine, Qt.RoundCap))
            p.drawLine(pt(a), pt(b))
    if head and ok[0] and ok[11] and ok[12]:
        r = float(np.hypot(*(xy[11] - xy[12]))) * 0.32
        p.setPen(QPen(HEAD, width)); p.setBrush(Qt.NoBrush); p.drawEllipse(pt(0), r, r)
    p.setBrush(QColor("#f6ad55")); p.setPen(Qt.NoPen)
    for i in range(25, len(xy)):
        if ok[i]:
            p.drawEllipse(pt(i), radius, radius)


class SkeletonCanvas(QWidget):
    """Draws one frame of a landmark clip as a 2D stick figure on a dark stage. The view box is
    fixed per clip (so the figure does not jump around) and fitted to the widget."""
    ASPECT = 4 / 3          # landmark x is normalised by width, y by height of a 4:3 camera frame

    def __init__(self, parent=None, head=True):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.frames = None; self.index = 0; self.head = head
        self.mirror = False          # display-only flip, like looking in a mirror
        self._box = (0.0, 0.0, 1.0, 1.0)
        self._timer = None; self.pos = 0.0

    def set_mirror(self, on):
        on = bool(on)
        if on != self.mirror:
            self.mirror = on; self.update()

    def set_frames(self, frames):
        self.frames = None if frames is None else np.asarray(frames, np.float32)
        self.index = 0; self.pos = 0.0
        self._box = self._fit_box(self.frames)
        self.update()

    @staticmethod
    def _fit_box(fr):
        if fr is None or len(fr) == 0:
            return (0.0, 0.0, 1.0, 1.0)
        xy = fr[:, :, :2].reshape(-1, 2); xy = xy[~np.isnan(xy).any(axis=1)]
        if len(xy) < 3:
            return (0.0, 0.0, 1.0, 1.0)
        lo, hi = np.percentile(xy, 1, axis=0), np.percentile(xy, 99, axis=0)
        pad = 0.08 * float(max(hi[0] - lo[0], hi[1] - lo[1], 0.2))
        return (float(lo[0] - pad), float(lo[1] - pad), float(hi[0] - lo[0] + 2 * pad), float(hi[1] - lo[1] + 2 * pad))

    def set_index(self, i):
        if self.frames is not None and len(self.frames):
            self.index = max(0, min(int(i), len(self.frames) - 1)); self.update()

    # simple self-driven looping (used by pending-sign cards)
    def play(self, on=True):
        if on and self.frames is not None and len(self.frames) > 1:
            if self._timer is None:
                self._timer = QTimer(self); self._timer.setInterval(40); self._timer.timeout.connect(self._step)
            self._timer.start()
        elif self._timer is not None:
            self._timer.stop()

    @property
    def is_playing(self):
        return self._timer is not None and self._timer.isActive()

    def _step(self):
        self.pos = (self.pos + 1) % len(self.frames); self.set_index(self.pos)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        path = QPainterPath(); path.addRoundedRect(QRectF(self.rect()), 14, 14)
        p.setClipPath(path)
        p.fillRect(self.rect(), QColor(theme.SCREEN))
        if self.frames is None or not len(self.frames):
            return
        x0, y0, bw, bh = self._box
        W, H = self.width(), self.height()
        s = min(W / (bw * self.ASPECT), H / bh)
        ox, oy = (W - bw * self.ASPECT * s) / 2, (H - bh * s) / 2
        f = self.frames[self.index]
        xy = np.stack([(f[:, 0] - x0) * self.ASPECT * s + ox, (f[:, 1] - y0) * s + oy], axis=1)
        ok = ~np.isnan(xy).any(axis=1)
        xy = np.nan_to_num(xy)
        if self.mirror:
            xy[:, 0] = W - xy[:, 0]
        _paint(p, xy, ok, width=max(2.0, min(W, H) / 110), radius=max(2.5, min(W, H) / 120),
               head=self.head)
        p.end()

    def sizeHint(self):
        return QSize(320, 320)


class CameraView(QLabel):
    """Shows the latest frame (or a dark canvas) with the landmark overlay on top."""

    def __init__(self):
        super().__init__()
        self.setMinimumSize(480, 360)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet(f"background:{theme.SCREEN}; border-radius:16px;")
        self._pm = None
        self._last = (None, None)
        # Selfie/mirror view like meeting apps. DISPLAY ONLY: the recogniser always receives the
        # un-mirrored landmarks (the model was trained on un-mirrored videos; mirroring its input
        # would swap left/right hands).
        self.mirror = True

    def set_mirror(self, on):
        self.mirror = bool(on)
        if self._last[0] is not None or self._last[1] is not None:
            self.update_view(*self._last)

    def update_view(self, frame_bgr, lm):
        self._last = (frame_bgr, None if lm is None else np.array(lm, dtype=float, copy=True))
        if lm is not None and self.mirror:
            lm = np.array(lm, dtype=float, copy=True)
            lm[..., 0] = 1.0 - lm[..., 0]            # normalised x -> mirrored x
        if frame_bgr is not None:
            if self.mirror:
                frame_bgr = frame_bgr[:, ::-1]
            f = np.ascontiguousarray(frame_bgr[:, :, ::-1])
            h, w = f.shape[:2]
            img = QImage(f.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        else:
            w, h = 640, 480
            img = QImage(w, h, QImage.Format_RGB888); img.fill(QColor(theme.SCREEN))
        p = QPainter(img); p.setRenderHint(QPainter.Antialiasing)
        if lm is not None:
            draw_skeleton(p, lm, w, h, width=max(2, w // 300), radius=max(2.5, w / 250))
        p.end()
        self._pm = QPixmap.fromImage(img)
        self._rescale()

    def _rescale(self):
        if self._pm:
            self.setPixmap(self._pm.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, e):
        super().resizeEvent(e); self._rescale()


def skeleton_thumbnail(frames, size=56, mirror=False) -> QPixmap:
    """Tiny icon: the most-detected frame of a clip, scaled to fit its own bounding box."""
    pm = QPixmap(size, size); pm.fill(QColor("#10141a"))
    frames = np.asarray(frames, float)
    if frames.ndim != 3 or len(frames) == 0:
        return pm
    f = frames[int(np.argmax((~np.isnan(frames[:, :, 0])).sum(axis=1)))]
    pts = f[~np.isnan(f[:, 0])]
    if len(pts) < 3:
        return pm
    lo, hi = pts[:, :2].min(0), pts[:, :2].max(0)
    span = max(float((hi - lo).max()), 1e-6)
    g = f.copy(); g[:, :2] = (g[:, :2] - lo) / span * 0.84 + 0.08
    if mirror:
        g[:, 0] = 1.0 - g[:, 0]
    p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
    draw_skeleton(p, g, size, size, width=1, radius=1.2); p.end()
    return pm


def frame_pixmap(frame, w, h, head=True, mirror=False) -> QPixmap:
    """Render ONE (67,3) landmark frame as a skeleton icon (bounding box fitted, aspect kept).
    mirror=True flips it left<->right (display only)."""
    pm = QPixmap(w, h); pm.fill(QColor(theme.SCREEN))
    f = np.asarray(frame, np.float32)
    ok = ~np.isnan(f[:, :2]).any(axis=1)
    if ok.sum() < 3:
        return pm
    xy = f[ok, :2] * np.array([SkeletonCanvas.ASPECT, 1.0])
    lo, hi = xy.min(0), xy.max(0)
    span = np.maximum(hi - lo, 1e-4)
    s = min(w * 0.88 / span[0], h * 0.88 / span[1])
    off = np.array([w, h]) / 2 - (lo + hi) / 2 * s
    pts = np.nan_to_num(f[:, :2]) * np.array([SkeletonCanvas.ASPECT, 1.0]) * s + off
    if mirror:
        pts[:, 0] = w - pts[:, 0]
    p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
    _paint(p, pts, ok, max(1.2, h / 55), max(1.4, h / 50), head=head)
    p.end()
    return pm


# ---------------------------------------------------------------------------- friendly building blocks
def btn(text, kind="", height=48, tip="", icon_name=None, size_px=None) -> QPushButton:
    """Rounded button. kind: '' | primary | good | warn | danger | chip | seg | link."""
    b = QPushButton(text)
    if kind:
        b.setProperty("kind", kind)
    b.setMinimumHeight(height); b.setCursor(Qt.PointingHandCursor)
    if tip:
        b.setToolTip(tip)
    if icon_name:
        from PySide6.QtGui import QIcon
        b.setIcon(QIcon(theme.icon(icon_name, 28, "#ffffff" if kind in ("primary", "good", "warn") else theme.TEXT)))
        b.setIconSize(QSize(24, 24))
    if size_px:
        b.setStyleSheet(f"font-size:{size_px}px;")
    return b


def set_kind(b, kind):
    b.setProperty("kind", kind); b.style().unpolish(b); b.style().polish(b)


def card(name="card") -> QFrame:
    f = QFrame(); f.setObjectName(name)
    return f


def heading(text, px=22, color=theme.TEXT) -> QLabel:
    return plain_label(text, wrap=True, style=f"font-size:{px}px;font-weight:bold;color:{color};")


def hint(text, px=14) -> QLabel:
    return plain_label(text, wrap=True, style=f"font-size:{px}px;color:{theme.MUTED};")


class Switch(QAbstractButton):
    """On/off switch (checkable). Use with a text label next to it."""

    def __init__(self, checked=False, parent=None):
        super().__init__(parent)
        self.setCheckable(True); self.setChecked(checked); self.setCursor(Qt.PointingHandCursor)
        self.setFixedSize(56, 32); self.setFocusPolicy(Qt.StrongFocus)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        on = self.isChecked()
        p.setPen(Qt.NoPen)
        col = QColor(theme.GOOD if on else "#cbd5e1")
        if not self.isEnabled():
            col.setAlpha(110)
        p.setBrush(col); p.drawRoundedRect(QRectF(0, 2, 56, 28), 14, 14)
        p.setBrush(QColor("#ffffff")); p.drawEllipse(QPointF(42 if on else 14, 16), 11, 11)
        if self.hasFocus():
            p.setBrush(Qt.NoBrush); p.setPen(QPen(QColor(theme.PRIMARY), 2)); p.drawRoundedRect(QRectF(1, 2, 54, 28), 14, 14)
        p.end()


class TickBox(QCheckBox):
    """Check box that draws its own box + tick (the stock indicator has no tick when styled)."""

    def sizeHint(self):
        return QSize(self.fontMetrics().horizontalAdvance(self.text()) + 40, 30)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        box = QRectF(2, (self.height() - 22) / 2, 22, 22)
        on = self.isChecked()
        p.setPen(QPen(QColor(theme.PRIMARY_DARK if on else "#94a3b8"), 2))
        p.setBrush(QColor(theme.PRIMARY if on else "#ffffff")); p.drawRoundedRect(box, 6, 6)
        if on:
            p.setPen(QPen(QColor("#ffffff"), 2.6, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)); p.setBrush(Qt.NoBrush)
            p.drawPolyline([QPointF(box.x() + 5, box.y() + 11.5), QPointF(box.x() + 9.5, box.y() + 16),
                            QPointF(box.x() + 17, box.y() + 6.5)])
        if self.hasFocus():
            p.setBrush(Qt.NoBrush); p.setPen(QPen(QColor(theme.PRIMARY), 2)); p.drawRoundedRect(box.adjusted(-2, -2, 2, 2), 8, 8)
        p.setPen(QColor(theme.TEXT if self.isEnabled() else "#94a3b8")); p.setFont(self.font())
        p.drawText(QRectF(34, 0, self.width() - 34, self.height()), Qt.AlignVCenter | Qt.AlignLeft, self.text())
        p.end()


class Collapsible(QWidget):
    """Section header with an arrow; the body is hidden until opened."""

    def __init__(self, title, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(6)
        self.toggle = QToolButton(); self.toggle.setText("  " + title); self.toggle.setCheckable(True)
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon); self.toggle.setArrowType(Qt.RightArrow)
        self.toggle.setCursor(Qt.PointingHandCursor)
        self.toggle.setStyleSheet(f"QToolButton{{border:none;font-size:15px;font-weight:bold;color:{theme.MUTED};padding:6px;}}")
        self.body = QWidget(); self.body.setVisible(False)
        self.body_layout = QVBoxLayout(self.body); self.body_layout.setContentsMargins(8, 0, 0, 0)
        self.toggle.toggled.connect(self.set_open)
        v.addWidget(self.toggle, 0, Qt.AlignLeft); v.addWidget(self.body)

    def set_title(self, title):
        self.toggle.setText("  " + title)

    def set_open(self, on):
        self.toggle.blockSignals(True); self.toggle.setChecked(bool(on)); self.toggle.blockSignals(False)
        self.toggle.setArrowType(Qt.DownArrow if on else Qt.RightArrow)
        self.body.setVisible(bool(on))

    @property
    def is_open(self):
        return self.body.isVisibleTo(self)


class NavButton(QAbstractButton):
    """Big sidebar button: drawn icon + plain word (+ optional count badge)."""

    def __init__(self, icon_name, text, parent=None):
        super().__init__(parent)
        self.icon_name, self.badge = icon_name, 0
        self.setText(text); self.setCheckable(True); self.setAutoExclusive(True)
        self.setMinimumHeight(60); self.setCursor(Qt.PointingHandCursor); self.setFocusPolicy(Qt.TabFocus)

    def set_badge(self, n):
        self.badge = int(n); self.update()

    def sizeHint(self):
        return QSize(170, 60)

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = QRectF(self.rect()).adjusted(8, 3, -8, -3)
        on = self.isChecked()
        if on:
            p.setPen(Qt.NoPen); p.setBrush(QColor(theme.PRIMARY)); p.drawRoundedRect(r, 14, 14)
        elif self.underMouse():
            p.setPen(Qt.NoPen); p.setBrush(QColor("#334155")); p.drawRoundedRect(r, 14, 14)
        if self.hasFocus():
            p.setBrush(Qt.NoBrush); p.setPen(QPen(QColor("#93c5fd"), 2)); p.drawRoundedRect(r, 14, 14)
        p.drawPixmap(int(r.x() + 12), int(r.center().y() - 16), theme.icon(self.icon_name, 32, "#ffffff"))
        p.setPen(QColor("#ffffff" if on else "#e2e8f0")); p.setFont(theme.make_font(17, on))
        p.drawText(QRectF(r.x() + 54, r.y(), r.width() - 60, r.height()), Qt.AlignVCenter | Qt.AlignLeft, self.text())
        if self.badge:
            bx = QRectF(r.x() + 32, r.center().y() - 22, 22, 22)       # small bubble on the icon's corner
            p.setPen(QPen(QColor(theme.NAVY if not on else theme.PRIMARY), 2)); p.setBrush(QColor(theme.WARN)); p.drawEllipse(bx)
            p.setPen(QColor("#ffffff")); p.setFont(theme.make_font(13, True))
            p.drawText(bx, Qt.AlignCenter, str(min(self.badge, 99)))
        p.end()


class Dots(QWidget):
    """'How sure am I' as three dots (0..3 filled) - no percentages."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.level = 0; self.color = theme.GOOD
        self.setFixedSize(66, 20)
        self.setAccessibleName("How sure I am")

    def set_level(self, n, color=None):
        self.level = max(0, min(3, int(n)))
        if color:
            self.color = color
        self.update()

    def paintEvent(self, e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing); p.setPen(Qt.NoPen)
        for i in range(3):
            p.setBrush(QColor(self.color if i < self.level else "#cbd5e1"))
            p.drawEllipse(QPointF(10 + i * 23, 10), 8, 8)
        p.end()


def mirror_button(parent=None) -> QPushButton:
    """Checkable 'Mirror: On / Off' button with an icon, meant to float on top of the camera.
    Call .refresh() after a programmatic setChecked() made with signals blocked."""
    from PySide6.QtGui import QIcon
    b = QPushButton("Mirror: On"); b.setCheckable(True)
    b.setAccessibleName("Mirror view"); b.setCursor(Qt.PointingHandCursor)
    b.setIcon(QIcon(theme.icon("mirror", 28, "#ffffff"))); b.setIconSize(QSize(22, 22))
    b.setMinimumHeight(42)
    # ON = solid green (a state, not a call to action); OFF = outlined and see-through
    b.setStyleSheet("QPushButton{background:rgba(15,23,42,60);color:#ffffff;border:2px solid rgba(255,255,255,170);"
                    "border-radius:21px;padding:4px 16px;font-size:15px;font-weight:bold;}"
                    "QPushButton:hover{background:rgba(30,41,59,200);}"
                    f"QPushButton:checked{{background:{theme.GOOD};border-color:#bbf7d0;}}")

    def refresh(*_):
        on = b.isChecked()
        b.setText("Mirror: On" if on else "Mirror: Off")
        b.setToolTip("Mirror is " + ("on" if on else "off") + ". It flips the camera, the sign pictures and the "
                     "animations, so you can copy them like in a mirror. I always read your real hands.")
    b.refresh = refresh
    b.toggled.connect(refresh); refresh()
    return b


class ClarifyPanel(QFrame):
    """'Which one did you mean?' Numbered buttons: 1) Skip  2) Retry  then up to 8 similar words, 3) ... 10) ...
    The signer answers by TAPPING a button, pressing its number (0 = 10), or HOLDING UP that many fingers
    (1-5 on one hand, 6-10 = five on one hand plus the rest on the other); `set_finger` shows the number being held."""
    confirmed = Signal(str)
    repeat = Signal()
    skip = Signal()
    MAX_WORDS = 8
    COLS = 3
    TITLE_FINGERS = "Not sure - tap a button, press its number, or hold up that many fingers (hold still)"
    TITLE_KEYS = "Not sure - tap a button or press its number"
    TITLE = TITLE_FINGERS
    SIZE = "font-size:22px;"
    PICK = "font-size:22px;border:5px solid #f59e0b;"

    def __init__(self):
        super().__init__()
        self.setObjectName("warncard")
        self.cands = []
        lay = QVBoxLayout(self); lay.setContentsMargins(16, 12, 16, 12); lay.setSpacing(8)
        self.title = plain_label(self.TITLE, wrap=True, style="font-size:16px;font-weight:bold;color:#92400e;")
        top = QHBoxLayout(); top.setSpacing(10)
        self.b_skip = btn("1) Skip", "", 60, tip="Cancel this one and carry on with the next sign  [1 finger or key 1 / S]")
        self.b_retry = btn("2) Retry", "", 60, tip="Sign it again  [2 fingers or key 2 / R]")
        for b in (self.b_skip, self.b_retry):
            b.setStyleSheet(self.SIZE); b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed); top.addWidget(b)
        self.b_skip.clicked.connect(self._skip); self.b_retry.clicked.connect(self._repeat)
        self.grid = QGridLayout(); self.grid.setSpacing(8)
        self.word_btns = []
        for i in range(self.MAX_WORDS):
            b = btn("", "primary", 60); b.setStyleSheet(self.SIZE); b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(lambda _=False, k=i: self._pick(k)); b.hide()
            self.word_btns.append(b); self.grid.addWidget(b, i // self.COLS, i % self.COLS)
        self.more = hint("For 6 to 10 fingers: show 5 on one hand and the rest on the other hand.", 13)
        self.carry = hint("Or just carry on signing: I skip this one after a moment.", 13)
        lay.addWidget(self.title); lay.addLayout(top); lay.addLayout(self.grid); lay.addWidget(self.more); lay.addWidget(self.carry)
        self.hide()

    fingers_on = True

    def set_fingers_enabled(self, on):
        """Fingers may answer (True) or only taps / keys (False): the wording follows."""
        self.fingers_on = bool(on)
        self.TITLE = self.TITLE_FINGERS if on else self.TITLE_KEYS
        self.title.setText(self.TITLE)
        self.more.setVisible(bool(on) and self.n_options > 5 and self.isVisible())
        self.carry.setVisible(bool(on) and self.isVisible())

    @property
    def n_options(self):
        return len(self.cands) + 2

    def show_candidates(self, cands):
        self.cands = [(str(l), float(p)) for l, p in cands][:self.MAX_WORDS]
        for i, b in enumerate(self.word_btns):
            b.setVisible(i < len(self.cands))
            if i < len(self.cands):
                n = i + 3
                b.setText(f"{n}) {self.cands[i][0]}")
                b.setToolTip(f"Hold up {n} finger{'s' if n != 1 else ''}, or press {n % 10}")
                auto_direction(b, self.cands[i][0])
        self.more.setVisible(self.n_options > 5 and self.fingers_on)
        self.carry.setVisible(self.fingers_on)
        self.set_finger(0, 0.0)
        self.show()

    def _button(self, n):
        """Button of option number n (1 = skip, 2 = retry, 3.. = word n-3)."""
        return self.b_skip if n == 1 else self.b_retry if n == 2 else (
            self.word_btns[n - 3] if 0 <= n - 3 < len(self.cands) else None)

    def set_finger(self, n, progress=0.0, seen=None, hand=False, options=0):
        """Highlight the button whose number the signer is holding up right now (n = 0: nothing) and say in words
        what is seen: the number read, or a hint when a hand is visible but no clear number can be read."""
        for k in range(1, self.n_options + 1):
            b = self._button(k)
            if b is not None:
                b.setStyleSheet(self.PICK if n == k else self.SIZE)
        top = options or self.n_options
        if n > 0:
            text = f"Holding up {n} - keep your hand steady..."
        elif seen:
            text = (f"I see {seen} finger{'s' if seen != 1 else ''} - hold up 1 to {top}" if seen > top
                    else f"I see {seen} finger{'s' if seen != 1 else ''} - keep steady")
        elif hand:
            text = "I can see your hand but not a clear number - palm toward the camera, fingers apart"
        else:
            text = self.TITLE
        self.title.setText(text)

    def press(self, n):
        """Key / finger number: 1 = skip, 2 = retry, 3.. = the suggested words."""
        if not self.isVisible():
            return
        if n == 1:
            self._skip()
        elif n == 2:
            self._repeat()
        elif 0 <= n - 3 < len(self.cands):
            self._pick(n - 3)

    def _pick(self, i):
        if self.isVisible() and i < len(self.cands):
            self.confirmed.emit(self.cands[i][0]); self.hide()

    def _skip(self):
        if self.isVisible():
            self.skip.emit(); self.hide()

    def _repeat(self):
        if self.isVisible():
            self.repeat.emit(); self.hide()
