"""Colours, the application stylesheet and small drawn icons (no image files needed)."""
from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen, QPixmap, QPolygonF

BG = "#f3f5fa"
CARD = "#ffffff"
LINE = "#e2e8f0"
TEXT = "#1e293b"
MUTED = "#64748b"
PRIMARY = "#2563eb"
PRIMARY_DARK = "#1d4ed8"
GOOD = "#16a34a"
GOOD_BG = "#dcfce7"
WARN = "#d97706"
WARN_BG = "#fef3c7"
BAD = "#dc2626"
BAD_BG = "#fee2e2"
INFO_BG = "#dbeafe"
NAVY = "#1e293b"
SCREEN = "#0f172a"

BASE_PX = 15

STYLESHEET = f"""
QMainWindow, QWidget#page {{ background:{BG}; }}
QWidget {{ color:{TEXT}; font-size:{BASE_PX}px; }}
QLabel {{ background:transparent; }}
QToolTip {{ background:{NAVY}; color:#ffffff; border:none; padding:6px; font-size:14px; }}
QFrame#card {{ background:{CARD}; border:1px solid {LINE}; border-radius:16px; }}
QFrame#softcard {{ background:{INFO_BG}; border:1px solid #bfdbfe; border-radius:16px; }}
QFrame#warncard {{ background:{WARN_BG}; border:1px solid #fcd34d; border-radius:16px; }}
QFrame#sidebar {{ background:{NAVY}; }}
QFrame#sidebar QLabel {{ color:#cbd5e1; }}
QScrollArea {{ border:none; background:transparent; }}
QScrollArea > QWidget > QWidget {{ background:transparent; }}
QScrollBar:vertical {{ background:transparent; width:12px; margin:2px; }}
QScrollBar::handle:vertical {{ background:#cbd5e1; border-radius:5px; min-height:30px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height:0; }}
QPushButton {{ background:#ffffff; border:1px solid #cbd5e1; border-radius:12px; padding:8px 16px;
  font-size:{BASE_PX}px; color:{TEXT}; }}
QPushButton:hover {{ background:#f1f5f9; border-color:#94a3b8; }}
QPushButton:pressed {{ background:#e2e8f0; }}
QPushButton:disabled {{ background:#f1f5f9; color:#94a3b8; border-color:#e2e8f0; }}
QPushButton:focus {{ border:2px solid {PRIMARY}; }}
QPushButton[kind="primary"] {{ background:{PRIMARY}; border:1px solid {PRIMARY_DARK}; color:#ffffff; font-weight:bold; }}
QPushButton[kind="primary"]:hover {{ background:{PRIMARY_DARK}; }}
QPushButton[kind="primary"]:disabled {{ background:#cbd5e1; border-color:#cbd5e1; color:#f8fafc; }}
QPushButton[kind="good"] {{ background:{GOOD}; border:1px solid #15803d; color:#ffffff; font-weight:bold; }}
QPushButton[kind="good"]:hover {{ background:#15803d; }}
QPushButton[kind="warn"] {{ background:{WARN}; border:1px solid #b45309; color:#ffffff; font-weight:bold; }}
QPushButton[kind="warn"]:hover {{ background:#b45309; }}
QPushButton[kind="danger"] {{ background:#ffffff; border:1px solid #fca5a5; color:{BAD}; }}
QPushButton[kind="danger"]:hover {{ background:{BAD_BG}; }}
QPushButton[kind="chip"] {{ border-radius:18px; padding:6px 10px; background:#ffffff; }}
QPushButton[kind="chip"]:checked {{ background:{NAVY}; color:#ffffff; border-color:{NAVY}; }}
QPushButton[kind="seg"] {{ border-radius:10px; padding:10px 14px; }}
QPushButton[kind="seg"]:checked {{ background:{PRIMARY}; color:#ffffff; border-color:{PRIMARY_DARK}; }}
QPushButton[kind="link"] {{ background:transparent; border:none; color:{PRIMARY}; text-decoration:underline; padding:4px; }}
QLineEdit, QComboBox, QPlainTextEdit {{ background:#ffffff; border:1px solid #cbd5e1; border-radius:10px;
  padding:6px 10px; selection-background-color:{PRIMARY}; selection-color:#ffffff; font-size:{BASE_PX}px; }}
QLineEdit:focus, QComboBox:focus, QPlainTextEdit:focus {{ border:2px solid {PRIMARY}; }}
QLineEdit:disabled, QComboBox:disabled, QPlainTextEdit:disabled {{ background:#f1f5f9; color:#94a3b8; }}
QComboBox QAbstractItemView {{ background:#ffffff; selection-background-color:{PRIMARY}; selection-color:#ffffff; }}
QListWidget {{ background:transparent; border:none; outline:none; }}
QCheckBox {{ spacing:10px; }}
QCheckBox::indicator {{ width:22px; height:22px; border:2px solid #94a3b8; border-radius:6px; background:#ffffff; }}
QCheckBox::indicator:checked {{ background:{PRIMARY}; border-color:{PRIMARY_DARK}; }}
QProgressBar {{ border:none; border-radius:4px; background:#e2e8f0; }}
QProgressBar::chunk {{ background:{PRIMARY}; border-radius:4px; }}
QSlider::groove:horizontal {{ height:8px; background:#e2e8f0; border-radius:4px; }}
QSlider::handle:horizontal {{ width:24px; height:24px; margin:-9px 0; border-radius:12px; background:{PRIMARY}; }}
QSlider::sub-page:horizontal {{ background:#93c5fd; border-radius:4px; }}
QSplitter::handle {{ background:transparent; }}
"""


def make_font(px, bold=False):
    f = QFont(); f.setPixelSize(px); f.setBold(bold)
    return f


def icon(name, size=40, color="#ffffff", bg=None) -> QPixmap:
    """Simple line icons drawn with QPainter. name in: talk signs teach settings help mirror camera
    toggle send check hand."""
    pm = QPixmap(size, size); pm.fill(Qt.transparent)
    p = QPainter(pm); p.setRenderHint(QPainter.Antialiasing)
    s = float(size)
    if bg:
        p.setPen(Qt.NoPen); p.setBrush(QColor(bg)); p.drawRoundedRect(QRectF(0, 0, s, s), s * .28, s * .28)
    col = QColor(color)
    pen = QPen(col, max(2.0, s / 15), Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    p.setPen(pen); p.setBrush(Qt.NoBrush)
    u = s / 10.0                                    # 10x10 grid
    R = lambda x, y, w, h: QRectF(x * u, y * u, w * u, h * u)
    P = lambda x, y: QPointF(x * u, y * u)
    if name == "talk":
        path = QPainterPath(); path.addRoundedRect(R(1.5, 2, 7, 5), u, u)
        p.drawPath(path)
        p.drawPolyline([P(3.5, 7), P(3, 9), P(5.5, 7)])
        p.setBrush(col)
        for x in (3.4, 5, 6.6):
            p.drawEllipse(P(x, 4.5), u * .35, u * .35)
    elif name == "signs":
        for x, y in ((1.5, 1.5), (5.4, 1.5), (1.5, 5.4), (5.4, 5.4)):
            p.drawRoundedRect(R(x, y, 3.1, 3.1), u * .6, u * .6)
    elif name == "teach":
        p.drawEllipse(R(1.2, 1.2, 7.6, 7.6)); p.drawLine(P(5, 3), P(5, 7)); p.drawLine(P(3, 5), P(7, 5))
    elif name == "settings":
        for y, kx in ((2.5, 3), (5, 7), (7.5, 4.5)):
            p.drawLine(P(1.5, y), P(8.5, y)); p.setBrush(QColor(bg or "#1e293b")); p.drawEllipse(P(kx, y), u * .9, u * .9)
            p.setBrush(Qt.NoBrush)
    elif name == "help":
        p.drawEllipse(R(1, 1, 8, 8)); p.setFont(make_font(int(s * .55), True)); p.drawText(R(1, 1, 8, 8), Qt.AlignCenter, "?")
    elif name == "mirror":
        p.drawLine(P(5, 1.5), P(5, 8.5))
        p.setBrush(col)
        p.drawPolygon(QPolygonF([P(4, 3), P(4, 7), P(1.2, 5)]))
        p.setBrush(Qt.NoBrush)
        p.drawPolygon(QPolygonF([P(6, 3), P(6, 7), P(8.8, 5)]))
    elif name == "camera":
        p.drawRoundedRect(R(1, 2.8, 8, 5.4), u * .8, u * .8); p.drawEllipse(P(5, 5.5), u * 1.4, u * 1.4)
    elif name == "toggle":
        p.drawRoundedRect(R(1, 3, 8, 4), u * 2, u * 2); p.setBrush(col); p.drawEllipse(P(7, 5), u * 1.2, u * 1.2)
    elif name == "send":
        p.drawPolygon(QPolygonF([P(1, 5), P(9, 1.5), P(6.5, 8.5), P(5, 5.8)])); p.drawLine(P(5, 5.8), P(9, 1.5))
    elif name == "check":
        p.drawPolyline([P(2, 5.3), P(4.2, 7.5), P(8.2, 2.8)])
    elif name == "hand":
        p.drawRoundedRect(R(2.6, 4.6, 4.8, 4), u * 1.4, u * 1.4)
        for x, top in ((2.9, 2.6), (4.1, 1.5), (5.3, 1.5), (6.5, 2.6)):
            p.drawLine(P(x, 4.8), P(x, top))
    p.end()
    return pm
