"""'My signs' screen: searchable cards with a skeleton icon + word, filter chips, and a simple edit panel.

Everything user-provided (labels, notes, aliases, categories) is shown as plain text only
(item text roles, QLabel PlainText, QPlainTextEdit). Thumbnails are rendered lazily in small time
slices and cached per (sign id, mirror state), so flipping the mirror never shows a stale icon.
MIRROR is display-only: the library data and what the recogniser gets are never flipped.
"""
import html
import re
import time
from pathlib import Path

from PySide6.QtCore import QEvent, QRect, QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPainter, QPainterPath, QPen, QPixmap
from PySide6.QtWidgets import (QAbstractItemView, QApplication, QButtonGroup, QComboBox, QDialog, QFileDialog, QHBoxLayout,
                               QFrame, QBoxLayout, QLabel, QLineEdit, QListView, QListWidget, QListWidgetItem, QMessageBox, QPlainTextEdit,
                               QPushButton, QScrollArea, QSplitter, QSplitterHandle, QStyle, QStyledItemDelegate, QVBoxLayout, QWidget)

from ..library import LibraryError, norm_text
from . import theme
from .recordings_panel import RecordingsPanel
from .widgets import (Collapsible, Switch, TickBox, auto_direction, btn, card, frame_pixmap, heading, hint, plain_label)

ROOT = Path(__file__).resolve().parents[2]
CELL = QSize(142, 138)
THUMB = QSize(124, 90)
ROLE_ID, ROLE_FLAGS = Qt.UserRole, Qt.UserRole + 1
GOAL_EXAMPLES = 3
MUTED = f"color:{theme.MUTED};font-size:14px;"
BULK_CONFIRM = 100                    # switching on more than this many signs at once asks first
UNDO_MS = 30000
SPLIT_KEY = "my_signs_split"          # saved setting: share of the height given to the signs list
SPLIT_DEFAULT = 0.6
HANDLE_H = 22
ALL_CATS = "All categories"
CHIPS = (("mine", "Recorded by me"), ("inuse", "Switched on"), ("all", "All dictionary"))


def my_examples(e):
    """How many examples were recorded by the user (dictionary signs keep one dictionary example)."""
    n = e.takes if e.source == "recorded" else max(e.takes - 1, 0)
    return max(n - getattr(e, "auto", 0), 0)          # examples learned automatically are not 'recorded by me'


def is_mine(e):
    """'Recorded by me': a sign I recorded from scratch, or a dictionary sign I added my own example to."""
    return e.source == "recorded" or my_examples(e) > 0


def recordings_count(e):
    """Examples that are not the built-in dictionary clip (recorded by me or learned from a correction)."""
    return e.takes if e.source == "recorded" else max(e.takes - 1, 0)


def examples_text(e):
    n = e.takes
    txt = f"{n} example{'s' if n != 1 else ''}"
    return txt + (" - add more to improve recognition" if my_examples(e) < GOAL_EXAMPLES else "")


class SignDelegate(QStyledItemDelegate):
    """Card = rounded skeleton icon with small badges, the word (elided) underneath."""

    def sizeHint(self, option, index):
        return CELL

    def paint(self, p, option, index):
        flags = index.data(ROLE_FLAGS) or {}
        p.save(); p.setRenderHint(QPainter.Antialiasing)
        r = option.rect.adjusted(4, 4, -4, -4)
        sel = bool(option.state & QStyle.State_Selected)
        if flags.get("hidden"):
            p.setOpacity(0.5)
        p.setBrush(QColor("#eff6ff") if sel else QColor("#ffffff"))
        p.setPen(QPen(QColor(theme.PRIMARY) if sel else QColor(theme.LINE), 3 if sel else 1))
        p.drawRoundedRect(r, 14, 14)
        tr = QRect(r.x() + (r.width() - THUMB.width()) // 2, r.y() + 6, THUMB.width(), THUMB.height())
        pm = index.data(Qt.DecorationRole)
        path = QPainterPath(); path.addRoundedRect(QRectF(tr), 10, 10)
        p.save(); p.setClipPath(path)
        if isinstance(pm, QPixmap):
            p.drawPixmap(tr, pm)
        else:
            p.fillRect(tr, QColor(theme.SCREEN))
        p.restore()
        y, f = tr.y() + 4, QFont(option.font); f.setPixelSize(12); f.setBold(True)
        p.setFont(f); fm = QFontMetrics(f)
        n_mine = int(flags.get("mine") or 0)               # how many recordings of mine this sign has
        for key, text, col in (("active", "On", theme.GOOD), ("mine", f"Mine {n_mine}" if n_mine > 1 else "Mine", "#7c3aed")):
            if flags.get(key):
                w = fm.horizontalAdvance(text) + 12
                p.setPen(Qt.NoPen); p.setBrush(QColor(col)); p.drawRoundedRect(QRect(tr.x() + 4, y, w, 17), 8, 8)
                p.setPen(QColor("#ffffff")); p.drawText(QRect(tr.x() + 4, y, w, 17), Qt.AlignCenter, text)
                y += 20
        f2 = QFont(option.font); f2.setPixelSize(16); f2.setBold(True); p.setFont(f2)
        fm2 = QFontMetrics(f2)
        txt = fm2.elidedText(str(index.data(Qt.DisplayRole) or ""), Qt.ElideRight, r.width() - 10)
        p.setPen(QColor(theme.TEXT))
        p.drawText(QRect(r.x() + 5, tr.bottom() + 5, r.width() - 10, 26), Qt.AlignHCenter | Qt.AlignVCenter | Qt.TextSingleLine, txt)
        p.restore()


class GripHandle(QSplitterHandle):
    """Drag bar between the signs list and the edit panel: a centred grip of three dots."""

    def __init__(self, orientation, parent):
        super().__init__(orientation, parent)
        self.hover = False
        self.setCursor(Qt.SplitVCursor); self.setMouseTracking(True)
        self.setToolTip("Drag to make the signs list taller or shorter\n(double-click to put it back)")
        self.setAccessibleName("Drag to make the signs list taller or shorter")

    def enterEvent(self, e):
        self.hover = True; self.update(); super().enterEvent(e)

    def leaveEvent(self, e):
        self.hover = False; self.update(); super().leaveEvent(e)

    def mouseReleaseEvent(self, e):
        super().mouseReleaseEvent(e)
        self.splitter().drag_finished()

    def mouseDoubleClickEvent(self, e):
        self.splitter().reset_ratio()

    def paintEvent(self, _e):
        p = QPainter(self); p.setRenderHint(QPainter.Antialiasing)
        r = self.rect().adjusted(0, 3, 0, -3)
        p.setPen(QPen(QColor("#c3ccdb"), 1))
        down = bool(self.hover or (QApplication.mouseButtons() & Qt.LeftButton and self.underMouse()))
        p.setBrush(QColor("#bfdbfe") if down else QColor("#dde3ee"))
        p.drawRoundedRect(r, 7, 7)
        p.setPen(Qt.NoPen); p.setBrush(QColor(theme.PRIMARY) if down else QColor("#64748b"))
        cx, cy = r.center().x(), r.center().y()
        for dx in (-14, 0, 14):
            p.drawEllipse(QRectF(cx + dx - 3, cy - 3, 6, 6))


class GripSplitter(QSplitter):
    """Vertical splitter with a visible grip. Remembers a ratio (top share) the owner persists."""
    drag_done = Signal()
    reset_done = Signal()

    def __init__(self, parent=None):
        super().__init__(Qt.Vertical, parent)
        self.setChildrenCollapsible(False); self.setHandleWidth(HANDLE_H)
        self.ratio = SPLIT_DEFAULT
        self._busy = False
        self.installEventFilter(self)

    def createHandle(self):
        return GripHandle(self.orientation(), self)

    def drag_finished(self):
        t, b = self.sizes()[:2]
        if t + b > 0:
            self.ratio = t / (t + b)
        self.drag_done.emit()

    def reset_ratio(self):
        self.ratio = SPLIT_DEFAULT; self.apply_ratio(); self.reset_done.emit()

    def apply_ratio(self):
        if self._busy or self.count() < 2:
            return
        total = sum(self.sizes())
        if total <= 0:
            return
        lo = self.widget(0).minimumHeight()
        hi = total - self.widget(1).minimumHeight()
        top = max(lo, min(int(round(total * self.ratio)), hi))
        if abs(top - self.sizes()[0]) > 1:
            self._busy = True
            try:
                self.setSizes([top, total - top])
            finally:
                self._busy = False

    def eventFilter(self, o, ev):
        if o is self and ev.type() == QEvent.Resize:
            QTimer.singleShot(0, self.apply_ratio)
        return False

    def showEvent(self, ev):
        super().showEvent(ev); QTimer.singleShot(0, self.apply_ratio)


class PageScroll(QScrollArea):
    """Gives its page the viewport size, but never less than the page's real minimum height (then it scrolls).
    (QScrollArea's own 'resizable' mode sizes wrapped-text pages by their preferred height instead.)"""

    def __init__(self):
        super().__init__()
        self.setFrameShape(QFrame.NoFrame); self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

    def setWidget(self, w):
        super().setWidget(w); w.installEventFilter(self); self.fit()

    def fit(self):
        w = self.widget()
        if w is None:
            return
        vp = self.viewport().size()
        h = max(vp.height(), w.minimumSizeHint().height())
        w.resize(vp.width(), h)

    def resizeEvent(self, e):
        super().resizeEvent(e); self.fit()

    def eventFilter(self, o, ev):
        if o is self.widget() and ev.type() in (QEvent.LayoutRequest, QEvent.Show):
            QTimer.singleShot(0, self.fit)
        return False


class AdaptiveRow(QWidget):
    """Text on the left, buttons on the right; when the column is narrow the buttons drop below the text."""

    def __init__(self, label, buttons, parent=None):
        super().__init__(parent)
        self.label, self.buttons = label, buttons
        self.box = QBoxLayout(QBoxLayout.LeftToRight, self); self.box.setContentsMargins(0, 0, 0, 0); self.box.setSpacing(8)
        self.box.addWidget(label, 1)
        self.btns = QWidget(); bl = QHBoxLayout(self.btns); bl.setContentsMargins(0, 0, 0, 0); bl.setSpacing(8)
        for b in buttons:
            b.setStyleSheet("padding:2px 10px;"); bl.addWidget(b)
        self.box.addWidget(self.btns)

    def resizeEvent(self, e):
        need = self.label.sizeHint().width() + self.btns.sizeHint().width() + 16
        d = QBoxLayout.LeftToRight if self.width() >= need else QBoxLayout.TopToBottom
        if self.box.direction() != d:
            self.box.setDirection(d)
        super().resizeEvent(e)


class LibraryTab(QWidget):
    play_entry = Signal(str)             # sign id -> main view plays its animation
    play_phrase = Signal(list, list)     # (sign ids, words not found)
    record_requested = Signal(str, str)  # ('replace' | 'take', sign id)
    changed = Signal()                   # an edit / switch was saved (templates + label_map must be refreshed)
    teach_first = Signal()               # 'teach your first sign' call to action
    message = Signal(str)
    form_filled = Signal()               # the detail form was refreshed (the main window adds the learned-examples list)
    watch_take = Signal(str, int)        # (sign id, example number): play that example of mine in the main view

    def __init__(self, library, editable=True, model_loaded=True, parent=None):
        super().__init__(parent)
        self.lib, self.editable, self.model_loaded = library, editable, model_loaded
        self.cur = None                  # id of the sign shown in the edit panel
        self.mirror = True               # display only
        self._cache = {}                 # (id, mirror) -> QPixmap
        self._queue = []                 # ids waiting for a thumbnail
        self._phrase = ([], [])
        self.confirm = self._ask         # tests replace these hooks (no modal dialogs)
        self.choose_dir = self._pick_dir
        self.confirm_many = self._ask_many
        self.undo_ms = UNDO_MS
        self._undo = None                # {ids: previous states} of the last bulk switch-off
        self._settings_io = None         # (get, set) installed by bind_settings; default: the window / the store
        self._mem_settings = {}
        self.category = None             # category chosen in the filter (None = all)
        self._debounce = QTimer(self); self._debounce.setSingleShot(True); self._debounce.setInterval(160)
        self._debounce.timeout.connect(self.refresh)
        self._vis = QTimer(self); self._vis.setSingleShot(True); self._vis.setInterval(30)
        self._vis.timeout.connect(self._collect_visible)
        self._fill = QTimer(self); self._fill.setInterval(8); self._fill.timeout.connect(self._fill_some)
        self._undo_timer = QTimer(self); self._undo_timer.setSingleShot(True)
        self._undo_timer.timeout.connect(self._hide_undo)
        self._build()
        self.split.ratio = self._load_ratio()
        self._pick_default_chip()
        self.refresh()

    # ------------------------------------------------------------------ library helpers (guarded)
    def _can_switch(self):
        return hasattr(self.lib, "set_active") and hasattr(self.lib, "is_active")

    def is_active(self, e):
        if hasattr(self.lib, "is_active"):
            try:
                return bool(self.lib.is_active(e.id))
            except Exception:
                return False
        return e.source == "recorded" or e.takes > 1

    def in_use_count(self):
        return sum(1 for e in self.lib.entries() if self.is_active(e))

    # ------------------------------------------------------------------ construction
    def _build(self):
        outer = QVBoxLayout(self); outer.setContentsMargins(0, 0, 0, 0)
        # the whole screen scrolls when the window is very small, so nothing is ever cut off
        self.page_scroll = PageScroll()
        body = QWidget(); body.setObjectName("page"); self.body = body
        self.page_scroll.setWidget(body); outer.addWidget(self.page_scroll)
        v = QVBoxLayout(body); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(8)
        v.addWidget(heading("My signs", 24))
        v.addWidget(hint("Tap a sign to watch it, then switch it on to use it."))
        self.search = QLineEdit(); self.search.setClearButtonEnabled(True); self.search.setMinimumHeight(40)
        self.search.setPlaceholderText("Search for a word, in English or Arabic (for example: mother, أم)")
        self.search.setStyleSheet("font-size:17px;")
        self.search.textChanged.connect(self._on_search_text)
        self.search.returnPressed.connect(self.on_enter)
        v.addWidget(self.search)

        fl = QHBoxLayout(); fl.setSpacing(8)
        self.chip_group = QButtonGroup(self); self.chip_group.setExclusive(True); self.chips = {}
        for key, text in CHIPS:
            b = btn(text, "chip", 34); b.setCheckable(True); b.setMinimumWidth(b.fontMetrics().horizontalAdvance(text) + 40)
            self.chip_group.addButton(b); self.chips[key] = b; fl.addWidget(b)
            b.toggled.connect(self._filters_changed)
        fl.addStretch(1)
        v.addLayout(fl)

        cr = QHBoxLayout(); cr.setSpacing(8)
        cr.addWidget(plain_label("Category", style=f"{MUTED}font-weight:bold;"))
        self.cat_combo = QComboBox(); self.cat_combo.setMinimumHeight(36); self.cat_combo.setMinimumWidth(180)
        self.cat_combo.setToolTip("Show only the signs of one category")
        self.cat_combo.currentIndexChanged.connect(self._category_changed)
        cr.addWidget(self.cat_combo, 1)
        self.count = plain_label("", style=MUTED); cr.addWidget(self.count)
        v.addLayout(cr)
        self._fill_categories()
        v.addWidget(self._build_bulk())

        pr = QHBoxLayout()
        self.phrase_btn = btn("Show the whole phrase", "primary", 42)
        self.phrase_btn.clicked.connect(self.play_current_phrase); self.phrase_btn.hide()
        self.missing = plain_label("", wrap=True, style=f"color:{theme.WARN};font-size:14px;"); self.missing.hide()
        pr.addWidget(self.phrase_btn); pr.addWidget(self.missing, 1)
        v.addLayout(pr)

        split = GripSplitter(); self.split = split
        top = QWidget(); tl = QVBoxLayout(top); tl.setContentsMargins(0, 0, 0, 0); tl.setSpacing(6)
        self.empty = plain_label("", wrap=True, style=f"color:{theme.MUTED};font-size:16px;padding:12px;")
        self.empty.setAlignment(Qt.AlignCenter); self.empty.hide()
        self.b_first = btn("Teach my first sign", "primary", 56, size_px=18)
        self.b_first.clicked.connect(self.teach_first); self.b_first.hide()
        self.list = QListWidget()
        self.list.setViewMode(QListView.IconMode); self.list.setResizeMode(QListView.Adjust)
        self.list.setMovement(QListView.Static); self.list.setUniformItemSizes(True)
        self.list.setGridSize(CELL); self.list.setSpacing(0); self.list.setWrapping(True)
        self.list.setVerticalScrollMode(QAbstractItemView.ScrollPerPixel)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setItemDelegate(SignDelegate(self.list)); self.list.setMinimumHeight(2 * CELL.height() + 4)
        self.list.itemClicked.connect(lambda it: self.play_entry.emit(it.data(ROLE_ID)))
        self.list.itemDoubleClicked.connect(lambda it: self.play_entry.emit(it.data(ROLE_ID)))
        self.list.itemActivated.connect(lambda it: self.play_entry.emit(it.data(ROLE_ID)))
        self.list.currentItemChanged.connect(self._on_current)
        self.list.verticalScrollBar().valueChanged.connect(lambda _: self._vis.start())
        tl.addWidget(self.empty); tl.addWidget(self.b_first, 0, Qt.AlignHCenter); tl.addWidget(self.list, 1)
        self.chk_hidden = TickBox("Show signs I hid"); self.chk_hidden.setStyleSheet(f"font-size:13px;color:{theme.MUTED};")
        self.chk_hidden.toggled.connect(self._filters_changed)
        tl.addWidget(self.chk_hidden, 0, Qt.AlignLeft)
        top.setMinimumHeight(2 * CELL.height() + 4 + 6 + self.chk_hidden.sizeHint().height())
        split.addWidget(top)
        panel = self._build_edit(); self.panel = panel
        panel.setMinimumHeight(self.PANEL_MIN)
        split.addWidget(panel)
        split.setStretchFactor(0, 1); split.setStretchFactor(1, 1)
        split.drag_done.connect(self._save_ratio); split.reset_done.connect(self._save_ratio)
        v.addWidget(split, 1)

    PANEL_MIN = 130          # the Save / Cancel row plus a sliver of the form

    # ------------------------------------------------------------------ remembered splitter position
    def bind_settings(self, get, set_):
        """Optional: main window hands over its own settings (get(key, default), set(key, value))."""
        self._settings_io = (get, set_)
        self.split.ratio = self._load_ratio(); self.split.apply_ratio()

    def _load_ratio(self):
        try:
            if self._settings_io:
                v = self._settings_io[0](SPLIT_KEY, None)
            else:
                st = getattr(self.lib, "store", None)
                v = (st.load_settings() if hasattr(st, "load_settings") else self._mem_settings).get(SPLIT_KEY)
            v = float(v)
            return v if 0.05 <= v <= 0.95 else SPLIT_DEFAULT
        except Exception:
            return SPLIT_DEFAULT

    def _save_ratio(self):
        v = round(float(self.split.ratio), 3)
        try:
            if self._settings_io:
                self._settings_io[1](SPLIT_KEY, v); return
            w = self.window()
            if w is not self and hasattr(w, "_save_setting"):     # the main window keeps the one true settings dict
                w._save_setting(SPLIT_KEY, v); return
            st = getattr(self.lib, "store", None)
            if hasattr(st, "save_settings") and hasattr(st, "load_settings"):
                cur = dict(st.load_settings() or {}); cur[SPLIT_KEY] = v; st.save_settings(cur)
                if hasattr(st, "audit"):
                    st.audit("settings_update", "settings", "signer", {"key": SPLIT_KEY})
                return
        except Exception:
            pass                                                  # never break the screen over a setting
        self._mem_settings[SPLIT_KEY] = v

    # ------------------------------------------------------------------ switch on / off in bulk
    def _build_bulk(self):
        box = card(); self.bulk_box = box
        bl = QVBoxLayout(box); bl.setContentsMargins(10, 6, 10, 6); bl.setSpacing(4)
        self.all_label = plain_label("", style="font-size:15px;font-weight:bold;")
        self.b_all_on = btn("Switch all on", "", 32); self.b_all_off = btn("Switch all off", "", 32)
        self.b_all_on.setToolTip("Switch on every sign in the dictionary (signs you hid stay as they are)")
        self.b_all_off.setToolTip("Switch off every sign, including the ones you recorded. You can undo it for a little while.")
        self.b_all_on.clicked.connect(lambda: self.bulk(None, True)); self.b_all_off.clicked.connect(lambda: self.bulk(None, False))
        self.b_groups = btn("Switch by group...", "", 32)
        self.b_groups.setToolTip("Switch a whole collection (recorded by me, switched on, dictionary) "
                                 "or any category on or off")
        self.b_groups.clicked.connect(self.open_groups)
        self.all_row = AdaptiveRow(self.all_label, [self.b_all_on, self.b_all_off, self.b_groups]); bl.addWidget(self.all_row)
        self.cat_label = plain_label("", style="font-size:15px;")
        self.b_cat_on = btn("Switch on", "", 32); self.b_cat_off = btn("Switch off", "", 32)
        self.b_cat_on.clicked.connect(lambda: self.bulk(self.category, True))
        self.b_cat_off.clicked.connect(lambda: self.bulk(self.category, False))
        self.cat_row = AdaptiveRow(self.cat_label, [self.b_cat_on, self.b_cat_off]); bl.addWidget(self.cat_row)
        # undo bar
        self.undo_bar = QFrame(); self.undo_bar.setObjectName("undobar")
        self.undo_bar.setStyleSheet(f"QFrame#undobar{{background:{theme.WARN_BG};border-radius:10px;}}")
        ul = QHBoxLayout(self.undo_bar); ul.setContentsMargins(10, 4, 6, 4); ul.setSpacing(8)
        self.undo_label = plain_label("", wrap=True, style="font-size:15px;font-weight:bold;color:#92400e;")
        self.b_undo = btn("Undo", "primary", 32); self.b_undo.clicked.connect(self.undo_bulk)
        ul.addWidget(self.undo_label, 1); ul.addWidget(self.b_undo)
        self.undo_bar.hide(); bl.addWidget(self.undo_bar)
        return box

    def _fill_categories(self):
        keep = self.category
        cats = [c for c in self.lib.categories() if c] if hasattr(self.lib, "categories") else []
        self.cat_combo.blockSignals(True)
        self.cat_combo.clear(); self.cat_combo.addItem(ALL_CATS, None)
        for c in cats:
            self.cat_combo.addItem(c, c)
        i = self.cat_combo.findData(keep) if keep else 0
        self.cat_combo.setCurrentIndex(max(i, 0)); self.category = self.cat_combo.currentData()
        self.cat_combo.blockSignals(False)

    def set_category(self, name):
        """Choose a category in the filter (None = all categories)."""
        i = self.cat_combo.findData(name) if name else 0
        if i >= 0:
            self.cat_combo.setCurrentIndex(i)

    def _category_changed(self, *_):
        self.category = self.cat_combo.currentData()
        self.refresh()

    def active_counts(self, category=None):
        if hasattr(self.lib, "active_counts"):
            return self.lib.active_counts(category)
        es = [e for e in self.lib.entries() if not category or e.category == category]
        return sum(1 for e in es if self.is_active(e)), len(es)

    def update_bulk(self):
        n_on, n = self.active_counts()
        self.all_label.setText(f"Signs switched on: {n_on:,} of {n:,}")
        can = self.editable and self._can_switch() and hasattr(self.lib, "set_active_many")
        self.b_all_on.setEnabled(can and n_on < n); self.b_all_off.setEnabled(can and n_on > 0)
        cat = self.category
        if cat:
            c_on, c_n = self.active_counts(cat)
            self.cat_label.setText(f"{cat}: {c_on:,} of {c_n:,} switched on")
            tip = f"Acts on all {c_n:,} signs of '{cat}', not only the ones shown in the list"
            self.b_cat_on.setEnabled(can and c_on < c_n); self.b_cat_off.setEnabled(can and c_on > 0)
            self.cat_label.setToolTip(tip)
            self.b_cat_on.setText(f"Switch on all {cat}"); self.b_cat_off.setText(f"Switch off all {cat}")
            self.cat_row.setVisible(True)
        else:
            tip = "Choose a category above to switch a whole category on or off"
            self.cat_label.setText("Pick a category above to use these buttons.")
            self.cat_label.setToolTip(tip)
            self.b_cat_on.setEnabled(False); self.b_cat_off.setEnabled(False)
            self.b_cat_on.setText("Switch on"); self.b_cat_off.setText("Switch off")
            self.cat_row.setVisible(False)                   # nothing to act on until a category is chosen
        self.cat_label.setStyleSheet(f"background:transparent;font-size:15px;" + ("" if cat else f"color:{theme.MUTED};"))
        self.b_cat_on.setToolTip(tip); self.b_cat_off.setToolTip(tip)
        if not cat:
            self.b_cat_on.setToolTip(tip); self.b_cat_off.setToolTip(tip)

    def bulk(self, category, on):
        """Switch every (not hidden) sign of a category - or of all - on or off in one go."""
        ids = [e.id for e in self.lib.entries(category=category)]
        return self.bulk_ids(ids, on)

    # ---- groups: three collections and every category
    COLLECTIONS = (("mine", "Recorded by me"), ("inuse", "Switched on"), ("all", "All dictionary"))

    def group_ids(self, kind, key=None):
        """Ids (not hidden) of one group: kind 'mine' | 'inuse' | 'all' | 'category' (key = the category name)."""
        if kind == "category":
            return [e.id for e in self.lib.entries(category=key)]
        es = self.lib.entries()
        if kind == "mine":
            return [e.id for e in es if is_mine(e)]
        if kind == "inuse":
            return [e.id for e in es if self.is_active(e)]
        return [e.id for e in es]

    def groups(self):
        """[(kind, key, title, n_on, n_total)] for the three collections, then every category."""
        out = []
        for kind, title in self.COLLECTIONS:
            ids = self.group_ids(kind)
            out.append((kind, None, title, sum(1 for i in ids if self.is_active_id(i)), len(ids)))
        for c in (self.lib.categories() if hasattr(self.lib, "categories") else []):
            if c:
                ids = self.group_ids("category", c)
                out.append(("category", c, c, sum(1 for i in ids if self.is_active_id(i)), len(ids)))
        return out

    def bulk_group(self, kind, key, on):
        return self.bulk_ids(self.group_ids(kind, key), on)

    def open_groups(self):
        dlg = GroupSwitchDialog(self)
        self._groups_dialog = dlg
        dlg.exec()

    def bulk_ids(self, ids, on):
        """Switch the given signs on or off with ONE write, ONE audit record, ONE templates push and an Undo."""
        if not self.editable or not hasattr(self.lib, "set_active_many"):
            return False
        changing = [i for i in ids if self.is_active_id(i) != on]
        if not changing:
            return False
        if on and len(changing) > BULK_CONFIRM:
            if not self.confirm_many(len(changing)):
                return False
        try:
            prev = self.lib.set_active_many(ids, on)
        except Exception as ex:
            self.message.emit(f"Could not change that: {ex}"); self.say(f"Could not change that: {ex}", err=True)
            return False
        n = len(changing)
        word = f"{n:,} sign{'s' if n != 1 else ''}"
        if on:
            self._show_undo(f"Switched on {word}", prev)
        else:
            mine = sum(1 for i in changing if self._is_mine(i))
            extra = f" (including {mine} of yours)" if mine else ""
            self._show_undo(f"Switched off {word}{extra}", prev)
        self.reload(); self.changed.emit()                  # one refresh, ONE templates push
        return True

    def _is_mine(self, id_):
        try:
            e = self.lib.get(id_)
            return e.source == "recorded" or my_examples(e) > 0
        except Exception:
            return False

    def is_active_id(self, id_):
        try:
            return bool(self.lib.is_active(id_))
        except Exception:
            return False

    def _show_undo(self, text, prev):
        self._undo = prev
        self.undo_label.setText(text + ("" if prev else "."))
        self.b_undo.setVisible(bool(prev)); self.undo_bar.show()
        self._undo_timer.start(self.undo_ms)

    def _hide_undo(self):
        self._undo = None; self.undo_bar.hide(); self._undo_timer.stop()

    def undo_bulk(self):
        prev = self._undo
        self._hide_undo()
        if not prev:
            return
        try:
            self.lib.restore_active(prev)
        except Exception as ex:
            self.say(f"Could not undo that: {ex}", err=True); return
        self.reload(); self.changed.emit()
        self._show_undo("Back the way it was", None)
        self._undo_timer.start(3000)

    def _ask_many(self, n):
        mb = QMessageBox(self); mb.setIcon(QMessageBox.Question); mb.setWindowTitle("Switch on a lot of signs?")
        mb.setText(f"You are about to switch on {n:,} signs.")
        mb.setInformativeText("Switching on very many dictionary signs makes me confuse one sign with another. "
                              "Signs you recorded yourself work best.\n\n"
                              "Tip: switch on one category at a time, or just the signs you need. "
                              "If you cancel, nothing changes.")
        ok = mb.addButton(f"Yes, switch on all {n:,}", QMessageBox.AcceptRole)
        cancel = mb.addButton("No, keep it as it is", QMessageBox.RejectRole); mb.setDefaultButton(cancel)
        mb.exec()
        return mb.clickedButton() is ok

    def _build_edit(self):
        wrap = QWidget(); wv = QVBoxLayout(wrap); wv.setContentsMargins(0, 0, 0, 0)
        self.pick_hint = hint("Tap a sign above to see it, switch it on, or record your own example.", 15)
        self.pick_hint.setAlignment(Qt.AlignCenter)
        wv.addWidget(self.pick_hint)
        sa = QScrollArea(); sa.setWidgetResizable(True); self.edit_scroll = sa; sa.setMinimumHeight(60)
        box = card(); self.edit_box = box
        g = QVBoxLayout(box); g.setContentsMargins(16, 14, 16, 14); g.setSpacing(8)
        self.ed_head = heading("", 24)
        self.ed_info = plain_label("", wrap=True, style=MUTED)
        g.addWidget(self.ed_head); g.addWidget(self.ed_info)
        self.need_mine = plain_label("", wrap=True, style=f"color:#92400e;background:{theme.WARN_BG};border-radius:10px;padding:8px 10px;font-size:14px;")
        g.addWidget(self.need_mine)
        self.b_take = btn("Record my own example", "primary", 54, size_px=18)
        self.b_take.clicked.connect(lambda: self._record("take"))
        g.addWidget(self.b_take)

        row = QHBoxLayout(); row.setSpacing(12)
        self.sw_active = Switch(False); self.sw_active.setAccessibleName("Switched on - use in conversation")
        self.sw_active.toggled.connect(self.set_active_cur)
        lab = plain_label("Switched on", style="font-size:17px;font-weight:bold;")
        row.addWidget(self.sw_active); row.addWidget(lab); row.addStretch(1)
        self.active_row = QWidget(); self.active_row.setLayout(row)
        g.addWidget(self.active_row)
        self.active_note = hint("When this is on, I will recognise this sign while you talk.", 15)
        g.addWidget(self.active_note)

        self.ed_label = QLineEdit(); self.ed_label.setMaxLength(80); self.ed_label.setMinimumHeight(44)
        self.ed_label.setPlaceholderText("The word that gets written when I see this sign")
        self.ed_label.setStyleSheet("font-size:17px;")
        self.ed_label.textChanged.connect(lambda t: auto_direction(self.ed_label, t))
        g.addWidget(plain_label("Word that gets written", style=MUTED)); g.addWidget(self.ed_label)
        sv = QHBoxLayout(); sv.setSpacing(8)
        self.b_save = btn("Save", "good", 46); self.b_cancel = btn("Cancel", "", 46)
        self.b_save.clicked.connect(self.save); self.b_cancel.clicked.connect(self._fill_form)
        sv.addWidget(self.b_save); sv.addWidget(self.b_cancel); sv.addStretch(1)
        self.ed_msg = plain_label("", wrap=True, style="font-size:14px;")

        more = Collapsible("More options"); self.more = more; ml = more.body_layout
        self.ed_notes = QPlainTextEdit(); self.ed_notes.setFixedHeight(60)
        self.ed_notes.setPlaceholderText("Notes about this sign (optional)")
        self.ed_alias = QLineEdit(); self.ed_alias.setPlaceholderText("Other words that find this sign, separated by commas")
        self.ed_cat = QComboBox(); self.ed_cat.setEditable(True); self.ed_cat.lineEdit().setMaxLength(40)
        self.ed_notes.textChanged.connect(lambda: auto_direction(self.ed_notes, self.ed_notes.toPlainText()))
        self.ed_alias.textChanged.connect(lambda t: auto_direction(self.ed_alias, t))
        for n, w in (("Notes", self.ed_notes), ("Other words", self.ed_alias), ("Category", self.ed_cat)):
            ml.addWidget(plain_label(n, style=MUTED)); ml.addWidget(w)
        self.b_rerec = btn("Record this sign again (replace)", "", 42)
        self.b_hide = btn("Hide", "", 42); self.b_reset = btn("Undo my changes", "", 42)
        self.b_delete = btn("Delete this sign", "danger", 42)
        self.b_export = btn("Save my examples for training...", "", 42)
        self.b_rerec.clicked.connect(lambda: self._record("replace"))
        self.b_hide.clicked.connect(self.toggle_hidden); self.b_reset.clicked.connect(self.reset_sign)
        self.b_delete.clicked.connect(self.delete_sign); self.b_export.clicked.connect(self.export_signs)
        r2 = QHBoxLayout(); r2.addWidget(self.b_hide); r2.addWidget(self.b_reset)
        ml.addWidget(self.b_rerec); ml.addLayout(r2); ml.addWidget(self.b_delete); ml.addWidget(self.b_export)
        ml.addWidget(hint("Your changes are stored encrypted on this computer. The dictionary files are never changed.", 13))
        g.addWidget(more)
        # 'My recordings': a second drop-down under 'More options' - the examples I recorded (or learned) of this sign
        self.rec_box = Collapsible("My recordings"); rl = self.rec_box.body_layout
        self.rec_panel = RecordingsPanel(self.lib, mirror=getattr(self, "mirror", True), editable=self.editable)
        self.rec_panel.watch.connect(self.watch_take)
        self.rec_panel.record_more.connect(lambda _sid: self._record("take"))
        self.rec_panel.changed.connect(lambda: self.cur is not None and self.after_change(self.cur))
        rl.addWidget(self.rec_panel)
        g.addWidget(self.rec_box)
        self.rec_box.hide()
        sa.setWidget(box); wv.addWidget(sa, 1)
        self.save_bar = QWidget(); self.save_bar.setLayout(sv)          # pinned below the scroll area: always visible
        sv.setContentsMargins(0, 4, 0, 0)
        wv.addWidget(self.ed_msg); wv.addWidget(self.save_bar)
        self._edit_widgets = [self.ed_label, self.ed_notes, self.ed_alias, self.ed_cat, self.b_save, self.b_cancel,
                              self.b_reset, self.b_hide, self.b_delete, self.b_rerec, self.b_take, self.sw_active]
        if not self.editable:
            self.say("The safe storage is not available, so signs can only be looked at "
                     "(editing and recording are switched off).", err=True)
            self.b_export.setEnabled(False)
        self._set_edit_enabled(False)
        sa.hide(); self.save_bar.hide(); self.ed_msg.hide()
        return wrap

    # ------------------------------------------------------------------ mirror (display only)
    def set_mirror(self, on):
        on = bool(on)
        if on == self.mirror:
            return
        self.mirror = on
        self.rec_panel.set_mirror(on)
        for i in range(self.list.count()):          # icons are re-fetched from the cache keyed by mirror state
            it = self.list.item(i)
            it.setData(Qt.DecorationRole, self._cache.get((it.data(ROLE_ID), on)))
        self._queue = []
        self._vis.start()

    # ------------------------------------------------------------------ filtering / search
    def _pick_default_chip(self):
        key = "inuse" if self.in_use_count() > 0 else "all"
        self.chips[key].setChecked(True)

    def chip(self):
        for k, b in self.chips.items():
            if b.isChecked():
                return k
        return "all"

    def set_chip(self, key):
        self.chips[key].setChecked(True)

    def _filters_changed(self, *_):
        self.refresh()

    def _on_search_text(self, t):
        auto_direction(self.search, t)
        self._debounce.start()

    def _query(self):
        return self.search.text().strip()

    def results(self):
        q = self._query()
        cat = self.category
        if q:                                    # searching looks through every sign (of the chosen category)
            res = self.lib.search(q, limit=600)
            return [e for e in res if e.category == cat] if cat else res
        res = self.lib.entries(self.chk_hidden.isChecked(), category=cat)
        c = self.chip()
        if c == "mine":
            res = [e for e in res if is_mine(e)]
        elif c == "inuse":
            res = [e for e in res if self.is_active(e)]
        return res

    def refresh(self, keep=None):
        """Rebuild the list from the current query / filters (cheap: items only, icons come lazily)."""
        self._debounce.stop()
        keep = keep or self.cur
        sb = self.list.verticalScrollBar(); pos = sb.value() if keep else 0
        res = self.results()
        self.list.setUpdatesEnabled(False); self.list.blockSignals(True)
        self.list.clear()
        sel = None
        seen, count = {}, {}
        for e in res:
            count[e.label.casefold()] = count.get(e.label.casefold(), 0) + 1
        for e in res:
            k = e.label.casefold(); seen[k] = seen.get(k, 0) + 1
            it = QListWidgetItem()
            it.setText(f"{e.label} ({seen[k]})" if count[k] > 1 else e.label)   # same word twice: told apart by number
            it.setData(ROLE_ID, e.id)
            n_rec = recordings_count(e)
            it.setData(ROLE_FLAGS, dict(active=self.is_active(e), mine=n_rec, hidden=e.hidden))
            tip = e.label + (f"\n{e.category}" if e.category else "") + \
                (("\n" + ", ".join(e.aliases)) if e.aliases else "") + \
                (f"\nYou have {n_rec} recording{'s' if n_rec != 1 else ''} of this sign: open 'My recordings' below to see or delete them" if n_rec else "")
            it.setToolTip(html.escape(tip).replace("\n", "<br>")); it.setSizeHint(CELL)
            pm = self._cache.get((e.id, self.mirror))
            if pm is not None:
                it.setData(Qt.DecorationRole, pm)
            self.list.addItem(it)
            if e.id == keep:
                sel = it
        if sel is not None:
            self.list.setCurrentItem(sel)
        self.list.blockSignals(False); self.list.setUpdatesEnabled(True)
        sb.setValue(pos)
        q = self._query(); n = len(res)
        self.count.setText(f"{n:,} sign{'s' if n != 1 else ''}" + (f" for '{q}'" if q else ""))
        self.update_bulk()
        self.empty.setVisible(n == 0); self.list.setVisible(n > 0); self.b_first.setVisible(False)
        if n == 0:
            if q:
                self.empty.setText(f"I couldn't find '{q}'" + (f" in {self.category}" if self.category else "") +
                                   ". Try another spelling, or look in 'All dictionary'.")
            elif self.chip() == "mine":
                self.empty.setText("You haven't taught me any signs yet. Teach your first sign and it will appear here.")
                self.b_first.setVisible(self.editable)
            elif self.chip() == "inuse":
                self.empty.setText("No signs are switched on yet. Open 'All dictionary', tap a sign and switch on "
                                   "'Switched on'.")
            else:
                self.empty.setText("No signs to show.")
        self._update_phrase(q)
        if sel is None and self.cur is not None:
            self.count.setText(self.count.text() + " (the selected sign is not in this list)")
        self._vis.start()

    def _update_phrase(self, q):
        self._phrase = ([], [])
        words = [w for w in re.split(r"\W+", norm_text(q)) if w]
        if len(words) >= 2:
            found, missing = self.lib.resolve_phrase(q)
            self._phrase = ([e.id for e in found], missing)
        ids, missing = self._phrase
        self.phrase_btn.setVisible(bool(ids) and len(words) >= 2)
        self.phrase_btn.setText(f"Show the whole phrase ({len(ids)} sign{'s' if len(ids) != 1 else ''})")
        self.missing.setVisible(bool(missing))
        self.missing.setText("No sign found for: " + ", ".join(missing) if missing else "")

    def play_current_phrase(self):
        ids, missing = self._phrase
        if ids:
            self.play_phrase.emit(list(ids), list(missing))

    def on_enter(self):
        """Enter in the search box: play the phrase (multi-word) or the top result."""
        self.refresh()
        ids, missing = self._phrase
        if len(ids) >= 2:
            self.play_current_phrase()
        elif self.list.count():
            self.play_entry.emit(self.list.item(0).data(ROLE_ID))
            self.list.setCurrentRow(0)

    # ------------------------------------------------------------------ lazy thumbnails
    def _collect_visible(self):
        vp = self.list.viewport().rect().adjusted(0, -CELL.height(), 0, CELL.height() * 2)
        rows = []
        for i in range(self.list.count()):
            it = self.list.item(i)
            if it.data(Qt.DecorationRole) is None and self.list.visualItemRect(it).intersects(vp):
                rows.append(it.data(ROLE_ID))
        if not rows and self.list.count() and not self.list.isVisible():
            rows = [self.list.item(i).data(ROLE_ID) for i in range(min(40, self.list.count()))
                    if self.list.item(i).data(Qt.DecorationRole) is None]
        self._queue = rows
        if rows:
            self._fill.start()

    def _thumb(self, id_, mirror):
        try:
            pm = frame_pixmap(self.lib.thumbnail_frame(id_), THUMB.width(), THUMB.height(), mirror=mirror)
        except Exception:
            pm = QPixmap(THUMB); pm.fill(QColor(theme.SCREEN))
            p = QPainter(pm); p.setPen(QColor("#94a3b8")); p.drawText(pm.rect(), Qt.AlignCenter, "no picture"); p.end()
        return pm

    def _fill_some(self):
        t0 = time.perf_counter()
        by_id = None
        mirror = self.mirror
        while self._queue and time.perf_counter() - t0 < 0.012:
            id_ = self._queue.pop(0)
            if (id_, mirror) not in self._cache:
                self._cache[(id_, mirror)] = self._thumb(id_, mirror)
            if by_id is None:
                by_id = {self.list.item(i).data(ROLE_ID): self.list.item(i) for i in range(self.list.count())}
            it = by_id.get(id_)
            if it is not None:
                it.setData(Qt.DecorationRole, self._cache[(id_, mirror)])
        if not self._queue:
            self._fill.stop()

    def pending_thumbnails(self):
        return len(self._queue)

    def invalidate(self, id_):
        for m in (True, False):
            self._cache.pop((id_, m), None)

    # ------------------------------------------------------------------ selection / edit panel
    def select(self, id_):
        for i in range(self.list.count()):
            if self.list.item(i).data(ROLE_ID) == id_:
                self.list.setCurrentRow(i); self.list.scrollToItem(self.list.item(i)); return True
        return False

    def _on_current(self, it, _prev):
        self.cur = it.data(ROLE_ID) if it is not None else None
        self._fill_form()

    def _set_edit_enabled(self, on):
        for w in self._edit_widgets:
            w.setEnabled(bool(on) and self.editable)

    def _fill_form(self, *_):
        self._fill_form_inner()
        self.form_filled.emit()

    def _fill_form_inner(self):
        if self.cur is None:
            self.edit_scroll.hide(); self.save_bar.hide(); self.ed_msg.hide(); self.pick_hint.show(); self._set_edit_enabled(False)
            self.rec_panel.set_sign(None); self.rec_box.hide()
            return
        try:
            e = self.lib.get(self.cur)
        except KeyError:
            self.cur = None; self._fill_form(); return
        self.edit_scroll.show(); self.save_bar.show(); self.ed_msg.setVisible(bool(self.ed_msg.text())); self.pick_hint.hide()
        self._set_edit_enabled(True)
        self.ed_head.setText(e.label); auto_direction(self.ed_head, e.label)
        mine = my_examples(e)
        kind = "Your own sign" if e.source == "recorded" else "Dictionary sign"
        self.ed_info.setText(f"{kind}  -  {examples_text(e)}")
        if mine == 0:
            self.need_mine.setText("To recognise this sign reliably, I need at least one example recorded by YOU. "
                                   "Tap the blue button to record yours.")
        elif mine < GOAL_EXAMPLES:
            self.need_mine.setText(f"You recorded {mine} example{'s' if mine != 1 else ''}. "
                                   f"{GOAL_EXAMPLES - mine} more will help me a lot.")
        self.need_mine.setVisible(mine < GOAL_EXAMPLES)
        self.b_take.setText("Record my own example" if mine == 0 else "Record another example")
        self.ed_label.setText(e.label); self.ed_notes.setPlainText(e.notes)
        self.ed_alias.setText(", ".join(e.aliases))
        self.ed_cat.clear(); self.ed_cat.addItems([c for c in self.lib.categories() if c])
        self.ed_cat.setCurrentText(e.category)
        self._fill_recordings(e)
        self.b_hide.setText("Show again" if e.hidden else "Hide this sign")
        self.b_delete.setEnabled(self.editable and e.source == "recorded")
        self.b_delete.setToolTip("Only signs you recorded can be deleted" if e.source != "recorded" else "")
        self.b_reset.setEnabled(self.editable and (e.edited or e.hidden))
        can = self._can_switch()
        self.active_row.setVisible(can); self.active_note.setVisible(can)
        self.sw_active.blockSignals(True); self.sw_active.setChecked(self.is_active(e)); self.sw_active.blockSignals(False)
        self.active_note.setText("On: I will recognise this sign while you talk." if self.is_active(e) else
                                 "Off: I will not look for this sign." + (
                                     "  It works best after you record your own example." if mine == 0 else ""))

    def _fill_recordings(self, e):
        """The 'My recordings' drop-down: only for signs that have an example of mine (recorded or learned)."""
        n = recordings_count(e)
        if n == 0:
            self.rec_panel.set_sign(None); self.rec_box.hide()
            return
        self.rec_panel.set_sign(e.id)
        self.rec_box.set_title(f"My recordings ({n})")
        self.rec_box.show()

    def say(self, text, err=False):
        self.ed_msg.setText(text); self.ed_msg.setVisible(bool(text) and self.edit_scroll.isVisibleTo(self))
        self.ed_msg.setStyleSheet(f"font-size:14px;color:{theme.BAD if err else theme.GOOD};")

    @staticmethod
    def parse_aliases(text):
        return [a.strip() for a in re.split(r"[,،;\n]", text) if a.strip()]

    def _do(self, fn, ok_msg):
        if self.cur is None or not self.editable:
            return False
        try:
            fn()
        except LibraryError as ex:
            self.say(str(ex), err=True); return False
        except Exception as ex:
            self.say(f"Could not save: {ex}", err=True); return False
        self.say(ok_msg)
        self.after_change(self.cur)
        return True

    def reload(self):
        """Re-read list / form after the library changed (also from outside this tab)."""
        keep_msg = self.ed_msg.text(), self.ed_msg.styleSheet()
        self._fill_categories()
        self.refresh(keep=self.cur)
        self._fill_form()
        self.ed_msg.setText(keep_msg[0]); self.ed_msg.setStyleSheet(keep_msg[1])
        self.ed_msg.setVisible(bool(keep_msg[0]) and self.edit_scroll.isVisibleTo(self))

    def after_change(self, id_):
        self.invalidate(id_)
        self.reload()
        self.changed.emit()

    def set_active_cur(self, on):
        i = self.cur
        if i is None or not self.editable or not self._can_switch():
            return
        try:
            self.lib.set_active(i, bool(on))
        except Exception as ex:
            self.say(f"Could not change that: {ex}", err=True)
            self.sw_active.blockSignals(True); self.sw_active.setChecked(not on); self.sw_active.blockSignals(False)
            return
        self.say("Switched on." if on else "Switched off.")
        self.reload(); self.changed.emit()

    def save(self):
        i = self.cur
        self._do(lambda: self.lib.update(i, label=self.ed_label.text(), notes=self.ed_notes.toPlainText(),
                                         aliases=self.parse_aliases(self.ed_alias.text()),
                                         category=self.ed_cat.currentText()),
                 "Saved. The new word is used in your messages right away.")

    def reset_sign(self):
        i = self.cur
        if self.confirm("Undo my changes", "Go back to the original word and picture for this sign?"):
            self._do(lambda: self.lib.reset(i), "Back to the original.")

    def toggle_hidden(self):
        i = self.cur
        if i is None:
            return
        hid = self.lib.get(i).hidden
        self._do(lambda: (self.lib.unhide if hid else self.lib.hide)(i), "This sign is visible again." if hid else
                 "Hidden. Tick 'Show signs I hid' to see it again.")

    def delete_sign(self):
        i = self.cur
        if i is None:
            return
        if self.lib.get(i).source != "recorded":
            self.say("Dictionary signs cannot be deleted. You can hide them instead.", err=True); return
        if self.confirm("Delete sign", "Delete this sign and all your examples of it for good?"):
            if self._do(lambda: self.lib.delete_recorded(i), "Deleted."):
                self.cur = None; self._fill_form()

    def _record(self, mode):
        if self.cur is not None and self.editable:
            self.record_requested.emit(mode, self.cur)

    def export_signs(self):
        if not self.editable:
            return
        if not self.confirm("Save my examples for training",
                            "This writes your recorded examples (body and hand points only, no video) as plain, "
                            "UNENCRYPTED files into a folder you choose, so they can be used to train the "
                            "recogniser.\n\nKeep that folder private. Continue?"):
            return
        d = self.choose_dir()
        if not d:
            return
        try:
            n = self.lib.export_user_signs(d)
            self.say(f"Saved {n} example(s) to {d}")
        except Exception as ex:
            self.say(f"That did not work: {ex}", err=True)

    # ------------------------------------------------------------------ dialogs (replaceable in tests)
    def _ask(self, title, text):
        return QMessageBox.question(self, title, text) == QMessageBox.Yes

    def _pick_dir(self):
        return QFileDialog.getExistingDirectory(self, "Choose a folder", str(ROOT / "data" / "user_signs"))


class GroupSwitchDialog(QDialog):
    """'Switch by group': one row per collection and per category with 'n of m switched on' and its own
    Switch on / Switch off buttons. Acts on ALL (not hidden) signs of the group, like the buttons on the
    My signs screen (same confirmation for very many signs, same Undo bar there)."""

    def __init__(self, tab, parent=None):
        super().__init__(parent or tab)
        self.tab = tab
        self.setWindowTitle("Switch signs on or off by group")
        self.setStyleSheet(f"QDialog{{background:{theme.BG};}}")       # (a dialog does not inherit the page background)
        self.resize(700, 660)
        self.rows = {}                                    # (kind, key) -> (label, b_on, b_off)
        v = QVBoxLayout(self); v.setContentsMargins(16, 14, 16, 14); v.setSpacing(8)
        v.addWidget(heading("Switch signs on or off by group", 20))
        v.addWidget(hint("A group switches ALL of its signs at once. You can undo the last change on the My signs screen.", 14))
        self.note = plain_label("", wrap=True, style=f"font-size:15px;font-weight:bold;color:{theme.GOOD};")
        v.addWidget(self.note)
        sa = QScrollArea(); sa.setWidgetResizable(True); sa.setFrameShape(QFrame.NoFrame)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        sa.setStyleSheet("QScrollArea{background:transparent;}")
        inner = QWidget(); inner.setStyleSheet("background:transparent;"); self.col = QVBoxLayout(inner); self.col.setContentsMargins(0, 0, 6, 0); self.col.setSpacing(6)
        self.col.setAlignment(Qt.AlignTop)
        sa.setWidget(inner); v.addWidget(sa, 1)
        self._build_rows()
        close = btn("Done", "primary", 44); close.clicked.connect(self.accept); v.addWidget(close)
        self.refresh()

    def _build_rows(self):
        self.col.addWidget(plain_label("Collections", style="font-size:16px;font-weight:bold;"))
        first_cat = True
        for kind, key, title, _on, _n in self.tab.groups():
            if kind == "category" and first_cat:
                self.col.addSpacing(8); self.col.addWidget(plain_label("Categories", style="font-size:16px;font-weight:bold;"))
                first_cat = False
            row = card(); h = QHBoxLayout(row); h.setContentsMargins(12, 8, 8, 8); h.setSpacing(8)
            label = plain_label("", wrap=True, style="font-size:15px;")
            b_on, b_off = btn("Switch on", "", 36), btn("Switch off", "", 36)
            b_on.setMinimumWidth(104); b_off.setMinimumWidth(104)
            b_on.clicked.connect(lambda _=False, k=kind, c=key: self._act(k, c, True))
            b_off.clicked.connect(lambda _=False, k=kind, c=key: self._act(k, c, False))
            h.addWidget(label, 1); h.addWidget(b_on); h.addWidget(b_off)
            self.col.addWidget(row)
            self.rows[(kind, key)] = (label, b_on, b_off)

    def refresh(self):
        can = self.tab.editable and self.tab._can_switch() and hasattr(self.tab.lib, "set_active_many")
        for kind, key, title, n_on, n in self.tab.groups():
            r = self.rows.get((kind, key))
            if r is None:
                continue
            label, b_on, b_off = r
            label.setText(f"Switched on right now: {n_on:,}" if kind == "inuse"
                          else f"{title}: {n_on:,} of {n:,} switched on")
            b_on.setEnabled(can and n_on < n); b_off.setEnabled(can and n_on > 0)
            tip = f"Acts on all {n:,} sign{'s' if n != 1 else ''} of '{title}'"
            b_on.setToolTip(tip); b_off.setToolTip(tip)

    def _act(self, kind, key, on):
        before = self.tab.undo_label.text()
        changed = self.tab.bulk_group(kind, key, on)
        if changed:
            self.note.setText(self.tab.undo_label.text())
        self.refresh()

