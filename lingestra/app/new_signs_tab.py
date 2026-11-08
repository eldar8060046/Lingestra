"""'Teach a sign' screen: teach a sign from the camera (the hands-free flow itself lives in
MainWindow / MainStage) and sort out signs the recogniser noticed but did not know ('pending cards').

Only landmarks are handled here, never images. All user text is plain text.
MIRROR is display-only: pending-card pictures are flipped like the rest of the view, the stored data is not."""
import time

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QDialog, QFrame, QMessageBox, QHBoxLayout, QLineEdit, QListWidget, QPushButton, QScrollArea,
                               QVBoxLayout, QWidget)

from ..library import LibraryError, prepare_animation
from . import theme
from .library_tab import GOAL_EXAMPLES, my_examples
from .widgets import SkeletonCanvas, auto_direction, btn, card, heading, hint, plain_label

MUTED = f"color:{theme.MUTED};font-size:14px;"
MAX_CARDS = 40


def _anim(arr):
    try:
        return prepare_animation(np.asarray(arr, np.float32))
    except Exception:
        return np.asarray(arr, np.float32)


class PickSignDialog(QDialog):
    """Search the library and pick one sign (used by 'Add to an existing sign')."""

    def __init__(self, library, parent=None):
        super().__init__(parent)
        self.lib = library; self.chosen = None
        self.setWindowTitle("Add to an existing sign"); self.resize(460, 480)
        self.setStyleSheet(f"QDialog{{background:{theme.BG};}}")
        v = QVBoxLayout(self); v.setContentsMargins(18, 18, 18, 18); v.setSpacing(10)
        v.addWidget(heading("Add to an existing sign", 20))
        v.addWidget(hint("Pick the sign that this movement is another example of:"))
        self.q = QLineEdit(); self.q.setPlaceholderText("Search..."); self.q.setClearButtonEnabled(True)
        self.q.setMinimumHeight(44)
        self.q.textChanged.connect(self.search)
        self.list = QListWidget(); self.list.itemDoubleClicked.connect(lambda _: self.accept_current())
        self.list.setStyleSheet("QListWidget{background:#ffffff;border:1px solid #cbd5e1;border-radius:10px;}"
                                f"QListWidget::item{{padding:8px;}}QListWidget::item:selected{{background:{theme.PRIMARY};color:#ffffff;}}")
        ok = btn("Add to this sign", "primary", 50); ok.clicked.connect(self.accept_current)
        v.addWidget(self.q); v.addWidget(self.list, 1); v.addWidget(ok)
        self.search("")

    def search(self, text):
        auto_direction(self.q, text)
        self.list.clear()
        res = self.lib.search(text, limit=60) if text.strip() else self.lib.entries()[:60]
        for e in res:
            self.list.addItem(e.label); self.list.item(self.list.count() - 1).setData(Qt.UserRole, e.id)

    def accept_current(self):
        it = self.list.currentItem() or (self.list.item(0) if self.list.count() else None)
        if it is not None:
            self.chosen = it.data(Qt.UserRole); self.accept()


class PendingCard(QFrame):
    def __init__(self, tab, id_, arr, meta):
        super().__init__()
        self.tab, self.id, self.arr, self.meta = tab, id_, arr, meta
        self.selected = False
        self.setObjectName("card")
        h = QHBoxLayout(self); h.setContentsMargins(14, 14, 14, 14); h.setSpacing(14)
        self.canvas = SkeletonCanvas(head=False); self.canvas.setFixedSize(130, 130)
        self.canvas.set_mirror(tab.mirror)
        anim = _anim(arr); self.canvas.set_frames(anim)
        if anim.ndim == 3 and len(anim):
            self.canvas.set_index(int(np.argmax((~np.isnan(anim[:, 25:, 0])).sum(axis=1))))
        lcol = QVBoxLayout(); lcol.setSpacing(8)
        lcol.addWidget(self.canvas); self.b_play = btn("Watch it", "", 42, tip="Watch this movement in the big view")
        lcol.addWidget(self.b_play); lcol.addStretch(1)
        h.addLayout(lcol)
        v = QVBoxLayout(); v.setSpacing(6); h.addLayout(v, 1)
        when = meta.get("created")
        ts = time.strftime("%d %b, %H:%M", time.localtime(when)) if isinstance(when, (int, float)) and when > 1e9 else "earlier"
        v.addWidget(plain_label(f"Seen {ts}", style=MUTED))
        self.name = QLineEdit(); self.name.setMaxLength(80); self.name.setMinimumHeight(44)
        self.name.setPlaceholderText("Type a name for this sign")
        self.name.setStyleSheet("font-size:17px;")
        if meta.get("label"):
            self.name.setText(str(meta["label"]))
        self.name.textChanged.connect(self._name_changed)
        self.name.returnPressed.connect(self.promote)
        v.addWidget(self.name)
        self.b_add = btn("Add as new sign", "primary", 48, size_px=17)
        self.b_add.clicked.connect(self.promote)
        v.addWidget(self.b_add)
        self.name_hint = hint("Type a name above to add it as a new sign.", 13)
        v.addWidget(self.name_hint)
        row = QHBoxLayout(); row.setSpacing(8)
        self.b_merge = btn("Add to an existing sign", "", 42, tip="It is another example of a sign I already have")
        self.b_discard = btn("Delete", "danger", 42)
        row.addWidget(self.b_merge, 1); row.addWidget(self.b_discard)
        v.addLayout(row)
        self.msg = plain_label("", wrap=True, style=f"color:{theme.BAD};font-size:14px;"); self.msg.hide(); v.addWidget(self.msg)
        self.b_play.clicked.connect(lambda: tab.play_array.emit(self.title(), _anim(self.arr)))
        self.b_merge.clicked.connect(self.merge_dialog); self.b_discard.clicked.connect(self.discard)
        if not tab.editable:
            for w in (self.name, self.b_merge):
                w.setEnabled(False)
        self._name_changed(self.name.text())

    def _name_changed(self, t):
        auto_direction(self.name, t)
        has = bool(t.strip()) and self.tab.editable
        self.b_add.setEnabled(has)
        self.b_add.setToolTip("" if has else "Type a name first")
        self.name_hint.setVisible(not t.strip() and self.tab.editable)

    def title(self):
        return self.name.text().strip() or "A sign I did not know"

    def enterEvent(self, e):
        self.canvas.play(True); super().enterEvent(e)

    def leaveEvent(self, e):
        if not self.selected:
            self.canvas.play(False)
        super().leaveEvent(e)

    def set_selected(self, on):
        self.selected = on; self.canvas.play(on)
        self.setStyleSheet(f"QFrame#card{{border:2px solid {theme.PRIMARY};}}" if on else "")

    def mousePressEvent(self, e):
        self.tab.select_card(self); super().mousePressEvent(e)

    def error(self, text):
        self.msg.setText(text); self.msg.setVisible(bool(text))

    # ---- actions
    def promote(self):
        self.tab.promote(self)

    def merge_dialog(self):
        dlg = PickSignDialog(self.tab.lib, self)
        if dlg.exec() and dlg.chosen:
            self.tab.merge(self, dlg.chosen)

    def discard(self):
        self.tab.discard(self)


class NewSignsTab(QWidget):
    teach_requested = Signal(str)          # optional name typed beforehand
    play_array = Signal(str, object)       # (title, frames) -> main view
    record_more = Signal(str)              # sign id -> record one more example of it
    library_changed = Signal()
    pending_changed = Signal(int)
    message = Signal(str)

    def __init__(self, library, store, editable=True, parent=None):
        super().__init__(parent)
        self.lib, self.store, self.editable = library, store, editable
        self.cards = []; self._arr = {}; self.total = 0
        self.mirror = True
        self._saved_id = None
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(12)
        v.addWidget(heading("Teach a sign", 26))

        hero = card(); hv = QVBoxLayout(hero); hv.setContentsMargins(18, 16, 18, 16); hv.setSpacing(10)
        self.how = hint("1.  Type the word (you can also do it after).\n2.  Press the big button and wait for 3 - 2 - 1.\n"
                        "3.  Raise your hands and sign.  Lower your hands when you finish.", 15)
        self.name = QLineEdit(); self.name.setMaxLength(80); self.name.setMinimumHeight(52)
        self.name.setStyleSheet("font-size:19px;")
        self.name.setPlaceholderText("The word for this sign, for example: Mother")
        self.name.textChanged.connect(lambda t: auto_direction(self.name, t))
        self.name.returnPressed.connect(self._teach)
        self.teach_btn = btn("Teach a new sign", "primary", 68, tip="Shortcut: N", icon_name="teach", size_px=22)
        self.teach_btn.clicked.connect(self._teach)
        self.status = plain_label("", wrap=True, style="font-size:15px;")
        hv.addWidget(self.how); hv.addWidget(self.name); hv.addWidget(self.teach_btn); hv.addWidget(self.status)
        v.addWidget(hero)
        self.status.setVisible(False)
        if not editable:
            self.teach_btn.setEnabled(False); self.status.setVisible(True)
            self.status.setText("I can't save signs right now because the safe storage could not be opened.")
            self.status.setStyleSheet(f"font-size:15px;color:{theme.BAD};")

        # 'record more examples' nudge after saving a sign
        self.nudge = card("softcard"); nv = QVBoxLayout(self.nudge); nv.setContentsMargins(18, 14, 18, 14)
        self.nudge_text = plain_label("", wrap=True, style=f"font-size:18px;font-weight:bold;color:{theme.TEXT};")
        nr = QHBoxLayout()
        self.b_more = btn("Record another example", "primary", 52, size_px=17)
        self.b_more.clicked.connect(self._more)
        self.b_dismiss = btn("Not now", "", 52); self.b_dismiss.clicked.connect(self.nudge.hide)
        nr.addWidget(self.b_more); nr.addWidget(self.b_dismiss); nr.addStretch(1)
        nv.addWidget(self.nudge_text); nv.addLayout(nr)
        self.nudge.hide(); v.addWidget(self.nudge)

        hr = QHBoxLayout()
        self.header = heading("", 20)
        self.b_refresh = btn("Check for new ones", "", 40); self.b_refresh.clicked.connect(self.refresh)
        self.b_clear = btn("Clear all", "danger", 40); self.b_clear.clicked.connect(self.clear_all)
        self.b_clear.setToolTip("Delete every movement I saved that you have not named. Your own signs are not touched.")
        self.confirm_clear = self._ask_clear               # tests replace this
        hr.addWidget(self.header, 1); hr.addWidget(self.b_clear); hr.addWidget(self.b_refresh)
        v.addLayout(hr)
        v.addWidget(hint("While you talk, I save movements I did not recognise. Name one to add it as a new sign, "
                         "or delete it.", 14))
        self.scroll = QScrollArea(); self.scroll.setWidgetResizable(True); self.scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.inner = QWidget(); self.cl = QVBoxLayout(self.inner); self.cl.setAlignment(Qt.AlignTop); self.cl.setSpacing(10)
        self.cl.setContentsMargins(0, 0, 6, 0)
        self.scroll.setWidget(self.inner)
        self.empty = plain_label("Nothing here yet. When I see a movement I don't know while you are talking, it will "
                                 "appear here so you can name it.", wrap=True,
                                 style=f"color:{theme.MUTED};font-size:15px;padding:10px;")
        v.addWidget(self.empty); v.addWidget(self.scroll, 1)
        v.addWidget(hint("Only the position of your hands and body is saved - never video or pictures. "
                         "Everything stays on this computer, encrypted.", 13))
        self.refresh()

    # ------------------------------------------------------------------
    def set_mirror(self, on):
        self.mirror = bool(on)
        for c in self.cards:
            c.canvas.set_mirror(self.mirror)

    def _teach(self):
        if self.editable:
            self.teach_requested.emit(self.name.text().strip())

    def _more(self):
        if self._saved_id:
            self.record_more.emit(self._saved_id)

    def say(self, text, err=False):
        self.status.setText(text); self.status.setVisible(bool(text))
        self.status.setStyleSheet(f"font-size:15px;color:{theme.BAD if err else theme.GOOD};")

    def show_saved(self, id_):
        """After a sign was saved: congratulate, and suggest recording more examples (3 is the goal)."""
        self._saved_id = id_
        try:
            e = self.lib.get(id_)
        except KeyError:
            self.nudge.hide(); return
        n = my_examples(e)
        if n >= GOAL_EXAMPLES:
            self.nudge_text.setText(f"Wonderful! '{e.label}' now has {n} of your examples. I will recognise it much better.")
            self.b_more.setVisible(False); self.b_dismiss.setText("OK")
        else:
            left = GOAL_EXAMPLES - n
            self.nudge_text.setText(f"Great! '{e.label}' is saved. Record {left} more example{'s' if left != 1 else ''} "
                                    "so I recognise it better.")
            self.b_more.setVisible(self.editable); self.b_dismiss.setText("Not now")
        auto_direction(self.nudge_text, self.nudge_text.text())
        self.nudge.show()

    def refresh(self):
        """Diff-based: existing cards (and any text typed into them) are kept, new captures are added."""
        items = []
        if self.store is not None:
            try:
                items = self.store.list_unknown()
            except Exception as ex:
                self.say(f"I could not read the saved movements: {ex}", err=True)
        items.sort(key=lambda m: m.get("created") if isinstance(m.get("created"), (int, float)) else 0, reverse=True)
        self.total = len(items)
        keep = {m["id"]: m for m in items[:MAX_CARDS]}
        for c in [c for c in self.cards if c.id not in keep]:
            c.canvas.play(False); self.cards.remove(c); c.setParent(None); c.deleteLater(); self._arr.pop(c.id, None)
        have = {c.id: c for c in self.cards}
        order = []
        for m in items[:MAX_CARDS]:
            c = have.get(m["id"])
            if c is None:
                try:
                    arr = self._arr.get(m["id"])
                    if arr is None:
                        arr = self._arr[m["id"]] = self.store.load_unknown(m["id"])[0]
                except Exception:
                    continue
                c = PendingCard(self, m["id"], arr, m)
            order.append(c)
        self.cards = order
        for i, c in enumerate(order):
            self.cl.insertWidget(i, c)
        shown = len(self.cards)
        self.header.setText(f"Signs I did not recognise ({self.total})" + (f" - showing the newest {shown}" if self.total > shown else ""))
        self.empty.setVisible(shown == 0); self.scroll.setVisible(shown > 0)
        self.b_clear.setVisible(self.total > 0); self.b_clear.setEnabled(self.editable)
        self.pending_changed.emit(self.total)

    def select_card(self, card):
        for c in self.cards:
            c.set_selected(c is card)

    def _remove(self, card):
        self._arr.pop(card.id, None)
        self.refresh()

    def promote(self, card):
        name = card.name.text().strip()
        if not name:
            card.error("Type a name first."); return
        try:
            nid = self.lib.add_recorded(name, card.arr, notes="captured automatically")
        except LibraryError as ex:
            card.error(str(ex)); return
        except Exception as ex:
            card.error(f"Could not save: {ex}"); return
        self._finish(card, f"Added '{name}' as a new sign.")
        self.show_saved(nid)
        return nid

    def merge(self, card, entry_id):
        try:
            self.lib.add_take(entry_id, card.arr)
            label = self.lib.get(entry_id).label
        except Exception as ex:
            card.error(str(ex)); return
        self._finish(card, f"Added as another example of '{label}'.")
        self.show_saved(entry_id)

    def discard(self, card):
        try:
            self.store.delete_unknown(card.id, actor="signer-admin")
        except Exception as ex:
            card.error(f"Could not delete: {ex}"); return
        self._remove(card); self.say("Deleted.")

    def _ask_clear(self, n):
        mb = QMessageBox(self); mb.setIcon(QMessageBox.Warning); mb.setWindowTitle("Clear all saved movements?")
        mb.setText(f"Delete all {n:,} saved movement{'s' if n != 1 else ''} that you have not named?")
        mb.setInformativeText("This cannot be undone. The signs you taught me are not touched.")
        yes = mb.addButton("Yes, delete them all", QMessageBox.DestructiveRole)
        mb.addButton("No, keep them", QMessageBox.RejectRole)
        mb.setDefaultButton(mb.buttons()[-1])
        mb.exec()
        return mb.clickedButton() is yes

    def clear_all(self):
        """Delete EVERY saved unrecognised movement (not only the newest ones that are shown), after asking."""
        if not self.editable or self.store is None:
            return False
        try:
            ids = [m["id"] for m in self.store.list_unknown()]
        except Exception as ex:
            self.say(f"I could not read the saved movements: {ex}", err=True); return False
        if not ids or not self.confirm_clear(len(ids)):
            return False
        failed = 0
        for i in ids:
            try:
                self.store.delete_unknown(i, actor="signer-admin")
            except Exception:
                failed += 1
        self._arr.clear(); self.refresh()
        done = len(ids) - failed
        self.say(f"Cleared {done:,} saved movement{'s' if done != 1 else ''}."
                 + (f" {failed} could not be deleted." if failed else ""), err=bool(failed))
        return failed == 0

    def _finish(self, card, msg):
        try:
            self.store.delete_unknown(card.id, actor="signer-admin")
        except Exception as ex:
            msg += f" (the saved copy could not be removed: {ex})"
        self._remove(card); self.say(msg); self.library_changed.emit()
