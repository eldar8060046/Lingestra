"""'My recordings': the drop-down in the sign detail (under 'More options') that lists the examples of a sign
that you recorded - or that I learned from your corrections - so you can watch and delete them.

Each example has a looping skeleton preview (hover over it), a Watch button (plays it in the big view) and a two-tap
Delete (nothing is deleted by accident). The built-in dictionary example is shown for context but cannot be deleted, and
neither can the only example of a sign you created (delete the sign itself in 'More options' instead).

Everything shown is plain text. Only the position of the hands and body is stored - never video or pictures.
"""
import time

from PySide6.QtCore import QTimer, Signal
from PySide6.QtWidgets import QFrame, QHBoxLayout, QVBoxLayout, QWidget

from ..library import LibraryError
from . import theme
from .widgets import SkeletonCanvas, btn, hint, plain_label

ARM_MS = 3000                      # the second tap on 'Delete' must come within this time
DIM = "QPushButton:disabled{color:#b6c0cf;border:1px solid #e2e8f0;background:#f8fafc;}"


def when(ts):
    return time.strftime("%d %b, %H:%M", time.localtime(ts))


class RecordingRow(QFrame):
    """One example: preview, what it is, Watch and Delete."""

    def __init__(self, panel, ex, number, frames, seconds, removable, why_not, mirror):
        super().__init__()
        self.setObjectName("softcard")
        self.panel, self.take, self.kind = panel, ex["take"], ex["kind"]
        v = QVBoxLayout(self); v.setContentsMargins(8, 6, 8, 6); v.setSpacing(4)
        top = QHBoxLayout(); top.setSpacing(10)
        self.canvas = SkeletonCanvas(head=False); self.canvas.setFixedSize(84, 74)
        self.canvas.set_frames(frames); self.canvas.set_mirror(mirror)
        top.addWidget(self.canvas)
        if ex["kind"] == "dictionary":
            title, sub = "Built-in example", "From the UAE sign dictionary"
        elif ex["kind"] == "learned":
            title = f"Example {number}"
            sub = "Learned from a correction" + (f" - {when(ex['time'])}" if ex.get("time") else "")
        else:
            title, sub = f"Example {number}", "Recorded by you"
        col = QVBoxLayout(); col.setSpacing(2)
        self.title = plain_label(title, style="font-size:16px;font-weight:bold;")
        self.sub = plain_label(f"{sub} - {seconds:.1f} s", wrap=True, style=f"font-size:13px;color:{theme.MUTED};")
        col.addWidget(self.title); col.addWidget(self.sub); col.addStretch(1)
        top.addLayout(col, 1)
        v.addLayout(top)
        bt = QHBoxLayout(); bt.setSpacing(8)
        self.b_watch = btn("Watch", "", 38); self.b_watch.setToolTip("Play it in the big view")
        self.b_watch.clicked.connect(lambda: panel.watch.emit(panel.sign_id, self.take))
        self.b_del = btn("Delete", "danger", 38)
        self.b_del.clicked.connect(self._delete_clicked)
        if not removable:
            self.b_del.setEnabled(False); self.b_del.setToolTip(why_not); self.b_del.setStyleSheet(DIM)
        bt.addWidget(self.b_watch); bt.addWidget(self.b_del)
        v.addLayout(bt)
        self.armed = False
        self._timer = QTimer(self); self._timer.setSingleShot(True); self._timer.timeout.connect(self.disarm)

    # the preview loops while the pointer is over the row
    def enterEvent(self, e):
        self.canvas.play(True); super().enterEvent(e)

    def leaveEvent(self, e):
        self.canvas.play(False); super().leaveEvent(e)

    def disarm(self):
        self.armed = False
        try:
            self.b_del.setText("Delete")
        except RuntimeError:
            pass                                      # the row is gone

    def _delete_clicked(self):
        if not self.armed:                            # first tap: ask
            self.armed = True; self.b_del.setText("Tap again to delete"); self._timer.start(ARM_MS)
            return
        self._timer.stop()
        self.panel.delete(self.take)


class RecordingsPanel(QWidget):
    changed = Signal()                 # an example was deleted (the library, the cards and the recogniser must refresh)
    watch = Signal(str, int)           # (sign id, take number): play that example in the main view
    record_more = Signal(str)          # sign id: 'Record another example'

    def __init__(self, library, mirror=True, editable=True, parent=None):
        super().__init__(parent)
        self.setObjectName("page")
        self.lib, self.sign_id, self.mirror, self.editable = library, None, bool(mirror), bool(editable)
        self.rows = []
        v = QVBoxLayout(self); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(6)
        v.addWidget(hint("These are the examples I use to recognise this sign. Hover over one to see it move. "
                         "Deleting an example cannot be undone.", 13))
        self.note = plain_label("", wrap=True, style=f"font-size:14px;font-weight:bold;color:{theme.GOOD};")
        v.addWidget(self.note)
        self.col = QVBoxLayout(); self.col.setSpacing(6); v.addLayout(self.col)
        self.empty = plain_label("You have not recorded an example of this sign yet.", wrap=True,
                                 style=f"font-size:14px;color:{theme.MUTED};")
        v.addWidget(self.empty)
        self.b_more = btn("Record another example", "primary", 42)
        self.b_more.clicked.connect(lambda: self.sign_id is not None and self.record_more.emit(self.sign_id))
        v.addWidget(self.b_more)

    # ------------------------------------------------------------------ content
    def count(self):
        """How many examples of the current sign are mine / learned (not the built-in clip)."""
        return sum(1 for r in self.rows if r.kind != "dictionary")

    def set_sign(self, sign_id):
        """Show another sign (or None); the message about the last action is cleared."""
        self.sign_id = sign_id
        self.note.setText("")
        self.refresh()

    def set_mirror(self, on):
        self.mirror = bool(on)
        for r in self.rows:
            r.canvas.set_mirror(self.mirror)

    def refresh(self):
        for r in self.rows:
            r.canvas.play(False); r._timer.stop(); r.setParent(None); r.deleteLater()
        self.rows = []
        if self.sign_id is None:
            self.empty.hide(); self.b_more.setEnabled(False)
            return
        try:
            e = self.lib.get(self.sign_id)
            exs = self.lib.examples(self.sign_id)
        except KeyError:
            self.empty.setText("This sign is gone."); self.empty.show(); self.b_more.setEnabled(False)
            return
        only_one = (e.source == "recorded" and len(exs) <= 1)
        number = 0
        for ex in exs:
            number += ex["kind"] != "dictionary"
            try:
                frames = self.lib.animation(self.sign_id, ex["take"])
            except Exception:
                continue
            removable, why = True, ""
            if ex["kind"] == "dictionary":
                removable, why = False, "The built-in example cannot be deleted"
            elif only_one:
                removable, why = False, "This is the only example of the sign. To remove the whole sign, use 'More options' above"
            elif not self.editable:
                removable, why = False, "Changes cannot be saved right now"
            row = RecordingRow(self, ex, number, frames, len(frames) / 25.0, removable, why, self.mirror)
            self.col.addWidget(row); self.rows.append(row)
        self.empty.setText("You have not recorded an example of this sign yet.")
        self.empty.setVisible(self.count() == 0)
        self.b_more.setEnabled(self.editable)

    def delete(self, take):
        if self.sign_id is None:
            return False
        try:
            self.lib.remove_take(self.sign_id, take)
        except (LibraryError, KeyError) as ex:
            self.note.setStyleSheet(f"font-size:14px;font-weight:bold;color:{theme.BAD};")
            self.note.setText(f"Could not delete that: {ex}")
            return False
        self.refresh()
        self.changed.emit()                          # (the parent refreshes the cards and may rebuild this panel ...)
        self.note.setStyleSheet(f"font-size:14px;font-weight:bold;color:{theme.GOOD};")
        self.note.setText("Deleted.")                # ... so the message is set last
        return True
