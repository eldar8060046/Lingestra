"""Main window. Four plain sections in a left sidebar: Talk, My signs, Teach a sign, Settings.

Talks to the recognition worker only through Qt signals and queue-based commands, and to the
SecureStore through its public interface. The SignLibrary lives on the UI thread; the worker only
ever receives plain data (label map, templates).

MIRROR MODE is display only: everything the user looks at (camera + overlay, sign icons, playback and
review animations, pending cards) is flipped, but the landmarks the worker/recogniser works on are
never touched.
"""
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PySide6.QtCore import QRect, QSize, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QFontMetrics, QKeySequence, QShortcut
from PySide6.QtWidgets import (QApplication, QButtonGroup, QComboBox, QFileDialog, QFrame, QHBoxLayout,
                               QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow, QPushButton,
                               QScrollArea, QSlider, QStackedWidget, QStyledItemDelegate, QVBoxLayout, QWidget)

from ..library import LibraryError, SignLibrary, prepare_animation
from . import theme
from .library_tab import LibraryTab, my_examples
from .new_signs_tab import NewSignsTab
from .stage import MainStage, Segment
from ..live.fingers import ASPECT, CAL_VERSION
from .finger_dialog import FingerDialog
from .widgets import (ClarifyPanel, Collapsible, Dots, NavButton, Switch, TickBox, auto_direction, btn, card, heading,
                      hint, plain_label)

ROOT = Path(__file__).resolve().parents[2]
MUTED_CSS = f"color:{theme.MUTED};font-size:14px;"
SECTIONS = (("talk", "talk", "Talk"), ("signs", "signs", "My signs"),
            ("teach", "teach", "Teach a sign"), ("settings", "settings", "Settings"))
# 'How sure must I be before writing a word' -> recogniser threshold
SURE_LEVELS = (("fast", "Faster", 0.65, "Writes words sooner, but may guess wrong more often."),
               ("balanced", "Balanced", 0.80, "A good mix of speed and care."),
               ("careful", "Fewer mistakes", 0.90, "Waits until I am quite sure. Fewer wrong words, a little slower."))
# 'When I hold still, write the word after...' -> how long a still hold ends the word before it (seconds)
HOLD_LEVELS = (("quick", "Quick", 1.0, "Writes the word after 1 second of holding still. A long pause in the MIDDLE of a sign may cut it in two."),
               ("normal", "Normal", 1.6, "Writes the word after about 1.5 seconds of holding still."),
               ("patient", "Patient", 2.5, "Waits 2.5 seconds. Safest if you pause a lot in the middle of signs."))
# What happens to a finished message (the three-way toggle on the Talk screen)
SEND_POLICIES = (
    ("off", "Don't copy", "Messages are only shown here. Nothing is copied or pasted."),
    ("auto", "Copy automatically", "When you lower your hands the message is copied and pasted into the window you have open "
                                   "(after a short countdown)."),
    ("manual", "Send with Space", "Nothing is sent until you press the Space bar (or Send now). The message keeps growing "
                                  "while you sign."))
PILL = {   # kind -> (background, text colour)
    "ready": ("#f1f5f9", "#475569"), "watching": ("#e0f2fe", "#075985"), "working": ("#e0f2fe", "#64748b"),
    "got": ("#dbeafe", "#1d4ed8"),
    "notsure": (theme.WARN_BG, "#92400e"), "unknown": (theme.WARN_BG, "#92400e"), "error": (theme.BAD_BG, "#991b1b"),
}


def _rearm(b):
    try:
        b.setProperty("armed", False); b.setText("Remove")
    except RuntimeError:
        pass                                   # the row is gone


@dataclass
class Config:
    source: str = "webcam"
    model: str = str(ROOT / "models" / "demo_family.keras")
    send_mode: str = "clipboard"
    threshold: float = 0.80
    loop: bool = False
    passphrase: str = None
    store_dir: str = None        # SecureStore directory (default: the app-data dir); tests use scratch dirs


def _row(title, sub, switch):
    """Settings row: switch on the left, bold title + small explanation on the right."""
    w = QWidget(); h = QHBoxLayout(w); h.setContentsMargins(0, 4, 0, 4); h.setSpacing(14)
    v = QVBoxLayout(); v.setSpacing(2)
    v.addWidget(plain_label(title, wrap=True, style="font-size:17px;font-weight:bold;"))
    if sub:
        v.addWidget(hint(sub, 14))
    h.addWidget(switch, 0, Qt.AlignTop); h.addLayout(v, 1)
    return w


class MainWindow(QMainWindow):
    def __init__(self, cfg: Config = None, worker_factory=None, store=None, library=None):
        super().__init__()
        self.cfg = cfg or Config()
        self.worker_factory = worker_factory   # tests inject a fake worker
        self.worker = None
        self.store = store
        self.store_error = ""
        self.library = library
        self.words = []                        # words of the sentence being built
        self._hole = None                      # index in self.words of the open question's placeholder
        self._answering = None
        self.last_msg = None                   # [text, status-words, ok] of the LAST message only (no log)
        self._signing = False                  # the engine says the hands are in the signing area
        self.settings = {}                     # persisted choices (encrypted store, when available)
        self.section = "talk"
        self.pill_state = "ready"
        self._capture = None                   # state of the 'teach a sign' flow
        self._stage_entry = None               # library id shown in the stage (example selector)
        self._learned = None                   # (sign id, example number, label) behind the 'Learned from you' bar
        self.setWindowTitle("Lingestra - talk with signs")
        self.resize(1360, 860); self.setMinimumSize(1000, 640)
        if QApplication.instance() is not None:
            QApplication.setStyle("Fusion")
        self.setStyleSheet(theme.STYLESHEET)
        self._open_store()
        self._load_settings()
        self.send_policy = self._initial_send_policy()
        self.finger_cal = self._valid_cal(self.settings.get("finger_cal"))     # the signer's taught finger poses
        self._finger_dlg = None
        self._aspect = ASPECT                  # camera frame width / height (the Check window reads fingers in 3D)
        self._open_library()
        self._build()
        self._pill_timer = QTimer(self); self._pill_timer.setSingleShot(True)
        self._pill_timer.timeout.connect(self._pill_back)
        self.set_pill("ready")
        self.clarify.set_fingers_enabled(self.finger_answers_sw.isChecked())
        for key, fn in tuple((str(k % 10), (lambda k=k: self.clarify.press(k))) for k in range(1, 11)) + (
                        ("R", self.clarify._repeat), ("S", self.clarify._skip), ("N", self._shortcut_teach),
                        ("Esc", self._on_escape)):
            QShortcut(QKeySequence(key), self, activated=fn)
        # Space sends the message ONLY in 'Send with Space' mode on the Talk screen (never while typing in a field:
        # text fields keep the Space key themselves)
        self.space_shortcut = QShortcut(QKeySequence(Qt.Key_Space), self, activated=self._on_space)
        self._update_space()

    # ------------------------------------------------------------------ data loading
    def _model_classes(self):
        try:
            return [c for c in json.loads(Path(self.cfg.model).with_suffix(".json").read_text())["classes"]]
        except Exception:
            return None

    def _open_store(self):
        if self.store is None:
            try:
                from ..security.store import SecureStore
                self.store = SecureStore(self.cfg.store_dir, passphrase=self.cfg.passphrase)
            except Exception as ex:
                self.store_error = (f"The safe storage could not be opened ({ex}). You can look at signs, "
                                    "but new signs cannot be saved.")

    def _load_settings(self):
        if self.store is not None and hasattr(self.store, "load_settings"):
            try:
                self.settings = dict(self.store.load_settings() or {})
            except Exception:
                self.settings = {}

    def _save_setting(self, key, value):
        self.settings[key] = value
        if self.store is not None and hasattr(self.store, "save_settings"):
            try:
                self.store.save_settings(dict(self.settings))
                if hasattr(self.store, "audit"):
                    self.store.audit("settings_update", "settings", "signer", {"key": str(key)})   # key name only
            except Exception:
                pass                              # a settings hiccup must never break the app

    def _open_library(self):
        classes = self._model_classes()
        self.model_loaded = classes is not None
        if self.library is None:
            try:
                self.library = SignLibrary(store=self.store, model_classes=classes or [])
            except Exception as ex:
                self.store_error = self.store_error or f"The sign library could not be opened ({ex})."
                self.library = SignLibrary(store=None, model_classes=[], data_root=ROOT / "scratch" / "_none")
        self.editable = self.store is not None and self.library.persistent

    # ------------------------------------------------------------------ UI construction
    def _build(self):
        central = QWidget(); central.setObjectName("page"); self.setCentralWidget(central)
        root = QHBoxLayout(central); root.setContentsMargins(0, 0, 0, 0); root.setSpacing(0)
        root.addWidget(self._build_sidebar())
        self.pages = QStackedWidget(); root.addWidget(self.pages, 1)

        # ---- workspace: big view on the left, the section's panel on the right
        work = QWidget(); work.setObjectName("page")
        wh = QHBoxLayout(work); wh.setContentsMargins(18, 18, 18, 18); wh.setSpacing(18)
        left = QWidget(); left.setObjectName("page"); lv = QVBoxLayout(left); lv.setContentsMargins(0, 0, 0, 0); lv.setSpacing(10)
        left.setMinimumWidth(440)
        self.stage = MainStage(); self.camera = self.stage.camera
        lv.addWidget(self.stage, 1)
        lv.addWidget(self._build_talk_strip())
        self.status = QLabel(self.store_error); self.status.setWordWrap(True); self.status.setTextFormat(Qt.PlainText)
        self.status.setStyleSheet(f"color:{theme.BAD if self.store_error else theme.MUTED};font-size:14px;")
        self.status.setVisible(bool(self.store_error))
        lv.addWidget(self.status)
        wh.addWidget(left, 5)
        self.panels = QStackedWidget(); self.panels.setMinimumWidth(420)
        wh.addWidget(self.panels, 4)
        st = self.stage
        st.take_changed.connect(self._on_take)
        st.keep.connect(self._capture_keep); st.retake.connect(self._capture_retake)
        st.discard.connect(self._capture_discard)
        st.capture_retry.connect(self._capture_retake); st.capture_cancel.connect(st.go_live)
        st.back_to_live.connect(self._on_stage_closed)
        st.mirror_toggled.connect(lambda on: self.set_mirror(on))

        self.talk_panel = self._build_talk_panel(); self.panels.addWidget(self.talk_panel)
        self.lib_tab = LibraryTab(self.library, editable=self.editable, model_loaded=self.model_loaded)
        self.lib_tab.play_entry.connect(self.play_entry); self.lib_tab.play_phrase.connect(self.play_phrase)
        self.lib_tab.record_requested.connect(self._on_record_requested)
        self.lib_tab.watch_take.connect(lambda sid, take: self.play_entry(sid, take))
        self.lib_tab.changed.connect(self._push_all)
        self.lib_tab.teach_first.connect(lambda: self.show_section("teach"))
        self.panels.addWidget(self.lib_tab)
        self.new_tab = NewSignsTab(self.library, self.store, editable=self.editable)
        self.new_tab.teach_requested.connect(lambda name: self.begin_capture("new", None, name))
        self.new_tab.record_more.connect(lambda id_: self.begin_capture("take", id_, ""))
        self.new_tab.play_array.connect(self._play_array)
        self.new_tab.library_changed.connect(self._on_library_changed)
        self.new_tab.pending_changed.connect(self._on_pending)
        self.panels.addWidget(self.new_tab)
        self.pages.addWidget(work)
        self.pages.addWidget(self._build_settings())
        self._on_pending(self.new_tab.total)
        self._update_empty()
        self.set_mirror(bool(self.settings.get("mirror", True)), save=False)
        self.show_section("talk")

    def _build_sidebar(self):
        sb = QFrame(); sb.setObjectName("sidebar"); sb.setFixedWidth(196)
        v = QVBoxLayout(sb); v.setContentsMargins(6, 18, 6, 14); v.setSpacing(4)
        title = plain_label("Lingestra", style="font-size:24px;font-weight:bold;color:#ffffff;padding-left:12px;")
        v.addWidget(title)
        v.addWidget(plain_label("Talk with signs", style="font-size:14px;color:#94a3b8;padding-left:12px;"))
        v.addSpacing(16)
        self.nav = {}
        for key, ic, text in SECTIONS:
            b = NavButton(ic, text); self.nav[key] = b
            b.clicked.connect(lambda _=False, k=key: self.show_section(k)); v.addWidget(b)
        v.addStretch(1)
        self.help_btn = NavButton("help", "How it works"); self.help_btn.setAutoExclusive(False)
        self.help_btn.clicked.connect(self.show_help); v.addWidget(self.help_btn)
        return sb

    def _build_talk_strip(self):
        """Under the camera on the Talk screen: friendly status pill, 'how sure' dots, 'did you mean' buttons."""
        self.talk_strip = w = QWidget(); v = QVBoxLayout(w); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(8)
        row = QHBoxLayout(); row.setSpacing(10)
        self.pill = QLabel(); self.pill.setTextFormat(Qt.PlainText); self.pill.setWordWrap(True)
        self.pill.setMinimumHeight(60)
        self.pill_btn = btn("Teach it", "primary", 52, size_px=17); self.pill_btn.hide()
        self.pill_btn.clicked.connect(lambda: self.show_section("teach"))
        self.dots = Dots()
        self.dots_box = QWidget(); db = QVBoxLayout(self.dots_box); db.setContentsMargins(0, 0, 0, 0); db.setSpacing(0)
        self.sure_word = plain_label("How sure I am", style=f"font-size:15px;font-weight:bold;color:{theme.MUTED};")
        db.addWidget(self.sure_word, 0, Qt.AlignHCenter)
        db.addWidget(self.dots, 0, Qt.AlignHCenter)
        self.dots_box.hide()
        row.addWidget(self.pill, 1); row.addWidget(self.pill_btn); row.addWidget(self.dots_box)
        v.addLayout(row)
        self.send_note_row = QWidget(); nr = QHBoxLayout(self.send_note_row); nr.setContentsMargins(0, 0, 0, 0)
        self.send_note = plain_label("", wrap=True, style=f"font-size:16px;font-weight:bold;color:{theme.WARN};")
        b_cancel = btn("Cancel", "", 44); b_cancel.clicked.connect(self._cancel_send)
        nr.addWidget(self.send_note, 1); nr.addWidget(b_cancel)
        self.send_note_row.hide(); v.addWidget(self.send_note_row)
        self.clarify = ClarifyPanel()
        self.clarify.confirmed.connect(self._on_confirm); self.clarify.repeat.connect(self._on_repeat)
        self.clarify.skip.connect(self._on_skip)
        v.addWidget(self.clarify)
        # 'Saved' bar (after a correction was saved as another example; Undo for ~15 s). It sits right under the
        # question panel, where the question just disappeared, so nothing below it jumps while the user signs.
        self.learn_bar = card("softcard"); lbh = QHBoxLayout(self.learn_bar); lbh.setContentsMargins(16, 8, 16, 8)
        self.learn_lbl = plain_label("", wrap=True, style="font-size:15px;")
        self.learn_undo = btn("Undo", "", 40); self.learn_undo.clicked.connect(self._undo_learn)
        lbh.addWidget(self.learn_lbl, 1); lbh.addWidget(self.learn_undo)
        self.learn_bar.hide(); v.addWidget(self.learn_bar)
        self._learn_timer = QTimer(self); self._learn_timer.setSingleShot(True)
        self._learn_timer.timeout.connect(self._hide_learn_bar)
        return w

    def _build_talk_panel(self):
        sa = QScrollArea(); sa.setWidgetResizable(True)
        sa.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)       # the panel is built to fit; no sideways scrolling
        inner = QWidget(); inner.setObjectName("page"); v = QVBoxLayout(inner); v.setContentsMargins(0, 0, 8, 0); v.setSpacing(12)
        v.setAlignment(Qt.AlignTop)
        sa.setWidget(inner)

        # first-run 'How it works'
        self.help_card = card("softcard"); hv = QVBoxLayout(self.help_card); hv.setContentsMargins(18, 14, 18, 14)
        hv.addWidget(heading("How it works", 22))
        steps = QHBoxLayout(); steps.setSpacing(10)
        for ic, t, d in (("teach", "1. Teach your signs", "Show me a sign on the camera. I remember it."),
                         ("toggle", "2. Turn them on", "In 'My signs', switch on the signs you want to use."),
                         ("send", "3. Sign and send", "Sign to the camera. I write the words and send them to your chat.")):
            col = QVBoxLayout(); col.setSpacing(4)
            pic = QLabel(); pic.setPixmap(theme.icon(ic, 56, "#ffffff", bg=theme.PRIMARY)); pic.setFixedSize(56, 56)
            col.addWidget(pic); col.addWidget(plain_label(t, wrap=True, style="font-size:16px;font-weight:bold;"))
            col.addWidget(hint(d, 13)); col.addStretch(1); steps.addLayout(col, 1)
        hv.addLayout(steps)
        self.help_ok = btn("Got it, let's start", "primary", 46); self.help_ok.clicked.connect(self.dismiss_help)
        hv.addWidget(self.help_ok, 0, Qt.AlignLeft)
        v.addWidget(self.help_card)
        self.help_card.setVisible(not self.settings.get("seen_help", False))

        # store problem
        self.store_card = card("warncard"); sv = QVBoxLayout(self.store_card); sv.setContentsMargins(18, 14, 18, 14)
        sv.addWidget(plain_label(self.store_error, wrap=True, style=f"font-size:15px;color:#92400e;"))
        self.store_card.setVisible(bool(self.store_error))
        v.addWidget(self.store_card)

        # nothing taught yet
        self.empty_card = card("softcard"); ev = QVBoxLayout(self.empty_card); ev.setContentsMargins(18, 14, 18, 14)
        ev.addWidget(heading("Let's teach me your first sign", 20))
        ev.addWidget(hint("I don't know any of your signs yet. It only takes a moment: show me a sign and give it a word.", 15))
        self.b_first = btn("Teach my first sign", "primary", 56, size_px=18)
        self.b_first.clicked.connect(lambda: self.show_section("teach")); ev.addWidget(self.b_first)
        v.addWidget(self.empty_card)

        # too many signs switched on: slower and more mix-ups
        self.many_card = card("warncard"); mv = QVBoxLayout(self.many_card); mv.setContentsMargins(18, 12, 18, 12); mv.setSpacing(6)
        self.many_text = plain_label("", wrap=True, style="font-size:15px;color:#92400e;")
        self.b_many = btn("Choose which signs to switch off", "", 44); self.b_many.clicked.connect(self._show_switched_on)
        mv.addWidget(self.many_text); mv.addWidget(self.b_many, 0, Qt.AlignLeft)
        self.many_card.hide(); v.addWidget(self.many_card)

        # sentence
        sc = card(); scv = QVBoxLayout(sc); scv.setContentsMargins(18, 14, 18, 14); scv.setSpacing(10)
        scv.addWidget(plain_label("My message", style=f"font-size:15px;color:{theme.MUTED};font-weight:bold;"))
        self.caption = QLabel(); self.caption.setWordWrap(True); self.caption.setTextFormat(Qt.PlainText)
        self.caption.setMinimumHeight(96); self.caption.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        scv.addWidget(self.caption)
        self.send_hint = hint("", 14)
        scv.addWidget(self.send_hint)
        scv.addWidget(plain_label("When a message is finished:", style=f"font-size:14px;font-weight:bold;color:{theme.MUTED};"))
        pr = QHBoxLayout(); pr.setSpacing(6)
        self.policy_group = QButtonGroup(self); self.policy_btns = {}
        for key, text, tip in SEND_POLICIES:
            b = btn(text, "seg", 44); b.setCheckable(True); b.setToolTip(tip); b.setFocusPolicy(Qt.NoFocus)
            b.setChecked(key == self.send_policy)
            self.policy_group.addButton(b); self.policy_btns[key] = b; pr.addWidget(b)
            b.clicked.connect(lambda _=False, k=key: self.set_send_policy(k))
        scv.addLayout(pr)
        br = QHBoxLayout(); br.setSpacing(8)
        self.b_send = btn("Send now", "primary", 60, icon_name="send", size_px=19)
        self.b_undo = btn("Undo last word", "", 60); self.b_clear = btn("Clear", "danger", 60)
        self.b_send.clicked.connect(self.send_now); self.b_undo.clicked.connect(self.undo_word)
        self.b_clear.clicked.connect(self.clear_sentence)
        for b_ in (self.b_send, self.b_undo, self.b_clear):
            b_.setFocusPolicy(Qt.NoFocus)             # the Space key must never press one of these by accident
        br.addWidget(self.b_send, 3); br.addWidget(self.b_undo, 2); br.addWidget(self.b_clear, 1)
        scv.addLayout(br)
        v.addWidget(sc)

        # the last message that was sent (only the last one - no log)
        hc = card(); hcv = QVBoxLayout(hc); hcv.setContentsMargins(18, 14, 18, 14); hcv.setSpacing(4)
        hcv.addWidget(plain_label("Last message sent", style=f"font-size:15px;color:{theme.MUTED};font-weight:bold;"))
        self.last_empty = hint("Nothing sent yet.", 15)
        self.last_text = QLabel(); self.last_text.setWordWrap(True); self.last_text.setTextFormat(Qt.PlainText)
        self.last_text.setStyleSheet(f"font-size:24px;font-weight:bold;color:{theme.TEXT};")
        self.last_status = QLabel(); self.last_status.setTextFormat(Qt.PlainText)
        hcv.addWidget(self.last_empty); hcv.addWidget(self.last_text); hcv.addWidget(self.last_status)
        self.last_text.hide(); self.last_status.hide()
        v.addWidget(hc)

        # technical details (off by default)
        self.tech = Collapsible("Technical details"); self.tech.set_open(True)
        tl = self.tech.body_layout
        self.fps_lbl, self.lat_lbl, self.scale_lbl = QLabel("-"), QLabel("-"), QLabel("-")
        self.skip_lbl, self.conf_lbl, self.src_lbl = QLabel("-"), QLabel("-"), QLabel("-")
        for n, w in (("Frames per second", self.fps_lbl), ("Latency", self.lat_lbl), ("Window scale", self.scale_lbl),
                     ("Frames skipped", self.skip_lbl), ("Confidence", self.conf_lbl), ("Recognised by", self.src_lbl)):
            r = QHBoxLayout(); r.addWidget(plain_label(n, style=MUTED_CSS)); r.addStretch(1); r.addWidget(w); tl.addLayout(r)
        self.tech.setVisible(bool(self.settings.get("tech", False)))
        v.addWidget(self.tech)
        self._refresh_sentence()
        return sa

    def _build_settings(self):
        sa = QScrollArea(); sa.setWidgetResizable(True)
        inner = QWidget(); inner.setObjectName("page"); outer = QHBoxLayout(inner); outer.setContentsMargins(24, 18, 24, 18)
        col = QWidget(); col.setObjectName("page"); col.setMaximumWidth(780)
        v = QVBoxLayout(col); v.setContentsMargins(0, 0, 0, 0); v.setSpacing(14); v.setAlignment(Qt.AlignTop)
        outer.addStretch(1); outer.addWidget(col, 10); outer.addStretch(1)
        sa.setWidget(inner)
        v.addWidget(heading("Settings", 26))

        # camera
        c1 = card(); c1v = QVBoxLayout(c1); c1v.setContentsMargins(18, 14, 18, 14); c1v.setSpacing(8)
        c1v.addWidget(heading("Camera", 19))
        self.cam_status = plain_label("Starting the camera...", wrap=True, style=f"font-size:15px;color:{theme.MUTED};")
        c1v.addWidget(self.cam_status)
        self.src_kind = QComboBox(); self.src_kind.setMinimumHeight(46)
        for data, text in (("webcam", "My webcam"), ("video", "A video file"), ("landmarks", "A practice recording")):
            self.src_kind.addItem(text, data)
        kind, _, path = self.cfg.source.partition(":")
        self.src_kind.setCurrentIndex(max(self.src_kind.findData(kind), 0))
        self.src_path = QLineEdit(path); self.src_path.setMinimumHeight(44); self.src_path.setPlaceholderText("File")
        self.b_browse = btn("Choose a file...", "", 44); self.b_browse.clicked.connect(self._browse)
        self.b_apply = btn("Restart camera", "primary", 48); self.b_apply.clicked.connect(self.restart_worker)
        self.mirror_cb = TickBox("Mirror my camera (like looking in a mirror)")
        self.mirror_cb.setChecked(True); self.mirror_cb.toggled.connect(lambda on: self.set_mirror(on))
        self.file_row = QWidget(); fr = QHBoxLayout(self.file_row); fr.setContentsMargins(0, 0, 0, 0)
        fr.addWidget(self.src_path, 1); fr.addWidget(self.b_browse)
        self.src_box = QWidget(); sb_ = QVBoxLayout(self.src_box); sb_.setContentsMargins(0, 0, 0, 0)   # technical: only with 'Show technical details'
        sb_.addWidget(self.src_kind); sb_.addWidget(self.file_row)
        c1v.addWidget(self.src_box); c1v.addWidget(self.b_apply, 0, Qt.AlignLeft)
        c1v.addWidget(self.mirror_cb)
        c1v.addWidget(hint("Mirror also flips the sign pictures and animations, so you can copy them like in a mirror. "
                           "I always read your real, unflipped hands.", 13))
        self.src_kind.currentIndexChanged.connect(self._src_changed); self._src_changed()
        v.addWidget(c1)

        # how sure
        c2 = card(); c2v = QVBoxLayout(c2); c2v.setContentsMargins(18, 14, 18, 14); c2v.setSpacing(8)
        c2v.addWidget(heading("How sure must I be before writing a word?", 19))
        sr = QHBoxLayout(); sr.setSpacing(8)
        self.sure_group = QButtonGroup(self); self.sure_btns = {}
        for key, text, val, tip in SURE_LEVELS:
            b = btn(text, "seg", 50); b.setCheckable(True); b.setToolTip(tip)
            self.sure_group.addButton(b); self.sure_btns[key] = b; sr.addWidget(b)
            b.clicked.connect(lambda _=False, v_=val: self.thr.setValue(int(round(v_ * 100))))
        c2v.addLayout(sr)
        self.sure_hint = hint("", 14); c2v.addWidget(self.sure_hint)
        c2v.addWidget(plain_label("When I hold my hands still, write the word after...", style="font-size:16px;font-weight:bold;"))
        hr_ = QHBoxLayout(); hr_.setSpacing(8)
        self.hold_group = QButtonGroup(self); self.hold_btns = {}
        cur = float(self.settings.get("hold_commit_s", 1.6))
        near_h = min(HOLD_LEVELS, key=lambda lv: abs(lv[2] - cur))
        for key, text, val, tip in HOLD_LEVELS:
            b = btn(text, "seg", 46); b.setCheckable(True); b.setToolTip(tip); b.setChecked(key == near_h[0])
            self.hold_group.addButton(b); self.hold_btns[key] = b; hr_.addWidget(b)
            b.clicked.connect(lambda _=False, v_=val: self.set_hold_commit(v_))
        c2v.addLayout(hr_)
        self.hold_hint = hint(near_h[3], 14); c2v.addWidget(self.hold_hint)
        self.continuous_sw = Switch(bool(self.settings.get("continuous", True)))
        self.continuous_sw.toggled.connect(self._on_continuous)
        c2v.addWidget(_row("I can sign several signs in a row without lowering my hands",
                           "Words appear while you keep signing. Turn this off if a single sign sometimes comes out as "
                           "two words - then I wait for a short pause or for you to lower your hands between signs.",
                           self.continuous_sw))
        v.addWidget(c2)

        # sending + model
        c3 = card(); c3v = QVBoxLayout(c3); c3v.setContentsMargins(18, 14, 18, 14); c3v.setSpacing(6)
        c3v.addWidget(hint("What happens to a finished message (copy it automatically, send it with the Space bar, or do "
                           "nothing) is chosen with the three buttons under 'My message' on the Talk screen. When I copy a "
                           "message I paste it into the window you have open after a 3 second countdown - so click your "
                           "chat box during the countdown.", 14))
        self.learn_sw = Switch(bool(self.settings.get("learn_corrections", True)))
        self.learn_sw.toggled.connect(self._on_learn_setting)
        c3v.addWidget(_row("Learn from my corrections",
                           "When I am not sure and you tap the right word, I remember how you signed it. "
                           "It stays private on this computer. See them in My signs, under each sign.", self.learn_sw))
        self.use_model_sw = Switch(bool(self.settings.get("use_model", False)))
        self.use_model_sw.toggled.connect(self._on_use_model)
        c3v.addWidget(_row("Use the built-in dictionary model (experimental)",
                           "Lets me also guess signs from the built-in dictionary. It was learned from one person, "
                           "so it often guesses wrong with your webcam. Your own signs work best.", self.use_model_sw))
        self.tech_sw = Switch(bool(self.settings.get("tech", False)))
        self.tech_sw.toggled.connect(self.set_tech)
        c3v.addWidget(_row("Show technical details", "Adds a small panel with speed and accuracy numbers to the Talk screen.",
                           self.tech_sw))
        v.addWidget(c3)

        # finger numbers
        cf = card(); cfv = QVBoxLayout(cf); cfv.setContentsMargins(18, 14, 18, 14); cfv.setSpacing(6)
        cfv.addWidget(heading("Choosing a word with your fingers", 19))
        cfv.addWidget(hint("When I am not sure which word you signed, hold up the number of the button you want (1 = skip, "
                           "2 = retry, 3 and up = the suggested words; keep your hand steady for half a second). If it does "
                           "not work for you, check what I see - or teach me how your fingers look. It only takes about "
                           "half a minute.", 14))
        self.finger_note = plain_label("", style=f"font-size:14px;font-weight:bold;color:{theme.MUTED};")
        cfv.addWidget(self.finger_note)
        fr_ = QHBoxLayout(); fr_.setSpacing(8)
        self.b_finger_check = btn("Check what I see", "", 48); self.b_finger_check.clicked.connect(lambda: self.open_finger_dialog(False))
        self.b_finger_teach = btn("Teach me my fingers", "primary", 48); self.b_finger_teach.clicked.connect(lambda: self.open_finger_dialog(True))
        fr_.addWidget(self.b_finger_check); fr_.addWidget(self.b_finger_teach); fr_.addStretch(1)
        cfv.addLayout(fr_)
        self.finger_answers_sw = Switch(bool(self.settings.get("finger_answers", True)))
        self.finger_answers_sw.toggled.connect(self._on_finger_answers)
        cfv.addWidget(_row("Answer questions with my fingers",
                           "While a question is showing, a hand HELD still with fingers up answers it, and carrying on signing "
                           "skips it. Turn this off if your own signs use finger shapes (number signs): then you answer with a tap "
                           "or a key only, and nothing you sign is ever taken for an answer.", self.finger_answers_sw))
        self._update_finger_note()
        v.addWidget(cf)

        # privacy
        c4 = card(); c4v = QVBoxLayout(c4); c4v.setContentsMargins(18, 14, 18, 14); c4v.setSpacing(6)
        c4v.addWidget(heading("Your privacy", 19))
        c4v.addWidget(hint("Your signs are kept only on this computer, locked with encryption. I save only the position of "
                           "your hands and body - never video or pictures of you.", 15))
        self.storage_note = plain_label(self.store_error or "Safe storage is working.", wrap=True,
                                        style=f"font-size:14px;color:{theme.BAD if self.store_error else theme.GOOD};")
        c4v.addWidget(self.storage_note)
        v.addWidget(c4)

        # advanced
        self.advanced = Collapsible("Advanced")
        al = self.advanced.body_layout
        self.thr = QSlider(Qt.Horizontal); self.thr.setRange(50, 99)
        self.thr.setValue(int(round(self.settings.get("threshold", self.cfg.threshold) * 100)))
        self.thr_lbl = plain_label("", style=MUTED_CSS)
        self.thr.valueChanged.connect(self._on_threshold); self._on_threshold(self.thr.value())
        self.pause = TickBox("Pause sending (messages are only shown, not pasted)"); self.pause.toggled.connect(self._on_pause)
        al.addWidget(plain_label("Exact certainty needed", style=MUTED_CSS)); al.addWidget(self.thr); al.addWidget(self.thr_lbl)
        al.addWidget(self.pause)
        v.addWidget(self.advanced)
        return sa

    # ------------------------------------------------------------------ navigation
    def show_section(self, name):
        self.section = name
        if hasattr(self, "space_shortcut"):
            self._update_space()
        self.nav[name].setChecked(True)
        if name == "settings":
            self.pages.setCurrentIndex(1)
            return
        self.pages.setCurrentIndex(0)
        self.panels.setCurrentWidget({"talk": self.talk_panel, "signs": self.lib_tab, "teach": self.new_tab}[name])
        self.talk_strip.setVisible(name == "talk")
        if name == "talk" and self.stage.mode == "playback":
            self.stage.go_live()
        if name == "signs":
            self.lib_tab._vis.start()
        if name == "talk":
            self._update_empty()

    def show_help(self):
        self.show_section("talk")
        self.help_card.show(); self._update_empty()
        self.talk_panel.verticalScrollBar().setValue(0)

    def dismiss_help(self):
        self.help_card.hide(); self._save_setting("seen_help", True); self._update_empty()

    def _on_pending(self, n):
        self.nav["teach"].set_badge(n)

    # ------------------------------------------------------------------ mirror (display only)
    def set_mirror(self, on, save=True):
        on = bool(on); self.mirror = on
        self.stage.set_mirror(on); self.lib_tab.set_mirror(on); self.new_tab.set_mirror(on)
        self.mirror_cb.blockSignals(True); self.mirror_cb.setChecked(on); self.mirror_cb.blockSignals(False)
        if save:
            self._save_setting("mirror", on)

    # ------------------------------------------------------------------ settings handlers
    def _src_changed(self, *_):
        self.file_row.setVisible(self.src_kind.currentData() != "webcam")
        self._update_src_box()

    def _update_src_box(self):
        # the practice-recording / file chooser is a debugging aid: shown only with technical details
        # (or when a non-webcam source was explicitly started, so the user can see what is running)
        self.src_box.setVisible(bool(self.settings.get("tech", False)) or self.src_kind.currentData() != "webcam")

    def _initial_send_policy(self):
        pol = self.settings.get("send_policy")
        if pol in ("off", "auto", "manual"):
            return pol
        if "auto_send" in self.settings:                       # saved by an older version: a plain on/off switch
            return "auto" if self.settings["auto_send"] else "off"
        return "off" if self.cfg.send_mode == "none" else "auto"

    def send_mode_value(self):
        """The worker's send mode: nothing is copied when 'Don't copy' is chosen."""
        return "none" if self.send_policy == "off" else "clipboard"

    def set_send_policy(self, policy):
        """'Don't copy' | 'Copy automatically' | 'Send with Space' (saved, applied now)."""
        if policy not in ("off", "auto", "manual"):
            return
        self.send_policy = policy
        for k, b in self.policy_btns.items():
            b.setChecked(k == policy)
        self._save_setting("send_policy", policy)
        self.cfg.send_mode = self.send_mode_value()
        self._push_send_policy()
        self._update_send_hint(); self._update_space(); self._refresh_sentence()

    def _push_send_policy(self):
        w = self.worker
        if w is None:
            return
        if hasattr(w, "send_mode"):
            w.send_mode = self.send_mode_value()
        if hasattr(w, "set_auto_finish"):
            w.set_auto_finish(self.send_policy != "manual")      # 'Send with Space': lowering the hands does not end it

    def _update_send_hint(self):
        text = {"off": "Copying is off: your message is only shown here.",
                "auto": "It is copied when you lower your hands (or press Send now).",
                "manual": ("Press the Space bar to send it. It keeps growing until you do." if self.words
                           else "Sign your message, then press the Space bar to send it.")}[self.send_policy]
        self.send_hint.setText(text)

    def _update_space(self):
        self.space_shortcut.setEnabled(self.send_policy == "manual" and getattr(self, "section", "talk") == "talk")

    def _on_space(self):
        if self.send_policy == "manual" and self.section == "talk" and self.words:
            self.send_now()

    def _on_use_model(self, on):
        self._save_setting("use_model", bool(on))
        self._push_use_model()
        self._update_empty()

    def _push_use_model(self):
        if self.worker is not None and hasattr(self.worker, "set_use_model"):
            self.worker.set_use_model(self.use_model_sw.isChecked())

    def _on_learn_setting(self, on):
        self._save_setting("learn_corrections", bool(on))
        self._push_learning()
        if not on:
            self._hide_learn_bar()

    def _push_learning(self):
        if self.worker is not None and hasattr(self.worker, "set_learning"):
            self.worker.set_learning(self.learn_sw.isChecked())

    # ---- learning from corrections
    def _on_learn_ready(self, hint):
        """A confirmed answer passed the engine's quality gate: keep it as another example (UI thread)."""
        if not self.learn_sw.isChecked() or not self.editable:
            return
        id_, label = hint.get("sign_id"), str(hint.get("label", ""))
        try:
            if not self.library.is_active(id_):
                return                                   # switched off meanwhile: teaches nothing
            n = self.library.add_auto_take(id_, hint["window"], fps=float(hint.get("fps") or 25.0))
        except (LibraryError, KeyError, ValueError):
            return
        except Exception:
            self._say("I could not save that example.")
            return
        self._learned = (id_, n, label)
        self._push_templates_now()                       # ONE refresh of what I look for
        self.lib_tab.invalidate(id_); self.lib_tab.reload()
        self.learn_lbl.setText(f"Saved. I will recognise {label} better now.")
        self.learn_bar.show(); self._learn_timer.start(15000)

    def _on_learn_skipped(self, reason):
        if reason == "movement too different":
            self._say("Not saved as an example: that movement was too different from this sign.")

    def _hide_learn_bar(self):
        self._learned = None
        self.learn_bar.hide(); self._learn_timer.stop()

    def _undo_learn(self):
        if self._learned is not None:
            id_, n, _ = self._learned
            try:
                self.library.remove_take(id_, n)
            except Exception:
                pass                                      # already gone (e.g. removed from My signs)
            self._push_templates_now()
            self.lib_tab.invalidate(id_); self.lib_tab.reload()
        self._hide_learn_bar()

    def _on_continuous(self, on):
        self._save_setting("continuous", bool(on))
        self._push_continuous()

    @staticmethod
    def _valid_cal(cal):
        """Taught finger poses as saved: {'1': [5 numbers], ...}; anything else is ignored."""
        try:
            cal = dict(cal)
            if int(cal.get("ver", 0)) != CAL_VERSION:           # measured the old (flat) way: not comparable, ignore
                return None
            ok = {str(k): [float(x) for x in v] for k, v in cal.items() if str(k).isdigit() and len(v) == 5}
        except Exception:
            return None
        return dict(ok, ver=CAL_VERSION) if ok else None

    def _on_finger_answers(self, on):
        self._save_setting("finger_answers", bool(on))
        self.clarify.set_fingers_enabled(bool(on))
        self._push_finger_answers()

    def _push_finger_answers(self):
        if self.worker is not None and hasattr(self.worker, "set_finger_answers"):
            self.worker.set_finger_answers(self.finger_answers_sw.isChecked())

    def _push_finger_cal(self):
        if self.worker is not None and hasattr(self.worker, "set_finger_cal"):
            self.worker.set_finger_cal(self.finger_cal)

    def set_finger_cal(self, cal):
        """Save (None = back to the standard way) and apply the taught finger poses."""
        self.finger_cal = self._valid_cal(cal)
        self._save_setting("finger_cal", self.finger_cal)
        self._push_finger_cal()
        self._update_finger_note()

    def _update_finger_note(self):
        self.finger_note.setText("I am using your own finger poses." if self.finger_cal
                                 else "I am using the standard way of counting fingers.")

    def open_finger_dialog(self, teach=False):
        self._finger_dlg = FingerDialog(self, teach=teach)
        self._finger_dlg.exec()
        self._finger_dlg = None

    def set_hold_commit(self, seconds):
        """How long holding still ends the word before it (saved, applied now)."""
        near = min(HOLD_LEVELS, key=lambda lv: abs(lv[2] - float(seconds)))
        for key, b in self.hold_btns.items():
            b.setChecked(key == near[0])
        self.hold_hint.setText(near[3])
        self._save_setting("hold_commit_s", float(seconds))
        self._push_hold_commit()

    def _push_hold_commit(self):
        if self.worker is not None and hasattr(self.worker, "set_hold_commit"):
            self.worker.set_hold_commit(float(self.settings.get("hold_commit_s", 1.6)))

    def _push_continuous(self):
        if self.worker is not None and hasattr(self.worker, "set_continuous"):
            self.worker.set_continuous(self.continuous_sw.isChecked())

    def set_tech(self, on):
        on = bool(on)
        self.tech.setVisible(on)
        self.tech_sw.blockSignals(True); self.tech_sw.setChecked(on); self.tech_sw.blockSignals(False)
        self._save_setting("tech", on)
        self._update_src_box()

    def _on_threshold(self, v):
        self.thr_lbl.setText(f"{v / 100:.2f}")
        near = min(SURE_LEVELS, key=lambda lv: abs(lv[2] * 100 - v))
        for key, b in self.sure_btns.items() if hasattr(self, "sure_btns") else ():
            b.setChecked(key == near[0])
        if hasattr(self, "sure_hint"):
            self.sure_hint.setText(near[3])
        if self.worker: self.worker.set_threshold(v / 100)
        if hasattr(self, "thr") and self.settings.get("threshold") != v / 100 and hasattr(self, "advanced"):
            self._save_setting("threshold", v / 100)

    def _on_pause(self, on):
        if self.worker: self.worker.paused = on

    # ------------------------------------------------------------------ friendly Talk state
    def set_pill(self, kind, text=None, tech_btn=False):
        """The one line that always says, in plain words, what is happening."""
        texts = {"ready": "Ready - raise your hands to sign", "watching": "Watching...", "working": "..."}
        self.pill_state = kind
        bg, fg = PILL[kind]
        self.pill.setText(text or texts.get(kind, ""))
        self.pill.setStyleSheet(f"background:{bg};color:{fg};font-size:24px;font-weight:bold;border-radius:16px;padding:8px 18px;")
        self.pill_btn.setVisible(tech_btn)
        if kind not in ("got", "notsure"):
            self.dots_box.hide()

    def _pill_back(self):
        if self.clarify.isVisible() or self.pill_state == "error":
            return
        self.set_pill("working" if self._signing else "ready")

    def _set_dots(self, conf, color=None):
        c = float(conf or 0)
        thr = self.thr.value() / 100
        lvl = 3 if c >= max(0.9, thr) else 2 if c >= thr else 1
        self.dots.set_level(lvl, color or (theme.GOOD if lvl >= 2 else theme.WARN)); self.dots_box.show()
        self.sure_word.setText({3: "Very sure", 2: "Sure", 1: "Not very sure"}[lvl])
        self.dots.setToolTip("How sure I am")

    def _n_in_use(self):
        return self.lib_tab.in_use_count()

    MANY_SIGNS = 150                      # more switched-on signs than this: warn (slower, more mix-ups)

    def _show_switched_on(self):
        self.show_section("signs"); self.lib_tab.set_chip("inuse")

    def _update_many(self):
        n = self._n_in_use()
        self.many_card.setVisible(n > self.MANY_SIGNS)
        if n > self.MANY_SIGNS:
            self.many_text.setText(f"{n:,} signs are switched on. That makes me slower and more likely to mix signs up - "
                                   "the words can take several seconds to appear. Signs you recorded yourself work best, so "
                                   "switch off the ones you do not use.")

    def _update_empty(self):
        """'Teach your first sign' call to action - shown after the intro card is dismissed."""
        self._update_many()
        self.empty_card.setVisible(self._n_in_use() == 0 and not self.use_model_sw.isChecked() and self.editable
                                   and not self.help_card.isVisibleTo(self.talk_panel))

    def _refresh_sentence(self):
        text = " ".join(self.words)
        if text:
            self.caption.setText(text); auto_direction(self.caption, text)
            self.caption.setStyleSheet(f"font-size:36px;font-weight:bold;color:{theme.TEXT};")
        else:
            self.caption.setText("Your message will appear here as you sign.")
            self.caption.setLayoutDirection(Qt.LeftToRight)
            self.caption.setStyleSheet("font-size:28px;font-weight:bold;color:#94a3b8;")
        has = bool(self.words)
        for b in (self.b_send, self.b_undo, self.b_clear):
            b.setEnabled(has)
        if hasattr(self, "send_hint"):
            self._update_send_hint()
            self.b_send.setText("Send now (Space)" if self.send_policy == "manual" else "Send now")
        self.b_send.setToolTip("" if has else "Sign something first")

    def send_now(self):
        if self.worker is not None and self.words:
            self.worker.flush()

    def undo_word(self):
        if self.words:
            if self._hole == len(self.words) - 1:
                self._hole = self._answering = None
            self.words.pop()
            if self.worker is not None and hasattr(self.worker, "undo_last_word"):
                self.worker.undo_last_word()
            self._refresh_sentence()

    def clear_sentence(self):
        self.words.clear(); self._hole = self._answering = None
        if self.worker is not None and hasattr(self.worker, "clear_sentence"):
            self.worker.clear_sentence()
        self._refresh_sentence()
        self.clarify.hide(); self.set_pill("ready")

    # ------------------------------------------------------------------ worker control
    def start_worker(self):
        if self.worker is not None:
            return
        spec = self._source_spec()
        factory = self.worker_factory
        if factory is None:
            from .worker import RecognitionWorker
            factory = RecognitionWorker
        self.worker = w = factory(spec, self.cfg.model, threshold=self.thr.value() / 100,
                                  send_mode=self.send_mode_value(), loop=self.cfg.loop, store=self.store)
        w.paused = self.pause.isChecked()
        w.preview.connect(self.on_preview); w.events.connect(self.on_events); w.stats.connect(self.on_stats)
        w.sent.connect(self.on_sent); w.failed.connect(self.on_failed)
        if hasattr(w, "countdown"):
            w.countdown.connect(self.on_countdown)
        w.unknown_saved.connect(lambda _id: self.new_tab.refresh())
        for sig, slot in (("learn_ready", self._on_learn_ready), ("learn_skipped", self._on_learn_skipped)):
            if hasattr(w, sig):
                getattr(w, sig).connect(slot)
        for sig, slot in (("capture_state", self._on_capture_state), ("capture_result", self._on_capture_result)):
            if hasattr(w, sig):
                getattr(w, sig).connect(slot)
        self._push_label_map(); self._push_templates_now(); self._update_empty()
        self._push_use_model()
        self._push_continuous()
        self._push_learning()
        self._push_finger_cal()
        self._push_finger_answers()
        self._push_hold_commit()
        self._push_send_policy()
        w.start()
        self.set_pill("ready")

    def stop_worker(self):
        if self.worker is not None:
            self.worker.stop(); self.worker.wait(3000); self.worker = None

    def restart_worker(self):
        self.stop_worker()
        self._cam_ok = False
        self.cam_status.setText("Starting the camera..."); self.cam_status.setStyleSheet(f"font-size:15px;color:{theme.MUTED};")
        if self._capture is not None:
            self._capture = None; self.stage.go_live()
        self.cfg.send_mode = self.send_mode_value()
        self.set_pill("ready")
        self.start_worker()

    def closeEvent(self, e):
        self.stop_worker(); super().closeEvent(e)

    def _source_spec(self):
        k = self.src_kind.currentData()
        return k if k == "webcam" else f"{k}:{self.src_path.text()}"

    def _browse(self):
        p, _ = QFileDialog.getOpenFileName(self, "Choose a file")
        if p:
            self.src_path.setText(p)

    def _push_label_map(self):
        """Edited output texts reach the recogniser (worker thread) as a plain dict via its queue."""
        if self.worker is not None and hasattr(self.worker, "set_label_map"):
            self.worker.set_label_map(self.library.label_map())

    def _push_templates(self):
        self._push_templates_now()

    def _push_templates_now(self):
        """The signs switched on for conversation reach the worker as plain data (raw, un-mirrored frames).
        A burst of changes is coalesced in the worker (only the newest list is applied)."""
        if self.worker is not None and hasattr(self.worker, "set_templates") and hasattr(self.library, "templates"):
            try:
                self.worker.set_templates(self.library.templates())
            except Exception as ex:
                self.status.setText(f"I could not update the list of signs I look for ({ex})"); self.status.show()

    def _check_learn_bar(self):
        """The 'Learned from you ... Undo' bar disappears when that example was deleted (e.g. in 'My recordings')."""
        if self._learned is not None:
            try:
                gone = self._learned[1] not in self.library.take_numbers(self._learned[0])
            except KeyError:
                gone = True
            if gone:
                self._hide_learn_bar()

    def _push_all(self):
        self._check_learn_bar()
        self._push_label_map(); self._push_templates(); self._update_empty()

    def _on_library_changed(self):
        self._push_all()
        self.lib_tab.reload()

    # ------------------------------------------------------------------ slots
    def on_preview(self, frame, lm):
        if not getattr(self, "_cam_ok", False):
            self._cam_ok = True
            self.cam_status.setText("Camera working." if self.src_kind.currentData() == "webcam"
                                    else "Playing your chosen recording.")
            self.cam_status.setStyleSheet(f"font-size:15px;font-weight:bold;color:{theme.GOOD};")
        if frame is not None and getattr(frame, "shape", None) is not None and frame.shape[0]:
            self._aspect = frame.shape[1] / frame.shape[0]
        if self._finger_dlg is not None and self._finger_dlg.isVisible():
            self._finger_dlg.feed(lm, self._aspect)                    # raw (un-mirrored) landmarks: what the recogniser sees
        if self.stage.camera_visible:          # no point scaling frames nobody sees
            self.camera.update_view(frame, lm)          # mirroring happens inside, on a copy
        if lm is not None and self.pill_state in ("ready", "watching", "working") and not self._pill_timer.isActive():
            seen = bool(np.any(~np.isnan(np.asarray(lm)[25:, 0])))
            want = "working" if self._signing else ("watching" if seen else "ready")
            if want != self.pill_state:
                self.set_pill(want)

    def on_stats(self, s):
        self._signing = bool(s.get("signing", False))
        if self.clarify.isVisible():
            f = s.get("finger") or (0, 0.0)
            self.clarify.set_finger(int(f[0]), float(f[1]), seen=s.get("finger_seen"),
                                    hand=bool(s.get("finger_hand")), options=int(s.get("finger_options") or 0))
        if self._signing and self.pill_state in ("ready", "watching") and not self._pill_timer.isActive():
            self.set_pill("working")                      # '...': you are signing and the next word is being worked out
        elif not self._signing and self.pill_state == "working" and not self._pill_timer.isActive():
            self.set_pill("ready")
        self.fps_lbl.setText(f"{s.get('fps', 0):.1f}")
        self.lat_lbl.setText(f"{s.get('latency_ms_avg', 0):.0f} ms")
        self.scale_lbl.setText(str(s.get("scale_used", "-")))
        self.skip_lbl.setText(str(s.get("frames_skipped", 0)))

    @staticmethod
    def _names(cands):
        return [str(c[0]) if isinstance(c, (tuple, list)) else str(c) for c in cands]

    def on_events(self, evs):
        for e in evs:
            if e.kind == "sign":
                if self._hole is not None and (self._answering or getattr(e, "answered", False)):
                    # the answer to the open question (tapped OR shown with fingers) takes the question's place
                    self.words[self._hole] = e.label
                    self._hole = self._answering = None
                    self.clarify.hide()
                elif self._hole is None:
                    self.clarify.hide()      # (a question that is still open keeps its card while signing goes on)
                    self.words.append(e.label)
                else:
                    self.words.append(e.label)
                self.set_pill("got", str(e.label)); self._set_dots(e.confidence)   # the word, big and blue, as soon as it is known
                self._pill_timer.start(1400)                                         # then back to '...' while the signing goes on
                self.conf_lbl.setText(f"{e.confidence:.0%}")
                self.src_lbl.setText({"personal": "my own signs", "model": "built-in model"}.get(
                    getattr(e, "source", None), "-"))
            elif e.kind == "clarify":
                names = self._names(e.candidates)[:2] or ([str(e.label)] if e.label else [])
                self.clarify.show_candidates(e.candidates)
                if self._hole is not None and self._hole < len(self.words):
                    self.words.pop(self._hole)       # a newer question replaces an unanswered one
                self._hole = None
                if self.continuous_sw.isChecked() and getattr(e, "source", None) == "personal":
                    self.words.append("..."); self._hole = len(self.words) - 1
                    h = self._hole
                    QTimer.singleShot(10500, lambda h=h: self._expire_hole(h))
                self.set_pill("notsure", "Not sure - did you mean " + " or ".join(names) + "?")
                self._set_dots(0, theme.WARN); self.dots.set_level(1, theme.WARN)
                self.conf_lbl.setText(f"{e.confidence:.0%}")
            elif e.kind == "skip":
                # answered '1' with the fingers: this sign is dropped, carry on with the next one
                self._drop_question()
                self.set_pill("ready", "No answer - I left that word out" if e.label == "timeout"
                              else "Skipped - carry on with the next sign"); self._pill_timer.start(1800)
            elif e.kind == "retry":
                # answered '2' with the fingers: the question and its placeholder go away, sign the word again
                if self._hole is not None and self._hole < len(self.words):
                    self.words.pop(self._hole)
                self._hole = self._answering = None
                self.clarify.hide()
                self.set_pill("ready", "OK - please sign it again"); self._pill_timer.start(2500)
            elif e.kind == "unknown":
                self.set_pill("unknown", "I don't know this sign yet - teach it?", tech_btn=True)
                self._pill_timer.start(6000)
                self.conf_lbl.setText(f"{e.confidence:.0%}")
            elif e.kind == "sentence":
                self.words.clear(); self._hole = self._answering = None
                self.clarify.hide()
                self._add_history(e.label)
            if e.latency_ms:
                self.lat_lbl.setText(f"{e.latency_ms:.0f} ms")
        self._refresh_sentence()

    # ---- the last message sent (only the last one is shown)
    def _add_history(self, text):
        self.last_msg = [text, "Sending...", None]
        self._show_last()

    def _show_last(self, detail=""):
        text, status, ok = self.last_msg
        self.last_empty.hide(); self.last_text.show(); self.last_status.show()
        self.last_text.setText(text); auto_direction(self.last_text, text)
        mark = "..." if ok is None else ("\u2713" if ok else "\u2717")
        self.last_status.setText(f"{mark}  {status}")
        color = theme.MUTED if ok is None else theme.GOOD if ok else theme.WARN
        self.last_status.setStyleSheet(f"font-size:16px;font-weight:bold;color:{color};")
        self.last_status.setToolTip("<p>" + html.escape(detail or "") + "</p>" if detail else "")

    def on_sent(self, text, ok, detail):
        if self.last_msg is None or self.last_msg[0] != text or self.last_msg[2] is not None:
            return
        if ok:
            status = "Sent"
        else:
            d = (detail or "").lower()
            status = ("Not sent - sending is paused" if "paused" in d else
                      "Not copied - copying is off" if "send mode" in d else
                      "Cancelled - copied, not pasted" if "cancelled" in d else "Not sent")
        self.last_msg[1] = status; self.last_msg[2] = bool(ok)
        self._show_last(detail)

    def on_countdown(self, left):
        """Visible paste countdown with a Cancel button (the worker re-checks Pause / auto-send too)."""
        if left > 0:
            self.send_note.setText(f"Pasting in {left:.0f} s - click your chat box now. Press Cancel to stop.")
            self.send_note_row.show()
        else:
            self.send_note_row.hide()

    def _cancel_send(self):
        if self.worker is not None and hasattr(self.worker, "cancel_send"):
            self.worker.cancel_send()
        self.send_note_row.hide()

    def on_failed(self, msg):
        low = (msg or "").lower()
        if any(k in low for k in ("camera", "webcam", "video", "capture", "open", "device")):
            text = "I can't find your camera. Check that it is plugged in and not used by another app, then restart it in Settings."
        else:
            text = "Something stopped working. Open Settings and press 'Restart camera'."
        self._pill_timer.stop()
        self._cam_ok = False
        self.cam_status.setText("No camera found - plug it in, then press Restart camera." if "camera" in text
                                else "Something stopped working - press Restart camera.")
        self.cam_status.setStyleSheet(f"font-size:15px;font-weight:bold;color:{theme.BAD};")
        self.set_pill("error", text)
        self.status.setText(msg); self.status.setStyleSheet(f"color:{theme.BAD};font-size:14px;")
        self.status.setVisible(bool(self.settings.get("tech", False)))

    def _expire_hole(self, h):
        """The engine drops an unanswered question after 10 s: so does the caption."""
        if self._hole == h and self._answering is None and h < len(self.words) and self.words[h] == "...":
            self.words.pop(h); self._hole = None; self.clarify.hide(); self._refresh_sentence()

    def _on_confirm(self, label):
        self._answering = label
        if self.worker: self.worker.confirm(label)
        self.set_pill("watching")

    def _drop_question(self):
        """The open question is gone (skip / retry): its placeholder leaves the message."""
        if self._hole is not None and self._hole < len(self.words):
            self.words.pop(self._hole)
        self._hole = self._answering = None
        self.clarify.hide()

    def _on_skip(self):
        self._drop_question(); self._refresh_sentence()
        if self.worker: self.worker.skip() if hasattr(self.worker, "skip") else self.worker.reject()
        self.set_pill("ready", "Skipped - carry on with the next sign"); self._pill_timer.start(1800)

    def _on_repeat(self):
        if self._hole is not None and self._hole < len(self.words):
            self.words.pop(self._hole); self._refresh_sentence()
        self._hole = self._answering = None
        if self.worker: self.worker.reject()
        self.set_pill("ready", "OK - please sign it again")

    def _say(self, text):
        """One quiet line of feedback under the camera."""
        self.status.setText(text); self.status.setStyleSheet(f"color:{theme.MUTED};font-size:14px;")
        self.status.setVisible(bool(text))

    # ------------------------------------------------------------------ playback in the main view
    def _busy_teaching(self):
        """True while a captured clip awaits Keep/Retake/Discard (never throw it away by accident)."""
        if self._capture is not None and self.stage.mode == "review":
            self._say("Keep, try again or delete the new recording first.")
            return True
        return False

    def _segment(self, id_, take=0):
        e = self.library.get(id_)
        return Segment(e.label, self.library.animation(id_, take), "" if e.category == "My signs" else e.category), e

    def play_entry(self, id_, take=0):
        if self._busy_teaching():
            return
        try:
            nums = self.library.take_numbers(id_)                 # example NUMBERS can have gaps after a deletion
            number = take if take in nums else nums[0]
            seg, e = self._segment(id_, number)
        except Exception as ex:
            self._say(f"I have no picture for this sign ({ex})"); return
        self._abort_capture()
        self._stage_entry = id_
        self.stage.play([seg], loop=True, takes=len(nums), take=nums.index(number))

    def play_phrase(self, ids, missing=()):
        if self._busy_teaching():
            return
        segs, missing = [], list(missing)
        for i in ids:
            try:
                segs.append(self._segment(i)[0])
            except Exception:
                try:
                    missing.append(self.library.get(i).label + " (no picture)")
                except KeyError:
                    pass
        note = ("No sign found for: " + ", ".join(missing)) if missing else ""
        if note:
            for s in segs:
                s.subtitle = note
            self._say(note)
        if not segs:
            return
        self._abort_capture()
        self._stage_entry = None
        self.stage.play(segs, loop=False)

    def _play_array(self, title, frames):
        if self._busy_teaching() or frames is None or not len(frames):
            return
        self._abort_capture(); self._stage_entry = None
        self.stage.play([Segment(title, frames, "A movement I saw but did not know")], loop=True)

    def _on_take(self, i):
        if self._stage_entry is None:
            return
        try:
            nums = self.library.take_numbers(self._stage_entry)    # the selector counts positions; examples have numbers
            seg, _ = self._segment(self._stage_entry, nums[i])
        except Exception as ex:
            self._say(f"Example {i + 1} is not available ({ex})"); return
        self.stage.replace_take(seg, i)

    def _on_escape(self):
        """Esc: leave the animation view / cancel recording."""
        if self.stage.mode == "review" and self.stage.name_edit.hasFocus():
            self.stage.name_edit.clearFocus()       # Esc while typing a name only leaves the field
            return
        if self.stage.mode != "live":
            self.stage.go_live()

    def _on_stage_closed(self):
        self._abort_capture()

    # ------------------------------------------------------------------ teach-a-sign flow
    def _shortcut_teach(self):
        if self.stage.mode in ("live", "playback"):
            self.begin_capture("new", None, self.new_tab.name.text().strip())

    def _on_record_requested(self, mode, id_):
        self.begin_capture(mode, id_, "")

    def _abort_capture(self):
        if self._capture is not None:
            self._capture = None
            if self.worker is not None and hasattr(self.worker, "cancel_capture"):
                self.worker.cancel_capture()

    def begin_capture(self, mode="new", id_=None, name=""):
        w = self.worker
        if not self.editable:
            self.new_tab.say("I can't save signs right now because the safe storage could not be opened.", err=True); return
        if w is not None and hasattr(w, "isRunning") and not w.isRunning():
            self.new_tab.say("The camera has stopped. Open Settings and press 'Restart camera' first.", err=True)
            self._say("Cannot record: the camera has stopped."); return
        if w is None or not hasattr(w, "start_capture"):
            self.new_tab.say("The camera is not running. Open Settings and press 'Restart camera' first.", err=True)
            self._say("Cannot record: the camera is not running."); return
        self._capture = dict(mode=mode, id=id_, name=name, clip=None, fps=25.0)
        if self.section in ("settings", "talk"):
            self.show_section("teach")
        self.stage.show_capture()
        self.clarify.hide()
        w.start_capture()
        self._capture["sid"] = getattr(w, "capture_gen", None)
        self.new_tab.say("")

    def _on_capture_state(self, st):
        if self._capture is None or self.stage.mode != "capture":
            return
        if st.state == "countdown":
            self.stage.set_overlay(str(max(1, math.ceil(st.seconds_left))), "Get ready...", "#fde047")
        elif st.state == "waiting":
            self.stage.set_overlay("Sign now", "Raise your hands and sign", "#86efac", compact=True)
        elif st.state == "recording":
            self.stage.set_overlay("Recording", "Lower your hands when you finish", "#fca5a5", compact=True)

    def _on_capture_result(self, st, fps, sid=None):
        if self._capture is None or self.stage.mode != "capture":
            return
        if sid is not None and self._capture.get("sid") not in (None, sid):
            return                      # stale result of an aborted capture
        if st.state == "done" and st.clip is not None:
            self._capture.update(clip=st.clip, fps=float(fps))
            try:
                anim = prepare_animation(st.clip, float(fps))
            except Exception:
                anim = np.asarray(st.clip, np.float32)
            mode = self._capture["mode"]
            title = "Check your recording"
            if mode != "new":
                try:
                    title = ("Recording again: " if mode == "replace" else "Another example of: ") + self.library.get(self._capture["id"]).label
                except KeyError:
                    pass
            self.stage.show_review(anim, title, self._capture["name"], need_name=(mode == "new"),
                                   subtitle="Watch it repeat. Does it look like your sign?")
        else:
            self.stage.set_overlay("Oops", st.message or "That did not work. Please try again.", "#fca5a5", retry=True)

    def _capture_retake(self):
        if self._capture is not None and self.worker is not None:
            self._capture["clip"] = None
            self.stage.show_capture()
            self.worker.start_capture()

    def _capture_discard(self):
        self._capture = None
        self.stage.go_live()
        self.new_tab.say("Deleted the recording.")

    def _capture_keep(self, name):
        c = self._capture
        if c is None or c.get("clip") is None:
            return
        lib = self.library
        try:
            if c["mode"] == "new":
                if not name:
                    self.stage.set_review_message("Please type the word for this sign first."); return
                id_ = lib.add_recorded(name, c["clip"], fps=c["fps"])
                msg = f"Saved '{name}' as a new sign."
            elif c["mode"] == "replace":
                id_ = c["id"]; lib.replace_animation(id_, c["clip"], fps=c["fps"])
                msg = "Saved your new recording of this sign."
            else:
                id_ = c["id"]; n = lib.add_take(id_, c["clip"], fps=c["fps"])
                msg = f"Saved example {n + 1}."
        except LibraryError as ex:
            self.stage.set_review_message(str(ex)); return
        except Exception as ex:
            self.stage.set_review_message(f"Could not save: {ex}"); return
        self._capture = None
        self.stage.go_live()
        self.lib_tab.invalidate(id_)
        self._on_library_changed()
        self.new_tab.say(msg)
        self.new_tab.show_saved(id_)
        if c["mode"] == "new":
            self.show_section("teach")
