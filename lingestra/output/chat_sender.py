"""Deliver a finished sentence to the signer's own meeting chat.

Default 'clipboard' mode: copy to the OS clipboard, then (optionally) simulate
paste + Enter in whichever window has focus. If keystroke injection is not
possible (Wayland, macOS without Accessibility permission, headless) we still
succeed with method='clipboard-only' and the signer presses Ctrl/Cmd+V.

'playwright' mode is EXPERIMENTAL and has NOT been tested against a live call.
Message text is never logged at INFO level (only at DEBUG, and not even there:
we log lengths only).
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

log = logging.getLogger(__name__)

# --- Playwright selectors, all in one place (accessible role/name, never jsname) ---
PLAYWRIGHT_SELECTORS = {
    "chat_input": {"role": "textbox", "name": "Send a message"},
    "send_button": {"role": "button", "name": "Send a message"},
    "open_chat_button": {"role": "button", "name": "Chat with everyone"},
}


@dataclass
class SendResult:
    ok: bool
    method: str   # 'clipboard+paste' | 'clipboard-only' | 'playwright' | 'none'
    detail: str


def app_data_dir() -> Path:
    """Per-user app data dir on Windows/macOS/Linux."""
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share"))
    return base / "Lingestra"


# ---------------------------------------------------------------- clipboard
class SystemClipboard:
    """pyperclip first, then native tools (wl-copy, xclip, xsel, pbcopy, clip)."""

    def copy(self, text: str) -> bool:
        try:
            import pyperclip
            pyperclip.copy(text)
            return True
        except Exception:  # missing backend raises PyperclipException
            pass
        for cmd in self._commands():
            try:
                subprocess.run(cmd, input=text.encode("utf-8"), check=True,
                               timeout=3, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL)
                return True
            except Exception:
                continue
        return False

    @staticmethod
    def _commands():
        if sys.platform == "darwin":
            cands = [["pbcopy"]]
        elif sys.platform.startswith("win"):
            cands = [["clip"]]
        else:
            cands = [["wl-copy"], ["xclip", "-selection", "clipboard"],
                     ["xsel", "--clipboard", "--input"]]
        return [c for c in cands if shutil.which(c[0])]


class PynputKeyboard:
    """Paste (+Enter) via pynput. Raises on any failure so caller can degrade."""

    def paste(self, press_enter: bool) -> None:
        # Wayland sessions ignore synthetic X11 events for native windows.
        if sys.platform.startswith("linux") and (
                os.environ.get("XDG_SESSION_TYPE") == "wayland"
                or os.environ.get("WAYLAND_DISPLAY")):
            raise RuntimeError("keystroke injection unavailable on Wayland")
        if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
            raise RuntimeError("no display")
        from pynput.keyboard import Controller, Key  # lazy: fails when headless
        kb = Controller()
        mod = Key.cmd if sys.platform == "darwin" else Key.ctrl
        with kb.pressed(mod):
            kb.press("v")
            kb.release("v")
        if press_enter:
            time.sleep(0.05)
            kb.press(Key.enter)
            kb.release(Key.enter)


# ------------------------------------------------------------------ sender
class ChatSender:
    def __init__(self, mode: str = "clipboard", press_enter: bool = True,
                 countdown_s: float = 0, clipboard=None, keyboard=None,
                 playwright_driver=None, sleep=time.sleep):
        """clipboard/keyboard/playwright_driver/sleep are injectable for tests."""
        if mode not in ("clipboard", "playwright"):
            raise ValueError(f"unknown mode {mode!r}")
        self.mode = mode
        self.press_enter = press_enter
        self.countdown_s = countdown_s
        self._clipboard = clipboard or SystemClipboard()
        self._keyboard = keyboard or PynputKeyboard()
        self._pw = playwright_driver  # created lazily in playwright mode
        self._sleep = sleep
        self.should_abort = None     # callable() -> bool, checked during / after the countdown
        self.on_tick = None          # callable(seconds_left) during the countdown

    def available_modes(self) -> list[str]:
        modes = ["clipboard"]
        try:
            import playwright  # noqa: F401
            modes.append("playwright")
        except Exception:
            pass
        return modes

    def send(self, text: str) -> SendResult:
        text = (text or "").strip()
        if not text:
            return SendResult(False, "none", "empty text")
        log.info("sending message (%d chars) via %s", len(text), self.mode)
        if self.mode == "playwright":
            res = self._send_playwright(text)
            if res.ok:
                return res
            log.warning("playwright failed (%s); falling back to clipboard", res.detail)
        return self._send_clipboard(text)

    # -- clipboard
    def _send_clipboard(self, text: str) -> SendResult:
        if not self._clipboard.copy(text):
            return SendResult(False, "none", "no clipboard backend available "
                              "(install xclip/xsel/wl-clipboard on Linux)")
        # Give the signer time to click into the chat box.
        if self.countdown_s > 0:
            left = float(self.countdown_s)
            while left > 0:                      # interruptible, so Pause / auto-off / Cancel really stop the paste
                if self.should_abort is not None and self.should_abort():
                    return SendResult(False, "cancelled", "cancelled before pasting (copied to clipboard)")
                step = min(0.1, left)
                self._sleep(step); left -= step
                if self.on_tick is not None:
                    self.on_tick(max(left, 0.0))
            if self.should_abort is not None and self.should_abort():
                return SendResult(False, "cancelled", "cancelled before pasting (copied to clipboard)")
        try:
            self._keyboard.paste(self.press_enter)
        except Exception as e:
            log.info("keystroke injection unavailable: %s", type(e).__name__)
            return SendResult(True, "clipboard-only", "copied; press Ctrl/Cmd+V")
        return SendResult(True, "clipboard+paste", "pasted into focused window")

    # -- playwright (EXPERIMENTAL, untested against a live call)
    def _send_playwright(self, text: str) -> SendResult:
        try:
            if self._pw is None:
                self._pw = PlaywrightDriver()
        except Exception as e:
            return SendResult(False, "playwright", f"unavailable: {type(e).__name__}")
        last = "unknown error"
        for _ in range(2):  # one retry
            try:
                if self._pw.send(text, self.press_enter):
                    return SendResult(True, "playwright", "sent and input cleared")
                last = "input did not clear"
            except Exception as e:
                last = type(e).__name__
        return SendResult(False, "playwright", last)


class PlaywrightDriver:
    """EXPERIMENTAL. Attaches to the signer's own browser profile (persistent
    context under the app data dir). Not tested against a live Meet call.
    The signer logs in and joins the meeting manually; we only type in chat."""

    def __init__(self, url: str | None = None):
        from playwright.sync_api import sync_playwright  # lazy import
        profile = app_data_dir() / "playwright_profile"
        profile.mkdir(parents=True, exist_ok=True)
        self._p = sync_playwright().start()
        self._ctx = self._p.chromium.launch_persistent_context(str(profile), headless=False)
        self._page = self._ctx.pages[0] if self._ctx.pages else self._ctx.new_page()
        if url:
            self._page.goto(url)

    def _locate(self, key: str):
        s = PLAYWRIGHT_SELECTORS[key]
        return self._page.get_by_role(s["role"], name=s["name"])

    def send(self, text: str, press_enter: bool) -> bool:
        box = self._locate("chat_input")
        if box.count() == 0:  # chat panel closed: try to open it
            self._locate("open_chat_button").first.click()
        box = self._locate("chat_input").first
        box.fill(text)
        if press_enter:
            box.press("Enter")
        else:
            self._locate("send_button").first.click()
        self._page.wait_for_timeout(300)
        return box.input_value() == ""  # verify the input cleared

    def close(self):
        self._ctx.close()
        self._p.stop()
