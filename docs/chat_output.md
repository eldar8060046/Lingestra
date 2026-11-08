# How text reaches the meeting chat

Lingestra never joins a meeting. The signer is already in the call with their own account. Lingestra only gets the finished sentence into the signer's chat box.

## The default: clipboard

1. The sentence is copied to the clipboard (pyperclip, with fallbacks: wl-copy / xclip / xsel on Linux, pbcopy on macOS, clip on Windows).
2. Optional countdown (`countdown_s`) so the signer can click into the chat box.
3. Lingestra presses Ctrl+V (Cmd+V on macOS) and then Enter in whatever window is focused.
4. If it cannot press keys (Wayland, macOS without Accessibility permission, no display), it stops after step 1 and says "copied; press Ctrl/Cmd+V". This still counts as success.

Risk: keystrokes go to the focused window, so use the countdown and keep the meeting chat focused. Set `press_enter=False` to review before sending.

## Experimental: Playwright

Drives a browser the signer logs into (persistent profile in the app data folder), finds the chat box by its accessible role and label, fills it, sends, and checks the box emptied. One retry, then it falls back to the clipboard. It is **EXPERIMENTAL and has not been tested against a live call**. Meeting sites can change their pages and their terms may restrict automation, so it is opt-in only.

## Why not an official API or bot?

Found in research:
- **Zoom**: the Meeting SDK policy forbids bots/automated participants in meetings.
- **Google Meet**: the REST API has no endpoint for sending chat messages.
- **Microsoft Teams**: bots cannot use the meeting chat in the way we need.

So the portable route is "type as the user", i.e. the clipboard.

## Fallback order

1. Playwright (only if the user chose it and it is installed)
2. Clipboard + simulated paste/Enter
3. Clipboard only (user presses Ctrl/Cmd+V)
4. Failure message in the dashboard (no clipboard tool found), with the text still shown on screen

Message text is never written to INFO logs.
