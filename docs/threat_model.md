# Lingestra threat model (STRIDE)

Status legend: **[Done]** implemented in code, **[Rec]** recommended, not yet implemented.

## Assets
| Asset | Why it matters |
|---|---|
| Webcam stream | Most sensitive: face, home, other people. Never leaves the laptop. |
| Landmark windows | Derived from the signer's body; can identify a person. |
| Sign library / labels | Wrong labels cause wrong sentences. |
| Clipboard contents | Contains the sentence while pasting. |
| Meeting chat session | Messages are sent as the signer. |
| Model files | Keras loading can execute code (Lambda layers, pickle). |
| Dictionary-derived training data | Third-party content (za.gov.ae), licence/terms apply. |

## Actors
Signer (trusted), other local apps/users (untrusted), meeting participants/platform (untrusted), dependency/package authors, a thief with the laptop, teammates labelling data.

## Trust boundaries
1. Camera -> Lingestra process (OS permission).
2. Lingestra -> clipboard / synthetic keystrokes (shared with every local app).
3. Lingestra -> disk (model files in, landmark store out).
4. Lingestra -> meeting client (focused window or browser).
5. Network -> dependencies and dictionary scraper (build time only).

## Threats and mitigations

### Spoofing
- Fake model file replacing the real one. Mitigation: load models only from the app `models/` dir and verify a SHA-256 manifest **[Rec]**.
- Someone else labelling as another user. Audit log records the actor name **[Done]**, but actor is self-declared (no authentication) - residual.

### Tampering
- Poisoned labels / training data: every label change is audited with old and new value, hash-chained so edits/reordering/mid-log deletion are detected by `verify_audit()` **[Done]**. Truncating the log tail is NOT detected without an external anchor (residual). Review labelled windows before retraining **[Rec]**.
- Tampered landmark files: Fernet (AES-128-CBC + HMAC) authenticates ciphertext; modification fails to decrypt **[Done]**.
- Malicious model: `.keras` files can contain code; load only trusted, hash-verified files, use `safe_mode=True` **[Rec]**. Never use pickle for anything we store; arrays are loaded with `allow_pickle=False` **[Done]**.
- Keystrokes injected into the wrong window: countdown, `press_enter=False` option, graceful fallback to clipboard-only **[Done]**; the user must keep the chat focused (residual).

### Repudiation
- Disputes about who labelled/deleted a sample: append-only hash-chained audit log with actor and timestamp **[Done]**. Not signed, so a local attacker with file access could rebuild the whole chain (residual; use HMAC with a key held in the keyring or ship logs off-device **[Rec]**).

### Information disclosure
- Video leakage: the store rejects image-like arrays (uint8, HxWx3, oversized) so video/frames cannot be saved via this API **[Done]**; the rest of the app must not write frames **[Rec: code review]**.
- Landmarks at rest: encrypted; key from scrypt passphrase, else OS keyring, else a 0600 keyfile; dirs 0700 where supported **[Done]**. Keyfile fallback only protects against other users/backups, not against malware running as the signer (residual).
- Clipboard snooping by other local apps/clipboard managers (history tools may retain the sentence): inherent to the clipboard route. Mitigation: keep clipboard time short; **[Rec]** clear/restore previous clipboard after paste. Residual risk accepted.
- Logs: message text is never logged at INFO **[Done]**.
- Meeting platform sees only the chat text the signer sends.

### Tampering by the signer's own mistakes: learning from corrections
- "Learn from my corrections" saves the movement of a confirmed 'Not sure - did you mean A or B?' answer as another encrypted example of the chosen sign. A wrong tap could therefore teach a wrong example and slowly poison a sign. Mitigations **[Done]**: only for signs that are switched on and only after an explicit tap (repeat/reject and unanswered questions teach nothing); quality gate (>= 0.4 s, hands visible in >= 60% of frames, distance to the sign's own examples <= 2x its acceptance distance, no duplicate within 2 s); at most 6 automatic examples per sign (oldest dropped first, the signer's own recordings are never dropped); a 15 s 'Undo' bar after each lesson; every automatic example is listed in the sign's detail with a 'Remove' button; one audit record per lesson / removal (ids and counts only, never landmarks); a settings switch (default on) turns it off. Examples are landmarks only, stored encrypted like every other example. Residual: the gate cannot tell two similar signs apart (the candidates of a question are close by construction), so a wrong tap between look-alike signs can still be learned until undone or evicted.

### Denial of service
- Huge arrays or id floods: array size capped at 20 MB **[Done]**. Recognizer adaptive frame skipping is owned by the recognizer. Disk quota for the unknown store **[Rec]**.
- Corrupt files are skipped in listings, not fatal **[Done]**.

### Elevation of privilege
- Path traversal via ids: ids must match `[A-Za-z0-9_-]{1,64}` **[Done]**.
- Dependency supply chain: pin versions with hashes (`pip-compile --generate-hashes`), run `pip-audit` **[Rec]**. pynput/pyperclip run in-process with user rights.
- Playwright pulls a full browser: experimental, opt-in, lazy import **[Done]**.
- Dictionary scraper downloads untrusted content: treat as data only, never execute **[Rec: validate formats]**.

## Third-party terms and privacy
- Automating a meeting client (Playwright) may violate the platform's terms; Zoom's Meeting SDK policy forbids bots; hence clipboard is the default and no bot joins. No code was taken from unlicensed third-party repos.
- Training data comes from the UAE Sign Language Dictionary; confirm reuse terms before distribution **[Rec]**.
- Signer privacy: video stays local, only landmarks/text are stored, encrypted, deletable with an audited delete. Landmarks can still be biometric-ish; keep retention short **[Rec]**.

## Not covered
Compromised OS/root, physical access to an unlocked laptop, side channels, and the security of Zoom/Meet/Teams themselves.
