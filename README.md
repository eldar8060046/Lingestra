# Lingestra

Local, cross-platform sign-to-text tool for a signer in online meetings. MediaPipe landmarks from the
webcam -> small Keras classifier (trained on the UAE Sign Language Dictionary) -> live captions with
confidence/latency -> clarification prompt when unsure -> sentence pasted into the signer's own
meeting chat. No bot joins the meeting; video never leaves the laptop.

## Setup
    python -m venv .venv && .venv/bin/pip install -r requirements.txt  
    
  
   # Windows: 
   python -m venv venv
   venv\Scripts\activate
  pip install -r requirements.txt

## Data pipeline (already run once; data/ is large and not meant for git)
    python -m lingestra.data.scrape_dictionary      # manifest + 1244 videos (1 req/s, resumable)
    python scripts/extract_dataset.py               # MediaPipe landmarks per clip
    python scripts/analyze_dataset.py               # data/dataset_report.md + data/vocabulary.json

## Train / run / test
    python scripts/train.py --vocab family --name my_model       # or --category Numbers --max-signs 12
    python -m lingestra.app --source webcam --model models/my_model.keras
    python -m lingestra.app --demo --model models/demo_family.keras --send-mode none
    pytest -q

## Using the dashboard
Four plain sections in the left sidebar (screenshots in `docs/ui_*.png`). A "How it works" card explains the three
steps on first start (teach your signs, turn them on, sign and send); the **How it works** button reopens it.
- **Talk**: the camera, a status panel under it that always says in words what is happening, and the message being
  built in large text with **Send now**, **Undo last word** and **Clear**. While you sign several words in a row it shows
  **"..."** (the app is working out the next word), then the word itself in **blue** the moment it is recognised, and
  the word is added to "My message" at once (nothing is sent yet). The message is sent when you **lower your hands**.
  Only the **last message sent** is shown, with a check mark ("Sent") or "Not sent".
  Under "My message" a **three-way toggle** decides what happens to a finished message: **Don't copy** (it is only shown),
  **Copy automatically** (copied and pasted after a short countdown when you lower your hands) or **Send with Space**
  (nothing is sent until you press the **Space bar** or *Send now*; lowering your hands does not end the message, it keeps
  growing). Space only works on the Talk screen and never while you type in a text field. The choice is remembered.
  When the app is not sure it shows numbered buttons: **1) Skip** (cancel this sign and carry on with the next one),
  **2) Retry** (sign it again), then up to 8 **similar signs** from your library, **3) ... 10) ...** (only signs that really
  look similar are offered, so often just a few buttons). Tap one, press its number (**0** = 10; **S** skips, **R**
  retries), or **hold up that many fingers**: 1-5 on one hand, 6-10 = five on one hand plus the rest on the other (keep
  the hand steady for half a second; the button lights up while you hold it). Your hands stay up the whole time and the
  answer takes the question's place in the message. If you lower your hands without answering, I wait 2 seconds, leave that
  word out and send the rest. The finger answer only works while a question is open, and it starts about 0.7 s after the
  question appears. It needs no training: I count your extended fingers from the hand landmarks (the thumb is ignored unless all four fingers are up - then thumb out = 5 - so a resting thumb never adds a finger) (a resting fist on your other hand is ignored, and the panel says what it sees, e.g. "I see 2 fingers"). If it does not read YOUR hand well, open **Settings > Choosing a word with your fingers**: **Check what I see** shows the number live and the raw per-finger measurements (works any time; **Copy my readings** puts a few numbers - never a picture - on the clipboard so you can paste them to whoever is helping), and **Teach me my fingers** learns your own 1, 2 and 3 in about half a minute (stored encrypted, only finger measurements). With nothing taught
  yet, a "Teach my first sign" button shows the next step. **Esc** goes back to the camera.
- **My signs**: every dictionary sign and your own as cards (skeleton picture + word). Chips: *Recorded by me* /
  *Switched on* / *All dictionary*; **Switch by group...** opens a window with one row per collection and per category
  (Alphabets, Animals, ...) and its own Switch on / Switch off buttons; the search box always looks through everything (English or Arabic, whole phrases play back to back).
  Cards of signs you have recorded examples of carry a purple **Mine** badge (with the number, e.g. *Mine 3*). Select one and the detail panel gets a **My recordings (3)** drop-down directly below **More options**: every example (also the ones I learned from your corrections) with a looping preview, **Watch** (plays it in the big view) and a two-tap **Delete**, plus **Record another example**. The built-in dictionary example and the only example of a sign you created cannot be deleted; deleting a re-recording of a dictionary sign brings the built-in one back. Click a card to watch it in the big view (Slow / Normal / Fast, Repeat, "Example 1 of 3"). The panel below has a
  **Use in conversation** switch (only switched-on signs are recognised), the big **Record my own example** button (a sign
  needs at least one example recorded by *you* to be recognised reliably; 3 is the goal) and a simple Save / Cancel form
  for the written word. Notes, other search words, hide, undo changes, delete and "save my examples for training" are under
  *More options*. Dictionary files are never modified; changes are stored encrypted and audit-logged.
- **Teach a sign**: type the word (optional), press **Teach a new sign** (key **N**): "Get ready... 3 2 1", "Raise your
  hands and sign", "Lower your hands when you finish". Then watch it repeat and choose **Keep** (K), **Try again** (T) or
  **Delete** (D) (Alt+K/T/D work while typing). After saving you are asked to record two more examples with one click.
  Movements the app noticed but did not know appear as cards: type a name and press **Add as new sign** (Enter works too),
  or **Add to an existing sign**, **Watch it**, **Delete**. **Clear all** (after asking) deletes every saved movement you have not named.
- **Mirror**: the **Mirror** button on the camera (also in Settings, remembered, on by default) flips everything you look at -
  camera and overlay, sign pictures, playback and review animations, pending cards - so you can copy signs like in a mirror.
  This is display only: the recogniser always receives the original, unflipped landmarks.
- **Settings**: camera/source and restart, mirror, *How sure must I be before writing a word* (Faster / Balanced / Fewer
  mistakes), send messages to chat automatically (copies the message and pastes it after a 3 second countdown, click your chat
  box meanwhile), the experimental built-in dictionary model (off by default), *Show technical details* (off by default;
  adds speed / delay / confidence numbers to the Talk screen), privacy note, and a collapsed *Advanced* part.
- Recording suspends recognition and sending. Only landmarks are ever stored, never video or images.
- Tests/demo without touching your real store: `python -m lingestra.app --demo --store-dir scratch/x --passphrase test`.
- Packages: PySide6, numpy, cryptography (already in requirements.txt); nothing new.

## Docs
- data/dataset_report.md - dataset analysis, label rules, limitations
- docs/threat_model.md   - STRIDE threat model (implemented vs recommended)
- docs/chat_output.md    - how text reaches the chat and why there is no bot

## Honest limitations
- ~1 clip per sign, probably one signer: accuracy measured on augmented training clips shows
  robustness to speed/angle only, NOT generalisation to a new signer.
- Webcam and Playwright/Meet modes are untested (no camera / live call in the dev sandbox).
- Demo phrases are scripted word sequences and need native-signer review.
