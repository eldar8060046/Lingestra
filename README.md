# Lingestra

Lingestra is a local, cross-platform sign-to-text tool for use during online meetings.

The recognition flow is:

1. MediaPipe extracts hand and body landmarks from the webcam.
2. A small Keras classifier, trained on the UAE Sign Language Dictionary, recognises signs.
3. Lingestra displays live captions with confidence and latency information.
4. If recognition is uncertain, Lingestra asks the signer to clarify.
5. The completed sentence can be pasted into the signer's own meeting chat.

No bot joins the meeting, and video never leaves the laptop.

## Setup

### Linux

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements.txt
```

### macOS

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -r requirements.txt
```

## Data pipeline

The pipeline has already been run once. The `data/` directory is large and is not meant for Git.

```bash
# Manifest and 1,244 videos; downloads are rate-limited and resumable.
python -m lingestra.data.scrape_dictionary

# Extract MediaPipe landmarks for each clip.
python scripts/extract_dataset.py

# Generate the dataset report and vocabulary.
python scripts/analyze_dataset.py
```

## Train, run, and test

```bash
# Train a model. You can also use --category Numbers --max-signs 12.
python scripts/train.py --vocab family --name my_model

# Run with a webcam and your trained model.
python -m lingestra.app --source webcam --model models/my_model.keras

# Run the safe demo mode without sending messages.
python -m lingestra.app --demo --model models/demo_family.keras --send-mode none

# Run all tests.
pytest -q
```

## Using the dashboard

The dashboard has four sections in the left sidebar. Screenshots are available in `docs/ui_*.png`.

On first start, a **How it works** card explains the basic flow: teach your signs, turn them on, then sign and send. The **How it works** button reopens the card later.

### Talk

- Shows the camera, a plain-language status panel, and the message being built.
- Provides **Send now**, **Undo last word**, and **Clear** controls.
- Displays `...` while the next word is being decoded, then shows recognised words in blue.
- A three-way message control lets you choose **Don't copy**, **Copy automatically**, or **Send with Space**.
- When recognition is uncertain, you can **Skip**, **Retry**, choose a suggested sign, use keyboard shortcuts, or answer with your fingers.
- Finger answers work only while a clarification question is open. The app can count up to ten fingers using both hands.
- **Settings > Choosing a word with your fingers** includes **Check what I see** and **Teach me my fingers**.
- Press **Esc** to return to the camera.

### My signs

- Browse dictionary signs and your own signs as cards with a skeleton preview and word label.
- Filter by **Recorded by me**, **Switched on**, or **All dictionary**.
- Search in English or Arabic; full phrases can be played back to back.
- Review, watch, add, or delete your personal recordings. The app keeps dictionary files unchanged.
- Use the **Use in conversation** switch to control which signs are recognised.
- Personal changes are stored encrypted and audit-logged.

### Teach a sign

- Enter a word and press **Teach a new sign** (or press **N**).
- Follow the countdown, sign naturally, and lower your hands when finished.
- Review the result and choose **Keep**, **Try again**, or **Delete**.
- Unknown movements can be named, added to an existing sign, watched, or deleted.
- **Clear all** removes unnamed movements after confirmation.

### Mirror

The **Mirror** button is also available in Settings and is enabled by default. It flips the camera, overlay, sign pictures, playback, and review animations so they are easier to copy.

This is display-only: recognition always receives the original, unflipped landmarks.

### Settings

Settings includes:

- Camera/source selection and camera restart.
- Mirror mode.
- Recognition certainty: **Faster**, **Balanced**, or **Fewer mistakes**.
- Message sending mode.
- The experimental built-in dictionary model.
- Technical details for speed, delay, and confidence information.
- Finger-answer calibration.
- Recognition speed: automatic or a selected number of CPU cores. Multi-core recognition can also be disabled.
- Privacy information and advanced options.

Recording suspends recognition and sending. Only landmarks are stored; video and images are never saved.

For tests and demos without touching your real store:

```bash
python -m lingestra.app --demo --store-dir scratch/x --passphrase test
```

The main dependencies are PySide6, NumPy, Numba, MediaPipe, TensorFlow, and cryptography; see `requirements.txt` for the complete list.

## Docs
- data/dataset_report.md - dataset analysis, label rules, limitations
- docs/threat_model.md   - STRIDE threat model (implemented vs recommended)
- docs/chat_output.md    - how text reaches the chat and why there is no bot

## Honest limitations
- ~1 clip per sign, probably one signer: accuracy measured on augmented training clips shows
  robustness to speed/angle only, NOT generalisation to a new signer.
- Webcam and Playwright/Meet modes are untested (no camera / live call in the dev sandbox).
- Demo phrases are scripted word sequences and need native-signer review.
