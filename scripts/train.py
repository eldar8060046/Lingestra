"""Train the Lingestra sign classifier on the scraped UAE dictionary landmarks.

The dictionary has ~1 clip per sign, so instead of splitting real clips we train on an endless
stream of augmented versions of each clip (random speed 0.5-2x + non-uniform time warp, rotation,
zoom, shift, jitter) and evaluate on freshly-augmented clips bucketed by speed. Where a sign has a
second real take (variant 02/03) that take is held out as a genuine test clip.

NOTE: all clips are from the same dictionary signer(s), so accuracy here measures robustness to
speed/position/angle, NOT generalisation to a new signer. That is stated in the report.

Usage:
  python scripts/train.py --category Numbers --max-signs 15
  python scripts/train.py --vocab family
  python scripts/train.py --labels "Mother,Father,Brother,Sister,Son,Daughter"
"""
import argparse, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingestra import features as F, labels as L

ap = argparse.ArgumentParser()
ap.add_argument("--category", action="append", help="dictionary category (repeatable); default all")
ap.add_argument("--labels", help="comma-separated explicit sign labels")
ap.add_argument("--vocab", help="name of a demo vocabulary set in data/vocabulary.json (e.g. family)")
ap.add_argument("--max-signs", type=int, default=15)
ap.add_argument("--epochs", type=int, default=60)
ap.add_argument("--steps", type=int, default=60, help="steps per epoch")
ap.add_argument("--batch", type=int, default=64)
ap.add_argument("--name", default="sign_model")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--include-user-signs", nargs="?", const="data/user_signs", default=None, metavar="DIR",
                help="add landmark takes exported by the app (SignLibrary.export_user_signs: *.npy + "
                     "labels.json) as extra TRAINING clips. A user take never counts as a real test clip "
                     "unless the label has >=3 user takes (the last one is then held out). Default dir "
                     "data/user_signs. Landmarks only, plaintext: keep that directory private.")
a = ap.parse_args()
rng = np.random.default_rng(a.seed)

man = json.load(open(F.ROOT / "data" / "manifest.json"))
have = {e["id"] for e in man if (F.ROOT / "data" / "landmarks" / f"{e['id']}.npy").exists()}
# skip clips whose hands/pose were rarely detected (thresholds: labels.MIN_HAND_FRAC etc.)
cands = [e for e in man if e["id"] in have and L.clip_quality(e["id"])[1]]
print(f"{len(have) - len(cands)} low-quality clips skipped (coverage is biased toward easily detected signs)")
for e in cands:
    e["label"] = L.canonical_label(e)   # normalised label (see lingestra/labels.py)
if a.vocab:
    voc = json.load(open(F.ROOT / "data" / "vocabulary.json", encoding="utf-8"))["sets"][a.vocab]["signs"]
    keep = {(s["category"], s["label"]) for s in voc}
    cands = [e for e in cands if (e["category"], e["label"]) in keep]
if a.category:
    cands = [e for e in cands if e["category"] in a.category]
if a.labels:
    want = [s.strip().lower() for s in a.labels.split(",")]
    cands = [e for e in cands if e["label"].lower() in want]
by_label = defaultdict(list)
for e in cands:
    by_label[e["label"]].append(e)
labels = sorted(by_label)[: a.max_signs] if not (a.labels or a.vocab) else sorted(by_label)
if len(labels) < 2:
    sys.exit(f"need >=2 signs with extracted landmarks, found {len(labels)} (run scripts/extract_dataset.py)")
user_raws = defaultdict(list)       # label -> [raw landmark takes] from the app (see --include-user-signs)
if a.include_user_signs:
    udir = Path(a.include_user_signs)
    udir = udir if udir.is_absolute() else F.ROOT / udir
    meta = json.load(open(udir / "labels.json", encoding="utf-8")) if (udir / "labels.json").exists() else {}
    known = {l.lower(): l for l in labels}
    want = {s.strip().lower() for s in a.labels.split(",")} if a.labels else None
    for fn, m in sorted(meta.items(), key=lambda kv: (kv[1]["label"], kv[1]["take"])):
        if "/" in fn or "\\" in fn or not (udir / fn).exists() or (want and m["label"].lower() not in want):
            continue
        lab = known.get(str(m.get("default_label") or "").lower()) or known.get(m["label"].lower(), m["label"])
        if lab not in known.values():
            labels.append(lab); known[lab.lower()] = lab
        user_raws[lab].append(np.load(udir / fn, allow_pickle=False))
    labels = sorted(labels)
    print(f"user signs: {sum(map(len, user_raws.values()))} takes over {len(user_raws)} labels")
classes = labels + ["none"]
print(f"{len(labels)} signs: {labels}")

# ---- load clips: first take -> train, extra takes -> real held-out test
train, real_test, idle = {}, [], []
extra_train = {}
for lab in labels:
    takes = sorted(by_label.get(lab, []), key=lambda e: e["variant"])
    raws = [np.load(F.ROOT / "data" / "landmarks" / f"{e['id']}.npy") for e in takes]
    urs = user_raws.get(lab, [])
    if len(urs) >= 3:                      # keep a genuine held-out user take
        real_test.append((lab, F.clip_to_tensor(urs[-1]))); urs = urs[:-1]
    if not raws:                           # user-only sign: first user take is the primary clip
        raws, urs = urs[:1], urs[1:]
    train[lab] = F.normalise(F.trim_for_model(raws[0]))
    extra_train[lab] = [F.normalise(F.trim_for_model(u)) for u in urs]
    for r in raws[1:]:
        real_test.append((lab, F.clip_to_tensor(r)))
    r = raws[0]
    act = np.where(F.hand_activity(r) > 0.25)[0]
    if len(act):
        for seg in (r[: max(act[0] - 3, 0)], r[act[-1] + 4:]):
            if len(seg) >= 15:
                idle.append(F.normalise(seg))
if not idle:
    idle = [np.zeros((30, F.FEAT), np.float32)]


def augment(lab, rng, max_speed=2.0):
    pool = [train[lab]] + extra_train[lab]
    x = F.time_warp(pool[rng.integers(len(pool))], rng, max_speed=max_speed)
    x = F.spatial_augment(x, rng)
    return F.add_velocity(F.resample(x))


def idle_sample(rng):
    seg = idle[rng.integers(len(idle))]
    n = rng.integers(15, len(seg) + 1) if len(seg) > 15 else len(seg)
    s = rng.integers(0, len(seg) - n + 1)
    x = F.spatial_augment(seg[s:s + n], rng, noise=0.02)
    return F.add_velocity(F.resample(x))


def batch(rng, n, max_speed=2.0):
    X, y = [], []
    for _ in range(n):
        if rng.random() < 1 / len(classes) * 1.5:
            X.append(idle_sample(rng)); y.append(len(labels))
        else:
            k = rng.integers(len(labels)); X.append(augment(labels[k], rng, max_speed)); y.append(k)
    return np.stack(X), np.array(y)


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
tf.random.set_seed(a.seed)
model = keras.Sequential([
    layers.Input((F.T, 2 * F.FEAT)),
    layers.Conv1D(96, 5, padding="same", activation="relu"),
    layers.SpatialDropout1D(0.2),
    layers.Bidirectional(layers.LSTM(96)),
    layers.Dropout(0.4),
    layers.Dense(len(classes), activation="softmax"),
])
model.compile(keras.optimizers.Adam(1e-3), "sparse_categorical_crossentropy", metrics=["accuracy"])


def gen():
    while True:
        yield batch(rng, a.batch)


Xv, yv = batch(np.random.default_rng(999), 600)
model.fit(gen(), steps_per_epoch=a.steps, epochs=a.epochs, validation_data=(Xv, yv), verbose=2,
          callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True, monitor="val_accuracy"),
                     keras.callbacks.ReduceLROnPlateau(patience=4, factor=0.5)])

# ---- evaluation: accuracy by playback speed (fresh augmentations, never seen in training)
print("\nTraining-clip robustness under re-timing/augmentation (NOT generalisation; same clips as training):")
erng = np.random.default_rng(12345)
for speed in (0.5, 0.75, 1.0, 1.5, 2.0):
    X, y = [], []
    for _ in range(300):
        k = erng.integers(len(labels))
        seq = train[labels[k]]
        n = max(int(len(seq) / speed), 6)
        x = F.spatial_augment(F.resample(seq, n), erng)
        X.append(F.add_velocity(F.resample(x))); y.append(k)
    p = model.predict(np.stack(X), verbose=0)
    print(f"  {speed:>4}x : top-1 {np.mean(p.argmax(1) == np.array(y)):.1%}")
if real_test:
    X = np.stack([t for _, t in real_test]); y = np.array([labels.index(l) for l, _ in real_test])
    p = model.predict(X, verbose=0)
    print(f"Real second-take clips ({len(y)}): top-1 {np.mean(p.argmax(1) == y):.1%}")
else:
    print("No second real takes among the chosen signs.")

out = F.MODELS; out.mkdir(exist_ok=True)
model.save(out / f"{a.name}.keras")
json.dump({"classes": classes, "T": F.T}, open(out / f"{a.name}.json", "w"), ensure_ascii=False)
print("saved", out / f"{a.name}.keras")
