"""Label normalisation + clip quality for the UAE Sign Language dictionary manifest.

Why this exists: the site's English titles are derived from URL slugs, which mix
  (a) take variants      'Brown-01' / 'Brown-02'   -> already parsed into entry['variant']
  (b) duplicate pages    two different page ids with the identical slug (e.g. Numbers '13' twice)
  (c) numbered suffixes  'Door 1', 'Gorilla 2', '1000000 1' (a distinct sign? another take? unknown)
  (d) casing / spacing   'brother' vs 'Brother'

RULE (conservative, see data/dataset_report.md for the evidence):
  1. clean: collapse whitespace, upper-case the first letter ('brother' -> 'Brother').
  2. group by (category, case-insensitive cleaned label); every entry in a group is a take of that sign.
  3. A trailing ' N' is NOT stripped automatically. Real numbers ('11', '100') and genuinely
     numbered signs ('National Center For Statistics 1/2') must survive, and merging two different
     signs into one class is worse than keeping two near-duplicate classes. The only exceptions are
     the explicit MERGE_TO table below: entries whose landmarks are near-identical to the base sign
     (mean |delta| of the normalised, resampled clip < SIMILAR_DIST, i.e. below the 5th percentile
     of random sign pairs). Extend it with data/label_overrides.json ({"<clip id>": "<label>"})
     once a native signer has reviewed the uncertain ones.
Translations are never invented: labels are exactly the dictionary's English slugs (typos such as
'Naphew' are kept; fix them via overrides, not in code).
"""
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import features as F

ROOT = F.ROOT
LANDMARK_DIR = ROOT / "data" / "landmarks"
OVERRIDES_PATH = ROOT / "data" / "label_overrides.json"

# Quality threshold used by train.py: fraction of the trimmed (active) frames in which at least
# one hand is detected, and the minimum active length. Clips below either are skipped.
MIN_HAND_FRAC = 0.60
MIN_POSE_FRAC = 0.80
MIN_ACTIVE_FRAMES = 8
SIMILAR_DIST = 0.10   # see rule 3

# Verified by scripts/analyze_dataset.py (suffix entries whose landmarks match their base sign).
# key = clip id, value = category-local label it is merged into.
MERGE_TO = {
    "6100": "1000000",   # '1000000 1' (dist 0.05 to '1000000' take)
    "5587": "Lieutenant",  # 'Lieutenant 2' (dist 0.08)
}

_LEFT = slice(F.N_POSE, F.N_POSE + 21)
_RIGHT = slice(F.N_POSE + 21, F.N_PTS)


def clean_label(label):
    s = re.sub(r"\s+", " ", label).strip()
    return s[:1].upper() + s[1:]


def _overrides():
    if OVERRIDES_PATH.exists():
        return json.load(open(OVERRIDES_PATH, encoding="utf-8"))
    return {}


def canonical_label(entry):
    """Canonical sign name for a manifest entry (see module docstring for the rule)."""
    ov = _overrides()
    eid = str(entry["id"])
    if eid in ov:
        return clean_label(ov[eid])
    if eid in MERGE_TO:
        return clean_label(MERGE_TO[eid])
    return clean_label(entry["label"])


def load_manifest():
    return json.load(open(ROOT / "data" / "manifest.json", encoding="utf-8"))


def group_takes(manifest=None):
    """{(category, canonical_label): [entry, ...]} sorted by (variant, id); entries are the
    manifest dicts. Case-insensitive within a category."""
    manifest = load_manifest() if manifest is None else manifest
    groups, names = defaultdict(list), {}
    for e in manifest:
        lab = canonical_label(e)
        key = (e["category"], lab.lower())
        names.setdefault(key, lab)
        groups[key].append(e)
    return {(c, names[(c, k)]): sorted(v, key=lambda e: (e["variant"], int(e["id"])))
            for (c, k), v in groups.items()}


def takes_by_label(manifest=None):
    """Per canonical label (ignoring category) -> list of takes. Labels that occur in several
    categories are pooled (same English word => same sign is assumed)."""
    out = defaultdict(list)
    for (_, lab), v in group_takes(manifest).items():
        out[lab].extend(v)
    return dict(out)


def clip_stats(raw):
    """Quality stats from a raw landmark array (frames, N_PTS, 3, NaN = undetected)."""
    nan = dict(frames=len(raw), active=0, left=0.0, right=0.0, any_hand=0.0, pose=0.0, has_hand=False)
    if len(raw) < 2 or np.isnan(raw[:, :F.N_POSE]).all():
        return nan
    act = F.trim_active(raw)
    det = lambda sl: ~np.isnan(act[:, sl, 0]).any(axis=1)
    l, r = det(_LEFT), det(_RIGHT)
    return dict(frames=len(raw), active=len(act), left=float(l.mean()), right=float(r.mean()),
                any_hand=float((l | r).mean()), pose=float((~np.isnan(act[:, :F.N_POSE, 0]).any(axis=1)).mean()),
                has_hand=bool((l | r).any()))


def is_good(stats):
    return (stats["any_hand"] >= MIN_HAND_FRAC and stats["pose"] >= MIN_POSE_FRAC
            and stats["active"] >= MIN_ACTIVE_FRAMES)


def clip_quality(clip_id):
    """(stats, good) for data/landmarks/<id>.npy; missing file -> (None, False)."""
    p = LANDMARK_DIR / f"{clip_id}.npy"
    if not p.exists():
        return None, False
    s = clip_stats(np.load(p))
    return s, is_good(s)
