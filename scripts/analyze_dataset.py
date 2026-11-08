"""Analyse all dictionary landmark clips -> data/dataset_report.md + data/vocabulary.json.

Usage: python scripts/analyze_dataset.py
Headless: only needs data/manifest.json and data/landmarks/*.npy (no webcam, no MediaPipe).
"""
import json, random, re, sys, warnings
from collections import Counter, defaultdict
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingestra import features as F, labels as L

warnings.filterwarnings("ignore")
D = F.ROOT / "data"
man = L.load_manifest()
ent = {e["id"]: e for e in man}

# ---------------- per-clip stats
EMPTY = dict(frames=0, active=0, left=0.0, right=0.0, any_hand=0.0, pose=0.0, has_hand=False)
stats, good = {}, {}
for e in man:
    s, g = L.clip_quality(e["id"])
    stats[e["id"]], good[e["id"]] = (s or EMPTY), g
bad = sorted((i for i in stats if not good[i]), key=lambda i: stats[i]["any_hand"])


def summ(ids):
    m = lambda k: float(np.mean([stats[i][k] for i in ids])) if ids else 0.0
    return dict(n=len(ids), frames=m("frames"), active=m("active"), left=m("left"), right=m("right"),
                any_hand=m("any_hand"), pose=m("pose"),
                nohand=sum(not stats[i]["has_hand"] for i in ids), bad=sum(not good[i] for i in ids))


cats = defaultdict(list)
for e in man:
    cats[e["category"]].append(e["id"])


# ---------------- label-rule evidence (landmark similarity)
def feat(i):
    r = np.load(L.LANDMARK_DIR / f"{i}.npy")
    if len(r) < 5 or np.isnan(r[:, :F.N_POSE]).all():
        return None
    return F.resample(F.normalise(F.trim_active(r)))


_cache = {}
def dist(a, b):
    for k in (a, b):
        if k not in _cache:
            _cache[k] = feat(k)
    x, y = _cache[a], _cache[b]
    return None if x is None or y is None else float(np.mean(np.abs(x - y)))


rng = random.Random(0)
ok_ids = [i for i in stats if stats[i]["active"] >= 8]
rand_d = [d for d in (dist(*rng.sample(ok_ids, 2)) for _ in range(500)) if d is not None]
rand_pct = np.percentile(rand_d, [5, 25, 50])
by_raw = defaultdict(list)
for e in man:
    by_raw[(e["category"], e["label"])].append(e["id"])
same_d = [d for v in by_raw.values() if len(v) > 1 for d in [dist(v[0], v[1])] if d is not None]
suffix_rows = []
for (cat, lab), v in sorted(by_raw.items()):
    m = re.match(r"^(.*\S) (\d+)$", lab)
    if m:
        base = by_raw.get((cat, m.group(1)))
        suffix_rows.append((cat, lab, v[0], m.group(1) if base else None,
                            dist(v[0], base[0]) if base else None))

groups = L.group_takes(man)
sign_cats = defaultdict(dict)
for (c, lab), takes in groups.items():
    sign_cats[c][lab] = takes
n_signs = len(groups)
takes_hist = Counter(len(t) for t in groups.values())
multi = sum(len(t) >= 2 for t in groups.values())
prof_all = Counter(parse_qs(urlparse(e["video_url"]).query).get("profile_id", ["none"])[0] for e in man)
prof_num = Counter(parse_qs(urlparse(e["video_url"]).query).get("profile_id", ["none"])[0]
                   for e in man if e["category"] == "Numbers")


# ---------------- vocabularies
def sign_rec(cat, lab):
    takes = sign_cats[cat][lab]
    gt = [e for e in takes if good[e["id"]]]
    q = max(((stats[e["id"]]["any_hand"] + stats[e["id"]]["pose"]) / 2 for e in gt), default=0.0)
    return dict(label=lab, category=cat, takes=len(takes), good_takes=len(gt),
                ids=[e["id"] for e in takes], quality=round(q, 3))


def pick(spec):
    out = []
    for cat, lab in spec:
        r = sign_rec(cat, lab)
        if r["good_takes"] and r["quality"] >= 0.75:
            out.append(r)
        else:
            print("skipped (quality)", cat, lab, r["quality"])
    return out


H, E_, C, N_, ED = "Health", "Education", "Common Verbs", "Numbers", "Measurement Units"
SETS = {
    "family": ("Family members",
               [("Family", x) for x in ["Mother", "Father", "Brother", "Sister", "Son", "Daughter", "Wife",
                                         "Uncle", "Aunt", "Grandmother", "Twins", "Couple", "Family", "Sibling"]],
               ["Mother Father", "Father Son", "Mother Daughter", "Brother Sister", "Wife Son Daughter",
                "Uncle Aunt", "Grandmother Mother Father", "Twins Brother Sister", "Couple Son Daughter",
                "Family Mother Father Sibling"]),
    "numbers_1_10": ("Numbers one to ten",
                     [(N_, x) for x in ["One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten"]],
                     ["One Two Three", "Three Four Five", "Five Six Seven Eight", "Eight Nine Ten",
                      "Ten Nine Eight", "Two Four Six Eight Ten", "One Three Five Seven Nine",
                      "Seven Five Two", "Nine Nine One", "Four Ten Six"]),
    "common_meeting": ("Meeting phrases / common words (greetings-style set; the dictionary has no 'Hello')",
                       [(H, "Doctor"), (H, "Medicine"), (H, "How Are You"), (H, "Fine"), (H, "Name"), (H, "Wait"),
                        (E_, "Repeat"), (E_, "Good"), (E_, "Very Good"), (C, "Helps"), (C, "Wants"), (C, "Speaks"),
                        (C, "Write"), (ED, "Time")],
                       ["How Are You Fine", "Name Write", "Wait Repeat", "Repeat Speaks", "Wants Helps",
                        "Time Wait Good", "Very Good Fine", "Doctor Wait", "Wants Doctor", "Helps Medicine"]),
    "days_time": ("Days of the week and time words",
                  [(ED, x) for x in ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday",
                                      "Week", "Year", "Time", "Yesterday"]],
                  ["Sunday Monday", "Monday Tuesday Wednesday", "Thursday Friday Saturday", "Yesterday Friday",
                   "Week Time", "Week Year", "Saturday Sunday", "Tuesday Time", "Year Yesterday", "Wednesday Thursday Friday"]),
}
vocab = {"_note": "DRAFT demo vocabularies and phrases generated by scripts/analyze_dataset.py. Phrases are "
                  "exact word sequences for scripted demos only and NEED NATIVE-SIGNER REVIEW (they are not "
                  "grammatical UAE Sign Language and are not translations).",
         "sets": {}}
for name, (desc, spec, phrases) in SETS.items():
    signs = pick(spec)
    have = {s["label"] for s in signs}
    vocab["sets"][name] = dict(description=desc, signs=signs,
                               phrases=[{"text": p, "status": "draft_needs_native_signer_review"} for p in phrases])


def check_phrase(p, labels):
    """Greedy longest-match: is phrase p an exact concatenation of set labels?"""
    toks, rest = [], p
    while rest:
        for lab in sorted(labels, key=len, reverse=True):
            if rest == lab or rest.startswith(lab + " "):
                toks.append(lab); rest = rest[len(lab):].strip(); break
        else:
            return None
    return toks


for name, v in vocab["sets"].items():
    labs = [s["label"] for s in v["signs"]]
    for ph in v["phrases"]:
        ph["signs"] = check_phrase(ph["text"], labs)
        assert ph["signs"], f"phrase {ph['text']!r} not buildable from set {name}"
json.dump(vocab, open(D / "vocabulary.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

# ---------------- report
pc = lambda x: f"{100 * x:.0f}%"
o = summ(list(stats))
R = []
w = R.append
w("# Lingestra dataset report\n")
w("Generated by `scripts/analyze_dataset.py` from `data/manifest.json` and `data/landmarks/*.npy`. "
  "Re-run the script to refresh.\n")
w("## 1. Overall\n")
w(f"- Clips: **{o['n']}**; mean {o['frames']:.0f} frames, mean active (trimmed) length {o['active']:.0f} frames.")
w(f"- In the trimmed part of a clip: left hand detected {pc(o['left'])}, right hand {pc(o['right'])}, "
  f"either hand {pc(o['any_hand'])}, pose {pc(o['pose'])} of frames (averages over clips).")
w(f"- Clips where no hand is ever detected: **{o['nohand']}**.")
w(f"- Quality filter (`labels.is_good`): either-hand >= {pc(L.MIN_HAND_FRAC)}, pose >= {pc(L.MIN_POSE_FRAC)}, "
  f"active >= {L.MIN_ACTIVE_FRAMES} frames. **{o['bad']} clips fail** ({100 * o['bad'] / o['n']:.1f}%).\n")
w("## 2. Per category\n")
w("| Category | Clips | Signs | Avg frames | Avg active | L hand | R hand | Any hand | Pose | No hand | Low quality |")
w("|---|---|---|---|---|---|---|---|---|---|---|")
for c in sorted(cats):
    s = summ(cats[c])
    w(f"| {c} | {s['n']} | {len(sign_cats[c])} | {s['frames']:.0f} | {s['active']:.0f} | {pc(s['left'])} | "
      f"{pc(s['right'])} | {pc(s['any_hand'])} | {pc(s['pose'])} | {s['nohand']} | {s['bad']} |")
w(f"| **All** | {o['n']} | {n_signs} | {o['frames']:.0f} | {o['active']:.0f} | {pc(o['left'])} | {pc(o['right'])} | "
  f"{pc(o['any_hand'])} | {pc(o['pose'])} | {o['nohand']} | {o['bad']} |\n")
w("(Signs are canonical labels per category; the same word can appear in two categories.)\n")
w(f"## 3. Low-quality clips ({len(bad)})\n")
w("Hands rarely detected in the trimmed segment, pose missing, or clip too short. Skipped by `train.py`.\n")
w("| id | Category | Label | Frames | Active | Any hand | Pose |")
w("|---|---|---|---|---|---|---|")
for i in bad:
    s, e = stats[i], ent[i]
    w(f"| {i} | {e['category']} | {e['label']} | {s['frames']} | {s['active']} | {pc(s['any_hand'])} | {pc(s['pose'])} |")
w("")
w("## 4. Label normalisation\n")
w("Rule (implemented in `lingestra/labels.py`):\n")
w("1. Clean whitespace and capitalise the first letter (`brother` -> `Brother`).")
w("2. Group by (category, case-insensitive label). Every clip in a group is a *take* of one sign. This covers "
  "`-01/-02` variants and duplicate pages with an identical slug (many Numbers appear twice).")
w("3. A trailing ` N` is **not** stripped automatically: real numbers (`11`, `100`) and genuinely numbered signs "
  "(`National Center For Statistics 1/2`, `Qasrul Emiraat 1/2`) must stay, and wrongly merging two different "
  "signs into one class hurts more than keeping two near-duplicate classes. Only entries whose landmarks are "
  f"near-identical to the base sign (mean abs. difference < {L.SIMILAR_DIST}) are merged via `labels.MERGE_TO`.\n")
w("Evidence: mean absolute difference between trimmed+resampled normalised clips (lower = more similar). "
  f"Random sign pairs: 5th pct {rand_pct[0]:.2f}, 25th {rand_pct[1]:.2f}, median {rand_pct[2]:.2f}. "
  f"Clips known to be the same label (n={len(same_d)}): median {np.median(same_d):.2f}, "
  f"25th {np.percentile(same_d, 25):.2f}, 75th {np.percentile(same_d, 75):.2f}.\n")
w("**Uncertainty:** the same-label and random distributions overlap strongly (the metric has no time "
  "alignment and signers/framing differ), so landmark similarity is weak evidence. Only very close matches are "
  "merged; everything else stays separate and should be reviewed by a native signer. "
  "Suffix entries found:\n")
w("| Category | Label | Clip | Base label present | Distance to base | Decision |")
w("|---|---|---|---|---|---|")
for cat, lab, cid, base, dd in suffix_rows:
    dec = f"merged into `{L.MERGE_TO[cid]}`" if cid in L.MERGE_TO else "kept separate"
    w(f"| {cat} | {lab} | {cid} | {base or '-'} | {'-' if dd is None else f'{dd:.2f}'} | {dec} |")
w("")
w("## 5. Signs, takes, signers\n")
w(f"- Distinct canonical signs (category, label): **{n_signs}** from {len(man)} clips "
  f"({len(L.takes_by_label(man))} distinct labels if categories are pooled).")
w("- Takes per sign: " + ", ".join(f"{k} take(s): {v}" for k, v in sorted(takes_hist.items())) + ".")
w(f"- Signs with >= 2 takes: **{multi}** ({100 * multi / n_signs:.0f}%); of these, "
  f"{sum(1 for t in groups.values() if len(t) >= 2 and sum(good[e['id']] for e in t) >= 2)} have >= 2 good-quality takes.\n")
w("| Category | Signs | Clips | Signs with >= 2 takes | Avg clips / sign |")
w("|---|---|---|---|---|")
for c in sorted(cats):
    sg = sign_cats[c]
    m2 = sum(len(t) >= 2 for t in sg.values())
    w(f"| {c} | {len(sg)} | {len(cats[c])} | {m2} | {len(cats[c]) / len(sg):.2f} |")
w("")
w("**Signers:** the number of distinct signers cannot be established from the data. The manifest has no signer "
  "field. The only hint is the Vimeo `profile_id` in the video URL, which exists only for some clips "
  f"(Numbers carry values {dict(prof_num)}) and probably identifies an upload account, not a person. "
  f"Over all clips: {dict(prof_all)}. Treat the corpus as **one signer (possibly a few)**.\n")
w("## 6. Limitations (be honest in the demo)\n")
w("- About one take per sign; the 'real held-out take' test exists for only a minority of signs.")
w("- One signer (probably), studio conditions: plain background, good lighting, frontal camera, no webcam "
  "noise. Accuracy on the training dictionary measures robustness to speed/position/angle augmentation, "
  "**not** generalisation to a new signer. Expect lower accuracy on a real user's webcam.")
w("- Landmark quality varies: some two-handed signs lose a hand in many frames.")
w("- Labels are the site's English slugs and contain typos (e.g. `Naphew`) and category mix-ups "
  "(e.g. `Absher Card` listed under Health). Not corrected, never translated.")
w("- Demo phrases are word sequences for scripting, not grammatical sentences.\n")
w("## 7. Suggested demo vocabularies (`data/vocabulary.json`)\n")
for name, v in vocab["sets"].items():
    w(f"### `{name}` - {v['description']} ({len(v['signs'])} signs)\n")
    w("| Sign | Category | Takes | Good takes | Quality |")
    w("|---|---|---|---|---|")
    for s in v["signs"]:
        w(f"| {s['label']} | {s['category']} | {s['takes']} | {s['good_takes']} | {s['quality']:.2f} |")
    w("\nDraft phrases (need native-signer review):\n")
    for ph in v["phrases"]:
        w(f"- {ph['text']}")
    w("")
w("Use with `python scripts/train.py --vocab <name>`.")
(D / "dataset_report.md").write_text("\n".join(R) + "\n", encoding="utf-8")
print("wrote report; clips", o["n"], "bad", o["bad"], "signs", n_signs, "multi", multi)
