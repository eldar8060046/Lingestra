"""Rank candidate demo SENTENCES made of dictionary signs by how well Lingestra decodes them.

For every candidate (a list of dictionary labels, case-insensitive) the script
  * picks the best-quality clip of each word (landmark quality, see lingestra/labels.py),
  * enrols the words plus N unrelated 'distractor' signs (so look-alike confusion is measured),
  * signs the sentence back to back WITHOUT lowering the hands (random speeds 0.6x/1x/1.6x, short
    pauses, jitter - the continuous-signing benchmark generator) and decodes it with the real Recogniser,
  * reports sentence exact-match rate and word accuracy.

IMPORTANT: this is the SAME-CLIP case (template = the clip that is replayed), i.e. an upper bound that
tells you which sentences are mechanically easy and which words get confused with each other. It is not
a prediction of accuracy on your webcam - record your own examples for that.

Usage: python scripts/find_sentence.py [--seqs 12] [--distractors 30] [--seed 9001]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lingestra import features as F          # noqa: E402
from lingestra import labels as L            # noqa: E402
import benchmark_continuous as C             # noqa: E402

CANDIDATES = [
    ["Mother", "Cooking", "Biryani"],             # Mother cooks biryani
    ["Father", "Love", "Mother"],                 # Father loves Mother
    ["Doctor", "helps", "Mother"],                # The doctor helps Mother
    ["Mother", "wants", "Medicine"],              # Mother wants medicine
    ["Mother", "Headache", "Call", "Doctor"],     # Mother has a headache, call the doctor
    ["Teacher", "Read", "Book"],                  # The teacher reads a book
    ["Sister", "Read", "Book"],                   # Sister reads a book
    ["Brother", "wants", "drink"],                # Brother wants a drink
    ["Father", "driving", "Hospital wards"],      # (control: odd)
    ["Teacher", "Explain", "Exam"],               # The teacher explains the exam
    ["Mother", "drink", "Medicine"],              # Mother drinks medicine
    ["Brother", "Write", "Name"],                 # Brother writes (his) name
    ["Doctor", "Wait"],                           # Doctor, wait
    ["How Are You", "Fine"],                      # How are you? Fine.
    ["Father", "work", "School"],                 # Father works (at the) school
    ["Mother", "Love", "Family"],                 # Mother loves (the) family
]


def load_pool():
    """{lower label: (label, [raw takes, best quality first])} for usable clips only."""
    man = L.load_manifest() if hasattr(L, "load_manifest") else json.load(open(ROOT / "data" / "manifest.json"))
    by = {}
    for e in man:
        try:
            st, ok = L.clip_quality(e["id"])
        except Exception:
            continue
        if not ok:
            continue
        raw = np.load(ROOT / "data" / "landmarks" / f"{e['id']}.npy")
        score = float(st["any_hand"]) * min(float(st["pose"]), 1.0)
        by.setdefault(e["label"].lower(), (e["label"], []))[1].append((score, raw, e["category"]))
    pool = {}
    for k, (lab, items) in by.items():
        items.sort(key=lambda t: -t[0])
        pool[k] = (lab, [t[1] for t in items], items[0][0])
    return pool


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", type=int, default=12)
    ap.add_argument("--distractors", type=int, default=30)
    ap.add_argument("--seed", type=int, default=9001)
    ap.add_argument("--only", help="comma-separated indexes of CANDIDATES to run")
    a = ap.parse_args()
    pool = load_pool()
    rng0 = np.random.default_rng(a.seed)
    keys = sorted(pool)
    rows = []
    sel = range(len(CANDIDATES)) if not a.only else [int(x) for x in a.only.split(",")]
    for ci in sel:
        words = CANDIDATES[ci]
        miss = [w for w in words if w.lower() not in pool]
        if miss:
            print(f"skip {words}: not usable/unknown: {miss}")
            continue
        used = {w.lower() for w in words}
        others = [k for k in keys if k not in used]
        pick = rng0.permutation(len(others))[: a.distractors]
        vocab = [(pool[w.lower()][0], pool[w.lower()][1]) for w in words]
        vocab += [(pool[others[i]][0], pool[others[i]][1]) for i in pick]
        rec = C.build_rec(vocab, True)
        rng = np.random.default_rng(a.seed + ci)
        ok = 0
        S = D = I = 0
        n = 0
        clarif = 0
        for _ in range(a.seqs):
            spans = [F.trim_active(vocab[i][1][0]) for i in range(len(words))]
            if any(len(s) < 12 for s in spans):
                break
            stream, _ = C.build_sequence(spans, vocab[0][1][0][0], vocab[len(words) - 1][1][0][-1], rng)
            if len(F.segment_signing(stream)) != 1:
                continue
            ev, _ = C.run_stream(rec, stream)
            hyp = C.words_of(ev)
            s, d, i = C.align([v[0] for v in vocab[: len(words)]], hyp)
            S, D, I, n = S + s, D + d, I + i, n + 1
            ok += hyp == [v[0] for v in vocab[: len(words)]]
            clarif += sum(e.kind in ("clarify", "unknown") for e in ev) > 0
        nw = max(n * len(words), 1)
        q = min(pool[w.lower()][2] for w in words)
        rows.append(dict(sentence=" ".join(words), n=n, exact=ok / max(n, 1), word_acc=1 - (S + D + I) / nw,
                         sub=S / nw, dele=D / nw, ins=I / nw, asked=clarif / max(n, 1), min_quality=q,
                         clips=[len(pool[w.lower()][1]) for w in words]))
        r = rows[-1]
        print(f"{r['sentence']:<34} exact {r['exact']:.0%}  words {r['word_acc']:.0%}  "
              f"(S{r['sub']:.0%} D{r['dele']:.0%} I{r['ins']:.0%})  asked {r['asked']:.0%}  "
              f"worst-clip quality {q:.2f}  clips/word {r['clips']}", flush=True)
    out = ROOT / "scratch" / "sentence_ranking.json"
    out.write_text(json.dumps(rows, indent=1))
    print("\nbest first:")
    for r in sorted(rows, key=lambda r: (-r["exact"], -r["word_acc"], -r["min_quality"])):
        print(f"  {r['exact']:.0%} exact | {r['word_acc']:.0%} words | {r['sentence']}")


if __name__ == "__main__":
    main()
