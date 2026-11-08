import sys
from pathlib import Path
ROOT = Path("/home/eldar/Desktop/Lingestra")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import numpy as np
from lingestra import features as F
import benchmark_continuous as C, benchmark_recognition as B
FPS = C.FPS; pool = B.load_pool()

def probe(vocab, rec, idx, take, where, hold_s, rng):
    span = F.trim_active(vocab[idx][1][take])
    if len(span) < 14: return None
    n = int(hold_s * FPS)
    if where == "end":
        chain = np.concatenate([span, C._pause(span[-1], n, rng)])
    else:
        m = len(span) // 2
        chain = np.concatenate([span[:m], C._pause(span[m], n, rng), span[m:]])
    stream, _ = C.build_sequence([chain], vocab[idx][1][take][0], vocab[idx][1][take][-1], rng, speeds=False, pauses=False)
    rows = []
    orig = rec.matcher.decode_sequence
    def spy(*a, **k):
        segs = orig(*a, **k)
        if rec.zone.hold_active and segs:
            sg = segs[-1]
            rows.append((sg.kind, sg.label, float(sg.ratio)))
        return segs
    rec.matcher.decode_sequence = spy
    rec.reset(); ev = []
    for i, f in enumerate(stream):
        ev += rec.push_frame(f, i / FPS)
    rec.matcher.decode_sequence = orig
    return rows, [e.label for e in ev if e.kind == "sign"], vocab[idx][0]

res = {"end": [], "mid": []}
for seed in (101, 102, 103, 104):
    vocab = C.make_vocab(pool, 30, seed); rec = C.build_rec(vocab, True)
    cand = C.matchable_signs(vocab)
    rng = np.random.default_rng(seed)
    for idx in cand[:8]:
        for where in ("end", "mid"):
            for hold in (1.0, 1.5):
                r = probe(vocab, rec, idx, 1, where, hold, rng)
                if r and r[0]:
                    rows, signs, truth = r
                    best = min((x[2] for x in rows if x[0] == "sign" and x[1] == truth), default=np.nan)
                    wrong = [x for x in rows if x[0] == "sign" and x[1] != truth]
                    res[where].append((best, len(wrong) > 0, signs == [truth], hold))
for where, label in (("end", "HOLD AT THE END of the sign (should commit)"), ("mid", "PAUSE IN THE MIDDLE of a sign (must NOT commit early)")):
    a = res[where]
    ratios = np.array([x[0] for x in a])
    print(f"{label}: n={len(a)}  ratio of the right word during the hold: median {np.nanmedian(ratios):.2f}, "
          f"p25 {np.nanpercentile(ratios,25):.2f}, p75 {np.nanpercentile(ratios,75):.2f}, no right-word match at all {np.isnan(ratios).mean():.0%}; "
          f"a WRONG word offered during the hold {np.mean([x[1] for x in a]):.0%}; final sentence exactly right {np.mean([x[2] for x in a]):.0%}", flush=True)
