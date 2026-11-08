"""Benchmark for CONTINUOUS signing: several signs in a row, hands never lowered.

    python scripts/benchmark_continuous.py                 # evaluation seeds, prints a markdown report
    python scripts/benchmark_continuous.py --tune          # tuning seeds (use these to choose parameters)
    python scripts/benchmark_continuous.py --no-continuous # the old behaviour (hold-only splitting)
    python scripts/benchmark_continuous.py --json out.json # also dump the numbers

Construction of a test stream (see docs/recognition.md, 'Continuous signing'):
  * a vocabulary of 30 signs (random, fixed seed) out of the dictionary signs with >= 2 good takes;
    the enrolled template is take 1 (all of it, the matcher trims it itself);
  * the query signs are the SIGNING SPANS (features.trim_active) of take 2 (cross-take) or of take 1
    again (same-take: the easy upper bound), 2 or 3 different signs per sequence;
  * the spans are joined into ONE landmark stream with NO hold: between the end pose of one sign and
    the start pose of the next the hands/arms are interpolated over 0.1-0.4 s (a synthetic
    'movement epenthesis'); the hands stay up. Optional per-sign speed (0.6x / 1x / 1.6x), short pauses
    (0.2-0.4 s, below the 0.6 s hold threshold) and landmark jitter;
  * a raise-from-rest lead-in and a lower-to-rest lead-out make the zone tracker see ONE episode
    (checked: sequences that do not give exactly one episode are counted and reported).
HONEST CAVEAT: every dictionary clip was performed in isolation, starting and ending at rest, so
these joins are a PROXY for real co-articulation (the real hand does not travel on a straight line
from one end pose to the next start pose). The real test is the user's webcam.

Metrics: sequence exact-match rate, word accuracy = 1 - (S + D + I) / N from a Levenshtein alignment of
the emitted 'sign' words with the reference, the share of sequences / reference words that produced a
'clarify' or 'unknown' event, single-sign false-split rate (a single sign, multi-pose included, that
yields more than one word / question / unknown event), and the decode time of the whole episode.
Everything is deterministic (fixed seeds); tuning seeds and evaluation seeds are disjoint.
"""
import argparse
import json
import sys
import time
import warnings
import zlib
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
warnings.filterwarnings("ignore")
from lingestra import features as F                                   # noqa: E402
import benchmark_recognition as B                                     # noqa: E402
from lingestra.live.recognizer import Recognizer                      # noqa: E402

FPS = 25.0
TUNE_SEEDS = (101, 102, 103)
# 2001-2005 were used while iterating (they are NOT held out any more); these are fresh. Do not tune on them.
EVAL_SEEDS = (7101, 7102, 7103, 7104, 7105)


# ----------------------------------------------------------------------------- stream construction
def _blend(a, b, n):
    """n frames moving from frame a towards frame b (exclusive of b); NaN follows the nearer frame."""
    w = np.linspace(0, 1, n + 1, endpoint=False)[1:, None, None].astype(np.float32)
    out = a[None] * (1 - w) + b[None] * w
    return np.where(np.isnan(out), np.where(w < 0.5, a[None], b[None]), out).astype(np.float32)


def _pause(frame, n, rng):
    """Hold a pose for n frames with the same small drift + jitter as real held hands."""
    drift = np.cumsum(rng.normal(0, 0.0015, (n, 1, 3)), 0).astype(np.float32) * np.array([1, 1, 0], np.float32)
    jit = rng.normal(0, 0.004, (n,) + frame.shape).astype(np.float32) * np.array([1, 1, 0], np.float32)
    return np.repeat(frame[None], n, 0) + drift + jit


def build_sequence(spans, rest_a, rest_b, rng, speeds=True, pauses=True, jitter=True):
    """spans: list of raw signing spans (n, 67, 3). Returns the stream (frames) with lead-in / lead-out
    plus the number of frames of the chain itself (for reporting)."""
    parts = []
    for i, s in enumerate(spans):
        sp = float(rng.choice([0.6, 1.0, 1.6])) if speeds else 1.0
        s = B.time_scale(s, sp) if sp != 1.0 else s
        if i > 0:
            prev = parts[-1][-1]
            if pauses and rng.random() < 0.4:
                parts.append(_pause(prev, int(rng.uniform(0.2, 0.4) * FPS), rng))
                prev = parts[-1][-1]
            parts.append(_blend(prev, s[0], max(int(rng.uniform(0.1, 0.4) * FPS), 2)))
        parts.append(s)
    chain = np.concatenate(parts)
    if jitter:
        chain = chain + rng.normal(0, 0.003, chain.shape).astype(np.float32) * np.array([1, 1, 0], np.float32)
    m = int(0.45 * FPS)
    lead = np.repeat(rest_a[None], int(1.0 * FPS), 0)
    tail = np.repeat(rest_b[None], int(2.4 * FPS), 0)
    stream = np.concatenate([lead, _blend(rest_a, chain[0], m), chain, _blend(chain[-1], rest_b, m), tail])
    return stream.astype(np.float32), len(chain)


# ----------------------------------------------------------------------------- scoring
def align(ref, hyp):
    """Levenshtein alignment -> (substitutions, deletions, insertions)."""
    n, m = len(ref), len(hyp)
    d = np.zeros((n + 1, m + 1), int)
    d[:, 0] = np.arange(n + 1)
    d[0, :] = np.arange(m + 1)
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i, j] = min(d[i - 1, j] + 1, d[i, j - 1] + 1, d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]))
    i, j, s, de, ins = n, m, 0, 0, 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i, j] == d[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            s += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and d[i, j] == d[i - 1, j] + 1:
            de += 1
            i -= 1
        else:
            ins += 1
            j -= 1
    return int(s), int(de), int(ins)


def run_stream(rec, stream):
    """Push a stream through the recogniser; returns (events, per-frame ms list, total ms)."""
    rec.reset()
    ev, lat = [], []
    for i, f in enumerate(stream):
        t0 = time.perf_counter()
        ev += rec.push_frame(f, i / FPS)
        lat.append((time.perf_counter() - t0) * 1000)
    return ev, lat


def words_of(ev):
    return [e.label for e in ev if e.kind == "sign"]


# ----------------------------------------------------------------------------- experiments
def make_vocab(pool, size, seed):
    eligible = [(lab, t) for lab, t in pool if len(t) >= 2]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(eligible))[:size]
    return [eligible[i] for i in idx]


def build_rec(vocab, continuous, mcfg=None):
    from lingestra.live.matcher import MatcherConfig
    rec = Recognizer(None, templates=[dict(id=str(i), label=lab, frames=[t[0]])
                                      for i, (lab, t) in enumerate(vocab)],
                     matcher_config=mcfg or MatcherConfig())
    if hasattr(rec, "continuous"):
        rec.continuous = bool(continuous)
    elif continuous:
        raise SystemExit("this recogniser has no continuous mode")
    return rec


def matchable_signs(vocab, mcfg=None):
    """Indices of vocabulary signs that the ordinary single-sign matcher answers correctly when the
    cross-take (take 2) clip is shown on its own. The dictionary's second take is often a different
    performance of the sign (see docs/recognition.md), so most cross-take failures are NOT about
    continuity; restricting the sequences to these signs isolates what continuous decoding adds
    (the other enrolled signs stay in the vocabulary as distractors)."""
    from lingestra.live.matcher import MatcherConfig, PersonalMatcher
    m = PersonalMatcher(mcfg or MatcherConfig())
    m.set_templates([dict(id=str(i), label=lab, frames=[t[0]]) for i, (lab, t) in enumerate(vocab)])
    return [i for i, (lab, t) in enumerate(vocab) if m.classify(F.trim_active(t[1]), trim=False).label == lab]


def run_sequences(pool, vocab, rec, n_signs, take, n_seq, seed, only=None, **kw):
    """Continuous sequences of n_signs different vocabulary signs. take 0 = same-take, 1 = cross-take.
    `only`: indices the sequence signs are drawn from (default all)."""
    rng = np.random.default_rng(seed)
    cand = np.array(list(range(len(vocab))) if only is None else only)
    if len(cand) < n_signs:
        return None
    S = D = I = N = exact = any_q = n_q = skipped = 0
    ep_bad = 0
    lat_all, dec_ms = [], []
    n_done = 0
    tries = 0
    while n_done < n_seq and tries < 4 * n_seq:
        tries += 1
        pick = cand[rng.permutation(len(cand))[:n_signs]]
        spans = [F.trim_active(vocab[i][1][take]) for i in pick]
        ref = [vocab[i][0] for i in pick]
        if len(set(ref)) < n_signs or any(len(s) < 12 for s in spans):
            continue
        stream, _ = build_sequence(spans, vocab[pick[0]][1][take][0], vocab[pick[-1]][1][take][-1], rng, **kw)
        if len(F.segment_signing(stream)) != 1:
            ep_bad += 1                      # the zone tracker did not see ONE episode: not a valid test stream
            continue
        n_done += 1
        ev, lat = run_stream(rec, stream)
        hyp = words_of(ev)
        s, d, i = align(ref, hyp)
        S, D, I, N = S + s, D + d, I + i, N + len(ref)
        exact += hyp == ref
        q = sum(e.kind in ("clarify", "unknown") for e in ev)
        any_q += q > 0
        n_q += q
        lat_all.append(max(lat))
        dec_ms.append(sum(lat))
    n = max(n_done, 1)
    return dict(n_seq=n_done, bad_episode=ep_bad, seq_acc=exact / n, word_acc=1 - (S + D + I) / max(N, 1),
                sub=S / max(N, 1), dele=D / max(N, 1), ins=I / max(N, 1),
                q_seq=any_q / n, q_per_word=n_q / max(N, 1), max_frame_ms=float(np.mean(lat_all)) if lat_all else 0,
                worst_frame_ms=float(np.max(lat_all)) if lat_all else 0)


def run_single(pool, vocab, rec, take, seed, unenrolled=0):
    """Single signs through the same stream builder (lead-in / out only). false_split = more than one
    word / clarify / unknown event for ONE sign; also exact-correct rate. `unenrolled` > 0 adds that many
    un-enrolled dictionary signs and reports how many produced a word (false accept)."""
    rng = np.random.default_rng(seed)
    n = split = ok = 0
    for lab, t in vocab:
        span = F.trim_active(t[take])
        if len(span) < 12:
            continue
        stream, _ = build_sequence([span], t[take][0], t[take][-1], rng, speeds=False, pauses=False)
        if len(F.segment_signing(stream)) != 1:
            continue
        ev, _ = run_stream(rec, stream)
        k = sum(e.kind in ("sign", "clarify", "unknown") for e in ev)
        n += 1
        split += k > 1
        ok += words_of(ev) == [lab]
    out = dict(n=n, false_split=split / max(n, 1), correct=ok / max(n, 1))
    if unenrolled:
        have = {lab for lab, _ in vocab}
        others = [(lab, t) for lab, t in pool if lab not in have]
        sel = np.random.default_rng(seed + 5).permutation(len(others))[:unenrolled]
        acc = words = 0
        for i in sel:
            lab, t = others[i]
            span = F.trim_active(t[0])
            if len(span) < 12:
                continue
            stream, _ = build_sequence([span], t[0][0], t[0][-1], rng, speeds=False, pauses=False)
            ev, _ = run_stream(rec, stream)
            w = words_of(ev)
            acc += len(w) > 0
            words += len(w)
        out.update(unenrolled=len(sel), unenrolled_word=acc / len(sel), unenrolled_words=words / len(sel))
    return out


def decode_timing(vocab_big, seconds=8.0):
    """Decode time for an ~8 s episode with 50 signs x 3 enrolled takes (direct matcher call). The query is a
    take that is NOT enrolled (pruning is not helped by an exact copy of a template)."""
    from lingestra.live.matcher import PersonalMatcher
    if not hasattr(PersonalMatcher, "decode_sequence"):
        return None
    elig = vocab_big[:50]
    m = PersonalMatcher()
    m.set_templates([dict(id=str(i), label=lab, frames=[t[0], t[0][::2], t[0][::3]]) for i, (lab, t) in enumerate(elig)])
    rng = np.random.default_rng(7)
    ts, worst = [], 0.0
    for r in range(6):
        pick = rng.permutation(len(elig))[:4]
        spans = [F.trim_active(elig[i][1][1]) for i in pick]
        st, _ = build_sequence(spans, elig[pick[0]][1][1][0], elig[pick[-1]][1][1][-1], rng)
        ep = F.segment_signing(st)[0].frames
        while len(ep) < seconds * FPS:                           # stretch to ~8 s by repeating signs
            spans.append(spans[len(spans) % 4])
            st, _ = build_sequence(spans, elig[pick[0]][1][1][0], elig[pick[-1]][1][1][-1], rng)
            ep = F.segment_signing(st)[0].frames
        ep = ep[:int(seconds * FPS)]
        runs = []
        for _ in range(3):
            t0 = time.perf_counter()
            m.decode_sequence(ep)
            runs.append((time.perf_counter() - t0) * 1000)
        ts.append(float(np.median(runs)))                         # median of 3 (no best-of-3 flattering)
        worst = max(worst, max(runs))
    return dict(ms_mean=float(np.mean(ts)), ms_max=float(worst), frames=len(ep), templates=m.n_takes)


def pct(x):
    return f"{100 * x:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="use the tuning seeds (never report these)")
    ap.add_argument("--no-continuous", action="store_true", help="old hold-only behaviour (the baseline)")
    ap.add_argument("--n", type=int, default=30, help="sequences per (kind, take, seed)")
    ap.add_argument("--vocab", type=int, default=30)
    ap.add_argument("--json", default=None)
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    seeds = TUNE_SEEDS if a.tune else EVAL_SEEDS
    if a.quick:
        seeds, a.n = seeds[:1], 10
    pool = B.load_pool()
    t_start = time.time()
    cont = not a.no_continuous
    rows = {}
    kinds = ((0, "same-take", False), (1, "cross-take", False), (1, "cross-take, matchable signs", True))
    for take, tname, only_m in kinds:
        for ns in (2, 3):
            acc = []
            for sd in seeds:
                vocab = make_vocab(pool, a.vocab, sd)
                rec = build_rec(vocab, cont)
                only = matchable_signs(vocab) if only_m else None
                r = run_sequences(pool, vocab, rec, ns, take, a.n, sd * 7 + ns + 10 * take, only=only)
                if r is not None:
                    acc.append(r)
            rows[(tname, ns)] = {k: float(np.mean([r[k] for r in acc])) for k in acc[0]}
            rows[(tname, ns)]["n_seq"] = int(sum(r["n_seq"] for r in acc))
            rows[(tname, ns)]["bad_episode"] = int(sum(r["bad_episode"] for r in acc))
            rows[(tname, ns)]["worst_frame_ms"] = float(max(r["worst_frame_ms"] for r in acc))
    # single signs
    single = {}
    for take, tname in ((0, "same-take"), (1, "cross-take")):
        acc = []
        for sd in seeds:
            vocab = make_vocab(pool, a.vocab, sd)
            rec = build_rec(vocab, cont)
            acc.append(run_single(pool, vocab, rec, take, sd, unenrolled=40 if take == 1 else 0))
        single[tname] = {k: float(np.mean([r[k] for r in acc])) for k in acc[0]}
    timing = decode_timing([(lab, t) for lab, t in pool if len(t) >= 2])
    mode = "continuous" if cont else "baseline (continuous OFF)"
    print(f"# Continuous-signing benchmark: {mode}, {'tuning' if a.tune else 'evaluation'} seeds {list(seeds)}, "
          f"vocabulary {a.vocab} signs, 1 enrolled take each ({time.time() - t_start:.0f} s)\n")
    hdr = ["stream", "sequences", "sequence exact", "word accuracy", "substituted", "deleted", "inserted",
           "with clarify/unknown", "worst frame ms"]
    print("| " + " | ".join(hdr) + " |\n|" + "---|" * len(hdr))
    for (tname, ns), r in rows.items():
        print(f"| {ns} signs, {tname} | {r['n_seq']} | {pct(r['seq_acc'])} | {pct(r['word_acc'])} | {pct(r['sub'])} | "
              f"{pct(r['dele'])} | {pct(r['ins'])} | {pct(r['q_seq'])} | {r['worst_frame_ms']:.0f} |")
    print("\n| single sign (no join) | signs | wrongly split into >1 events | exactly the right word | un-enrolled -> a word |\n|---|---|---|---|---|")
    for tname, r in single.items():
        print(f"| {tname} | {r['n']:.0f} per vocab | {pct(r['false_split'])} | {pct(r['correct'])} | "
              f"{pct(r['unenrolled_word']) if 'unenrolled_word' in r else '-'} |")
    if timing:
        print(f"\ndecode_sequence, {timing['frames']} frames, {timing['templates']} templates: "
              f"{timing['ms_mean']:.0f} ms mean / {timing['ms_max']:.0f} ms max")
    if a.json:
        Path(a.json).write_text(json.dumps(dict(rows={f"{k[0]}|{k[1]}": v for k, v in rows.items()},
                                                single=single, timing=timing), indent=1))


if __name__ == "__main__":
    main()
