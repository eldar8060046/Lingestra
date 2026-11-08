"""Optional multi-core, compiled version of the SPOTTING step of continuous signing (matcher.spot_batch).

Spotting compares the live movement with every stored example of every sign (subsequence DTW). In numpy it is a few
thousand small array operations that all hold Python's global lock, so more Python threads do NOT help (measured: 0.6-1.0x).
Here the same maths is compiled with numba and the stored examples are split across all cores (`prange`), with no
(examples x frames x frames) cost array in memory.

  * Optional: if numba is not installed, `available()` is False and the numpy code in matcher.py is used unchanged.
  * Same result: identical local cost, step set, step penalty and tie-breaking as `matcher._spot_chunk`
    (tests/test_fastspot.py compares them); differences are float rounding only.
  * `LINGESTRA_NO_NUMBA=1` switches it off; `LINGESTRA_THREADS=n` sets the number of cores it may use (default: up to 8, always
    leaving a few cores for the camera / hand tracking / window).
"""
import os
import threading

import numpy as np

try:                                                   # optional dependency
    if os.environ.get("LINGESTRA_NO_NUMBA"):
        raise ImportError("switched off by LINGESTRA_NO_NUMBA")
    import numba
    from numba import njit, prange
    _OK = True
except Exception:                                      # numba missing or broken: the numpy path is used
    _OK = False

INF = np.float32(1e9)
_LOCK = threading.Lock()                               # numba's default threading layer is not re-entrant
_WARM = {"started": False, "done": threading.Event(), "ok": False}


def available() -> bool:
    return _OK


def ready() -> bool:
    """The compiled kernel exists and works (warm_up finished): safe to use without a compile stall."""
    return _OK and _WARM["ok"]


def n_threads() -> int:
    if not _OK:
        return 1
    want = os.environ.get("LINGESTRA_THREADS")
    cores = os.cpu_count() or 2
    n = int(want) if want and want.isdigit() else max(1, min(8, cores - 4))
    return max(1, min(n, numba.config.NUMBA_NUM_THREADS))


if _OK:
    @njit(parallel=True, cache=True, nogil=True)
    def _kernel(DX, DM, dlens, QX, QM, W, SC, S0, S1, c_miss, pen, out_d, out_s):
        """out_d[k, j] / out_s[k, j]: normalised distance of the best alignment of example k that ENDS at query point j,
        and the point where it starts (identical to matcher._spot_chunk)."""
        N = DX.shape[0]
        Lq = QX.shape[0]
        NBk = W.shape[0]
        for k in prange(N):
            L = dlens[k]
            D = np.empty(Lq, np.float32)
            P = np.empty(Lq, np.float32)
            S = np.empty(Lq, np.int64)
            D2 = np.empty(Lq, np.float32)
            P2 = np.empty(Lq, np.float32)
            S2 = np.empty(Lq, np.int64)
            row = np.empty(Lq, np.float32)
            for i in range(L):
                # local cost of template frame i against every query frame (same as matcher.frame_cost)
                for j in range(Lq):
                    num = 0.0
                    den = 0.0
                    for b in range(NBk):
                        a_has = DM[k, i, b]
                        q_has = QM[j, b]
                        if a_has and q_has:
                            acc = 0.0
                            for d in range(S0[b], S1[b]):
                                diff = DX[k, i, d] - QX[j, d]
                                acc += diff * diff
                            num += W[b] * np.sqrt(acc) * SC[b]
                            den += W[b]
                        elif a_has or q_has:
                            num += W[b] * c_miss
                            den += W[b]
                    row[j] = num / max(den, 1e-6)
                if i == 0:
                    for j in range(Lq):
                        D[j] = row[j]
                        S[j] = j
                        P[j] = 1.0
                else:
                    for j in range(Lq):
                        # the three predecessors, in the order of matcher._spot_chunk: stay (+pen), diagonal, skip two (+pen);
                        # ties go to the first (np.argmin)
                        best = D[j] + pen
                        bp = P[j]
                        bs = S[j]
                        if j >= 1:
                            v = D[j - 1]
                            if v < best:
                                best = v
                                bp = P[j - 1]
                                bs = S[j - 1]
                        if j >= 2:
                            v = D[j - 2] + pen
                            if v < best:
                                best = v
                                bp = P[j - 2]
                                bs = S[j - 2]
                        else:
                            pass
                        D2[j] = best + row[j]
                        P2[j] = bp + 1.0
                        S2[j] = bs
                    for j in range(Lq):
                        D[j] = D2[j]
                        P[j] = P2[j]
                        S[j] = S2[j]
            for j in range(Lq):
                out_d[k, j] = D[j] / P[j]
                out_s[k, j] = S[j]


def _tables(cfg=None):
    """Block constants of the sign features (weights, 1/(sqrt(dim)*T), slice bounds)."""
    from lingestra.live import matcher as M
    SC = np.array([1.0 / (np.sqrt(sl.stop - sl.start) * float(t)) for sl, t in zip(M._SL, M._T)], np.float64)
    return (np.asarray(M._W, np.float64), SC,
            np.array([sl.start for sl in M._SL], np.int64), np.array([sl.stop for sl in M._SL], np.int64))


def spot_batch(DX, DM, dlens, Q, cfg):
    """Drop-in for matcher.spot_batch: (dist (N, Lq) float32, start (N, Lq) int)."""
    N, Lq = len(DX), len(Q.X)
    out_d = np.empty((N, Lq), np.float32)
    out_s = np.empty((N, Lq), np.int64)
    W, SC, S0, S1 = _tables()
    DXc = np.ascontiguousarray(DX, np.float32)
    DMc = np.ascontiguousarray(DM, np.bool_)
    QXc = np.ascontiguousarray(Q.X, np.float32)
    QMc = np.ascontiguousarray(Q.M, np.bool_)
    dl = np.ascontiguousarray(dlens, np.int64)
    with _LOCK:
        numba.set_num_threads(n_threads())
        _kernel(DXc, DMc, dl, QXc, QMc, W, SC, S0, S1, float(cfg.c_miss), np.float32(cfg.step_pen), out_d, out_s)
    return out_d, out_s


def warm_up(background=True):
    """Compile (or load from the on-disk cache) the kernel with a tiny example so that the first real decode does not
    stall for seconds. Safe to call many times."""
    if not _OK or _WARM["started"]:
        if not background:
            _WARM["done"].wait()
        return
    _WARM["started"] = True

    def run():
        try:
            from lingestra.live import matcher as M
            nb, dim = M.NB, M.F.SIGN_DIM
            DX = np.zeros((2, 4, dim), np.float32)
            DM = np.ones((2, 4, nb), np.bool_)
            Q = M._Seq(np.zeros((6, dim), np.float32), np.ones((6, nb), np.bool_))
            spot_batch(DX, DM, np.array([4, 3]), Q, M.MatcherConfig())
            _WARM["ok"] = True
        except Exception:
            _WARM["ok"] = False                       # the numpy path keeps working
        finally:
            _WARM["done"].set()
    if background:
        threading.Thread(target=run, name="lingestra-warm", daemon=True).start()
    else:
        run()


def wait_warm(timeout=None):
    return _WARM["done"].wait(timeout)
