import numpy as np
from lingestra.live.matcher import *
from lingestra.live.matcher import _SL,_W,_T,_D
def cost_matrices(TX, TM, Q, c_miss):
    """Local costs (N, Lt, Lq) between every template frame and every query frame."""
    N, Lt, _ = TX.shape
    Lq = len(Q.X)
    num = np.zeros((N, Lt, Lq), np.float32)
    den = np.zeros((N, Lt, Lq), np.float32)
    for b in range(NB):
        sl = _SL[b]
        A = TX[:, :, sl].reshape(N * Lt, -1)
        B = Q.X[:, sl]
        sq = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
        d = np.sqrt(np.maximum(sq, 0.0) / (sl.stop - sl.start)).reshape(N, Lt, Lq) / _T[b]
        ta = TM[:, :, b][:, :, None]
        qa = Q.M[:, b][None, None, :]
        both = ta & qa
        one = ta ^ qa
        num += _W[b] * (np.where(both, d, 0.0) + np.where(one, c_miss, 0.0))
        den += _W[b] * (ta | qa)
    return num / np.maximum(den, 1e-6)


