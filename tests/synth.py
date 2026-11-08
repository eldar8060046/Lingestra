"""Synthetic landmark streams for the engine tests (no data files, deterministic).

A 'sign' is a list of key poses (right-wrist position in image coords, finger curl 0..1); the
stream is rest -> raise -> key poses (linear moves, optional holds) -> lower -> rest. The body is a
fixed standing skeleton so the shoulder/hip reference is stable."""
import numpy as np

from lingestra import features as F

FPS = 25.0
REST_R = np.array([0.60, 0.74])       # right wrist hanging at the hip
REST_L = np.array([0.40, 0.74])
NOSE = np.array([0.50, 0.20])


def hand_points(wrist, curl, spread=0.0, scale=0.05):
    """21 hand landmarks (x, y, z) in image coords: palm up, fingers curling with `curl`."""
    pts = np.zeros((21, 3), np.float32)
    pts[0, :2] = wrist
    mcp_y = -scale
    for f, dx in enumerate((-0.6, -0.3, 0.0, 0.3, 0.6)):      # thumb .. pinky base (f=0 thumb)
        base = np.array([dx * scale * (1 + spread), mcp_y * (0.5 if f == 0 else 1.0)])
        ln = scale * (0.8 if f in (0, 4) else 1.0)
        base_idx = 1 + 4 * f
        for j in range(4):
            frac = (j + 1) / 4
            ext = base + np.array([0.0, -ln * frac * (1 - curl)])
            ext[1] += ln * frac * curl * 1.4 * frac          # curled tips fall back toward the palm
            pts[base_idx + j, :2] = wrist + ext + np.array([0, 0])
    pts[:, 2] = np.linspace(0, -0.02, 21).astype(np.float32) * (1 - 0.5 * curl)
    return pts


def frame(wr_r=REST_R, wr_l=REST_L, curl_r=0.0, curl_l=0.0, show_hands=True):
    f = np.full((F.N_PTS, 3), np.nan, np.float32)
    f[:, 2] = 0.0
    f[0, :2] = NOSE
    f[F.L_SH, :2], f[F.R_SH, :2] = (0.60, 0.32), (0.40, 0.32)     # person's left shoulder is image right
    f[F.L_HIP, :2], f[F.R_HIP, :2] = (0.57, 0.72), (0.43, 0.72)
    f[13, :2], f[14, :2] = (0.65, 0.5), (0.35, 0.5)
    f[F.L_WR, :2], f[F.R_WR, :2] = wr_l, wr_r
    f[13, :2] = (np.array(f[F.L_SH, :2]) + wr_l) / 2
    f[14, :2] = (np.array(f[F.R_SH, :2]) + wr_r) / 2
    f[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], :2] = NOSE + np.array([0.0, 0.0])
    for i in (17, 18, 19, 20, 21, 22):
        f[i, :2] = wr_r if i % 2 == 0 else wr_l
    if show_hands:
        f[F.N_POSE:F.N_POSE + 21] = hand_points(wr_l, curl_l)
        f[F.N_POSE + 21:F.N_PTS] = hand_points(wr_r, curl_r)
    return f


def _lerp(a, b, n):
    w = np.linspace(0, 1, n, endpoint=False)[:, None]
    return a[None] * (1 - w) + b[None] * w


def moves(keys, move_frames, hold_frames=0, hold_at=()):
    """Frames through key poses (x, y, curl); `hold_frames` repeated at the key indices in hold_at."""
    out = []
    for k in range(len(keys) - 1):
        out.append(_lerp(np.asarray(keys[k], float), np.asarray(keys[k + 1], float), move_frames))
        if (k + 1) in hold_at:
            out.append(np.repeat(np.asarray(keys[k + 1], float)[None], hold_frames, 0))
    out.append(np.asarray(keys[-1], float)[None])
    return np.concatenate(out)


def sign_frames(keys, speed=1.0, pause_s=0.0, pause_at=(), move_s=0.5, noise=0.0, rng=None, edge_hold_s=0.25):
    """Frames (n, 67, 3) of just the signing part."""
    mf = max(int(move_s * FPS / speed), 2)
    path = moves(keys, mf, int(pause_s * FPS), pause_at)
    eh = max(int(edge_hold_s * FPS), 1)                # the hand settles at the first / last pose
    path = np.concatenate([np.repeat(path[:1], eh, 0), path, np.repeat(path[-1:], eh, 0)])
    rng = rng or np.random.default_rng(0)
    out = []
    for x, y, c in path:
        w = np.array([x, y]) + (rng.normal(0, noise, 2) if noise else 0)
        out.append(frame(wr_r=w, curl_r=float(c)))
    return np.stack(out)


def stream(sign, lead_s=0.8, raise_s=0.5, lower_s=0.5, tail_s=1.0, rest_pose=None):
    """rest + raise + sign + lower + rest. The raise/lower are straight moves of the whole hand."""
    first, last = sign[0], sign[-1]
    rest = frame()
    parts = [np.repeat(rest[None], int(lead_s * FPS), 0)]
    n = int(raise_s * FPS)
    w = np.linspace(0, 1, n, endpoint=False)[:, None, None]
    parts.append(rest[None] * (1 - w) + first[None] * w)
    parts.append(sign)
    n = int(lower_s * FPS)
    w = np.linspace(0, 1, n, endpoint=False)[:, None, None]
    parts.append(last[None] * (1 - w) + rest[None] * w)
    parts.append(np.repeat(rest[None], int(tail_s * FPS), 0))
    return np.concatenate(parts).astype(np.float32), int(lead_s * FPS), int(raise_s * FPS), len(sign), n


# --- a small vocabulary of distinct synthetic signs: (x, y, curl) key poses of the right hand
SIGNS = {
    "alpha": [(0.62, 0.42, 0.0), (0.50, 0.24, 0.0), (0.38, 0.42, 0.0)],
    "beta": [(0.70, 0.50, 1.0), (0.70, 0.36, 1.0), (0.56, 0.36, 0.5)],
    "gamma": [(0.42, 0.28, 0.2), (0.56, 0.46, 0.8), (0.42, 0.54, 0.2), (0.64, 0.40, 0.9)],
    "delta": [(0.30, 0.50, 1.0), (0.30, 0.30, 0.0), (0.30, 0.50, 1.0), (0.45, 0.50, 0.0)],
}


def sign(name, **kw):
    return sign_frames(SIGNS[name], **kw)
