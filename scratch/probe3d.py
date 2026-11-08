import sys, json
sys.path.insert(0, "/home/eldar/Desktop/Lingestra")
import numpy as np
from lingestra.live import fingers as FG
man = json.load(open("/home/eldar/Desktop/Lingestra/data/manifest.json"))
want = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
TIP = (8, 12, 16, 20); MCP = (5, 9, 13, 17)
ASPECT = 640 / 360
def feats(h, mode):
    h = h.astype(float).copy()
    if mode == "2d":
        w = h[0, :2]; d = lambda a: np.linalg.norm(a[:2] - w)
    else:                                   # 3D: x,z scaled by the aspect ratio, y as is
        h[:, 0] *= ASPECT; h[:, 2] *= ASPECT
        w = h[0]; d = lambda a: np.linalg.norm(a - w)
    return np.array([d(h[t]) / d(h[m]) for t, m in zip(TIP, MCP)])
for e in man:
    if e["category"] == "Numbers" and e["label"] in want:
        raw = np.load(f"/home/eldar/Desktop/Lingestra/data/landmarks/{e['id']}.npy")
        hs = [f[46:67] for f in raw if FG.hands_count(f) == want[e["label"]]]
        for mode in ("2d", "3d"):
            v = np.array([feats(h, mode) for h in hs])
            print(f"{e['label']:<5} {mode}: median {np.round(np.median(v, 0), 2)}  frame-to-frame std {np.round(v.std(0), 2)}")
# rotate a hand (pointing toward the camera): 2D shrinks, 3D does not
e = next(e for e in man if e["category"] == "Numbers" and e["label"] == "One")
raw = np.load(f"/home/eldar/Desktop/Lingestra/data/landmarks/{e['id']}.npy")
h = next(f[46:67] for f in raw if FG.hands_count(f) == 1).astype(float)
for deg in (0, 30, 50, 65):
    a = np.deg2rad(deg); c, s = np.cos(a), np.sin(a)
    g = h.copy(); g[:, 0] *= ASPECT; g[:, 2] *= ASPECT
    cen = g[0].copy(); g -= cen
    R = np.array([[1, 0, 0], [0, c, -s], [0, s, c]])        # tilt the hand toward the camera (rotate about the x axis)
    g = g @ R.T + cen
    g[:, 0] /= ASPECT; g[:, 2] /= ASPECT
    f2 = feats(g, "2d"); f3 = feats(g, "3d")
    print(f"index finger, hand tilted {deg:>2} deg toward the camera: 2D ratio {f2[0]:.2f} (up needs >=1.5)   3D ratio {f3[0]:.2f}")
