import sys, json
sys.path.insert(0, "/home/eldar/Desktop/Lingestra")
import numpy as np
from lingestra.live import fingers as FG
man = json.load(open("/home/eldar/Desktop/Lingestra/data/manifest.json"))
want = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
TIP = (8, 12, 16, 20); MCP = (5, 9, 13, 17)
def feats(h, aspect):
    h = h.copy(); h[:, 0] *= aspect
    w = h[0, :2]
    r = [np.linalg.norm(h[t, :2] - w) / np.linalg.norm(h[m, :2] - w) for t, m in zip(TIP, MCP)]
    palm = np.linalg.norm(h[5, :2] - h[17, :2])
    return np.array(r + [np.linalg.norm(h[4, :2] - h[5, :2]) / palm])
for e in man:
    if e["category"] == "Numbers" and e["label"] in want:
        raw = np.load(f"/home/eldar/Desktop/Lingestra/data/landmarks/{e['id']}.npy")
        hs = [f[46:67] for f in raw if FG.hands_count(f) == want[e["label"]]]
        if not hs: continue
        for aspect in (1.0, 640 / 360, 640 / 480):
            v = np.median([feats(h, aspect) for h in hs], 0)
            print(f"{e['label']:<5} aspect {aspect:.2f}: idx/mid/ring/pink {np.round(v[:4], 2)} thumb {v[4]:.2f}")
