import sys, json
sys.path.insert(0, "/home/eldar/Desktop/Lingestra")
import numpy as np
from lingestra.live import fingers as FG
man = json.load(open("/home/eldar/Desktop/Lingestra/data/manifest.json"))
want = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
TIP = (8, 12, 16, 20); MCP = (5, 9, 13, 17)
def count_2d(h):      # the PREVIOUS flat reading (same thresholds, x/y only, thumb rule as now)
    w = h[0, :2]; r = [np.linalg.norm(h[t, :2] - w) / np.linalg.norm(h[m, :2] - w) for t, m in zip(TIP, MCP)]
    palm = np.linalg.norm(h[5, :2] - h[17, :2]); thumb = np.linalg.norm(h[4, :2] - h[5, :2]) / palm
    v = np.array(r + [thumb])
    return FG._standard_count(v)
def count_3d(h):
    return FG.count_fingers(h, None, 16 / 9)
WIN = 13
tot = {"2d": [0, 0], "3d": [0, 0]}
for e in man:
    if e["category"] == "Numbers" and e["label"] in want:
        raw = np.load(f"/home/eldar/Desktop/Lingestra/data/landmarks/{e['id']}.npy")
        hands = [f[46:67] for f in raw]
        ok = [not np.isnan(h[:, :3]).any() for h in hands]
        n = want[e["label"]]
        c2 = [count_2d(h) if o else None for h, o in zip(hands, ok)]
        c3 = [count_3d(h) if o else None for h, o in zip(hands, ok)]
        hit = [i for i in range(len(hands)) if c2[i] == n or c3[i] == n]
        if not hit: continue
        lo, hi = hit[0], hit[-1]
        res = {}
        for name, c in (("2d", c2), ("3d", c3)):
            wins = [all(c[j] == n for j in range(i, i + WIN)) for i in range(lo, hi - WIN + 2)]
            tot[name][0] += sum(wins); tot[name][1] += len(wins); res[name] = f"{np.mean(wins):.0%}" if wins else "-"
            res[name + "_frames"] = f"{np.mean([c[i] == n for i in range(lo, hi + 1)]):.0%}"
        print(f"{e['label']:<5} pose phase {hi-lo+1:>3} frames | correct on single frames: 2D {res['2d_frames']:>4}  3D {res['3d_frames']:>4} | "
              f"held steadily for 0.5 s: 2D {res['2d']:>4}  3D {res['3d']:>4}")
print(f"ALL    steady 0.5 s windows read correctly: 2D {tot['2d'][0]}/{tot['2d'][1]} = {tot['2d'][0]/max(tot['2d'][1],1):.0%}   3D {tot['3d'][0]}/{tot['3d'][1]} = {tot['3d'][0]/max(tot['3d'][1],1):.0%}")
