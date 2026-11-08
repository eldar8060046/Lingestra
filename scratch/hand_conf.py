import sys, json, glob
sys.path.insert(0, "/home/eldar/Desktop/Lingestra")
import cv2, numpy as np
from lingestra import features as F
man = json.load(open("/home/eldar/Desktop/Lingestra/data/manifest.json"))
want = {"One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5}
ids = [e for e in man if e["category"] == "Numbers" and e["label"] in want]
for conf in (0.5, 0.35, 0.2):
    ex = F.LandmarkExtractor(video_mode=True, hand_conf=conf)
    tot = det = n_false = 0
    per = {}
    for e in ids:
        cap = cv2.VideoCapture(f"/home/eldar/Desktop/Lingestra/data/raw_videos/{e['file']}")
        raw0 = np.load(f"/home/eldar/Desktop/Lingestra/data/landmarks/{e['id']}.npy")
        act = F.trim_active(raw0)
        # active span frames (use the stored 0.5-conf landmarks to find the signing span)
        have = ~np.isnan(raw0[:, F.N_POSE:, 0]).all(axis=1)
        idx = np.where(have)[0]
        lo, hi = (idx[0], idx[-1]) if len(idx) else (0, 0)
        i = 0; got = 0; span = 0
        ex2 = F.LandmarkExtractor(video_mode=True, hand_conf=conf)
        while True:
            ok, bgr = cap.read()
            if not ok: break
            out = ex2.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), int(i * 40))
            if lo <= i <= hi:
                span += 1; got += int(not np.isnan(out[F.N_POSE:, 0]).all())
            i += 1
        ex2.close(); cap.release()
        per[e["label"]] = got / max(span, 1)
    print(f"hand confidence {conf}: hand found in the signing frames: " + "  ".join(f"{k} {v:.0%}" for k, v in per.items()) + f"   mean {np.mean(list(per.values())):.0%}", flush=True)
