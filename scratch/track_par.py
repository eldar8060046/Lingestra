import sys, time, glob
sys.path.insert(0, "/home/eldar/Desktop/Lingestra")
import cv2, numpy as np
from lingestra import features as F
vid = sorted(glob.glob("/home/eldar/Desktop/Lingestra/data/raw_videos/*.mp4"))[3]
cap = cv2.VideoCapture(vid); frames = []
while len(frames) < 150:
    ok, bgr = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
res = {}
for par in (False, True, False, True):
    ex = F.LandmarkExtractor(video_mode=True, parallel=par)
    for i in range(10): ex.process(frames[i], i * 40)          # warm up
    ex2 = F.LandmarkExtractor(video_mode=True, parallel=par)
    t0 = time.perf_counter(); out = [ex2.process(f, i * 40) for i, f in enumerate(frames)]; dt = (time.perf_counter() - t0) / len(frames) * 1000
    res.setdefault(par, []).append((dt, np.stack(out)))
    ex.close(); ex2.close()
ser = np.mean([d for d, _ in res[False]]); par = np.mean([d for d, _ in res[True]])
same = np.allclose(res[False][0][1], res[True][0][1], equal_nan=True)
print(f"serial {ser:.1f} ms/frame, parallel {par:.1f} ms/frame -> {ser/par:.2f}x faster ({1000/par:.0f} frames/s); landmarks identical: {same}")
