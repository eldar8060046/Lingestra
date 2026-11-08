import sys, time, glob
sys.path.insert(0, "/home/eldar/Desktop/Lingestra")
import cv2, numpy as np
from concurrent.futures import ThreadPoolExecutor
from lingestra import features as F
vid = sorted(glob.glob("/home/eldar/Desktop/Lingestra/data/raw_videos/*.mp4"))[3]
cap = cv2.VideoCapture(vid); frames = []
while len(frames) < 120:
    ok, bgr = cap.read()
    if not ok: break
    frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
ex = F.LandmarkExtractor(video_mode=True)
import mediapipe as mp
def timed(fn, n=100):
    t0 = time.perf_counter()
    for i in range(n): fn(i)
    return (time.perf_counter() - t0) / n * 1000
both = timed(lambda i: ex.process(frames[i], int(i * 40)))
def pose_only(i):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frames[i])); ex.pose.detect_for_video(img, 100000 + int(i * 40))
def hands_only(i):
    img = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(frames[i])); ex.hands.detect_for_video(img, 100000 + int(i * 40))
p = timed(pose_only); h = timed(hands_only)
print(f"per frame on this machine: pose+hands together {both:.1f} ms  (pose alone {p:.1f} ms, hands alone {h:.1f} ms)  -> max {1000/both:.0f} frames/s")
print("camera delivers 25-30 frames/s = 33-40 ms per frame")
