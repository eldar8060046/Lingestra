"""Extract MediaPipe landmarks for every downloaded dictionary video (resumable).

Stores raw per-clip landmarks in data/landmarks/<id>.npy (frames, N_PTS, 3) so training can
re-run preprocessing/augmentation without re-running MediaPipe.
"""
import json, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lingestra.features import ROOT, LandmarkExtractor, extract_video

man = json.load(open(ROOT / "data" / "manifest.json"))
out_dir = ROOT / "data" / "landmarks"; out_dir.mkdir(exist_ok=True)
done = 0
for e in man:
    src, dst = ROOT / "data" / "raw_videos" / e["file"], out_dir / f"{e['id']}.npy"
    if dst.exists() or not src.exists():
        continue
    ex = LandmarkExtractor(video_mode=True)     # fresh timestamps per video
    raw, fps = extract_video(src, ex); ex.close()
    np.save(dst, raw); done += 1
    print(f"{e['id']:>6} {e['category']:<22} {e['label']:<24} frames={len(raw)} fps={fps:.0f} "
          f"hands_seen={(~np.isnan(raw[:,25:,0])).any(axis=1).mean():.0%}", flush=True)
print("extracted", done)
