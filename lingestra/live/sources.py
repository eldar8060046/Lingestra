"""Frame sources. Each is an iterable yielding (raw_landmarks (67,3) float32, t_seconds).

All sources run the same landmark pipeline (features.LandmarkExtractor) so what the recognizer
sees live is identical to what training saw. Sources are lazy: nothing heavy is imported or
opened until iteration starts.
"""
import sys
import time
from pathlib import Path

import numpy as np

from lingestra import features as F


class _CvSource:
    """Shared video->landmarks loop for webcam and video-file sources."""
    last_frame_bgr = None          # most recent BGR frame, for the UI preview
    aspect = None                  # frame width / height (finger readings are 3D and need it)
    _cap = None

    def close(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def __iter__(self):
        import cv2
        ex = F.LandmarkExtractor(video_mode=True)
        try:
            for bgr, t, ts_ms in self._frames():
                self.last_frame_bgr = bgr
                self.aspect = bgr.shape[1] / max(bgr.shape[0], 1)
                yield ex.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), ts_ms), t
        finally:
            ex.close()
            self.close()


class WebcamSource(_CvSource):
    """Live webcam. Timestamps are wall-clock seconds since start.

    Camera backends: we first try cv2.CAP_ANY (works on Linux V4L2, macOS AVFoundation and
    usually Windows). On Windows, if opening or reading is slow/fails (common with MSMF), the
    code retries with cv2.CAP_DSHOW automatically.
    """

    def __init__(self, index=0, width=640, height=480):
        self.index, self.width, self.height = index, width, height

    def _open(self):
        import cv2
        backends = [cv2.CAP_ANY] + ([cv2.CAP_DSHOW] if sys.platform.startswith("win") else [])
        for be in backends:
            cap = cv2.VideoCapture(self.index, be)
            if cap.isOpened():
                ok, _ = cap.read()
                if ok:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                    return cap
            cap.release()
        raise RuntimeError(f"cannot open camera {self.index}")

    def _frames(self):
        self._cap = self._open()
        t0 = time.monotonic()
        while self._cap is not None:
            ok, bgr = self._cap.read()
            if not ok:
                break
            t = time.monotonic() - t0
            yield bgr, t, int(t * 1000)


class VideoFileSource(_CvSource):
    """Video file. `speed` scales timestamps: speed=2 plays the sign twice as fast (t halves),
    speed=0.5 twice as slow. No real-time sleeping: frames are produced as fast as processed."""

    def __init__(self, path, speed=1.0):
        self.path, self.speed = str(path), float(speed)

    def _frames(self):
        import cv2
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"cannot open video {self.path}")
        fps = self._cap.get(cv2.CAP_PROP_FPS) or 25.0
        i = 0
        while self._cap is not None:
            ok, bgr = self._cap.read()
            if not ok:
                break
            # MediaPipe video mode needs strictly increasing ms timestamps (unscaled)
            yield bgr, i / fps / self.speed, int(i * 1000 / fps)
            i += 1


class LandmarkFileSource:
    """Replays saved raw landmarks (data/landmarks/<id>.npy, frames x 67 x 3).
    Also accepts an ndarray instead of a path (used by tests to build idle/concatenated streams)."""
    last_frame_bgr = None
    aspect = 16 / 9                # the dictionary videos are 640x360

    def __init__(self, npy_path, fps=25, speed=1.0):
        self.data = np.asarray(npy_path) if isinstance(npy_path, np.ndarray) else np.load(Path(npy_path))
        self.fps, self.speed = float(fps), float(speed)

    def close(self):
        pass

    def __iter__(self):
        for i, frame in enumerate(self.data):
            yield frame.astype(np.float32), i / (self.fps * self.speed)
