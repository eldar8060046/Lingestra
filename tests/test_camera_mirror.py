"""Mirror (selfie) view is display-only: image and overlay flip, the recogniser's data does not."""
import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
import numpy as np
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication

from lingestra.app.widgets import CameraView

_app = QApplication.instance() or QApplication([])


def _frame():
    f = np.zeros((60, 80, 3), np.uint8)
    f[:, :40] = (0, 0, 255)          # BGR red on the LEFT half
    return f


def _left_right_red(view):
    img = view._pm.toImage()
    left = img.pixelColor(10, 30).red(); right = img.pixelColor(img.width() - 10, 30).red()
    return left, right


def test_mirrored_by_default_swaps_image_halves():
    v = CameraView(); v.update_view(_frame(), None)
    assert v.mirror is True
    left, right = _left_right_red(v)
    assert right > 200 and left < 50          # red moved to the right


def test_mirror_off_keeps_raw_direction():
    v = CameraView(); v.mirror = False; v.update_view(_frame(), None)
    left, right = _left_right_red(v)
    assert left > 200 and right < 50


def test_input_arrays_are_not_modified():
    lm = np.random.default_rng(0).random((67, 3)).astype(np.float32)
    before = lm.copy(); frame = _frame(); fcopy = frame.copy()
    v = CameraView(); v.update_view(frame, lm)
    assert np.array_equal(lm, before) and np.array_equal(frame, fcopy)   # recogniser data untouched


def test_overlay_flips_with_image():
    # a full synthetic pose squeezed into the left 20% of the image must be drawn on the far
    # right when mirrored, and stay on the far left when not
    rng = np.random.default_rng(1)
    lm = rng.random((67, 3)).astype(np.float32)
    lm[:, 0] = 0.02 + lm[:, 0] * 0.18
    lm[:, 1] = 0.2 + lm[:, 1] * 0.6
    for mirror in (True, False):
        v = CameraView(); v.mirror = mirror; v.update_view(None, lm)
        img = v._pm.toImage(); w, h = img.width(), img.height()
        bg = img.pixelColor(2, 2).name()
        xs = [x for x in range(0, w, 2) for y in range(0, h, 2) if img.pixelColor(x, y).name() != bg]
        assert xs, "nothing was drawn"
        if mirror:
            assert min(xs) > 0.75 * w
        else:
            assert max(xs) < 0.25 * w


def test_set_mirror_rerenders_last_frame_and_never_touches_input():
    v = CameraView(); frame = _frame(); fcopy = frame.copy()
    v.update_view(frame, None)
    assert _left_right_red(v)[1] > 200
    v.set_mirror(False)
    left, right = _left_right_red(v)
    assert left > 200 and right < 50 and np.array_equal(frame, fcopy)


def test_skeleton_icon_and_canvas_flip_horizontally_when_mirrored():
    from lingestra.app.widgets import SkeletonCanvas, frame_pixmap
    rng = np.random.default_rng(3)
    lm = rng.random((67, 3)).astype(np.float32); lm[:, 0] = 0.1 + lm[:, 0] * 0.8
    before = lm.copy()
    a = frame_pixmap(lm, 120, 90, mirror=False).toImage()
    b = frame_pixmap(lm, 120, 90, mirror=True).toImage()
    assert np.array_equal(lm, before)                                  # data untouched

    def arr(img):
        img = img.convertToFormat(QImage.Format_RGB888)
        return np.frombuffer(img.constBits(), np.uint8).reshape(img.height(), img.bytesPerLine())[:, :img.width() * 3] \
            .reshape(img.height(), img.width(), 3).astype(int)
    A, B = arr(a), arr(b)
    assert np.abs(A - B).mean() > 1.0                                  # visibly different
    assert np.abs(A[:, ::-1] - B).mean() < np.abs(A - B).mean() * 0.35   # ...and it is the mirror image

    c = SkeletonCanvas(); c.resize(300, 200); c.set_frames(np.stack([lm, lm])); c.set_index(0)
    im1 = c.grab().toImage(); c.set_mirror(True); im2 = c.grab().toImage()
    A, B = arr(im1), arr(im2)
    assert np.abs(A - B).mean() > 0.5 and np.array_equal(c.frames[0], lm)
