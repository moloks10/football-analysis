import os

import cv2
import numpy as np
import pytest

FIXTURE_DIR = "tests/fixtures"


def _make_test_video(path, n_frames=10, w=640, h=480):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(path, fourcc, 24, (w, h))
    for _ in range(n_frames):
        out.write(np.zeros((h, w, 3), dtype=np.uint8))
    out.release()


def test_read_video_returns_frames():
    from football_analysis.utils.video_utils import read_video

    path = os.path.join(FIXTURE_DIR, "tiny.avi")
    _make_test_video(path, n_frames=10)
    frames = read_video(path)
    assert len(frames) == 10
    assert frames[0].shape == (480, 640, 3)
    os.remove(path)


def test_save_video_creates_file():
    from football_analysis.utils.video_utils import save_video

    frames = [np.zeros((480, 640, 3), dtype=np.uint8) for _ in range(5)]
    out_path = os.path.join(FIXTURE_DIR, "out.avi")
    save_video(frames, out_path)
    assert os.path.exists(out_path)
    os.remove(out_path)


def test_bbox_helpers():
    from football_analysis.utils.bbox_utils import (
        get_bbox_width,
        get_center_of_bbox,
        get_foot_position,
        measure_distance,
        measure_xy_distance,
    )

    bbox = [100, 200, 300, 400]
    assert get_center_of_bbox(bbox) == (200, 300)
    assert get_bbox_width(bbox) == 200
    assert get_foot_position(bbox) == (200, 400)
    assert abs(measure_distance((0, 0), (3, 4)) - 5.0) < 1e-9
    assert measure_xy_distance((10, 20), (13, 24)) == (-3, -4)
