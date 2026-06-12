import os
import pickle

import cv2
import numpy as np

from football_analysis.utils.bbox_utils import measure_distance, measure_xy_distance


class CameraMovementEstimator:
    def __init__(self, first_frame):
        self.minimum_distance = 5
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        gray = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
        mask = np.zeros_like(gray)
        mask[:, 0:20] = 1
        mask[:, 900:1050] = 1
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=3,
            blockSize=7,
            mask=mask,
        )

    def get_camera_movement(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        camera_movement = [[0.0, 0.0]] * len(frames)
        old_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        old_features = cv2.goodFeaturesToTrack(old_gray, **self.feature_params)

        for frame_num in range(1, len(frames)):
            new_gray = cv2.cvtColor(frames[frame_num], cv2.COLOR_BGR2GRAY)
            new_features, _, _ = cv2.calcOpticalFlowPyrLK(
                old_gray, new_gray, old_features, None, **self.lk_params
            )
            max_dist = 0
            cam_x, cam_y = 0.0, 0.0
            for new_pt, old_pt in zip(new_features, old_features):
                dist = measure_distance(new_pt.ravel(), old_pt.ravel())
                if dist > max_dist:
                    max_dist = dist
                    cam_x, cam_y = measure_xy_distance(old_pt.ravel(), new_pt.ravel())

            if max_dist > self.minimum_distance:
                camera_movement[frame_num] = [cam_x, cam_y]
                old_features = cv2.goodFeaturesToTrack(new_gray, **self.feature_params)

            old_gray = new_gray.copy()

        if stub_path:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(camera_movement, f)

        return camera_movement

    def add_adjust_positions_to_tracks(self, tracks, camera_movement_per_frame):
        for obj, obj_tracks in tracks.items():
            for frame_num, frame_track in enumerate(obj_tracks):
                dx, dy = camera_movement_per_frame[frame_num]
                for track_id, info in frame_track.items():
                    x, y = info["position"]
                    tracks[obj][frame_num][track_id]["position_adjusted"] = (x - dx, y - dy)

    def draw_camera_movement(self, frames, camera_movement_per_frame):
        output_frames = []
        for frame_num, frame in enumerate(frames):
            frame = frame.copy()
            overlay = frame.copy()
            cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), cv2.FILLED)
            cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
            dx, dy = camera_movement_per_frame[frame_num]
            cv2.putText(frame, f"Camera Movement X: {dx:.2f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            cv2.putText(frame, f"Camera Movement Y: {dy:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
            output_frames.append(frame)
        return output_frames
