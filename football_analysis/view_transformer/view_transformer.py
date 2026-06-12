import cv2
import numpy as np


class ViewTransformer:
    def __init__(self):
        court_width = 68
        court_length = 23.32

        # Four pitch-line corners in pixel coords, calibrated for 08fd33_4.mp4.
        # Re-pick these points if using a different camera angle.
        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915],
        ], dtype=np.float32)

        self.target_vertices = np.array([
            [0, court_width],
            [0, 0],
            [court_length, 0],
            [court_length, court_width],
        ], dtype=np.float32)

        self.M = cv2.getPerspectiveTransform(self.pixel_vertices, self.target_vertices)

    def transform_point(self, point):
        p = (int(point[0]), int(point[1]))
        if cv2.pointPolygonTest(self.pixel_vertices, p, False) < 0:
            return None
        pt = np.array([[point]], dtype=np.float32)
        transformed = cv2.perspectiveTransform(pt, self.M)
        return transformed.reshape(-1, 2)

    def add_transformed_position_to_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            for frame_num, frame_track in enumerate(obj_tracks):
                for track_id, info in frame_track.items():
                    pos = info.get("position_adjusted")
                    if pos is None:
                        tracks[obj][frame_num][track_id]["position_transformed"] = None
                        continue
                    transformed = self.transform_point(pos)
                    tracks[obj][frame_num][track_id]["position_transformed"] = (
                        transformed.squeeze().tolist() if transformed is not None else None
                    )
