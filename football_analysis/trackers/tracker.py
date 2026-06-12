import os
import pickle

import cv2
import numpy as np
import pandas as pd
import supervision as sv
from ultralytics import YOLO

from football_analysis.utils.bbox_utils import get_center_of_bbox, get_bbox_width, get_foot_position


class Tracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()

    # ── Detection ──────────────────────────────────────────────────────────

    def detect_frames(self, frames):
        detections = []
        batch_size = 20
        for i in range(0, len(frames), batch_size):
            batch = self.model.predict(frames[i : i + batch_size], conf=0.1)
            detections += batch
        return detections

    # ── Tracking ───────────────────────────────────────────────────────────

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path, "rb") as f:
                return pickle.load(f)

        detections = self.detect_frames(frames)

        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            cls_names = detection.names
            cls_names_inv = {v: k for k, v in cls_names.items()}

            det_sv = sv.Detections.from_ultralytics(detection)

            # treat goalkeeper as player so ByteTrack assigns them an ID
            for idx, cls_id in enumerate(det_sv.class_id):
                if cls_names[cls_id] == "goalkeeper":
                    det_sv.class_id[idx] = cls_names_inv["player"]

            det_with_tracks = self.tracker.update_with_detections(det_sv)

            tracks["players"].append({})
            tracks["referees"].append({})
            tracks["ball"].append({})

            for frame_det in det_with_tracks:
                bbox = frame_det[0].tolist()
                cls_id = frame_det[3]
                track_id = frame_det[4]

                if cls_id == cls_names_inv.get("player"):
                    tracks["players"][frame_num][track_id] = {"bbox": bbox}
                if cls_id == cls_names_inv.get("referee"):
                    tracks["referees"][frame_num][track_id] = {"bbox": bbox}

            # ball has no stable ID from ByteTrack; always store under key 1
            for frame_det in det_sv:
                bbox = frame_det[0].tolist()
                cls_id = frame_det[3]
                if cls_id == cls_names_inv.get("ball"):
                    tracks["ball"][frame_num][1] = {"bbox": bbox}

        if stub_path:
            os.makedirs(os.path.dirname(stub_path), exist_ok=True)
            with open(stub_path, "wb") as f:
                pickle.dump(tracks, f)

        return tracks

    def add_position_to_tracks(self, tracks):
        for obj, obj_tracks in tracks.items():
            for frame_num, frame_track in enumerate(obj_tracks):
                for track_id, info in frame_track.items():
                    bbox = info["bbox"]
                    position = (
                        get_center_of_bbox(bbox)
                        if obj == "ball"
                        else get_foot_position(bbox)
                    )
                    tracks[obj][frame_num][track_id]["position"] = position

    def interpolate_ball_positions(self, ball_positions):
        positions = [x.get(1, {}).get("bbox", []) for x in ball_positions]
        df = pd.DataFrame(positions, columns=["x1", "y1", "x2", "y2"])
        df = df.interpolate().bfill()
        return [{1: {"bbox": row}} for row in df.to_numpy().tolist()]

    # ── Drawing ────────────────────────────────────────────────────────────

    def _draw_ellipse(self, frame, bbox, color, track_id=None):
        y2 = int(bbox[3])
        x_center, _ = get_center_of_bbox(bbox)
        width = get_bbox_width(bbox)
        cv2.ellipse(
            frame,
            center=(x_center, y2),
            axes=(int(width), int(0.35 * width)),
            angle=0.0,
            startAngle=-45,
            endAngle=235,
            color=color,
            thickness=2,
            lineType=cv2.LINE_4,
        )
        if track_id is not None:
            rw, rh = 40, 20
            x1r = x_center - rw // 2
            x2r = x_center + rw // 2
            y1r = y2 - rh // 2 + 15
            y2r = y2 + rh // 2 + 15
            cv2.rectangle(frame, (int(x1r), int(y1r)), (int(x2r), int(y2r)), color, cv2.FILLED)
            x_text = x1r + (2 if track_id > 99 else 12)
            cv2.putText(
                frame, str(track_id), (int(x_text), int(y1r + 15)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2,
            )
        return frame

    def _draw_triangle(self, frame, bbox, color):
        y = int(bbox[1])
        x, _ = get_center_of_bbox(bbox)
        pts = np.array([[x, y], [x - 10, y - 20], [x + 10, y - 20]])
        cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
        cv2.drawContours(frame, [pts], 0, (0, 0, 0), 2)
        return frame

    def _draw_team_ball_control(self, frame, frame_num, team_ball_control):
        overlay = frame.copy()
        cv2.rectangle(overlay, (1350, 850), (1900, 970), (255, 255, 255), cv2.FILLED)
        cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
        history = team_ball_control[: frame_num + 1]
        t1 = int(np.sum(history == 1))
        t2 = int(np.sum(history == 2))
        total = t1 + t2 or 1
        cv2.putText(
            frame, f"Team 1 Ball Control: {t1 / total * 100:.1f}%",
            (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3,
        )
        cv2.putText(
            frame, f"Team 2 Ball Control: {t2 / total * 100:.1f}%",
            (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3,
        )
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        output_frames = []
        for frame_num, frame in enumerate(video_frames):
            frame = frame.copy()
            for track_id, player in tracks["players"][frame_num].items():
                color = player.get("team_color", (0, 0, 255))
                frame = self._draw_ellipse(frame, player["bbox"], color, track_id)
                if player.get("has_ball"):
                    frame = self._draw_triangle(frame, player["bbox"], (0, 255, 0))
            for _, ref in tracks["referees"][frame_num].items():
                frame = self._draw_ellipse(frame, ref["bbox"], (0, 255, 255))
            for _, ball in tracks["ball"][frame_num].items():
                frame = self._draw_triangle(frame, ball["bbox"], (0, 255, 0))
            frame = self._draw_team_ball_control(frame, frame_num, team_ball_control)
            output_frames.append(frame)
        return output_frames
