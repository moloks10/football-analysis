import cv2

from football_analysis.utils.bbox_utils import measure_distance, get_foot_position


class SpeedAndDistanceEstimator:
    def __init__(self, frame_rate=24, frame_window=5):
        self.frame_rate = frame_rate
        self.frame_window = frame_window

    def add_speed_and_distance_to_tracks(self, tracks):
        total_distance = {}
        for obj, obj_tracks in tracks.items():
            if obj in ("ball", "referees"):
                continue
            total_distance.setdefault(obj, {})
            n_frames = len(obj_tracks)

            for start in range(0, n_frames, self.frame_window):
                end = min(start + self.frame_window, n_frames - 1)
                for track_id in obj_tracks[start]:
                    if track_id not in obj_tracks[end]:
                        continue
                    p_start = obj_tracks[start][track_id].get("position_transformed")
                    p_end = obj_tracks[end][track_id].get("position_transformed")
                    if p_start is None or p_end is None:
                        continue

                    dist = measure_distance(p_start, p_end)
                    elapsed = (end - start) / self.frame_rate
                    speed_kmh = (dist / elapsed) * 3.6

                    total_distance[obj].setdefault(track_id, 0)
                    total_distance[obj][track_id] += dist

                    for fn in range(start, end):
                        if track_id not in obj_tracks[fn]:
                            continue
                        obj_tracks[fn][track_id]["speed"] = speed_kmh
                        obj_tracks[fn][track_id]["distance"] = total_distance[obj][track_id]

    def draw_speed_and_distance(self, frames, tracks):
        for frame_num, frame in enumerate(frames):
            for obj, obj_tracks in tracks.items():
                if obj in ("ball", "referees"):
                    continue
                for track_id, info in obj_tracks[frame_num].items():
                    if "speed" not in info:
                        continue
                    foot = list(get_foot_position(info["bbox"]))
                    foot[1] += 40
                    pos = tuple(map(int, foot))
                    cv2.putText(frame, f"{info['speed']:.1f} km/h", pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                    cv2.putText(frame, f"{info['distance']:.1f} m", (pos[0], pos[1] + 20),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        return frames
