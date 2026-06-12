import csv
import gc
import os
from collections import defaultdict

import numpy as np

from football_analysis.camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from football_analysis.heatmap_generator.heatmap_generator import HeatmapGenerator
from football_analysis.player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from football_analysis.speed_distance_estimator.speed_distance_estimator import SpeedAndDistanceEstimator
from football_analysis.team_assigner.team_assigner import TeamAssigner
from football_analysis.trackers.tracker import Tracker
from football_analysis.utils.video_utils import read_video, save_video
from football_analysis.view_transformer.view_transformer import ViewTransformer


def run_pipeline(
    input_path,
    output_video_path,
    heatmap_dir,
    stats_path,
    model_path="models/best.pt",
    track_stub="stubs/track_stubs.pkl",
    cam_stub="stubs/camera_movement_stub.pkl",
):
    import cv2

    use_track_stub = bool(track_stub and os.path.exists(track_stub))
    use_cam_stub = bool(cam_stub and os.path.exists(cam_stub))

    # Load all frames only when YOLO or optical flow actually needs them.
    # With both stubs present we only need the first frame.
    if not use_track_stub or not use_cam_stub:
        print("Reading video (full load required for YOLO / camera movement)...")
        frames = read_video(input_path)
        frame0 = frames[0]
    else:
        print("Reading video (first frame only — stubs available)...")
        cap0 = cv2.VideoCapture(input_path)
        ret, frame0 = cap0.read()
        cap0.release()
        frames = None

    print("Tracking objects...")
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(
        frames if frames is not None else [],
        read_from_stub=True, stub_path=track_stub,
    )
    tracker.add_position_to_tracks(tracks)

    print("Estimating camera movement...")
    cam_est = CameraMovementEstimator(frame0)
    camera_movement = cam_est.get_camera_movement(
        frames if frames is not None else [],
        read_from_stub=True, stub_path=cam_stub,
    )
    cam_est.add_adjust_positions_to_tracks(tracks, camera_movement)

    # Free bulk frames as soon as YOLO + camera movement are done.
    if frames is not None:
        del frames
        gc.collect()

    print("Applying perspective transform...")
    ViewTransformer().add_transformed_position_to_tracks(tracks)

    print("Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    print("Computing speed and distance...")
    SpeedAndDistanceEstimator().add_speed_and_distance_to_tracks(tracks)

    # Team assignment — stream frames one at a time from disk.
    print("Assigning teams...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frame0, tracks["players"][0])
    cap = cv2.VideoCapture(input_path)
    for frame_num, player_track in enumerate(tracks["players"]):
        ret, frame = cap.read()
        if not ret:
            break
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frame, track["bbox"], player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]
    cap.release()

    print("Assigning ball possession...")
    pba = PlayerBallAssigner()
    team_ball_control = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox = tracks["ball"][frame_num][1]["bbox"]
        assigned = pba.assign_ball_to_player(player_track, ball_bbox)
        if assigned != -1:
            tracks["players"][frame_num][assigned]["has_ball"] = True
            team_ball_control.append(tracks["players"][frame_num][assigned]["team"])
        else:
            team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
    team_ball_control = np.array(team_ball_control)

    print("Drawing annotations and saving video...")
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    _stream_annotated_video(
        output_video_path, input_path, tracks, team_ball_control, camera_movement, tracker, cam_est
    )

    print("Generating heatmaps...")
    HeatmapGenerator().generate(tracks, heatmap_dir)

    print("Exporting stats CSV...")
    _export_stats(tracks, team_ball_control, stats_path)

    print("Done.")
    return tracks, team_ball_control


def _stream_annotated_video(output_path, input_path, tracks, team_ball_control, camera_movement, tracker, cam_est):
    """Read source video and write annotations one frame at a time to stay within RAM limits."""
    import cv2
    cap = cv2.VideoCapture(input_path)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, 24, (w, h))
    speed_est = SpeedAndDistanceEstimator()
    frame_num = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        single = [frame]
        single_tracks = {k: [v[frame_num]] for k, v in tracks.items()}

        annotated = tracker.draw_annotations(single, single_tracks, team_ball_control, frame_offset=frame_num)
        annotated = cam_est.draw_camera_movement(annotated, [camera_movement[frame_num]])
        speed_est.draw_speed_and_distance(annotated, single_tracks)

        out.write(annotated[0])
        frame_num += 1

    cap.release()
    out.release()


def _export_stats(tracks, team_ball_control, output_path):
    player_data = defaultdict(lambda: {"speeds": [], "distance": 0.0, "team": 0})
    for frame_tracks in tracks["players"]:
        for pid, info in frame_tracks.items():
            if "speed" in info:
                player_data[pid]["speeds"].append(info["speed"])
            player_data[pid]["distance"] = max(
                player_data[pid]["distance"], info.get("distance", 0.0)
            )
            player_data[pid]["team"] = info.get("team", 0)

    poss = {
        1: int(np.sum(team_ball_control == 1)),
        2: int(np.sum(team_ball_control == 2)),
    }
    total = sum(poss.values()) or 1

    rows = []
    for pid, data in player_data.items():
        speeds = data["speeds"]
        team = data["team"]
        rows.append({
            "player_id": pid,
            "team": team,
            "avg_speed_kmh": round(float(np.mean(speeds)), 2) if speeds else 0.0,
            "max_speed_kmh": round(float(max(speeds)), 2) if speeds else 0.0,
            "total_distance_m": round(data["distance"], 2),
            "possession_pct": round(poss.get(team, 0) / total * 100, 2),
        })

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    run_pipeline(
        input_path="input_videos/08fd33_4.mp4",
        output_video_path="output_videos/output_video.avi",
        heatmap_dir="output_videos/heatmaps",
        stats_path="output_videos/match_stats.csv",
    )
