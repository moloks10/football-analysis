import csv
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
    print("Reading video...")
    frames = read_video(input_path)

    print("Tracking objects...")
    tracker = Tracker(model_path)
    tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path=track_stub)
    tracker.add_position_to_tracks(tracks)

    print("Estimating camera movement...")
    cam_est = CameraMovementEstimator(frames[0])
    camera_movement = cam_est.get_camera_movement(frames, read_from_stub=True, stub_path=cam_stub)
    cam_est.add_adjust_positions_to_tracks(tracks, camera_movement)

    print("Applying perspective transform...")
    ViewTransformer().add_transformed_position_to_tracks(tracks)

    print("Interpolating ball positions...")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    print("Computing speed and distance...")
    SpeedAndDistanceEstimator().add_speed_and_distance_to_tracks(tracks)

    print("Assigning teams...")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(frames[0], tracks["players"][0])
    for frame_num, player_track in enumerate(tracks["players"]):
        for player_id, track in player_track.items():
            team = team_assigner.get_player_team(frames[frame_num], track["bbox"], player_id)
            tracks["players"][frame_num][player_id]["team"] = team
            tracks["players"][frame_num][player_id]["team_color"] = team_assigner.team_colors[team]

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
        output_video_path, frames, tracks, team_ball_control, camera_movement, tracker, cam_est
    )

    print("Generating heatmaps...")
    HeatmapGenerator().generate(tracks, heatmap_dir)

    print("Exporting stats CSV...")
    _export_stats(tracks, team_ball_control, stats_path)

    print("Done.")
    return tracks, team_ball_control


def _stream_annotated_video(output_path, frames, tracks, team_ball_control, camera_movement, tracker, cam_est):
    """Draw annotations and write one frame at a time to stay within RAM limits."""
    import cv2
    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_path, fourcc, 24, (w, h))
    speed_est = SpeedAndDistanceEstimator()

    for frame_num, frame in enumerate(frames):
        single = [frame]
        single_tracks = {k: [v[frame_num]] for k, v in tracks.items()}

        annotated = tracker.draw_annotations(single, single_tracks, team_ball_control, frame_offset=frame_num)
        annotated = cam_est.draw_camera_movement(annotated, [camera_movement[frame_num]])
        speed_est.draw_speed_and_distance(annotated, single_tracks)

        out.write(annotated[0])

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
