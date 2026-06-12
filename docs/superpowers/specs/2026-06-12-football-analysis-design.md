# Football Analysis Pipeline — Design Spec

**Date:** 2026-06-12  
**Reference repo:** https://github.com/abdullahtarek/football_analysis  
**Goal:** Learning-focused rebuild of a full football CV/ML pipeline, extended with heatmaps, stats export, and a Streamlit web app for LinkedIn/resume demo.

---

## 1. Approach

Notebook-first → clean package → Streamlit (Approach A).

Each module is built in a Jupyter notebook first so every component is understood before it is refactored into a clean Python class. The notebooks become portfolio artifacts alongside the working app.

---

## 2. Project Structure

```
footy yolo/
├── notebooks/
│   ├── 01_yolo_detection.ipynb
│   ├── 02_tracking.ipynb
│   ├── 03_team_assignment.ipynb
│   ├── 04_camera_movement.ipynb
│   ├── 05_perspective_transform.ipynb
│   ├── 06_speed_distance.ipynb
│   └── 07_heatmaps_and_stats.ipynb
├── football_analysis/
│   ├── __init__.py
│   ├── trackers/
│   │   ├── __init__.py
│   │   └── tracker.py
│   ├── team_assigner/
│   │   ├── __init__.py
│   │   └── team_assigner.py
│   ├── camera_movement_estimator/
│   │   ├── __init__.py
│   │   └── camera_movement_estimator.py
│   ├── view_transformer/
│   │   ├── __init__.py
│   │   └── view_transformer.py
│   ├── speed_distance_estimator/
│   │   ├── __init__.py
│   │   └── speed_distance_estimator.py
│   ├── player_ball_assigner/
│   │   ├── __init__.py
│   │   └── player_ball_assigner.py
│   ├── heatmap_generator/
│   │   ├── __init__.py
│   │   └── heatmap_generator.py
│   └── utils/
│       ├── __init__.py
│       ├── video_utils.py
│       └── bbox_utils.py
├── training/
│   └── yolo_football_training.ipynb
├── stubs/                        # pickle cache — skip re-running YOLO during dev
├── models/                       # YOLO weights (best.pt)
├── input_videos/
├── output_videos/
├── app.py                        # Streamlit web app
├── main.py                       # CLI entry point
└── requirements.txt
```

---

## 3. Build Order

1. YOLO fine-tuning (Colab) — produces `best.pt`
2. Tracker — YOLO inference + ByteTrack IDs
3. Team Assigner — K-means on jersey crop
4. Camera Movement Estimator — optical flow compensation
5. View Transformer — pixel → real-world metres
6. Speed & Distance Estimator — km/h + total distance per player
7. Player-Ball Assigner — possession tracking
8. Heatmap Generator — position history → pitch heatmap PNGs
9. Stats Export — CSV with per-player speed, distance, possession %
10. Annotator — draws all overlays onto frames
11. Streamlit App — wraps the full pipeline with a web UI

---

## 4. Data Flow

```
Input video
  → YOLO detection        (bboxes: players, goalkeeper, referee, ball per frame)
  → ByteTrack             (assign stable IDs across frames)
  → Team Assigner         (K-means on jersey crop → team 1 / team 2)
  → Camera Movement       (optical flow → dx, dy per frame)
  → adjusted position     = raw_tracker_position − camera_displacement
  → View Transformer      (homography: pixel coords → pitch metres)
  → Speed & Distance      (metres/frame → km/h; cumulative distance)
  → Player-Ball Assigner  (closest player to ball → possession flag)
  → Annotator             (draw boxes, IDs, team colours, speed labels)
  → Output video          (annotated MP4)
  → Heatmap Generator     (position history → per-player PNG heatmaps)
  → Stats CSV             (speed, distance, possession % per player)
```

**Key invariant:** camera displacement must be subtracted from raw tracker positions before the view transformer sees them. Skipping this step corrupts all speed and distance values whenever the camera pans.

Ball positions are interpolated with pandas to fill frames where the ball is occluded.

---

## 5. Module Responsibilities

### `trackers/tracker.py` — `Tracker`
- Loads YOLOv8 model (`best.pt`)
- Runs inference on every frame; filters by class (player, goalkeeper, referee, ball)
- Runs ByteTrack (via `supervision`) to assign stable IDs
- Exposes `get_object_tracks(frames)` → `dict[str, list[dict]]`
- Exposes `interpolate_ball_positions(tracks)`
- Exposes `draw_annotations(frames, tracks, team_ball_control)` → annotated frames
- Pickle stub: saves/loads tracks to skip re-inference during development

### `team_assigner/team_assigner.py` — `TeamAssigner`
- Crops the jersey region from each player's bounding box
- Runs K-means (k=2) on pixel colours to find the two dominant jersey colours
- Assigns each player to team 1 or team 2; caches per-player assignment to avoid flicker
- Exposes `assign_team_color(frame, player_tracks)` and `get_player_team(frame, bbox, player_id)`

### `camera_movement_estimator/camera_movement_estimator.py` — `CameraMovementEstimator`
- Uses Lucas-Kanade optical flow on good features to track (corners)
- Computes (dx, dy) displacement per frame relative to the first frame
- Exposes `get_camera_movement(frames)` → `list[tuple[float, float]]`
- Exposes `add_adjust_positions_to_tracks(tracks, camera_movement)` — subtracts displacement in-place
- Pickle stub for camera movement data
- Exposes `draw_camera_movement(frames, camera_movement)` → frames with displacement overlay

### `view_transformer/view_transformer.py` — `ViewTransformer`
- Hardcodes 4 reference points on the pitch (pixel) and their real-world counterparts (metres)
- Computes homography matrix with OpenCV
- Exposes `add_transformed_position_to_tracks(tracks)` — adds `position_transformed` to each track entry

### `speed_distance_estimator/speed_distance_estimator.py` — `SpeedAndDistanceEstimator`
- Computes frame-to-frame displacement in metres using `position_transformed`
- Converts to km/h using video FPS
- Accumulates total distance per player
- Exposes `add_speed_and_distance_to_tracks(tracks)` — adds `speed` and `distance` to track entries
- Exposes `draw_speed_and_distance(frames, tracks)` → frames with speed labels

### `player_ball_assigner/player_ball_assigner.py` — `PlayerBallAssigner`
- For each frame, finds the player whose bounding box centre is closest to the ball centre
- Returns -1 if no player is within a threshold distance
- Exposes `assign_ball_to_player(player_tracks, ball_bbox)` → player ID or -1

### `heatmap_generator/heatmap_generator.py` — `HeatmapGenerator`
- Collects `position_transformed` values across all frames per player
- Renders a 2D Gaussian KDE heatmap on a pitch outline using matplotlib
- Exports per-player PNG and an aggregate team PNG
- Exposes `generate(tracks, output_dir)`

### `utils/video_utils.py`
- `read_video(path)` → list of BGR frames
- `save_video(frames, path)` → writes AVI/MP4

### `utils/bbox_utils.py`
- `get_center_of_bbox(bbox)`, `get_bbox_width(bbox)`, `get_foot_position(bbox)`
- `measure_distance(p1, p2)`

---

## 6. Stats Export

After the pipeline runs, write `output_videos/match_stats.csv` with columns:

| player_id | team | avg_speed_kmh | max_speed_kmh | total_distance_m | possession_pct |

Possession % = frames the player held the ball / total frames with possession assigned.

---

## 7. Streamlit App (`app.py`)

1. File uploader — accepts MP4/AVI, max 200 MB
2. Runs the full pipeline on upload (with a progress bar)
3. Displays annotated output video (st.video)
4. Displays per-player heatmap PNGs (st.image, two columns)
5. Displays match_stats.csv as a sortable table (st.dataframe)
6. Download buttons for the output video and stats CSV

---

## 8. YOLO Fine-Tuning (`training/yolo_football_training.ipynb`)

- Run on Google Colab (free GPU)
- Dataset: Roboflow football-players-detection dataset
- Base model: `yolov8x.pt`
- Classes: player, goalkeeper, referee, ball
- Export: `best.pt` → download and place in `models/`
- This notebook is a standalone portfolio artifact — it documents the training run with loss curves and before/after detection samples

---

## 9. Development Stubs

During development, YOLO inference and camera movement computation are expensive. Both modules support a pickle stub pattern:

```python
tracker.get_object_tracks(frames, read_from_stub=True, stub_path='stubs/track_stubs.pkl')
camera_movement_estimator.get_camera_movement(frames, read_from_stub=True, stub_path='stubs/camera_movement_stub.pkl')
```

On first run with `read_from_stub=False`, results are computed and saved. Subsequent runs load instantly from the pickle file.

---

## 10. Testing

- No unit tests during the notebook learning phase — visual inspection per cell is sufficient
- The clean package gets a single smoke test: run the full pipeline on a 10-frame clip and assert that output video exists and tracks dict has expected keys
- Streamlit validates: file extension (mp4/avi), file size (≤ 200 MB) before triggering the pipeline

---

## 11. Tech Stack

| Concern | Library |
|---|---|
| Detection | `ultralytics` (YOLOv8) |
| Tracking | `supervision` (ByteTrack) |
| CV / optical flow | `opencv-python` |
| Team assignment | `scikit-learn` (K-means) |
| Data / interpolation | `pandas`, `numpy` |
| Visualisation | `matplotlib` |
| Web app | `streamlit` |
| Colab training | `roboflow`, `ultralytics` |
