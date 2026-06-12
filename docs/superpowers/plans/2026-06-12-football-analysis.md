# Football Analysis Pipeline — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a full football video analysis pipeline from scratch — YOLO detection, ByteTrack tracking, team assignment, camera movement compensation, perspective transform, speed/distance estimation, heatmaps, stats export, and a Streamlit web app.

**Architecture:** Notebook-first: each module is explored in a Jupyter notebook first, then refactored into a clean Python package (`football_analysis/`). A CLI `main.py` wires the full pipeline; `app.py` wraps it in Streamlit.

**Tech Stack:** `ultralytics` (YOLOv8), `supervision` (ByteTrack), `opencv-python`, `scikit-learn`, `pandas`, `numpy`, `matplotlib`, `scipy`, `streamlit`

---

## File Map

| File | Purpose |
|---|---|
| `requirements.txt` | Pinned dependencies |
| `football_analysis/__init__.py` | Package root (empty) |
| `football_analysis/utils/video_utils.py` | `read_video`, `save_video` |
| `football_analysis/utils/bbox_utils.py` | Bounding box math helpers |
| `football_analysis/trackers/tracker.py` | `Tracker` — YOLO inference + ByteTrack |
| `football_analysis/team_assigner/team_assigner.py` | `TeamAssigner` — K-means on jersey |
| `football_analysis/camera_movement_estimator/camera_movement_estimator.py` | `CameraMovementEstimator` — optical flow |
| `football_analysis/view_transformer/view_transformer.py` | `ViewTransformer` — homography |
| `football_analysis/speed_distance_estimator/speed_distance_estimator.py` | `SpeedAndDistanceEstimator` |
| `football_analysis/player_ball_assigner/player_ball_assigner.py` | `PlayerBallAssigner` |
| `football_analysis/heatmap_generator/heatmap_generator.py` | `HeatmapGenerator` |
| `main.py` | CLI — wires all modules, exports stats CSV |
| `app.py` | Streamlit web app |
| `tests/test_utils.py` | Smoke test for video read/write |
| `notebooks/01_yolo_detection.ipynb` | Explore YOLO inference |
| `notebooks/02_tracking.ipynb` | Explore ByteTrack |
| `notebooks/03_team_assignment.ipynb` | Explore K-means jersey |
| `notebooks/04_camera_movement.ipynb` | Explore optical flow |
| `notebooks/05_perspective_transform.ipynb` | Explore homography |
| `notebooks/06_speed_distance.ipynb` | Explore speed/distance |
| `notebooks/07_heatmaps_and_stats.ipynb` | Explore heatmaps + stats |
| `training/yolo_football_training.ipynb` | Colab fine-tuning (standalone) |

---

## Task 1: Project Setup

**Files:**
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: all `__init__.py` files
- Create: empty placeholder dirs (`stubs/`, `models/`, `input_videos/`, `output_videos/`)

- [ ] **Step 1: Initialise git and create directory structure**

```bash
cd ~/footy\ yolo
git init
mkdir -p notebooks training stubs models input_videos output_videos
mkdir -p football_analysis/utils
mkdir -p football_analysis/trackers
mkdir -p football_analysis/team_assigner
mkdir -p football_analysis/camera_movement_estimator
mkdir -p football_analysis/view_transformer
mkdir -p football_analysis/speed_distance_estimator
mkdir -p football_analysis/player_ball_assigner
mkdir -p football_analysis/heatmap_generator
mkdir -p tests/fixtures
```

- [ ] **Step 2: Write `requirements.txt`**

```
ultralytics==8.2.18
supervision==0.20.0
opencv-python==4.9.0.80
scikit-learn==1.4.2
pandas==2.2.2
numpy==1.26.4
matplotlib==3.8.4
scipy==1.13.0
streamlit==1.34.0
roboflow==1.1.37
pytest==8.2.0
```

- [ ] **Step 3: Write `.gitignore`**

```
__pycache__/
*.py[cod]
*.pkl
*.pt
models/
input_videos/
output_videos/
stubs/
.env
.DS_Store
```

- [ ] **Step 4: Create all `__init__.py` files**

Create each of the following as an empty file:
- `football_analysis/__init__.py`
- `football_analysis/utils/__init__.py`
- `football_analysis/trackers/__init__.py`
- `football_analysis/team_assigner/__init__.py`
- `football_analysis/camera_movement_estimator/__init__.py`
- `football_analysis/view_transformer/__init__.py`
- `football_analysis/speed_distance_estimator/__init__.py`
- `football_analysis/player_ball_assigner/__init__.py`
- `football_analysis/heatmap_generator/__init__.py`

```bash
touch football_analysis/__init__.py \
      football_analysis/utils/__init__.py \
      football_analysis/trackers/__init__.py \
      football_analysis/team_assigner/__init__.py \
      football_analysis/camera_movement_estimator/__init__.py \
      football_analysis/view_transformer/__init__.py \
      football_analysis/speed_distance_estimator/__init__.py \
      football_analysis/player_ball_assigner/__init__.py \
      football_analysis/heatmap_generator/__init__.py
```

- [ ] **Step 5: Install dependencies**

```bash
pip install -r requirements.txt
```

Expected: all packages install without error. May take several minutes.

- [ ] **Step 6: Add `.gitkeep` to empty dirs and commit**

```bash
touch stubs/.gitkeep models/.gitkeep input_videos/.gitkeep output_videos/.gitkeep
git add .
git commit -m "chore: project scaffold — dirs, requirements, gitignore"
```

---

## Task 2: Utils + Smoke Test

**Files:**
- Create: `football_analysis/utils/video_utils.py`
- Create: `football_analysis/utils/bbox_utils.py`
- Create: `tests/test_utils.py`

- [ ] **Step 1: Write the failing smoke test**

`tests/test_utils.py`:
```python
import os
import numpy as np
import cv2
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
        get_center_of_bbox,
        get_bbox_width,
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
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
pytest tests/test_utils.py -v
```

Expected: `ImportError` — modules not found yet.

- [ ] **Step 3: Write `football_analysis/utils/video_utils.py`**

```python
import cv2


def read_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def save_video(output_video_frames, output_video_path):
    if not output_video_frames:
        return
    h, w = output_video_frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    out = cv2.VideoWriter(output_video_path, fourcc, 24, (w, h))
    for frame in output_video_frames:
        out.write(frame)
    out.release()
```

- [ ] **Step 4: Write `football_analysis/utils/bbox_utils.py`**

```python
import numpy as np


def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    return bbox[2] - bbox[0]


def get_foot_position(bbox):
    x1, _, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def measure_distance(p1, p2):
    return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5


def measure_xy_distance(p1, p2):
    return p1[0] - p2[0], p1[1] - p2[1]
```

- [ ] **Step 5: Run tests to confirm they pass**

```bash
pytest tests/test_utils.py -v
```

Expected: 3 tests PASS.

- [ ] **Step 6: Commit**

```bash
git add football_analysis/utils/ tests/test_utils.py
git commit -m "feat: add video utils and bbox helpers with smoke tests"
```

---

## Task 3: Tracker Module

**Files:**
- Create: `football_analysis/trackers/tracker.py`
- Create: `notebooks/01_yolo_detection.ipynb`
- Create: `notebooks/02_tracking.ipynb`

- [ ] **Step 1: Write `football_analysis/trackers/tracker.py`**

```python
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
        t1 = np.sum(history == 1)
        t2 = np.sum(history == 2)
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
```

- [ ] **Step 2: Create `notebooks/01_yolo_detection.ipynb`**

Create a notebook with these cells (in order). Use VS Code's Jupyter extension.

**Cell 1 — Markdown:**
```
# Notebook 01: YOLO Detection

Explore raw YOLO detections on a single frame before adding any tracking.
```

**Cell 2 — Install (only needed if not already installed):**
```python
# %pip install ultralytics supervision opencv-python
```

**Cell 3 — Load video and pick a frame:**
```python
import cv2
import sys
sys.path.insert(0, '..')

from football_analysis.utils.video_utils import read_video

frames = read_video('../input_videos/08fd33_4.mp4')
print(f"Loaded {len(frames)} frames, shape: {frames[0].shape}")
```

**Cell 4 — Run YOLO on one frame:**
```python
from ultralytics import YOLO

model = YOLO('../models/best.pt')
result = model.predict(frames[0], conf=0.1)[0]

print("Classes detected:", result.names)
print("Number of detections:", len(result.boxes))
for box in result.boxes:
    print(f"  class={result.names[int(box.cls)]:12s}  conf={float(box.conf):.2f}  bbox={box.xyxy[0].tolist()}")
```

**Cell 5 — Visualise:**
```python
import matplotlib.pyplot as plt

annotated = result.plot()
plt.figure(figsize=(16, 9))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Raw YOLO detections — frame 0')
plt.show()
```

- [ ] **Step 3: Create `notebooks/02_tracking.ipynb`**

**Cell 1 — Markdown:**
```
# Notebook 02: ByteTrack Tracking

Add persistent player IDs across frames using ByteTrack from the supervision library.
```

**Cell 2:**
```python
import sys
sys.path.insert(0, '..')
from football_analysis.trackers.tracker import Tracker
from football_analysis.utils.video_utils import read_video
import cv2

frames = read_video('../input_videos/08fd33_4.mp4')
tracker = Tracker('../models/best.pt')
```

**Cell 3 — Get tracks (reads from stub if available):**
```python
tracks = tracker.get_object_tracks(
    frames,
    read_from_stub=True,
    stub_path='../stubs/track_stubs.pkl'
)
print("Keys:", list(tracks.keys()))
print("Frames tracked:", len(tracks['players']))
print("Players in frame 0:", list(tracks['players'][0].keys()))
```

**Cell 4 — Add foot positions:**
```python
tracker.add_position_to_tracks(tracks)
# Each player now has a 'position' key
sample = list(tracks['players'][0].values())[0]
print("Sample player data:", sample)
```

**Cell 5 — Visualise frame 0 with IDs:**
```python
import numpy as np
import matplotlib.pyplot as plt

# Draw without team colours yet — pass zeros for team_ball_control
dummy_control = np.zeros(len(frames), dtype=int)
annotated = tracker.draw_annotations([frames[0]], 
                                      {k: [v[0]] for k, v in tracks.items()},
                                      dummy_control)
plt.figure(figsize=(16, 9))
plt.imshow(cv2.cvtColor(annotated[0], cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Frame 0 with ByteTrack IDs')
plt.show()
```

- [ ] **Step 4: Commit**

```bash
git add football_analysis/trackers/ notebooks/01_yolo_detection.ipynb notebooks/02_tracking.ipynb
git commit -m "feat: add Tracker module (YOLO + ByteTrack) and exploration notebooks"
```

---

## Task 4: Team Assigner Module

**Files:**
- Create: `football_analysis/team_assigner/team_assigner.py`
- Create: `notebooks/03_team_assignment.ipynb`

- [ ] **Step 1: Write `football_analysis/team_assigner/team_assigner.py`**

```python
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self):
        self.team_colors = {}
        self.player_team_dict = {}
        self.kmeans = None

    def _get_clustering_model(self, image):
        image_2d = image.reshape(-1, 3)
        kmeans = KMeans(n_clusters=2, init="k-means++", n_init=1)
        kmeans.fit(image_2d)
        return kmeans

    def _get_player_color(self, frame, bbox):
        crop = frame[int(bbox[1]) : int(bbox[3]), int(bbox[0]) : int(bbox[2])]
        top_half = crop[: crop.shape[0] // 2, :]
        kmeans = self._get_clustering_model(top_half)
        labels = kmeans.labels_.reshape(top_half.shape[0], top_half.shape[1])
        # corners are background; non-corner cluster is the player jersey
        corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        bg_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 - bg_cluster
        return kmeans.cluster_centers_[player_cluster]

    def assign_team_color(self, frame, player_detections):
        player_colors = [
            self._get_player_color(frame, info["bbox"])
            for info in player_detections.values()
        ]
        if len(player_colors) < 2:
            return
        self.kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
        self.kmeans.fit(player_colors)
        self.team_colors[1] = self.kmeans.cluster_centers_[0]
        self.team_colors[2] = self.kmeans.cluster_centers_[1]

    def get_player_team(self, frame, player_bbox, player_id):
        if player_id in self.player_team_dict:
            return self.player_team_dict[player_id]
        color = self._get_player_color(frame, player_bbox)
        team_id = int(self.kmeans.predict(color.reshape(1, -1))[0]) + 1
        self.player_team_dict[player_id] = team_id
        return team_id
```

- [ ] **Step 2: Create `notebooks/03_team_assignment.ipynb`**

**Cell 1 — Markdown:**
```
# Notebook 03: Team Assignment via K-Means

Jersey colour is the signal. We crop the top half of each player's bounding box
(avoids pitch colour at the bottom), run K-means with k=2 to find the two 
jersey colours, then assign every player to a team.
```

**Cell 2:**
```python
import sys
sys.path.insert(0, '..')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from football_analysis.utils.video_utils import read_video
from football_analysis.trackers.tracker import Tracker
from football_analysis.team_assigner.team_assigner import TeamAssigner

frames = read_video('../input_videos/08fd33_4.mp4')
tracker = Tracker('../models/best.pt')
tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path='../stubs/track_stubs.pkl')
```

**Cell 3 — Assign team colours from first frame:**
```python
team_assigner = TeamAssigner()
team_assigner.assign_team_color(frames[0], tracks['players'][0])

print("Team 1 colour (BGR):", team_assigner.team_colors[1])
print("Team 2 colour (BGR):", team_assigner.team_colors[2])
```

**Cell 4 — Label all players across all frames:**
```python
for frame_num, player_track in enumerate(tracks['players']):
    for player_id, track in player_track.items():
        team = team_assigner.get_player_team(frames[frame_num], track['bbox'], player_id)
        tracks['players'][frame_num][player_id]['team'] = team
        tracks['players'][frame_num][player_id]['team_color'] = team_assigner.team_colors[team]

# Spot-check
sample = list(tracks['players'][0].items())[:3]
for pid, info in sample:
    print(f"Player {pid}: team={info['team']}  color={info['team_color']}")
```

**Cell 5 — Visualise teams on frame 0:**
```python
dummy_control = np.zeros(len(frames), dtype=int)
annotated = tracker.draw_annotations([frames[0]], 
                                      {k: [v[0]] for k, v in tracks.items()},
                                      dummy_control)
plt.figure(figsize=(16, 9))
plt.imshow(cv2.cvtColor(annotated[0], cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title('Frame 0 — team colours assigned')
plt.show()
```

- [ ] **Step 3: Commit**

```bash
git add football_analysis/team_assigner/ notebooks/03_team_assignment.ipynb
git commit -m "feat: add TeamAssigner module (K-means jersey colour) and notebook"
```

---

## Task 5: Camera Movement Estimator

**Files:**
- Create: `football_analysis/camera_movement_estimator/camera_movement_estimator.py`
- Create: `notebooks/04_camera_movement.ipynb`

- [ ] **Step 1: Write `football_analysis/camera_movement_estimator/camera_movement_estimator.py`**

```python
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
        # mask: only track features on left/right sideline strips (stable background)
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
```

- [ ] **Step 2: Create `notebooks/04_camera_movement.ipynb`**

**Cell 1 — Markdown:**
```
# Notebook 04: Camera Movement Estimation

The camera pans to follow the ball. We measure this pan using Lucas-Kanade 
optical flow on stable background features (the sideline strips). 
The result (dx, dy) per frame is subtracted from every player position 
before we compute real-world coordinates.
```

**Cell 2:**
```python
import sys
sys.path.insert(0, '..')
import cv2
import matplotlib.pyplot as plt
from football_analysis.utils.video_utils import read_video
from football_analysis.trackers.tracker import Tracker
from football_analysis.camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator

frames = read_video('../input_videos/08fd33_4.mp4')
tracker = Tracker('../models/best.pt')
tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path='../stubs/track_stubs.pkl')
tracker.add_position_to_tracks(tracks)
```

**Cell 3 — Estimate camera movement:**
```python
estimator = CameraMovementEstimator(frames[0])
camera_movement = estimator.get_camera_movement(
    frames, read_from_stub=True, stub_path='../stubs/camera_movement_stub.pkl'
)
print(f"Frames: {len(camera_movement)}")
print("First 5 movements:", camera_movement[:5])
```

**Cell 4 — Adjust positions:**
```python
estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
# Compare raw vs adjusted for player 1 in frame 5
frame5 = tracks['players'][5]
if frame5:
    pid = list(frame5.keys())[0]
    print(f"Player {pid}:")
    print("  raw position    :", frame5[pid]['position'])
    print("  adjusted position:", frame5[pid]['position_adjusted'])
```

**Cell 5 — Plot camera movement over time:**
```python
import numpy as np
dx = [m[0] for m in camera_movement]
dy = [m[1] for m in camera_movement]
plt.figure(figsize=(14, 4))
plt.plot(dx, label='X displacement')
plt.plot(dy, label='Y displacement')
plt.xlabel('Frame')
plt.ylabel('Pixels')
plt.title('Camera movement per frame')
plt.legend()
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Commit**

```bash
git add football_analysis/camera_movement_estimator/ notebooks/04_camera_movement.ipynb
git commit -m "feat: add CameraMovementEstimator (Lucas-Kanade optical flow) and notebook"
```

---

## Task 6: View Transformer

**Files:**
- Create: `football_analysis/view_transformer/view_transformer.py`
- Create: `notebooks/05_perspective_transform.ipynb`

- [ ] **Step 1: Write `football_analysis/view_transformer/view_transformer.py`**

```python
import cv2
import numpy as np


class ViewTransformer:
    def __init__(self):
        # Real pitch dimensions (metres) for the visible section
        court_width = 68
        court_length = 23.32

        # Four pitch-line corners as they appear in the video (pixel coords).
        # These are calibrated for the reference video (08fd33_4.mp4).
        # Re-calibrate for a different camera angle by picking 4 known pitch points.
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
        # Only transform points that fall inside the calibrated quadrilateral
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
```

- [ ] **Step 2: Create `notebooks/05_perspective_transform.ipynb`**

**Cell 1 — Markdown:**
```
# Notebook 05: Perspective Transform

A homography matrix maps from the distorted camera view to a top-down "bird's eye"
view of the pitch in real-world metres. We pick 4 known points in the pixel image 
and their known real-world coordinates, then let OpenCV compute the transformation.

Once positions are in metres, we can compute actual speed and distance.
```

**Cell 2:**
```python
import sys
sys.path.insert(0, '..')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from football_analysis.utils.video_utils import read_video
from football_analysis.trackers.tracker import Tracker
from football_analysis.camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from football_analysis.view_transformer.view_transformer import ViewTransformer

frames = read_video('../input_videos/08fd33_4.mp4')
tracker = Tracker('../models/best.pt')
tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path='../stubs/track_stubs.pkl')
tracker.add_position_to_tracks(tracks)

estimator = CameraMovementEstimator(frames[0])
camera_movement = estimator.get_camera_movement(frames, read_from_stub=True, stub_path='../stubs/camera_movement_stub.pkl')
estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
```

**Cell 3 — Apply transform:**
```python
vt = ViewTransformer()
vt.add_transformed_position_to_tracks(tracks)

# Show a player's journey in real-world coordinates
for pid, info in tracks['players'][10].items():
    print(f"Player {pid}:")
    print("  pixel (adjusted):", info.get('position_adjusted'))
    print("  real-world (m):  ", info.get('position_transformed'))
    break
```

**Cell 4 — Scatter real-world positions of all players in one frame:**
```python
xs, ys, teams = [], [], []
for frame_tracks in tracks['players']:
    for pid, info in frame_tracks.items():
        pos = info.get('position_transformed')
        if pos:
            xs.append(pos[0])
            ys.append(pos[1])

plt.figure(figsize=(10, 7))
plt.scatter(xs, ys, s=2, alpha=0.1)
plt.xlim(0, 23.32)
plt.ylim(0, 68)
plt.xlabel('Length (m)')
plt.ylabel('Width (m)')
plt.title('All player positions — real-world coordinates')
plt.tight_layout()
plt.show()
```

- [ ] **Step 3: Commit**

```bash
git add football_analysis/view_transformer/ notebooks/05_perspective_transform.ipynb
git commit -m "feat: add ViewTransformer (homography, pixel → pitch metres) and notebook"
```

---

## Task 7: Speed & Distance Estimator

**Files:**
- Create: `football_analysis/speed_distance_estimator/speed_distance_estimator.py`
- Create: `notebooks/06_speed_distance.ipynb`

- [ ] **Step 1: Write `football_analysis/speed_distance_estimator/speed_distance_estimator.py`**

```python
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
```

- [ ] **Step 2: Create `notebooks/06_speed_distance.ipynb`**

**Cell 1 — Markdown:**
```
# Notebook 06: Speed & Distance

We now have real-world positions in metres. Speed = displacement over time.
We batch frames into windows of 5 to smooth out per-frame noise.
```

**Cell 2 — Setup (same pipeline setup as notebook 05):**
```python
import sys
sys.path.insert(0, '..')
from football_analysis.utils.video_utils import read_video
from football_analysis.trackers.tracker import Tracker
from football_analysis.camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from football_analysis.view_transformer.view_transformer import ViewTransformer
from football_analysis.speed_distance_estimator.speed_distance_estimator import SpeedAndDistanceEstimator

frames = read_video('../input_videos/08fd33_4.mp4')
tracker = Tracker('../models/best.pt')
tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path='../stubs/track_stubs.pkl')
tracker.add_position_to_tracks(tracks)
estimator = CameraMovementEstimator(frames[0])
camera_movement = estimator.get_camera_movement(frames, read_from_stub=True, stub_path='../stubs/camera_movement_stub.pkl')
estimator.add_adjust_positions_to_tracks(tracks, camera_movement)
tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
ViewTransformer().add_transformed_position_to_tracks(tracks)
```

**Cell 3 — Compute speed/distance:**
```python
speed_est = SpeedAndDistanceEstimator()
speed_est.add_speed_and_distance_to_tracks(tracks)

# Print top-speed players
from collections import defaultdict
player_max_speed = defaultdict(float)
for frame_tracks in tracks['players']:
    for pid, info in frame_tracks.items():
        player_max_speed[pid] = max(player_max_speed[pid], info.get('speed', 0))

print("Top 5 players by max speed:")
for pid, spd in sorted(player_max_speed.items(), key=lambda x: -x[1])[:5]:
    print(f"  Player {pid}: {spd:.1f} km/h")
```

- [ ] **Step 3: Commit**

```bash
git add football_analysis/speed_distance_estimator/ notebooks/06_speed_distance.ipynb
git commit -m "feat: add SpeedAndDistanceEstimator and notebook"
```

---

## Task 8: Player-Ball Assigner

**Files:**
- Create: `football_analysis/player_ball_assigner/player_ball_assigner.py`

- [ ] **Step 1: Write `football_analysis/player_ball_assigner/player_ball_assigner.py`**

```python
from football_analysis.utils.bbox_utils import get_center_of_bbox, measure_distance


class PlayerBallAssigner:
    def __init__(self, max_distance=70):
        self.max_distance = max_distance

    def assign_ball_to_player(self, players, ball_bbox):
        ball_center = get_center_of_bbox(ball_bbox)
        min_dist = float("inf")
        assigned_player = -1
        for player_id, player in players.items():
            bbox = player["bbox"]
            # check distance from both feet-corners of the bounding box
            dist = min(
                measure_distance((bbox[0], bbox[3]), ball_center),
                measure_distance((bbox[2], bbox[3]), ball_center),
            )
            if dist < self.max_distance and dist < min_dist:
                min_dist = dist
                assigned_player = player_id
        return assigned_player
```

- [ ] **Step 2: Commit**

```bash
git add football_analysis/player_ball_assigner/
git commit -m "feat: add PlayerBallAssigner module"
```

---

## Task 9: Heatmap Generator

**Files:**
- Create: `football_analysis/heatmap_generator/heatmap_generator.py`
- Create: `notebooks/07_heatmaps_and_stats.ipynb`

- [ ] **Step 1: Write `football_analysis/heatmap_generator/heatmap_generator.py`**

```python
import os

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


class HeatmapGenerator:
    PITCH_LENGTH = 23.32  # metres (visible section width)
    PITCH_WIDTH = 68.0    # metres

    def generate(self, tracks, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        team_positions = {1: [], 2: []}

        for frame_tracks in tracks["players"]:
            for info in frame_tracks.values():
                pos = info.get("position_transformed")
                team = info.get("team")
                if pos is None or team not in team_positions:
                    continue
                team_positions[team].append(pos)

        for team_id, positions in team_positions.items():
            if len(positions) < 5:
                continue
            arr = np.array(positions)
            self._render(arr[:, 0], arr[:, 1], team_id, output_dir)

    def _render(self, x, y, team_id, output_dir):
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_facecolor("#4CAF50")
        ax.set_xlim(0, self.PITCH_LENGTH)
        ax.set_ylim(0, self.PITCH_WIDTH)

        # pitch outline
        rect = patches.Rectangle(
            (0, 0), self.PITCH_LENGTH, self.PITCH_WIDTH,
            linewidth=2, edgecolor="white", facecolor="none",
        )
        ax.add_patch(rect)
        ax.axvline(x=self.PITCH_LENGTH / 2, color="white", linewidth=1.5)

        try:
            kde = gaussian_kde(np.vstack([x, y]))
            xi, yi = np.mgrid[0 : self.PITCH_LENGTH : 100j, 0 : self.PITCH_WIDTH : 100j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()]))
            ax.contourf(xi, yi, zi.reshape(xi.shape), alpha=0.6, cmap="hot")
        except Exception:
            ax.scatter(x, y, alpha=0.1, s=5, c="red")

        ax.set_title(f"Team {team_id} Heatmap", fontsize=16, color="white")
        fig.patch.set_facecolor("#2d5a27")
        plt.savefig(os.path.join(output_dir, f"team_{team_id}_heatmap.png"),
                    dpi=150, bbox_inches="tight")
        plt.close()
```

- [ ] **Step 2: Create `notebooks/07_heatmaps_and_stats.ipynb`**

**Cell 1 — Markdown:**
```
# Notebook 07: Heatmaps & Stats

Two portfolio-ready outputs:
1. Team heatmaps — where each team spent time on the pitch
2. Match stats CSV — per-player speed, distance, possession
```

**Cell 2 — Run full pipeline up to this point:**
```python
import sys
sys.path.insert(0, '..')
import numpy as np
from football_analysis.utils.video_utils import read_video
from football_analysis.trackers.tracker import Tracker
from football_analysis.team_assigner.team_assigner import TeamAssigner
from football_analysis.camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from football_analysis.view_transformer.view_transformer import ViewTransformer
from football_analysis.speed_distance_estimator.speed_distance_estimator import SpeedAndDistanceEstimator
from football_analysis.player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from football_analysis.heatmap_generator.heatmap_generator import HeatmapGenerator

frames = read_video('../input_videos/08fd33_4.mp4')
tracker = Tracker('../models/best.pt')
tracks = tracker.get_object_tracks(frames, read_from_stub=True, stub_path='../stubs/track_stubs.pkl')
tracker.add_position_to_tracks(tracks)

cam = CameraMovementEstimator(frames[0])
cam_mv = cam.get_camera_movement(frames, read_from_stub=True, stub_path='../stubs/camera_movement_stub.pkl')
cam.add_adjust_positions_to_tracks(tracks, cam_mv)

ViewTransformer().add_transformed_position_to_tracks(tracks)
tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
SpeedAndDistanceEstimator().add_speed_and_distance_to_tracks(tracks)

ta = TeamAssigner()
ta.assign_team_color(frames[0], tracks['players'][0])
for frame_num, pt in enumerate(tracks['players']):
    for pid, t in pt.items():
        team = ta.get_player_team(frames[frame_num], t['bbox'], pid)
        tracks['players'][frame_num][pid]['team'] = team
        tracks['players'][frame_num][pid]['team_color'] = ta.team_colors[team]

pba = PlayerBallAssigner()
team_ball_control = []
for frame_num, pt in enumerate(tracks['players']):
    ball_bbox = tracks['ball'][frame_num][1]['bbox']
    assigned = pba.assign_ball_to_player(pt, ball_bbox)
    if assigned != -1:
        tracks['players'][frame_num][assigned]['has_ball'] = True
        team_ball_control.append(tracks['players'][frame_num][assigned]['team'])
    else:
        team_ball_control.append(team_ball_control[-1] if team_ball_control else 0)
team_ball_control = np.array(team_ball_control)
```

**Cell 3 — Generate and display heatmaps:**
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

hg = HeatmapGenerator()
hg.generate(tracks, '../output_videos/heatmaps')

fig, axes = plt.subplots(1, 2, figsize=(20, 8))
for i, team_id in enumerate([1, 2]):
    img = mpimg.imread(f'../output_videos/heatmaps/team_{team_id}_heatmap.png')
    axes[i].imshow(img)
    axes[i].axis('off')
    axes[i].set_title(f'Team {team_id}', fontsize=14)
plt.tight_layout()
plt.show()
```

**Cell 4 — Build and display stats:**
```python
import pandas as pd
from collections import defaultdict

player_stats = defaultdict(lambda: {"speeds": [], "distance": 0, "team": 0})
for frame_tracks in tracks['players']:
    for pid, info in frame_tracks.items():
        if "speed" in info:
            player_stats[pid]["speeds"].append(info["speed"])
        player_stats[pid]["distance"] = max(player_stats[pid]["distance"], info.get("distance", 0))
        player_stats[pid]["team"] = info.get("team", 0)

possession_counts = {1: int(np.sum(team_ball_control == 1)),
                     2: int(np.sum(team_ball_control == 2))}
total_poss = sum(possession_counts.values()) or 1

rows = []
for pid, data in player_stats.items():
    speeds = data["speeds"]
    team = data["team"]
    rows.append({
        "player_id": pid,
        "team": team,
        "avg_speed_kmh": round(np.mean(speeds), 2) if speeds else 0,
        "max_speed_kmh": round(max(speeds), 2) if speeds else 0,
        "total_distance_m": round(data["distance"], 2),
        "possession_pct": round(possession_counts.get(team, 0) / total_poss * 100, 2),
    })

df = pd.DataFrame(rows).sort_values("max_speed_kmh", ascending=False)
df.to_csv('../output_videos/match_stats.csv', index=False)
display(df.head(10))
```

- [ ] **Step 3: Commit**

```bash
git add football_analysis/heatmap_generator/ notebooks/07_heatmaps_and_stats.ipynb
git commit -m "feat: add HeatmapGenerator module and final notebook"
```

---

## Task 10: CLI (`main.py`)

**Files:**
- Create: `main.py`

- [ ] **Step 1: Write `main.py`**

```python
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


def run_pipeline(input_path, output_video_path, heatmap_dir, stats_path,
                 model_path="models/best.pt",
                 track_stub="stubs/track_stubs.pkl",
                 cam_stub="stubs/camera_movement_stub.pkl"):

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

    print("Drawing annotations...")
    output_frames = tracker.draw_annotations(frames, tracks, team_ball_control)
    output_frames = cam_est.draw_camera_movement(output_frames, camera_movement)
    SpeedAndDistanceEstimator().draw_speed_and_distance(output_frames, tracks)

    print(f"Saving video to {output_video_path}...")
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    save_video(output_frames, output_video_path)

    print("Generating heatmaps...")
    HeatmapGenerator().generate(tracks, heatmap_dir)

    print("Exporting stats CSV...")
    _export_stats(tracks, team_ball_control, stats_path)

    print("Done.")
    return tracks, team_ball_control


def _export_stats(tracks, team_ball_control, output_path):
    player_data = defaultdict(lambda: {"speeds": [], "distance": 0.0, "team": 0})
    for frame_tracks in tracks["players"]:
        for pid, info in frame_tracks.items():
            if "speed" in info:
                player_data[pid]["speeds"].append(info["speed"])
            player_data[pid]["distance"] = max(player_data[pid]["distance"], info.get("distance", 0.0))
            player_data[pid]["team"] = info.get("team", 0)

    poss = {1: int(np.sum(team_ball_control == 1)),
            2: int(np.sum(team_ball_control == 2))}
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
```

- [ ] **Step 2: Run the full pipeline end-to-end**

```bash
python main.py
```

Expected output (takes several minutes on first run):
```
Reading video...
Tracking objects...
Estimating camera movement...
Applying perspective transform...
Interpolating ball positions...
Computing speed and distance...
Assigning teams...
Assigning ball possession...
Drawing annotations...
Saving video to output_videos/output_video.avi...
Generating heatmaps...
Exporting stats CSV...
Done.
```

Check outputs exist:
```bash
ls output_videos/
# output_video.avi  heatmaps/  match_stats.csv
```

- [ ] **Step 3: Commit**

```bash
git add main.py
git commit -m "feat: add CLI main.py — wires full pipeline, exports stats CSV"
```

---

## Task 11: Streamlit App

**Files:**
- Create: `app.py`

- [ ] **Step 1: Write `app.py`**

```python
import os
import tempfile

import numpy as np
import streamlit as st

st.set_page_config(page_title="Football Analysis", layout="wide")
st.title("Football Match Analysis")
st.caption("Upload a match clip to get an annotated video, player heatmaps, and match stats.")

uploaded = st.file_uploader("Upload a match video", type=["mp4", "avi"], help="Max 200 MB")

if uploaded is not None:
    if uploaded.size > 200 * 1024 * 1024:
        st.error("File exceeds the 200 MB limit. Please upload a shorter clip.")
        st.stop()

    with tempfile.TemporaryDirectory() as tmpdir:
        input_path = os.path.join(tmpdir, "input" + os.path.splitext(uploaded.name)[1])
        output_video_path = os.path.join(tmpdir, "output.avi")
        heatmap_dir = os.path.join(tmpdir, "heatmaps")
        stats_path = os.path.join(tmpdir, "match_stats.csv")

        with open(input_path, "wb") as f:
            f.write(uploaded.read())

        with st.spinner("Running analysis pipeline… this may take several minutes."):
            from main import run_pipeline
            run_pipeline(
                input_path=input_path,
                output_video_path=output_video_path,
                heatmap_dir=heatmap_dir,
                stats_path=stats_path,
                # no stubs in app mode — process fresh each time
                track_stub=None,
                cam_stub=None,
            )

        st.success("Analysis complete!")

        # ── Annotated video ───────────────────────────────────────────────
        st.subheader("Annotated Video")
        if os.path.exists(output_video_path):
            with open(output_video_path, "rb") as f:
                st.download_button(
                    "Download annotated video",
                    f,
                    file_name="output_video.avi",
                    mime="video/avi",
                )
            st.info("Open the downloaded file in VLC or QuickTime to watch it.")

        # ── Heatmaps ──────────────────────────────────────────────────────
        st.subheader("Player Heatmaps")
        heatmap_files = (
            sorted(f for f in os.listdir(heatmap_dir) if f.endswith(".png"))
            if os.path.isdir(heatmap_dir)
            else []
        )
        if heatmap_files:
            cols = st.columns(len(heatmap_files))
            for col, hf in zip(cols, heatmap_files):
                col.image(
                    os.path.join(heatmap_dir, hf),
                    caption=hf.replace("_", " ").replace(".png", "").title(),
                    use_column_width=True,
                )
        else:
            st.info("No heatmaps generated — not enough position data in this clip.")

        # ── Stats table ───────────────────────────────────────────────────
        if os.path.exists(stats_path):
            import pandas as pd

            st.subheader("Match Statistics")
            df = pd.read_csv(stats_path)
            st.dataframe(df.sort_values("max_speed_kmh", ascending=False), use_container_width=True)
            with open(stats_path, "rb") as f:
                st.download_button(
                    "Download stats CSV",
                    f,
                    file_name="match_stats.csv",
                    mime="text/csv",
                )
```

- [ ] **Step 2: Run the Streamlit app**

```bash
streamlit run app.py
```

Open http://localhost:8501 in a browser. Upload a short clip and verify:
- Progress spinner appears while pipeline runs
- Download button appears for the annotated video
- Heatmap images display
- Stats table displays and is sortable
- CSV download button works

- [ ] **Step 3: Commit**

```bash
git add app.py
git commit -m "feat: add Streamlit web app (upload → annotated video + heatmaps + stats)"
```

---

## Task 12: Training Notebook (Colab — run separately)

**Files:**
- Create: `training/yolo_football_training.ipynb`

This notebook runs on Google Colab with a free GPU. It is standalone — run it once to produce `best.pt`, then download and place in `models/`.

- [ ] **Step 1: Create `training/yolo_football_training.ipynb`**

Create a notebook with these cells:

**Cell 1 — Markdown:**
```
# YOLO Football Fine-Tuning

Fine-tune YOLOv8x on a Roboflow football detection dataset.
Classes: player, goalkeeper, referee, ball.

Run this on Google Colab (Runtime → Change runtime type → T4 GPU).
Download best.pt when done and place it in models/.
```

**Cell 2 — Install:**
```python
!pip install ultralytics roboflow -q
```

**Cell 3 — Download dataset:**
```python
from roboflow import Roboflow

rf = Roboflow(api_key="YOUR_ROBOFLOW_API_KEY")
project = rf.workspace("roboflow-jvuqo").project("football-players-detection-3zvbc")
version = project.version(1)
dataset = version.download("yolov8")
```

**Cell 4 — Train:**
```python
from ultralytics import YOLO

model = YOLO("yolov8x.pt")
results = model.train(
    data=f"{dataset.location}/data.yaml",
    epochs=100,
    imgsz=640,
    batch=16,
    project="football_training",
    name="run1",
    device=0,
)
```

**Cell 5 — Evaluate and visualise results:**
```python
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

metrics = model.val()
print(f"mAP50:    {metrics.box.map50:.3f}")
print(f"mAP50-95: {metrics.box.map:.3f}")

# Show training curves
img = mpimg.imread('football_training/run1/results.png')
plt.figure(figsize=(16, 8))
plt.imshow(img)
plt.axis('off')
plt.title('Training curves')
plt.show()
```

**Cell 6 — Download weights:**
```python
from google.colab import files
files.download('football_training/run1/weights/best.pt')
```

- [ ] **Step 2: Commit**

```bash
git add training/yolo_football_training.ipynb
git commit -m "feat: add Colab YOLO fine-tuning notebook"
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] Project setup + directory structure → Task 1
- [x] Utils (video_utils, bbox_utils) + smoke test → Task 2
- [x] Tracker (YOLO + ByteTrack + stubs + draw_annotations) → Task 3
- [x] Team Assigner (K-means jersey) → Task 4
- [x] Camera Movement Estimator (optical flow, adjusted positions) → Task 5
- [x] View Transformer (homography, pixel → metres) → Task 6
- [x] Speed & Distance Estimator → Task 7
- [x] Player-Ball Assigner → Task 8
- [x] Heatmap Generator → Task 9
- [x] main.py + stats CSV export → Task 10
- [x] Streamlit app (upload, video, heatmaps, table, downloads) → Task 11
- [x] YOLO fine-tuning notebook (Colab) → Task 12
- [x] 7 learning notebooks → Tasks 3–9

**Key invariant confirmed:** `add_adjust_positions_to_tracks` in Task 5 subtracts camera displacement before `add_transformed_position_to_tracks` in Task 6 runs. The `position_adjusted` key is the input to the view transformer, not `position`.

**Type consistency:** `position_adjusted` (set in Task 5, read in Task 6), `position_transformed` (set in Task 6, read in Tasks 7 and 9), `speed`/`distance` (set in Task 7, read in Tasks 10–11) — all consistent across tasks.
