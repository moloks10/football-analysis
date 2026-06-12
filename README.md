# Football Match Analysis

End-to-end computer vision pipeline for analysing football match footage. Detects and tracks players, estimates ball possession, measures speed and distance covered, and produces a Streamlit web app for interactive exploration.

## Demo

| Annotated video | Team heatmaps |
|---|---|
| Player IDs, team colours, ball possession %, speed/distance overlays | Gaussian KDE heatmaps per team on a pitch outline |

## Features

- **Object detection** — YOLOv8 fine-tuned on football footage (players, goalkeepers, referees, ball)
- **Multi-object tracking** — ByteTrack for stable player IDs across frames
- **Team assignment** — K-means clustering on jersey colours
- **Camera movement compensation** — Lucas-Kanade optical flow on pitch sideline features
- **Real-world positioning** — homography / perspective transform (pixel → metres)
- **Speed & distance** — derived from real-world positions at 24 fps, displayed in km/h
- **Ball possession** — per-frame nearest-player assignment, shown as running %
- **Heatmaps** — Gaussian KDE of each team's positions on a pitch outline
- **Stats export** — CSV with per-player avg speed, max speed, distance, possession %
- **Streamlit app** — upload a clip, download annotated video, view heatmaps and stats table

## Architecture

```
main.py                   ← pipeline entry point
app.py                    ← Streamlit web UI
football_analysis/
  trackers/               ← YOLO detection + ByteTrack
  team_assigner/          ← K-means jersey clustering
  camera_movement_estimator/  ← Lucas-Kanade optical flow
  view_transformer/       ← perspective transform
  speed_distance_estimator/   ← km/h from real-world coords
  player_ball_assigner/   ← ball → nearest player
  heatmap_generator/      ← Gaussian KDE heatmaps
  utils/                  ← video I/O, bbox helpers
tests/                    ← smoke tests
```

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Place your model and video
#    models/best.pt          ← YOLOv8 weights
#    input_videos/clip.mp4   ← match footage

# 3. Run the pipeline
python main.py

# 4. Or launch the web app
streamlit run app.py
```

Outputs are written to `output_videos/` (annotated AVI, heatmap PNGs, match_stats.csv).

## Memory-efficient design

Processing a 750-frame 1080p clip on a MacBook Air (8 GB unified memory) required careful memory management:

- **Streaming annotation**: frames are never buffered as a list during drawing — `cv2.VideoCapture` reads one frame at a time, `cv2.VideoWriter` writes immediately
- **Lazy frame loading**: when tracking/camera stubs are cached, only the first frame is loaded into RAM; all subsequent steps stream from disk
- **Explicit GC**: bulk frame arrays are `del`'d and `gc.collect()`'d before memory-heavy annotation passes

## Tech stack

`ultralytics` · `supervision` · `opencv-python` · `scikit-learn` · `scipy` · `pandas` · `numpy` · `matplotlib` · `streamlit`

## Credits

Inspired by [abdullahtarek/football_analysis](https://github.com/abdullahtarek/football_analysis). Extended with heatmaps, stats CSV, Streamlit UI, and memory-efficient streaming.
