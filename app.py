import os
import tempfile

import streamlit as st

st.set_page_config(page_title="Football Analysis", layout="wide")
st.title("Football Match Analysis")
st.caption("Upload a match clip to get an annotated video, player heatmaps, and match stats.")

uploaded = st.file_uploader("Upload a match video", type=["mp4", "avi"], help="Max 200 MB")

if uploaded is not None:
    if uploaded.size > 200 * 1024 * 1024:
        st.error("File exceeds the 200 MB limit. Please upload a shorter clip.")
        st.stop()

    # Use session_state so Streamlit reruns don't re-run the whole pipeline.
    cache_key = f"{uploaded.name}_{uploaded.size}"
    if st.session_state.get("cache_key") != cache_key:
        st.session_state.cache_key = cache_key
        st.session_state.result_dir = None

    if st.session_state.result_dir is None:
        result_dir = tempfile.mkdtemp()
        input_path = os.path.join(result_dir, "input" + os.path.splitext(uploaded.name)[1])
        output_video_path = os.path.join(result_dir, "output.avi")
        heatmap_dir = os.path.join(result_dir, "heatmaps")
        stats_path = os.path.join(result_dir, "match_stats.csv")

        with open(input_path, "wb") as f:
            f.write(uploaded.read())

        with st.spinner("Running analysis pipeline… this may take several minutes."):
            from main import run_pipeline
            run_pipeline(
                input_path=input_path,
                output_video_path=output_video_path,
                heatmap_dir=heatmap_dir,
                stats_path=stats_path,
                track_stub=None,
                cam_stub=None,
            )

        st.session_state.result_dir = result_dir
        st.session_state.output_video_path = output_video_path
        st.session_state.heatmap_dir = heatmap_dir
        st.session_state.stats_path = stats_path

    st.success("Analysis complete!")

    output_video_path = st.session_state.output_video_path
    heatmap_dir = st.session_state.heatmap_dir
    stats_path = st.session_state.stats_path

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
        st.info("Open the downloaded file in VLC to watch it.")

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
