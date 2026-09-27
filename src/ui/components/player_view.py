"""Delivery and export component for inspecting rendered 9:16 clips and downloading assets."""

from pathlib import Path
import streamlit as st

from src.ui.state import AppStateManager


def render_export_player_view() -> None:
    """Displays rendered video players, package download buttons, and social copy."""
    job = AppStateManager.get_job()

    st.subheader("🎉 Production Delivery & Export")
    st.success("All approved clips have been rendered in 1080x1920 portrait format with kinetic subtitles!")

    workspace_dir = Path("outputs") / job.job_id
    mp4_files = sorted(list(workspace_dir.glob("*.mp4"))) if workspace_dir.exists() else []

    if not mp4_files:
        st.info("Rendered files are located in your output directory.")

    tabs = st.tabs([f"Clip {idx + 1}" for idx in range(max(1, len(mp4_files)))])

    for idx, tab in enumerate(tabs):
        with tab:
            if idx < len(mp4_files):
                video_file = mp4_files[idx]
                col_video, col_meta = st.columns([1, 1])

                with col_video:
                    st.video(str(video_file))
                    with open(video_file, "rb") as f:
                        video_bytes = f.read()

                    st.download_button(
                        label=f"⬇ Download Video ({video_file.name})",
                        data=video_bytes,
                        file_name=video_file.name,
                        mime="video/mp4",
                        use_container_width=True,
                    )

                with col_meta:
                    metadata_file = video_file.with_name(video_file.stem + "_metadata.json")
                    if metadata_file.exists():
                        with open(metadata_file, "r", encoding="utf-8") as mf:
                            meta_content = mf.read()

                        st.markdown("#### Social Media Copy")
                        st.code(meta_content, language="json")

                        st.download_button(
                            label="⬇ Download SEO Package (JSON)",
                            data=meta_content,
                            file_name=f"{video_file.stem}_seo.json",
                            mime="application/json",
                            use_container_width=True,
                        )
            else:
                st.info("No video file generated for this slot.")

    st.divider()
    if st.button("🔄 Process Another Video", type="secondary"):
        AppStateManager.reset_job()
        st.rerun()
