"""Non-blocking live progress tracking, GPU indicator, and event monitor view."""

import time
import streamlit as st
from src.config import config
from src.core.contracts import JobStatus
from src.ui.state import AppStateManager


@st.fragment(run_every="1s")
def render_live_progress() -> None:
    """
    Isolated reactive progress fragment.
    Polls events every 1 second without triggering full page reruns,
    and advances the main view when a phase transition completes.
    """
    job = AppStateManager.get_job()
    status_changed = AppStateManager.drain_and_apply_events()

    # If status transitioned to a review or terminal state, rerun outer page to switch views
    if job.status not in {JobStatus.DOWNLOADING, JobStatus.TRANSCRIBING, JobStatus.SCORING, JobStatus.RENDERING}:
        st.rerun()
        return

    col_title, col_gpu = st.columns([2, 1])
    with col_title:
        st.subheader("⚡ Processing Video Pipeline")
    with col_gpu:
        st.markdown(
            f"<div style='text-align: right; padding-top: 10px; color: #10b981; font-weight: 600;'>"
            f"🚀 {config.gpu_device_name}"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Progress bar and phase display
    progress_val = min(1.0, max(0.0, job.progress_percentage / 100.0))
    st.progress(progress_val, text=f"{job.status_message} ({job.progress_percentage:.1f}%)")

    # Visual status card
    with st.status(f"Phase: {job.current_phase.value.replace('_', ' ').title()}", expanded=True) as status_box:
        col_st1, col_st2 = st.columns(2)
        with col_st1:
            st.write(f"**Status:** `{job.status.value.upper()}`")
        with col_st2:
            st.write(f"**Hardware:** `{config.video_codec}`")

        st.info(f"📍 **Current Activity:** {job.status_message}")

        # Live terminal output stream: show latest 8 log lines inside the status card
        if job.logs:
            st.markdown("**Live Activity Stream:**")
            recent_logs = job.logs[-8:]
            formatted_logs = "\n".join(recent_logs)
            st.code(formatted_logs, language="bash")

        if job.status in {JobStatus.DOWNLOADING, JobStatus.TRANSCRIBING, JobStatus.SCORING, JobStatus.RENDERING}:
            status_box.update(label=f"Active ({job.progress_percentage:.1f}%): {job.status_message}", state="running")
        elif job.status == JobStatus.COMPLETED:
            status_box.update(label="Complete! (100%)", state="complete")
        elif job.status == JobStatus.FAILED:
            status_box.update(label="Failed", state="error")

    # Full history logs expander placed outside st.status
    if job.logs:
        with st.expander(f"Full Execution History ({len(job.logs)} entries)", expanded=False):
            st.code("\n".join(job.logs), language="log")
