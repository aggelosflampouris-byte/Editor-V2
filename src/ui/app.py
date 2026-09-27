"""Main Streamlit entrypoint for Antigravity Video Engine."""

from pathlib import Path
import shutil
from typing import Tuple
import streamlit as st

# Patch torch.classes.__getattr__ for Streamlit's local_sources_watcher compatibility
try:
    import torch
    orig_classes_getattr = torch._classes._Classes.__getattr__

    def _safe_torch_getattr(self, attr: str):
        if attr in ("__path__", "__file__"):
            raise AttributeError(attr)
        return orig_classes_getattr(self, attr)

    torch._classes._Classes.__getattr__ = _safe_torch_getattr
except Exception:
    pass

from src.config import config
from src.core.contracts import JobStatus
from src.ui.components.clip_card import render_hitl_workspace
from src.ui.components.input_form import render_input_form
from src.ui.components.player_view import render_export_player_view
from src.ui.components.progress_view import render_live_progress
from src.ui.state import AppStateManager


@st.cache_resource
def get_device_diagnostics() -> Tuple[bool, str]:
    """Inspect GPU capability once per server lifetime."""
    try:
        import torch
        if torch.cuda.is_available():
            return True, torch.cuda.get_device_name(0)
    except Exception:
        pass
    return False, "CPU Mode"


def setup_page() -> None:
    """Configure modern high-density page styling."""
    st.set_page_config(
        page_title="Antigravity Video Engine",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom styling
    st.markdown(
        """
        <style>
        .stButton>button {
            border-radius: 8px;
            font-weight: 600;
        }
        .main-header {
            font-size: 2.2rem;
            font-weight: 800;
            background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render system diagnostics and environment capabilities."""
    with st.sidebar:
        st.markdown("### ⚙️ Engine Diagnostics")

        # Hardware & dependencies status
        has_ffmpeg = (
            shutil.which(config.ffmpeg_binary) is not None
            or Path(config.ffmpeg_binary).is_file()
        )
        has_ytdlp = shutil.which("yt-dlp") is not None
        try:
            import yt_dlp
            has_ytdlp = True
        except ImportError:
            pass

        st.markdown(f"**FFmpeg:** {'✅ Available' if has_ffmpeg else '⚠️ Not Found'}")
        st.markdown(f"**yt-dlp:** {'✅ Available' if has_ytdlp else '⚠️ Not Found'}")
        st.markdown(f"**Whisper Model:** `{config.whisper_model}`")
        st.markdown(f"**Target Format:** `1080x1920 (9:16)`")

        st.markdown(f"**GPU Hardware:** `{config.gpu_device_name}`")
        st.markdown(f"**Hardware Encoder:** `{config.video_codec}`")

        st.divider()
        st.markdown("### 🤖 Multimodal AI (Gemini)")
        api_key_status = "✅ Active" if config.gemini_api_key else "⚠️ Optional / Inactive"
        st.markdown(f"**Gemini AI ({config.gemini_model}):** {api_key_status}")

        user_key = st.text_input(
            "Gemini API Key",
            value=config.gemini_api_key or "",
            type="password",
            placeholder="AIzaSy...",
            help="Enables macro-context Greek transcript cleanup and semantic 4-pillar narrative scanning.",
            key="gemini_api_key_input",
        )
        if user_key and user_key != config.gemini_api_key:
            config.set_gemini_api_key(user_key)
            st.success("API Key updated!")
            st.rerun()

        st.divider()
        st.caption("Decoupled Architecture • Pydantic v2 • Zero-Rerun State Machine")


def render_failed_view() -> None:
    """Render diagnostic details when a job fails."""
    job = AppStateManager.get_job()
    st.error("❌ Pipeline Execution Encountered an Error")
    st.markdown(f"**Reason:** {job.status_message}")

    if job.error_log:
        st.code(job.error_log, language="text")

    if job.logs:
        with st.expander("View Full Execution Logs"):
            st.code("\n".join(job.logs), language="log")

    if st.button("🔄 Reset and Retry", type="primary"):
        AppStateManager.reset_job()
        st.rerun()


def main() -> None:
    """Main application router managing stage transitions."""
    setup_page()
    render_sidebar()

    # Header banner
    st.markdown('<div class="main-header">⚡ Antigravity Video Engine</div>', unsafe_allow_html=True)
    st.caption("Automated Greek-Aware Multimodal Video Clipping & 9:16 Reframing Pipeline")
    st.divider()

    # Initialize state
    AppStateManager.initialize_session()
    job = AppStateManager.get_job()

    # Router based on state machine
    if job.status == JobStatus.IDLE:
        render_input_form()

    elif job.status in {JobStatus.DOWNLOADING, JobStatus.TRANSCRIBING, JobStatus.SCORING, JobStatus.RENDERING}:
        render_live_progress()

    elif job.status == JobStatus.READY_FOR_REVIEW:
        render_hitl_workspace()

    elif job.status == JobStatus.COMPLETED:
        render_export_player_view()

    elif job.status == JobStatus.FAILED:
        render_failed_view()


if __name__ == "__main__":
    main()
