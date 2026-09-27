"""Input form and configuration component for initiating pipeline jobs."""

import re
import streamlit as st

from src.core.contracts import JobStatus, PipelineConfig, PipelineJob
from src.engine import PipelineOrchestrator
from src.core.events import JobContext
from src.ui.state import AppStateManager, get_event_emitter, get_thread_pool


URL_REGEX = re.compile(
    r"^(https?://(?:www\.)?(?:youtube\.com|youtu\.be|vimeo\.com)/.+|.+\.(?:mp4|mkv|mov|webm))$",
    re.IGNORECASE,
)


DOMAIN_PRESETS = {
    "Automotive & Mechanics": {
        "hotwords": "βαλβίδα PCV, αναθυμιάσεις, κινητήρας, λάδια, συνεργείο, κάρτερ, εκκεντροφόρος, μπουζί, συμπίεση, φίλτρο, τουρμπίνα, σασμάν",
        "prompt": "Ελληνική συνομιλία, ακριβής ορθογραφία, σημεία στίξης, τόνοι, μηχανικά μέρη αυτοκινήτων, βαλβίδα PCV, βαλβίδα αναθυμιάσεων, κινητήρας, συνεργείο, λάδια.",
    },
    "Technology & Software": {
        "hotwords": "Python, JavaScript, API, backend, frontend, database, Docker, Kubernetes, AI, machine learning, cloud",
        "prompt": "Ελληνική συνομιλία, προγραμματισμός, τεχνολογία, λογισμικό, ανάπτυξη εφαρμογών, κώδικας, βάσεις δεδομένων.",
    },
    "Business & Finance": {
        "hotwords": "επένδυση, κέρδος, μετοχές, εταιρεία, μάρκετινγκ, στρατηγική, πωλήσεις, χρηματοοικονομικά, startup",
        "prompt": "Ελληνική συνομιλία, επιχειρηματικότητα, οικονομία, επενδύσεις, στρατηγική, ανάπτυξη πωλήσεων.",
    },
    "General Conversation & Podcast": {
        "hotwords": "συζήτηση, άποψη, εμπειρία, ιστορία, συνέντευξη, podcast",
        "prompt": "Ελληνική συνομιλία, ακριβής ορθογραφία, σημεία στίξης, τόνοι, φυσική ροή λόγου.",
    },
}


def render_input_form() -> None:
    """Render URL input and pipeline configuration controls."""
    st.subheader("🎬 New Video Extraction Job")
    st.caption("Enter a YouTube, Vimeo, or local video path to extract high-virality 9:16 clips.")

    with st.form("pipeline_input_form", clear_on_submit=False):
        video_url = st.text_input(
            "Video URL or Local File Path",
            placeholder="https://www.youtube.com/watch?v=... or C:/path/to/video.mp4",
            help="Supports YouTube, Vimeo links, or direct local video files.",
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            max_clips = st.number_input(
                "Max Clips",
                min_value=1,
                max_value=10,
                value=3,
                help="Maximum number of candidate clips to score and propose.",
            )
        with col2:
            min_duration = st.slider(
                "Min Clip Duration (s)",
                min_value=15,
                max_value=60,
                value=25,
            )
        with col3:
            max_duration = st.slider(
                "Max Clip Duration (s)",
                min_value=30,
                max_value=120,
                value=60,
            )

        col4, col5 = st.columns(2)
        with col4:
            enable_greek_guardrail = st.toggle(
                "Greek Orthographic Guardrail (LLM)",
                value=True,
                help="Applies contextual Greek grammar and accent (τόνοι) normalization with Gemini without altering timestamps.",
            )
            enable_virality_scoring = st.toggle(
                "AI Semantic Scanning & Narrative Scoring",
                value=True,
                help="When enabled, analyzes narrative units with Gemini for 4-pillar logic, standalone sense, and hook strength.",
            )
        with col5:
            virality_threshold = st.slider(
                "Virality / Coherence Threshold",
                min_value=50,
                max_value=95,
                value=65,
                help="Candidate story units below this score will be pruned (only applies if AI scoring is enabled).",
            )

        col6, col7 = st.columns(2)
        with col6:
            whisper_model_choice = st.selectbox(
                "Whisper Speech Model",
                options=["large-v3-turbo (Recommended)", "medium", "small", "base"],
                index=0,
                help="large-v3-turbo provides state-of-the-art Greek accuracy, punctuation, and technical vocabulary.",
            )
            domain_choice = st.selectbox(
                "Domain Vocabulary Preset",
                options=list(DOMAIN_PRESETS.keys()),
                index=0,
                help="Pre-tunes Whisper and Gemini with specialized technical vocabulary.",
            )
        with col7:
            preset_data = DOMAIN_PRESETS.get(domain_choice, DOMAIN_PRESETS["Automotive & Mechanics"])
            custom_hotwords = st.text_input(
                "Vocabulary Hotwords (Bias ASR)",
                value=preset_data["hotwords"],
                help="Comma-separated keywords directly biasing Whisper's acoustic decoder toward correct terms.",
            )
            custom_prompt = st.text_input(
                "Domain Context / Prompt (Optional)",
                value=preset_data["prompt"],
                help="Contextual hints guiding Whisper and Gemini on Greek terminology.",
            )

        submitted = st.form_submit_button("🚀 Start Pipeline", use_container_width=True, type="primary")

        if submitted:
            clean_url = video_url.strip()
            if not clean_url:
                st.error("Please provide a valid video URL or file path.")
                return

            if min_duration >= max_duration:
                st.error("Minimum duration must be less than maximum duration.")
                return

            selected_model = whisper_model_choice.split(" ")[0].strip()

            # Instantiate config
            cfg = PipelineConfig(
                max_clips=int(max_clips),
                min_clip_duration=float(min_duration),
                max_clip_duration=float(max_duration),
                enable_greek_guardrail=enable_greek_guardrail,
                enable_virality_scoring=enable_virality_scoring,
                virality_threshold=int(virality_threshold) if enable_virality_scoring else None,
                whisper_model=selected_model,
                domain_preset=domain_choice,
                hotwords=custom_hotwords.strip() if custom_hotwords.strip() else None,
                transcription_prompt=custom_prompt.strip() if custom_prompt.strip() else None,
            )

            # Create new job
            job = PipelineJob(
                url=clean_url,
                config=cfg,
                status=JobStatus.DOWNLOADING,
                status_message="Initiating background pipeline...",
            )
            AppStateManager.update_job(job)

            # Dispatch background worker
            _dispatch_discovery_job(job)
            st.rerun()


def _dispatch_discovery_job(job: PipelineJob) -> None:
    """Submit the discovery phase (Phase 1-3) to the background thread pool."""
    pool = get_thread_pool()
    emitter = get_event_emitter()
    emitter.clear()
    context = JobContext(job_id=job.job_id, emitter=emitter)

    def worker_task():
        orchestrator = PipelineOrchestrator(context=context)
        orchestrator.run_discovery_phase(job)

    pool.submit(worker_task)
