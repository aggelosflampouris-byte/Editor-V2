"""Human-in-the-Loop (HITL) review cards and candidate approval workspace."""

from typing import List
import streamlit as st

from src.core.contracts import ClipProposal, JobStatus, LayoutMode
from src.core.events import JobContext
from src.engine import PipelineOrchestrator
from src.ui.state import AppStateManager, get_event_emitter, get_thread_pool


@st.fragment
def render_clip_card(clip: ClipProposal) -> None:
    """
    Renders an editable card for an individual candidate clip.
    Isolated via @st.fragment to prevent whole-page re-evaluations during text editing.
    """
    with st.container(border=True):
        col_header, col_status = st.columns([4, 1])
        with col_header:
            virality_text = f"Virality: **{clip.virality_score}/100**" if clip.virality_score is not None else "Virality: *Optional*"
            st.markdown(f"#### Clip `{clip.clip_id}` ({clip.duration:.1f}s) — {virality_text}")
        with col_status:
            is_approved = st.checkbox(
                "Approve",
                value=clip.is_approved,
                key=f"approve_{clip.clip_id}",
                help="Include this clip in the final hardware render batch.",
            )

        col_left, col_right = st.columns([1, 1])
        with col_left:
            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                hook_display = f"{clip.hook_score}/100" if clip.hook_score is not None else "N/A"
                st.metric("Hook", hook_display)
            with col_m2:
                semantic_display = f"{clip.semantic_score}/100" if clip.semantic_score is not None else "N/A"
                st.metric("Logic / Sense", semantic_display)
            with col_m3:
                payoff_display = f"{clip.payoff_score}/100" if clip.payoff_score is not None else "N/A"
                st.metric("Payoff", payoff_display)

            st.caption(f"**Timing:** `{clip.start_time:.2f}s` to `{clip.end_time:.2f}s`")
            st.caption(f"**Narrative Assessment:** {clip.reasoning}")

            layout_options = [e.value for e in LayoutMode]
            selected_layout = st.selectbox(
                "Framing Layout Mode",
                options=layout_options,
                index=layout_options.index(clip.layout_mode.value),
                key=f"layout_{clip.clip_id}",
            )

        with col_right:
            hook_text = st.text_input(
                "Opening Hook Caption",
                value=clip.hook_text,
                key=f"hook_{clip.clip_id}",
                help="Opening 3-second kinetic hook text displayed on screen.",
            )
            seo_title = st.text_input(
                "SEO Title",
                value=clip.seo_title,
                key=f"title_{clip.clip_id}",
            )
            seo_description = st.text_area(
                "SEO Description",
                value=clip.seo_description,
                key=f"desc_{clip.clip_id}",
                height=80,
            )

            # Available tags
            tags_str = ", ".join(clip.seo_tags)
            updated_tags_raw = st.text_input(
                "SEO Tags (comma-separated)",
                value=tags_str,
                key=f"tags_{clip.clip_id}",
            )
            updated_tags = [t.strip() for t in updated_tags_raw.split(",") if t.strip()]

        # Persist changes to session state
        AppStateManager.update_clip_proposal(
            clip_id=clip.clip_id,
            hook_text=hook_text,
            seo_title=seo_title,
            seo_description=seo_description,
            seo_tags=updated_tags,
            is_approved=is_approved,
            layout_mode=LayoutMode(selected_layout),
        )


def render_hitl_workspace() -> None:
    """Render the staged HITL review workspace after Phase 3 completion."""
    job = AppStateManager.get_job()

    st.subheader("🎯 Human-in-the-Loop Review Workspace")
    st.info(
        "Phase 1–3 completed. Inspect proposed clips, verify Greek accents and phrasing, "
        "and approve the candidates you wish to render in 9:16 portrait format."
    )

    if not job.proposed_clips:
        st.warning("No candidate clips were discovered meeting the virality criteria.")
        if st.button("⬅ Back to Input", type="secondary"):
            AppStateManager.reset_job()
            st.rerun()
        return

    # Render each clip inside its isolated fragment
    for clip in job.proposed_clips:
        render_clip_card(clip)

    approved_count = sum(1 for c in job.proposed_clips if c.is_approved)

    st.divider()
    col_back, col_render = st.columns([1, 2])
    with col_back:
        if st.button("Cancel / Restart", use_container_width=True):
            AppStateManager.reset_job()
            st.rerun()

    with col_render:
        render_clicked = st.button(
            f"🎬 Render {approved_count} Approved Clip(s)",
            type="primary",
            use_container_width=True,
            disabled=(approved_count == 0),
        )

        if render_clicked:
            _dispatch_rendering_job(job)
            st.rerun()


def _dispatch_rendering_job(job) -> None:
    """Dispatch post-production render phase (Phase 4-6) to background thread."""
    approved_clips = [c for c in job.proposed_clips if c.is_approved]
    AppStateManager.transition_state(JobStatus.RENDERING, f"Rendering {len(approved_clips)} approved clips...")

    pool = get_thread_pool()
    emitter = get_event_emitter()
    emitter.clear()
    context = JobContext(job_id=job.job_id, emitter=emitter)

    def render_worker():
        orchestrator = PipelineOrchestrator(context=context)
        orchestrator.run_render_phase(job, approved_clips)

    pool.submit(render_worker)
