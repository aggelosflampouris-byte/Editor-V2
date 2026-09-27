"""Streamlit session state management, state machine transitions, and thread-safe worker bridge."""

from __future__ import annotations

import concurrent.futures
from typing import Dict, List, Optional, Set
import streamlit as st

from src.core.contracts import (
    ClipProposal,
    JobStatus,
    PipelineConfig,
    PipelineJob,
    PipelinePhase,
    RenderedClip,
    LayoutMode,
)
from src.core.events import EventEmitter, JobContext, PipelineEvent, PipelineEventType
from src.core.exceptions import InvalidStateTransitionError


# Allowed state transitions enforcing the pipeline state machine
ALLOWED_TRANSITIONS: Dict[JobStatus, Set[JobStatus]] = {
    JobStatus.IDLE: {JobStatus.DOWNLOADING, JobStatus.FAILED},
    JobStatus.DOWNLOADING: {JobStatus.TRANSCRIBING, JobStatus.FAILED},
    JobStatus.TRANSCRIBING: {JobStatus.SCORING, JobStatus.FAILED},
    JobStatus.SCORING: {JobStatus.READY_FOR_REVIEW, JobStatus.FAILED},
    JobStatus.READY_FOR_REVIEW: {JobStatus.RENDERING, JobStatus.IDLE, JobStatus.FAILED},
    JobStatus.RENDERING: {JobStatus.COMPLETED, JobStatus.FAILED},
    JobStatus.COMPLETED: {JobStatus.IDLE},
    JobStatus.FAILED: {JobStatus.IDLE},
}


@st.cache_resource
def get_thread_pool() -> concurrent.futures.ThreadPoolExecutor:
    """Singleton thread pool executor for background pipeline tasks."""
    return concurrent.futures.ThreadPoolExecutor(max_workers=2, thread_name_prefix="pipeline_worker")


@st.cache_resource
def get_event_emitter() -> EventEmitter:
    """Singleton thread-safe event emitter bridge."""
    return EventEmitter()


class AppStateManager:
    """Enforces deterministic state transitions and manages the UI-engine boundary."""

    SESSION_JOB_KEY = "pipeline_job"
    SESSION_EVENTS_KEY = "pipeline_events"
    SESSION_INITIALIZED_KEY = "state_initialized"

    @classmethod
    def _get_store(cls):
        try:
            _ = st.session_state.get(cls.SESSION_INITIALIZED_KEY, False)
            return st.session_state
        except Exception:
            if not hasattr(cls, "_test_store"):
                cls._test_store = {}
            return cls._test_store

    @classmethod
    def initialize_session(cls) -> None:
        """Initialize session state keys once per user session."""
        store = cls._get_store()
        if not store.get(cls.SESSION_INITIALIZED_KEY, False):
            store[cls.SESSION_JOB_KEY] = PipelineJob(
                url="",
                status=JobStatus.IDLE,
                status_message="System initialized. Ready for input.",
            )
            store[cls.SESSION_EVENTS_KEY] = []
            store[cls.SESSION_INITIALIZED_KEY] = True

    @classmethod
    def get_job(cls) -> PipelineJob:
        """Get the current typed PipelineJob from session state."""
        cls.initialize_session()
        store = cls._get_store()
        return store[cls.SESSION_JOB_KEY]

    @classmethod
    def update_job(cls, job: PipelineJob) -> None:
        """Store the updated PipelineJob into session state."""
        store = cls._get_store()
        store[cls.SESSION_JOB_KEY] = job

    @classmethod
    def transition_state(cls, target_status: JobStatus, status_message: Optional[str] = None) -> None:
        """Validate and transition the job status according to the state machine contract."""
        job = cls.get_job()
        allowed = ALLOWED_TRANSITIONS.get(job.status, set())
        if target_status not in allowed:
            raise InvalidStateTransitionError(
                current_state=job.status.value,
                target_state=target_status.value,
            )

        job.status = target_status
        if status_message:
            job.status_message = status_message
        cls.update_job(job)

    @classmethod
    def drain_and_apply_events(cls) -> bool:
        """
        Drain pending events from the background emitter into session state.
        Returns True if a major status change occurred that warrants a view rerender.
        """
        emitter = get_event_emitter()
        events: List[PipelineEvent] = emitter.drain_events()
        if not events:
            return False

        job = cls.get_job()
        status_changed = False

        for event in events:
            if event.event_type == PipelineEventType.LOG_EMITTED:
                job.logs.append(event.message)
            elif event.event_type == PipelineEventType.PROGRESS_UPDATED:
                job.current_phase = event.phase
                job.progress_percentage = event.progress
                job.status_message = event.message
            elif event.event_type == PipelineEventType.STATUS_CHANGED and event.status:
                if event.status != job.status and event.status in ALLOWED_TRANSITIONS.get(job.status, set()):
                    job.status = event.status
                    job.status_message = event.message
                    status_changed = True
            elif event.event_type == PipelineEventType.TRANSCRIPT_READY:
                if event.transcript:
                    job.transcript = list(event.transcript)
            elif event.event_type == PipelineEventType.CLIPS_PROPOSED:
                job.status = JobStatus.READY_FOR_REVIEW
                job.current_phase = PipelinePhase.HITL_GATE
                job.status_message = event.message
                if event.proposed_clips:
                    job.proposed_clips = list(event.proposed_clips)
                status_changed = True
            elif event.event_type == PipelineEventType.CLIPS_RENDERED:
                if event.rendered_clip:
                    job.rendered_clips.append(event.rendered_clip)
            elif event.event_type == PipelineEventType.JOB_COMPLETED:
                job.status = JobStatus.COMPLETED
                job.progress_percentage = 100.0
                job.status_message = event.message
                status_changed = True
            elif event.event_type == PipelineEventType.JOB_FAILED:
                job.status = JobStatus.FAILED
                job.status_message = event.message
                job.error_log = event.message
                status_changed = True

        cls.update_job(job)
        return status_changed

    @classmethod
    def update_clip_proposal(
        cls,
        clip_id: str,
        hook_text: str,
        seo_title: str,
        seo_description: str,
        seo_tags: List[str],
        is_approved: bool,
        layout_mode: Optional[LayoutMode] = None,
    ) -> None:
        """Update a specific clip proposal modified by the human operator."""
        job = cls.get_job()
        for idx, clip in enumerate(job.proposed_clips):
            if clip.clip_id == clip_id:
                updates = {
                    "hook_text": hook_text,
                    "seo_title": seo_title,
                    "seo_description": seo_description,
                    "seo_tags": seo_tags,
                    "is_approved": is_approved,
                }
                if layout_mode is not None:
                    updates["layout_mode"] = layout_mode
                updated_clip = clip.model_copy(
                    update=updates
                )
                job.proposed_clips[idx] = updated_clip
                break
        cls.update_job(job)

    @classmethod
    def reset_job(cls) -> None:
        """Reset the session back to IDLE state."""
        emitter = get_event_emitter()
        emitter.clear()
        store = cls._get_store()
        store[cls.SESSION_JOB_KEY] = PipelineJob(
            url="",
            status=JobStatus.IDLE,
            status_message="Ready for new job.",
        )
