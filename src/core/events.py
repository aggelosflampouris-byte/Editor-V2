"""Thread-safe event bridge, observer pattern, and job execution context."""

from __future__ import annotations

import queue
import threading
import time
import uuid
from enum import Enum
from typing import Callable, List, Optional

from pydantic import BaseModel, ConfigDict, Field

from src.core.contracts import (
    ClipProposal,
    JobStatus,
    PipelinePhase,
    RenderedClip,
    TranscriptSegment,
)


class PipelineEventType(str, Enum):
    """Categorized event types emitted during pipeline execution."""
    STATUS_CHANGED = "status_changed"
    PROGRESS_UPDATED = "progress_updated"
    LOG_EMITTED = "log_emitted"
    TRANSCRIPT_READY = "transcript_ready"
    CLIPS_PROPOSED = "clips_proposed"
    CLIPS_RENDERED = "clips_rendered"
    JOB_COMPLETED = "job_completed"
    JOB_FAILED = "job_failed"


class PipelineEvent(BaseModel):
    """Immutable event payload transferred across threads."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    job_id: str
    event_type: PipelineEventType
    phase: PipelinePhase
    timestamp: float = Field(default_factory=time.time)
    message: str
    progress: float = Field(default=0.0, ge=0.0, le=100.0)
    status: Optional[JobStatus] = None
    proposed_clips: Optional[List[ClipProposal]] = None
    transcript: Optional[List[TranscriptSegment]] = None
    rendered_clip: Optional[RenderedClip] = None


class EventEmitter:
    """Thread-safe event dispatcher and event queue broker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._listeners: List[Callable[[PipelineEvent], None]] = []
        self._event_queue: queue.Queue[PipelineEvent] = queue.Queue()

    def subscribe(self, listener: Callable[[PipelineEvent], None]) -> None:
        """Register a synchronous event consumer callback."""
        with self._lock:
            if listener not in self._listeners:
                self._listeners.append(listener)

    def unsubscribe(self, listener: Callable[[PipelineEvent], None]) -> None:
        """Deregister an existing event consumer."""
        with self._lock:
            if listener in self._listeners:
                self._listeners.remove(listener)

    def emit(self, event: PipelineEvent) -> None:
        """Push an event to the queue and broadcast to all registered listeners."""
        self._event_queue.put(event)
        with self._lock:
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(event)
            except Exception:
                # Event dispatching must never crash the emitting engine thread
                pass

    def drain_events(self, max_items: Optional[int] = None) -> List[PipelineEvent]:
        """Drain events from the thread-safe queue non-blockingly."""
        events: List[PipelineEvent] = []
        count = 0
        while True:
            if max_items is not None and count >= max_items:
                break
            try:
                event = self._event_queue.get_nowait()
                events.append(event)
                count += 1
            except queue.Empty:
                break
        return events

    def clear(self) -> None:
        """Discard any unconsumed events in the queue."""
        while not self._event_queue.empty():
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                break


class JobContext:
    """Scoped execution context passed into engine modules to report progress and artifacts."""

    def __init__(self, job_id: str, emitter: Optional[EventEmitter] = None) -> None:
        self.job_id = job_id
        self.emitter = emitter or EventEmitter()
        self._cancel_requested = threading.Event()
        self._lock = threading.Lock()
        self._transcript: List[TranscriptSegment] = []
        self._proposed_clips: List[ClipProposal] = []
        self._rendered_clips: List[RenderedClip] = []
        self._current_phase = PipelinePhase.INITIALIZATION
        self._current_status = JobStatus.IDLE
        self._progress = 0.0

    @property
    def is_cancelled(self) -> bool:
        return self._cancel_requested.is_set()

    def request_cancellation(self) -> None:
        self._cancel_requested.set()
        self.log_warning("Cancellation requested by operator.")

    def set_status(self, status: JobStatus, message: str) -> None:
        with self._lock:
            self._current_status = status
        event = PipelineEvent(
            job_id=self.job_id,
            event_type=PipelineEventType.STATUS_CHANGED,
            phase=self._current_phase,
            status=status,
            message=message,
            progress=self._progress,
        )
        self.emitter.emit(event)

    def update_progress(self, phase: PipelinePhase, percentage: float, message: str) -> None:
        with self._lock:
            self._current_phase = phase
            self._progress = round(max(0.0, min(100.0, percentage)), 2)
        event = PipelineEvent(
            job_id=self.job_id,
            event_type=PipelineEventType.PROGRESS_UPDATED,
            phase=phase,
            status=self._current_status,
            message=message,
            progress=self._progress,
        )
        self.emitter.emit(event)

    def log_info(self, message: str) -> None:
        self._emit_log(f"[INFO] {message}")

    def log_warning(self, message: str) -> None:
        self._emit_log(f"[WARN] {message}")

    def log_error(self, message: str) -> None:
        self._emit_log(f"[ERROR] {message}")

    def _emit_log(self, formatted_message: str) -> None:
        event = PipelineEvent(
            job_id=self.job_id,
            event_type=PipelineEventType.LOG_EMITTED,
            phase=self._current_phase,
            status=self._current_status,
            message=formatted_message,
            progress=self._progress,
        )
        self.emitter.emit(event)

    def set_transcript(self, segments: List[TranscriptSegment]) -> None:
        with self._lock:
            self._transcript = list(segments)
        event = PipelineEvent(
            job_id=self.job_id,
            event_type=PipelineEventType.TRANSCRIPT_READY,
            phase=self._current_phase,
            status=self._current_status,
            message=f"Transcript ready with {len(segments)} segments.",
            progress=self._progress,
            transcript=list(segments),
        )
        self.emitter.emit(event)

    def get_transcript(self) -> List[TranscriptSegment]:
        with self._lock:
            return list(self._transcript)

    def set_proposed_clips(self, clips: List[ClipProposal]) -> None:
        with self._lock:
            self._proposed_clips = list(clips)
        event = PipelineEvent(
            job_id=self.job_id,
            event_type=PipelineEventType.CLIPS_PROPOSED,
            phase=PipelinePhase.HITL_GATE,
            status=JobStatus.READY_FOR_REVIEW,
            message=f"{len(clips)} candidate clips extracted and awaiting operator review.",
            progress=self._progress,
            proposed_clips=list(clips),
        )
        self.emitter.emit(event)

    def get_proposed_clips(self) -> List[ClipProposal]:
        with self._lock:
            return list(self._proposed_clips)

    def add_rendered_clip(self, clip: RenderedClip) -> None:
        with self._lock:
            self._rendered_clips.append(clip)
        event = PipelineEvent(
            job_id=self.job_id,
            event_type=PipelineEventType.CLIPS_RENDERED,
            phase=self._current_phase,
            status=self._current_status,
            message=f"Rendered clip {clip.clip_id} complete ({clip.resolution}).",
            progress=self._progress,
            rendered_clip=clip,
        )
        self.emitter.emit(event)

    def get_rendered_clips(self) -> List[RenderedClip]:
        with self._lock:
            return list(self._rendered_clips)

    def complete(self, message: str = "Pipeline completed successfully.") -> None:
        with self._lock:
            self._current_status = JobStatus.COMPLETED
            self._progress = 100.0
        event = PipelineEvent(
            job_id=self.job_id,
            event_type=PipelineEventType.JOB_COMPLETED,
            phase=PipelinePhase.FINAL_PACKAGING,
            status=JobStatus.COMPLETED,
            message=message,
            progress=100.0,
        )
        self.emitter.emit(event)

    def fail(self, error_message: str) -> None:
        with self._lock:
            self._current_status = JobStatus.FAILED
        event = PipelineEvent(
            job_id=self.job_id,
            event_type=PipelineEventType.JOB_FAILED,
            phase=self._current_phase,
            status=JobStatus.FAILED,
            message=error_message,
            progress=self._progress,
        )
        self.emitter.emit(event)
