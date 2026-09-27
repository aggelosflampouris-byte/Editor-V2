"""Domain-specific exception hierarchy for the video clipping and editing pipeline."""

from typing import Optional


class PipelineError(Exception):
    """Base exception for all pipeline errors."""

    def __init__(self, message: str, phase: Optional[str] = None, details: Optional[str] = None):
        super().__init__(message)
        self.message = message
        self.phase = phase
        self.details = details

    def __str__(self) -> str:
        prefix = f"[{self.phase}] " if self.phase else ""
        if self.details:
            return f"{prefix}{self.message} | Details: {self.details}"
        return f"{prefix}{self.message}"


class InvalidStateTransitionError(PipelineError):
    """Raised when an illegal state machine transition is attempted."""

    def __init__(self, current_state: str, target_state: str):
        super().__init__(
            message=f"Invalid state transition from '{current_state}' to '{target_state}'",
            phase="STATE_MACHINE",
        )
        self.current_state = current_state
        self.target_state = target_state


class IngestionError(PipelineError):
    """Raised when video/audio ingestion or ffmpeg normalization fails."""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message=message, phase="INGESTION", details=details)


class AlignmentError(PipelineError):
    """Raised when transcription, phoneme alignment, or Greek orthography guardrails fail."""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message=message, phase="SPEECH_ALIGNMENT", details=details)


class IntelligenceError(PipelineError):
    """Raised when scene boundary detection or multimodal virality scoring fails."""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message=message, phase="INTELLIGENCE", details=details)


class VisionError(PipelineError):
    """Raised during face tracking, active speaker identification, or framing computation."""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message=message, phase="VISION", details=details)


class RenderingError(PipelineError):
    """Raised during subtitle generation, audio ducking, or FFmpeg hardware rendering."""

    def __init__(self, message: str, details: Optional[str] = None):
        super().__init__(message=message, phase="RENDERING", details=details)
