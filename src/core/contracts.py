"""Strict Pydantic v2 data models, state contracts, and domain schemas."""

from __future__ import annotations

import uuid
from enum import Enum
from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class JobStatus(str, Enum):
    """Execution status for the end-to-end pipeline state machine."""
    IDLE = "idle"
    DOWNLOADING = "downloading"
    TRANSCRIBING = "transcribing"
    SCORING = "scoring"
    READY_FOR_REVIEW = "ready_for_review"
    RENDERING = "rendering"
    COMPLETED = "completed"
    FAILED = "failed"


class PipelinePhase(str, Enum):
    """Granular execution phases of the pipeline engine."""
    INITIALIZATION = "initialization"
    INGESTION = "ingestion"
    TRANSCRIPTION = "transcription"
    ALIGNMENT = "alignment"
    ORTHOGRAPHY_GUARDRAIL = "orthography_guardrail"
    SCENE_DETECTION = "scene_detection"
    VIRALITY_SCORING = "virality_scoring"
    HITL_GATE = "hitl_gate"
    VISION_TRACKING = "vision_tracking"
    AUDIO_ENGINEERING = "audio_engineering"
    SUBTITLE_GENERATION = "subtitle_generation"
    VIDEO_ASSEMBLY = "video_assembly"
    FINAL_PACKAGING = "final_packaging"


class LayoutMode(str, Enum):
    """Framing layout modes for 16:9 to 9:16 portrait reformatting."""
    SINGLE_SPEAKER_SMOOTH = "single_speaker_smooth"
    SPLIT_SCREEN = "split_screen"
    DYNAMIC_PUNCH_IN = "dynamic_punch_in"


class WordTiming(BaseModel):
    """Sub-second word timing and alignment score."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    word: str = Field(..., min_length=1, description="Word token")
    start: float = Field(..., ge=0.0, description="Start timestamp in seconds")
    end: float = Field(..., ge=0.0, description="End timestamp in seconds")
    score: float = Field(default=1.0, ge=0.0, le=1.0, description="Phoneme alignment confidence score")

    @model_validator(mode="after")
    def validate_time_range(self) -> WordTiming:
        if self.start > self.end:
            raise ValueError(f"Start time ({self.start}s) cannot be greater than end time ({self.end}s)")
        return self


class TranscriptSegment(BaseModel):
    """Contiguous sentence or phrase level segment with token-level alignments."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    id: int = Field(..., ge=0, description="Zero-indexed segment sequence id")
    text: str = Field(..., min_length=1, description="Full segment text")
    start: float = Field(..., ge=0.0, description="Segment start timestamp in seconds")
    end: float = Field(..., ge=0.0, description="Segment end timestamp in seconds")
    words: List[WordTiming] = Field(default_factory=list, description="Ordered word token alignments")

    @model_validator(mode="after")
    def validate_segment_boundaries(self) -> TranscriptSegment:
        if self.start > self.end:
            raise ValueError(f"Segment start ({self.start}s) exceeds segment end ({self.end}s)")
        return self


class SceneBoundary(BaseModel):
    """Visual shot boundary detected via content-aware thresholding."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    shot_id: int = Field(..., ge=0)
    start_time: float = Field(..., ge=0.0)
    end_time: float = Field(..., ge=0.0)
    start_frame: int = Field(..., ge=0)
    end_frame: int = Field(..., ge=0)


class ClipProposal(BaseModel):
    """Proposed candidate short-form video clip extracted during Phase 3."""
    model_config = ConfigDict(extra="forbid")

    clip_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8], description="Unique clip identifier")
    start_time: float = Field(..., ge=0.0, description="In seconds, adjusted to natural boundaries")
    end_time: float = Field(..., ge=0.0, description="In seconds, adjusted to natural boundaries")
    virality_score: Optional[int] = Field(default=None, ge=0, le=100, description="Overall virality rating (optional)")
    hook_score: Optional[int] = Field(default=None, ge=0, le=100, description="Opening 3-5s retention score (optional)")
    semantic_score: Optional[int] = Field(default=None, ge=0, le=100, description="Logical completeness & standalone sense (optional)")
    payoff_score: Optional[int] = Field(default=None, ge=0, le=100, description="Conclusion / payoff strength (optional)")
    hook_text: str = Field(default="", description="Opening phrase or caption hook")
    reasoning: str = Field(default="", description="Rationale for candidate selection")
    layout_mode: LayoutMode = Field(default=LayoutMode.SINGLE_SPEAKER_SMOOTH)
    seo_title: str = Field(..., min_length=1, description="Targeted YouTube Shorts / TikTok title")
    seo_description: str = Field(..., min_length=1, description="SEO-optimized description")
    seo_tags: List[str] = Field(default_factory=list, description="Platform tags e.g. #shorts, #viral")
    is_approved: bool = Field(default=True, description="Human-in-the-loop operator approval switch")

    @model_validator(mode="after")
    def validate_clip_duration(self) -> ClipProposal:
        if self.start_time >= self.end_time:
            raise ValueError(f"start_time ({self.start_time}s) must be strictly less than end_time ({self.end_time}s)")
        return self

    @property
    def duration(self) -> float:
        return round(self.end_time - self.start_time, 2)


class CropBox(BaseModel):
    """Bounding crop trajectory coordinate at a discrete timestamp."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    timestamp: float = Field(..., ge=0.0)
    crop_x: int = Field(..., ge=0, description="Horizontal top-left pixel offset")
    crop_y: int = Field(..., ge=0, description="Vertical top-left pixel offset")
    crop_w: int = Field(..., gt=0, description="Crop width in pixels")
    crop_h: int = Field(..., gt=0, description="Crop height in pixels")


class RenderedClip(BaseModel):
    """Artifact bundle produced following Phase 6 hardware encoding."""
    model_config = ConfigDict(frozen=True, extra="forbid")

    clip_id: str
    video_path: str
    thumbnail_path: Optional[str] = None
    metadata_path: str
    duration: float = Field(..., gt=0.0)
    resolution: str = Field(default="1080x1920")


class PipelineConfig(BaseModel):
    """Operator configuration parameters for the execution run."""
    model_config = ConfigDict(extra="forbid")

    max_clips: int = Field(default=5, ge=1, le=20, description="Maximum number of clips to extract")
    min_clip_duration: float = Field(default=25.0, ge=10.0, le=120.0, description="Minimum clip length in seconds")
    max_clip_duration: float = Field(default=60.0, ge=15.0, le=180.0, description="Maximum clip length in seconds")
    enable_greek_guardrail: bool = Field(default=True, description="Run LLM orthographic and tone check")
    enable_virality_scoring: bool = Field(default=True, description="Enable AI virality scoring and hook evaluation")
    virality_threshold: Optional[int] = Field(default=65, ge=0, le=100, description="Minimum score to propose clip if virality scoring is enabled")
    background_music_genre: str = Field(default="none", description="Background music theme for ducking")
    watermark_text: Optional[str] = Field(default=None, description="Optional subtle brand watermark")
    whisper_model: str = Field(default="large-v3-turbo", description="Whisper model architecture")
    domain_preset: str = Field(default="Automotive & Mechanics", description="Predefined domain vocabulary pack")
    hotwords: Optional[str] = Field(
        default="βαλβίδα PCV, αναθυμιάσεις, κινητήρας, λάδια, συνεργείο, κάρτερ, εκκεντροφόρος",
        description="Biasing hotwords for Whisper decoder",
    )
    transcription_prompt: Optional[str] = Field(
        default="Ελληνική συνομιλία, ακριβής ορθογραφία, σημεία στίξης, τόνοι, μηχανικά μέρη αυτοκινήτων, βαλβίδα PCV, βαλβίδα αναθυμιάσεων, κινητήρας, συνεργείο, λάδια, εξαρτήματα.",
        description="Initial domain prompt for Whisper",
    )

    @model_validator(mode="after")
    def validate_duration_range(self) -> PipelineConfig:
        if self.min_clip_duration >= self.max_clip_duration:
            raise ValueError(
                f"min_clip_duration ({self.min_clip_duration}s) must be strictly less than "
                f"max_clip_duration ({self.max_clip_duration}s)"
            )
        return self


class PipelineJob(BaseModel):
    """Top-level pipeline job entity holding state, inputs, and artifacts."""
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    url: str = Field(default="", description="Source video URL or local file path")
    config: PipelineConfig = Field(default_factory=PipelineConfig)
    status: JobStatus = Field(default=JobStatus.IDLE)
    current_phase: PipelinePhase = Field(default=PipelinePhase.INITIALIZATION)
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)
    status_message: str = Field(default="Ready")
    transcript: List[TranscriptSegment] = Field(default_factory=list)
    proposed_clips: List[ClipProposal] = Field(default_factory=list)
    rendered_clips: List[RenderedClip] = Field(default_factory=list)
    logs: List[str] = Field(default_factory=list)
    error_log: Optional[str] = Field(default=None)

    @field_validator("progress_percentage")
    @classmethod
    def clamp_progress(cls, v: float) -> float:
        return round(max(0.0, min(100.0, v)), 2)
