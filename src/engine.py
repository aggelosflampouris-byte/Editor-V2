"""Headless pipeline orchestrator executing video processing phases asynchronously."""

from pathlib import Path
from typing import List, Optional

from src.core.logger import logger
from src.core.contracts import (
    ClipProposal,
    JobStatus,
    PipelineJob,
    PipelinePhase,
    RenderedClip,
    TranscriptSegment,
)
from src.core.events import JobContext
from src.core.workspace import JobWorkspace
from src.modules.ingestion.downloader import VideoIngestionEngine
from src.modules.intelligence.scorer import MultimodalIntelligenceEngine
from src.modules.postprod.renderer import PostProductionEngine
from src.modules.speech.transcriber import SpeechAlignmentEngine
from src.modules.vision.framing import VisionFramingEngine


class PipelineOrchestrator:
    """Coordinates video extraction, AI intelligence, and post-production rendering."""

    def __init__(self, context: JobContext):
        self.context = context
        self.workspace = JobWorkspace(job_id=context.job_id)
        self.ingestion_engine = VideoIngestionEngine(context)
        self.speech_engine = SpeechAlignmentEngine(context)
        self.intelligence_engine = MultimodalIntelligenceEngine(context)
        self.vision_engine = VisionFramingEngine(context)
        self.postprod_engine = PostProductionEngine(context)

    def run_discovery_phase(self, job: PipelineJob) -> List[ClipProposal]:
        """
        Execute Phase 1 through Phase 3:
        1. Ingestion & CFR Normalization
        2. Speech Alignment & Greek Orthographic Guardrail
        3. Multimodal Virality Scoring & Scene Boundary Snapping
        Stops at HITL Gate (JobStatus.READY_FOR_REVIEW).
        """
        try:
            self.workspace.setup()
            logger.info(f"Initialized job workspace for job {job.job_id}: {self.workspace.root}")
            self.context.log_info(f"Initialized job workspace: {self.workspace.root}")

            # Phase 1: Ingestion
            self.context.set_status(JobStatus.DOWNLOADING, "Downloading and normalizing AV streams...")
            self.context.update_progress(PipelinePhase.INGESTION, 10.0, "Extracting video and audio streams...")
            norm_video, norm_audio = self.ingestion_engine.process(
                url=job.url,
                output_video=self.workspace.source_video_path,
                output_audio=self.workspace.normalized_audio_path,
            )

            if self.context.is_cancelled:
                logger.warning(f"Job {job.job_id} cancelled during Ingestion.")
                self.context.fail("Job cancelled by operator during Ingestion.")
                return []

            # Phase 2: Speech Alignment & Greek Guardrail
            self.context.set_status(JobStatus.TRANSCRIBING, "Transcribing and aligning Greek speech...")
            self.context.update_progress(PipelinePhase.TRANSCRIPTION, 35.0, f"Running Whisper alignment ({job.config.whisper_model})...")
            segments: List[TranscriptSegment] = self.speech_engine.transcribe_and_align(
                audio_path=norm_audio,
                enable_guardrail=job.config.enable_greek_guardrail,
                whisper_model=job.config.whisper_model,
                transcription_prompt=job.config.transcription_prompt,
                hotwords=job.config.hotwords,
            )

            if self.context.is_cancelled:
                logger.warning(f"Job {job.job_id} cancelled during Transcription.")
                self.context.fail("Job cancelled by operator during Transcription.")
                return []

            # Phase 3: Multimodal Virality Scoring
            self.context.set_status(JobStatus.SCORING, "Detecting scenes and scoring clip virality...")
            self.context.update_progress(PipelinePhase.VIRALITY_SCORING, 65.0, "Analyzing hook strength & pacing...")
            proposals: List[ClipProposal] = self.intelligence_engine.process(
                video_path=norm_video,
                segments=segments,
                pipeline_config=job.config,
            )

            self.context.set_proposed_clips(proposals)
            self.context.update_progress(
                PipelinePhase.HITL_GATE,
                75.0,
                f"Candidate clips ready for operator review ({len(proposals)} found).",
            )
            return proposals

        except Exception as exc:
            logger.exception(f"Pipeline discovery phase failed for job {job.job_id}")
            self.context.log_error(f"Pipeline discovery phase failed: {exc}")
            self.context.fail(f"Discovery phase failed: {exc}")
            raise

    def run_render_phase(
        self,
        job: PipelineJob,
        approved_clips: List[ClipProposal],
    ) -> List[RenderedClip]:
        """
        Execute Phase 4 through Phase 6 on approved clips:
        4. Vision & 9:16 Framing Trajectory
        5. Kinetic Subtitles & Audio Ducking
        6. Hardware-Accelerated Video Encoding & Final Packaging
        """
        try:
            self.context.set_status(JobStatus.RENDERING, "Rendering approved short-form clips...")
            self.context.update_progress(PipelinePhase.VIDEO_ASSEMBLY, 78.0, "Starting post-production rendering...")

            source_video = self.workspace.source_video_path
            segments = self.context.get_transcript() or job.transcript
            rendered_clips: List[RenderedClip] = []

            total_clips = len(approved_clips)
            for idx, clip in enumerate(approved_clips):
                if self.context.is_cancelled:
                    logger.warning(f"Job {job.job_id} cancelled during Rendering.")
                    self.context.fail("Job cancelled by operator during Rendering.")
                    break

                self.context.log_info(f"Processing clip {idx + 1}/{total_clips} (ID: {clip.clip_id})")

                # Phase 4: Vision Tracking & 9:16 Crop Trajectory
                self.context.update_progress(
                    PipelinePhase.VISION_TRACKING,
                    78.0 + (idx / total_clips) * 10.0,
                    f"Computing 9:16 crop trajectory for clip {idx + 1}...",
                )
                trajectory = self.vision_engine.compute_crop_trajectory(
                    video_path=source_video,
                    start_time=clip.start_time,
                    end_time=clip.end_time,
                    layout_mode=clip.layout_mode,
                )

                # Phase 5 & 6: Subtitles, Assembly, and Render
                self.context.update_progress(
                    PipelinePhase.VIDEO_ASSEMBLY,
                    88.0 + (idx / total_clips) * 10.0,
                    f"Encoding hardware-accelerated video for clip {idx + 1}...",
                )
                output_video = self.workspace.get_clip_render_path(clip.clip_id)
                output_metadata = self.workspace.get_clip_metadata_path(clip.clip_id)

                rendered = self.postprod_engine.render_clip(
                    source_video=source_video,
                    clip=clip,
                    trajectory=trajectory,
                    segments=segments,
                    output_video=output_video,
                    output_metadata=output_metadata,
                    clip_index=idx + 1,
                    total_clips=total_clips,
                )
                rendered_clips.append(rendered)
                self.context.add_rendered_clip(rendered)

            self.workspace.cleanup_scratch()
            self.context.complete(f"Successfully rendered {len(rendered_clips)} short clips.")
            return rendered_clips

        except Exception as exc:
            logger.exception(f"Pipeline render phase failed for job {job.job_id}")
            self.context.log_error(f"Pipeline render phase failed: {exc}")
            self.context.fail(f"Render phase failed: {exc}")
            raise
