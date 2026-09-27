"""Temporary workspace and scratch space lifecycle management."""

import shutil
from pathlib import Path
from typing import Optional
from src.config import config


class JobWorkspace:
    """Manages the disk lifecycle of a single pipeline job execution."""

    def __init__(self, job_id: str, base_dir: Optional[Path] = None):
        self.job_id = job_id
        self.root = (base_dir or config.workspace_dir) / job_id
        self.audio_dir = self.root / "audio"
        self.video_dir = self.root / "video"
        self.subtitles_dir = self.root / "subtitles"
        self.scratch_dir = self.root / "scratch"
        self.output_dir = (base_dir or config.output_dir) / job_id

    def setup(self) -> None:
        """Initialize all required directories for this job."""
        for directory in [
            self.root,
            self.audio_dir,
            self.video_dir,
            self.subtitles_dir,
            self.scratch_dir,
            self.output_dir,
        ]:
            directory.mkdir(parents=True, exist_ok=True)

    @property
    def source_video_path(self) -> Path:
        return self.video_dir / "source_video.mp4"

    @property
    def normalized_audio_path(self) -> Path:
        return self.audio_dir / "source_audio.wav"

    def get_clip_render_path(self, clip_id: str) -> Path:
        return self.output_dir / f"clip_{clip_id}.mp4"

    def get_clip_metadata_path(self, clip_id: str) -> Path:
        return self.output_dir / f"clip_{clip_id}_metadata.json"

    def cleanup_scratch(self) -> None:
        """Clean intermediate scratch files while preserving final outputs."""
        if self.scratch_dir.exists():
            shutil.rmtree(self.scratch_dir, ignore_errors=True)

    def teardown(self) -> None:
        """Purge entire working directory."""
        if self.root.exists():
            shutil.rmtree(self.root, ignore_errors=True)
