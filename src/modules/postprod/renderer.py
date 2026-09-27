"""Hardware-accelerated video assembly, FFmpeg filtergraph construction, and export packaging."""

import json
from pathlib import Path
import re
import subprocess
from typing import List, Optional

from src.config import config
from src.core.logger import logger
from src.core.contracts import ClipProposal, CropBox, PipelinePhase, RenderedClip, TranscriptSegment
from src.core.events import JobContext
from src.core.exceptions import RenderingError
from src.modules.postprod.subtitles import SubtitleGenerator


class PostProductionEngine:
    """Builds consolidated FFmpeg filtergraphs and executes GPU hardware-accelerated renders."""

    def __init__(self, context: JobContext):
        self.context = context
        self.subtitle_generator = SubtitleGenerator()

    def render_clip(
        self,
        source_video: Path,
        clip: ClipProposal,
        trajectory: List[CropBox],
        segments: List[TranscriptSegment],
        output_video: Path,
        output_metadata: Path,
        clip_index: int = 1,
        total_clips: int = 1,
    ) -> RenderedClip:
        """Render a single 9:16 clip with subtitles, framing, and metadata package."""
        self.context.log_info(
            f"Rendering clip {clip_index}/{total_clips} (ID: {clip.clip_id}) | "
            f"Duration: {clip.duration:.1f}s | Encoder: {config.video_codec} ({config.gpu_device_name})"
        )
        output_video.parent.mkdir(parents=True, exist_ok=True)

        # Step 1: Generate ASS subtitle file
        ass_path = output_video.parent / f"subtitles_{clip.clip_id}.ass"
        self.subtitle_generator.generate_ass(
            segments=segments,
            clip_start=clip.start_time,
            clip_end=clip.end_time,
            output_path=ass_path,
        )
        self.context.log_info(f"Generated kinetic Greek ASS subtitles for clip {clip.clip_id}")

        # Step 2: Compute representative crop box (or trajectory center)
        if trajectory:
            avg_x = int(sum(c.crop_x for c in trajectory) / len(trajectory))
            crop_w = trajectory[0].crop_w
            crop_h = trajectory[0].crop_h
        else:
            crop_w = 608
            crop_h = 1080
            avg_x = (1920 - crop_w) // 2

        # Step 3: Build FFmpeg filtergraph
        # Windows-safe path escaping for FFmpeg subtitles filter
        escaped_ass_path = str(ass_path).replace("\\", "/").replace(":", "\\:")
        video_filters = [
            f"crop={crop_w}:{crop_h}:{avg_x}:0",
            f"scale={config.target_width}:{config.target_height}",
            f"subtitles='{escaped_ass_path}'",
        ]
        vf_string = ",".join(video_filters)

        # Hardware-accelerated encoder with software fallback
        codec = config.video_codec if config.use_hardware_accel else config.fallback_video_codec
        success = self._execute_ffmpeg_render_with_progress(
            source_video=source_video,
            clip=clip,
            vf_string=vf_string,
            codec=codec,
            output_video=output_video,
            clip_index=clip_index,
            total_clips=total_clips,
        )

        if not success and codec != config.fallback_video_codec:
            self.context.log_warning(f"Hardware codec '{codec}' failed. Retrying with '{config.fallback_video_codec}'...")
            success = self._execute_ffmpeg_render_with_progress(
                source_video=source_video,
                clip=clip,
                vf_string=vf_string,
                codec=config.fallback_video_codec,
                output_video=output_video,
                clip_index=clip_index,
                total_clips=total_clips,
            )

        if not success:
            logger.error(f"Rendering failed for clip {clip.clip_id}")
            raise RenderingError(f"Rendering failed for clip {clip.clip_id}")

        # Step 4: Write SEO and metadata bundle
        metadata = {
            "clip_id": clip.clip_id,
            "title": clip.seo_title,
            "description": clip.seo_description,
            "tags": clip.seo_tags,
            "hook_text": clip.hook_text,
            "virality_score": clip.virality_score,
            "duration": clip.duration,
            "layout_mode": clip.layout_mode.value,
            "resolution": f"{config.target_width}x{config.target_height}",
            "hardware_encoder": codec,
        }
        with open(output_metadata, "w", encoding="utf-8") as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)

        rendered_clip = RenderedClip(
            clip_id=clip.clip_id,
            video_path=str(output_video),
            thumbnail_path=None,
            metadata_path=str(output_metadata),
            duration=clip.duration,
            resolution=f"{config.target_width}x{config.target_height}",
        )
        self.context.add_rendered_clip(rendered_clip)
        self.context.log_info(f"Clip {clip.clip_id} successfully encoded ({output_video.stat().st_size / (1024*1024):.2f} MB)")
        return rendered_clip

    def _execute_ffmpeg_render_with_progress(
        self,
        source_video: Path,
        clip: ClipProposal,
        vf_string: str,
        codec: str,
        output_video: Path,
        clip_index: int,
        total_clips: int,
    ) -> bool:
        """Run FFmpeg hardware render subprocess with real-time progression logging."""
        hw_args = list(config.hwaccel_flags) if codec != config.fallback_video_codec else []

        cmd = [
            config.ffmpeg_binary,
            "-y",
            *hw_args,
            "-ss", f"{clip.start_time:.3f}",
            "-to", f"{clip.end_time:.3f}",
            "-i", str(source_video),
            "-vf", vf_string,
            "-c:v", codec,
            "-c:a", "aac",
            "-b:a", "192k",
            "-progress", "pipe:1",
            str(output_video),
        ]

        render_err_log = output_video.with_suffix(".render_err.log")
        try:
            with open(render_err_log, "w", encoding="utf-8", errors="replace") as render_err_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=render_err_file,  # Redirect to disk file to prevent OS pipe deadlocks
                    text=True,
                    bufsize=1,
                )

                time_pattern = re.compile(r"out_time_us=(\d+)")
                speed_pattern = re.compile(r"speed=\s*([\d\.]+)x")

                current_speed = "1.0x"
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    line = line.strip()
                    if line.startswith("speed="):
                        m_speed = speed_pattern.search(line)
                        if m_speed:
                            current_speed = f"{m_speed.group(1)}x"
                    elif line.startswith("out_time_us="):
                        m_time = time_pattern.search(line)
                        if m_time:
                            micros = int(m_time.group(1))
                            seconds = micros / 1_000_000.0
                            clip_progress = min(1.0, seconds / max(0.1, clip.duration))

                            # Overall render progression between 78% and 98%
                            base_pct = 78.0 + ((clip_index - 1) / max(1, total_clips)) * 20.0
                            clip_pct_contribution = (clip_progress / max(1, total_clips)) * 20.0
                            total_pct = min(98.0, base_pct + clip_pct_contribution)

                            self.context.update_progress(
                                PipelinePhase.VIDEO_ASSEMBLY,
                                round(total_pct, 1),
                                f"Encoding Clip {clip_index}/{total_clips} [{codec}]: {seconds:.1f}s / {clip.duration:.1f}s ({clip_progress*100:.0f}%, speed: {current_speed})",
                            )

                process.wait(timeout=config.rendering_timeout_seconds)

            if process.returncode == 0:
                render_err_log.unlink(missing_ok=True)
                return True
            else:
                err_msg = render_err_log.read_text(encoding="utf-8", errors="replace")[-500:] if render_err_log.exists() else ""
                logger.error(f"Render failed with codec {codec}: {err_msg}")
                self.context.log_warning(f"Render failed with codec {codec}: {err_msg}")
                return False

        except Exception as exc:
            logger.exception(f"FFmpeg render error: {exc}")
            self.context.log_error(f"FFmpeg render error: {exc}")
            return False
