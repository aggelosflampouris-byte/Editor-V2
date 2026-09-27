"""Module for video ingestion, audio extraction, GPU hardware normalization, and live progression."""

import os
from pathlib import Path
import re
import shutil
import subprocess
import subprocess
from typing import Tuple

from src.core.logger import logger
from src.config import config
from src.core.contracts import PipelinePhase
from src.core.events import JobContext
from src.core.exceptions import IngestionError


class VideoIngestionEngine:
    """Handles downloading via yt-dlp and normalizing AV streams with GPU hardware acceleration."""

    def __init__(self, context: JobContext):
        self.context = context

    def process(self, url: str, output_video: Path, output_audio: Path) -> Tuple[Path, Path]:
        """
        Download or extract media, normalize framerate to CFR 30fps with GPU acceleration,
        and normalize audio loudness using EBU R128 with detailed live progress.
        """
        self.context.log_info(f"Targeting media URL / path: {url}")
        self.context.log_info(f"GPU Hardware Mode: {config.gpu_device_name} (Encoder: {config.video_codec})")
        output_video.parent.mkdir(parents=True, exist_ok=True)
        output_audio.parent.mkdir(parents=True, exist_ok=True)

        # Check if URL is local file or web link
        is_local_file = False
        try:
            is_local_file = Path(url).is_file()
        except (OSError, ValueError):
            pass

        if is_local_file:
            source_file = Path(url)
            self.context.log_info(f"Using local video source: {source_file.name} ({source_file.stat().st_size / (1024*1024):.1f} MB)")
            raw_video = source_file
        else:
            raw_video = output_video.parent / "raw_download.mp4"
            if not raw_video.exists():
                self._download_stream_with_progress(url, raw_video)

                # If yt-dlp left separate streams, merge them
                if not raw_video.exists():
                    video_candidates = [f for f in raw_video.parent.glob("*.mp4") if f != output_video]
                    audio_candidates = list(raw_video.parent.glob("*.m4a")) + list(raw_video.parent.glob("*.webm"))
                    if video_candidates and audio_candidates:
                        self.context.log_info("Merging split streams into single container with FFmpeg...")
                        self._merge_separate_streams(video_candidates[0], audio_candidates[0], raw_video)
                    elif video_candidates:
                        raw_video = video_candidates[0]
            else:
                self.context.log_info(f"Existing raw download found: {raw_video.name} ({raw_video.stat().st_size / (1024*1024):.1f} MB)")

        if not raw_video.exists():
            raise IngestionError(f"Raw downloaded video not found at: {raw_video}")

        self._normalize_video_cfr(raw_video, output_video)
        self._extract_normalized_audio(output_video, output_audio)

        self.context.log_info(f"Ingestion complete: Video ({output_video.stat().st_size / (1024*1024):.1f} MB), Audio ({output_audio.stat().st_size / (1024*1024):.1f} MB)")
        return output_video, output_audio

    def _download_stream_with_progress(self, url: str, target_path: Path) -> None:
        """Download stream using yt-dlp with real-time percentage and speed feedback."""
        self.context.log_info("Initializing yt-dlp stream extraction engine...")
        self.context.update_progress(PipelinePhase.INGESTION, 5.0, "Querying video metadata from URL...")

        try:
            import yt_dlp

            last_logged_pct = [-1]

            def ytdl_hook(d):
                if d["status"] == "downloading":
                    total = d.get("total_bytes") or d.get("total_bytes_estimate") or 0
                    downloaded = d.get("downloaded_bytes") or 0
                    if total > 0:
                        pct = (downloaded / total) * 100.0
                        scaled_pct = 5.0 + (pct * 0.15)  # 5% to 20%
                        speed = d.get("_speed_str", "").strip()
                        eta = d.get("_eta_str", "").strip()
                        int_pct = int(pct)
                        if int_pct % 10 == 0 and int_pct != last_logged_pct[0]:
                            last_logged_pct[0] = int_pct
                            self.context.log_info(f"Downloading stream: {pct:.1f}% complete ({speed}, ETA {eta})")

                        self.context.update_progress(
                            PipelinePhase.INGESTION,
                            round(scaled_pct, 1),
                            f"Downloading stream: {pct:.1f}% ({speed}, ETA {eta})",
                        )
                elif d["status"] == "finished":
                    self.context.log_info("Stream download finished. Processing container...")

            ydl_opts = {
                "format": "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]",
                "merge_output_format": "mp4",
                "outtmpl": str(target_path),
                "noplaylist": True,
                "ffmpeg_location": config.ffmpeg_binary,
                "progress_hooks": [ytdl_hook],
                "quiet": True,
                "no_warnings": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                ydl.download([url])

        except Exception as exc:
            logger.warning(f"Python yt-dlp API fallback to subprocess: {exc}")
            self.context.log_warning(f"Python yt-dlp API fallback to subprocess: {exc}")
            self._download_stream_subprocess(url, target_path)

    def _download_stream_subprocess(self, url: str, target_path: Path) -> None:
        """Fallback subprocess downloader."""
        cmd = [
            *config.get_ytdlp_base_cmd(),
            "--ffmpeg-location", config.ffmpeg_binary,
            "-f", "bestvideo[height<=1080][ext=mp4]+bestaudio[ext=m4a]/best[height<=1080]",
            "--merge-output-format", "mp4",
            "-o", str(target_path),
            "--no-playlist",
            url,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=config.ingestion_timeout_seconds, check=False)
        if result.returncode != 0:
            err_msg = result.stderr[-400:] if result.stderr else 'Unknown error'
            logger.error(f"yt-dlp subprocess failed: {err_msg}")
            raise IngestionError(f"yt-dlp failed: {err_msg}")

    def _merge_separate_streams(self, video_file: Path, audio_file: Path, output_file: Path) -> None:
        """Merge split audio and video tracks into unified MP4."""
        cmd = [
            config.ffmpeg_binary,
            "-y",
            "-i", str(video_file),
            "-i", str(audio_file),
            "-c:v", "copy",
            "-c:a", "aac",
            str(output_file),
        ]
        subprocess.run(cmd, capture_output=True, text=True, timeout=config.ingestion_timeout_seconds, check=False)

    def _normalize_video_cfr(self, input_path: Path, output_path: Path) -> None:
        """
        Enforce CFR 30fps.
        If already 30fps constant rate, avoids costly re-encoding.
        If re-encoding is required, uses deadlock-proof file-buffered stderr.
        """
        source_fps = 30.0
        total_duration_sec = 60.0
        try:
            import cv2
            cap = cv2.VideoCapture(str(input_path))
            if cap.isOpened():
                source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
                if source_fps > 0 and frame_count > 0:
                    total_duration_sec = frame_count / source_fps
                cap.release()
        except Exception:
            pass

        self.context.log_info(
            f"Source video metrics: {total_duration_sec:.1f}s ({total_duration_sec/60:.1f} min), {source_fps:.2f} fps"
        )

        # Fast path: If already ~30fps CFR (+-0.2fps), fast-copy to output_path
        if abs(source_fps - 30.0) < 0.2:
            self.context.log_info(f"Source video is already 30fps constant frame rate ({source_fps:.2f} fps). Fast-linking container...")
            self.context.update_progress(PipelinePhase.INGESTION, 28.0, "Framerate validated (30fps CFR). Fast-linking...")
            if input_path.resolve() != output_path.resolve():
                shutil.copyfile(input_path, output_path)
            self.context.log_info("CFR validation completed.")
            return

        self.context.log_info(f"Normalizing variable framerate to CFR 30fps using {config.gpu_device_name}...")
        self.context.update_progress(PipelinePhase.INGESTION, 22.0, "Normalizing video framerate (CFR 30fps)...")

        hw_args = list(config.hwaccel_flags)
        encoder_args = ["-c:v", config.video_codec] if config.use_hardware_accel else ["-c:v", config.fallback_video_codec, "-preset", "fast", "-crf", "18"]

        cmd = [
            config.ffmpeg_binary,
            "-y",
            *hw_args,
            "-i", str(input_path),
            "-filter:v", "fps=fps=30",
            "-vsync", "cfr",
            *encoder_args,
            "-c:a", "copy",
            "-progress", "pipe:1",
            str(output_path),
        ]

        err_log_path = output_path.with_suffix(".cfr_err.log")
        try:
            with open(err_log_path, "w", encoding="utf-8", errors="replace") as err_file:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=err_file,  # Redirect to disk file to completely prevent OS pipe deadlocks
                    text=True,
                    bufsize=1,
                )

                time_pattern = re.compile(r"out_time_us=(\d+)")
                speed_pattern = re.compile(r"speed=\s*([\d\.]+)x")

                current_speed = "1.0x"
                last_reported_sec = 0.0

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
                            pct = min(1.0, seconds / max(1.0, total_duration_sec))
                            scaled = 22.0 + (pct * 8.0)

                            if (seconds - last_reported_sec) >= 10.0 or pct >= 1.0:
                                last_reported_sec = seconds
                                self.context.update_progress(
                                    PipelinePhase.INGESTION,
                                    round(scaled, 1),
                                    f"Normalizing CFR 30fps: {seconds:.1f}s / {total_duration_sec:.1f}s ({pct*100:.0f}%, speed: {current_speed})",
                                )
                                self.context.log_info(f"CFR Normalization: {seconds:.1f}s / {total_duration_sec:.1f}s processed ({current_speed})")

                process.wait(timeout=config.ingestion_timeout_seconds)

            if process.returncode != 0:
                err_content = err_log_path.read_text(encoding="utf-8", errors="replace")[-600:] if err_log_path.exists() else "Unknown error"
                logger.error(f"CFR Normalization failed. FFmpeg output: {err_content}")
                raise IngestionError("FFmpeg CFR normalization failed", details=err_content)

            err_log_path.unlink(missing_ok=True)

        except subprocess.TimeoutExpired as exc:
            logger.error(f"FFmpeg CFR normalization timed out: {exc}")
            raise IngestionError("FFmpeg CFR normalization timed out", details=str(exc)) from exc

    def _extract_normalized_audio(self, video_path: Path, audio_path: Path) -> None:
        """Extract audio stream normalized to 16kHz mono PCM s16le and EBU R128 loudness."""
        self.context.log_info("Extracting audio with EBU R128 loudness normalization (-16 LUFS)...")
        self.context.update_progress(PipelinePhase.INGESTION, 31.0, "Extracting audio and normalizing loudness...")

        cmd = [
            config.ffmpeg_binary,
            "-y",
            "-i", str(video_path),
            "-vn",
            "-ar", "16000",
            "-ac", "1",
            "-c:a", "pcm_s16le",
            "-filter:a", "loudnorm=I=-16:TP=-1.5:LRA=11",
            str(audio_path),
        ]

        err_log = audio_path.with_suffix(".audio_err.log")
        with open(err_log, "w", encoding="utf-8", errors="replace") as err_f:
            res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=err_f, timeout=config.ingestion_timeout_seconds, check=False)

        if res.returncode != 0:
            err_details = err_log.read_text(encoding="utf-8", errors="replace")[-500:] if err_log.exists() else "Unknown error"
            logger.error(f"Audio extraction failed. FFmpeg output: {err_details}")
            raise IngestionError("FFmpeg audio extraction failed", details=err_details)

        err_log.unlink(missing_ok=True)
        self.context.update_progress(PipelinePhase.INGESTION, 34.0, "Audio extraction and loudness normalization complete.")
