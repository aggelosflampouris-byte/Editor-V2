"""Typed application settings, GPU hardware detection, and environment configuration."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import List, Optional, Tuple
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from src.core.logger import logger

# Automatically load environment variables from .env if present
_env_path = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(_env_path)
load_dotenv()


def _resolve_ffmpeg() -> str:
    """Resolve FFmpeg binary location from PATH or imageio_ffmpeg."""
    path_ffmpeg = shutil.which("ffmpeg")
    if path_ffmpeg:
        return path_ffmpeg
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        pass
    return "ffmpeg"


def _resolve_ytdlp() -> str:
    """Resolve yt-dlp binary or Python runner."""
    path_ytdlp = shutil.which("yt-dlp")
    if path_ytdlp:
        return path_ytdlp
    return sys.executable


def detect_hardware_acceleration(ffmpeg_path: str) -> Tuple[str, str, List[str]]:
    """
    Detect available GPU hardware encoder and hardware decoding flags.
    Returns: (gpu_name, video_encoder, hwaccel_args)
    """
    try:
        # Check AMD AMF hardware encoder
        test_amf = subprocess.run(
            [ffmpeg_path, "-y", "-hwaccel", "d3d11va", "-f", "lavfi", "-i", "testsrc=duration=1:size=640x360:rate=30", "-c:v", "h264_amf", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if test_amf.returncode == 0:
            return "AMD Radeon AMF (GPU Accelerated)", "h264_amf", ["-hwaccel", "d3d11va"]

        # Check NVIDIA NVENC hardware encoder
        test_nvenc = subprocess.run(
            [ffmpeg_path, "-y", "-hwaccel", "cuda", "-f", "lavfi", "-i", "testsrc=duration=1:size=640x360:rate=30", "-c:v", "h264_nvenc", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if test_nvenc.returncode == 0:
            return "NVIDIA NVENC (GPU Accelerated)", "h264_nvenc", ["-hwaccel", "cuda"]

        # Check Intel QSV hardware encoder
        test_qsv = subprocess.run(
            [ffmpeg_path, "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=640x360:rate=30", "-c:v", "h264_qsv", "-f", "null", "-"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if test_qsv.returncode == 0:
            return "Intel QSV (GPU Accelerated)", "h264_qsv", ["-hwaccel", "qsv"]

    except Exception:
        pass

    return "CPU Software Mode", "libx264", []


class AppConfig(BaseModel):
    """Global configuration settings for Antigravity Video Engine."""

    # Project directories
    base_dir: Path = Field(default_factory=lambda: Path(os.getcwd()))
    workspace_dir: Path = Field(default_factory=lambda: Path(os.getcwd()) / "workspace")
    output_dir: Path = Field(default_factory=lambda: Path(os.getcwd()) / "outputs")
    cache_dir: Path = Field(default_factory=lambda: Path(os.getcwd()) / ".cache")

    # API Keys & LLM Models
    gemini_api_key: Optional[str] = Field(
        default_factory=lambda: os.getenv("GEMINI_API_KEY", "")
    )
    gemini_model: str = Field(default="gemini-1.5-pro")

    # Speech Alignment Models
    whisper_model: str = Field(default="large-v3")
    greek_alignment_model: str = Field(
        default="lighteternal/wav2vec2-large-xlsr-53-greek"
    )

    # Subprocess execution & timeouts (seconds)
    ffmpeg_binary: str = Field(default_factory=_resolve_ffmpeg)
    ytdlp_binary: str = Field(default_factory=_resolve_ytdlp)
    ingestion_timeout_seconds: int = Field(default=900)
    transcription_timeout_seconds: int = Field(default=1200)
    rendering_timeout_seconds: int = Field(default=900)

    # Hardware acceleration & rendering constraints
    target_width: int = Field(default=1080)
    target_height: int = Field(default=1920)
    target_fps: int = Field(default=30)
    use_hardware_accel: bool = Field(default=True)
    gpu_device_name: str = Field(default="Detecting...")
    video_codec: str = Field(default="h264_amf")
    hwaccel_flags: List[str] = Field(default_factory=list)
    fallback_video_codec: str = Field(default="libx264")

    def init_hardware(self) -> None:
        """Detect and set GPU capabilities."""
        name, codec, hw_flags = detect_hardware_acceleration(self.ffmpeg_binary)
        self.gpu_device_name = name
        self.video_codec = codec
        self.hwaccel_flags = hw_flags
        logger.info(f"Hardware initialization complete: GPU={name}, Codec={codec}, Flags={hw_flags}")

    def set_gemini_api_key(self, api_key: str) -> None:
        """Update runtime Gemini API key, set environment variable, and persist to .env."""
        clean_key = api_key.strip()
        self.gemini_api_key = clean_key
        os.environ["GEMINI_API_KEY"] = clean_key

        # Persist to local .env file
        try:
            env_file = Path(__file__).resolve().parent.parent / ".env"
            lines = []
            if env_file.exists():
                with open(env_file, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.startswith("GEMINI_API_KEY="):
                            lines.append(line)
            
            # Ensure the last line before appending has a newline
            if lines and not lines[-1].endswith("\n"):
                lines[-1] = lines[-1] + "\n"
                
            lines.append(f"GEMINI_API_KEY={clean_key}\n")
            with open(env_file, "w", encoding="utf-8") as f:
                f.writelines(lines)
            logger.info("Successfully persisted updated GEMINI_API_KEY to .env file.")
        except Exception as e:
            logger.error(f"Failed to persist GEMINI_API_KEY to .env: {e}", exc_info=True)

    def get_ytdlp_base_cmd(self) -> List[str]:
        """Return executable command prefix for invoking yt-dlp."""
        if shutil.which("yt-dlp"):
            return ["yt-dlp"]
        return [sys.executable, "-m", "yt_dlp"]

    def ensure_directories(self) -> None:
        """Create necessary project directories if they do not exist."""
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# Default singleton instance initialized with hardware detection
config = AppConfig()
config.init_hardware()
