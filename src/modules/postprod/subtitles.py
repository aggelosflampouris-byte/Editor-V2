"""Kinetic Greek subtitle generation using Advanced SubStation Alpha (.ass) format."""

from pathlib import Path
from typing import List

from src.core.logger import logger
from src.core.contracts import TranscriptSegment, WordTiming


class SubtitleGenerator:
    """Generates high-contrast kinetic ASS subtitles with word-level karaoke highlights."""

    ASS_HEADER_TEMPLATE = """[Script Info]
Title: Antigravity Greek Kinetic Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601
PlayResX: 1080
PlayResY: 1920

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Montserrat,68,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,6,2,2,40,40,480,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""

    def generate_ass(
        self,
        segments: List[TranscriptSegment],
        clip_start: float,
        clip_end: float,
        output_path: Path,
        words_per_window: int = 3,
    ) -> Path:
        """
        Produce a .ass file with maximum 3 words on screen at any time and
        karaoke-style active word highlighting (yellow active color: &H0000FFFF& in BGR ASS).
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Collect words falling inside clip range
        words_in_clip: List[WordTiming] = []
        for seg in segments:
            for w in seg.words:
                if w.end >= clip_start and w.start <= clip_end:
                    words_in_clip.append(w)

        events: List[str] = []

        # Group words into windows of max 3 words
        for i in range(0, len(words_in_clip), words_per_window):
            window = words_in_clip[i : i + words_per_window]
            if not window:
                continue

            window_start = max(0.0, window[0].start - clip_start)
            window_end = max(window_start + 0.1, window[-1].end - clip_start)

            # For each word in this window, create a sub-event where it is highlighted
            for active_idx, target_word in enumerate(window):
                if active_idx == 0:
                    word_sub_start = max(0.0, target_word.start - clip_start)
                else:
                    word_sub_start = max(0.0, window[active_idx - 1].end - clip_start)

                if active_idx == len(window) - 1:
                    word_sub_end = max(word_sub_start + 0.1, window[-1].end - clip_start)
                else:
                    word_sub_end = max(word_sub_start + 0.1, target_word.end - clip_start)

                start_formatted = self._format_ass_time(word_sub_start)
                end_formatted = self._format_ass_time(word_sub_end)

                # Build line where active word has yellow highlight {\c&H0000FFFF&}
                # and inactive words remain white {\c&H00FFFFFF&}
                line_parts: List[str] = []
                for idx, w in enumerate(window):
                    if idx == active_idx:
                        line_parts.append(r"{\c&H0022FFFF\b1}" + w.word + r"{\r}")
                    else:
                        line_parts.append(r"{\c&H00FFFFFF\b1}" + w.word + r"{\r}")

                line_text = " ".join(line_parts)
                events.append(
                    f"Dialogue: 0,{start_formatted},{end_formatted},Default,,0,0,0,,{line_text}"
                )

        ass_content = self.ASS_HEADER_TEMPLATE + "\n".join(events) + "\n"
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(ass_content)

        return output_path

    @staticmethod
    def _format_ass_time(seconds: float) -> str:
        """Format seconds to ASS timestamp format: H:MM:SS.cs"""
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        centiseconds = int(round((seconds - int(seconds)) * 100))
        if centiseconds >= 100:
            centiseconds = 99
        return f"{hours}:{minutes:02d}:{secs:02d}.{centiseconds:02d}"
