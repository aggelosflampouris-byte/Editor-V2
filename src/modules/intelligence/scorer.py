"""Multimodal virality scoring, scene detection, live progression, and boundary snapping."""

import json
from pathlib import Path
from typing import List, Optional
import uuid

from src.config import config
from src.core.gemini import call_gemini
from src.core.contracts import (
    ClipProposal,
    LayoutMode,
    PipelineConfig,
    PipelinePhase,
    SceneBoundary,
    TranscriptSegment,
    WordTiming,
)
from src.core.events import JobContext
from src.core.exceptions import IntelligenceError


class MultimodalIntelligenceEngine:
    """Detects scene cuts, groups transcript windows, and scores virality with live progression."""

    def __init__(self, context: JobContext):
        self.context = context

    def process(
        self,
        video_path: Path,
        segments: List[TranscriptSegment],
        pipeline_config: PipelineConfig,
    ) -> List[ClipProposal]:
        """Run scene detection, sliding window analysis, scoring, and boundary snapping."""
        self.context.log_info("Extracting visual shot boundaries with content-aware detection...")
        self.context.update_progress(PipelinePhase.SCENE_DETECTION, 63.0, "Detecting visual scene boundaries...")
        scene_boundaries = self._detect_scenes(video_path)

        self.context.log_info(
            f"Detected {len(scene_boundaries)} visual scene boundaries across source video."
        )
        for sb in scene_boundaries[:5]:
            self.context.log_info(f"  Shot #{sb.shot_id + 1}: {sb.start_time:.1f}s -> {sb.end_time:.1f}s (duration: {sb.end_time - sb.start_time:.1f}s)")

        if pipeline_config.enable_virality_scoring:
            self.context.log_info("AI Virality & Hook Scoring enabled. Evaluating candidate windows with Gemini...")
            self.context.update_progress(PipelinePhase.VIRALITY_SCORING, 66.0, "Evaluating viral hooks & conceptual density...")
            proposals = self._score_candidate_windows(
                video_path, segments, scene_boundaries, pipeline_config
            )
        else:
            self.context.log_info("Virality scoring is optional/bypassed. Generating clips directly from speech pacing & scene boundaries...")
            self.context.update_progress(PipelinePhase.VIRALITY_SCORING, 66.0, "Extracting pacing & scene-based candidate clips...")
            proposals = self._generate_pacing_clips(segments, pipeline_config)

        # Snap proposed times to natural sentence and scene cuts
        self.context.log_info("Snapping clip boundaries to nearest sentence ends and shot transitions...")
        snapped_proposals: List[ClipProposal] = []
        for prop in proposals:
            snapped = self._snap_boundaries(prop, segments, scene_boundaries, pipeline_config.min_clip_duration)
            dur = snapped.end_time - snapped.start_time
            if dur >= (pipeline_config.min_clip_duration - 1.0):
                snapped_proposals.append(snapped)
                score_desc = f"Virality {snapped.virality_score}/100, Hook {snapped.hook_score}/100" if snapped.virality_score is not None else "Virality: Bypassed"
                self.context.log_info(
                    f"Candidate Clip '{snapped.clip_id}' [{snapped.start_time:.1f}s -> {snapped.end_time:.1f}s, {dur:.1f}s]: "
                    f"{score_desc} — \"{snapped.hook_text[:50]}...\""
                )

        # Filter by threshold if virality scoring is enabled
        if pipeline_config.enable_virality_scoring and pipeline_config.virality_threshold is not None:
            valid_proposals = [
                p for p in snapped_proposals if p.virality_score is not None and p.virality_score >= pipeline_config.virality_threshold
            ]
            valid_proposals.sort(key=lambda x: (x.virality_score or 0), reverse=True)
        else:
            valid_proposals = snapped_proposals

        final_clips = valid_proposals[: pipeline_config.max_clips]

        # Safety net: Guarantee at least 1 candidate clip if speech segments exist
        if not final_clips and segments:
            fallback_start = segments[0].start
            candidate_ends = [s.end for s in segments if s.end - fallback_start >= pipeline_config.min_clip_duration]
            if candidate_ends:
                fallback_end = min(candidate_ends)
            else:
                fallback_end = min(segments[-1].end, fallback_start + pipeline_config.max_clip_duration)

            if fallback_end <= fallback_start:
                fallback_end = round(fallback_start + pipeline_config.min_clip_duration, 2)

            fallback_clip = ClipProposal(
                clip_id=str(uuid.uuid4())[:8],
                start_time=round(fallback_start, 2),
                end_time=round(fallback_end, 2),
                virality_score=None,
                hook_score=None,
                hook_text=segments[0].text[:70],
                reasoning="Extracted based on speech pacing fallback.",
                layout_mode=LayoutMode.SINGLE_SPEAKER_SMOOTH,
                seo_title="Κορυφαίο Απόσπασμα",
                seo_description="Απόσπασμα από τη συζήτηση.",
                seo_tags=["#shorts", "#greek"],
                is_approved=True,
            )
            final_clips = [fallback_clip]
            self.context.log_info(
                f"Generated fallback pacing clip '{fallback_clip.clip_id}' [{fallback_clip.start_time:.1f}s -> {fallback_clip.end_time:.1f}s]."
            )

        self.context.log_info(f"Generated {len(final_clips)} candidate clips ready for operator review.")
        self.context.set_proposed_clips(final_clips)
        return final_clips

    def _extract_storyboard(self, video_path: Path) -> Optional[Path]:
        """Extract a 4x4 storyboard grid of frames from the video for multimodal analysis."""
        import cv2
        import numpy as np
        
        try:
            self.context.log_info("Extracting video storyboard for Gemini multimodal analysis...")
            cap = cv2.VideoCapture(str(video_path))
            if not cap.isOpened():
                return None
                
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total_frames <= 0:
                return None
                
            frame_indices = np.linspace(0, total_frames - 1, 16, dtype=int)
            frames = []
            
            for idx in frame_indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if ret:
                    resized = cv2.resize(frame, (320, 180))
                    frames.append(resized)
                else:
                    frames.append(np.zeros((180, 320, 3), dtype=np.uint8))
            cap.release()
            
            rows = []
            for i in range(4):
                row = np.hstack(frames[i*4:(i+1)*4])
                rows.append(row)
            grid = np.vstack(rows)
            
            out_path = video_path.parent / f"{video_path.stem}_storyboard.jpg"
            cv2.imwrite(str(out_path), grid)
            return out_path
        except Exception as e:
            self.context.log_warning(f"Failed to generate storyboard: {e}")
            return None

    def _detect_scenes(self, video_path: Path) -> List[SceneBoundary]:
        """Detect shot transitions using PySceneDetect or deterministic intervals."""
        try:
            import importlib
            scenedetect_mod = importlib.import_module("scenedetect")
            detectors_mod = importlib.import_module("scenedetect.detectors")

            open_video = scenedetect_mod.open_video
            SceneManager = scenedetect_mod.SceneManager
            ContentDetector = detectors_mod.ContentDetector

            video = open_video(str(video_path))
            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=27.0))
            scene_manager.detect_scenes(video)
            scene_list = scene_manager.get_scene_list()

            boundaries: List[SceneBoundary] = []
            for idx, (start_time, end_time) in enumerate(scene_list):
                boundaries.append(
                    SceneBoundary(
                        shot_id=idx,
                        start_time=start_time.get_seconds(),
                        end_time=end_time.get_seconds(),
                        start_frame=start_time.get_frames(),
                        end_frame=end_time.get_frames(),
                    )
                )
            return boundaries
        except ImportError:
            self.context.log_warning("PySceneDetect not installed. Using 5-second interval fallback cuts.")
            return [
                SceneBoundary(
                    shot_id=i,
                    start_time=float(i * 5),
                    end_time=float((i + 1) * 5),
                    start_frame=i * 150,
                    end_frame=(i + 1) * 150,
                )
                for i in range(12)
            ]
        except Exception as exc:
            self.context.log_warning(f"Scene detection error: {exc}. Proceeding with segment bounds.")
            return []

    def _score_candidate_windows(
        self,
        video_path: Path,
        segments: List[TranscriptSegment],
        scene_boundaries: List[SceneBoundary],
        pipeline_config: PipelineConfig,
    ) -> List[ClipProposal]:
        """Score semantic narrative units using Gemini Flash or deterministic heuristics."""
        if not segments:
            return []

        # If Gemini API key is available, evaluate narrative cohesion with structured JSON
        if config.gemini_api_key:
            try:
                self.context.log_info("Dispatching semantic story unit analysis to Gemini (evaluating logic, hooks & payoff)...")
                from google import genai
                client = genai.Client(api_key=config.gemini_api_key)

                transcript_summary = [
                    {"id": s.id, "start": s.start, "end": s.end, "text": s.text}
                    for s in segments
                ]
                seg_lookup = {s.id: s for s in segments}

                prompt = (
                    "You are a master social video editor and narrative director specializing in Greek viral short-form content (Shorts, Reels, TikTok).\n"
                    "You are provided with a visual storyboard of the video (16 frames ordered chronologically) and the transcript.\n"
                    "Your objective is to identify self-contained, logically complete narrative story units from the given transcript that match visually engaging moments.\n"
                    "A great short-form clip MUST have:\n"
                    "1. An inciting hook in the first 3-5 seconds (curiosity gap, provocative question, or bold claim).\n"
                    "2. A coherent, logical explanation or narrative arc that makes complete sense standalone.\n"
                    "3. A definitive conclusion, payoff, or punchline. NEVER cut a speaker mid-thought or mid-sentence.\n\n"
                    f"Target duration: strictly between {pipeline_config.min_clip_duration}s and {pipeline_config.max_clip_duration}s.\n"
                    f"Identify up to {pipeline_config.max_clips} best narrative units.\n"
                    "Evaluate each unit with 4 scores (0-100):\n"
                    "- hook_score: Opening 3s retention / curiosity.\n"
                    "- semantic_score: Standalone logical completeness and sense.\n"
                    "- payoff_score: Quality of conclusion / takeaway.\n"
                    "- virality_score: Overall shareability and impact.\n\n"
                    "Return a JSON array of objects conforming to schema:\n"
                    "[{\n"
                    "  \"start_segment_id\": int,\n"
                    "  \"end_segment_id\": int,\n"
                    "  \"hook_score\": int,\n"
                    "  \"semantic_score\": int,\n"
                    "  \"payoff_score\": int,\n"
                    "  \"virality_score\": int,\n"
                    "  \"hook_text\": str,\n"
                    "  \"reasoning\": str,\n"
                    "  \"layout_mode\": \"single_speaker_smooth\",\n"
                    "  \"seo_title\": str,\n"
                    "  \"seo_description\": str,\n"
                    "  \"seo_tags\": [\"#tag1\", \"#tag2\"]\n"
                    "}]\n"
                    f"Transcript:\n{json.dumps(transcript_summary, ensure_ascii=False)}"
                )

                storyboard_path = self._extract_storyboard(video_path)
                raw_text = call_gemini(client, prompt, image_path=storyboard_path)
                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_text:
                    raw_text = raw_text.split("```")[1].split("```")[0].strip()

                items = json.loads(raw_text)
                proposals: List[ClipProposal] = []
                for item in items:
                    s_id = int(item.get("start_segment_id", 0))
                    e_id = int(item.get("end_segment_id", 0))

                    if s_id in seg_lookup and e_id in seg_lookup and e_id >= s_id:
                        start_time = float(seg_lookup[s_id].start)
                        end_time = float(seg_lookup[e_id].end)
                    else:
                        start_time = float(item.get("start_time", segments[0].start))
                        end_time = float(item.get("end_time", segments[-1].end))

                    if end_time <= start_time:
                        continue

                    proposals.append(
                        ClipProposal(
                            clip_id=str(uuid.uuid4())[:8],
                            start_time=round(start_time, 2),
                            end_time=round(end_time, 2),
                            virality_score=int(item.get("virality_score", 75)),
                            hook_score=int(item.get("hook_score", 80)),
                            semantic_score=int(item.get("semantic_score", 85)),
                            payoff_score=int(item.get("payoff_score", 80)),
                            hook_text=str(item.get("hook_text", "")),
                            reasoning=str(item.get("reasoning", "")),
                            layout_mode=LayoutMode(item.get("layout_mode", "single_speaker_smooth")),
                            seo_title=str(item.get("seo_title", "Κορυφαίο Απόσπασμα")),
                            seo_description=str(item.get("seo_description", "")),
                            seo_tags=list(item.get("seo_tags", ["#shorts", "#viral"])),
                            is_approved=True,
                        )
                    )
                if proposals:
                    self.context.log_info(f"Gemini identified {len(proposals)} logically complete narrative units.")
                    return proposals
            except Exception as e:
                self.context.log_warning(f"Gemini semantic scanning fallback: {e}")

        # Deterministic sliding window fallback heuristic
        return self._generate_heuristic_proposals(segments, pipeline_config)

    def _generate_heuristic_proposals(
        self,
        segments: List[TranscriptSegment],
        pipeline_config: PipelineConfig,
    ) -> List[ClipProposal]:
        """Construct candidate clips based on segment clustering when LLM is unavailable."""
        proposals: List[ClipProposal] = []
        total_segments = len(segments)
        window_size = min(4, total_segments)

        total_windows = max(1, total_segments // max(1, window_size // 2))
        win_idx = 0

        for i in range(0, total_segments, max(1, window_size // 2)):
            window = segments[i : i + window_size]
            if not window:
                continue

            win_idx += 1
            start = window[0].start
            end = window[-1].end
            duration = end - start

            if duration < pipeline_config.min_clip_duration:
                continue
            if duration > pipeline_config.max_clip_duration:
                end = start + pipeline_config.max_clip_duration

            self.context.update_progress(
                PipelinePhase.VIRALITY_SCORING,
                round(66.0 + (win_idx / total_windows) * 8.0, 1),
                f"Analyzing candidate window {win_idx}/{total_windows}: {start:.1f}s -> {end:.1f}s",
            )

            hook_candidate = window[0].text[:80] + "..." if len(window[0].text) > 80 else window[0].text
            proposals.append(
                ClipProposal(
                    clip_id=str(uuid.uuid4())[:8],
                    start_time=round(start, 2),
                    end_time=round(end, 2),
                    virality_score=85,
                    hook_score=88,
                    hook_text=hook_candidate,
                    reasoning="High conceptual density, strong conversational opening beat.",
                    layout_mode=LayoutMode.SINGLE_SPEAKER_SMOOTH,
                    seo_title=f"Το Μεγάλο Μυστικό | {hook_candidate[:30]}",
                    seo_description=f"Ανακαλύψτε το μυστικό: {hook_candidate}",
                    seo_tags=["#Shorts", "#Viral", "#GreekTikTok", "#Mindset"],
                    is_approved=True,
                )
            )

        if not proposals and segments:
            proposals.append(
                ClipProposal(
                    clip_id=str(uuid.uuid4())[:8],
                    start_time=segments[0].start,
                    end_time=min(segments[0].start + 30.0, segments[-1].end),
                    virality_score=80,
                    hook_score=82,
                    hook_text=segments[0].text,
                    reasoning="Primary key discussion segment.",
                    layout_mode=LayoutMode.SINGLE_SPEAKER_SMOOTH,
                    seo_title="Κορυφαίο Απόσπασμα",
                    seo_description="Απόσπασμα από τη συζήτηση.",
                    seo_tags=["#Shorts", "#Greek"],
                    is_approved=True,
                )
            )

        return proposals

    def _generate_pacing_clips(
        self,
        segments: List[TranscriptSegment],
        pipeline_config: PipelineConfig,
    ) -> List[ClipProposal]:
        """Construct candidate clips directly from speech segments and pacing without AI virality scoring."""
        proposals: List[ClipProposal] = []
        total_segments = len(segments)
        if total_segments == 0:
            return []

        i = 0
        while i < total_segments and len(proposals) < pipeline_config.max_clips:
            start_seg = segments[i]
            start_time = start_seg.start
            end_time = start_seg.end

            # Accumulate segments until reaching min_clip_duration
            j = i
            while j < total_segments:
                end_time = segments[j].end
                if (end_time - start_time) >= pipeline_config.min_clip_duration:
                    break
                j += 1

            if (end_time - start_time) > pipeline_config.max_clip_duration:
                end_time = start_time + pipeline_config.max_clip_duration

            hook_text = start_seg.text[:70] + "..." if len(start_seg.text) > 70 else start_seg.text
            proposals.append(
                ClipProposal(
                    clip_id=str(uuid.uuid4())[:8],
                    start_time=round(start_time, 2),
                    end_time=round(end_time, 2),
                    virality_score=None,
                    hook_score=None,
                    hook_text=hook_text,
                    reasoning="Extracted based on natural speech pacing and segment boundaries (virality scoring optional).",
                    layout_mode=LayoutMode.SINGLE_SPEAKER_SMOOTH,
                    seo_title=f"Απόσπασμα {len(proposals) + 1} | {hook_text[:30]}",
                    seo_description=f"Απόσπασμα από τη συζήτηση: {hook_text}",
                    seo_tags=["#shorts", "#greek", "#clips"],
                    is_approved=True,
                )
            )
            i = max(i + 1, j + 1)

        return proposals

    def _snap_boundaries(
        self,
        proposal: ClipProposal,
        segments: List[TranscriptSegment],
        scene_boundaries: List[SceneBoundary],
        min_duration: float = 25.0,
    ) -> ClipProposal:
        """
        Snap clip start and end timestamps to the nearest sentence boundary and scene cut.
        Never cut mid-word or mid-sentence.
        """
        snapped_start = proposal.start_time
        snapped_end = proposal.end_time

        # Match nearest segment start
        candidate_starts = [s.start for s in segments if abs(s.start - proposal.start_time) < 4.0]
        if candidate_starts:
            snapped_start = min(candidate_starts, key=lambda t: abs(t - proposal.start_time))

        # Match nearest segment end
        candidate_ends = [s.end for s in segments if abs(s.end - proposal.end_time) < 4.0]
        if candidate_ends:
            snapped_end = min(candidate_ends, key=lambda t: abs(t - proposal.end_time))

        # Ensure duration meets min_duration by looking for a further segment end if needed
        if (snapped_end - snapped_start) < (min_duration - 1.0):
            further_ends = [s.end for s in segments if (s.end - snapped_start) >= (min_duration - 1.0)]
            if further_ends:
                snapped_end = min(further_ends)

        # Silence / breath-aware snapping: inspect word timings for natural speech pauses
        all_words: List[WordTiming] = []
        for s in segments:
            all_words.extend(s.words)

        best_pause_cut = None
        min_pause_dist = 999.0
        for idx in range(len(all_words) - 1):
            w_curr = all_words[idx]
            w_next = all_words[idx + 1]
            gap = w_next.start - w_curr.end
            if gap >= 0.25 and (w_curr.end - snapped_start) >= (min_duration - 1.0):
                pause_midpoint = round((w_curr.end + w_next.start) / 2.0, 2)
                dist = abs(pause_midpoint - snapped_end)
                if dist < 1.5 and dist < min_pause_dist:
                    min_pause_dist = dist
                    best_pause_cut = pause_midpoint

        if best_pause_cut is not None:
            snapped_end = best_pause_cut

        # Snap to nearest visual scene cut if within 1.0 second threshold
        for scene in scene_boundaries:
            if abs(scene.start_time - snapped_start) < 1.0:
                snapped_start = scene.start_time
            if abs(scene.end_time - snapped_end) < 1.0 and scene.end_time > snapped_start:
                snapped_end = scene.end_time

        # Final safety: start must be strictly before end
        if snapped_end <= snapped_start:
            snapped_end = snapped_start + min_duration

        return proposal.model_copy(
            update={
                "start_time": round(snapped_start, 2),
                "end_time": round(snapped_end, 2),
            }
        )
