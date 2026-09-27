"""Unit tests verifying event payload propagation into AppStateManager session state."""

import unittest
from pathlib import Path

from src.core.contracts import (
    ClipProposal,
    JobStatus,
    LayoutMode,
    PipelineConfig,
    PipelinePhase,
    RenderedClip,
    TranscriptSegment,
    WordTiming,
)
from src.core.events import JobContext
from src.modules.intelligence.scorer import MultimodalIntelligenceEngine
from src.ui.state import AppStateManager, get_event_emitter


class TestEventPropagation(unittest.TestCase):

    def setUp(self):
        AppStateManager.initialize_session()
        AppStateManager.reset_job()
        self.emitter = get_event_emitter()
        self.emitter.clear()
        self.context = JobContext(job_id="test_job", emitter=self.emitter)

    def test_clips_proposed_propagates_to_job(self):
        sample_clips = [
            ClipProposal(
                clip_id="c1",
                start_time=10.0,
                end_time=35.0,
                virality_score=None,
                hook_score=None,
                hook_text="Δοκιμή απόσπασμα",
                reasoning="Test reasoning",
                layout_mode=LayoutMode.SINGLE_SPEAKER_SMOOTH,
                seo_title="Τίτλος",
                seo_description="Περιγραφή",
                seo_tags=["#tag1"],
                is_approved=True,
            )
        ]
        self.context.set_proposed_clips(sample_clips)

        changed = AppStateManager.drain_and_apply_events()
        self.assertTrue(changed)

        job = AppStateManager.get_job()
        self.assertEqual(job.status, JobStatus.READY_FOR_REVIEW)
        self.assertEqual(len(job.proposed_clips), 1)
        self.assertEqual(job.proposed_clips[0].clip_id, "c1")
        self.assertIsNone(job.proposed_clips[0].virality_score)

    def test_transcript_propagates_to_job(self):
        words = [WordTiming(word="Δοκιμή", start=1.0, end=1.5, score=0.99)]
        segments = [
            TranscriptSegment(
                id=0,
                text="Δοκιμή",
                start=1.0,
                end=1.5,
                words=words,
            )
        ]
        self.context.set_transcript(segments)

        AppStateManager.drain_and_apply_events()
        job = AppStateManager.get_job()
        self.assertEqual(len(job.transcript), 1)
        self.assertEqual(job.transcript[0].text, "Δοκιμή")

    def test_rendered_clip_propagates_to_job(self):
        rendered = RenderedClip(
            clip_id="c1",
            video_path="outputs/c1.mp4",
            metadata_path="outputs/c1.json",
            duration=25.0,
            resolution="1080x1920",
        )
        self.context.add_rendered_clip(rendered)

        AppStateManager.drain_and_apply_events()
        job = AppStateManager.get_job()
        self.assertEqual(len(job.rendered_clips), 1)
        self.assertEqual(job.rendered_clips[0].clip_id, "c1")

    def test_scorer_bypassed_virality_produces_clips(self):
        engine = MultimodalIntelligenceEngine(self.context)
        # Create 10 mock segments spanning 50 seconds
        segments = []
        for i in range(10):
            w = WordTiming(word="λέξη", start=float(i * 5), end=float((i + 1) * 5), score=1.0)
            segments.append(
                TranscriptSegment(
                    id=i,
                    text=f"Πρόταση {i}",
                    start=float(i * 5),
                    end=float((i + 1) * 5),
                    words=[w],
                )
            )

        cfg = PipelineConfig(
            max_clips=2,
            min_clip_duration=20.0,
            max_clip_duration=35.0,
            enable_virality_scoring=False,
        )

        clips = engine.process(
            video_path=Path("non_existent_fake.mp4"),
            segments=segments,
            pipeline_config=cfg,
        )

        self.assertGreaterEqual(len(clips), 1)
        for clip in clips:
            self.assertIsNone(clip.virality_score)
            self.assertGreaterEqual(clip.end_time - clip.start_time, 19.0)


if __name__ == "__main__":
    unittest.main()
