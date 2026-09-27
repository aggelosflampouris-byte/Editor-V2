"""Unit tests for Pydantic v2 domain models and validation rules."""

import unittest
from pydantic import ValidationError

from src.core.contracts import (
    ClipProposal,
    JobStatus,
    LayoutMode,
    PipelineConfig,
    PipelineJob,
    WordTiming,
)


class TestContracts(unittest.TestCase):

    def test_word_timing_valid(self):
        wt = WordTiming(word="δοκιμή", start=1.0, end=2.0, score=0.95)
        self.assertEqual(wt.word, "δοκιμή")
        self.assertEqual(wt.start, 1.0)
        self.assertEqual(wt.end, 2.0)

    def test_word_timing_invalid_order(self):
        with self.assertRaises(ValidationError):
            WordTiming(word="λάθος", start=3.0, end=1.0)

    def test_clip_proposal_duration(self):
        clip = ClipProposal(
            start_time=10.0,
            end_time=35.5,
            virality_score=85,
            hook_score=90,
            hook_text="Δείτε αυτό τώρα!",
            reasoning="Strong opening hook.",
            layout_mode=LayoutMode.SINGLE_SPEAKER_SMOOTH,
            seo_title="Κορυφαίο Βίντεο",
            seo_description="Περιγραφή",
            seo_tags=["#shorts", "#greek"],
        )
        self.assertEqual(clip.duration, 25.5)
        self.assertTrue(clip.is_approved)

    def test_clip_proposal_invalid_times(self):
        with self.assertRaises(ValidationError):
            ClipProposal(
                start_time=40.0,
                end_time=30.0,
                virality_score=80,
                hook_score=80,
                hook_text="Hook",
                reasoning="Reason",
                layout_mode=LayoutMode.SINGLE_SPEAKER_SMOOTH,
                seo_title="Title",
                seo_description="Desc",
            )

    def test_pipeline_config_duration_validation(self):
        with self.assertRaises(ValidationError):
            PipelineConfig(min_clip_duration=60.0, max_clip_duration=30.0)

    def test_pipeline_job_initial_state(self):
        job = PipelineJob(url="https://www.youtube.com/watch?v=test")
        self.assertEqual(job.status, JobStatus.IDLE)
        self.assertEqual(job.progress_percentage, 0.0)
        self.assertEqual(len(job.proposed_clips), 0)

    def test_clip_proposal_optional_virality(self):
        clip = ClipProposal(
            start_time=5.0,
            end_time=35.0,
            virality_score=None,
            hook_score=None,
            hook_text="Optional hook text",
            reasoning="Pacing-based extraction",
            seo_title="Title",
            seo_description="Desc",
        )
        self.assertIsNone(clip.virality_score)
        self.assertIsNone(clip.hook_score)
        self.assertEqual(clip.duration, 30.0)

    def test_pipeline_config_optional_virality(self):
        cfg = PipelineConfig(enable_virality_scoring=False, virality_threshold=None)
        self.assertFalse(cfg.enable_virality_scoring)
        self.assertIsNone(cfg.virality_threshold)


if __name__ == "__main__":
    unittest.main()
