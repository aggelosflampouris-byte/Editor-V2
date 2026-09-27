"""Unit tests for semantic scanning, multi-dimensional scoring, and pause-aware snapping."""

import unittest
from pathlib import Path

from src.core.contracts import (
    ClipProposal,
    LayoutMode,
    PipelineConfig,
    TranscriptSegment,
    WordTiming,
)
from src.core.events import JobContext
from src.modules.intelligence.scorer import MultimodalIntelligenceEngine
from src.modules.speech.transcriber import SpeechAlignmentEngine


class TestSemanticScanner(unittest.TestCase):

    def setUp(self):
        self.context = JobContext(job_id="test_semantic_job")

    def test_clip_proposal_multi_dimensional_scores(self):
        proposal = ClipProposal(
            clip_id="c_sem",
            start_time=5.0,
            end_time=32.5,
            virality_score=88,
            hook_score=92,
            semantic_score=95,
            payoff_score=85,
            hook_text="Γιατί αν κολλήσει αυτή η βαλβίδα θα σου κάψει λάδι;",
            reasoning="Complete explanation of PCV failure with clear monetary payoff.",
            layout_mode=LayoutMode.SINGLE_SPEAKER_SMOOTH,
            seo_title="Η Βαλβίδα των 15€ που Σώζει το Μοτέρ",
            seo_description="Ανάλυση για τη βαλβίδα αναθυμιάσεων PCV.",
            seo_tags=["#μηχανική", "#αυτοκίνητο", "#pcv"],
            is_approved=True,
        )
        self.assertEqual(proposal.hook_score, 92)
        self.assertEqual(proposal.semantic_score, 95)
        self.assertEqual(proposal.payoff_score, 85)
        self.assertEqual(proposal.duration, 27.5)

    def test_pipeline_config_domain_presets(self):
        cfg = PipelineConfig(
            domain_preset="Automotive & Mechanics",
            hotwords="βαλβίδα PCV, αναθυμιάσεις, κινητήρας",
        )
        self.assertEqual(cfg.domain_preset, "Automotive & Mechanics")
        self.assertIn("βαλβίδα PCV", cfg.hotwords)

    def test_pause_aware_boundary_snapping(self):
        engine = MultimodalIntelligenceEngine(self.context)
        # Create segments with a distinct silence pause at 28.0s -> 28.5s
        words_seg1 = [
            WordTiming(word="Πρώτη", start=0.0, end=1.0, score=1.0),
            WordTiming(word="πρόταση", start=1.1, end=2.0, score=1.0),
        ]
        words_seg2 = [
            WordTiming(word="Μέση", start=10.0, end=15.0, score=1.0),
            WordTiming(word="εξήγηση", start=15.1, end=27.9, score=1.0),
        ]
        # Notice 0.7s gap between 27.9s and 28.6s
        words_seg3 = [
            WordTiming(word="Επόμενο", start=28.6, end=35.0, score=1.0),
        ]

        segments = [
            TranscriptSegment(id=0, text="Πρώτη πρόταση", start=0.0, end=2.0, words=words_seg1),
            TranscriptSegment(id=1, text="Μέση εξήγηση", start=10.0, end=27.9, words=words_seg2),
            TranscriptSegment(id=2, text="Επόμενο", start=28.6, end=35.0, words=words_seg3),
        ]

        prop = ClipProposal(
            clip_id="test_pause",
            start_time=0.0,
            end_time=28.2,
            hook_text="Hook",
            seo_title="Title",
            seo_description="Desc",
        )

        snapped = engine._snap_boundaries(prop, segments, [], min_duration=20.0)
        # Snapped end should be in the silence gap between 27.9 and 28.6
        self.assertGreaterEqual(snapped.end_time, 27.9)
        self.assertLessEqual(snapped.end_time, 28.6)

    def test_rule_fallback_phonetic_normalization(self):
        from src.config import config
        orig_key = config.gemini_api_key
        try:
            config.gemini_api_key = ""
            engine = SpeechAlignmentEngine(self.context)
            words = [
                WordTiming(word="Λυσπό", start=1.0, end=1.5, score=0.9),
                WordTiming(word="πίβησί", start=1.6, end=2.2, score=0.9),
                WordTiming(word="θυμιάσεων", start=2.3, end=3.0, score=0.9),
            ]
            seg = TranscriptSegment(id=0, text="Λυσπό πίβησί θυμιάσεων", start=1.0, end=3.0, words=words)
            corrected = engine._apply_rule_fallback_segment(seg)

            self.assertEqual(corrected.words[0].word, "Λοιπόν")
            self.assertEqual(corrected.words[1].word, "PCV")
            self.assertEqual(corrected.words[2].word, "αναθυμιάσεων")
        finally:
            config.gemini_api_key = orig_key

    def test_vision_framing_trajectory_no_cascade(self):
        from src.modules.vision.framing import VisionFramingEngine
        engine = VisionFramingEngine(self.context)
        # Verify fallback trajectory produces valid 9:16 CropBox instances without crashing
        res = engine._fallback_center_crops(0.0, 2.0, frame_width=1920, frame_height=1080)
        self.assertGreater(len(res), 0)
        self.assertEqual(res[0].crop_h, 1080)
        self.assertEqual(res[0].crop_w, 608)
        self.assertEqual(res[0].crop_y, 0)
        self.assertTrue(res[0].crop_x % 2 == 0)


if __name__ == "__main__":
    unittest.main()
