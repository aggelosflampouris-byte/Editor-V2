"""Unit tests for the Greek speech orthography guardrail."""

import unittest
from src.core.contracts import TranscriptSegment, WordTiming
from src.core.events import JobContext
from src.modules.speech.transcriber import SpeechAlignmentEngine


class TestGreekGuardrail(unittest.TestCase):

    def test_greek_rule_normalization(self):
        from src.config import config
        original_key = config.gemini_api_key
        try:
            config.gemini_api_key = ""
            ctx = JobContext(job_id="test_job")
            engine = SpeechAlignmentEngine(context=ctx)

            tokens = ["γιατι", "ειναι", "αυτο", "εδω", "σημαντικο"]
            corrected = engine._llm_correct_greek_tokens(tokens)

            self.assertEqual(len(corrected), len(tokens))
            self.assertEqual(corrected[0], "γιατί")
            self.assertEqual(corrected[1], "είναι")
            self.assertEqual(corrected[2], "αυτό")
            self.assertEqual(corrected[3], "εδώ")
        finally:
            config.gemini_api_key = original_key

    def test_guardrail_preserves_token_count_and_timing(self):
        from src.config import config
        original_key = config.gemini_api_key
        try:
            config.gemini_api_key = ""
            ctx = JobContext(job_id="test_job")
            engine = SpeechAlignmentEngine(context=ctx)

            words = [
                WordTiming(word="γιατι", start=0.5, end=0.9, score=0.9),
                WordTiming(word="ειναι", start=1.0, end=1.4, score=0.9),
            ]
            seg = TranscriptSegment(
                id=0,
                text="γιατι ειναι",
                start=0.5,
                end=1.4,
                words=words,
            )

            result = engine._apply_orthographic_guardrail([seg])
            self.assertEqual(len(result), 1)
            self.assertEqual(result[0].words[0].word, "γιατί")
            self.assertEqual(result[0].words[0].start, 0.5)
            self.assertEqual(result[0].words[0].end, 0.9)
            self.assertEqual(result[0].words[1].word, "είναι")
            self.assertEqual(result[0].words[1].start, 1.0)
            self.assertEqual(result[0].words[1].end, 1.4)
        finally:
            config.gemini_api_key = original_key


    def test_transcriber_call_gemini_fallback_on_404(self):
        class MockModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, model, contents):
                self.calls.append(model)
                if model == "gemini-3.5-flash":
                    raise Exception("404 NOT_FOUND. Model is unavailable.")
                class Response:
                    text = '["αποτέλεσμα"]'
                return Response()

        class MockGenAIClient:
            def __init__(self):
                self.models = MockModels()

        ctx = JobContext(job_id="test_job")
        engine = SpeechAlignmentEngine(context=ctx)
        client = MockGenAIClient()
        res = engine._call_gemini(client, "test prompt")
        self.assertEqual(res, '["αποτέλεσμα"]')
        self.assertIn("gemini-3.5-flash-lite", client.models.calls)

    def test_scorer_call_gemini_fallback_on_404(self):
        from src.modules.intelligence.scorer import MultimodalIntelligenceEngine

        class MockModels:
            def __init__(self):
                self.calls = []

            def generate_content(self, model, contents):
                self.calls.append(model)
                if model == "gemini-3.5-flash":
                    raise Exception("503 UNAVAILABLE. High demand spike.")
                class Response:
                    text = '[{"start_segment_id": 0, "end_segment_id": 1}]'
                return Response()

        class MockGenAIClient:
            def __init__(self):
                self.models = MockModels()

        ctx = JobContext(job_id="test_job")
        scorer = MultimodalIntelligenceEngine(context=ctx)
        client = MockGenAIClient()
        res = scorer._call_gemini(client, "test prompt")
        self.assertIn("start_segment_id", res)
        self.assertIn("gemini-3.5-flash-lite", client.models.calls)


if __name__ == "__main__":
    unittest.main()
