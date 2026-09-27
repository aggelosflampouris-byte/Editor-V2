"""Greek-aware speech transcription, phoneme alignment, live progression, and orthographic guardrail."""

import os
from pathlib import Path
from typing import List, Optional
import json

from src.config import config
from src.core.logger import logger
from src.core.gemini import call_gemini
from src.core.contracts import PipelinePhase, TranscriptSegment, WordTiming
from src.core.events import JobContext
from src.core.exceptions import AlignmentError


class SpeechAlignmentEngine:
    """Orchestrates Whisper transcription, phoneme alignment, and Greek LLM guardrails with live streaming feedback."""

    def __init__(self, context: JobContext):
        self.context = context

    def transcribe_and_align(
        self,
        audio_path: Path,
        enable_guardrail: bool = True,
        whisper_model: Optional[str] = None,
        transcription_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> List[TranscriptSegment]:
        """Transcribe audio with live progress per segment, perform phoneme alignment, and apply Greek orthographic guardrails."""
        target_model = whisper_model or config.whisper_model
        if " " in target_model:
            target_model = target_model.split(" ")[0].strip()

        self.context.log_info(f"Loading speech alignment models (Whisper: {target_model})...")
        self.context.update_progress(PipelinePhase.TRANSCRIPTION, 35.0, f"Loading Whisper speech model ({target_model})...")

        # Step 1: Run Whisper transcription with live segment feedback
        raw_segments = self._run_whisper_transcription(
            audio_path=audio_path,
            model_name=target_model,
            transcription_prompt=transcription_prompt,
            hotwords=hotwords,
        )

        # Step 2: Apply Greek Orthographic Guardrail if enabled
        if enable_guardrail:
            self.context.log_info(f"Applying Greek orthographic & semantic guardrail on {len(raw_segments)} segments...")
            self.context.update_progress(PipelinePhase.ORTHOGRAPHY_GUARDRAIL, 60.0, "Verifying Greek grammar, accents, and terminology...")
            validated_segments = self._apply_orthographic_guardrail(raw_segments)
        else:
            validated_segments = raw_segments

        self.context.set_transcript(validated_segments)
        self.context.log_info(f"Speech alignment finalized: {len(validated_segments)} segments mapped.")
        return validated_segments

    def _run_whisper_transcription(
        self,
        audio_path: Path,
        model_name: str = "large-v3",
        transcription_prompt: Optional[str] = None,
        hotwords: Optional[str] = None,
    ) -> List[TranscriptSegment]:
        """Transcribe using WhisperX with wav2vec 2.0 forced phoneme alignment."""
        try:
            import whisperx
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            compute_type = "float16" if device == "cuda" else "int8"
            
            self.context.log_info(f"Loading WhisperX model '{model_name}' on {device.upper()}...")
            
            # 1. Transcribe with Whisper
            model = whisperx.load_model(
                model_name,
                device,
                compute_type=compute_type,
                language="el"
            )
            
            self.context.log_info("Starting batch transcription (language: 'el')...")
            
            # WhisperX transcribe
            result = model.transcribe(
                str(audio_path),
                batch_size=8 if device == "cuda" else 2,
                language="el"
            )
            
            total_duration = 60.0
            self.context.log_info("Transcription complete. Loading Greek wav2vec 2.0 alignment model...")
            self.context.update_progress(PipelinePhase.TRANSCRIPTION, 45.0, "Transcription complete. Aligning phonemes to audio waves...")
            
            # Free up VRAM
            import gc
            del model
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

            # 2. Align with wav2vec 2.0
            align_model_name = config.greek_alignment_model
            self.context.log_info(f"Using forced alignment model: {align_model_name}")
            model_a, metadata = whisperx.load_align_model(
                language_code="el",
                model_name=align_model_name,
                device=device
            )
            
            aligned_result = whisperx.align(
                result["segments"],
                model_a,
                metadata,
                str(audio_path),
                device,
                return_char_alignments=False
            )
            
            self.context.log_info("Phoneme forced alignment complete.")

            segments: List[TranscriptSegment] = []
            for idx, seg in enumerate(aligned_result["segments"]):
                word_timings: List[WordTiming] = []
                if "words" in seg:
                    for w in seg["words"]:
                        clean_word = w.get("word", "").strip()
                        # WhisperX alignment might sometimes skip words or lack timestamps if completely muted
                        if clean_word and "start" in w and "end" in w and w["start"] <= w["end"]:
                            word_timings.append(
                                WordTiming(
                                    word=clean_word,
                                    start=round(float(w["start"]), 2),
                                    end=round(float(w["end"]), 2),
                                    score=round(float(w.get("score", 1.0)), 2),
                                )
                            )

                if word_timings:
                    seg_item = TranscriptSegment(
                        id=idx,
                        text=seg["text"].strip(),
                        start=word_timings[0].start,
                        end=word_timings[-1].end,
                        words=word_timings,
                    )
                    segments.append(seg_item)
                    
                    self.context.log_info(f"[{seg_item.start:.1f}s -> {seg_item.end:.1f}s] \"{seg_item.text}\"")

            if not segments:
                self.context.log_warning("No speech segments detected in audio file.")
            
            self.context.update_progress(PipelinePhase.TRANSCRIPTION, 58.0, "Phoneme alignment finalized.")
            return segments

        except ImportError:
            logger.warning("WhisperX library not installed; generating deterministic placeholder alignment.")
            self.context.log_warning("WhisperX library not installed; generating deterministic placeholder alignment.")
            return self._generate_fallback_segments()
        except Exception as exc:
            logger.exception("WhisperX transcription/alignment failed")
            raise AlignmentError("WhisperX transcription failed", details=str(exc)) from exc

    def _apply_orthographic_guardrail(
        self, segments: List[TranscriptSegment]
    ) -> List[TranscriptSegment]:
        """
        Run targeted Greek orthographic and grammatical check with macro-context Gemini batching.
        Strict Rule: Must preserve 1:1 token length and word timestamps.
        """
        if not segments:
            return []

        # Batch process segments in groups of 20 for rich context
        batch_size = 20
        total_segments = len(segments)
        validated: List[TranscriptSegment] = []

        for b_start in range(0, total_segments, batch_size):
            b_segments = segments[b_start : b_start + batch_size]
            b_corrected = self._correct_segment_batch_with_llm(b_segments)
            validated.extend(b_corrected)

            progress_val = round(58.0 + (len(validated) / total_segments) * 4.0, 1)
            self.context.update_progress(
                PipelinePhase.ORTHOGRAPHY_GUARDRAIL,
                progress_val,
                f"Verifying Greek grammar: segments {len(validated)}/{total_segments}",
            )

        return validated

    def _correct_segment_batch_with_llm(
        self, batch: List[TranscriptSegment]
    ) -> List[TranscriptSegment]:
        """Send a conversational batch of segments to Gemini for contextual grammar/vocabulary correction."""
        original_token_map = {seg.id: [w.word for w in seg.words] for seg in batch}

        if config.gemini_api_key:
            try:
                from google import genai
                client = genai.Client(api_key=config.gemini_api_key)

                payload = [
                    {"id": seg.id, "text": seg.text, "words": [w.word for w in seg.words]}
                    for seg in batch
                ]

                prompt = (
                    "You are a Greek technical editor and linguist specializing in automotive and conversational transcripts.\n"
                    "Given this batch of consecutive transcribed segments from a Greek discussion:\n"
                    "1. Correct technical jargon and phonetics (e.g. 'Λυσπό' -> 'Λοιπόν', 'βαλβίδα θυμιάσεων' -> 'βαλβίδα αναθυμιάσεων', 'Πίβησί' -> 'PCV').\n"
                    "2. Add correct Greek accents (τόνοι), standard casing, and grammatical inflection.\n"
                    "3. CRITICAL: For each segment, you MUST return a list of words with the EXACT same count as the input words array so audio timestamps stay in sync.\n"
                    "Output JSON schema: [{\"id\": int, \"words\": [\"word1\", \"word2\", ...]}]\n"
                    f"Segments:\n{json.dumps(payload, ensure_ascii=False)}"
                )

                raw_text = call_gemini(client, prompt)
                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_text:
                    raw_text = raw_text.split("```")[1].split("```")[0].strip()

                corrected_items = json.loads(raw_text)
                corrected_lookup = {item["id"]: item["words"] for item in corrected_items if "id" in item and "words" in item}

                result: List[TranscriptSegment] = []
                for seg in batch:
                    orig_words = original_token_map[seg.id]
                    llm_words = corrected_lookup.get(seg.id, [])
                    if len(llm_words) == len(orig_words):
                        new_timings = [
                            WordTiming(word=cw, start=tw.start, end=tw.end, score=tw.score)
                            for cw, tw in zip(llm_words, seg.words)
                        ]
                        result.append(
                            TranscriptSegment(
                                id=seg.id,
                                text=" ".join(llm_words),
                                start=seg.start,
                                end=seg.end,
                                words=new_timings,
                            )
                        )
                    else:
                        result.append(self._apply_rule_fallback_segment(seg))
                return result

            except Exception as e:
                logger.warning(f"Batch LLM guardrail fallback to rules: {e}")
                self.context.log_warning(f"Batch LLM guardrail fallback to rules: {e}")

        # Fallback to rule-based per segment
        return [self._apply_rule_fallback_segment(seg) for seg in batch]



    def _apply_rule_fallback_segment(self, segment: TranscriptSegment) -> TranscriptSegment:
        """Fallback deterministic phonetic rule correction for a segment."""
        orig_words = [w.word for w in segment.words]
        corrected_words = self._llm_correct_greek_tokens(orig_words)
        new_timings = [
            WordTiming(word=cw, start=tw.start, end=tw.end, score=tw.score)
            for cw, tw in zip(corrected_words, segment.words)
        ]
        return TranscriptSegment(
            id=segment.id,
            text=" ".join(corrected_words),
            start=segment.start,
            end=segment.end,
            words=new_timings,
        )

    def _llm_correct_greek_tokens(self, tokens: List[str]) -> List[str]:
        """
        Call Gemini or standard Greek orthographic rule normalization.
        Ensures accents (τόνοι) and standard orthography without changing count.
        """
        if not tokens:
            return []

        # If Gemini API key is configured, prompt Gemini with strict schema
        if config.gemini_api_key:
            try:
                from google import genai
                client = genai.Client(api_key=config.gemini_api_key)
                prompt = (
                    "You are a Greek orthography and grammar normalizer. "
                    "You are given a JSON list of Greek words. "
                    "Fix missing accents (τόνοι), common phonetic errors, and capitalization. "
                    "CRITICAL: You MUST return a JSON list with EXACTLY the same number of words in the exact same order. "
                    "Do NOT add, remove, split, or merge words. "
                    f"Input words: {json.dumps(tokens, ensure_ascii=False)}"
                )
                raw_text = call_gemini(client, prompt)
                if "```json" in raw_text:
                    raw_text = raw_text.split("```json")[1].split("```")[0].strip()
                elif "```" in raw_text:
                    raw_text = raw_text.split("```")[1].split("```")[0].strip()
                result = json.loads(raw_text)
                if isinstance(result, list) and len(result) == len(tokens):
                    return [str(x) for x in result]
            except Exception as e:
                self.context.log_warning(f"LLM Greek guardrail fallback to rules: {e}")

        # Deterministic standard Greek orthographic rule correction fallback
        greek_accents_dict = {
            "ειναι": "είναι",
            "γιατι": "γιατί",
            "ποιος": "ποιός",
            "ποια": "ποιά",
            "ποιο": "ποιό",
            "αυτος": "αυτός",
            "αυτη": "αυτή",
            "αυτο": "αυτό",
            "οταν": "όταν",
            "τωρα": "τώρα",
            "εδω": "εδώ",
            "εκει": "εκεί",
            "ομως": "όμως",
            "λοιπον": "λοιπόν",
            "παντα": "πάντα",
            "ποτε": "ποτέ",
            "πολυ": "πολύ",
            "μονο": "μόνο",
            "μετα": "μετά",
            "μαζι": "μαζί",
            "λυσπο": "λοιπόν",
            "λυσπό": "λοιπόν",
            "πιβησι": "PCV",
            "πίβησί": "PCV",
            "πισυβι": "PCV",
            "πίσυβί": "PCV",
            "πισηβι": "PCV",
            "θυμιασεων": "αναθυμιάσεων",
            "θυμιάσεων": "αναθυμιάσεων",
            "ξαιθελα": "ήθελα",
            "ξαίθελα": "ήθελα",
            "ξαβγαλα": "βγάλω",
            "ξαβγάλα": "βγάλω",
        }

        corrected: List[str] = []
        for word in tokens:
            lower = word.lower()
            if lower in greek_accents_dict:
                replacement = greek_accents_dict[lower]
                if word.isupper():
                    corrected.append(replacement.upper())
                elif word[0].isupper():
                    corrected.append(replacement.capitalize())
                else:
                    corrected.append(replacement)
            else:
                corrected.append(word)

        return corrected

    def _generate_fallback_segments(self) -> List[TranscriptSegment]:
        """Provides deterministic transcript segments for test/mock runs."""
        words_sample = [
            ("Αυτό", 0.5, 0.8),
            ("είναι", 0.9, 1.2),
            ("το", 1.3, 1.5),
            ("πιο", 1.6, 1.8),
            ("σημαντικό", 1.9, 2.5),
            ("μυστικό", 2.6, 3.1),
            ("για", 3.2, 3.4),
            ("την", 3.5, 3.7),
            ("επιτυχία", 3.8, 4.5),
        ]
        timings = [
            WordTiming(word=w, start=s, end=e, score=0.98)
            for w, s, e in words_sample
        ]
        return [
            TranscriptSegment(
                id=0,
                text="Αυτό είναι το πιο σημαντικό μυστικό για την επιτυχία.",
                start=0.5,
                end=4.5,
                words=timings,
            )
        ]
