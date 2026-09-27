"""Unit tests for the thread-safe EventEmitter and JobContext observer bridge."""

import unittest
from src.core.contracts import JobStatus, PipelinePhase
from src.core.events import EventEmitter, JobContext, PipelineEvent, PipelineEventType


class TestEvents(unittest.TestCase):

    def test_event_emitter_drain(self):
        emitter = EventEmitter()
        ev1 = PipelineEvent(
            job_id="test1",
            event_type=PipelineEventType.LOG_EMITTED,
            phase=PipelinePhase.INGESTION,
            message="Log 1",
        )
        ev2 = PipelineEvent(
            job_id="test1",
            event_type=PipelineEventType.PROGRESS_UPDATED,
            phase=PipelinePhase.INGESTION,
            message="Progress 50%",
            progress=50.0,
        )

        emitter.emit(ev1)
        emitter.emit(ev2)

        drained = emitter.drain_events()
        self.assertEqual(len(drained), 2)
        self.assertEqual(drained[0].message, "Log 1")
        self.assertEqual(drained[1].progress, 50.0)

        # Queue should be empty now
        self.assertEqual(len(emitter.drain_events()), 0)

    def test_job_context_emission(self):
        emitter = EventEmitter()
        ctx = JobContext(job_id="job_abc", emitter=emitter)

        ctx.set_status(JobStatus.DOWNLOADING, "Starting download")
        ctx.update_progress(PipelinePhase.INGESTION, 25.0, "Downloading 25%")
        ctx.log_info("Info message")

        events = emitter.drain_events()
        self.assertEqual(len(events), 3)
        self.assertEqual(events[0].event_type, PipelineEventType.STATUS_CHANGED)
        self.assertEqual(events[1].event_type, PipelineEventType.PROGRESS_UPDATED)
        self.assertEqual(events[2].event_type, PipelineEventType.LOG_EMITTED)


if __name__ == "__main__":
    unittest.main()
