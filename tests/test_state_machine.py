"""Unit tests for the state machine transitions and invalid transition guards."""

import unittest
from src.core.contracts import JobStatus
from src.core.exceptions import InvalidStateTransitionError
from src.ui.state import ALLOWED_TRANSITIONS, AppStateManager


class TestStateMachine(unittest.TestCase):

    def setUp(self):
        AppStateManager.initialize_session()
        AppStateManager.reset_job()

    def test_allowed_transitions_structure(self):
        self.assertIn(JobStatus.IDLE, ALLOWED_TRANSITIONS)
        self.assertIn(JobStatus.DOWNLOADING, ALLOWED_TRANSITIONS[JobStatus.IDLE])
        self.assertNotIn(JobStatus.COMPLETED, ALLOWED_TRANSITIONS[JobStatus.IDLE])

    def test_valid_sequential_transitions(self):
        job = AppStateManager.get_job()
        self.assertEqual(job.status, JobStatus.IDLE)

        # Valid: IDLE -> DOWNLOADING
        AppStateManager.transition_state(JobStatus.DOWNLOADING)
        self.assertEqual(AppStateManager.get_job().status, JobStatus.DOWNLOADING)

        # Valid: DOWNLOADING -> TRANSCRIBING
        AppStateManager.transition_state(JobStatus.TRANSCRIBING)
        self.assertEqual(AppStateManager.get_job().status, JobStatus.TRANSCRIBING)

        # Valid: TRANSCRIBING -> SCORING
        AppStateManager.transition_state(JobStatus.SCORING)
        self.assertEqual(AppStateManager.get_job().status, JobStatus.SCORING)

        # Valid: SCORING -> READY_FOR_REVIEW
        AppStateManager.transition_state(JobStatus.READY_FOR_REVIEW)
        self.assertEqual(AppStateManager.get_job().status, JobStatus.READY_FOR_REVIEW)

    def test_invalid_transition_raises(self):
        self.assertEqual(AppStateManager.get_job().status, JobStatus.IDLE)

        # Attempt illegal skip from IDLE directly to COMPLETED
        with self.assertRaises(InvalidStateTransitionError):
            AppStateManager.transition_state(JobStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
