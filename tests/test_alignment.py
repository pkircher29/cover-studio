import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import AsyncMock, patch
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from align_lyrics import groups, review_overlaps
import server
import whisper_transcribe


class AlignmentTests(unittest.TestCase):
    def test_groups_do_not_cross_long_silence_or_long_audio_windows(self):
        words = [{'start': i, 'end': i+1} for i in range(45)]
        result = list(groups(words))
        self.assertEqual(sum(result, []), list(range(45)))
        self.assertTrue(all(words[g[-1]]['end'] - words[g[0]]['start'] <= 20 for g in result))
        self.assertEqual(list(groups([{'start': 0, 'end': 1}, {'start': 8, 'end': 9}])), [[0], [1]])

    def test_conflicting_refinements_restore_originals_even_when_reversion_cascades(self):
        words = [dict(start=0, end=2, original_start=0, original_end=1, alignment_status='aligned'),
                 dict(start=2, end=3, original_start=1, original_end=2, alignment_status='aligned'),
                 dict(start=2.5, end=4, original_start=2, original_end=4, alignment_status='review')]
        review_overlaps(words)
        self.assertTrue(all(a['end'] <= b['start'] for a, b in zip(words, words[1:])))
        self.assertTrue(all(w['alignment_status'] == 'review' for w in words))


class AlignmentJobTests(unittest.IsolatedAsyncioTestCase):
    async def test_failure_preserves_previous_timing_and_lyrics(self):
        session = server.Session('test', 'original.wav', 'folder', 'test', lyrics='Home',
            transcription={'text': 'Home', 'words': []}, ready=True)
        before = session.transcription
        with patch.object(server.whisper_transcribe, 'refine', AsyncMock(side_effect=RuntimeError('test failure'))), patch.object(server.session_store, 'save'):
            job = server.Job('test', 'transcribe_lyrics')
            await server._run_whisper_job(job, session, timing_only=True)
        self.assertIs(session.transcription, before)
        self.assertEqual(session.lyrics, 'Home')
        self.assertEqual(job.status, 'error')
        self.assertIsNone(session.active_job_id)

    async def test_alignment_cache_does_not_repeat_or_drift(self):
        data = {'alignment_method': 'ctc-forced-alignment-v1', 'words': [{'start': 1, 'end': 2}]}
        self.assertIs(await whisper_transcribe.refine('unused', 'unused', data), data)
