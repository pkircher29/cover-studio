import copy
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
import server
import session_store
from whisper_transcribe import correct_words, timing_view


class WordCorrectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_section_edits_persist_without_moving_word_times(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / 'song'
            (folder / 'stems').mkdir(parents=True)
            (folder / 'source.wav').write_bytes(b'original')
            (folder / 'stems/vocals.wav').write_bytes(b'vocals')
            session = server.Session('sections-test', str(folder/'source.wav'), str(folder), 'Song',
                vocals_path=str(folder/'stems/vocals.wav'), lyrics='Hide across', ready=True,
                transcription=copy.deepcopy(self.data))
            server.SESSIONS[session.id] = session
            try:
                edited = '[Verse 1]\nHide\n\n[Chorus]\nacross'
                result = await server.save_session_lyrics(session.id, server.LyricsUpdate(lyrics=edited))
                self.assertTrue(result['lyric_timing']['current'])
                self.assertEqual(session.transcription['words'], self.data['words'])
                restored = session_store.load_all(Path(root))[session.id]
                self.assertTrue(restored.lyrics_reviewed)
                self.assertEqual(server._session_result(restored)['generation_lyrics']['lyrics'], edited)
                # Removing every section is also an intentional edit.
                await server.save_session_lyrics(session.id, server.LyricsUpdate(lyrics='Hide across'))
                self.assertEqual(server._session_result(session)['generation_lyrics']['lyrics'], 'Hide across')
            finally:
                server.SESSIONS.pop(session.id, None)

    def setUp(self):
        self.data = {'model': 'whisper-large-v3', 'text': 'Hide across',
            'alignment_method': 'ctc-forced-alignment-v1',
            'words': [{'text': 'Hide', 'start': 17.38, 'end': 18.58, 'original_start': 17.38,
                       'original_end': 18.58, 'alignment_score': .3, 'alignment_status': 'review'},
                      {'text': 'across', 'start': 19.739, 'end': 20.28}]}

    async def test_save_replaces_word_and_retains_all_timing_after_reopen(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / 'song'
            (folder / 'stems').mkdir(parents=True)
            (folder / 'source.wav').write_bytes(b'original')
            (folder / 'stems/vocals.wav').write_bytes(b'vocals')
            session = server.Session('word-edit-test', str(folder/'source.wav'), str(folder), 'Song',
                vocals_path=str(folder/'stems/vocals.wav'), lyrics='Hide across', ready=True,
                transcription=copy.deepcopy(self.data))
            server.SESSIONS[session.id] = session
            try:
                response = await server.save_session_lyrics(session.id,
                    server.LyricsUpdate(lyrics='High across', expected_lyrics='Hide across'))
                self.assertTrue(response['lyric_timing']['current'])
                restored = session_store.load_all(Path(root))[session.id]
                self.assertEqual(restored.lyrics, 'High across')
                for old, new in zip(self.data['words'], restored.transcription['words']):
                    self.assertEqual({k:v for k,v in old.items() if k != 'text'},
                        {k:v for k,v in new.items() if k not in ('text','text_edited','recognized_text')})
                self.assertEqual(restored.transcription['words'][0]['recognized_text'], 'Hide')
                with self.assertRaises(server.HTTPException) as conflict:
                    await server.save_session_lyrics(session.id,
                        server.LyricsUpdate(lyrics='Home across', expected_lyrics='Hide across'))
                self.assertEqual(conflict.exception.status_code, 409)
            finally:
                server.SESSIONS.pop(session.id, None)

    def test_inserting_words_does_not_reassign_existing_intervals(self):
        result = correct_words(self.data, 'Hide across', 'Hide away across')
        self.assertIs(result, self.data)
        self.assertFalse(timing_view(result, 'Hide away across', '.')['current'])

    def test_punctuation_and_line_breaks_keep_word_intervals(self):
        result = correct_words(self.data, 'Hide across', 'High!\nAcross.')
        self.assertEqual(result['text'], 'High!\nAcross.')
        self.assertEqual(result['words'][1]['start'], 19.739)
        self.assertEqual(self.data['text'], 'Hide across')
