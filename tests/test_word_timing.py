import json
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from whisper_transcribe import timing_view
import session_store


class WordTimingTests(unittest.TestCase):
    def test_one_word_can_span_multiple_notes_and_edits_hide_stale_links(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'scores/vocal-melody/analysis-complete.json'
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({'events': [
                {'time': 2, 'values': {'melody': [{'track': 0, 'pitch': 60, 'end_time': 3}]}},
                {'time': 3, 'values': {'melody': [{'track': 0, 'pitch': 62, 'end_time': 4},
                                                 {'track': 1, 'pitch': 40, 'end_time': 4}]}}]}))
            result = {'model': 'whisper-large-v3', 'text': 'Home',
                      'words': [{'text': 'Home', 'start': 2, 'end': 4}]}
            view = timing_view(result, 'home!', root)
            self.assertEqual(view['words'][0]['note_indices'], [0, 1])
            self.assertEqual(len(view['notes']), 2)
            stale = timing_view(result, 'Going home', root)
            self.assertFalse(stale['current'])
            self.assertEqual(stale['words'], [])

    def test_restart_preserves_timestamps(self):
        with tempfile.TemporaryDirectory() as root:
            folder = Path(root) / 'song'
            (folder / 'stems').mkdir(parents=True)
            (folder / 'source.wav').write_bytes(b'original')
            (folder / 'stems/vocals.wav').write_bytes(b'vocals')
            session = session_store.Session('id', str(folder / 'source.wav'), str(folder), 'Song',
                vocals_path=str(folder / 'stems/vocals.wav'), lyrics='Home', ready=True,
                transcription={'model': 'whisper-large-v3', 'text': 'Home',
                               'words': [{'text': 'Home', 'start': 2, 'end': 4}]})
            session_store.save(session)
            self.assertEqual(session_store.load_all(Path(root))['id'].transcription, session.transcription)
