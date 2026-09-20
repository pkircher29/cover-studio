import sys
import tempfile
from contextlib import nullcontext
from types import SimpleNamespace
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
import sheetsage2_transcribe as st

class DirectScoreTests(TestCase):
    def setUp(self):
        torch = patch.dict(sys.modules, {'torch': SimpleNamespace(inference_mode=nullcontext)})
        torch.start()
        self.addCleanup(torch.stop)
    def test_one_full_song_call_and_verbatim_export_then_cached(self):
        result={'abc':'EXACT native score\n','events':[], 'warnings':[]}
        model=Mock()
        model.transcribe.return_value=result
        with tempfile.TemporaryDirectory() as root, patch.object(st,'_load_model',return_value=model), patch.object(st,'rebuild_notation') as repair:
            first=st.transcribe_cover('vocals.wav','original.wav',root)
            second=st.transcribe_cover('vocals.wav','original.wav',root)
            model.transcribe.assert_called_once_with('original.wav',output_dir=str(Path(root)/'direct-song'),melody_only=True)
            self.assertEqual(first['abc'],result['abc'])
            self.assertEqual(second,first)
            repair.assert_not_called()
    def test_export_failure_is_not_repaired_or_cached(self):
        model=Mock()
        model.transcribe.return_value={'abc_error':'invalid grid'}
        with tempfile.TemporaryDirectory() as root, patch.object(st,'_load_model',return_value=model), patch.object(st,'rebuild_notation') as repair:
            with self.assertRaisesRegex(RuntimeError,'invalid grid'):
                st.transcribe_cover('vocals.wav','original.wav',root)
            self.assertFalse((Path(root)/'direct-song/analysis-complete.json').exists())
            repair.assert_not_called()
