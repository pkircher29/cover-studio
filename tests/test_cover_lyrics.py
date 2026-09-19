import copy
from pathlib import Path
import sys
import tempfile
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from cover_lyrics import prepare


class CoverLyricsTests(unittest.TestCase):
    def test_sections_pauses_and_cross_boundary_word_preserve_every_word(self):
        with tempfile.TemporaryDirectory() as folder:
            score = Path(folder) / 'scores/melody'
            score.mkdir(parents=True)
            (score / 'structure.lab').write_text('0 4 verse\n4 8 chorus\n8 12 bridge\n')
            data = {'text': 'Stay here sing with me', 'words': [
                {'text': w, 'start': a, 'end': b} for w,a,b in
                [('Stay',1,2),('here',3,4.5),('sing',5,5.5),('with',5.6,6),('me',9,10)]]}
            before = copy.deepcopy(data)
            result = prepare(data['text'],data,folder)
            self.assertEqual(result['lyrics'],'[Verse]\nStay\nhere\n\n[Chorus]\nsing with\n\n[Bridge]\nme')
            self.assertEqual(data,before)
            self.assertIsNone(result['error'])
            self.assertTrue(prepare('extra '+data['text'],data,folder)['error'])
    def test_manual_sections_retained(self):
        text='[Verse]\nMy words\n[Chorus]\nMy chorus'
        self.assertEqual(prepare(text,None,'.')['lyrics'],text)
    def test_missing_analysis_does_not_invent_structure(self):
        self.assertTrue(prepare('hello',None,'.')['error'])
