import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'app'))
from score_grid import fit_note_intervals


class ScoreGridTests(unittest.TestCase):
    def test_short_note_gets_one_cell_without_losing_pitch_or_original_timing(self):
        notes = [(44.409896, 44.509896, 78), (44.56, 44.72, 76)]
        original = copy.deepcopy(notes)
        result, changes = fit_note_intervals(notes, [44.30, 44.46, 44.62, 44.78, 44.94])
        self.assertEqual(notes, original)
        self.assertEqual(result, [(44.46, 44.62, 78), (44.62, 44.78, 76)])
        self.assertEqual(changes[0]['original_start'], 44.409896)

    def test_quantized_collisions_preserve_all_notes_in_order(self):
        result, _ = fit_note_intervals([(0.01, .04, 60), (.05, .08, 62), (.09, .30, 64)], [0, .1, .2, .3, .4])
        self.assertEqual([n[2] for n in result], [60, 62, 64])
        self.assertTrue(all(a[1] <= b[0] for a,b in zip(result,result[1:])))
        self.assertTrue(all(n[1] > n[0] for n in result))

    def test_insufficient_grid_fails_instead_of_silently_dropping_notes(self):
        with self.assertRaisesRegex(ValueError, 'preserve all melody notes'):
            fit_note_intervals([(0, .01, 60), (.02, .03, 62)], [0, .1])

    def test_exact_grid_intervals_are_unchanged(self):
        notes = [(0, .2, 60), (.3, .4, 62)]
        result, changes = fit_note_intervals(notes, [0, .1, .2, .3, .4])
        self.assertEqual(result, notes)
        self.assertEqual(changes, [])
