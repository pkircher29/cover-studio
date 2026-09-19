"""Notation-only quantization; never rewrite source audio or analysis timings."""
from bisect import bisect_left
import importlib
import json
from pathlib import Path


def fit_note_intervals(notes, grid):
    """Place ordered monophonic notes in nonzero cells without dropping pitches."""
    if len(grid) < 2 or any(a >= b for a, b in zip(grid, grid[1:])):
        raise ValueError('Notation grid must have increasing boundaries')
    boundaries = [(a + b) / 2 for a, b in zip(grid, grid[1:])]
    placed, changes, previous_end = [], [], 0
    for start, end, pitch in sorted(notes):
        start_index = max(previous_end, min(bisect_left(boundaries, start), len(grid) - 2))
        end_index = max(start_index + 1, bisect_left(boundaries, end))
        if end_index >= len(grid):
            raise ValueError('Not enough notation grid cells to preserve all melody notes')
        placed.append((grid[start_index], grid[end_index], pitch))
        if abs(grid[start_index] - start) > 1e-6 or abs(grid[end_index] - end) > 1e-6:
            changes.append({'pitch': pitch, 'original_start': start, 'original_end': end,
                            'score_start': grid[start_index], 'score_end': grid[end_index]})
        previous_end = end_index
    return placed, changes


def rebuild_notation(exporter, directory, melody_only):
    import pretty_midi
    notation = importlib.import_module(exporter.__package__ + '.notation_sheetsage2')
    directory = Path(directory)
    folder = directory / 'notation'
    beats = notation.read_beats(folder / 'song_beats.txt')
    measures, _ = notation.infer_measures(beats)
    grid, _, _ = notation._build_grid(beats, measures)
    grid = [float(t) for t in grid]
    midi = pretty_midi.PrettyMIDI(str(folder / 'song_melody.mid'))
    changes = []
    for instrument in midi.instruments:
        ordered = sorted(instrument.notes, key=lambda n: (n.start, n.end, n.pitch))
        intervals, adjustments = fit_note_intervals([(n.start, n.end, n.pitch) for n in ordered], grid)
        for note, (start, end, _) in zip(ordered, intervals):
            note.start, note.end = start, end
        changes.extend(dict(voice=instrument.name, **item) for item in adjustments)
    text, score = notation.generate_abc_from_data(midi, beats,
        notation.read_chords(folder / 'song_chords.txt'),
        notation.read_keys(folder / 'song_keys.txt'),
        notation.read_structures(folder / 'song_structures.txt'), melody_only=melody_only)
    midi.write(str(folder / 'song_melody-grid.mid'))
    (folder / 'grid-adjustments.json').write_text(json.dumps(changes, indent=2), encoding='utf-8')
    temporary = directory / 'score.abc.tmp'
    temporary.write_text(text, encoding='utf-8')
    temporary.replace(directory / 'score.abc')
    return text, [f'Notation only: fitted {len(changes)} note intervals to the full-song grid; original timings preserved.', *score.diagnostics]
