"""Prepare sectioned lyric text for YuE2 without changing saved word timing."""
from pathlib import Path
import re

TAGS = {'intro': 'Intro', 'verse': 'Verse', 'pre-chorus': 'Pre-Chorus',
        'chorus': 'Chorus', 'bridge': 'Bridge', 'outro': 'Outro',
        'interlude': 'Interlude', 'instrumental': 'Instrumental', 'solo': 'Solo'}
SECTION = re.compile(r'^\s*\[(?:intro|verse|pre-chorus|chorus|bridge|outro|interlude|instrumental|solo)(?:\s+\d+)?\]\s*$', re.I | re.M)


def prepare(lyrics, transcription, folder):
    """Return a reviewable generation input, or an actionable validation error."""
    if SECTION.search(lyrics):
        return {'lyrics': lyrics, 'error': None, 'method': 'manual-sections'}
    words = (transcription or {}).get('words', [])
    text = re.findall(r'\w+', lyrics.casefold())
    recognized = re.findall(r'\w+', (transcription or {}).get('text', '').casefold())
    if not words or text != recognized or len(lyrics.split()) != len(words):
        return {'lyrics': '', 'error': 'Save matching word corrections before generating, or add section labels such as [Verse] and [Chorus] to your reviewed lyrics.', 'method': None}
    path = Path(folder) / 'scores/melody/structure.lab'
    if not path.is_file():
        return {'lyrics': '', 'error': 'Song section analysis is missing. Resume preparation or add section labels to your reviewed lyrics.', 'method': None}
    sections = []
    try:
        for row in path.read_text(encoding='utf-8').splitlines():
            start, end, label = row.split()
            start, end = float(start), float(end)
            label = label.lower()
            if label not in TAGS or end <= start or (sections and start < sections[-1][1]):
                raise ValueError('Invalid or unsupported song section')
            sections.append((start, end, label))
        if not sections:
            raise ValueError('No song sections')
        lines, phrase, previous, last_section = [], [], None, None
        def flush():
            if phrase:
                lines.append(' '.join(phrase))
                phrase.clear()
        for token, word in zip(lyrics.split(), words):
            # Assign by onset: a held word crossing a boundary stays in its phrase.
            onset = float(word['start'])
            index = max((i for i, s in enumerate(sections) if s[0] <= onset), default=0)
            label = sections[index][2]
            if label != last_section:
                flush()
                if lines:
                    lines.append('')
                lines.append('[' + TAGS[label] + ']')
                last_section = label
            elif previous is not None and (onset - float(previous['end']) >= .45 or len(phrase) >= 12):
                flush()
            phrase.append(token)
            if token.endswith(('.', '!', '?', ';')):
                flush()
            previous = word
        flush()
        return {'lyrics': '\n'.join(lines), 'error': None, 'method': 'score-sections-and-word-onsets'}
    except (ValueError, KeyError, TypeError) as exc:
        return {'lyrics': '', 'error': f'Cannot prepare sectioned lyrics: {exc}', 'method': None}
