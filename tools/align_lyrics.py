"""Refine existing English word timing using wav2vec2 CTC acoustic alignment."""
import argparse
import faulthandler
import json
from pathlib import Path
import re


def groups(words):
    group = []
    for i, word in enumerate(words):
        if group and (word['end'] - words[group[0]]['start'] > 20 or
                      word['start'] - words[group[-1]]['end'] > 2):
            yield group
            group = []
        group.append(i)
    if group:
        yield group


def align(audio, original, models):
    print('Loading acoustic alignment runtime', flush=True)
    import torch
    import torchaudio
    import soundfile as sf
    print('Loading wav2vec2 alignment model', flush=True)
    torch.set_num_threads(8)
    bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
    # Avoid allocating and initializing a second copy of this 360 MB model.
    # Bundle internals are pinned to the torchaudio 2.5.1 Demucs environment.
    with torch.device('meta'):
        model = torchaudio.models.wav2vec2_model(**bundle._params)
    checkpoint = Path(models) / bundle._path
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    if not checkpoint.is_file():
        temporary = checkpoint.with_suffix('.download')
        torch.hub.download_url_to_file('https://download.pytorch.org/torchaudio/models/' + bundle._path, str(temporary))
        temporary.replace(checkpoint)
    state = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
    # Match the bundle's removal of unused fairseq output labels.
    for name in ('aux.weight', 'aux.bias'):
        state[name] = torch.stack([row for i, row in enumerate(state[name]) if i not in bundle._remove_aux_axis])
    model.load_state_dict(state, assign=True)
    del state
    model.eval()
    print('Reading isolated vocals', flush=True)
    dictionary = {c: i for i, c in enumerate(bundle.get_labels())}
    samples, rate = sf.read(audio, dtype='float32', always_2d=True)
    waveform = torch.from_numpy(samples.mean(axis=1)).unsqueeze(0)
    waveform = torchaudio.functional.resample(waveform, rate, bundle.sample_rate)
    rate = bundle.sample_rate
    duration = waveform.shape[-1] / rate
    words = [dict(w, original_start=w['start'], original_end=w['end'],
                  alignment_status='review', alignment_score=None) for w in original['words']]
    for group in groups(words):
        print(f'Aligning words {group[0] + 1}-{group[-1] + 1}', flush=True)
        left = max(0, words[group[0]]['start'] - 0.75)
        right = min(duration, words[group[-1]]['end'] + 0.75)
        if group[0]:
            left = max(left, words[group[0] - 1]['original_end'])
        if group[-1] + 1 < len(words):
            right = min(right, words[group[-1] + 1]['original_start'])
        clean = [re.sub("[^A-Z']", '', words[i]['text'].upper()) for i in group]
        if any(not word for word in clean) or any(re.search(r'\d', words[i]['text']) for i in group) or right <= left or right - left > 30:
            continue
        tokens = []
        ranges = []
        for word in clean:
            begin = len(tokens)
            tokens.extend(dictionary[c] for c in word)
            ranges.append((begin, len(tokens)))
            tokens.append(dictionary['|'])
        tokens.pop()
        clip = waveform[:, int(left * rate):int(right * rate)]
        with torch.inference_mode():
            emissions, _ = model(clip)
            emissions = emissions.log_softmax(-1)
            try:
                path, scores = torchaudio.functional.forced_align(emissions,
                    torch.tensor([tokens], dtype=torch.int32), blank=0)
            except RuntimeError:
                continue
        spans = torchaudio.functional.merge_tokens(path[0], scores[0].exp())
        if len(spans) != len(tokens):
            continue
        scale = clip.shape[-1] / rate / emissions.shape[1]
        for index, (begin, end) in zip(group, ranges):
            chars = spans[begin:end]
            score = sum(s.score * (s.end - s.start) for s in chars) / sum(s.end - s.start for s in chars)
            start = left + chars[0].start * scale
            finish = left + chars[-1].end * scale
            word = words[index]
            word['alignment_score'] = round(float(score), 4)
            if score >= 0.15 and finish > start and abs(start - word['original_start']) <= 1.5 and abs(finish - word['original_end']) <= 1.5:
                word.update(start=round(start, 3), end=round(finish, 3), alignment_status='aligned')
    # Never let an accepted boundary overlap an unchanged neighbor.
    review_overlaps(words)
    return {**original, 'words': words, 'alignment_model': 'WAV2VEC2_ASR_BASE_960H',
            'alignment_method': 'ctc-forced-alignment-v1',
            'review_words': sum(w['alignment_status'] != 'aligned' for w in words)}


def review_overlaps(words):
    for _ in range(len(words)):
        changed = False
        for i in range(1, len(words)):
            if words[i]['start'] < words[i-1]['end']:
                for word in (words[i-1], words[i]):
                    changed |= word['alignment_status'] == 'aligned'
                    word.update(start=word['original_start'], end=word['original_end'], alignment_status='review')
        if not changed:
            break


def main():
    faulthandler.enable()
    parser = argparse.ArgumentParser()
    parser.add_argument('audio')
    parser.add_argument('transcript')
    parser.add_argument('output')
    parser.add_argument('--models', required=True)
    args = parser.parse_args()
    original = json.loads(Path(args.transcript).read_text(encoding='utf-8'))
    result = align(args.audio, original, args.models)
    output = Path(args.output)
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2), encoding='utf-8')
    temporary.replace(output)


if __name__ == '__main__':
    main()
