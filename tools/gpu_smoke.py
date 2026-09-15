"""Exercise real YuE2 generation and validate the returned WAV (stdlib only)."""
import argparse
import array
import base64
import io
import json
from pathlib import Path
import time
import urllib.request
import urllib.error
import wave


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8181')
    parser.add_argument('--output', type=Path, default=Path('.verification/gpu-smoke.wav'))
    parser.add_argument('--max-tokens', type=int)
    args = parser.parse_args()
    health = json.load(urllib.request.urlopen(args.url + '/health', timeout=10))
    print('Engine:', health, flush=True)
    assert health.get('backend') in ('vulkan', 'cuda'), 'Test requires a GPU backend'
    body = {'model': 'yue2', 'request': {
        'lyrics': '[Verse]\nLight across the water.\nWe are coming home.',
        'seed': 42, 'num_inference_steps': 4,
        'options': {'style': 'English, indie pop, acoustic guitar, soft drums', 'cot': 'off'},
    }}
    if args.max_tokens:
        body['request']['options']['semantic_max_tokens'] = args.max_tokens
    started = time.monotonic()
    request = urllib.request.Request(args.url + '/v1/tasks/run',
                                     data=json.dumps(body).encode(),
                                     headers={'Content-Type': 'application/json'})
    try:
        response = json.load(urllib.request.urlopen(request, timeout=1800))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f'Engine HTTP {exc.code}: {exc.read().decode()}') from exc
    audio = base64.b64decode(response['audio'], validate=True)
    with wave.open(io.BytesIO(audio)) as wav:
        duration = wav.getnframes() / wav.getframerate()
        assert duration > 1, 'Output must contain more than one second of audio'
        assert wav.getnchannels() == 2 and wav.getframerate() == 48000, 'Expected 48 kHz stereo'
        assert wav.getsampwidth() == 2, 'Expected PCM16 audio'
        samples = array.array('h', wav.readframes(wav.getnframes()))
        assert any(samples), 'Output is entirely silent'
        print('WAV:', wav.getnchannels(), 'channels,', wav.getframerate(), 'Hz,', duration, 'seconds')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(audio)
    print('PASS:', args.output, len(audio), 'bytes;', round(time.monotonic() - started, 1), 'seconds elapsed')


if __name__ == '__main__':
    main()
