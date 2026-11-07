"""Exercise the actual optional singing API without playing through speakers."""
import base64
import json
from pathlib import Path
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'evidence/integrated-singer'
BASE = 'http://127.0.0.1:8791'


def call(path, data=None):
    request = urllib.request.Request(BASE+path,
        None if data is None else json.dumps(data).encode(),
        {} if data is None else {'Content-Type': 'application/json'})
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def render(name, request):
    started = time.perf_counter();job = call('/api/song', request)
    phases = [];last = None
    try:
        while time.perf_counter()-started < 240:
            data = call('/api/song?id='+job['id'])
            phase = data.get('phase', data['status'])
            if phase != last:
                phases.append({'phase': phase, 'seconds': round(time.perf_counter()-started, 3)})
                print(name, phases[-1], flush=True);last=phase
            if data['status'] == 'error':
                raise RuntimeError(data['error'])
            if data['status'] == 'ready':
                result = data['result'];raw = base64.b64decode(result.pop('audio'))
                result.pop('frames', None)
                result.update(request=request, id=job['id'], phases=phases,
                              wallSeconds=round(time.perf_counter()-started, 3))
                (OUT/(name+'.wav')).write_bytes(raw)
                (OUT/(name+'.json')).write_text(json.dumps(result, indent=2))
                transcription = urllib.request.Request(BASE+'/api/transcribe', raw, {'Content-Type': 'audio/wav'})
                with urllib.request.urlopen(transcription, timeout=60) as response:
                    result['asr'] = json.load(response)
                (OUT/(name+'.json')).write_text(json.dumps(result, indent=2))
                return {k:v for k,v in result.items() if k!='score'}
            time.sleep(.5)
        raise TimeoutError('Singing test deadline exceeded')
    finally:
        call('/api/song-cancel', {'id': job['id']})


def main():
    OUT.mkdir(exist_ok=True)
    cases = [
        ('heart-spoken', {'lyrics': 'Little lights are shining\nHere with you tonight',
            'speaker': 'af_heart', 'referenceKind': 'speech', 'tempo': 104, 'melody': 'glow', 'transpose': 0}),
        ('heart-sung', {'lyrics': 'We can hear the ocean\nCalling us back home',
            'speaker': 'af_heart', 'referenceKind': 'sung', 'tempo': 120, 'melody': 'lullaby', 'transpose': 5}),
        ('michael-spoken', {'lyrics': 'Keep a little spark inside\nLet it shine for you',
            'speaker': 'am_michael', 'referenceKind': 'speech', 'tempo': 88, 'melody': 'glow', 'transpose': -12}),
    ]
    rows = []
    for name, request in cases:
        try:
            result = render(name, {'engine': 'soulx', **request})
            rows.append({'name': name, **result})
        except Exception as error:
            rows.append({'name': name, 'request': request, 'error': str(error)})
        (OUT/'report.json').write_text(json.dumps(rows, indent=2))
        print(json.dumps(rows[-1]), flush=True)


if __name__ == '__main__':
    main()
