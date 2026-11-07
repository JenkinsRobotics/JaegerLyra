"""Fixed synthetic vowel references for output-instrument research; no playback."""
import base64
import hashlib
import json
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/neural-vowels/references'


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    model = ROOT / 'evidence/neural-waveform/axis-detail-v2/projected-r1e-7.npz'
    report = {'scope': 'Synthetic Kokoro sound references; intended vowel labels require listening confirmation',
              'candidate': str(model.relative_to(ROOT)),
              'candidateSha256': hashlib.sha256(model.read_bytes()).hexdigest(),
              'source': 'Existing Kokoro English speech API; source-derived teacher audio, not independently sourced training',
              'clips': [], 'complete': False}
    for speaker in ('af_heart', 'am_michael', 'af_bella'):
        for label, text in (('ah', 'Ah, ah, ah.'), ('ee', 'Eee, eee, eee.'), ('oo', 'Ooh, ooh, ooh.')):
            name = speaker + '-' + label
            path = OUT / (name + '.wav')
            if not path.exists():
                request = urllib.request.Request('http://127.0.0.1:8791/api/voice',
                    json.dumps({'engine': 'expressive', 'speaker': speaker, 'text': text}).encode(),
                    {'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=60) as response:
                    result = json.load(response)
                path.write_bytes(base64.b64decode(result['audio']))
            report['clips'].append({'name': name, 'speaker': speaker, 'intendedVowel': label,
                'text': text, 'file': str(path.relative_to(ROOT)),
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
            (OUT / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')
            print(name, flush=True)
    report['complete'] = True
    (OUT / 'manifest.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
