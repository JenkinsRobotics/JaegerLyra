"""Original synthetic training utterances for a neural decoder capacity study.

Kokoro is an explicit teacher. These are not independently sourced training
weights/data or a clean-room corpus. The decoder itself trains only on PCM.
"""
import base64,hashlib,json,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-waveform/training-inputs'
LINES=[
    'A quiet morning gives us time to notice the small things.',
    'The bright red bicycle waited beside the garden gate.',
    'Would you like another cup of tea before we leave?',
    'I found a smooth white stone near the edge of the river.',
    'Fresh bread and warm soup were ready on the wooden table.',
    'The wind pushed the curtains gently into the empty room.',
    'We can start slowly and make the next step together.',
    'Three little birds flew past the window after sunrise.',
    'That sounds wonderful, but let me check the time first.',
    'Please bring the green notebook and a pencil from upstairs.',
    'A tiny silver bell rang twice behind the closed door.',
    'Some days are difficult, and a short rest can help.',
]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Teacher-generated audio for neural waveform reconstruction research; no text is supplied to the trained decoder',
        'teacher':'Existing local Kokoro, Heart and Michael voices',
        'provenance':'Synthetic teacher imitation, not independent corpus or clean-room weights',
        'heldout':'All existing source-detail evaluation clips, including Bella and human readers, excluded from fitting',
        'clips':[]}
    for speaker in ('af_heart','am_michael'):
        for index,text in enumerate(LINES):
            name=f'{speaker}-{index:02d}';path=OUT/(name+'.wav')
            if not path.exists():
                request=urllib.request.Request('http://127.0.0.1:8791/api/voice',
                    json.dumps({'engine':'expressive','speaker':speaker,'text':text}).encode(),
                    {'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response)
                path.write_bytes(base64.b64decode(result['audio']))
            report['clips'].append({'name':name,'speaker':speaker,'text':text,
                'file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
            (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
            print(name,flush=True)


if __name__=='__main__':main()
