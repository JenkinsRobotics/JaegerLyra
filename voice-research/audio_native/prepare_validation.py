"""New synthetic validation utterances, chosen after the first feature experiment.

No tuning is performed on these clips. Uses the existing local Kokoro endpoint;
does not play audio. They are evaluation fixtures, not independent-engine data.
"""
import base64
import hashlib
import json
from pathlib import Path
import time
import urllib.request
import numpy as np
import soundfile as sf
from streaming_voicebox import process

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'evidence/streaming-voicebox/validation'
LINES = [
    'Wait, did you really hear that? I thought the room was empty.',
    'Slowly now. Take a breath, and let the last note fade away.',
]


def main():
    OUT.mkdir(exist_ok=True)
    manifest={'purpose':'Post-experiment validation without changing features, model architecture, learning rates, seeds or training clips',
        'training':'Only the original Heart and Michael reference clips; no validation training before evaluation',
        'novelVoice':'af_bella', 'origin':'Original text, synthetic local Kokoro output; no human recording or microphone', 'clips':[]}
    for speaker in ('af_heart','am_michael','af_bella'):
        for index,line in enumerate(LINES):
            name=f'{speaker}-{index}'
            path=OUT/(name+'.wav')
            if not path.exists():
                request=urllib.request.Request('http://127.0.0.1:8791/api/voice',json.dumps({'engine':'expressive','speaker':speaker,'text':line}).encode(),{'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response)
                path.write_bytes(base64.b64decode(result['audio']))
            audio, rate=sf.read(path)
            assert rate==24000
            started=time.perf_counter()
            output,frames=process(audio)
            exact=bool(np.array_equal(np.rint(output*32768).astype(np.int16),np.rint(audio*32768).astype(np.int16)))
            assert exact
            np.savez(OUT/(name+'-features.npz'),features=np.array([f.features() for f in frames]),
                     available_at=np.array([f.available_at for f in frames]),f0=np.array([f.f0 for f in frames]))
            row={'name':name,'speaker':speaker,'text':line,'file':str(path.relative_to(ROOT)),
                 'featureFile':name+'-features.npz','sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                 'duration':len(audio)/rate,'pcm16Identical':exact,'processingSeconds':time.perf_counter()-started}
            manifest['clips'].append(row)
            (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
            print(json.dumps(row),flush=True)


if __name__=='__main__':
    main()
