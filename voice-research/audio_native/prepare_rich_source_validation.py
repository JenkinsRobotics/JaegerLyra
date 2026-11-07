"""Fresh evaluation clips selected after the richer-source design was fixed.

Synthetic Kokoro baseline only. No playback or microphone. These clips never
fit the PCA basis; all variants retain the original reference training clips.
"""
import base64,hashlib,json,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/rich-source/fresh-inputs'
LINES=[
    'I almost forgot the little blue umbrella by the door.',
    'Are you ready? Three, two, one, let the music begin!',
]


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    report={'purpose':'Prospective different-sentence validation after source-ceiling/rich-source design; no model retuning or target weight fitting',
        'origin':'Original sentences synthesized by local Kokoro; no human voice recording',
        'clips':[]}
    for speaker in ('af_heart','am_michael'):
        for index,text in enumerate(LINES):
            name=f'{speaker}-{index}';path=OUT/(name+'.wav')
            if not path.exists():
                request=urllib.request.Request('http://127.0.0.1:8791/api/voice',json.dumps({'engine':'expressive','speaker':speaker,'text':text}).encode(),{'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response)
                path.write_bytes(base64.b64decode(result['audio']))
            row={'name':name,'speaker':speaker,'text':text,'file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            report['clips'].append(row)
            (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
