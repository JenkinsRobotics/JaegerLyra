"""Fresh output-only teacher comparisons, generated after candidate selection."""
import base64,hashlib,json,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-waveform/output-validation'
LINES=[
    'Soft rain tapped against the glass while Lucy folded a yellow scarf.',
    'Could you move those five boxes closer to the bookshelf, please?',
    'Oh, I thought you were joking! That is a lovely surprise.',
]


def main():
    development=json.loads((ROOT/'evidence/neural-waveform/axis-v1/development.json').read_text())
    if not development['complete']:raise ValueError('Select candidate before creating final evaluation')
    candidate=ROOT/'evidence/neural-waveform/axis-v1'/(development['selected']+'.npz')
    OUT.mkdir(exist_ok=True,parents=True)
    report={'scope':'Fresh synthetic output-fidelity evaluation; never used to fit or select this candidate',
        'candidate':str(candidate.relative_to(ROOT)),
        'candidateSha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),
        'teacher':'Existing Kokoro, synthetic research audio; not independent training provenance',
        'clips':[]}
    for speaker in ('af_heart','am_michael','af_bella'):
        for index,line in enumerate(LINES):
            name=f'{speaker}-{index}';path=OUT/(name+'.wav')
            if not path.exists():
                request=urllib.request.Request('http://127.0.0.1:8791/api/voice',
                    json.dumps({'engine':'expressive','speaker':speaker,'text':line}).encode(),
                    {'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response)
                path.write_bytes(base64.b64decode(result['audio']))
            report['clips'].append({'name':name,'file':str(path.relative_to(ROOT)),
                'speaker':speaker,'text':line,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'reference':f'evidence/voice-fixtures/{speaker}-reference.wav' if speaker!='af_bella'
                    else 'evidence/source-detail/fresh-inputs/af_bella-0.wav'})
            (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
            print(name,flush=True)


if __name__=='__main__':main()
