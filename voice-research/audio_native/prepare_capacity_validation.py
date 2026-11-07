"""Fresh output comparisons only after the capacity candidate has been frozen."""
import base64,hashlib,json,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-waveform/capacity-validation'
LINES=[
    'Please switch off the kitchen fan before you close the screen door.',
    'The crisp morning breeze brushed through the branches beside the porch.',
    'Wait, was that your idea? I absolutely love it!',
]


def main():
    folder=ROOT/'evidence/neural-waveform/capacity-v3'
    development=json.loads((folder/'development.json').read_text())
    if not development['complete']:raise ValueError('Finish selection before creating final recordings')
    row=next(r for r in development['results'] if r['name']==development['selected'])
    candidate=folder/(row['name']+'.npz')
    digest=hashlib.sha256(candidate.read_bytes()).hexdigest()
    if digest!=row['modelSha256']:raise ValueError('Frozen candidate changed')
    OUT.mkdir(exist_ok=True,parents=True)
    report={'scope':'Reconstruction fidelity from target-derived controls; no independent new speech',
        'candidate':str(candidate.relative_to(ROOT)),'candidateSha256':digest,'expectedClips':12,
        'evaluationDescription':'Nine fresh Kokoro recordings generated after capacity-v3 selection; three existing SoulX songs and four human clips are explicitly reused regressions, excluded from model fitting',
        'teacher':'Synthetic Kokoro evaluation; existing pretrained SoulX songs; not independent model-training provenance',
        'controlScalarsPerSecond':row['controlScalarsPerSecond'],'clips':[]}
    for speaker in ('af_heart','am_michael','af_bella'):
        for index,line in enumerate(LINES):
            name=f'{speaker}-capacity-{index}';path=OUT/(name+'.wav')
            if not path.exists():
                request=urllib.request.Request('http://127.0.0.1:8791/api/voice',
                    json.dumps({'engine':'expressive','speaker':speaker,'text':line}).encode(),
                    {'Content-Type':'application/json'})
                with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response)
                path.write_bytes(base64.b64decode(result['audio']))
            report['clips'].append({'name':name,'kind':'fresh-kokoro','file':str(path.relative_to(ROOT)),
                'speaker':speaker,'text':line,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
                'reference':f'evidence/voice-fixtures/{speaker}-reference.wav' if speaker!='af_bella'
                    else 'evidence/source-detail/fresh-inputs/af_bella-0.wav'})
            (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
            print(name,flush=True)
    songs=json.loads((ROOT/'evidence/integrated-singer/report.json').read_text())
    for song in songs:
        path=ROOT/'evidence/integrated-singer'/(song['name']+'.wav')
        report['clips'].append({'name':'sung-'+song['name'],'kind':'sung-regression',
            'file':str(path.relative_to(ROOT)),'speaker':song['speaker'],'text':song['request']['lyrics'],
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'reference':f"evidence/voice-fixtures/{song['speaker']}-reference.wav",
            'provenance':'Reused pretrained SoulX output; source singing identity/naturalness already unaccepted'})
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('READY',len(report['clips']),flush=True)


if __name__=='__main__':main()
