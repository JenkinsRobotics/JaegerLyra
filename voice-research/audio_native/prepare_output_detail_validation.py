"""Fresh consonant-rich speech and unseen existing sung output for frozen v2."""
import base64,hashlib,json,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-waveform/output-detail-validation'
LINES=[
    'Sasha chose a soft blue scarf and placed her glasses on the shelf.',
    'Bright sparks flickered briefly before the little fire settled.',
    "Really? Oh, that is fantastic! I was worried we had missed it.",
]


def main():
    folder=ROOT/'evidence/neural-waveform/axis-detail-v2'
    development=json.loads((folder/'development.json').read_text())
    if not development['complete']:raise ValueError('Freeze output candidate before fresh evaluation')
    candidate=folder/(development['selected']+'.npz')
    OUT.mkdir(exist_ok=True,parents=True)
    report={'scope':'Output reconstruction, not new-speech or singing composition',
        'candidate':str(candidate.relative_to(ROOT)),
        'candidateSha256':hashlib.sha256(candidate.read_bytes()).hexdigest(),'expectedClips':12,
        'evaluationDescription':'Nine fresh Kokoro recordings after v2 selection; three existing SoulX songs unseen in neural fitting/selection; human panel is reused regression',
        'teacher':'Synthetic Kokoro speech; separate existing pretrained SoulX singing sources; not independent model provenance',
        'clips':[]}
    for speaker in ('af_heart','am_michael','af_bella'):
        for index,line in enumerate(LINES):
            name=f'{speaker}-detail-{index}';path=OUT/(name+'.wav')
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
    singing=json.loads((ROOT/'evidence/integrated-singer/report.json').read_text())
    for song in singing:
        path=ROOT/'evidence/integrated-singer'/(song['name']+'.wav')
        report['clips'].append({'name':'sung-'+song['name'],'kind':'sung-heldout',
            'file':str(path.relative_to(ROOT)),'speaker':song['speaker'],'text':song['request']['lyrics'],
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'reference':f"evidence/voice-fixtures/{song['speaker']}-reference.wav",
            'provenance':'Existing pretrained SoulX output; comparison assesses preservation, not generated melody or cloned identity'})
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('READY',len(report['clips']),flush=True)


if __name__=='__main__':main()
