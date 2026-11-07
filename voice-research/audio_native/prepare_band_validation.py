"""Fresh speech/song outputs plus explicitly reused SoulX regressions."""
import base64,hashlib,json,urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-waveform/band-validation'
LINES=[
    'The silver wind chimes shifted softly as the back door opened.',
    'She packed six peaches, fresh bread, and a small jar of raspberry jam.',
    "Oh, that's wonderful news! Shall we celebrate after dinner?",
]


def main():
    development=json.loads((ROOT/'evidence/neural-waveform/band-v4/development.json').read_text())
    candidate=ROOT/development['candidate']
    digest=hashlib.sha256(candidate.read_bytes()).hexdigest()
    if not development['complete'] or digest!=development['candidateSha256']:
        raise ValueError('Freeze the decoder assembly before fresh evaluation')
    songs=json.loads((OUT/'singing-manifest.json').read_text())
    if not songs['complete'] or len(songs['clips'])!=3 or songs['candidateSha256']!=digest:
        raise ValueError('Finish fresh teacher songs under this frozen assembly')
    report={'scope':'Waveform reconstruction from target-derived controls; output fidelity, not new speech composition',
        'candidate':development['candidate'],'candidateSha256':digest,'candidateType':'band-preserving-neural',
        'expectedClips':15,'controlScalarsPerSecond':44900,
        'evaluationDescription':'Nine fresh Kokoro speech clips and three fresh score-adapter songs generated after band-v4 freeze; three prior SoulX songs and the human panel are explicitly reused regressions',
        'teacher':'Synthetic Kokoro and score-adapter outputs; reused pretrained SoulX material; not independent training provenance',
        'clips':[]}
    for speaker in ('af_heart','am_michael','af_bella'):
        for index,line in enumerate(LINES):
            name=f'{speaker}-band-{index}';path=OUT/(name+'.wav')
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
    report['clips'].extend(songs['clips'])
    prior=json.loads((ROOT/'evidence/integrated-singer/report.json').read_text())
    for song in prior:
        path=ROOT/'evidence/integrated-singer'/(song['name']+'.wav')
        report['clips'].append({'name':'reused-sung-'+song['name'],'kind':'sung-regression',
            'file':str(path.relative_to(ROOT)),'speaker':song['speaker'],'text':song['request']['lyrics'],
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'reference':f"evidence/voice-fixtures/{song['speaker']}-reference.wav",
            'provenance':'Reused SoulX output including the clip exposing the capacity-v3 lyric regression; not fresh validation'})
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print('READY',len(report['clips']),flush=True)


if __name__=='__main__':main()
