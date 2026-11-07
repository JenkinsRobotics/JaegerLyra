"""Local ASR diagnostic for source, reconstructed and controlled syllables.

Homophone equivalence is declared before evaluation because these are sounds,
not a spelling task. Raw transcripts and exact-spelling matches are retained.
No pronunciation, identity or naturalness acceptance follows from ASR alone.
"""
import hashlib
import json
import re
import urllib.request
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-articulation/v1'
EQUIVALENTS={'me':{'me'},'see':{'see','sea','c'},'bee':{'bee','be','b'}}


def main(out=OUT,reuse_sources=False):
    capture=json.loads((OUT/'capture-report.json').read_text())
    renders=json.loads((out/'report.json').read_text())
    if not capture['complete'] or not renders['complete']:raise ValueError('Incomplete articulation panel')
    cases=[];reused=[]
    for unit in capture['units']:
        for kind,key in (('source','trimmedSource'),('decoded-source','decodedSource')):
            cases.append({'name':unit['name']+'-'+kind,'speaker':unit['speaker'],'unit':unit['unit'],
                'mode':kind,'file':unit[key]})
    if reuse_sources:
        prior=json.loads((OUT/'word-report.json').read_text())
        if not prior['complete']:raise ValueError('Incomplete source ASR study')
        for case in cases:
            row=next(r for r in prior['results'] if r['name']==case['name'])
            if hashlib.sha256((ROOT/case['file']).read_bytes()).hexdigest()!=row['sha256']:
                raise ValueError('Source changed before ASR reuse')
            reused.append(row)
        cases=[]
    cases += [{k:r[k] for k in ('name','speaker','unit','mode','file')} for r in renders['results']]
    report={'scope':'Local Whisper lexical diagnostic on original/reconstructed short words and captured onset-to-sustain outputs',
        'expectedCases':len(cases)+len(reused),'reusedSourceCases':len(reused),'equivalents':{k:sorted(v) for k,v in EQUIVALENTS.items()},
        'model':'Existing local /api/transcribe English Whisper endpoint, beam_size=1 and VAD enabled',
        'results':list(reused),'complete':False,
        'limits':['Single-word ASR is sensitive to duration and VAD; raw errors are not human judgments.',
            'Homophone matches assess the intended sound class, not spelling.',
            'No listening or naturalness acceptance is established.']}
    for case in cases:
        file=ROOT/case['file'];payload=file.read_bytes()
        request=urllib.request.Request('http://127.0.0.1:8791/api/transcribe',payload,{'Content-Type':'audio/wav'})
        with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response)
        if 'text' not in result:raise ValueError('Missing ASR transcript: '+str(result))
        words=re.findall(r"[a-z]+",result['text'].lower());normalized=' '.join(words)
        row={**case,'sha256':hashlib.sha256(payload).hexdigest(),'transcript':result['text'],
            'normalized':normalized,'exactSpellingMatch':normalized==case['unit'],
            'homophoneEquivalentMatch':normalized in EQUIVALENTS[case['unit']]}
        report['results'].append(row)
        (out/'word-report.json').write_text(json.dumps(report,indent=2)+'\n')
        print(case['name'],repr(result['text']),row['homophoneEquivalentMatch'],flush=True)
    report['complete']=len(report['results'])==report['expectedCases']
    (out/'word-report.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':
    import sys
    if '--short' in sys.argv:main(out=ROOT/'evidence/neural-articulation/short-v1',reuse_sources=True)
    else:main()
