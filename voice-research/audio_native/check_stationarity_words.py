"""Reuse baseline transcripts by hash; evaluate all new sustain ablations."""
import hashlib
import json
import re
import urllib.request
from pathlib import Path
from check_articulation_words import EQUIVALENTS

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-articulation/stationarity-v1'


def main():
    study=json.loads((OUT/'report.json').read_text())
    original=json.loads((ROOT/'evidence/neural-articulation/v1/word-report.json').read_text())
    short=json.loads((ROOT/'evidence/neural-articulation/short-v1/word-report.json').read_text())
    if not all(r['complete'] for r in (study,original,short)):raise ValueError('Incomplete input study')
    lookup={r['file']:r for r in original['results']+short['results']}
    report={'scope':'Development-panel local Whisper diagnostic for original/fixed/reselected sustains',
        'expectedCases':study['expectedCases'],'expectedNewTranscriptions':study['expectedNewRenders'],
        'results':[],'complete':False,'equivalents':{k:sorted(v) for k,v in EQUIVALENTS.items()},
        'limits':original['limits']+['The same nine development source words are reused; no held-out pronunciation claim.']}
    def persist():(OUT/'word-report.json').write_text(json.dumps(report,indent=2)+'\n')
    for case in study['results']:
        payload=(ROOT/case['file']).read_bytes();sha=hashlib.sha256(payload).hexdigest()
        if case['audioReused']:
            old=lookup[case['file']]
            if sha!=old['sha256']:raise ValueError('Changed baseline audio')
            transcript=old['transcript']
        else:
            request=urllib.request.Request('http://127.0.0.1:8791/api/transcribe',payload,{'Content-Type':'audio/wav'})
            with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response)
            if 'text' not in result:raise ValueError('Missing local transcript')
            transcript=result['text']
        normalized=' '.join(re.findall(r'[a-z]+',transcript.lower()))
        report['results'].append({k:case[k] for k in ('name','speaker','unit','mode','method','file')}
            | {'sha256':sha,'transcript':transcript,'normalized':normalized,'transcriptReused':case['audioReused'],
                'exactSpellingMatch':normalized==case['unit'],
                'homophoneEquivalentMatch':normalized in EQUIVALENTS[case['unit']]})
        persist();print(case['name'],repr(transcript),normalized in EQUIVALENTS[case['unit']],flush=True)
    report['complete']=len(report['results'])==report['expectedCases'] and sum(not r['transcriptReused'] for r in report['results'])==report['expectedNewTranscriptions']
    persist()


if __name__=='__main__':main()
