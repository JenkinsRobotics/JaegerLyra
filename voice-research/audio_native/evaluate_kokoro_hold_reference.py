"""Pitch and lexical diagnostics for the pretrained held-vowel comparator."""
import hashlib
import json
import re
import urllib.request
from pathlib import Path
import numpy as np
import soundfile as sf
import parselmouth
import pyworld
from compact_vocal_instrument import RATE
from neural_vowel_instrument import VowelScore
from benchmark_neural_vowels import pitch_metrics
from check_articulation_words import EQUIVALENTS

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-articulation/kokoro-reference-v1'


def main():
    source=json.loads((OUT/'report.json').read_text())
    comparison=json.loads((ROOT/'evidence/neural-articulation/stationarity-v1/report.json').read_text())
    if not source['complete'] or not comparison['complete']:raise ValueError('Incomplete reference')
    report={'scope':'Pretrained Kokoro controlled-vowel reference; not independent neural synthesis or a perceptual acceptance test',
        'expectedCases':18,'results':[],'complete':False,
        'limits':source['limits']+['Single-word ASR can misinterpret long tones; exact-spelling and fixed homophone-class scores are both retained.']}
    for row in source['results']:
        file=ROOT/row['file'];payload=file.read_bytes()
        if hashlib.sha256(payload).hexdigest()!=row['sha256']:raise ValueError('Changed teacher output')
        audio,rate=sf.read(file);assert rate==RATE;score=VowelScore(**row['score'])
        reference=next(r for r in comparison['results'] if r['speaker']==row['speaker'] and r['unit']==row['unit'] and r['mode']==row['mode'])
        start=max(row['vowelStart']+.080,reference['commonAnalysisHoldStartSeconds']+.080)
        end=min(row['vowelEnd']-.030,score.seconds-.050)
        p=parselmouth.Sound(audio,sampling_frequency=RATE).to_pitch_ac(time_step=.01,pitch_floor=50.,pitch_ceiling=900.)
        wf,wt=pyworld.harvest(np.ascontiguousarray(audio),RATE,f0_floor=50.,f0_ceil=900.,frame_period=10.)
        metrics={}
        for name,times,f0 in (('praat',p.xs(),p.selected_array['frequency']),('world',wt,wf)):
            keep=(times>=start)&(times<=end)
            metrics[name]=pitch_metrics(times[keep],f0[keep],score)
            metrics[name]['scope']='Common post-onset interval within the teacher-authored vowel boundaries'
        request=urllib.request.Request('http://127.0.0.1:8791/api/transcribe',payload,{'Content-Type':'audio/wav'})
        with urllib.request.urlopen(request,timeout=60) as response:result=json.load(response)
        if 'text' not in result:raise ValueError('Missing ASR transcript')
        normalized=' '.join(re.findall(r'[a-z]+',result['text'].lower()))
        report['results'].append({**row,**metrics,'analysisStartSeconds':start,'analysisEndSeconds':end,
            'transcript':result['text'],'normalized':normalized,'exactSpellingMatch':normalized==row['unit'],
            'homophoneEquivalentMatch':normalized in EQUIVALENTS[row['unit']],
            'rawPeak':float(max(abs(audio))),'finite':bool(np.isfinite(audio).all())})
        (OUT/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(row['name'],repr(result['text']),normalized in EQUIVALENTS[row['unit']],flush=True)
    report['complete']=len(report['results'])==report['expectedCases']
    (OUT/'evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
