"""Level-match all captured-articulation cases and preserve source guides."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/neural-articulation/v1/audition'


def main():
    capture=json.loads((OUT.parent/'capture-report.json').read_text())
    long=json.loads((OUT.parent/'report.json').read_text())
    short=json.loads((ROOT/'evidence/neural-articulation/short-v1/report.json').read_text())
    if not all(r['complete'] for r in (capture,long,short)):raise ValueError('Incomplete articulation studies')
    OUT.mkdir(parents=True,exist_ok=True);groups=[];decoded={}
    for unit in capture['units']:
        items=[('Original Kokoro word','source',unit['trimmedSource']),
            ('Frozen neural reconstruction','decoded-source',unit['decodedSource'])]
        items += [('Controlled '+r['mode'],r['mode'],r['file']) for r in long['results']+short['results']
            if r['speaker']==unit['speaker'] and r['unit']==unit['unit']]
        loaded=[]
        for label,mode,file in items:
            audio,rate=sf.read(ROOT/file)
            # A fixed 200 ms loudness gate supports the shorter original words.
            lufs=pyln.Meter(rate,block_size=.2).integrated_loudness(audio)
            loaded.append((label,mode,file,audio,rate,lufs))
        target=min(-20.,*(lufs+20*np.log10(.9/max(abs(audio))) for _,_,_,audio,_,lufs in loaded))
        clips=[]
        for label,mode,source,audio,rate,lufs in loaded:
            gain=10**((target-lufs)/20);file=OUT/(unit['name']+'-'+mode+'.wav')
            sf.write(file,audio*gain,rate,subtype='PCM_24');saved,_=sf.read(file)
            actual=pyln.Meter(rate,block_size=.2).integrated_loudness(saved)
            if abs(actual-target)>.01 or max(abs(saved))>.901:raise ValueError('Invalid audition levels')
            decoded[(unit['speaker'],unit['unit'],mode)]=saved
            clips.append({'label':label,'mode':mode,'file':str(file.relative_to(ROOT)),
                'source':source,'sourceSha256':hashlib.sha256((ROOT/source).read_bytes()).hexdigest(),
                'gain':gain,'lufs200MsGate':actual,'samplePeak':float(max(abs(saved)))})
        groups.append({'speaker':unit['speaker'],'unit':unit['unit'],'clips':clips})
    sequences=[]
    for speaker in ('af_heart','am_michael','af_bella'):
        for mode in ('short-native','native','melody'):
            pieces=[]
            for i,unit in enumerate(('me','see','bee')):
                if i:pieces.append(np.zeros(6000))
                pieces.append(decoded[(speaker,unit,mode)])
            file=OUT/(speaker+'-three-syllables-'+mode+'.wav')
            sf.write(file,np.concatenate(pieces),24000,subtype='PCM_24')
            sequences.append({'speaker':speaker,'mode':mode,'file':str(file.relative_to(ROOT)),
                'construction':'Three separately rendered captured units, me / see / bee, with 250 ms silence between; not connected speech or live unit switching'})
    report={'scope':'All nine captured units and all 72 controlled renders; short and long cases retained',
        'processing':'Whole clips, constant gain only, common level per source-unit group; no pitch correction, denoising or limiting. Loudness uses a disclosed 200 ms gate for short words.',
        'latency':'WAVs are aligned offline and do not demonstrate the fixed 20 ms stream delay or device latency.',
        'truePeakMeasured':False,'listeningAccepted':False,'groups':groups,'sequences':sequences}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print('Saved',sum(len(g['clips']) for g in groups),'matched clips and',len(sequences),'labelled concatenations; no playback.')


if __name__=='__main__':main()
