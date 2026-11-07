"""Matched complete clips for the sustain comparison; all development units."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'public/research/sustain-comparison'


def main():
    study=json.loads((ROOT/'evidence/neural-articulation/stationarity-v1/report.json').read_text())
    teacher=json.loads((ROOT/'evidence/neural-articulation/kokoro-reference-v1/report.json').read_text())
    capture=json.loads((ROOT/'evidence/neural-articulation/v1/capture-report.json').read_text())
    if not all(r['complete'] for r in (study,teacher,capture)):raise ValueError('Incomplete comparison')
    OUT.mkdir(parents=True,exist_ok=True);groups=[];files=[]
    for unit in capture['units']:
        rows=[r for r in study['results'] if r['speaker']==unit['speaker'] and r['unit']==unit['unit']]
        references=[r for r in teacher['results'] if r['speaker']==unit['speaker'] and r['unit']==unit['unit']]
        items=[('original','source',unit['trimmedSource'])]
        items += [(r['mode'],r['method'],r['file']) for r in rows]
        items += [(r['mode'],'kokoro',r['file']) for r in references]
        loaded=[]
        for mode,method,path in items:
            audio,rate=sf.read(ROOT/path);lufs=pyln.Meter(rate,block_size=.2).integrated_loudness(audio)
            loaded.append((mode,method,path,audio,rate,lufs))
        target=min(-20.,*(lufs+20*np.log10(.9/max(abs(audio))) for _,_,_,audio,_,lufs in loaded))
        packed={}
        for mode,method,path,audio,rate,lufs in loaded:
            gain=10**((target-lufs)/20);file=OUT/(unit['name']+'-'+mode+'-'+method+'.wav')
            sf.write(file,audio*gain,rate,subtype='PCM_24');saved,_=sf.read(file)
            actual=pyln.Meter(rate,block_size=.2).integrated_loudness(saved)
            if abs(actual-target)>.01 or max(abs(saved))>.901:raise ValueError('Invalid comparison level')
            record={'file':file.name,'source':path,'sourceSha256':hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
                'gain':gain,'lufs200MsGate':actual,'samplePeak':float(max(abs(saved))),'mode':mode,'method':method}
            packed[(mode,method)]=record;files.append(record)
        for mode,title in (('native','Held note'),('melody','Melody'),('short-native','Short word')):
            reference=packed[('original','source')]
            groups.append({'speaker':unit['speaker'],'word':unit['unit'],'mode':mode,'title':title,
                'loop':packed[(mode,'baseline-loop')],'steady':packed[(mode,'stationary-fixed')],
                'reference':reference,'controlReference':None if mode=='short-native' else packed[(mode,'kokoro')],
                'fixedOriginal':packed[(mode,'fixed-original')],'stationaryLoop':packed[(mode,'stationary-loop')],
                'referenceTitle':'Original Kokoro speech',
                'referenceDescription':'The original short word, with its own pitch and timing. Use this as the normal speech-quality guide.'})
    manifest={'scope':'All nine development units, all three study modes, all sustain ablations and pretrained Kokoro references',
        'processing':'Whole clips with constant gain; common integrated loudness per source unit using a disclosed 200 ms gate. No limiting, denoising or pitch correction.',
        'labels':'Open labels, not a blinded or independent listening study','truePeakMeasured':False,
        'listeningAccepted':False,'groups':groups,'files':files}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('Saved',len(files),'matched clips in',len(groups),'comparison groups; no playback.')


if __name__=='__main__':main()
