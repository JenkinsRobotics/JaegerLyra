"""Matched, whole-phrase performance controls; no playback or source replacement."""
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'evidence/neural-performance/v2'
OUT=BASE/'audition'


def main():
    report=json.loads((BASE/'report.json').read_text())
    if not report['complete']:raise ValueError('Performance evaluation incomplete')
    fixture=report['originals'][0]['fixture']
    rows=[r for r in report['results'] if r['fixture']==fixture]
    if len(rows)!=7 or any(r['status']!='ok' for r in rows):raise ValueError('Do not hide a failed showcase case')
    loaded=[]
    for row in rows:
        audio,rate=sf.read(ROOT/row['file']);lufs=pyln.Meter(rate).integrated_loudness(audio)
        loaded.append((row,audio,rate,lufs))
    target=min(-20.,*(lufs+20*np.log10(.9/max(abs(audio))) for _,audio,_,lufs in loaded))
    OUT.mkdir(exist_ok=True);files=[]
    for row,audio,rate,lufs in loaded:
        gain=10**((target-lufs)/20);path=OUT/(row['mode']+'.wav')
        sf.write(path,audio*gain,rate,subtype='PCM_24');saved,_=sf.read(path)
        measured=pyln.Meter(rate).integrated_loudness(saved)
        if abs(measured-target)>.01 or max(abs(saved))>.901:raise ValueError('Audition gain validation failed')
        files.append({'mode':row['mode'],'controls':row['controls'],'source':row['file'],
            'file':str(path.relative_to(ROOT)),'outputLufs':measured,'gain':gain,
            'samplePeak':float(max(abs(saved))),'durationSeconds':len(saved)/rate})
    manifest={'scope':'Captured articulation with authored output controls; not independent lyrics or cloned identity',
        'selection':'First predefined Heart fixture, every tested control; whole clips, no excerpt selection',
        'fixture':fixture,'method':'One constant gain per complete clip to matched BS.1770 loudness; no pitch correction, denoising or limiting',
        'targetLufs':target,'truePeakMeasured':False,'files':files}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps({'files':len(files),'folder':str(OUT)}))


if __name__=='__main__':main()
