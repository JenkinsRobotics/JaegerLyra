"""Whole-clip, level-matched live/planned vowel comparisons; no playback."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/live-neural-grains/v1/audition'


def main():
    report=json.loads((OUT.parent/'report.json').read_text())
    sources=json.loads((ROOT/'evidence/neural-vowels/references/manifest.json').read_text())
    if not report['complete']:raise ValueError('Incomplete live study')
    OUT.mkdir(parents=True,exist_ok=True);groups=[]
    def package(items):
        loaded=[]
        for label,name,path in items:
            audio,rate=sf.read(ROOT/path)
            loudness=pyln.Meter(rate).integrated_loudness(audio)
            loaded.append((label,name,path,audio,rate,loudness))
        target=min(-20.,*(lufs+20*np.log10(.9/max(abs(audio))) for _,_,_,audio,_,lufs in loaded))
        result=[]
        for label,name,path,audio,rate,lufs in loaded:
            gain=10**((target-lufs)/20);file=OUT/(name+'.wav')
            sf.write(file,audio*gain,rate,subtype='PCM_24')
            saved,_=sf.read(file);actual=pyln.Meter(rate).integrated_loudness(saved)
            if abs(actual-target)>.01 or max(abs(saved))>.901:raise ValueError('Invalid audition levels')
            result.append({'label':label,'file':str(file.relative_to(ROOT)),
                'source':path,'sourceSha256':hashlib.sha256((ROOT/path).read_bytes()).hexdigest(),
                'gain':gain,'lufs':actual,'samplePeak':float(max(abs(saved)))})
        return result
    for speaker in ('af_heart','am_michael','af_bella'):
        for kind,mode in (('held','native'),('transition','melody')):
            row=next(r for r in report['results'] if r['speaker']==speaker and r['kind']==kind
                and r['mode']==mode and (kind=='transition' or r['vowel']=='ah'))
            name=speaker+'-'+('held' if kind=='held' else 'melody')
            groups.append({'speaker':speaker,'kind':kind,'mode':mode,
                'clips':package([('Immediate-control neural grains',name+'-live',row['file']),
                    ('Earlier complete-score neural grains',name+'-planned',row['offlineReference'])])})
    guides=[]
    for speaker in ('af_heart','am_michael','af_bella'):
        source=next(c for c in sources['clips'] if c['name']==speaker+'-ah')
        guides+=package([('Original Kokoro voice guide',speaker+'-original',source['file'])])
    manifest={'scope':'All three captured synthetic voices; held ah and vowel melody, immediate controls versus earlier complete-score reference',
        'processing':'Whole clips, constant gain to shared integrated loudness per pair. No limiting, denoising or pitch correction.',
        'latency':'Audition WAVs are aligned offline. Live renderer has a fixed 20 ms waveform delay plus computation/transport/device time.',
        'truePeakMeasured':False,'listeningAccepted':False,'groups':groups,'originalGuides':guides}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('Saved 12 matched comparison clips and 3 original voice guides; no playback.')


if __name__=='__main__':main()
