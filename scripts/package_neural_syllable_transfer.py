"""Level-matched teacher, reconstruction and live-control comparisons.

Includes failed diagnostics and the source with a rejected capture. No playback.
"""
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import pyloudnorm as pyln

ROOT=Path(__file__).resolve().parents[1]
STUDY=ROOT/'evidence/neural-syllables/teacher-transfer-v1'
OUT=STUDY/'audition'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    capture=json.loads((STUDY/'capture.json').read_text())
    render=json.loads((STUDY/'report.json').read_text())
    if not capture['complete'] or not render['complete']:raise ValueError('Incomplete sources')
    cases=capture['references']+capture['reconstructions']+render['results']
    OUT.mkdir(exist_ok=True);files=[];reels=[];meter=pyln.Meter(24000,block_size=.2)
    for source in capture['sources']:
        rows=[{**source,'variant':'original','method':'ordinary-speech','mode':'source'}]
        rows += [r for r in cases if (r['speaker'],r['word'])==(source['speaker'],source['word'])]
        loaded=[]
        for row in rows:
            file=ROOT/row['file']
            if sha(file)!=row['sha256']:raise ValueError('Changed source audio')
            audio,rate=sf.read(file);assert rate==24000
            lufs=meter.integrated_loudness(audio)
            if not np.isfinite(lufs):raise ValueError('Non-finite loudness')
            loaded.append((row,audio,lufs))
        target=min(-20.,*(lufs+20*np.log10(.85/max(abs(audio))) for _,audio,lufs in loaded))-.2
        packed={};levelled={}
        for row,audio,lufs in loaded:
            gain=10**((target-lufs)/20)
            for _ in range(20):
                measured=meter.integrated_loudness(audio*gain)
                if abs(measured-target)<.002:break
                gain=min(gain*10**((target-measured)/20),.85/max(abs(audio)))
            file=OUT/(row['name']+'.wav');sf.write(file,audio*gain,24000,subtype='PCM_24')
            saved,_=sf.read(file);measured=meter.integrated_loudness(saved)
            if abs(measured-target)>.01 or max(abs(saved))>.851:raise ValueError('Invalid audition level')
            record={k:row[k] for k in ('name','speaker','word','variant','method','mode')}
            record.update(file=str(file.relative_to(ROOT)),sha256=sha(file),source=row['file'],sourceSha256=row['sha256'],
                auditionGain=gain,targetLufs200MsGate=target,measuredLufs200MsGate=measured,samplePeak=float(max(abs(saved))))
            files.append(record);key=(row['variant'],row['method'],row['mode'])
            packed[key]=record;levelled[key]=saved
        for variant in ('baseline','level'):
            sequence=[('original','ordinary-speech','source'),(variant,'teacher','native'),
                (variant,'reconstruction','native'),(variant,'live-neural','native'),(variant,'live-neural','melody')]
            parts=[];segments=[];missing=[];offset=0
            for key in sequence:
                if key not in packed:
                    missing.append({'variant':key[0],'method':key[1],'mode':key[2]});continue
                audio=levelled[key]
                segments.append({'startSeconds':offset/24000,'endSeconds':(offset+len(audio))/24000,
                    **{k:packed[key][k] for k in ('variant','method','mode','file')}})
                parts.extend([audio,np.zeros(6000)]);offset+=len(audio)+6000
            file=OUT/(source['name']+'-'+variant+'-comparison.wav')
            sf.write(file,np.concatenate(parts[:-1]),24000,subtype='PCM_24')
            reels.append({'speaker':source['speaker'],'word':source['word'],'variant':variant,
                'file':str(file.relative_to(ROOT)),'sha256':sha(file),'segments':segments,'missingCases':missing,
                'scope':'Labelled offline concatenation with 250 ms gaps; not connected speech or live latency.'})
    manifest={'scope':'Every teacher, reconstruction and eligible live-render condition with ordinary source speech',
        'processing':'One constant audition gain per whole clip; common source-specific target across both variants and all methods, using 200 ms loudness gating. No limiter or pitch correction.',
        'files':files,'reels':reels,'captureFailures':capture['failures'],
        'maximumSamplePeak':max(r['samplePeak'] for r in files),
        'maximumLoudnessErrorLu':max(abs(r['measuredLufs200MsGate']-r['targetLufs200MsGate']) for r in files),
        'truePeakMeasured':False,'autoplay':False,'listeningAccepted':False,'complete':True}
    (OUT/'manifest.json').write_text(json.dumps(manifest,indent=2,allow_nan=False)+'\n')
    print('Packaged',len(files),'clips and',len(reels),'labelled reels; no playback.')


if __name__=='__main__':main()
