"""Independent WORLD pitch checks and measured vibrato for the output controls."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
import pyworld
import soundfile as sf
from benchmark_voicebox import f0_track
from neural_waveform import RATE
from neural_performance import load_performance

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-performance/v2/control-checks.json'


def main():
    report=json.loads((OUT.parent/'report.json').read_text())
    if not report['complete']:raise ValueError('Complete the performance panel first')
    originals={r['fixture']:r for r in report['originals']}
    result={'scope':'Output control accuracy; not a naturalness or singing-quality acceptance',
        'world':'Harvest F0 on rendered audio; independent of the Praat pulse-analysis implementation',
        'vibrato':'Fit source-relative detected cents to sine/cosine across 2–8 Hz; no interpolation across unvoiced source frames',
        'results':[]}
    for row in report['results']:
        if row['status']!='ok' or row['mode'] not in ('four-notes','vibrato'):continue
        output,rate=sf.read(ROOT/row['file']);assert rate==RATE
        original,_=sf.read(ROOT/originals[row['fixture']]['file'])
        item={'fixture':row['fixture'],'mode':row['mode']}
        if row['mode']=='four-notes':
            pitch,times=pyworld.harvest(np.ascontiguousarray(output),RATE,f0_floor=50.,f0_ceil=800.,frame_period=10.)
            source_pitch,source_times=pyworld.harvest(np.ascontiguousarray(original),RATE,f0_floor=50.,f0_ceil=800.,frame_period=10.)
            expected=np.interp(times,row['controls']['note_times'],row['controls']['note_hz'])
            nearest=np.clip(np.rint(times/.01).astype(int),0,len(source_pitch)-1)
            expected[source_pitch[nearest]<=0]=0.
            joint=(pitch>0)&(expected>0)
            cents=abs(1200*np.log2(pitch[joint]/expected[joint]))
            item.update({'medianPitchErrorCents':float(np.median(cents)) if len(cents) else None,
                'p95PitchErrorCents':float(np.percentile(cents,95)) if len(cents) else None,
                'voicingAgreement':float(np.mean((pitch>0)==(expected>0))),'jointVoicedFrames':int(sum(joint))})
            performance,_=load_performance(ROOT/originals[row['fixture']]['performance'])
            pulses=performance.pulses/RATE*row['controls']['duration_scale']
            interior=np.zeros(len(times),dtype=bool)
            for group in np.split(pulses,np.flatnonzero(np.diff(pulses)>.020*row['controls']['duration_scale'])+1):
                if len(group)>=3:interior|=(times>=group[0]+.025)&(times<=group[-1]-.025)
            target=np.interp(times,row['controls']['note_times'],row['controls']['note_hz'])
            trusted=interior&(pitch>0)
            interior_cents=abs(1200*np.log2(pitch[trusted]/target[trusted]))
            item['pulseInterior']={'scope':'Secondary diagnostic: predetermined source-pulse regions, excluding 25 ms at each boundary; no selection by measured error',
                'eligibleFrames':int(sum(interior)),'detectedVoicedFrames':int(sum(trusted)),
                'fractionOfAllFrames':float(np.mean(interior)),
                'medianPitchErrorCents':float(np.median(interior_cents)) if len(interior_cents) else None,
                'p95PitchErrorCents':float(np.percentile(interior_cents,95)) if len(interior_cents) else None}
            praat_times,praat_pitch=f0_track(output)
            np.savez(OUT.parent/(row['fixture']+'-note-tracks.npz'),times=times,world=pitch,target=target,
                sourceWorld=source_pitch,pulseInterior=interior,praatTimes=praat_times,praat=praat_pitch)
        else:
            times,pitch=f0_track(output);source_times,source_pitch=f0_track(original)
            if not np.array_equal(times,source_times):raise ValueError('Vibrato changed duration/timing')
            joint=(pitch>0)&(source_pitch>0)
            t=times[joint];cents=1200*np.log2(pitch[joint]/source_pitch[joint])
            if len(t)<20:raise ValueError('Insufficient voiced samples for vibrato check')
            candidates=[]
            for frequency in np.linspace(2.,8.,241):
                design=np.c_[np.sin(2*np.pi*frequency*t),np.cos(2*np.pi*frequency*t),np.ones(len(t))]
                coefficient=np.linalg.lstsq(design,cents,rcond=None)[0]
                mse=float(np.mean((cents-design@coefficient)**2))
                candidates.append((mse,float(frequency),coefficient))
            mse,frequency,coefficient=min(candidates,key=lambda r:r[0])
            item.update({'requestedHz':row['controls']['vibrato_hz'],'measuredHz':frequency,
                'requestedCents':row['controls']['vibrato_cents'],
                'measuredCentsAmplitude':float(np.hypot(coefficient[0],coefficient[1])),
                'fittedOffsetCents':float(coefficient[2]),'residualRmsCents':float(np.sqrt(mse)),
                'jointVoicedFrames':len(t)})
            robust=[];delta=20.
            for frequency in np.linspace(2.,8.,241):
                design=np.c_[np.sin(2*np.pi*frequency*t),np.cos(2*np.pi*frequency*t),np.ones(len(t))]
                coef=np.linalg.lstsq(design,cents,rcond=None)[0]
                for _ in range(12):
                    residual=cents-design@coef;weight=np.minimum(1.,delta/np.maximum(abs(residual),1e-9))
                    root=np.sqrt(weight);coef=np.linalg.lstsq(design*root[:,None],cents*root,rcond=None)[0]
                error=abs(cents-design@coef)
                loss=float(np.mean(np.where(error<=delta,error**2,2*delta*error-delta**2)))
                robust.append((loss,float(frequency),coef,error))
            loss,frequency,coef,error=min(robust,key=lambda r:r[0])
            item['robustDiagnostic']={'method':'Huber IRLS with 20-cent residual threshold; all frames retained, raw estimate above remains authoritative evidence too',
                'measuredHz':frequency,'measuredCentsAmplitude':float(np.hypot(coef[0],coef[1])),
                'downweightedFrames':int(sum(error>delta)),'totalFrames':len(error),
                'maximumResidualCents':float(max(error))}
        result['results'].append(item);print(json.dumps(item),flush=True)
    OUT.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
