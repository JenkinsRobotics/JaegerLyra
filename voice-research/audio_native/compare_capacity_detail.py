"""Fresh-frame error diagnostics for both frozen decoders; no model selection."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from analyze_neural_output_error import frames
from neural_waveform import RATE,SIZE,HOP
from benchmark_voicebox import f0_track

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-waveform/capacity-v3'


def main():
    report=json.loads((OUT/'evaluation/report.json').read_text())
    if not report['complete']:raise ValueError('Finish all frozen-model evaluations first')
    originals=[r for r in report['originals'] if r['kind']=='fresh-kokoro']
    bands=((0,1000),(1000,4000),(4000,8000),(8000,12001))
    frequency=np.fft.rfftfreq(SIZE,1/RATE);window=np.hanning(SIZE)
    result={'scope':'Post-selection error localization on all nine fresh speech clips; diagnostic only',
        'bandsHz':bands,'fileGain':report['auditionGain'],'results':[],'aggregate':{}}
    for original in originals:
        source,rate=sf.read(ROOT/original['file']);assert rate==RATE
        truth=frames(source);rms=np.sqrt(np.mean(truth*truth,axis=1))
        times=np.arange(len(truth))*HOP/RATE
        pt,pitch=f0_track(source);voiced=np.interp(times,pt,(pitch>0).astype(float))>.5
        active=rms>.003
        masks={'active-voiced':active&voiced,'active-unvoiced':active&~voiced,
            'quiet':(rms>1e-6)&~active,'near-silent':rms<=1e-6}
        for mode in ('structured-v2','capacity-v3'):
            row=next(r for r in report['results'] if r['fixture']==original['fixture'] and r['mode']==mode)
            if row['status']!='ok':raise ValueError('Do not omit an unsuccessful comparison')
            audio,rate=sf.read(ROOT/row['file']);audio/=report['auditionGain']
            error=frames(source-audio)
            power=abs(np.fft.rfft(error*window,axis=1))**2;power[:,1:-1]*=2
            record={'fixture':original['fixture'],'mode':mode,
                'bandErrorEnergy':[float(np.sum(power[:,(frequency>=a)&(frequency<b)])) for a,b in bands],
                'categories':{}}
            for name,mask in masks.items():
                signal=float(np.sum(truth[mask]**2));noise=float(np.sum(error[mask]**2))
                record['categories'][name]={'frames':int(sum(mask)),'signalEnergy':signal,'errorEnergy':noise,
                    'snrDb':float(10*np.log10(max(signal,1e-30)/max(noise,1e-30))) if signal>1e-12 else None}
            result['results'].append(record)
    for mode in ('structured-v2','capacity-v3'):
        rows=[r for r in result['results'] if r['mode']==mode]
        result['aggregate'][mode]={'bandErrorEnergy':np.sum([r['bandErrorEnergy'] for r in rows],axis=0).tolist(),'categories':{}}
        for name in masks:
            signal=sum(r['categories'][name]['signalEnergy'] for r in rows)
            noise=sum(r['categories'][name]['errorEnergy'] for r in rows)
            result['aggregate'][mode]['categories'][name]={'signalEnergy':signal,'errorEnergy':noise,
                'snrDb':float(10*np.log10(max(signal,1e-30)/max(noise,1e-30))) if signal>1e-12 else None}
    old=np.array(result['aggregate']['structured-v2']['bandErrorEnergy'])
    new=np.array(result['aggregate']['capacity-v3']['bandErrorEnergy'])
    result['bandErrorReductionDb']=(10*np.log10(old/np.maximum(new,1e-30))).tolist()
    (OUT/'detail-comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'bandErrorReductionDb':result['bandErrorReductionDb'],'aggregate':result['aggregate']},indent=2))


if __name__=='__main__':main()
