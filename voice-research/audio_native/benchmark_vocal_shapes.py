"""Controlled vowel transitions: waveform blend vs external spectral reference."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import parselmouth
import pyworld
import soundfile as sf
from neural_waveform import RATE,load_model
from neural_performance import load_performance
from neural_vowel_instrument import VowelScore
from vocal_shape_control import (analyse_shape,ShapePath,shape_controls,render_world_shapes,
    render_grain_crossfade,save_shapes,load_shapes,FFT_SIZE,FREQUENCIES)
from benchmark_neural_vowels import pitch_metrics

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/vocal-shapes/v1'


def evaluate(audio,score,path,profiles):
    sound=parselmouth.Sound(audio,sampling_frequency=RATE)
    praat=sound.to_pitch_ac(time_step=.01,pitch_floor=50.,pitch_ceiling=900.)
    pt=praat.xs();pf=praat.selected_array['frequency']
    wf,wt=pyworld.harvest(np.ascontiguousarray(audio),RATE,f0_floor=50.,f0_ceil=900.,frame_period=10.)
    commanded=score.frequency(wt)
    power=pyworld.cheaptrick(np.ascontiguousarray(audio),commanded,wt,RATE,fft_size=FFT_SIZE)
    target,_=shape_controls(profiles,path,wt,score.seconds)
    band=(FREQUENCIES>=80)&(FREQUENCIES<=6000)
    def centered(value):
        db=10*np.log10(np.maximum(value[:,band],1e-20))
        return db-db.mean(axis=1,keepdims=True)
    observed=centered(power);expected=centered(target)
    interior=(wt>=.05)&(wt<=score.seconds-.05)
    weights=path.at(wt);dominant=np.argmax(weights,axis=1)
    # Plateau eligibility is determined only by authored control weights,
    # including fixed margins on both sides of a transition.
    plateau=interior&(np.max(weights,axis=1)>1-1e-9)
    for delta in (-.05,.05):
        other=path.at(wt+delta)
        plateau&=(np.argmax(other,axis=1)==dominant)&(np.max(other,axis=1)>1-1e-9)
    corners=centered(np.exp(np.array([p.log_power for p in profiles])))
    distances=np.mean(abs(observed[:,None,:]-corners[None,:,:]),axis=2)
    closest=np.argmin(distances,axis=1)
    formant=sound.to_formant_burg(time_step=.01,max_number_of_formants=5.,
        maximum_formant=5500.,window_length=.025,pre_emphasis_from=50.)
    formants=[]
    for label in range(len(profiles)):
        t=wt[plateau&(dominant==label)]
        row=[]
        for number in (1,2,3):
            values=np.array([formant.get_value_at_time(number,at) for at in t])
            row.append(float(np.nanmedian(values)) if np.isfinite(values).any() else None)
        formants.append(row)
    result={'praat':pitch_metrics(pt,pf,score),'world':pitch_metrics(wt,wf,score),
        'shapeSelfConsistency':{'scope':'WORLD CheapTrick analysis against prescribed profile shapes, with per-frame mean dB removed over 80–6000 Hz; shares algorithms with external WORLD renderer, not independent quality evidence',
            'meanAbsoluteDb':float(np.mean(abs(observed[interior]-expected[interior]))),
            'plateauMeanAbsoluteDb':float(np.mean(abs(observed[plateau]-expected[plateau]))),
            'plateauFrames':int(sum(plateau)),'plateauNearestShapeAccuracy':float(np.mean(closest[plateau]==dominant[plateau]))},
        'plateauFormantsHz':formants,
        'rawPeak':float(abs(audio).max()),'overFullScaleSamples':int(sum(abs(audio)>=1.)),
        'finite':bool(np.isfinite(audio).all())}
    tracks={'times':wt,'world':wf,'targetPitch':commanded,'praatTimes':pt,'praat':pf,
        'weights':weights,'plateau':plateau,'closestShape':closest,
        'shapeErrorDb':np.mean(abs(observed-expected),axis=1)}
    return result,tracks


def main():
    prior=json.loads((ROOT/'evidence/neural-vowels/v1/report.json').read_text())
    if not prior['complete']:raise ValueError('Complete the captured-vowel panel first')
    model=ROOT/prior['model']
    if hashlib.sha256(model.read_bytes()).hexdigest()!=prior['modelSha256']:raise ValueError('Frozen model changed')
    _,decoder,_=load_model(model)
    weight_hash=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest()
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Output-only authored transitions among three captured vowel shapes; no learned articulation or new speech',
        'model':prior['model'],'modelSha256':prior['modelSha256'],
        'methods':{'grain-crossfade':'Direct PCM blend of our existing frozen-neural captured-vowel instruments',
            'world-shapes':'Continuous log-spectral-shape and aperiodicity controls, synthesized by external WORLD via pyworld 0.3.5; offline comparator, not our neural or clean-room engine'},
        'shapeOrder':['ah','ee','oo'],'expectedCases':24,'profiles':[],'results':[],'complete':False,
        'timingScope':'Whole-utterance offline render only, including controls but excluding profile compilation; not streaming or first-audio latency',
        'limits':['Intended vowels are not human-verified phonetic labels.',
            'WORLD shape self-consistency uses the same analysis family as its synthesis controls.',
            'Praat formants are a secondary diagnostic, not exact vocal-tract measurements.',
            'A smooth numeric path and good pitch are not evidence of natural coarticulation or Kokoro quality.']}
    for speaker in ('af_heart','am_michael','af_bella'):
        refs=[next(r for r in prior['references'] if r['name']==speaker+'-'+label) for label in report['shapeOrder']]
        patches=[load_performance(ROOT/r['artifact'])[0] for r in refs]
        profiles=[];metadata=[]
        for patch in patches:
            profile,meta=analyse_shape(patch,decoder);profiles.append(profile);metadata.append(meta)
        bank=OUT/(speaker+'-shapes.npz')
        save_shapes(bank,profiles,{'speaker':speaker,'shapeOrder':report['shapeOrder'],'sourceArtifacts':[r['artifact'] for r in refs],
            'modelSha256':prior['modelSha256'],'analysis':metadata})
        profiles,_=load_shapes(bank)
        report['profiles'].append({'speaker':speaker,'file':str(bank.relative_to(ROOT)),
            'analysis':metadata,'sourceFormantsHz':[r['sourceFormantsHz'] for r in refs]})
        base=110. if speaker=='am_michael' else 220.
        times=(0.,.8,1.05,1.8,2.05,2.8,3.05,4.)
        ah=(1.,0.,0.);ee=(0.,1.,0.);oo=(0.,0.,1.)
        path=ShapePath(times,(ah,ah,ee,ee,oo,oo,ah,ah))
        fast=ShapePath(tuple(t*.5 for t in times),path.weights)
        modes={'steady':(VowelScore(seconds=4.,hz=(base,)),path),
            'vibrato':(VowelScore(seconds=4.,hz=(base,),vibrato_cents=35.),path),
            'melody':(VowelScore(seconds=4.,times=(0.,.95,1.05,1.95,2.05,2.95,3.05,4.),
                hz=tuple(base*2**(np.array([0.,0.,2.,2.,4.,4.,0.,0.])/12))),path),
            'faster':(VowelScore(seconds=2.,hz=(base,)),fast)}
        for mode,(score,curve) in modes.items():
            for method in report['methods']:
                row={'speaker':speaker,'mode':mode,'method':method,'score':score.__dict__,'shapePath':curve.__dict__}
                try:
                    start=time.perf_counter()
                    if method=='grain-crossfade':audio=render_grain_crossfade(patches,decoder,score,curve)
                    else:audio=render_world_shapes(profiles,score,curve)
                    elapsed=time.perf_counter()-start
                    file=OUT/(speaker+'-'+mode+'-'+method+'.wav')
                    sf.write(file,audio,RATE,subtype='FLOAT')
                    metrics,tracks=evaluate(audio,score,curve,profiles)
                    trackpath=file.with_suffix('.npz');np.savez(trackpath,**tracks)
                    row.update(status='ok',file=str(file.relative_to(ROOT)),tracks=str(trackpath.relative_to(ROOT)),
                        renderSeconds=elapsed,computeRealTimeFactor=elapsed/score.seconds,**metrics)
                except ValueError as error:
                    row.update(status='failed',error=str(error))
                report['results'].append(row)
                (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                print(speaker,mode,method,row['status'],row.get('shapeSelfConsistency',{}).get('meanAbsoluteDb'),flush=True)
    if weight_hash!=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest():raise AssertionError('Neural weights changed')
    report['weightsUnchanged']=True;report['complete']=True
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
