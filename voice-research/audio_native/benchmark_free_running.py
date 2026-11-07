"""Drive actual PCM synthesis with local forecasts during withheld-audio gaps.

Gap controls are never passed to the controller. Precomputed targets are retained
only for evaluation. Context-overlap masking prevents analysis lookahead leakage.
All training uses two original synthetic references; no pretrained decoder/text
model enters synthesis. This is acoustic continuation, not language generation.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ['HF_HUB_OFFLINE']='1'
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import soundfile as sf
from aligned_analysis import AlignedAnalysis
from continuous_source import ContinuousSource
from excitation import GeneratedVoicebox,pulse_profile
from streaming_voicebox import VoiceboxConfig,LPC_VERSION
from predictive_controls import ControlSpace,FreeRunningControls,AnchoredControls,visible_outside_gap
from auditory_prediction import LAGS
from lab import Config,LocalPredictor,LinearPredictor
from benchmark_source_ceiling import spectral_error
from benchmark_voicebox import errors,transcribe,f0_track

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/free-running-controls/lpc-v2'
CFG=VoiceboxConfig(order=32)


def analyze(audio):
    source=AlignedAnalysis(CFG)
    return source.push(audio)+source.finish()


def save_audio(path,audio):
    # One fixed gain for every variant; never normalize using the missing target.
    saved=audio*.5
    if np.abs(saved).max()>=1:
        raise ValueError(f'Fixed audition gain would clip (raw peak {np.abs(audio).max():.6f}); preserve as failure')
    sf.write(path,saved,24000,subtype='PCM_24')


def spectral_db(original,generated):
    value=spectral_error(original,generated)['meanAbsoluteDb']
    return value if np.isfinite(value) else None


def render(space,frames,features,visible,controller=None,filter_mode='direct'):
    sink=GeneratedVoicebox(profile=pulse_profile(64),config=CFG,seed=79,filter_mode=filter_mode)
    audio=[];vectors=[];times=[];predicted_poles=[]
    for i,frame in enumerate(frames):
        start=time.perf_counter()
        if controller is None:
            value=features[i]
        else:
            value,_=controller.step(features[i] if visible[i] else None)
        controls,harmonics,mix=space.decode(value,frame.start,len(frame.residual))
        if not visible[i]:predicted_poles.append(float(np.max(np.abs(np.roots(controls.coefficients)))))
        audio.append(sink.push(controls,harmonics=harmonics,voicing=mix))
        vectors.append(value)
        times.append((time.perf_counter()-start)*1000)
    return np.concatenate(audio),np.array(vectors),times,predicted_poles


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--validation',action='store_true')
    parser.add_argument('--anchored',action='store_true',help='Direct multi-horizon forecasts without recursive synthetic context')
    parser.add_argument('--output-energy',action='store_true',help='Predict desired output RMS and account for stationary filter/source gain')
    parser.add_argument('--lattice',action='store_true',help='Normalized scattering synthesis and normalized excitation ramp')
    args=parser.parse_args()
    if args.lattice:OUT=OUT/'normalized-lattice'
    if args.output_energy:OUT=OUT/'output-energy'
    if args.anchored:OUT=OUT/'anchored'
    if args.validation:OUT=OUT/'validation'
    OUT.mkdir(parents=True,exist_ok=True)
    filter_mode='normalized-lattice' if args.lattice else 'direct'
    preparation=0.
    if args.lattice:
        from lattice_filter import normalized_lattice
        started=time.perf_counter()
        normalized_lattice(np.zeros(1),np.zeros(CFG.order),np.zeros(CFG.order))
        preparation=(time.perf_counter()-started)*1000
    training_frames=[];training_records=[]
    for speaker in ('af_heart','am_michael'):
        path=ROOT/f'evidence/voice-fixtures/{speaker}-reference.wav'
        audio,rate=sf.read(path);assert rate==24000
        training_frames.append(analyze(audio))
        training_records.append({'file':str(path.relative_to(ROOT)),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
    source,source_count=ContinuousSource.fit(sum(training_frames,[]),dimensions=48,harmonics=64)
    space=ControlSpace(source,output_energy=args.output_energy)
    training=[np.array([space.encode(frame) for frame in frames]) for frames in training_frames]
    all_features=np.concatenate(training)
    center=all_features.mean(axis=0);scale=np.maximum(all_features.std(axis=0),.03)
    normalized=(all_features-center)/scale
    lower=normalized.min(axis=0)-1.;upper=normalized.max(axis=0)+1.
    np.savez(OUT/'reference-model.npz',source_mean=source.mean,source_basis=source.basis,
             center=center,scale=scale,lower=lower,upper=upper)
    models={'hold':None}
    parameters=Config(history=space.size*len(LAGS),block=space.size,hidden=96,
                      input_gain=1.,settling_steps=12,learning_rate=.04)
    horizons=(1,2,3,5,8,13,21,34,55)
    def build(kind,config):
        return {h:kind(config) for h in horizons} if args.anchored else kind(config)
    models['linear']=build(LinearPredictor,parameters)
    for seed in (17,29,41):
        models[f'pc-{seed}']=build(LocalPredictor,Config(**{**parameters.__dict__,'seed':seed}))
    for name,model in models.items():
        if model is None:continue
        for features in training:
            session=(AnchoredControls if args.anchored else FreeRunningControls)(model,center,scale,lower,upper,adaptive=True)
            for feature in features:session.step(feature)
        tensors={}
        entries=[(f'h{h}_',m) for h,m in model.items()] if args.anchored else [('',model)]
        for prefix,predictor in entries:
            tensors[prefix+'W']=predictor.W
            if hasattr(predictor,'V'):tensors[prefix+'V']=predictor.V
        np.savez(OUT/f'{name}-weights.npz',**tensors)
    variants=[('hold','hold',False),('linear-adaptive','linear',True)]
    variants += [(f'pc-{mode}-{seed}',f'pc-{seed}',mode=='adaptive')
                 for mode in ('frozen','adaptive') for seed in (17,29,41)]
    if args.validation:
        cases=json.loads((ROOT/'evidence/streaming-voicebox/validation/manifest.json').read_text())['clips']
    else:
        cases=[]
        for speaker in ('af_heart','am_michael'):
            path=ROOT/f'evidence/voice-fixtures/{speaker}-heldout.wav'
            cases.append({'name':speaker,'file':str(path.relative_to(ROOT)),
                          'text':json.loads(path.with_suffix('.json').read_text())['text']})
    report={'scope':'Short free-running acoustic gap continuation, no autonomous words or semantic understanding',
        'analysisVersion':LPC_VERSION,
        'training':training_records,'sharedSourceBasisVoicedFrames':source_count,
        'continuousFeatures':space.size,'historyMs':max(LAGS)*10,
        'energyControl':'Desired output RMS / stationary source-filter gain' if args.output_energy else 'Excitation residual RMS',
        'synthesisFilter':filter_mode,'jitPreparationMs':preparation,
        'learner':parameters.__dict__,'trainingPasses':1,'globalBackpropagation':False,
        'validation':args.validation,'fixtureOrigin':'Previously generated synthetic Kokoro sentences',
        'forecastMode':'direct anchored horizons' if args.anchored else 'recursive one-frame continuation',
        'horizonFrames':list(horizons) if args.anchored else [1],
        'gapStartSeconds':1.2,'gapDurationsMs':[100,250,500],
        'exclusion':'Every analysis window intersecting withheld PCM; nominal gap plus 40 ms of control frames',
        'learnDuringGap':False,
        'recoveryLearning':('Only observed targets update previously emitted, wholly observed contexts; new forecasts resume after full observed context' if args.anchored else 'Only after every context frame is observed again'),
        'bounds':'Reference-only min/max plus one training standard deviation; only generated learned predictions clipped',
        'auditionGain':.5,'metrics':'Feature error, gap spectral error, pitch and whole-clip ASR; no human quality acceptance',
        'timingScope':'Control prediction, decoding, synthesis and diagnostic pole checks only; excludes analysis, encoding, reference training, transport and devices',
        'undefinedMetrics':'Spectral error is null when its reference region has no active analysis frames',
        'originals':[],'results':[]}
    for case in cases:
        path=ROOT/case['file'];audio,rate=sf.read(path);assert rate==24000
        frames=analyze(audio);features=np.array([space.encode(frame) for frame in frames])
        oracle,_,_,_=render(space,frames,features,np.ones(len(frames),dtype=bool),filter_mode=filter_mode)
        oracle_path=OUT/(case['name']+'-observed-controls.wav');save_audio(oracle_path,oracle)
        oracle_text=transcribe(oracle_path)
        pitch_times,oracle_pitch=f0_track(oracle)
        report['originals'].append({'name':case['name'],'file':case['file'],
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'observedControlsFile':str(oracle_path.relative_to(ROOT)),
            'observedControlsText':oracle_text,**errors(case['text'],oracle_text),
            'reconstructionSpectralErrorDb':spectral_db(audio,oracle)})
        for milliseconds in (100,250,500):
            gap_start=28800;gap_end=gap_start+milliseconds*24
            visible=visible_outside_gap(frames,gap_start,gap_end,CFG)
            missing=np.flatnonzero(~visible)
            start_sample=frames[missing[0]].start
            end_sample=frames[missing[-1]].start+CFG.hop
            window=slice(start_sample,min(len(audio),end_sample+2400))
            pitch_region=(pitch_times>=start_sample/24000)&(pitch_times<(end_sample+2400)/24000)
            for label,model_name,adaptive in variants:
                model=copy.deepcopy(models[model_name])
                constructor=AnchoredControls if args.anchored and model is not None else FreeRunningControls
                session=constructor(model,center,scale,lower,upper,adaptive=adaptive)
                row={'fixture':case['name'],'mode':label,'gapMs':milliseconds,
                    'withheldControlFrames':int((~visible).sum()),
                    'withheldControlRangeSamples':[start_sample,end_sample]}
                try:
                    generated,values,latencies,poles=render(space,frames,features,visible,session,filter_mode=filter_mode)
                    output=OUT/f'{case["name"]}-{milliseconds}ms-{label}.wav';save_audio(output,generated)
                    text=transcribe(output)
                    _,output_pitch=f0_track(generated)
                    joint=pitch_region&(oracle_pitch>0)&(output_pitch>0)
                    cents=np.abs(1200*np.log2(output_pitch[joint]/oracle_pitch[joint]))
                    row.update({'status':'ok','file':str(output.relative_to(ROOT)),
                        'text':text,**errors(case['text'],text),
                        'featureMse':float(np.mean(((values[~visible]-features[~visible])/scale)**2)),
                        'spectralErrorVsObservedDb':spectral_db(oracle[window],generated[window]),
                        'spectralErrorVsOriginalDb':spectral_db(audio[window],generated[window]),
                        'medianPitchErrorCents':float(np.median(cents)) if len(cents) else None,
                        'pitchVoicingAgreement':float(np.mean((oracle_pitch[pitch_region]>0)==(output_pitch[pitch_region]>0))),
                        'rawPeak':float(np.abs(generated).max()),'maximumPredictedPoleRadius':max(poles),
                        'generatedFrames':session.generated_frames,'observedLearningFrames':session.learned_frames,
                        'clampedPredictionValues':session.clamped_values,
                        'processingMsP95':float(np.percentile(latencies,95)),
                        'processingRealTimeFactor':sum(latencies)/1000/(len(audio)/24000)})
                except ValueError as error:
                    row.update({'status':'failed','error':str(error)})
                report['results'].append(row)
                (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                print(json.dumps({k:v for k,v in row.items() if k not in ('file','text')}),flush=True)
    print(json.dumps({'done':len(report['results']),'failed':sum(r['status']!='ok' for r in report['results'])}),flush=True)


if __name__=='__main__':main()
