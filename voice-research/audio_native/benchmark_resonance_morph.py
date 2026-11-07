"""Learn compact diphthong timing on nine words, evaluate eighteen fresh words.

Reference pitch, duration and amplitude are supplied to all candidates to
isolate spectral articulation. Those oracle controls are not learned speech.
No original target waveform is supplied to any candidate renderer.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import parselmouth
import pyworld
import soundfile as sf
from neural_waveform import load_model,ControlMatrix,RATE
from compiled_neural_decoder import compile_decoder
from neural_performance import compile_performance,EncodedPerformance,save_performance,load_performance
from live_neural_grains import LiveNeuralGrainStream,DELAY
from compact_vocal_instrument import VocalFrame,BLOCK
from resonance_morph import ResonancePath,NeuralResonanceStream,FREQUENCIES,FFT

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/resonance-morph/v1'
MODEL=ROOT/'evidence/neural-waveform/axis-detail-v2/projected-r1e-7.npz'
EXPECTED_MODEL='ac2debf521a0936a2ee9b7c9cb5cac86d550612de78a7d57233363b28b58a510'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save_json(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def source(item,encoder):
    path=ROOT/item['file']
    if sha(path)!=item['sha256']:raise ValueError('Changed source')
    audio,rate=sf.read(path);assert rate==RATE and audio.ndim==1
    performance=compile_performance(audio,encoder)
    groups=[g for g in np.split(performance.pulses,np.flatnonzero(np.diff(performance.pulses)>RATE*.020)+1) if len(g)>=10]
    if not groups:raise ValueError('No voiced nucleus with at least ten pulses')
    group=max(groups,key=lambda g:g[-1]-g[0])
    start=int(np.ceil(group[0]));stop=int(np.floor(group[-1]))
    core=audio[start:stop];pulses=group-start
    pt=(pulses[:-1]+pulses[1:])/(2*RATE);pf=RATE/np.diff(pulses)
    if min(pf)<60 or max(pf)>800:raise ValueError('Nucleus pitch outside control range')
    times=np.arange(int(np.ceil(len(core)/BLOCK)))*.005
    f0=np.interp(times,pt,pf)
    power=pyworld.cheaptrick(np.ascontiguousarray(core),f0,times,RATE,fft_size=FFT)
    rms=np.sqrt(np.convolve(core**2,np.ones(480)/480,mode='same'))
    if len(rms)!=len(core):raise ValueError('Voiced nucleus too short')
    return core,pulses,{'sourceStartSample':start,'sourceStopSample':stop,'times':times,'pitch':f0,
        'rms':rms,'power':power,'seconds':len(core)/RATE}


def endpoint(core,pulses,position,encoder,decoder,neighbours,minimum_seconds):
    neighbours=max(neighbours,int(np.ceil(minimum_seconds*RATE/(2*np.median(np.diff(pulses))))))
    if len(pulses)<2*neighbours+1:raise ValueError('Nucleus too short for endpoint analysis window')
    index=int(np.argmin(abs(pulses-position*len(core))))
    index=int(np.clip(index,neighbours,len(pulses)-neighbours-1))
    a=int(np.floor(pulses[index-neighbours]));b=int(np.ceil(pulses[index+neighbours]))+1
    a=max(0,a);b=min(len(core),b)
    selected=pulses[index-neighbours+1:index+neighbours]-a
    if b-a>RATE*.2 or len(selected)<4 or selected[0]<0 or selected[-1]>=b-a:raise ValueError('Endpoint support invalid')
    matrix=encoder.encode(core[a:b]);decoded=decoder.render(matrix)
    rms=float(np.sqrt(np.mean(decoded**2)))
    if rms<1e-4:raise ValueError('Quiet endpoint')
    patch=EncodedPerformance(ControlMatrix(matrix.values,matrix.gains*(.1/rms),matrix.sample_count),selected.copy())
    audio=decoder.render(patch.matrix);t=np.arange(0,len(audio)/RATE,.005)
    f=np.interp(t,(selected[:-1]+selected[1:])/(2*RATE),RATE/np.diff(selected))
    spectrum=pyworld.cheaptrick(np.ascontiguousarray(audio),f,t,RATE,fft_size=FFT)
    interior=(t>=.015)&(t<=len(audio)/RATE-.015)
    if sum(interior)<3:raise ValueError('Insufficient endpoint interior')
    shape=np.median(np.log(np.maximum(spectrum[interior],1e-20)),axis=0)
    formant=parselmouth.Sound(audio,sampling_frequency=RATE).to_formant_burg(time_step=.005,
        max_number_of_formants=5.,maximum_formant=5500.,window_length=.025,pre_emphasis_from=50.)
    landmarks=np.array([np.nanmedian([formant.get_value_at_time(k,at) for at in t[interior]]) for k in (1,2,3)])
    return patch,shape,landmarks,{'startSample':a,'stopSample':b,'centerFraction':float(pulses[index]/len(core)),
        'normalizationGain':.1/rms,'landmarksHz':landmarks.tolist(),'pulseCount':len(selected),'pulseNeighbours':neighbours}


def spectral_features(power):
    centers=np.geomspace(100,6000,82);bank=[]
    for a,b,c in zip(centers[:-2],centers[1:-1],centers[2:]):
        weights=np.maximum(0,np.minimum((FREQUENCIES-a)/(b-a),(c-FREQUENCIES)/(c-b)))
        bank.append(weights/max(weights.sum(),1e-20))
    db=10*np.log10(np.maximum(np.asarray(power)@np.array(bank).T,1e-20))
    return db-db.mean(axis=-1,keepdims=True)


def fit_path(power,times,seconds,path,config):
    lo,hi=config['endpointPositions'];mask=(times/seconds>=lo)&(times/seconds<=hi)
    phase=(times[mask]/seconds-lo)/(hi-lo)
    grid=np.linspace(0,1,config['trajectoryGrid'])
    expected=spectral_features(np.exp(np.array([path.at(a) for a in grid])))
    observed=spectral_features(power[mask])
    cost=np.mean((observed[:,None,:]-expected[None,:,:])**2,axis=2)
    if len(cost)<4:raise ValueError('Too few interior articulation frames')
    previous=np.full(len(grid),np.inf);previous[0]=cost[0,0];back=[]
    for row in cost[1:]:
        indices=np.array([np.argmin(previous[:j+1]) for j in range(len(grid))])
        previous=row+previous[indices];back.append(indices)
    chosen=[len(grid)-1]
    for indices in reversed(back):chosen.append(int(indices[chosen[-1]]))
    chosen=grid[np.array(chosen[::-1])]
    knots=np.linspace(0,1,config['trajectoryKnots']);fitted=np.interp(knots,phase,chosen)
    fitted[0]=0;fitted[-1]=1
    return fitted,{'phase':phase.tolist(),'selectedAlpha':chosen.tolist(),'trainingFrames':len(cost)}


def capture(config,refs):
    if sha(MODEL)!=EXPECTED_MODEL:raise ValueError('Changed decoder')
    encoder,decoder,_=load_model(MODEL);decoder=compile_decoder(decoder)
    report={'model':str(MODEL.relative_to(ROOT)),'modelSha256':EXPECTED_MODEL,
        'sourceManifest':'evidence/resonance-morph/v1/references.json','units':[],'failures':[],
        'trajectoryScope':'Monotonic grid fit to development spectral envelopes, averaged over three voices; no neural training.',
        'complete':False}
    for item in refs['clips']:
        if item['split']!='train':continue
        try:
            core,pulses,meta=source(item,encoder);patches=[];shapes=[];landmarks=[];endpoint_meta=[]
            for j,position in enumerate(config['endpointPositions']):
                patch,shape,landmark,detail=endpoint(core,pulses,position,encoder,decoder,config['endpointPulseNeighbours'],config['minimumEndpointSeconds'])
                file=OUT/(item['name']+'-endpoint-'+str(j)+'.npz')
                save_performance(file,patch,{'sourceSha256':item['sha256'],'modelSha256':EXPECTED_MODEL,**detail})
                patches.append({'file':str(file.relative_to(ROOT)),'sha256':sha(file)})
                shapes.append(shape);landmarks.append(landmark);endpoint_meta.append(detail)
            path=ResonancePath(shapes,landmarks)
            fitted,fit_details=fit_path(meta['power'],meta['times'],meta['seconds'],path,config)
            profile=OUT/(item['name']+'-path.npz');np.savez(profile,log_power=path.log_power,landmarks=path.landmarks)
            report['units'].append({'speaker':item['speaker'],'group':item['group'],'source':item['name'],
                'patches':patches,'profile':str(profile.relative_to(ROOT)),'profileSha256':sha(profile),
                'endpointDetails':endpoint_meta,'fittedAlpha':fitted.tolist(),'fit':fit_details})
            print('Captured',item['name'],flush=True)
        except ValueError as error:
            report['failures'].append({'source':item['name'],'error':str(error)});print('Rejected',item['name'],str(error),flush=True)
        save_json('capture.json',report)
    trajectories={}
    for group in config['groups']:
        units=[u for u in report['units'] if u['group']==group]
        # Do not hide a missing training voice by averaging fewer identities.
        if len(units)==len(config['speakers']):trajectories[group]=np.mean([u['fittedAlpha'] for u in units],axis=0).tolist()
    report['trajectories']=trajectories;report['complete']=True;save_json('capture.json',report)


def render(config,refs):
    capture=json.loads((OUT/'capture.json').read_text())
    assert capture['complete'] and sha(MODEL)==EXPECTED_MODEL
    encoder,decoder,_=load_model(MODEL);decoder=compile_decoder(decoder)
    report={'scope':'Held-out word nuclei, oracle source pitch/duration/RMS, development endpoint timbre and articulation path',
        'expectedCases':18*len(config['methods']),'results':[],'references':[],'failures':[],'complete':False,
        'noTargetWaveformAtRender':True,'neuralWeightsUnchanged':True,'newNeuralTraining':False}
    for item in refs['clips']:
        if item['split']!='validation':continue
        try:
            unit=next(u for u in capture['units'] if u['speaker']==item['speaker'] and u['group']==item['group'])
            if item['group'] not in capture['trajectories']:raise ValueError('Incomplete group trajectory')
            patches=[]
            for row in unit['patches']:
                if sha(ROOT/row['file'])!=row['sha256']:raise ValueError('Changed endpoint')
                patches.append(load_performance(ROOT/row['file'])[0])
            if sha(ROOT/unit['profile'])!=unit['profileSha256']:raise ValueError('Changed envelope profile')
            with np.load(ROOT/unit['profile']) as z:path=ResonancePath(z['log_power'],z['landmarks'])
            core,_,meta=source(item,encoder);count=len(core)
            fade=np.ones(count);n=min(240,count//2);fade[:n]=np.linspace(0,1,n);fade[-n:]=np.linspace(1,0,n)
            original=OUT/(item['name']+'-nucleus.wav');sf.write(original,core*fade,RATE,subtype='FLOAT')
            direct=decoder.render(encoder.encode(core));reconstruction=OUT/(item['name']+'-reconstruction.wav')
            sf.write(reconstruction,direct*fade,RATE,subtype='FLOAT')
            controls=OUT/(item['name']+'-controls.npz')
            np.savez(controls,times=meta['times'],pitch=meta['pitch'],rms=meta['rms'],power=meta['power'])
            report['references'].append({**item,'nucleus':str(original.relative_to(ROOT)),
                'reconstruction':str(reconstruction.relative_to(ROOT)),'controls':str(controls.relative_to(ROOT)),
                'sourceStartSample':meta['sourceStartSample'],'sourceStopSample':meta['sourceStopSample'],'seconds':meta['seconds']})
            for method in config['methods']:
                morph=method.startswith('filter')
                stream=NeuralResonanceStream(patches[0],decoder,path,config['filterTaps'],config['maximumFilterGainDb']) if morph else LiveNeuralGrainStream(patches,decoder)
                parts=[];timings=[];alphas=[]
                for i,seconds in enumerate(meta['times']):
                    lo,hi=config['endpointPositions'];phase=float(np.clip((seconds/meta['seconds']-lo)/(hi-lo),0,1))
                    alpha=float(np.interp(phase,np.linspace(0,1,config['trajectoryKnots']),capture['trajectories'][item['group']])) if method.endswith('learned') else phase
                    hz=float(meta['pitch'][i]);begin=time.perf_counter()
                    block=stream.push(hz,alpha) if morph else stream.push(VocalFrame(hz,(1-alpha,alpha)))
                    timings.append((time.perf_counter()-begin)*1000);parts.append(block);alphas.append(alpha)
                parts.append(stream.finish());audio=np.concatenate(parts)[DELAY:DELAY+count]
                # The same reference amplitude curve is supplied to every method.
                audio=audio*(meta['rms']/.1)*fade
                if len(audio)!=count or not np.isfinite(audio).all():raise ValueError('Invalid output')
                file=OUT/(item['name']+'-'+method+'.wav');sf.write(file,audio,RATE,subtype='FLOAT')
                report['results'].append({'name':item['name'],'speaker':item['speaker'],'group':item['group'],'method':method,
                    'file':str(file.relative_to(ROOT)),'sha256':sha(file),'alpha':alphas,'rawPeak':float(max(abs(audio))),
                    'overFullScaleSamples':int(sum(abs(audio)>=1)),'blockP95Ms':float(np.percentile(timings,95)),
                    'maximumBlockMs':max(timings),'blocksOver5Ms':int(sum(t>5 for t in timings)),
                    'fixedWaveformDelayMs':20,'filterLookaheadSamples':0,'filterHistorySamples':511 if morph else 0,
                    'computeScope':'Resident prepared controls; no loading, HTTP, browser or device latency included.'})
                print('Rendered',item['name'],method,flush=True);save_json('render.json',report)
        except (ValueError,StopIteration) as error:
            report['failures'].append({'source':item['name'],'error':str(error)});print('Rejected',item['name'],str(error),flush=True)
            save_json('render.json',report)
    report['complete']=True;save_json('render.json',report)


def stft_features(audio):
    pad=480;window=np.hanning(960);padded=np.pad(audio,(pad,pad))
    frames=np.lib.stride_tricks.sliding_window_view(padded,960)[::120]
    power=abs(np.fft.rfft(frames*window,n=FFT))**2
    return np.arange(len(frames))*.005,spectral_features(power)


def evaluate(config):
    render=json.loads((OUT/'render.json').read_text());assert render['complete']
    oracle=json.loads((OUT/'oracle-envelope.json').read_text()) if (OUT/'oracle-envelope.json').exists() else {'results':[],'complete':True}
    assert oracle['complete']
    report={'scope':'Fresh-word spectral/pitch diagnostics; oracle prosody, not listener acceptance','results':[],'complete':False}
    for reference in render['references']:
        target,_=sf.read(ROOT/reference['nucleus']);times,features=stft_features(target)
        with np.load(ROOT/reference['controls']) as z:ct=z['times'].copy();cf=z['pitch'].copy()
        entries=[{'method':'teacher','file':reference['nucleus']},{'method':'reconstruction','file':reference['reconstruction']}]+[r for r in render['results'] if r['name']==reference['name']]
        entries += [r for r in oracle['results'] if r['name']==reference['name']]
        for entry in entries:
            audio,rate=sf.read(ROOT/entry['file']);assert rate==RATE
            at,observed=stft_features(audio);assert len(times)==len(at)
            error=np.mean(abs(features-observed),axis=1)
            core=(times>=.03)&(times<=reference['seconds']-.03)
            transition=core&(times/reference['seconds']>=.25)&(times/reference['seconds']<=.75)
            pitch=parselmouth.Sound(audio,sampling_frequency=RATE).to_pitch_ac(time_step=.005,pitch_floor=50.,pitch_ceiling=900.)
            wf,wt=pyworld.harvest(np.ascontiguousarray(audio),RATE,f0_floor=50.,f0_ceil=900.,frame_period=5.)
            metrics={}
            for label,t,f in [('praat',pitch.xs(),pitch.selected_array['frequency']),('world',wt,wf)]:
                eligible=(t>=.03)&(t<=reference['seconds']-.03);voiced=eligible&(f>0)
                cents=abs(1200*np.log2(f[voiced]/np.interp(t[voiced],ct,cf)))
                coverage=float(sum(voiced)/max(1,sum(eligible)));p95=float(np.percentile(cents,95)) if len(cents) else None
                metrics[label]={'voicedFraction':coverage,'p95ErrorCents':p95,'passed':coverage>=.95 and p95 is not None and p95<=50}
            row={'name':reference['name'],'speaker':reference['speaker'],'group':reference['group'],'method':entry['method'],
                'file':entry['file'],'sha256':sha(ROOT/entry['file']),
                'centeredSpectrumMaeDb':float(np.mean(error[core])),
                'transitionSpectrumMaeDb':float(np.mean(error[transition])),
                'rmsErrorDb':float(20*np.log10((np.sqrt(np.mean(audio**2))+1e-12)/(np.sqrt(np.mean(target**2))+1e-12))),
                'pitch':metrics}
            track=OUT/(reference['name']+'-'+entry['method']+'-metrics.npz')
            np.savez(track,times=times,reference=features,observed=observed,spectrumError=error,transition=transition,
                praatTimes=pitch.xs(),praat=pitch.selected_array['frequency'],worldTimes=wt,world=wf,controlTimes=ct,controlPitch=cf)
            row['tracks']=str(track.relative_to(ROOT));report['results'].append(row);save_json('evaluation.json',report)
            print('Evaluated',reference['name'],entry['method'],round(row['transitionSpectrumMaeDb'],3),flush=True)
    report['complete']=True;save_json('evaluation.json',report)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['capture','render','evaluate']);args=parser.parse_args()
    config=json.loads((OUT/'config.json').read_text());refs=json.loads((OUT/'references.json').read_text())
    assert refs['complete'] and sha(OUT/'config.json')==refs['configSha256']
    if args.stage=='capture':capture(config,refs)
    elif args.stage=='render':render(config,refs)
    else:evaluate(config)
