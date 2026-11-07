"""Diagnose the remaining source/filter reconstruction gap with richer controls.

All controls still come from observed audio. More detail is not new autonomous
speech generation. The explicitly labelled oracle variant uses OFFLINE Praat
pitch and therefore cannot substantiate streaming latency. No model downloads,
microphone access, or audio playback. English synthetic references only.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ['HF_HUB_OFFLINE']='1'
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch,torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from aligned_analysis import AlignedAnalysis
from streaming_voicebox import VoiceboxConfig
from continuous_source import ContinuousSource,descriptor
from excitation import GeneratedVoicebox,VocalControls,pulse_profile
from benchmark_voicebox import errors,transcribe,f0_track

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/source-ceiling'
FIXTURES=ROOT/'evidence/voice-fixtures'


def analyze(samples,order=20):
    source=AlignedAnalysis(VoiceboxConfig(order=order));frames=[]
    for at in range(0,len(samples),480):frames.extend(source.push(samples[at:at+480]))
    return frames+source.finish()


def measured_source(frame,harmonics):
    """Aligned harmonic shape and estimated harmonic/total residual power.

    Narrowband harmonic sampling is approximate, not an orthogonal separation;
    the ratio is capped and is not a probability or calibrated breathiness.
    """
    if frame.f0<=0:return None,0.
    signal=frame.excitation_window
    window=np.hanning(len(signal));k=np.arange(1,harmonics+1)
    c=2*(np.exp(-2j*np.pi*k[:,None]*frame.f0*np.arange(len(signal))/24000)
         @(window*signal))/sum(window)
    c[k*frame.f0>=24000*.47]=0
    harmonic_power=.5*np.sum(np.abs(c)**2)
    total_power=np.sum(window*signal*signal)/sum(window)
    if harmonic_power<1e-10 or abs(c[0])/np.sqrt(harmonic_power)<.05:return None,0.
    aligned=c*np.exp(-1j*k*np.angle(c[0]))/np.sqrt(harmonic_power)
    return aligned,float(np.clip(harmonic_power/max(total_power,1e-10),0.,.98))


def spectral_error(original,generated):
    """Active-frame mean absolute log-magnitude error; not a perceptual score."""
    rows=[]
    for size in (256,512,1024):
        hop=size//4;window=np.hanning(size)
        def transform(audio):
            padded=np.pad(audio,(size//2,size//2))
            frames=np.lib.stride_tricks.sliding_window_view(padded,size)[::hop]
            return np.abs(np.fft.rfft(frames*window,axis=1)),np.sqrt(np.mean(frames*frames,axis=1))
        truth,energy=transform(original);output,_=transform(generated)
        mask=energy>.003
        frequency=np.fft.rfftfreq(size,1/24000)
        band=(frequency>=65)&(frequency<=10000)
        floor=max(float(truth.max())*10**(-70/20),1e-9)
        difference=np.abs(20*np.log10(np.maximum(truth,floor))-20*np.log10(np.maximum(output,floor)))
        rows.append(float(np.mean(difference[mask][:,band])))
    return {'windowSizes':[256,512,1024],'meanAbsoluteDbByWindow':rows,'meanAbsoluteDb':float(np.mean(rows))}


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--validation',action='store_true');args=parser.parse_args()
    if args.validation:OUT=OUT/'validation'
    OUT.mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4)
    directory=ROOT/'models/speaker-eval'
    evaluator=EncoderClassifier.from_hparams(source=str(directory),savedir=str(directory),
        overrides={'pretrained_path':str(directory)},run_opts={'device':'cpu'})
    def embedding(audio):
        data=torchaudio.functional.resample(torch.from_numpy(audio.astype(np.float32)).unsqueeze(0),24000,16000)
        with torch.inference_mode():value=evaluator.encode_batch(data).flatten()
        return value/torch.linalg.vector_norm(value)
    report={'scope':'Diagnostic representation/synthesis ceiling, not autonomous speech generation',
        'decoder':'Existing generated harmonic/noise voicebox; no original residual in decoder',
        'sourceInputs':'Observed continuous harmonic descriptors; only PCA baseline is compressed',
        'oracle':'Praat uses full original audio to supply pitch/voicing. Oracle is an offline diagnostic, never a streaming result.',
        'metrics':'ASR words, Praat pitch, ECAPA speech cosine, and active-frame log-magnitude spectral error. None is human naturalness acceptance.',
        'validation':args.validation,'results':[]}
    modes=[('pca24',32,20,False,False),('full32',32,20,False,False),
           ('full64',64,20,False,False),('full64-measured-mix',64,20,True,False),
           ('full64-order32',64,32,True,False),('oracle64-order32',64,32,True,True)]
    for speaker in ('af_heart','am_michael'):
        reference,rate=sf.read(FIXTURES/f'{speaker}-reference.wav');assert rate==24000
        anchor=embedding(reference)
        cache=np.load(ROOT/f'evidence/generated-excitation/continuous-source/smooth/{speaker}-source.npz')
        pca=ContinuousSource(cache['mean'],cache['basis'])
        ref_frames={order:analyze(reference,order) for order in (20,32)}
        fallback={}
        for harmonics,order in ((32,20),(64,20),(64,32)):
            rows=[value for f in ref_frames[order] if (value:=descriptor(f,harmonics=harmonics)) is not None]
            mean=np.mean(rows,axis=0);fallback[(harmonics,order)]=mean[:harmonics]+1j*mean[harmonics:]
        if args.validation:
            manifest=json.loads((ROOT/'evidence/streaming-voicebox/validation/manifest.json').read_text())
            cases=[(c['name'],ROOT/c['file'],c['text']) for c in manifest['clips'] if c['speaker']==speaker]
        else:
            path=FIXTURES/f'{speaker}-heldout.wav'
            cases=[(speaker,path,json.loads(path.with_suffix('.json').read_text())['text'])]
        for label,path,text in cases:
            audio,rate=sf.read(path);assert rate==24000
            times,original_f0=f0_track(audio)
            original_vector=embedding(audio);original_identity=float(torch.dot(original_vector,anchor))
            analyzed={order:analyze(audio,order) for order in (20,32)}
            for mode,harmonics,order,use_mix,oracle in modes:
                started=time.perf_counter();encoded=[];mixes=[]
                for source_frame in analyzed[order]:
                    frame=copy.copy(source_frame)
                    if oracle:
                        at=(frame.start+len(frame.residual)/2)/rate
                        frame.f0=float(original_f0[np.argmin(np.abs(times-at))])
                        frame.periodicity=max(frame.periodicity,.65) if frame.f0 else 0.
                    control=VocalControls.from_frame(frame)
                    if mode=='pca24':value=pca.decode(pca.encode(frame));mix=None
                    elif use_mix:
                        value,mix=measured_source(frame,harmonics)
                    else:
                        raw=descriptor(frame,harmonics=harmonics)
                        value=None if raw is None else raw[:harmonics]+1j*raw[harmonics:]
                        mix=None
                    if value is None:value=fallback[(harmonics,order)]
                    encoded.append((control,value,mix));mixes.append(mix)
                encoded_seconds=time.perf_counter()-started
                source=GeneratedVoicebox(profile=pulse_profile(harmonics),config=VoiceboxConfig(order=order))
                outputs=[];durations=[]
                for control,value,mix in encoded:
                    started=time.perf_counter()
                    outputs.append(source.push(control,harmonics=value,voicing=mix))
                    durations.append((time.perf_counter()-started)*1000)
                output=np.concatenate(outputs)
                raw_peak=float(np.max(np.abs(output)))
                gain=min(float(np.sqrt(np.mean(audio**2))/max(1e-9,np.sqrt(np.mean(output**2)))),.88/max(1e-9,raw_peak))
                output*=gain
                saved=OUT/f'{label}-{mode}.wav'
                sf.write(saved,np.rint(output*32768).astype(np.int16),rate,subtype='PCM_16')
                heard=transcribe(saved)
                _,generated_f0=f0_track(output)
                joint=(original_f0>0)&(generated_f0>0)
                pitch_error=np.abs(1200*np.log2(generated_f0[joint]/original_f0[joint]))
                result={'fixture':label,'speaker':speaker,'mode':mode,'file':str(saved.relative_to(ROOT)),
                    'oracleOfflinePitch':oracle,'identityCosine':float(torch.dot(embedding(output),anchor)),
                    'originalIdentityCosine':original_identity,'text':heard,**errors(text,heard),
                    'medianPitchErrorCents':float(np.median(pitch_error)) if any(joint) else None,
                    'voicingAgreement':float(np.mean((original_f0>0)==(generated_f0>0))),
                    'spectral':spectral_error(audio,output),'rawPeak':raw_peak,'auditionGain':gain,
                    'encodingSeconds':encoded_seconds,'decoderMsP95':float(np.percentile(durations,95)),
                    'measuredVoicingPowerMean':float(np.mean([v for v in mixes if v is not None])) if use_mix else None}
                report['results'].append(result)
                (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
                print(json.dumps(result),flush=True)


if __name__=='__main__':main()
