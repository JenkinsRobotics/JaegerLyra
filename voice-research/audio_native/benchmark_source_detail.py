"""Isolate resonance motion and noise-color effects on reconstructed speech.

All controls come from observed input audio. This is not autonomous generation,
voice cloning or accepted naturalness. No playback and no microphone access.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ['HF_HUB_OFFLINE']='1'
import argparse,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch,torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from continuous_source import ContinuousSource
from predictive_controls import ControlSpace
from excitation import GeneratedVoicebox,pulse_profile
from noise_shape import noise_descriptor
from lattice_filter import normalized_lattice
from benchmark_free_running import analyze,save_audio,CFG
from streaming_voicebox import LPC_VERSION
from benchmark_voicebox import errors,transcribe,f0_track
from benchmark_vocal_energy import envelope_error

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/source-detail/lpc-v2'
MODES={
    'baseline':{'filter':'direct','smooth':False,'noise':'none'},
    'lattice-static':{'filter':'normalized-lattice','smooth':False,'noise':'none'},
    'lattice-smooth':{'filter':'normalized-lattice','smooth':True,'noise':'none'},
    'colored-noise':{'filter':'direct','smooth':False,'noise':'all'},
    'unvoiced-noise':{'filter':'direct','smooth':False,'noise':'unvoiced'},
    'combined':{'filter':'normalized-lattice','smooth':True,'noise':'all'},
}


def segmented_spectral_error(original,generated,pitch_times,pitch):
    by_size=[]
    for size in (256,512,1024):
        hop=size//4;window=np.hanning(size)
        def transform(audio):
            padded=np.pad(audio,(size//2,size//2))
            frames=np.lib.stride_tricks.sliding_window_view(padded,size)[::hop]
            return np.abs(np.fft.rfft(frames*window,axis=1)),np.sqrt(np.mean(frames**2,axis=1))
        truth,energy=transform(original);output,_=transform(generated)
        times=np.arange(len(truth))*hop/24000
        positions=np.searchsorted(pitch_times,times)
        positions=np.clip(positions,0,len(pitch_times)-1)
        left=np.maximum(positions-1,0)
        positions=np.where(abs(times-pitch_times[left])<abs(times-pitch_times[positions]),left,positions)
        voiced=pitch[positions]>0
        active=energy>.003
        frequency=np.fft.rfftfreq(size,1/24000);band=(frequency>=65)&(frequency<=10000)
        floor=max(float(truth.max())*10**(-70/20),1e-9)
        difference=np.abs(20*np.log10(np.maximum(truth,floor)/np.maximum(output,floor)))[:,band]
        row={'window':size}
        for name,mask in (('all',active),('voiced',active&voiced),('unvoiced',active&~voiced)):
            row[name]=float(np.mean(difference[mask])) if mask.any() else None
            row[name+'Frames']=int(mask.sum())
        by_size.append(row)
    result={'byWindow':by_size,'classification':'Nearest offline Praat voicing frame; evaluation only'}
    for name in ('all','voiced','unvoiced'):
        values=[r[name] for r in by_size if r[name] is not None]
        result[name]=float(np.mean(values)) if values else None
    return result


def render_detail(space,frames,features,noise,mode):
    sink=GeneratedVoicebox(pulse_profile(64),CFG,seed=79,filter_mode=mode['filter'],
                           smooth_filter=mode['smooth'],shape_noise=mode['noise']!='none')
    output=[];times=[]
    for frame,feature,color in zip(frames,features,noise):
        started=time.perf_counter()
        control,harmonics,mix=space.decode(feature,frame.start,len(frame.residual))
        bands=None
        if mode['noise']!='none':
            bands=np.zeros_like(color) if mode['noise']=='unvoiced' and frame.f0 else color
        output.append(sink.push(control,harmonics=harmonics,voicing=mix,noise_bands=bands))
        times.append((time.perf_counter()-started)*1000)
    return np.concatenate(output),times


def main():
    global OUT
    parser=argparse.ArgumentParser();sets=parser.add_mutually_exclusive_group()
    sets.add_argument('--fresh',action='store_true');sets.add_argument('--human',action='store_true')
    parser.add_argument('--modes',nargs='+',choices=MODES,default=list(MODES))
    args=parser.parse_args()
    if args.fresh:OUT=OUT/'fresh'
    if args.human:OUT=OUT/'human'
    OUT.mkdir(exist_ok=True,parents=True);torch.set_num_threads(4)
    started=time.perf_counter()
    normalized_lattice(np.zeros(1),np.zeros(32),np.zeros(32))
    normalized_lattice(np.zeros(1),np.zeros(32),np.zeros(32),previous_reflection=np.zeros(32))
    jit_ms=(time.perf_counter()-started)*1000
    refs=[]
    for voice in ('af_heart','am_michael'):
        audio,sr=sf.read(ROOT/f'evidence/voice-fixtures/{voice}-reference.wav');assert sr==24000
        refs+=analyze(audio)
    source,count=ContinuousSource.fit(refs,dimensions=48,harmonics=64)
    space=ControlSpace(source)
    directory=ROOT/'models/speaker-eval'
    evaluator=EncoderClassifier.from_hparams(source=str(directory),savedir=str(directory),
        overrides={'pretrained_path':str(directory)},run_opts={'device':'cpu'})
    def embedding(audio):
        data=torchaudio.functional.resample(torch.from_numpy(audio.astype(np.float32)).unsqueeze(0),24000,16000)
        with torch.inference_mode():value=evaluator.encode_batch(data).flatten()
        return value/torch.linalg.vector_norm(value)
    if args.fresh or args.human:
        kind='human' if args.human else 'fresh'
        cases=json.loads((ROOT/f'evidence/source-detail/{kind}-inputs/manifest.json').read_text())['clips']
    else:
        cases=[]
        for voice in ('af_heart','am_michael'):
            path=ROOT/f'evidence/voice-fixtures/{voice}-heldout.wav'
            cases.append({'name':voice,'speaker':voice,'file':str(path.relative_to(ROOT)),
                          'text':json.loads(path.with_suffix('.json').read_text())['text']})
        cases+=json.loads((ROOT/'evidence/streaming-voicebox/validation/manifest.json').read_text())['clips']
    report={'scope':'Observed-control reconstruction, not autonomous speech or naturalness acceptance',
        'fresh':args.fresh or args.human,'analysisVersion':LPC_VERSION,
        'fixtureOrigin':'Public LibriSpeech dev-clean evaluation subset' if args.human else 'Synthetic Kokoro speech',
        'basisTraining':'Only original Heart/Michael reference clips; frozen PCA weights',
        'basisVoicedFrames':count,'modes':{k:MODES[k] for k in args.modes},'expectedCases':len(cases)*len(args.modes),
        'controls':'Original excitation RMS; same 84 continuous controls plus eight noise-color values only when enabled',
        'noiseAnalysis':'40 ms existing residual window, harmonic-neighbor exclusion with broad-band fallback; approximate color, not calibrated aperiodicity',
        'noiseBoundsDb':[-12,12],'noiseFirTaps':64,'noiseHistorySamples':63,
        'filterInterpolation':'Optional linear ramp in scattering angles from preceding to current frame; may shift articulation within the 10 ms frame',
        'lookaheadMs':15,'auditionGain':.5,'jitPreparationAfterImportsMs':jit_ms,
        'timingScope':'Warm control decode and synthesis, including noise-filter construction; analysis/encoding measured separately; excludes devices, transport and learned content generation',
        'metrics':'ASR, ECAPA speech cosine, spectral/envelope/pitch diagnostics; no listener acceptance',
        'inputs':[],'results':[]}
    anchors={}
    for case in cases:
        original,rate=sf.read(ROOT/case['file']);assert rate==24000
        anchor_file=(ROOT/case['reference'] if 'reference' in case else
                     ROOT/f'evidence/voice-fixtures/{case["speaker"]}-reference.wav'
                     if case['speaker']!='af_bella' else
                     ROOT/f'evidence/streaming-voicebox/validation/af_bella-{1-int(case["name"][-1])}.wav')
        if anchor_file not in anchors:
            audio,rate=sf.read(anchor_file);assert rate==24000;anchors[anchor_file]=embedding(audio)
        anchor=anchors[anchor_file]
        original_identity=float(torch.dot(embedding(original),anchor))
        pitch_times,original_pitch=f0_track(original)
        started=time.perf_counter();frames=analyze(original)
        analysis_seconds=time.perf_counter()-started
        started=time.perf_counter();features=np.array([space.encode(f) for f in frames])
        base_encoding=time.perf_counter()-started
        started=time.perf_counter();noise=np.array([noise_descriptor(f) for f in frames])
        noise_encoding=time.perf_counter()-started
        original_heard=transcribe(ROOT/case['file']) if args.fresh or args.human else None
        report['inputs'].append({**case,'anchor':str(anchor_file.relative_to(ROOT)),
            'analysisSeconds':analysis_seconds,'controlEncodingSeconds':base_encoding,
            'noiseEncodingSeconds':noise_encoding,'durationSeconds':len(original)/24000,
            'originalIdentityCosine':original_identity,
            'originalRecognition':{'text':original_heard,**errors(case['text'],original_heard)} if original_heard is not None else None})
        for label in args.modes:
            row={'fixture':case['name'],'mode':label,'source':case['file']}
            try:
                generated,times=render_detail(space,frames,features,noise,MODES[label])
                path=OUT/f'{case["name"]}-{label}.wav';save_audio(path,generated)
                text=transcribe(path);_,pitch=f0_track(generated);joint=(original_pitch>0)&(pitch>0)
                cents=np.abs(1200*np.log2(pitch[joint]/original_pitch[joint]))
                row.update({'status':'ok','file':str(path.relative_to(ROOT)),'text':text,**errors(case['text'],text),
                    'identityCosine':float(torch.dot(embedding(generated),anchor)),
                    'originalIdentityCosine':original_identity,
                    'spectral':segmented_spectral_error(original,generated,pitch_times,original_pitch),
                    'envelope':envelope_error(original,generated),
                    'medianPitchErrorCents':float(np.median(cents)) if len(cents) else None,
                    'pitchVoicingAgreement':float(np.mean((original_pitch>0)==(pitch>0))),
                    'rawPeak':float(abs(generated).max()),'decoderMsP95':float(np.percentile(times,95)),
                    'decoderRealTimeFactor':sum(times)/1000/(len(original)/24000)})
                if label=='baseline' and not (args.fresh or args.human):
                    before,_=sf.read(ROOT/f'evidence/vocal-energy/{case["name"]}-residual-energy.wav')
                    after,_=sf.read(path)
                    row['previousBaselinePcm24Identical']=bool(np.array_equal(before,after))
                    row['previousBaselineMaximumSavedSampleDifference']=float(abs(before-after).max())
            except ValueError as error:
                row.update({'status':'failed','error':str(error)})
            report['results'].append(row)
            (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ('spectral','envelope','file','source','text')}),flush=True)


if __name__=='__main__':main()
