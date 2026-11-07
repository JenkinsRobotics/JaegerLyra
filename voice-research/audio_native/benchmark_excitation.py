"""Generated-source ablation: words, pitch, identity, timing and latency apart.

Runs silently. Synthetic original references only. ASR, Praat and ECAPA are
evaluation-only, outside the audio-native synthesis path. No human ratings.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
os.environ['HF_HUB_OFFLINE']='1'
import argparse,hashlib,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
import torch,torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from streaming_voicebox import StreamingAnalysis
from aligned_analysis import AlignedAnalysis
from excitation import VocalControls,ReferenceExcitation,GeneratedVoicebox,pulse_profile
from benchmark_voicebox import transcribe,errors,f0_track

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/generated-excitation'
FIXTURES=ROOT/'evidence/voice-fixtures'


def analyze(audio,aligned=False):
    source=AlignedAnalysis() if aligned else StreamingAnalysis()
    frames=[]
    for at in range(0,len(audio),480):frames.extend(source.push(audio[at:at+480]))
    return frames+source.finish()


def synthesize(controls,profile,ratio=1.):
    source=GeneratedVoicebox(profile)
    samples=[];times=[]
    for control in controls:
        start=time.perf_counter()
        samples.append(source.push(control,pitch_ratio=ratio))
        times.append((time.perf_counter()-start)*1000)
    return np.concatenate(samples),times


def main():
    global OUT
    parser=argparse.ArgumentParser();parser.add_argument('--aligned',action='store_true');args=parser.parse_args()
    if args.aligned:OUT=OUT/'aligned'
    OUT.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    model_dir=ROOT/'models/speaker-eval'
    evaluator=EncoderClassifier.from_hparams(source=str(model_dir),savedir=str(model_dir),
        overrides={'pretrained_path':str(model_dir)},run_opts={'device':'cpu'})
    def embedding(audio,rate=24000):
        samples=torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        if rate!=16000:samples=torchaudio.functional.resample(samples,rate,16000)
        with torch.inference_mode():result=evaluator.encode_batch(samples).flatten()
        return result/torch.linalg.vector_norm(result)
    anchors={};profiles={}
    report={'scope':'Analysis/resynthesis from continuous LPC/pitch/energy controls, with generated harmonic/noise excitation; no original PCM residual available to decoder, no linguistic or autonomous speech generation',
        'learning':'Running harmonic power and phase statistics from short reference; no neural model or global backpropagation',
        'evaluation':'Whisper text, Praat pitch and ECAPA cosine are diagnostics, not listener acceptance; ECAPA is not validated for singing',
        'limits':['Filter/content/timing controls still extracted from target input waveform','Reference source shape alone does not imply speaker cloning','Saved auditions use whole-clip RMS matching and peak limiting; not a live loudness controller'],
        'analysisMode':'15ms lookahead, centered window' if args.aligned else '0ms lookahead, trailing window',
        'firstFrameBufferMs':25 if args.aligned else 10,'references':[],'results':[]}
    for speaker in ('af_heart','am_michael'):
        path=FIXTURES/f'{speaker}-reference.wav'
        audio,rate=sf.read(path);assert rate==24000
        started=time.perf_counter();fit=ReferenceExcitation()
        for frame in analyze(audio,args.aligned):fit.observe(frame)
        profile=fit.profile();profiles[speaker]=profile
        (OUT/f'{speaker}-profile.json').write_text(json.dumps(profile,indent=2)+'\n')
        anchors[speaker]=embedding(audio)
        report['references'].append({'speaker':speaker,'file':str(path.relative_to(ROOT)),
            'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'duration':len(audio)/rate,
            'analysisAndFitSeconds':time.perf_counter()-started,'voicedFrames':profile['voicedFrames'],'realProfileScalars':64})
    fixtures=[]
    for speaker in ('af_heart','am_michael'):
        path=FIXTURES/f'{speaker}-heldout.wav'
        fixtures.append((speaker,path,json.loads(path.with_suffix('.json').read_text())['text']))
    fixtures.append(('song',ROOT/'evidence/singer-residency/rendered-song.wav','Little lights are shining stay with me tonight'))
    for label,path,text in fixtures:
        audio,rate=sf.read(path);assert rate==24000
        started=time.perf_counter();frames=analyze(audio,args.aligned);analysis_seconds=time.perf_counter()-started
        controls=[VocalControls.from_frame(frame) for frame in frames]
        # Discard frames/residuals before invoking any generated-source decoder.
        del frames
        matched='af_heart' if label=='song' else label
        reference=profiles[matched]
        _,source_f0=f0_track(audio)
        original_embedding=embedding(audio)
        source_identity={name:float(torch.dot(original_embedding,anchor)) for name,anchor in anchors.items()}
        candidates=[('pulse',pulse_profile(),1.),('reference',reference,1.),('reference-up3',reference,2**.25)]
        if label!='song':candidates.append(('other-source',profiles['am_michael' if label=='af_heart' else 'af_heart'],1.))
        for mode,profile,ratio in candidates:
            generated,times=synthesize(controls,profile,ratio)
            assert len(generated)==len(audio)
            peak=float(np.max(np.abs(generated)))
            gain=min(float(np.sqrt(np.mean(audio**2))/max(1e-9,np.sqrt(np.mean(generated**2)))),.88/max(peak,1e-9))
            generated*=gain
            result_path=OUT/f'{label}-{mode}.wav'
            sf.write(result_path,np.rint(generated*32768).astype(np.int16),rate,subtype='PCM_16')
            heard=transcribe(result_path)
            _,output_f0=f0_track(generated)
            joint=(source_f0>0)&(output_f0>0)
            difference=1200*np.log2(output_f0[joint]/(source_f0[joint]*ratio))
            vector=embedding(generated)
            row={'fixture':label,'mode':mode,'file':str(result_path.relative_to(ROOT)),
                'text':heard,**errors(text,heard),'duration':len(audio)/rate,'pitchRatio':ratio,
                'medianAbsolutePitchErrorCents':float(np.median(np.abs(difference))) if any(joint) else None,
                'p90AbsolutePitchErrorCents':float(np.percentile(np.abs(difference),90)) if any(joint) else None,
                'jointVoicedFrames':int(sum(joint)),'voicingAgreement':float(np.mean((source_f0>0)==(output_f0>0))),
                'identityCosine':{name:float(torch.dot(vector,anchor)) for name,anchor in anchors.items()},
                'originalIdentityCosine':source_identity,'rawPeak':peak,'auditionGain':gain,
                'analysisSeconds':analysis_seconds,'synthesisMsP95':float(np.percentile(times,95)),
                'synthesisRealTimeFactor':sum(times)/1000/(len(audio)/rate),
                'continuousScalarsPer10ms':24,'originalResidualInDecoder':False}
            report['results'].append(row)
            (OUT/'report.json').write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(row),flush=True)


if __name__=='__main__':main()
