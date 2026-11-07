"""Control-only neural waveform reconstruction: notes/vowels, then held-out speech.

Controls are fitted/encoded from each test signal. Authored note/vowel tests
therefore measure decoder capacity, NOT learned pitch/articulator disentanglement.
Original PCM is used by the encoder and evaluator only, never neural decoding.
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
from scipy.signal import butter,sosfilt
from speechbrain.inference.classifiers import EncoderClassifier
from neural_waveform import load_model,ControlMatrix,overlap_add,RATE
from lab import voicebox
from benchmark_free_running import save_audio
from benchmark_voicebox import errors,transcribe,f0_track
from benchmark_source_ceiling import spectral_error

ROOT=Path(__file__).resolve().parents[2]


def authored_signals():
    size=36000;t=np.arange(size)/RATE
    fade=np.minimum(np.minimum(t/.03,(t[-1]-t)/.03),1.)
    notes={name:np.full(size,f0) for name,f0 in (('note-110',110.),('note-220',220.),('note-440',440.))}
    notes['glide']=np.exp(np.linspace(np.log(110),np.log(440),size))
    notes['vibrato']=220.*2**(.3*np.sin(2*np.pi*5*t)/12)
    result=[]
    for name,f0 in notes.items():
        audio=.15*np.sin(2*np.pi*np.cumsum(f0)/RATE)*fade
        result.append((name,audio,f0))
    for name,formants in (('vowel-open',[700.,1100.,2500.]),('vowel-front',[300.,2300.,3000.]),('vowel-rounded',[350.,850.,2200.])):
        f0=np.full(size,160.)
        result.append((name,voicebox(f0,np.tile(formants,(size,1)),breath=.01),f0))
    noise=sosfilt(butter(4,[4000,9000],btype='bandpass',fs=RATE,output='sos'),np.random.default_rng(79).normal(size=size))
    envelope=np.clip((t-.2)/.03,0,1)*np.clip((1.3-t)/.03,0,1)
    result.append(('noise-burst',noise*envelope*.1,None))
    result.append(('two-note',.075*(np.sin(2*np.pi*220*t)+np.sin(2*np.pi*330*t))*fade,None))
    return result


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--model',default='evidence/neural-waveform/d256-h768')
    args=parser.parse_args();directory=ROOT/args.model;out=directory/'evaluation';out.mkdir(exist_ok=True,parents=True)
    (out/'controls').mkdir(exist_ok=True);(out/'authored').mkdir(exist_ok=True)
    encoder,ridge,_=load_model(directory/'neural-ridge.npz')
    pc_encoder,pc,_=load_model(directory/'neural-local-pc.npz')
    assert np.array_equal(encoder.basis,pc_encoder.basis)
    models={'pca-ceiling':None,'neural-ridge':ridge,'neural-local-pc':pc}
    torch.set_num_threads(4);modeldir=ROOT/'models/speaker-eval'
    evaluator=EncoderClassifier.from_hparams(source=str(modeldir),savedir=str(modeldir),
        overrides={'pretrained_path':str(modeldir)},run_opts={'device':'cpu'})
    def embedding(audio):
        data=torchaudio.functional.resample(torch.from_numpy(audio.astype(np.float32)).unsqueeze(0),RATE,16000)
        with torch.inference_mode():value=evaluator.encode_batch(data).flatten()
        return value/torch.linalg.vector_norm(value)
    cases=[]
    for name,audio,f0 in authored_signals():
        path=out/'authored'/(name+'.wav');sf.write(path,audio,RATE,subtype='PCM_24')
        cases.append({'name':name,'kind':'authored','file':str(path.relative_to(ROOT)),'authoredF0':f0})
    for folder,kind in (('fresh-inputs','synthetic'),('human-inputs','human')):
        for item in json.loads((ROOT/f'evidence/source-detail/{folder}/manifest.json').read_text())['clips']:
            cases.append({**item,'name':kind+'-'+item['name'],'kind':kind})
    for speaker in ('af_heart','am_michael'):
        path=ROOT/f'evidence/voice-fixtures/{speaker}-heldout.wav'
        cases.append({'name':'original-'+speaker,'kind':'synthetic','speaker':speaker,
            'file':str(path.relative_to(ROOT)),'reference':f'evidence/voice-fixtures/{speaker}-reference.wav',
            'text':json.loads(path.with_suffix('.json').read_text())['text']})
    report={'scope':'Neural decoder capacity from target-encoded continuous controls; not autonomous speech, cloning or learned articulator control',
        'modelDirectory':args.model,'trainingAudioExcluded':True,'models':list(models),
        'controlsPerFrame':len(encoder.basis)+1,'frameMs':20,'hopMs':10,
        'controlMeaning':'Learned acoustic basis coordinates plus gain; not labelled/disentangled phonotypes',
        'decoderInputs':'Saved continuous controls, gains and output duration only',
        'auditionGain':.5,'expectedCases':len(cases)*len(models),
        'timingScope':'Offline encoding and decoder compute, excluding transport, devices and any control-generating brain',
        'pitch':'Praat against authored F0 for single-note tests; original detected F0 for speech. No unique fundamental assigned to two-note/noise controls.',
        'originals':[],'results':[]}
    anchors={}
    for case in cases:
        original,rate=sf.read(ROOT/case['file']);assert rate==RATE
        started=time.perf_counter();matrix=encoder.encode(original);encoding=time.perf_counter()-started
        control_path=out/'controls'/(case['name']+'.npz')
        np.savez(control_path,values=matrix.values,gains=matrix.gains,sample_count=matrix.sample_count)
        # Rendering below consumes the reloaded control artifact, not audio.
        with np.load(control_path,allow_pickle=False) as data:
            matrix=ControlMatrix(data['values'],data['gains'],int(data['sample_count']))
        pitch_times,original_pitch=f0_track(original)
        expected_pitch=original_pitch
        if case.get('authoredF0') is not None:
            expected_pitch=np.interp(pitch_times,np.arange(len(original))/RATE,case['authoredF0'])
            expected_pitch[(pitch_times<.05)|(pitch_times>(len(original)/RATE-.05))]=0
        identity=None
        if case['kind']!='authored':
            ref=case['reference']
            if ref not in anchors:
                audio,rate=sf.read(ROOT/ref);assert rate==RATE;anchors[ref]=embedding(audio)
            identity=anchors[ref]
            heard=transcribe(ROOT/case['file'])
        report['originals'].append({'fixture':case['name'],'kind':case['kind'],'file':case['file'],
            'controlFile':str(control_path.relative_to(ROOT)),'encodingSeconds':encoding,
            'originalRecognition':{'text':heard,**errors(case['text'],heard)} if identity is not None else None,
            'originalIdentityCosine':float(torch.dot(embedding(original),identity)) if identity is not None else None})
        for label,decoder in models.items():
            row={'fixture':case['name'],'kind':case['kind'],'mode':label}
            try:
                started=time.perf_counter()
                generated=encoder.reconstruct_linear(matrix) if decoder is None else decoder.render(matrix)
                elapsed=time.perf_counter()-started
                path=out/(case['name']+'-'+label+'.wav');save_audio(path,generated)
                _,pitch=f0_track(generated);joint=(pitch>0)&(expected_pitch>0)
                cents=np.abs(1200*np.log2(pitch[joint]/expected_pitch[joint]))
                signal=float(np.sum(original**2));error=float(np.sum((generated-original)**2))
                row.update({'status':'ok','file':str(path.relative_to(ROOT)),
                    'waveformSnrDb':float(10*np.log10(max(signal,1e-20)/max(error,1e-20))),
                    'spectralErrorDb':spectral_error(original,generated)['meanAbsoluteDb'],
                    'medianPitchErrorCents':float(np.median(cents)) if len(cents) and not (case['kind']=='authored' and case.get('authoredF0') is None) else None,
                    'pitchVoicingAgreement':float(np.mean((pitch>0)==(expected_pitch>0))),
                    'rawPeak':float(abs(generated).max()),'decoderSeconds':elapsed,
                    'decoderRealTimeFactor':elapsed/(len(original)/RATE)})
                if identity is not None:
                    text=transcribe(path);row.update({'text':text,**errors(case['text'],text),
                        'identityCosine':float(torch.dot(embedding(generated),identity))})
            except ValueError as error:
                row.update({'status':'failed','error':str(error)})
            report['results'].append(row)
            (out/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ('file','text')}),flush=True)


if __name__=='__main__':main()
