"""Frozen output candidate vs equal-training random decoder and linear reference."""
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
from neural_waveform import load_model,ControlMatrix
from benchmark_neural_waveform import authored_signals
from benchmark_free_running import save_audio
from benchmark_voicebox import errors,transcribe,f0_track
from benchmark_source_ceiling import spectral_error
from develop_neural_fidelity import snr

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-waveform/axis-v1/evaluation'


def main():
    global OUT
    parser=argparse.ArgumentParser()
    parser.add_argument('--manifest',default='evidence/neural-waveform/output-validation/manifest.json')
    parser.add_argument('--output',default='evidence/neural-waveform/axis-v1/evaluation')
    parser.add_argument('--baseline',default='evidence/neural-waveform/fidelity-v2/d256-h768-g0.7-r0.001.npz')
    parser.add_argument('--baseline-label',default='random-neural')
    parser.add_argument('--candidate-label',default='axis-neural')
    args=parser.parse_args();OUT=ROOT/args.output
    if args.baseline_label==args.candidate_label or 'linear-reference' in (args.baseline_label,args.candidate_label):
        raise ValueError('Distinct model labels required')
    manifest=json.loads((ROOT/args.manifest).read_text())
    if len(manifest['clips'])!=manifest.get('expectedClips',9):raise ValueError('Validation corpus incomplete')
    candidate=ROOT/manifest['candidate']
    if hashlib.sha256(candidate.read_bytes()).hexdigest()!=manifest['candidateSha256']:
        raise ValueError('Candidate changed after final evaluation was defined')
    if manifest.get('candidateType')=='band-preserving-neural':
        from band_preserving_decoder import load_band_preserving
        axis_encoder,axis,_=load_band_preserving(candidate)
    else:
        axis_encoder,axis,_=load_model(candidate)
    baseline=ROOT/args.baseline
    old_encoder,old,_=load_model(baseline)
    models={args.baseline_label:(old_encoder,old),'linear-reference':(axis_encoder,None),args.candidate_label:(axis_encoder,axis)}
    for sub in ('controls','authored'): (OUT/sub).mkdir(exist_ok=True,parents=True)
    torch.set_num_threads(4);directory=ROOT/'models/speaker-eval'
    evaluator=EncoderClassifier.from_hparams(source=str(directory),savedir=str(directory),
        overrides={'pretrained_path':str(directory)},run_opts={'device':'cpu'})
    def embedding(audio):
        tensor=torchaudio.functional.resample(torch.from_numpy(audio.astype(np.float32)).unsqueeze(0),24000,16000)
        with torch.inference_mode():value=evaluator.encode_batch(tensor).flatten()
        return value/torch.linalg.vector_norm(value)
    cases=[{**c,'name':'fresh-'+c['name'],'kind':c.get('kind','fresh-kokoro')} for c in manifest['clips']]
    for c in json.loads((ROOT/'evidence/source-detail/human-inputs/manifest.json').read_text())['clips']:
        cases.append({**c,'name':'human-'+c['name'],'kind':'human-regression'})
    for name,audio,f0 in authored_signals():
        path=OUT/'authored'/(name+'.wav');sf.write(path,audio,24000,subtype='PCM_24')
        cases.append({'name':name,'kind':'authored','file':str(path.relative_to(ROOT)),'authoredF0':f0})
    trained=json.loads((ROOT/'evidence/neural-waveform/axis-v1/development.json').read_text())['corpus']
    training_hashes={c['sha256'] for c in trained}
    report={'scope':'Output reconstruction; controls are encoded from complete targets, not generated content',
        'candidate':manifest['candidate'],'candidateSha256':manifest['candidateSha256'],
        'candidateType':manifest.get('candidateType','single-neural-decoder'),
        'baseline':str(baseline.relative_to(ROOT)),'baselineSha256':hashlib.sha256(baseline.read_bytes()).hexdigest(),
        'trainingComparison':'Both neural decoders fitted the same 20 teacher clips; development excluded from fitting',
        'freshEvaluation':manifest.get('evaluationDescription','Nine new Kokoro recordings created after development selection; human panel is reused regression'),
        'models':list(models),'auditionGain':.5,'expectedCases':len(cases)*len(models),
        'timingScope':'Decoder compute only, not control analysis, transport, streaming scheduling or audio devices',
        'qualityAccepted':False,'originals':[],'results':[],'complete':False}
    anchors={}
    for case in cases:
        path=ROOT/case['file'];digest=hashlib.sha256(path.read_bytes()).hexdigest()
        if digest in training_hashes:raise ValueError('Evaluation audio overlaps fitting/development corpus')
        if case.get('sha256',digest)!=digest:raise ValueError('Evaluation audio changed')
        original,rate=sf.read(path)
        if rate!=24000:raise ValueError('Expected 24 kHz')
        times,expected_pitch=f0_track(original)
        if case.get('authoredF0') is not None:
            expected_pitch=np.interp(times,np.arange(len(original))/rate,case['authoredF0'])
            expected_pitch[(times<.05)|(times>len(original)/rate-.05)]=0
        identity=None;original_heard=None
        if case['kind']!='authored':
            ref=case['reference']
            if ref not in anchors:
                ref_audio,ref_rate=sf.read(ROOT/ref)
                if ref_rate!=rate:raise ValueError('Expected same reference rate')
                anchors[ref]=embedding(ref_audio)
            identity=anchors[ref];original_heard=transcribe(path)
        report['originals'].append({'fixture':case['name'],'kind':case['kind'],'file':case['file'],'sha256':digest,
            'recognition':{'text':original_heard,**errors(case['text'],original_heard)} if identity is not None else None,
            'identityCosine':float(torch.dot(embedding(original),identity)) if identity is not None else None})
        for mode,(encoder,decoder) in models.items():
            control_file=OUT/'controls'/(case['name']+'-'+mode+'.npz')
            encoded=encoder.encode(original)
            np.savez(control_file,values=encoded.values,gains=encoded.gains,sample_count=encoded.sample_count)
            del encoded
            with np.load(control_file,allow_pickle=False) as saved:
                matrix=ControlMatrix(saved['values'],saved['gains'],int(saved['sample_count']))
            row={'fixture':case['name'],'kind':case['kind'],'mode':mode}
            try:
                started=time.perf_counter()
                audio=encoder.reconstruct_linear(matrix) if decoder is None else decoder.render(matrix)
                elapsed=time.perf_counter()-started;wav=OUT/(case['name']+'-'+mode+'.wav');save_audio(wav,audio)
                _,pitch=f0_track(audio);joint=(pitch>0)&(expected_pitch>0)
                cents=abs(1200*np.log2(pitch[joint]/expected_pitch[joint]))
                row.update({'status':'ok','file':str(wav.relative_to(ROOT)),'waveformSnrDb':snr(original,audio),
                    'spectralErrorDb':spectral_error(original,audio)['meanAbsoluteDb'],'rawPeak':float(abs(audio).max()),
                    'decoderSeconds':elapsed,'decoderRealTimeFactor':elapsed/(len(original)/rate),
                    'medianPitchErrorCents':float(np.median(cents)) if len(cents) and not
                        (case['kind']=='authored' and case.get('authoredF0') is None) else None,
                    'pitchVoicingAgreement':float(np.mean((pitch>0)==(expected_pitch>0)))})
                if identity is not None:
                    heard=transcribe(wav);row.update({'text':heard,**errors(case['text'],heard),
                        'identityCosine':float(torch.dot(embedding(audio),identity))})
            except ValueError as error:
                row.update({'status':'failed','error':str(error)})
            report['results'].append(row)
            (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(json.dumps({k:v for k,v in row.items() if k not in ('file','text')}),flush=True)
    report['complete']=True
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
