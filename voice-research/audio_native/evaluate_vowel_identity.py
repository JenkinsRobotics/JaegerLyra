"""Speaker-embedding diagnostic with a leave-one-vowel-out teacher baseline.

Sustained synthetic vowels are outside normal speech evaluation conditions.
Scores and nearest-speaker labels are diagnostics, never perceptual acceptance
or a claim of human-reference cloning. Original-teacher failures remain visible.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier
from neural_waveform import load_model, RATE
from neural_performance import load_performance, _sample
from neural_vowel_instrument import VowelScore

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-vowels/v1'


def main():
    source=json.loads((ROOT/'evidence/neural-vowels/references/manifest.json').read_text())
    performances=json.loads((OUT/'report.json').read_text())
    if not performances['complete']:raise ValueError('Finish all vowel renders first')
    torch.set_num_threads(4)
    directory=ROOT/'models/speaker-eval'
    model=EncoderClassifier.from_hparams(source=str(directory),savedir=str(directory),
        overrides={'pretrained_path':str(directory)},run_opts={'device':'cpu'})
    def embed_audio(audio,rate):
        samples=torchaudio.functional.resample(torch.from_numpy(audio.astype(np.float32)).unsqueeze(0),rate,16000)
        with torch.inference_mode():value=model.encode_batch(samples).flatten()
        return value/torch.linalg.vector_norm(value)
    def embed(path):
        audio,rate=sf.read(path)
        return embed_audio(audio,rate)
    original={s['name']:embed(ROOT/s['file']) for s in source['clips']}
    byname={s['name']:s for s in source['clips']}
    speakers=sorted({s['speaker'] for s in source['clips']})
    def compare(vector,name):
        fixture=byname[name];scores={}
        for speaker in speakers:
            references=[original[s['name']] for s in source['clips']
                if s['speaker']==speaker and s['intendedVowel']!=fixture['intendedVowel']]
            anchor=torch.stack(references).mean(0);anchor/=torch.linalg.vector_norm(anchor)
            scores[speaker]=float(torch.dot(vector,anchor))
        ordered=sorted(scores,key=scores.get,reverse=True)
        return {'referenceVowelExcluded':fixture['intendedVowel'], 'cosines':scores,
            'nearestSpeaker':ordered[0], 'matchesSourceSpeaker':ordered[0]==fixture['speaker'],
            'sourceSpeakerCosine':scores[fixture['speaker']],
            'sourceMargin':scores[fixture['speaker']]-max(v for k,v in scores.items() if k!=fixture['speaker']),
            'sameVowelOriginalCosine':float(torch.dot(vector,original[name]))}
    # Preserve the initial evidence before adding matched sustained controls.
    earlier=OUT/'identity.json'
    if earlier.exists() and not (OUT/'identity-initial.json').exists():
        (OUT/'identity-initial.json').write_bytes(earlier.read_bytes())
    report={'scope':'Leave-one-vowel-out speaker embedding diagnostic across three synthetic Kokoro voices',
        'limits':'Vowels are not normal speech; original teacher references are a necessary baseline. Scores do not establish perceived identity or general cloning.',
        'originals':[{'fixture':name,**compare(vector,name)} for name,vector in original.items()],
        'results':[], 'sustainedBaselines':[], 'complete':False}
    for row in performances['results']:
        if row['status']!='ok':
            report['results'].append({'fixture':row['fixture'],'mode':row['mode'],'status':'render-failed'})
            continue
        report['results'].append({'fixture':row['fixture'],'mode':row['mode'],'status':'ok',
            **compare(embed(ROOT/row['file']),row['fixture'])})
        (OUT/'identity.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(row['fixture'],row['mode'],flush=True)
    encoder,decoder,_=load_model(ROOT/performances['model'])
    for row in performances['references']:
        if row['status']!='ok':continue
        name=row['name'];audio,rate=sf.read(ROOT/byname[name]['file']);assert rate==RATE
        patch,_=load_performance(ROOT/row['artifact'])
        crop=audio[round(row['sourceStartSeconds']*RATE):round(row['sourceEndSeconds']*RATE)]
        generated=decoder.render(patch.matrix)
        count=RATE*2;first,last=patch.pulses[[0,-1]]
        positions=first+np.arange(count)%(last-first)
        envelope=VowelScore(seconds=2.).envelope(np.arange(count),count)
        baselines={'whole-source-neural':decoder.render(encoder.encode(audio)),
            'original-patch-loop':_sample(crop,positions)*envelope,
            'neural-patch-loop':_sample(generated,positions)*envelope}
        for mode,samples in baselines.items():
            path=OUT/(name+'-'+mode+'.wav');sf.write(path,samples,RATE,subtype='FLOAT')
            report['sustainedBaselines'].append({'fixture':name,'mode':mode,
                'scope':'Diagnostic only: repeating an unchanged multi-period patch, or decoding the full original phrase; not an alternate accepted voice',
                'file':str(path.relative_to(ROOT)),**compare(embed_audio(samples,RATE),name)})
        (OUT/'identity.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    report['complete']=True
    (OUT/'identity.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
