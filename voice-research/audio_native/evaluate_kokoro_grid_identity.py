"""Independent speaker-embedding and boundary diagnostics for grid cancellation.

ECAPA is a pretrained evaluator, never a synthesis input. Sustained synthetic
words are outside its ordinary speech setting; no perceptual pass threshold.
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/kokoro-control-voicebox/upsampler-grid-v1'
VOWEL_GROUP={'calm':'ah','palm':'ah','paw':'aw','raw':'aw','blue':'oo','true':'oo'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    study=json.loads((OUT/'report.json').read_text())
    if not study['complete']:raise ValueError('Incomplete study')
    torch.set_num_threads(4);directory=ROOT/'models/speaker-eval'
    model=EncoderClassifier.from_hparams(source=str(directory),savedir=str(directory),
        overrides={'pretrained_path':str(directory)},run_opts={'device':'cpu'})
    cache=OUT/'speaker-embeddings';cache.mkdir(exist_ok=True)
    def embedding(row):
        file=ROOT/row['file']
        if sha(file)!=row['sha256']:raise ValueError('Changed audio')
        stored=cache/(row['sha256']+'.npy')
        earlier_cache=ROOT/'evidence/kokoro-control-voicebox/upsampler-grid-v1/speaker-embeddings'/stored.name
        if stored.exists():value=np.load(stored,allow_pickle=False)
        elif earlier_cache.exists():
            value=np.load(earlier_cache,allow_pickle=False);np.save(stored,value)
        else:
            audio,rate=sf.read(file);samples=torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            samples=torchaudio.functional.resample(samples,rate,16000)
            with torch.inference_mode():vector=model.encode_batch(samples).flatten()
            value=(vector/torch.linalg.vector_norm(vector)).numpy();np.save(stored,value)
        if value.ndim!=1 or not np.isfinite(value).all() or abs(np.linalg.norm(value)-1)>.0001:
            raise ValueError('Invalid saved embedding')
        return value
    originals={s['name']:embedding(s) for s in study['sources']}
    speakers=sorted({s['speaker'] for s in study['sources']})
    def compare(row,vector):
        excluded=VOWEL_GROUP[row['word']];scores={}
        for speaker in speakers:
            references=[originals[s['name']] for s in study['sources']
                if s['speaker']==speaker and VOWEL_GROUP[s['word']]!=excluded]
            anchor=np.mean(references,axis=0);anchor/=np.linalg.norm(anchor)
            scores[speaker]=float(np.dot(vector,anchor))
        nearest=max(scores,key=scores.get)
        source=originals[row['speaker']+'-'+row['word']]
        return {'referenceVowelClassExcluded':excluded,'cosines':scores,'nearestSpeaker':nearest,
            'matchesSourceSpeaker':nearest==row['speaker'],'sourceSpeakerCosine':scores[row['speaker']],
            'sourceSpeakerMargin':scores[row['speaker']]-max(v for k,v in scores.items() if k!=row['speaker']),
            'sameWordOriginalCosine':float(np.dot(vector,source))}
    report={'scope':'Leave-vowel-class-out ECAPA diagnostic across three existing synthetic voices, plus outside-hold waveform comparisons',
        'model':'speechbrain/spkrec-ecapa-voxceleb, local models/speaker-eval',
        'modelFiles':{p.name:sha(p) for p in directory.glob('*.ckpt')},
        'sources':[],'results':[],'boundaryComparisons':[],'complete':False,
        'limits':['Speech-trained embeddings are not validated here for sustained vowels or singing and are not naturalness scores.',
            'Reference anchors exclude both words from the same intended vowel class; original-source results remain the baseline.',
            'Three synthetic voices do not establish general human-reference cloning or perceived identity.',
            'Correction is gated in intermediate features; later convolutions may affect waveform outside those exact feature boundaries.',
            'Boundary comparisons use a disclosed 200 ms guard and do not prove absence of audible transition changes.']}
    for source in study['sources']:
        report['sources'].append({'name':source['name'],'speaker':source['speaker'],'word':source['word'],
            'file':source['file'],'sha256':source['sha256'],**compare(source,originals[source['name']])})
    for row in study['results']:
        report['results'].append({k:row[k] for k in ('name','speaker','word','mode','method','file','sha256')}|
            compare(row,embedding(row)))
        if row['method']!='baseline':
            baseline=next(r for r in study['results'] if r['method']=='baseline' and
                (r['speaker'],r['word'],r['mode'])==(row['speaker'],row['word'],row['mode']))
            a,rate=sf.read(ROOT/baseline['file']);b,rb=sf.read(ROOT/row['file']);assert rate==rb==24000
            time=np.arange(len(a))/rate;outside=(time<row['plan']['holdStartSeconds']-.2)|(time>row['plan']['holdEndSeconds']+.2)
            difference=b[outside]-a[outside]
            report['boundaryComparisons'].append({'name':row['name'],'guardSeconds':.2,'outsideSamples':int(sum(outside)),
                'maximumDifference':float(max(abs(difference))),'rmsDifference':float(np.sqrt(np.mean(difference**2)))})
        (OUT/'identity-evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(row['name'],round(report['results'][-1]['sourceSpeakerCosine'],4),report['results'][-1]['nearestSpeaker'],flush=True)
    report['groups']=[]
    methods=list(dict.fromkeys(r['method'] for r in study['results']))
    for method,rows in [('source',report['sources'])]+[(m,[r for r in report['results'] if r['method']==m]) for m in methods]:
        report['groups'].append({'method':method,'cases':len(rows),'nearestSpeakerMatches':sum(r['matchesSourceSpeaker'] for r in rows),
            'medianSourceSpeakerCosine':float(np.median([r['sourceSpeakerCosine'] for r in rows])),
            'medianSourceSpeakerMargin':float(np.median([r['sourceSpeakerMargin'] for r in rows])),
            'medianSameWordOriginalCosine':float(np.median([r['sameWordOriginalCosine'] for r in rows]))})
    report['complete']=(len(report['sources'])==len(study['sources']) and len(report['results'])==len(study['results'])
        and len(report['boundaryComparisons'])==sum(r['method']!='baseline' for r in study['results']))
    (OUT/'identity-evaluation.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(report['groups'],flush=True)


if __name__=='__main__':
    import sys
    if len(sys.argv)>1:OUT=ROOT/sys.argv[1]
    main()
