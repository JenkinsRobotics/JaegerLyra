"""Pretrained evaluator only: identity diagnostics, not synthesis or acceptance."""
import os
os.environ['HF_HUB_OFFLINE']='1'
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
OUT=ROOT/'evidence/resonance-morph/v1'
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    refs=json.loads((OUT/'references.json').read_text());render=json.loads((OUT/'render.json').read_text())
    oracle=json.loads((OUT/'oracle-envelope.json').read_text()) if (OUT/'oracle-envelope.json').exists() else {'results':[],'complete':True}
    assert oracle['complete']
    assert refs['complete'] and render['complete']
    directory=ROOT/'models/speaker-eval'
    model_hashes={p.name:sha(p) for p in directory.glob('*.ckpt')}
    earlier=json.loads((ROOT/'evidence/neural-syllables/grain-windows-v1/identity-evaluation.json').read_text())
    if model_hashes!=earlier['modelFiles']:raise ValueError('Changed evaluation model')
    torch.set_num_threads(4)
    model=EncoderClassifier.from_hparams(source=str(directory),savedir=str(directory),
        overrides={'pretrained_path':str(directory)},run_opts={'device':'cpu'})
    cache=OUT/'embeddings';cache.mkdir(exist_ok=True)
    def embed(file):
        file=ROOT/file;key=sha(file);saved=cache/(key+'.npy')
        if saved.exists():return np.load(saved,allow_pickle=False)
        audio,rate=sf.read(file);samples=torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
        samples=torchaudio.functional.resample(samples,rate,16000)
        with torch.inference_mode():vector=model.encode_batch(samples).flatten()
        vector=(vector/torch.linalg.vector_norm(vector)).numpy();np.save(saved,vector)
        return vector
    training=[r for r in refs['clips'] if r['split']=='train']
    vectors={r['name']:embed(r['file']) for r in training}
    speakers=sorted({r['speaker'] for r in training})
    anchors={}
    for group in sorted({r['group'] for r in training}):
        for speaker in speakers:
            vector=np.mean([vectors[r['name']] for r in training if r['group']!=group and r['speaker']==speaker],axis=0)
            anchors[(group,speaker)]=vector/np.linalg.norm(vector)
    report={'scope':'Leave-diphthong-group-out ECAPA diagnostics over three synthetic identities',
        'modelFiles':model_hashes,'results':[],'complete':False,
        'limits':['Short vowel nuclei are outside normal speech evaluation; these are not naturalness or perceived-identity ratings.',
            'Only two different development diphthongs per identity supply each anchor. The evaluated group and all held-out words are excluded.',
            'Oracle target pitch, duration and amplitude are provided to every renderer. No general voice cloning claim.']}
    for reference in render['references']:
        original=embed(reference['nucleus'])
        items=[{'method':'teacher','file':reference['nucleus']},{'method':'reconstruction','file':reference['reconstruction']}]+[r for r in render['results'] if r['name']==reference['name']]
        items += [r for r in oracle['results'] if r['name']==reference['name']]
        for item in items:
            vector=original if item['method']=='teacher' else embed(item['file'])
            scores={s:float(np.dot(vector,anchors[(reference['group'],s)])) for s in speakers}
            nearest=max(scores,key=scores.get)
            report['results'].append({'name':reference['name'],'speaker':reference['speaker'],'group':reference['group'],
                'method':item['method'],'nearestSpeaker':nearest,'matchesSourceSpeaker':nearest==reference['speaker'],
                'cosines':scores,'sourceSpeakerCosine':scores[reference['speaker']],
                'sameNucleusCosine':float(np.dot(original,vector))})
        (OUT/'identity.json').write_text(json.dumps(report,indent=2)+'\n');print(reference['name'],flush=True)
    report['complete']=True;(OUT/'identity.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
