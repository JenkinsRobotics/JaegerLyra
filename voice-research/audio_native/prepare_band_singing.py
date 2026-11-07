"""Fresh synthetic singing fixtures after freezing the complementary decoder."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import hashlib,json,sys,wave
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'evidence/neural-waveform/band-validation'


def main():
    development=json.loads((ROOT/'evidence/neural-waveform/band-v4/development.json').read_text())
    candidate=ROOT/development['candidate']
    if not development['complete'] or hashlib.sha256(candidate.read_bytes()).hexdigest()!=development['candidateSha256']:
        raise ValueError('Freeze the assembly before generating final fixtures')
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    from kokoro_singing import generate
    mx.set_cache_limit(128*1024*1024)
    model=load_model(ROOT/'models/expressive/Kokoro-82M-bf16')
    cases=[('af_heart','Silver clouds are drifting softly by'),
        ('am_michael','Hold a little light beside your heart'),
        ('af_bella','We will find a brighter morning soon')]
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Fresh reconstruction-only singing evaluation; synthetic score-conditioned Kokoro adapter, not a singing-trained model',
        'candidate':development['candidate'],'candidateSha256':development['candidateSha256'],
        'adapterSha256':hashlib.sha256((ROOT/'kokoro_singing.py').read_bytes()).hexdigest(),
        'clips':[],'complete':False}
    for speaker,line in cases:
        name=speaker+'-new-song';path=OUT/(name+'.wav');score_path=OUT/(name+'.score.json')
        if not path.exists():
            audio,score=generate({'speaker':speaker,'lyrics':line,'tempo':104,'melody':'glow'},model=model)
            if not np.isfinite(audio).all():raise ValueError('Invalid teacher waveform')
            with wave.open(str(path),'wb') as wav:
                wav.setparams((1,2,24000,0,'NONE','not compressed'))
                wav.writeframes((np.clip(audio,-1,1)*32767).astype('<i2').tobytes())
            score_path.write_text(json.dumps(score,indent=2)+'\n')
        report['clips'].append({'name':name,'kind':'fresh-kokoro-sung','file':str(path.relative_to(ROOT)),
            'speaker':speaker,'text':line,'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'score':str(score_path.relative_to(ROOT)),'sourceFormat':'24 kHz PCM16, matching the existing Kokoro adapter output convention',
            'reference':f'evidence/voice-fixtures/{speaker}-reference.wav' if speaker!='af_bella'
                else 'evidence/source-detail/fresh-inputs/af_bella-0.wav',
            'provenance':'Fresh output of the existing Kokoro score adapter, generated after candidate assembly was frozen; not a claim that our output model composes singing'})
        (OUT/'singing-manifest.json').write_text(json.dumps(report,indent=2)+'\n')
        print(name,flush=True)
    report['complete']=True
    (OUT/'singing-manifest.json').write_text(json.dumps(report,indent=2)+'\n')


if __name__=='__main__':main()
