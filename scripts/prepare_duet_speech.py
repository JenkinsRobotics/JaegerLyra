"""Prepare a local user-authored duet's speech references with optional Kokoro.

Run in the separate MLX environment. This stage generates SPEECH, not singing.
The subsequent Lyra-only renderer supplies the authored singing pitch/timing.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import hashlib,json,sys
from pathlib import Path
import numpy as np
from scipy.io import wavfile

ROOT=Path(__file__).resolve().parents[1]


def main():
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    path=Path(sys.argv[1]).resolve();score=json.loads(path.read_text())
    out=path.parent/'speech';out.mkdir(exist_ok=True)
    mx.set_cache_limit(128*1024*1024)
    model=load_model(ROOT/'models/expressive/Kokoro-82M-bf16')
    entries=[]
    for row in score['phrases']:
        name=row['id'];speaker=row['speaker'];text=row['text']
        pipeline=model._get_pipeline(speaker[0]);pack=pipeline.load_voice(speaker)
        _,tokens=pipeline.g2p(text)
        segments=list(pipeline.en_tokenize(tokens))
        if len(segments)!=1:raise ValueError('Use short lyric phrases')
        phonemes=''.join(p for p in segments[0][1] if p in model.vocab)
        mx.random.seed(42)
        result=model(phonemes,pack[len(phonemes)-1],return_output=True)
        mx.eval(result.audio,result.pred_dur)
        samples=np.asarray(result.audio,dtype=np.float32).reshape(-1)
        durations=np.asarray(result.pred_dur).astype(int).reshape(-1)
        if len(durations)!=len(phonemes)+2 or not np.isfinite(samples).all():
            raise ValueError('Unexpected pronunciation alignment')
        target=out/(name+'.wav');wavfile.write(target,24000,samples)
        entries.append({**row,'phonemes':phonemes,'duration_frames':durations.tolist(),
            'file':str(target.relative_to(path.parent)),'sha256':hashlib.sha256(target.read_bytes()).hexdigest(),
            'sample_count':len(samples),'origin':'Kokoro natural speech, seed 42; not original song audio'})
        (out/'manifest.json').write_text(json.dumps({'score_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
            'phrases':entries,'complete':len(entries)==len(score['phrases'])},indent=2)+'\n')
        print(name,'phonemes',phonemes,'seconds',round(len(samples)/24000,3),flush=True)
        mx.clear_cache()


if __name__=='__main__':main()
