"""Capture and replay the actual pretrained Kokoro waveform-decoder inputs.

Uses the installed mlx-audio 0.6.4 model, not an independently implemented or
trained speech model. A temporary decoder wrapper observes normal inference
in this isolated process. The app worker and its model are never modified.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import hashlib
import importlib.metadata
import json
import time
from pathlib import Path
import numpy as np
from scipy.io import wavfile

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/kokoro-control-voicebox/capture-v1'
VOWELS=set('ɑɐɒæəɛɜɚɝɪiɔʊuʌʉɨɯɶœøyᵊAEIOUYaeo')
KEYS=('asr','f0','noise','style')


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    model_path=ROOT/'models/expressive/Kokoro-82M-bf16'
    OUT.mkdir(parents=True,exist_ok=True);mx.set_cache_limit(128*1024*1024)
    start=time.perf_counter();model=load_model(model_path)
    upstream=Path(__import__('mlx_audio.tts.models.kokoro.kokoro',fromlist=['__file__']).__file__)
    report={'scope':'Normal pretrained Kokoro inference and exact saved-control replay; no independent model or new neural training',
        'expectedCases':9,'results':[],'complete':False,'seed':42,'modelLoadSeconds':time.perf_counter()-start,
        'model':str(model_path.relative_to(ROOT)),'modelWeightsSha256':sha(model_path/'kokoro-v1_0.safetensors'),
        'mlxAudioVersion':importlib.metadata.version('mlx-audio'),'upstreamForwardSha256':sha(upstream),
        'provenance':'Uses the existing pretrained Kokoro model and installed mlx-audio implementation (MIT; licenses/mlx-audio-LICENSE.txt). Controls originate from its text/acoustic model. Replay calls its original neural waveform decoder directly.',
        'limits':['Whole-utterance controls are captured offline; this is not live streaming.',
            'Replay does not receive text, but the captured controls were produced by a pretrained text model.',
            'Saved decoder features are not human articulators or proof of independent voice cloning.',
            'Raw float WAVs have no app high-pass filtering, levelling, added silence or fades.']}
    dtype_map={'float32':mx.float32,'float16':mx.float16,'bfloat16':mx.bfloat16}
    for speaker in ('af_heart','am_michael','af_bella'):
        pipeline=model._get_pipeline(speaker[0]);pack=pipeline.load_voice(speaker)
        for word in ('me','see','bee'):
            text=word.capitalize()+'.';_,tokens=pipeline.g2p(text)
            segments=list(pipeline.en_tokenize(tokens))
            if len(segments)!=1:raise ValueError('Expected one source segment')
            phonemes=''.join(p for p in segments[0][1] if p in model.vocab)
            ref=pack[len(phonemes)-1];native_decoder=model.decoder;captured={};dtypes={}
            def observe(asr,f0,noise,style):
                for key,value in zip(KEYS,(asr,f0,noise,style)):
                    mx.eval(value);dtypes[key]=str(value.dtype).split('.')[-1]
                    if dtypes[key] not in dtype_map:raise ValueError('Unsupported control dtype')
                    captured[key]=np.array(value.astype(mx.float32))
                return native_decoder(asr,f0,noise,style)
            mx.random.seed(42);start=time.perf_counter();model.decoder=observe
            try:normal=model(phonemes,ref,return_output=True)
            finally:model.decoder=native_decoder
            mx.eval(normal.audio,normal.pred_dur)
            audio=np.array(normal.audio.astype(mx.float32)).reshape(-1)
            natural=np.array(normal.pred_dur).astype(int)
            normal_seconds=time.perf_counter()-start
            name=speaker+'-'+word;file=OUT/(name+'-controls.npz')
            np.savez(file,**captured,naturalDurationFrames=natural,
                     metadata=json.dumps({'dtypes':dtypes,'phonemes':phonemes,'speaker':speaker,'seed':42}))
            # Read from disk and restore the original dtypes; don't reuse live tensors.
            with np.load(file,allow_pickle=False) as saved:
                stored=json.loads(str(saved['metadata']));args=[mx.array(saved[key],dtype=dtype_map[stored['dtypes'][key]]) for key in KEYS]
            mx.random.seed(42);start=time.perf_counter();replayed=native_decoder(*args)[0];mx.eval(replayed)
            replay=np.array(replayed.astype(mx.float32)).reshape(-1);replay_seconds=time.perf_counter()-start
            if len(audio)!=len(replay) or not np.isfinite(replay).all():raise ValueError('Invalid replay')
            difference=replay.astype(float)-audio.astype(float)
            original_file=OUT/(name+'-native.wav');replay_file=OUT/(name+'-replayed.wav')
            wavfile.write(original_file,24000,audio);wavfile.write(replay_file,24000,replay)
            edges=np.r_[0,np.cumsum(natural)]/40
            chars=['']+list(phonemes)+[''];nuclei=[i for i,c in enumerate(chars) if c in VOWELS]
            if len(nuclei)!=1:raise ValueError('Expected one intended vowel')
            nucleus=nuclei[0];a=int(round(edges[nucleus]*80));b=int(round(edges[nucleus+1]*80))
            f0=captured['f0'][0];noise=captured['noise'][0]
            row={'name':name,'speaker':speaker,'word':word,'text':text,'phonemes':phonemes,'nucleus':nucleus,
                'naturalDurationFrames':natural.tolist(),'vowelStartSeconds':float(edges[nucleus]),
                'vowelEndSeconds':float(edges[nucleus+1]),'seconds':len(audio)/24000,
                'controls':str(file.relative_to(ROOT)),'controlsSha256':sha(file),'controlShapes':{k:list(v.shape) for k,v in captured.items()},
                'controlDtypes':dtypes,'nativeAudio':str(original_file.relative_to(ROOT)),'nativeAudioSha256':sha(original_file),
                'replayAudio':str(replay_file.relative_to(ROOT)),'replayAudioSha256':sha(replay_file),
                'maximumReplayDifference':float(np.max(abs(difference))),'replayRmsDifference':float(np.sqrt(np.mean(difference**2))),
                'normalWholeComputeSeconds':normal_seconds,'replayWholeComputeSeconds':replay_seconds,
                'vowelF0Above55Fraction':float(np.mean(f0[a:b]>55)),
                'vowelF0Minimum':float(min(f0[a:b])),'vowelF0Maximum':float(max(f0[a:b])),
                'vowelNoiseMinimum':float(min(noise[a:b])),'vowelNoiseMaximum':float(max(noise[a:b])),
                'rawPeak':float(max(abs(audio))),'finite':bool(np.isfinite(audio).all())}
            report['results'].append(row)
            (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(name,'replay error',row['maximumReplayDifference'],'vowel voiced controls',row['vowelF0Above55Fraction'],flush=True)
            mx.clear_cache()
    report['complete']=len(report['results'])==report['expectedCases']
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
