"""Matched long-vowel reference using the existing source-derived Kokoro adapter.

This explicitly uses Kokoro's pretrained text/acoustic model and waveform
decoder. It is not the independent waveform model, a streaming engine or a
singing-trained model. The adapter provenance remains in kokoro_singing.py.
"""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import hashlib
import json
import sys
import time
import wave
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfilt

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
OUT=ROOT/'evidence/neural-articulation/kokoro-reference-v1'


def main():
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    import kokoro_singing as adapter
    source=json.loads((ROOT/'evidence/neural-articulation/v1/report.json').read_text())
    if not source['complete']:raise ValueError('Incomplete original controls')
    cases=[r for r in source['results'] if r['mode'] in ('native','melody')]
    OUT.mkdir(parents=True,exist_ok=True);mx.set_cache_limit(128*1024*1024)
    start=time.perf_counter();model=load_model(ROOT/'models/expressive/Kokoro-82M-bf16')
    report={'scope':'Offline pretrained Kokoro waveform reference at the same long-word duration and pitch curves',
        'expectedCases':18,'results':[],'complete':False,'modelLoadSeconds':time.perf_counter()-start,
        'adapter':'kokoro_singing.py','adapterSha256':hashlib.sha256((ROOT/'kokoro_singing.py').read_bytes()).hexdigest(),
        'provenance':'Existing adapter forward pass derives from mlx-audio 0.6.4 MIT code; model is the existing local Kokoro-82M-bf16. No independent model or new neural training.',
        'limits':['The vowel duration is lengthened inside Kokoro acoustic controls, not by repeating generated waveform cycles.',
            'Whole-word rendering requires all controls and uses the pretrained text/acoustic model.',
            'Consonant timing is Kokoro-predicted; boundary durations are two 40 Hz frames and punctuation/stress one frame, so exact onset timing differs from captured-unit renders.',
            'Voicing uses the prior adapter threshold of 55 Hz; a long held vowel is not proof of trained singing quality.']}
    original_plan=adapter.plan_score
    for case in cases:
        score=case['score'];details={}
        def plan(phonemes,natural,*args,**kwargs):
            chars=['']+list(phonemes)+[''];nuclei=[i for i,p in enumerate(chars) if p in adapter.VOWELS]
            if len(nuclei)!=1:raise ValueError('Expected one vowel in '+phonemes)
            durations=np.maximum(np.asarray(natural,dtype=int),1).copy()
            durations[0]=durations[-1]=2
            for i,p in enumerate(chars):
                if p and p in ' ˈˌ.,!?;:':durations[i]=1
            target_frames=round(score['seconds']*40);nucleus=nuclei[0]
            remaining=target_frames-int(durations.sum()-durations[nucleus])
            if remaining<6:raise ValueError('Insufficient vowel duration')
            durations[nucleus]=remaining
            times=np.arange(target_frames*2)/80
            pitch=np.interp(times,score['times'],np.log2(score['hz']))
            pitch+=score['vibrato_cents']/1200*np.sin(2*np.pi*score['vibrato_hz']*times)
            target=(2**pitch).astype(np.float32)
            edges=np.r_[0,np.cumsum(durations)]
            note={'vowelStart':float(edges[nucleus]/40),'vowelEnd':float(edges[nucleus+1]/40)}
            details.update(phonemes=phonemes,naturalDurationFrames=np.asarray(natural).tolist(),
                authoredDurationFrames=durations.tolist(),nucleus=nucleus,**note)
            return durations,target,[note]
        start=time.perf_counter();mx.random.seed(42)
        pipeline=model._get_pipeline(case['speaker'][0]);pack=pipeline.load_voice(case['speaker'])
        adapter.plan_score=plan
        try:
            samples,meta=adapter.render_line(model,pipeline,case['text'],pack,
                {'tempo':104,'melody':'glow','speaker':case['speaker']},0)
        finally:adapter.plan_score=original_plan
        samples=sosfilt(butter(2,60,fs=24000,btype='highpass',output='sos'),samples)
        peak=float(max(abs(samples)));rms=float(np.sqrt(np.mean(samples**2)))
        if peak>1e-6:samples*=min(2.5,.11/max(rms,.001),.89/peak)
        fade=min(192,len(samples)//2);samples[:fade]*=np.linspace(0,1,fade);samples[-fade:]*=np.linspace(1,0,fade)
        if not np.isfinite(samples).all():raise ValueError('Nonfinite teacher output')
        file=OUT/(case['name']+'-kokoro-decoder.wav')
        with wave.open(str(file),'wb') as wav:
            wav.setparams((1,2,24000,0,'NONE','not compressed'))
            wav.writeframes((np.clip(samples,-1,1)*32767).astype('<i2').tobytes())
        report['results'].append({'name':case['name']+'-kokoro-decoder','speaker':case['speaker'],'unit':case['unit'],
            'text':case['text'],'mode':case['mode'],'score':score,'file':str(file.relative_to(ROOT)),
            'sha256':hashlib.sha256(file.read_bytes()).hexdigest(),'seconds':len(samples)/24000,
            'wholeRenderSeconds':time.perf_counter()-start,**details})
        (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
        print(case['name'],'seconds',len(samples)/24000,flush=True);mx.clear_cache()
    report['complete']=len(report['results'])==report['expectedCases']
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
