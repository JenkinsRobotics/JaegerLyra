"""Offline pretrained control-sustain experiment, using captured normal inputs."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from kokoro_control_sustain import stretch_controls

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/kokoro-control-voicebox/sustain-v1'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    capture=json.loads((ROOT/'evidence/kokoro-control-voicebox/capture-v1/report.json').read_text())
    original=json.loads((ROOT/'evidence/neural-articulation/v1/report.json').read_text())
    if not capture['complete'] or not original['complete']:raise ValueError('Incomplete input evidence')
    if max(r['maximumReplayDifference'] for r in capture['results'])>1e-6:raise ValueError('Native control replay not verified')
    model_path=ROOT/capture['model']
    if sha(model_path/'kokoro-v1_0.safetensors')!=capture['modelWeightsSha256']:raise ValueError('Changed pretrained model')
    mx.set_cache_limit(128*1024*1024);start=time.perf_counter();model=load_model(model_path)
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Pretrained Kokoro decoder driven by edited captured controls; offline, source-derived and not independent neural synthesis',
        'captureReport':'evidence/kokoro-control-voicebox/capture-v1/report.json',
        'modelWeightsSha256':capture['modelWeightsSha256'],'modelLoadSeconds':time.perf_counter()-start,
        'selectionAndRetimingSha256':sha(Path(__file__).with_name('kokoro_control_sustain.py')),
        'expectedCases':36,'results':[],'complete':False,
        'limits':['Whole controls are known before rendering; not low-latency acoustic streaming.',
            'Pretrained Kokoro weights and captured speech-model controls remain necessary; no new neural training.',
            'The N input is treated as an opaque learned acoustic control, not asserted to be breath or loudness in physical units.',
            'Both methods retain original control timing outside the changed vowel region or inserted state; audio at boundaries may still change.',
            'Authored F0 replaces only control frames whose source F0 is above 55 Hz.',
            'These are the earlier nine development units; no fresh-source quality or identity acceptance.']}
    dtype_map={'float32':mx.float32,'float16':mx.float16,'bfloat16':mx.bfloat16}
    for source in capture['results']:
        file=ROOT/source['controls']
        if sha(file)!=source['controlsSha256']:raise ValueError('Changed captured controls')
        with np.load(file,allow_pickle=False) as data:
            controls={key:data[key].copy() for key in ('asr','f0','noise','style')}
            durations=data['naturalDurationFrames'].copy();metadata=json.loads(str(data['metadata']))
        cases=[r for r in original['results'] if r['speaker']==source['speaker'] and r['unit']==source['word'] and r['mode'] in ('native','melody')]
        if len(cases)!=2:raise ValueError('Missing authored control cases')
        for case in cases:
            score=case['score'];times=np.arange(round(score['seconds']*80))/80
            target=2**np.interp(times,score['times'],np.log2(score['hz']))
            for method in ('whole-vowel','interior-hold'):
                start=time.perf_counter();edited,plan,maps=stretch_controls(controls,durations,source['nucleus'],score['seconds'],method,target)
                prepare_seconds=time.perf_counter()-start
                args=[mx.array(edited[key],dtype=dtype_map[metadata['dtypes'][key]]) for key in ('asr','f0','noise','style')]
                mx.random.seed(42);start=time.perf_counter();wave=model.decoder(*args)[0];mx.eval(wave)
                audio=np.array(wave.astype(mx.float32)).reshape(-1);render_seconds=time.perf_counter()-start
                if len(audio)!=round(score['seconds']*24000) or not np.isfinite(audio).all():raise ValueError('Invalid generated waveform')
                name=case['name']+'-'+method;output=OUT/(name+'.wav');wavfile.write(output,24000,audio)
                saved=OUT/(name+'-controls.npz');np.savez(saved,**edited,**maps,requestedF0=target,metadata=json.dumps(plan))
                begin=round((plan['holdStartSeconds']+.080)*80);end=round((plan['holdEndSeconds']-.080)*80)
                report['results'].append({'name':name,'speaker':source['speaker'],'word':source['word'],'mode':case['mode'],
                    'method':method,'score':score,'file':str(output.relative_to(ROOT)),'sha256':sha(output),
                    'controls':str(saved.relative_to(ROOT)),'controlsSha256':sha(saved),'plan':plan,
                    'originalControlReport':source['name'],'prepareControlsSeconds':prepare_seconds,
                    'wholeDecoderComputeSeconds':render_seconds,'computeRealTimeFactor':render_seconds/score['seconds'],
                    'rawPeak':float(max(abs(audio))),'overFullScaleSamples':int(np.sum(abs(audio)>=1.)),
                    'finite':bool(np.isfinite(audio).all()),'heldControlVoicedFraction':float(np.mean(edited['f0'][0,begin:end]>55)),
                    'heldNRange':[float(min(edited['noise'][0,begin:end])),float(max(edited['noise'][0,begin:end]))]})
                (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
                print(name,'control voicing',report['results'][-1]['heldControlVoicedFraction'],'render seconds',round(render_seconds,3),flush=True)
                mx.clear_cache()
    report['complete']=len(report['results'])==report['expectedCases']
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
