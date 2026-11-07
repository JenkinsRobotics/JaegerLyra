"""Isolate excitation and cancel the first upsampler's held-input grid response.

Pretrained, offline, source-derived Kokoro ablation. The app and model weights
are unchanged. Each temporary hook is restored even when inference fails.
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY','1')
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import validate_kokoro_control_generalization as common
from periodic_hold_correction import correct_periodic_hold

ROOT=common.ROOT
PRIOR=ROOT/'evidence/kokoro-control-voicebox/generator-hold-v1'
OUT=ROOT/'evidence/kokoro-control-voicebox/upsampler-grid-v1'


def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')


def protocol():
    prior=json.loads((PRIOR/'report.json').read_text()); old=json.loads((PRIOR/'protocol.json').read_text())
    if not prior['complete']:raise ValueError('Incomplete generator study')
    config={**old,'scope':'Post-validation first-upsampler constant-response ablation; pretrained and offline',
        'methods':['baseline','half','full'],'maximumRenders':108,
        'intervention':'Subtract 0%, 50% or 100% of the mean-zero stride-phase response from the first learned upsampler during the known constant-input hold. No output-audio filtering or neural-weight changes.',
        'priorReportSha256':common.sha(PRIOR/'report.json'),
        'correctionCodeSha256':common.sha(Path(__file__).with_name('periodic_hold_correction.py')),
        'primaryReference':'https://arxiv.org/abs/2010.14356',
        'referenceScope':'Prior work describes tonal artifacts from neural audio upsampling; the present held-activation intervention is a local diagnostic.',
        'newControlledRenders':72,'baselineReplays':36,'newNeuralTraining':False,
        'correctionScope':'Fit on held first-upsampler features, not on output waveform, target metrics or transcripts; uses known complete hold boundaries.',
        'sourceObservation':'Capture the unchanged harmonic source before the generator; compare its steady-note spectrum with output.',
        'appDefaultChanged':False,'listeningAccepted':False}
    OUT.mkdir(parents=True,exist_ok=True)
    if (OUT/'protocol.json').exists():
        if json.loads((OUT/'protocol.json').read_text())!=config:raise ValueError('Changed protocol')
    else:write('protocol.json',config)
    common.OUT=OUT
    return config,prior


@contextmanager
def observed_generator(generator,case,strength):
    import mlx.core as mx
    source=generator.m_source;up=generator.ups[0];captured={}
    def source_call(f0):
        result=source(f0);mx.eval(result[0]);captured['harmonicSource']=np.array(result[0].astype(mx.float32)).reshape(-1)
        return result
    def up_call(x,conv):
        result=up(x,conv);mx.eval(x,result)
        inputs=np.array(x.astype(mx.float32))[0];values=np.array(result.astype(mx.float32))[0]
        rate=80*int(up.stride);begin=case['plan']['holdStartSeconds'];end=case['plan']['holdEndSeconds']
        interior=inputs[int(np.ceil((begin+.1)*80)):int(np.floor((end-.1)*80))]
        variation=float(np.max(np.ptp(interior,axis=0)))
        if variation>1e-6:raise ValueError('First-upsampler input is not constant in the fitted region')
        corrected,template,gate,info=correct_periodic_hold(values,int(up.stride),rate,begin,end,strength)
        captured.update(template=template,gate=gate,info={**info,'inputTemporalPeakToPeakMaximum':variation,
            'inputShape':list(x.shape),'outputShape':list(result.shape)},
            beforeFeatureEnergy=np.mean(values.astype(float)**2,axis=1),
            afterFeatureEnergy=np.mean(corrected.astype(float)**2,axis=1))
        if strength==0:return result
        return mx.array(corrected[None],dtype=result.dtype)
    generator.m_source=source_call;generator.ups[0]=up_call
    try:yield captured
    finally:
        generator.m_source=source;generator.ups[0]=up
        if generator.m_source is not source or generator.ups[0] is not up:raise AssertionError('Hooks not restored')


def render(config,prior):
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    from kokoro_normalization_reference import NormalizationReference
    if common.sha(ROOT/config['model']/'kokoro-v1_0.safetensors')!=config['modelWeightsSha256']:raise ValueError('Changed model')
    mx.set_cache_limit(128*1024*1024);model=load_model(ROOT/config['model']);generator=model.decoder.generator
    report={'protocolSha256':common.sha(OUT/'protocol.json'),'sources':prior['sources'],'failures':[],
        'results':[],'baselineChecks':[],'sourceCaptures':[],'complete':False}
    if (OUT/'report.json').exists():report=common.read('report.json')
    if report['complete']:print('Already complete; retained.',flush=True);return
    done={r['name']:r for r in report['results']}
    dtypes={'float32':mx.float32,'float16':mx.float16,'bfloat16':mx.bfloat16}
    for source in prior['sources']:
        for field,hash_field in [('statistics','statisticsSha256'),('generatorControls','generatorControlsSha256')]:
            if common.sha(ROOT/source[field])!=source[hash_field]:raise ValueError('Changed reference')
        with np.load(ROOT/source['generatorControls'],allow_pickle=False) as data:types=json.loads(str(data['metadata']))['dtypes']
        reference=NormalizationReference.load(generator,ROOT/source['statistics'])
        cases=[r for r in prior['results'] if r['speaker']==source['speaker'] and r['word']==source['word'] and r['method']=='reference']
        for case in cases:
            if common.sha(ROOT/case['controls'])!=case['controlsSha256']:raise ValueError('Changed controls')
            with np.load(ROOT/case['controls'],allow_pickle=False) as data:
                args=[mx.array(data[k],dtype=dtypes[types[k]]) for k in ('x','style','f0')]
            for method,strength in [('baseline',0.),('half',.5),('full',1.)]:
                name=case['name'] if method=='baseline' else case['name']+'-grid-'+method
                if name in done:
                    if common.sha(ROOT/done[name]['file'])!=done[name]['sha256']:raise ValueError('Changed existing render')
                    continue
                mx.random.seed(config['seed']);start=time.perf_counter()
                with observed_generator(generator,case,strength) as captured:
                    with reference.apply('reference'):
                        wave=generator(*args)[0];mx.eval(wave)
                elapsed=time.perf_counter()-start;audio=np.array(wave.astype(mx.float32)).reshape(-1)
                if not np.isfinite(audio).all() or len(audio)!=round(config['seconds']*24000):raise ValueError('Invalid waveform')
                features=OUT/(name+'-features.npz')
                np.savez(features,template=captured['template'],gate=captured['gate'],
                    beforeFeatureEnergy=captured['beforeFeatureEnergy'],afterFeatureEnergy=captured['afterFeatureEnergy'])
                if method=='baseline':
                    rate,original=wavfile.read(ROOT/case['file']);assert rate==24000
                    difference=float(np.max(abs(audio-original)))
                    if difference>1e-6:raise ValueError('Hook observation changed baseline')
                    report['baselineChecks'].append({'name':name,'maximumDifference':difference,'hooksRestored':True})
                    output=ROOT/case['file']
                    harmonic=OUT/(case['name']+'-harmonic-source.wav');wavfile.write(harmonic,24000,captured['harmonicSource'])
                    report['sourceCaptures'].append({'name':case['name'],'mode':case['mode'],
                        'speaker':case['speaker'],'word':case['word'],'score':case['score'],'plan':case['plan'],
                        'file':str(harmonic.relative_to(ROOT)),'sha256':common.sha(harmonic)})
                else:
                    output=OUT/(name+'.wav');wavfile.write(output,24000,audio)
                report['results'].append({**case,'name':name,'method':method,'file':str(output.relative_to(ROOT)),
                    'sha256':common.sha(output),'correction':captured['info'],'featureDiagnostics':str(features.relative_to(ROOT)),
                    'wholeDecoderComputeSeconds':elapsed,'rawPeak':float(max(abs(audio))),
                    'overFullScaleSamples':int(np.sum(abs(audio)>=1.)),'finite':True})
                write('report.json',report);print(name,'feature variation',round(captured['info']['beforeTemporalRms'],4),
                    '->',round(captured['info']['afterTemporalRms'],4),'peak',round(max(abs(audio)),3),flush=True)
                mx.clear_cache()
    report['expectedEligibleRenders']=108;report['complete']=len(report['results'])==108 and len(report['baselineChecks'])==36
    write('report.json',report)


if __name__=='__main__':
    config,prior=protocol();stage=sys.argv[1]
    if stage=='render':render(config,prior)
    elif stage=='evaluate':
        if not (OUT/'evaluation.json').exists():
            previous=json.loads((PRIOR/'evaluation.json').read_text())
            if not previous['complete']:raise ValueError('Incomplete baseline evaluation')
            reused=[{**r,'method':'baseline'} for r in previous['results'] if r['method']=='reference']
            for row in reused+previous['sources']:
                if common.sha(ROOT/row['file'])!=row['sha256']:raise ValueError('Changed reused audio')
            write('evaluation.json',{'protocolSha256':common.sha(OUT/'protocol.json'),'sources':previous['sources'],
                'results':reused,'reusedAsrEvaluations':54,'complete':False})
        common.evaluate(config)
    elif stage=='summarize':common.summarize(config)
    else:raise ValueError('Unknown stage')
