"""Post-validation ablation: hold captured vocoder inputs instead of ASR inputs.

Uses Kokoro's pretrained generator without changing any neural weights. This
tests whether earlier decoder processing destroys an otherwise usable sustain.
The source selection, pitch, duration, random seed and evaluation stay fixed.
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
import json
import sys
import time
from pathlib import Path
import numpy as np
from scipy.io import wavfile
import validate_kokoro_control_generalization as common

ROOT = common.ROOT
OUT = ROOT/'evidence/kokoro-control-voicebox/generator-hold-v1'
PRIOR = ROOT/'evidence/kokoro-control-voicebox/fresh-vowels-v1'


def write(name, value):
    (OUT/name).write_text(json.dumps(value, indent=2, allow_nan=False)+'\n')


def setup():
    prior = json.loads((PRIOR/'report.json').read_text())
    if not prior['complete']: raise ValueError('Incomplete fresh validation')
    config = json.loads((PRIOR/'protocol.json').read_text())
    config.update(scope='Post-validation generator-input hold ablation on the same fresh-vowel panel; pretrained, not another held-out test',
        originalProtocolSha256=common.sha(PRIOR/'protocol.json'),
        originalReportSha256=common.sha(PRIOR/'report.json'),
        intervention='Hold the captured 80 Hz generator feature vector instead of extending upstream ASR/N controls; same source frame map and authored F0.',
        generatorSource='Installed mlx-audio Kokoro istftnet.Generator; unchanged pretrained weights',
        newNeuralTraining=False)
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'protocol.json').exists():
        if json.loads((OUT/'protocol.json').read_text()) != config: raise ValueError('Changed ablation protocol')
    else: write('protocol.json', config)
    common.OUT = OUT
    return config, prior


def render(config, prior):
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    from kokoro_normalization_reference import NormalizationReference
    for name, expected in config['frozenCode'].items():
        if common.sha(Path(__file__).parent/name) != expected: raise ValueError('Changed frozen method')
    if common.sha(ROOT/config['model']/'kokoro-v1_0.safetensors') != config['modelWeightsSha256']:
        raise ValueError('Changed weights')
    report = {'protocolSha256': common.sha(OUT/'protocol.json'), 'sources': [], 'failures': [], 'results': [], 'complete': False}
    if (OUT/'report.json').exists(): report = common.read('report.json')
    if report['complete']: print('Already complete; retained.', flush=True); return
    mx.set_cache_limit(128*1024*1024); model = load_model(ROOT/config['model'])
    dtypes = {'float32':mx.float32, 'float16':mx.float16, 'bfloat16':mx.bfloat16}
    done = {r['name']:r for r in report['results']}
    for source in prior['sources']:
        file = ROOT/source['controls']
        if common.sha(file) != source['controlsSha256']: raise ValueError('Changed source controls')
        with np.load(file, allow_pickle=False) as saved:
            meta = json.loads(str(saved['metadata']))
            args = [mx.array(saved[k], dtype=dtypes[meta['dtypes'][k]]) for k in common.KEYS]
        generator = model.decoder.generator; captured = {}; types = {}
        def observe(x, style, f0):
            for key, value in zip(('x','style','f0'), (x,style,f0)):
                mx.eval(value); types[key] = str(value.dtype).split('.')[-1]
                captured[key] = np.array(value.astype(mx.float32))
            return generator(x,style,f0)
        mx.random.seed(config['seed']); model.decoder.generator = observe
        try: native = model.decoder(*args)[0]; mx.eval(native)
        finally: model.decoder.generator = generator
        control_file = OUT/(source['name']+'-generator-controls.npz')
        np.savez(control_file, **captured, metadata=json.dumps({'dtypes':types,'speaker':source['speaker']}))
        with np.load(control_file, allow_pickle=False) as saved:
            gen_args = [mx.array(saved[k],dtype=dtypes[types[k]]) for k in ('x','style','f0')]
        if captured['x'].shape != (1,512,captured['f0'].shape[1]): raise ValueError('Unexpected generator control dimensions')
        reference = NormalizationReference(generator); mx.random.seed(config['seed'])
        with reference.apply('capture'):
            replay = generator(*gen_args)[0]; mx.eval(replay)
        mx.random.seed(config['seed'])
        with reference.apply('reference'):
            frozen = generator(*gen_args)[0]; mx.eval(frozen)
        rate, original = wavfile.read(ROOT/source['file']); assert rate == 24000
        error = float(max(np.max(abs(np.array(a.astype(mx.float32)).reshape(-1)-original)) for a in (native,replay,frozen)))
        if error > 1e-6: raise ValueError('Generator native replay failed: '+str(error))
        stats = OUT/(source['name']+'-generator-normalization.npz'); info = reference.save(stats)
        reference = NormalizationReference.load(generator, stats)
        new_source = {**source, 'generatorControls':str(control_file.relative_to(ROOT)),
            'generatorControlsSha256':common.sha(control_file),'generatorControlShapes':{k:list(v.shape) for k,v in captured.items()},
            'nativeReplayMaximumDifference':error,'statistics':str(stats.relative_to(ROOT)),
            'statisticsSha256':common.sha(stats),'statisticsInfo':info}
        report['sources'] = [r for r in report['sources'] if r['name'] != source['name']] + [new_source]
        cases = [r for r in prior['results'] if r['speaker']==source['speaker'] and r['word']==source['word'] and r['method']=='normal']
        for case in cases:
            if common.sha(ROOT/case['controls']) != case['controlsSha256']: raise ValueError('Changed authored controls')
            with np.load(ROOT/case['controls'],allow_pickle=False) as saved:
                mapping = saved['sourceFrameMap80Hz']; f0 = saved['f0'].copy()
            x = np.array([np.interp(mapping,np.arange(captured['x'].shape[-1]),row) for row in captured['x'][0]])[None]
            if x.shape != (1,512,round(config['seconds']*80)): raise ValueError('Invalid edited generator controls')
            x_file = OUT/(source['name']+'-'+case['mode']+'-generator-edited.npz')
            np.savez(x_file,x=x,style=captured['style'],f0=f0,sourceFrameMap80Hz=mapping)
            edited = [mx.array(value,dtype=dtypes[types[key]]) for key,value in [('x',x),('style',captured['style']),('f0',f0)]]
            for method in config['methods']:
                name = source['name']+'-'+case['mode']+'-generator-'+method
                if name in done:
                    if common.sha(ROOT/done[name]['file']) != done[name]['sha256']: raise ValueError('Changed prior output')
                    continue
                mx.random.seed(config['seed']); start = time.perf_counter()
                if method == 'reference':
                    with reference.apply('reference'):
                        wave = generator(*edited)[0]; mx.eval(wave)
                else: wave = generator(*edited)[0]; mx.eval(wave)
                elapsed = time.perf_counter()-start; audio = np.array(wave.astype(mx.float32)).reshape(-1)
                if len(audio) != round(config['seconds']*24000) or not np.isfinite(audio).all(): raise ValueError('Invalid output')
                output = OUT/(name+'.wav'); wavfile.write(output,24000,audio)
                report['results'].append({**case,'name':name,'method':method,'heldFeatureStage':'generator-input',
                    'file':str(output.relative_to(ROOT)),'sha256':common.sha(output),
                    'controls':str(x_file.relative_to(ROOT)),'controlsSha256':common.sha(x_file),
                    'rawPeak':float(max(abs(audio))),'overFullScaleSamples':int(np.sum(abs(audio)>=1)),
                    'wholeDecoderComputeSeconds':elapsed,'finite':True})
                write('report.json',report); print(name,'peak',round(max(abs(audio)),3),flush=True)
                mx.clear_cache()
    report['expectedEligibleRenders'] = len(prior['results'])
    report['complete'] = len(report['sources']) == 18 and len(report['results']) == len(prior['results'])
    write('report.json',report)


if __name__ == '__main__':
    config, prior = setup(); stage = sys.argv[1]
    if stage == 'render': render(config,prior)
    elif stage == 'evaluate':
        if not (OUT/'evaluation.json').exists():
            prior_eval = json.loads((PRIOR/'evaluation.json').read_text())
            if not prior_eval['complete']: raise ValueError('Incomplete source evaluation')
            for source in prior_eval['sources']:
                if common.sha(ROOT/source['file']) != source['sha256']: raise ValueError('Changed reused ASR source')
            write('evaluation.json',{'protocolSha256':common.sha(OUT/'protocol.json'),'sources':prior_eval['sources'],
                'reusedSourceAsrEvaluations':len(prior_eval['sources']),'results':[],'complete':False})
        common.evaluate(config)
    elif stage == 'summarize': common.summarize(config)
    else: raise ValueError('Unknown stage')
