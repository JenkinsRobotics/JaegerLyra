"""Factorial follow-up: same control edits, normal-source normalization stats."""
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
from kokoro_normalization_reference import NormalizationReference

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/kokoro-control-voicebox/reference-norm-v2'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    capture=json.loads((ROOT/'evidence/kokoro-control-voicebox/capture-v1/report.json').read_text())
    prior=json.loads((ROOT/'evidence/kokoro-control-voicebox/sustain-v1/report.json').read_text())
    if not capture['complete'] or not prior['complete']:raise ValueError('Incomplete prior study')
    model_path=ROOT/capture['model']
    if sha(model_path/'kokoro-v1_0.safetensors')!=capture['modelWeightsSha256']:raise ValueError('Changed pretrained weights')
    mx.set_cache_limit(128*1024*1024);model=load_model(model_path);OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Pretrained Kokoro per-utterance normalization-statistic ablation; no independent model or new training',
        'expectedCases':36,'expectedNativeChecks':9,'nativeChecks':[],'results':[],'complete':False,
        'modelWeightsSha256':capture['modelWeightsSha256'],
        'normalizationCodeSha256':sha(Path(__file__).with_name('kokoro_normalization_reference.py')),
        'limits':['Statistics come from each complete normal source utterance, not a generalized speaker embedding.',
            'Only normalization statistics change relative to the prior control edits; captured controls and neural weights are unchanged.',
            'This is offline full-clip decoding, not causal streaming or general reference cloning.',
            'All hooks are restored in an isolated process; the app worker and installed source files are unchanged.']}
    dtypes={'float32':mx.float32,'float16':mx.float16,'bfloat16':mx.bfloat16};keys=('asr','f0','noise','style')
    for source in capture['results']:
        file=ROOT/source['controls']
        if sha(file)!=source['controlsSha256']:raise ValueError('Changed source controls')
        with np.load(file,allow_pickle=False) as data:
            metadata=json.loads(str(data['metadata']))
            args=[mx.array(data[key],dtype=dtypes[metadata['dtypes'][key]]) for key in keys]
        reference=NormalizationReference(model.decoder)
        mx.random.seed(42)
        with reference.apply('capture'):
            original=model.decoder(*args)[0];mx.eval(original)
        mx.random.seed(42)
        with reference.apply('reference'):
            neutral=model.decoder(*args)[0];mx.eval(neutral)
        a=np.array(original.astype(mx.float32));b=np.array(neutral.astype(mx.float32))
        difference=float(np.max(abs(a-b)))
        if difference>1e-6:raise ValueError('Reference normalization does not reproduce normal decoder: '+str(difference))
        stats_file=OUT/(source['name']+'-normalization.npz');stats_info=reference.save(stats_file)
        report['nativeChecks'].append({'name':source['name'],'maximumDifference':difference,
            'statistics':str(stats_file.relative_to(ROOT)),'statisticsSha256':sha(stats_file),**stats_info})
        cases=[r for r in prior['results'] if r['speaker']==source['speaker'] and r['word']==source['word']]
        for case in cases:
            if sha(ROOT/case['controls'])!=case['controlsSha256']:raise ValueError('Changed edited controls')
            with np.load(ROOT/case['controls'],allow_pickle=False) as data:
                edited=[mx.array(data[key],dtype=dtypes[metadata['dtypes'][key]]) for key in keys]
            mx.random.seed(42);start=time.perf_counter()
            with reference.apply('reference'):
                generated=model.decoder(*edited)[0];mx.eval(generated)
            elapsed=time.perf_counter()-start;audio=np.array(generated.astype(mx.float32)).reshape(-1)
            if not np.isfinite(audio).all():raise ValueError('Nonfinite referenced output')
            name=case['name']+'-reference-norm';output=OUT/(name+'.wav');wavfile.write(output,24000,audio)
            report['results'].append({**case,'name':name,'method':case['method']+'-reference-norm',
                'file':str(output.relative_to(ROOT)),'sha256':sha(output),'normalizationStatistics':str(stats_file.relative_to(ROOT)),
                'normalizationStatisticsSha256':sha(stats_file),'wholeDecoderComputeSeconds':elapsed,
                'computeRealTimeFactor':elapsed/case['score']['seconds'],'rawPeak':float(max(abs(audio))),
                'overFullScaleSamples':int(np.sum(abs(audio)>=1.)),'finite':bool(np.isfinite(audio).all())})
            (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
            print(name,'peak',round(report['results'][-1]['rawPeak'],3),'native difference',difference,flush=True)
            mx.clear_cache()
    report['complete']=len(report['results'])==36 and len(report['nativeChecks'])==9
    (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
