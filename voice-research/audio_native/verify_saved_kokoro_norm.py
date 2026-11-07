"""Reopen source statistics and reproduce every edited pretrained output."""
import os
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
os.environ['TOKENIZERS_PARALLELISM']='false'
import hashlib
import json
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
    study=json.loads((OUT/'report.json').read_text())
    if not study['complete']:raise ValueError('Incomplete study')
    model_path=ROOT/capture['model']
    if sha(model_path/'kokoro-v1_0.safetensors')!=capture['modelWeightsSha256']:raise ValueError('Changed weights')
    mx.set_cache_limit(128*1024*1024);model=load_model(model_path)
    dtypes={'float32':mx.float32,'float16':mx.float16,'bfloat16':mx.bfloat16};report={
        'scope':'Replay all edited clips from saved acoustic controls and saved normalization statistics, without original speech-model inference',
        'expectedCases':36,'results':[],'complete':False,'absoluteTolerance':1e-6,
        'normalizationHelperSha256':sha(Path(__file__).with_name('kokoro_normalization_reference.py'))}
    for case in study['results']:
        source=next(r for r in capture['results'] if r['speaker']==case['speaker'] and r['word']==case['word'])
        for path,expected in [(case['controls'],case['controlsSha256']),
                              (case['normalizationStatistics'],case['normalizationStatisticsSha256']),
                              (case['file'],case['sha256'])]:
            if sha(ROOT/path)!=expected:raise ValueError('Changed replay input')
        ref=NormalizationReference.load(model.decoder,ROOT/case['normalizationStatistics'])
        with np.load(ROOT/case['controls'],allow_pickle=False) as data:
            args=[mx.array(data[key],dtype=dtypes[source['controlDtypes'][key]]) for key in ('asr','f0','noise','style')]
        mx.random.seed(42)
        with ref.apply('reference'):
            result=model.decoder(*args)[0];mx.eval(result)
        audio=np.array(result.astype(mx.float32)).reshape(-1);rate,expected=wavfile.read(ROOT/case['file'])
        if rate!=24000 or audio.shape!=expected.shape:raise ValueError('Replay format changed')
        difference=float(np.max(abs(audio-expected)))
        if difference>report['absoluteTolerance']:raise ValueError('Saved-state replay differs: '+case['name'])
        report['results'].append({'name':case['name'],'maximumDifference':difference,'samples':len(audio)})
        (OUT/'saved-state-replay.json').write_text(json.dumps(report,indent=2)+'\n');mx.clear_cache()
    report['complete']=len(report['results'])==36
    (OUT/'saved-state-replay.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Replayed',len(report['results']),'saved cases; maximum error',max(r['maximumDifference'] for r in report['results']))


if __name__=='__main__':main()
