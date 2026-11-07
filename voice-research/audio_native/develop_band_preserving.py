"""Fixed 8 kHz complementary detail branch, checked on development audio.

The cutoff follows the earlier high-band error diagnosis; no cutoff/tap sweep is
selected on the now-reused capacity-v3 final panel. Fresh validation is required
after this composition is frozen. Neural branch weights remain untouched.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib,json,time
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import ControlMatrix,RATE
from band_preserving_decoder import load_band_preserving
from develop_neural_fidelity import snr
from benchmark_source_ceiling import spectral_error

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-waveform/band-v4'


def main():
    capacity=json.loads((ROOT/'evidence/neural-waveform/capacity-v3/development.json').read_text())
    if not capacity['complete']:raise ValueError('Finish capacity selection first')
    base=ROOT/capacity['baseline'];detail=ROOT/'evidence/neural-waveform/capacity-v3'/(capacity['selected']+'.npz')
    spec={'type':'band-preserving-neural','version':1,
        'base':{'file':str(base.relative_to(ROOT)),'sha256':hashlib.sha256(base.read_bytes()).hexdigest()},
        'detail':{'file':str(detail.relative_to(ROOT)),'sha256':hashlib.sha256(detail.read_bytes()).hexdigest()},
        'cutoffHz':8000.,'taps':129,'window':'Kaiser beta 8.0',
        'designOrigin':'Add the richer decoder only above the 8 kHz boundary identified by the earlier error audit; motivated by capacity-v3 low-band/lyric regressions',
        'scope':'Two frozen target-control waveform decoders with a complementary FIR; no novel speech, compact articulator code or independent model provenance',
        'additionalFirDelaySamples':64,'controlScalarsPerSecond':44900}
    OUT.mkdir(parents=True,exist_ok=True);path=OUT/'decoder.json'
    path.write_text(json.dumps(spec,indent=2)+'\n')
    encoder,decoder,_=load_band_preserving(path)
    report={'scope':spec['scope'],'candidate':str(path.relative_to(ROOT)),
        'candidateSha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'design':'One fixed cutoff/tap condition; no search over fresh evaluation audio',
        'controlPrefixExact':True,'results':[],'complete':False}
    for clip in capacity['corpus']:
        if clip['split']!='development':continue
        source=ROOT/clip['file']
        if hashlib.sha256(source.read_bytes()).hexdigest()!=clip['sha256']:raise ValueError('Development audio changed')
        audio,rate=sf.read(source);assert rate==RATE
        matrix=encoder.encode(audio)
        for mode,network,controls in (
                ('stable-v2',decoder.base,ControlMatrix(matrix.values[:,:decoder.base_dimensions],matrix.gains,matrix.sample_count)),
                ('capacity-v3',decoder.detail,matrix),('band-preserving-v4',decoder,matrix)):
            started=time.perf_counter();generated=network.render(controls);elapsed=time.perf_counter()-started
            row={'fixture':clip['name'],'mode':mode,'waveformSnrDb':snr(audio,generated),
                'spectralErrorDb':spectral_error(audio,generated)['meanAbsoluteDb'],
                'rawPeak':float(abs(generated).max()),'computeRealTimeFactor':elapsed/(len(audio)/RATE)}
            report['results'].append(row);print(json.dumps(row),flush=True)
    report['complete']=True
    (OUT/'development.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')


if __name__=='__main__':main()
