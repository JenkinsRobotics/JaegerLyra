"""Immediate neural-grain controls against the prior complete-score reference."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import load_model,SIZE
from compiled_neural_decoder import compile_decoder
from neural_performance import load_performance
from neural_vowel_instrument import VowelScore
from vocal_shape_control import ShapePath,load_shapes
from benchmark_vocal_shapes import evaluate
from benchmark_compact_vocal_instrument import held_metrics
from compact_vocal_instrument import VocalFrame,RATE,BLOCK
from live_neural_grains import LiveNeuralGrainStream,DELAY

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/live-neural-grains/v1'


def controlled_render(patches,decoder,score,path):
    score.validate();path.validate(len(patches),score.seconds)
    start=time.perf_counter();stream=LiveNeuralGrainStream(patches,decoder)
    initialization=time.perf_counter()-start
    count=round(score.seconds*RATE);pieces=[];timings=[]
    for start in range(0,count,BLOCK):
        begin=time.perf_counter();at=(start+BLOCK-1)/RATE
        pieces.append(stream.push(VocalFrame(float(score.frequency(at)),tuple(path.at(at)))))
        timings.append(time.perf_counter()-begin)
    begin=time.perf_counter();pieces.append(stream.finish());tail_time=time.perf_counter()-begin
    begin=time.perf_counter();audio=np.concatenate(pieces)[DELAY:DELAY+count]
    audio*=score.envelope(np.arange(count),count);consumer_time=time.perf_counter()-begin
    try:stream.push(VocalFrame(220.,tuple(path.at(0.))))
    except RuntimeError:closed=True
    else:closed=False
    return audio,{'fixedWaveformDelayMs':1000*DELAY/RATE,
        'firstPhysicalPacketComputeMs':1000*(initialization+timings[0]),
        'packetP95Ms':float(1000*np.percentile(timings,95)),
        'maximumPacketMs':float(1000*max(timings)),
        'packetsOver5Ms':sum(t>.005 for t in timings),'tailComputeMs':1000*tail_time,
        'alignmentAndEnvelopeMs':1000*consumer_time,
        'computeRealTimeFactor':(initialization+sum(timings)+tail_time+consumer_time)/score.seconds,
        'packets':len(timings),'generatedPulses':stream.generated_pulses,
        'maximumCachedWaveformSamplesUpperBound':sum(r.maximum_cache_frames for r in stream.readers)*SIZE,
        'residentSourceControlScalars':sum(r.matrix.values.size+r.matrix.gains.size for r in stream.readers),
        'decodedFrames':sum(r.decoded_frames for r in stream.readers),
        'closedPreventsOutput':closed,
        'closedClearsHistory':not any(np.any(a) for a in (stream.pending,stream.weights,stream.controls))
            and not any(r.cache for r in stream.readers),
        'timingScope':'Resident frozen compiled neural decoder and encoded <=200 ms patches. Includes reader creation, lazy neural decoding and immediate 5 ms controls. Fixed 20 ms waveform delay is additional. Excludes source capture, model loading, compilation, transport and device latency; tail and offline alignment/envelope measured separately.'}


def main():
    prior=json.loads((ROOT/'evidence/neural-vowels/v1/report.json').read_text())
    old=json.loads((ROOT/'evidence/compact-vocal/v1/report.json').read_text())
    shapes=json.loads((ROOT/'evidence/vocal-shapes/v1/report.json').read_text())
    if not all(r['complete'] for r in (prior,old,shapes)):raise ValueError('Incomplete source study')
    model=ROOT/prior['model']
    if hashlib.sha256(model.read_bytes()).hexdigest()!=prior['modelSha256']:raise ValueError('Changed decoder')
    _,decoder,_=load_model(model);decoder=compile_decoder(decoder)
    weights_before=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest()
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Immediate controlled vowel output using the frozen neural captured-grain reference',
        'model':prior['model'],'modelSha256':prior['modelSha256'],'expectedCases':39,
        'profiles':[],'results':[],'complete':False,
        'limits':['Short captured vowel microvariation is repeated; no new phonemes, words or general cloning.',
            'No new neural weights are trained; lazy waveform decoding uses the established frozen v2 network.',
            'Vowel mixing occurs after independent overlap/add normalization; captured phases need not align across vowels.',
            'Right grain width uses current pitch instead of an unknown future pulse interval, so fast changes may differ from offline output.',
            'Additional breath control is explicitly unsupported.',
            'Acoustic diagnostics and stream timings do not establish listening quality or device performance.']}
    def persist():(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    source_hashes={}
    for speaker in ('af_heart','am_michael','af_bella'):
        refs=[next(r for r in prior['references'] if r['name']==speaker+'-'+v) for v in ('ah','ee','oo')]
        patches=[]
        for ref in refs:
            file=ROOT/ref['artifact'];source_hashes[ref['artifact']]=hashlib.sha256(file.read_bytes()).hexdigest()
            patch,_=load_performance(file);patches.append(patch)
        shape=next(r for r in shapes['profiles'] if r['speaker']==speaker)
        profiles,_=load_shapes(ROOT/shape['file'])
        report['profiles'].append({'speaker':speaker,'patches':[r['artifact'] for r in refs],
            'sourceFormantsHz':[r['sourceFormantsHz'] for r in refs]})
        for reference in [r for r in old['results'] if r['speaker']==speaker
                and r['method']=='compact-harmonic-noise' and r['kind']!='breath']:
            score=VowelScore(**reference['score']);path=ShapePath(**reference['shapePath'])
            audio,timing=controlled_render(patches,decoder,score,path)
            row={k:reference[k] for k in ('speaker','kind','mode','vowel','score','shapePath') if k in reference}
            row.update(method='live-neural-grains',name=reference['name']+'-live-neural-grains')
            file=OUT/(row['name']+'.wav');sf.write(file,audio,RATE,subtype='FLOAT')
            if row['kind']=='transition':
                metrics,tracks=evaluate(audio,score,path,profiles)
                track=file.with_suffix('.npz');np.savez(track,**tracks);row['tracks']=str(track.relative_to(ROOT))
                comparison=next(r for r in shapes['results'] if r['speaker']==speaker and r['mode']==row['mode'] and r['method']=='grain-crossfade')
            else:
                i=('ah','ee','oo').index(row['vowel']);metrics=held_metrics(audio,score,refs[i]['sourceFormantsHz'])
                comparison=next(r for r in prior['results'] if r['fixture']==speaker+'-'+row['vowel'] and r['mode']==row['mode'])
            original,rate=sf.read(ROOT/comparison['file']);assert rate==RATE and len(original)==len(audio)
            row.update(status='ok',file=str(file.relative_to(ROOT)),streaming=timing,**metrics,
                offlineReference=comparison['file'],
                offlineWaveformDifference={'maximumAbsolute':float(max(abs(original-audio))),
                    'interiorRms':float(np.sqrt(np.mean((original[1200:-1200]-audio[1200:-1200])**2)))})
            report['results'].append(row);persist()
            print(row['name'],'Praat',row['praat']['p95ErrorCents'],'packet max',timing['maximumPacketMs'],flush=True)
    if weights_before!=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest():raise AssertionError('Decoder changed')
    for file,sha in source_hashes.items():
        if hashlib.sha256((ROOT/file).read_bytes()).hexdigest()!=sha:raise AssertionError('Source changed')
    report['decoderAndSourcesUnchanged']=True;report['sourceHashes']=source_hashes
    report['complete']=len(report['results'])==report['expectedCases'];persist()


if __name__=='__main__':main()
