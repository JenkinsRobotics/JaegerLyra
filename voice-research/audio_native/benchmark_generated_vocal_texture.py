"""Render autonomous learned/iid texture with fixed seeds and retained cases."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from compact_vocal_instrument import load_bank,GRID,RATE
from generated_vocal_texture import load_models,GeneratedTextureStream
from benchmark_compact_vocal_instrument import controlled_render,held_metrics
from benchmark_vocal_shapes import evaluate
from neural_vowel_instrument import VowelScore
from vocal_shape_control import ShapePath,load_shapes

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/generated-texture/v1'
VARIANTS={'learned-seed-0':(True,0),'learned-seed-1':(True,1),'iid-seed-0':(False,0)}


def main():
    fit=json.loads((OUT/'fit-report.json').read_text())
    old=json.loads((ROOT/'evidence/compact-vocal/v1/report.json').read_text())
    if not fit['complete'] or not old['complete']:raise ValueError('Incomplete input study')
    report={'scope':'Autonomous generated amplitude texture with immediate vowel/pitch controls; no source trajectory or target recording at runtime',
        'fitReport':'evidence/generated-texture/v1/fit-report.json','variants':VARIANTS,
        'expectedCases':126,'results':[],'complete':False,
        'limits':['Three-dimensional linear stochastic states, not a newly trained neural speech model.',
            'Only harmonic amplitude varies; phase, noise and total harmonic power use the compact comparator.',
            'Stationary Gaussian innovation is a modelling assumption, not a learned human phonation process.',
            'Two seeds test limited reproducibility, not broad probabilistic robustness.',
            'Captured vowel labels and voice identity remain unverified by listening.']}
    def persist():(OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    bank_hashes={}
    for speaker in ('af_heart','am_michael','af_bella'):
        source=next(p for p in old['profiles'] if p['speaker']==speaker)
        shapes,metadata=load_bank(ROOT/source['bank'])
        profiles,_=load_shapes(ROOT/metadata['sourceBank'])
        bank=next(p for p in fit['banks'] if p['speaker']==speaker)
        file=ROOT/bank['file']
        if hashlib.sha256(file.read_bytes()).hexdigest()!=bank['sha256']:raise ValueError('Changed dynamics bank')
        bank_hashes[bank['file']]=bank['sha256'];models,_=load_models(file)
        for reference in [r for r in old['results'] if r['speaker']==speaker and r['method']=='compact-harmonic-noise']:
            score=VowelScore(**reference['score'])
            path=ShapePath(**reference['shapePath']) if 'shapePath' in reference else ShapePath((0.,),((1.,0.,0.),))
            for name,(correlated,chosen_seed) in VARIANTS.items():
                instances=[]
                def factory(shapes,seed):
                    stream=GeneratedTextureStream(shapes,models,seed=chosen_seed,correlated=correlated)
                    instances.append(stream);return stream
                audio,timing=controlled_render(shapes,score,path,reference.get('breath',0.),stream_factory=factory)
                row={k:reference[k] for k in ('speaker','kind','mode','vowel','score','shapePath','breath') if k in reference}
                row.update(method=name,name=reference['name']+'-'+name)
                file=OUT/(row['name']+'.wav');sf.write(file,audio,RATE,subtype='FLOAT')
                if reference['kind']=='transition':
                    metrics,tracks=evaluate(audio,score,path,profiles)
                    track=file.with_suffix('.npz');np.savez(track,**tracks);row['tracks']=str(track.relative_to(ROOT))
                else:
                    i=('ah','ee','oo').index(reference['vowel'])
                    metrics=held_metrics(audio,score,source['sourceFormantsHz'][i])
                stream=instances[0]
                row.update(status='ok',file=str(file.relative_to(ROOT)),streaming=timing,**metrics,
                    generatedTextureFrames=stream.generated_frames,
                    clippedEnvelopeFraction=stream.clipped_grid_values/max(1,stream.generated_frames*len(GRID)),
                    cancelledLatentCleared=not np.any(stream.state),
                    modelBank=bank['file'],modelBankSha256=bank['sha256'])
                report['results'].append(row);persist()
                print(row['name'],'Praat p95',row['praat']['p95ErrorCents'],'clipped',row['clippedEnvelopeFraction'],flush=True)
    for file,sha in bank_hashes.items():
        if hashlib.sha256((ROOT/file).read_bytes()).hexdigest()!=sha:raise AssertionError('Dynamics bank changed')
    report['banksUnchanged']=True;report['complete']=len(report['results'])==report['expectedCases'];persist()


if __name__=='__main__':main()
