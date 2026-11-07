"""Amplitude/phase factorial ablation with all nine captured vowel sources."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import functools
import hashlib
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from compact_vocal_instrument import load_bank as load_compact_bank
from captured_harmonic_voice import fit_harmonics,save_bank,load_bank,CapturedHarmonicStream
from benchmark_compact_vocal_instrument import controlled_render,held_metrics
from benchmark_vocal_shapes import evaluate
from neural_waveform import load_model,RATE
from neural_performance import load_performance
from neural_vowel_instrument import VowelScore
from vocal_shape_control import load_shapes,ShapePath

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/captured-harmonic/v1'
VARIANTS={'captured-amplitude':(True,False),'captured-phase':(False,True),'captured-both':(True,True)}


def main():
    prior=json.loads((ROOT/'evidence/compact-vocal/v1/report.json').read_text())
    vowels=json.loads((ROOT/'evidence/neural-vowels/v1/report.json').read_text())
    sources=json.loads((ROOT/'evidence/neural-vowels/references/manifest.json').read_text())
    if not prior['complete'] or not vowels['complete']:raise ValueError('Incomplete prior study')
    model=ROOT/vowels['model']
    if hashlib.sha256(model.read_bytes()).hexdigest()!=vowels['modelSha256']:raise ValueError('Changed neural model')
    _,decoder,_=load_model(model)
    before=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest()
    OUT.mkdir(parents=True,exist_ok=True)
    report={'scope':'Captured harmonic amplitude/phase controls; output-only finite vowel bank, not new speech',
        'neuralModel':vowels['model'],'neuralModelSha256':vowels['modelSha256'],
        'variants':VARIANTS,'expectedCases':120,'profiles':[],'results':[],'complete':False,
        'sourceFit':'Even source cycles fitted; odd cycles held out. Artifacts keep the even-cycle fit, without refitting on odd cycles.',
        'limits':['Static harmonic capture excludes cycle-to-cycle variation and changing articulation.',
            'Amplitude and residual-phase interpolation at absolute frequency are hypotheses, not a guarantee of resonance preservation.',
            'Noise amount and noise filter remain identical to the compact comparator.',
            'Burg formants and WORLD shape self-consistency do not establish listening quality or identity.',
            'Sources and control schedules are reused; no held-out-speaker claim or general cloning.']}
    def persist():
        (OUT/'report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    for speaker in ('af_heart','am_michael','af_bella'):
        compact_source=next(p for p in prior['profiles'] if p['speaker']==speaker)
        compact,metadata=load_compact_bank(ROOT/compact_source['bank'])
        profiles,_=load_shapes(ROOT/metadata['sourceBank'])
        refs=[next(r for r in vowels['references'] if r['name']==speaker+'-'+v) for v in ('ah','ee','oo')]
        harmonic=[];diagnostics=[]
        for shape,ref in zip(compact,refs):
            patch,_=load_performance(ROOT/ref['artifact']);decoded=decoder.render(patch.matrix)
            captured,diagnostic=fit_harmonics(decoded,patch.pulses,shape)
            source=next(s for s in sources['clips'] if s['name']==ref['name'])
            original,rate=sf.read(ROOT/source['file']);assert rate==RATE
            original=original[round(ref['sourceStartSeconds']*RATE):round(ref['sourceEndSeconds']*RATE)]
            _,original_diagnostic=fit_harmonics(original,patch.pulses,shape)
            diagnostic.update(source=ref['name'],sourcePatch=ref['artifact'],
                originalPcmFit=original_diagnostic,
                neuralPatchSnrDb=float(10*np.log10(np.sum(original**2)/np.sum((original-decoded)**2))))
            harmonic.append(captured);diagnostics.append(diagnostic)
        bank=OUT/(speaker+'-bank.npz')
        save_bank(bank,harmonic,{'speaker':speaker,'compactBank':compact_source['bank'],
            'compactBankSha256':compact_source['sha256'],'neuralModelSha256':vowels['modelSha256'],
            'capture':diagnostics})
        harmonic,_=load_bank(bank)
        report['profiles'].append({'speaker':speaker,'bank':str(bank.relative_to(ROOT)),
            'sha256':hashlib.sha256(bank.read_bytes()).hexdigest(),'capture':diagnostics,
            'profileScalars':sum(105+1+2*len(s.log_amplitudes) for s in harmonic),
            'sourceFormantsHz':compact_source['sourceFormantsHz']})
        persist()
        for old in [r for r in prior['results'] if r['speaker']==speaker and r['method']=='compact-harmonic-noise']:
            score=VowelScore(**old['score'])
            path=ShapePath(**old['shapePath']) if 'shapePath' in old else ShapePath((0.,),((1.,0.,0.),))
            for name,(magnitude,phase) in VARIANTS.items():
                if old['kind']=='breath' and name!='captured-both':continue
                row={k:old[k] for k in ('speaker','kind','mode','vowel','score','shapePath','breath') if k in old}
                row.update(method=name,name=old['name']+'-'+name)
                factory=functools.partial(CapturedHarmonicStream,magnitude=magnitude,phase=phase)
                audio,timing=controlled_render(harmonic,score,path,old.get('breath',0.),stream_factory=factory)
                file=OUT/(row['name']+'.wav');sf.write(file,audio,RATE,subtype='FLOAT')
                if old['kind']=='transition':
                    metrics,tracks=evaluate(audio,score,path,profiles)
                    track=file.with_suffix('.npz');np.savez(track,**tracks);row['tracks']=str(track.relative_to(ROOT))
                else:
                    ref=next(r for r in refs if r['name']==speaker+'-'+old['vowel'])
                    metrics=held_metrics(audio,score,ref['sourceFormantsHz'])
                row.update(status='ok',file=str(file.relative_to(ROOT)),streaming=timing,**metrics)
                report['results'].append(row);persist()
                print(row['name'],'Praat p95',row['praat']['p95ErrorCents'],'peak',row['rawPeak'],flush=True)
    if before!=hashlib.sha256(decoder.W.tobytes()+decoder.V.tobytes()).hexdigest():raise AssertionError('Neural weights changed')
    report['neuralWeightsUnchanged']=True;report['complete']=len(report['results'])==report['expectedCases'];persist()


if __name__=='__main__':main()
