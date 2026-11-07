"""Autonomous texture results with predictor, clipping and seed limitations."""
import json
import re
import statistics as stats
from pathlib import Path
import numpy as np
import soundfile as sf
from summarize_compact_vocal_instrument import pitch_summary
from summarize_captured_harmonic_voice import spectral_shape
from compact_vocal_instrument import RATE

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/generated-texture/v1'


def main():
    report=json.loads((OUT/'report.json').read_text())
    fit=json.loads((OUT/'fit-report.json').read_text())
    if not report['complete'] or len(report['results'])!=report['expectedCases']:raise ValueError('Incomplete texture render study')
    old=json.loads((ROOT/'evidence/compact-vocal/v1/summary.json').read_text())
    previous=json.loads((ROOT/'evidence/compact-vocal/v1/report.json').read_text())
    vowels=json.loads((ROOT/'evidence/neural-vowels/v1/report.json').read_text())
    originals=json.loads((ROOT/'evidence/neural-vowels/references/manifest.json').read_text())
    sources={p['speaker']:p['sourceFormantsHz'] for p in previous['profiles']}
    transitions=[];held=[]
    for method in report['variants']:
        rows=[r for r in report['results'] if r['method']==method and r['kind']=='transition']
        drift=[]
        for row in rows:
            for i,(observed,target) in enumerate(zip(row['plateauFormantsHz'],sources[row['speaker']])):
                for j,(value,reference) in enumerate(zip(observed,target)):
                    if value is not None and reference is not None:
                        drift.append({'percent':100*abs(value-reference)/reference,'speaker':row['speaker'],
                            'mode':row['mode'],'vowel':('ah','ee','oo')[i],'formant':j+1})
        transitions.append({'method':method,'cases':len(rows),**pitch_summary(rows),
            'medianSourceFormantDriftPercent':stats.median(d['percent'] for d in drift),
            'worstSourceFormantDrift':max(drift,key=lambda d:d['percent']),
            'meanShapeSelfConsistencyErrorDb':stats.mean(r['shapeSelfConsistency']['meanAbsoluteDb'] for r in rows),
            'maximumRawPeak':max(r['rawPeak'] for r in rows)})
        rows=[r for r in report['results'] if r['method']==method and r['kind']=='held']
        drift=[{'percent':value,'speaker':r['speaker'],'vowel':r['vowel'],'mode':r['mode'],'formant':j+1}
            for r in rows for j,value in enumerate(r['formantDriftPercent']) if value is not None]
        held.append({'method':method,'cases':len(rows),**pitch_summary(rows),
            'medianSourceFormantDriftPercent':stats.median(d['percent'] for d in drift),
            'worstSourceFormantDrift':max(drift,key=lambda d:d['percent']),
            'maximumRawPeak':max(r['rawPeak'] for r in rows)})
    # Reuse all prior static-reference summaries as historical comparisons.
    transitions+=old['transitions'];held+=old['heldVowels']
    native=[r for r in report['results']+previous['results'] if r['kind']=='held' and r['mode']=='native']
    for row in vowels['results']:
        if row['mode']=='native':
            speaker,vowel=row['fixture'].rsplit('-',1)
            native.append({**row,'speaker':speaker,'vowel':vowel,'method':'neural-grain'})
    spectral=[];tracks={}
    for source in originals['clips']:
        ref=next(r for r in vowels['references'] if r['name']==source['name'])
        audio,rate=sf.read(ROOT/source['file']);assert rate==RATE
        audio=audio[round(ref['sourceStartSeconds']*RATE):round(ref['sourceEndSeconds']*RATE)]
        target,hz=spectral_shape(audio);tracks[source['name']+'-original']=target
        for row in [r for r in native if r['speaker']==source['speaker'] and r['vowel']==source['intendedVowel']]:
            audio,rate=sf.read(ROOT/row['file']);assert rate==RATE
            observed,_=spectral_shape(audio[int(.05*RATE):-int(.05*RATE)])
            spectral.append({'source':source['name'],'method':row['method'],'meanAbsoluteDb':float(np.mean(abs(observed-target)))})
            tracks[source['name']+'-'+row['method']]=observed
    np.savez(OUT/'native-spectral-shapes.npz',frequencyHz=hz,**tracks)
    spectral_groups=[{'method':method,'cases':len(rows),'meanAbsoluteDb':stats.mean(r['meanAbsoluteDb'] for r in rows),
        'worstCase':max(rows,key=lambda r:r['meanAbsoluteDb'])}
        for method in sorted({r['method'] for r in spectral}) for rows in [[r for r in spectral if r['method']==method]]]
    validation=[]
    for horizon in (1,5,10):
        rows=[{'source':source['name'],**next(v for v in source['validation'] if v['horizonFrames']==horizon)} for source in fit['sources']]
        eligible=[r for r in rows if r['eligibleFrames']]
        validation.append({'horizonMs':horizon*5,'sources':len(eligible),'eligibleFrames':sum(r['eligibleFrames'] for r in eligible),
            'beatsMean':sum(r['rmse']['learned']<r['rmse']['mean'] for r in eligible),
            'beatsPersistence':sum(r['rmse']['learned']<r['rmse']['persistence'] for r in eligible),
            'pooledRmse':{method:float(np.sqrt(sum(r['rmse'][method]**2*r['eligibleFrames'] for r in eligible)/sum(r['eligibleFrames'] for r in eligible)))
                for method in ('mean','persistence','learned')},'allCases':rows})
    stream=[r['streaming'] for r in report['results']]
    summary={'scope':report['scope'],'completedCases':len(report['results']),
        'model':'Three PCA directions of log-spectral shape and a bounded linear stochastic transition; generated innovations, no recorded trajectory',
        'sourceTraining':{'clips':len(fit['sources']),'seconds':sum(r['sourceSeconds'] for r in fit['sources']),
            'trainFrames':sum(r['fit']['trainingFrames'] for r in fit['sources']),
            'heldSegmentFrames':sum(d['frames'] for r in fit['sources'] for d in r['analysisSegments'] if d['split']=='validation'),
            'retainedVarianceRange':[min(r['fit']['retainedVarianceFraction'] for r in fit['sources']),max(r['fit']['retainedVarianceFraction'] for r in fit['sources'])],
            'split':fit['analysis']['split'],'decoderSha256':fit['decoderSha256'],'fitReport':report['fitReport']},
        'forecastValidation':validation,'transitions':transitions,'heldVowels':held,
        'nativeSourceSpectrum':{'scope':'Original-PCM native-pitch centered log-mel spectrum diagnostic; same definition as captured-harmonic study, not a perceptual or identity score',
            'groups':spectral_groups,'allCases':spectral},
        'modulationLimits':{'logAmplitudeLimit':.5,'maximumPreNormalizationFactor':float(np.exp(.5)),
            'medianClippedGridFraction':stats.median(r['clippedEnvelopeFraction'] for r in report['results']),
            'maximumClippedGridFraction':max(r['clippedEnvelopeFraction'] for r in report['results']),
            'meaning':'Fraction of spectral-grid values at frame endpoints that exceed the bound before clipping; not PCM clipping. Power normalization follows.'},
        'breathCases':[{'speaker':r['speaker'],'method':r['method'],**pitch_summary([r]),'rawPeak':r['rawPeak'],
            'formantDriftPercent':r['formantDriftPercent']} for r in report['results'] if r['kind']=='breath'],
        'streaming':{'cases':len(stream),'medianFirstPacketComputeMs':stats.median(r['firstPacketComputeMs'] for r in stream),
            'maximumFirstPacketComputeMs':max(r['firstPacketComputeMs'] for r in stream),
            'maximumPacketP95Ms':max(r['packetP95Ms'] for r in stream),'maximumSinglePacketMs':max(r['maximumPacketMs'] for r in stream),
            'maximumComputeRealTimeFactor':max(r['computeRealTimeFactor'] for r in stream),
            'allCancellationChecksPassed':all(r['cancelPreventsOutput'] and r['cancelClearsHistory'] for r in stream)
                and all(r['cancelledLatentCleared'] for r in report['results']),
            'timingScope':stream[0]['timingScope'],'state':'Fixed model arrays, three latent values per vowel, independent RNG, previous frame/filter/harmonics and 256 noise samples; no trajectory growth'},
        'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in report['results']),
        'banksUnchanged':report['banksUnchanged'],
        'decision':'Retain autonomous state generation as a working control capability, without selecting a quality replacement. Both learned seeds improve native source-spectrum error, but independent-frame variation improves it more; learned timing has not earned a listening-quality advantage. Modulation clipping and resonance/pitch-tracker outliers remain.',
        'listening':{'accepted':False,'url':'http://127.0.0.1:8791/voicebox-texture-lab.html',
            'manifest':'public/research/generated-texture/manifest.json','scope':'All three voices, held-ah and melody, four main methods plus a second learned take and original Kokoro guides'},
        'limitations':report['limits']+[
            'Training cepstral tracks also contain articulation and analysis-window effects; learned variation is not isolated natural roughness.',
            'Modulation clipping changes the learned statistics; stationary innovation calibration is a modelling assumption.',
            'No listening acceptance or Kokoro-level naturalness is established.'],'completionProven':False}
    log=ROOT/'evidence/audio-native-texture-tests.txt'
    if log.exists():
        contents=log.read_text();match=re.search(r'Ran (\d+) tests',contents)
        if match is None or not contents.rstrip().endswith('OK'):
            raise ValueError('The research test log is incomplete or has failures')
        summary['tests']={'passed':int(match.group(1)),'log':str(log.relative_to(ROOT))}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for r in validation:print('forecast',r['horizonMs'],'beats persistence',r['beatsPersistence'],'/',r['sources'],'RMSE',r['pooledRmse'])
    for r in spectral_groups:print('native source spectrum',r['method'],round(r['meanAbsoluteDb'],3))
    for r in held[:3]:print('held',r['method'],'formant median/worst',round(r['medianSourceFormantDriftPercent'],2),round(r['worstSourceFormantDrift']['percent'],2))
    print('Modulation limits',summary['modulationLimits'])


if __name__=='__main__':main()
