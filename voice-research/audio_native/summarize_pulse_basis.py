"""Retain pulse-basis reconstruction, generated-output and latency evidence separately."""
import json
import re
import statistics as stats
from pathlib import Path
import numpy as np
import soundfile as sf
from compact_vocal_instrument import RATE
from summarize_compact_vocal_instrument import pitch_summary
from summarize_captured_harmonic_voice import spectral_shape

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/pulse-basis/v1'


def main():
    report=json.loads((OUT/'report.json').read_text())
    fit=json.loads((OUT/'fit-report.json').read_text())
    if not report['complete'] or len(report['results'])!=report['expectedCases']:raise ValueError('Incomplete pulse study')
    old=json.loads((ROOT/'evidence/compact-vocal/v1/report.json').read_text())
    vowels=json.loads((ROOT/'evidence/neural-vowels/v1/report.json').read_text())
    originals=json.loads((ROOT/'evidence/neural-vowels/references/manifest.json').read_text())
    sources={p['speaker']:p['sourceFormantsHz'] for p in old['profiles']}
    groups=[]
    for kind in ('held','transition'):
        for method in report['variants']:
            rows=[r for r in report['results'] if r['method']==method and r['kind']==kind]
            drift=[]
            for r in rows:
                if kind=='held':
                    drift.extend({'percent':v,'name':r['name'],'formant':i+1}
                        for i,v in enumerate(r['formantDriftPercent']) if v is not None)
                else:
                    for i,(observed,target) in enumerate(zip(r['plateauFormantsHz'],sources[r['speaker']])):
                        drift.extend({'percent':100*abs(v-t)/t,'name':r['name'],
                            'vowel':('ah','ee','oo')[i],'formant':j+1}
                            for j,(v,t) in enumerate(zip(observed,target)) if v is not None and t is not None)
            groups.append({'kind':kind,'method':method,'cases':len(rows),**pitch_summary(rows),
                'medianSourceFormantDriftPercent':stats.median(d['percent'] for d in drift),
                'worstSourceFormantDrift':max(drift,key=lambda d:d['percent']),
                'maximumRawPeak':max(r['rawPeak'] for r in rows),
                'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in rows)})
    native=[r for r in report['results']+old['results'] if r['kind']=='held' and r['mode']=='native']
    for r in vowels['results']:
        if r['mode']=='native':
            speaker,vowel=r['fixture'].rsplit('-',1)
            native.append({**r,'speaker':speaker,'vowel':vowel,'method':'neural-grain'})
    spectral=[]
    for source in originals['clips']:
        ref=next(r for r in vowels['references'] if r['name']==source['name'])
        audio,rate=sf.read(ROOT/source['file']);assert rate==RATE
        audio=audio[round(ref['sourceStartSeconds']*RATE):round(ref['sourceEndSeconds']*RATE)]
        target,_=spectral_shape(audio)
        for r in [r for r in native if r['speaker']==source['speaker'] and r['vowel']==source['intendedVowel']]:
            audio,rate=sf.read(ROOT/r['file']);assert rate==RATE
            observed,_=spectral_shape(audio[1200:-1200])
            spectral.append({'source':source['name'],'method':r['method'],
                'meanAbsoluteDb':float(np.mean(abs(observed-target)))})
    spectral_groups=[{'method':method,'cases':len(rows),
        'meanAbsoluteDb':stats.mean(r['meanAbsoluteDb'] for r in rows),
        'worstCase':max(rows,key=lambda r:r['meanAbsoluteDb'])}
        for method in sorted({r['method'] for r in spectral})
        for rows in [[r for r in spectral if r['method']==method]]]
    streams=[r['streaming'] for r in report['results']]
    held=[{'source':r['name'],**r['validation']} for r in fit['sources']]
    log=ROOT/'evidence/audio-native-pulse-tests.txt'
    contents=log.read_text();match=re.search(r'Ran (\d+) tests',contents)
    if match is None or not contents.rstrip().endswith('OK'):raise ValueError('Incomplete or failing test log')
    summary={'scope':report['scope'],'completedCases':len(report['results']),
        'training':{'sources':len(fit['sources']),
            'trainingPulses':sum(r['fit']['trainingPulses'] for r in fit['sources']),
            'heldPulses':sum(r['validation']['pulses'] for r in fit['sources']),
            'parametersPerVowel':{'waveformTemplateScalars':fit['sources'][0]['fit']['waveformTemplateScalars'],
                'statisticalStateScalars':fit['sources'][0]['fit']['stateScalars'],'pitchAndRms':2},
            'decoderSha256':fit['decoderSha256'],'fitReport':report['fitReport'],
            'capturedWaveformTemplates':True,'newNeuralModel':False},
        'heldRepresentation':{'scope':held[0]['scope'],'medianMeanTemplateSnrDb':stats.median(r['meanTemplateSnrDb'] for r in held),
            'medianProjectedEightModeSnrDb':stats.median(r['projectedEightModeSnrDb'] for r in held),
            'minimumProjectedEightModeSnrDb':min(r['projectedEightModeSnrDb'] for r in held),'allCases':held},
        'generatedOutput':groups,
        'nativeSourceSpectrum':{'scope':'Centered log-mel shape against original short source PCM; diagnostic only. Pulse fitting uses different repeated-vowel recordings.',
            'groups':spectral_groups,'allCases':spectral},
        'streaming':{'cases':len(streams),'fixedWaveformDelayMs':streams[0]['outputDelayMs'],
            'medianFirstPhysicalPacketComputeMs':stats.median(r['firstPhysicalPacketComputeMs'] for r in streams),
            'maximumFirstPhysicalPacketComputeMs':max(r['firstPhysicalPacketComputeMs'] for r in streams),
            'maximumPacketP95Ms':max(r['packetP95Ms'] for r in streams),
            'maximumSinglePacketMs':max(r['maximumPacketMs'] for r in streams),
            'maximumComputeRealTimeFactor':max(r['computeRealTimeFactor'] for r in streams),
            'allClosureChecksPassed':all(r['closedPreventsOutput'] and r['closedClearsHistory'] for r in streams),
            'timingScope':streams[0]['timingScope']},
        'allFinite':all(r['finite'] for r in report['results']),
        'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in report['results']),
        'banksUnchanged':report['banksUnchanged'],'excludedCases':report['excludedCases'],
        'tests':{'passed':int(match.group(1)),'log':str(log.relative_to(ROOT))},
        'decision':'Do not adopt pulse-basis variation as the quality default. A representation that reconstructs captured pulses is not enough: generated coefficients also need coherent excitation and accurate pitch. Inspect phase alignment and timing before expanding articulation.',
        'limitations':report['limits']+['Stationary Gaussian state covariance is prescribed, not a learned human phonation process.',
            'Held pulse coefficients are encoded from the target for the representation test only.',
            '20 ms output delay is separate from the measured packet computation time.',
            'No listening acceptance or Kokoro-level naturalness established.'],
        'listeningAccepted':False,'completionProven':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('completed',summary['completedCases'],'finite',summary['allFinite'],'over-full-scale',summary['overFullScaleSamples'])
    print('held representation',summary['heldRepresentation']['medianProjectedEightModeSnrDb'])
    for r in groups:print(r['kind'],r['method'],'Praat/WORLD worst',r['praat']['worstPerCaseP95Cents'],r['world']['worstPerCaseP95Cents'],
        'voiced',r['praat']['minimumVoicedFraction'],r['world']['minimumVoicedFraction'])
    for r in spectral_groups:print('native spectrum',r['method'],r['meanAbsoluteDb'])
    print('streaming',summary['streaming'])


if __name__=='__main__':main()
