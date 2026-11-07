"""Compare phase correction and immediate neural grains without dropping failures."""
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
OUT=ROOT/'evidence/live-neural-grains/v1'


def main():
    paths={'pulse-original':'evidence/pulse-basis/v1/report.json',
        'pulse-canonical':'evidence/pulse-basis/coherent-v2/report.json',
        'pulse-periodic-only':'evidence/pulse-basis/periodic-only-v2/report.json',
        'live':'evidence/live-neural-grains/v1/report.json',
        'compact':'evidence/compact-vocal/v1/report.json'}
    reports={k:json.loads((ROOT/v).read_text()) for k,v in paths.items()}
    if not all(r['complete'] and len(r['results'])==r['expectedCases'] for r in reports.values()):raise ValueError('Incomplete comparison')
    vowels=json.loads((ROOT/'evidence/neural-vowels/v1/report.json').read_text())
    transitions=json.loads((ROOT/'evidence/vocal-shapes/v1/report.json').read_text())
    originals=json.loads((ROOT/'evidence/neural-vowels/references/manifest.json').read_text())
    sources={r['speaker']:r['sourceFormantsHz'] for r in reports['compact']['profiles']}
    all_rows=[]
    for study,report in reports.items():
        for row in report['results']:
            if row['kind']=='breath':continue
            all_rows.append({**row,'comparisonMethod':study+'/'+row['method']})
    for row in vowels['results']:
        if row['mode'] not in ('native','down-three','up-three'):continue
        speaker,vowel=row['fixture'].rsplit('-',1)
        all_rows.append({**row,'speaker':speaker,'vowel':vowel,'kind':'held',
            'comparisonMethod':'planned-neural-grains','overFullScaleSamples':0})
    for row in transitions['results']:
        if row['method']=='grain-crossfade':all_rows.append({**row,'kind':'transition','comparisonMethod':'planned-neural-grains'})
    groups=[]
    for method in sorted({r['comparisonMethod'] for r in all_rows}):
        for kind in ('held','transition'):
            rows=[r for r in all_rows if r['comparisonMethod']==method and r['kind']==kind]
            if not rows:continue
            drift=[]
            for r in rows:
                if kind=='held':
                    drift.extend({'percent':v,'speaker':r['speaker'],'mode':r['mode'],'vowel':r['vowel'],'formant':i+1}
                        for i,v in enumerate(r['formantDriftPercent']) if v is not None)
                else:
                    for i,(observed,target) in enumerate(zip(r['plateauFormantsHz'],sources[r['speaker']])):
                        drift.extend({'percent':100*abs(v-t)/t,'speaker':r['speaker'],'mode':r['mode'],
                            'vowel':('ah','ee','oo')[i],'formant':j+1}
                            for j,(v,t) in enumerate(zip(observed,target)) if v is not None and t is not None)
            groups.append({'method':method,'kind':kind,'cases':len(rows),**pitch_summary(rows),
                'medianSourceFormantDriftPercent':stats.median(d['percent'] for d in drift),
                'worstSourceFormantDrift':max(drift,key=lambda d:d['percent']),
                'maximumRawPeak':max(r['rawPeak'] for r in rows)})
    spectral=[]
    for source in originals['clips']:
        ref=next(r for r in vowels['references'] if r['name']==source['name'])
        audio,rate=sf.read(ROOT/source['file']);assert rate==RATE
        target,_=spectral_shape(audio[round(ref['sourceStartSeconds']*RATE):round(ref['sourceEndSeconds']*RATE)])
        for r in [r for r in all_rows if r['kind']=='held' and r['mode']=='native'
                and r['speaker']==source['speaker'] and r['vowel']==source['intendedVowel']]:
            audio,rate=sf.read(ROOT/r['file']);assert rate==RATE
            observed,_=spectral_shape(audio[1200:-1200])
            spectral.append({'source':source['name'],'method':r['comparisonMethod'],
                'meanAbsoluteDb':float(np.mean(abs(observed-target)))})
    spectral_groups=[{'method':method,'cases':len(rows),'meanAbsoluteDb':stats.mean(r['meanAbsoluteDb'] for r in rows),
        'worstCase':max(rows,key=lambda r:r['meanAbsoluteDb'])}
        for method in sorted({r['method'] for r in spectral})
        for rows in [[r for r in spectral if r['method']==method]]]
    live=reports['live'];streams=[r['streaming'] for r in live['results']]
    held=[r for r in live['results'] if r['kind']=='held'];changed=[r for r in live['results'] if r['kind']=='transition']
    log=ROOT/'evidence/audio-native-live-grain-tests.txt';contents=log.read_text();match=re.search(r'Ran (\d+) tests',contents)
    if match is None or not contents.rstrip().endswith('OK'):raise ValueError('Incomplete or failing tests')
    new_rows=[r for name in ('pulse-canonical','pulse-periodic-only','live') for r in reports[name]['results']]
    summary={'scope':'Phase-correction ablations followed by immediate-control streaming of the established neural captured-grain reference',
        'reports':paths,'newCompletedCases':len(new_rows),'liveCases':len(live['results']),
        'acousticGroups':groups,
        'nativeSourceSpectrum':{'scope':'Native-pitch centered log-mel diagnostic against short original PCM; not a listening-quality or identity score',
            'groups':spectral_groups,'allCases':spectral},
        'heldWaveformComparison':{'cases':len(held),
            'maximumInteriorRmsDifference':max(r['offlineWaveformDifference']['interiorRms'] for r in held),
            'maximumWholeClipAbsoluteDifference':max(r['offlineWaveformDifference']['maximumAbsolute'] for r in held),
            'scope':'Comparison with saved float WAVs of complete-score neural grains; excludes 50 ms edges only for interior RMS'},
        'transitionWaveformComparison':{'cases':len(changed),
            'maximumInteriorRmsDifference':max(r['offlineWaveformDifference']['interiorRms'] for r in changed),
            'scope':'Immediate 5 ms controls and current-frequency grain width differ from the prior 1 kHz planned pulse schedule; not an exact-equivalence claim'},
        'streaming':{'cases':len(streams),'fixedWaveformDelayMs':20,
            'medianFirstPhysicalPacketComputeMs':stats.median(r['firstPhysicalPacketComputeMs'] for r in streams),
            'maximumFirstPhysicalPacketComputeMs':max(r['firstPhysicalPacketComputeMs'] for r in streams),
            'maximumPacketP95Ms':max(r['packetP95Ms'] for r in streams),
            'maximumSinglePacketMs':max(r['maximumPacketMs'] for r in streams),
            'packetsOver5Ms':sum(r['packetsOver5Ms'] for r in streams),
            'maximumComputeRealTimeFactor':max(r['computeRealTimeFactor'] for r in streams),
            'maximumCachedWaveformSamplesUpperBound':max(r['maximumCachedWaveformSamplesUpperBound'] for r in streams),
            'residentSourceControlScalarsRange':[min(r['residentSourceControlScalars'] for r in streams),max(r['residentSourceControlScalars'] for r in streams)],
            'allClosureChecksPassed':all(r['closedPreventsOutput'] and r['closedClearsHistory'] for r in streams),
            'timingScope':streams[0]['timingScope']},
        'allNewOutputsFinite':all(r['finite'] for r in new_rows),
        'newOverFullScaleSamples':sum(r['overFullScaleSamples'] for r in new_rows),
        'decoderAndSourcesUnchanged':live['decoderAndSourcesUnchanged'],
        'tests':{'passed':int(match.group(1)),'log':str(log.relative_to(ROOT))},
        'decision':'Keep immediate neural grains as a live-control reference using the strongest prior captured timbre, without requiring a full future score. Do not promote phase-canonical stochastic pulses: alignment alone has not resolved their generated texture/pitch problems.',
        'limitations':live['limits']+['The phase-correction experiment changes central-cycle periodization and Nyquist restriction as well as phase; a paired periodic-only seed-0 control isolates the phase step.',
            'All source vowels remain synthetic captured references, not independent articulation learning or human-reference cloning.',
            'No listening acceptance or Kokoro-level naturalness is established.'],
        'listeningAccepted':False,'completionProven':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('new cases',len(new_rows),'tests',summary['tests'],'finite',summary['allNewOutputsFinite'],'over full scale',summary['newOverFullScaleSamples'])
    for r in groups:
        if r['method'] in ('live/live-neural-grains','planned-neural-grains','pulse-canonical/correlated-seed-0','pulse-periodic-only/correlated-seed-0'):
            print(r['method'],r['kind'],'Praat/WORLD',r['praat']['worstPerCaseP95Cents'],r['world']['worstPerCaseP95Cents'],
                'formant median/worst',r['medianSourceFormantDriftPercent'],r['worstSourceFormantDrift']['percent'])
    print('native spectrum',[(r['method'],round(r['meanAbsoluteDb'],4)) for r in spectral_groups])
    print('streaming',summary['streaming']);print('held waveform',summary['heldWaveformComparison'])


if __name__=='__main__':main()
