"""Compare all compact-instrument cases, retaining negative diagnostics."""
import json
import re
import statistics as stats
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/compact-vocal/v1'


def pitch_summary(rows):
    result={}
    for estimator in ('praat','world'):
        values=[r[estimator]['p95ErrorCents'] for r in rows if r[estimator]['p95ErrorCents'] is not None]
        result[estimator]={'medianPerCaseP95Cents':stats.median(values),
            'worstPerCaseP95Cents':max(values),'minimumVoicedFraction':min(r[estimator]['voicedFraction'] for r in rows)}
    return result


def main():
    report=json.loads((OUT/'report.json').read_text())
    if not report['complete'] or len(report['results'])!=report['expectedCases']:
        raise ValueError('Incomplete compact study')
    prior=json.loads((ROOT/'evidence/vocal-shapes/v1/report.json').read_text())
    vowels=json.loads((ROOT/'evidence/neural-vowels/v1/report.json').read_text())
    sources={r['speaker']:r['sourceFormantsHz'] for r in report['profiles']}
    transition_groups=[]
    rows=[r for r in report['results'] if r['kind']=='transition']+prior['results']
    for method in ('compact-harmonic-noise','grain-crossfade','world-shapes'):
        selected=[r for r in rows if r['method']==method and r['status']=='ok']
        drift=[]
        for row in selected:
            for vowel,(observed,target) in enumerate(zip(row['plateauFormantsHz'],sources[row['speaker']])):
                for formant,(value,reference) in enumerate(zip(observed,target)):
                    if value is not None and reference is not None:
                        drift.append({'percent':100*abs(value-reference)/reference,'speaker':row['speaker'],
                            'mode':row['mode'],'vowel':('ah','ee','oo')[vowel],'formant':formant+1,
                            'sourceHz':reference,'measuredHz':value})
        transition_groups.append({'method':method,'cases':len(selected),**pitch_summary(selected),
            'meanShapeSelfConsistencyErrorDb':stats.mean(r['shapeSelfConsistency']['meanAbsoluteDb'] for r in selected),
            'medianEstimatedFormantDriftPercent':stats.median(d['percent'] for d in drift),
            'worstEstimatedFormantDrift':max(drift,key=lambda r:r['percent']),
            'maximumRawPeak':max(r['rawPeak'] for r in selected),
            'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in selected)})
    held=[r for r in report['results'] if r['kind']=='held']
    for row in vowels['results']:
        if row['mode'] not in ('native','down-three','up-three') or row['status']!='ok':continue
        speaker,vowel=row['fixture'].rsplit('-',1)
        held.append({**row,'speaker':speaker,'vowel':vowel,'method':'neural-grain',
            'overFullScaleSamples':0})
    held_groups=[]
    for method in ('compact-harmonic-noise','world-shapes','neural-grain'):
        selected=[r for r in held if r['method']==method]
        drift=[];pitch_drift=[]
        for row in selected:
            for formant,value in enumerate(row['formantDriftPercent']):
                if value is not None:drift.append({'percent':value,'speaker':row['speaker'],
                    'vowel':row['vowel'],'mode':row['mode'],'formant':formant+1})
            if row['mode']=='native':continue
            native=next(r for r in selected if r['speaker']==row['speaker'] and r['vowel']==row['vowel'] and r['mode']=='native')
            for formant,(value,reference) in enumerate(zip(row['measuredFormantsHz'],native['measuredFormantsHz'])):
                if value is not None and reference is not None:
                    pitch_drift.append({'percent':100*abs(value-reference)/reference,'speaker':row['speaker'],
                        'vowel':row['vowel'],'mode':row['mode'],'formant':formant+1})
        held_groups.append({'method':method,'cases':len(selected),**pitch_summary(selected),
            'medianSourceFormantDriftPercent':stats.median(d['percent'] for d in drift),
            'worstSourceFormantDrift':max(drift,key=lambda d:d['percent']),
            'medianPitchInducedFormantDriftPercent':stats.median(d['percent'] for d in pitch_drift),
            'worstPitchInducedFormantDrift':max(pitch_drift,key=lambda d:d['percent']),
            'maximumRawPeak':max(r['rawPeak'] for r in selected),
            'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in selected)})
    streaming=[r['streaming'] for r in report['results'] if r['method']=='compact-harmonic-noise']
    capture=[c for p in report['profiles'] for c in p['capture']]
    breath=[{'speaker':r['speaker'],**pitch_summary([r]),'rawPeak':r['rawPeak'],
        'formantDriftPercent':r['formantDriftPercent']} for r in report['results'] if r['kind']=='breath']
    summary={'scope':report['scope'],'method':report['method'],'completedCases':len(report['results']),
        'priorReusedComparisons':len(prior['results'])+27,
        'profileParameters':report['profileParameters'],'liveControls':report['liveControls'],
        'capture':{'meanEnvelopeErrorDb':stats.mean(c['meanEnvelopeErrorDb'] for c in capture),
            'worstPerVowelP95EnvelopeErrorDb':max(c['p95EnvelopeErrorDb'] for c in capture)},
        'transitions':transition_groups,'heldVowels':held_groups,'breathCases':breath,
        'streaming':{'cases':len(streaming),
            'medianFirstPacketComputeMs':stats.median(r['firstPacketComputeMs'] for r in streaming),
            'maximumFirstPacketComputeMs':max(r['firstPacketComputeMs'] for r in streaming),
            'maximumPacketP95Ms':max(r['packetP95Ms'] for r in streaming),
            'maximumSinglePacketMs':max(r['maximumPacketMs'] for r in streaming),
            'maximumComputeRealTimeFactor':max(r['computeRealTimeFactor'] for r in streaming),
            'allCancellationChecksPassed':all(r['cancelPreventsOutput'] and r['cancelClearsHistory'] for r in streaming),
            'timingScope':streaming[0]['timingScope']},
        'neuralWeightsUnchanged':report['neuralWeightsUnchanged'],
        'tests':{'passed':int(re.search(r'Ran (\d+) tests',
            (ROOT/'evidence/audio-native-compact-tests.txt').read_text()).group(1)),
            'log':'evidence/audio-native-compact-tests.txt'},
        'decision':'Keep the compact immediate-control instrument as a research option, not a quality replacement. Pitch accuracy and control latency improve over captured grains, but estimated resonance shifts are larger and a breath case causes an octave disagreement. The next quality work must address excitation texture and captured envelope/phase, not merely sharpen pitch metrics.',
        'listening':{'accepted':False,'url':'http://127.0.0.1:8791/voicebox-instrument-lab.html',
            'manifest':'public/research/compact-vocal/manifest.json',
            'selection':'All three voices, held-ah and melody; source guides and all three breath controls retained'},
        'limitations':report['limits']+['Finite-bank compact synthesis is not waveform reconstruction or a general compression result.',
            'Comparisons reuse previous vowel sources and control schedules; no new held-out speaker claim.',
            'No listening acceptance has been recorded; objective pitch/resonance results do not prove Kokoro quality.'],
        'completionProven':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    for group in transition_groups:
        print('transition',group['method'],'Praat p95 worst',round(group['praat']['worstPerCaseP95Cents'],3),
            'formant median/worst',round(group['medianEstimatedFormantDriftPercent'],2),round(group['worstEstimatedFormantDrift']['percent'],2))
    for group in held_groups:
        print('held',group['method'],'source drift median/worst',round(group['medianSourceFormantDriftPercent'],2),
            round(group['worstSourceFormantDrift']['percent'],2),'pitch-induced median/worst',
            round(group['medianPitchInducedFormantDriftPercent'],2),round(group['worstPitchInducedFormantDrift']['percent'],2))


if __name__=='__main__':main()
