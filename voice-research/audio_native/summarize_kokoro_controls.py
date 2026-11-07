"""Separate pretrained control improvements from independent-engine claims."""
import hashlib
import json
import re
import statistics as stats
from pathlib import Path
import numpy as np
from summarize_neural_articulation import pitch_group

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/kokoro-control-voicebox/reference-norm-v2'


def read(path):return json.loads((ROOT/path).read_text())
def test_count(path):
    content=(ROOT/path).read_text();match=re.search(r'Ran (\d+) tests',content)
    if match is None or not content.rstrip().endswith('OK'):raise ValueError('Missing successful tests: '+path)
    return int(match.group(1))


def main():
    capture=read('evidence/kokoro-control-voicebox/capture-v1/report.json')
    first=read('evidence/kokoro-control-voicebox/sustain-v1/report.json')
    revised=read('evidence/kokoro-control-voicebox/reference-norm-v2/report.json')
    evaluated=read('evidence/kokoro-control-voicebox/reference-norm-v2/evaluation.json')
    replay=read('evidence/kokoro-control-voicebox/reference-norm-v2/saved-state-replay.json')
    audition=read('public/research/control-sustain/manifest.json')
    if not all(j['complete'] for j in (capture,first,revised,evaluated,replay)):raise ValueError('Incomplete evidence')
    rows=evaluated['results'];groups=[]
    for mode in ('native','melody'):
        for method in ('legacy-kokoro','captured-cycle','whole-vowel','interior-hold',
                       'whole-vowel-reference-norm','interior-hold-reference-norm'):
            cases=[r for r in rows if r['mode']==mode and r['method']==method]
            if len(cases)!=9:raise ValueError('Missing comparison group')
            groups.append({'mode':mode,'method':method,'cases':len(cases),
                'praat':pitch_group(cases,'praat'),'world':pitch_group(cases,'world'),
                'medianHeldRms':stats.median(r['heldRms'] for r in cases),
                'homophoneEquivalentMatches':sum(r['homophoneEquivalentMatch'] for r in cases),
                'exactSpellingMatches':sum(r['exactSpellingMatch'] for r in cases),
                'transcripts':[{'name':r['name'],'transcript':r['transcript']} for r in cases]})
    chosen=[r for r in rows if r['method']=='interior-hold-reference-norm']
    previous=[r for r in rows if r['method']=='interior-hold']
    gains=[]
    for row in chosen:
        old=next(r for r in previous if r['speaker']==row['speaker'] and r['word']==row['word'] and r['mode']==row['mode'])
        gains.append({'name':row['name'],'heldRmsChangeDb':float(20*np.log10(row['heldRms']/old['heldRms']))})
    disagreements=[{'name':r['name'],'praatP95Cents':r['praat']['p95ErrorCents'],
        'worldP95Cents':r['world']['p95ErrorCents'],'file':r['file']} for r in chosen
        if abs(r['praat']['p95ErrorCents']-r['world']['p95ErrorCents'])>600]
    new=first['results']+revised['results']
    for row in audition['files']:
        if hashlib.sha256((ROOT/row['source']).read_bytes()).hexdigest()!=row['sourceSha256']:raise ValueError('Changed audition source')
        if not (ROOT/'public/research/control-sustain'/row['file']).exists():raise ValueError('Missing audition file')
    summary={'scope':revised['scope'],'controlledNewRenders':len(new),'comparativeCases':len(rows),
        'newAsrEvaluations':sum(not r['transcriptReused'] for r in rows)+len(evaluated['nativeSources']),
        'reusedAsrEvaluations':sum(r['transcriptReused'] for r in rows),
        'ordinaryControlReplay':{'cases':len(capture['results']),
            'maximumDifference':max(r['maximumReplayDifference'] for r in capture['results']),
            'vowelF0Above55FractionRange':[min(r['vowelF0Above55Fraction'] for r in capture['results']),max(r['vowelF0Above55Fraction'] for r in capture['results'])]},
        'nativeNormalizationParity':{'cases':len(revised['nativeChecks']),
            'maximumDifference':max(r['maximumDifference'] for r in revised['nativeChecks']),
            'normalizationModulesPerSource':sorted(set(r['modules'] for r in revised['nativeChecks'])),
            'statisticScalarsPerSource':sorted(set(r['statisticScalars'] for r in revised['nativeChecks']))},
        'savedStateReplay':{'cases':len(replay['results']),'maximumDifference':max(r['maximumDifference'] for r in replay['results']),
            'scope':replay['scope'],'report':'evidence/kokoro-control-voicebox/reference-norm-v2/saved-state-replay.json'},
        'groups':groups,
        'revisedInterior':{'cases':len(chosen),'minimumPraatVoicedFraction':min(r['praat']['voicedFraction'] for r in chosen),
            'minimumWorldVoicedFraction':min(r['world']['voicedFraction'] for r in chosen),
            'medianRawHeldRmsBefore':stats.median(r['heldRms'] for r in previous),
            'medianRawHeldRmsAfter':stats.median(r['heldRms'] for r in chosen),
            'medianPairedHeldRmsChangeDb':stats.median(r['heldRmsChangeDb'] for r in gains),'pairedAmplitudeChanges':gains,
            'octaveDisagreements':disagreements,
            'worstWorldPerCaseP95Cents':max(r['world']['p95ErrorCents'] for r in chosen),
            'homophoneEquivalentMatches':sum(r['homophoneEquivalentMatch'] for r in chosen),
            'previousInteriorHomophoneEquivalentMatches':sum(r['homophoneEquivalentMatch'] for r in previous)},
        'nativeSpeechDiagnostic':{'cases':len(evaluated['nativeSources']),
            'homophoneEquivalentMatches':sum(r['homophoneEquivalentMatch'] for r in evaluated['nativeSources']),
            'sources':evaluated['nativeSources']},
        'offlineDecoderTiming':{'referenceNormCases':len(revised['results']),
            'medianWholeClipComputeMs':1000*stats.median(r['wholeDecoderComputeSeconds'] for r in revised['results']),
            'maximumWholeClipComputeMs':1000*max(r['wholeDecoderComputeSeconds'] for r in revised['results']),
            'clipSeconds':2.5,'scope':'Resident pretrained model and prepared controls/source statistics; includes reference normalization hook and whole-clip decoder evaluation. Excludes loading, source-stat capture, control preparation, transport and devices. Not first-packet or streaming latency.'},
        'allNewFinite':all(r['finite'] for r in new),'maximumNewRawPeak':max(r['rawPeak'] for r in new),
        'newOverFullScaleSamples':sum(r['overFullScaleSamples'] for r in new),
        'tests':{'research':test_count('evidence/audio-native-kokoro-control-tests.txt'),
            'researchLog':'evidence/audio-native-kokoro-control-tests.txt',
            'mlxHook':test_count('evidence/kokoro-normalization-hook-tests.txt'),'mlxHookLog':'evidence/kokoro-normalization-hook-tests.txt'},
        'auditions':{'page':'http://127.0.0.1:8791/voicebox-control-lab.html',
            'manifest':'public/research/control-sustain/manifest.json','files':len(audition['files']),
            'groups':len(audition['groups']),'plots':len(audition['groups']),
            'maximumMeasuredLevelErrorLu':max(abs(r['lufs200MsGate']-r['targetLufs200MsGate']) for r in audition['files']),
            'noAutoplay':True,'listeningAccepted':False},
        'decision':'Retain the revised pretrained reference for listening and fresh-source validation. Interior-state holding and source normalization together yield fully tracked voiced sustains, but two octave disagreements and weak single-word ASR remain. This does not establish naturalness, identity or independent-engine parity.',
        'nextExperiment':'Freeze the control-hold and normalization rules, test new vowels and source words, then evaluate whether these more stationary teacher references improve the independent waveform instrument. Keep pretrained synthesis, independent reconstruction and later control learning separate. Input understanding remains deferred.',
        'limitations':evaluated['limits']+[
            'This is a source-derived adaptation of a pretrained model, not clean-room code or new neural speech-model training.',
            'Normalization statistics depend on a complete source utterance and occupy 37,396 scalars per source; not a compact universal voice identity.',
            'N is treated as an opaque acoustic control; it is not asserted to be a physical vocal-tract parameter.',
            'No semantic audio input, general voice cloning, causal streaming or listener acceptance is demonstrated by this study.'],
        'appDefaultChanged':False,'listeningAccepted':False,'completionProven':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('New controlled renders',len(new),'comparison cases',len(rows),'new ASR',summary['newAsrEvaluations'])
    print('Revised interior coverage',summary['revisedInterior']['minimumPraatVoicedFraction'],summary['revisedInterior']['minimumWorldVoicedFraction'])
    print('Octave disagreements',len(disagreements),'median paired RMS change dB',summary['revisedInterior']['medianPairedHeldRmsChangeDb'])
    print('Saved replay',summary['savedStateReplay']);print('Timing',summary['offlineDecoderTiming'])


if __name__=='__main__':main()
