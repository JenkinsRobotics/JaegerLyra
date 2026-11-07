"""Keep acoustic, streaming and lexical evidence separate for captured syllables."""
import json
import re
import statistics as stats
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-articulation/v1'


def pitch_group(rows,key):
    valid=[r for r in rows if r[key]['eligibleFrames']>0]
    measured=[r for r in valid if r[key]['p95ErrorCents'] is not None]
    return {'eligibleCases':len(valid),'casesWithoutEligibleFrames':len(rows)-len(valid),
        'minimumEligibleFrames':min((r[key]['eligibleFrames'] for r in valid),default=0),
        'minimumVoicedFraction':min((r[key]['voicedFraction'] for r in valid),default=None),
        'medianPerCaseP95Cents':stats.median(r[key]['p95ErrorCents'] for r in measured) if measured else None,
        'worstCase':max(({'name':r['name'],'p95ErrorCents':r[key]['p95ErrorCents']} for r in measured),
            key=lambda d:d['p95ErrorCents'],default=None)}


def main():
    capture=json.loads((OUT/'capture-report.json').read_text())
    long=json.loads((OUT/'report.json').read_text())
    short=json.loads((ROOT/'evidence/neural-articulation/short-v1/report.json').read_text())
    words=json.loads((OUT/'word-report.json').read_text())
    short_words=json.loads((ROOT/'evidence/neural-articulation/short-v1/word-report.json').read_text())
    if not all(r['complete'] for r in (capture,long,short,words,short_words)):raise ValueError('Incomplete articulation study')
    rows=long['results']+short['results'];groups=[]
    for mode in dict.fromkeys(r['mode'] for r in rows):
        selected=[r for r in rows if r['mode']==mode]
        drift=[{'name':r['name'],'formant':i+1,'percent':v} for r in selected
            for i,v in enumerate(r['formantDriftPercent']) if v is not None]
        groups.append({'mode':mode,'cases':len(selected),
            **{key:pitch_group(selected,key) for key in ('praat','world','praatVoicedAttack','worldVoicedAttack')},
            'formantComparisons':len(drift),'possibleFormantComparisons':len(selected)*3,
            'medianFormantDriftPercent':stats.median(r['percent'] for r in drift) if drift else None,
            'worstFormantDrift':max(drift,key=lambda d:d['percent'],default=None)})
    word_rows=words['results']+[r for r in short_words['results'] if r['mode'].startswith('short-')]
    lexical=[]
    for mode in dict.fromkeys(r['mode'] for r in word_rows):
        selected=[r for r in word_rows if r['mode']==mode]
        lexical.append({'mode':mode,'cases':len(selected),
            'exactSpellingMatches':sum(r['exactSpellingMatch'] for r in selected),
            'homophoneEquivalentMatches':sum(r['homophoneEquivalentMatch'] for r in selected),
            'mismatches':[{'name':r['name'],'transcript':r['transcript']} for r in selected if not r['homophoneEquivalentMatch']]})
    preserved=[]
    for unit in capture['units']:
        source=next(r for r in word_rows if r['name']==unit['name']+'-source')
        decoded=next(r for r in word_rows if r['name']==unit['name']+'-decoded-source')
        preserved.append({'name':unit['name'],'source':source['transcript'],'decoded':decoded['transcript'],
            'sameNormalizedTranscript':source['normalized']==decoded['normalized']})
    streams=[r['streaming'] for r in rows]
    unvoiced=[r for r in rows if r['strictUnvoicedPrefix']['samples']]
    log=ROOT/'evidence/audio-native-articulation-tests.txt';contents=log.read_text();match=re.search(r'Ran (\d+) tests',contents)
    if match is None or not contents.rstrip().endswith('OK'):raise ValueError('Incomplete or failing research tests')
    summary={'scope':long['scope'],'completedRenders':len(rows),'longCases':len(long['results']),
        'durationMatchedCases':len(short['results']),'uniqueAsrEvaluations':len(word_rows),
        'capture':{'units':len(capture['units']),'model':capture['model'],'modelSha256':capture['modelSha256'],
            'medianReconstructionSnrDb':stats.median(r['neuralSourceReconstructionSnrDb'] for r in capture['units']),
            'minimumReconstructionSnrDb':min(r['neuralSourceReconstructionSnrDb'] for r in capture['units']),
            'missingSourceFormantUnits':[r['name'] for r in capture['units'] if not any(v is not None for v in r['sourceFormantsHz'])],
            'selection':capture['selection'],'captureReport':long['captureReport']},
        'acousticGroups':groups,
        'strictUnvoicedPreservation':{'eligibleCases':len(unvoiced),'totalCases':len(rows),
            'maximumAbsoluteDifferenceFromNeuralSource':max(r['strictUnvoicedPrefix']['maximumAbsoluteDifference'] for r in unvoiced),
            'scope':'Only source-detected unvoiced prefixes at least 20 ms before the first valid voiced pulse group and after the initial 20 ms. Compared to neural source reconstruction, not original PCM or human annotations.'},
        'lexicalDiagnostic':{'groups':lexical,'sourceReconstructionPairs':preserved,
            'equivalents':words['equivalents'],'report':'evidence/neural-articulation/v1/word-report.json',
            'durationMatchedReport':'evidence/neural-articulation/short-v1/word-report.json',
            'limits':words['limits']},
        'streaming':{'cases':len(streams),'fixedWaveformDelayMs':20.,
            'medianFirstPhysicalPacketComputeMs':stats.median(r['firstPhysicalPacketComputeMs'] for r in streams),
            'maximumFirstPhysicalPacketComputeMs':max(r['firstPhysicalPacketComputeMs'] for r in streams),
            'maximumPacketP95Ms':max(r['packetP95Ms'] for r in streams),
            'maximumSinglePacketMs':max(r['maximumPacketMs'] for r in streams),
            'packetsOver5Ms':sum(r['packetsOver5Ms'] for r in streams),
            'maximumComputeRealTimeFactor':max(r['computeRealTimeFactor'] for r in streams),
            'maximumCachedWaveformSamples':max(r['maximumCachedWaveformSamples'] for r in streams),
            'residentSourceControlScalarsRange':[min(r['residentSourceControlScalars'] for r in streams),max(r['residentSourceControlScalars'] for r in streams)],
            'allClosureChecksPassed':all(r['closedPreventsOutput'] and r['closedClearsHistory'] for r in streams),
            'timingScope':streams[0]['timingScope']},
        'allFinite':all(r['finite'] for r in rows),'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in rows),
        'maximumRawPeak':max(r['rawPeak'] for r in rows),
        'decoderAndSourcesUnchanged':long['decoderAndSourcesUnchanged'] and short['decoderAndSourcesUnchanged'],
        'tests':{'passed':int(match.group(1)),'log':str(log.relative_to(ROOT))},
        'decision':'Retain captured onset-to-sustain control as an output capability, without declaring pronunciation or naturalness parity. Source ASR already misses intended words, and controlled long sustains produce further lexical errors. Use the duration-matched follow-up and raw auditions to distinguish onset preservation from repeated-vowel artifacts.',
        'limitations':long['limits']+[
            'Three short source nuclei have no formant estimate under the fixed analysis margins; missing measurements are explicit.',
            'Pitch/resonance diagnostics on short follow-ups may have very few eligible frames.',
            'The native condition means a constant selected nucleus pitch, not replay of the original source intonation.',
            'No new neural weights or general pronunciation model; the synthetic source units remain necessary.',
            'No listener acceptance, identity guarantee or Kokoro-level quality established.'],
        'auditions':'evidence/neural-articulation/v1/audition/manifest.json',
        'listeningAccepted':False,'completionProven':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('renders',len(rows),'ASR',len(word_rows),'tests',summary['tests']['passed'],'peak',summary['maximumRawPeak'])
    for r in lexical:print(r['mode'],r['homophoneEquivalentMatches'],'/',r['cases'])
    print('unvoiced',summary['strictUnvoicedPreservation']);print('streaming',summary['streaming'])
    for r in groups:print(r['mode'],'held',r['praat']['worstCase'],r['world']['worstCase'],'formant comparisons',r['formantComparisons'])


if __name__=='__main__':main()
