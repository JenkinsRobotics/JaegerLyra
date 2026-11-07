"""Keep control accuracy, reconstruction and identity diagnostics separate."""
import json
import statistics
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-vowels/v1'


def main():
    report=json.loads((OUT/'report.json').read_text())
    identity=json.loads((OUT/'identity.json').read_text())
    if not report['complete'] or not identity['complete']:
        raise ValueError('Incomplete vowel evidence')
    if len(report['results'])!=report['expectedCases']:
        raise ValueError('Missing requested control cases')
    rows=report['results'];good=[r for r in rows if r['status']=='ok']
    def speaker_stats(values):
        return {'cases':len(values),'nearestSpeakerMatches':sum(x['matchesSourceSpeaker'] for x in values),
            'meanSourceSpeakerCosine':statistics.mean(x['sourceSpeakerCosine'] for x in values),
            'meanSameVowelOriginalCosine':statistics.mean(x['sameVowelOriginalCosine'] for x in values)}
    modes=[]
    for mode in ('native','down-three','up-three','vibrato','melody','long-hold'):
        selected=[r for r in good if r['mode']==mode]
        item={'mode':mode,'completedCases':len(selected)}
        for method in ('praat','world'):
            item[method]={'medianOfPerClipP95ErrorCents':statistics.median(r[method]['p95ErrorCents'] for r in selected),
                'worstPerClipP95ErrorCents':max(r[method]['p95ErrorCents'] for r in selected),
                'minimumVoicedFraction':min(r[method]['voicedFraction'] for r in selected)}
        item['identityDiagnostic']=speaker_stats([r for r in identity['results'] if r['mode']==mode and r['status']=='ok'])
        modes.append(item)
    originals=[r for r in report['references'] if r['status']=='ok']
    summary={'scope':'New sustained note performances from short captured vowel sounds; a sample-based instrument around frozen neural waveform decoding',
        'model':report['model'],'modelSha256':report['modelSha256'],'weightsUnchanged':report['weightsUnchanged'],
        'completedCases':len(rows),'successfulExecutions':len(good),'referenceCount':len(originals),
        'minimumSourceSeconds':min(r['retainedSeconds'] for r in originals),
        'maximumSourceSeconds':max(r['retainedSeconds'] for r in originals),
        'maximumOutputSeconds':max(r['durationSeconds'] for r in good),
        'meanPatchReconstructionSnrDb':statistics.mean(r['patchReconstructionSnrDb'] for r in originals),
        'maximumEstimatedFormantDriftPercent':max(v for r in good for v in r['formantDriftPercent'] if v is not None),
        'modes':modes,
        'identityOriginals':speaker_stats(identity['originals']),
        'identityBaselines':{mode:speaker_stats([r for r in identity['sustainedBaselines'] if r['mode']==mode])
            for mode in ('whole-source-neural','original-patch-loop','neural-patch-loop')},
        'identityInterpretation':'The large score drop also occurs when looping original PCM, while full-phrase neural decoding remains close to the original. This exposes sustained-vowel evaluation/domain limitations; it does not establish perceptual identity preservation or excuse renderer artifacts.',
        'streaming':{'timingScope':report['timingScope'],
            'medianFirstBlockComputeMs':statistics.median(r['firstBlockComputeMs'] for r in good),
            'maximumFirstBlockComputeMs':max(r['firstBlockComputeMs'] for r in good),
            'maximumPacketMsP95':max(r['packetMsP95'] for r in good),
            'maximumSinglePacketMs':max(r['packetMsMaximum'] for r in good),
            'maximumComputeRealTimeFactor':max(r['computeRealTimeFactor'] for r in good),
            'maximumCachedWaveformSamples':max(r['maximumCachedWaveformSamples'] for r in good),
            'memoryScope':'Encoded short patch and duration-dependent pulse metadata resident; only decoded waveform cache bounded'},
        'tests':{'passed':89,'log':'evidence/audio-native-vowel-tests.txt'},
        'limitations':report['limits']+[
            'Resonance estimates drift by up to 22.43%; estimator uncertainty and actual timbre changes remain unresolved.',
            'Arbitrary new words, learned phonemes, cross-vowel articulation and general short-reference cloning are unimplemented.',
            'No listening acceptance or Kokoro naturalness parity is established.'],
        'completionProven':False}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('Saved',len(rows),'control cases and',len(identity['sustainedBaselines']),'identity baseline cases')


if __name__=='__main__':main()
