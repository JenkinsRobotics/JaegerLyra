"""Combined grid-intervention evidence, including amplitude and identity costs."""
import json
import re
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
GRID=ROOT/'evidence/kokoro-control-voicebox/upsampler-grid-v1'
LEVEL=ROOT/'evidence/kokoro-control-voicebox/grid-level-v1'


def read(path):return json.loads(path.read_text())


def main():
    rendered=read(GRID/'report.json'); gain_rendered=read(LEVEL/'report.json')
    evaluated=read(GRID/'evaluation.json'); gain_evaluated=read(LEVEL/'evaluation.json')
    identity=read(GRID/'identity-evaluation.json');gain_identity=read(LEVEL/'identity-evaluation.json')
    spectra=read(GRID/'spectral-evaluation.json');manifest=read(LEVEL/'audition/manifest.json')
    if not all(r['complete'] for r in (rendered,gain_rendered,evaluated,gain_evaluated,identity,gain_identity,spectra)):
        raise ValueError('Incomplete evidence')
    log=ROOT/'evidence/audio-native-upsampler-grid-tests.txt';text=log.read_text();tests=re.search(r'Ran (\d+) tests',text)
    if tests is None or not text.rstrip().endswith('OK'):raise ValueError('Research checks did not pass')
    rows=evaluated['results']+[r for r in gain_evaluated['results'] if r['method']=='level']
    waves=rendered['results']+[r for r in gain_rendered['results'] if r['method']=='level']
    speakers=identity['results']+[r for r in gain_identity['results'] if r['method']=='level']
    groups=[];pairs=[]
    for method in ('baseline','half','full','level'):
        selected=[r for r in rows if r['method']==method];audio=[r for r in waves if r['method']==method]
        ids=[r for r in speakers if r['method']==method]
        groups.append({'method':method,'cases':len(selected),
            'bothPitchChecksPassed':sum(r['bothPitchChecksPassed'] for r in selected),
            'praatVoicedAtLeast95Percent':sum(r['praat']['voicedFraction']>=.95 for r in selected),
            'worldVoicedAtLeast95Percent':sum(r['world']['voicedFraction']>=.95 for r in selected),
            'homophoneEquivalentMatches':sum(r['homophoneEquivalentMatch'] for r in selected),
            'medianHeldRms':float(np.median([r['heldRms'] for r in selected])),
            'maximumRawPeak':max(r['rawPeak'] for r in audio),'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in audio),
            'speakerIdentity':{'nearestSpeakerMatches':sum(r['matchesSourceSpeaker'] for r in ids),
                'medianSourceSpeakerCosine':float(np.median([r['sourceSpeakerCosine'] for r in ids])),
                'medianSourceSpeakerMargin':float(np.median([r['sourceSpeakerMargin'] for r in ids])),
                'medianSameWordOriginalCosine':float(np.median([r['sameWordOriginalCosine'] for r in ids]))},
            'pitchFailures':[{'name':r['name'],'praat':r['praat'],'world':r['world']} for r in selected if not r['bothPitchChecksPassed']]})
        if method=='baseline':continue
        match=[]
        for after in selected:
            before=next(r for r in rows if r['method']=='baseline' and
                (r['speaker'],r['word'],r['mode'])==(after['speaker'],after['word'],after['mode']))
            match.append((before,after))
        pairs.append({'method':method,'pitchPassGains':sum(not a['bothPitchChecksPassed'] and b['bothPitchChecksPassed'] for a,b in match),
            'pitchPassLosses':sum(a['bothPitchChecksPassed'] and not b['bothPitchChecksPassed'] for a,b in match),
            'wordMatchGains':sum(not a['homophoneEquivalentMatch'] and b['homophoneEquivalentMatch'] for a,b in match),
            'wordMatchLosses':sum(a['homophoneEquivalentMatch'] and not b['homophoneEquivalentMatch'] for a,b in match)})
    gains=[r['levelCompensation']['gainDb'] for r in gain_rendered['results'] if r['method']=='level']
    checks=rendered['baselineChecks'];boundaries=identity['boundaryComparisons']+gain_identity['boundaryComparisons']
    summary={'scope':'First-upsampler grid-response cancellation and a separately labelled offline gain-only diagnostic',
        'sources':18,'newControlledRenders':108,'comparativeControlledCases':144,'baselineReplays':36,
        'newLocalAsrEvaluations':108,'reusedOriginalSources':18,'reusedBaselineEvaluations':36,
        'sourceSpeakerDiagnostic':identity['groups'][0],
        'unchangedBaselineMaximumDifference':max(r['maximumDifference'] for r in checks),
        'allHooksRestored':all(r['hooksRestored'] for r in checks),
        'firstUpsamplerConstantInputMaximumVariation':max(r['correction']['inputTemporalPeakToPeakMaximum'] for r in rendered['results']),
        'fullCorrectionMaximumRemainingFeatureTemporalRms':max(r['correction']['afterTemporalRms'] for r in rendered['results'] if r['method']=='full'),
        'groups':groups,'pairedChanges':pairs,
        'spectralDiagnostic':{'report':'evidence/kokoro-control-voicebox/upsampler-grid-v1/spectral-evaluation.json',
            'groups':spectra['groups'],'sidebandsReducedCases':spectra['sidebandsReducedCases'],
            'medianPairedSidebandChangeDb':spectra['medianPairedSidebandChangeDb']},
        'gainDiagnostic':{'rangeDb':[min(gains),max(gains)],'medianDb':float(np.median(gains)),
            'wholeRenderedClipsRequired':True,'notStreaming':True,
            'scope':'Constant gain in the steady core with guarded cosine ramps, calibrated to the uncorrected core RMS; no pitch correction or filtering.'},
        'outsideHoldBoundaryCheck':{'guardSeconds':.2,'maximumSampleDifference':max(r['maximumDifference'] for r in boundaries),
            'scope':'Baseline compared to each candidate outside the hold plus a 200 ms guard; this does not guarantee unchanged audible joins.'},
        'auditions':{'manifest':'evidence/kokoro-control-voicebox/grid-level-v1/audition/manifest.json',
            'clips':len(manifest['files']),'plots':len(manifest['groups']),'labelledReels':len(manifest['reels']),
            'maximumLevelErrorLu':max(abs(r['targetLufs200MsGate']-r['measuredLufs200MsGate']) for r in manifest['files']),
            'maximumSamplePeak':max(r['samplePeak'] for r in manifest['files']),
            'audioPlayed':False,'listeningAccepted':False,'processing':manifest['processing']},
        'tests':{'research':int(tests.group(1)),'log':str(log.relative_to(ROOT))},
        'primaryReference':'https://arxiv.org/abs/2010.14356',
        'decision':'The first upsampler contributes materially to the held-output 80 Hz sidebands: cancelling its constant-input phase response reduces them in 17/18 steady notes. Full cancellation also suppresses some vowel levels and shifts speaker embeddings. Offline held-gain compensation improves pitch/word diagnostics but does not restore baseline embedding similarity. Retain all conditions as controlled pretrained references; no quality default is promoted.',
        'nextExperiment':'Transfer the uncorrected and corrected/gain-compensated sustained references into the independent waveform instrument. Compare direct teacher output, independent reconstruction and immediate-control rendering with the same authored pitch and source intervals; retain pitch, identity and waveform-fidelity failures separately. Keep pretrained capture out of the runtime and input understanding deferred.',
        'limitations':['The panel is post-validation development on the same three synthetic voices and six words.',
            'All teacher outputs still use Kokoro pretrained weights; this is not independent neural speech-model training.',
            'Removing a periodic intermediate response can also remove useful voice detail. Lower sideband energy is not naturalness acceptance.',
            'Gain compensation uses complete rendered audio and known hold boundaries, not a causal level controller.',
            'ASR of isolated sustained words and ECAPA on singing/vowels are diagnostic; original teacher sources are imperfect baselines.',
            'The transient feature hooks include CPU observation/correction and are not an optimized streaming implementation.',
            'No perceptual identity, human-reference cloning or Kokoro-level independent output has been accepted.'],
        'appDefaultChanged':False,'listeningAccepted':False,'completionProven':False}
    (LEVEL/'combined-summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print('Saved combined grid summary:',[(g['method'],g['bothPitchChecksPassed'],g['homophoneEquivalentMatches']) for g in groups])


if __name__=='__main__':main()
