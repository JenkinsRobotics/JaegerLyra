import hashlib
import json
import re
from pathlib import Path
from statistics import median
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/resonance-morph/v1'


def main():
    def read(name):return json.loads((OUT/name).read_text())
    render=read('render.json');evaluation=read('evaluation.json');identity=read('identity.json')
    oracle=read('oracle-envelope.json');capture=read('capture.json');audition=read('audition.json')
    assert all(r['complete'] for r in (render,evaluation,identity,oracle,capture,audition))
    methods=['teacher','reconstruction','grain-linear','grain-learned','filter-linear','filter-learned','oracle-envelope']
    groups=[];paired=[]
    for method in methods:
        rows=[r for r in evaluation['results'] if r['method']==method]
        ids=[r for r in identity['results'] if r['method']==method]
        raw=[r for r in render['results']+oracle['results'] if r['method']==method]
        assert len(rows)==len(ids)==18
        group={'method':method,'cases':len(rows),
            'medianTransitionSpectrumMaeDb':median(r['transitionSpectrumMaeDb'] for r in rows),
            'medianWholeNucleusSpectrumMaeDb':median(r['centeredSpectrumMaeDb'] for r in rows),
            'bothPitchChecksPassed':sum(all(p['passed'] for p in r['pitch'].values()) for r in rows),
            'nearestSpeakerMatches':sum(r['matchesSourceSpeaker'] for r in ids),
            'medianSourceSpeakerCosine':median(r['sourceSpeakerCosine'] for r in ids),
            'medianSameNucleusCosine':median(r['sameNucleusCosine'] for r in ids),
            'pitchFailures':[{'name':r['name'],**r['pitch']} for r in rows if not all(p['passed'] for p in r['pitch'].values())]}
        if raw:group.update(maximumRawPeak=max(r['rawPeak'] for r in raw),
            overFullScaleSamples=sum(r['overFullScaleSamples'] for r in raw),
            overFullScaleCases=[r['name'] for r in raw if r['overFullScaleSamples']])
        groups.append(group)
        if method not in ('teacher','reconstruction','grain-linear'):
            changes=[];identity_changes=[]
            for row in rows:
                baseline=next(r for r in evaluation['results'] if r['name']==row['name'] and r['method']=='grain-linear')
                changes.append(row['transitionSpectrumMaeDb']-baseline['transitionSpectrumMaeDb'])
                x=next(r for r in ids if r['name']==row['name'])
                y=next(r for r in identity['results'] if r['name']==row['name'] and r['method']=='grain-linear')
                identity_changes.append(x['sameNucleusCosine']-y['sameNucleusCosine'])
            paired.append({'method':method,'reference':'grain-linear','medianPairedSpectrumErrorChangeDb':median(changes),
                'spectrumImprovedCases':sum(c<0 for c in changes),'spectrumWorsenedCases':sum(c>0 for c in changes),
                'medianPairedSameNucleusCosineChange':median(identity_changes)})
    snrs=[]
    for row in render['references']:
        source,_=sf.read(ROOT/row['nucleus']);decoded,_=sf.read(ROOT/row['reconstruction'])
        snrs.append(float(10*np.log10((source@source)/max((source-decoded)@(source-decoded),1e-30))))
    testlog=ROOT/'evidence/audio-native-resonance-morph-tests.txt';text=testlog.read_text()
    match=re.search(r'Ran (\d+) tests in ([\d.]+)s',text);assert match and text.rstrip().endswith('OK')
    for row in audition['files']+audition['plots']:
        assert hashlib.sha256((ROOT/row['file']).read_bytes()).hexdigest()==row['sha256']
    summary={'scope':'Continuous spectral-resonance movement and a compact fitted timing curve, evaluated on eighteen new diphthong words',
        'trainingSources':9,'freshValidationSources':18,'capturedUnits':len(capture['units']),
        'captureRepair':read('capture-repair.json'),'controlledRenders':len(render['results']),
        'oracleDiagnosticRenders':len(oracle['results']),'comparisonCases':len(evaluation['results']),
        'trajectoryParameters':sum(len(v) for v in capture['trajectories'].values()),
        'newNeuralTraining':False,'newFittedTimingControls':True,
        'medianFreshReconstructionSnrDb':median(snrs),'minimumFreshReconstructionSnrDb':min(snrs),
        'groups':groups,'pairedChanges':paired,
        'runtime':{'controlPacketMs':5,'fixedSourceDelayMs':20,'filterLookaheadSamples':0,'filterHistorySamples':511,
            'maximumBlockMs':max(r['maximumBlockMs'] for r in render['results']),
            'blocksOver5Ms':sum(r['blocksOver5Ms'] for r in render['results']),
            'excludes':'Offline preparation, model loading, transport, browser buffering and devices; oracle probe is offline.'},
        'tests':{'count':int(match[1]),'passed':True,'log':str(testlog.relative_to(ROOT))},
        'audition':{'url':'http://127.0.0.1:8791/voicebox-resonance-lab.html','clips':len(audition['files']),
            'plots':len(audition['plots']),'manifest':'evidence/resonance-morph/v1/audition.json',
            'maximumPeak':max(r['peak'] for r in audition['files']),
            'maximumRmsError':max(r['rmsError'] for r in audition['files']),'autoplay':False},
        'browserCheck':{'sourceSelectionAndSevenPlayersVerified':True,'audioPlayed':False,
            'screenshot':'evidence/resonance-morph/v1/browser-check.jpg'},
        'listeningAccepted':False,'kokoroParityAccepted':False,'qualityWinnerPromoted':False,
        'appDefaultChanged':False,'inputUnderstandingDeferred':True,'completionProven':False,
        'decision':'Retain this as an experimental articulation controller. Moving resonances modestly improves spectral error over blending, but does not establish naturalness or identity. Fitted timing is not promoted. A full reference-envelope diagnostic improves spectrum and identity more, but remains far from direct reconstruction and loses pitch coverage.',
        'limits':['Every candidate receives target pitch, duration and RMS; this does not generate new pronunciation or prosody independently.',
            'Endpoint data come from synthetic Kokoro development words. No general short-reference or human voice cloning evidence.',
            'The nine-parameter curves are fitted spectral timing controls, not new neural phoneme learning or predictive coding.',
            'WORLD envelope and Praat formant extraction are external offline analysis; runtime combines the existing neural decoder with project DSP.',
            'The complete-envelope condition is a post-validation oracle diagnostic and must not be counted as independent generalization.',
            'Six of eighteen natural teacher nuclei fail the same strict dual pitch gate; pass counts are diagnostics, not listening judgments.',
            'Speech-trained speaker embeddings are uncertain on short vowel nuclei. Original and direct-reconstruction baselines remain visible.',
            'Raw over-full-scale samples are preserved. Equal-RMS audition gains provide playback headroom without repairing the underlying amplitude control.',
            'The 18 dB envelope-ratio clamp precedes energy normalization and is not a final waveform peak or frequency-response bound.',
            'No audible listening acceptance; the user reported they have not listened yet.'],
        'nextExperiment':'Using the same complete-envelope diagnostic, isolate phase changes introduced by the time-varying minimum-phase filter with a fixed-delay linear-phase alternative. Keep excitation, envelope and amplitude controls unchanged; separately inspect common carrier crest factor before any promotion.'}
    files=[ROOT/'voice-research/audio_native'/name for name in ['resonance_morph.py','test_resonance_morph.py','prepare_morph_references.py',
        'benchmark_resonance_morph.py','probe_morph_envelope_ceiling.py','evaluate_resonance_identity.py','summarize_resonance_morph.py']]
    summary['implementationHashes']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('controlledRenders','oracleDiagnosticRenders','medianFreshReconstructionSnrDb','pairedChanges','runtime','tests','audition')},indent=2))


if __name__=='__main__':main()
