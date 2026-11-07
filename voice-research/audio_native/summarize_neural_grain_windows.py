"""Summarize window alternatives with paired changes and separate diagnostics."""
import json
import re
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-syllables/grain-windows-v1'


def read(name):return json.loads((OUT/name).read_text())
def median(values):return float(np.median(values))


def main():
    render=read('report.json');evaluation=read('evaluation.json');identity=read('identity-evaluation.json');spectral=read('spectral-evaluation.json')
    if not all(r['complete'] for r in (render,evaluation,identity,spectral)):raise ValueError('Incomplete study')
    identities={r['name']:r for r in identity['results']};groups=[];paired=[]
    for method in ('teacher','reconstruction','live-neural','live-pcm','source-neural','source-pcm','wide-neural','wide-pcm'):
        rows=[r for r in evaluation['results'] if r['method']==method]
        if len(rows)!=36:raise ValueError('Incomplete condition')
        shape=[r for r in spectral['results'] if r['method']==method]
        group={'method':method,'cases':len(rows),'bothPitchChecksPassed':sum(r['bothPitchChecksPassed'] for r in rows),
            'homophoneEquivalentMatches':sum(r['homophoneEquivalentMatch'] for r in rows),
            'nearestSpeakerMatches':sum(identities[r['name']]['matchesSourceSpeaker'] for r in rows),
            'medianSourceSpeakerCosine':median([identities[r['name']]['sourceSpeakerCosine'] for r in rows]),
            'pitchFailures':[{'name':r['name'],**{tracker:{k:r[tracker][k] for k in ('voicedFraction','p95ErrorCents','pitchCheckPassed')}
                for tracker in ('praat','world')}} for r in rows if not r['bothPitchChecksPassed']],
            'byMode':[{'mode':m,'cases':sum(r['mode']==m for r in rows),
                'bothPitchChecksPassed':sum(r['bothPitchChecksPassed'] for r in rows if r['mode']==m),
                'homophoneEquivalentMatches':sum(r['homophoneEquivalentMatch'] for r in rows if r['mode']==m)} for m in ('native','melody')]}
        if shape:group['nativeSpectrum']={'cases':len(shape),'medianMeanAbsoluteShapeDifferenceDb':median([r['meanAbsoluteShapeDifferenceDb'] for r in shape]),
            'medianHeldRmsRatioToTeacherDb':median([r['heldRmsRatioDb'] for r in shape])}
        groups.append(group)
        if method.startswith(('source-','wide-')):
            changes={'method':method,'wordMatchesGained':[],'wordMatchesLost':[],'pitchPassesGained':[],'pitchPassesLost':[]};cosines=[]
            for row in rows:
                baseline=next(r for r in evaluation['results'] if r['method']=='live-'+method.split('-')[-1] and
                    all(r[k]==row[k] for k in ('speaker','word','mode')))
                for field,gain,loss in [('homophoneEquivalentMatch','wordMatchesGained','wordMatchesLost'),
                        ('bothPitchChecksPassed','pitchPassesGained','pitchPassesLost')]:
                    if row[field] and not baseline[field]:changes[gain].append(row['name'])
                    if baseline[field] and not row[field]:changes[loss].append(row['name'])
                cosines.append(identities[row['name']]['sourceSpeakerCosine']-identities[baseline['name']]['sourceSpeakerCosine'])
            changes['medianPairedSourceSpeakerCosineDelta']=median(cosines);paired.append(changes)
    pairs=[]
    for row in render['results']:
        if row['decoderKind']!='neural':continue
        other=next(r for r in render['results'] if r['decoderKind']=='pcm' and r['windowPolicy']==row['windowPolicy']
            and all(r[k]==row[k] for k in ('speaker','word','mode')))
        a,ra=sf.read(ROOT/other['file']);b,rb=sf.read(ROOT/row['file']);assert ra==rb==24000
        pairs.append({'name':row['name'],'windowPolicy':row['windowPolicy'],
            'waveformSnrDb':float(10*np.log10(max(float(a@a),1e-30)/max(float((a-b)@(a-b)),1e-30)))})
    fresh=[r for r in render['results'] if not r['reused']];neural=[r for r in fresh if r['decoderKind']=='neural'];timings=[r['streaming'] for r in neural]
    log=ROOT/'evidence/audio-native-grain-window-tests.txt';text=log.read_text();match=re.search(r'Ran (\d+) tests? in ([\d.]+)s',text)
    if not match or not text.rstrip().endswith('OK'):raise ValueError('Missing passing test evidence')
    manifest=read('audition/manifest.json')
    summary={'scope':read('protocol.json')['scope'],'sourcePanel':read('protocol.json')['sourcePanel'],
        'sourceIdentities':18,'newRenders':len(fresh),'reusedControlledRenders':sum(r['reused'] for r in render['results']),
        'comparativeCases':len(evaluation['results']),
        'newLocalAsrEvaluations':sum(not r.get('parentEvaluationReused',False) for r in evaluation['results']),
        'reusedEvaluations':sum(r.get('parentEvaluationReused',False) for r in evaluation['results']),
        'groups':groups,'pairedChanges':paired,'decoderComparisons':pairs,
        'decoderMedianSnrDbByPolicy':{p:median([r['waveformSnrDb'] for r in pairs if r['windowPolicy']==p]) for p in ('minimum','source','wide')},
        'outsideHoldGuardMaximumError':max(r['outsideHoldGuardMaximumError'] for r in fresh),
        'outsideHoldGuardMs':20,'maximumGrainRadiusSamples':max(r['maximumGrainRadiusSamples'] for r in fresh),
        'radiusClippedGrains':sum(r['radiusClippedGrains'] for r in fresh),
        'maximumNewRawPeak':max(r['rawPeak'] for r in fresh),'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in fresh),
        'streaming':{'neuralCases':len(neural),'medianFirstPhysicalPacketComputeMs':median([r['firstPhysicalPacketComputeMs'] for r in timings]),
            'medianPacketP95Ms':median([r['packetP95Ms'] for r in timings]),'maximumPacketMs':max(r['maximumPacketMs'] for r in timings),
            'packetsOver5Ms':sum(r['packetsOver5Ms'] for r in timings),'waveformDelayMs':20,
            'maximumCachedWaveformSamples':max(r['maximumCachedWaveformSamples'] for r in timings),
            'maximumResidentSourceControlScalars':max(r['residentSourceControlScalars'] for r in timings),
            'allClosedClearHistory':all(r['closedClearsHistory'] for r in timings),
            'excludes':'Preparation, loading, transport and devices; fixed waveform delay is additional.'},
        'tests':{'count':int(match[1]),'passed':True,'log':str(log.relative_to(ROOT))},
        'audition':{'manifest':str((OUT/'audition/manifest.json').relative_to(ROOT)),'clips':len(manifest['files']),
            'plots':len(manifest['plots']),'reels':len(manifest['reels']),'autoplay':False},
        'listeningAccepted':False,'kokoroParityAccepted':False,'newNeuralTraining':False,'appDefaultChanged':False,
        'inputUnderstandingDeferred':True,'qualityWinnerPromoted':False,
        'limits':read('protocol.json')['limits']+identity['limits'],
        'nextExperiment':'Retain the minimum-period renderer as the established reference. Build an actual live waveform instrument and a natural-speech reconstruction comparison for direct listening; keep alternatives experimental and require fresh-source validation before promotion.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('groups','pairedChanges','decoderMedianSnrDbByPolicy','outsideHoldGuardMaximumError','radiusClippedGrains','streaming','tests')},indent=2))


if __name__=='__main__':main()
