"""Report the frozen context ablation without promoting an ASR-only winner."""
import hashlib
import json
import re
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-syllables/context-v1'


def read(name):return json.loads((OUT/name).read_text())
def median(values):return float(np.median(values))


def main():
    capture=read('capture.json');render=read('report.json');evaluation=read('evaluation.json');identity=read('identity-evaluation.json')
    if not all(d['complete'] for d in (capture,render,evaluation,identity)):raise ValueError('Incomplete study')
    identities={r['name']:r for r in identity['results']}
    methods=['teacher','reconstruction','live-neural','live-pcm','prefix-neural','prefix-pcm','release-neural','release-pcm','both-neural','both-pcm']
    groups=[];paired=[]
    for method in methods:
        rows=[r for r in evaluation['results'] if r['method']==method]
        if len(rows)!=36:raise ValueError('Incomplete condition')
        group={'method':method,'cases':len(rows),'bothPitchChecksPassed':sum(r['bothPitchChecksPassed'] for r in rows),
            'homophoneEquivalentMatches':sum(r['homophoneEquivalentMatch'] for r in rows),
            'nearestSpeakerMatches':sum(identities[r['name']]['matchesSourceSpeaker'] for r in rows),
            'medianSourceSpeakerCosine':median([identities[r['name']]['sourceSpeakerCosine'] for r in rows]),
            'byMode':[]}
        for mode in ('native','melody'):
            subset=[r for r in rows if r['mode']==mode]
            group['byMode'].append({'mode':mode,'cases':len(subset),
                'bothPitchChecksPassed':sum(r['bothPitchChecksPassed'] for r in subset),
                'homophoneEquivalentMatches':sum(r['homophoneEquivalentMatch'] for r in subset)})
        groups.append(group)
        if method.startswith(('prefix-','release-','both-')):
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
        pcm=next(r for r in render['results'] if r['decoderKind']=='pcm' and r['contextPolicy']==row['contextPolicy']
            and all(r[k]==row[k] for k in ('speaker','word','mode')))
        a,ra=sf.read(ROOT/pcm['file']);b,rb=sf.read(ROOT/row['file']);assert ra==rb==24000
        pairs.append({'name':row['name'],'contextPolicy':row['contextPolicy'],
            'waveformSnrDb':float(10*np.log10(max(float(a@a),1e-30)/max(float((a-b)@(a-b)),1e-30)))})
    fresh=[r for r in render['results'] if not r['reused']];neural=[r for r in fresh if r['decoderKind']=='neural']
    transition=[]
    for policy in ('prefix','release','both'):
        rows=[r for r in neural if r['contextPolicy']==policy]
        for join in (['hold'] if policy=='prefix' else ['release'] if policy=='release' else ['hold','release']):
            changes=[(r['joinEnvelope'][join]['minimumRelativeRmsDb']-r['baselineJoinEnvelope'][join]['minimumRelativeRmsDb'],r['name']) for r in rows]
            worst=min(changes)
            transition.append({'contextPolicy':policy,'join':join,'cases':len(rows),
                'medianChangeInMinimumRelativeRmsDb':median([v for v,n in changes]),
                'worstChangeInMinimumRelativeRmsDb':worst[0],'worstCase':worst[1],
                'scope':'Exploratory 20 ms envelope diagnostic; lower values indicate a deeper local dip, not a listening verdict.'})
    log=ROOT/'evidence/audio-native-context-tests.txt';text=log.read_text();match=re.search(r'Ran (\d+) tests? in ([\d.]+)s',text)
    if not match or not text.rstrip().endswith('OK'):raise ValueError('Missing passing test result')
    manifest=read('audition/manifest.json');timings=[r['streaming'] for r in neural]
    summary={'scope':read('protocol.json')['scope'],'sourcePanel':read('protocol.json')['sourcePanel'],
        'sourceIdentities':18,'newRenders':len(fresh),'reusedControlledRenders':sum(r['reused'] for r in render['results']),
        'comparativeCases':len(evaluation['results']),
        'newLocalAsrEvaluations':sum(not r.get('parentEvaluationReused',False) for r in evaluation['results']),
        'reusedEvaluations':sum(r.get('parentEvaluationReused',False) for r in evaluation['results']),
        'groups':groups,'pairedChanges':paired,'decoderComparisons':pairs,
        'decoderMedianSnrDbByPolicy':{p:median([r['waveformSnrDb'] for r in pairs if r['contextPolicy']==p]) for p in ('retune-all','prefix','release','both')},
        'preservedContextMaximumError':max(r['preservedContextMaximumError'] for r in fresh),
        'untouchedRegionMaximumError':max(r['untouchedRegionMaximumError'] for r in fresh),
        'transitionDiagnostics':transition,
        'maximumNewRawPeak':max(r['rawPeak'] for r in fresh),'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in fresh),
        'streaming':{'neuralCases':len(neural),'medianFirstPhysicalPacketComputeMs':median([r['firstPhysicalPacketComputeMs'] for r in timings]),
            'medianPacketP95Ms':median([r['packetP95Ms'] for r in timings]),'maximumPacketMs':max(r['maximumPacketMs'] for r in timings),
            'packetsOver5Ms':sum(r['packetsOver5Ms'] for r in timings),'waveformDelayMs':20,
            'maximumCachedWaveformSamples':max(r['maximumCachedWaveformSamples'] for r in timings),
            'maximumResidentSourceControlScalars':max(r['residentSourceControlScalars'] for r in timings),
            'allClosedClearHistory':all(r['closedClearsHistory'] for r in timings),
            'excludes':'Preparation, loading, transport and devices; fixed waveform delay is additional.'},
        'tests':{'passed':True,'count':int(match[1]),'log':str(log.relative_to(ROOT))},
        'audition':{'manifest':str((OUT/'audition/manifest.json').relative_to(ROOT)),'clips':len(manifest['files']),
            'plots':len(manifest['plots']),'reels':len(manifest['reels']),'autoplay':False},
        'listeningAccepted':False,'kokoroParityAccepted':False,'newNeuralTraining':False,'appDefaultChanged':False,
        'inputUnderstandingDeferred':True,'limits':read('protocol.json')['limits']+identity['limits'],
        'nextExperiment':'Keep context policies optional, not promoted. Isolate held-grain window/phase effects with the same lossless PCM comparator, then validate any selected change on fresh sources. Expose actual live control without changing the accepted Kokoro speech default.'}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:summary[k] for k in ('groups','pairedChanges','decoderMedianSnrDbByPolicy','preservedContextMaximumError','untouchedRegionMaximumError','streaming','tests')},indent=2))


if __name__=='__main__':main()
