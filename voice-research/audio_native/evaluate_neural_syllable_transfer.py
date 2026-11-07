"""Identity and paired-decoder diagnostics for captured syllable transfer.

This evaluator is never loaded by the independent synthesis runtime.
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE','1')
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
import re
import sys
from pathlib import Path
import numpy as np
import soundfile as sf

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'evidence/neural-syllables/teacher-transfer-v1'
VOWELS={'calm':'ah','palm':'ah','paw':'aw','raw':'aw','blue':'oo','true':'oo'}


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name):return json.loads((OUT/name).read_text())
def write(name,data):(OUT/name).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def median(rows,key):return float(np.median([r[key] for r in rows]))


def identity():
    import torch
    import torchaudio
    from speechbrain.inference.classifiers import EncoderClassifier
    capture=read('capture.json');render=read('report.json')
    if not capture['complete'] or not render['complete']:raise ValueError('Incomplete rendering')
    cases=capture['references']+capture['reconstructions']+render['results']
    directory=ROOT/'models/speaker-eval';model_hashes={p.name:sha(p) for p in directory.glob('*.ckpt')}
    torch.set_num_threads(4)
    model=EncoderClassifier.from_hparams(source=str(directory),savedir=str(directory),
        overrides={'pretrained_path':str(directory)},run_opts={'device':'cpu'})
    cache=OUT/'speaker-embeddings';cache.mkdir(exist_ok=True)
    known={}
    for folder in ['evidence/kokoro-control-voicebox/upsampler-grid-v1',
                   'evidence/kokoro-control-voicebox/grid-level-v1']:
        report=json.loads((ROOT/folder/'identity-evaluation.json').read_text())
        if report['modelFiles']!=model_hashes:raise ValueError('Earlier evaluator weights differ')
        for row in report['sources']+report['results']:
            known[row['sha256']]=ROOT/folder/'speaker-embeddings'/(row['sha256']+'.npy')
    metadata=cache/'model-files.json'
    if metadata.exists() and json.loads(metadata.read_text())!=model_hashes:raise ValueError('Changed cached evaluator')
    metadata.write_text(json.dumps(model_hashes,indent=2)+'\n')
    def embedding(row):
        file=ROOT/row['file']
        if sha(file)!=row['sha256']:raise ValueError('Changed audio')
        stored=cache/(row['sha256']+'.npy')
        if stored.exists():value=np.load(stored,allow_pickle=False)
        elif row['sha256'] in known and known[row['sha256']].exists():
            value=np.load(known[row['sha256']],allow_pickle=False);np.save(stored,value)
        else:
            audio,rate=sf.read(file)
            samples=torchaudio.functional.resample(torch.from_numpy(audio.astype(np.float32)).unsqueeze(0),rate,16000)
            with torch.inference_mode():vector=model.encode_batch(samples).flatten()
            value=(vector/torch.linalg.vector_norm(vector)).numpy();np.save(stored,value)
        if value.ndim!=1 or not np.isfinite(value).all() or abs(np.linalg.norm(value)-1)>.0001:
            raise ValueError('Invalid embedding')
        return value
    sources=capture['sources'];originals={s['name']:embedding(s) for s in sources}
    speakers=sorted({s['speaker'] for s in sources})
    def compare(row):
        vector=embedding(row);scores={};excluded=VOWELS[row['word']]
        for speaker in speakers:
            anchor=np.mean([originals[s['name']] for s in sources
                if s['speaker']==speaker and VOWELS[s['word']]!=excluded],axis=0)
            anchor/=np.linalg.norm(anchor);scores[speaker]=float(np.dot(vector,anchor))
        nearest=max(scores,key=scores.get)
        return {'referenceVowelClassExcluded':excluded,'cosines':scores,'nearestSpeaker':nearest,
            'matchesSourceSpeaker':nearest==row['speaker'],'sourceSpeakerCosine':scores[row['speaker']],
            'sameWordOriginalCosine':float(np.dot(vector,originals[row['speaker']+'-'+row['word']]))}
    report={'scope':'Leave-vowel-class-out ECAPA comparisons for teacher, reconstructed and live syllables',
        'modelFiles':model_hashes,'sources':[],'results':[],'complete':False,
        'limits':['Speech-trained embeddings are not validated as singing-naturalness or perceived-identity scores.',
            'Only three existing synthetic voices; this does not establish general human voice cloning.',
            'Anchors exclude both ordinary source words of the same intended vowel class.']}
    for row in sources:
        report['sources'].append({k:row[k] for k in ('name','speaker','word','file','sha256')}|compare(row))
    for row in cases:
        report['results'].append({k:row[k] for k in ('name','speaker','word','variant','mode','method','file','sha256')}|compare(row))
        write('identity-evaluation.json',report)
        print(row['name'],round(report['results'][-1]['sourceSpeakerCosine'],4),flush=True)
    report['complete']=len(report['results'])==len(cases);write('identity-evaluation.json',report)


def summarize():
    capture=read('capture.json');render=read('report.json');evaluation=read('evaluation.json');identity=read('identity-evaluation.json')
    if not all(d['complete'] for d in (capture,render,evaluation,identity)):raise ValueError('Incomplete evaluation')
    identities={r['name']:r for r in identity['results']}
    unit_sets=[{(r['speaker'],r['word']) for r in capture['units'] if r['variant']==v} for v in ('baseline','level')]
    common=set.intersection(*unit_sets)
    def group(rows):
        return {'cases':len(rows),'bothPitchChecksPassed':sum(r['bothPitchChecksPassed'] for r in rows),
            'homophoneEquivalentMatches':sum(r['homophoneEquivalentMatch'] for r in rows),
            'nearestSpeakerMatches':sum(identities[r['name']]['matchesSourceSpeaker'] for r in rows),
            'medianSourceSpeakerCosine':float(np.median([identities[r['name']]['sourceSpeakerCosine'] for r in rows]))}
    groups=[]
    for variant in ('baseline','level'):
        for method in ('teacher','reconstruction','live-neural','live-pcm'):
            rows=[r for r in evaluation['results'] if (r['variant'],r['method'])==(variant,method)]
            groups.append({'variant':variant,'method':method,'plannedCases':36,**group(rows),
                'missingCases':36-len(rows),'commonEligibleSources':group([r for r in rows if (r['speaker'],r['word']) in common])})
    pairs=[]
    for row in render['results']:
        if row['method']!='live-neural':continue
        counterpart=next(r for r in render['results'] if r['method']=='live-pcm' and
            all(r[k]==row[k] for k in ('speaker','word','variant','mode')))
        a,ra=sf.read(ROOT/counterpart['file']);b,rb=sf.read(ROOT/row['file']);assert ra==rb==24000
        pairs.append({'name':row['name'],'variant':row['variant'],'mode':row['mode'],
            'waveformSnrDb':float(10*np.log10(max(float(a@a),1e-30)/max(float((a-b)@(a-b)),1e-30))),
            'maximumDifference':float(max(abs(a-b)))})
    neural=[r for r in render['results'] if r['method']=='live-neural']
    timings=[r['streaming'] for r in neural]
    log=ROOT/'evidence/audio-native-syllable-transfer-tests.txt';testtext=log.read_text()
    testmatch=re.search(r'Ran (\d+) tests? in ([\d.]+)s',testtext)
    if not testmatch or not testtext.rstrip().endswith('OK'):raise ValueError('Missing passing test evidence')
    manifest=read('audition/manifest.json')
    summary={'scope':'Captured teacher waveforms transferred to a frozen independent decoder and onset/hold/release controller',
        'protocolSha256':sha(OUT/'protocol.json'),'sourceIdentities':len(capture['sources']),
        'teacherClips':len(capture['references']),'newReconstructedClips':len(capture['reconstructions']),
        'capturedUnits':len(capture['units']),'plannedUnits':36,'captureFailures':capture['failures'],
        'newLiveRenders':len(render['results']),'plannedLiveRenders':144,'comparativeCases':len(evaluation['results']),
        'newLocalAsrEvaluations':sum(not r['teacherEvaluationReused'] for r in evaluation['results']),
        'groups':groups,'commonEligibleSourceIdentities':len(common),
        'reconstruction':[], 'pairedLiveDecoderComparison':pairs,
        'pairedLiveDecoderMedianSnrDb':median(pairs,'waveformSnrDb'),
        'pairedLiveDecoderMinimumSnrDb':min(r['waveformSnrDb'] for r in pairs),
        'streaming':{'cases':len(neural),'medianFirstPhysicalPacketComputeMs':median(timings,'firstPhysicalPacketComputeMs'),
            'maximumPacketMs':max(r['maximumPacketMs'] for r in timings),
            'medianPacketP95Ms':median(timings,'packetP95Ms'),'packetsOver5Ms':sum(r['packetsOver5Ms'] for r in timings),
            'waveformDelayMs':20,'maximumCachedWaveformSamples':max(r['maximumCachedWaveformSamples'] for r in timings),
            'maximumResidentSourceControlScalars':max(r['residentSourceControlScalars'] for r in timings),
            'closedClearsHistory':all(r['closedClearsHistory'] for r in timings),
            'excludes':'Capture, loading, transport and audio devices; fixed waveform delay is additional.'},
        'maximumNewRawPeak':max(r['rawPeak'] for r in capture['reconstructions']+render['results']),
        'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in capture['reconstructions']+render['results']),
        'tests':{'count':int(testmatch[1]),'passed':True,'log':str(log.relative_to(ROOT))},
        'audition':{'manifest':'evidence/neural-syllables/teacher-transfer-v1/audition/manifest.json',
            'clips':len(manifest['files']),'reels':len(manifest['reels']),'autoplay':False},
        'listeningAccepted':False,'kokoroParityAccepted':False,'newNeuralTraining':False,
        'appDefaultChanged':False,'inputUnderstandingDeferred':True,
        'limits':read('protocol.json')['limits']+identity['limits'],
        'nextExperiment':'Localize pitch and pronunciation loss in source capture, phase, held trajectory and release; retain exact PCM comparison. Test fresh sources before promoting control changes.'}
    for variant in ('baseline','level'):
        rows=[r for r in capture['reconstructions'] if r['variant']==variant]
        summary['reconstruction'].append({'variant':variant,'cases':len(rows),
            'medianWaveformSnrDb':median(rows,'waveformSnrDb'),'minimumWaveformSnrDb':min(r['waveformSnrDb'] for r in rows),
            'medianLinearReferenceSnrDb':median(rows,'linearReferenceSnrDb')})
    write('summary.json',summary)
    print(json.dumps({k:summary[k] for k in ('capturedUnits','newLiveRenders','groups','reconstruction','pairedLiveDecoderMedianSnrDb','streaming','tests')},indent=2))


if __name__=='__main__':{'identity':identity,'summarize':summarize}[sys.argv[1]]()
