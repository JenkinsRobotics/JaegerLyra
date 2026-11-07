"""Frozen sustain validation on fresh synthetic words, with a PCM control.

Run stages separately: capture, render, words, summarize. The protocol and
algorithm hashes are saved before capture. Capture/selection failures count;
neither output measurements nor transcripts alter selection or rendering.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import base64
import hashlib
import json
import re
import statistics as stats
import sys
import urllib.request
from functools import partial
from pathlib import Path
import numpy as np
import soundfile as sf
from neural_waveform import load_model, RATE
from compiled_neural_decoder import compile_decoder
from neural_performance import save_performance, load_performance
from neural_articulation import ArticulationUnit, NeuralArticulationStream, compile_unit
from vowel_stationarity import select_stationary_unit
from neural_vowel_instrument import VowelScore
from benchmark_neural_vowels import formants
from benchmark_neural_articulation import controlled_render, metrics
from pcm_reference_decoder import pcm_performance, PcmFrameDecoder
from summarize_neural_articulation import pitch_group

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/neural-articulation/fresh-sustain-v1'
SPEAKERS = ('af_heart', 'am_michael', 'af_bella')
WORDS = ('spa', 'bra', 'saw', 'law', 'sue', 'zoo')
EQUIVALENTS = {w: {w} for w in WORDS}
EQUIVALENTS['sue'].add('sioux')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def read(name):
    return json.loads((OUT / name).read_text())


def protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    model = 'evidence/neural-waveform/axis-detail-v2/projected-r1e-7.npz'
    files = ('neural_articulation.py', 'vowel_stationarity.py', 'benchmark_neural_articulation.py',
             'pcm_reference_decoder.py')
    config = {
        'scope': 'Fresh-word validation of frozen source selection and sustain; existing three synthetic voices, no new-speaker or human-cloning claim',
        'words': list(WORDS), 'speakers': list(SPEAKERS), 'expectedSources': 18,
        'modes': ['native', 'melody', 'short-native'],
        'methods': ['neural-loop', 'neural-fixed', 'pcm-loop', 'pcm-fixed'],
        'maximumRenders': 216, 'model': model, 'modelSha256': sha(ROOT / model),
        'frozenCode': {name: sha(Path(__file__).parent / name) for name in files},
        'baseline': 'Original pitch/energy-selected span with moving source-pulse loop',
        'candidate': 'Source spectral-stationarity-selected span with fixed first cycle; unchanged development rule',
        'pcmControl': 'Identical pulses, selections, controls and stream; lossless waveform-frame reader replaces neural approximation. Explicitly retains original waveform data.',
        'shortDuration': 'max(trimmed source duration, original hold start + 150 ms)',
        'longDurationSeconds': 2.5,
        'melodyTimes': [0., .6, 1., 1.4, 1.8, 2.2], 'melodySemitones': [0, 0, 3, 3, 7, 0],
        'selectionFailurePolicy': 'Record and count failures; no fallback or threshold tuning',
        'asrEquivalents': {w: sorted(v) for w, v in EQUIVALENTS.items()},
        'evaluation': 'Original speech, neural reconstruction and every controlled clip; local English Whisper diagnostic. Matched original/decoded word pairs separately from sustained-word accuracy.',
        'naturalnessAccepted': False, 'newNeuralTraining': False,
    }
    path = OUT / 'protocol.json'
    if path.exists() and read('protocol.json') != config:
        raise ValueError('Frozen protocol changed; use a new versioned experiment')
    if not path.exists(): write('protocol.json', config)
    return config


def capture(config):
    encoder, decoder, _ = load_model(ROOT / config['model'])
    report = {'protocolSha256': sha(OUT / 'protocol.json'), 'expectedSources': 18,
              'sources': [], 'units': [], 'failures': [], 'complete': False}
    if (OUT / 'capture.json').exists():
        existing = read('capture.json')
        if existing['complete']:
            if existing['protocolSha256'] != report['protocolSha256']: raise ValueError('Changed protocol')
            print('Capture already complete; retained.', flush=True); return
    for speaker in SPEAKERS:
        for word in WORDS:
            name = speaker + '-' + word; text = word.capitalize() + '.'
            file = OUT / (name + '-original.wav')
            if not file.exists():
                request = urllib.request.Request('http://127.0.0.1:8791/api/voice',
                    json.dumps({'engine': 'expressive', 'speaker': speaker, 'text': text}).encode(),
                    {'Content-Type': 'application/json'})
                with urllib.request.urlopen(request, timeout=60) as response: result = json.load(response)
                file.write_bytes(base64.b64decode(result['audio']))
            audio, rate = sf.read(file)
            if rate != RATE or audio.ndim != 1 or not np.isfinite(audio).all(): raise ValueError('Invalid teacher audio')
            source = {'name': name, 'speaker': speaker, 'unit': word, 'text': text,
                      'file': str(file.relative_to(ROOT)), 'sha256': sha(file), 'seconds': len(audio)/RATE}
            report['sources'].append(source)
            try:
                rms = np.sqrt(np.convolve(audio**2, np.ones(120)/120, mode='same'))
                active = np.flatnonzero(rms > max(1e-5, .003*max(rms)))
                if not len(active): raise ValueError('Silent source')
                first = max(0, int(active[0])-240); last = min(len(audio), int(active[-1])+241)
                trimmed = audio[first:last]
                original, meta = compile_unit(trimmed, encoder)
                short = max(len(trimmed)/RATE, meta['holdStartSample']/RATE + .150)
                selected, selection = select_stationary_unit(trimmed, original, (short-.150)*RATE)
                artifact = OUT / (name + '-unit.npz')
                save_performance(artifact, original.performance, meta)
                trimmed_file = OUT / (name + '-trimmed.wav'); sf.write(trimmed_file, trimmed, RATE, subtype='DOUBLE')
                decoded = decoder.render(original.performance.matrix)
                decoded_file = OUT / (name + '-decoded.wav'); sf.write(decoded_file, decoded, RATE, subtype='DOUBLE')
                source_formants = formants(trimmed[int(meta['holdStartSample']):int(meta['holdEndSample'])])
                report['units'].append({**source, **meta, 'trimStart': first, 'trimEnd': last,
                    'trimmedSource': str(trimmed_file.relative_to(ROOT)), 'trimmedSha256': sha(trimmed_file),
                    'decodedSource': str(decoded_file.relative_to(ROOT)), 'decodedSha256': sha(decoded_file),
                    'artifact': str(artifact.relative_to(ROOT)), 'artifactSha256': sha(artifact),
                    'selectedFirst': int(selected.hold_first), 'selectedLast': int(selected.hold_last),
                    'selection': selection, 'shortDurationSeconds': short, 'sourceFormantsHz': source_formants,
                    'neuralReconstructionSnrDb': float(10*np.log10(np.sum(trimmed**2)/max(np.sum((trimmed-decoded)**2), 1e-30)))})
                print(name, 'captured', flush=True)
            except ValueError as exc:
                report['failures'].append({**source, 'reason': str(exc)})
                print(name, 'capture failed:', exc, flush=True)
            write('capture.json', report)
    report['complete'] = len(report['sources']) == 18 and len(report['units']) + len(report['failures']) == 18
    write('capture.json', report)


def render(config):
    captured = read('capture.json')
    if not captured['complete']: raise ValueError('Incomplete capture')
    _, neural, _ = load_model(ROOT / config['model']); neural = compile_decoder(neural)
    pcm_decoder = PcmFrameDecoder()
    report = {'protocolSha256': sha(OUT / 'protocol.json'), 'expectedSources': 18,
              'eligibleSources': len(captured['units']), 'captureFailures': captured['failures'],
              'expectedEligibleRenders': len(captured['units'])*12, 'results': [], 'complete': False}
    if (OUT / 'report.json').exists():
        report = read('report.json')
        if report['protocolSha256'] != sha(OUT / 'protocol.json'): raise ValueError('Changed protocol')
    done = {r['name']: r for r in report['results']}
    for source in captured['units']:
        for file, expected in ((source['artifact'], source['artifactSha256']),
                               (source['trimmedSource'], source['trimmedSha256']),
                               (source['decodedSource'], source['decodedSha256'])):
            if sha(ROOT / file) != expected: raise ValueError('Changed captured source')
        performance, _ = load_performance(ROOT / source['artifact'])
        trimmed, rate = sf.read(ROOT / source['trimmedSource']); assert rate == RATE
        decoded, _ = sf.read(ROOT / source['decodedSource'])
        original = ArticulationUnit(performance, source['holdFirstPulse'], source['holdLastPulse'])
        selected = ArticulationUnit(performance, source['selectedFirst'], source['selectedLast'])
        pcm = pcm_performance(trimmed, performance.pulses)
        pcm_original = ArticulationUnit(pcm, original.hold_first, original.hold_last)
        pcm_selected = ArticulationUnit(pcm, selected.hold_first, selected.hold_last)
        if np.max(abs(pcm_decoder.render(pcm.matrix)-trimmed)) > 1e-12: raise ValueError('PCM control is not lossless')
        base = source['sourceMedianPitchHz']
        scores = {'native': VowelScore(seconds=2.5, hz=(base,)),
                  'melody': VowelScore(seconds=2.5, times=tuple(config['melodyTimes']),
                     hz=tuple(base*2**(n/12) for n in config['melodySemitones'])),
                  'short-native': VowelScore(seconds=source['shortDurationSeconds'], hz=(base,))}
        common = max(performance.pulses[original.hold_first], performance.pulses[selected.hold_first])/RATE
        methods = {'neural-loop': (original, neural, 'loop', decoded),
                   'neural-fixed': (selected, neural, 'fixed-first', decoded),
                   'pcm-loop': (pcm_original, pcm_decoder, 'loop', trimmed),
                   'pcm-fixed': (pcm_selected, pcm_decoder, 'fixed-first', trimmed)}
        for mode, score in scores.items():
            for method, (unit, decoder, hold_mode, reconstruction) in methods.items():
                name = source['name'] + '-' + mode + '-' + method
                if name in done:
                    if sha(ROOT / done[name]['file']) != done[name]['sha256']: raise ValueError('Changed render')
                    continue
                audio, timing = controlled_render(unit, decoder, score,
                    stream_factory=partial(NeuralArticulationStream, hold_mode=hold_mode))
                if method.startswith('pcm'):
                    timing['timingScope'] = 'Explicit lossless waveform-frame reference with identical controller; retains original source waveform, not a neural model. Loading, capture and device latency excluded; 20 ms waveform delay is additional.'
                actual = float(performance.pulses[unit.hold_first])
                changed = {**source, 'holdStartSample': actual}
                evaluated, tracks = metrics(audio, score, unit, changed, reconstruction, analysis_hold_start=common)
                evaluated['strictUnvoicedPrefix']['reference'] = 'original PCM' if method.startswith('pcm') else 'neural reconstruction'
                evaluated['strictUnvoicedPrefix']['rmsDifferenceFromReference'] = evaluated['strictUnvoicedPrefix'].pop('rmsDifferenceFromNeuralSource')
                file = OUT / (name+'.wav'); sf.write(file, audio, RATE, subtype='DOUBLE')
                track = OUT / (name+'.tracks.npz'); np.savez(track, **tracks)
                row = {'name': name, 'speaker': source['speaker'], 'unit': source['unit'], 'text': source['text'],
                       'mode': mode, 'method': method, 'score': score.__dict__, 'file': str(file.relative_to(ROOT)),
                       'sha256': sha(file), 'tracks': str(track.relative_to(ROOT)),
                       'commonAnalysisHoldStartSeconds': common, 'actualHoldStartSeconds': actual/RATE,
                       'streaming': timing, **evaluated}
                report['results'].append(row); write('report.json', report)
                print(name, 'Praat p95', row['praat']['p95ErrorCents'], flush=True)
    protocol()
    report['frozenProtocolAndCodeVerified'] = True
    report['complete'] = len(report['results']) == report['expectedEligibleRenders']
    write('report.json', report)


def words(config):
    captured = read('capture.json'); study = read('report.json')
    if not captured['complete'] or not study['complete']: raise ValueError('Incomplete capture/render')
    cases = [{**source, 'name': source['name']+'-original-source', 'mode': 'original-source', 'method': 'original-source'}
             for source in captured['sources']]
    for source in captured['units']:
        for mode, field, hash_field in (('source', 'trimmedSource', 'trimmedSha256'),
                                       ('decoded', 'decodedSource', 'decodedSha256')):
            cases.append({**source, 'name': source['name']+'-'+mode, 'mode': mode, 'method': mode,
                          'file': source[field], 'sha256': source[hash_field]})
    cases += study['results']
    report = {'protocolSha256': sha(OUT/'protocol.json'), 'expectedCases': len(cases), 'results': [], 'complete': False,
              'equivalents': config['asrEquivalents'],
              'limits': ['Short-word and sustained-tone ASR are imperfect; this is not listener acceptance.',
                         'Fresh source words in existing synthetic voices; not a human-reference or unseen-speaker test.']}
    if (OUT/'word-report.json').exists(): report = read('word-report.json')
    if report['protocolSha256'] != sha(OUT/'protocol.json'): raise ValueError('Changed protocol')
    done = {r['name']: r for r in report['results']}
    for case in cases:
        file = ROOT/case['file']; payload = file.read_bytes()
        if sha(file) != case['sha256']: raise ValueError('Changed audio')
        if case['name'] in done:
            if done[case['name']]['sha256'] != case['sha256']: raise ValueError('Changed ASR source')
            continue
        request = urllib.request.Request('http://127.0.0.1:8791/api/transcribe', payload, {'Content-Type':'audio/wav'})
        with urllib.request.urlopen(request, timeout=60) as response: result = json.load(response)
        if 'text' not in result: raise ValueError('Missing ASR result')
        transcript = result['text']; normalized = ' '.join(re.findall(r'[a-z]+', transcript.lower()))
        report['results'].append({k: case[k] for k in ('name','speaker','unit','mode','method','file','sha256')} |
            {'transcript': transcript, 'normalized': normalized, 'exactSpellingMatch': normalized == case['unit'],
             'homophoneEquivalentMatch': normalized in EQUIVALENTS[case['unit']]})
        write('word-report.json', report); print(case['name'], repr(transcript), flush=True)
    report['complete'] = len(report['results']) == len(cases); write('word-report.json', report)


def summarize(config):
    captured = read('capture.json'); study = read('report.json'); lexical = read('word-report.json')
    if not all(j['complete'] for j in (captured, study, lexical)): raise ValueError('Incomplete study')
    groups = []
    for mode in ('original-source', 'source', 'decoded', *config['modes']):
        methods = [mode] if mode in ('original-source','source','decoded') else config['methods']
        for method in methods:
            rows = [r for r in lexical['results'] if r['mode'] == mode and r['method'] == method]
            acoustic = [r for r in study['results'] if r['mode'] == mode and r['method'] == method]
            group = {'mode': mode, 'method': method, 'cases': len(rows),
                     'soundClassMatches': sum(r['homophoneEquivalentMatch'] for r in rows),
                     'exactSpellingMatches': sum(r['exactSpellingMatch'] for r in rows),
                     'mismatches': [{'name':r['name'],'transcript':r['transcript']} for r in rows if not r['homophoneEquivalentMatch']]}
            if acoustic:
                group.update(praat=pitch_group(acoustic,'praat'), world=pitch_group(acoustic,'world'))
            groups.append(group)
    paired = []
    for row in study['results']:
        if not row['method'].startswith('neural'): continue
        other = next(r for r in study['results'] if r['name'] == row['name'].replace('-neural-', '-pcm-'))
        a,_ = sf.read(ROOT/row['file']); b,_ = sf.read(ROOT/other['file'])
        neural_words = next(r for r in lexical['results'] if r['name'] == row['name'])
        pcm_words = next(r for r in lexical['results'] if r['name'] == other['name'])
        paired.append({'name': row['name'], 'snrDb': float(10*np.log10(np.sum(b*b)/max(np.sum((a-b)**2),1e-30))),
                       'sameNormalizedTranscript': neural_words['normalized'] == pcm_words['normalized'],
                       'neuralTranscript': neural_words['transcript'], 'pcmTranscript': pcm_words['transcript']})
    streams = [r['streaming'] for r in study['results'] if r['method'].startswith('neural')]
    source_pairs=[]
    for source in captured['units']:
        versions={mode:next(r for r in lexical['results'] if r['name']==source['name']+'-'+mode)
                  for mode in ('original-source','source','decoded')}
        source_pairs.append({'name':source['name'],
            'original':versions['original-source']['transcript'], 'trimmed':versions['source']['transcript'],
            'decoded':versions['decoded']['transcript'],
            'originalSoundClassMatch':versions['original-source']['homophoneEquivalentMatch'],
            'sameAfterTrimming':versions['original-source']['normalized']==versions['source']['normalized'],
            'sameAfterNeuralReconstruction':versions['source']['normalized']==versions['decoded']['normalized']})
    summary = {'scope': config['scope'], 'protocol': str((OUT/'protocol.json').relative_to(ROOT)),
        'sourceCount':len(captured['sources']), 'eligibleSources':len(captured['units']), 'captureFailures':captured['failures'],
        'completedRenders':len(study['results']), 'localAsrEvaluations':len(lexical['results']), 'groups':groups,
        'eligibleSourcePairs':{'cases':len(source_pairs),
            'originalSoundClassMatches':sum(r['originalSoundClassMatch'] for r in source_pairs),
            'sameNormalizedTranscriptAfterTrimming':sum(r['sameAfterTrimming'] for r in source_pairs),
            'sameNormalizedTranscriptAfterNeuralReconstruction':sum(r['sameAfterNeuralReconstruction'] for r in source_pairs),
            'rows':source_pairs},
        'pcmControlComparison':{'pairs':len(paired),'medianSnrDb':stats.median(r['snrDb'] for r in paired),
            'minimumSnrDb':min(r['snrDb'] for r in paired),'sameNormalizedTranscripts':sum(r['sameNormalizedTranscript'] for r in paired),
            'rows':paired,'scope':'Identical source-pulse and performance controls; isolates neural waveform approximation from the resynthesis method.'},
        'streamingNeuralOnly':{'cases':len(streams),'fixedWaveformDelayMs':20.,
            'medianFirstPacketComputeMs':stats.median(r['firstPhysicalPacketComputeMs'] for r in streams),
            'maximumPacketMs':max(r['maximumPacketMs'] for r in streams),'packetsOver5Ms':sum(r['packetsOver5Ms'] for r in streams),
            'maximumCachedWaveformSamples':max(r['maximumCachedWaveformSamples'] for r in streams),
            'scope':streams[0]['timingScope']},
        'allFinite':all(r['finite'] for r in study['results']),
        'maximumRawPeak':max(r['rawPeak'] for r in study['results']),
        'overFullScaleSamples':sum(r['overFullScaleSamples'] for r in study['results']),
        'frozenProtocolAndCodeVerified':study['frozenProtocolAndCodeVerified'],
        'limitations':lexical['limits']+['Source pulse detection and span selection may identify the wrong sound.',
            'PCM-reference inference deliberately uses original waveform data; it is an experimental control, not an independent generator.',
            'No new neural training, general pronunciation learning, human cloning or listening acceptance.'],
        'listeningAccepted':False,'completionProven':False}
    write('summary.json', summary)
    for r in groups: print(r['mode'],r['method'],r['soundClassMatches'],'/',r['cases'])
    print('Neural/PCM median SNR',summary['pcmControlComparison']['medianSnrDb'],
          'same transcripts',summary['pcmControlComparison']['sameNormalizedTranscripts'],'/',len(paired))
    print('streaming',summary['streamingNeuralOnly'])


if __name__ == '__main__':
    config = protocol()
    {'capture':capture,'render':render,'words':words,'summarize':summarize}[sys.argv[1]](config)
