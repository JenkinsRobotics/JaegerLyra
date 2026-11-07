"""Fresh-vowel validation of frozen pretrained Kokoro control/normalization rules.

Stages: render (.voice-venv), evaluate (.research-venv), summarize (either).
This is a pretrained reference, not an independent neural speech model.
"""
import os
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('HF_HUB_DISABLE_TELEMETRY', '1')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS', '1')
import hashlib
import json
import re
import sys
import time
import urllib.request
from pathlib import Path
import numpy as np
from scipy.io import wavfile
from capture_kokoro_controls import VOWELS, KEYS
from kokoro_control_sustain import interior_state, stretch_controls

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'evidence/kokoro-control-voicebox/fresh-vowels-v1'
WORDS = ('calm', 'palm', 'paw', 'raw', 'blue', 'true')
SPEAKERS = ('af_heart', 'am_michael', 'af_bella')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(name): return json.loads((OUT / name).read_text())
def write(name, data):
    (OUT / name).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def protocol():
    OUT.mkdir(parents=True, exist_ok=True)
    model = 'models/expressive/Kokoro-82M-bf16'
    config = {
        'scope': 'Frozen interior-hold and reference-normalization validation on new words and vowel classes in three existing synthetic voices',
        'words': list(WORDS), 'speakers': list(SPEAKERS), 'expectedSources': 18,
        'modes': ['native', 'melody'], 'methods': ['normal', 'reference'],
        'maximumRenders': 72, 'seconds': 3., 'seed': 42,
        'model': model, 'modelWeightsSha256': sha(ROOT / model / 'kokoro-v1_0.safetensors'),
        'frozenCode': {name: sha(Path(__file__).parent / name) for name in (
            'kokoro_control_sustain.py', 'kokoro_normalization_reference.py')},
        'pitchRule': 'Use the frozen interior-state F0 as base; native holds it, melody interpolates semitones across inserted-hold fractions.',
        'melodyHoldFractions': [0., .15, .35, .55, .75, .90, 1.],
        'melodySemitones': [0, 0, 4, 7, 4, 0, 0],
        'analysis': 'Identical inserted-hold interval for both methods, excluding 80 ms at each end; Praat and WORLD independently.',
        'pitchCheck': {'minimumVoicedFraction': .95, 'maximumP95ErrorCents': 50.},
        'nativeParityMaximumSampleDifference': 1e-6,
        'failurePolicy': 'Record and count source, selection and rendering failures; do not change thresholds or substitute words.',
        'asrEquivalents': {word: sorted({word, 'blew'} if word == 'blue' else {word}) for word in WORDS},
        'limits': ['Existing pretrained decoder and complete captured controls; not independent training or causal streaming.',
            'New source words and vowel classes relative to the me/see/bee control study; not unseen-speaker validation.',
            'No source-audio clipping, levelling or post-filtering; raw float WAVs.',
            'Sustained-word ASR is diagnostic only; pitch checks cannot establish naturalness or identity.',
            'Normal-source statistics are per utterance, not a generalized speaker embedding.'],
        'listeningAccepted': False, 'appDefaultChanged': False,
    }
    if (OUT / 'protocol.json').exists():
        if read('protocol.json') != config: raise ValueError('Frozen protocol or code changed')
    else: write('protocol.json', config)
    return config


def render(config):
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    from kokoro_normalization_reference import NormalizationReference
    mx.set_cache_limit(128 * 1024 * 1024)
    report = {'protocolSha256': sha(OUT / 'protocol.json'), 'sources': [], 'failures': [], 'results': [], 'complete': False}
    if (OUT / 'report.json').exists(): report = read('report.json')
    if report['protocolSha256'] != sha(OUT / 'protocol.json'): raise ValueError('Changed protocol')
    if report['complete']:
        print('Render already complete; retained.', flush=True); return
    model = load_model(ROOT / config['model'])
    dtypes = {'float32': mx.float32, 'float16': mx.float16, 'bfloat16': mx.bfloat16}
    done = {r['name']: r for r in report['results']}
    failed = {r['name'] for r in report['failures']}
    for speaker in SPEAKERS:
        pipeline = model._get_pipeline(speaker[0]); pack = pipeline.load_voice(speaker)
        for word in WORDS:
            name = speaker + '-' + word
            if name in failed: continue
            source = next((r for r in report['sources'] if r['name'] == name), None)
            if source is None:
                text = word.capitalize() + '.'; _, tokens = pipeline.g2p(text)
                segments = list(pipeline.en_tokenize(tokens))
                if len(segments) != 1: raise ValueError('Unexpected source segmentation')
                phones = ''.join(p for p in segments[0][1] if p in model.vocab)
                ref = pack[len(phones)-1]; native = model.decoder; captured = {}; types = {}
                def observe(*args):
                    for key, value in zip(KEYS, args):
                        mx.eval(value); types[key] = str(value.dtype).split('.')[-1]
                        if types[key] not in dtypes: raise ValueError('Unsupported dtype')
                        captured[key] = np.array(value.astype(mx.float32))
                    return native(*args)
                mx.random.seed(config['seed']); model.decoder = observe
                try: normal = model(phones, ref, return_output=True)
                finally: model.decoder = native
                mx.eval(normal.audio, normal.pred_dur)
                audio = np.array(normal.audio.astype(mx.float32)).reshape(-1)
                duration = np.array(normal.pred_dur).astype(int)
                file = OUT / (name + '-source.wav'); wavfile.write(file, 24000, audio)
                saved = OUT / (name + '-controls.npz')
                np.savez(saved, **captured, naturalDurationFrames=duration,
                         metadata=json.dumps({'dtypes': types, 'phonemes': phones, 'speaker': speaker, 'seed': config['seed']}))
                source = {'name': name, 'speaker': speaker, 'word': word, 'text': text, 'phonemes': phones,
                    'file': str(file.relative_to(ROOT)), 'sha256': sha(file),
                    'controls': str(saved.relative_to(ROOT)), 'controlsSha256': sha(saved),
                    'seconds': len(audio)/24000, 'naturalDurationFrames': duration.tolist()}
                report['sources'].append(source); write('report.json', report)
            saved = ROOT / source['controls']
            if sha(saved) != source['controlsSha256'] or sha(ROOT / source['file']) != source['sha256']:
                raise ValueError('Changed captured source')
            with np.load(saved, allow_pickle=False) as data:
                controls = {k: data[k].copy() for k in KEYS}; duration = data['naturalDurationFrames']
                metadata = json.loads(str(data['metadata']))
            try:
                chars = [''] + list(source['phonemes']) + ['']; nuclei = [i for i, c in enumerate(chars) if c in VOWELS]
                if len(nuclei) != 1: raise ValueError('Expected one vowel, got ' + repr(nuclei))
                nucleus = nuclei[0]; frame, selection = interior_state(controls, duration, nucleus)
                extra = round(config['seconds'] * 40) - int(duration.sum())
                if extra < 16: raise ValueError('Insufficient inserted hold for common evaluation')
            except ValueError as exc:
                report['failures'].append({'name': name, 'stage': 'source-selection', 'reason': str(exc)})
                write('report.json', report); print(name, 'REJECTED', exc, flush=True); continue
            args = [mx.array(controls[k], dtype=dtypes[metadata['dtypes'][k]]) for k in KEYS]
            reference = NormalizationReference(model.decoder); mx.random.seed(config['seed'])
            with reference.apply('capture'):
                normal = model.decoder(*args)[0]; mx.eval(normal)
            mx.random.seed(config['seed'])
            with reference.apply('reference'):
                replay = model.decoder(*args)[0]; mx.eval(replay)
            normal = np.array(normal.astype(mx.float32)).reshape(-1)
            replay = np.array(replay.astype(mx.float32)).reshape(-1)
            rate, source_audio = wavfile.read(ROOT / source['file']); assert rate == 24000
            parity = float(max(np.max(abs(normal-source_audio)), np.max(abs(replay-source_audio))))
            if parity > config['nativeParityMaximumSampleDifference']: raise ValueError('Native parity failed')
            statistics = OUT / (name + '-normalization.npz'); stats_info = reference.save(statistics)
            # Reload the saved values so every controlled render uses persisted state.
            reference = NormalizationReference.load(model.decoder, statistics)
            source.update(nucleus=nucleus, selection=selection, nativeReplayMaximumDifference=parity,
                statistics=str(statistics.relative_to(ROOT)), statisticsSha256=sha(statistics), statisticsInfo=stats_info)
            begin = frame/40; end = (frame+extra)/40; base = selection['sourceF0']
            for mode in config['modes']:
                if mode == 'native': times = [0.]; hz = [base]
                else:
                    times = [0.] + [begin + f*(end-begin) for f in config['melodyHoldFractions']]
                    hz = [base] + [base*2**(n/12) for n in config['melodySemitones']]
                target = 2**np.interp(np.arange(round(config['seconds']*80))/80, times, np.log2(hz))
                edited, plan, maps = stretch_controls(controls, duration, nucleus, config['seconds'], 'interior-hold', target)
                controls_file = OUT / (name+'-'+mode+'-edited.npz')
                np.savez(controls_file, **edited, **maps, requestedF0=target, metadata=json.dumps(plan))
                edited_args = [mx.array(edited[k], dtype=dtypes[metadata['dtypes'][k]]) for k in KEYS]
                for method in config['methods']:
                    case = name+'-'+mode+'-'+method
                    if case in done:
                        if sha(ROOT / done[case]['file']) != done[case]['sha256']: raise ValueError('Changed prior render')
                        continue
                    mx.random.seed(config['seed']); start = time.perf_counter()
                    if method == 'reference':
                        with reference.apply('reference'):
                            wave = model.decoder(*edited_args)[0]; mx.eval(wave)
                    else: wave = model.decoder(*edited_args)[0]; mx.eval(wave)
                    elapsed = time.perf_counter()-start; audio = np.array(wave.astype(mx.float32)).reshape(-1)
                    if len(audio) != round(config['seconds']*24000) or not np.isfinite(audio).all(): raise ValueError('Invalid controlled audio')
                    file = OUT / (case+'.wav'); wavfile.write(file, 24000, audio)
                    report['results'].append({'name': case, 'speaker': speaker, 'word': word, 'mode': mode, 'method': method,
                        'file': str(file.relative_to(ROOT)), 'sha256': sha(file), 'plan': plan,
                        'score': {'seconds': config['seconds'], 'times': times, 'hz': hz},
                        'controls': str(controls_file.relative_to(ROOT)), 'controlsSha256': sha(controls_file),
                        'rawPeak': float(max(abs(audio))), 'overFullScaleSamples': int(np.sum(abs(audio)>=1.)),
                        'finite': True, 'wholeDecoderComputeSeconds': elapsed})
                    write('report.json', report); print(case, 'peak', round(max(abs(audio)), 3), flush=True)
                    mx.clear_cache()
    protocol()
    report['expectedEligibleRenders'] = (len(report['sources'])-len(report['failures']))*4
    report['complete'] = len(report['sources']) == 18 and len(report['results']) == report['expectedEligibleRenders']
    write('report.json', report)


def evaluate(config):
    import parselmouth
    import pyworld
    from neural_vowel_instrument import VowelScore
    from benchmark_neural_vowels import pitch_metrics
    study = read('report.json')
    if not study['complete']: raise ValueError('Incomplete renders')
    report = {'protocolSha256': sha(OUT/'protocol.json'), 'results': [], 'sources': [], 'complete': False}
    if (OUT/'evaluation.json').exists(): report = read('evaluation.json')
    if report['protocolSha256'] != sha(OUT/'protocol.json'): raise ValueError('Changed protocol')
    for group, cases in [('sources', study['sources']), ('results', study['results'])]:
        done = {r['name']: r for r in report[group]}
        for row in cases:
            file = ROOT / row['file']
            if sha(file) != row['sha256']: raise ValueError('Changed source/audio')
            if row['name'] in done:
                if done[row['name']]['sha256'] != row['sha256']: raise ValueError('Changed prior evaluation')
                continue
            rate, audio = wavfile.read(file); audio = np.asarray(audio, dtype=np.float64); assert rate == 24000
            evaluated = {k: row[k] for k in ('name', 'speaker', 'word', 'file', 'sha256')}
            if group == 'results':
                score = VowelScore(**row['score']); score.validate()
                start = row['plan']['holdStartSeconds']+.08; end = row['plan']['holdEndSeconds']-.08
                praat = parselmouth.Sound(audio, sampling_frequency=rate).to_pitch_ac(time_step=.01,pitch_floor=50.,pitch_ceiling=900.)
                wf, wt = pyworld.harvest(np.ascontiguousarray(audio), rate, f0_floor=50.,f0_ceil=900.,frame_period=10.)
                tracks = {}
                for name, times, f0 in [('praat', praat.xs(), praat.selected_array['frequency']), ('world', wt, wf)]:
                    keep = (times>=start)&(times<=end); measured = pitch_metrics(times[keep], f0[keep], score)
                    measured['pitchCheckPassed'] = bool(measured['voicedFraction'] >= config['pitchCheck']['minimumVoicedFraction']
                        and measured['p95ErrorCents'] is not None and measured['p95ErrorCents'] <= config['pitchCheck']['maximumP95ErrorCents'])
                    evaluated[name] = measured; tracks[name+'Times'] = times; tracks[name+'Pitch'] = f0
                tracks['requestedTimes'] = np.arange(round(score.seconds*100))/100
                tracks['requestedF0'] = score.frequency(tracks['requestedTimes'])
                track = OUT/(row['name']+'-tracks.npz'); np.savez(track, **tracks)
                region = audio[round(start*rate):round(end*rate)]
                evaluated.update(mode=row['mode'], method=row['method'], analysisStartSeconds=start, analysisEndSeconds=end,
                    tracks=str(track.relative_to(ROOT)), heldRms=float(np.sqrt(np.mean(region**2))),
                    bothPitchChecksPassed=evaluated['praat']['pitchCheckPassed'] and evaluated['world']['pitchCheckPassed'])
            request = urllib.request.Request('http://127.0.0.1:8791/api/transcribe', file.read_bytes(), {'Content-Type':'audio/wav'})
            with urllib.request.urlopen(request, timeout=60) as response: result = json.load(response)
            transcript = result['text']; normalized = ' '.join(re.findall(r'[a-z]+', transcript.lower()))
            evaluated.update(transcript=transcript, exactSpellingMatch=normalized==row['word'],
                homophoneEquivalentMatch=normalized in config['asrEquivalents'][row['word']])
            report[group].append(evaluated); write('evaluation.json', report)
            print(row['name'], repr(transcript), evaluated.get('bothPitchChecksPassed'), flush=True)
    report['complete'] = len(report['results']) == len(study['results']) and len(report['sources']) == 18
    write('evaluation.json', report)


def summarize(config):
    study = read('report.json'); evaluation = read('evaluation.json')
    if not study['complete'] or not evaluation['complete']: raise ValueError('Incomplete study')
    groups = []
    for mode in config['modes']:
        for method in config['methods']:
            rows = [r for r in evaluation['results'] if r['mode']==mode and r['method']==method]
            groups.append({'mode': mode, 'method': method, 'cases': len(rows),
                'bothPitchChecksPassed': sum(r['bothPitchChecksPassed'] for r in rows),
                'minimumPraatVoicedFraction': min(r['praat']['voicedFraction'] for r in rows),
                'minimumWorldVoicedFraction': min(r['world']['voicedFraction'] for r in rows),
                'medianHeldRms': float(np.median([r['heldRms'] for r in rows])),
                'homophoneEquivalentMatches': sum(r['homophoneEquivalentMatch'] for r in rows),
                'failedPitchCases': [{k:r[k] for k in ('name','praat','world')} for r in rows if not r['bothPitchChecksPassed']]})
    result = {'scope': config['scope'], 'protocol': str((OUT/'protocol.json').relative_to(ROOT)),
        'sources': len(study['sources']), 'eligibleSources': len(study['sources'])-len(study['failures']),
        'selectionFailures': study['failures'], 'renders': len(study['results']),
        'localAsrEvaluations': len(evaluation['sources'])+len(evaluation['results']), 'groups': groups,
        'nativeMaximumReplayDifference': max(r['nativeReplayMaximumDifference'] for r in study['sources'] if 'nativeReplayMaximumDifference' in r),
        'sourceWordMatches': sum(r['homophoneEquivalentMatch'] for r in evaluation['sources']),
        'allFinite': all(r['finite'] for r in study['results']), 'maximumRawPeak': max(r['rawPeak'] for r in study['results']),
        'overFullScaleSamples': sum(r['overFullScaleSamples'] for r in study['results']),
        'referenceWholeClipComputeMsMedian': float(np.median([r['wholeDecoderComputeSeconds']*1000 for r in study['results'] if r['method']=='reference'])),
        'timingScope': 'Resident pretrained decoder and prepared controls/statistics, whole three-second clip; excludes loading/preparation/devices, not streaming latency.',
        'frozenProtocolAndCodeVerified': True, 'limitations': config['limits'], 'listeningAccepted': False,
        'completionProven': False, 'appDefaultChanged': False}
    write('summary.json', result); print(json.dumps(result, indent=2))


if __name__ == '__main__':
    config = protocol()
    {'render': render, 'evaluate': evaluate, 'summarize': summarize}[sys.argv[1]](config)
