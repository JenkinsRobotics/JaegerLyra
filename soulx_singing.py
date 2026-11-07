"""Explicit experimental SoulX adapter, separate from accepted Kokoro speech.

Uses pinned upstream SoulX code/weights and synthetic Kokoro references. This is
not an independent or clean-room engine. Heavy imports only run in research venv.
"""
import json
import math
import re
import sys
from pathlib import Path
from kokoro_singing import VOICES, MELODIES

ROOT = Path(__file__).resolve().parent
REFERENCE_FOLDER = ROOT / '.cache/singing-references'


def validate_request(data):
    lyrics = data.get('lyrics', '')
    if not isinstance(lyrics, str):
        raise ValueError('Enter English lyrics.')
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    if not lines or len(lyrics) > 180 or len(lines) > 4 or any(len(line)>90 for line in lines):
        raise ValueError('Use 1–4 short English lines, up to 180 characters.')
    if re.search(r"[^A-Za-z\s.,!?;:'’\-()\"]", lyrics) or not re.search('[A-Za-z]', lyrics):
        raise ValueError('Use English words; write numbers out in words.')
    speaker = data.get('speaker', 'af_heart')
    melody = data.get('melody', 'glow')
    kind = data.get('referenceKind', 'speech')
    if speaker not in VOICES or melody not in MELODIES or kind not in ('speech', 'sung'):
        raise ValueError('Choose a listed voice, melody and reference type.')
    try:
        tempo = float(data.get('tempo', 104))
        transpose = float(data.get('transpose', 0))
    except (ValueError, TypeError):
        raise ValueError('Choose a valid tempo and pitch range.') from None
    if not math.isfinite(tempo) or not 72 <= tempo <= 132:
        raise ValueError('Tempo must be 72–132 BPM.')
    if not math.isfinite(transpose) or not transpose.is_integer() or not -12 <= transpose <= 12:
        raise ValueError('Pitch range must be within one octave.')
    return {'engine': 'soulx', 'version': 1, 'lyrics': '\n'.join(lines),
            'speaker': speaker, 'melody': melody, 'tempo': tempo,
            'transpose': int(transpose), 'referenceKind': kind, 'steps': 32, 'cfg': 3.0}


def reference_base(request):
    # Validated symbolic names only; never accept a client filesystem path.
    request = validate_request(request)
    return REFERENCE_FOLDER / ('v1-'+request['speaker']+'-'+request['referenceKind'])


def generate(request, progress=lambda message: None):
    progress('Preparing the singing reference…')
    import os
    os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    os.environ['NLTK_DATA'] = str(ROOT/'.cache/nltk')
    from scripts.try_soulx import make_prompt, make_sung_prompt, SOURCE
    import copy
    import time
    import numpy as np
    import torch
    from omegaconf import OmegaConf
    from singing_model import load_singer
    from soulxsinger.utils.data_processor import DataProcessor
    from g2p_en import G2p
    from english_singing_score import plan_english
    request = validate_request(request)
    reference = reference_base(request)
    prompt, reference_wav = (make_sung_prompt(reference) if request['referenceKind']=='sung'
                             else make_prompt(fixture=reference))
    target = plan_english(request['lyrics'], G2p(), request['tempo'],
                          request['melody'], request['transpose'])
    torch.set_num_threads(4)
    torch.manual_seed(42)
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    config = OmegaConf.load(ROOT/'models/soulx-singer/config.yaml')
    start = time.perf_counter()
    progress('Loading the experimental singer…')
    model = load_singer(ROOT/'models/soulx-singer/model.pt', config, device=device)
    loaded = time.perf_counter()
    processor = DataProcessor(config.audio.hop_size, config.audio.sample_rate,
        str(SOURCE/'soulxsinger/utils/phoneme/phone_set.json'), device=device)
    data = {'prompt': processor.process(copy.deepcopy(prompt), str(reference_wav)),
            'target': processor.process(copy.deepcopy(target), None)}
    progress('Singing the lyrics…')
    with torch.inference_mode():
        audio = model.infer(data, auto_shift=False, pitch_shift=0,
            n_steps=request['steps'], cfg=request['cfg'], control='score').squeeze().cpu().numpy()
    if not np.isfinite(audio).all() or not len(audio):
        raise RuntimeError('The singer returned invalid audio.')
    # One gain over the entire clip; no note-by-note loudness pumping.
    rms = float(np.sqrt(np.mean(audio*audio)))
    peak = float(np.max(np.abs(audio)))
    audio *= min(.06/max(rms, .001), .88/max(peak, .001), 2.5)
    details = {'engine': 'soulx-singing', 'speaker': request['speaker'],
               'referenceKind': request['referenceKind'], 'score': target,
               'loadSeconds': round(loaded-start, 3),
               'renderSeconds': round(time.perf_counter()-loaded, 3),
               'device': device, 'loader': 'empty-parameters-mmap-assign',
               'steps': request['steps'], 'cfg': request['cfg'],
               'duration': len(audio)/24000,
               'notice': 'Experimental reference-conditioned singer. Identity and pronunciation can change.'}
    return audio, details


if __name__ == '__main__':
    import soundfile as sf
    path = Path(sys.argv[1])
    def progress(message):
        temporary=path.with_suffix('.progress.tmp')
        temporary.write_text(json.dumps({'phase':message}))
        temporary.replace(path.with_suffix('.progress.json'))
    try:
        audio, result = generate(json.loads(path.read_text()), progress)
    except ValueError as error:
        path.with_suffix('.error.json').write_text(json.dumps({'error': str(error)}))
        raise
    output = path.with_suffix('.wav')
    sf.write(output, audio, 24000, subtype='PCM_16')
    result['path'] = str(output)
    temporary = path.with_suffix('.result.tmp')
    temporary.write_text(json.dumps(result))
    temporary.replace(path.with_suffix('.result.json'))
    print(json.dumps({k:v for k,v in result.items() if k!='score'}), flush=True)
