"""Prepare a synthetic Kokoro reference, with alignment, in the MLX venv.

Runs as its own cancellable stage before loading the research singer. No uploaded
human reference, STT, forced alignment or per-speaker weight training is used.
"""
import json
import os
import sys
import wave
from pathlib import Path
import numpy as np
from soulx_singing import validate_request, reference_base
from kokoro_singing import ROOT


def main():
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['HF_HUB_CACHE'] = str(ROOT/'models/hf-cache')
    request = validate_request(json.loads(Path(sys.argv[1]).read_text()))
    base = reference_base(request)
    base.parent.mkdir(parents=True, exist_ok=True)
    if request['referenceKind'] == 'sung':
        from kokoro_singing import generate
        audio, score = generate({'lyrics': 'We can watch the stars\nShine in the sky',
            'speaker': request['speaker'], 'tempo': 104, 'melody': 'lullaby'})
        metadata = {'score': score, 'duration': len(audio)/24000,
                    'speaker': request['speaker'], 'origin': 'Synthetic Kokoro score adapter'}
        meta_path = base.with_suffix('.result.json')
    else:
        import mlx.core as mx
        from mlx_audio.tts.utils import load_model
        mx.set_cache_limit(128*1024*1024)
        mx.random.seed(42)
        model = load_model(ROOT/'models/expressive/Kokoro-82M-bf16')
        pipeline = model._get_pipeline(request['speaker'][0])
        text = 'You light up my day. Stay here close to me. We can watch the stars shine in the sky.'
        parts, words, offset = [], [], 0.
        for result in pipeline(text, voice=request['speaker'], speed=1.):
            part = np.asarray(result.audio).reshape(-1)
            parts.append(part)
            for token in result.tokens or []:
                words.append({'text': token.text,
                    'start': None if token.start_ts is None else offset+token.start_ts,
                    'end': None if token.end_ts is None else offset+token.end_ts})
            offset += len(part)/24000
        audio = np.concatenate(parts)
        metadata = {'words': words, 'text': text, 'duration': len(audio)/24000,
                    'speaker': request['speaker'], 'origin': 'Synthetic Kokoro speech, model-predicted word timing'}
        meta_path = base.with_suffix('.json')
    if not np.isfinite(audio).all():
        raise RuntimeError('Invalid reference waveform')
    temporary = base.with_suffix('.wav.tmp')
    with wave.open(str(temporary), 'wb') as wav:
        wav.setparams((1, 2, 24000, 0, 'NONE', 'not compressed'))
        wav.writeframes((np.clip(audio, -1, 1)*32767).astype('<i2').tobytes())
    temporary.replace(base.with_suffix('.wav'))
    temporary = meta_path.with_suffix('.tmp')
    temporary.write_text(json.dumps(metadata))
    temporary.replace(meta_path)
    print('Prepared', request['speaker'], request['referenceKind'], metadata['duration'], flush=True)


if __name__ == '__main__':
    main()
