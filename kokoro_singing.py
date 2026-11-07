"""Experimental score-conditioned Kokoro inference, with the original voice decoder.

The model forward pass is adapted from mlx-audio 0.6.4 (MIT, Prince Canuma;
see licenses/mlx-audio-LICENSE.txt). Our score planner replaces phoneme timing
and voiced pitch. This is an inference extension, not a singing-trained model.
"""
import json
import os
import re
import sys
import time
import wave
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
VOICES = ('af_heart', 'af_bella', 'af_nicole', 'am_michael', 'bf_emma', 'bm_george')
VOWELS = set('ɑɐɒæəɛɜɚɝɪiɔʊuʌʉɨɯɶœøyᵊAEIOUYaeo')
MELODIES = {'glow': [60, 62, 64, 67, 65, 64, 62, 60],
            'lullaby': [64, 62, 60, 62, 64, 62, 59, 60]}


def validate_request(data):
    lyrics = re.sub(r'\[[^\]]*\]', '', str(data.get('lyrics', ''))).strip()
    lines = [line.strip() for line in lyrics.splitlines() if line.strip()]
    if not lines or len(lyrics) > 180 or len(lines) > 4:
        raise ValueError('Use 1–4 short English lines, up to 180 characters.')
    if any(len(line) > 90 for line in lines):
        raise ValueError('Keep each lyric line under 90 characters.')
    speaker = data.get('speaker', 'af_heart')
    melody = data.get('melody', 'glow')
    tempo = float(data.get('tempo', 104))
    if speaker not in VOICES or melody not in MELODIES:
        raise ValueError('Choose a listed Kokoro voice and melody.')
    if not np.isfinite(tempo) or not 72 <= tempo <= 132:
        raise ValueError('Tempo must be between 72 and 132 BPM.')
    return {'engine': 'kokoro', 'version': 1, 'lyrics': '\n'.join(lines),
            'speaker': speaker, 'melody': melody, 'tempo': tempo}


def plan_score(phonemes, natural, tempo=104, melody='glow', transpose=0, line=0):
    """Assign one note per vowel; extend vowels while keeping consonants short."""
    chars = [''] + list(phonemes) + ['']
    nuclei = [i for i, p in enumerate(chars) if p in VOWELS]
    if not nuclei or len(nuclei) > 16:
        raise ValueError('Use a short lyric line with 1–16 sung syllables.')
    durations = np.clip(np.asarray(natural, dtype=int), 1, 8)
    durations[0] = 3; durations[-1] = 5
    for i, p in enumerate(chars):
        if p and p in ' ˈˌ.,!?;:': durations[i] = 1
    boundaries = [1] + [(a+b)//2+1 for a,b in zip(nuclei, nuclei[1:])] + [len(chars)-1]
    phrase = MELODIES[melody]
    notes = []
    for j, nucleus in enumerate(nuclei):
        left, right = boundaries[j:j+2]
        beats = 1.7 if j == len(nuclei)-1 else (0.8 if j % 2 else 1.)
        frames = round(40 * 60 / tempo * beats)
        other = int(durations[left:right].sum() - durations[nucleus])
        durations[nucleus] = max(6, frames - other)
        midi = phrase[j % len(phrase)] + transpose
        if j == len(nuclei)-1: midi = (60 if line % 2 else 64) + transpose
        notes.append({'midi': midi, 'nucleus': nucleus, 'left': left, 'right': right})
    offsets = np.r_[0, np.cumsum(durations)]
    # F0 has twice the phoneme-alignment frame rate (80 Hz vs 40 Hz).
    pitch = np.full(int(offsets[-1])*2, float(notes[0]['midi']))
    previous = notes[0]['midi']
    for note in notes:
        a, b = int(offsets[note['left']])*2, int(offsets[note['right']])*2
        t = np.arange(b-a)/80
        glide = np.clip(t/.055, 0, 1)
        vibrato = .13*np.sin(2*np.pi*5.1*t)*np.clip((t-.18)/.2, 0, 1)
        pitch[a:b] = previous + (note['midi']-previous)*glide + vibrato
        note.update(start=round(a/80, 4), end=round(b/80, 4),
                    vowelStart=round(float(offsets[note['nucleus']])/40, 4),
                    vowelEnd=round(float(offsets[note['nucleus']+1])/40, 4))
        previous = note['midi']
    pitch[int(offsets[-2])*2:] = previous
    return durations, (440*2**((pitch-69)/12)).astype(np.float32), notes


def render_line(model, pipeline, line, voice_pack, request, line_number):
    import mlx.core as mx
    _, tokens = pipeline.g2p(line)
    segments = list(pipeline.en_tokenize(tokens))
    if len(segments) != 1: raise ValueError('Shorten this lyric line.')
    phonemes = ''.join(p for p in segments[0][1] if p in model.vocab)
    if not phonemes or len(phonemes) > 180: raise ValueError('Shorten this lyric line.')
    ref = voice_pack[len(phonemes)-1]
    ids = mx.array([[0, *[model.vocab[p] for p in phonemes], 0]])
    lengths = mx.array([ids.shape[1]])
    mask = mx.zeros(ids.shape, dtype=mx.bool_)
    bert, _ = model.bert(ids, attention_mask=(~mask).astype(mx.int32))
    d_en = model.bert_encoder(bert).transpose(0,2,1)
    style = ref[:,128:]
    encoded = model.predictor.text_encoder(d_en, style, lengths, mask)
    x, _ = model.predictor.lstm(encoded)
    natural = mx.clip(mx.round(mx.sigmoid(model.predictor.duration_proj(x)).sum(-1)),1,100)[0]
    mx.eval(natural)
    natural = np.array(natural).astype(int)
    durations, target, notes = plan_score(phonemes, natural, request['tempo'],
        request['melody'], -12 if request['speaker'][1]=='m' else 0, line_number)
    # Predict articulation at its trained speech timing, then stretch its controls.
    # Predicting prosody at long held timings can distort articulation.
    original_indices = mx.array(np.repeat(np.arange(len(natural)),natural))
    original_alignment = mx.zeros((ids.shape[1],len(original_indices)))
    original_alignment[original_indices,mx.arange(len(original_indices))] = 1
    original_en = encoded.transpose(0,2,1) @ original_alignment[None,:]
    original_f0, original_noise = model.predictor.F0Ntrain(original_en,style)
    mx.eval(original_f0,original_noise)
    edges = np.r_[0,np.cumsum(natural)]*2
    warp = np.concatenate([np.linspace(edges[i],edges[i+1]-1,int(n)*2) for i,n in enumerate(durations)])
    axis = np.arange(original_f0.shape[-1])
    f0 = mx.array(np.interp(warp,axis,np.array(original_f0)[0]).astype(np.float32))[None,:]
    noise = mx.array(np.interp(warp,axis,np.array(original_noise)[0]).astype(np.float32))[None,:]
    indices = mx.array(np.repeat(np.arange(len(durations)), durations))
    alignment = mx.zeros((ids.shape[1],len(indices)))
    alignment[indices,mx.arange(len(indices))] = 1
    alignment = alignment[None,:]
    if f0.shape[-1] != len(target): raise RuntimeError('Unsupported Kokoro frame layout')
    # Leave unvoiced consonants unvoiced. The same neural decoder creates the waveform.
    sung_f0 = mx.where(f0>55, mx.array(target)[None,:], f0)
    text = model.text_encoder(ids,lengths,mask) @ alignment
    audio = model.decoder(text,sung_f0,noise,ref[:,:128])[0]
    mx.eval(audio)
    samples = np.array(audio).reshape(-1).astype(np.float32)
    return samples, {'text':line,'phonemes':phonemes,'notes':notes,'duration':len(samples)/24000}


def generate(request, model=None):
    import mlx.core as mx
    from mlx_audio.tts.utils import load_model
    from scipy.signal import butter, sosfilt
    request = validate_request(request)
    mx.set_cache_limit(128*1024*1024); mx.random.seed(42)
    if model is None: model = load_model(ROOT/'models/expressive/Kokoro-82M-bf16')
    pipeline = model._get_pipeline(request['speaker'][0])
    pack = pipeline.load_voice(request['speaker'])
    chunks=[]; score=[]; offset=.12
    chunks.append(np.zeros(round(offset*24000),np.float32))
    for n,line in enumerate(request['lyrics'].splitlines()):
        samples, phrase = render_line(model,pipeline,line,pack,request,n)
        fade=min(240,len(samples)//2)
        samples[:fade]*=np.linspace(0,1,fade);samples[-fade:]*=np.linspace(1,0,fade)
        phrase['offset']=round(offset,4);score.append(phrase)
        chunks.extend([samples,np.zeros(6000,np.float32)])
        offset += len(samples)/24000+.25
        mx.clear_cache()
    audio = sosfilt(butter(2,55,fs=24000,btype='highpass',output='sos'),np.concatenate(chunks)).astype(np.float32)
    rms=float(np.sqrt(np.mean(audio*audio)));peak=float(np.abs(audio).max())
    if peak>1e-6:audio*=min(.09/max(rms,.001),.87/peak,2.5)
    if not np.isfinite(audio).all():raise RuntimeError('Invalid Kokoro song output')
    return audio,score


if __name__=='__main__':
    os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
    os.environ['TOKENIZERS_PARALLELISM']='false'
    path=Path(sys.argv[1]);started=time.perf_counter()
    samples,score=generate(json.loads(path.read_text()))
    output=path.with_suffix('.wav')
    with wave.open(str(output),'wb') as f:
        f.setparams((1,2,24000,0,'NONE','not compressed'))
        f.writeframes((np.clip(samples,-1,1)*32767).astype('<i2').tobytes())
    result={'path':str(output),'engine':'kokoro-singing','score':score,
            'synthMs':round((time.perf_counter()-started)*1000),'duration':len(samples)/24000}
    path.with_suffix('.result.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='score'}),flush=True)
