"""MLX voice process. Models stay loaded; JSON on stdout, diagnostics on stderr."""
import base64, contextlib, io, json, os, sys, time, wave, signal
from collections import OrderedDict
from pathlib import Path
import numpy as np
os.environ['HF_HUB_CACHE']=str(Path(__file__).resolve().parent/'models/hf-cache')
import mlx.core as mx
mx.set_cache_limit(128*1024*1024)
from mlx_audio.tts.utils import load_model
from scipy.signal import butter, sosfilt

ROOT=Path(__file__).resolve().parent
MODELS={'expressive':'Kokoro-82M-bf16','clone':'chatterbox-turbo-8bit'}
loaded={};conditionals=OrderedDict();CANCELLED=False;PROTOCOL=sys.stdout
def cancel_signal(*_):
 global CANCELLED
 CANCELLED=True
if hasattr(signal,'SIGUSR1'):signal.signal(signal.SIGUSR1,cancel_signal)
def reference_path(data):
 reference=ROOT/'.cache/voice-references'/str(data.get('reference',''))
 if reference.parent!=ROOT/'.cache/voice-references' or not reference.is_file():raise ValueError('Voice reference not found')
 return str(reference)
def load_engine(engine):
 if engine not in MODELS:raise ValueError('Unknown voice engine')
 if engine not in loaded:
  loaded[engine]=load_model(ROOT/'models/expressive'/MODELS[engine])
  if engine=='clone':
   path=ROOT/'models/hf-cache/models--mlx-community--S3TokenizerV2/snapshots/e0c9886f0e1c35ae85b1f27277416fb19fc72bec/model.safetensors'
   if not path.is_file():raise ValueError('The local speech tokenizer is missing; run scripts/setup_voice.py.')
   tokenizer=loaded[engine]._s3tokenizer;weights=mx.load(str(path))
   if hasattr(tokenizer,'sanitize'):weights=tokenizer.sanitize(weights)
   tokenizer.load_weights(list(weights.items()),strict=False);mx.eval(tokenizer.parameters())
 return loaded[engine]
def prepare_clone(model,data):
 reference=reference_path(data)
 if reference not in conditionals:
  model.prepare_conditionals(reference,exaggeration=0.,norm_loudness=True)
  conditionals[reference]=model._conds
  if len(conditionals)>3:conditionals.popitem(last=False)
 else:conditionals.move_to_end(reference);model._conds=conditionals[reference]
def encode(samples,sr):
 wav=io.BytesIO()
 with wave.open(wav,'wb') as f:f.setparams((1,2,sr,0,'NONE','not compressed'));f.writeframes((np.clip(samples,-1,1)*32767).astype('<i2').tobytes())
 return base64.b64encode(wav.getvalue()).decode()
def generate(data):
 engine=data.get('engine','expressive');start=time.perf_counter()
 with contextlib.redirect_stdout(sys.stderr):
  model=load_engine(engine);mx.random.seed(42)
  text=str(data['text'])[:700]
  emotion=data.get('emotion','neutral')
  if engine=='clone':
   prepare_clone(model,data);mx.random.seed(42)
   tag={'laughing':'[chuckle] ','surprised':'[gasp] ','sad':'[sigh] '}.get(emotion,'')
   results=model.generate(text=tag+text,ref_audio=None,temperature=.65,exaggeration=0.,cfg_weight=0.,max_tokens=min(750,max(180,len(text)*3)),verbose=False)
  else:
   speaker=data.get('speaker','af_heart')
   if speaker not in ('af_heart','af_bella','af_nicole','am_michael','bf_emma','bm_george'):speaker='af_heart'
   speed={'joy':1.02,'excited':1.08,'sad':.92,'thinking':.95,'shy':.94,'soft':.92,'bright':1.06}.get(emotion,1.)
   # Retain the voice pack and bounded MLX cache between conversational clauses.
   # mlx-audio's convenience generate() resets both on every call. This calls
   # the same Kokoro pipeline with the same voice/style and acoustic weights.
   results=model._get_pipeline(speaker[0])(text,voice=speaker,speed=speed)
  parts=[];sr=24000
  for result in results:parts.append(np.array(result.audio).reshape(-1));sr=getattr(result,'sample_rate',24000)
 if not parts:raise ValueError('The voice produced no audio')
 samples=np.concatenate(parts).astype(np.float32)
 if not np.isfinite(samples).all():raise ValueError('Invalid audio samples')
 # Remove DC/rumble, equalize level, keep headroom and soften clip boundaries.
 samples=sosfilt(butter(2,60,fs=sr,btype='highpass',output='sos'),samples)
 rms=float(np.sqrt(np.mean(samples*samples)));peak=float(np.max(np.abs(samples)))
 if peak>.0001:samples*=min(2.5,.11/max(rms,.001),.89/peak)
 n=min(int(sr*.008),len(samples)//2);samples[:n]*=np.linspace(0,1,n);samples[-n:]*=np.linspace(1,0,n)
 return {'audio':encode(samples,sr),'sampleRate':sr,'duration':len(samples)/sr,'synthMs':round((time.perf_counter()-start)*1000,1),'engine':engine}

def stream_generate(data):
 if data.get('engine')!='clone':raise ValueError('Native waveform streaming is available for the reference matcher')
 start=time.perf_counter();previous=start;index=0;state=None
 with contextlib.redirect_stdout(sys.stderr):
  model=load_engine('clone');prepare_clone(model,data);mx.random.seed(42)
  text=str(data['text'])[:700]
  text={'laughing':'[chuckle] ','surprised':'[gasp] ','sad':'[sigh] '}.get(data.get('emotion'),'')+text
  results=model.generate(text=text,ref_audio=None,temperature=.65,exaggeration=0.,cfg_weight=0.,max_tokens=min(750,max(180,len(text)*3)),stream=True,streaming_interval=1.28,split_pattern=None,verbose=False)
  for result in results:
   samples=np.array(result.audio).reshape(-1).astype(np.float32);sr=result.sample_rate
   if CANCELLED:break
   if not len(samples):continue
   if not np.isfinite(samples).all():raise ValueError('Invalid stream audio')
   if state is None:
    sos=butter(2,60,fs=sr,btype='highpass',output='sos');state=np.zeros((len(sos),2))
   samples,state=sosfilt(sos,samples,zi=state)
   # Continuous filter state and fixed gain preserve chunk joins and dynamics.
   # Do not independently normalize/fade each acoustic chunk.
   if index==0:
    n=min(int(sr*.008),len(samples));samples[:n]*=np.linspace(0,1,n)
   now=time.perf_counter()
   yield {'type':'audio','audio':encode(samples,sr),'sampleRate':sr,'duration':len(samples)/sr,'synthMs':round((now-previous)*1000,1),'elapsedMs':round((now-start)*1000,1),'index':index,'engine':'clone-stream'}
   previous=now;index+=1
 if not CANCELLED and not index:raise ValueError('The matcher produced no audio')

def main():
 global CANCELLED
 for line in sys.stdin:
  CANCELLED=False
  try:
   data=json.loads(line)
   if data.get('control')=='prepare-singing':
    # Retain the small accepted speech model, release the heavier matcher and
    # its reference encodings before the separate singing process allocates.
    import gc
    conditionals.clear();loaded.pop('clone',None);gc.collect();mx.clear_cache()
    result={'type':'singing-ready','kokoroReady':'expressive' in loaded,
            'loadedEngines':list(loaded),'activeBytes':mx.get_active_memory()}
   elif data.get('stream'):
    for result in stream_generate(data):print(json.dumps(result),file=PROTOCOL,flush=True)
    result={'type':'done','cancelled':CANCELLED}
   else:result=generate(data)
  except Exception as e:
   import traceback;traceback.print_exc(file=sys.stderr);result={'error':str(e)}
  print(json.dumps(result),file=PROTOCOL,flush=True)

if __name__=='__main__':main()
