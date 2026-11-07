"""Compare Chatterbox's complete and incremental waveform paths, silently.

Uses a synthetic Kokoro Michael reference, with its text held out from the
generated sentence. Conditionals are encoded once, reused without training.
"""
import json,os,time,wave
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_CACHE']=str(ROOT/'models/hf-cache')
import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model
mx.set_cache_limit(128*1024*1024)

def main():
 folder=ROOT/'evidence/clone-streaming';folder.mkdir(exist_ok=True)
 start=time.perf_counter();model=load_model(ROOT/'models/expressive/chatterbox-turbo-8bit')
 path=ROOT/'models/hf-cache/models--mlx-community--S3TokenizerV2/snapshots/e0c9886f0e1c35ae85b1f27277416fb19fc72bec/model.safetensors'
 tokenizer=model._s3tokenizer;weights=mx.load(str(path))
 if hasattr(tokenizer,'sanitize'):weights=tokenizer.sanitize(weights)
 tokenizer.load_weights(list(weights.items()),strict=False);mx.eval(tokenizer.parameters())
 load=time.perf_counter()-start;start=time.perf_counter()
 model.prepare_conditionals(str(ROOT/'evidence/voice-fixtures/am_michael-reference.wav'),exaggeration=0.,norm_loudness=True)
 preparation=time.perf_counter()-start;rows=[]
 text='This is my new English voice. I hope it feels a little more like me.'
 for interval in [None,.64,1.28]:
  mx.random.seed(42);start=time.perf_counter();chunks=[];parts=[]
  for result in model.generate(text=text,ref_audio=None,temperature=.65,exaggeration=0.,cfg_weight=0.,max_tokens=300,split_pattern=None,stream=interval is not None,streaming_interval=interval or 2.):
   samples=np.asarray(result.audio).reshape(-1).astype(np.float32);parts.append(samples);sr=result.sample_rate
   chunks.append({'readyMs':round((time.perf_counter()-start)*1000,1),'duration':len(samples)/sr})
   print('chunk',interval,chunks[-1],flush=True)
  audio=np.concatenate(parts);name='full' if interval is None else 'stream-'+str(interval)
  with wave.open(str(folder/(name+'.wav')),'wb') as f:
   f.setparams((1,2,sr,0,'NONE','not compressed'));f.writeframes((np.clip(audio,-1,1)*32767).astype('<i2').tobytes())
  row={'name':name,'text':text,'reference':'Kokoro Michael, synthetic, different text','loadSeconds':round(load,3),'referenceEncodingSeconds':round(preparation,3),'firstChunkMs':chunks[0]['readyMs'],'totalMs':chunks[-1]['readyMs'],'duration':len(audio)/sr,'peak':float(np.max(np.abs(audio))),'chunks':chunks}
  # Minimum initial buffering needed so every later chunk arrives by playback.
  prior=0.;required=0.
  for chunk in chunks:required=max(required,chunk['readyMs']/1000-prior);prior+=chunk['duration']
  row['minimumStartBufferSeconds']=round(required,3)
  rows.append(row);(folder/'report.json').write_text(json.dumps(rows,indent=2));print(json.dumps(row),flush=True)

if __name__=='__main__':main()
