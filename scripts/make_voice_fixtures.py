"""Original, synthetic Kokoro fixtures with model-predicted word alignment.

Research fixtures only: no person's uploaded voice or third-party song recording.
Run with .voice-venv/bin/python.
"""
import json,os,sys,time,wave
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_CACHE']=str(ROOT/'models/hf-cache')
import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model

def main():
 folder=ROOT/'evidence/voice-fixtures';folder.mkdir(exist_ok=True)
 model=load_model(ROOT/'models/expressive/Kokoro-82M-bf16');pipeline=model._get_pipeline('a')
 texts={
  'reference':'You light up my day. Stay here close to me. We can watch the stars shine in the sky.',
  'heldout':'This is my new English voice. I hope it feels a little more like me.',
 }
 for voice in ['af_heart','am_michael']:
  for kind,text in texts.items():
   start=time.perf_counter();parts=[];words=[];offset=0.;mx.random.seed(42)
   for result in pipeline(text,voice=voice,speed=1.):
    audio=np.asarray(result.audio).reshape(-1);parts.append(audio)
    for token in result.tokens or []:
     words.append({'text':token.text,'phonemes':token.phonemes,'start':None if token.start_ts is None else offset+token.start_ts,'end':None if token.end_ts is None else offset+token.end_ts})
    offset+=len(audio)/24000
   audio=np.concatenate(parts);name=voice+'-'+kind
   with wave.open(str(folder/(name+'.wav')),'wb') as wav:
    wav.setparams((1,2,24000,0,'NONE','not compressed'));wav.writeframes((np.clip(audio,-1,1)*32767).astype('<i2').tobytes())
   meta={'speaker':voice,'text':text,'duration':len(audio)/24000,'sampleRate':24000,'elapsed':round(time.perf_counter()-start,3),'words':words,'origin':'Kokoro synthetic reference; original text; source-model predicted word timing'}
   (folder/(name+'.json')).write_text(json.dumps(meta,indent=2));print(name,meta['duration'],meta['elapsed'],flush=True)

if __name__=='__main__':main()
