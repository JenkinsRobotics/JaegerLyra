"""Check that retaining Kokoro's voice pack/cache changes speed, not samples."""
import json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HUB_OFFLINE']='1';os.environ['HF_HUB_CACHE']=str(ROOT/'models/hf-cache')
import mlx.core as mx
import numpy as np
from mlx_audio.tts.utils import load_model
mx.set_cache_limit(128*1024*1024)
model=load_model(ROOT/'models/expressive/Kokoro-82M-bf16');pipeline=model._get_pipeline('a')
texts=['Hello, the little lights are on.','I am glad we have this quiet moment together.','Can you hear the rain tapping gently on the window?']
def run(text,direct):
 mx.random.seed(42);start=time.perf_counter()
 results=pipeline(text,voice='af_heart',speed=1.) if direct else model.generate(text=text,voice='af_heart',lang_code='a',speed=1.)
 audio=np.concatenate([np.asarray(result.audio).reshape(-1) for result in results]);elapsed=time.perf_counter()-start
 return audio,elapsed
run('Hello, I am ready.',False)
rows=[]
for text in texts:
 old,old_time=run(text,False);new,new_time=run(text,True)
 rows.append({'text':text,'wrapperSeconds':round(old_time,4),'retainedPipelineSeconds':round(new_time,4),'samples':len(old),'exactSampleMatch':bool(np.array_equal(old,new)),'maximumAbsoluteDifference':float(np.max(np.abs(old-new))) if old.shape==new.shape else None})
print(json.dumps(rows,indent=2));(ROOT/'evidence/kokoro-pipeline-comparison.json').write_text(json.dumps(rows,indent=2))
