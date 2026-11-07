"""SoulX-Singer research adapter: original score + synthetic Kokoro reference.

Uses upstream code and weights; not a clean-room model. No per-voice training.
Run with .research-venv/bin/python scripts/try_soulx.py --steps 8 --device mps
"""
import argparse,copy,json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'voice-tools/soulx-singer'
os.environ['PYTORCH_ENABLE_MPS_FALLBACK']='1'
os.environ['HF_HOME']=str(ROOT/'models/hf-research')
os.environ['NLTK_DATA']=str(ROOT/'.cache/nltk')
sys.path.insert(0,str(SOURCE))
sys.path.insert(0,str(ROOT))
import numpy as np
import soundfile as sf
import pyworld
import torch
from omegaconf import OmegaConf
from cli.inference import build_model
from soulxsinger.utils.data_processor import DataProcessor

PHONES={'you':'Y-UW1','light':'L-AY1-T','up':'AH1-P','my':'M-AY1','day':'D-EY1',
        'stay':'S-T-EY1','here':'HH-IY1-R','close':'K-L-OW1-S','to':'T-UW1','me':'M-IY1',
        'we':'W-IY1','can':'K-AE1-N','watch':'W-AA1-CH','the':'DH-AH0',
        'stars':'S-T-AA1-R-Z','shine':'SH-AY1-N','in':'IH0-N','sky':'S-K-AY1'}

def metadata(items):
 duration=sum(item[1] for item in items)
 return {'index':0,'language':'English','time':[0,round(duration*1000)],
         'text':' '.join(item[0] for item in items),
         'phoneme':' '.join('<SP>' if item[0]=='<SP>' else 'en_'+PHONES[item[0].lower()] for item in items),
         'duration':' '.join(f'{item[1]:.5f}' for item in items),
         'note_pitch':' '.join(str(item[2]) for item in items),
         'note_type':' '.join('1' if item[0]=='<SP>' else '2' for item in items)}

def make_prompt(sung=False,fixture=None):
 if sung:return make_sung_prompt()
 fixture=fixture or ROOT/'evidence/voice-fixtures/af_heart-reference'
 data=json.loads(fixture.with_suffix('.json').read_text());audio,sr=sf.read(fixture.with_suffix('.wav'))
 f0,times=pyworld.harvest(audio.astype(np.float64),sr,f0_floor=70,f0_ceil=600,frame_period=20)
 f0=pyworld.stonemask(audio.astype(np.float64),f0,times,sr)
 items=[];previous=0
 for word in data['words']:
  if word['start'] is None:continue
  start,end=word['start'],word['end'];text=word['text'].lower()
  if start>previous+.001:items.append(('<SP>',start-previous,0))
  if text not in PHONES:items.append(('<SP>',end-start,0))
  else:
   pitch=f0[(times>=start)&(times<end)&(f0>0)]
   note=int(round(69+12*np.log2(np.median(pitch)/440))) if len(pitch) else 60
   items.append((text,end-start,note))
  previous=end
 if data['duration']>previous:items.append(('<SP>',data['duration']-previous,0))
 meta=metadata(items);meta['f0']=' '.join(f'{pitch:.2f}' for pitch in f0[:int(data['duration']*50)])
 return meta,fixture.with_suffix('.wav')

def make_sung_prompt(fixture=None):
 fixture=fixture or ROOT/'evidence/voice-fixtures/heart-sung-reference'
 result=json.loads(fixture.with_suffix('.result.json').read_text())
 audio,sr=sf.read(fixture.with_suffix('.wav'))
 f0,times=pyworld.harvest(audio.astype(np.float64),sr,f0_floor=70,f0_ceil=800,frame_period=20)
 f0=pyworld.stonemask(audio.astype(np.float64),f0,times,sr)
 items=[];previous=0
 for phrase in result['score']:
  words=phrase['text'].split()
  if len(words)!=len(phrase['notes']):raise ValueError('This fixture requires one-syllable words')
  for word,note in zip(words,phrase['notes']):
   start,end=phrase['offset']+note['start'],phrase['offset']+note['end']
   if start>previous+.001:items.append(('<SP>',start-previous,0))
   pitch=f0[(times>=start)&(times<end)&(f0>0)]
   midi=int(round(69+12*np.log2(np.median(pitch)/440))) if len(pitch) else note['midi']
   items.append((word.lower(),end-start,midi));previous=end
 if len(audio)/sr>previous:items.append(('<SP>',len(audio)/sr-previous,0))
 meta=metadata(items);meta['f0']=' '.join(f'{pitch:.2f}' for pitch in f0[:int(len(audio)/sr*50)])
 return meta,fixture.with_suffix('.wav')

def main():
 parser=argparse.ArgumentParser();parser.add_argument('--device',default='mps');parser.add_argument('--steps',type=int,default=8);parser.add_argument('--cfg',type=float,default=3.0);parser.add_argument('--sweep',action='store_true');parser.add_argument('--sung-reference',action='store_true');parser.add_argument('--score-request',type=Path);args=parser.parse_args()
 torch.set_num_threads(4);torch.manual_seed(42)
 folder=ROOT/'evidence/soulx';folder.mkdir(exist_ok=True)
 prompt,reference=make_prompt(args.sung_reference)
 target=metadata([('<SP>',.2,0),('You',.45,60),('light',.5,62),('up',.35,64),('my',.4,67),('day',.9,65),('<SP>',.25,0),('Stay',.55,64),('here',.45,62),('close',.45,60),('to',.35,62),('me',.95,60),('<SP>',.3,0)])
 if args.score_request:
  from english_singing_score import plan_english
  from g2p_en import G2p
  request=json.loads(args.score_request.read_text());target=plan_english(request['lyrics'],G2p(),tempo=request.get('tempo',104),melody=request.get('melody','glow'))
 (folder/('prompt-sung.json' if args.sung_reference else 'prompt.json')).write_text(json.dumps([prompt],indent=2));(folder/('target-general.json' if args.score_request else 'target.json')).write_text(json.dumps([target],indent=2))
 config=OmegaConf.load(ROOT/'models/soulx-singer/config.yaml');start=time.perf_counter()
 model=build_model(str(ROOT/'models/soulx-singer/model.pt'),config,device=args.device)
 load_seconds=time.perf_counter()-start;print('load_seconds',load_seconds,flush=True)
 processor=DataProcessor(config.audio.hop_size,config.audio.sample_rate,str(SOURCE/'soulxsinger/utils/phoneme/phone_set.json'),device=args.device)
 data={'prompt':processor.process(copy.deepcopy(prompt),str(reference)),'target':processor.process(copy.deepcopy(target),None)}
 for steps,cfg in ([(16,2.),(32,2.),(32,3.)] if args.sweep else [(args.steps,args.cfg)]):
  torch.manual_seed(42);start=time.perf_counter()
  with torch.inference_mode():audio=model.infer(data,auto_shift=False,pitch_shift=0,n_steps=steps,cfg=cfg,control='score').squeeze().cpu().numpy()
  elapsed=time.perf_counter()-start
  if not np.isfinite(audio).all():raise ValueError('Non-finite model audio')
  kind='sung' if args.sung_reference else 'speech'
  name=f'kokoro-{"sung-" if args.sung_reference else ""}reference-{args.device}-{steps}steps-cfg{cfg:g}'
  if args.score_request:name='general-'+args.score_request.stem+'-'+name
  sf.write(folder/(name+'.wav'),audio,24000)
  report={'name':name,'code':'81aeb3ae772c70093c3de74dc23c92d983801ae4','weights':'Soul-AILab/SoulX-Singer@40493ad90286056c7a9095035164434a79daa8c9','device':args.device,'steps':steps,'cfg':cfg,'reference':str(reference),'referenceKind':'Synthetic Kokoro '+kind+' with model alignment','lyrics':target.get('lyrics','You light up my day. Stay here close to me.'),'score':target if args.score_request else None,'loadSeconds':round(load_seconds,3),'synthSeconds':round(elapsed,3),'duration':len(audio)/24000,'peak':float(np.max(np.abs(audio))),'status':'Awaiting lyric, pitch and perceptual evaluation'}
  (folder/(name+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)

if __name__=='__main__':main()
