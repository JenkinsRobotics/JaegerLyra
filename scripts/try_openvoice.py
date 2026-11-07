"""Source-derived OpenVoice V2 experiment using only synthetic Kokoro references.

No training or per-speaker fine-tuning. This is NOT a clean-room implementation.
Run with .research-venv/bin/python.
"""
import json,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'models/hf-research')
os.environ['TORCH_HOME']=str(ROOT/'models/torch-research')
sys.path.insert(0,str(ROOT/'voice-tools/openvoice'))
import numpy as np
import torch
import soundfile as sf
from openvoice.api import ToneColorConverter

def main():
 torch.set_num_threads(4);torch.manual_seed(42)
 output=ROOT/'evidence/openvoice';output.mkdir(exist_ok=True)
 fixtures=ROOT/'evidence/voice-fixtures';start=time.perf_counter()
 converter=ToneColorConverter(str(ROOT/'models/openvoice-v2/converter/config.json'),device='cpu')
 converter.load_ckpt(str(ROOT/'models/openvoice-v2/converter/checkpoint.pth'))
 report={'code':'74a1d147b17a8c3092dd5430504bd83ef6c7eb23','weights':'myshell-ai/OpenVoiceV2@f36e7edfe1684461a8343844af60babc2efbb727','device':'cpu','loadSeconds':round(time.perf_counter()-start,3),'trials':[]}
 source=converter.extract_se(str(fixtures/'af_heart-reference.wav'))
 for target in ['af_heart','am_michael']:
  start=time.perf_counter();embedding=converter.extract_se(str(fixtures/(target+'-reference.wav')))
  embedding_seconds=time.perf_counter()-start
  for tau in ([.3] if target=='af_heart' else [.3,.1]):
   start=time.perf_counter();audio=converter.convert(str(fixtures/'af_heart-heldout.wav'),src_se=source,tgt_se=embedding,tau=tau,message='KIN1')
   name=f'heart-to-{target}-tau{tau}';sf.write(output/(name+'.wav'),audio,converter.hps.data.sampling_rate)
   trial={'name':name,'reference':target+'-reference.wav','text':json.loads((fixtures/'af_heart-heldout.json').read_text())['text'],'seconds':round(time.perf_counter()-start,3),'referenceSeconds':round(embedding_seconds,3),'duration':len(audio)/converter.hps.data.sampling_rate,'peak':float(np.max(np.abs(audio))),'finite':bool(np.isfinite(audio).all()),'watermark':'KIN1 (32 bits, one block)'}
   report['trials'].append(trial);print(json.dumps(trial),flush=True)
 # Ablation: reference-matched pitch followed by learned timbre conversion.
 # Pitch-only fixture uses the reference pair, never the held-out target audio.
 shifted=ROOT/'evidence/voice-identity/pitch-only.wav'
 if shifted.exists():
  shifted_source=converter.extract_se(str(shifted));target=converter.extract_se(str(fixtures/'am_michael-reference.wav'))
  start=time.perf_counter();audio=converter.convert(str(shifted),src_se=shifted_source,tgt_se=target,tau=.1,message='KIN1')
  sf.write(output/'pitch-plus-openvoice-michael.wav',audio,converter.hps.data.sampling_rate)
  report['trials'].append({'name':'pitch-plus-openvoice-michael','seconds':round(time.perf_counter()-start,3),'source':'WORLD pitch-only from reference-derived ratio','target':'am_michael-reference.wav','watermark':'KIN1'})
 # Singing-to-timbre conversion is a separate research question. Keep note
 # trajectory and lyric content while trying to move toward Heart's identity.
 heart=converter.extract_se(str(fixtures/'af_heart-reference.wav'))
 for name in ['kokoro-reference-mps-8steps-cfg3','kokoro-reference-mps-32steps-cfg3']:
  song=ROOT/'evidence/soulx'/(name+'.wav')
  if not song.exists():continue
  song_source=converter.extract_se(str(song));start=time.perf_counter()
  audio=converter.convert(str(song),src_se=song_source,tgt_se=heart,tau=.1,message='KIN1')
  sf.write(output/(name+'-heart-transfer.wav'),audio,converter.hps.data.sampling_rate)
  trial={'name':name+'-heart-transfer','seconds':round(time.perf_counter()-start,3),'source':str(song),'target':'af_heart-reference.wav','watermark':'KIN1'};report['trials'].append(trial);print(json.dumps(trial),flush=True)
 (output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
