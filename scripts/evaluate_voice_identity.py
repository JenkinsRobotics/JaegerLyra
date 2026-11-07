"""Pitch-only ablation and independent speaker-embedding comparison.

All references are synthetic Kokoro. ECAPA scores are a relative diagnostic,
not a probability, a naturalness score, or a validated singing metric.
"""
import json,os,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['HF_HOME']=str(ROOT/'models/hf-research')
import numpy as np
import pyworld
import soundfile as sf
import torch,torchaudio
from speechbrain.inference.classifiers import EncoderClassifier

def analyze(path):
 audio,sr=sf.read(path);audio=np.asarray(audio,dtype=np.float64)
 f0,times=pyworld.harvest(audio,sr,f0_floor=60,f0_ceil=600,frame_period=5.)
 f0=pyworld.stonemask(audio,f0,times,sr)
 return audio,sr,f0,times

def main():
 torch.set_num_threads(4)
 fixtures=ROOT/'evidence/voice-fixtures';output=ROOT/'evidence/voice-identity';output.mkdir(exist_ok=True)
 source,sr,f0,times=analyze(fixtures/'af_heart-heldout.wav')
 _,_,heart_f0,_=analyze(fixtures/'af_heart-reference.wav');_,_,michael_f0,_=analyze(fixtures/'am_michael-reference.wav')
 ratio=float(np.median(michael_f0[michael_f0>0])/np.median(heart_f0[heart_f0>0]))
 spectrum=pyworld.cheaptrick(source,f0,times,sr);aperiodicity=pyworld.d4c(source,f0,times,sr)
 for label,pitch_ratio,formant_ratio in [('world-self',1.,1.),('pitch-only',ratio,1.),('pitch-and-formant',ratio,.85)]:
  envelope=spectrum if formant_ratio==1 else np.asarray([np.interp(np.arange(len(frame))/formant_ratio,np.arange(len(frame)),frame,right=1e-12) for frame in spectrum])
  audio=pyworld.synthesize(np.ascontiguousarray(f0*pitch_ratio),np.ascontiguousarray(envelope),aperiodicity,sr,frame_period=5.)[:len(source)]
  peak=np.max(np.abs(audio));audio*=min(1.,.9/max(peak,1e-8))
  sf.write(output/(label+'.wav'),audio,sr)
 model_dir=ROOT/'models/speaker-eval'
 classifier=EncoderClassifier.from_hparams(source=str(model_dir),savedir=str(model_dir),overrides={'pretrained_path':str(model_dir)},run_opts={'device':'cpu'})
 def embedding(path):
  audio,sr=sf.read(path);samples=torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
  if sr!=16000:samples=torchaudio.functional.resample(samples,sr,16000)
  with torch.inference_mode():result=classifier.encode_batch(samples).flatten()
  return result/torch.linalg.vector_norm(result)
 anchors={name:embedding(fixtures/(name+'.wav')) for name in ['af_heart-reference','am_michael-reference','af_heart-heldout','am_michael-heldout']}
 paths=list(output.glob('*.wav'))+list((ROOT/'evidence/openvoice').glob('*.wav'))+list((ROOT/'evidence/soulx').glob('*.wav'))
 paths += [p for p in (ROOT/'evidence/clone-streaming').glob('*.wav') if 'cancel' not in p.name and 'recovery' not in p.name]
 paths += [fixtures/'af_heart-heldout.wav',fixtures/'am_michael-heldout.wav']
 rows=[]
 for path in paths:
  item=embedding(path);row={'file':str(path.relative_to(ROOT)),'cosine':{name:round(float(torch.dot(item,anchor)),4) for name,anchor in anchors.items()}}
  rows.append(row);print(json.dumps(row),flush=True)
 report={'metric':'ECAPA-TDNN cosine similarity (speechbrain/spkrec-ecapa-voxceleb); relative evidence only, no pass threshold. Speech-trained and not validated here for singing. Synthetic speakers and one held-out sentence are insufficient for general cloning claims.',
         'pitchRatioFromReferences':ratio,'pitchSemitones':float(12*np.log2(ratio)),
         'formantBaseline':'Fixed 0.85 spectral-envelope frequency scaling; not fitted to held-out audio.',
         'rows':rows}
 (output/'report.json').write_text(json.dumps(report,indent=2))

if __name__=='__main__':main()
