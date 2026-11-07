"""Build explicit, pre-rendered research auditions with avatar mouth frames."""
import base64,io,json,shutil,sys,wave
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from audio_analysis import analyze
folder=ROOT/'public/audio/research';folder.mkdir(parents=True,exist_ok=True)
sources={
 'heart':('evidence/voice-fixtures/af_heart-heldout.wav','This is my new English voice. I hope it feels a little more like me.','speech','Kokoro Heart'),
 'singer-a':('evidence/soulx/kokoro-reference-mps-16steps-cfg2.wav','You light up my day. Stay here close to me.','sing','SoulX-Singer; Kokoro speech reference'),
 'singer-b':('evidence/soulx/kokoro-sung-reference-mps-32steps-cfg3.wav','You light up my day. Stay here close to me.','sing','SoulX-Singer; experimental Kokoro sung reference'),
}
manifest={'notice':'Prepared research samples, not live singing generation. Voice resemblance and naturalness need listening review.','samples':{}}
for name,(source,text,mode,engine) in sources.items():
 with wave.open(str(ROOT/source),'rb') as f:sr=f.getframerate();audio=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768
 rms=float(np.sqrt(np.mean(audio*audio)));peak=float(np.max(np.abs(audio)));audio*=min(.06/max(.001,rms),.85/max(.001,peak),3.)
 data=io.BytesIO()
 with wave.open(data,'wb') as f:f.setparams((1,2,sr,0,'NONE','not compressed'));f.writeframes((audio*32767).astype('<i2').tobytes())
 packed={'audio':base64.b64encode(data.getvalue()).decode(),'sampleRate':sr,'duration':len(audio)/sr,'frames':analyze(audio,sr),'synthMs':0,'analysisMs':0,'engine':engine,'prerendered':True}
 (folder/(name+'.json')).write_text(json.dumps(packed));(folder/(name+'.wav')).write_bytes(data.getvalue())
 manifest['samples'][name]={'text':text,'mode':mode,'url':'/audio/research/'+name+'.json','source':source,'engine':engine}
reference=ROOT/'evidence/voice-fixtures/af_heart-reference'
shutil.copy2(reference.with_suffix('.wav'),folder/'heart-reference.wav')
manifest['reference']={'url':'/audio/research/heart-reference.wav','text':json.loads(reference.with_suffix('.json').read_text())['text'],'kind':'Synthetic Kokoro Heart reference; different words from live target'}
(folder/'manifest.json').write_text(json.dumps(manifest,indent=2))
print('Packaged',list(manifest['samples']))
