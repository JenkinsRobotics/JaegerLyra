"""Local English zero-shot audition; separate from KIN's default Kokoro voice."""
import os,sys,json,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];SOURCE=ROOT/'voice-tools/gpt-sovits'
os.environ['NLTK_DATA']=str(ROOT/'.cache/nltk');Path(os.environ['NLTK_DATA']).mkdir(parents=True,exist_ok=True)
os.environ['HF_HOME']=str(ROOT/'models/sovits-hf-cache');os.environ['TOKENIZERS_PARALLELISM']='false';os.environ['language']='en_US'
os.chdir(SOURCE);sys.path[:0]=[str(SOURCE),str(SOURCE/'GPT_SoVITS')]
import torch,numpy as np,soundfile as sf
# Language resources are installed once by setup_sovits.py.
os.environ['HF_HUB_OFFLINE']='1'
os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
torch.set_num_threads(4)
from TTS_infer_pack.TTS import TTS,TTS_Config
base=SOURCE/'GPT_SoVITS/pretrained_models'
config={'device':'cpu','is_half':False,'version':'v2ProPlus','t2s_weights_path':str(base/'s1v3.ckpt'),'vits_weights_path':str(base/'v2Pro/s2Gv2ProPlus.pth'),'bert_base_path':str(base/'chinese-roberta-wwm-ext-large'),'cnhuhbert_base_path':str(base/'chinese-hubert-base')}
start=time.perf_counter();tts=TTS(TTS_Config({'custom':config}));loaded=time.perf_counter()
request={'text':'This is my new English voice. I hope it feels a little more like me.','text_lang':'en','ref_audio_path':str(ROOT/'evidence/kokoro-af_heart.wav'),'prompt_text':"Hi, I'm Vita. It's good to see you. Let's make a little magic together, one expression at a time.",'prompt_lang':'en','seed':42,'parallel_infer':True,'batch_size':1,'text_split_method':'cut0','speed_factor':1.0,'use_cuda_graph':False,'use_flash_attention':False}
parts=[]
for rate,samples in tts.run(request):parts.append(samples)
audio=np.concatenate(parts);sf.write(ROOT/'evidence/gpt-sovits-english-match.wav',audio,rate)
result={'model':'GPT-SoVITS v2ProPlus','device':'cpu','duration':len(audio)/rate,'loadMs':round((loaded-start)*1000),'synthMs':round((time.perf_counter()-loaded)*1000),'text':request['text']}
(ROOT/'evidence/gpt-sovits-english-match.json').write_text(json.dumps(result,indent=2));print(json.dumps(result),flush=True)
