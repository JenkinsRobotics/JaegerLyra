"""Bounded local song jobs. One music render at a time, sharing the voice GPU lock."""
import atexit,base64,hashlib,io,json,os,subprocess,threading,time,wave
from pathlib import Path
import numpy as np
import expressive_voice
from kokoro_singing import validate_request
import soulx_singing
from audio_analysis import analyze
ROOT=Path(__file__).resolve().parent
FOLDER=ROOT/'.cache/songs';FOLDER.mkdir(parents=True,exist_ok=True)
JOBS={};GUARD=threading.Lock();CLOSING=threading.Event()
DEFAULT_STYLE='A cheerful anime character singing a clear intimate female solo vocal melody, gentle upbeat pop, a cappella, no drums, no instruments, clean studio vocal recording.'
def available(engine='ace-step'):
 if engine=='kokoro':return expressive_voice.available()
 if engine=='soulx':return expressive_voice.available() and all(p.is_file() for p in (ROOT/'.research-venv/bin/python',ROOT/'models/soulx-singer/model.pt',ROOT/'voice-tools/soulx-singer/cli/inference.py'))
 return (ROOT/'.music-venv/bin/python').is_file() and (ROOT/'voice-tools/ace-step/checkpoints/acestep-v15-turbo/model.safetensors').is_file()
def start(data):
 if CLOSING.is_set():raise ValueError('The voice server is shutting down.')
 engine=data.get('engine','kokoro')
 if engine=='kokoro':request=validate_request(data)
 elif engine=='soulx':request=soulx_singing.validate_request(data)
 elif engine=='ace-step':
  lyrics=str(data.get('lyrics','')).strip()[:500]
  if not lyrics:raise ValueError('Enter a few lines of original lyrics')
  request={'lyrics':lyrics,'caption':str(data.get('caption') or DEFAULT_STYLE)[:500],'duration':15}
 else:raise ValueError('Unknown singing engine')
 key=hashlib.sha256(json.dumps(request,sort_keys=True).encode()).hexdigest()[:24]
 with GUARD:
  if (FOLDER/(key+'.audio.json')).exists():return {'id':key,'status':'ready'}
  if not available(engine):
   if engine=='soulx':raise ValueError('The experimental singer is not installed on this machine. Choose the Kokoro melody comparison instead.')
   raise ValueError('The singing engine is not installed. Run scripts/setup_voice.py'+(' --singing' if engine=='ace-step' else '')+'.')
  if key in JOBS and JOBS[key]['status'] in ('queued','generating','ready'):return {'id':key,'status':JOBS[key]['status']}
  if any(j['status'] in ('queued','generating') for j in JOBS.values()):raise ValueError('A song is already generating. Stop it or wait for it to finish.')
  job={'status':'queued','process':None}
  JOBS[key]=job
  # Retain a small status history; finished audio is already stored on disk.
  for old in list(JOBS):
   if len(JOBS)<=64:break
   if JOBS[old]['status'] not in ('queued','generating'):del JOBS[old]
  expressive_voice.MUSIC_ACTIVE.set();threading.Thread(target=run,args=(key,request,job),daemon=True).start()
 return {'id':key,'status':'queued'}
def run_stage(job,interpreter,worker,path,phase):
 error_path=path.with_suffix('.error.json');error_path.unlink(missing_ok=True)
 path.with_suffix('.progress.json').unlink(missing_ok=True)
 with (ROOT/'evidence/singing-worker.log').open('a') as log:
  with GUARD:
   if job['status']=='cancelled':return False
   job['phase']=phase
   process=subprocess.Popen([str(interpreter),'-u',str(worker),str(path)],cwd=ROOT,stdout=log,stderr=log,env={**os.environ,'HF_HUB_OFFLINE':'1','HF_HUB_DISABLE_TELEMETRY':'1','TOKENIZERS_PARALLELISM':'false','PYTORCH_ENABLE_MPS_FALLBACK':'1'})
   job['process']=process
  try:code=process.wait(timeout=360)
  except subprocess.TimeoutExpired:process.kill();process.wait();raise RuntimeError('Song generation timed out')
 if job['status']=='cancelled':return False
 if code:
  if error_path.exists():raise ValueError(json.loads(error_path.read_text())['error'])
  raise RuntimeError('The singing engine failed. See evidence/singing-worker.log.')
 return True

def run(key,request,job):
 try:
  with expressive_voice.LOCK:
   if job['status']=='cancelled':return
   # SoulX's bounded process can coexist with small Kokoro weights. Release
   # the much larger reference matcher; do not make speaking reload afterward.
   if request.get('engine')=='soulx':job['speechResidency']=expressive_voice.prepare_for_singing()
   else:expressive_voice.close()
   with GUARD:
    if job['status']=='cancelled':return
    job['status']='generating'
   path=FOLDER/(key+'.json');path.write_text(json.dumps(request));started=time.perf_counter()
   kokoro=request.get('engine')=='kokoro'
   soulx=request.get('engine')=='soulx'
   if soulx:
    reference=soulx_singing.reference_base(request)
    metadata=reference.with_suffix('.result.json' if request['referenceKind']=='sung' else '.json')
    if not (reference.with_suffix('.wav').is_file() and metadata.is_file()):
     if not run_stage(job,ROOT/'.voice-venv/bin/python',ROOT/'singing_reference.py',path,'Preparing the Kokoro '+request['referenceKind']+' reference…'):return
   interpreter=ROOT/('.research-venv/bin/python' if soulx else '.voice-venv/bin/python' if kokoro else '.music-venv/bin/python')
   worker=ROOT/('soulx_singing.py' if soulx else 'kokoro_singing.py' if kokoro else 'singing_worker.py')
   phase='Loading and rendering the experimental singer…' if soulx else 'Preparing notes and vowels…'
   if not run_stage(job,interpreter,worker,path,phase):return
   details=json.loads(path.with_suffix('.result.json').read_text());wav_path=Path(details['path'])
   result=pack_audio(wav_path,started)
   if kokoro:result.update(engine='kokoro-singing',speaker=request['speaker'],score=details['score'])
   if soulx:result.update({k:v for k,v in details.items() if k not in ('path','duration')})
   with GUARD:
    if job['status']=='cancelled':return
    temporary=FOLDER/(key+'.audio.tmp');temporary.write_text(json.dumps(result));temporary.replace(FOLDER/(key+'.audio.json'));job['status']='ready'
 except Exception as e:
  if job['status']!='cancelled':job.update(status='error',error=str(e))
 finally:
  # A cancelled job can finish after its replacement was queued.
  with GUARD:
   if not any(j['status'] in ('queued','generating') for j in JOBS.values()):expressive_voice.MUSIC_ACTIVE.clear()
  # Restore quick conversational speech after the separate singing process exits.
  if not CLOSING.is_set():expressive_voice.warm()
def status(key):
 if not isinstance(key,str) or len(key)!=24 or any(c not in '0123456789abcdef' for c in key):raise ValueError('Unknown song')
 path=FOLDER/(key+'.audio.json')
 if path.exists():return {'status':'ready','result':json.loads(path.read_text())}
 job=JOBS.get(key)
 if not job:raise ValueError('Unknown song')
 result={k:v for k,v in job.items() if k!='process'}
 if job['status']=='generating':
  try:
   progress=json.loads((FOLDER/(key+'.progress.json')).read_text())
   phase=progress.get('phase') if isinstance(progress,dict) else None
   if isinstance(phase,str):result['phase']=phase[:200]
  except (OSError,ValueError):pass
 return result
def cancel(key):
 process=None
 with GUARD:
  job=JOBS.get(key)
  if job and job['status'] in ('queued','generating'):
   job['status']='cancelled'
   process=job['process']
 if process and process.poll() is None:
  process.terminate()
  try:process.wait(timeout=3)
  except subprocess.TimeoutExpired:process.kill();process.wait()
 return {'ok':True}
def close():
 CLOSING.set()
 for key in list(JOBS):cancel(key)
atexit.register(close)

def pack_audio(wav_path,started):
 # ACE writes float WAV. PyAV (already used by Whisper) decodes it without another runtime.
 import av
 container=av.open(str(wav_path));resampler=av.AudioResampler(format='s16',layout='mono',rate=24000);chunks=[]
 for frame in container.decode(audio=0):
  for frame in resampler.resample(frame):chunks.append(frame.to_ndarray().reshape(-1))
 for frame in resampler.resample(None):chunks.append(frame.to_ndarray().reshape(-1))
 container.close();pcm=np.concatenate(chunks).astype('<i2');samples=pcm.astype(np.float32)/32768;out=io.BytesIO()
 with wave.open(out,'wb') as f:f.setparams((1,2,24000,0,'NONE','not compressed'));f.writeframes(pcm.tobytes())
 result={'audio':base64.b64encode(out.getvalue()).decode(),'sampleRate':24000,'duration':len(pcm)/24000,'frames':analyze(samples,24000),'synthMs':round((time.perf_counter()-started)*1000),'analysisMs':0,'engine':'ace-step'}
 return result
