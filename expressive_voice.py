"""Local quality voice adapter, separate from the lightweight Piper environment."""
import atexit,base64,hashlib,io,json,os,select,signal,subprocess,threading,time,wave
from pathlib import Path
import numpy as np
from audio_analysis import analyze
from voice_requests import register, finish, check, VoiceCancelled
ROOT=Path(__file__).resolve().parent
LOCK=threading.Lock();PROCESS=None;MUSIC_ACTIVE=threading.Event()
BUFFER=bytearray();CAN_SIGNAL=False
WARMING=threading.Event();READY=threading.Event()
CACHE=ROOT/'.cache/english-voice-v1';CACHE.mkdir(parents=True,exist_ok=True)
REFERENCES=ROOT/'.cache/voice-references';REFERENCES.mkdir(parents=True,exist_ok=True)
def close(timeout=3):
 global PROCESS,CAN_SIGNAL
 if PROCESS and PROCESS.poll() is None:
  PROCESS.terminate()
  try:PROCESS.wait(timeout=timeout)
  except subprocess.TimeoutExpired:PROCESS.kill();PROCESS.wait()
 PROCESS=None
 BUFFER.clear();CAN_SIGNAL=False
 READY.clear()
atexit.register(close)
def available():return (ROOT/'.voice-venv/bin/python').is_file() and any((ROOT/'models/expressive/Kokoro-82M-bf16').glob('*.safetensors'))
def start_worker():
 global PROCESS,CAN_SIGNAL
 if not available():raise ValueError('Expressive voice is not installed. Run scripts/setup_voice.py.')
 if not PROCESS or PROCESS.poll() is not None:
  BUFFER.clear();CAN_SIGNAL=False
  env={**os.environ,'HF_HUB_OFFLINE':'1','TOKENIZERS_PARALLELISM':'false'}
  with (ROOT/'evidence/expressive-worker.log').open('a') as log:
   PROCESS=subprocess.Popen([str(ROOT/'.voice-venv/bin/python'),'-u',str(ROOT/'expressive_worker.py')],cwd=ROOT,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=log,bufsize=0,env=env)
def send_worker(request):
 PROCESS.stdin.write((json.dumps(request)+'\n').encode());PROCESS.stdin.flush()
def read_worker(event,state):
 """Read complete protocol lines without TextIO/select read-ahead races."""
 global CAN_SIGNAL
 while True:
  now=time.monotonic()
  if event.is_set():
   if state.get('cancelledAt') is None:
    state['cancelledAt']=now
    if state.get('stream') and CAN_SIGNAL and hasattr(signal,'SIGUSR1') and PROCESS.poll() is None:PROCESS.send_signal(signal.SIGUSR1)
   if now-state['cancelledAt']>.6:close(timeout=.2);raise VoiceCancelled('Speech cancelled')
  if now>state['deadline']:close();raise TimeoutError('Voice generation took too long')
  if b'\n' in BUFFER:
   line,_,tail=BUFFER.partition(b'\n');BUFFER[:]=tail;CAN_SIGNAL=True
   return json.loads(line)
  if select.select([PROCESS.stdout],[],[],.025)[0]:
   chunk=os.read(PROCESS.stdout.fileno(),65536)
   if not chunk:
    close();check(event);raise RuntimeError('Voice process stopped; try again')
   BUFFER.extend(chunk)
def with_frames(result):
 start=time.perf_counter()
 with wave.open(io.BytesIO(base64.b64decode(result['audio'])),'rb') as f:samples=np.frombuffer(f.readframes(f.getnframes()),dtype='<i2').astype(np.float32)/32768;sr=f.getframerate()
 result['frames']=analyze(samples,sr);result['analysisMs']=round((time.perf_counter()-start)*1000,1)
 return result
def prepare_for_singing():
 """Caller holds LOCK. Keep Kokoro warm while dropping clone-model memory."""
 if not PROCESS:READY.clear();return False
 if PROCESS.poll() is not None:close(timeout=.2);return False
 try:
  send_worker({'control':'prepare-singing'})
  result=read_worker(threading.Event(),{'deadline':time.monotonic()+5})
  if result.get('type')!='singing-ready':raise RuntimeError('Unexpected voice control response')
  if result.get('kokoroReady'):READY.set()
  else:READY.clear()
  return result
 except Exception:
  close(timeout=.2)
  return False
def synthesize(data):
 request_id=data.get('requestId');event=register(request_id)
 try:return _synthesize(data,event)
 finally:finish(request_id)
def _synthesize(data,event):
 global PROCESS
 check(event)
 request={k:data.get(k) for k in ('text','engine','speaker','emotion','reference','referenceText') if data.get(k) is not None}
 key=hashlib.sha256(json.dumps({'version':'clone-4' if request.get('engine')=='clone' else 'english-3',**request},sort_keys=True).encode()).hexdigest();path=CACHE/(key+'.json')
 if not data.get('_warm') and path.exists():result=json.loads(path.read_text());result['cached']=True;return result
 if MUSIC_ACTIVE.is_set():raise ValueError('A song is generating. Stop it before preparing new speech.')
 while not LOCK.acquire(timeout=.05):check(event)
 try:
  check(event)
  if MUSIC_ACTIVE.is_set():raise ValueError('A song is generating. Stop it before preparing new speech.')
  if not data.get('_warm') and path.exists():result=json.loads(path.read_text());result['cached']=True;return result
  start_worker();send_worker(request)
  result=read_worker(event,{'deadline':time.monotonic()+240})
  if result.get('error'):raise ValueError(result['error'])
  if request.get('engine','expressive')=='expressive':READY.set()
  check(event)
  with_frames(result)
  # Generated audition lines may be reused by reels; no reference clips leave this folder.
  temporary=path.with_suffix('.tmp');temporary.write_text(json.dumps(result));temporary.replace(path);return result
 finally:LOCK.release()
def stream(data):
 request_id=data.get('requestId');event=register(request_id);acquired=False;sent=False;done=False
 state={'deadline':time.monotonic()+240,'stream':True}
 try:
  if data.get('engine')!='clone':raise ValueError('Choose reference matching for native audio streaming')
  check(event)
  while not LOCK.acquire(timeout=.05):check(event)
  acquired=True;check(event)
  if MUSIC_ACTIVE.is_set():raise ValueError('A song is generating. Stop it before preparing new speech.')
  request={k:data.get(k) for k in ('text','engine','emotion','reference','referenceText') if data.get(k) is not None}
  start_worker();send_worker({**request,'stream':True});sent=True
  while True:
   result=read_worker(event,state)
   if result.get('error'):done=True;raise ValueError(result['error'])
   if result.get('type')=='done':done=True;check(event);yield result;return
   if not event.is_set():yield with_frames(result)
 finally:
  if acquired:
   # A disconnected browser may close this generator between chunks. Drain
   # the matching terminal message before handing the shared worker to a new
   # request; otherwise old audio could be mistaken for the next character.
   if sent and not done and PROCESS and PROCESS.poll() is None:
    event.set()
    try:
     while True:
      result=read_worker(event,state)
      if result.get('type')=='done' or result.get('error'):break
    except Exception:close(timeout=.2)
   LOCK.release()
  finish(request_id)
def warm():
 """Load English Kokoro before the first conversation, without playing audio."""
 if WARMING.is_set() or READY.is_set() or MUSIC_ACTIVE.is_set() or not available():return
 WARMING.set()
 def run():
  try:
   # Bypass disk cache: a cached WAV does not warm model weights or the G2P.
   _synthesize({'text':'Hello, I am ready to talk.','engine':'expressive','speaker':'af_heart','_warm':True},threading.Event())
  except Exception as e:print('Voice warm-up:',e,flush=True)
  finally:WARMING.clear()
 threading.Thread(target=run,daemon=True).start()
def reference(data):
 raw=base64.b64decode(data.get('audio',''),validate=True)
 with wave.open(io.BytesIO(raw),'rb') as f:
  duration=f.getnframes()/f.getframerate()
  if f.getnchannels()!=1 or f.getsampwidth()!=2 or not 5<=duration<=30:raise ValueError('Use 5–30 seconds of mono WAV audio')
 transcript=str(data.get('text','')).strip()[:1500]
 if not transcript:raise ValueError('Enter the words spoken in the reference clip')
 name=hashlib.sha256(raw).hexdigest()+'.wav';(REFERENCES/name).write_bytes(raw)
 return {'reference':name,'referenceText':transcript,'duration':duration}
