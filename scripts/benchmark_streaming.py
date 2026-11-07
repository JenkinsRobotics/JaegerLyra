"""HTTP-level Kokoro latency/cancellation. Never plays audio through speakers."""
import base64
import concurrent.futures
import json
from pathlib import Path
import statistics
import time
import urllib.error
import urllib.request
import uuid

ROOT=Path(__file__).resolve().parents[1]
BASE='http://127.0.0.1:8791'
LINES=[
 'The first stars are appearing above the rooftops.',
 'I found a tiny blue feather beside the garden gate.',
 'Take a breath. We have plenty of time to begin.',
 'When the little lantern lights the room,',
 'we can sit together and watch the stars.',
 'Dr. Miller will arrive at three fifteen tomorrow.',
 'That was a lovely surprise, and I cannot stop smiling.',
 'Which little adventure shall we try next?',
 'The rain is soft, the tea is warm, and you are here.',
 'Let me think about that for a moment.',
 'You can interrupt me whenever you need to.',
 'I will be right here when you come back.',
]

def post(path,data):
 request=urllib.request.Request(BASE+path,json.dumps(data).encode(),{'Content-Type':'application/json'})
 try:
  with urllib.request.urlopen(request,timeout=240) as response:return response.status,json.load(response)
 except urllib.error.HTTPError as error:return error.code,json.load(error)

def speech(text,**extra):
 start=time.perf_counter();status,data=post('/api/voice',{'text':text,'engine':'expressive','speaker':'af_heart','emotion':'neutral',**extra})
 return {'text':text,'status':status,'requestMs':round((time.perf_counter()-start)*1000,1),**data}

def main():
 folder=ROOT/'evidence/streaming';folder.mkdir(exist_ok=True)
 deadline=time.monotonic()+60
 while time.monotonic()<deadline:
  with urllib.request.urlopen(BASE+'/api/status',timeout=5) as response:state=json.load(response)
  if state['kokoroReady']:break
  time.sleep(.25)
 else:raise RuntimeError('Kokoro did not warm up')
 rows=[]
 for index,text in enumerate(LINES):
  row=speech(text)
  if row['status']!=200:raise RuntimeError(row)
  (folder/f'phrase-{index:02}.wav').write_bytes(base64.b64decode(row.pop('audio')))
  row.pop('frames');row['rtf']=round(row['requestMs']/1000/row['duration'],4);rows.append(row);print(json.dumps(row),flush=True)
 key=str(uuid.uuid4())
 with concurrent.futures.ThreadPoolExecutor() as pool:
  future=pool.submit(speech,'This particular sentence should be interrupted before its voice reaches the listener.',requestId=key)
  time.sleep(.08);start=time.perf_counter();status,cancel=post('/api/voice-cancel',{'requestId':key})
  result=future.result();release_ms=round((time.perf_counter()-start)*1000,1)
 recovery=speech('A fresh reply can begin as soon as the previous thought is stopped.')
 recovery.pop('audio',None);recovery.pop('frames',None)
 uncached=[r for r in rows if not r.get('cached')];times=sorted(r['requestMs'] for r in uncached)
 report={'scope':'Local HTTP WAV readiness, not first audible browser sample. Warm Kokoro Heart; avatar tab may be open. Cached rows excluded from percentiles.',
         'rows':rows,'uncachedCount':len(times),'p50Ms':statistics.median(times) if times else None,
         'p95Ms':times[min(len(times)-1,int(.95*len(times)))] if times else None,
         'cancellation':{'response':cancel,'releaseMs':release_ms,'result':result,'recovery':recovery}}
 (folder/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='rows'}),flush=True)

if __name__=='__main__':main()
