"""Exercise actual native-streaming endpoint, cancellation and worker reuse."""
import base64,io,json,time,urllib.request,uuid,wave
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];BASE='http://127.0.0.1:8791'
def request(path,data):return urllib.request.Request(BASE+path,json.dumps(data).encode(),{'Content-Type':'application/json'})
def post(path,data):
 with urllib.request.urlopen(request(path,data),timeout=240) as r:return json.load(r)
def main():
 folder=ROOT/'evidence/clone-streaming';fixture=ROOT/'evidence/voice-fixtures/am_michael-reference'
 reference=post('/api/voice-reference',{'audio':base64.b64encode(fixture.with_suffix('.wav').read_bytes()).decode(),'text':json.loads(fixture.with_suffix('.json').read_text())['text']})
 rows=[]
 for name,text,cancel in [('http-cold','This is my new English voice. I hope it feels a little more like me.',False),('http-warm','A quiet voice can still carry a whole world of feeling.',False),('http-cancel','This longer sentence is here to test a clean interruption before the next thought begins, without repeating a single word from the old stream.',True)]:
  key=str(uuid.uuid4());start=time.perf_counter();chunks=[];pcm=[];cancel_at=None;terminal=None
  with urllib.request.urlopen(request('/api/voice-stream',{'engine':'clone','requestId':key,'text':text,**reference}),timeout=240) as response:
   for line in response:
    event=json.loads(line)
    if event.get('type')=='error':raise RuntimeError(event)
    if event.get('type')=='done':terminal=event;break
    chunks.append({'readyMs':round((time.perf_counter()-start)*1000,1),'duration':event['duration']})
    with wave.open(io.BytesIO(base64.b64decode(event['audio'])),'rb') as f:sr=f.getframerate();pcm.append(f.readframes(f.getnframes()))
    if cancel and cancel_at is None:cancel_at=time.perf_counter();post('/api/voice-cancel',{'requestId':key})
  row={'name':name,'text':text,'chunks':chunks,'terminal':terminal,'firstChunkMs':chunks[0]['readyMs'],'totalMs':round((time.perf_counter()-start)*1000,1),'cancellationMs':None if cancel_at is None else round((time.perf_counter()-cancel_at)*1000,1)}
  with wave.open(str(folder/(name+'.wav')),'wb') as f:f.setparams((1,2,sr,0,'NONE','not compressed'));f.writeframes(b''.join(pcm))
  rows.append(row);print(json.dumps(row),flush=True)
 start=time.perf_counter();recovery=post('/api/voice',{'engine':'expressive','speaker':'af_heart','text':'The next character gets a fresh thought, with no leftover sound.'})
 with wave.open(io.BytesIO(base64.b64decode(recovery['audio'])),'rb') as f:pcm=f.readframes(f.getnframes());sr=f.getframerate()
 with wave.open(str(folder/'http-recovery.wav'),'wb') as f:f.setparams((1,2,sr,0,'NONE','not compressed'));f.writeframes(pcm)
 report={'scope':'Actual localhost HTTP, no speaker playback. Cold includes model/reference encoding. Warm excludes those one-time costs.','reference':reference,'rows':rows,'kokoroRecoveryMs':round((time.perf_counter()-start)*1000,1),'recoveryEngine':recovery['engine']}
 (folder/'http-report.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ('rows','reference')}),flush=True)
if __name__=='__main__':main()
