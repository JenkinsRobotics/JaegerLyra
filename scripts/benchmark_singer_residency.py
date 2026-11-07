"""Actual clone -> song -> speech test, plus cancellation after rendering starts."""
import base64,json,subprocess,time,urllib.request
from pathlib import Path
from benchmark_integrated_singer import call
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/singer-residency'


def workers(name, key=None):
    rows=subprocess.check_output(['ps','-Ao','pid,command'],text=True).splitlines()
    return [int(row.strip().split(None,1)[0]) for row in rows
            if str(ROOT/name) in row and (key is None or key+'.json' in row)]


def main():
    OUT.mkdir(exist_ok=True)
    report={'scope':'Real API, no speaker playback; exact audio identity checked separately by the loader benchmark'}
    started=time.perf_counter()
    while not call('/api/status')['kokoroReady']:
        if time.perf_counter()-started>30:raise TimeoutError('Kokoro did not warm')
        time.sleep(.2)
    raw=(ROOT/'evidence/voice-fixtures/am_michael-reference.wav').read_bytes()
    reference=call('/api/voice-reference',{'audio':base64.b64encode(raw).decode(),
        'text':'You light up my day. Stay here close to me. We can watch the stars shine in the sky.'})
    # Deliberately load the heavier model so the song must trim it.
    clone=call('/api/voice',{'engine':'clone',**reference,'text':'Before we sing, this is a brief reference voice check.'})
    report['cloneEngine']=clone['engine'];report['speechWorkerBefore']=workers('expressive_worker.py')
    requests=[{'lyrics':'Little lights are shining\nStay with me tonight','tempo':103},
              {'lyrics':'The tiny lantern sings tonight','tempo':105}]
    report['jobs']=[]
    for index,body in enumerate(requests):
        started=time.perf_counter();job=call('/api/song',{'engine':'soulx','speaker':'af_heart',**body})
        row={'id':job['id'],'request':body,'phases':[]};last=None
        try:
            while time.perf_counter()-started<120:
                data=call('/api/song?id='+job['id']);phase=data.get('phase',data['status'])
                if data.get('speechResidency'):row['speechResidency']=data['speechResidency']
                if phase!=last:
                    row['phases'].append({'phase':phase,'seconds':round(time.perf_counter()-started,3)})
                    print(index, row['phases'][-1],flush=True);last=phase
                if data['status']=='error':raise RuntimeError(data['error'])
                if index==1 and phase=='Singing the lyrics…':
                    time.sleep(.3)
                    row['workerPidsBeforeCancel']=workers('soulx_singing.py',job['id'])
                    tick=time.perf_counter();call('/api/song-cancel',{'id':job['id']})
                    row['cancelMs']=round((time.perf_counter()-tick)*1000,1)
                    row['afterCancel']=call('/api/song?id='+job['id'])['status']
                    row['remainingWorkers']=workers('soulx_singing.py',job['id'])
                    assert row['afterCancel']=='cancelled' and not row['remainingWorkers']
                    break
                if data['status']=='ready':
                    if index==1:raise RuntimeError('Missed the render-stage cancellation window')
                    result=data['result'];raw=base64.b64decode(result.pop('audio'));result.pop('frames',None)
                    row.update({k:v for k,v in result.items() if k!='score'})
                    row['wallSeconds']=round(time.perf_counter()-started,3)
                    (OUT/'rendered-song.wav').write_bytes(raw)
                    (OUT/'rendered-song.json').write_text(json.dumps(result,indent=2))
                    request=urllib.request.Request('http://127.0.0.1:8791/api/transcribe',raw,{'Content-Type':'audio/wav'})
                    with urllib.request.urlopen(request,timeout=30) as response:row['asr']=json.load(response)
                    break
                time.sleep(.1)
            else:raise TimeoutError('Song deadline exceeded')
        finally:call('/api/song-cancel',{'id':job['id']})
        row['speechWorkerAfter']=workers('expressive_worker.py')
        row['kokoroReadyAfter']=call('/api/status')['kokoroReady']
        text=('That little melody is ready, and I can still speak right away.' if index==0
              else 'The singing stopped, and my speaking voice stayed ready.')
        tick=time.perf_counter();speech=call('/api/voice',{'engine':'expressive','speaker':'af_heart','text':text})
        row['speechRecoveryMs']=round((time.perf_counter()-tick)*1000,1)
        row['speechCached']=speech.get('cached',False)
        assert row['speechWorkerAfter']==report['speechWorkerBefore'] and row['kokoroReadyAfter']
        assert row['speechResidency']['loadedEngines']==['expressive']
        (OUT/f'speech-after-{index}.wav').write_bytes(base64.b64decode(speech['audio']))
        report['jobs'].append(row);(OUT/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(row),flush=True)


if __name__=='__main__':main()
