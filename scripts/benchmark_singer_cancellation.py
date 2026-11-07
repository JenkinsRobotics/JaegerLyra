"""Cancel a real singer, restart the same request, cancel again, recover speech."""
import base64
import json
from pathlib import Path
import subprocess
import time
from benchmark_integrated_singer import call

ROOT=Path(__file__).resolve().parents[1]
request={'engine':'soulx','lyrics':'A quiet river hums\nUnder the silver moon',
         'speaker':'af_heart','referenceKind':'speech','tempo':104,'transpose':0}


def worker_for(key):
    rows=subprocess.check_output(['ps','-Ao','pid,command'],text=True).splitlines()
    return [int(row.strip().split(None,1)[0]) for row in rows
            if 'soulx_singing.py' in row and key+'.json' in row]


def main():
    report={'request':request,'cancellations':[]}
    for attempt in range(2):
        job=call('/api/song',request);started=time.perf_counter();workers=[]
        while time.perf_counter()-started<30:
            status=call('/api/song?id='+job['id'])
            if status['status'] in ('ready','error'):raise RuntimeError('Expected an active uncached job: '+status['status'])
            workers=worker_for(job['id'])
            if workers:break
            time.sleep(.1)
        if not workers:raise RuntimeError('No live singer worker found')
        started=time.perf_counter();call('/api/song-cancel',{'id':job['id']})
        row={'attempt':attempt+1,'id':job['id'],'workerPids':workers,
             'cancelMs':round((time.perf_counter()-started)*1000,1),
             'status':call('/api/song?id='+job['id'])['status'],
             'remainingWorkerPids':worker_for(job['id'])}
        assert row['status']=='cancelled' and not row['remainingWorkerPids']
        report['cancellations'].append(row);print(json.dumps(row),flush=True)
    started=time.perf_counter()
    result=call('/api/voice',{'engine':'expressive','speaker':'af_heart','emotion':'neutral',
         'text':'The song has stopped, and I am ready to speak again.'})
    raw=base64.b64decode(result['audio'])
    folder=ROOT/'evidence/integrated-singer-cancellation';folder.mkdir(exist_ok=True)
    (folder/'recovered-speech.wav').write_bytes(raw)
    report['speechRecovery']={'wallMs':round((time.perf_counter()-started)*1000,1),
                              'engine':result['engine'],'duration':result['duration'],
                              'cached':result.get('cached',False)}
    (folder/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report['speechRecovery']),flush=True)


if __name__=='__main__':main()
