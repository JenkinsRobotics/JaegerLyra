"""Silent end-to-end checks against a running loopback demo; never plays audio."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import soundfile as sf
from runtime import Engine, Session, ROOT, RATE


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--url',default='http://127.0.0.1:8794')
    args=parser.parse_args()
    base=args.url.rstrip('/')
    if not base.startswith(('http://127.0.0.1:','http://localhost:')):
        raise ValueError('Only the local research server is supported')
    out=ROOT/'evidence/live-voicebox/v1'
    report={'scope':'HTTP and waveform verification, with no audio playback or listening judgment',
        'url':base,'complete':False,'phrases':[],'autoplay':False,'listeningAccepted':False}

    def request(path,body=None,headers=None):
        payload=None if body is None else json.dumps(body).encode()
        return urlopen(Request(base+path,data=payload,
            headers={'Content-Type':'application/json',**(headers or {})}),timeout=10)

    def api(path,body=None):
        with request(path,body) as response:return json.load(response)

    def rejects(path,body=None,headers=None):
        try:
            with request(path,body,headers):pass
        except HTTPError as exc:
            assert exc.code==400,(path,exc.code)
            return
        raise AssertionError('Invalid request accepted: '+path)

    def closed(identifier):
        deadline=time.monotonic()+3
        while time.monotonic()<deadline:
            result=api('/api/status?id='+identifier)
            if result['closed']:return result
            time.sleep(.04)
        raise AssertionError('Session did not close')

    catalog=api('/api/catalog')
    assert catalog['activeSessions']==0,'Finish any browser performance before this check'
    assert len(catalog['units'])==18 and len(catalog['phrases'])==9
    engine=Engine()
    for item in catalog['phrases']:
        saved=engine.items[('speech',item['id'])]
        for choice in item['choices']:
            expected=next(c for c in saved['choices'] if c['choice']==choice['choice'])
            with request(choice['url']) as response:
                assert hashlib.sha256(response.read()).hexdigest()==expected['sha256']
    for path in ('/','/app.js','/pcm-player.js','/pcm-ring.js','/pcm-bytes.js','/style.css'):
        with request(path) as response:assert response.status==200 and len(response.read())>0
    report['catalogAndAssetsPassed']=True

    for item in catalog['phrases']:
        start=time.perf_counter()
        session=api('/api/session',{'mode':'speech','item':item['id'],'gain':.75})
        with request('/api/stream?id='+session['id']) as response:
            assert response.headers['Transfer-Encoding']=='chunked'
            assert response.headers['X-Sample-Rate']==str(RATE)
            first=response.read(1920)
            first_ms=(time.perf_counter()-start)*1000
            raw=first+response.read()
        wave=np.frombuffer(raw,dtype='<f4')
        saved=engine.items[('speech',item['id'])]
        source,rate=sf.read(ROOT/saved['reconstructionFile'],dtype='float32')
        expected=source.astype(float)*saved['streamGain']*.75
        assert rate==RATE and len(wave)==len(expected) and np.isfinite(wave).all()
        error=float(np.max(np.abs(wave-expected)))
        assert error<1e-6,error
        state=closed(session['id'])
        assert state['completed'] and state['generatedSamples']==len(wave)
        report['phrases'].append({'id':item['id'],'samples':len(wave),'maximumError':error,
            'firstPacketMs':first_ms,'blockP95Ms':state['packetP95Ms'],
            'maximumBlockMs':state['maximumPacketMs'],'blocksOver5Ms':state['packetsOver5Ms']})
        print('Speech verified:',item['id'],flush=True)

    item=catalog['units'][0]
    session=api('/api/session',{'mode':'syllable','item':item['id'],'hz':item['baseHz'],'gain':.7})
    raw=bytearray();sent=set()
    with request('/api/stream?id='+session['id']) as response:
        while True:
            packet=response.read(1920)
            if not packet:break
            raw.extend(packet);seconds=len(raw)/4/RATE
            for seq,at,change in [(1,.4,{'hz':261.6256}),(2,1.1,{'hz':349.2282,'gain':.5}),(3,1.6,{'release':True})]:
                if seconds>=at and seq not in sent:
                    api('/api/control',{'id':session['id'],'sequence':seq,**change});sent.add(seq)
    state=closed(session['id']);assert state['completed'] and sent=={1,2,3}
    performance,metadata=engine.performances[('syllable',item['id'])]
    replay=Session('syllable',engine.items[('syllable',item['id'])],performance,metadata,engine.decoder,item['baseHz'],.7)
    events={e['appliedSample']:e for e in state['events']}
    parts=[]
    while not replay.closed:
        event=events.pop(replay.position,None)
        if event is not None:
            replay.control({'sequence':event['sequence'],'hz':event['hz'],'gain':event['gain'],
                'release':event['releaseRequested']})
        part=replay._next()
        if part is not None:parts.append(part)
    actual=np.frombuffer(raw,dtype='<f4');expected=np.concatenate(parts)
    assert len(actual)==len(expected) and np.isfinite(actual).all()
    error=float(np.max(np.abs(actual-expected)));assert error<1e-6
    assert not events and [e['sequence'] for e in state['events']]==[1,2,3]
    sf.write(out/'live-controls-http.wav',actual,RATE,subtype='FLOAT')
    report['liveControls']={'samples':len(actual),'maximumReplayError':error,'events':state['events'],
        'releaseSample':state['releaseSample'],'blockP95Ms':state['packetP95Ms'],
        'maximumBlockMs':state['maximumPacketMs'],'blocksOver5Ms':state['packetsOver5Ms']}

    for mode in ('explicit-stop','disconnect'):
        session=api('/api/session',{'mode':'syllable','item':item['id']})
        response=request('/api/stream?id='+session['id']);assert response.read(1920)
        rejects('/api/stream?id='+session['id'])
        if mode=='explicit-stop':
            stopped=api('/api/stop',{'id':session['id']});assert stopped['closed']
            remaining=response.read();assert len(remaining)<=1920*5
        response.close()
        state=closed(session['id']);assert not state['completed']
        report[mode]={'closed':True,'generatedSamples':state['generatedSamples']}
    rejects('/api/catalog',headers={'Origin':'https://example.com'})
    rejects('/api/session',{'mode':'speech','item':'missing'})
    rejects('/api/session',{'mode':'syllable','item':item['id'],'gain':-1})
    assert api('/api/catalog')['activeSessions']==0
    report['invalidRequestsRejected']=True
    report['activeSessionsAfter']=0
    report['complete']=True
    report['latencyScope']='Local session creation and transport to first packet; audio buffering/device latency excluded. No new text generation.'
    (out/'http-check.json').write_text(json.dumps(report,indent=2)+'\n')
    print('All HTTP checks passed; no playback.',flush=True)


if __name__=='__main__':main()
