"""Verify the standalone alpha, without playing audio or downloading models."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('VECLIB_MAXIMUM_THREADS','1')
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request,urlopen

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/lyra-verification'
sys.path.insert(0,str(ROOT/'voice-research/live_voicebox'))
from runtime import Engine,Session,RATE
import numpy as np
import soundfile as sf


def collect(session):
    parts=[]
    while packet:=session.next_chunk():parts.append(np.frombuffer(packet,dtype='<f4'))
    assert session.completed
    return np.concatenate(parts)


def run(label,command):
    result=subprocess.run(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
    (OUT/(label+'.log')).write_text(result.stdout)
    if result.returncode:print(result.stdout);raise SystemExit(label+' failed')
    print(label+': passed',flush=True)


def isolation_check():
    # Only files shipped with the alpha are present here. In particular there
    # are no evidence archives, model caches, external tools or avatar files.
    with tempfile.TemporaryDirectory(prefix='lyra-standalone-') as folder:
        isolated=Path(folder)
        for name in ('jaeger_lyra','voice-research/audio_native','voice-research/live_voicebox','demo/bundle'):
            shutil.copytree(ROOT/name,isolated/name,ignore=shutil.ignore_patterns('__pycache__'))
        with (isolated/'server.log').open('w+') as log:
            env={**os.environ};env.pop('PYTHONPATH',None)
            process=subprocess.Popen([sys.executable,'-u','-m','jaeger_lyra','--port','0'],cwd=isolated,
                env=env,stdout=log,stderr=subprocess.STDOUT)
            try:
                address=None
                # A fresh macOS runner may need more than ten seconds for
                # numerical-library imports and bundle loading. This is a
                # bounded readiness check, not an audio latency benchmark.
                started=time.monotonic();deadline=started+60
                while time.monotonic()<deadline:
                    log.seek(0);output=log.read()
                    if 'http://127.0.0.1:' in output:
                        address=output.split('http://127.0.0.1:',1)[1].split('/')[0];break
                    if process.poll() is not None:raise AssertionError('Standalone server exited:\n'+output)
                    time.sleep(.1)
                assert address is not None,'Standalone server was not ready within 60 seconds. See isolated-server.log.\n'+output
                startup_s=time.monotonic()-started
                base='http://127.0.0.1:'+address
                with urlopen(base+'/api/catalog',timeout=10) as r:catalog=json.load(r)
                assert catalog['project']=='Jaeger Lyra' and catalog['demonstrates']=='reconstruction-playback'
                with urlopen(base+'/api/duet',timeout=10) as r:duet=json.load(r)
                assert duet=={'available':False},'Local user media must not be required by a fresh checkout'
                request=Request(base+'/api/session',data=json.dumps({'mode':'word','item':'af_heart-palm'}).encode(),headers={'Content-Type':'application/json'})
                with urlopen(request,timeout=10) as r:session=json.load(r)
                with urlopen(base+'/api/stream?id='+session['id'],timeout=10) as r:audio=r.read()
                with urlopen(base+'/api/status?id='+session['id'],timeout=10) as r:state=json.load(r)
                assert state['completed'] and len(audio)>1000
                return {'passed':True,'outsideProjectFilesRequired':False,'samples':len(audio)//4,'startup_s':startup_s}
            finally:
                process.terminate()
                try:process.wait(timeout=5)
                except subprocess.TimeoutExpired:process.kill();process.wait()
                log.seek(0);(OUT/'isolated-server.log').write_text(log.read())


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    checks=json.loads((ROOT/'demo/bundle/checksums.json').read_text())
    for row in checks['files']:
        path=(ROOT/row['path']).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file()
        assert path.stat().st_size==row['bytes'] and hashlib.sha256(path.read_bytes()).hexdigest()==row['sha256'],row['path']
    print('Bundle hashes:',len(checks['files']),'passed',flush=True)
    engine=Engine();phrases=[]
    for item in engine.manifest['phrases']:
        performance,metadata=engine.performances[('speech',item['id'])]
        assert len(performance.pulses)>3
        neutral=collect(Session('speech',item,performance,metadata,engine.decoder,gain=1.))
        reference,rate=sf.read(ROOT/item['reconstructionFile']);assert rate==RATE
        error=float(max(abs(neutral-reference*item['streamGain'])));assert error<1e-6
        vibrato=collect(Session('speech',item,performance,metadata,engine.decoder,gain=1.,shape={'vibrato':45.}))
        effect=float(np.sqrt(np.mean((vibrato-neutral)**2)))
        assert len(vibrato)==len(neutral) and np.isfinite(vibrato).all() and effect>1e-4
        phrases.append({'id':item['id'],'neutralMaximumError':error,'vibratoDifferenceRms':effect,'samples':len(neutral)})
    forbidden=[name for name in sys.modules if name.split('.')[0] in ('kokoro','mlx','mlx_audio','torch','transformers')]
    assert not forbidden,forbidden
    print('Nine neutral/vibrato phrase checks: passed; no pretrained engine imported',flush=True)
    run('core-tests',[sys.executable,'-m','unittest','discover','-s','voice-research/audio_native','-p','test_*.py','-v'])
    run('runtime-tests',[sys.executable,'-m','unittest','discover','-s','voice-research/live_voicebox','-p','test_*.py','-v'])
    run('legacy-adapter-tests',[sys.executable,'-m','unittest','discover','-s','tests','-p','test_*.py','-v'])
    run('browser-audio-tests',['node','--test','voice-research/live_voicebox/test_pcm.mjs'])
    run('browser-syntax',['node','--check','voice-research/live_voicebox/public/app.js'])
    run('duet-browser-syntax',['node','--check','voice-research/live_voicebox/public/duet.js'])
    run('voice-pair-tests',['node','--test','voice-research/live_voicebox/test_voice_pair.mjs'])
    isolated=isolation_check();print('Isolated source-checkout playback: passed',flush=True)
    report={'complete':True,'audioPlayed':False,'bundleFiles':len(checks['files']),
        'phrases':phrases,'pretrainedEngineImports':forbidden,'isolatedCheckout':isolated}
    (OUT/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Standalone alpha verification passed. No audio played.',flush=True)


if __name__=='__main__':main()
