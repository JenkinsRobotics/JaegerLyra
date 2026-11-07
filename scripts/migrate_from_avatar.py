"""One-time local extraction; preserve the avatar client with path links.

This moves existing files on the same disk. It does not publish data, remove
archives, download models, or overwrite destination content. The demo's tracked
bundle is built separately; model caches and research outputs stay gitignored.
"""
import argparse
import json
import os
from pathlib import Path

AUDIO_PREFIXES=('anchored-','articulation-','audio-','band-','captured-','chatterbox-','clone-',
    'coherent-','compact-','compiled-','english-','expressive-','free-','fresh-','generated-',
    'gpt-','harmonic-','integrated-singer','kokoro-','lattice-','live-','neural-','openvoice',
    'output-energy','pulse-','qwen-','resonance-','rich-source','seed-','singer-','singing-',
    'soulx','source-','speaker-','stationarity','streaming','sung-','variable-','vocal-','voice-')


def main():
    parser=argparse.ArgumentParser();parser.add_argument('source',type=Path);args=parser.parse_args()
    source=args.source.resolve();target=Path(__file__).resolve().parents[1]
    assert (source/'voice-research/live_voicebox/runtime.py').is_file()
    assert (target/'.git').is_dir()
    paths=['voice-research','models','voice-tools','checkpoints',
        '.research-venv','.voice-venv','.music-venv','.seed-vc-venv','.sovits-venv']
    paths += ['audio_analysis.py','english_singing_score.py','expressive_voice.py','expressive_worker.py',
        'kokoro_singing.py','singing.py','singing_model.py','singing_reference.py','singing_worker.py',
        'soulx_singing.py','voice_requests.py']
    paths += [p.name for p in source.glob('requirements-*-lock.txt') if p.name!='requirements-lock.txt']
    paths += [str(p.relative_to(source)) for p in (source/'scripts').iterdir()
        if p.is_file() and p.suffix=='.py' and p.name not in ('probe_brain.py','setup.py','verify_local.py')]
    paths += ['public/research','public/audio/research']
    paths += [str(p.relative_to(source)) for p in (source/'public').glob('voicebox-*.html')]
    paths += [str(p.relative_to(source)) for p in (source/'evidence').iterdir()
        if (p.name.startswith(AUDIO_PREFIXES) or p.name in ('laugh.wav','sing.wav','original-song-demo.wav'))
        and p.name!='voice-goal-baseline.zip']
    paths += [str(p.relative_to(source)) for p in (source/'licenses').iterdir() if p.is_file()
        and not p.name.startswith(('face-','mediapipe','three','momo','nori','gemma'))]
    paths += ['tests/'+name for name in ('test_english_singing_score.py','test_kokoro_score.py',
        'test_soulx_singing.py','test_voice_cancellation.py','test_voice_studio.py')]
    paths += ['.cache/'+name for name in ('acestep','english-voice-v1','expressive-voice','huggingface',
        'nltk','singing-references','songs','tmp','voice-references')]
    paths=sorted(set(p for p in paths if (source/p).exists()))
    for name in paths:
        assert not (target/name).exists() and not (target/name).is_symlink(), 'Destination exists: '+name
        assert not (source/name).is_symlink(), 'Already linked: '+name
    report={'source':str(source),'destination':str(target),'moves':[],
        'purpose':'Extract all audio research, demo, evidence and optional reference infrastructure; retain avatar path compatibility',
        'excluded':'Avatar renderer/assets/server, npm and base studio environment, full-avatar backup ZIPs',
        'published':False,'complete':False}
    out=target/'evidence/migration-v1';out.mkdir(parents=True,exist_ok=True)
    for name in paths:
        old=source/name;new=target/name;new.parent.mkdir(parents=True,exist_ok=True)
        before=old.stat();kind='directory' if old.is_dir() else 'file'
        old.rename(new)
        assert new.stat().st_ino==before.st_ino and new.stat().st_dev==before.st_dev
        old.symlink_to(os.path.relpath(new,old.parent),target_is_directory=kind=='directory')
        assert old.resolve()==new.resolve()
        report['moves'].append({'path':name,'kind':kind,'inodePreserved':True,'avatarPathLinked':True})
        (out/'local-transfer.json').write_text(json.dumps(report,indent=2)+'\n')
    report['complete']=True;(out/'local-transfer.json').write_text(json.dumps(report,indent=2)+'\n')
    print('Moved',len(report['moves']),'paths into JaegerLyra; original avatar paths now link to them.')


if __name__=='__main__':main()
