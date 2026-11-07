#!/usr/bin/env python3
"""Install isolated local Apple Silicon speech; --singing also installs ACE-Step."""
import argparse,json,shutil,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--singing',action='store_true');args=parser.parse_args()
def run(*args):subprocess.run([str(a) for a in args],cwd=ROOT,check=True)
def environment(folder,version,lock):
 py=ROOT/folder/'bin/python'
 if not py.exists():
  system=shutil.which('python'+version);fallback=Path.home()/'.pyenv/versions'/('3.11.9' if version=='3.11' else '3.13.7')/'bin/python'
  if not system and fallback.exists():system=str(fallback)
  if not system:raise SystemExit('Install Python '+version+' first, then run this script again.')
  run(system,'-m','venv',ROOT/folder)
 run(py,'-m','pip','install','-r',ROOT/lock);return py
py=environment('.voice-venv','3.13','requirements-voice-lock.txt')
models=[('mlx-community/Kokoro-82M-bf16','a71e4d38b236d968966a2002c4c895dbd12b1c3c'),('mlx-community/chatterbox-turbo-8bit','2f2e21a03863f86a1274d1060dcc188e7cde77e1')]
for repo,revision in models:
 run(py,'-c','from huggingface_hub import snapshot_download; import sys; snapshot_download(sys.argv[1],revision=sys.argv[2],local_dir=sys.argv[3])',repo,revision,ROOT/'models/expressive'/repo.split('/')[-1])
run(py,'-c','from huggingface_hub import hf_hub_download; import sys; hf_hub_download("mlx-community/S3TokenizerV2","model.safetensors",revision="e0c9886f0e1c35ae85b1f27277416fb19fc72bec",cache_dir=sys.argv[1])',ROOT/'models/hf-cache')
if args.singing:
 environment('.music-venv','3.11','requirements-music-lock.txt')
 source=ROOT/'voice-tools/ace-step'
 if not source.exists():
  run('git','clone','https://github.com/ace-step/ACE-Step-1.5.git',source)
  run('git','-C',source,'checkout','ca1e85fe9430179831e6bc6be790c332190a3866')
 run(py,'-c','from huggingface_hub import snapshot_download; import sys; snapshot_download("ACE-Step/Ace-Step1.5",revision="19671f406d603126926c1b7e2adc169acbcade22",local_dir=sys.argv[1])',source/'checkpoints')
print('Local voice setup complete. Restart KIN to use it.')
