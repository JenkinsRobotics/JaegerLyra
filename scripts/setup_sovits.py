#!/usr/bin/env python3
"""Reproduce the isolated English GPT-SoVITS comparison on Apple Silicon."""
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'voice-tools/gpt-sovits'
COMMIT = '04a44eb977586645f33594b72af13a035a2056c6'
REVISION = '8f8857563c040d542de108a2c18b0425abd6c9c5'


def run(*args):
    subprocess.run([str(a) for a in args], cwd=ROOT, check=True)


python = ROOT / '.sovits-venv/bin/python'
if not python.exists():
    system = shutil.which('python3.11')
    fallback = Path.home() / '.pyenv/versions/3.11.9/bin/python'
    if not system and fallback.exists():
        system = fallback
    if not system:
        raise SystemExit('Install Python 3.11, then run this script again.')
    run(system, '-m', 'venv', ROOT / '.sovits-venv')
run(python, '-m', 'pip', 'install', '-r', ROOT / 'requirements-sovits-lock.txt')
if not SOURCE.exists():
    run('git', 'clone', 'https://github.com/RVC-Boss/GPT-SoVITS.git', SOURCE)
    run('git', '-C', SOURCE, 'checkout', COMMIT)
else:
    actual = subprocess.check_output(['git', '-C', str(SOURCE), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise SystemExit('Existing GPT-SoVITS checkout differs from the tested revision; preserved it unchanged.')
run(python, '-c', '''
from huggingface_hub import snapshot_download
import sys
snapshot_download('lj1995/GPT-SoVITS', revision=sys.argv[1], local_dir=sys.argv[2], allow_patterns=[
 's1v3.ckpt', 'v2Pro/s2Gv2ProPlus.pth', 'sv/pretrained_eres2netv2w24s4ep4.ckpt',
 'chinese-roberta-wwm-ext-large/config.json', 'chinese-roberta-wwm-ext-large/tokenizer.json',
 'chinese-roberta-wwm-ext-large/pytorch_model.bin', 'chinese-hubert-base/config.json',
 'chinese-hubert-base/pytorch_model.bin', 'chinese-hubert-base/preprocessor_config.json'])
''', REVISION, SOURCE / 'GPT_SoVITS/pretrained_models')
run(python, '-c', '''
import nltk,sys
for corpus in ['averaged_perceptron_tagger','averaged_perceptron_tagger_eng','cmudict']:
 if not nltk.download(corpus, download_dir=sys.argv[1], quiet=True):
  raise RuntimeError('Could not install '+corpus)
''', ROOT / '.cache/nltk')
print('Ready: .sovits-venv/bin/python scripts/try_gpt_sovits.py')
