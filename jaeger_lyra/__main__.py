"""Run the checked-out reconstruction demo: python -m jaeger_lyra --port 8794."""
from pathlib import Path
import runpy
import sys


def main():
    root=Path(__file__).resolve().parents[1]
    server=root/'voice-research/live_voicebox/server.py'
    if not server.is_file():raise SystemExit('Run this research demo from a JaegerLyra source checkout.')
    sys.path.insert(0,str(server.parent))
    runpy.run_path(str(server),run_name='__main__')


if __name__=='__main__':main()
