"""Whole-clip listening comparison for the decoder capacity gate; never plays."""
import json
from pathlib import Path
import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'evidence/source-detail/lpc-v2/audition'
CLIPS={
    'kokoro-original':'evidence/source-detail/fresh-inputs/af_heart-0.wav',
    'voicebox-before-fix':'evidence/source-detail/fresh/af_heart-0-baseline.wav',
    'voicebox-corrected':'evidence/source-detail/lpc-v2/fresh/af_heart-0-baseline.wav',
    'voicebox-colored-noise':'evidence/source-detail/lpc-v2/fresh/af_heart-0-colored-noise.wav',
}


def main():
    OUT.mkdir(parents=True,exist_ok=True);loaded={}
    for name,file in CLIPS.items():
        audio,rate=sf.read(ROOT/file);lufs=pyln.Meter(rate).integrated_loudness(audio)
        loaded[name]=(audio,rate,lufs,float(abs(audio).max()))
    target=min(-20.,*(lufs+20*np.log10(.9/peak) for _,_,lufs,peak in loaded.values()))
    rows=[]
    for name,(audio,rate,lufs,peak) in loaded.items():
        gain=10**((target-lufs)/20);path=OUT/(name+'.wav')
        sf.write(path,audio*gain,rate,subtype='PCM_24');saved,_=sf.read(path)
        measured=pyln.Meter(rate).integrated_loudness(saved)
        assert abs(measured-target)<.01 and abs(saved).max()<=.901
        rows.append({'name':name,'source':CLIPS[name],'file':str(path.relative_to(ROOT)),
            'gain':gain,'originalLUFS':lufs,'outputLUFS':measured,'samplePeak':float(abs(saved).max())})
    report={'scope':'Matched-loudness reconstruction auditions, not new words or accepted naturalness',
        'selection':'First fresh Heart sentence, selected by index; whole clips',
        'targetLUFS':target,'method':'Integrated BS.1770, fixed gain only; no denoising, limiting or pitch correction',
        'truePeakMeasured':False,'files':rows}
    (OUT/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
