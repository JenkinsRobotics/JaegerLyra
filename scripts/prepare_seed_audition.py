"""Loudness-match existing synthetic comparison clips without automatic playback.

Only a fixed gain is applied to each whole clip. No denoising, pitch correction,
limiting, clipping or selection of a favorable subsection conceals differences.
"""
import json
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "evidence/seed-vc/audition"
CLIPS = {
    "heart-speaking-reference": "evidence/voice-fixtures/af_heart-reference.wav",
    "heart-original-singer": "evidence/soulx/kokoro-reference-mps-32steps-cfg3.wav",
    "heart-reference-converted": "evidence/seed-vc/validation-30steps-cfg0.7/heart-second-song.wav",
    "michael-speaking-reference": "evidence/voice-fixtures/am_michael-reference.wav",
    "michael-original-singer": "evidence/integrated-singer/michael-spoken.wav",
    "michael-reference-converted": "evidence/seed-vc/validation-30steps-cfg0.7/michael-song.wav",
}


def main():
    OUT.mkdir(exist_ok=True, parents=True)
    loaded = {}
    for name, relative in CLIPS.items():
        audio, rate = sf.read(ROOT / relative)
        lufs = pyln.Meter(rate).integrated_loudness(audio)
        peak = np.abs(audio).max()
        assert np.isfinite(lufs) and peak > 0
        loaded[name] = (audio, rate, lufs, peak)
    target = min(-20., *(lufs + 20 * np.log10(.9 / peak) for _, _, lufs, peak in loaded.values()))
    rows = []
    for name, (audio, rate, lufs, peak) in loaded.items():
        gain = 10 ** ((target - lufs) / 20)
        output = OUT / f"{name}.wav"
        sf.write(output, audio * gain, rate, subtype="PCM_24")
        saved, _ = sf.read(output)
        measured = pyln.Meter(rate).integrated_loudness(saved)
        assert abs(measured - target) < .01 and np.abs(saved).max() <= .901
        rows.append({"name": name, "source": CLIPS[name], "file": str(output.relative_to(ROOT)),
            "originalLUFS": lufs, "gain": gain, "outputLUFS": measured,
            "samplePeak": float(np.abs(saved).max()), "duration": len(saved) / rate})
    report = {"purpose": "Whole-clip fixed-gain listening comparison; no listener acceptance yet",
        "targetLUFS": target, "algorithm": "pyloudnorm integrated BS.1770 loudness; common level chosen for <=0.9 sample peaks",
        "truePeakMeasured": False, "files": rows}
    (OUT / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
