"""Independent diagnostics for Seed-VC: lyrics, two pitch estimators, identity.

Run in .research-venv after conversion exits, so the evaluators do not share GPU
memory with synthesis. ASR and ECAPA are evaluators only, never synthesis inputs.
"""
import argparse
import json
import os
from pathlib import Path
import re
import urllib.request

os.environ["HF_HUB_OFFLINE"] = "1"
import numpy as np
import parselmouth
import pyworld
import soundfile as sf
import torch
import torchaudio
from speechbrain.inference.classifiers import EncoderClassifier

ROOT = Path(__file__).resolve().parents[1]


def errors(expected, actual):
    a, b = [re.findall(r"[a-z']+", text.lower()) for text in (expected, actual)]
    previous = list(range(len(b) + 1))
    for i, word in enumerate(a, 1):
        current = [i]
        for j, heard in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[-1] + 1,
                               previous[j - 1] + (word != heard)))
        previous = current
    return {"wordErrors": previous[-1], "referenceWords": len(a),
            "wer": previous[-1] / max(1, len(a))}


def transcribe(path):
    request = urllib.request.Request("http://127.0.0.1:8791/api/transcribe", path.read_bytes(),
                                     {"Content-Type": "audio/wav"})
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)["text"]


def pitch_tracks(audio, sr):
    sound = parselmouth.Sound(audio, sr).to_pitch_ac(time_step=.01, pitch_floor=65, pitch_ceiling=800)
    f0, time = pyworld.harvest(audio.astype(np.float64), sr, f0_floor=65, f0_ceil=800, frame_period=10.)
    f0 = pyworld.stonemask(audio.astype(np.float64), f0, time, sr)
    return {"praat": (sound.xs(), sound.selected_array["frequency"]), "world": (time, f0)}


def note_errors(score, tracks):
    result = {key: [] for key in tracks}
    at = 0.
    for word, duration, midi in zip(score["text"].split(), map(float, score["duration"].split()),
                                     map(int, score["note_pitch"].split())):
        if midi:
            hz = 440 * 2 ** ((midi - 69) / 12)
            for name, (times, pitch) in tracks.items():
                window = pitch[(times >= at + duration * .25) & (times < at + duration * .75)]
                voiced = window[window > 0]
                cents = 1200 * np.log2(voiced / hz)
                result[name].append({"word": word, "midi": midi, "start": at,
                    "voicedFraction": len(voiced) / max(1, len(window)),
                    "medianAbsoluteCents": float(np.median(np.abs(cents))) if len(voiced) else None,
                    "medianSignedCents": float(np.median(cents)) if len(voiced) else None})
        at += duration
    return {name: {"notes": notes,
        "medianNoteErrorCents": float(np.median([x["medianAbsoluteCents"] for x in notes if x["medianAbsoluteCents"] is not None])),
        "meanNoteVoicedFraction": float(np.mean([x["voicedFraction"] for x in notes]))}
        for name, notes in result.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    args = parser.parse_args()
    folder = ROOT / args.folder
    source_report = json.loads((folder / "report.json").read_text())
    torch.set_num_threads(4)
    modeldir = ROOT / "models/speaker-eval"
    evaluator = EncoderClassifier.from_hparams(source=str(modeldir), savedir=str(modeldir),
        overrides={"pretrained_path": str(modeldir)}, run_opts={"device": "cpu"})
    cache = {}

    def embedding(path):
        if path not in cache:
            audio, rate = sf.read(path)
            value = torch.from_numpy(audio.astype(np.float32)).unsqueeze(0)
            value = torchaudio.functional.resample(value, rate, 16000)
            with torch.inference_mode():
                value = evaluator.encode_batch(value).flatten()
            cache[path] = value / torch.linalg.vector_norm(value)
        return cache[path]

    report = {"scope": "Synthetic-voice diagnostics, no subjective acceptance or general cloning claim",
        "identityMetric": "ECAPA cosine: speech-trained, not a validated singing identity/naturalness measure",
        "pitchMetric": "Praat and WORLD, 65–800 Hz, no octave correction; score uses middle half of each note",
        "conversionReport": str((folder / "report.json").relative_to(ROOT)), "rows": []}
    for trial in source_report["rows"]:
        reference = ROOT / trial["reference"]
        anchor = embedding(reference)
        heldout = ROOT / f"evidence/voice-fixtures/{trial['targetSpeaker']}-heldout.wav"
        row = {"name": trial["name"], "expectedText": trial["expectedText"],
               "targetHeldoutIdentity": float(torch.dot(embedding(heldout), anchor))}
        tracks = {}
        for role in ("source", "converted"):
            path = ROOT / trial["source" if role == "source" else "file"]
            audio, sr = sf.read(path)
            tracks[role] = pitch_tracks(audio, sr)
            text = transcribe(path)
            item = {"file": str(path.relative_to(ROOT)), "text": text,
                    **errors(trial["expectedText"], text), "duration": len(audio) / sr,
                    "identityCosine": float(torch.dot(embedding(path), anchor)),
                    "peak": float(np.abs(audio).max()), "fullScaleFraction": float(np.mean(np.abs(audio) >= .9999))}
            if "score" in trial:
                item["scorePitch"] = note_errors(trial["score"], tracks[role])
            row[role] = item
        row["sourcePitchPreservation"] = {}
        for name in tracks["source"]:
            st, sf0 = tracks["source"][name]
            ot, of0 = tracks["converted"][name]
            mask = (st >= ot[0]) & (st <= ot[-1])
            st, sf0 = st[mask], sf0[mask]
            nearest = np.clip(np.searchsorted(ot, st), 1, len(ot) - 1)
            nearest -= (np.abs(ot[nearest - 1] - st) < np.abs(ot[nearest] - st)).astype(int)
            of0 = of0[nearest]
            joint = (sf0 > 0) & (of0 > 0)
            cents = 1200 * np.log2(of0[joint] / sf0[joint])
            row["sourcePitchPreservation"][name] = {
                "voicingAgreement": float(np.mean((sf0 > 0) == (of0 > 0))),
                "jointVoicedFrames": int(joint.sum()),
                "medianAbsoluteCents": float(np.median(np.abs(cents))) if len(cents) else None,
                "medianSignedCents": float(np.median(cents)) if len(cents) else None}
        report["rows"].append(row)
        (folder / "evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({"case": row["name"],
            "sourceIdentity": row["source"]["identityCosine"],
            "convertedIdentity": row["converted"]["identityCosine"],
            "convertedWordErrors": row["converted"]["wordErrors"],
            "pitch": row["sourcePitchPreservation"]}), flush=True)


if __name__ == "__main__":
    main()
