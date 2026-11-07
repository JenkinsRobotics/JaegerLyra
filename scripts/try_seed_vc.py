"""Source-derived, offline singing conversion using a pinned Seed-VC checkout.

Run with .seed-vc-venv/bin/python after prepare_seed_vc.py. Keeps upstream source
unchanged, resolves all weights locally and records complete-clip latency. No
transcript is consumed by conversion. This is not the independent voice engine.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_HOME"] = str(ROOT / "models/seed-vc/hf")
os.environ["MPLCONFIGDIR"] = str(ROOT / "models/seed-vc/matplotlib")
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.insert(0, str(ROOT / "voice-tools/seed-vc"))

import numpy as np
import soundfile as sf
import torch
import yaml
import inference as upstream

MODELS = ROOT / "models/seed-vc"
OUT = ROOT / "evidence/seed-vc"
CHECKPOINT = MODELS / "seed/DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth"


def local_weight(repo_id, model_filename="pytorch_model.bin", config_filename=None):
    folders = {"lj1995/VoiceConversionWebUI": "pitch", "funasr/campplus": "speaker"}
    path = MODELS / folders[repo_id] / model_filename
    if config_filename is not None or not path.is_file():
        raise ValueError(f"Unprepared weight: {repo_id}/{model_filename}")
    return str(path)


def synchronize():
    if upstream.device.type == "mps":
        torch.mps.synchronize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=["initial", "validation"], default="initial")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cfg", type=float, default=.7)
    parser.add_argument("--case", help="Run only this named case within the suite")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    destination = OUT / f"{args.suite}-{args.steps}steps-cfg{args.cfg}"
    destination.mkdir(exist_ok=True)
    config = yaml.safe_load((MODELS / "seed/config_dit_mel_seed_uvit_whisper_base_f0_44k.yml").read_text())
    config["model_params"]["vocoder"]["name"] = str(MODELS / "vocoder")
    config["model_params"]["speech_tokenizer"]["name"] = str(MODELS / "content")
    config_path = destination / "local-paths.yml"
    config_path.write_text(yaml.safe_dump(config))
    upstream.load_custom_model_from_hf = local_weight
    checkpoint_audit = []
    original_loader = upstream.load_checkpoint

    def checked_load(model, optimizer, path, **kwargs):
        state = torch.load(path, map_location="cpu", weights_only=True, mmap=True)["net"]
        for name, module in model.items():
            supplied = {key.removeprefix("module."): value for key, value in state[name].items()}
            required = dict(module.named_parameters())
            missing = [key for key, parameter in required.items()
                       if key not in supplied or supplied[key].shape != parameter.shape]
            if missing:
                raise RuntimeError(f"Missing active checkpoint parameters: {name}: {missing}")
            expected = module.state_dict()
            unmatched = [key for key, value in supplied.items()
                         if key not in expected or expected[key].shape != value.shape]
            # Upstream expands this deterministic arange buffer from the older
            # checkpoint. It is not a missing trained positional parameter.
            buffer_issues = [key for key in expected if key not in required
                             and (key not in supplied or expected[key].shape != supplied[key].shape)]
            if buffer_issues:
                assert set(buffer_issues) <= {"estimator.input_pos", "estimator.t_embedder.freqs"}, buffer_issues
                if "estimator.input_pos" in buffer_issues:
                    key = "estimator.input_pos"
                    assert torch.equal(expected[key], torch.arange(len(expected[key])))
                    assert torch.equal(supplied[key], torch.arange(len(supplied[key])))
                if "estimator.t_embedder.freqs" in buffer_issues:
                    key = "estimator.t_embedder.freqs"
                    half = len(expected[key])
                    assert torch.equal(expected[key], torch.exp(-math.log(10000) * torch.arange(half, dtype=torch.float32) / half))
            checkpoint_audit.append({"module": name, "matchedParameterTensors": len(required),
                "missingActiveParameters": missing, "unusedOrResizedCheckpointEntries": unmatched,
                "deterministicRegeneratedBuffers": buffer_issues})
        del state
        return original_loader(model, optimizer, path, **kwargs)

    upstream.load_checkpoint = checked_load
    options = SimpleNamespace(f0_condition=True, fp16=False, checkpoint=str(CHECKPOINT),
        config=str(config_path), auto_f0_adjust=False, semi_tone_shift=0,
        diffusion_steps=args.steps, length_adjust=1.0, inference_cfg_rate=args.cfg,
        output=str(destination))
    torch.set_num_threads(4)
    torch.manual_seed(42)
    started = time.perf_counter()
    loaded = upstream.load_models(options)
    # RMVPE returns NumPy float64; MPS rejects conversion of that dtype.
    # Keep its estimates while representing Hz at float32 precision.
    original_f0 = loaded[2]
    def f0_float32(*values, **keywords):
        return np.asarray(original_f0(*values, **keywords), dtype=np.float32)
    loaded = (*loaded[:2], f0_float32, *loaded[3:])
    synchronize()
    load_seconds = time.perf_counter() - started
    upstream.load_models = lambda _: loaded
    cases = ([
        ("heart-song", "evidence/integrated-singer/heart-spoken.wav", "af_heart"),
        ("heart-speech-to-michael", "evidence/voice-fixtures/af_heart-heldout.wav", "am_michael"),
    ] if args.suite == "initial" else [
        ("heart-second-song", "evidence/soulx/kokoro-reference-mps-32steps-cfg3.wav", "af_heart"),
        ("michael-song", "evidence/integrated-singer/michael-spoken.wav", "am_michael"),
    ])
    if args.case:
        cases = [case for case in cases if case[0] == args.case]
        if not cases:
            raise ValueError(f"Unknown case in {args.suite}: {args.case}")
    report = {"scope": "Offline pretrained reference conversion, not clean-room or local-learning synthesis",
        "manifest": "licenses/seed-vc-research.json", "device": str(upstream.device),
        "precision": "CFM, RMVPE and vocoder float32; upstream Whisper encoder float16",
        "compatibilityAdapter": "Cast RMVPE output Hz from float64 to float32 before MPS transfer",
        "checkpointAudit": checkpoint_audit,
        "steps": args.steps, "cfg": args.cfg, "autoPitchAdjust": False, "semitoneShift": 0,
        "loadSeconds": load_seconds, "rows": []}
    (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"loadedSeconds": load_seconds, "device": report["device"]}), flush=True)
    for name, source_file, speaker in cases:
        source = ROOT / source_file
        reference = ROOT / f"evidence/voice-fixtures/{speaker}-reference.wav"
        options.source, options.target = str(source), str(reference)
        torch.manual_seed(42)
        np.random.seed(42)
        synchronize()
        started = time.perf_counter()
        upstream.main(options)
        synchronize()
        seconds = time.perf_counter() - started
        generated = destination / f"vc_{source.stem}_{reference.stem}_1.0_{args.steps}_{args.cfg}.wav"
        output = destination / f"{name}.wav"
        generated.replace(output)
        audio, rate = sf.read(output)
        assert np.isfinite(audio).all() and len(audio) > rate
        source_meta = json.loads(source.with_suffix(".json").read_text())
        row = {"name": name, "source": source_file, "reference": str(reference.relative_to(ROOT)),
            "targetSpeaker": speaker, "file": str(output.relative_to(ROOT)),
            "seconds": seconds, "duration": len(audio) / rate, "sampleRate": rate,
            "processingRealTimeFactor": seconds / (len(audio) / rate), "peak": float(np.abs(audio).max()),
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
            "sourceSha256": hashlib.sha256(source.read_bytes()).hexdigest(),
            "referenceSha256": hashlib.sha256(reference.read_bytes()).hexdigest()}
        if "score" in source_meta:
            row["score"] = source_meta["score"]
            row["expectedText"] = source_meta["score"]["lyrics"]
        elif "target" in source_meta:
            row["score"] = json.loads((ROOT / source_meta["target"]).read_text())[0]
        else:
            row["expectedText"] = source_meta.get("text")
        # Older prepared song metadata records its score in the shared target.
        if "song" in name and "score" not in row:
            row["score"] = json.loads((ROOT / "evidence/soulx/target.json").read_text())[0]
            row["expectedText"] = "You light up my day Stay here close to me"
        if "score" in row and "expectedText" not in row:
            row["expectedText"] = "You light up my day Stay here close to me"
        output.with_suffix(".json").write_text(json.dumps(row, indent=2) + "\n")
        report["rows"].append(row)
        (destination / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps({k: v for k, v in row.items() if k != "score"}), flush=True)


if __name__ == "__main__":
    main()
