"""Pin the official Seed-VC singing comparator and its inference checkpoints.

Separate research only; no account/gate acceptance, mirror, or app-default change.
The source checkout is recorded independently from model repository revisions.
"""
import hashlib
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "models/seed-vc"
SOURCES = [
    ("seed", "Plachta/Seed-VC", "257283f9f41585055e8f858fba4fd044e5caed6e", "GPL-3.0", [
        "DiT_seed_v2_uvit_whisper_base_f0_44k_bigvgan_pruned_ft_ema_v2.pth",
        "config_dit_mel_seed_uvit_whisper_base_f0_44k.yml", "README.md"]),
    ("pitch", "lj1995/VoiceConversionWebUI", "e6d0c1a17da07c33557852f9dfa2bd44cc75737d", "MIT (repository declaration)", ["rmvpe.pt", "README.md"]),
    ("speaker", "funasr/campplus", "e4b6ede7ce16997aff4ae69fbca1f0175e2afede", "Apache-2.0", ["campplus_cn_common.bin", "README.md"]),
    ("vocoder", "nvidia/bigvgan_v2_44khz_128band_512x", "95a9d1dcb12906c03edd938d77b9333d6ded7dfb", "MIT", ["config.json", "bigvgan_generator.pt", "LICENSE", "README.md"]),
    ("content", "openai/whisper-small", "973afd24965f72e36ca33b3055d56a652f456b4d", "Apache-2.0", ["config.json", "model.safetensors", "preprocessor_config.json", "README.md"]),
]


def download(source):
    folder, repo, revision, license_name, files = source
    record = {"repository": repo, "revision": revision,
              "upstreamLicenseDeclaration": license_name, "files": []}
    for name in files:
        path = Path(hf_hub_download(repo_id=repo, filename=name, revision=revision,
                                   local_dir=DEST / folder))
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        record["files"].append({"path": str(path.relative_to(ROOT)),
                                "bytes": path.stat().st_size, "sha256": digest})
        print(json.dumps({"downloaded": f"{repo}/{name}", "bytes": path.stat().st_size}), flush=True)
    return record


def main():
    with ThreadPoolExecutor(max_workers=3) as pool:
        models = list(pool.map(download, SOURCES))
    report = {"name": "Seed-VC v1 singing conversion comparator", "cleanRoom": False,
              "code": "https://github.com/Plachtaa/seed-vc", "codeLicense": "GPL-3.0",
              "commit": subprocess.check_output(["git", "-C", str(ROOT / "voice-tools/seed-vc"), "rev-parse", "HEAD"], text=True).strip(),
              "models": models, "use": "Isolated research, no app integration",
              "limits": "Conventional pretrained models. Whisper continuous encoder features, no transcript at conversion; quantized F0 conditioning. Not the independent local-learning audio core."}
    (ROOT / "licenses/seed-vc-research.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
