# Model and audio provenance

## Included reconstruction bundle

`demo/bundle/manifest.json` identifies the selected decoder, each prepared acoustic matrix, source/reconstruction recordings, and their SHA-256 values. `demo/bundle/checksums.json` covers every packaged file. `scripts/verify.py` checks those hashes and exercises the decoder.

The independent decoder is original project code with locally fitted weights. Its research training audio and the demonstrated source performances were generated with existing Kokoro weights. It is therefore **Kokoro-derived research**, not an independent speech corpus. The live decoder does not load Kokoro weights and does not replay source WAVs. Reference players intentionally play WAVs for comparison.

The bundle contains synthetic Heart, Michael and Bella examples from the existing experiments. No newly uploaded personal voice recording was added during this migration. Original manifests and research provenance remain in the local archive. `voice-research/audio_native/PROVENANCE.json` records the research methods, external references and limits.

During packaging, the nine phrase artifacts gained offline Praat pulse timing from their own source recordings so that vibrato is effective. Their encoded acoustic matrices and the selected decoder weights were preserved. Neutral reconstruction is checked against the earlier saved reconstructions.

## External reference infrastructure

Kokoro, GPT-SoVITS, OpenVoice, SoulX-Singer, Seed-VC, Whisper and other downloaded comparison components retain their existing provenance records in `licenses/`. Downloaded model weights, upstream source checkouts, caches, reference uploads and virtual environments are local-only and ignored by Git. They are not required for the packaged reconstruction demo.

Historical evaluation includes LibriSpeech material; its archive, attribution and selection records remain local, with notices under `licenses/`. LibriSpeech audio is not included in the prepared demo bundle.

The original implementation notes explicitly do not claim strict clean-room development. The repository's Apache-2.0 license does not replace third-party model/data terms.
