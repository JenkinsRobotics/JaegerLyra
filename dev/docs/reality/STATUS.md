# Jaeger Lyra — current status

Updated November 7, 2026. Version 0.1.0-alpha.1 is a research prototype, not a production TTS/STT release.

This alpha is an R&D handoff. Further model development is deferred.

## Working

- Self-contained local reconstruction demo: `python -m jaeger_lyra --port 8794`.
- Independent neural waveform decoding of prepared Kokoro-derived acoustic controls.
- Eighteen whole-word captures, explicit held-vowel mode and nine phrase comparisons.
- Pitch, volume, experimental resonance/brightness, pace and vibrato controls.
- Automatic whole-word completion, cancellation, bounded browser audio queues and no autoplay.
- Included hash-checked demo bundle. No avatar application, Kokoro model, network inference or account required at playback.
- Source/reconstruction labels and implemented-versus-planned architecture are visible in the demo.
- Research, legacy reference infrastructure and local evidence migrated into this repository; old avatar paths link here.

## Not established

- Natural independent generation of unseen speech or new articulatory sequences.
- General singing, short-reference voice cloning or robust speaker matching.
- Semantic audio understanding or a unified end-to-end audio reasoning model.
- A general non-backpropagation learning solution, strict clean-room provenance or Kokoro-level new-speech quality.
- A JaegerOS module/slot adapter or installable production inference API (planned).

CI uses a Linux runner; the initially inherited macOS runner configuration was rejected by GitHub before any test steps. Local validation was performed on Apple Silicon.

## Validation

Local alpha checks passed: **163 core + 14 runtime/colour + 26 legacy-adapter + 6 browser-audio = 209 automated tests**. All 119 bundled file hashes passed, as did nine neutral/vibrato phrase checks, isolated-checkout playback, complete-word/hold HTTP checks and 198 shaped catalogue renders. Silent browser checks of a word and a shaped phrase completed automatically with zero observed underruns.

The migration verifier checks the packaged file hashes, all core numerical tests, live runtime tests, browser PCM tests, neutral reconstruction and actual phrase-vibrato effects. Live HTTP and silent browser checks are recorded in the local `evidence/lyra-verification/` directory. See the final migration validation record for observed counts and timings; historical research measurements remain in their original files.

The demo's runtime dependencies are pinned in `requirements.txt`. Optional historical experiments keep their original separate lock files and model/data notices.
