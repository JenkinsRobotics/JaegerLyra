# Jaeger Lyra — current status

Updated November 7, 2026. Version 0.1.0-alpha.1 is a research prototype, not a production TTS/STT release.

This alpha is an R&D handoff. Further model development is deferred.

Duet playback correction: “Both voices” now schedules two independent Web Audio sources from the lead and response files on the same clock. Each has its own gain, mute and download. Solo changes do not restart or merge the sources; play, pause, seeking and Stop act on both. The previous merged WAV remains only as a local diagnostic artifact and is not requested by the player. Seven transport tests cover shared scheduling, separate buffers/gains, solo without restart, synchronized seeking, pause/stop, cancellation while loading and paired completion/failure.

Post-alpha local demo addition: a prepared two-voice, vocals-only singing study. Natural pronunciation is generated offline by the optional Kokoro environment; the fixed Lyra decoder and a new piecewise timing/pitch renderer produce the sung stems. No model weights were trained. The music video, supplied lyrics, prepared controls and audio output remain local. The melody is an authored arrangement, not an exact transcription of the recording. Browser playback uses saved output with solo selection, phrase seeking and automatic completion. This does not establish independent text-to-singing or artist voice imitation.

Local duet validation: six scored-renderer tests and seven independent-player tests pass alongside the existing 209 tests (222 total). The 30.0875-second render contains 57 scored vowels; all measured median vowel-pitch errors are below 13 cents using broad-range Praat estimation without octave correction. Both stems sum to the diagnostic mix within PCM24 rounding error; peak amplitude is 0.558, with no clipping. Local HTTP hashes, byte-range seeking and refusal of non-allowlisted media paths pass. These measurements establish note control and transport behavior, not human-rated singing quality. The current singing was judged too robotic; naturalness remains unresolved. Optional preparation requires the separate local MLX/Kokoro environment; ordinary Lyra playback still uses its own `.venv`.

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

**macOS is a core supported platform.** Local validation passed on macOS 26.6.2 / Apple Silicon. CI now covers `macos-26` ARM64 and Linux independently with fail-fast disabled. The earlier `macos-14` configuration was rejected by GitHub; that runner configuration issue did not remove Mac application support. Intel Mac compatibility remains unverified.

## Validation

Final local alpha handoff checks passed: **169 core + 14 runtime/colour + 26 legacy-adapter + 6 browser-audio + 7 independent-player = 222 automated tests**. All 119 bundled file hashes and nine neutral/vibrato phrase checks passed again. Isolated-checkout playback works without the local research archives or duet assets. Earlier complete-word/hold HTTP checks, 198 shaped catalogue renders and silent browser checks of a word and a shaped phrase passed, with automatic completion and zero observed browser underruns.

The migration verifier checks the packaged file hashes, all core numerical tests, live runtime tests, browser PCM tests, neutral reconstruction and actual phrase-vibrato effects. Live HTTP and silent browser checks are recorded in the local `evidence/lyra-verification/` directory. See the final migration validation record for observed counts and timings; historical research measurements remain in their original files.

The demo's runtime dependencies are pinned in `requirements.txt`. Optional historical experiments keep their original separate lock files and model/data notices.

Hosted verification passed on both **macOS 26 ARM64 and Linux** for code revision `c0f5185`, including a fresh isolated checkout and complete-word reconstruction: [verification run](https://github.com/JenkinsRobotics/JaegerLyra/actions/runs/42412672746). The fresh macOS runner exposed the harness's overly short ten-second startup limit. The harness now allows up to 60 seconds for cold imports and bundle loading, records actual startup time, bounds HTTP requests and retains the server log on failure. This startup allowance does not change audio generation or playback timing.
