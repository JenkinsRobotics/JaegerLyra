# Jaeger Lyra reconstruction playback demo

Standalone R&D alpha at **http://127.0.0.1:8794/**, owned by JaegerLyra. This demonstrates reconstruction playback of captured Kokoro-derived performances; the avatar studio is a separate client.

The demo exposes the existing independent neural waveform decoder as a playable output instrument. **Whole word is the default.** Choose Heart, Michael or Bella and one of six words, then press **Reconstruct word**. It reconstructs the original complete word and finishes automatically. At neutral settings, every decoded sample is preserved in order. Pitch is selected before playback; volume can change during playback. Stop or Escape interrupts either mode.

**Voice character:** resonance (0.85–1.18× spectral envelope), brightness (±4 dB/octave), pace (0.75–1.35×), and vibrato (0–60 cents at 5 Hz) are selected before playback. Warm, Light and Gentle vibrato provide starting points. Reset controls restores natural pitch and all four neutral shaping values. Pace changes duration independently of pitch; pitch alone preserves duration. These four controls also apply to **Reconstruct phrase live**, while the recorded A/B comparisons stay fixed. Hold mode supports resonance, brightness and vibrato; pace applies only to complete words and phrases.

`voice_colour.py` is a new experimental DSP wrapper over decoded windows. It shifts a smoothed spectral envelope while retaining the harmonic frequency grid/phase, and applies a separate brightness tilt. It is not a physical vocal-tract simulation or learned speaker matching. Frame energy is normalized and rare peaks above 0.92 are smoothly bounded below 0.98; the current catalogue sweep did not activate that peak protection. All neutral processing is bypassed. Timing/vibrato reuse the existing pulse-synchronous performance stream. No model was retrained.

**Hold a note** is an explicit alternative for sustained vowels and live pitch bends. In that mode, A/S/D/F/G/H/J/K play notes and **Finish word** releases the held vowel into its ending. Early release waits for the onset. Holding is limited to one minute. This mode deliberately uses the earlier stretched source units; ordinary playback uses separate, naturally timed word captures.

Nine natural-speech pairs compare Kokoro originals with saved/reopened neural reconstructions, using matched whole-clip loudness and hidden labels. **Reconstruct phrase live** decodes a prepared phrase in small packets. Kokoro produced the source recordings; their detailed acoustic information was encoded offline. Kokoro weights and original WAVs are absent from the live synthesis path, but the captured performances remain Kokoro-derived. This is not independently generated new speech. Playback starts only on an explicit button press; the Silent check checkbox mutes the live monitor while keeping the audio processor running. It does not mute the separate A/B players.

## Launch and checks

From the `JaegerLyra` checkout:

```sh
.venv/bin/python -m jaeger_lyra --port 8794
```

The current self-contained bundle is `demo/bundle/manifest.json`; original v1/v2 evidence is retained locally. Regeneration is offline and uses existing research fixtures:

```sh
.seed-vc-venv/bin/python voice-research/live_voicebox/prepare.py
.research-venv/bin/python voice-research/live_voicebox/prepare_words.py
.research-venv/bin/python -m unittest discover -s voice-research/live_voicebox -p 'test_*.py' -v
node --test voice-research/live_voicebox/test_pcm.mjs
.research-venv/bin/python -u voice-research/live_voicebox/check_words.py
.research-venv/bin/python -u voice-research/live_voicebox/check_shapes.py
```

The HTTP check requires the server to be running and no active browser performance. It verifies automatic completion and sample fidelity for all 18 words without a release command, checks pitched words at the same duration, and verifies that holding occurs only in the explicit sustain mode. It never plays audio. The older `check_http.py` covers the separate prepared-phrase and sustain transport paths.

## Architecture and limits

- `prepare.py` verifies and compiles captured controls. Original WAVs are only required offline and for the A/B comparison endpoint.
- `runtime.py` loads the frozen independent decoder and saved continuous controls, without Kokoro weights or original WAVs in the synthesis path. It retains dense captured acoustic detail; this is neither compact articulatory modeling nor general voice cloning.
- `server.py` binds loopback only. It synthesizes 5 ms blocks and sends paced 20 ms float32 PCM packets over HTTP. At most two sessions may be active. Controls carry sequence numbers to reject stale updates.
- The browser AudioWorklet uses a bounded two-second ring with a 40 ms initial buffer and 10 ms start/stop/recovery fades. Held-voice synthesis has an additional 20 ms waveform delay. Preparation, transport, buffering and device latency must be distinguished from decoder compute time.
- The independent model, source-derived fixture preparation, and accepted Kokoro speech baseline remain distinct. No new neural training or predictive-coding claim is made by this demo.

## Verified November 7, 2026

**Voice-character controls:** 14 Python tests and six browser-audio tests pass. Synthetic vowel tests verify that resonance moves spectral peaks while the 100 Hz harmonic spacing stays fixed; brightness changes high/low energy balance. Pace and vibrato match the separate offline renderer. All 198 catalogue renders (18 words × 11 settings) finish with finite, unclipped output; neutral settings retain the earlier reconstruction. Four HTTP checks cover shaped Palm in all three voices and one shaped phrase. The sweep's maximum p95 block computation was 1.22 ms; HTTP checks observed up to 3.30 ms. These are individual compute measurements, not a hard real-time guarantee. Evidence is in `evidence/live-voicebox/voice-character-v3/`. Listening acceptance of the new colour edits remains open.

Conceptual background: [Praat's independent pitch, formant and duration controls](https://fon.hum.uva.nl/praat/manual/Sound__Change_gender___.html) and [DDSP's separation of pitch, loudness and timbre](https://magenta.tensorflow.org/ddsp). The colour wrapper is project DSP code and does not implement or train DDSP.

**Whole-word correction:** 18 original words are prepared separately from the stretched sustain fixtures. Quiet leading/trailing edges are trimmed with 20 ms padding; the interior is unchanged. Durations are **0.417–0.618 seconds**. Eight runtime tests and six browser-audio tests pass. HTTP checks confirm all 18 native words reproduce their offline neural reconstructions and complete automatically, plus three pitched Palm examples retain exactly the same sample counts. An explicit hold still waits for release. Muted browser checks of Heart/Palm at native pitch and +3 semitones returned to **Word finished**, with zero observed underruns and an empty queue. Evidence is under `evidence/live-voicebox/word-playback-v2/`.

The following describes the earlier v1 validation:

Five runtime tests and six JavaScript tests pass. All nine HTTP speech streams match saved decoder reconstructions within **4.36e-8** maximum sample difference; a three-event live control sequence matches its deterministic replay exactly. Explicit Stop and disconnect both close sessions. All 18 comparison WAV hashes match the manifest.

Muted browser checks exercised changing notes, keyboard controls, release, natural speech, Escape and comparison labels. Observed first-packet delivery was **38 ms** for the first held voice and **30 ms** for the speech phrase, with zero observed buffer underruns. These are individual local checks, not end-to-end acoustic latency guarantees. The HTTP run observed 12 blocks above the 5 ms compute budget, with a maximum of **8.49 ms**, so a hard real-time guarantee is not established.

The nine speech reconstructions have median waveform SNR **32.91 dB** against their Kokoro sources. This is a reused evaluation panel and a waveform metric, not listening acceptance or proof of Kokoro-level new speech. Recorded evidence, screenshots and test logs are in `evidence/live-voicebox/v1/`. Quality parity, natural new articulation, broad singing and short-reference cloning remain unresolved.

The engineering curriculum remains: faithful reconstruction; stable notes and expressive pitch; vowels and breath; consonant attacks/releases; smooth new syllable transitions; unseen connected speech and singing. Chords can be tested with multiple voice instances later. This sequence is a development strategy, not evidence that the model learns like a human.

## Alpha migration

The repository README is the standalone setup entry point. Run `python scripts/verify.py` after installing `requirements-dev.txt`. Packaging adds offline pulse timing to the nine phrase artifacts while preserving their encoded acoustic matrices, allowing phrase vibrato to take effect. The R&D snapshot does not extend model development beyond migration and correctness checks.
