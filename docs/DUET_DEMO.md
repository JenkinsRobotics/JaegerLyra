# Local two-voice singing study

The optional panel at `http://127.0.0.1:8794/#duet` plays a prepared, dry two-voice performance. Both voices means **two independent vocal tracks running simultaneously**, each with its own audio source, volume and mute. They share one audio clock and start/seek at an identical offset. Solo selection only changes their individual gains, keeping both tracks synchronized. Separate voice WAV downloads are available. There is no autoplay. Escape or Stop resets both voices; other app playback pauses both.

This is a **hybrid text-and-score experiment**:

1. `scripts/prepare_duet_speech.py` uses the optional existing MLX/Kokoro environment to synthesize natural speech and predicted phone durations from supplied lyrics.
2. `scripts/render_duet.py encode` encodes those synthetic recordings with the frozen Lyra encoder, captures pulses and prepares a score. It corrects the existing Kokoro pipeline's three-frame timestamp offset and uses nearby actually voiced nuclei to avoid stretching plosive gaps or vocal fry into long notes. This is prepared source assistance, not general phoneme recognition.
3. `scripts/render_duet.py render` opens only the frozen Lyra model, encoded controls and score. Its decoder generates waveform samples; `scored_resynthesis.py` applies piecewise articulation timing and absolute pitch using pulse-synchronous DSP. No original-song PCM, Kokoro model or synthetic source WAV is loaded in this render stage.
4. The browser fetches only `lead.wav` and `reply.wav` and decodes them into distinct buffers. It schedules two Web Audio sources on the same clock and routes each through its own gain. It does not load the premixed `duet.wav` or construct a combined buffer. The sources meet only at the audio output. This is synchronized playback of prepared voices, not text-to-singing or live neural synthesis on a button press.

The current local arrangement uses two synthetic voices and a custom E-minor melody at 96 BPM. It is not an exact transcription of the supplied recording and does not clone the original performers. Pitch checks assess note control, not naturalness or faithful pronunciation. Listening acceptance remains a separate user judgment.

## Local preparation

The local score and all generated assets are in the ignored `evidence/duet-demo/v1/` folder. Supplied media is in ignored `media/`. Neither the song-specific lyrics nor recordings are part of the public demo bundle. A fresh checkout simply hides this optional panel and keeps the ordinary reconstruction demo working.

```bash
# Optional offline pronunciation preparation; needs the separately installed
# MLX/Kokoro environment and local reference model on Apple Silicon.
.voice-venv/bin/python scripts/prepare_duet_speech.py evidence/duet-demo/v1/score.json

# These stages use the standalone Lyra environment.
.venv/bin/python scripts/render_duet.py encode evidence/duet-demo/v1
.venv/bin/python scripts/render_duet.py render evidence/duet-demo/v1
.venv/bin/python -m jaeger_lyra --port 8794
# In another terminal, with the server running:
.venv/bin/python scripts/check_duet.py evidence/duet-demo/v1
```

The score JSON contains `title`, `attribution`, `arrangement`, `tempo_bpm`, and `phrases`. Each phrase has a unique `id`, `part` (`lead` or `reply`), a supported Kokoro `speaker`, `text`, `start_beat`, `duration_beats` and `notes_midi` (one note for each predicted vowel). This is an internal experimental file, not a stable external interface. The current vowel planner is limited to short English phrases with usable voiced nuclei and rejects unsupported captures.

Outputs include `output/lead.wav`, `output/reply.wav` and a manifest recording provenance, note timing and hashes. The earlier offline `output/duet.wav` remains a diagnostic artifact, unused by the player. The render uses fixed phrase gain, short edge fades and modest stereo placement. It adds no instruments, backing track, reverberation, vocal doubling or artist recording. Only these named output files are exposed by the loopback server.
