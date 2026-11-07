# Lyra's audio path

The long-term framework has four responsibilities. They are conceptual boundaries inside one project, not four separate repositories or a promise of four separately trained networks.

1. **Audio encoder — ears.** Accept waveform samples and preserve acoustic detail in continuous features: voicing, pitch, rhythm, spectral structure and speaker cues. Today's encoder is an offline acoustic representation for reconstruction; it is not a semantic hearing system.
2. **Perception and inference (planned).** Interpret sound and maintain context: what happened, who is speaking, what a phrase means and how it was expressed. A waveform codec or pitch tracker alone cannot establish this capability.
3. **Performance controller.** Produce a time-varying acoustic plan. It needs to preserve or generate articulation, timing, pitch, intensity, resonance and source texture. Today, captured controls and manual transformations supply that plan. Generating a new, natural utterance is planned.
4. **Waveform decoder — voicebox.** Convert the plan into waveform samples. This is the part currently demonstrated through reconstruction and controlled resynthesis.

## Implemented path

`neural_waveform.py` supplies the experimental continuous encoder and neural decoder. `compiled_neural_decoder.py` executes the selected sparse-input decoder without changing its weights. The live runtime reads prepared acoustic matrices, not raw reference audio. `neural_performance_stream.py` adds pulse-synchronous pitch, duration and vibrato; `voice_colour.py` applies experimental spectral-envelope and brightness edits. Neutral settings bypass these edits.

Original recordings remain available to the A/B comparison endpoint. They are not the live synthesis input. This distinction is deliberate: independent waveform reconstruction is working; independent generation of the performance controls is still open.

The demo binds loopback and sends 24 kHz mono float32 PCM: 5 ms computation blocks, grouped into paced 20 ms HTTP chunks. Browser playback uses an AudioWorklet and a bounded queue. See the live demo README for implementation details and measured limits.

## Planned audio-to-audio path

Audio encoder → perception/state → controller → waveform decoder. This can preserve prosody and other acoustic information without forcing every interaction through a transcript. Optional text, scripts and musical scores can be additional inputs to the controller. A feedback/evaluation path can compare the generated waveform with intended properties.

Research into local learning and predictive coding is already archived in `voice-research/audio_native/`. It does not establish that the full framework learns like a human, that backpropagation is unnecessary for all components, or that a small set of pitch/formant controls suffices to imitate any speaker.

## Integration boundary

Aether Avatar is a client of Lyra's audio capability. JaegerKokoroTTS and JaegerWhisperSTT remain separate reference/adapter projects. Lyra does not rename those pretrained models or advertise their capabilities as its own. A JaegerOS module/slot adapter and a stable versioned inference API are planned after the experimental audio path is established.
