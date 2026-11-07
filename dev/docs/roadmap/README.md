# Lyra roadmap

The stages below are planned research work, not implemented product features. Current implementation status is maintained in [STATUS](../reality/STATUS.md).

1. **Output fidelity and expressive control:** evaluate reconstruction and manipulation on unseen recordings, preserve consonant timing and speaker detail, and measure audible artifacts plus compute/device latency.
2. **Performance generation:** generate smooth new control trajectories and articulatory transitions rather than relying on a complete captured utterance. Test unseen words and connected speech.
3. **Voice adaptation and singing:** separate speaker characteristics from content/prosody, test short-reference adaptation, and add controllable note durations, transitions and articulation.
4. **Ears and perception:** develop streaming acoustic representations and evaluate what information is retained, then add audio meaning/context inference.
5. **Audio-to-audio integration:** connect perception to the controller and waveform decoder, with interruption and timing control. Evaluate local/predictive learning against explicit baselines.
6. **Application integration:** establish a tested JaegerOS adapter and versioned API for Aether Avatar and other clients.
