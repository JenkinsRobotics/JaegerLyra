"""Explicit lossless frame reference for output-resynthesis comparisons.

This retains source waveform frames. It is not a neural model, compression,
independent sound generation, or an implementation used by the app voice.
"""
import numpy as np
from types import SimpleNamespace
from neural_waveform import ControlMatrix, SIZE, waveform_frames, overlap_add
from neural_performance import EncodedPerformance


class PcmFrameDecoder:
    config = SimpleNamespace(history=SIZE + 1)
    def frame(self, values, gain):
        return np.asarray(values) * gain

    def render(self, matrix):
        return overlap_add(matrix.values * matrix.gains[:, None], matrix.sample_count)


def pcm_performance(audio, pulses):
    values, gains = waveform_frames(np.asarray(audio, dtype=float))
    return EncodedPerformance(ControlMatrix(values, gains, len(audio)), pulses.copy())
