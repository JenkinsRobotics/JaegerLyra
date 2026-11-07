"""Bounded neural output from a stream of continuous control frames.

No source audio, future control sequence or full utterance duration is accepted.
The producer supplies the same overlapping frame convention as neural_waveform:
one priming frame centered at time zero, then one frame for every 10 ms of
output. The last push may specify fewer valid samples. No playback limiter or
audio-device transport is implemented here.
"""
import numpy as np
from neural_waveform import HOP, WINDOW


class NeuralOutputStream:
    def __init__(self, decoder):
        self.decoder = decoder
        self.tail = None
        self.frames = 0
        self.samples = 0
        self.closed = False
        self.cancelled = False

    def push(self, controls, gain=1., emit_samples=HOP):
        if self.closed: raise RuntimeError('Output stream is closed')
        if (not isinstance(emit_samples, (int, np.integer)) or isinstance(emit_samples, bool)
                or not 1 <= emit_samples <= HOP or (not self.frames and emit_samples != HOP)):
            raise ValueError('Only the final output frame can have a partial sample count')
        window = self.decoder.frame(controls, gain) * WINDOW
        if self.tail is None:
            output = np.empty(0)
        else:
            weight = WINDOW[HOP:]**2 + WINDOW[:HOP]**2
            output = ((self.tail + window[:HOP]) / weight)[:emit_samples].copy()
        self.tail = window[HOP:].copy()
        self.frames += 1
        self.samples += len(output)
        if emit_samples < HOP:
            self.tail = None
            self.closed = True
        return output

    def finish(self):
        if self.frames == 1 and not self.cancelled:
            raise ValueError('A priming frame requires a terminating overlap frame')
        self.tail = None
        self.closed = True
        return np.empty(0)

    def cancel(self):
        self.tail = None
        self.closed = True
        self.cancelled = True
