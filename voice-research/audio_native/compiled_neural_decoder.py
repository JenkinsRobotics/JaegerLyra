"""Exact sparse-input execution for structured tanh waveform networks.

Only networks with at most one nonzero control coefficient per hidden unit are
eligible. This skips multiplications by exact zero; it does not prune, quantize,
retrain, change activations or retain encoder/source audio. The dense decoder
remains the independent reference and serialization format.
"""
import numpy as np
from neural_waveform import NeuralDecoder


class CompiledAxisDecoder(NeuralDecoder):
    def __init__(self, config, W, V):
        super().__init__(config, W, V)
        control_weights = self.W[:, :-1]
        if np.any(np.count_nonzero(control_weights, axis=1) > 1):
            raise ValueError('Dense or cross-coordinate hidden unit cannot use the axis compiler')
        self._columns = np.argmax(abs(control_weights), axis=1)
        self._scales = control_weights[np.arange(len(control_weights)), self._columns].copy()
        self._biases = self.W[:, -1].copy()
        for value in (self._columns, self._scales, self._biases):
            value.setflags(write=False)

    def frame(self, controls, gain=1.):
        controls = np.asarray(controls, dtype=np.float64)
        if (controls.shape != (self.config.history-1,) or not np.isfinite(controls).all()
                or not np.isfinite(gain) or gain < 0):
            raise ValueError('Invalid neural synthesis controls')
        scaled = controls * self.config.input_gain
        hidden = np.tanh(self._scales * scaled[self._columns] + self._biases * self.config.input_gain)
        result = self.V @ hidden / self.config.input_gain * gain
        if not np.isfinite(result).all():
            raise ValueError('Nonfinite generated waveform')
        return result


def compile_decoder(decoder):
    if np.any(np.count_nonzero(decoder.W[:, :-1], axis=1) > 1):
        return decoder
    return CompiledAxisDecoder(decoder.config, decoder.W, decoder.V)
