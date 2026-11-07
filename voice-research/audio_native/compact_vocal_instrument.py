"""Incremental harmonic/noise instrument with captured compact vocal shapes.

This is a project implementation of established harmonic-plus-noise synthesis,
not a newly trained neural vocoder. Offline compilation may use neural-decoded
references and external WORLD analysis. Runtime requires only compact profiles
and immediate controls, never PCM, neural weights, text or a complete score.
Algorithm reference: https://magenta.tensorflow.org/ddsp (no code copied).
"""
from dataclasses import dataclass
import json
import numpy as np
from scipy.signal import lfilter

RATE = 24000
BLOCK = 120
FFT = 2048
CEPSTRA = 96
TAPS = 257
GRID = np.fft.rfftfreq(FFT, 1 / RATE)
NOISE_HZ = np.array([0., 500., 1000., 2000., 4000., 6000., 9000., 12000.])
HARMONICS = np.arange(1, int((RATE / 2) / 60) + 1)


@dataclass(frozen=True)
class CompactShape:
    cepstrum: np.ndarray
    noise: np.ndarray
    rms: float

    def validate(self):
        if (self.cepstrum.shape != (CEPSTRA,) or self.noise.shape != NOISE_HZ.shape
                or not np.isfinite(self.cepstrum).all() or np.max(abs(self.cepstrum)) > 30
                or not np.isfinite(self.noise).all() or np.any(self.noise < 0)
                or np.any(self.noise > 1) or not np.isfinite(self.rms)
                or not 0 < self.rms <= .4):
            raise ValueError('Invalid compact vocal shape')


def compile_shape(profile, rms):
    """Capture a static envelope, not a source waveform or learned phoneme."""
    profile.validate()
    c = np.fft.irfft(.5 * profile.log_power, n=FFT)[:CEPSTRA].copy()
    noise = np.interp(NOISE_HZ, GRID, profile.aperiodicity)
    result = CompactShape(c, noise, float(rms))
    result.validate()
    for array in (c, noise):
        array.setflags(write=False)
    recovered, _ = envelope(result.cepstrum)
    band = (GRID >= 80) & (GRID <= 8000)
    error_db = 20 / np.log(10) * (recovered - .5 * profile.log_power)
    return result, {'retainedCepstralValues': CEPSTRA, 'noiseBandValues': len(NOISE_HZ),
        'rmsValues': 1, 'meanEnvelopeErrorDb': float(np.mean(abs(error_db[band]))),
        'p95EnvelopeErrorDb': float(np.percentile(abs(error_db[band]), 95))}


def envelope(cepstrum):
    # Real even cepstrum defines magnitude; doubling positive quefrencies
    # gives the corresponding minimum-phase complex log response.
    minimum = np.zeros(FFT)
    minimum[0] = cepstrum[0]
    minimum[1:len(cepstrum)] = 2 * cepstrum[1:]
    response = np.fft.rfft(minimum)
    return response.real, response.imag


def save_bank(path, shapes, metadata):
    for shape in shapes:
        shape.validate()
    if not shapes:
        raise ValueError('Empty shape bank')
    np.savez(path, cepstrum=np.array([s.cepstrum for s in shapes]),
        noise=np.array([s.noise for s in shapes]), rms=np.array([s.rms for s in shapes]),
        metadata=json.dumps(metadata))


def load_bank(path):
    with np.load(path, allow_pickle=False) as data:
        c, n, r = data['cepstrum'], data['noise'], data['rms']
        metadata = json.loads(str(data['metadata']))
    if c.ndim != 2 or n.shape != (len(c), len(NOISE_HZ)) or r.shape != (len(c),) or not len(c):
        raise ValueError('Invalid shape-bank dimensions')
    shapes = [CompactShape(a.copy(), b.copy(), float(v)) for a, b, v in zip(c, n, r)]
    for shape in shapes:
        shape.validate()
        shape.cepstrum.setflags(write=False); shape.noise.setflags(write=False)
    return shapes, metadata


@dataclass(frozen=True)
class VocalFrame:
    hz: float
    weights: tuple
    gain: float = 1.
    breath: float = 0.

    def validate(self, shapes):
        w = np.asarray(self.weights, dtype=float)
        if (w.shape != (shapes,) or not np.isfinite(w).all() or np.any(w < 0)
                or not np.isclose(w.sum(), 1., rtol=0., atol=1e-12)
                or not np.isfinite((self.hz, self.gain, self.breath)).all()
                or not 60 <= self.hz <= 800 or not 0 <= self.gain <= 2
                or not 0 <= self.breath <= 1):
            raise ValueError('Invalid immediate vocal controls')


class CompactVocalStream:
    """One 5 ms packet per supplied frame, with no future-score dependency.

    Frame values are endpoints, interpolated from the preceding frame. The
    first frame is constant. Noise uses a seeded local RNG; phase, prior filter
    and 256 noise samples persist. Cancel clears synthesis state and prevents
    future output. Synchronous cancel cannot remove consumer-queued audio.
    """
    def __init__(self, shapes, seed=0):
        if not shapes:
            raise ValueError('Empty shape bank')
        for shape in shapes:
            shape.validate()
        self.cepstra = np.array([s.cepstrum for s in shapes])
        self.noises = np.array([s.noise for s in shapes])
        self.log_rms = np.log([s.rms for s in shapes])
        for array in (self.cepstra, self.noises, self.log_rms):
            array.setflags(write=False)
        self.rng = np.random.default_rng(seed)
        self.noise_history = np.zeros(TAPS - 1)
        self.phase = 0.
        self.previous = None
        self.previous_harmonics = None
        self.previous_filter = None
        self.position = 0
        self.closed = False

    def _parameters(self, frame):
        weights = np.asarray(frame.weights)
        log_amplitude, phase = envelope(weights @ self.cepstra)
        amplitude = np.exp(np.clip(log_amplitude, -30, 15))
        noise = np.interp(GRID, NOISE_HZ, weights @ self.noises)
        noise_power = np.clip(noise ** 2 + frame.breath ** 2, 0., 1.)
        power = amplitude ** 2
        fraction = float(np.sum(power * noise_power) / max(np.sum(power), 1e-30))
        rms = float(np.exp(weights @ self.log_rms))
        frequencies = HARMONICS * frame.hz
        amp = np.interp(frequencies, GRID, amplitude) * np.sqrt(
            1 - np.interp(frequencies, GRID, noise_power))
        # A smooth last-500-Hz rolloff avoids an abruptly disappearing partial.
        amp *= np.clip((RATE / 2 - frequencies) / 500., 0., 1.)
        amp *= rms * np.sqrt(2 * (1 - fraction) / max(np.sum(amp ** 2), 1e-30))
        harmonics = amp * np.exp(1j * np.interp(frequencies, GRID, phase))

        noise_amplitude = amplitude * np.sqrt(noise_power)
        noise_amplitude *= np.clip(GRID / 60., 0., 1.)
        # Minimum-phase FIR from the requested noise magnitude. The impulse
        # is truncated explicitly; normalization preserves requested variance.
        c = np.fft.irfft(np.log(np.maximum(noise_amplitude, 1e-20)), n=FFT)
        causal = np.zeros(FFT); causal[0] = c[0]
        causal[1:FFT//2] = 2 * c[1:FFT//2]; causal[FFT//2] = c[FFT//2]
        impulse = np.fft.irfft(np.exp(np.fft.rfft(causal)), n=FFT)[:TAPS].copy()
        impulse *= rms * np.sqrt(fraction / max(np.sum(impulse ** 2), 1e-30))
        return harmonics, impulse

    def push(self, frame):
        if self.closed:
            raise RuntimeError('Instrument cancelled')
        frame.validate(len(self.cepstra))
        frame = VocalFrame(float(frame.hz), tuple(frame.weights), float(frame.gain), float(frame.breath))
        harmonics, impulse = self._parameters(frame)
        previous = frame if self.previous is None else self.previous
        previous_harmonics = harmonics if self.previous is None else self.previous_harmonics
        previous_filter = impulse if self.previous is None else self.previous_filter
        progress = (np.arange(BLOCK) + 1) / BLOCK
        frequency = np.exp(np.log(previous.hz) + progress * np.log(frame.hz / previous.hz))
        phases = self.phase + np.cumsum(frequency) / RATE
        h = previous_harmonics[None, :] * (1 - progress[:, None]) + harmonics[None, :] * progress[:, None]
        # Also suppress old partials during an abrupt upward pitch transition.
        h *= np.clip((RATE / 2 - frequency[:, None] * HARMONICS) / 500., 0., 1.)
        audio = np.real(np.sum(h * np.exp(2j * np.pi * phases[:, None] * HARMONICS), axis=1))
        raw_noise = self.rng.normal(size=BLOCK)
        noise = np.r_[self.noise_history, raw_noise]
        a = lfilter(previous_filter, [1.], noise)[TAPS - 1:]
        b = lfilter(impulse, [1.], noise)[TAPS - 1:]
        audio += a * (1 - progress) + b * progress
        audio *= previous.gain * (1 - progress) + frame.gain * progress
        self.noise_history = noise[-(TAPS - 1):].copy()
        self.phase = float(phases[-1] % 1)
        self.previous = frame; self.previous_harmonics = harmonics; self.previous_filter = impulse
        self.position += BLOCK
        return audio

    def cancel(self):
        self.closed = True; self.noise_history.fill(0)
        self.phase = 0.; self.previous = None
        self.previous_harmonics = None; self.previous_filter = None


def render_score(shapes, score, path, breath=0., seed=0):
    """Convenience driver; the instrument itself receives only each next frame."""
    score.validate(); path.validate(len(shapes), score.seconds)
    stream = CompactVocalStream(shapes, seed=seed)
    count = round(score.seconds * RATE)
    pieces = []
    for start in range(0, count, BLOCK):
        at = (start + BLOCK - 1) / RATE
        frame = VocalFrame(float(score.frequency(at)), tuple(path.at(at)), 1., breath)
        pieces.append(stream.push(frame))
    return np.concatenate(pieces)[:count] * score.envelope(np.arange(count), count)
