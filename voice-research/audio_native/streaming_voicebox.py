"""Continuous source/filter research scaffold, not a speech or identity model.

Standard autocorrelation LPC, explicit residual PCM, and causal stateful filters.
No text, discrete codebook, pretrained weights or gradient training. Keeping the
full residual is essential to exact reconstruction; this is NOT compression.
The equations follow classical LPC/source-filter DSP, not a novel invention.
References and provenance are recorded in FOUNDATION.md.
"""
from dataclasses import dataclass
import numpy as np
from scipy.signal import lfilter, lfiltic, find_peaks, freqz

LPC_VERSION='2-unclipped-levinson'


@dataclass(frozen=True)
class VoiceboxConfig:
    sample_rate: int = 24000
    hop: int = 240
    history: int = 960
    order: int = 20
    bandwidth: float = .99

    def __post_init__(self):
        if (self.sample_rate != 24000 or self.hop < 32 or self.history < self.hop
                or not 2 <= self.order < min(self.hop, self.history)
                or not 0 < self.bandwidth < 1):
            raise ValueError('Invalid source/filter configuration')


def mono(samples):
    value = np.asarray(samples, dtype=np.float64)
    if value.ndim != 1 or not np.isfinite(value).all():
        raise ValueError('Expected finite mono PCM')
    return value


def lpc(history, config):
    """Levinson recursion with regularization and bandwidth expansion.

    This fits an envelope to already observed audio. It does not train a neural
    predictor, establish physiological parameters, or isolate speaker identity.
    """
    signal = (history-np.mean(history))*np.hanning(len(history))
    autocorrelation = np.array([signal[k:]@signal[:len(signal)-k]
                                for k in range(config.order+1)])
    autocorrelation[0] += 1e-6 + autocorrelation[0]*1e-4
    coefficients = np.array([1.])
    error = autocorrelation[0]
    for order in range(1, config.order+1):
        reflection = -(autocorrelation[order]
                       + coefficients[1:]@autocorrelation[order-1:0:-1])/error
        # The error recurrence assumes the unconstrained covariance solution.
        # Clipping k inside this solve invalidates later stages and can create
        # enormous coefficients despite every individually clipped |k| < 1.
        # Positive diagonal loading makes this autocorrelation system positive
        # definite; bandwidth expansion is applied only after solving it.
        if not np.isfinite(reflection) or abs(reflection)>=1:
            raise ValueError('LPC covariance solve lost positive definiteness')
        coefficients = np.r_[coefficients, 0.] + reflection*np.r_[0., coefficients[::-1]]
        error *= 1-reflection*reflection
    return coefficients*config.bandwidth**np.arange(config.order+1)


def pitch(history, rate):
    """Causal, normalized autocorrelation diagnostic; not an accepted F0 tracker.

    Reports zero for uncertain/unvoiced frames. Octave errors and breathy speech
    failures must be measured, not inferred away by the voicing threshold.
    """
    signal = history-np.mean(history)
    if np.sqrt(np.mean(signal*signal)) < .003:
        return 0., 0.
    low, high = int(rate/500), min(int(rate/65), len(signal)//2)
    correlations = np.array([signal[k:]@signal[:-k]
        / np.sqrt(max(1e-20, (signal[k:]@signal[k:])*(signal[:-k]@signal[:-k])))
        for k in range(low, high+1)])
    peaks, _ = find_peaks(correlations)
    if not len(peaks):
        return 0., 0.
    best = float(np.max(correlations[peaks]))
    # Prefer the first strong period rather than a later integer multiple.
    choices = peaks[correlations[peaks] >= max(.60, best*.94)]
    if not len(choices):
        return 0., float(max(0., best))
    index = choices[0]
    a, b, c = correlations[index-1:index+2]
    offset = .5*(a-c)/(a-2*b+c) if abs(a-2*b+c)>1e-12 else 0.
    return float(rate/(low+index+np.clip(offset, -.5, .5))), float(b)


@dataclass
class AudioFrame:
    start: int
    available_at: int
    coefficients: np.ndarray
    residual: np.ndarray
    f0: float
    periodicity: float
    rms: float

    def features(self, sample_rate=24000):
        """Continuous envelope/prosody diagnostics, not phoneme/semantic codes."""
        frequencies = np.geomspace(100, 8000, 16)
        _, response = freqz([1.], self.coefficients,
                            worN=2*np.pi*frequencies/sample_rate)
        envelope_db = 20*np.log10(np.maximum(np.abs(response), 1e-8))
        envelope_db -= np.mean(envelope_db)
        return np.r_[envelope_db, 20*np.log10(max(self.rms, 1e-5)),
                     np.log2(self.f0/160) if self.f0 else 0., self.periodicity]


class StreamingAnalysis:
    """Emits 10 ms frames as received, using up to 40 ms of observed history.

    No future beyond a frame's available_at may influence it. The analysis
    history grows at startup without waiting for 40 ms. A partial final frame
    is emitted only on finish(). Transport chunk boundaries do not change data.
    """
    def __init__(self, config=VoiceboxConfig()):
        self.config = config
        self.pending = np.empty(0)
        self.history = np.zeros(config.history)
        self.input_history = np.zeros(config.order)
        self.position = 0
        self.closed = False

    def _frame(self, samples):
        cfg = self.config
        self.history = np.r_[self.history, samples][-cfg.history:]
        coefficients = lpc(self.history, cfg)
        # Recompute direct-form filter state from actual past samples whenever
        # coefficients change; carrying old lfilter zi here creates artifacts.
        initial = lfiltic(coefficients, [1.], [], self.input_history[::-1])
        residual, _ = lfilter(coefficients, [1.], samples, zi=initial)
        self.input_history = np.r_[self.input_history, samples][-cfg.order:]
        f0, periodicity = pitch(self.history, cfg.sample_rate)
        frame = AudioFrame(self.position, self.position+len(samples), coefficients,
                           residual, f0, periodicity, float(np.sqrt(np.mean(samples*samples))))
        self.position += len(samples)
        return frame

    def push(self, samples):
        if self.closed:
            raise ValueError('Analysis has finished')
        self.pending = np.r_[self.pending, mono(samples)]
        frames = []
        while len(self.pending) >= self.config.hop:
            frames.append(self._frame(self.pending[:self.config.hop]))
            self.pending = self.pending[self.config.hop:]
        # Copy the small tail so it cannot retain a large caller allocation.
        self.pending = self.pending.copy()
        return frames

    def finish(self):
        if self.closed:
            return []
        self.closed = True
        frames = [self._frame(self.pending)] if len(self.pending) else []
        self.pending = np.empty(0)
        return frames


def resonance_coefficients(coefficients, factor):
    """Warp pole angles, retaining radii; pitch/source timing is not resampled.

    This is an approximate spectral color control, NOT speaker cloning or
    independent articulator control. Per-frame stable poles alone do not prove
    stability of every time-varying filter sequence.
    """
    if not np.isfinite(factor) or not .85 <= factor <= 1.15:
        raise ValueError('Resonance factor must be 0.85–1.15')
    if factor == 1.:
        return coefficients.copy()
    poles = np.roots(coefficients)
    angles = np.angle(poles)
    shifted = np.clip(angles*factor, -np.pi, np.pi)
    poles = np.abs(poles)*np.exp(1j*shifted)
    return np.real(np.poly(poles))


class StreamingSynthesis:
    """Reconstruct continuous PCM, keeping output history across every frame.

    Unchanged filter/residual is algebraically reversible. Resonance edits are
    experimental; invalid output raises rather than silently clipping audio.
    """
    def __init__(self, config=VoiceboxConfig()):
        self.config = config
        self.output_history = np.zeros(config.order)
        self.position = 0

    def push(self, frame, resonance=1., residual=None):
        if frame.start != self.position:
            raise ValueError('Frames must arrive once, in order')
        coefficients = resonance_coefficients(frame.coefficients, resonance)
        excitation = frame.residual if residual is None else mono(residual)
        if len(excitation) != len(frame.residual):
            raise ValueError('Excitation length must match the frame')
        initial = lfiltic([1.], coefficients, self.output_history[::-1])
        samples, _ = lfilter([1.], coefficients, excitation, zi=initial)
        if not np.isfinite(samples).all() or np.max(np.abs(samples), initial=0) > 10:
            raise ValueError('Unstable or excessive source/filter output')
        self.output_history = np.r_[self.output_history, samples][-self.config.order:]
        self.position += len(samples)
        return samples


def process(samples, chunk_sizes=(127, 480, 53, 1024), resonance=1., residual_mode='original'):
    """File fixture adapter; the analysis/synthesis classes themselves stream."""
    if not chunk_sizes or min(chunk_sizes)<1 or residual_mode not in ('original', 'noise'):
        raise ValueError('Invalid fixture processing options')
    samples = mono(samples)
    analysis, synthesis = StreamingAnalysis(), StreamingSynthesis()
    frames, output = [], []
    rng = np.random.default_rng(79)
    def consume(items):
        for frame in items:
            frames.append(frame)
            residual = None
            if residual_mode == 'noise':
                residual = rng.normal(0, np.sqrt(np.mean(frame.residual**2)), len(frame.residual))
            output.append(synthesis.push(frame, resonance=resonance, residual=residual))
    at, index = 0, 0
    while at < len(samples):
        chunk = samples[at:at+chunk_sizes[index % len(chunk_sizes)]]
        consume(analysis.push(chunk))
        at += len(chunk)
        index += 1
    consume(analysis.finish())
    return (np.concatenate(output) if output else np.empty(0)), frames
