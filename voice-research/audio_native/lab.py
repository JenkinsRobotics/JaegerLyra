"""Small continuous-signal experiments; not a speech/dialogue model.

No transcript, phoneme tokenizer, audio codebook, autograd, or pretrained weights.
The local learner uses explicit energy gradients and symmetric feedback. Avoiding
global backpropagation does not make it gradient-free or biologically established.
"""
from dataclasses import dataclass
import numpy as np


def cosine_basis(size):
    """Full orthonormal DCT-II basis. Keeping every coefficient preserves phase."""
    sample = np.arange(size)[None, :]
    frequency = np.arange(size)[:, None]
    basis = np.sqrt(2 / size) * np.cos(np.pi / size * (sample + .5) * frequency)
    basis[0] /= np.sqrt(2)
    return basis


def continuous_roundtrip(pcm, frame_size=240):
    """Reversible framed analysis, zero codebook/compression; ~10 ms at 24 kHz."""
    pcm = np.asarray(pcm, dtype=np.float64)
    if pcm.ndim != 1 or not np.isfinite(pcm).all():
        raise ValueError('Expected finite mono PCM')
    if not len(pcm):
        return pcm.copy()
    basis = cosine_basis(frame_size)
    padding = (-len(pcm)) % frame_size
    frames = np.pad(pcm, (0, padding)).reshape(-1, frame_size)
    coefficients = frames @ basis.T
    return (coefficients @ basis).reshape(-1)[:len(pcm)]


@dataclass(frozen=True)
class Config:
    history: int = 192
    block: int = 32
    hidden: int = 64
    settling_steps: int = 12
    learning_rate: float = .04
    input_gain: float = 4.
    seed: int = 17


class LocalPredictor:
    """Two-layer supervised predictive-coding toy with a clamped PCM target.

    For past audio x, hidden state h and the newly observed next block y:
      p = tanh(W x)
      E = .5 ||h - p||^2 + .5 ||y - V h||^2
      dh = -(h - p) + V.T (y - V h)
      dW = ((h - p) * (1 - p^2)) outer x
      dV = (y - V h) outer h

    Inference settles h while W,V stay fixed. Then adjacent-layer errors update
    weights once. Layer-norm step scaling is an engineering stabilizer; this is
    not a strict biological-locality claim. The benchmark scores V tanh(W x)
    BEFORE clamping y; target-conditioned reconstruction is reported separately.
    """
    def __init__(self, config=Config()):
        self.config = config
        rng = np.random.default_rng(config.seed)
        self.W = rng.normal(0, 1 / np.sqrt(config.history),
                            (config.hidden, config.history))
        self.V = np.zeros((config.block, config.hidden))

    def _input(self, past):
        x = np.asarray(past, dtype=np.float64) * self.config.input_gain
        if x.shape != (self.config.history,) or not np.isfinite(x).all():
            raise ValueError('Invalid PCM history')
        return x

    def predict(self, past):
        x = self._input(past)
        return self.V @ np.tanh(self.W @ x) / self.config.input_gain

    def observe(self, past, observed, learn=True):
        x = self._input(past)
        y = np.asarray(observed, dtype=np.float64) * self.config.input_gain
        if y.shape != (self.config.block,) or not np.isfinite(y).all():
            raise ValueError('Invalid observed PCM block')
        p = np.tanh(self.W @ x)
        h = p.copy()
        def energy(state):
            return .5 * (np.sum((state-p)**2) + np.sum((y-self.V@state)**2))
        initial = energy(h)
        # 1 + ||V||_F^2 bounds the Hessian's largest eigenvalue.
        step = .9 / (1 + np.sum(self.V**2))
        for _ in range(self.config.settling_steps):
            h += step * (self.V.T @ (y - self.V @ h) - (h - p))
        final = energy(h)
        conditioned = self.V @ h / self.config.input_gain
        if learn:
            # Both errors use the same settled states and pre-update weights.
            upper_error = h - p
            lower_error = y - self.V @ h
            self.W += self.config.learning_rate * np.outer(
                upper_error * (1-p*p), x) / (1 + x @ x)
            self.V += self.config.learning_rate * np.outer(
                lower_error, h) / (1 + h @ h)
        return {'initialEnergy': initial, 'finalEnergy': final,
                'conditionedReconstruction': conditioned}


class LinearPredictor:
    """Direct normalized-LMS comparator; no hidden-state inference."""
    def __init__(self, config=Config()):
        self.config = config
        self.W = np.zeros((config.block, config.history))

    def predict(self, past):
        return self.W @ past

    def observe(self, past, observed, learn=True):
        if learn:
            error = observed - self.predict(past)
            self.W += .15 * np.outer(error, past) / (.01 + past @ past)


def voicebox(f0, formants, sample_rate=24000, breath=.015, seed=19):
    """A controllable harmonic/noise source with an analytic spectral envelope.

    Formants are explicit continuous trajectories, not learned speaker identity.
    This only makes vowel-like sounds. It cannot pronounce words or clone a voice.
    No neural residual and no learned physiological model are claimed.
    """
    f0 = np.asarray(f0, dtype=np.float64)
    formants = np.asarray(formants, dtype=np.float64)
    if f0.ndim != 1 or formants.shape != (len(f0), 3):
        raise ValueError('Expected one pitch and three formants per sample')
    if (not np.isfinite(f0).all() or not np.isfinite(formants).all()
            or np.min(f0) < 60 or np.max(f0) > 800
            or np.min(formants) <= 0 or np.max(formants) >= sample_rate / 2):
        raise ValueError('Pitch or formants outside this toy synthesizer range')
    phase = 2*np.pi*np.cumsum(f0)/sample_rate
    audio = np.zeros(len(f0))
    for harmonic in range(1, min(100, int(sample_rate / (2*np.min(f0))))):
        frequency = harmonic*f0
        envelope = np.full(len(f0), .01)
        for j, width in enumerate((90, 130, 180)):
            envelope += np.exp(-.5*((frequency-formants[:, j])/width)**2)
        # Smooth taper prevents harmonics crossing Nyquist from aliasing abruptly.
        taper = np.clip((sample_rate*.49-frequency)/(sample_rate*.04), 0, 1)
        audio += np.sin(harmonic*phase)*envelope*taper / harmonic**.8
    noise = np.random.default_rng(seed).normal(0, 1, len(f0))
    audio += breath * (noise-np.roll(noise, 1)) * .5
    fade = min(int(sample_rate*.03), len(audio)//2)
    audio[:fade] *= np.linspace(0, 1, fade)
    audio[-fade:] *= np.linspace(1, 0, fade)
    audio *= .12 / max(1e-9, np.sqrt(np.mean(audio**2)))
    return audio
