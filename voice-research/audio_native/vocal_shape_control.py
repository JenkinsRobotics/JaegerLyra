"""Continuous vowel-shape controls and an explicitly external WORLD reference.

Profiles are estimated from short neural-decoded captured vowels. The control
path interpolates spectral shape and aperiodicity, rather than crossfading PCM.
WORLD performs reference synthesis; this is neither a new neural vocoder nor a
clean-room engine. The reference renderer is offline, not a streaming backend.
"""
from dataclasses import dataclass
import json
import numpy as np
import pyworld
from neural_waveform import RATE
from neural_vowel_instrument import NeuralVowelInstrument

FFT_SIZE = 2048
FREQUENCIES = np.fft.rfftfreq(FFT_SIZE, 1/RATE)


@dataclass(frozen=True)
class VocalShape:
    log_power: np.ndarray
    aperiodicity: np.ndarray

    def validate(self):
        if (self.log_power.shape != (FFT_SIZE//2+1,)
                or self.aperiodicity.shape != self.log_power.shape
                or not np.isfinite(self.log_power).all()
                or not np.isfinite(self.aperiodicity).all()
                or np.any(abs(self.log_power) > 60)
                or np.any(self.aperiodicity < 0) or np.any(self.aperiodicity > 1)):
            raise ValueError('Invalid vocal-shape profile')


def analyse_shape(patch, decoder):
    """Offline shape compilation; no new neural weights or phoneme recognition."""
    decoded = np.ascontiguousarray(decoder.render(patch.matrix))
    times = np.arange(0, len(decoded)/RATE, .005)
    periods = np.diff(patch.pulses)
    pitch_times = (patch.pulses[:-1]+patch.pulses[1:])/(2*RATE)
    f0 = np.interp(times, pitch_times, RATE/periods)
    power = pyworld.cheaptrick(decoded, f0, times, RATE, fft_size=FFT_SIZE)
    # References were already selected as stable voiced regions. Keep their
    # estimated aperiodicity rather than applying a second voicing rejection.
    noise = pyworld.d4c(decoded, f0, times, RATE, fft_size=FFT_SIZE, threshold=0.)
    interior = (times >= .025) & (times <= len(decoded)/RATE-.025)
    if sum(interior) < 4:
        raise ValueError('Insufficient interior frames for a stable vowel shape')
    profile = VocalShape(np.median(np.log(np.maximum(power[interior], 1e-20)), axis=0),
                         np.median(noise[interior], axis=0))
    profile.validate()
    profile.log_power.setflags(write=False); profile.aperiodicity.setflags(write=False)
    return profile, {'analysisFrames': int(sum(interior)), 'fftSize': FFT_SIZE,
        'sourceSeconds': len(decoded)/RATE, 'neuralDecodedReference': True,
        'voicingThreshold': 0., 'sourcePitchHz': float(np.median(f0[interior]))}


@dataclass(frozen=True)
class ShapePath:
    times: tuple
    weights: tuple

    def validate(self, shapes, seconds):
        t = np.asarray(self.times, dtype=float); w = np.asarray(self.weights, dtype=float)
        if (t.ndim != 1 or not len(t) or w.shape != (len(t), shapes)
                or not np.isfinite(t).all() or not np.isfinite(w).all()
                or t[0] != 0 or np.any(np.diff(t) <= 0) or t[-1] > seconds
                or np.any(w < 0) or not np.allclose(w.sum(axis=1), 1., atol=1e-12, rtol=0.)):
            raise ValueError('Expected timed convex vowel weights')

    def at(self, seconds):
        t = np.asarray(self.times); weights = np.asarray(self.weights)
        query = np.asarray(seconds)
        if len(t) == 1:
            return np.broadcast_to(weights[0], (*query.shape, weights.shape[1])).copy()
        right = np.clip(np.searchsorted(t, query, side='right'), 1, len(t)-1)
        left = right-1
        progress = np.clip((query-t[left])/(t[right]-t[left]), 0., 1.)
        smooth = progress*progress*(3-2*progress)
        return weights[left]*(1-smooth[...,None])+weights[right]*smooth[...,None]


def shape_controls(profiles, path, times, seconds):
    path.validate(len(profiles), seconds)
    for profile in profiles:
        profile.validate()
    weights = path.at(times)
    log_power = weights @ np.array([p.log_power for p in profiles])
    # Aperiodicity is interpolated as the documented WORLD control value.
    # No claim is made that it is a physical vocal-fold parameter.
    aperiodicity = weights @ np.array([p.aperiodicity for p in profiles])
    return np.exp(log_power), aperiodicity


def render_world_shapes(profiles, score, path):
    """Offline external-vocoder comparator; never presented as our neural core."""
    score.validate(); path.validate(len(profiles), score.seconds)
    count = round(score.seconds*RATE)
    times = np.arange(int(np.ceil(count/120))+1)*.005
    power, noise = shape_controls(profiles, path, times, score.seconds)
    pitch = np.ascontiguousarray(score.frequency(times))
    audio = pyworld.synthesize(pitch, np.ascontiguousarray(power), np.ascontiguousarray(noise),
        RATE, frame_period=5.)
    if len(audio) < count:
        raise ValueError('External vocoder returned insufficient samples')
    audio = audio[:count] * score.envelope(np.arange(count), count)
    if not np.isfinite(audio).all():
        raise ValueError('Nonfinite external reference output')
    return audio


def render_grain_crossfade(patches, decoder, score, path):
    """Baseline: direct waveform crossfade of existing neural vowel instruments."""
    score.validate(); path.validate(len(patches), score.seconds)
    count = round(score.seconds*RATE)
    weights = path.at(np.arange(count)/RATE)
    audio = np.zeros(count)
    for i, patch in enumerate(patches):
        stream = NeuralVowelInstrument(patch, decoder, score)
        for packet in stream:
            sl = slice(packet.start, packet.start+len(packet.samples))
            audio[sl] += packet.samples * weights[sl, i]
    return audio


def save_shapes(path, profiles, metadata):
    for profile in profiles:
        profile.validate()
    np.savez(path, log_power=np.array([p.log_power for p in profiles]),
        aperiodicity=np.array([p.aperiodicity for p in profiles]), metadata=json.dumps(metadata))


def load_shapes(path):
    with np.load(path, allow_pickle=False) as data:
        log_power = data['log_power']; noise = data['aperiodicity']
        metadata = json.loads(str(data['metadata']))
    if log_power.ndim != 2 or log_power.shape != noise.shape or not len(log_power):
        raise ValueError('Invalid shape-bank dimensions')
    profiles = [VocalShape(p.copy(), n.copy()) for p, n in zip(log_power, noise)]
    for profile in profiles:
        profile.validate()
        profile.log_power.setflags(write=False); profile.aperiodicity.setflags(write=False)
    return profiles, metadata
